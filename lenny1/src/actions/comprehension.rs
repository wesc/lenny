use anyhow::Result;
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures_util::StreamExt;
use lancedb::embeddings::sentence_transformers::SentenceTransformersEmbeddings;
use lancedb::embeddings::{EmbeddingDefinition, EmbeddingFunction};
use lancedb::query::{ExecutableQuery, QueryBase};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::Config;
use crate::context;
use crate::once;

/// Approximate token count via whitespace word count.
fn approx_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Move a file from `src` to `dst`, creating parent dirs as needed.
fn move_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::rename(src, dst).is_err() {
        fs::copy(src, dst)?;
        fs::remove_file(src)?;
    }
    Ok(())
}

/// File with its content and metadata for comprehension decisions.
pub struct DynamicFile {
    pub path: PathBuf,
    pub relative: PathBuf,
    pub content: String,
    pub tokens: usize,
}

/// Collect files from dynamic/, read contents, compute token counts.
pub fn collect_dynamic_files(config: &Config) -> Result<Vec<DynamicFile>> {
    let paths = context::collect_files(&config.dynamic_dir)?;
    let mut files = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(&config.dynamic_dir)
            .unwrap_or(&path)
            .to_path_buf();
        let content = fs::read_to_string(&path)?;
        let tokens = approx_tokens(&content);
        files.push(DynamicFile {
            path,
            relative,
            content,
            tokens,
        });
    }
    Ok(files)
}

/// Select oldest files to pop so remaining tokens <= min_context_tokens.
/// Returns (files_to_pop, files_to_keep).
pub fn select_files_to_pop(
    mut files: Vec<DynamicFile>,
    max_tokens: usize,
    min_tokens: usize,
) -> (Vec<DynamicFile>, Vec<DynamicFile>) {
    let total: usize = files.iter().map(|f| f.tokens).sum();
    if total <= max_tokens {
        return (vec![], files);
    }

    files.sort_by(|a, b| a.relative.cmp(&b.relative));

    let mut remaining = total;
    let mut split_idx = 0;
    for (i, file) in files.iter().enumerate() {
        if remaining <= min_tokens {
            break;
        }
        remaining -= file.tokens;
        split_idx = i + 1;
    }

    let keep = files.split_off(split_idx);
    (files, keep)
}

/// Move popped files from dynamic/ to references/, preserving subdirectory structure.
pub fn move_to_references(config: &Config, files: &[DynamicFile]) -> Result<()> {
    for file in files {
        let dest = config.references_dir().join(&file.relative);
        move_file(&file.path, &dest)?;
    }
    Ok(())
}

/// A single comprehension entry produced by the LLM.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ComprehensionEntry {
    /// The comprehension text.
    pub text: String,
    /// Unix timestamp (seconds) for when this event/information occurred.
    pub timestamp: i64,
}

/// Structured output from the LLM for each file.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ComprehensionOutput {
    /// Updated running summary incorporating this file.
    pub summary: String,
    /// Comprehension entries extracted from this file.
    pub comprehensions: Vec<ComprehensionEntry>,
}

const COMPREHENSION_PREAMBLE: &str = "\
You are a knowledge extraction assistant. You are extracting knowledge for an AI assistant. \
You are given a running summary of files processed so far, and the content of a new file being archived.

These files are from the assistant's knowledge base. When files use \"you\" or \"your\", they refer to \
the assistant, not a human user. For example, \"your favorite color is chartreuse\" means the assistant's \
favorite color is chartreuse.

Produce a JSON object with:
1. \"summary\": an updated running summary (2-4 sentences) incorporating the new file's key information.
2. \"comprehensions\": an array of comprehension entries, each with:
   - \"text\": a self-contained statement capturing a key fact, decision, event, or insight from the file. \
Use third person (e.g. \"The assistant's favorite car is the Toyota Pineapple\").
   - \"timestamp\": the unix timestamp (seconds) of when the event/information occurred, extracted from \
the file content. Use your best judgment; if no timestamp is apparent, use 0.

Extract as many distinct comprehension entries as are warranted by the content. Each entry should \
be independently understandable without needing to read the source file.";

/// Write comprehension entries to LanceDB.
pub async fn write_to_lancedb(
    db_path: &Path,
    entries: Vec<(String, i64, String)>, // (summary, timestamp, file_reference)
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    let embedding = Arc::new(SentenceTransformersEmbeddings::builder().build()?);

    let db = lancedb::connect(db_path.to_str().unwrap())
        .execute()
        .await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    let summaries: Vec<&str> = entries.iter().map(|(s, _, _)| s.as_str()).collect();
    let timestamps: Vec<i64> = entries.iter().map(|(_, t, _)| *t).collect();
    let file_refs: Vec<&str> = entries.iter().map(|(_, _, f)| f.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("summary", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("file_reference", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(summaries)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(file_refs)),
        ],
    )?;

    let table_exists = db.open_table("comprehensions").execute().await.is_ok();

    if table_exists {
        let table = db.open_table("comprehensions").execute().await?;
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        table.add(reader).execute().await?;
    } else {
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        db.create_table("comprehensions", reader)
            .add_embedding(EmbeddingDefinition::new(
                "summary",
                "sentence-transformers",
                Some("summary_embedding"),
            ))?
            .execute()
            .await?;
    }

    Ok(())
}

/// Result of a retrieval query against the comprehensions DB.
pub struct RetrieveResult {
    /// Formatted context string with matched comprehensions.
    pub context: String,
}

/// Retrieve relevant comprehensions for a query, optionally filtered by time range.
/// `time_range` is `Some((start_unix, end_unix))` for inclusive filtering.
pub async fn retrieve(
    db_path: &Path,
    query: &str,
    top_k: usize,
    time_range: Option<(i64, i64)>,
) -> Result<RetrieveResult> {
    if !db_path.exists() {
        return Ok(RetrieveResult {
            context: String::new(),
        });
    }

    let embedding = Arc::new(SentenceTransformersEmbeddings::builder().build()?);

    let db = lancedb::connect(db_path.to_str().unwrap())
        .execute()
        .await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    let table = match db.open_table("comprehensions").execute().await {
        Ok(t) => t,
        Err(_) => {
            return Ok(RetrieveResult {
                context: String::new(),
            });
        }
    };

    let query_arr = Arc::new(StringArray::from_iter_values(std::iter::once(query)));
    let query_vector = embedding.compute_query_embeddings(query_arr)?;

    let mut search = table
        .vector_search(query_vector)?
        .column("summary_embedding")
        .limit(top_k);

    if let Some((start, end)) = time_range {
        search = search.only_if(format!("timestamp >= {start} AND timestamp <= {end}"));
    }

    let mut results = search.execute().await?;

    let mut entries = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;

        let summary_col = batch
            .column_by_name("summary")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let file_ref_col = batch
            .column_by_name("file_reference")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let ts_col = batch
            .column_by_name("timestamp")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let summary = summary_col.value(i);
            let file_ref = file_ref_col.value(i);
            let ts = ts_col.value(i);
            entries.push(format!("[{file_ref} @ {ts}] {summary}"));
        }
    }

    let context = entries.join("\n\n");

    Ok(RetrieveResult { context })
}

/// Dump all comprehension entries from LanceDB as JSON to stdout.
pub async fn dump_json(config: &Config) -> Result<()> {
    let db_path = config.comprehensions_dir();

    if !db_path.exists() {
        println!("[]");
        return Ok(());
    }

    let db = lancedb::connect(db_path.to_str().unwrap())
        .execute()
        .await?;

    let table = match db.open_table("comprehensions").execute().await {
        Ok(t) => t,
        Err(_) => {
            println!("[]");
            return Ok(());
        }
    };

    let mut results = table
        .query()
        .select(lancedb::query::Select::columns(&[
            "summary",
            "timestamp",
            "file_reference",
        ]))
        .execute()
        .await?;

    let mut entries = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;

        let summary_col = batch
            .column_by_name("summary")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let ts_col = batch
            .column_by_name("timestamp")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let file_ref_col = batch
            .column_by_name("file_reference")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            entries.push(serde_json::json!({
                "summary": summary_col.value(i),
                "timestamp": ts_col.value(i),
                "file_reference": file_ref_col.value(i),
            }));
        }
    }

    println!("{}", serde_json::to_string_pretty(&entries)?);
    Ok(())
}

/// Search comprehensions by semantic similarity and print top N matches as JSON.
pub async fn search_json(config: &Config, query: &str, top_k: usize) -> Result<()> {
    let db_path = config.comprehensions_dir();

    if !db_path.exists() {
        println!("[]");
        return Ok(());
    }

    let embedding = Arc::new(SentenceTransformersEmbeddings::builder().build()?);

    let db = lancedb::connect(db_path.to_str().unwrap())
        .execute()
        .await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    let table = match db.open_table("comprehensions").execute().await {
        Ok(t) => t,
        Err(_) => {
            println!("[]");
            return Ok(());
        }
    };

    let query_arr = Arc::new(StringArray::from_iter_values(std::iter::once(query)));
    let query_vector = embedding.compute_query_embeddings(query_arr)?;

    let mut results = table
        .vector_search(query_vector)?
        .column("summary_embedding")
        .limit(top_k)
        .execute()
        .await?;

    let mut entries = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;

        let summary_col = batch
            .column_by_name("summary")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let ts_col = batch
            .column_by_name("timestamp")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let file_ref_col = batch
            .column_by_name("file_reference")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            entries.push(serde_json::json!({
                "summary": summary_col.value(i),
                "timestamp": ts_col.value(i),
                "file_reference": file_ref_col.value(i),
            }));
        }
    }

    println!("{}", serde_json::to_string_pretty(&entries)?);
    Ok(())
}

/// Run comprehension on dynamic/. Returns Ok(true) if comprehension occurred.
/// If `force` is true, skip the token threshold check and comprehend all files.
pub async fn run(config: &Config, force: bool) -> Result<bool> {
    let files = collect_dynamic_files(config)?;

    if files.is_empty() {
        return Ok(false);
    }

    let (to_pop, _keep) = if force {
        (files, vec![])
    } else {
        let total_tokens: usize = files.iter().map(|f| f.tokens).sum();
        if total_tokens <= config.max_context_tokens {
            return Ok(false);
        }
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens)
    };

    if to_pop.is_empty() {
        return Ok(false);
    }

    eprintln!(
        "[comprehension] popping {} file(s), asking LLM for comprehension...",
        to_pop.len()
    );
    for file in &to_pop {
        eprintln!(
            "[comprehension]   {} ({} tokens)",
            file.relative.display(),
            file.tokens
        );
    }

    let mut all_entries: Vec<(String, i64, String)> = Vec::new();
    let mut summary = String::new();

    for file in &to_pop {
        let file_reference = format!("references/{}", file.relative.display());

        let prompt = format!(
            "Running summary so far:\n{summary}\n\n\
             File ({file_reference}):\n{}",
            file.content
        );

        let output: ComprehensionOutput =
            once::run_completion_typed(config, COMPREHENSION_PREAMBLE, &prompt).await?;

        summary = output.summary;

        for entry in output.comprehensions {
            all_entries.push((entry.text, entry.timestamp, file_reference.clone()));
        }
    }

    let db_path = config.comprehensions_dir();
    write_to_lancedb(&db_path, all_entries).await?;
    eprintln!(
        "[comprehension] wrote comprehensions to LanceDB: {}",
        db_path.display()
    );

    move_to_references(config, &to_pop)?;
    eprintln!(
        "[comprehension] moved {} file(s) to references/",
        to_pop.len()
    );

    Ok(true)
}
