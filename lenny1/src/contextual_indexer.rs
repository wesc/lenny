use anyhow::Result;
use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures_util::StreamExt;
use lancedb::embeddings::sentence_transformers::SentenceTransformersEmbeddings;
use lancedb::embeddings::{EmbeddingDefinition, EmbeddingFunction};
use lancedb::query::{ExecutableQuery, QueryBase};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::once;

/// Output from the fold step: running summary + contextual enrichment.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FoldOutput {
    /// Updated running summary incorporating this document.
    #[serde(alias = "updated_summary")]
    pub summary: String,
    /// Contextual enrichment: what this document is about in the context of the corpus.
    #[serde(alias = "contextual_enrichment")]
    pub enrichment: String,
}

const FOLD_PREAMBLE: &str = "\
You are a document analyst. Given a running summary of documents seen so far, \
a target document, and a window of surrounding documents, produce two things:
1. \"summary\": an updated summary (2-3 sentences) incorporating the new document.
2. \"enrichment\": a contextual enrichment (1-2 sentences) describing what the target \
document is specifically about, placed in context of the surrounding documents.

You MUST respond with a JSON object containing exactly these two fields:
{\"summary\": \"...\", \"enrichment\": \"...\"}
No other text, no markdown, just the JSON object.";

/// Return a slice of `documents` representing a window of `size` centered on index `i`.
pub fn window<T>(documents: &[T], i: usize, size: usize) -> &[T] {
    if documents.is_empty() {
        return &[];
    }
    let half = size / 2;
    let start = i.saturating_sub(half);
    let end = (start + size).min(documents.len());
    let start = end.saturating_sub(size);
    &documents[start..end]
}

/// Deduplicate window docs by name (first occurrence wins), concatenate contents.
pub fn flatten_windows(windows: Vec<Vec<(String, String)>>) -> String {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for window in windows {
        for (name, content) in window {
            if seen.insert(name.clone()) {
                result.push(format!("## {name}\n{content}"));
            }
        }
    }
    result.join("\n\n")
}

/// Build the contextual index. For each document, generate an enrichment via LLM
/// and store in LanceDB with an optional timestamp per document.
pub async fn build_index(
    db_path: &Path,
    documents: &[(String, String)],
    timestamps: &[String],
    config: &Config,
    window_size: usize,
) -> Result<()> {
    let embedding = Arc::new(SentenceTransformersEmbeddings::builder().build()?);

    let db = lancedb::connect(db_path.to_str().unwrap())
        .execute()
        .await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    let mut ids = Vec::new();
    let mut doc_names = Vec::new();
    let mut enrichments = Vec::new();
    let mut window_docs_json = Vec::new();
    let mut ts_col = Vec::new();

    let mut summary = String::new();

    for (i, (name, content)) in documents.iter().enumerate() {
        let w = window(documents, i, window_size);
        let window_entries: Vec<serde_json::Value> = w
            .iter()
            .map(|(n, c)| serde_json::json!({"name": n, "content": c}))
            .collect();
        let window_json = serde_json::to_string(&window_entries)?;

        let prompt = format!(
            "Running summary so far:\n{summary}\n\n\
             Target document ({name}):\n{content}\n\n\
             Surrounding documents:\n{window_json}"
        );

        let output: FoldOutput = once::run_completion_typed(config, FOLD_PREAMBLE, &prompt).await?;

        summary = output.summary;

        ids.push(i as i32);
        doc_names.push(name.clone());
        enrichments.push(output.enrichment);
        window_docs_json.push(window_json);
        ts_col.push(timestamps.get(i).cloned().unwrap_or_default());
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("doc_name", DataType::Utf8, false),
        Field::new("enrichment", DataType::Utf8, false),
        Field::new("window_docs", DataType::Utf8, false),
        Field::new("timestamp", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(doc_names)),
            Arc::new(StringArray::from(enrichments)),
            Arc::new(StringArray::from(window_docs_json)),
            Arc::new(StringArray::from(ts_col)),
        ],
    )?;

    let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

    db.create_table("entries", reader)
        .add_embedding(EmbeddingDefinition::new(
            "enrichment",
            "sentence-transformers",
            Some("enrichment_embedding"),
        ))?
        .execute()
        .await?;

    Ok(())
}

/// Result of a retrieval query.
pub struct RetrieveResult {
    /// Doc names of the rows that matched the vector search.
    pub matched_docs: Vec<String>,
    /// Deduplicated, flattened window contents.
    pub context: String,
}

/// Retrieve relevant context for a query, optionally filtered by time range.
/// `time_range` is `Some((start, end))` where both are ISO 8601 strings for inclusive filtering.
pub async fn retrieve(
    db_path: &Path,
    query: &str,
    top_k: usize,
    time_range: Option<(&str, &str)>,
) -> Result<RetrieveResult> {
    let db_uri = db_path.to_str().unwrap();

    if !db_path.exists() {
        return Ok(RetrieveResult {
            matched_docs: Vec::new(),
            context: String::new(),
        });
    }

    let embedding = Arc::new(SentenceTransformersEmbeddings::builder().build()?);

    let db = lancedb::connect(db_uri).execute().await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    let table = match db.open_table("entries").execute().await {
        Ok(t) => t,
        Err(_) => {
            return Ok(RetrieveResult {
                matched_docs: Vec::new(),
                context: String::new(),
            });
        }
    };

    let query_arr = Arc::new(StringArray::from_iter_values(std::iter::once(query)));
    let query_vector = embedding.compute_query_embeddings(query_arr)?;

    let mut search = table
        .vector_search(query_vector)?
        .column("enrichment_embedding")
        .limit(top_k);

    if let Some((start, end)) = time_range {
        search = search.only_if(format!("timestamp >= '{start}' AND timestamp <= '{end}'"));
    }

    let mut results = search.execute().await?;

    let mut matched_docs = Vec::new();
    let mut windows: Vec<Vec<(String, String)>> = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;

        // Collect matched doc names
        let doc_name_col = batch
            .column_by_name("doc_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for val in doc_name_col.iter().flatten() {
            matched_docs.push(val.to_string());
        }

        // Collect window docs
        let window_col = batch
            .column_by_name("window_docs")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for val in window_col.iter().flatten() {
            let entries: Vec<serde_json::Value> = serde_json::from_str(val)?;
            let parsed: Vec<(String, String)> = entries
                .into_iter()
                .map(|e| {
                    (
                        e["name"].as_str().unwrap_or("").to_string(),
                        e["content"].as_str().unwrap_or("").to_string(),
                    )
                })
                .collect();
            windows.push(parsed);
        }
    }

    let context = if windows.is_empty() {
        String::new()
    } else {
        flatten_windows(windows)
    };

    Ok(RetrieveResult {
        matched_docs,
        context,
    })
}
