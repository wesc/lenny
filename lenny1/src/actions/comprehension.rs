use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::context;
use crate::embed;
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

/// Write comprehension entries to sqlite-vec DB.
pub async fn write_to_db(
    db_path: &Path,
    entries: Vec<(String, i64, String)>, // (summary, timestamp, file_reference)
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    // If db_path is a directory (old lancedb), remove it first
    if db_path.is_dir() {
        fs::remove_dir_all(db_path)?;
    }

    let summary_strs: Vec<String> = entries.iter().map(|(s, _, _)| s.clone()).collect();

    // Batch-embed summaries
    let embeddings =
        tokio::task::spawn_blocking(move || embed::embed_batch(summary_strs)).await??;

    let db_path = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = embed::open_db(&db_path)?;

        // Create tables if not exist
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS comprehensions (
                summary TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                file_reference TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS comprehensions_vec USING vec0(embedding float[384]);",
        )?;

        // Find max rowid to continue from
        let max_rowid: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(rowid), 0) FROM comprehensions",
                [],
                |row| row.get(0),
            )?;

        let tx = conn.unchecked_transaction()?;
        for (i, (summary, ts, file_ref)) in entries.iter().enumerate() {
            let rowid = max_rowid + (i as i64) + 1;
            tx.execute(
                "INSERT INTO comprehensions (rowid, summary, timestamp, file_reference)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![rowid, summary, ts, file_ref],
            )?;
            tx.execute(
                "INSERT INTO comprehensions_vec (rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![rowid, embed::f32_to_bytes(&embeddings[i])],
            )?;
        }
        tx.commit()?;

        Ok(())
    })
    .await??;

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
    if !db_path.exists() || db_path.is_dir() {
        return Ok(RetrieveResult {
            context: String::new(),
        });
    }

    let query = query.to_string();
    let db_path = db_path.to_path_buf();

    tokio::task::spawn_blocking(move || -> Result<RetrieveResult> {
        let query_vec = embed::embed_query(&query)?;

        let conn = embed::open_db(&db_path)?;

        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='comprehensions'")?
            .exists([])?;
        if !table_exists {
            return Ok(RetrieveResult {
                context: String::new(),
            });
        }

        let fetch_limit = if time_range.is_some() {
            top_k * 5
        } else {
            top_k
        };

        let mut stmt = conn.prepare(
            "SELECT e.summary, e.timestamp, e.file_reference
             FROM comprehensions_vec v
             JOIN comprehensions e ON e.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let query_bytes = embed::f32_to_bytes(&query_vec);

        struct Row {
            summary: String,
            timestamp: i64,
            file_reference: String,
        }

        let rows: Vec<Row> = stmt
            .query_map(rusqlite::params![query_bytes, fetch_limit as i64], |row| {
                Ok(Row {
                    summary: row.get(0)?,
                    timestamp: row.get(1)?,
                    file_reference: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let filtered: Vec<&Row> = if let Some((start, end)) = time_range {
            rows.iter()
                .filter(|r| r.timestamp >= start && r.timestamp <= end)
                .take(top_k)
                .collect()
        } else {
            rows.iter().collect()
        };

        let entries: Vec<String> = filtered
            .iter()
            .map(|r| format!("[{} @ {}] {}", r.file_reference, r.timestamp, r.summary))
            .collect();

        let context = entries.join("\n\n");

        Ok(RetrieveResult { context })
    })
    .await?
}

/// Dump all comprehension entries as JSON to stdout.
pub async fn dump_json(config: &Config) -> Result<()> {
    let db_path = config.comprehensions_dir();

    if !db_path.exists() || db_path.is_dir() {
        return Ok(());
    }

    let db_path_owned = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = embed::open_db(&db_path_owned)?;

        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='comprehensions'")?
            .exists([])?;
        if !table_exists {
            return Ok(());
        }

        let mut stmt =
            conn.prepare("SELECT summary, timestamp, file_reference FROM comprehensions")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let entry = serde_json::json!({
                "summary": row.get_ref(0)?.as_str()?,
                "timestamp": row.get::<_, i64>(1)?,
                "file_reference": row.get_ref(2)?.as_str()?,
            });
            println!("{}", serde_json::to_string(&entry)?);
        }
        Ok(())
    })
    .await??;

    Ok(())
}

/// Search comprehensions by semantic similarity and print top N matches as JSON.
pub async fn search_json(config: &Config, query: &str, top_k: usize) -> Result<()> {
    let db_path = config.comprehensions_dir();

    if !db_path.exists() || db_path.is_dir() {
        return Ok(());
    }

    let query = query.to_string();
    let db_path_owned = db_path.to_path_buf();

    tokio::task::spawn_blocking(move || -> Result<()> {
        let query_vec = embed::embed_query(&query)?;
        let conn = embed::open_db(&db_path_owned)?;

        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='comprehensions'")?
            .exists([])?;
        if !table_exists {
            return Ok(());
        }

        let mut stmt = conn.prepare(
            "SELECT e.summary, e.timestamp, e.file_reference
             FROM comprehensions_vec v
             JOIN comprehensions e ON e.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let query_bytes = embed::f32_to_bytes(&query_vec);
        let mut rows = stmt.query(rusqlite::params![query_bytes, top_k as i64])?;
        while let Some(row) = rows.next()? {
            let entry = serde_json::json!({
                "summary": row.get_ref(0)?.as_str()?,
                "timestamp": row.get::<_, i64>(1)?,
                "file_reference": row.get_ref(2)?.as_str()?,
            });
            println!("{}", serde_json::to_string(&entry)?);
        }
        Ok(())
    })
    .await??;

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
    write_to_db(&db_path, all_entries).await?;
    eprintln!(
        "[comprehension] wrote comprehensions to {}",
        db_path.display()
    );

    move_to_references(config, &to_pop)?;
    eprintln!(
        "[comprehension] moved {} file(s) to references/",
        to_pop.len()
    );

    Ok(true)
}
