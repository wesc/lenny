use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::context;
use crate::embed;
use crate::once;
use crate::tokens;

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

/// Expected filename format: `YYYY-MM-DD_HH-MM-SS_<id>.ext`
/// Returns the timestamp portion as an ISO 8601 string, or an error.
pub fn parse_filename_timestamp(filename: &str) -> Result<String> {
    // Strip extension to get stem
    let stem = filename
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(filename);

    // Need at least "YYYY-MM-DD_HH-MM-SS" = 19 chars
    if stem.len() < 19 {
        anyhow::bail!("filename too short for timestamp: {filename}");
    }

    let date_part = &stem[..10]; // YYYY-MM-DD
    let time_part = &stem[11..19]; // HH-MM-SS

    if stem.as_bytes()[10] != b'_' {
        anyhow::bail!("expected '_' at position 10 in filename: {filename}");
    }

    // Validate date
    chrono::NaiveDate::parse_from_str(date_part, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("bad date in filename {filename}: {e}"))?;

    // Convert HH-MM-SS to HH:MM:SS
    let time_colon = time_part.replace('-', ":");
    chrono::NaiveTime::parse_from_str(&time_colon, "%H:%M:%S")
        .map_err(|e| anyhow::anyhow!("bad time in filename {filename}: {e}"))?;

    Ok(format!("{date_part}T{time_colon}Z"))
}

/// File with its content and metadata.
pub struct DynamicFile {
    pub path: PathBuf,
    pub relative: PathBuf,
    pub content: String,
    pub tokens: usize,
    /// ISO 8601 timestamp extracted from the filename.
    pub created_at: String,
}

/// Collect files from dynamic/, read contents, compute token counts.
/// Only includes files matching the expected filename timestamp format.
pub fn collect_dynamic_files(config: &Config) -> Result<Vec<DynamicFile>> {
    let paths = context::collect_files(&config.dynamic_dir)?;
    let mut files = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(&config.dynamic_dir)
            .unwrap_or(&path)
            .to_path_buf();

        let filename = relative.file_name().and_then(|f| f.to_str()).unwrap_or("");

        let created_at = match parse_filename_timestamp(filename) {
            Ok(ts) => ts,
            Err(e) => {
                tracing::debug!("skipping {}: {e}", relative.display());
                continue;
            }
        };

        let content = fs::read_to_string(&path)?;
        let toks = tokens::count_tokens(&content);
        files.push(DynamicFile {
            path,
            relative,
            content,
            tokens: toks,
            created_at,
        });
    }
    Ok(files)
}

/// Select oldest files to pop so remaining tokens <= min_context_tokens.
/// Files are sorted by `created_at` (oldest first).
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

    // Sort oldest first by timestamp
    files.sort_by(|a, b| a.created_at.cmp(&b.created_at));

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

/// Structured output from the LLM for fact extraction.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FactOutput {
    /// Extracted facts from the file.
    pub facts: Vec<String>,
}

pub const FACT_PREAMBLE: &str = "\
You are a knowledge extraction assistant. You are extracting facts for an AI assistant's long-term memory.
You are given the content of a file being archived from the assistant's knowledge base.

When files use \"you\" or \"your\", they refer to the assistant, not a human user. \
For example, \"your favorite color is chartreuse\" means the assistant's favorite color is chartreuse.

Produce a JSON object with:
- \"facts\": an array of strings, each a self-contained factual statement extracted from the file.
  Use third person (e.g. \"The assistant's favorite car is the Toyota Pineapple\").
  Each fact should be independently understandable without needing the source file.
  Extract as many distinct facts as are warranted by the content. \
  If the file contains no meaningful facts, return an empty array.";

/// A single fact entry ready for DB insertion.
struct FactEntry {
    created_at: String,
    fact: String,
    file_reference: String,
    token_count: usize,
}

/// Write fact entries to sqlite-vec DB at the given path.
async fn write_to_db(db_path: &Path, entries: Vec<FactEntry>) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    let fact_strs: Vec<String> = entries.iter().map(|e| e.fact.clone()).collect();

    // Batch-embed facts
    let embeddings = tokio::task::spawn_blocking(move || embed::embed_batch(fact_strs)).await??;

    let db_path = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = embed::open_db(&db_path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
                created_at TEXT NOT NULL,
                fact TEXT NOT NULL,
                file_reference TEXT NOT NULL,
                token_count INTEGER NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[384]);",
        )?;

        let max_rowid: i64 =
            conn.query_row("SELECT COALESCE(MAX(rowid), 0) FROM facts", [], |row| {
                row.get(0)
            })?;

        let tx = conn.unchecked_transaction()?;
        for (i, entry) in entries.iter().enumerate() {
            let rowid = max_rowid + (i as i64) + 1;
            tx.execute(
                "INSERT INTO facts (rowid, created_at, fact, file_reference, token_count)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    rowid,
                    entry.created_at,
                    entry.fact,
                    entry.file_reference,
                    entry.token_count as i64,
                ],
            )?;
            tx.execute(
                "INSERT INTO facts_vec (rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![rowid, embed::f32_to_bytes(&embeddings[i])],
            )?;
        }
        tx.commit()?;

        Ok(())
    })
    .await??;

    Ok(())
}

/// Result of a retrieval query against the facts DB.
pub struct RetrieveResult {
    pub context: String,
}

/// Retrieve relevant facts for a query.
pub async fn retrieve(db_path: &Path, query: &str, top_k: usize) -> Result<RetrieveResult> {
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
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'")?
            .exists([])?;
        if !table_exists {
            return Ok(RetrieveResult {
                context: String::new(),
            });
        }

        let mut stmt = conn.prepare(
            "SELECT e.fact, e.created_at, e.file_reference
             FROM facts_vec v
             JOIN facts e ON e.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let query_bytes = embed::f32_to_bytes(&query_vec);

        let mut entries: Vec<String> = Vec::new();
        let mut rows = stmt.query(rusqlite::params![query_bytes, top_k as i64])?;
        while let Some(row) = rows.next()? {
            let fact: String = row.get(0)?;
            let created_at: String = row.get(1)?;
            let file_ref: String = row.get(2)?;
            entries.push(format!("[{file_ref} @ {created_at}] {fact}"));
        }

        let context = entries.join("\n\n");
        Ok(RetrieveResult { context })
    })
    .await?
}

/// Dump all fact entries as JSON to stdout.
pub async fn dump_json(config: &Config) -> Result<()> {
    let db_path = config.memory_db();

    if !db_path.exists() || db_path.is_dir() {
        return Ok(());
    }

    let db_path_owned = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = embed::open_db(&db_path_owned)?;

        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'")?
            .exists([])?;
        if !table_exists {
            return Ok(());
        }

        let mut stmt =
            conn.prepare("SELECT created_at, fact, file_reference, token_count FROM facts")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let entry = serde_json::json!({
                "created_at": row.get_ref(0)?.as_str()?,
                "fact": row.get_ref(1)?.as_str()?,
                "file_reference": row.get_ref(2)?.as_str()?,
                "token_count": row.get::<_, i64>(3)?,
            });
            println!("{}", serde_json::to_string(&entry)?);
        }
        Ok(())
    })
    .await??;

    Ok(())
}

/// Search facts by semantic similarity and print top N matches as JSON.
pub async fn search_json(config: &Config, query: &str, top_k: usize) -> Result<()> {
    let db_path = config.memory_db();

    if !db_path.exists() || db_path.is_dir() {
        return Ok(());
    }

    let query = query.to_string();
    let db_path_owned = db_path.to_path_buf();

    tokio::task::spawn_blocking(move || -> Result<()> {
        let query_vec = embed::embed_query(&query)?;
        let conn = embed::open_db(&db_path_owned)?;

        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'")?
            .exists([])?;
        if !table_exists {
            return Ok(());
        }

        let mut stmt = conn.prepare(
            "SELECT e.created_at, e.fact, e.file_reference, e.token_count
             FROM facts_vec v
             JOIN facts e ON e.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let query_bytes = embed::f32_to_bytes(&query_vec);
        let mut rows = stmt.query(rusqlite::params![query_bytes, top_k as i64])?;
        while let Some(row) = rows.next()? {
            let entry = serde_json::json!({
                "created_at": row.get_ref(0)?.as_str()?,
                "fact": row.get_ref(1)?.as_str()?,
                "file_reference": row.get_ref(2)?.as_str()?,
                "token_count": row.get::<_, i64>(3)?,
            });
            println!("{}", serde_json::to_string(&entry)?);
        }
        Ok(())
    })
    .await??;

    Ok(())
}

/// Digest a set of dynamic files: extract facts via LLM, embed, store in DB, move to references.
pub async fn digest(config: &Config, files: &[DynamicFile]) -> Result<usize> {
    if files.is_empty() {
        return Ok(0);
    }

    let db_path = config.memory_db();
    let mut all_entries: Vec<FactEntry> = Vec::new();

    for file in files {
        let file_reference = file.relative.display().to_string();
        let ref_token_count = tokens::count_tokens(&file.content);

        let prompt = format!("File ({file_reference}):\n{}", file.content);

        let output: FactOutput = once::run_completion_typed_with_model(
            config,
            config.provider.comprehension_model(),
            FACT_PREAMBLE,
            &prompt,
        )
        .await?;

        eprintln!(
            "[fact] {} -> {} fact(s)",
            file_reference,
            output.facts.len()
        );

        for fact in output.facts {
            all_entries.push(FactEntry {
                created_at: file.created_at.clone(),
                fact,
                file_reference: file_reference.clone(),
                token_count: ref_token_count,
            });
        }
    }

    let count = all_entries.len();
    write_to_db(&db_path, all_entries).await?;

    if count > 0 {
        eprintln!("[fact] wrote {count} fact(s) to {}", db_path.display());
    }

    move_to_references(config, files)?;
    eprintln!("[fact] moved {} file(s) to references/", files.len());

    Ok(count)
}

/// Run fact digest on dynamic/. Returns Ok(true) if any facts were extracted.
/// If `force` is true, skip the token threshold check and digest all files.
pub async fn run(config: &Config, force: bool) -> Result<bool> {
    let files = collect_dynamic_files(config)?;

    if files.is_empty() {
        return Ok(false);
    }

    let (to_pop, _keep) = if force {
        // Still sort oldest first even in force mode
        let mut all = files;
        all.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        (all, vec![])
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

    eprintln!("[fact] digesting {} file(s)...", to_pop.len());
    for file in &to_pop {
        eprintln!(
            "[fact]   {} ({} tokens, created {})",
            file.relative.display(),
            file.tokens,
            file.created_at,
        );
    }

    let count = digest(config, &to_pop).await?;
    Ok(count > 0)
}
