use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::config::Config;
use crate::embed;
use crate::once;

const MAX_CHUNK_WORDS: usize = 750;

/// Split large documents into chunks on paragraph boundaries.
/// Documents under MAX_CHUNK_WORDS pass through unchanged.
/// Large documents get chunks named "name/001", "name/002", etc.
/// Each chunk inherits the parent's timestamp.
fn chunk_documents(
    documents: &[(String, String)],
    timestamps: &[String],
) -> (Vec<(String, String)>, Vec<String>) {
    let mut out_docs = Vec::new();
    let mut out_ts = Vec::new();

    for (i, (name, content)) in documents.iter().enumerate() {
        let ts = timestamps.get(i).cloned().unwrap_or_default();
        let word_count = content.split_whitespace().count();

        if word_count <= MAX_CHUNK_WORDS {
            out_docs.push((name.clone(), content.clone()));
            out_ts.push(ts);
            continue;
        }

        let normalized = content.replace("\r\n", "\n");
        let paragraphs: Vec<&str> = normalized.split("\n\n").collect();
        let mut chunk_text = String::new();
        let mut chunk_words = 0usize;
        let mut chunk_idx = 1u32;

        for para in &paragraphs {
            let para_words = para.split_whitespace().count();

            if chunk_words > 0 && chunk_words + para_words > MAX_CHUNK_WORDS {
                out_docs.push((format!("{name}/{chunk_idx:03}"), chunk_text.clone()));
                out_ts.push(ts.clone());
                chunk_idx += 1;
                chunk_text.clear();
                chunk_words = 0;
            }

            if !chunk_text.is_empty() {
                chunk_text.push_str("\n\n");
            }
            chunk_text.push_str(para);
            chunk_words += para_words;
        }

        if !chunk_text.is_empty() {
            out_docs.push((format!("{name}/{chunk_idx:03}"), chunk_text));
            out_ts.push(ts.clone());
        }
    }

    (out_docs, out_ts)
}

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
2. \"enrichment\": a contextual enrichment (1-2 sentences) stating the specific topics, \
arguments, and key terms in the target document. Write directly about the content \
(e.g. \"Paine argues hereditary monarchy violates natural rights\"), never use \
meta-language like \"this document discusses\" or \"within the surrounding documents\".

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
/// and store in sqlite-vec with an optional timestamp per document.
pub async fn build_index(
    db_path: &Path,
    documents: &[(String, String)],
    timestamps: &[String],
    config: &Config,
    window_size: usize,
) -> Result<()> {
    let (documents, timestamps) = chunk_documents(documents, timestamps);

    let mut doc_names = Vec::new();
    let mut contents: Vec<String> = Vec::new();
    let mut enrichments = Vec::new();
    let mut window_docs_json = Vec::new();
    let mut ts_col = Vec::new();

    let mut summary = String::new();

    for (i, (name, content)) in documents.iter().enumerate() {
        let w = window(&documents, i, window_size);
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

        doc_names.push(name.clone());
        contents.push(content.clone());
        enrichments.push(output.enrichment);
        window_docs_json.push(window_json);
        ts_col.push(timestamps.get(i).cloned().unwrap_or_default());
    }

    // Batch-embed enrichments (synchronous, run on blocking thread)
    let enrichment_strs: Vec<String> = enrichments.clone();
    let embeddings =
        tokio::task::spawn_blocking(move || embed::embed_batch(enrichment_strs)).await??;

    // Write to sqlite-vec
    let db_path = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = embed::open_db(&db_path)?;

        conn.execute_batch(
            "DROP TABLE IF EXISTS entries_vec;
             DROP TABLE IF EXISTS entries;",
        )?;
        conn.execute_batch(
            "CREATE TABLE entries (
                doc_name TEXT NOT NULL,
                content TEXT NOT NULL,
                enrichment TEXT NOT NULL,
                window_docs TEXT NOT NULL,
                timestamp TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE entries_vec USING vec0(embedding float[384]);",
        )?;

        let tx = conn.unchecked_transaction()?;
        for i in 0..doc_names.len() {
            tx.execute(
                "INSERT INTO entries (rowid, doc_name, content, enrichment, window_docs, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    (i + 1) as i64,
                    doc_names[i],
                    contents[i],
                    enrichments[i],
                    window_docs_json[i],
                    ts_col[i],
                ],
            )?;
            tx.execute(
                "INSERT INTO entries_vec (rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![(i + 1) as i64, embed::f32_to_bytes(&embeddings[i])],
            )?;
        }
        tx.commit()?;

        Ok(())
    })
    .await??;

    Ok(())
}

/// One candidate's rerank stats for debugging.
pub struct RerankStat {
    pub doc_name: String,
    pub enrichment_prefix: String,
    pub score: f32,
}

/// Result of a retrieval query.
pub struct RetrieveResult {
    /// Doc names of the rows that matched the vector search.
    pub matched_docs: Vec<String>,
    /// Deduplicated, flattened window contents.
    pub context: String,
    /// Rerank scores for all candidates (sorted by score descending).
    pub rerank_stats: Vec<RerankStat>,
    /// Total candidates after time filter (before gap filter).
    pub candidates_considered: usize,
    /// candidates_considered - matched_docs.len()
    pub candidates_filtered: usize,
}

/// Retrieve relevant context for a query, optionally filtered by time range.
/// `time_range` is `Some((start, end))` where both are ISO 8601 strings for inclusive filtering.
pub async fn retrieve(
    db_path: &Path,
    query: &str,
    top_k: usize,
    time_range: Option<(&str, &str)>,
    min_score_range: f32,
    score_gap_threshold: f32,
) -> Result<RetrieveResult> {
    if !db_path.exists() {
        return Ok(RetrieveResult {
            matched_docs: Vec::new(),
            context: String::new(),
            rerank_stats: Vec::new(),
            candidates_considered: 0,
            candidates_filtered: 0,
        });
    }

    let query = query.to_string();
    let db_path = db_path.to_path_buf();
    let time_range = time_range.map(|(s, e)| (s.to_string(), e.to_string()));

    tokio::task::spawn_blocking(move || -> Result<RetrieveResult> {
        let query_vec = embed::embed_query(&query)?;

        let conn = embed::open_db(&db_path)?;

        // Check if table exists
        let table_exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries'")?
            .exists([])?;
        if !table_exists {
            return Ok(RetrieveResult {
                matched_docs: Vec::new(),
                context: String::new(),
                rerank_stats: Vec::new(),
                candidates_considered: 0,
                candidates_filtered: 0,
            });
        }

        // Always over-fetch for reranking
        let fetch_limit = top_k * 5;

        let mut stmt = conn.prepare(
            "SELECT e.doc_name, e.window_docs, e.timestamp, e.content
             FROM entries_vec v
             JOIN entries e ON e.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let query_bytes = embed::f32_to_bytes(&query_vec);

        struct Row {
            doc_name: String,
            window_docs: String,
            timestamp: String,
            content: String,
        }

        let rows: Vec<Row> = stmt
            .query_map(rusqlite::params![query_bytes, fetch_limit as i64], |row| {
                Ok(Row {
                    doc_name: row.get(0)?,
                    window_docs: row.get(1)?,
                    timestamp: row.get(2)?,
                    content: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Apply time filter in Rust
        let filtered: Vec<&Row> = if let Some((start, end)) = &time_range {
            rows.iter()
                .filter(|r| {
                    r.timestamp.as_str() >= start.as_str() && r.timestamp.as_str() <= end.as_str()
                })
                .collect()
        } else {
            rows.iter().collect()
        };

        // Rerank by document content
        let enrichments: Vec<&str> = filtered.iter().map(|r| r.content.as_str()).collect();
        let reranked = if enrichments.is_empty() {
            Vec::new()
        } else {
            embed::rerank(&query, &enrichments)?
        };

        // Build rerank stats for all candidates
        let rerank_stats: Vec<RerankStat> = reranked
            .iter()
            .map(|r| {
                let row = filtered[r.index];
                let prefix: String = row.content.chars().take(20).collect();
                RerankStat {
                    doc_name: row.doc_name.clone(),
                    enrichment_prefix: prefix,
                    score: r.score,
                }
            })
            .collect();

        // Range gate + gap trim
        let candidates_considered = filtered.len();
        let score_range = if reranked.is_empty() {
            0.0
        } else {
            reranked[0].score - reranked[reranked.len() - 1].score
        };

        let search_depth = reranked.len().min(top_k + 1);
        let cutoff = if score_range < min_score_range {
            // Tight cluster → all noise → return nothing
            0
        } else if reranked.len() <= top_k {
            // Few candidates survived filtering; return all (nothing to trim)
            reranked.len()
        } else {
            // Signal exists: scan for gap, else return top_k
            let mut c = search_depth.min(top_k);
            for i in 1..search_depth {
                let gap = reranked[i - 1].score - reranked[i].score;
                if gap >= score_gap_threshold {
                    c = i;
                    break;
                }
            }
            c
        };
        let top_indices: Vec<usize> = reranked[..cutoff].iter().map(|r| r.index).collect();

        let mut matched_docs = Vec::new();
        let mut windows: Vec<Vec<(String, String)>> = Vec::new();

        for idx in &top_indices {
            let row = filtered[*idx];
            matched_docs.push(row.doc_name.clone());

            let entries: Vec<serde_json::Value> = serde_json::from_str(&row.window_docs)?;
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

        let context = if windows.is_empty() {
            String::new()
        } else {
            flatten_windows(windows)
        };

        let candidates_filtered = candidates_considered - matched_docs.len();
        Ok(RetrieveResult {
            matched_docs,
            context,
            rerank_stats,
            candidates_considered,
            candidates_filtered,
        })
    })
    .await?
}
