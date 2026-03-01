use anyhow::Result;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::config::Config;

const MAX_TOKENS: usize = 1000;
const MAX_AGE_SECS: i64 = 7 * 24 * 60 * 60; // 1 week

/// Approximate token count: split on whitespace, count words.
/// Roughly 1 token ≈ 0.75 words for English, but we use 1:1 for a conservative bound.
fn approx_tokens(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Collect all NDJSON lines from references/chats/*.json and
/// references/matrix/**/*.json, filter to last 7 days, keep most
/// recent up to ~1000 tokens, write to dynamic/00-session.json.
pub fn run(config: &Config) -> Result<bool> {
    let now = chrono::Utc::now().timestamp();
    let cutoff = now - MAX_AGE_SECS;

    let mut utterances: Vec<(i64, String)> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    // Read from references/chats/*.json
    let chats_dir = config.references_dir.join("chats");
    if chats_dir.exists() {
        collect_ndjson_lines(&chats_dir, false, cutoff, &mut seen_ids, &mut utterances)?;
    }

    // Read from references/matrix/**/*.json (host subdirectories)
    let matrix_dir = config.references_dir.join("matrix");
    if matrix_dir.exists() {
        for host_entry in fs::read_dir(&matrix_dir)? {
            let host_entry = host_entry?;
            let host_path = host_entry.path();
            if host_path.is_dir() && !is_hidden(&host_path) {
                collect_ndjson_lines(&host_path, false, cutoff, &mut seen_ids, &mut utterances)?;
            }
        }
    }

    if utterances.is_empty() {
        return Ok(false);
    }

    // Sort by timestamp ascending (oldest first)
    utterances.sort_by_key(|(ts, _)| *ts);

    // Keep most recent lines up to MAX_TOKENS.
    // Walk backwards from the end, accumulating tokens.
    let mut budget = MAX_TOKENS;
    let mut keep_start = utterances.len();
    for i in (0..utterances.len()).rev() {
        let line_tokens = approx_tokens(&utterances[i].1);
        if line_tokens > budget {
            break;
        }
        budget -= line_tokens;
        keep_start = i;
    }

    let selected = &utterances[keep_start..];

    // Build NDJSON output
    let output: String = selected
        .iter()
        .map(|(_, line)| line.as_str())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    // Check if content actually changed
    let dest = config.dynamic_dir.join("00-session.json");
    if dest.exists() {
        let existing = fs::read_to_string(&dest)?;
        if existing == output {
            return Ok(false);
        }
    }

    // Atomic write into dynamic/
    fs::create_dir_all(&config.dynamic_dir)?;
    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = config.dynamic_dir.join(&tmp_name);
    fs::write(&tmp_path, &output)?;
    fs::rename(&tmp_path, &dest)?;

    Ok(true)
}

/// Read all NDJSON *.json files from `dir` (non-recursive), collecting lines
/// whose normalized timestamp >= `cutoff`.
fn collect_ndjson_lines(
    dir: &Path,
    _recursive: bool,
    cutoff: i64,
    seen_ids: &mut HashSet<String>,
    out: &mut Vec<(i64, String)>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        if is_hidden(&path) {
            continue;
        }
        let content = fs::read_to_string(&path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                // De-duplicate by id: skip lines whose id we've already seen
                if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
                    if !seen_ids.insert(id.to_string()) {
                        continue;
                    }
                }
                let ts = extract_timestamp(&obj);
                if ts >= cutoff {
                    out.push((ts, line.to_string()));
                }
            }
        }
    }
    Ok(())
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
}

/// Millisecond threshold: timestamps above this are treated as milliseconds.
const MS_THRESHOLD: i64 = 10_000_000_000;

fn extract_timestamp(obj: &serde_json::Value) -> i64 {
    // Accept "timestamp" as either a unix int or a float
    if let Some(ts) = obj.get("timestamp") {
        if let Some(n) = ts.as_i64() {
            return normalize_timestamp(n);
        }
        if let Some(n) = ts.as_f64() {
            return normalize_timestamp(n as i64);
        }
    }
    0
}

/// If `ts` looks like milliseconds (> ~year 2286 in seconds), convert to seconds.
fn normalize_timestamp(ts: i64) -> i64 {
    if ts > MS_THRESHOLD { ts / 1000 } else { ts }
}
