use anyhow::Result;
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

/// Collect all NDJSON lines from references/chats/*.json,
/// filter to last 7 days, keep most recent up to ~1000 tokens,
/// write to dynamic/00-session.json.
pub fn run(config: &Config) -> Result<bool> {
    let chats_dir = config.references_dir.join("chats");
    if !chats_dir.exists() {
        return Ok(false);
    }

    let now = chrono::Utc::now().timestamp();
    let cutoff = now - MAX_AGE_SECS;

    // Collect all utterance lines with their timestamps
    let mut utterances: Vec<(i64, String)> = Vec::new();

    for entry in fs::read_dir(&chats_dir)? {
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
            // Parse just enough to extract the timestamp
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                let ts = extract_timestamp(&obj);
                if ts >= cutoff {
                    utterances.push((ts, line.to_string()));
                }
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

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
}

fn extract_timestamp(obj: &serde_json::Value) -> i64 {
    // Accept "timestamp" as either a unix int or a float
    if let Some(ts) = obj.get("timestamp") {
        if let Some(n) = ts.as_i64() {
            return n;
        }
        if let Some(n) = ts.as_f64() {
            return n as i64;
        }
    }
    0
}
