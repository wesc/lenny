use anyhow::Result;
use std::path::PathBuf;

use crate::config::Config;

use super::{Check, ContextualEval};

const EVALS: &[ContextualEval] = &[
    // --- positive: correct cluster is retrieved ---
    ContextualEval {
        name: "retrieve_programming",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["01", "02", "03", "04", "05"]),
    },
    ContextualEval {
        name: "retrieve_cooking",
        query: "how to make pasta from scratch",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["06", "07", "08", "09", "10"]),
    },
    ContextualEval {
        name: "retrieve_travel",
        query: "best places to visit in japan",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["11", "12", "13", "14", "15"]),
    },
    ContextualEval {
        name: "retrieve_music",
        query: "favorite albums and concerts",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["16", "17", "18", "19", "20"]),
    },
    // --- negative: top-2 results for one topic exclude distant clusters ---
    ContextualEval {
        name: "programming_excludes_music",
        query: "rust borrow checker errors",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["16", "17", "18", "19", "20"]),
    },
    ContextualEval {
        name: "cooking_excludes_music",
        query: "how to make pasta from scratch",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["16", "17", "18", "19", "20"]),
    },
    ContextualEval {
        name: "travel_excludes_programming",
        query: "best places to visit in japan",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["01", "02", "03", "04", "05"]),
    },
    ContextualEval {
        name: "music_excludes_programming",
        query: "favorite albums and concerts",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["01", "02", "03", "04", "05"]),
    },
    // --- time-range filtering ---
    ContextualEval {
        name: "time_filter_programming_nov1",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: Some(("2025-11-01T00:00:00Z", "2025-11-01T23:59:59Z")),
        check: Check::MatchedContains(&["01", "02"]),
    },
    ContextualEval {
        name: "time_filter_excludes_early",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: Some(("2025-11-08T00:00:00Z", "2025-11-10T23:59:59Z")),
        check: Check::MatchedExcludes(&["01", "02", "03", "04", "05"]),
    },
    ContextualEval {
        name: "time_filter_music_late",
        query: "favorite albums and concerts",
        top_k: 5,
        time_range: Some(("2025-11-08T00:00:00Z", "2025-11-10T23:59:59Z")),
        check: Check::MatchedContains(&["16", "17", "18", "19", "20"]),
    },
    ContextualEval {
        name: "time_filter_cooking_too_early",
        query: "how to make pasta from scratch",
        top_k: 5,
        time_range: Some(("2025-11-01T00:00:00Z", "2025-11-02T23:59:59Z")),
        check: Check::MatchedExcludes(&["06", "07", "08", "09", "10"]),
    },
];

/// Load all fixture NDJSON files as (name, content, timestamp) tuples.
fn load_fixture_documents() -> Result<Vec<(String, String, String)>> {
    let fixture_dir = PathBuf::from("tests/fixtures/chat-data");
    let mut entries: Vec<_> = std::fs::read_dir(&fixture_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "ndjson"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut documents = Vec::new();
    for entry in entries {
        let name = entry
            .path()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let content = std::fs::read_to_string(entry.path())?;

        let timestamp = content
            .lines()
            .next()
            .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .and_then(|v| v["timestamp"].as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        documents.push((name, content, timestamp));
    }
    Ok(documents)
}

pub async fn run(base_config: &Config) -> Result<()> {
    let docs_with_ts = load_fixture_documents()?;
    let documents: Vec<(String, String)> = docs_with_ts
        .iter()
        .map(|(n, c, _)| (n.clone(), c.clone()))
        .collect();
    let timestamps: Vec<String> = docs_with_ts.iter().map(|(_, _, t)| t.clone()).collect();

    super::run_contextual_evals("chats", &documents, &timestamps, EVALS, base_config).await?;

    Ok(())
}
