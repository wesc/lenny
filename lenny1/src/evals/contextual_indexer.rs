use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::Config;
use crate::contextual_indexer;

use crate::contextual_indexer::RetrieveResult;

enum Check {
    /// At least 2 of these files should appear in the retrieved window content.
    ContainsCluster(&'static [&'static str]),
    /// At most 1 of these files may appear in window content.
    ExcludesCluster(&'static [&'static str]),
    /// At least 2 of these doc names should be among the matched rows.
    MatchedContains(&'static [&'static str]),
    /// None of these doc names should be among the matched rows.
    MatchedExcludes(&'static [&'static str]),
}

struct Eval {
    name: &'static str,
    query: &'static str,
    top_k: usize,
    time_range: Option<(&'static str, &'static str)>,
    check: Check,
}

const EVALS: &[Eval] = &[
    // --- positive: correct cluster is retrieved ---
    Eval {
        name: "retrieve_programming",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["01", "02", "03", "04", "05"]),
    },
    Eval {
        name: "retrieve_cooking",
        query: "how to make pasta from scratch",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["06", "07", "08", "09", "10"]),
    },
    Eval {
        name: "retrieve_travel",
        query: "best places to visit in japan",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["11", "12", "13", "14", "15"]),
    },
    Eval {
        name: "retrieve_music",
        query: "favorite albums and concerts",
        top_k: 5,
        time_range: None,
        check: Check::ContainsCluster(&["16", "17", "18", "19", "20"]),
    },
    // --- negative: top-2 results for one topic exclude distant clusters ---
    // Each negative test uses the most distant cluster to avoid window bleed
    // at adjacent cluster boundaries.
    Eval {
        name: "programming_excludes_music",
        query: "rust borrow checker errors",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["16", "17", "18", "19", "20"]),
    },
    Eval {
        name: "cooking_excludes_music",
        query: "how to make pasta from scratch",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["16", "17", "18", "19", "20"]),
    },
    Eval {
        name: "travel_excludes_programming",
        query: "best places to visit in japan",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["01", "02", "03", "04", "05"]),
    },
    Eval {
        name: "music_excludes_programming",
        query: "favorite albums and concerts",
        top_k: 2,
        time_range: None,
        check: Check::ExcludesCluster(&["01", "02", "03", "04", "05"]),
    },
    // --- time-range filtering (checks matched rows, not window content) ---
    // Programming files have timestamps 2025-11-01 to 2025-11-03.
    // Filtering to Nov 1 only should match programming docs from that day.
    Eval {
        name: "time_filter_programming_nov1",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: Some(("2025-11-01T00:00:00Z", "2025-11-01T23:59:59Z")),
        check: Check::MatchedContains(&["01", "02"]),
    },
    // Filtering to Nov 8+ should NOT match any programming docs.
    Eval {
        name: "time_filter_excludes_early",
        query: "rust borrow checker errors",
        top_k: 5,
        time_range: Some(("2025-11-08T00:00:00Z", "2025-11-10T23:59:59Z")),
        check: Check::MatchedExcludes(&["01", "02", "03", "04", "05"]),
    },
    // Music docs are Nov 8-10. Filtering to that range should match them.
    Eval {
        name: "time_filter_music_late",
        query: "favorite albums and concerts",
        top_k: 5,
        time_range: Some(("2025-11-08T00:00:00Z", "2025-11-10T23:59:59Z")),
        check: Check::MatchedContains(&["16", "17", "18", "19", "20"]),
    },
    // Filtering to Nov 1-2 for cooking query should NOT match any cooking docs
    // (cooking is Nov 3-5).
    Eval {
        name: "time_filter_cooking_too_early",
        query: "how to make pasta from scratch",
        top_k: 5,
        time_range: Some(("2025-11-01T00:00:00Z", "2025-11-02T23:59:59Z")),
        check: Check::MatchedExcludes(&["06", "07", "08", "09", "10"]),
    },
];

/// Load all fixture NDJSON files as (name, content, timestamp) tuples.
/// Timestamp is extracted from the first message's `timestamp` field.
fn load_fixture_documents() -> Result<Vec<(String, String, String)>> {
    let fixture_dir = PathBuf::from("tests/fixtures/chat_data");
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

        // Extract timestamp from the first NDJSON line
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

/// Check for a document name header (e.g. "## 01\n") as produced by flatten_windows.
fn has_doc(text: &str, name: &str) -> bool {
    text.contains(&format!("## {name}\n"))
}

fn run_check(result: &RetrieveResult, check: &Check) -> (bool, String) {
    match check {
        Check::ContainsCluster(expected) => {
            let found: Vec<&&str> = expected
                .iter()
                .filter(|f| has_doc(&result.context, f))
                .collect();
            let pass = found.len() >= 2;
            let reason = if pass {
                format!("found {}/{} expected files", found.len(), expected.len())
            } else {
                format!(
                    "only found {}/{} expected files: {:?}",
                    found.len(),
                    expected.len(),
                    found
                )
            };
            (pass, reason)
        }
        Check::ExcludesCluster(excluded) => {
            let found: Vec<&&str> = excluded
                .iter()
                .filter(|f| has_doc(&result.context, f))
                .collect();
            let pass = found.len() <= 1;
            let reason = if pass {
                format!(
                    "correctly excluded cluster ({}/{} found)",
                    found.len(),
                    excluded.len()
                )
            } else {
                format!(
                    "wrong cluster leaked: {}/{} found: {:?}",
                    found.len(),
                    excluded.len(),
                    found
                )
            };
            (pass, reason)
        }
        Check::MatchedContains(expected) => {
            let found: Vec<&&str> = expected
                .iter()
                .filter(|f| result.matched_docs.iter().any(|d| d == **f))
                .collect();
            let pass = found.len() >= 2;
            let reason = if pass {
                format!(
                    "matched {}/{} expected docs (matched: {:?})",
                    found.len(),
                    expected.len(),
                    result.matched_docs
                )
            } else {
                format!(
                    "only matched {}/{} expected docs (matched: {:?})",
                    found.len(),
                    expected.len(),
                    result.matched_docs
                )
            };
            (pass, reason)
        }
        Check::MatchedExcludes(excluded) => {
            let found: Vec<&&str> = excluded
                .iter()
                .filter(|f| result.matched_docs.iter().any(|d| d == **f))
                .collect();
            let pass = found.is_empty();
            let reason = if pass {
                format!(
                    "correctly excluded docs (matched: {:?})",
                    result.matched_docs
                )
            } else {
                format!(
                    "wrong docs matched: {:?} (all matched: {:?})",
                    found, result.matched_docs
                )
            };
            (pass, reason)
        }
    }
}

pub async fn run(base_config: &Config) -> Result<()> {
    let docs_with_ts = load_fixture_documents()?;
    let documents: Vec<(String, String)> = docs_with_ts
        .iter()
        .map(|(n, c, _)| (n.clone(), c.clone()))
        .collect();
    let timestamps: Vec<String> = docs_with_ts.iter().map(|(_, _, t)| t.clone()).collect();

    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("test-index");

    eprintln!(
        "  Building contextual index ({} documents)...",
        documents.len()
    );
    let build_start = Instant::now();
    contextual_indexer::build_index(&db_path, &documents, &timestamps, base_config, 5).await?;
    let build_elapsed = build_start.elapsed().as_secs_f64();
    eprintln!("  Index built in {build_elapsed:.1}s\n");

    let mut results = Vec::new();
    let mut passed = 0;
    let total = EVALS.len();
    let run_start = Instant::now();

    for eval in EVALS {
        eprint!("  {} ... ", eval.name);

        let eval_start = Instant::now();
        let outcome =
            match contextual_indexer::retrieve(&db_path, eval.query, eval.top_k, eval.time_range)
                .await
            {
                Ok(result) => {
                    let elapsed = eval_start.elapsed().as_secs_f64();
                    let (pass, reason) = run_check(&result, &eval.check);
                    if pass {
                        passed += 1;
                    }
                    eprintln!("{} ({:.1}s)", if pass { "PASS" } else { "FAIL" }, elapsed);
                    json!({
                        "name": eval.name,
                        "query": eval.query,
                        "pass": pass,
                        "reason": reason,
                        "elapsed_s": (elapsed * 10.0).round() / 10.0,
                    })
                }
                Err(e) => {
                    let elapsed = eval_start.elapsed().as_secs_f64();
                    eprintln!("ERROR ({elapsed:.1}s)");
                    json!({
                        "name": eval.name,
                        "query": eval.query,
                        "pass": false,
                        "reason": format!("error: {e}"),
                        "elapsed_s": (elapsed * 10.0).round() / 10.0,
                    })
                }
            };

        results.push(outcome);
    }

    let total_elapsed = run_start.elapsed().as_secs_f64();
    eprintln!("\n  {passed}/{total} passed in {total_elapsed:.1}s");

    let output = json!({
        "passed": passed,
        "total": total,
        "build_elapsed_s": (build_elapsed * 10.0).round() / 10.0,
        "elapsed_s": (total_elapsed * 10.0).round() / 10.0,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
