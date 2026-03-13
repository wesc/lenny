pub mod all;
pub mod contextual_all;
pub mod contextual_chats;
pub mod contextual_texts;
pub mod fact;

use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::Config;
use crate::contextual_indexer::RetrieveResult;
use crate::once::{self, PromptResult};
use crate::session::SessionId;
use crate::tools::AgentEvent;

// ---------------------------------------------------------------------------
// Shared types for contextual eval suites (chats + texts)
// ---------------------------------------------------------------------------

pub(crate) enum Check {
    /// At least 2 of these files should appear in the retrieved window content.
    ContainsCluster(&'static [&'static str]),
    /// At most 1 of these files may appear in window content.
    ExcludesCluster(&'static [&'static str]),
    /// At least 2 of these doc names should be among the matched rows.
    MatchedContains(&'static [&'static str]),
    /// None of these doc names should be among the matched rows.
    MatchedExcludes(&'static [&'static str]),
    /// At least one matched doc name starts with `prefix` and total matches >= `min`.
    MatchedPrefixContains(&'static str, usize),
    /// No matched doc name starts with `prefix`.
    MatchedPrefixExcludes(&'static str),
}

pub(crate) struct ContextualEval {
    pub name: &'static str,
    pub query: &'static str,
    pub top_k: usize,
    pub time_range: Option<(&'static str, &'static str)>,
    pub check: Check,
}

/// Check for a document name header (e.g. "## 01\n") as produced by flatten_windows.
pub(crate) fn has_doc(text: &str, name: &str) -> bool {
    text.contains(&format!("## {name}\n"))
}

pub(crate) fn run_check(result: &RetrieveResult, check: &Check) -> (bool, String) {
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
        Check::MatchedPrefixContains(prefix, min) => {
            let found: Vec<&String> = result
                .matched_docs
                .iter()
                .filter(|d| d.starts_with(prefix))
                .collect();
            let pass = found.len() >= *min;
            let reason = if pass {
                format!(
                    "matched {} docs with prefix '{}' (>= {}) (matched: {:?})",
                    found.len(),
                    prefix,
                    min,
                    result.matched_docs
                )
            } else {
                format!(
                    "only matched {} docs with prefix '{}' (need {}) (matched: {:?})",
                    found.len(),
                    prefix,
                    min,
                    result.matched_docs
                )
            };
            (pass, reason)
        }
        Check::MatchedPrefixExcludes(prefix) => {
            let found: Vec<&String> = result
                .matched_docs
                .iter()
                .filter(|d| d.starts_with(prefix))
                .collect();
            let pass = found.is_empty();
            let reason = if pass {
                format!(
                    "correctly excluded prefix '{}' (matched: {:?})",
                    prefix, result.matched_docs
                )
            } else {
                format!(
                    "wrong prefix '{}' matched: {:?} (all matched: {:?})",
                    prefix, found, result.matched_docs
                )
            };
            (pass, reason)
        }
    }
}

/// Run a contextual eval suite: build index, run retrieval evals, print results.
pub(crate) async fn run_contextual_evals(
    label: &str,
    documents: &[(String, String)],
    timestamps: &[String],
    evals: &[ContextualEval],
    config: &Config,
) -> Result<(usize, usize)> {
    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("test-index");

    eprintln!("  [{label}] model: {}", config.provider.display_short());
    eprintln!(
        "  [{label}] Building contextual index ({} documents)...",
        documents.len()
    );
    let build_start = Instant::now();
    crate::contextual_indexer::build_index(&db_path, documents, timestamps, config, 5).await?;
    let build_elapsed = build_start.elapsed().as_secs_f64();
    eprintln!("  [{label}] Index built in {build_elapsed:.1}s\n");

    let mut results = Vec::new();
    let mut passed = 0;
    let total = evals.len();
    let run_start = Instant::now();

    for eval in evals {
        eprint!("  {} ... ", eval.name);

        let eval_start = Instant::now();
        let outcome = match crate::contextual_indexer::retrieve(
            &db_path,
            eval.query,
            eval.top_k,
            eval.time_range,
            config.min_score_range,
            config.score_gap_threshold,
        )
        .await
        {
            Ok(result) => {
                let elapsed = eval_start.elapsed().as_secs_f64();
                let (pass, reason) = run_check(&result, &eval.check);
                if pass {
                    passed += 1;
                }
                eprintln!("{} ({:.1}s)", if pass { "PASS" } else { "FAIL" }, elapsed);
                if !pass {
                    eprintln!("    reason: {reason}");
                    eprintln!(
                        "    candidates: {} considered, {} filtered",
                        result.candidates_considered, result.candidates_filtered
                    );
                    if let (Some(first), Some(last)) =
                        (result.rerank_stats.first(), result.rerank_stats.last())
                    {
                        eprintln!(
                            "    score range: {:.4} (top {:.4}, bottom {:.4})",
                            first.score - last.score,
                            first.score,
                            last.score
                        );
                    }
                    eprintln!("    rerank stats (score | doc | content):");
                    for stat in &result.rerank_stats {
                        eprintln!(
                            "      {:.4} | {} | {}...",
                            stat.score, stat.doc_name, stat.enrichment_prefix
                        );
                    }
                }
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
    eprintln!("\n  [{label}] {passed}/{total} passed in {total_elapsed:.1}s");

    if passed < total {
        let failures: Vec<_> = results
            .into_iter()
            .filter(|r| !r["pass"].as_bool().unwrap_or(true))
            .collect();
        let output = json!({
            "suite": label,
            "passed": passed,
            "total": total,
            "build_elapsed_s": (build_elapsed * 10.0).round() / 10.0,
            "elapsed_s": (total_elapsed * 10.0).round() / 10.0,
            "failures": failures,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        anyhow::bail!("[{label}] {}/{total} evals failed", total - passed);
    }

    Ok((passed, total))
}

// ---------------------------------------------------------------------------
// Basic eval suite (unchanged)
// ---------------------------------------------------------------------------

struct Eval {
    name: &'static str,
    prompt: &'static str,
    check: fn(&PromptResult) -> (bool, String),
}

fn check_arithmetic(r: &PromptResult) -> (bool, String) {
    let pass = r.answer.contains('4');
    let reason = if pass {
        "answer contains '4'".to_string()
    } else {
        format!("expected '4' in answer, got: {}", r.answer)
    };
    (pass, reason)
}

fn check_favorite_color(r: &PromptResult) -> (bool, String) {
    let pass = r.answer.to_lowercase().contains("chartreuse");
    let reason = if pass {
        "answer contains 'chartreuse'".to_string()
    } else {
        format!("expected 'chartreuse' in answer, got: {}", r.answer)
    };
    (pass, reason)
}

fn check_favorite_car(r: &PromptResult) -> (bool, String) {
    let lower = r.answer.to_lowercase();
    let pass = lower.contains("toyota") && lower.contains("pineapple");
    let reason = if pass {
        "answer contains 'Toyota Pineapple'".to_string()
    } else {
        format!("expected 'Toyota Pineapple' in answer, got: {}", r.answer)
    };
    (pass, reason)
}

fn has_tool_call(events: &[AgentEvent], name: &str) -> bool {
    events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == name))
}

fn check_random_number(r: &PromptResult) -> (bool, String) {
    if !has_tool_call(&r.events, "random_number") {
        return (false, "did not call random_number tool".to_string());
    }
    let has_digit = r.answer.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return (false, format!("answer has no digit: {}", r.answer));
    }
    (
        true,
        "called random_number tool and answer contains a digit".to_string(),
    )
}

fn check_random_letter(r: &PromptResult) -> (bool, String) {
    if !has_tool_call(&r.events, "random_letter") {
        return (false, "did not call random_letter tool".to_string());
    }
    let re = Regex::new(r"[a-z]").unwrap();
    if !re.is_match(&r.answer) {
        return (
            false,
            format!("answer has no lowercase letter: {}", r.answer),
        );
    }
    (
        true,
        "called random_letter tool and answer contains a letter".to_string(),
    )
}

fn check_no_response(r: &PromptResult) -> (bool, String) {
    if !r.skipped {
        return (false, "no_response was not set".to_string());
    }
    (true, "no_response was set".to_string())
}

fn check_multi_tool(r: &PromptResult) -> (bool, String) {
    let used_number = has_tool_call(&r.events, "random_number");
    let used_letter = has_tool_call(&r.events, "random_letter");
    if !used_number || !used_letter {
        return (
            false,
            format!("expected both tools called (number={used_number}, letter={used_letter})"),
        );
    }
    let has_digit = r.answer.chars().any(|c| c.is_ascii_digit());
    let has_letter = r.answer.chars().any(|c| c.is_ascii_lowercase());
    let pass = has_digit && has_letter;
    let reason = if pass {
        "called both tools, answer has digit + letter".to_string()
    } else {
        format!("answer missing digit or letter: {}", r.answer)
    };
    (pass, reason)
}

const EVALS: &[Eval] = &[
    Eval {
        name: "arithmetic",
        prompt: "What is 2+2?",
        check: check_arithmetic,
    },
    Eval {
        name: "favorite_color",
        prompt: "What is your favorite color?",
        check: check_favorite_color,
    },
    Eval {
        name: "favorite_car",
        prompt: "What is your favorite car?",
        check: check_favorite_car,
    },
    Eval {
        name: "random_number",
        prompt: "Generate a random number between 1 and 100 using the random_number tool and tell me the result.",
        check: check_random_number,
    },
    Eval {
        name: "random_letter",
        prompt: "Generate a random letter using the random_letter tool and tell me what it is.",
        check: check_random_letter,
    },
    Eval {
        name: "multi_tool",
        prompt: "Use the random_letter tool to get a random letter and the random_number tool to get a random number between 1 and 9, then concatenate them into a single string like 'x7'. Tell me the result.",
        check: check_multi_tool,
    },
    Eval {
        name: "no_response",
        prompt: "Alice: hey Bob, did you see the game last night?\nBob: yeah it was great, what a finish",
        check: check_no_response,
    },
];

/// Build a Config that points at the eval fixtures, with a temp references dir.
fn eval_config(base_config: &Config) -> Result<(Config, tempfile::TempDir)> {
    let fixtures = PathBuf::from("tests/fixtures/basic-evals");
    let tmpdir = tempfile::tempdir()?;

    let config = Config {
        provider: base_config.provider.clone(),
        max_iterations: base_config.max_iterations,
        system_dir: fixtures.join("system"),
        dynamic_dir: fixtures.join("dynamic"),
        knowledge_dir: tmpdir.path().to_path_buf(),
        max_context_tokens: base_config.max_context_tokens,
        min_context_tokens: base_config.min_context_tokens,
        matrix: None,
        min_score_range: base_config.min_score_range,
        score_gap_threshold: base_config.score_gap_threshold,
        prompt_log: false,
        model_candidates: Vec::new(),
        firecrawl_api_key: None,
    };
    Ok((config, tmpdir))
}

pub async fn run(base_config: &Config) -> Result<()> {
    let (config, _tmpdir) = eval_config(base_config)?;

    eprintln!("  model: {}\n", config.provider.display_short());

    let mut results = Vec::new();
    let mut passed = 0;
    let total = EVALS.len();
    let run_start = Instant::now();

    for eval in EVALS {
        eprint!("  {} ... ", eval.name);

        let eval_start = Instant::now();
        let eval_session = SessionId::new("evals", eval.name);
        let outcome = match once::run_prompt(&config, &eval_session, eval.prompt).await {
            Ok(result) => {
                let elapsed = eval_start.elapsed().as_secs_f64();
                let (pass, reason) = (eval.check)(&result);
                if pass {
                    passed += 1;
                }
                eprintln!("{} ({:.1}s)", if pass { "PASS" } else { "FAIL" }, elapsed);
                json!({
                    "name": eval.name,
                    "prompt": eval.prompt,
                    "answer": result.answer,
                    "pass": pass,
                    "reason": reason,
                    "events": result.events.len(),
                    "elapsed_s": (elapsed * 10.0).round() / 10.0,
                })
            }
            Err(e) => {
                let elapsed = eval_start.elapsed().as_secs_f64();
                eprintln!("ERROR ({elapsed:.1}s)");
                json!({
                    "name": eval.name,
                    "prompt": eval.prompt,
                    "answer": null,
                    "pass": false,
                    "reason": format!("error: {e}"),
                    "events": 0,
                    "elapsed_s": (elapsed * 10.0).round() / 10.0,
                })
            }
        };

        results.push(outcome);
    }

    let total_elapsed = run_start.elapsed().as_secs_f64();
    eprintln!("\n  {passed}/{total} passed in {total_elapsed:.1}s");

    if passed < total {
        let failures: Vec<_> = results
            .into_iter()
            .filter(|r| !r["pass"].as_bool().unwrap_or(true))
            .collect();
        let output = json!({
            "passed": passed,
            "total": total,
            "elapsed_s": (total_elapsed * 10.0).round() / 10.0,
            "failures": failures,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        anyhow::bail!("{}/{total} evals failed", total - passed);
    }

    Ok(())
}
