use anyhow::Result;
use serde_json::json;
use std::time::Instant;

use crate::config::Config;
use crate::once::{self, PromptResult};
use crate::session::SessionId;
use crate::tools::{self, AgentEvent};

struct Eval {
    name: &'static str,
    prompt: &'static str,
    /// Channel to run against (default: "cli").
    channel: &'static str,
    check: fn(&PromptResult) -> (bool, String),
}

fn has_tool_call(events: &[AgentEvent], name: &str) -> bool {
    events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == name))
}

fn check_tool(r: &PromptResult, tool_name: &str) -> (bool, String) {
    if has_tool_call(&r.events, tool_name) {
        (true, format!("called {tool_name} tool"))
    } else {
        (false, format!("did not call {tool_name} tool"))
    }
}

fn check_random_number(r: &PromptResult) -> (bool, String) {
    check_tool(r, "random_number")
}
fn check_random_letter(r: &PromptResult) -> (bool, String) {
    check_tool(r, "random_letter")
}
fn check_context_search(r: &PromptResult) -> (bool, String) {
    check_tool(r, "context_search")
}
fn check_lookup_reference(r: &PromptResult) -> (bool, String) {
    check_tool(r, "lookup_reference")
}
fn check_extract_url(r: &PromptResult) -> (bool, String) {
    check_tool(r, "extract_url")
}
fn check_web_search(r: &PromptResult) -> (bool, String) {
    check_tool(r, "web_search")
}
fn check_extract_url_to_note(r: &PromptResult) -> (bool, String) {
    check_tool(r, "extract_url_to_note")
}
fn check_write_note(r: &PromptResult) -> (bool, String) {
    check_tool(r, "write_note")
}
fn check_read_note(r: &PromptResult) -> (bool, String) {
    check_tool(r, "read_note")
}
fn check_bluesky_trending(r: &PromptResult) -> (bool, String) {
    check_tool(r, "bluesky_trending")
}

fn check_multi_tool(r: &PromptResult) -> (bool, String) {
    let has_number = has_tool_call(&r.events, "random_number");
    let has_letter = has_tool_call(&r.events, "random_letter");
    if has_number && has_letter {
        (
            true,
            "called both random_number and random_letter".to_string(),
        )
    } else {
        (
            false,
            format!("expected both tools: random_number={has_number}, random_letter={has_letter}"),
        )
    }
}

fn check_no_response(r: &PromptResult) -> (bool, String) {
    if r.skipped {
        (true, "no_response was set".to_string())
    } else {
        (false, "no_response was not set".to_string())
    }
}

const EVALS: &[Eval] = &[
    Eval {
        name: "random_number",
        prompt: "Use random_number to pick a number between 1 and 100, then tell me the result.",
        channel: "cli",
        check: check_random_number,
    },
    Eval {
        name: "random_letter",
        prompt: "Use random_letter to generate a random letter and tell me what you got.",
        channel: "cli",
        check: check_random_letter,
    },
    Eval {
        name: "context_search",
        prompt: "Use context_search to look up what you know about cooking recipes.",
        channel: "cli",
        check: check_context_search,
    },
    Eval {
        name: "lookup_reference",
        prompt: "Use lookup_reference to read the file at turns/20260101-example.json.",
        channel: "cli",
        check: check_lookup_reference,
    },
    Eval {
        name: "web_search",
        prompt: "Use web_search to find information about the Rust programming language.",
        channel: "cli",
        check: check_web_search,
    },
    Eval {
        name: "extract_url",
        prompt: "Use extract_url to find out what https://example.com is about.",
        channel: "cli",
        check: check_extract_url,
    },
    Eval {
        name: "extract_url_to_note",
        prompt: "Use extract_url_to_note to extract the main points from https://example.com/article into a file called research.md.",
        channel: "cli",
        check: check_extract_url_to_note,
    },
    Eval {
        name: "write_note",
        prompt: "Use write_note to save a note called weather.md with the content 'Sunny and warm today.'",
        channel: "cli",
        check: check_write_note,
    },
    Eval {
        name: "read_note",
        prompt: "Use read_note to read the file named research.md and tell me what it says.",
        channel: "cli",
        check: check_read_note,
    },
    Eval {
        name: "bluesky_trending",
        prompt: "Use bluesky_trending to find trending links about artificial intelligence.",
        channel: "cli",
        check: check_bluesky_trending,
    },
    Eval {
        name: "multi_tool",
        prompt: "Use random_number to pick a number between 1 and 26, then use random_letter to pick a letter. Tell me both results.",
        channel: "cli",
        check: check_multi_tool,
    },
    Eval {
        name: "no_response",
        prompt: "Alice: hey Bob, did you see the game last night?\nBob: yeah it was great, what a finish",
        channel: "matrix",
        check: check_no_response,
    },
];

/// Build a Config with temp dirs so evals don't accumulate state.
fn eval_config(base_config: &Config) -> Result<(Config, tempfile::TempDir)> {
    let tmpdir = tempfile::tempdir()?;

    let config = Config {
        provider: base_config.provider.clone(),
        max_iterations: base_config.max_iterations,
        system_dir: base_config.system_dir.clone(),
        dynamic_dir: tmpdir.path().join("dynamic"),
        knowledge_dir: tmpdir.path().join("knowledge"),
        max_context_tokens: base_config.max_context_tokens,
        min_context_tokens: base_config.min_context_tokens,
        matrix: None,
        min_score_range: base_config.min_score_range,
        score_gap_threshold: base_config.score_gap_threshold,
        prompt_log: false,
        model_candidates: Vec::new(),
        firecrawl_api_key: None,
    };

    std::fs::create_dir_all(&config.dynamic_dir)?;
    std::fs::create_dir_all(&config.knowledge_dir)?;

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

        // no_response eval doesn't need mock tools (it tests response behavior)
        let outcome = if eval.channel == "matrix" {
            once::run_prompt(&config, eval.channel, &eval_session, eval.prompt, None).await
        } else {
            once::run_prompt_with_tools(
                &config,
                eval.channel,
                &eval_session,
                eval.prompt,
                tools::build_mock_tools(),
            )
            .await
        };

        let outcome = match outcome {
            Ok(result) => {
                let elapsed = eval_start.elapsed().as_secs_f64();
                let (pass, reason) = (eval.check)(&result);
                if pass {
                    passed += 1;
                }
                eprintln!("{} ({:.1}s)", if pass { "PASS" } else { "FAIL" }, elapsed);
                if !pass {
                    eprintln!("    answer: {}", result.answer);
                    let tool_names: Vec<String> = result
                        .events
                        .iter()
                        .filter_map(|e| {
                            if let AgentEvent::ToolCall { tool, .. } = e {
                                Some(tool.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    eprintln!("    tools called: {tool_names:?}");
                }
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
