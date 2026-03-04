pub mod contextual_indexer;

use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::Config;
use crate::once::{self, PromptResult};
use crate::tools::AgentEvent;

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
        thinking: base_config.thinking,
        system_dir: fixtures.join("system"),
        dynamic_dir: fixtures.join("dynamic"),
        knowledge_dir: tmpdir.path().to_path_buf(),
        max_context_tokens: base_config.max_context_tokens,
        min_context_tokens: base_config.min_context_tokens,
        matrix: None,
    };
    Ok((config, tmpdir))
}

pub async fn run(base_config: &Config) -> Result<()> {
    let (config, _tmpdir) = eval_config(base_config)?;

    let mut results = Vec::new();
    let mut passed = 0;
    let total = EVALS.len();
    let run_start = Instant::now();

    for eval in EVALS {
        eprint!("  {} ... ", eval.name);

        let eval_start = Instant::now();
        let outcome = match once::run_prompt(&config, eval.prompt).await {
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

    let output = json!({
        "passed": passed,
        "total": total,
        "elapsed_s": (total_elapsed * 10.0).round() / 10.0,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
