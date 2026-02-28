use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::path::PathBuf;

use crate::config::Config;
use crate::once;
use crate::tools::AgentEvent;

struct Eval {
    name: &'static str,
    prompt: &'static str,
    check: fn(&str, &[AgentEvent]) -> (bool, String),
}

fn check_arithmetic(answer: &str, _events: &[AgentEvent]) -> (bool, String) {
    let pass = answer.contains('4');
    let reason = if pass {
        "answer contains '4'".to_string()
    } else {
        format!("expected '4' in answer, got: {answer}")
    };
    (pass, reason)
}

fn check_favorite_color(answer: &str, _events: &[AgentEvent]) -> (bool, String) {
    let pass = answer.to_lowercase().contains("chartreuse");
    let reason = if pass {
        "answer contains 'chartreuse'".to_string()
    } else {
        format!("expected 'chartreuse' in answer, got: {answer}")
    };
    (pass, reason)
}

fn check_favorite_car(answer: &str, _events: &[AgentEvent]) -> (bool, String) {
    let lower = answer.to_lowercase();
    let pass = lower.contains("toyota") && lower.contains("pineapple");
    let reason = if pass {
        "answer contains 'Toyota Pineapple'".to_string()
    } else {
        format!("expected 'Toyota Pineapple' in answer, got: {answer}")
    };
    (pass, reason)
}

fn check_random_number(answer: &str, events: &[AgentEvent]) -> (bool, String) {
    let used_tool = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == "random_number"));
    if !used_tool {
        return (false, "did not call random_number tool".to_string());
    }
    let has_digit = answer.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return (false, format!("answer has no digit: {answer}"));
    }
    (
        true,
        "called random_number tool and answer contains a digit".to_string(),
    )
}

fn check_random_letter(answer: &str, events: &[AgentEvent]) -> (bool, String) {
    let used_tool = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == "random_letter"));
    if !used_tool {
        return (false, "did not call random_letter tool".to_string());
    }
    let re = Regex::new(r"[a-z]").unwrap();
    if !re.is_match(answer) {
        return (false, format!("answer has no lowercase letter: {answer}"));
    }
    (
        true,
        "called random_letter tool and answer contains a letter".to_string(),
    )
}

fn check_multi_tool(answer: &str, events: &[AgentEvent]) -> (bool, String) {
    let used_number = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == "random_number"));
    let used_letter = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolCall { tool, .. } if tool == "random_letter"));
    if !used_number || !used_letter {
        return (
            false,
            format!("expected both tools called (number={used_number}, letter={used_letter})"),
        );
    }
    let has_digit = answer.chars().any(|c| c.is_ascii_digit());
    let has_letter = answer.chars().any(|c| c.is_ascii_lowercase());
    let pass = has_digit && has_letter;
    let reason = if pass {
        "called both tools, answer has digit + letter".to_string()
    } else {
        format!("answer missing digit or letter: {answer}")
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
];

/// Build a Config that points at the eval fixtures, with a temp references dir.
fn eval_config(base_config: &Config) -> Result<(Config, tempfile::TempDir)> {
    let fixtures = PathBuf::from("evals/fixtures/basic");
    let tmpdir = tempfile::tempdir()?;

    let config = Config {
        model: base_config.model.clone(),
        max_iterations: base_config.max_iterations,
        thinking: base_config.thinking,
        system_dir: fixtures.join("system"),
        dynamic_dir: fixtures.join("dynamic"),
        references_dir: tmpdir.path().to_path_buf(),
        ollama_url: base_config.ollama_url.clone(),
    };
    Ok((config, tmpdir))
}

pub async fn run(base_config: &Config) -> Result<()> {
    let (config, _tmpdir) = eval_config(base_config)?;

    let mut results = Vec::new();
    let mut passed = 0;
    let total = EVALS.len();

    for eval in EVALS {
        eprint!("  {} ... ", eval.name);

        let outcome = match once::run_prompt(&config, eval.prompt).await {
            Ok(result) => {
                let (pass, reason) = (eval.check)(&result.answer, &result.events);
                if pass {
                    passed += 1;
                }
                eprintln!("{}", if pass { "PASS" } else { "FAIL" });
                json!({
                    "name": eval.name,
                    "prompt": eval.prompt,
                    "answer": result.answer,
                    "pass": pass,
                    "reason": reason,
                    "events": result.events.len(),
                })
            }
            Err(e) => {
                eprintln!("ERROR");
                json!({
                    "name": eval.name,
                    "prompt": eval.prompt,
                    "answer": null,
                    "pass": false,
                    "reason": format!("error: {e}"),
                    "events": 0,
                })
            }
        };

        results.push(outcome);
    }

    eprintln!("\n  {passed}/{total} passed");

    let output = json!({
        "passed": passed,
        "total": total,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
