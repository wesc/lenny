use crate::datasets::{Dataset, DatasetKind, Eval};
use crate::runner::RunResult;
use crate::tools;

fn has_tool_call(result: &RunResult, name: &str) -> bool {
    result.tool_events.iter().any(|e| e.tool == name)
}

/// Get the result value returned by a specific tool call.
fn tool_result_for(result: &RunResult, name: &str) -> Option<String> {
    result
        .tool_events
        .iter()
        .find(|e| e.tool == name)
        .map(|e| e.result.clone())
}

fn check_random_number(r: &RunResult) -> (bool, String) {
    if !has_tool_call(r, "random_number") {
        return (false, "did not call random_number tool".into());
    }
    let tool_value = match tool_result_for(r, "random_number") {
        Some(v) => v,
        None => return (false, "no tool result found".into()),
    };
    if !r.answer.contains(&tool_value) {
        return (
            false,
            format!(
                "answer does not contain tool result '{tool_value}': {}",
                r.answer
            ),
        );
    }
    (
        true,
        format!("called random_number, answer contains tool result '{tool_value}'"),
    )
}

fn check_random_letter(r: &RunResult) -> (bool, String) {
    if !has_tool_call(r, "random_letter") {
        return (false, "did not call random_letter tool".into());
    }
    let tool_value = match tool_result_for(r, "random_letter") {
        Some(v) => v,
        None => return (false, "no tool result found".into()),
    };
    if !r.answer.contains(&tool_value) {
        return (
            false,
            format!(
                "answer does not contain tool result '{tool_value}': {}",
                r.answer
            ),
        );
    }
    (
        true,
        format!("called random_letter, answer contains tool result '{tool_value}'"),
    )
}

fn check_multi_tool(r: &RunResult) -> (bool, String) {
    let used_number = has_tool_call(r, "random_number");
    let used_letter = has_tool_call(r, "random_letter");
    if !used_number || !used_letter {
        return (
            false,
            format!("expected both tools (number={used_number}, letter={used_letter})"),
        );
    }
    let num_val = match tool_result_for(r, "random_number") {
        Some(v) => v,
        None => return (false, "no random_number result found".into()),
    };
    let letter_val = match tool_result_for(r, "random_letter") {
        Some(v) => v,
        None => return (false, "no random_letter result found".into()),
    };
    if !r.answer.contains(&num_val) {
        return (
            false,
            format!(
                "answer missing number tool result '{num_val}': {}",
                r.answer
            ),
        );
    }
    if !r.answer.contains(&letter_val) {
        return (
            false,
            format!(
                "answer missing letter tool result '{letter_val}': {}",
                r.answer
            ),
        );
    }
    (
        true,
        format!("both tools called, answer contains '{letter_val}' and '{num_val}'"),
    )
}

fn check_number_range(r: &RunResult) -> (bool, String) {
    if !has_tool_call(r, "random_number") {
        return (false, "did not call random_number tool".into());
    }
    // Check the tool was called with correct range args
    let mut correct_args = false;
    for event in &r.tool_events {
        if event.tool == "random_number" {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(&event.args) {
                let min = args["min"].as_i64().unwrap_or(-1);
                let max = args["max"].as_i64().unwrap_or(-1);
                if min == 50 && max == 60 {
                    correct_args = true;
                } else {
                    return (
                        false,
                        format!("wrong range: min={min}, max={max} (expected 50,60)"),
                    );
                }
            }
        }
    }
    if !correct_args {
        return (false, "could not verify tool args".into());
    }
    // Also check the answer contains the actual tool result
    let tool_value = match tool_result_for(r, "random_number") {
        Some(v) => v,
        None => return (false, "no tool result found".into()),
    };
    if !r.answer.contains(&tool_value) {
        return (
            false,
            format!(
                "answer does not contain tool result '{tool_value}': {}",
                r.answer
            ),
        );
    }
    (
        true,
        format!("correct range [50,60], answer contains tool result '{tool_value}'"),
    )
}

pub fn dataset() -> Dataset {
    Dataset {
        name: "tool_use",
        system: "You are a helpful assistant. Use the available tools when asked.",
        kind: DatasetKind::ToolUse {
            tools: tools::all_tools(),
            evals: vec![
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
                    name: "number_range",
                    prompt: "Generate a random number between 50 and 60 (inclusive) using the random_number tool. Tell me the result.",
                    check: check_number_range,
                },
            ],
        },
    }
}
