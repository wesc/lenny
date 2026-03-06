use crate::datasets::{Dataset, DatasetKind, Eval};
use crate::runner::RunResult;
use crate::tools;

/// System prompt that mirrors lenny's: instructs the model to search first,
/// then produce JSON with the answer.
const SYSTEM: &str = "\
You are a helpful assistant with access to a knowledge base via the context_search tool.

IMPORTANT — Knowledge base procedure:
BEFORE answering, you MUST call context_search with a query derived from the user's message.
The only exceptions are purely procedural messages (e.g. \"hello\", \"thanks\") that clearly need no factual lookup.

After searching and using any other tools you need, respond with a JSON object:
{\"no_response\": bool, \"answer\": string, \"slug\": string}

- If the message needs no reply: {\"no_response\": true, \"answer\": \"\", \"slug\": \"\"}
- Otherwise: set no_response to false, answer using what you found in the knowledge base \
combined with your own knowledge, and set slug to a short 2-4 word lowercase hyphenated topic summary.
- If context_search returns no relevant results, say you don't have that information in your knowledge base.
- Your ENTIRE response must be valid JSON. No text before or after the JSON object.";

fn has_tool_call(result: &RunResult, name: &str) -> bool {
    result.tool_events.iter().any(|e| e.tool == name)
}

/// Check that the model called context_search AND did not hallucinate specifics.
fn check_no_hallucination(r: &RunResult) -> (bool, String) {
    if !has_tool_call(r, "context_search") {
        return (false, "did not call context_search tool".into());
    }

    let answer_lower = r.answer.to_lowercase();

    // The answer should indicate lack of knowledge, not fabricate details.
    // Look for hallucination signals: specific tech names that would indicate fabrication.
    let hallucination_signals = [
        "python",
        "rust",
        "javascript",
        "typescript",
        "react",
        "django",
        "flask",
        "node",
        "postgresql",
        "mongodb",
        "redis",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure",
        "java",
        "go ",
        "golang",
        "ruby",
        "rails",
        "vue",
        "angular",
        "next.js",
        "fastapi",
        "express",
        "spring",
    ];

    for signal in &hallucination_signals {
        if answer_lower.contains(signal) {
            return (
                false,
                format!(
                    "hallucinated tech stack detail '{}' despite empty search results. answer: {}",
                    signal, r.answer
                ),
            );
        }
    }

    // Should express uncertainty or lack of info
    let uncertainty_signals = [
        "don't have",
        "no relevant",
        "don't know",
        "no information",
        "not found",
        "unable to find",
        "couldn't find",
        "no results",
        "knowledge base",
        "not available",
        "no data",
    ];

    let expresses_uncertainty = uncertainty_signals.iter().any(|s| answer_lower.contains(s));

    if !expresses_uncertainty {
        return (
            false,
            format!(
                "answer does not express uncertainty despite empty search. answer: {}",
                r.answer
            ),
        );
    }

    (
        true,
        "called context_search, acknowledged lack of info".into(),
    )
}

pub fn dataset() -> Dataset {
    Dataset {
        name: "empty_search",
        system: SYSTEM,
        kind: DatasetKind::ToolUse {
            tools: tools::empty_search_tools(),
            evals: vec![Eval {
                name: "tech_stack_no_hallucinate",
                prompt: "What's your tech stack?",
                check: check_no_hallucination,
            }],
        },
    }
}
