use anyhow::{Result, anyhow};
use rig::OneOrMany;
use rig::completion::{
    CompletionModel as CompletionModelTrait, CompletionRequest, message::AssistantContent,
};
use rig::providers::openrouter;
use serde::de::DeserializeOwned;
use std::fs;
use std::path::{Path, PathBuf};

use crate::agent::{self, Agent, Message, ToolDef};
use crate::config::Config;
use crate::context;
use crate::tools::{
    AgentEvent, AgentOutput, BlueskyTrendingTool, ContextSearchTool, ExtractToNoteTool, LennyHook,
    LookupReferenceTool, RandomLetterTool, RandomNumberTool, ReadNoteTool, ScrapeUrlTool,
    SummarizeUrlTool, WebScrapeTool, WriteNoteTool,
};

/// Result of running a single prompt through the agent.
pub struct PromptResult {
    pub answer: String,
    pub slug: String,
    pub skipped: bool,
    pub events: Vec<AgentEvent>,
}

/// Instructions for the reasoning phase: tells the model to use tools.
const TOOL_INSTRUCTIONS: &str = "\
IMPORTANT — Knowledge base procedure:
BEFORE answering, you MUST call context_search with a query derived from the user's message. \
This applies to virtually every question — about yourself, your preferences, past events, facts, \
people, decisions, or anything that might have been discussed before. \
The only exceptions are purely procedural messages (e.g. \"hello\", \"thanks\") that clearly \
need no factual lookup. When in doubt, search. You may call context_search multiple times \
with different queries if the topic is broad.";

/// Instructions for the response phase: JSON format + anti-hallucination guardrail.
/// No mention of tools — the response model doesn't have any.
const RESPONSE_INSTRUCTIONS: &str = "\
Respond with a JSON object:
{\"no_response\": bool, \"answer\": string, \"slug\": string}

- Ensure that all the fields in the response JSON are the correct type.
- If the message is not directed at you or needs no reply, you MUST respond: {\"no_response\": true, \"answer\": \"\", \"slug\": \"\"}
- Otherwise: set no_response to false, answer the question, \
and set slug to a short 2-4 word lowercase hyphenated topic summary.
- For questions about personal details, preferences, history, or specific facts about people and projects: \
only use what is in your provided context or tool results. If that information was not found, \
say you don't have it in your knowledge base rather than guessing.
- Your ENTIRE response must be valid JSON. No text before or after the JSON object.";

/// Returns the prompt log path if logging is enabled, `None` otherwise.
fn prompt_log_path_for(config: &Config) -> Option<PathBuf> {
    if !config.prompt_log {
        return None;
    }
    Some(
        config
            .dynamic_dir
            .parent()
            .unwrap_or(&config.dynamic_dir)
            .join("prompt-log.txt"),
    )
}

fn build_tools(config: &Config) -> Vec<ToolDef> {
    let lookup_ref = LookupReferenceTool {
        references_dir: config.references_dir(),
    };
    let context_search = ContextSearchTool {
        db_path: config.memory_db(),
    };
    let write_note = WriteNoteTool {
        dynamic_dir: config.dynamic_dir.clone(),
    };
    let read_note = ReadNoteTool {
        dynamic_dir: config.dynamic_dir.clone(),
    };
    let mut tools = vec![
        lookup_ref.tool_def(),
        context_search.tool_def(),
        RandomNumberTool.tool_def(),
        RandomLetterTool.tool_def(),
        WebScrapeTool.tool_def(),
        BlueskyTrendingTool.tool_def(),
        write_note.tool_def(),
        read_note.tool_def(),
    ];
    if let Some(ref api_key) = config.firecrawl_api_key {
        if let Ok(firecrawl) = firecrawl::FirecrawlApp::new(api_key) {
            tools.push(
                SummarizeUrlTool {
                    firecrawl: firecrawl.clone(),
                }
                .tool_def(),
            );
            tools.push(
                ScrapeUrlTool {
                    firecrawl: firecrawl.clone(),
                }
                .tool_def(),
            );
            tools.push(
                ExtractToNoteTool {
                    firecrawl,
                    dynamic_dir: config.dynamic_dir.clone(),
                }
                .tool_def(),
            );
        }
    }
    tools
}

/// Run a prompt through the agent and return structured result (no printing).
/// Uses the base `config.system_dir`. For channel-specific system prompts,
/// use `run_prompt_with_system_dir`.
pub async fn run_prompt(config: &Config, user_prompt: &str) -> Result<PromptResult> {
    run_prompt_with_system_dir(config, &config.system_dir, user_prompt).await
}

/// Run a prompt with a custom system directory for context assembly.
pub async fn run_prompt_with_system_dir(
    config: &Config,
    system_dir: &Path,
    user_prompt: &str,
) -> Result<PromptResult> {
    let preamble = context::assemble_context(system_dir, &config.dynamic_dir)?;
    let eastern = chrono::Utc::now().with_timezone(&chrono_tz::US::Eastern);
    let now_str = eastern.format("%A, %B %-d, %Y %-I:%M %p %Z");
    let preamble = format!("{preamble}\n\nCurrent date/time: {now_str}");
    let reasoning_system = format!("{preamble}\n\n{TOOL_INSTRUCTIONS}\n\n{RESPONSE_INSTRUCTIONS}");
    let response_system = format!("{preamble}\n\n{RESPONSE_INSTRUCTIONS}");

    let client = agent::build_client(config)?;
    let tools = build_tools(config);
    let prompt_log_path = prompt_log_path_for(config);

    let state = crate::tools::AgentState::new();

    let agent = Agent::builder(&client, config)
        .system(&reasoning_system)
        .response_system(&response_system)
        .tools(&tools)
        .build();

    let mut hook = LennyHook {
        state: state.clone(),
        prompt_log_path,
        preamble: Some(reasoning_system.clone()),
    };

    let result = agent.run(user_prompt, &mut hook).await?;
    let output: AgentOutput = parse_agent_output(&result.answer)?;

    let events = std::mem::take(&mut state.lock().unwrap().events);

    if output.no_response {
        Ok(PromptResult {
            answer: output.answer,
            slug: "no-response".to_string(),
            skipped: true,
            events,
        })
    } else {
        Ok(PromptResult {
            answer: output.answer,
            slug: sanitize_slug(&output.slug),
            skipped: false,
            events,
        })
    }
}

/// Simple completion using the response model. No tools, no hooks.
pub async fn run_completion_typed<T: DeserializeOwned + Send>(
    config: &Config,
    preamble: &str,
    prompt: &str,
) -> Result<T> {
    run_completion_typed_with_model(config, config.provider.response_model(), preamble, prompt)
        .await
}

/// Simple completion with an explicit model name. No tools, no hooks.
/// Sends json_object + response-healing to guarantee valid JSON.
pub async fn run_completion_typed_with_model<T: DeserializeOwned + Send>(
    config: &Config,
    model_name: &str,
    preamble: &str,
    prompt: &str,
) -> Result<T> {
    let client = agent::build_client(config)?;
    let model = openrouter::CompletionModel::new(client, model_name);

    let request = CompletionRequest {
        model: None,
        preamble: Some(preamble.to_string()),
        chat_history: OneOrMany::one(Message::user(prompt)),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            "plugins": [{"id": "response-healing"}],
        })),
        output_schema: None,
    };

    let response = model
        .completion(request)
        .await
        .map_err(|e| anyhow!("completion error: {e}"))?;

    let raw = response
        .choice
        .iter()
        .filter_map(|c| {
            if let AssistantContent::Text(t) = c {
                Some(t.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    parse_json_response(&raw)
}

/// The `once` CLI command: run prompt, print JSON, save turn.
pub async fn run(config: &Config, user_prompt: &str) -> Result<()> {
    let result = run_prompt(config, user_prompt).await?;

    let tool_calls: Vec<&AgentEvent> = result
        .events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolCall { .. }))
        .collect();

    let output = serde_json::json!({
        "prompt": user_prompt,
        "answer": result.answer,
        "slug": result.slug,
        "skipped": result.skipped,
        "tool_calls": tool_calls,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    save_turn(config, user_prompt, &result)?;
    Ok(())
}

/// Parse AgentOutput from a raw LLM response string.
fn parse_agent_output(raw: &str) -> Result<AgentOutput> {
    parse_json_response(raw)
}

/// Parse JSON from an LLM response into T.
/// Uses llm_json to repair malformed JSON from LLMs (missing quotes,
/// trailing commas, prose wrapping, markdown fences, etc).
fn parse_json_response<T: DeserializeOwned>(raw: &str) -> Result<T> {
    let repaired = llm_json::repair_json(raw, &llm_json::RepairOptions::default())
        .map_err(|e| anyhow!("Failed to repair LLM JSON: {e}\nRaw response: {raw}"))?;
    serde_json::from_str(&repaired).map_err(|e| {
        anyhow!("Failed to parse repaired JSON: {e}\nRepaired: {repaired}\nRaw: {raw}")
    })
}

pub fn sanitize_slug(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect()
}

fn save_turn(config: &Config, prompt: &str, result: &PromptResult) -> Result<()> {
    let turns_dir = config.references_dir().join("turns");
    fs::create_dir_all(&turns_dir)?;

    let timestamp = chrono::Utc::now();
    let time_str = timestamp.format("%Y%m%d-%H%M%S").to_string();
    let filename = format!("{}-{}.json", time_str, result.slug);

    let mut lines = Vec::new();

    lines.push(serde_json::to_string(&serde_json::json!({
        "type": "prompt",
        "content": prompt,
        "timestamp": timestamp.to_rfc3339(),
    }))?);

    for event in &result.events {
        lines.push(serde_json::to_string(event)?);
    }

    lines.push(serde_json::to_string(&serde_json::json!({
        "type": "answer",
        "content": result.answer,
        "slug": result.slug,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))?);

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = turns_dir.join(&tmp_name);
    let final_path = turns_dir.join(&filename);

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    eprintln!("Saved turn: {}", final_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Simple {
        answer: String,
    }

    #[test]
    fn parse_clean_json() {
        let r: Simple = parse_json_response(r#"{"answer": "hello"}"#).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_with_whitespace() {
        let r: Simple = parse_json_response("  \n{\"answer\": \"hello\"}\n  ").unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_markdown_fences() {
        let input = "```json\n{\"answer\": \"hello\"}\n```";
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_leading_prose() {
        let input = "Here is the result:\n{\"answer\": \"hello\"}";
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_trailing_prose() {
        let input = "{\"answer\": \"hello\"}\n\nI hope that helps!";
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_leading_and_trailing_prose() {
        let input = "Sure! Here you go:\n{\"answer\": \"hello\"}\nLet me know if you need more.";
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_nested_braces_in_string() {
        let input = r#"{"answer": "use {x} for templating"}"#;
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "use {x} for templating");
    }

    #[test]
    fn parse_escaped_quotes() {
        let input = r#"{"answer": "she said \"hi\""}"#;
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "she said \"hi\"");
    }

    #[test]
    fn parse_fails_on_garbage() {
        let result: Result<Simple> = parse_json_response("not json at all");
        assert!(result.is_err());
    }
}
