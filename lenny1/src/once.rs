use anyhow::{Result, anyhow};
use openrouter_rs::{
    OpenRouterClient,
    api::chat::{ChatCompletionRequest, Message, Plugin},
    types::{ResponseFormat, Role},
};
use serde::de::DeserializeOwned;
use std::fs;
use std::path::PathBuf;

use crate::agent::{Agent, ToolDef};
use crate::config::{Config, ProviderConfig};
use crate::context;
use crate::tools::{
    AgentEvent, AgentOutput, ContextSearchTool, LennyHook, LookupReferenceTool, RandomLetterTool,
    RandomNumberTool, WebScrapeTool,
};

/// Result of running a single prompt through the agent.
pub struct PromptResult {
    pub answer: String,
    pub slug: String,
    pub skipped: bool,
    pub events: Vec<AgentEvent>,
}

const OUTPUT_INSTRUCTIONS: &str = "\
IMPORTANT — Knowledge base procedure:
BEFORE answering, you MUST call context_search with a query derived from the user's message. \
This applies to virtually every question — about yourself, your preferences, past events, facts, \
people, decisions, or anything that might have been discussed before. \
The only exceptions are purely procedural messages (e.g. \"hello\", \"thanks\") that clearly \
need no factual lookup. When in doubt, search. You may call context_search multiple times \
with different queries if the topic is broad.

After searching and using any other tools you need, respond with a JSON object:
{\"no_response\": bool, \"answer\": string, \"slug\": string}

- If the message is not directed at you or needs no reply: {\"no_response\": true, \"answer\": \"\", \"slug\": \"\"}
- Otherwise: set no_response to false, answer using what you found in the knowledge base, \
and set slug to a short 2-4 word lowercase hyphenated topic summary.
- If context_search returns no relevant results, say you don't have that information in your knowledge base.
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

fn build_client(config: &Config) -> Result<OpenRouterClient> {
    let ProviderConfig::OpenRouter { ref api_key, .. } = config.provider;
    Ok(OpenRouterClient::builder().api_key(api_key).build()?)
}

fn build_tools(config: &Config) -> Vec<ToolDef> {
    let lookup_ref = LookupReferenceTool {
        references_dir: config.references_dir(),
    };
    let context_search = ContextSearchTool {
        db_path: config.knowledge_dir.join("comprehensions"),
    };
    vec![
        lookup_ref.tool_def(),
        context_search.tool_def(),
        RandomNumberTool.tool_def(),
        RandomLetterTool.tool_def(),
        WebScrapeTool.tool_def(),
    ]
}

/// Run a prompt through the agent and return structured result (no printing).
pub async fn run_prompt(config: &Config, user_prompt: &str) -> Result<PromptResult> {
    let preamble = context::assemble_context(&config.system_dir, &config.dynamic_dir)?;
    let full_preamble = format!("{preamble}\n\n{OUTPUT_INSTRUCTIONS}");

    let client = build_client(config)?;
    let tools = build_tools(config);
    let prompt_log_path = prompt_log_path_for(config);

    let state = crate::tools::AgentState::new();

    let agent = Agent::builder(&client, config)
        .system(&full_preamble)
        .tools(&tools)
        .build();

    let mut hook = LennyHook {
        state: state.clone(),
        prompt_log_path,
        preamble: Some(full_preamble.clone()),
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

/// Simple completion: no tools, no hooks. For internal use (e.g. comprehension).
/// Uses the response model for a single non-agentic call.
/// Sends json_object + response-healing to guarantee valid JSON.
pub async fn run_completion_typed<T: DeserializeOwned + Send>(
    config: &Config,
    preamble: &str,
    prompt: &str,
) -> Result<T> {
    let client = build_client(config)?;
    let model = config.provider.response_model();

    let request = ChatCompletionRequest::builder()
        .model(model)
        .messages(vec![
            Message::new(Role::System, preamble),
            Message::new(Role::User, prompt),
        ])
        .response_format(ResponseFormat::json_object())
        .plugins(vec![Plugin::new("response-healing")])
        .max_tokens(1024u32)
        .build()?;

    let response = client.send_chat_completion(&request).await?;
    let choice = response
        .choices
        .first()
        .ok_or_else(|| anyhow!("no choices in response"))?;

    let raw = choice.content().unwrap_or("");
    parse_json_response(raw)
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
/// Relies on provider-level structured output (OpenRouter json_object + response-healing)
/// to guarantee valid JSON. Falls back to trimming markdown fences.
fn parse_json_response<T: DeserializeOwned>(raw: &str) -> Result<T> {
    // First try direct parse (works when provider enforces JSON schema)
    if let Ok(val) = serde_json::from_str(raw.trim()) {
        return Ok(val);
    }
    // Fallback: strip markdown code fences
    let trimmed = raw.trim();
    let json_str = if trimmed.starts_with("```") {
        let after_first = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        after_first
            .strip_suffix("```")
            .unwrap_or(after_first)
            .trim()
    } else {
        trimmed
    };
    serde_json::from_str(json_str)
        .map_err(|e| anyhow!("Failed to parse LLM response as JSON: {e}\nRaw response: {raw}"))
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
