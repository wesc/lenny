use anyhow::{Result, anyhow};
use regex::Regex;
use rig::OneOrMany;
use rig::completion::{
    CompletionModel as CompletionModelTrait, CompletionRequest, message::AssistantContent,
};
use rig::providers::openrouter;
use serde::de::DeserializeOwned;
use std::fs;
use std::path::PathBuf;

use crate::agent::{self, Agent, Message, ToolDef};
use crate::config::Config;
use crate::context::{self, TemplateContext};
use crate::session::{self, SessionId};
use crate::tools::{
    AgentEvent, BlueskyTrendingTool, ContextSearchTool, ExtractUrlToNoteTool, ExtractUrlTool,
    LennyHook, LookupReferenceTool, RandomLetterTool, RandomNumberTool, ReadNoteTool,
    WebSearchTool, WriteNoteTool,
};

/// Caller-provided hooks for prompt execution events.
/// All methods default to no-ops.
#[allow(unused_variables)]
pub trait PromptHooks: Send {
    /// Called before each LLM request is sent.
    fn on_request(&mut self) {}
    /// Called before a tool is executed.
    fn on_tool_start(&mut self, name: &str, args: &str) {}
    /// Called after a tool returns a result.
    fn on_tool_result(&mut self, name: &str, result: &str) {}
}

/// Result of running a single prompt through the agent.
pub struct PromptResult {
    pub answer: String,
    pub slug: String,
    pub skipped: bool,
    pub events: Vec<AgentEvent>,
}

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
        BlueskyTrendingTool.tool_def(),
        write_note.tool_def(),
        read_note.tool_def(),
    ];
    if let Some(ref api_key) = config.firecrawl_api_key {
        if let Ok(firecrawl) = firecrawl::FirecrawlApp::new(api_key) {
            tools.push(
                WebSearchTool {
                    firecrawl: firecrawl.clone(),
                }
                .tool_def(),
            );
            tools.push(
                ExtractUrlTool {
                    firecrawl: firecrawl.clone(),
                }
                .tool_def(),
            );
            tools.push(
                ExtractUrlToNoteTool {
                    firecrawl,
                    dynamic_dir: config.dynamic_dir.clone(),
                }
                .tool_def(),
            );
        }
    }
    tools
}

/// Build a TemplateContext with the current date/time in Eastern timezone.
fn build_template_context(channel: &str) -> TemplateContext {
    let eastern = chrono::Utc::now().with_timezone(&chrono_tz::US::Eastern);
    let now_str = eastern.format("%A, %B %-d, %Y %-I:%M %p %Z").to_string();
    TemplateContext {
        channel_name: channel.to_string(),
        current_datetime: now_str,
    }
}

/// Extract slug from answer text. Looks for `[slug: topic-name]` at end.
/// Falls back to deriving slug from first few words.
fn extract_slug(answer: &str) -> String {
    let re = Regex::new(r"\[slug:\s*([a-z0-9-]+)\]\s*$").unwrap();
    if let Some(caps) = re.captures(answer) {
        return caps[1].to_string();
    }
    // Fallback: first 4 words, lowercased, hyphenated
    let slug: String = answer
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    if slug.is_empty() {
        "unknown".to_string()
    } else {
        slug
    }
}

/// Strip the slug tag from the answer text if present.
fn strip_slug_tag(answer: &str) -> String {
    let re = Regex::new(r"\s*\[slug:\s*[a-z0-9-]+\]\s*$").unwrap();
    re.replace(answer, "").to_string()
}

/// Detect no-response convention.
fn is_no_response(answer: &str) -> bool {
    answer.contains("[no-response]")
}

/// Run a prompt through the agent with a channel name for system prompt selection.
/// Loads session history + documents from disk, runs agent, persists the turn.
pub async fn run_prompt(
    config: &Config,
    channel: &str,
    session_id: &SessionId,
    user_prompt: &str,
    hook: Option<&mut dyn PromptHooks>,
) -> Result<PromptResult> {
    run_prompt_inner(config, channel, session_id, user_prompt, None, hook).await
}

/// Run a prompt with custom tools (for evals with mock tool handlers).
pub async fn run_prompt_with_tools(
    config: &Config,
    channel: &str,
    session_id: &SessionId,
    user_prompt: &str,
    tools: Vec<ToolDef>,
) -> Result<PromptResult> {
    run_prompt_inner(config, channel, session_id, user_prompt, Some(tools), None).await
}

async fn run_prompt_inner(
    config: &Config,
    channel: &str,
    session_id: &SessionId,
    user_prompt: &str,
    custom_tools: Option<Vec<ToolDef>>,
    prompt_hook: Option<&mut dyn PromptHooks>,
) -> Result<PromptResult> {
    // Step 1: Assemble system prompt from channel directory
    let system_dir = config.system_dir.join(channel);
    let ctx = build_template_context(channel);
    let preamble = context::assemble_system_prompt(&system_dir, &ctx)?;

    // Step 2: Load session state from disk
    let session_ctx = session::load_session(&config.dynamic_dir, session_id)?;

    let client = agent::build_client(config)?;
    let tools = custom_tools.unwrap_or_else(|| build_tools(config));
    let prompt_log_path = prompt_log_path_for(config);

    let hook_state = crate::tools::AgentState::new();

    let agent = Agent::builder(&client, config)
        .system(&preamble)
        .tools(&tools)
        .documents(session_ctx.documents)
        .build();

    let mut hook = LennyHook {
        state: hook_state.clone(),
        prompt_log_path,
        preamble: Some(preamble.clone()),
        prompt_hook,
    };

    // Step 3: Run agent with history from disk
    let result = agent
        .run(user_prompt, session_ctx.history, &mut hook)
        .await?;

    let events = std::mem::take(&mut hook_state.lock().unwrap().events);

    let answer = strip_slug_tag(&result.answer);
    let skipped = is_no_response(&answer);
    let slug = if skipped {
        "no-response".to_string()
    } else {
        extract_slug(&result.answer)
    };
    let answer = if skipped { String::new() } else { answer };

    // Step 4: Persist turn to session directory
    let new_messages = extract_new_messages(&result.messages, user_prompt);
    if let Err(e) = session::save_turn(&config.dynamic_dir, session_id, &new_messages, &slug) {
        tracing::warn!(error = %e, "failed to save session turn");
    }

    Ok(PromptResult {
        answer,
        slug,
        skipped,
        events,
    })
}

/// Extract only the new messages produced during this run.
/// History messages are already persisted; we only want the user prompt + assistant turns.
fn extract_new_messages(all_messages: &[Message], user_prompt: &str) -> Vec<Message> {
    // Find the last user message matching the prompt — everything from there onwards is new.
    // Walk backwards to find it.
    for (i, msg) in all_messages.iter().enumerate().rev() {
        if let Message::User { content } = msg {
            let text: String = content
                .iter()
                .filter_map(|c| {
                    if let rig::completion::message::UserContent::Text(t) = c {
                        Some(t.text.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            if text == user_prompt {
                return all_messages[i..].to_vec();
            }
        }
    }
    // Fallback: return all messages
    all_messages.to_vec()
}

/// Simple completion using the reasoning model. No tools, no hooks.
pub async fn run_completion_typed<T: DeserializeOwned + Send>(
    config: &Config,
    preamble: &str,
    prompt: &str,
) -> Result<T> {
    run_completion_typed_with_model(config, config.provider.agent_model(), preamble, prompt).await
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

/// The `once` CLI command: run prompt, print output, save turn.
pub async fn run(config: &Config, user_prompt: &str) -> Result<()> {
    let session_id = SessionId::new("once", "default");
    let result = run_prompt(config, "cli", &session_id, user_prompt, None).await?;

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

    save_turn_to_references(config, user_prompt, &result)?;
    Ok(())
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

#[allow(dead_code)]
pub fn sanitize_slug(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect()
}

/// Save turn to references/turns/ for backward compatibility with fact digest.
fn save_turn_to_references(config: &Config, prompt: &str, result: &PromptResult) -> Result<()> {
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

    #[test]
    fn extract_slug_from_tag() {
        let answer = "The answer is 42. [slug: meaning-of-life]";
        assert_eq!(extract_slug(answer), "meaning-of-life");
    }

    #[test]
    fn extract_slug_fallback() {
        let answer = "Hello world from here";
        assert_eq!(extract_slug(answer), "hello-world-from-here");
    }

    #[test]
    fn extract_slug_empty() {
        assert_eq!(extract_slug(""), "unknown");
    }

    #[test]
    fn strip_slug_tag_present() {
        let answer = "The answer is 42. [slug: meaning-of-life]";
        assert_eq!(strip_slug_tag(answer), "The answer is 42.");
    }

    #[test]
    fn strip_slug_tag_absent() {
        let answer = "The answer is 42.";
        assert_eq!(strip_slug_tag(answer), "The answer is 42.");
    }

    #[test]
    fn no_response_detection() {
        assert!(is_no_response("[no-response]"));
        assert!(is_no_response("  [no-response]  "));
        assert!(is_no_response("Some preamble\n\n[no-response]"));
        assert!(!is_no_response("Hello"));
    }

    #[test]
    fn parse_clean_json() {
        use serde::Deserialize;
        #[derive(Debug, Deserialize, PartialEq)]
        struct Simple {
            answer: String,
        }
        let r: Simple = parse_json_response(r#"{"answer": "hello"}"#).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_markdown_fences() {
        use serde::Deserialize;
        #[derive(Debug, Deserialize, PartialEq)]
        struct Simple {
            answer: String,
        }
        let input = "```json\n{\"answer\": \"hello\"}\n```";
        let r: Simple = parse_json_response(input).unwrap();
        assert_eq!(r.answer, "hello");
    }

    #[test]
    fn parse_fails_on_garbage() {
        use serde::Deserialize;
        #[derive(Debug, Deserialize)]
        struct Simple {
            #[allow(dead_code)]
            answer: String,
        }
        let result: Result<Simple> = parse_json_response("not json at all");
        assert!(result.is_err());
    }
}
