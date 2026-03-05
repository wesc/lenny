mod context_search;
mod lookup_reference;
mod random_letter;
mod random_number;
mod web_scrape;

pub use context_search::ContextSearchTool;
pub use lookup_reference::LookupReferenceTool;
pub use random_letter::RandomLetterTool;
pub use random_number::RandomNumberTool;
pub use web_scrape::WebScrapeTool;

use rig::agent::{HookAction, PromptHook};
use rig::completion::CompletionModel;
use rig::message::Message;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ---- Structured output ----

/// The agent's structured response, enforced via output schema.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentOutput {
    /// Set to true when the conversation does not need a response from you.
    #[serde(default)]
    pub no_response: bool,
    /// Your complete final answer to the user's question.
    #[serde(default)]
    pub answer: String,
    /// A short slug (2-4 words, lowercase, hyphens) summarizing the topic.
    #[serde(default)]
    pub slug: String,
}

// ---- Shared agent state ----

/// A single event in the agent's iteration history.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    #[serde(rename = "tool_call")]
    ToolCall {
        tool: String,
        args: serde_json::Value,
        timestamp: String,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool: String,
        result: String,
        timestamp: String,
    },
}

/// Shared mutable state passed to the hook.
pub struct AgentState {
    pub events: Vec<AgentEvent>,
}

impl AgentState {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self { events: Vec::new() }))
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Format a message's content field for the prompt log.
/// If it's a string, print it directly. Otherwise print as JSON.
fn format_content(content: &serde_json::Value, out: &mut String) {
    match content {
        serde_json::Value::String(s) => out.push_str(s),
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                } else if let Ok(json) = serde_json::to_string_pretty(item) {
                    out.push_str(&json);
                }
                out.push('\n');
            }
        }
        other => {
            if let Ok(json) = serde_json::to_string_pretty(other) {
                out.push_str(&json);
            }
        }
    }
}

// ---- AgentHook: captures tool events ----

#[derive(Clone)]
pub struct AgentHook {
    pub state: Arc<Mutex<AgentState>>,
    pub prompt_log_path: Option<PathBuf>,
    pub preamble: Option<String>,
}

impl<M: CompletionModel> PromptHook<M> for AgentHook {
    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        if let Some(path) = &self.prompt_log_path {
            let mut out = String::new();
            if let Some(preamble) = &self.preamble {
                out.push_str("[SYSTEM]\n");
                out.push_str(preamble);
                out.push_str("\n\n");
            }
            for msg in history.iter().chain(std::iter::once(prompt)) {
                if let Ok(v) = serde_json::to_value(msg) {
                    let role = v["role"].as_str().unwrap_or("unknown").to_uppercase();
                    out.push_str(&format!("[{role}]\n"));
                    format_content(&v["content"], &mut out);
                    out.push_str("\n\n");
                }
            }
            let _ = std::fs::write(path, out.trim_end());
        }
        HookAction::cont()
    }

    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> rig::agent::ToolCallHookAction {
        let args_value =
            serde_json::from_str(args).unwrap_or(serde_json::Value::String(args.to_string()));
        tracing::debug!(tool = tool_name, args, "tool call");
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolCall {
                tool: tool_name.to_string(),
                args: args_value,
                timestamp: now(),
            });
        }
        rig::agent::ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        let truncated = &result[..result.len().min(200)];
        tracing::debug!(tool = tool_name, result = truncated, "tool result");
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolResult {
                tool: tool_name.to_string(),
                result: result.to_string(),
                timestamp: now(),
            });
        }
        HookAction::cont()
    }
}
