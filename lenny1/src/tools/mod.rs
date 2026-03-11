mod bluesky_trending;
mod context_search;
mod extract_to_note;
mod lookup_reference;
mod notes;
mod random_letter;
mod random_number;
mod read_note;
mod scrape_url;
mod summarize_url;
mod web_scrape;
mod write_note;

pub use bluesky_trending::{BlueskyTrendingTool, fetch_trending};
pub use context_search::ContextSearchTool;
pub use extract_to_note::ExtractToNoteTool;
pub use lookup_reference::LookupReferenceTool;
pub use random_letter::RandomLetterTool;
pub use random_number::RandomNumberTool;
pub use read_note::ReadNoteTool;
pub use scrape_url::ScrapeUrlTool;
pub use summarize_url::SummarizeUrlTool;
pub use web_scrape::WebScrapeTool;
pub use write_note::WriteNoteTool;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::agent;

// ---- Structured output ----

/// The agent's structured response, enforced via output schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// ---- LennyHook: captures tool events + prompt logging ----

pub struct LennyHook {
    pub state: Arc<Mutex<AgentState>>,
    pub prompt_log_path: Option<PathBuf>,
    pub preamble: Option<String>,
}

impl agent::AgentHook for LennyHook {
    fn on_request(
        &mut self,
        _iteration: usize,
        request: &openrouter_rs::api::chat::ChatCompletionRequest,
    ) {
        if let Some(path) = &self.prompt_log_path {
            let mut out = String::new();
            if let Some(preamble) = &self.preamble {
                out.push_str("[SYSTEM]\n");
                out.push_str(preamble);
                out.push_str("\n\n");
            }
            if let Ok(json) = serde_json::to_string_pretty(request) {
                out.push_str("[REQUEST]\n");
                out.push_str(&json);
                out.push_str("\n\n");
            }
            let _ = std::fs::write(path, out.trim_end());
        }
    }

    fn on_tool_call(&mut self, _iteration: usize, name: &str, args: &str, result: &str) {
        let args_value =
            serde_json::from_str(args).unwrap_or(serde_json::Value::String(args.to_string()));
        tracing::debug!(tool = name, args, "tool call");
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolCall {
                tool: name.to_string(),
                args: args_value,
                timestamp: now(),
            });
            let truncated = &result[..result.len().min(200)];
            tracing::debug!(tool = name, result = truncated, "tool result");
            state.events.push(AgentEvent::ToolResult {
                tool: name.to_string(),
                result: result.to_string(),
                timestamp: now(),
            });
        }
    }
}
