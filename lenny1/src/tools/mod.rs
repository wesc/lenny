mod bluesky_trending;
mod context_search;
mod extract_url;
mod extract_url_to_note;
mod lookup_reference;
mod notes;
mod random_letter;
mod random_number;
mod read_note;
mod web_search;
mod write_note;

pub use bluesky_trending::{BlueskyTrendingTool, fetch_trending};
pub use context_search::ContextSearchTool;
pub use extract_url::ExtractUrlTool;
pub use extract_url_to_note::ExtractUrlToNoteTool;
pub use lookup_reference::LookupReferenceTool;
pub use random_letter::RandomLetterTool;
pub use random_number::RandomNumberTool;
pub use read_note::ReadNoteTool;
pub use web_search::WebSearchTool;
pub use write_note::WriteNoteTool;

/// Build mock versions of all tools (same definitions, stub handlers). Used by evals.
pub fn build_mock_tools() -> Vec<crate::agent::ToolDef> {
    vec![
        random_number::mock_tool_def(),
        random_letter::mock_tool_def(),
        context_search::mock_tool_def(),
        lookup_reference::mock_tool_def(),
        extract_url::mock_tool_def(),
        web_search::mock_tool_def(),
        extract_url_to_note::mock_tool_def(),
        write_note::mock_tool_def(),
        read_note::mock_tool_def(),
        bluesky_trending::mock_tool_def(),
    ]
}

use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::agent;

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

pub struct LennyHook<'a> {
    pub state: Arc<Mutex<AgentState>>,
    pub prompt_log_path: Option<PathBuf>,
    pub preamble: Option<String>,
    pub prompt_hook: Option<&'a mut dyn crate::once::PromptHooks>,
}

impl agent::AgentHook for LennyHook<'_> {
    fn on_request(&mut self, _iteration: usize, request: &rig::completion::CompletionRequest) {
        if let Some(ref mut hook) = self.prompt_hook {
            hook.on_request();
        }
        if let Some(path) = &self.prompt_log_path {
            let mut out = String::new();
            if let Some(preamble) = &self.preamble {
                out.push_str("[SYSTEM]\n");
                out.push_str(preamble);
                out.push_str("\n\n");
            }
            // CompletionRequest doesn't implement Serialize, so log a summary
            let summary = agent::RequestSummary::from(request);
            if let Ok(json) = serde_json::to_string_pretty(&summary) {
                out.push_str("[REQUEST]\n");
                out.push_str(&json);
                out.push_str("\n\n");
            }
            let _ = std::fs::write(path, out.trim_end());
        }
    }

    fn on_tool_start(&mut self, _iteration: usize, name: &str, args: &str) {
        let args_value =
            serde_json::from_str(args).unwrap_or(serde_json::Value::String(args.to_string()));
        tracing::debug!(tool = name, args, "tool call");
        if let Some(ref mut hook) = self.prompt_hook {
            hook.on_tool_start(name, args);
        }
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolCall {
                tool: name.to_string(),
                args: args_value,
                timestamp: now(),
            });
        }
    }

    fn on_tool_call(&mut self, _iteration: usize, name: &str, _args: &str, result: &str) {
        let truncated = &result[..result.len().min(200)];
        tracing::debug!(tool = name, result = truncated, "tool result");
        if let Some(ref mut hook) = self.prompt_hook {
            hook.on_tool_result(name, result);
        }
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolResult {
                tool: name.to_string(),
                result: result.to_string(),
                timestamp: now(),
            });
        }
    }
}
