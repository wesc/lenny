mod lookup_reference;
mod random_letter;
mod random_number;
mod web_scrape;

pub use lookup_reference::LookupReferenceTool;
pub use random_letter::RandomLetterTool;
pub use random_number::RandomNumberTool;
pub use web_scrape::WebScrapeTool;

use rig::agent::{HookAction, PromptHook};
use rig::completion::CompletionModel;
use serde::{Deserialize, Serialize};
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

// ---- AgentHook: captures tool events ----

#[derive(Clone)]
pub struct AgentHook {
    pub state: Arc<Mutex<AgentState>>,
}

impl<M: CompletionModel> PromptHook<M> for AgentHook {
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
