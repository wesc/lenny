mod final_answer;
mod lookup_reference;
mod no_response;
mod random_letter;
mod random_number;

pub use final_answer::FinalAnswerTool;
pub use lookup_reference::LookupReferenceTool;
pub use no_response::NoResponseTool;
pub use random_letter::RandomLetterTool;
pub use random_number::RandomNumberTool;

use rig::agent::{HookAction, PromptHook};
use rig::completion::CompletionModel;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerData {
    pub answer: String,
    pub slug: String,
}

/// Shared mutable state passed to the hook and tools.
pub struct AgentState {
    pub final_answer: Option<FinalAnswerData>,
    pub no_response: Option<String>,
    pub events: Vec<AgentEvent>,
}

impl AgentState {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            final_answer: None,
            no_response: None,
            events: Vec::new(),
        }))
    }
}

pub(crate) fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

// ---- AgentHook: captures events + terminates on final_answer ----

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
        if let Ok(mut state) = self.state.lock() {
            state.events.push(AgentEvent::ToolResult {
                tool: tool_name.to_string(),
                result: result.to_string(),
                timestamp: now(),
            });
        }
        if tool_name == "final_answer" {
            HookAction::terminate("final_answer called")
        } else if tool_name == "no_response" {
            HookAction::terminate("no_response called")
        } else {
            HookAction::cont()
        }
    }
}
