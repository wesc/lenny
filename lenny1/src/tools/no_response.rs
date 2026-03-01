use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::AgentState;

#[derive(Debug, Deserialize)]
pub struct NoResponseArgs {
    pub reason: String,
}

#[derive(Debug, Error)]
#[error("NoResponse error: {0}")]
pub struct NoResponseError(String);

pub struct NoResponseTool {
    pub state: Arc<Mutex<AgentState>>,
}

impl Tool for NoResponseTool {
    const NAME: &'static str = "no_response";

    type Error = NoResponseError;
    type Args = NoResponseArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "no_response".to_string(),
            description: "Call this when the conversation does not require your response. Use this to opt out of replying to messages that are not directed at you or do not need a reply.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason why no response is needed."
                    }
                },
                "required": ["reason"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let reason = args.reason.clone();
        let mut state = self
            .state
            .lock()
            .map_err(|e| NoResponseError(e.to_string()))?;
        state.no_response = Some(reason.clone());
        Ok(reason)
    }
}
