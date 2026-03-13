use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

pub struct RandomLetterTool;

#[async_trait]
impl ToolHandler for RandomLetterTool {
    async fn call(&self, _args: &serde_json::Value) -> Result<String> {
        use rand::Rng;
        let c = rand::rng().random_range(b'a'..=b'z') as char;
        tracing::debug!(letter = %c, "generated random letter");
        Ok(c.to_string())
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "random_letter".to_string(),
        description: "Generate a random lowercase letter (a-z). Takes no arguments.".to_string(),
        parameters: json!({"type": "object", "properties": {}}),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "q".to_string(),
        }),
    }
}

impl RandomLetterTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
