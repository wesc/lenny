use anyhow::Result;
use async_trait::async_trait;
use openrouter_rs::types::Tool;
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

impl RandomLetterTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: Tool::new(
                "random_letter",
                "Generate a random lowercase letter (a-z).",
                json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
            handler: Box::new(self),
        }
    }
}
