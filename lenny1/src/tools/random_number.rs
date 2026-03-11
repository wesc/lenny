use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct RandomNumberArgs {
    min: i64,
    max: i64,
}

pub struct RandomNumberTool;

#[async_trait]
impl ToolHandler for RandomNumberTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        use rand::Rng;
        let args: RandomNumberArgs = serde_json::from_value(args.clone())?;
        if args.min > args.max {
            anyhow::bail!("min must be <= max");
        }
        let n = rand::rng().random_range(args.min..=args.max);
        tracing::debug!(
            min = args.min,
            max = args.max,
            result = n,
            "generated random number"
        );
        Ok(n.to_string())
    }
}

impl RandomNumberTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "random_number".to_string(),
                description: "Generate a random integer between min and max (inclusive)."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "min": { "type": "integer", "description": "Minimum value (inclusive)" },
                        "max": { "type": "integer", "description": "Maximum value (inclusive)" }
                    },
                    "required": ["min", "max"]
                }),
            },
            handler: Box::new(self),
        }
    }
}
