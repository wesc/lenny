use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct RandomNumberArgs {
    pub min: i64,
    pub max: i64,
}

#[derive(Debug, Error)]
#[error("RandomNumber error: {0}")]
pub struct RandomNumberError(String);

pub struct RandomNumberTool;

impl Tool for RandomNumberTool {
    const NAME: &'static str = "random_number";

    type Error = RandomNumberError;
    type Args = RandomNumberArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "random_number".to_string(),
            description: "Generate a random integer between min and max (inclusive).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "min": { "type": "integer", "description": "Minimum value (inclusive)" },
                    "max": { "type": "integer", "description": "Maximum value (inclusive)" }
                },
                "required": ["min", "max"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        use rand::Rng;
        if args.min > args.max {
            return Err(RandomNumberError("min must be <= max".to_string()));
        }
        let n = rand::rng().random_range(args.min..=args.max);
        Ok(n)
    }
}
