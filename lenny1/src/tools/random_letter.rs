use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct RandomLetterArgs {}

#[derive(Debug, Error)]
#[error("RandomLetter error: {0}")]
pub struct RandomLetterError(String);

pub struct RandomLetterTool;

impl Tool for RandomLetterTool {
    const NAME: &'static str = "random_letter";

    type Error = RandomLetterError;
    type Args = RandomLetterArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "random_letter".to_string(),
            description: "Generate a random lowercase letter (a-z).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        use rand::Rng;
        let c = rand::rng().random_range(b'a'..=b'z') as char;
        tracing::debug!(letter = %c, "generated random letter");
        Ok(c.to_string())
    }
}
