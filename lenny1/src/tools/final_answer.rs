use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::{AgentState, FinalAnswerData};

#[derive(Debug, Deserialize)]
pub struct FinalAnswerArgs {
    pub answer: String,
    pub slug: String,
}

#[derive(Debug, Error)]
#[error("FinalAnswer error: {0}")]
pub struct FinalAnswerError(String);

pub struct FinalAnswerTool {
    pub state: Arc<Mutex<AgentState>>,
}

impl Tool for FinalAnswerTool {
    const NAME: &'static str = "final_answer";

    type Error = FinalAnswerError;
    type Args = FinalAnswerArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "final_answer".to_string(),
            description: "Submit your final answer. Provide the answer text and a short slug (2-4 words, lowercase, hyphens) summarizing the topic.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete final answer to the user's question."
                    },
                    "slug": {
                        "type": "string",
                        "description": "A short slug (2-4 words, lowercase, hyphens) summarizing the topic. Example: 'rust-ownership-basics'"
                    }
                },
                "required": ["answer", "slug"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let answer = args.answer.clone();
        let mut state = self
            .state
            .lock()
            .map_err(|e| FinalAnswerError(e.to_string()))?;
        state.final_answer = Some(FinalAnswerData {
            answer: args.answer,
            slug: args.slug,
        });
        Ok(answer)
    }
}
