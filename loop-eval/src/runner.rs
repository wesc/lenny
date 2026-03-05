use anyhow::Result;
use openrouter_rs::{
    OpenRouterClient,
    api::chat::{ChatCompletionRequest, Message},
    types::Role,
};
use serde::Deserialize;

use crate::agent::{Agent, NoopHook};
use crate::config::Config;
use crate::tools::ToolDef;

// Re-export from agent so downstream imports don't break.
pub use crate::agent::RunResult;

/// Result of running a response-only eval (no reasoning loop).
#[derive(Debug)]
pub struct ResponseOnlyResult {
    /// Final answer from the response model.
    pub answer: String,
}

/// Run a single eval: reasoning loop then response phase.
pub async fn run_eval(
    client: &OpenRouterClient,
    config: &Config,
    system: &str,
    user_prompt: &str,
    tool_defs: &[ToolDef],
) -> Result<RunResult> {
    let agent = Agent::builder(client, config)
        .system(system)
        .tools(tool_defs)
        .build();

    agent.run(user_prompt, &mut NoopHook).await
}

/// Run a response-only eval: skip reasoning loop, use pre-fabricated tool context.
pub async fn run_response_only(
    client: &OpenRouterClient,
    config: &Config,
    system: &str,
    user_prompt: &str,
    tool_context: &str,
) -> Result<ResponseOnlyResult> {
    let agent = Agent::builder(client, config).system(system).build();

    let answer = agent
        .run_response_only(user_prompt, tool_context, &mut NoopHook)
        .await?;
    Ok(ResponseOnlyResult { answer })
}

// ---------------------------------------------------------------------------
// Judge (unchanged)
// ---------------------------------------------------------------------------

/// Result of an LLM judge evaluation.
#[derive(Debug)]
pub struct JudgeResult {
    pub pass: bool,
    pub reason: String,
}

#[derive(Deserialize)]
struct JudgeResponse {
    pass: bool,
    reason: String,
}

const JUDGE_SYSTEM: &str = "\
You are an evaluation judge. You will receive an answer and grading criteria. \
Evaluate whether the answer meets the criteria. \
Respond with ONLY a JSON object: {\"pass\": true/false, \"reason\": \"...\"}";

/// Use an LLM to judge whether an answer meets the given criteria.
pub async fn judge(
    client: &OpenRouterClient,
    judge_model: &str,
    answer: &str,
    criteria: &str,
) -> Result<JudgeResult> {
    let user_prompt = format!(
        "## Answer\n{answer}\n\n## Criteria\n{criteria}\n\n\
         Respond with ONLY a JSON object: {{\"pass\": true/false, \"reason\": \"...\"}}"
    );

    let request = ChatCompletionRequest::builder()
        .model(judge_model)
        .messages(vec![
            Message::new(Role::System, JUDGE_SYSTEM),
            Message::new(Role::User, user_prompt.as_str()),
        ])
        .max_tokens(256u32)
        .build()?;

    let response = client.send_chat_completion(&request).await?;

    let raw = response
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("no choices in judge response"))?
        .content()
        .unwrap_or("")
        .to_string();

    match serde_json::from_str::<JudgeResponse>(&raw) {
        Ok(parsed) => Ok(JudgeResult {
            pass: parsed.pass,
            reason: parsed.reason,
        }),
        Err(_) => Ok(JudgeResult {
            pass: false,
            reason: format!("judge returned unparseable response: {raw}"),
        }),
    }
}
