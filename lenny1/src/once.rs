use anyhow::Result;
use rig::client::{CompletionClient, Nothing};
use rig::completion::{Prompt, TypedPrompt};
use rig::providers::{ollama, openrouter};
use serde_json::json;
use std::fs;

use crate::config::{Config, ProviderConfig};
use crate::context;
use crate::tools::{
    AgentEvent, AgentHook, AgentOutput, AgentState, LookupReferenceTool, RandomLetterTool,
    RandomNumberTool, WebScrapeTool,
};

/// Result of running a single prompt through the agent.
pub struct PromptResult {
    pub answer: String,
    pub slug: String,
    pub skipped: bool,
    pub events: Vec<AgentEvent>,
}

const OUTPUT_INSTRUCTIONS: &str = "\
After using any tools you need, you MUST respond with a JSON object matching this schema:
{\"no_response\": bool, \"answer\": string, \"slug\": string}

- If the message is not directed at you or needs no reply, respond: {\"no_response\": true, \"answer\": \"\", \"slug\": \"\"}
- Otherwise, set no_response to false, answer with your full response, and slug with a short 2-4 word lowercase hyphenated topic summary.
- Your ENTIRE response must be valid JSON. No text before or after the JSON object.";

/// Build agent from any CompletionClient, run prompt, return structured result.
async fn run_with_client<C: CompletionClient>(
    client: C,
    model: &str,
    config: &Config,
    preamble: &str,
    user_prompt: &str,
    additional_params: Option<serde_json::Value>,
) -> Result<PromptResult> {
    let state = AgentState::new();

    let lookup_ref = LookupReferenceTool {
        references_dir: config.references_dir(),
    };

    let full_preamble = format!("{preamble}\n\n{OUTPUT_INSTRUCTIONS}");

    let mut builder = client
        .agent(model)
        .preamble(&full_preamble)
        .tool(lookup_ref)
        .tool(RandomNumberTool)
        .tool(RandomLetterTool)
        .tool(WebScrapeTool)
        .default_max_turns(config.max_iterations);

    if let Some(params) = additional_params {
        builder = builder.additional_params(params);
    }

    let agent = builder.build();

    let hook = AgentHook {
        state: state.clone(),
    };
    let output: AgentOutput = agent.prompt_typed(user_prompt).with_hook(hook).await?;

    let events = std::mem::take(&mut state.lock().unwrap().events);

    if output.no_response {
        Ok(PromptResult {
            answer: output.answer,
            slug: "no-response".to_string(),
            skipped: true,
            events,
        })
    } else {
        Ok(PromptResult {
            answer: output.answer,
            slug: sanitize_slug(&output.slug),
            skipped: false,
            events,
        })
    }
}

/// Run a prompt through the agent and return structured result (no printing).
pub async fn run_prompt(config: &Config, user_prompt: &str) -> Result<PromptResult> {
    let preamble = context::assemble_context(&config.system_dir, &config.dynamic_dir)?;

    match &config.provider {
        ProviderConfig::Ollama { url, model } => {
            let client: ollama::Client = ollama::Client::builder()
                .api_key(Nothing)
                .base_url(url)
                .build()?;
            let params = Some(json!({"think": config.thinking}));
            run_with_client(client, model, config, &preamble, user_prompt, params).await
        }
        ProviderConfig::OpenRouter { api_key, model } => {
            let client: openrouter::Client = openrouter::Client::new(api_key)?;
            run_with_client(client, model, config, &preamble, user_prompt, None).await
        }
    }
}

/// Simple completion: no tools, no hooks. For internal use (e.g. compaction).
pub async fn run_completion(config: &Config, preamble: &str, prompt: &str) -> Result<String> {
    match &config.provider {
        ProviderConfig::Ollama { url, model } => {
            let client: ollama::Client = ollama::Client::builder()
                .api_key(Nothing)
                .base_url(url)
                .build()?;
            let agent = client.agent(model).preamble(preamble).build();
            let response = agent.prompt(prompt).await?;
            Ok(response)
        }
        ProviderConfig::OpenRouter { api_key, model } => {
            let client: openrouter::Client = openrouter::Client::new(api_key)?;
            let agent = client.agent(model).preamble(preamble).build();
            let response = agent.prompt(prompt).await?;
            Ok(response)
        }
    }
}

/// The `once` CLI command: run prompt, print JSON, save turn.
pub async fn run(config: &Config, user_prompt: &str) -> Result<()> {
    let result = run_prompt(config, user_prompt).await?;

    let output = json!({
        "prompt": user_prompt,
        "answer": result.answer,
        "slug": result.slug,
        "skipped": result.skipped,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    save_turn(config, user_prompt, &result)?;
    Ok(())
}

pub fn sanitize_slug(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect()
}

fn save_turn(config: &Config, prompt: &str, result: &PromptResult) -> Result<()> {
    let turns_dir = config.references_dir().join("turns");
    fs::create_dir_all(&turns_dir)?;

    let timestamp = chrono::Utc::now();
    let time_str = timestamp.format("%Y%m%d-%H%M%S").to_string();
    let filename = format!("{}-{}.json", time_str, result.slug);

    let mut lines = Vec::new();

    lines.push(serde_json::to_string(&json!({
        "type": "prompt",
        "content": prompt,
        "timestamp": timestamp.to_rfc3339(),
    }))?);

    for event in &result.events {
        lines.push(serde_json::to_string(event)?);
    }

    lines.push(serde_json::to_string(&json!({
        "type": "answer",
        "content": result.answer,
        "slug": result.slug,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))?);

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = turns_dir.join(&tmp_name);
    let final_path = turns_dir.join(&filename);

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    eprintln!("Saved turn: {}", final_path.display());
    Ok(())
}
