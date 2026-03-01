use anyhow::{Result, bail};
use rig::client::{CompletionClient, Nothing};
use rig::completion::{Prompt, PromptError};
use rig::providers::{ollama, openrouter};
use serde_json::json;
use std::fs;

use crate::config::{Config, ProviderConfig};
use crate::context;
use crate::tools::{
    AgentEvent, AgentHook, AgentState, FinalAnswerData, FinalAnswerTool, LookupReferenceTool,
    NoResponseTool, RandomLetterTool, RandomNumberTool,
};

/// Result of running a single prompt through the agent.
pub struct PromptResult {
    pub answer: String,
    pub slug: String,
    pub skipped: bool,
    pub events: Vec<AgentEvent>,
}

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

    let final_answer = FinalAnswerTool {
        state: state.clone(),
    };
    let no_response = NoResponseTool {
        state: state.clone(),
    };
    let lookup_ref = LookupReferenceTool {
        references_dir: config.references_dir.clone(),
    };

    let mut builder = client
        .agent(model)
        .preamble(preamble)
        .tool(final_answer)
        .tool(no_response)
        .tool(lookup_ref)
        .tool(RandomNumberTool)
        .tool(RandomLetterTool)
        .default_max_turns(config.max_iterations);

    if let Some(params) = additional_params {
        builder = builder.additional_params(params);
    }

    let agent = builder.build();

    let hook = AgentHook {
        state: state.clone(),
    };
    let result = agent.prompt(user_prompt).with_hook(hook).await;

    match result {
        Err(PromptError::PromptCancelled { .. }) => {
            let mut st = state.lock().unwrap();
            if let Some(reason) = st.no_response.take() {
                let events = std::mem::take(&mut st.events);
                Ok(PromptResult {
                    answer: reason,
                    slug: "no-response".to_string(),
                    skipped: true,
                    events,
                })
            } else if let Some(mut data) = st.final_answer.take() {
                data.slug = sanitize_slug(&data.slug);
                let events = std::mem::take(&mut st.events);
                Ok(PromptResult {
                    answer: data.answer,
                    slug: data.slug,
                    skipped: false,
                    events,
                })
            } else {
                bail!("hook fired but no data captured");
            }
        }
        Ok(text) => {
            let mut st = state.lock().unwrap();
            let data = st.final_answer.take().unwrap_or_else(|| {
                let (answer, slug) = extract_answer_from_text(&text);
                FinalAnswerData { answer, slug }
            });
            let events = std::mem::take(&mut st.events);
            Ok(PromptResult {
                answer: data.answer,
                slug: data.slug,
                skipped: false,
                events,
            })
        }
        Err(PromptError::MaxTurnsError { max_turns, .. }) => {
            bail!("reached max turns limit ({max_turns})");
        }
        Err(e) => {
            bail!("{e}");
        }
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

/// Parse the LLM's raw text when it tried to "call" final_answer inline rather
/// than via a proper tool call. Strips the final_answer block and extracts slug.
fn extract_answer_from_text(text: &str) -> (String, String) {
    let lines: Vec<&str> = text.lines().collect();

    // Find where the LLM started emitting "final_answer" as plain text
    let fa_index = lines
        .iter()
        .position(|l| l.trim().eq_ignore_ascii_case("final_answer"));

    if let Some(idx) = fa_index {
        let answer = lines[..idx].join("\n").trim().to_string();

        // Look for a slug line in the tail after "final_answer"
        let tail = &lines[idx..];
        let slug = tail
            .iter()
            .rev()
            .find_map(|l| {
                let t = l.trim();
                if let Some(s) = t.strip_prefix("slug:") {
                    let s = sanitize_slug(s.trim());
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
                if t.contains('-') && !t.contains(' ') && t.len() < 60 {
                    let s = sanitize_slug(t);
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
                None
            })
            .unwrap_or_else(|| slugify_prompt(&answer));

        let answer = if answer.is_empty() {
            text.trim().to_string()
        } else {
            answer
        };
        return (answer, slug);
    }

    // No "final_answer" block — look for a trailing "slug:" line
    for i in (0..lines.len()).rev() {
        let trimmed = lines[i].trim();
        if let Some(slug) = trimmed.strip_prefix("slug:") {
            let slug = sanitize_slug(slug.trim());
            if !slug.is_empty() {
                let answer = lines[..i].join("\n").trim().to_string();
                let answer = if answer.is_empty() {
                    text.trim().to_string()
                } else {
                    answer
                };
                return (answer, slug);
            }
        }
    }

    (text.trim().to_string(), slugify_prompt(text))
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

fn slugify_prompt(text: &str) -> String {
    let slug: String = text
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    if slug.is_empty() {
        "response".to_string()
    } else {
        slug
    }
}

fn save_turn(config: &Config, prompt: &str, result: &PromptResult) -> Result<()> {
    let turns_dir = config.references_dir.join("turns");
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
