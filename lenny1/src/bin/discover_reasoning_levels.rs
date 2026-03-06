use anyhow::Result;
use openrouter_rs::OpenRouterClient;
use std::path::PathBuf;

use lenny1::agent::{EFFORT_LEVELS, probe_min_effort};
use lenny1::config::{Config, ProviderConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.yaml".to_string());
    let config = Config::load(PathBuf::from(&config_path).as_path())?;

    let ProviderConfig::OpenRouter { ref api_key, .. } = config.provider;
    let client = OpenRouterClient::builder().api_key(api_key).build()?;

    if config.model_candidates.is_empty() {
        eprintln!("No model_candidates configured in {config_path}");
        return Ok(());
    }

    eprintln!(
        "Probing {} models for reasoning effort support...\n",
        config.model_candidates.len()
    );

    for model in &config.model_candidates {
        eprint!("  {model} ... ");

        // Find the minimum effort level accepted
        let min = probe_min_effort(&client, model).await;

        match min {
            Ok(min_effort) => {
                // Also discover all supported levels
                let mut supported = Vec::new();
                for effort in &EFFORT_LEVELS {
                    let mut builder = openrouter_rs::api::chat::ChatCompletionRequest::builder();
                    builder
                        .model(model.as_str())
                        .messages(vec![openrouter_rs::api::chat::Message::new(
                            openrouter_rs::types::Role::User,
                            "Say OK.",
                        )])
                        .max_tokens(4u32)
                        .reasoning_effort(effort.clone());
                    if matches!(effort, openrouter_rs::types::Effort::None) {
                        builder.reasoning_max_tokens(0u32);
                    }
                    let request = builder.build()?;

                    match client.send_chat_completion(&request).await {
                        Ok(_) => supported.push(format!("{effort}")),
                        Err(_) => supported.push(format!("({effort})")),
                    }
                }
                eprintln!("min={min_effort}  levels: {}", supported.join(" "));
            }
            Err(e) => {
                eprintln!("FAILED: {e}");
            }
        }
    }

    eprintln!();
    Ok(())
}
