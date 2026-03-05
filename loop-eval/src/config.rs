use anyhow::{Result, bail};
use openrouter_rs::types::Effort;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    provider: RawProvider,
}

#[derive(Debug, Clone, Deserialize)]
struct RawProvider {
    api_key: Option<String>,
    reasoning_model: Option<String>,
    response_model: Option<String>,
    judge_model: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub reasoning_model: String,
    pub response_model: String,
    pub judge_model: String,
    pub max_turns: usize,
    pub response_effort: Effort,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let raw: RawConfig = serde_yaml::from_reader(file)?;

        let api_key = raw
            .provider
            .api_key
            .ok_or_else(|| anyhow::anyhow!("missing api_key"))?;

        let fallback = raw.provider.model.as_deref().unwrap_or("");
        let reasoning_model = raw
            .provider
            .reasoning_model
            .unwrap_or_else(|| fallback.to_string());
        let response_model = raw
            .provider
            .response_model
            .unwrap_or_else(|| fallback.to_string());

        if reasoning_model.is_empty() {
            bail!("no reasoning_model or model configured");
        }
        if response_model.is_empty() {
            bail!("no response_model or model configured");
        }

        let judge_model = raw
            .provider
            .judge_model
            .unwrap_or_else(|| "openai/gpt-4o".to_string());

        Ok(Config {
            api_key,
            reasoning_model,
            response_model,
            judge_model,
            max_turns: 10,
            response_effort: Effort::None,
        })
    }
}
