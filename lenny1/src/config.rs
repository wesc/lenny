use anyhow::Result;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    #[serde(rename = "openrouter")]
    OpenRouter {
        api_key: String,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        reasoning_model: Option<String>,
        #[serde(default)]
        response_model: Option<String>,
    },
}

impl ProviderConfig {
    /// Dummy provider for tests that don't call the LLM.
    #[allow(dead_code)]
    pub fn test_default() -> Self {
        ProviderConfig::OpenRouter {
            api_key: "test-key".to_string(),
            model: Some("test-model".to_string()),
            reasoning_model: None,
            response_model: None,
        }
    }

    fn model_fields(&self) -> (&Option<String>, &Option<String>, &Option<String>) {
        let ProviderConfig::OpenRouter {
            model,
            reasoning_model,
            response_model,
            ..
        } = self;
        (model, reasoning_model, response_model)
    }

    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        let (model, _, _) = self.model_fields();
        model.as_deref().unwrap_or_else(|| {
            let (_, r, resp) = self.model_fields();
            r.as_deref()
                .or(resp.as_deref())
                .expect("no model configured")
        })
    }

    /// Model for tool-calling / reasoning phase. Falls back to `model`.
    pub fn reasoning_model(&self) -> &str {
        let (model, reasoning, _) = self.model_fields();
        reasoning
            .as_deref()
            .or(model.as_deref())
            .expect("no reasoning_model or model configured")
    }

    /// Model for final response phase. Falls back to `model`.
    pub fn response_model(&self) -> &str {
        let (model, _, response) = self.model_fields();
        response
            .as_deref()
            .or(model.as_deref())
            .expect("no response_model or model configured")
    }

    /// True when separate reasoning + response models are configured.
    pub fn is_dual_model(&self) -> bool {
        let (_, reasoning, response) = self.model_fields();
        reasoning.is_some() && response.is_some()
    }

    /// Short display string like "openrouter/gpt-oss-120b:nitro" or
    /// "openrouter/gpt-oss-120b:nitro + gemini-2.5-flash-lite" for dual model.
    pub fn display_short(&self) -> String {
        if self.is_dual_model() {
            format!(
                "openrouter/{} + {}",
                self.reasoning_model(),
                self.response_model()
            )
        } else {
            format!("openrouter/{}", self.reasoning_model())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RespondTo {
    Mention,
    All,
    None,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub username: String,
    pub password: String,
    pub store_path: PathBuf,
    #[serde(default = "default_respond_to")]
    pub respond_to: RespondTo,
}

fn default_respond_to() -> RespondTo {
    RespondTo::Mention
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub provider: ProviderConfig,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_system_dir")]
    pub system_dir: PathBuf,
    #[serde(default = "default_dynamic_dir")]
    pub dynamic_dir: PathBuf,
    #[serde(default = "default_knowledge_dir")]
    pub knowledge_dir: PathBuf,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
    #[serde(default = "default_min_context_tokens")]
    pub min_context_tokens: usize,
    #[serde(default)]
    pub matrix: Option<MatrixConfig>,
    #[serde(default = "default_min_score_range")]
    pub min_score_range: f32,
    #[serde(default = "default_score_gap_threshold")]
    pub score_gap_threshold: f32,
    #[serde(default = "default_prompt_log")]
    pub prompt_log: bool,
    #[serde(default)]
    pub model_candidates: Vec<String>,
}

fn default_prompt_log() -> bool {
    true
}
fn default_max_iterations() -> usize {
    5
}
fn default_system_dir() -> PathBuf {
    PathBuf::from("data/system")
}
fn default_dynamic_dir() -> PathBuf {
    PathBuf::from("data/dynamic")
}
fn default_knowledge_dir() -> PathBuf {
    PathBuf::from("data/knowledge")
}
fn default_max_context_tokens() -> usize {
    4000
}
fn default_min_context_tokens() -> usize {
    2000
}
fn default_min_score_range() -> f32 {
    2.0
}
fn default_score_gap_threshold() -> f32 {
    0.5
}

impl Config {
    pub fn references_dir(&self) -> PathBuf {
        self.knowledge_dir.join("references")
    }

    pub fn comprehensions_dir(&self) -> PathBuf {
        self.knowledge_dir.join("comprehensions")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let config: Config = serde_yaml::from_reader(file)?;
        Ok(config)
    }
}
