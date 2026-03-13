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
        /// Fallback model name (lowest priority).
        #[serde(default)]
        model: Option<String>,
        /// Primary model for tool-calling and response generation.
        #[serde(default)]
        agent_model: Option<String>,
        #[serde(default)]
        comprehension_model: Option<String>,
    },
}

impl ProviderConfig {
    /// Dummy provider for tests that don't call the LLM.
    #[allow(dead_code)]
    pub fn test_default() -> Self {
        ProviderConfig::OpenRouter {
            api_key: "test-key".to_string(),
            model: Some("test-model".to_string()),
            agent_model: None,
            comprehension_model: None,
        }
    }

    /// Primary model for tool-calling and response generation.
    /// Falls back to `model` if `agent_model` is not set.
    pub fn agent_model(&self) -> &str {
        let ProviderConfig::OpenRouter {
            model, agent_model, ..
        } = self;
        agent_model
            .as_deref()
            .or(model.as_deref())
            .expect("no agent_model or model configured")
    }

    /// Model for fact extraction / comprehension. Falls back to `agent_model()`.
    pub fn comprehension_model(&self) -> &str {
        let ProviderConfig::OpenRouter {
            comprehension_model,
            ..
        } = self;
        comprehension_model
            .as_deref()
            .unwrap_or_else(|| self.agent_model())
    }

    /// Short display string like "openrouter/model-name".
    pub fn display_short(&self) -> String {
        format!("openrouter/{}", self.agent_model())
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
    #[serde(default)]
    pub firecrawl_api_key: Option<String>,
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

    pub fn memory_db(&self) -> PathBuf {
        self.knowledge_dir.join("memory.db")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let config: Config = serde_yaml::from_reader(file)?;
        Ok(config)
    }
}
