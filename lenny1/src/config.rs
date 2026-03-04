use anyhow::Result;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    #[serde(rename = "ollama")]
    Ollama {
        #[serde(default = "default_ollama_url")]
        url: String,
        #[serde(default = "default_model")]
        model: String,
    },
    #[serde(rename = "openrouter")]
    OpenRouter { api_key: String, model: String },
}

impl ProviderConfig {
    /// Dummy provider for tests that don't call the LLM.
    #[allow(dead_code)]
    pub fn test_default() -> Self {
        ProviderConfig::Ollama {
            url: default_ollama_url(),
            model: default_model(),
        }
    }

    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        match self {
            ProviderConfig::Ollama { model, .. } => model,
            ProviderConfig::OpenRouter { model, .. } => model,
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
    #[serde(default)]
    pub thinking: bool,
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
}

fn default_model() -> String {
    "qwen3:1.7b".to_string()
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
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
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
