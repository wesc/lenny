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

impl Default for ProviderConfig {
    fn default() -> Self {
        ProviderConfig::Ollama {
            url: default_ollama_url(),
            model: default_model(),
        }
    }
}

impl ProviderConfig {
    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        match self {
            ProviderConfig::Ollama { model, .. } => model,
            ProviderConfig::OpenRouter { model, .. } => model,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub username: String,
    pub password: String,
    pub store_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
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
    #[serde(default = "default_min_rerank_score")]
    pub min_rerank_score: f32,
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
fn default_min_rerank_score() -> f32 {
    0.01
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: ProviderConfig::default(),
            max_iterations: default_max_iterations(),
            thinking: false,
            system_dir: default_system_dir(),
            dynamic_dir: default_dynamic_dir(),
            knowledge_dir: default_knowledge_dir(),
            max_context_tokens: default_max_context_tokens(),
            min_context_tokens: default_min_context_tokens(),
            matrix: None,
            min_rerank_score: default_min_rerank_score(),
        }
    }
}

impl Config {
    pub fn references_dir(&self) -> PathBuf {
        self.knowledge_dir.join("references")
    }

    pub fn comprehensions_dir(&self) -> PathBuf {
        self.knowledge_dir.join("comprehensions")
    }

    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let file = File::open(path)?;
            let config: Config = serde_yaml::from_reader(file)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }
}
