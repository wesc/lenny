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
    #[serde(default = "default_references_dir")]
    pub references_dir: PathBuf,
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
fn default_references_dir() -> PathBuf {
    PathBuf::from("data/references")
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
            references_dir: default_references_dir(),
        }
    }
}

impl Config {
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
