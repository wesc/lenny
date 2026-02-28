use anyhow::Result;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub model: String,
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
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
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
            model: default_model(),
            max_iterations: default_max_iterations(),
            thinking: false,
            system_dir: default_system_dir(),
            dynamic_dir: default_dynamic_dir(),
            references_dir: default_references_dir(),
            ollama_url: default_ollama_url(),
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
