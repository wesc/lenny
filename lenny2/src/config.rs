use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub openrouter_api_key: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub firecrawl_api_key: String,
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
}

fn default_model() -> String {
    "anthropic/claude-haiku-4.5".to_string()
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}

impl Config {
    pub fn pages_dir(&self) -> PathBuf {
        self.data_dir.join("pages")
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("lenny2.db")
    }

    pub fn load() -> Result<Self> {
        let path = Path::new("config.yaml");
        if path.exists() {
            let file = std::fs::File::open(path)?;
            let config: Config = serde_yaml::from_reader(file)?;
            Ok(config)
        } else {
            // Fall back to environment variables
            Ok(Config {
                openrouter_api_key: std::env::var("OPENROUTER_API_KEY")
                    .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?,
                model: std::env::var("LENNY2_MODEL").unwrap_or_else(|_| default_model()),
                firecrawl_api_key: std::env::var("FIRECRAWL_API_KEY")
                    .map_err(|_| anyhow::anyhow!("FIRECRAWL_API_KEY not set"))?,
                data_dir: default_data_dir(),
            })
        }
    }
}
