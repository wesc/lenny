use anyhow::Result;

use crate::config::Config;

pub async fn run(base_config: &Config) -> Result<()> {
    super::contextual_chats::run(base_config).await?;
    eprintln!();
    super::contextual_texts::run(base_config).await?;
    Ok(())
}
