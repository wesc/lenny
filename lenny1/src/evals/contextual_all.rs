use anyhow::Result;

use crate::config::Config;

pub async fn run(base_config: &Config) -> Result<()> {
    let r1 = super::contextual_chats::run(base_config).await;
    eprintln!();
    let r2 = super::contextual_texts::run(base_config).await;
    r1?;
    r2?;
    Ok(())
}
