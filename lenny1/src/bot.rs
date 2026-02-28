use anyhow::Result;
use std::fs;
use std::io::{BufRead, Write};

use crate::config::Config;
use crate::once;

/// Run an interactive chat loop, reading from `input` and writing to `output`.
///
/// Each line triggers a fresh agent call via `once::run_prompt`. Chat history
/// is persisted as NDJSON to `references/chats/{session_id}.json` after each
/// turn so the dream watcher can sessionize it into dynamic context before the
/// user's next input.
pub async fn chat_loop<R: BufRead, W: Write>(
    config: &Config,
    input: &mut R,
    output: &mut W,
) -> Result<()> {
    let session_id = format!("cli-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    let mut lines: Vec<String> = Vec::new();

    loop {
        write!(output, "> ")?;
        output.flush()?;

        let mut line = String::new();
        let n = input.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        let ts = chrono::Utc::now().timestamp();
        let result = once::run_prompt(config, line).await?;

        writeln!(output, "{}", result.answer)?;

        lines.push(serde_json::to_string(&serde_json::json!({
            "timestamp": ts,
            "sender": "user",
            "body": line,
        }))?);
        lines.push(serde_json::to_string(&serde_json::json!({
            "timestamp": chrono::Utc::now().timestamp(),
            "sender": "lennybot",
            "body": result.answer,
        }))?);

        save_chat_file(config, &session_id, &lines)?;
    }

    Ok(())
}

/// Atomically write accumulated NDJSON chat lines to `references/chats/{session_id}.json`.
pub fn save_chat_file(config: &Config, session_id: &str, lines: &[String]) -> Result<()> {
    let chats_dir = config.references_dir.join("chats");
    fs::create_dir_all(&chats_dir)?;

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = chats_dir.join(&tmp_name);
    let final_path = chats_dir.join(format!("{session_id}.json"));

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    Ok(())
}
