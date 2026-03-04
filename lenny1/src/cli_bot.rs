use anyhow::Result;
use std::fs;
use std::io::{BufRead, Write};

use crate::config::Config;
use crate::once;

/// Run an interactive chat loop, reading from `input` and writing to `output`.
///
/// Each line triggers a fresh agent call via `once::run_prompt`. Chat history
/// is persisted as NDJSON to `dynamic/cli-bot/{session_id}.json` so it appears
/// directly in assembled context.
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
        let result = match once::run_prompt(config, line).await {
            Ok(r) => r,
            Err(e) => {
                writeln!(output, "\n\x1b[38;5;245m[no response: {e}]\x1b[0m\n")?;
                continue;
            }
        };

        lines.push(serde_json::to_string(&serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "timestamp": ts,
            "sender": "user",
            "body": line,
        }))?);

        if !result.skipped {
            writeln!(output, "\n\x1b[38;5;245m{}\x1b[0m\n", result.answer)?;

            lines.push(serde_json::to_string(&serde_json::json!({
                "id": uuid::Uuid::new_v4().to_string(),
                "timestamp": chrono::Utc::now().timestamp(),
                "sender": "lennybot",
                "body": result.answer,
            }))?);
        }

        save_dynamic_chat(config, "cli-bot", &session_id, &lines)?;
    }

    Ok(())
}

/// Atomically write accumulated NDJSON chat lines to `references/chats/{session_id}.json`.
pub fn save_chat_file(config: &Config, session_id: &str, lines: &[String]) -> Result<()> {
    let chats_dir = config.references_dir().join("chats");
    fs::create_dir_all(&chats_dir)?;

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = chats_dir.join(&tmp_name);
    let final_path = chats_dir.join(format!("{session_id}.json"));

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    Ok(())
}

/// Atomically write accumulated NDJSON chat lines to `dynamic/{subdir}/{session_id}.json`.
pub fn save_dynamic_chat(
    config: &Config,
    subdir: &str,
    session_id: &str,
    lines: &[String],
) -> Result<()> {
    let target_dir = config.dynamic_dir.join(subdir);
    fs::create_dir_all(&target_dir)?;

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = target_dir.join(&tmp_name);
    let final_path = target_dir.join(format!("{session_id}.json"));

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    Ok(())
}
