use anyhow::Result;
use std::io::{BufRead, Write};

use crate::config::Config;
use crate::once::{self, PromptHooks};
use crate::session::SessionId;

struct CliHook;

impl PromptHooks for CliHook {
    fn on_tool_start(&mut self, name: &str, args: &str) {
        eprintln!("\x1b[38;5;245m  → {name}({args})\x1b[0m");
    }
}

/// Run an interactive chat loop, reading from `input` and writing to `output`.
///
/// Each line triggers a fresh agent call via `once::run_prompt`. Chat history
/// is persisted as session NDJSON files by `run_prompt`, so it appears in
/// the agent's context on subsequent turns.
pub async fn chat_loop<R: BufRead, W: Write>(
    config: &Config,
    input: &mut R,
    output: &mut W,
) -> Result<()> {
    let session_name = chrono::Utc::now()
        .format("%Y-%m-%d_%H-%M-%S_cli")
        .to_string();
    let session_id = SessionId::new("cli", &session_name);

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

        let mut hook = CliHook;
        let result = match once::run_prompt(config, "cli", &session_id, line, Some(&mut hook)).await
        {
            Ok(r) => r,
            Err(e) => {
                writeln!(output, "\n\x1b[38;5;245m[no response: {e}]\x1b[0m\n")?;
                continue;
            }
        };

        if !result.skipped {
            writeln!(output, "\n\x1b[38;5;245m{}\x1b[0m\n", result.answer)?;
        }
    }

    Ok(())
}
