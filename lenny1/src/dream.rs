use anyhow::Result;
use notify_debouncer_full::{DebounceEventResult, new_debouncer, notify::RecursiveMode};
use std::path::Path;
use std::time::Duration;

use crate::actions;
use crate::config::Config;

const DEBOUNCE_SECS: u64 = 1;

fn is_hidden_path(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

pub async fn run(config: &Config, force_digest: bool) -> Result<()> {
    // Run fact digest at startup
    if actions::fact::run(config, force_digest).await? {
        eprintln!("Initial fact digest completed");
    }

    if force_digest {
        eprintln!("Force digest completed, exiting.");
        return Ok(());
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(100);

    let mut debouncer = new_debouncer(
        Duration::from_secs(DEBOUNCE_SECS),
        None,
        move |result: DebounceEventResult| {
            let _ = tx.blocking_send(result);
        },
    )?;

    for dir in [
        &config.system_dir,
        &config.dynamic_dir,
        &config.knowledge_dir,
    ] {
        if dir.exists() {
            debouncer.watch(dir, RecursiveMode::Recursive)?;
            eprintln!("Watching: {}", dir.display());
        } else {
            eprintln!("Directory not found, skipping: {}", dir.display());
        }
    }

    eprintln!("Dream mode active. Watching for changes...");

    while let Some(result) = rx.recv().await {
        match result {
            Ok(events) => {
                let has_relevant = events
                    .iter()
                    .any(|e| e.event.paths.iter().any(|p| !is_hidden_path(p)));
                if has_relevant {
                    eprintln!("Changes detected, running fact digest...");
                    if let Err(e) = actions::fact::run(config, false).await {
                        eprintln!("Fact digest error: {e}");
                    }
                }
            }
            Err(errors) => {
                for e in errors {
                    eprintln!("Watch error: {e}");
                }
            }
        }
    }

    Ok(())
}
