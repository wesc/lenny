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

pub async fn run(config: &Config, force_comprehension: bool) -> Result<()> {
    // Run comprehension at startup
    if actions::comprehension::run(config, force_comprehension).await? {
        eprintln!("Initial comprehension completed");
    }

    if force_comprehension {
        eprintln!("Force comprehension completed, exiting.");
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
                    eprintln!("Changes detected, running comprehension...");
                    if let Err(e) = actions::comprehension::run(config, false).await {
                        eprintln!("Comprehension error: {e}");
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
