use anyhow::Result;
use notify_debouncer_full::{DebounceEventResult, new_debouncer, notify::RecursiveMode};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use crate::actions;
use crate::config::Config;

const DEBOUNCE_SECS: u64 = 1;

fn is_hidden_path(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

pub fn run(config: &Config) -> Result<()> {
    // Run all actions once at startup
    eprintln!("Running initial actions...");
    let n = actions::run_all(config)?;
    if n > 0 {
        eprintln!("{n} action(s) produced changes");
    }

    let (tx, rx) = mpsc::channel();

    let mut debouncer = new_debouncer(
        Duration::from_secs(DEBOUNCE_SECS),
        None,
        move |result: DebounceEventResult| {
            let _ = tx.send(result);
        },
    )?;

    for dir in [
        &config.system_dir,
        &config.dynamic_dir,
        &config.references_dir,
    ] {
        if dir.exists() {
            debouncer.watch(dir, RecursiveMode::Recursive)?;
            eprintln!("Watching: {}", dir.display());
        } else {
            eprintln!("Directory not found, skipping: {}", dir.display());
        }
    }

    eprintln!("Dream mode active. Watching for changes...");

    for result in rx {
        match result {
            Ok(events) => {
                let dominated_events = events
                    .into_iter()
                    .filter(|e| e.event.paths.iter().any(|p| !is_hidden_path(p)));
                let has_relevant = dominated_events.count() > 0;
                if has_relevant {
                    eprintln!("Changes detected, running actions...");
                    if let Err(e) = actions::run_all(config) {
                        eprintln!("Action error: {e}");
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
