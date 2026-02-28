use anyhow::Result;
use notify::{Event, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;

use crate::config::Config;

/// Returns true if any component of the path starts with '.'
fn is_hidden_path(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

fn should_ignore_event(event: &Event) -> bool {
    event.paths.iter().all(|p| is_hidden_path(p))
}

pub fn run(config: &Config) -> Result<()> {
    let (tx, rx) = mpsc::channel::<notify::Result<Event>>();

    let mut watcher = notify::recommended_watcher(tx)?;

    // Watch configured directories if they exist
    for dir in [
        &config.system_dir,
        &config.dynamic_dir,
        &config.references_dir,
    ] {
        if dir.exists() {
            watcher.watch(dir, RecursiveMode::Recursive)?;
            eprintln!("Watching: {}", dir.display());
        } else {
            eprintln!("Directory not found, skipping: {}", dir.display());
        }
    }

    eprintln!("Dream mode active. Watching for changes...");

    for result in rx {
        match result {
            Ok(event) => {
                if !should_ignore_event(&event) {
                    println!("Change detected: {event:?}");
                }
            }
            Err(e) => {
                eprintln!("Watch error: {e}");
            }
        }
    }

    Ok(())
}
