pub mod comprehension;
pub mod sessionize_chats;

use anyhow::Result;

use crate::config::Config;

type Action = (&'static str, fn(&Config) -> Result<bool>);

/// Run all dream actions. Returns the number of actions that produced changes.
pub fn run_all(config: &Config) -> Result<usize> {
    let actions: &[Action] = &[("sessionize-chats", sessionize_chats::run)];

    let mut changed = 0;
    for (name, action) in actions {
        match action(config) {
            Ok(true) => {
                eprintln!("  [{name}] updated");
                changed += 1;
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("  [{name}] error: {e}");
            }
        }
    }
    Ok(changed)
}
