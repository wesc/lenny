use anyhow::Result;
use std::time::Instant;

use crate::config::Config;

pub async fn run(config: &Config) -> Result<()> {
    let start = Instant::now();
    let mut suites_passed = 0;
    let mut suites_failed = 0;

    eprintln!("\n=== eval basic ===\n");
    match super::run(config).await {
        Ok(()) => suites_passed += 1,
        Err(e) => {
            eprintln!("  FAILED: {e}");
            suites_failed += 1;
        }
    }

    eprintln!("\n=== eval contextual-chats ===\n");
    match super::contextual_chats::run(config).await {
        Ok(()) => suites_passed += 1,
        Err(e) => {
            eprintln!("  FAILED: {e}");
            suites_failed += 1;
        }
    }

    eprintln!("\n=== eval contextual-texts ===\n");
    match super::contextual_texts::run(config).await {
        Ok(()) => suites_passed += 1,
        Err(e) => {
            eprintln!("  FAILED: {e}");
            suites_failed += 1;
        }
    }

    eprintln!("\n=== eval fact ===\n");
    match super::fact::run(config).await {
        Ok(()) => suites_passed += 1,
        Err(e) => {
            eprintln!("  FAILED: {e}");
            suites_failed += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total = suites_passed + suites_failed;
    eprintln!("\n=== ALL EVALS: {suites_passed}/{total} suites passed in {elapsed:.1}s ===");

    if suites_failed > 0 {
        anyhow::bail!("{suites_failed}/{total} eval suites failed");
    }

    Ok(())
}
