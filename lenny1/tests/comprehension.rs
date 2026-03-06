use lenny1::actions::comprehension::{
    collect_dynamic_files, move_to_references, select_files_to_pop,
};
use lenny1::config::{Config, ProviderConfig};
use std::fs;

fn make_config(tmpdir: &tempfile::TempDir) -> Config {
    Config {
        provider: ProviderConfig::test_default(),
        max_iterations: 1,
        system_dir: tmpdir.path().join("system"),
        dynamic_dir: tmpdir.path().join("dynamic"),
        knowledge_dir: tmpdir.path().join("knowledge"),
        max_context_tokens: 100,
        min_context_tokens: 50,
        matrix: None,
        min_score_range: 2.0,
        score_gap_threshold: 0.5,
        prompt_log: false,
        model_candidates: Vec::new(),
    }
}

/// Write a file with approximately `n_words` words to dynamic/.
fn write_dynamic_file(config: &Config, relative: &str, n_words: usize) {
    let path = config.dynamic_dir.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let content: String = (0..n_words)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(path, content).unwrap();
}

#[test]
fn below_threshold_noop() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    // Write 50 words — under the max_context_tokens of 100
    write_dynamic_file(&config, "a-small.txt", 50);

    let files = collect_dynamic_files(&config).unwrap();
    let total: usize = files.iter().map(|f| f.tokens).sum();
    assert!(total <= config.max_context_tokens);

    let (to_pop, keep) =
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens);
    assert!(to_pop.is_empty());
    assert_eq!(keep.len(), 1);
}

#[test]
fn pops_oldest_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    // Create 4 files with 40 words each = 160 total, over max of 100
    // Lexicographic order determines age: 01 is oldest
    write_dynamic_file(&config, "01-first.txt", 40);
    write_dynamic_file(&config, "02-second.txt", 40);
    write_dynamic_file(&config, "03-third.txt", 40);
    write_dynamic_file(&config, "04-fourth.txt", 40);

    let files = collect_dynamic_files(&config).unwrap();
    let total: usize = files.iter().map(|f| f.tokens).sum();
    assert!(total > config.max_context_tokens);

    let (to_pop, keep) =
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens);

    // Need to pop until remaining <= 50 (min_context_tokens)
    // 160 total, each 40 words: pop 3 files (remaining = 40 <= 50)
    assert!(!to_pop.is_empty());
    let remaining: usize = keep.iter().map(|f| f.tokens).sum();
    assert!(
        remaining <= config.min_context_tokens,
        "remaining {remaining} > min {}",
        config.min_context_tokens
    );

    // The oldest files should be popped
    let popped_names: Vec<String> = to_pop
        .iter()
        .map(|f| f.relative.display().to_string())
        .collect();
    assert!(popped_names.contains(&"01-first.txt".to_string()));
}

#[test]
fn preserves_subdir_structure() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    fs::create_dir_all(config.references_dir()).unwrap();

    // Create a file in a subdirectory
    write_dynamic_file(&config, "cli-bot/session.json", 40);

    let files = collect_dynamic_files(&config).unwrap();

    // Manually move to references to test structure preservation
    move_to_references(&config, &files).unwrap();

    let dest = config.references_dir().join("cli-bot/session.json");
    assert!(
        dest.exists(),
        "file should exist at references/cli-bot/session.json"
    );

    let src = config.dynamic_dir.join("cli-bot/session.json");
    assert!(!src.exists(), "file should be removed from dynamic/");
}

#[test]
fn atomic_write_no_temp_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    fs::create_dir_all(config.references_dir()).unwrap();

    write_dynamic_file(&config, "01-old.txt", 40);
    write_dynamic_file(&config, "02-new.txt", 40);

    let files = collect_dynamic_files(&config).unwrap();
    move_to_references(&config, &files).unwrap();

    // No temp files should remain in references/
    let entries: Vec<String> = fs::read_dir(config.references_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    for entry in &entries {
        assert!(
            !entry.starts_with(".tmp-"),
            "temp file left behind: {entry}"
        );
    }
}

#[test]
fn writes_comprehension_with_llm() {
    // This test requires a live LLM — skip in CI
    // Tested manually: populate dynamic/ over threshold, run comprehension, verify
    // comprehension file appears in comprehensions/
}
