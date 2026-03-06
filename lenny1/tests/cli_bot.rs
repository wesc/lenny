use lenny1::cli_bot::{save_chat_file, save_dynamic_chat};
use lenny1::config::{Config, ProviderConfig};
use std::fs;

fn make_config(tmpdir: &tempfile::TempDir) -> Config {
    Config {
        provider: ProviderConfig::test_default(),
        max_iterations: 1,
        system_dir: tmpdir.path().join("system"),
        dynamic_dir: tmpdir.path().join("dynamic"),
        knowledge_dir: tmpdir.path().join("knowledge"),
        max_context_tokens: 4000,
        min_context_tokens: 2000,
        matrix: None,
        min_score_range: 2.0,
        score_gap_threshold: 0.5,
        prompt_log: false,
        model_candidates: Vec::new(),
    }
}

fn chat_line(ts: i64, sender: &str, body: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "timestamp": ts,
        "sender": sender,
        "body": body,
    }))
    .unwrap()
}

#[test]
fn writes_valid_ndjson() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let lines = vec![
        chat_line(1740000000, "user", "hello"),
        chat_line(1740000001, "lennybot", "hi there"),
    ];

    save_chat_file(&config, "cli-test", &lines).unwrap();

    let path = config.references_dir().join("chats/cli-test.json");
    let content = fs::read_to_string(&path).unwrap();
    let parsed: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0]["sender"], "user");
    assert_eq!(parsed[0]["body"], "hello");
    assert_eq!(parsed[0]["timestamp"], 1740000000);
    assert_eq!(parsed[1]["sender"], "lennybot");
    assert_eq!(parsed[1]["body"], "hi there");
    assert_eq!(parsed[1]["timestamp"], 1740000001);
}

#[test]
fn accumulates_lines() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let mut lines = vec![
        chat_line(1740000000, "user", "first"),
        chat_line(1740000001, "lennybot", "reply1"),
    ];
    save_chat_file(&config, "cli-accum", &lines).unwrap();

    let path = config.references_dir().join("chats/cli-accum.json");
    let content = fs::read_to_string(&path).unwrap();
    let count = content.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(count, 2);

    // Add more lines and rewrite
    lines.push(chat_line(1740000010, "user", "second"));
    lines.push(chat_line(1740000011, "lennybot", "reply2"));
    save_chat_file(&config, "cli-accum", &lines).unwrap();

    let content = fs::read_to_string(&path).unwrap();
    let count = content.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(count, 4);
}

#[test]
fn creates_chats_dir() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    // references/chats/ does not exist yet
    assert!(!config.references_dir().join("chats").exists());

    let lines = vec![chat_line(1740000000, "user", "hi")];
    save_chat_file(&config, "cli-mkdir", &lines).unwrap();

    assert!(
        config
            .references_dir()
            .join("chats/cli-mkdir.json")
            .exists()
    );
}

#[test]
fn atomic_write_no_temp_files_left() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let lines = vec![
        chat_line(1740000000, "user", "hello"),
        chat_line(1740000001, "lennybot", "world"),
    ];
    save_chat_file(&config, "cli-atomic", &lines).unwrap();

    let chats_dir = config.references_dir().join("chats");
    let entries: Vec<_> = fs::read_dir(&chats_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    // Only the final file should exist, no .tmp- leftovers
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], "cli-atomic.json");
}

// --- save_dynamic_chat tests ---

#[test]
fn dynamic_writes_valid_ndjson() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let lines = vec![
        chat_line(1740000000, "user", "hello"),
        chat_line(1740000001, "lennybot", "hi there"),
    ];

    save_dynamic_chat(&config, "cli-bot", "cli-test", &lines).unwrap();

    let path = config.dynamic_dir.join("cli-bot/cli-test.json");
    let content = fs::read_to_string(&path).unwrap();
    let parsed: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0]["sender"], "user");
    assert_eq!(parsed[0]["body"], "hello");
    assert_eq!(parsed[1]["sender"], "lennybot");
    assert_eq!(parsed[1]["body"], "hi there");
}

#[test]
fn dynamic_accumulates_lines() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let mut lines = vec![
        chat_line(1740000000, "user", "first"),
        chat_line(1740000001, "lennybot", "reply1"),
    ];
    save_dynamic_chat(&config, "cli-bot", "cli-accum", &lines).unwrap();

    let path = config.dynamic_dir.join("cli-bot/cli-accum.json");
    let content = fs::read_to_string(&path).unwrap();
    let count = content.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(count, 2);

    lines.push(chat_line(1740000010, "user", "second"));
    lines.push(chat_line(1740000011, "lennybot", "reply2"));
    save_dynamic_chat(&config, "cli-bot", "cli-accum", &lines).unwrap();

    let content = fs::read_to_string(&path).unwrap();
    let count = content.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(count, 4);
}

#[test]
fn dynamic_creates_subdir() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    assert!(!config.dynamic_dir.join("cli-bot").exists());

    let lines = vec![chat_line(1740000000, "user", "hi")];
    save_dynamic_chat(&config, "cli-bot", "cli-mkdir", &lines).unwrap();

    assert!(config.dynamic_dir.join("cli-bot/cli-mkdir.json").exists());
}

#[test]
fn dynamic_atomic_write_no_temp_files_left() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    let lines = vec![
        chat_line(1740000000, "user", "hello"),
        chat_line(1740000001, "lennybot", "world"),
    ];
    save_dynamic_chat(&config, "cli-bot", "cli-atomic", &lines).unwrap();

    let target_dir = config.dynamic_dir.join("cli-bot");
    let entries: Vec<_> = fs::read_dir(&target_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], "cli-atomic.json");
}
