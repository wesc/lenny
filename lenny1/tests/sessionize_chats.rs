use lenny1::actions::sessionize_chats;
use lenny1::config::{Config, ProviderConfig};
use std::fs;

fn make_config(tmpdir: &tempfile::TempDir) -> Config {
    let knowledge = tmpdir.path().join("knowledge");
    let dynamic = tmpdir.path().join("dynamic");
    fs::create_dir_all(knowledge.join("references/chats")).unwrap();
    fs::create_dir_all(&dynamic).unwrap();

    Config {
        provider: ProviderConfig::default(),
        max_iterations: 1,
        thinking: false,
        system_dir: tmpdir.path().join("system"),
        dynamic_dir: dynamic,
        knowledge_dir: knowledge,
        max_context_tokens: 4000,
        min_context_tokens: 2000,
        matrix: None,
    }
}

fn now_ts() -> i64 {
    chrono::Utc::now().timestamp()
}

fn write_ndjson(dir: &std::path::Path, filename: &str, lines: &[serde_json::Value]) {
    fs::create_dir_all(dir).unwrap();
    let content: String = lines
        .iter()
        .map(|v| serde_json::to_string(v).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(dir.join(filename), content).unwrap();
}

fn write_chat(config: &Config, filename: &str, lines: &[serde_json::Value]) {
    write_ndjson(&config.references_dir().join("chats"), filename, lines);
}

fn read_session(config: &Config) -> Vec<serde_json::Value> {
    let path = config.dynamic_dir.join("10-session.json");
    let content = fs::read_to_string(path).unwrap();
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn basic_sessionize() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    write_chat(
        &config,
        "room.json",
        &[
            serde_json::json!({"timestamp": now - 100, "sender": "alice", "body": "hello"}),
            serde_json::json!({"timestamp": now - 50, "sender": "bob", "body": "hi there"}),
            serde_json::json!({"timestamp": now, "sender": "alice", "body": "how are you"}),
        ],
    );

    let changed = sessionize_chats::run(&config).unwrap();
    assert!(changed, "should report a change");

    let session = read_session(&config);
    assert_eq!(session.len(), 3);
    // Verify sorted by timestamp
    assert_eq!(session[0]["sender"], "alice");
    assert_eq!(session[1]["sender"], "bob");
    assert_eq!(session[2]["sender"], "alice");
}

#[test]
fn filters_old_messages() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();
    let two_weeks_ago = now - 14 * 24 * 60 * 60;

    write_chat(
        &config,
        "room.json",
        &[
            serde_json::json!({"timestamp": two_weeks_ago, "sender": "old", "body": "ancient message"}),
            serde_json::json!({"timestamp": now - 10, "sender": "recent", "body": "fresh message"}),
        ],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 1);
    assert_eq!(session[0]["sender"], "recent");
}

#[test]
fn respects_token_limit() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    // Create lines with ~200 words each. 6 lines = ~1200 words, exceeding the 1000 token limit.
    let fat_body = "word ".repeat(200);
    let mut lines = Vec::new();
    for i in 0..6 {
        lines.push(serde_json::json!({
            "timestamp": now - 600 + i * 100,
            "sender": format!("user{i}"),
            "body": fat_body.trim(),
        }));
    }

    write_chat(&config, "room.json", &lines);
    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    // Should have kept only the most recent lines that fit in ~1000 tokens.
    // Each line is ~200+ words (body + json overhead), so we expect ~4-5 lines.
    assert!(
        session.len() < 6,
        "should have truncated, got {}",
        session.len()
    );
    assert!(!session.is_empty(), "should have kept some lines");
    // The kept lines should be the most recent ones
    let last = &session[session.len() - 1];
    assert_eq!(last["sender"], "user5", "most recent should be kept");
}

#[test]
fn merges_multiple_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    write_chat(
        &config,
        "room1.json",
        &[serde_json::json!({"timestamp": now - 200, "sender": "alice", "body": "from room1"})],
    );
    write_chat(
        &config,
        "room2.json",
        &[serde_json::json!({"timestamp": now - 100, "sender": "bob", "body": "from room2"})],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 2);
    // Should be sorted by timestamp: room1 first, then room2
    assert_eq!(session[0]["sender"], "alice");
    assert_eq!(session[1]["sender"], "bob");
}

#[test]
fn skips_hidden_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    write_chat(
        &config,
        ".hidden.json",
        &[serde_json::json!({"timestamp": now, "sender": "hidden", "body": "should not appear"})],
    );
    write_chat(
        &config,
        "visible.json",
        &[serde_json::json!({"timestamp": now, "sender": "visible", "body": "should appear"})],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 1);
    assert_eq!(session[0]["sender"], "visible");
}

#[test]
fn no_change_when_content_same() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    write_chat(
        &config,
        "room.json",
        &[serde_json::json!({"timestamp": now, "sender": "alice", "body": "hello"})],
    );

    let first = sessionize_chats::run(&config).unwrap();
    assert!(first, "first run should change");

    let second = sessionize_chats::run(&config).unwrap();
    assert!(!second, "second run with same data should not change");
}

#[test]
fn no_chats_dir_returns_false() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic = tmpdir.path().join("dynamic");
    fs::create_dir_all(&dynamic).unwrap();

    let config = Config {
        provider: ProviderConfig::default(),
        max_iterations: 1,
        thinking: false,
        system_dir: tmpdir.path().join("system"),
        dynamic_dir: dynamic,
        knowledge_dir: tmpdir.path().join("knowledge"), // no chats/ subdir
        max_context_tokens: 4000,
        min_context_tokens: 2000,
        matrix: None,
    };

    let changed = sessionize_chats::run(&config).unwrap();
    assert!(!changed);
}

#[test]
fn normalizes_millisecond_timestamps() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();
    let now_ms = now * 1000; // Matrix-style millisecond timestamp

    // Write a message with ms timestamp
    write_chat(
        &config,
        "matrix-room.json",
        &[serde_json::json!({"timestamp": now_ms, "sender": "alice", "body": "ms timestamp"})],
    );
    // Write a message with normal seconds timestamp
    write_chat(
        &config,
        "other-room.json",
        &[serde_json::json!({"timestamp": now - 10, "sender": "bob", "body": "sec timestamp"})],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 2);
    // bob (now-10) should come before alice (now)
    assert_eq!(session[0]["sender"], "bob");
    assert_eq!(session[1]["sender"], "alice");
}

#[test]
fn reads_matrix_messages() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();
    let now_ms = now * 1000;

    // Write to references/matrix/example.org/room.json
    let matrix_host_dir = config.references_dir().join("matrix").join("example.org");
    write_ndjson(
        &matrix_host_dir,
        "room.json",
        &[serde_json::json!({
            "timestamp": now_ms - 5000,
            "sender": "@alice:example.org",
            "body": "from matrix host 1",
            "room": "test-room",
        })],
    );

    // Write to references/matrix/other.org/room.json
    let matrix_host_dir2 = config.references_dir().join("matrix").join("other.org");
    write_ndjson(
        &matrix_host_dir2,
        "room.json",
        &[serde_json::json!({
            "timestamp": now_ms,
            "sender": "@bob:other.org",
            "body": "from matrix host 2",
            "room": "other-room",
        })],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 2);
    // First host message (now-5s) should come before second host message (now)
    assert_eq!(session[0]["sender"], "@alice:example.org");
    assert_eq!(session[1]["sender"], "@bob:other.org");
}

#[test]
fn deduplicates_by_id() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    // Two lines with the same id but different timestamps/bodies
    write_chat(
        &config,
        "room.json",
        &[
            serde_json::json!({"id": "dup-1", "timestamp": now - 100, "sender": "alice", "body": "first"}),
            serde_json::json!({"id": "dup-1", "timestamp": now - 50, "sender": "alice", "body": "duplicate"}),
            serde_json::json!({"id": "dup-2", "timestamp": now, "sender": "bob", "body": "unique"}),
        ],
    );

    sessionize_chats::run(&config).unwrap();

    let session = read_session(&config);
    assert_eq!(session.len(), 2, "duplicate id should be removed");
    assert_eq!(session[0]["body"], "first", "first occurrence kept");
    assert_eq!(session[1]["body"], "unique");
}

#[test]
fn skips_hidden_matrix_host_dirs() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    let now = now_ts();

    // Hidden host dir should be skipped
    let hidden_dir = config.references_dir().join("matrix").join(".hidden-host");
    write_ndjson(
        &hidden_dir,
        "room.json",
        &[
            serde_json::json!({"timestamp": now * 1000, "sender": "hidden", "body": "should not appear"}),
        ],
    );

    let changed = sessionize_chats::run(&config).unwrap();
    assert!(!changed, "hidden matrix dir should be ignored");
}
