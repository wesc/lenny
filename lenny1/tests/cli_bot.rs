use lenny1::agent::Message;
use lenny1::session::{self, SessionId};
use std::fs;

#[test]
fn session_save_and_load_round_trip() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic_dir = tmpdir.path();
    let session_id = SessionId::new("cli", "test-session");

    let messages = vec![Message::user("hello"), Message::assistant("hi there")];

    let path = session::save_turn(dynamic_dir, &session_id, &messages, "greeting").unwrap();
    assert!(path.exists());
    assert!(path.to_string_lossy().ends_with(".ndjson"));

    let ctx = session::load_session(dynamic_dir, &session_id).unwrap();
    assert_eq!(ctx.history.len(), 2);
    assert!(ctx.documents.is_empty());
}

#[test]
fn session_creates_dir() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic_dir = tmpdir.path();
    let session_id = SessionId::new("cli", "mkdir-test");

    assert!(!dynamic_dir.join("cli/mkdir-test").exists());

    let messages = vec![Message::user("hi")];
    session::save_turn(dynamic_dir, &session_id, &messages, "test").unwrap();

    assert!(dynamic_dir.join("cli/mkdir-test").exists());
}

#[test]
fn session_atomic_write_no_temp_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic_dir = tmpdir.path();
    let session_id = SessionId::new("cli", "atomic-test");

    let messages = vec![Message::user("hello"), Message::assistant("world")];
    session::save_turn(dynamic_dir, &session_id, &messages, "test").unwrap();

    let session_dir = dynamic_dir.join("cli/atomic-test");
    let entries: Vec<_> = fs::read_dir(&session_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    // Only the final .ndjson file should exist, no .tmp- leftovers
    assert_eq!(entries.len(), 1);
    assert!(entries[0].ends_with(".ndjson"));
    assert!(!entries[0].starts_with(".tmp"));
}

#[test]
fn session_accumulates_turns() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic_dir = tmpdir.path();
    let session_id = SessionId::new("cli", "accum-test");

    // First turn
    let messages1 = vec![Message::user("first"), Message::assistant("reply1")];
    session::save_turn(dynamic_dir, &session_id, &messages1, "turn1").unwrap();

    let ctx = session::load_session(dynamic_dir, &session_id).unwrap();
    assert_eq!(ctx.history.len(), 2);

    // Second turn
    let messages2 = vec![Message::user("second"), Message::assistant("reply2")];
    session::save_turn(dynamic_dir, &session_id, &messages2, "turn2").unwrap();

    let ctx = session::load_session(dynamic_dir, &session_id).unwrap();
    assert_eq!(ctx.history.len(), 4);
}

#[test]
fn session_documents_exclude_session_dir() {
    let tmpdir = tempfile::tempdir().unwrap();
    let dynamic_dir = tmpdir.path();
    let session_id = SessionId::new("cli", "doc-test");

    // Save a session turn
    let messages = vec![Message::user("hello")];
    session::save_turn(dynamic_dir, &session_id, &messages, "test").unwrap();

    // Create a non-session file
    let notes_dir = dynamic_dir.join("notes");
    fs::create_dir_all(&notes_dir).unwrap();
    fs::write(notes_dir.join("note.md"), "some note content").unwrap();

    let ctx = session::load_session(dynamic_dir, &session_id).unwrap();
    assert_eq!(ctx.history.len(), 1);
    assert_eq!(ctx.documents.len(), 1);
    assert_eq!(ctx.documents[0].id, "notes/note.md");
}
