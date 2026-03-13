use anyhow::Result;
use rig::completion::request::Document;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::agent::Message;
use crate::context::collect_files;

/// Identifies a session by its relative path under the dynamic/ directory.
pub struct SessionId {
    /// Relative path under dynamic/, e.g. "cli/2026-03-12_10-00-00_cli"
    pub path: PathBuf,
}

impl SessionId {
    /// Construct a session ID from a subdirectory and name.
    /// e.g. `SessionId::new("cli", "2026-03-12_10-00-00_cli")`
    pub fn new(subdir: &str, name: &str) -> Self {
        Self {
            path: PathBuf::from(subdir).join(name),
        }
    }
}

/// Context loaded from disk for a session: chat history + general knowledge documents.
pub struct SessionContext {
    /// Chat history messages deserialized from session NDJSON files.
    pub history: Vec<Message>,
    /// General knowledge documents from non-session files in dynamic/.
    pub documents: Vec<Document>,
}

/// Load session state from disk.
///
/// - Reads `*.ndjson` files from `dynamic/{session_id.path}/`, sorted by filename,
///   deserializing each line as a `Message` into history.
/// - Reads all other non-hidden files under `dynamic/` (excluding the session dir)
///   and creates `Document` entries for general knowledge.
pub fn load_session(dynamic_dir: &Path, session_id: &SessionId) -> Result<SessionContext> {
    let session_dir = dynamic_dir.join(&session_id.path);
    let history = load_history(&session_dir)?;
    let documents = load_documents(dynamic_dir, &session_dir)?;
    Ok(SessionContext { history, documents })
}

/// Read all `*.ndjson` files in the session directory, sorted by filename,
/// and deserialize each line as a Message.
fn load_history(session_dir: &Path) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    if !session_dir.exists() {
        return Ok(messages);
    }

    let mut ndjson_files: Vec<PathBuf> = fs::read_dir(session_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("ndjson"))
        .collect();
    ndjson_files.sort();

    for path in &ndjson_files {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Message>(line) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    tracing::warn!(
                        file = %path.display(),
                        error = %e,
                        "skipping malformed message line"
                    );
                }
            }
        }
    }

    Ok(messages)
}

/// Load all non-session files under dynamic/ as Documents for general knowledge.
fn load_documents(dynamic_dir: &Path, session_dir: &Path) -> Result<Vec<Document>> {
    let mut documents = Vec::new();
    let all_files = collect_files(dynamic_dir)?;

    // Canonicalize session_dir for comparison (may not exist yet)
    let session_prefix = if session_dir.exists() {
        Some(fs::canonicalize(session_dir)?)
    } else {
        // If session dir doesn't exist, nothing to exclude
        None
    };

    for path in all_files {
        // Skip .ndjson files in the session directory
        if let Some(ref prefix) = session_prefix {
            if let Ok(canonical) = fs::canonicalize(&path) {
                if canonical.starts_with(prefix) {
                    continue;
                }
            }
        }

        // Skip binary-looking files
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel_path = path
            .strip_prefix(dynamic_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        let source_type = source_type_from_path(&rel_path);

        let mut additional_props = HashMap::new();
        additional_props.insert("source_type".to_string(), source_type.to_string());

        documents.push(Document {
            id: rel_path,
            text: content,
            additional_props,
        });
    }

    Ok(documents)
}

/// Classify a file by its directory path.
fn source_type_from_path(rel_path: &str) -> &str {
    if rel_path.starts_with("notes/") {
        "note"
    } else if rel_path.starts_with("cli/") {
        "cli_session"
    } else if rel_path.starts_with("matrix/") {
        "matrix_session"
    } else {
        "document"
    }
}

/// Persist a turn's messages to disk as NDJSON.
///
/// Writes to `dynamic/{session_id.path}/{timestamp}_{slug}.ndjson`.
/// Filters out `AssistantContent::Reasoning` blocks before writing.
/// Uses atomic write (tmp file + rename).
pub fn save_turn(
    dynamic_dir: &Path,
    session_id: &SessionId,
    messages: &[Message],
    slug: &str,
) -> Result<PathBuf> {
    let session_dir = dynamic_dir.join(&session_id.path);
    fs::create_dir_all(&session_dir)?;

    let timestamp = chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("{timestamp}_{slug}.ndjson");

    let filtered = filter_reasoning(messages);
    let mut lines = Vec::new();
    for msg in &filtered {
        lines.push(serde_json::to_string(msg)?);
    }

    let content = lines.join("\n") + "\n";

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = session_dir.join(&tmp_name);
    let final_path = session_dir.join(&filename);

    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    tracing::info!(path = %final_path.display(), "saved session turn");
    Ok(final_path)
}

/// Strip Reasoning blocks from assistant messages before persisting.
/// They can be very large and aren't useful on reload.
fn filter_reasoning(messages: &[Message]) -> Vec<Message> {
    use rig::completion::message::AssistantContent;

    messages
        .iter()
        .map(|msg| match msg {
            Message::Assistant { id, content } => {
                let filtered: Vec<AssistantContent> = content
                    .iter()
                    .filter(|c| !matches!(c, AssistantContent::Reasoning(_)))
                    .cloned()
                    .collect();
                if filtered.is_empty() {
                    // Keep at least an empty text if everything was reasoning
                    Message::assistant("")
                } else {
                    Message::Assistant {
                        id: id.clone(),
                        content: rig::OneOrMany::many(filtered).unwrap_or_else(|_| {
                            rig::OneOrMany::one(AssistantContent::Text(
                                rig::completion::message::Text {
                                    text: String::new(),
                                },
                            ))
                        }),
                    }
                }
            }
            other => other.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn source_type_classification() {
        assert_eq!(source_type_from_path("notes/foo.md"), "note");
        assert_eq!(source_type_from_path("cli/session.json"), "cli_session");
        assert_eq!(
            source_type_from_path("matrix/host/room.json"),
            "matrix_session"
        );
        assert_eq!(source_type_from_path("other/file.txt"), "document");
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = TempDir::new().unwrap();
        let dynamic_dir = tmp.path();
        let session_id = SessionId::new("test", "session1");

        let messages = vec![Message::user("hello"), Message::assistant("hi there")];

        let path = save_turn(dynamic_dir, &session_id, &messages, "greeting").unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().ends_with(".ndjson"));

        let ctx = load_session(dynamic_dir, &session_id).unwrap();
        assert_eq!(ctx.history.len(), 2);
        // No other files, so no documents
        assert!(ctx.documents.is_empty());
    }

    #[test]
    fn load_empty_session() {
        let tmp = TempDir::new().unwrap();
        let session_id = SessionId::new("nonexistent", "session");
        let ctx = load_session(tmp.path(), &session_id).unwrap();
        assert!(ctx.history.is_empty());
        assert!(ctx.documents.is_empty());
    }

    #[test]
    fn documents_exclude_session_dir() {
        let tmp = TempDir::new().unwrap();
        let dynamic_dir = tmp.path();
        let session_id = SessionId::new("cli", "mysession");

        // Create session dir with an ndjson file
        let session_dir = dynamic_dir.join("cli").join("mysession");
        fs::create_dir_all(&session_dir).unwrap();
        fs::write(
            session_dir.join("2026-01-01_00-00-00_test.ndjson"),
            "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}\n",
        )
        .unwrap();

        // Create a non-session file
        let notes_dir = dynamic_dir.join("notes");
        fs::create_dir_all(&notes_dir).unwrap();
        fs::write(notes_dir.join("note.md"), "some note content").unwrap();

        let ctx = load_session(dynamic_dir, &session_id).unwrap();
        assert_eq!(ctx.history.len(), 1);
        assert_eq!(ctx.documents.len(), 1);
        assert_eq!(ctx.documents[0].id, "notes/note.md");
        assert_eq!(
            ctx.documents[0]
                .additional_props
                .get("source_type")
                .unwrap(),
            "note"
        );
    }
}
