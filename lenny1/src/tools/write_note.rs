use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

use super::notes;
use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct WriteNoteArgs {
    filename: String,
    content: String,
}

pub struct WriteNoteTool {
    pub dynamic_dir: PathBuf,
}

#[async_trait]
impl ToolHandler for WriteNoteTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: WriteNoteArgs = serde_json::from_value(args.clone())?;
        let filename = notes::sanitize_filename(&args.filename)?;

        let dir = notes::notes_dir(&self.dynamic_dir)?;
        let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
        let tmp_path = dir.join(&tmp_name);
        let final_path = dir.join(&filename);

        std::fs::write(&tmp_path, &args.content)?;
        std::fs::rename(&tmp_path, &final_path)?;

        tracing::debug!(path = %final_path.display(), "write_note: saved");
        Ok(format!("Saved to notes/{filename}"))
    }
}

impl WriteNoteTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "write_note".to_string(),
                description: "Write a note or report to a file in the notes directory (overwrites if exists). Use this to save final reports or any information worth remembering long-term. The file will be automatically incorporated into the knowledge base over time.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename for the note (e.g. 'bluesky-research-2026-03-10.md'). No path separators."
                        },
                        "content": {
                            "type": "string",
                            "description": "The full content to write to the file."
                        }
                    },
                    "required": ["filename", "content"]
                }),
            },
            handler: Box::new(self),
        }
    }
}
