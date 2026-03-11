use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

use super::notes;
use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct ReadNoteArgs {
    filename: String,
}

pub struct ReadNoteTool {
    pub dynamic_dir: PathBuf,
}

#[async_trait]
impl ToolHandler for ReadNoteTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: ReadNoteArgs = serde_json::from_value(args.clone())?;
        let filename = notes::sanitize_filename(&args.filename)?;

        let path = notes::notes_dir(&self.dynamic_dir)?.join(&filename);
        let content = std::fs::read_to_string(&path)
            .map_err(|_| anyhow::anyhow!("Note not found: notes/{filename}"))?;

        tracing::debug!(path = %path.display(), len = content.len(), "read_note");
        Ok(content)
    }
}

impl ReadNoteTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "read_note".to_string(),
                description: "Read a note file from the notes directory. Use this to read back accumulated research or other notes you have previously written.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename to read (e.g. 'research-draft.md')."
                        }
                    },
                    "required": ["filename"]
                }),
            },
            handler: Box::new(self),
        }
    }
}
