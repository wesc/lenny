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
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                tracing::debug!(path = %path.display(), len = content.len(), "read_note");
                Ok(content)
            }
            Err(_) => Ok(format!("Note not found: notes/{filename}")),
        }
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
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
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "Contents of research.md:\n\nSome research notes here.".to_string(),
        }),
    }
}

impl ReadNoteTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
