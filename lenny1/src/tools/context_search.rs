use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

use crate::actions::comprehension;
use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct ContextSearchArgs {
    query: String,
    #[serde(default)]
    start_time: Option<String>,
    #[serde(default)]
    end_time: Option<String>,
}

pub struct ContextSearchTool {
    pub db_path: PathBuf,
}

/// Parse an ISO 8601 string to unix seconds. Returns None on failure.
fn iso_to_unix(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp())
}

#[async_trait]
impl ToolHandler for ContextSearchTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: ContextSearchArgs = serde_json::from_value(args.clone())?;

        let time_range = match (&args.start_time, &args.end_time) {
            (Some(start), Some(end)) => match (iso_to_unix(start), iso_to_unix(end)) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            },
            _ => None,
        };

        tracing::debug!(
            query = %args.query,
            start_time = ?args.start_time,
            end_time = ?args.end_time,
            "context search"
        );

        let result = comprehension::retrieve(&self.db_path, &args.query, 5, time_range).await?;

        if result.context.is_empty() {
            Ok("No relevant context found.".to_string())
        } else {
            Ok(result.context)
        }
    }
}

impl ContextSearchTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "context_search".to_string(),
                description: "Search past conversations and documents by semantic similarity. \
                     Optionally filter by a time range. Returns relevant context passages."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query describing what information to find"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Optional start of time range (ISO 8601, e.g. '2025-11-01T00:00:00Z')"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Optional end of time range (ISO 8601, e.g. '2025-11-05T00:00:00Z')"
                        }
                    },
                    "required": ["query"]
                }),
            },
            handler: Box::new(self),
        }
    }
}
