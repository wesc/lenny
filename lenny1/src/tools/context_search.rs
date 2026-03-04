use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use thiserror::Error;

use crate::actions::comprehension;

#[derive(Debug, Deserialize)]
pub struct ContextSearchArgs {
    /// The search query describing what information to find.
    pub query: String,
    /// Optional start of time range (ISO 8601, e.g. "2025-11-01T00:00:00Z").
    #[serde(default)]
    pub start_time: Option<String>,
    /// Optional end of time range (ISO 8601, e.g. "2025-11-05T00:00:00Z").
    #[serde(default)]
    pub end_time: Option<String>,
}

#[derive(Debug, Error)]
pub enum ContextSearchError {
    #[error("{0}")]
    Search(String),
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

impl Tool for ContextSearchTool {
    const NAME: &'static str = "context_search";

    type Error = ContextSearchError;
    type Args = ContextSearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
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
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

        let result = comprehension::retrieve(&self.db_path, &args.query, 5, time_range)
            .await
            .map_err(|e| ContextSearchError::Search(e.to_string()))?;

        if result.context.is_empty() {
            Ok("No relevant context found.".to_string())
        } else {
            Ok(result.context)
        }
    }
}
