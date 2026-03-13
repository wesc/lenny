use anyhow::Result;
use async_trait::async_trait;
use firecrawl::FirecrawlApp;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

use super::notes;
use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct ExtractUrlToNoteArgs {
    url: String,
    prompt: String,
    filename: String,
}

pub struct ExtractUrlToNoteTool {
    pub firecrawl: FirecrawlApp,
    pub dynamic_dir: PathBuf,
}

#[async_trait]
impl ToolHandler for ExtractUrlToNoteTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: ExtractUrlToNoteArgs = serde_json::from_value(args.clone())?;
        let filename = notes::sanitize_filename(&args.filename)?;

        tracing::debug!(url = %args.url, prompt = %args.prompt, "extract_url_to_note: starting");

        let params = firecrawl::extract::ExtractParams {
            urls: Some(vec![args.url.clone()]),
            prompt: Some(args.prompt.clone()),
            ..Default::default()
        };

        let job = self.firecrawl.async_extract(params).await?;

        // Poll until completion
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let status = self.firecrawl.get_extract_status(&job.id).await?;
            match status.status.as_str() {
                "completed" => {
                    let data = status
                        .data
                        .ok_or_else(|| anyhow::anyhow!("extract completed but no data"))?;
                    let text = serde_json::to_string_pretty(&data)?;

                    let section =
                        format!("\n## {}\n\nSource: {}\n\n{}\n", args.prompt, args.url, text);
                    let path = notes::append_to_note(&self.dynamic_dir, &filename, &section)?;

                    tracing::debug!(path = %path.display(), "extract_url_to_note: appended");
                    return Ok(format!(
                        "Extracted and appended summary of {} to notes/{filename}",
                        args.url
                    ));
                }
                "failed" => {
                    let msg = status.error.unwrap_or_else(|| "unknown error".to_string());
                    return Ok(format!("Extract failed for {}: {msg}", args.url));
                }
                _ => continue,
            }
        }
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "extract_url_to_note".to_string(),
        description: "Extract information from a URL and save it directly to a note file in one step. Combines URL extraction with file writing — the extracted content is appended to the named file. Use this instead of extract_url + write_note when you want to save extracted content to a file. Ideal for research pipelines gathering information from multiple URLs.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to extract from."
                },
                "prompt": {
                    "type": "string",
                    "description": "Question or extraction prompt (e.g. 'Summarize the key points of this article')."
                },
                "filename": {
                    "type": "string",
                    "description": "Note file to append results to (e.g. 'research-draft.md'). Created if it doesn't exist."
                }
            },
            "required": ["url", "prompt", "filename"]
        }),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "Extracted content appended to research.md.".to_string(),
        }),
    }
}

impl ExtractUrlToNoteTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
