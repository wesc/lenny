use anyhow::Result;
use async_trait::async_trait;
use firecrawl::FirecrawlApp;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct ExtractUrlArgs {
    url: String,
    prompt: String,
}

pub struct ExtractUrlTool {
    pub firecrawl: FirecrawlApp,
}

#[async_trait]
impl ToolHandler for ExtractUrlTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: ExtractUrlArgs = serde_json::from_value(args.clone())?;
        let url = &args.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            anyhow::bail!("Invalid URL: {url}");
        }

        tracing::debug!(url, prompt = %args.prompt, "extract_url: starting");

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
                    tracing::debug!(url, len = text.len(), "extract_url: done");
                    return Ok(format!(
                        "{text}\n\n(Partial extraction from {}. The source page may contain additional information not covered by this query. Use extract_url again with a different prompt to retrieve more.)",
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
        name: "extract_url".to_string(),
        description: "Extract information from a URL by asking a question about its content. \
            Each call returns only information relevant to your specific prompt — the source page \
            likely contains more. Query the same URL again with a different prompt to retrieve \
            different details. Use this whenever you need to read, summarize, or query a web page."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract from (must start with http:// or https://)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Question or extraction prompt (e.g. 'Summarize the key points' or 'What is the main topic?')"
                }
            },
            "required": ["url", "prompt"]
        }),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "The page discusses recent developments in artificial intelligence."
                .to_string(),
        }),
    }
}

impl ExtractUrlTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
