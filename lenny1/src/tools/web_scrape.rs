use anyhow::Result;
use async_trait::async_trait;
use openrouter_rs::types::Tool;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct WebScrapeArgs {
    url: String,
}

pub struct WebScrapeTool;

const MAX_WORDS: usize = 2000;

#[async_trait]
impl ToolHandler for WebScrapeTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: WebScrapeArgs = serde_json::from_value(args.clone())?;
        let url = &args.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            anyhow::bail!("Invalid URL: {url}");
        }

        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")
            .build()?;

        tracing::debug!(url, "fetching web page");

        let response = client.get(url).send().await?;

        let status = response.status();
        tracing::debug!(status = status.as_u16(), url, "received response");
        if !status.is_success() {
            anyhow::bail!("Non-success status: {}", status.as_u16());
        }

        let body = response.text().await?;

        tracing::debug!(body_len = body.len(), url, "fetched body");

        let markdown = html_to_markdown_rs::convert(&body, None)
            .map_err(|e| anyhow::anyhow!("HTML conversion failed: {e}"))?;

        // Truncate to MAX_WORDS to avoid blowing up context
        let words: Vec<&str> = markdown.split_whitespace().collect();
        if words.len() > MAX_WORDS {
            Ok(words[..MAX_WORDS].join(" ") + "\n\n[truncated]")
        } else {
            Ok(markdown)
        }
    }
}

impl WebScrapeTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: Tool::new(
                "web_scrape",
                "Fetch a web page and return its content as markdown. Use this to read URLs.",
                json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (must start with http:// or https://)"
                        }
                    },
                    "required": ["url"]
                }),
            ),
            handler: Box::new(self),
        }
    }
}
