use anyhow::Result;
use async_trait::async_trait;
use firecrawl::FirecrawlApp;
use firecrawl::scrape::{ScrapeFormats, ScrapeOptions};
use openrouter_rs::types::Tool;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct ScrapeUrlArgs {
    url: String,
}

pub struct ScrapeUrlTool {
    pub firecrawl: FirecrawlApp,
}

const MAX_WORDS: usize = 8000;

#[async_trait]
impl ToolHandler for ScrapeUrlTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: ScrapeUrlArgs = serde_json::from_value(args.clone())?;
        let url = &args.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            anyhow::bail!("Invalid URL: {url}");
        }

        tracing::debug!(url, "scrape_url: scraping");

        let options = ScrapeOptions {
            formats: Some(vec![ScrapeFormats::Markdown]),
            only_main_content: Some(true),
            ..Default::default()
        };

        let doc = self.firecrawl.scrape_url(url, options).await?;

        let title = doc
            .metadata
            .title
            .as_deref()
            .or(doc.metadata.og_title.as_deref())
            .unwrap_or("(no title)");
        let markdown = doc.markdown.as_deref().unwrap_or("");

        let words: Vec<&str> = markdown.split_whitespace().collect();
        let content = if words.len() > MAX_WORDS {
            words[..MAX_WORDS].join(" ") + "\n\n[truncated]"
        } else {
            markdown.to_string()
        };

        Ok(format!("# {title}\n\n{content}"))
    }
}

impl ScrapeUrlTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: Tool::new(
                "scrape_url",
                "Scrape a web page using Firecrawl and return its content as markdown.",
                json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape (must start with http:// or https://)"
                        }
                    },
                    "required": ["url"]
                }),
            ),
            handler: Box::new(self),
        }
    }
}
