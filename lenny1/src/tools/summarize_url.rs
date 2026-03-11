use anyhow::Result;
use async_trait::async_trait;
use firecrawl::FirecrawlApp;
use firecrawl::scrape::{ScrapeFormats, ScrapeOptions};
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct SummarizeUrlArgs {
    url: String,
}

pub struct SummarizeUrlTool {
    pub firecrawl: FirecrawlApp,
}

const MAX_WORDS: usize = 8000;

#[async_trait]
impl ToolHandler for SummarizeUrlTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: SummarizeUrlArgs = serde_json::from_value(args.clone())?;
        let url = &args.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            anyhow::bail!("Invalid URL: {url}");
        }

        tracing::debug!(url, "summarize_url: scraping");

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

        // Truncate to MAX_WORDS
        let words: Vec<&str> = markdown.split_whitespace().collect();
        let content = if words.len() > MAX_WORDS {
            words[..MAX_WORDS].join(" ") + "\n\n[truncated]"
        } else {
            markdown.to_string()
        };

        Ok(format!("# {title}\n\n{content}"))
    }
}

impl SummarizeUrlTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "summarize_url".to_string(),
                description: "Fetch a web page using Firecrawl and return its main content as markdown for summarization. Use this to read and summarize URLs.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch and summarize (must start with http:// or https://)"
                        }
                    },
                    "required": ["url"]
                }),
            },
            handler: Box::new(self),
        }
    }
}
