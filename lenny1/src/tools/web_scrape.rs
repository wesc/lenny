use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct WebScrapeArgs {
    pub url: String,
}

#[derive(Debug, Error)]
pub enum WebScrapeError {
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    #[error("Request failed: {0}")]
    Request(String),
    #[error("Non-success status: {0}")]
    Status(u16),
    #[error("HTML conversion failed: {0}")]
    Conversion(String),
}

pub struct WebScrapeTool;

const MAX_WORDS: usize = 8000;

impl Tool for WebScrapeTool {
    const NAME: &'static str = "web_scrape";

    type Error = WebScrapeError;
    type Args = WebScrapeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "web_scrape".to_string(),
            description:
                "Fetch a web page and return its content as markdown. Use this to read URLs."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = &args.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(WebScrapeError::InvalidUrl(url.clone()));
        }

        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")
            .build()
            .map_err(|e| WebScrapeError::Request(e.to_string()))?;

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| WebScrapeError::Request(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            return Err(WebScrapeError::Status(status.as_u16()));
        }

        let body = response
            .text()
            .await
            .map_err(|e| WebScrapeError::Request(e.to_string()))?;

        let markdown = html_to_markdown_rs::convert(&body, None)
            .map_err(|e| WebScrapeError::Conversion(e.to_string()))?;

        // Truncate to MAX_WORDS to avoid blowing up context
        let words: Vec<&str> = markdown.split_whitespace().collect();
        if words.len() > MAX_WORDS {
            Ok(words[..MAX_WORDS].join(" ") + "\n\n[truncated]")
        } else {
            Ok(markdown)
        }
    }
}
