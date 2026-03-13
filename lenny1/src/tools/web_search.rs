use anyhow::Result;
use async_trait::async_trait;
use firecrawl::FirecrawlApp;
use firecrawl::search::SearchParams;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: u32,
}

fn default_limit() -> u32 {
    5
}

pub struct WebSearchTool {
    pub firecrawl: FirecrawlApp,
}

#[async_trait]
impl ToolHandler for WebSearchTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: WebSearchArgs = serde_json::from_value(args.clone())?;

        tracing::debug!(query = %args.query, limit = args.limit, "web_search: searching");

        let params = SearchParams {
            query: args.query.clone(),
            limit: Some(args.limit.min(20)),
            ..Default::default()
        };

        let response = self.firecrawl.search(&args.query, params).await?;

        if response.data.is_empty() {
            return Ok("No results found.".to_string());
        }

        let results: Vec<serde_json::Value> = response
            .data
            .iter()
            .map(|doc| {
                json!({
                    "title": doc.title,
                    "url": doc.url,
                    "description": doc.description,
                })
            })
            .collect();

        Ok(serde_json::to_string_pretty(&results)?)
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "web_search".to_string(),
        description: "Search the web for information. Returns a list of results with titles, \
            URLs, and descriptions. Use this to find URLs, answer questions about current events, \
            or research any topic."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-20, default 5)"
                }
            },
            "required": ["query"]
        }),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: r#"[{"title": "Example", "url": "https://example.com", "description": "An example page."}]"#.to_string(),
        }),
    }
}

impl WebSearchTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
