use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

use crate::agent::{ToolDef, ToolHandler};

const API_BASE: &str = "https://api.bsky.app/xrpc";

const POSTS_PER_QUERY: u32 = 50;

// ---------------------------------------------------------------------------
// Bluesky API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SearchResponse {
    posts: Vec<Post>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct Post {
    uri: String,
    author: Author,
    record: Record,
    embed: Option<EmbedView>,
    like_count: Option<u64>,
    repost_count: Option<u64>,
    quote_count: Option<u64>,
    indexed_at: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct Author {
    handle: String,
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct Record {
    text: String,
    created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "$type")]
enum EmbedView {
    #[serde(rename = "app.bsky.embed.external#view")]
    External { external: ExternalView },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct ExternalView {
    uri: String,
    title: String,
    description: String,
}

// ---------------------------------------------------------------------------
// Collected link
// ---------------------------------------------------------------------------

struct CollectedLink {
    url: String,
    title: String,
    description: String,
    posts: Vec<LinkPost>,
}

struct LinkPost {
    likes: u64,
    reposts: u64,
    quotes: u64,
}

impl LinkPost {
    fn engagement(&self) -> u64 {
        self.likes + self.reposts * 2 + self.quotes * 3
    }
}

// ---------------------------------------------------------------------------
// Public: fetch trending links from Bluesky
// ---------------------------------------------------------------------------

/// Fetch trending links from Bluesky for the given queries. Returns JSON value with the links.
pub async fn fetch_trending(
    queries: &[String],
    since: &str,
    until: &str,
) -> Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .user_agent("lenny1/0.1")
        .build()?;

    let mut link_map: HashMap<String, CollectedLink> = HashMap::new();

    for query in queries {
        let url = format!(
            "{API_BASE}/app.bsky.feed.searchPosts?q={}&sort=top&limit={POSTS_PER_QUERY}&since={since}&until={until}",
            urlencoded(query)
        );

        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }

        let body: SearchResponse = resp.json().await?;

        for post in body.posts {
            let Some(EmbedView::External { external }) = post.embed else {
                continue;
            };

            if external.uri.contains("bsky.app") {
                continue;
            }

            let lp = LinkPost {
                likes: post.like_count.unwrap_or(0),
                reposts: post.repost_count.unwrap_or(0),
                quotes: post.quote_count.unwrap_or(0),
            };

            let entry = link_map
                .entry(external.uri.clone())
                .or_insert_with(|| CollectedLink {
                    url: external.uri.clone(),
                    title: external.title,
                    description: external.description,
                    posts: Vec::new(),
                });
            entry.posts.push(lp);
        }
    }

    // Rank links by total engagement
    let mut links: Vec<CollectedLink> = link_map.into_values().collect();
    for link in &mut links {
        link.posts
            .sort_by_key(|p| std::cmp::Reverse(p.engagement()));
    }
    links.sort_by_key(|link| {
        std::cmp::Reverse(link.posts.iter().map(|p| p.engagement()).sum::<u64>())
    });

    let items: Vec<serde_json::Value> = links
        .iter()
        .take(15)
        .map(|link| {
            let total_engagement: u64 = link.posts.iter().map(|p| p.engagement()).sum();
            json!({
                "url": link.url,
                "title": link.title,
                "description": link.description,
                "engagement": total_engagement,
                "shares": link.posts.len(),
            })
        })
        .collect();

    Ok(json!({
        "source": "bluesky",
        "queries": queries,
        "since": since,
        "until": until,
        "fetched_at": chrono::Utc::now().to_rfc3339(),
        "link_count": items.len(),
        "links": items,
    }))
}

// ---------------------------------------------------------------------------
// Agent tool
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BlueskyTrendingArgs {
    queries: Vec<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    until: Option<String>,
}

pub struct BlueskyTrendingTool;

#[async_trait]
impl ToolHandler for BlueskyTrendingTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: BlueskyTrendingArgs = serde_json::from_value(args.clone())?;

        let now = chrono::Utc::now();
        let since = args
            .since
            .map(|d| format!("{d}T00:00:00Z"))
            .unwrap_or_else(|| {
                (now - chrono::Duration::hours(24))
                    .format("%Y-%m-%dT%H:%M:%SZ")
                    .to_string()
            });
        let until = args
            .until
            .map(|d| {
                let date = chrono::NaiveDate::parse_from_str(&d, "%Y-%m-%d")
                    .expect("until should be YYYY-MM-DD");
                let next_day = date + chrono::Duration::days(1);
                format!("{next_day}T00:00:00Z")
            })
            .unwrap_or_else(|| now.format("%Y-%m-%dT%H:%M:%SZ").to_string());

        tracing::debug!(since = %since, until = %until, "bluesky_trending: fetching");

        let output = fetch_trending(&args.queries, &since, &until).await?;
        Ok(serde_json::to_string_pretty(&output)?)
    }
}

impl BlueskyTrendingTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: ToolDefinition {
                name: "bluesky_trending".to_string(),
                description: "Search Bluesky for trending links matching the given keywords. Returns a JSON object with top shared links ranked by engagement. Each query is searched separately and results are merged and ranked.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Keywords to search for (e.g. [\"rust programming\", \"cargo build system\"]). Each string is a separate search query."
                        },
                        "since": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD). Defaults to 24 hours ago."
                        },
                        "until": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD). Defaults to now."
                        }
                    },
                    "required": ["queries"]
                }),
            },
            handler: Box::new(self),
        }
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}
