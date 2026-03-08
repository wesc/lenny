use anyhow::Result;
use async_trait::async_trait;
use openrouter_rs::types::Tool;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

use crate::agent::{ToolDef, ToolHandler};

const API_BASE: &str = "https://api.bsky.app/xrpc";

const QUERIES: &[&str] = &[
    "machine learning",
    "artificial intelligence",
    "large language model",
    "neural network",
    "deep learning",
];

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
struct Author {
    handle: String,
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
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
    author_handle: String,
    author_name: Option<String>,
    text: String,
    likes: u64,
    reposts: u64,
    quotes: u64,
    created_at: String,
    post_uri: String,
}

impl LinkPost {
    fn engagement(&self) -> u64 {
        self.likes + self.reposts * 2 + self.quotes * 3
    }
}

// ---------------------------------------------------------------------------
// Public: fetch trending links from Bluesky
// ---------------------------------------------------------------------------

/// Fetch trending AI/ML links from Bluesky. Returns JSON value with the links.
pub async fn fetch_trending(since: &str, until: &str) -> Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .user_agent("lenny1/0.1")
        .build()?;

    let mut link_map: HashMap<String, CollectedLink> = HashMap::new();

    for query in QUERIES {
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
                author_handle: post.author.handle,
                author_name: post.author.display_name,
                text: post.record.text,
                likes: post.like_count.unwrap_or(0),
                reposts: post.repost_count.unwrap_or(0),
                quotes: post.quote_count.unwrap_or(0),
                created_at: post.record.created_at.unwrap_or(post.indexed_at),
                post_uri: post.uri,
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
        .take(30)
        .map(|link| {
            let total_engagement: u64 = link.posts.iter().map(|p| p.engagement()).sum();
            let top_post = &link.posts[0];
            json!({
                "url": link.url,
                "title": link.title,
                "description": link.description,
                "total_engagement": total_engagement,
                "shared_by_count": link.posts.len(),
                "top_post": {
                    "author": top_post.author_name.as_deref()
                        .unwrap_or(&top_post.author_handle),
                    "handle": top_post.author_handle,
                    "text": top_post.text,
                    "likes": top_post.likes,
                    "reposts": top_post.reposts,
                    "quotes": top_post.quotes,
                    "created_at": top_post.created_at,
                    "uri": top_post.post_uri,
                },
            })
        })
        .collect();

    Ok(json!({
        "source": "bluesky",
        "queries": QUERIES,
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

        let output = fetch_trending(&since, &until).await?;
        Ok(serde_json::to_string_pretty(&output)?)
    }
}

impl BlueskyTrendingTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: Tool::new(
                "bluesky_trending",
                "Search Bluesky for trending AI/ML links. Returns a JSON object with top shared links ranked by engagement. Optionally filter by date range.",
                json!({
                    "type": "object",
                    "properties": {
                        "since": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD). Defaults to 24 hours ago."
                        },
                        "until": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD). Defaults to now."
                        }
                    },
                    "required": []
                }),
            ),
            handler: Box::new(self),
        }
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}
