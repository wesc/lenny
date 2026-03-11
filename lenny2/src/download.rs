use anyhow::Result;
use chrono::Local;
use firecrawl::FirecrawlApp;
use firecrawl::scrape::ScrapeFormats;
use rig::client::CompletionClient;
use rig::providers::openrouter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use crate::config::Config;
use crate::tokens;

/// Structured output: the LLM's exploration of a scraped page.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ExplorationResult {
    /// 10 exploratory questions as chain-of-thought scaffolding
    pub questions: Vec<String>,
    /// 3-5 concrete search queries derived from the reasoning
    pub search_queries: Vec<String>,
}

pub async fn run(config: &Config, seed_url: &str, max_pages: usize) -> Result<()> {
    let rumination_id = Local::now().format("%Y%m%d-%H%M%S").to_string();
    let pages_dir = config.pages_dir().join(&rumination_id);
    std::fs::create_dir_all(&pages_dir)?;

    println!("Rumination ID: {rumination_id}");
    println!("Saving pages to: {}", pages_dir.display());

    let firecrawl = FirecrawlApp::new(&config.firecrawl_api_key)?;
    let or_client = openrouter::Client::new(&config.openrouter_api_key)?;
    let extractor = or_client
        .extractor::<ExplorationResult>(&config.model)
        .preamble(
            "You are a research assistant. Given a scraped webpage, generate 10 exploratory \
             questions that dive deep into the topic (as chain-of-thought reasoning), then \
             produce 3-5 concrete search queries that would find the most relevant related pages. \
             Focus on key concepts, entities, and ideas worth exploring further.",
        )
        .build();

    let mut queue: VecDeque<String> = VecDeque::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut page_count: usize = 0;

    queue.push_back(seed_url.to_string());
    seen.insert(seed_url.to_string());

    while let Some(url) = queue.pop_front() {
        if page_count >= max_pages {
            break;
        }

        println!("[{}/{}] Scraping: {url}", page_count + 1, max_pages);

        // Scrape the page
        let content = match scrape_page(&firecrawl, &url).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Failed to scrape {url}: {e}");
                continue;
            }
        };

        // Save to disk
        let filename = format!("{page_count:04}.md");
        let filepath = pages_dir.join(&filename);
        save_page(&filepath, &url, &content)?;
        page_count += 1;
        let tok_count = tokens::count_tokens(&content);
        println!(
            "  Saved: {filename} ({} chars, ~{tok_count} tokens)",
            content.len()
        );

        // Don't explore further if we're near the limit
        if page_count >= max_pages {
            break;
        }

        // Generate exploration queries via structured LLM output
        let truncated = truncate_content(&content, 6000);
        let exploration = match extractor.extract(&truncated).await {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  Failed to generate exploration: {e}");
                continue;
            }
        };

        println!(
            "  Generated {} questions, {} search queries",
            exploration.questions.len(),
            exploration.search_queries.len()
        );
        for (i, q) in exploration.questions.iter().enumerate() {
            println!("    Q{}: {q}", i + 1);
        }

        // Search for each query and enqueue new URLs
        for query in &exploration.search_queries {
            if page_count + queue.len() >= max_pages {
                break;
            }

            println!("  Searching: \"{query}\"");
            match search_firecrawl(&firecrawl, query).await {
                Ok(urls) => {
                    if urls.is_empty() {
                        println!("    No results");
                    }
                    for found_url in urls {
                        if !seen.contains(&found_url) {
                            println!("    + {found_url}");
                            seen.insert(found_url.clone());
                            queue.push_back(found_url);
                        } else {
                            println!("    (already seen) {found_url}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("    Search failed: {e}");
                }
            }
        }

        println!("  Queue size: {}", queue.len());
    }

    println!(
        "\nDownload complete: {page_count} pages saved to {}",
        pages_dir.display()
    );
    Ok(())
}

async fn scrape_page(firecrawl: &FirecrawlApp, url: &str) -> Result<String> {
    let options = firecrawl::scrape::ScrapeOptions {
        formats: Some(vec![ScrapeFormats::Markdown]),
        only_main_content: Some(true),
        ..Default::default()
    };

    let result = firecrawl.scrape_url(url, options).await?;
    let markdown = result.markdown.unwrap_or_default();
    Ok(markdown)
}

async fn search_firecrawl(firecrawl: &FirecrawlApp, query: &str) -> Result<Vec<String>> {
    let params = firecrawl::search::SearchParams {
        query: query.to_string(),
        limit: Some(1),
        ..Default::default()
    };

    let response = firecrawl.search(query, params).await?;
    let urls: Vec<String> = response.data.into_iter().map(|doc| doc.url).collect();
    Ok(urls)
}

fn save_page(filepath: &PathBuf, url: &str, content: &str) -> Result<()> {
    let with_header = format!("<!-- url: {url} -->\n\n{content}");
    std::fs::write(filepath, with_header)?;
    Ok(())
}

/// Truncate content to approximately `max_words` words.
fn truncate_content(content: &str, max_words: usize) -> String {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() <= max_words {
        content.to_string()
    } else {
        words[..max_words].join(" ")
    }
}
