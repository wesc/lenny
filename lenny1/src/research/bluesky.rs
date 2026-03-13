use anyhow::Result;
use firecrawl::FirecrawlApp;
use std::fs;

use crate::agent::{self, Agent, ToolDef};
use crate::config::Config;
use crate::tools::{BlueskyTrendingTool, ExtractUrlTool};

const RESEARCH_SYSTEM: &str = "\
You are a deep research agent that analyzes trending AI/ML content from Bluesky.

Your task:
1. Call bluesky_trending to get the latest trending AI/ML links.
2. For each of the top links (up to 10), call extract_url to extract the key content.
3. After gathering all content, produce a comprehensive research summary.

When you are done gathering data, respond with your final analysis as plain text. \
Do NOT call any more tools — just provide your summary directly.

Your final response should synthesize the key themes, notable developments, and important \
takeaways from the trending content. Group related items into themes. \
Highlight particularly significant developments. \
Include links as inline citations for every claim or development you mention — \
use the original URLs from the trending data and scraped pages.";

pub async fn run(config: &Config, since: &str, until: &str) -> Result<()> {
    eprintln!("  Deep research: {since} -> {until}");

    let firecrawl_api_key = config
        .firecrawl_api_key
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("firecrawl_api_key is required for deep research"))?;

    let client = agent::build_client(config)?;
    let firecrawl = FirecrawlApp::new(firecrawl_api_key)?;

    // Step 1: Fetch trending data directly (not via agent) so we can save the raw output
    eprintln!("  Fetching trending links...");
    let default_queries: Vec<String> = [
        "machine learning",
        "artificial intelligence",
        "large language model",
        "neural network",
        "deep learning",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let trending = crate::tools::fetch_trending(&default_queries, since, until).await?;
    let link_count = trending["link_count"].as_u64().unwrap_or(0);
    eprintln!("  Found {link_count} trending links");

    // Step 2: Build agent with extract_url tool + bluesky_trending for the research loop
    let tools: Vec<ToolDef> = vec![
        BlueskyTrendingTool.tool_def(),
        ExtractUrlTool {
            firecrawl: firecrawl.clone(),
        }
        .tool_def(),
    ];

    // Build a prompt that includes the trending data so the agent can pick URLs to scrape
    let trending_json = serde_json::to_string_pretty(&trending)?;
    let prompt = format!(
        "Here are the trending AI/ML links from Bluesky ({since} to {until}):\n\n\
         {trending_json}\n\n\
         Now extract the key content from the top links (up to 10) using extract_url, \
         then provide a comprehensive research summary of all the trends and developments."
    );

    let agent = Agent::builder(&client, config)
        .system(RESEARCH_SYSTEM)
        .tools(&tools)
        .max_iterations(15)
        .build();

    eprintln!("  Running deep research agent...");
    let mut hook = ResearchHook;
    let result = agent.run(&prompt, vec![], &mut hook).await?;

    let summary = result.answer.clone();

    // Write output: two JSON lines (trending + summary)
    let out_dir = config.dynamic_dir.join("research").join("bluesky");
    fs::create_dir_all(&out_dir)?;

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let final_name = format!("{timestamp}-deep-research.txt");
    let tmp_path = out_dir.join(&tmp_name);
    let final_path = out_dir.join(&final_name);

    let line1 = serde_json::to_string(&trending)?;
    let summary_json = serde_json::json!({
        "type": "research_summary",
        "since": since,
        "until": until,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "reasoning_turns": result.reasoning_turns,
        "tool_events": result.tool_events.len(),
        "summary": summary,
    });
    let line2 = serde_json::to_string(&summary_json)?;

    let content = format!(
        "Trending AI/ML links from Bluesky ({since} to {until}):\n{line1}\n\n\
         Research summary for the above trending links:\n{line2}\n"
    );
    fs::write(&tmp_path, &content)?;
    fs::rename(&tmp_path, &final_path)?;

    eprintln!("\n  Deep research written to {}", final_path.display());
    eprintln!(
        "  ({} reasoning turns, {} tool events)",
        result.reasoning_turns,
        result.tool_events.len()
    );

    // Also print the summary to stdout
    println!("{summary}");

    Ok(())
}

struct ResearchHook;

impl crate::agent::AgentHook for ResearchHook {
    fn on_response(&mut self, iteration: usize, _content: Option<&str>, tool_calls: usize) {
        if tool_calls > 0 {
            eprintln!("    iteration {iteration}: {tool_calls} tool call(s)");
        } else {
            eprintln!("    iteration {iteration}: reasoning done");
        }
    }

    fn on_tool_call(&mut self, _iteration: usize, name: &str, args: &str, _result: &str) {
        let short_args: String = args.chars().take(80).collect();
        eprintln!("      -> {name}({short_args})");
    }
}
