mod actions;
mod agent;
mod cli_bot;
mod config;
mod context;
mod contextual_indexer;
mod dream;
mod embed;
mod evals;
mod matrix_bot;
mod once;
mod research;
mod tools;

use clap::{Parser, Subcommand};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process;

/// Lenny1 — a CAA agent that assembles context from files, runs an LLM loop, and saves results.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to the config file
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a single prompt through the agent and print the answer
    Once {
        /// The prompt to send to the agent
        #[arg(required = true, trailing_var_arg = true)]
        prompt: Vec<String>,
    },
    /// Watch system/, dynamic/, and references/ for changes
    Dream {
        /// Force comprehension immediately without waiting for token thresholds
        #[arg(long)]
        force_comprehension: bool,
    },
    /// Run eval suites
    Eval {
        #[command(subcommand)]
        suite: EvalSuite,
    },
    /// Matrix chat bot
    Matrix {
        #[command(subcommand)]
        cmd: MatrixCmd,
    },
    /// CLI chat bot
    Cli {
        #[command(subcommand)]
        cmd: CliCmd,
    },
    /// Comprehension index operations
    Comprehension {
        #[command(subcommand)]
        cmd: ComprehensionCmd,
    },
    /// Run research tasks
    Research {
        #[command(subcommand)]
        cmd: ResearchCmd,
    },
    /// Extract information from a URL using Firecrawl
    FirecrawlExtract {
        /// URL to extract from
        url: String,
        /// Question / extraction prompt
        #[arg(required = true, trailing_var_arg = true)]
        prompt: Vec<String>,
    },
    /// Estimate token counts for files in a directory or a single file
    EstimateTokens {
        /// File or directory to scan
        path: PathBuf,
    },
    /// Probe model_candidates for supported reasoning effort levels
    DiscoverReasoningLevels,
}

#[derive(Subcommand)]
enum ResearchCmd {
    /// Search Bluesky for trending AI/ML links
    Bluesky {
        /// Start date (YYYY-MM-DD), defaults to 24 hours ago
        #[arg(long)]
        since: Option<String>,
        /// End date (YYYY-MM-DD), defaults to now
        #[arg(long)]
        until: Option<String>,
    },
}

#[derive(Subcommand)]
enum EvalSuite {
    /// Run basic eval battery
    Basic,
    /// Run contextual eval suite against chat fixture data
    ContextualChats,
    /// Run contextual eval suite against text fixture data
    ContextualTexts,
    /// Run both contextual eval suites (chats + texts)
    ContextualAll,
}

#[derive(Subcommand)]
enum MatrixCmd {
    /// Start the Matrix chat bot
    Bot {
        /// Clear sync state and re-sync from scratch (keeps device identity)
        #[arg(long)]
        reset: bool,
    },
}

#[derive(Subcommand)]
enum CliCmd {
    /// Start the interactive CLI bot
    Bot,
}

#[derive(Subcommand)]
enum ComprehensionCmd {
    /// Dump all comprehension entries as JSON
    Dump,
    /// Search comprehensions by semantic similarity
    Search {
        /// The search query
        #[arg(required = true, trailing_var_arg = true)]
        query: Vec<String>,
        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        n: usize,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let config = config::Config::load(Path::new(&cli.config)).unwrap_or_else(|e| {
        eprintln!("Error loading config: {e}");
        process::exit(1);
    });

    let result = match cli.command {
        Command::Once { prompt } => {
            let prompt = prompt.join(" ");
            once::run(&config, &prompt).await
        }
        Command::Dream {
            force_comprehension,
        } => dream::run(&config, force_comprehension).await,
        Command::Eval { suite } => match suite {
            EvalSuite::Basic => evals::run(&config).await,
            EvalSuite::ContextualChats => evals::contextual_chats::run(&config).await,
            EvalSuite::ContextualTexts => evals::contextual_texts::run(&config).await,
            EvalSuite::ContextualAll => evals::contextual_all::run(&config).await,
        },
        Command::Matrix { cmd } => match cmd {
            MatrixCmd::Bot { reset } => matrix_bot::run(&config, reset).await,
        },
        Command::Cli { cmd } => match cmd {
            CliCmd::Bot => {
                let stdin = io::stdin();
                let mut input = BufReader::new(stdin.lock());
                let mut output = io::stdout();
                cli_bot::chat_loop(&config, &mut input, &mut output).await
            }
        },
        Command::Comprehension { cmd } => match cmd {
            ComprehensionCmd::Dump => actions::comprehension::dump_json(&config).await,
            ComprehensionCmd::Search { query, n } => {
                let query = query.join(" ");
                actions::comprehension::search_json(&config, &query, n).await
            }
        },
        Command::Research { cmd } => match cmd {
            ResearchCmd::Bluesky { since, until } => {
                let now = chrono::Utc::now();
                let since = since.map(|d| format!("{d}T00:00:00Z")).unwrap_or_else(|| {
                    (now - chrono::Duration::hours(24))
                        .format("%Y-%m-%dT%H:%M:%SZ")
                        .to_string()
                });
                let until = until
                    .map(|d| {
                        // Next day at midnight so the entire end date is included
                        let date = chrono::NaiveDate::parse_from_str(&d, "%Y-%m-%d")
                            .expect("--until should be YYYY-MM-DD");
                        let next_day = date + chrono::Duration::days(1);
                        format!("{next_day}T00:00:00Z")
                    })
                    .unwrap_or_else(|| now.format("%Y-%m-%dT%H:%M:%SZ").to_string());
                research::bluesky::run(&config, &since, &until).await
            }
        },
        Command::FirecrawlExtract { url, prompt } => {
            firecrawl_extract(&config, &url, &prompt.join(" ")).await
        }
        Command::EstimateTokens { path } => estimate_tokens(&path),
        Command::DiscoverReasoningLevels => discover_reasoning_levels(&config).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

async fn firecrawl_extract(config: &config::Config, url: &str, prompt: &str) -> anyhow::Result<()> {
    let api_key = config
        .firecrawl_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("firecrawl_api_key not set in config"))?;
    let app = firecrawl::FirecrawlApp::new(api_key)?;

    let params = firecrawl::extract::ExtractParams {
        urls: Some(vec![url.to_string()]),
        prompt: Some(prompt.to_string()),
        ..Default::default()
    };

    eprintln!("Extracting from {url}...");
    let job = app.async_extract(params).await?;
    eprintln!("Job {} started, polling...", job.id);

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let status = app.get_extract_status(&job.id).await?;
        eprintln!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "status": status.status,
                "success": status.success,
            }))?
        );
        match status.status.as_str() {
            "completed" => {
                if let Some(data) = status.data {
                    println!("{}", serde_json::to_string(&data)?);
                } else {
                    eprintln!("No data returned");
                }
                return Ok(());
            }
            "failed" => {
                let msg = status.error.unwrap_or_else(|| "unknown error".to_string());
                anyhow::bail!("Extract failed: {msg}");
            }
            _ => continue,
        }
    }
}

fn estimate_tokens(path: &Path) -> anyhow::Result<()> {
    fn approx_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
        if path.is_file() {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)?.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let p = entry.path();
                if p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
                {
                    continue;
                }
                collect_files(&p, &mut *files)?;
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(path, &mut files)?;

    let mut total = 0usize;
    for file in &files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue, // skip binary / unreadable files
        };
        let tokens = approx_tokens(&content);
        total += tokens;
        println!("{tokens:>8}  {}", file.display());
    }
    println!("{total:>8}  total");

    Ok(())
}

async fn discover_reasoning_levels(config: &config::Config) -> anyhow::Result<()> {
    use config::ProviderConfig;
    use openrouter_rs::OpenRouterClient;

    if config.model_candidates.is_empty() {
        anyhow::bail!("No model_candidates configured");
    }

    let ProviderConfig::OpenRouter { ref api_key, .. } = config.provider;
    let client = OpenRouterClient::builder().api_key(api_key).build()?;

    eprintln!(
        "Probing {} model(s) for reasoning effort support...\n",
        config.model_candidates.len()
    );

    for model in &config.model_candidates {
        eprint!("  {model} ... ");

        match agent::probe_effort_range(&client, model).await {
            Ok(range) => {
                let supported_strs: Vec<String> =
                    range.supported.iter().map(|e| format!("{e}")).collect();
                let levels: Vec<String> = agent::EFFORT_LEVELS
                    .iter()
                    .map(|e| {
                        let s = format!("{e}");
                        if supported_strs.contains(&s) {
                            s
                        } else {
                            format!("({s})")
                        }
                    })
                    .collect();
                eprintln!(
                    "min={} max={}  levels: {}",
                    range.min,
                    range.max,
                    levels.join(" ")
                );
            }
            Err(e) => {
                eprintln!("FAILED: {e}");
            }
        }
    }

    Ok(())
}
