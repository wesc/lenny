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
        Command::DiscoverReasoningLevels => discover_reasoning_levels(&config).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
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
