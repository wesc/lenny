mod actions;
mod cli_bot;
mod config;
mod context;
mod contextual_indexer;
mod dream;
mod evals;
mod matrix_bot;
mod once;
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
    /// Run basic eval battery against fixture data and print results as JSON
    EvalBasic,
    /// Run contextual indexer eval suite against chat fixture data
    EvalContextualIndexer,
    /// Start the Matrix chat bot
    MatrixBot {
        /// Clear sync state and re-sync from scratch (keeps device identity)
        #[arg(long)]
        reset: bool,
    },
    /// Start the interactive CLI bot
    CliBot,
    /// Dump all comprehension entries from LanceDB as JSON
    DumpComprehensions,
    /// Search comprehensions by semantic similarity and print top matches as JSON
    SearchComprehensions {
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

    match cli.command {
        Command::Once { prompt } => {
            let prompt = prompt.join(" ");
            if let Err(e) = once::run(&config, &prompt).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::Dream { force_comprehension } => {
            if let Err(e) = dream::run(&config, force_comprehension).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::EvalBasic => {
            if let Err(e) = evals::run(&config).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::EvalContextualIndexer => {
            if let Err(e) = evals::contextual_indexer::run(&config).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::MatrixBot { reset } => {
            if let Err(e) = matrix_bot::run(&config, reset).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::DumpComprehensions => {
            if let Err(e) = actions::comprehension::dump_json(&config).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::SearchComprehensions { query, n } => {
            let query = query.join(" ");
            if let Err(e) =
                actions::comprehension::search_json(&config, &query, n).await
            {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        Command::CliBot => {
            let stdin = io::stdin();
            let mut input = BufReader::new(stdin.lock());
            let mut output = io::stdout();
            if let Err(e) = cli_bot::chat_loop(&config, &mut input, &mut output).await {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
    }
}
