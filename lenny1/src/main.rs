mod actions;
mod cli_bot;
mod config;
mod context;
mod contextual_indexer;
mod dream;
mod embed;
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
        Command::Dream { force_comprehension } => {
            dream::run(&config, force_comprehension).await
        }
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
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
