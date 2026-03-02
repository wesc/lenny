mod actions;
mod cli_bot;
mod config;
mod context;
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
    Dream,
    /// Run basic eval battery against fixture data and print results as JSON
    EvalBasic,
    /// Start the Matrix chat bot
    MatrixBot {
        /// Clear sync state and re-sync from scratch (keeps device identity)
        #[arg(long)]
        reset: bool,
    },
    /// Start the interactive CLI bot
    CliBot,
}

#[tokio::main]
async fn main() {
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
        Command::Dream => {
            if let Err(e) = dream::run(&config) {
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
        Command::MatrixBot { reset } => {
            if let Err(e) = matrix_bot::run(&config, reset).await {
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
