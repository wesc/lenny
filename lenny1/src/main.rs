mod config;
mod context;
mod dream;
mod evals;
mod once;
mod tools;

use clap::{Parser, Subcommand};
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
    /// Start the Matrix chat bot (stub)
    BotMatrix,
    /// Start the interactive CLI bot (stub)
    BotCli,
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
        Command::BotMatrix => {
            println!("Hello, world!");
        }
        Command::BotCli => {
            println!("Hello, world!");
        }
    }
}
