mod build_context;
mod cluster;
mod config;
mod db;
mod download;
mod embed;
mod query;
mod tokens;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lenny2", about = "The Interrobot")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the download phase: scrape pages starting from a seed URL.
    Download {
        /// Seed URL to start from
        url: String,
        /// Maximum number of pages to download
        #[arg(long, default_value_t = 100)]
        max_pages: usize,
    },
    /// Cluster a downloaded page set and generate summaries.
    Cluster {
        /// Rumination ID (datetime directory name under pages/)
        rumination_id: String,
        /// Number of clusters (default: sqrt of page count)
        #[arg(long)]
        k: Option<usize>,
    },
    /// One-shot query against the knowledge base.
    Query {
        /// The prompt to query
        prompt: String,
    },
    /// Interactive chat loop against the knowledge base.
    Chat,
    /// Return the N closest cluster summaries to a prompt.
    BuildContext {
        /// The prompt to match against
        prompt: String,
        /// Number of summaries to return
        #[arg(short, default_value_t = 3)]
        n: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::load()?;

    match cli.command {
        Command::Download { url, max_pages } => {
            download::run(&config, &url, max_pages).await?;
        }
        Command::Cluster { rumination_id, k } => {
            cluster::run(&config, &rumination_id, k).await?;
        }
        Command::Query { prompt } => {
            query::run_once(&config, &prompt).await?;
        }
        Command::Chat => {
            query::run_chat(&config).await?;
        }
        Command::BuildContext { prompt, n } => {
            build_context::run(&config, &prompt, n).await?;
        }
    }

    Ok(())
}
