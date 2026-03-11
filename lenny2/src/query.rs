use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message, Prompt};
use rig::providers::openrouter;
use std::io::Write;

use crate::config::Config;
use crate::db;
use crate::embed;

/// Build context string from the N nearest cluster summaries.
async fn build_context(conn: &rusqlite::Connection, prompt: &str, n: usize) -> Result<String> {
    let query_embedding = tokio::task::spawn_blocking({
        let prompt = prompt.to_string();
        move || embed::embed_query(&prompt)
    })
    .await??;

    let results = db::find_nearest_clusters(conn, &query_embedding, n)?;

    let mut context = String::new();
    for (i, result) in results.iter().enumerate() {
        context.push_str(&format!(
            "=== Topic {} (distance: {:.4}) ===\n{}\n\n",
            i + 1,
            result.distance,
            result.summary
        ));
    }
    Ok(context)
}

fn system_prompt(context: &str) -> String {
    format!(
        "You are Lenny, an Interrobot — you explore and interrogate knowledge and beliefs. \
         Use the following research context to inform your answers. Be direct, insightful, \
         and willing to present multiple perspectives.\n\n\
         # Research Context\n\n{context}"
    )
}

/// One-shot query.
pub async fn run_once(config: &Config, prompt: &str) -> Result<()> {
    let conn = db::open(&config.db_path())?;
    let context = build_context(&conn, prompt, 3).await?;

    let or_client = openrouter::Client::new(&config.openrouter_api_key)?;
    let agent = or_client
        .agent(&config.model)
        .preamble(&system_prompt(&context))
        .build();

    let response = agent.prompt(prompt).await?;
    println!("{response}");
    Ok(())
}

/// Interactive chat loop.
pub async fn run_chat(config: &Config) -> Result<()> {
    let conn = db::open(&config.db_path())?;
    let or_client = openrouter::Client::new(&config.openrouter_api_key)?;

    let mut history: Vec<Message> = Vec::new();

    println!("Lenny2 chat (type 'quit' to exit)\n");

    loop {
        // Read user input
        let mut input = String::new();
        eprint!("> ");
        std::io::stderr().flush()?;
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Build context from prompt + recent history
        let context_query = if history.len() > 2 {
            let recent: String = history
                .iter()
                .rev()
                .take(4)
                .filter_map(|m| match m {
                    Message::User { content } => Some(format!("{content:?}")),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{recent} {input}")
        } else {
            input.to_string()
        };

        let context = build_context(&conn, &context_query, 3).await?;

        let agent = or_client
            .agent(&config.model)
            .preamble(&system_prompt(&context))
            .build();

        match agent.chat(input, history.clone()).await {
            Ok(response) => {
                println!("\n{response}\n");
                history.push(Message::user(input));
                history.push(Message::assistant(&response));
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
    }

    Ok(())
}
