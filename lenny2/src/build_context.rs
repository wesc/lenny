use anyhow::Result;

use crate::config::Config;
use crate::db;
use crate::embed;

pub async fn run(config: &Config, prompt: &str, n: usize) -> Result<()> {
    let conn = db::open(&config.db_path())?;

    let query_embedding = tokio::task::spawn_blocking({
        let prompt = prompt.to_string();
        move || embed::embed_query(&prompt)
    })
    .await??;

    let results = db::find_nearest_clusters(&conn, &query_embedding, n)?;

    if results.is_empty() {
        println!("No clusters found in the database. Run `download` and `cluster` first.");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        println!(
            "=== Match {} (rumination: {}, cluster: {}, distance: {:.4}) ===\n",
            i + 1,
            result.rumination_id,
            result.cluster_index,
            result.distance
        );
        println!("{}\n", result.summary);
    }

    Ok(())
}
