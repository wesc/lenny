use anyhow::{Context, Result};
use linfa::DatasetBase;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rig::client::CompletionClient;
use rig::providers::openrouter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::Config;
use crate::db;
use crate::embed;

/// Structured output for cluster summaries.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ClusterSummary {
    /// 5 core mental models that describe this topic
    pub mental_models: Vec<String>,
    /// 3 strong statements supporting the key ideas, with reasoning
    pub supporting_statements: Vec<Statement>,
    /// 3 strong statements disagreeing with the key ideas, with reasoning
    pub dissenting_statements: Vec<Statement>,
    /// A detailed summary paying attention to key entities and concepts
    pub summary: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct Statement {
    pub statement: String,
    pub reasoning: String,
}

impl std::fmt::Display for ClusterSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "## Mental Models")?;
        for (i, m) in self.mental_models.iter().enumerate() {
            writeln!(f, "{}. {}", i + 1, m)?;
        }
        writeln!(f, "\n## Supporting Statements")?;
        for s in &self.supporting_statements {
            writeln!(f, "- **{}**\n  {}", s.statement, s.reasoning)?;
        }
        writeln!(f, "\n## Dissenting Statements")?;
        for s in &self.dissenting_statements {
            writeln!(f, "- **{}**\n  {}", s.statement, s.reasoning)?;
        }
        writeln!(f, "\n## Summary\n{}", self.summary)?;
        Ok(())
    }
}

pub async fn run(config: &Config, rumination_id: &str, k_override: Option<usize>) -> Result<()> {
    let pages_dir = config.pages_dir().join(rumination_id);
    if !pages_dir.exists() {
        anyhow::bail!("Pages directory not found: {}", pages_dir.display());
    }

    // Load all pages
    let pages = load_pages(&pages_dir)?;
    let n = pages.len();
    if n == 0 {
        anyhow::bail!("No pages found in {}", pages_dir.display());
    }
    println!("Loaded {n} pages from {}", pages_dir.display());

    let k = k_override.unwrap_or_else(|| {
        let auto_k = (n as f64).sqrt().round() as usize;
        auto_k.max(2)
    });
    println!("Clustering into {k} clusters");

    // Embed all pages
    println!("Embedding pages...");
    let texts: Vec<String> = pages
        .iter()
        .map(|p| truncate_for_embedding(&p.content))
        .collect();
    let embeddings = tokio::task::spawn_blocking(move || embed::embed_batch(texts)).await??;

    // Build ndarray matrix for linfa
    let dim = embeddings[0].len();
    let flat: Vec<f64> = embeddings.iter().flatten().map(|&v| v as f64).collect();
    let matrix = Array2::from_shape_vec((n, dim), flat)?;
    let dataset = DatasetBase::from(matrix);

    // Run k-means
    println!("Running k-means...");
    let rng = Xoshiro256Plus::seed_from_u64(42);
    let model = KMeans::params_with_rng(k, rng)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("K-means failed: {e}"))?;

    let predictions = model.predict(&dataset);
    let assignments = predictions.as_targets();

    // Group pages by cluster
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (page_idx, &cluster_idx) in assignments.iter().enumerate() {
        clusters.entry(cluster_idx).or_default().push(page_idx);
    }

    // Compute centroids as mean of member embeddings (in f32 for sqlite-vec)
    let mut centroids: HashMap<usize, Vec<f32>> = HashMap::new();
    for (&cluster_idx, member_indices) in &clusters {
        let mut centroid = vec![0.0f32; dim];
        for &page_idx in member_indices {
            for (j, val) in embeddings[page_idx].iter().enumerate() {
                centroid[j] += val;
            }
        }
        let count = member_indices.len() as f32;
        for val in &mut centroid {
            *val /= count;
        }
        centroids.insert(cluster_idx, centroid);
    }

    // Generate summaries and store in DB
    let or_client = openrouter::Client::new(&config.openrouter_api_key)?;
    let extractor = or_client
        .extractor::<ClusterSummary>(&config.model)
        .preamble(
            "You are a research analyst. Given a collection of scraped web pages that belong \
             to the same topic cluster, produce a detailed structured summary.\n\n\
             IMPORTANT: You must return valid JSON matching the schema exactly.\n\
             - mental_models: a JSON array of exactly 5 strings\n\
             - supporting_statements: a JSON array of exactly 3 objects, each with \"statement\" and \"reasoning\" string fields\n\
             - dissenting_statements: a JSON array of exactly 3 objects, each with \"statement\" and \"reasoning\" string fields\n\
             - summary: a single string, approximately 5000 tokens, covering key entities, concepts, and relationships",
        )
        .retries(2)
        .build();

    let conn = db::open(&config.db_path())?;

    // Clear previous clusters for this rumination
    db::delete_clusters_for_rumination(&conn, rumination_id)?;

    for (cluster_idx, member_indices) in &clusters {
        println!("\nCluster {cluster_idx} ({} pages):", member_indices.len());

        // Assemble cluster content for the LLM
        let mut cluster_text = String::new();
        let mut page_paths: Vec<String> = Vec::new();
        for &page_idx in member_indices {
            let page = &pages[page_idx];
            page_paths.push(page.filename.clone());
            let truncated = truncate_content(&page.content, 2000);
            cluster_text.push_str(&format!(
                "\n---\nPage: {} ({})\n{}\n",
                page.filename, page.url, truncated
            ));
            println!("  - {} ({})", page.filename, page.url);
        }

        // Generate summary
        println!("  Generating summary...");
        let summary = match extractor.extract(&cluster_text).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Failed to generate summary for cluster {cluster_idx}: {e}");
                continue;
            }
        };

        let summary_text = summary.to_string();
        let centroid = centroids.get(cluster_idx).context("missing centroid")?;

        db::insert_cluster(
            &conn,
            rumination_id,
            *cluster_idx,
            &summary_text,
            &page_paths,
            centroid,
        )?;

        println!("  Summary stored ({} chars)", summary_text.len());
    }

    println!("\nClustering complete: {k} clusters stored for rumination {rumination_id}");
    Ok(())
}

struct Page {
    filename: String,
    url: String,
    content: String,
}

fn load_pages(dir: &PathBuf) -> Result<Vec<Page>> {
    let mut pages = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let content = std::fs::read_to_string(entry.path())?;
        let filename = entry.file_name().to_string_lossy().to_string();

        // Extract URL from header comment if present
        let url = content
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("<!-- url: "))
            .and_then(|line| line.strip_suffix(" -->"))
            .unwrap_or("unknown")
            .to_string();

        pages.push(Page {
            filename,
            url,
            content,
        });
    }

    Ok(pages)
}

/// Truncate for embedding input (fastembed has token limits).
fn truncate_for_embedding(content: &str) -> String {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() <= 500 {
        content.to_string()
    } else {
        words[..500].join(" ")
    }
}

/// Truncate content to approximately `max_words` words.
fn truncate_content(content: &str, max_words: usize) -> String {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() <= max_words {
        content.to_string()
    } else {
        words[..max_words].join(" ")
    }
}
