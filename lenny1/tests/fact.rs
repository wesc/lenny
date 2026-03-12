use lenny1::actions::fact::{
    self, collect_dynamic_files, move_to_references, parse_filename_timestamp, select_files_to_pop,
};
use lenny1::config::{Config, ProviderConfig};
use lenny1::embed;
use lenny1::tokens;
use std::fs;

fn make_config(tmpdir: &tempfile::TempDir) -> Config {
    Config {
        provider: ProviderConfig::test_default(),
        max_iterations: 1,
        system_dir: tmpdir.path().join("system"),
        dynamic_dir: tmpdir.path().join("dynamic"),
        knowledge_dir: tmpdir.path().join("knowledge"),
        max_context_tokens: 100,
        min_context_tokens: 50,
        matrix: None,
        min_score_range: 2.0,
        score_gap_threshold: 0.5,
        prompt_log: false,
        model_candidates: Vec::new(),
        firecrawl_api_key: None,
    }
}

/// Write a file with approximately `n_words` words to dynamic/.
/// Uses the required timestamp filename format.
fn write_dynamic_file(config: &Config, filename: &str, n_words: usize) {
    let path = config.dynamic_dir.join(filename);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let content: String = (0..n_words)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(path, content).unwrap();
}

// ---------------------------------------------------------------------------
// Filename timestamp parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_timestamp_valid() {
    let ts = parse_filename_timestamp("2025-11-15_10-30-00_abc123.txt").unwrap();
    assert_eq!(ts, "2025-11-15T10:30:00Z");
}

#[test]
fn parse_timestamp_no_extension() {
    let ts = parse_filename_timestamp("2025-11-15_10-30-00_abc123").unwrap();
    assert_eq!(ts, "2025-11-15T10:30:00Z");
}

#[test]
fn parse_timestamp_invalid_date() {
    assert!(parse_filename_timestamp("2025-13-01_10-30-00_abc.txt").is_err());
}

#[test]
fn parse_timestamp_too_short() {
    assert!(parse_filename_timestamp("short.txt").is_err());
}

#[test]
fn parse_timestamp_bad_separator() {
    assert!(parse_filename_timestamp("2025-11-15X10-30-00_abc.txt").is_err());
}

// ---------------------------------------------------------------------------
// File collection and threshold tests
// ---------------------------------------------------------------------------

#[test]
fn skips_files_without_timestamp_format() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    // One valid, one invalid filename
    write_dynamic_file(&config, "2025-11-15_10-00-00_valid.txt", 20);
    write_dynamic_file(&config, "no-timestamp.txt", 20);

    let files = collect_dynamic_files(&config).unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].relative.to_string_lossy().contains("valid"));
}

#[test]
fn below_threshold_noop() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    write_dynamic_file(&config, "2025-11-15_10-00-00_small.txt", 20);

    let files = collect_dynamic_files(&config).unwrap();
    let total: usize = files.iter().map(|f| f.tokens).sum();
    assert!(total <= config.max_context_tokens);

    let (to_pop, keep) =
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens);
    assert!(to_pop.is_empty());
    assert_eq!(keep.len(), 1);
}

#[test]
fn pops_oldest_files_by_timestamp() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    // Create 4 files with timestamps — note the oldest is first by timestamp, not filename
    write_dynamic_file(&config, "2025-11-01_08-00-00_aaa.txt", 40);
    write_dynamic_file(&config, "2025-11-02_09-00-00_bbb.txt", 40);
    write_dynamic_file(&config, "2025-11-03_10-00-00_ccc.txt", 40);
    write_dynamic_file(&config, "2025-11-04_11-00-00_ddd.txt", 40);

    let files = collect_dynamic_files(&config).unwrap();
    let total: usize = files.iter().map(|f| f.tokens).sum();
    assert!(total > config.max_context_tokens);

    let (to_pop, keep) =
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens);

    assert!(!to_pop.is_empty());
    let remaining: usize = keep.iter().map(|f| f.tokens).sum();
    assert!(
        remaining <= config.min_context_tokens,
        "remaining {remaining} > min {}",
        config.min_context_tokens
    );

    // The oldest files (by timestamp) should be popped first
    assert_eq!(to_pop[0].created_at, "2025-11-01T08:00:00Z");
}

#[test]
fn created_at_extracted_from_filename() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);

    write_dynamic_file(&config, "2025-12-25_14-30-00_xmas.txt", 10);

    let files = collect_dynamic_files(&config).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].created_at, "2025-12-25T14:30:00Z");
}

#[test]
fn preserves_subdir_structure() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    fs::create_dir_all(config.references_dir()).unwrap();

    write_dynamic_file(&config, "cli/2025-11-15_10-00-00_session.txt", 40);

    let files = collect_dynamic_files(&config).unwrap();
    move_to_references(&config, &files).unwrap();

    let dest = config
        .references_dir()
        .join("cli/2025-11-15_10-00-00_session.txt");
    assert!(dest.exists(), "file should exist at references/cli/...");

    let src = config
        .dynamic_dir
        .join("cli/2025-11-15_10-00-00_session.txt");
    assert!(!src.exists(), "file should be removed from dynamic/");
}

#[test]
fn atomic_write_no_temp_files() {
    let tmpdir = tempfile::tempdir().unwrap();
    let config = make_config(&tmpdir);
    fs::create_dir_all(config.references_dir()).unwrap();

    write_dynamic_file(&config, "2025-11-01_10-00-00_old.txt", 40);
    write_dynamic_file(&config, "2025-11-02_10-00-00_new.txt", 40);

    let files = collect_dynamic_files(&config).unwrap();
    move_to_references(&config, &files).unwrap();

    let entries: Vec<String> = fs::read_dir(config.references_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    for entry in &entries {
        assert!(
            !entry.starts_with(".tmp-"),
            "temp file left behind: {entry}"
        );
    }
}

// ---------------------------------------------------------------------------
// DB round-trip tests (uses fastembed, no LLM)
// ---------------------------------------------------------------------------

/// Helper: insert facts directly into the DB (bypasses LLM, uses fastembed).
fn insert_facts(db_path: &std::path::Path, facts: &[(&str, &str, &str)]) {
    let fact_texts: Vec<String> = facts.iter().map(|(f, _, _)| f.to_string()).collect();
    let embeddings = embed::embed_batch(fact_texts).unwrap();
    let conn = embed::open_db(db_path).unwrap();

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS facts (
            created_at TEXT NOT NULL,
            fact TEXT NOT NULL,
            file_reference TEXT NOT NULL,
            token_count INTEGER NOT NULL
         );
         CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[384]);",
    )
    .unwrap();

    let max_rowid: i64 = conn
        .query_row("SELECT COALESCE(MAX(rowid), 0) FROM facts", [], |row| {
            row.get(0)
        })
        .unwrap();

    let tx = conn.unchecked_transaction().unwrap();
    for (i, (fact_text, file_ref, created_at)) in facts.iter().enumerate() {
        let rowid = max_rowid + (i as i64) + 1;
        let token_count = tokens::count_tokens(fact_text);
        tx.execute(
            "INSERT INTO facts (rowid, created_at, fact, file_reference, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![rowid, created_at, fact_text, file_ref, token_count as i64],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO facts_vec (rowid, embedding) VALUES (?1, ?2)",
            rusqlite::params![rowid, embed::f32_to_bytes(&embeddings[i])],
        )
        .unwrap();
    }
    tx.commit().unwrap();
}

#[tokio::test]
async fn db_roundtrip_retrieve() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("memory.db");

    let facts = &[
        (
            "The assistant's favorite color is chartreuse",
            "preferences.txt",
            "2025-11-15T10:00:00Z",
        ),
        (
            "The assistant's favorite car is a Toyota Pineapple",
            "preferences.txt",
            "2025-11-15T10:00:00Z",
        ),
        (
            "Project Alpha deadline is March 15, 2026",
            "project-alpha.txt",
            "2025-11-16T09:00:00Z",
        ),
        (
            "The team uses PostgreSQL for the primary database",
            "tech-decisions.txt",
            "2025-11-17T14:00:00Z",
        ),
        (
            "The assistant visited Tokyo in October 2024",
            "travel.txt",
            "2025-11-18T08:00:00Z",
        ),
        (
            "The assistant is allergic to shellfish",
            "cooking-notes.txt",
            "2025-11-19T12:00:00Z",
        ),
    ];

    insert_facts(&db_path, facts);

    // Search for color — should find chartreuse
    let result = fact::retrieve(&db_path, "what is your favorite color", 3)
        .await
        .unwrap();
    assert!(
        result.context.to_lowercase().contains("chartreuse"),
        "expected 'chartreuse' in results, got: {}",
        result.context
    );

    // Search for database — should find postgresql
    let result = fact::retrieve(&db_path, "what database do we use", 3)
        .await
        .unwrap();
    assert!(
        result.context.to_lowercase().contains("postgresql"),
        "expected 'postgresql' in results, got: {}",
        result.context
    );

    // Search for travel — should find tokyo
    let result = fact::retrieve(&db_path, "tell me about Japan trip", 3)
        .await
        .unwrap();
    assert!(
        result.context.to_lowercase().contains("tokyo"),
        "expected 'tokyo' in results, got: {}",
        result.context
    );
}

#[tokio::test]
async fn db_retrieve_file_reference() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("memory.db");

    let facts = &[
        (
            "The CI/CD pipeline runs on GitHub Actions",
            "tech-decisions.txt",
            "2025-11-15T10:00:00Z",
        ),
        (
            "Sprint planning happens every Monday at 10am",
            "project-alpha.txt",
            "2025-11-16T10:00:00Z",
        ),
    ];

    insert_facts(&db_path, facts);

    let result = fact::retrieve(&db_path, "CI/CD pipeline", 2).await.unwrap();
    assert!(
        result.context.contains("tech-decisions.txt"),
        "expected 'tech-decisions.txt' file reference in: {}",
        result.context
    );
}

#[tokio::test]
async fn db_retrieve_empty_db() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("nonexistent.db");

    let result = fact::retrieve(&db_path, "anything", 5).await.unwrap();
    assert!(
        result.context.is_empty(),
        "expected empty context for missing DB"
    );
}

#[tokio::test]
async fn db_token_count_stored() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("memory.db");

    let fact_text = "The assistant prefers dark mode in all applications";
    insert_facts(
        &db_path,
        &[(fact_text, "preferences.txt", "2025-11-15T10:00:00Z")],
    );

    let conn = embed::open_db(&db_path).unwrap();
    let stored_count: i64 = conn
        .query_row("SELECT token_count FROM facts WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .unwrap();

    let expected = tokens::count_tokens(fact_text);
    assert_eq!(
        stored_count, expected as i64,
        "stored token count {stored_count} != expected {expected}"
    );
}

#[tokio::test]
async fn db_created_at_stored() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("memory.db");

    insert_facts(
        &db_path,
        &[("Some fact", "file.txt", "2025-12-25T14:30:00Z")],
    );

    let conn = embed::open_db(&db_path).unwrap();
    let stored: String = conn
        .query_row("SELECT created_at FROM facts WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .unwrap();

    assert_eq!(stored, "2025-12-25T14:30:00Z");
}

#[tokio::test]
async fn db_search_relevance_across_topics() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("memory.db");

    let facts = &[
        // Cooking cluster
        (
            "The secret to pizza dough is 72 hour cold fermentation",
            "cooking.txt",
            "2025-11-15T10:00:00Z",
        ),
        (
            "The assistant's sourdough starter is named Bubbles",
            "cooking.txt",
            "2025-11-15T10:00:00Z",
        ),
        (
            "Thai green curry uses homemade paste with lemongrass",
            "cooking.txt",
            "2025-11-15T10:00:00Z",
        ),
        // Tech cluster
        (
            "Kubernetes is used for container orchestration",
            "tech.txt",
            "2025-11-16T10:00:00Z",
        ),
        (
            "The CI/CD pipeline runs on GitHub Actions",
            "tech.txt",
            "2025-11-16T10:00:00Z",
        ),
        (
            "Monitoring uses Grafana dashboards with Prometheus",
            "tech.txt",
            "2025-11-16T10:00:00Z",
        ),
    ];

    insert_facts(&db_path, facts);

    // Search cooking — top result should be from cooking, not tech
    let result = fact::retrieve(&db_path, "bread baking fermentation dough", 2)
        .await
        .unwrap();
    let lower = result.context.to_lowercase();
    assert!(
        lower.contains("fermentation") || lower.contains("sourdough"),
        "cooking search should return cooking facts, got: {}",
        result.context
    );
    assert!(
        !lower.contains("kubernetes"),
        "cooking search should not return kubernetes"
    );

    // Search tech — top result should be from tech, not cooking
    let result = fact::retrieve(&db_path, "deployment infrastructure containers", 2)
        .await
        .unwrap();
    let lower = result.context.to_lowercase();
    assert!(
        lower.contains("kubernetes")
            || lower.contains("github actions")
            || lower.contains("grafana"),
        "tech search should return tech facts, got: {}",
        result.context
    );
    assert!(
        !lower.contains("sourdough") && !lower.contains("pizza"),
        "tech search should not return cooking facts"
    );
}
