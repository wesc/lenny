use lenny1::contextual_indexer::{flatten_windows, window};

fn make_docs(n: usize) -> Vec<(String, String)> {
    (0..n)
        .map(|i| (format!("doc-{i:02}"), format!("content of doc {i}")))
        .collect()
}

#[test]
fn test_window_centered() {
    let docs = make_docs(20);
    let w = window(&docs, 10, 5);
    assert_eq!(w.len(), 5);
    assert_eq!(w[0].0, "doc-08");
    assert_eq!(w[4].0, "doc-12");
}

#[test]
fn test_window_at_start() {
    let docs = make_docs(20);
    let w = window(&docs, 0, 5);
    assert_eq!(w.len(), 5);
    assert_eq!(w[0].0, "doc-00");
    assert_eq!(w[4].0, "doc-04");
}

#[test]
fn test_window_at_end() {
    let docs = make_docs(20);
    let w = window(&docs, 19, 5);
    assert_eq!(w.len(), 5);
    assert_eq!(w[0].0, "doc-15");
    assert_eq!(w[4].0, "doc-19");
}

#[test]
fn test_window_small_collection() {
    let docs = make_docs(3);
    let w = window(&docs, 1, 5);
    assert_eq!(w.len(), 3); // Can't get 5 from 3 docs
    assert_eq!(w[0].0, "doc-00");
    assert_eq!(w[2].0, "doc-02");
}

#[test]
fn test_window_empty() {
    let docs: Vec<(String, String)> = Vec::new();
    let w = window(&docs, 0, 5);
    assert!(w.is_empty());
}

#[test]
fn test_flatten_deduplication() {
    let windows = vec![
        vec![
            ("a".to_string(), "content-a".to_string()),
            ("b".to_string(), "content-b".to_string()),
        ],
        vec![
            ("b".to_string(), "content-b".to_string()),
            ("c".to_string(), "content-c".to_string()),
        ],
    ];
    let result = flatten_windows(windows);
    // "b" should appear only once
    assert_eq!(result.matches("## b").count(), 1);
    assert!(result.contains("## a"));
    assert!(result.contains("## c"));
}

#[test]
fn test_flatten_preserves_order() {
    let windows = vec![
        vec![
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string()),
        ],
        vec![
            ("y".to_string(), "2".to_string()),
            ("z".to_string(), "3".to_string()),
        ],
    ];
    let result = flatten_windows(windows);
    let x_pos = result.find("## x").unwrap();
    let y_pos = result.find("## y").unwrap();
    let z_pos = result.find("## z").unwrap();
    assert!(x_pos < y_pos);
    assert!(y_pos < z_pos);
}

#[tokio::test]
async fn test_empty_retrieve() {
    let tmpdir = tempfile::tempdir().unwrap();
    let db_path = tmpdir.path().join("nonexistent-db");
    let result = lenny1::contextual_indexer::retrieve(&db_path, "anything", 5, None)
        .await
        .unwrap();
    assert!(result.matched_docs.is_empty());
    assert!(result.context.is_empty());
}
