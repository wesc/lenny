use anyhow::Result;
use std::path::PathBuf;

use crate::config::Config;

use super::{Check, ContextualEval};

const EVALS: &[ContextualEval] = &[
    // --- positive: queries about Common Sense topics should match its chunks ---
    ContextualEval {
        name: "monarchy_hereditary",
        query: "monarchy and hereditary succession arguments against kings",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "government_vs_society",
        query: "difference between government and society, necessary evil",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "american_independence",
        query: "American independence from Britain, separation from England",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "reconciliation_britain",
        query: "reconciliation with Britain is impossible, arguments against peace",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "military_naval_ability",
        query: "America's military and naval ability to defend itself",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "english_constitution",
        query: "English constitution, House of Commons, House of Lords",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "scriptural_kings",
        query: "scripture Bible arguments against kings, Israel wanting a king",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    ContextualEval {
        name: "continental_union",
        query: "continental union, charter for the colonies, representative government",
        top_k: 5,
        time_range: None,
        check: Check::MatchedPrefixContains("thomas-paine-common-sense", 1),
    },
    // --- negative: unrelated queries should NOT match Common Sense chunks ---
    ContextualEval {
        name: "neg_cooking",
        query: "how to make pasta carbonara from scratch",
        top_k: 3,
        time_range: None,
        check: Check::MatchedPrefixExcludes("thomas-paine-common-sense"),
    },
    ContextualEval {
        name: "neg_programming",
        query: "rust borrow checker lifetime errors and debugging",
        top_k: 3,
        time_range: None,
        check: Check::MatchedPrefixExcludes("thomas-paine-common-sense"),
    },
];

/// Load .txt files from tests/fixtures/texts/ as documents.
/// Uses file stem as name, empty timestamp.
fn load_text_documents() -> Result<Vec<(String, String)>> {
    let fixture_dir = PathBuf::from("tests/fixtures/texts");
    let mut entries: Vec<_> = std::fs::read_dir(&fixture_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut documents = Vec::new();
    for entry in entries {
        let name = entry
            .path()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let content = std::fs::read_to_string(entry.path())?;
        documents.push((name, content));
    }
    Ok(documents)
}

pub async fn run(base_config: &Config) -> Result<()> {
    let documents = load_text_documents()?;
    let timestamps: Vec<String> = vec![String::new(); documents.len()];

    super::run_contextual_evals("texts", &documents, &timestamps, EVALS, base_config).await?;

    Ok(())
}
