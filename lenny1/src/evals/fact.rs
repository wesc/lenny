use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

use crate::actions::fact::{self, DynamicFile};
use crate::config::Config;

// ---------------------------------------------------------------------------
// Eval types
// ---------------------------------------------------------------------------

struct FactEval {
    name: &'static str,
    query: &'static str,
    top_k: usize,
    check: FactCheck,
}

enum FactCheck {
    /// At least `min` of these keywords appear across the returned facts.
    ContainsAny {
        keywords: &'static [&'static str],
        min: usize,
    },
    /// None of these keywords appear in the returned facts.
    ExcludesAll { keywords: &'static [&'static str] },
    /// At least one returned fact references a file matching this substring.
    HasFileRef(&'static str),
}

fn run_check(context: &str, check: &FactCheck) -> (bool, String) {
    let lower = context.to_lowercase();
    match check {
        FactCheck::ContainsAny { keywords, min } => {
            let found: Vec<&&str> = keywords
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .collect();
            let pass = found.len() >= *min;
            let reason = if pass {
                format!(
                    "found {}/{} keywords: {:?}",
                    found.len(),
                    keywords.len(),
                    found
                )
            } else {
                format!(
                    "only found {}/{} keywords (need {}): {:?}",
                    found.len(),
                    keywords.len(),
                    min,
                    found
                )
            };
            (pass, reason)
        }
        FactCheck::ExcludesAll { keywords } => {
            let found: Vec<&&str> = keywords
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .collect();
            let pass = found.is_empty();
            let reason = if pass {
                "correctly excluded all keywords".to_string()
            } else {
                format!("unexpected keywords found: {found:?}")
            };
            (pass, reason)
        }
        FactCheck::HasFileRef(substring) => {
            let pass = lower.contains(&substring.to_lowercase());
            let reason = if pass {
                format!("found file reference containing '{substring}'")
            } else {
                format!("no file reference containing '{substring}' in results")
            };
            (pass, reason)
        }
    }
}

// ---------------------------------------------------------------------------
// Extraction quality evals
// ---------------------------------------------------------------------------

/// Check that facts extracted from a file contain expected content.
struct ExtractionEval {
    name: &'static str,
    /// Which fixture file to digest (relative path under fact-data/).
    fixture_file: &'static str,
    /// Keywords that should appear in at least one extracted fact.
    expected_keywords: &'static [&'static str],
    /// Minimum number of expected keywords that must appear.
    min_keywords: usize,
    /// Minimum number of facts we expect to be extracted.
    min_facts: usize,
}

const EXTRACTION_EVALS: &[ExtractionEval] = &[
    ExtractionEval {
        name: "extract_preferences",
        fixture_file: "2025-11-01_10-00-00_preferences.txt",
        expected_keywords: &[
            "chartreuse",
            "toyota",
            "pineapple",
            "dark mode",
            "rust",
            "earl grey",
            "hofstadter",
        ],
        min_keywords: 5,
        min_facts: 4,
    },
    ExtractionEval {
        name: "extract_project",
        fixture_file: "2025-11-02_10-00-00_project-alpha.txt",
        expected_keywords: &[
            "sarah",
            "march 15",
            "postgresql",
            "redis",
            "collaboration",
            "staging",
        ],
        min_keywords: 4,
        min_facts: 4,
    },
    ExtractionEval {
        name: "extract_tech_decisions",
        fixture_file: "2025-11-03_10-00-00_tech-decisions.txt",
        expected_keywords: &[
            "postgresql",
            "grpc",
            "kubernetes",
            "auth0",
            "github actions",
            "grafana",
            "trunk-based",
        ],
        min_keywords: 5,
        min_facts: 4,
    },
    ExtractionEval {
        name: "extract_meeting",
        fixture_file: "2025-11-04_10-00-00_meeting-notes.txt",
        expected_keywords: &[
            "budget",
            "migration",
            "component library",
            "mobile",
            "november 22",
        ],
        min_keywords: 3,
        min_facts: 3,
    },
    ExtractionEval {
        name: "extract_cooking",
        fixture_file: "2025-11-05_10-00-00_cooking-notes.txt",
        expected_keywords: &[
            "pizza",
            "fermentation",
            "thai",
            "shellfish",
            "sourdough",
            "aglio e olio",
        ],
        min_keywords: 4,
        min_facts: 4,
    },
    ExtractionEval {
        name: "extract_travel",
        fixture_file: "2025-11-06_10-00-00_travel-experiences.txt",
        expected_keywords: &[
            "tokyo",
            "shimokitazawa",
            "fuji",
            "patagonia",
            "23 countries",
            "ramen",
        ],
        min_keywords: 4,
        min_facts: 4,
    },
];

// ---------------------------------------------------------------------------
// Retrieval evals (run after digesting all fixtures)
// ---------------------------------------------------------------------------

const RETRIEVAL_EVALS: &[FactEval] = &[
    // --- positive: correct topic retrieved ---
    FactEval {
        name: "search_favorite_color",
        query: "what is your favorite color",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["chartreuse"],
            min: 1,
        },
    },
    FactEval {
        name: "search_favorite_car",
        query: "what car do you like",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["toyota", "pineapple"],
            min: 1,
        },
    },
    FactEval {
        name: "search_project_deadline",
        query: "when is the project deadline",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["march 15", "project alpha"],
            min: 1,
        },
    },
    FactEval {
        name: "search_team_members",
        query: "who is on the engineering team",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["sarah", "marcus", "priya", "jake"],
            min: 2,
        },
    },
    FactEval {
        name: "search_database_choice",
        query: "why did we choose our database",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["postgresql", "json", "full-text"],
            min: 1,
        },
    },
    FactEval {
        name: "search_ci_cd",
        query: "what CI/CD system do we use",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["github actions", "jenkins"],
            min: 1,
        },
    },
    FactEval {
        name: "search_travel_tokyo",
        query: "tell me about the trip to Tokyo",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["tokyo", "shimokitazawa", "fuji"],
            min: 2,
        },
    },
    FactEval {
        name: "search_cooking_pizza",
        query: "how to make pizza",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["pizza", "fermentation", "tipo 00"],
            min: 1,
        },
    },
    FactEval {
        name: "search_food_allergies",
        query: "do you have any food allergies",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &["shellfish", "allergic"],
            min: 1,
        },
    },
    FactEval {
        name: "search_meeting_action_items",
        query: "what were the action items from the meeting",
        top_k: 5,
        check: FactCheck::ContainsAny {
            keywords: &[
                "migration plan",
                "component library",
                "november 22",
                "december",
            ],
            min: 1,
        },
    },
    // --- file reference checks ---
    FactEval {
        name: "ref_preferences",
        query: "favorite color",
        top_k: 3,
        check: FactCheck::HasFileRef("preferences"),
    },
    FactEval {
        name: "ref_project",
        query: "project deadline budget",
        top_k: 3,
        check: FactCheck::HasFileRef("project-alpha"),
    },
    FactEval {
        name: "ref_travel",
        query: "tokyo japan travel",
        top_k: 3,
        check: FactCheck::HasFileRef("travel"),
    },
    // --- negative: irrelevant queries should not match specific topics ---
    FactEval {
        name: "neg_cooking_excludes_kubernetes",
        query: "pizza dough fermentation sourdough",
        top_k: 3,
        check: FactCheck::ExcludesAll {
            keywords: &["kubernetes", "github actions", "grpc"],
        },
    },
    FactEval {
        name: "neg_tech_excludes_cooking",
        query: "kubernetes container orchestration deployment",
        top_k: 3,
        check: FactCheck::ExcludesAll {
            keywords: &["pizza", "sourdough", "thai curry", "shellfish"],
        },
    },
    FactEval {
        name: "neg_travel_excludes_project",
        query: "tokyo japan mount fuji hiking",
        top_k: 3,
        check: FactCheck::ExcludesAll {
            keywords: &["sprint planning", "staging environment", "budget"],
        },
    },
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn load_fixture_files() -> Result<Vec<DynamicFile>> {
    let fixture_dir = PathBuf::from("tests/fixtures/fact-data");
    let mut entries: Vec<_> = std::fs::read_dir(&fixture_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut files = Vec::new();
    for entry in entries {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        let created_at = fact::parse_filename_timestamp(&filename)?;
        let relative: PathBuf = filename.into();
        let content = std::fs::read_to_string(&path)?;
        let tokens = crate::tokens::count_tokens(&content);
        files.push(DynamicFile {
            path,
            relative,
            content,
            tokens,
            created_at,
        });
    }
    Ok(files)
}

pub async fn run(base_config: &Config) -> Result<()> {
    let tmpdir = tempfile::tempdir()?;
    let config = Config {
        provider: base_config.provider.clone(),
        max_iterations: base_config.max_iterations,
        system_dir: PathBuf::from("tests/fixtures/basic-evals/system"),
        dynamic_dir: tmpdir.path().join("dynamic"),
        knowledge_dir: tmpdir.path().join("knowledge"),
        max_context_tokens: base_config.max_context_tokens,
        min_context_tokens: base_config.min_context_tokens,
        matrix: None,
        min_score_range: base_config.min_score_range,
        score_gap_threshold: base_config.score_gap_threshold,
        prompt_log: false,
        model_candidates: Vec::new(),
        firecrawl_api_key: None,
    };

    // Create knowledge dir for memory.db
    std::fs::create_dir_all(config.knowledge_dir.join("references"))?;

    eprintln!(
        "  [fact] comprehension model: {}",
        config.provider.comprehension_model()
    );

    // Phase 1: Extraction quality
    eprintln!("\n  === Phase 1: Fact Extraction Quality ===\n");

    let mut extraction_passed = 0;
    let extraction_total = EXTRACTION_EVALS.len();
    let mut extraction_results = Vec::new();
    let extraction_start = Instant::now();

    for eval in EXTRACTION_EVALS {
        eprint!("  {} ... ", eval.name);
        let eval_start = Instant::now();

        let fixture_path = PathBuf::from("tests/fixtures/fact-data").join(eval.fixture_file);
        let content = std::fs::read_to_string(&fixture_path)?;
        let tokens = crate::tokens::count_tokens(&content);

        // Create a temp dynamic dir for this single file
        let file_tmpdir = tempfile::tempdir()?;
        let file_config = Config {
            knowledge_dir: file_tmpdir.path().join("knowledge"),
            dynamic_dir: file_tmpdir.path().join("dynamic"),
            ..config.clone()
        };
        std::fs::create_dir_all(file_config.knowledge_dir.join("references"))?;

        let created_at = fact::parse_filename_timestamp(eval.fixture_file)?;
        let dyn_file = DynamicFile {
            path: fixture_path.clone(),
            relative: PathBuf::from(eval.fixture_file),
            content: content.clone(),
            tokens,
            created_at,
        };

        // We can't use digest() because it moves files. Instead, use the LLM directly.
        let output: fact::FactOutput = crate::once::run_completion_typed_with_model(
            &config,
            config.provider.comprehension_model(),
            super::super::actions::fact::FACT_PREAMBLE,
            &format!("File ({}):\n{}", eval.fixture_file, dyn_file.content),
        )
        .await?;

        let elapsed = eval_start.elapsed().as_secs_f64();

        // Check extraction quality
        let all_facts = output.facts.join(" ").to_lowercase();
        let found_keywords: Vec<&&str> = eval
            .expected_keywords
            .iter()
            .filter(|kw| all_facts.contains(&kw.to_lowercase()))
            .collect();

        let enough_keywords = found_keywords.len() >= eval.min_keywords;
        let enough_facts = output.facts.len() >= eval.min_facts;
        let pass = enough_keywords && enough_facts;

        if pass {
            extraction_passed += 1;
        }

        eprintln!(
            "{} ({:.1}s) [{} facts, {}/{} keywords]",
            if pass { "PASS" } else { "FAIL" },
            elapsed,
            output.facts.len(),
            found_keywords.len(),
            eval.expected_keywords.len(),
        );

        if !pass {
            if !enough_facts {
                eprintln!(
                    "    too few facts: got {}, need {}",
                    output.facts.len(),
                    eval.min_facts
                );
            }
            if !enough_keywords {
                let missing: Vec<&&str> = eval
                    .expected_keywords
                    .iter()
                    .filter(|kw| !all_facts.contains(&kw.to_lowercase()))
                    .collect();
                eprintln!("    missing keywords: {missing:?}");
            }
            eprintln!("    extracted facts:");
            for (i, f) in output.facts.iter().enumerate() {
                eprintln!("      {}. {}", i + 1, f);
            }
        }

        extraction_results.push(json!({
            "name": eval.name,
            "pass": pass,
            "fact_count": output.facts.len(),
            "keywords_found": found_keywords.len(),
            "keywords_total": eval.expected_keywords.len(),
            "elapsed_s": (elapsed * 10.0).round() / 10.0,
        }));
    }

    let extraction_elapsed = extraction_start.elapsed().as_secs_f64();
    eprintln!(
        "\n  [extraction] {extraction_passed}/{extraction_total} passed in {extraction_elapsed:.1}s"
    );

    // Phase 2: Digest all fixtures into a single DB, then run retrieval evals
    eprintln!("\n  === Phase 2: Fact Retrieval Quality ===\n");

    eprint!("  Digesting all fixture files... ");
    let digest_start = Instant::now();
    let files = load_fixture_files()?;
    let file_count = files.len();

    // Copy fixture files to temp dynamic dir so digest() can move them
    std::fs::create_dir_all(&config.dynamic_dir)?;
    for file in &files {
        let dest = config.dynamic_dir.join(&file.relative);
        std::fs::copy(&file.path, &dest)?;
    }

    // Re-collect from the temp dir so paths point to movable copies
    let movable_files = fact::collect_dynamic_files(&config)?;
    let fact_count = fact::digest(&config, &movable_files).await?;
    let digest_elapsed = digest_start.elapsed().as_secs_f64();
    eprintln!("done ({file_count} files, {fact_count} facts in {digest_elapsed:.1}s)\n");

    let mut retrieval_passed = 0;
    let retrieval_total = RETRIEVAL_EVALS.len();
    let mut retrieval_results = Vec::new();
    let retrieval_start = Instant::now();

    let db_path = config.memory_db();

    for eval in RETRIEVAL_EVALS {
        eprint!("  {} ... ", eval.name);
        let eval_start = Instant::now();

        let result = fact::retrieve(&db_path, eval.query, eval.top_k).await?;
        let elapsed = eval_start.elapsed().as_secs_f64();

        let (pass, reason) = run_check(&result.context, &eval.check);
        if pass {
            retrieval_passed += 1;
        }

        eprintln!("{} ({:.1}s)", if pass { "PASS" } else { "FAIL" }, elapsed);
        if !pass {
            eprintln!("    reason: {reason}");
            eprintln!(
                "    context preview: {}...",
                &result.context[..result.context.len().min(200)]
            );
        }

        retrieval_results.push(json!({
            "name": eval.name,
            "query": eval.query,
            "pass": pass,
            "reason": reason,
            "elapsed_s": (elapsed * 10.0).round() / 10.0,
        }));
    }

    let retrieval_elapsed = retrieval_start.elapsed().as_secs_f64();
    eprintln!(
        "\n  [retrieval] {retrieval_passed}/{retrieval_total} passed in {retrieval_elapsed:.1}s"
    );

    // Summary
    let total_passed = extraction_passed + retrieval_passed;
    let total_evals = extraction_total + retrieval_total;

    eprintln!("\n  === TOTAL: {total_passed}/{total_evals} passed ===");

    if total_passed < total_evals {
        let extraction_failures: Vec<_> = extraction_results
            .iter()
            .filter(|r| !r["pass"].as_bool().unwrap_or(true))
            .cloned()
            .collect();
        let retrieval_failures: Vec<_> = retrieval_results
            .iter()
            .filter(|r| !r["pass"].as_bool().unwrap_or(true))
            .cloned()
            .collect();
        let output = json!({
            "extraction_passed": extraction_passed,
            "extraction_total": extraction_total,
            "retrieval_passed": retrieval_passed,
            "retrieval_total": retrieval_total,
            "total_passed": total_passed,
            "total_evals": total_evals,
            "digest_elapsed_s": (digest_elapsed * 10.0).round() / 10.0,
            "extraction_failures": extraction_failures,
            "retrieval_failures": retrieval_failures,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        anyhow::bail!(
            "{}/{total_evals} fact evals failed",
            total_evals - total_passed
        );
    }

    Ok(())
}
