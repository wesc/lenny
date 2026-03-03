use anyhow::Result;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use crate::context;
use crate::once;

/// Approximate token count via whitespace word count.
fn approx_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Move a file from `src` to `dst`, creating parent dirs as needed.
/// Uses rename if possible, falls back to copy+delete for cross-device moves.
fn move_file(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::rename(src, dst).is_err() {
        // Rename can fail across devices; fall back to copy+delete
        fs::copy(src, dst)?;
        fs::remove_file(src)?;
    }
    Ok(())
}

/// File with its content and metadata for compaction decisions.
pub struct DynamicFile {
    pub path: PathBuf,
    pub relative: PathBuf,
    pub content: String,
    pub tokens: usize,
}

/// Collect files from dynamic/, read contents, compute token counts.
pub fn collect_dynamic_files(config: &Config) -> Result<Vec<DynamicFile>> {
    let paths = context::collect_files(&config.dynamic_dir)?;
    let mut files = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(&config.dynamic_dir)
            .unwrap_or(&path)
            .to_path_buf();
        let content = fs::read_to_string(&path)?;
        let tokens = approx_tokens(&content);
        files.push(DynamicFile {
            path,
            relative,
            content,
            tokens,
        });
    }
    Ok(files)
}

/// Select oldest files to pop so remaining tokens <= min_context_tokens.
/// Returns (files_to_pop, files_to_keep).
pub fn select_files_to_pop(
    mut files: Vec<DynamicFile>,
    max_tokens: usize,
    min_tokens: usize,
) -> (Vec<DynamicFile>, Vec<DynamicFile>) {
    let total: usize = files.iter().map(|f| f.tokens).sum();
    if total <= max_tokens {
        return (vec![], files);
    }

    // Sort by relative path (lexicographic = timestamp order)
    files.sort_by(|a, b| a.relative.cmp(&b.relative));

    let mut remaining = total;
    let mut split_idx = 0;
    for (i, file) in files.iter().enumerate() {
        if remaining <= min_tokens {
            break;
        }
        remaining -= file.tokens;
        split_idx = i + 1;
    }

    let keep = files.split_off(split_idx);
    (files, keep)
}

/// Move popped files from dynamic/ to references/, preserving subdirectory structure.
pub fn move_to_references(config: &Config, files: &[DynamicFile]) -> Result<()> {
    for file in files {
        let dest = config.references_dir().join(&file.relative);
        move_file(&file.path, &dest)?;
    }
    Ok(())
}

/// Build the compaction prompt from popped file contents.
fn build_compaction_prompt(files: &[DynamicFile]) -> String {
    let mut prompt = String::from(
        "The following files are being moved from working memory to references/. \
         Please produce a concise comprehension that distills the key information, \
         patterns, and insights from these files. Include citations to the source \
         files using their references/ paths (e.g. references/cli-bot/session.json).\n\n",
    );

    for file in files {
        let ref_path = format!("references/{}", file.relative.display());
        prompt.push_str(&format!("--- {ref_path} ---\n"));
        prompt.push_str(&file.content);
        if !file.content.ends_with('\n') {
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt
}

/// Atomic-write a comprehension file to the comprehensions directory.
fn write_comprehension(config: &Config, content: &str) -> Result<PathBuf> {
    let comp_dir = config.comprehensions_dir();
    fs::create_dir_all(&comp_dir)?;

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let filename = format!("{timestamp}-compact.md");
    let final_path = comp_dir.join(&filename);

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = comp_dir.join(&tmp_name);

    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, &final_path)?;

    Ok(final_path)
}

/// Run compaction on dynamic/. Returns Ok(true) if compaction occurred.
pub async fn run(config: &Config) -> Result<bool> {
    let files = collect_dynamic_files(config)?;
    let total_tokens: usize = files.iter().map(|f| f.tokens).sum();

    if total_tokens <= config.max_context_tokens {
        return Ok(false);
    }

    let (to_pop, _keep) =
        select_files_to_pop(files, config.max_context_tokens, config.min_context_tokens);

    if to_pop.is_empty() {
        return Ok(false);
    }

    eprintln!(
        "[compact] popping {} file(s), asking LLM for comprehension...",
        to_pop.len()
    );

    let prompt = build_compaction_prompt(&to_pop);
    let preamble = "You are a knowledge distillation assistant. Produce concise, well-organized \
                     comprehensions that preserve key facts, decisions, and context. Always cite \
                     source files by their references/ path.";

    let comprehension = once::run_completion(config, preamble, &prompt).await?;

    let comp_path = write_comprehension(config, &comprehension)?;
    eprintln!("[compact] wrote comprehension: {}", comp_path.display());

    move_to_references(config, &to_pop)?;
    eprintln!("[compact] moved {} file(s) to references/", to_pop.len());

    Ok(true)
}
