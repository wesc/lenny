use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;

use crate::tokens::count_tokens;

/// Warn when assembled context exceeds this many tokens.
/// At 30K tokens the system prompt alone consumes ~23% of a 128K context window,
/// leaving less room for the multi-turn conversation, tool calls, and tool results.
const CONTEXT_TOKEN_WARN_THRESHOLD: usize = 30_000;

/// Template variables available in system prompt files.
#[derive(Debug, Clone, Serialize)]
pub struct TemplateContext {
    pub channel_name: String,
    pub current_datetime: String,
}

/// Recursively collect all non-hidden file paths under `dir`.
pub(crate) fn collect_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    collect_files_recursive(dir, &mut files)?;
    Ok(files)
}

fn collect_files_recursive(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden files/directories and Emacs backup files
        if name_str.starts_with('.') || name_str.ends_with('~') {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

/// Assemble system prompt from a channel directory.
/// Each file is read, rendered through minijinja with the template context,
/// then concatenated with `--- {filename} ---` headers.
/// Files are sorted lexically by relative path.
pub fn assemble_system_prompt(system_dir: &Path, ctx: &TemplateContext) -> Result<String> {
    let mut system_files = collect_files(system_dir)?;
    system_files.sort();

    let env = minijinja::Environment::new();
    let ctx_value = minijinja::value::Value::from_serialize(ctx);

    let mut context = String::new();

    for path in &system_files {
        let label = path
            .strip_prefix(system_dir)
            .map(|p| format!("system/{}", p.display()))
            .unwrap_or_else(|_| path.display().to_string());

        let raw_content = fs::read_to_string(path)?;

        // Render through minijinja template engine
        let content = env
            .render_str(&raw_content, &ctx_value)
            .unwrap_or_else(|e| {
                tracing::warn!(file = %label, error = %e, "template render failed, using raw content");
                raw_content.clone()
            });

        context.push_str(&format!("--- {label} ---\n"));
        context.push_str(&content);
        if !content.ends_with('\n') {
            context.push('\n');
        }
        context.push('\n');
    }

    let token_count = count_tokens(&context);
    if token_count > CONTEXT_TOKEN_WARN_THRESHOLD {
        tracing::warn!(
            tokens = token_count,
            threshold = CONTEXT_TOKEN_WARN_THRESHOLD,
            "assembled system prompt is large — may degrade tool-calling reliability"
        );
    } else {
        tracing::info!(tokens = token_count, "assembled system prompt");
    }

    Ok(context)
}
