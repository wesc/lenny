use anyhow::Result;
use std::fs;
use std::path::Path;

/// Recursively collect all non-hidden file paths under `dir`.
fn collect_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
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

/// Assemble context from system and dynamic directories.
/// Returns a single string with section markers for each file.
/// System files come first, then dynamic files, each sorted by relative path.
pub fn assemble_context(system_dir: &Path, dynamic_dir: &Path) -> Result<String> {
    let mut system_files = collect_files(system_dir)?;
    let mut dynamic_files = collect_files(dynamic_dir)?;
    system_files.sort();
    dynamic_files.sort();

    let all_files: Vec<(&str, &Path, &std::path::PathBuf)> = system_files
        .iter()
        .map(|f| ("system", system_dir, f))
        .chain(dynamic_files.iter().map(|f| ("dynamic", dynamic_dir, f)))
        .collect();

    let mut context = String::new();
    for (label, base_dir, path) in &all_files {
        let display = path
            .strip_prefix(base_dir)
            .map(|p| format!("{}/{}", label, p.display()))
            .unwrap_or_else(|_| path.display().to_string());

        let content = fs::read_to_string(path)?;
        context.push_str(&format!("--- {display} ---\n"));
        context.push_str(&content);
        if !content.ends_with('\n') {
            context.push('\n');
        }
        context.push('\n');
    }

    Ok(context)
}
