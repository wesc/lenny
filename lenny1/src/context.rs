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

        // Skip hidden files and directories
        if name_str.starts_with('.') {
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
pub fn assemble_context(system_dir: &Path, dynamic_dir: &Path) -> Result<String> {
    let mut all_files = Vec::new();

    for file in collect_files(system_dir)? {
        all_files.push(("system", file));
    }
    for file in collect_files(dynamic_dir)? {
        all_files.push(("dynamic", file));
    }

    // Sort by full path within each category
    all_files.sort_by(|a, b| a.1.cmp(&b.1));

    let mut context = String::new();
    for (label, path) in &all_files {
        // Build a display path like "system/00-identity.md"
        let display = if *label == "system" {
            path.strip_prefix(system_dir)
                .map(|p| format!("{}/{}", label, p.display()))
                .unwrap_or_else(|_| path.display().to_string())
        } else {
            path.strip_prefix(dynamic_dir)
                .map(|p| format!("{}/{}", label, p.display()))
                .unwrap_or_else(|_| path.display().to_string())
        };

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
