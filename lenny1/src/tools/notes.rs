use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Sanitize a user-provided filename: strip separators, reject empty/hidden.
pub fn sanitize_filename(raw: &str) -> Result<String> {
    let name = raw
        .trim()
        .replace(['/', '\\'], "-")
        .trim_start_matches('.')
        .to_string();
    if name.is_empty() {
        anyhow::bail!("filename must not be empty");
    }
    Ok(name)
}

/// Return the notes directory, creating it if needed.
pub fn notes_dir(dynamic_dir: &Path) -> Result<PathBuf> {
    let dir = dynamic_dir.join("notes");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Append text to a file in the notes directory using atomic write.
pub fn append_to_note(dynamic_dir: &Path, filename: &str, content: &str) -> Result<PathBuf> {
    let dir = notes_dir(dynamic_dir)?;
    let path = dir.join(filename);

    let mut existing = fs::read_to_string(&path).unwrap_or_default();
    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(content);

    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let tmp_path = dir.join(&tmp_name);
    fs::write(&tmp_path, &existing)?;
    fs::rename(&tmp_path, &path)?;

    Ok(path)
}
