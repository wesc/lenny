use anyhow::{Result, bail};
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::{Component, Path, PathBuf};

use crate::agent::{ToolDef, ToolHandler};

#[derive(Debug, Deserialize)]
struct LookupReferenceArgs {
    path: String,
}

pub struct LookupReferenceTool {
    pub references_dir: PathBuf,
}

/// Resolve `user_path` against `base_dir`, rejecting any traversal outside `base_dir`.
///
/// Returns the joined absolute path on success. This works purely on lexical
/// components (no filesystem access) so it cannot be tricked by symlinks that
/// don't exist yet, TOCTOU races, or non-existent base directories.
fn safe_resolve(base_dir: &Path, user_path: &str) -> Result<PathBuf> {
    // Reject absolute paths outright — the input must be relative.
    if Path::new(user_path).is_absolute() {
        bail!("Path traversal detected: {user_path}");
    }

    // Walk the user-supplied components, tracking logical depth inside base_dir.
    // Any attempt to go above depth 0 (i.e. escape the base) is rejected.
    let mut resolved = base_dir.to_path_buf();
    let mut depth: usize = 0;

    for component in Path::new(user_path).components() {
        match component {
            Component::Normal(seg) => {
                resolved.push(seg);
                depth += 1;
            }
            Component::CurDir => {
                // "." — no-op
            }
            Component::ParentDir => {
                if depth == 0 {
                    bail!("Path traversal detected: {user_path}");
                }
                resolved.pop();
                depth -= 1;
            }
            // Prefix (Windows drive) or RootDir — should not appear in relative paths
            _ => {
                bail!("Path traversal detected: {user_path}");
            }
        }
    }

    // Must reference an actual file, not the base dir itself.
    if depth == 0 {
        bail!("Path traversal detected: {user_path}");
    }

    Ok(resolved)
}

#[async_trait]
impl ToolHandler for LookupReferenceTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        let args: LookupReferenceArgs = serde_json::from_value(args.clone())?;
        let full_path = safe_resolve(&self.references_dir, &args.path)?;

        tracing::debug!(path = %full_path.display(), "looking up reference");
        let content = std::fs::read_to_string(&full_path)?;
        Ok(content)
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "lookup_reference".to_string(),
        description: "Read a file from the references directory by its relative path. Use this to retrieve past conversation turns, uploaded documents, or any reference material. Always use this tool when asked to look up or read a reference file.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the reference file (e.g. 'turns/20260228-120000-some-topic.json')"
                }
            },
            "required": ["path"]
        }),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "File not found.".to_string(),
        }),
    }
}

impl LookupReferenceTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn base() -> PathBuf {
        PathBuf::from("/data/knowledge/references")
    }

    // --- should succeed ---

    #[test]
    fn simple_file() {
        let result = safe_resolve(&base(), "file.txt").unwrap();
        assert_eq!(result, base().join("file.txt"));
    }

    #[test]
    fn nested_path() {
        let result = safe_resolve(&base(), "turns/2025-01-01.json").unwrap();
        assert_eq!(result, base().join("turns/2025-01-01.json"));
    }

    #[test]
    fn dot_in_middle() {
        let result = safe_resolve(&base(), "subdir/./file.txt").unwrap();
        assert_eq!(result, base().join("subdir/file.txt"));
    }

    #[test]
    fn parent_within_subtree() {
        let result = safe_resolve(&base(), "a/b/../c.txt").unwrap();
        assert_eq!(result, base().join("a/c.txt"));
    }

    // --- should reject ---

    #[test]
    fn bare_dotdot() {
        assert!(safe_resolve(&base(), "..").is_err());
    }

    #[test]
    fn leading_dotdot() {
        assert!(safe_resolve(&base(), "../etc/passwd").is_err());
    }

    #[test]
    fn multiple_dotdot() {
        assert!(safe_resolve(&base(), "../../../etc/shadow").is_err());
    }

    #[test]
    fn dotdot_after_descend() {
        assert!(safe_resolve(&base(), "a/../../etc/passwd").is_err());
    }

    #[test]
    fn absolute_path() {
        assert!(safe_resolve(&base(), "/etc/passwd").is_err());
    }

    #[test]
    fn empty_path() {
        assert!(safe_resolve(&base(), "").is_err());
    }

    #[test]
    fn just_dot() {
        assert!(safe_resolve(&base(), ".").is_err());
    }

    #[test]
    fn dotdot_disguised_in_longer_chain() {
        assert!(safe_resolve(&base(), "a/b/c/../../../..").is_err());
    }

    #[test]
    fn reads_real_file() {
        let tmpdir = tempfile::tempdir().unwrap();
        let base = tmpdir.path().to_path_buf();
        std::fs::write(base.join("hello.txt"), "world").unwrap();

        let resolved = safe_resolve(&base, "hello.txt").unwrap();
        assert_eq!(std::fs::read_to_string(resolved).unwrap(), "world");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_resolves_lexically() {
        let tmpdir = tempfile::tempdir().unwrap();
        let base = tmpdir.path().join("refs");
        std::fs::create_dir_all(&base).unwrap();

        let outside = tmpdir.path().join("secret.txt");
        std::fs::write(&outside, "sensitive").unwrap();

        std::os::unix::fs::symlink(&outside, base.join("escape")).unwrap();
        // Lexical resolution sees "escape" as depth=1 inside base — allowed.
        // The defense is that the LLM can only request paths, not create symlinks.
        let resolved = safe_resolve(&base, "escape").unwrap();
        assert_eq!(resolved, base.join("escape"));
    }
}
