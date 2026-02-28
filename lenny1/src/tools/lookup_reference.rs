use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct LookupReferenceArgs {
    pub path: String,
}

#[derive(Debug, Error)]
pub enum LookupReferenceError {
    #[error("Path traversal detected: {0}")]
    PathTraversal(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct LookupReferenceTool {
    pub references_dir: PathBuf,
}

impl Tool for LookupReferenceTool {
    const NAME: &'static str = "lookup_reference";

    type Error = LookupReferenceError;
    type Args = LookupReferenceArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "lookup_reference".to_string(),
            description: "Look up a reference file by its relative path within the references directory. Returns the file contents.".to_string(),
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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let requested = std::path::Path::new(&args.path);
        for component in requested.components() {
            if let std::path::Component::ParentDir = component {
                return Err(LookupReferenceError::PathTraversal(args.path));
            }
        }

        let full_path = self.references_dir.join(&args.path);

        let canonical_refs = self
            .references_dir
            .canonicalize()
            .unwrap_or_else(|_| self.references_dir.clone());
        let canonical_target = full_path.canonicalize()?;
        if !canonical_target.starts_with(&canonical_refs) {
            return Err(LookupReferenceError::PathTraversal(args.path));
        }

        let content = std::fs::read_to_string(&full_path)?;
        Ok(content)
    }
}
