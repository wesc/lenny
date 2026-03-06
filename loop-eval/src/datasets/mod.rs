pub mod empty_search;
pub mod hallucination;
pub mod tool_use;

use crate::runner::{ResponseOnlyResult, RunResult};
use crate::tools::ToolDef;

/// A single eval case for tool-use datasets (reasoning loop + response).
pub struct Eval {
    pub name: &'static str,
    pub prompt: &'static str,
    pub check: fn(&RunResult) -> (bool, String),
}

/// How a response-only eval is checked.
#[allow(dead_code)]
pub enum CheckMethod {
    /// LLM judge evaluates the answer against these criteria.
    Judge(&'static str),
    /// Deterministic check function (for exact arithmetic, format validation, etc.)
    Exact(fn(&ResponseOnlyResult) -> (bool, String)),
}

/// A single eval case for response-only datasets (no reasoning loop).
pub struct ResponseOnlyEval {
    pub name: &'static str,
    pub prompt: &'static str,
    pub tool_context: &'static str,
    pub check: CheckMethod,
}

/// Distinguishes datasets that run the full reasoning loop from those that
/// skip it and inject pre-fabricated tool context.
pub enum DatasetKind {
    /// Reasoning loop + response. Existing behavior.
    ToolUse {
        tools: Vec<ToolDef>,
        evals: Vec<Eval>,
    },
    /// Response-only with pre-fabricated tool context.
    ResponseOnly { evals: Vec<ResponseOnlyEval> },
}

/// A dataset is a named collection of evals with a shared system prompt.
pub struct Dataset {
    pub name: &'static str,
    pub system: &'static str,
    pub kind: DatasetKind,
}

/// Return all registered datasets.
pub fn all_datasets() -> Vec<Dataset> {
    vec![
        tool_use::dataset(),
        hallucination::dataset(),
        empty_search::dataset(),
    ]
}

/// Valid dataset names (for CLI validation).
pub fn dataset_names() -> Vec<&'static str> {
    vec!["tool_use", "hallucination", "empty_search"]
}
