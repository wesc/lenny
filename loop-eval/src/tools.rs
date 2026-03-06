use anyhow::{Result, anyhow};
use openrouter_rs::types::{Tool, ToolCall};
use rand::RngExt;
use serde_json::json;

/// A registered tool: its OpenRouter definition + a handler function.
pub struct ToolDef {
    pub tool: Tool,
    pub handler: fn(&serde_json::Value) -> Result<String>,
}

fn handle_random_number(args: &serde_json::Value) -> Result<String> {
    let min = args["min"].as_i64().ok_or_else(|| anyhow!("missing min"))?;
    let max = args["max"].as_i64().ok_or_else(|| anyhow!("missing max"))?;
    if min > max {
        return Ok("error: min must be <= max".to_string());
    }
    let n = rand::rng().random_range(min..=max);
    Ok(n.to_string())
}

fn handle_random_letter(_args: &serde_json::Value) -> Result<String> {
    let c = rand::rng().random_range(b'a'..=b'z') as char;
    Ok(c.to_string())
}

fn handle_context_search(_args: &serde_json::Value) -> Result<String> {
    Ok("No relevant context found.".to_string())
}

/// All available tools.
pub fn all_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            tool: Tool::new(
                "random_number",
                "Generate a random integer between min and max (inclusive).",
                json!({
                    "type": "object",
                    "properties": {
                        "min": { "type": "integer", "description": "Minimum value (inclusive)" },
                        "max": { "type": "integer", "description": "Maximum value (inclusive)" }
                    },
                    "required": ["min", "max"]
                }),
            ),
            handler: handle_random_number,
        },
        ToolDef {
            tool: Tool::new(
                "random_letter",
                "Generate a random lowercase letter (a-z).",
                json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
            handler: handle_random_letter,
        },
    ]
}

/// Tools for the empty_search dataset: context_search that always returns nothing.
pub fn empty_search_tools() -> Vec<ToolDef> {
    vec![ToolDef {
        tool: Tool::new(
            "context_search",
            "Search past conversations and documents by semantic similarity. Returns relevant context passages.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what information to find"
                    }
                },
                "required": ["query"]
            }),
        ),
        handler: handle_context_search,
    }]
}

/// Look up a tool's handler by name and execute it.
pub fn dispatch(tool_defs: &[ToolDef], call: &ToolCall) -> Result<String> {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .unwrap_or(serde_json::Value::Object(Default::default()));

    for def in tool_defs {
        if def.tool.function.name == call.function.name {
            return (def.handler)(&args);
        }
    }
    Ok(format!("unknown tool: {}", call.function.name))
}
