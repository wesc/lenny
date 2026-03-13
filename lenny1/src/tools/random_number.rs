use anyhow::Result;
use async_trait::async_trait;
use rig::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

use crate::agent::{ToolDef, ToolHandler};

/// Deserialize a number that might arrive as a JSON string (e.g. "100" instead of 100).
fn deserialize_lenient_i64<'de, D>(deserializer: D) -> std::result::Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct LenientI64;
    impl<'de> de::Visitor<'de> for LenientI64 {
        type Value = i64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an integer or string-encoded integer")
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<i64, E> {
            Ok(v)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<i64, E> {
            Ok(v as i64)
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<i64, E> {
            Ok(v as i64)
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<i64, E> {
            v.parse::<i64>().map_err(de::Error::custom)
        }
    }
    deserializer.deserialize_any(LenientI64)
}

#[derive(Debug, Deserialize)]
struct RandomNumberArgs {
    #[serde(default = "default_min", deserialize_with = "deserialize_lenient_i64")]
    min: i64,
    #[serde(default = "default_max", deserialize_with = "deserialize_lenient_i64")]
    max: i64,
}

fn default_min() -> i64 {
    1
}
fn default_max() -> i64 {
    100
}

pub struct RandomNumberTool;

#[async_trait]
impl ToolHandler for RandomNumberTool {
    async fn call(&self, args: &serde_json::Value) -> Result<String> {
        use rand::Rng;
        let args: RandomNumberArgs = serde_json::from_value(args.clone())?;
        if args.min > args.max {
            anyhow::bail!("min must be <= max");
        }
        let n = rand::rng().random_range(args.min..=args.max);
        tracing::debug!(
            min = args.min,
            max = args.max,
            result = n,
            "generated random number"
        );
        Ok(n.to_string())
    }
}

pub fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "random_number".to_string(),
        description:
            "Generate a random integer. You MUST provide both 'min' and 'max' as integer arguments."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "min": { "type": "integer", "description": "Minimum value (inclusive). Example: 1" },
                "max": { "type": "integer", "description": "Maximum value (inclusive). Example: 100" }
            },
            "required": ["min", "max"]
        }),
    }
}

pub fn mock_tool_def() -> ToolDef {
    ToolDef {
        tool: tool_definition(),
        handler: Box::new(crate::agent::MockHandler {
            response: "42".to_string(),
        }),
    }
}

impl RandomNumberTool {
    pub fn tool_def(self) -> ToolDef {
        ToolDef {
            tool: tool_definition(),
            handler: Box::new(self),
        }
    }
}
