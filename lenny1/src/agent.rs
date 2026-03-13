use anyhow::{Result, bail};
use async_trait::async_trait;
use rig::OneOrMany;
use rig::completion::{
    CompletionModel as CompletionModelTrait, CompletionRequest, ToolDefinition,
    message::{AssistantContent, Text},
};
use rig::message::ToolChoice;
use rig::providers::openrouter;
use serde::Serialize;

use rig::completion::request::Document;

use crate::config::Config;

/// Re-export rig's Message type for use across the codebase.
pub use rig::completion::Message;
/// Re-export rig's ToolCall type.
pub use rig::completion::message::ToolCall;

const MAX_RETRIES: usize = 3;
const RETRY_BASE_DELAY_MS: u64 = 1000;

/// Send a completion request with retries on transient errors.
/// The request is cloned for each retry since `completion()` consumes it.
macro_rules! send_with_retry {
    ($model:expr, $request:expr) => {{
        let mut _last_err = None;
        let mut _result = None;
        let req = $request;
        for _attempt in 0..MAX_RETRIES {
            match $model.completion(req.clone()).await {
                Ok(response) => {
                    _result = Some(response);
                    break;
                }
                Err(e) => {
                    let msg = e.to_string();
                    eprintln!(
                        "API error (attempt {}/{}): {msg}",
                        _attempt + 1,
                        MAX_RETRIES,
                    );
                    _last_err = Some(e);
                    tokio::time::sleep(std::time::Duration::from_millis(
                        RETRY_BASE_DELAY_MS * (1 << _attempt),
                    ))
                    .await;
                }
            }
        }
        match _result {
            Some(r) => Ok(r),
            None => Err(anyhow::anyhow!("{}", _last_err.unwrap())),
        }
    }};
}

// ---------------------------------------------------------------------------
// Async tool handler
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn call(&self, args: &serde_json::Value) -> Result<String>;
}

pub struct ToolDef {
    pub tool: ToolDefinition,
    pub handler: Box<dyn ToolHandler>,
}

/// Look up a tool's handler by name and execute it.
pub async fn dispatch(tool_defs: &[ToolDef], call: &ToolCall) -> Result<String> {
    // rig-core provides arguments as serde_json::Value directly
    let args = &call.function.arguments;

    for def in tool_defs {
        if def.tool.name == call.function.name {
            return def.handler.call(args).await;
        }
    }
    Ok(format!("unknown tool: {}", call.function.name))
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single tool call event recorded during the reasoning loop.
#[derive(Debug, Clone)]
pub struct ToolEvent {
    pub tool: String,
    pub args: String,
    pub result: String,
}

/// Result of running a single eval through reasoning + response.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RunResult {
    /// Final answer from the response model.
    pub answer: String,
    /// Tool calls and their results from the reasoning loop.
    pub tool_events: Vec<ToolEvent>,
    /// The formatted tool context that was passed to the response model.
    pub tool_context: String,
    /// Number of assistant turns in the reasoning loop.
    pub reasoning_turns: usize,
    /// All messages produced during this run (user input + assistant + tool results).
    /// Used for disk persistence by the session module.
    pub messages: Vec<Message>,
}

/// What a single `once()` iteration produced.
#[allow(dead_code)]
pub enum IterationOutcome {
    ToolCalls { tool_count: usize },
    Done { text: Option<String> },
    MaxIterations,
}

/// Mutable conversation state across turns.
pub struct AgentState {
    /// System prompt, passed as `preamble` on each request.
    pub preamble: String,
    /// Chat messages (user, assistant, tool results). No system message here.
    pub messages: Vec<Message>,
    pub tool_events: Vec<ToolEvent>,
    pub iterations: usize,
    pub done: bool,
}

impl AgentState {
    pub fn new(system: &str, prompt: &str) -> Self {
        Self {
            preamble: system.to_string(),
            messages: vec![Message::user(prompt)],
            tool_events: Vec::new(),
            iterations: 0,
            done: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Effort (local replacement for openrouter_rs::types::Effort)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
}

impl Effort {
    pub fn as_str(&self) -> &str {
        match self {
            Effort::None => "none",
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Hook trait
// ---------------------------------------------------------------------------

/// Observability hook for the agent loop. All methods default to no-ops.
#[allow(unused_variables)]
pub trait AgentHook: Send {
    fn on_request(&mut self, iteration: usize, request: &CompletionRequest) {}
    fn on_response(&mut self, iteration: usize, content: Option<&str>, tool_calls: usize) {}
    fn on_tool_call(&mut self, iteration: usize, name: &str, args: &str, result: &str) {}
    fn on_reasoning_done(&mut self, state: &AgentState, outcome: &IterationOutcome) {}
    fn on_response_phase_request(&mut self, request: &CompletionRequest) {}
    fn on_response_phase_done(&mut self, answer: &str) {}
}

/// No-op hook implementation.
pub struct NoopHook;
impl AgentHook for NoopHook {}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Immutable agent config. Short-lived, one per prompt.
pub struct Agent<'a> {
    client: openrouter::Client,
    reasoning_model: &'a str,
    response_model: &'a str,
    /// System prompt for the reasoning phase (includes tool-calling instructions).
    system: &'a str,
    /// System prompt for the response phase (no tool instructions). Falls back to `system`.
    response_system: Option<&'a str>,
    tool_defs: &'a [ToolDef],
    /// Documents to include in completion requests (general knowledge from dynamic/).
    documents: Vec<Document>,
    max_iterations: usize,
    reasoning_max_tokens: u64,
    response_max_tokens: u64,
    response_effort: Effort,
}

/// Builder for Agent.
pub struct AgentBuilder<'a> {
    client: openrouter::Client,
    reasoning_model: &'a str,
    response_model: &'a str,
    system: &'a str,
    response_system: Option<&'a str>,
    tool_defs: &'a [ToolDef],
    documents: Vec<Document>,
    max_iterations: usize,
    reasoning_max_tokens: u64,
    response_max_tokens: u64,
    response_effort: Effort,
}

impl<'a> AgentBuilder<'a> {
    pub fn system(mut self, system: &'a str) -> Self {
        self.system = system;
        self
    }

    /// Set a separate system prompt for the response phase (without tool instructions).
    pub fn response_system(mut self, system: &'a str) -> Self {
        self.response_system = Some(system);
        self
    }

    pub fn tools(mut self, tool_defs: &'a [ToolDef]) -> Self {
        self.tool_defs = tool_defs;
        self
    }

    #[allow(dead_code)]
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    #[allow(dead_code)]
    pub fn response_effort(mut self, effort: Effort) -> Self {
        self.response_effort = effort;
        self
    }

    pub fn documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = documents;
        self
    }

    pub fn build(self) -> Agent<'a> {
        Agent {
            client: self.client,
            reasoning_model: self.reasoning_model,
            response_model: self.response_model,
            system: self.system,
            response_system: self.response_system,
            tool_defs: self.tool_defs,
            documents: self.documents,
            max_iterations: self.max_iterations,
            reasoning_max_tokens: self.reasoning_max_tokens,
            response_max_tokens: self.response_max_tokens,
            response_effort: self.response_effort,
        }
    }
}

impl<'a> Agent<'a> {
    pub fn builder(client: &openrouter::Client, config: &'a Config) -> AgentBuilder<'a> {
        AgentBuilder {
            client: client.clone(),
            reasoning_model: config.provider.reasoning_model(),
            response_model: config.provider.response_model(),
            system: "",
            response_system: None,
            tool_defs: &[],
            documents: vec![],
            max_iterations: config.max_iterations,
            reasoning_max_tokens: 2048,
            response_max_tokens: 1024,
            response_effort: Effort::Medium,
        }
    }

    /// Run one reasoning iteration: build request, send, dispatch tools.
    pub async fn once(
        &self,
        state: &mut AgentState,
        hook: &mut dyn AgentHook,
    ) -> Result<IterationOutcome> {
        state.iterations += 1;
        if state.iterations > self.max_iterations {
            let outcome = IterationOutcome::MaxIterations;
            state.done = true;
            hook.on_reasoning_done(state, &outcome);
            return Ok(outcome);
        }

        let tools: Vec<ToolDefinition> = self.tool_defs.iter().map(|td| td.tool.clone()).collect();

        let chat_history = OneOrMany::many(state.messages.clone())
            .unwrap_or_else(|_| OneOrMany::one(Message::user("")));

        let request = CompletionRequest {
            model: Some(self.reasoning_model.to_string()),
            preamble: Some(state.preamble.clone()),
            chat_history,
            documents: self.documents.clone(),
            tools,
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Auto),
            additional_params: Some(serde_json::json!({
                "max_tokens": self.reasoning_max_tokens,
            })),
            output_schema: None,
        };

        hook.on_request(state.iterations, &request);

        let model = openrouter::CompletionModel::new(self.client.clone(), self.reasoning_model);
        let response = send_with_retry!(model, request)?;

        // Extract tool calls and text from response
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut text_parts: Vec<String> = Vec::new();

        for content in response.choice.iter() {
            match content {
                AssistantContent::ToolCall(tc) => tool_calls.push(tc.clone()),
                AssistantContent::Text(t) => text_parts.push(t.text.clone()),
                _ => {} // ignore reasoning blocks, images
            }
        }

        // Model wants to call tools — execute and loop.
        if !tool_calls.is_empty() {
            let content_text = text_parts.join("");
            let tool_count = tool_calls.len();
            let content_opt = if content_text.is_empty() {
                None
            } else {
                Some(content_text.as_str())
            };
            hook.on_response(state.iterations, content_opt, tool_count);

            // Push the full assistant message (preserves text + tool calls)
            let assistant_content: Vec<AssistantContent> = response.choice.into_iter().collect();
            state.messages.push(Message::Assistant {
                id: response.message_id,
                content: OneOrMany::many(assistant_content)
                    .expect("response had tool calls, so non-empty"),
            });

            for tc in &tool_calls {
                let args_str = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                let result = dispatch(self.tool_defs, tc).await?;
                hook.on_tool_call(state.iterations, &tc.function.name, &args_str, &result);
                state.tool_events.push(ToolEvent {
                    tool: tc.function.name.clone(),
                    args: args_str,
                    result: result.clone(),
                });
                state.messages.push(Message::tool_result(&tc.id, &result));
            }

            return Ok(IterationOutcome::ToolCalls { tool_count });
        }

        // No tool calls — model is done reasoning.
        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };
        hook.on_response(state.iterations, text.as_deref(), 0);

        if let Some(ref t) = text {
            state.messages.push(Message::assistant(t.as_str()));
        }

        state.done = true;
        let outcome = IterationOutcome::Done { text };
        hook.on_reasoning_done(state, &outcome);
        Ok(outcome)
    }

    /// Run full reasoning loop + response phase.
    ///
    /// `history` is prepended to messages (previous turns from disk).
    /// The user prompt is appended after history.
    pub async fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
        hook: &mut dyn AgentHook,
    ) -> Result<RunResult> {
        let mut state = AgentState::new(self.system, prompt);

        // Prepend history before the user prompt
        if !history.is_empty() {
            let user_msg = state.messages.pop(); // the user prompt we just added
            state.messages = history;
            if let Some(msg) = user_msg {
                state.messages.push(msg);
            }
        }

        loop {
            let outcome = self.once(&mut state, hook).await?;
            match outcome {
                IterationOutcome::ToolCalls { .. } => continue,
                IterationOutcome::Done { .. } => break,
                IterationOutcome::MaxIterations => {
                    bail!("reasoning loop exceeded {} turns", self.max_iterations);
                }
            }
        }

        let reasoning_turns = state
            .messages
            .iter()
            .filter(|m| matches!(m, Message::Assistant { .. }))
            .count();

        let tool_context = format_tool_context(&state.tool_events);
        let answer = self.response_phase(prompt, &tool_context, hook).await?;

        Ok(RunResult {
            answer,
            tool_events: state.tool_events,
            tool_context,
            reasoning_turns,
            messages: state.messages,
        })
    }

    /// Response phase only (for hallucination evals).
    #[allow(dead_code)]
    pub async fn run_response_only(
        &self,
        prompt: &str,
        tool_context: &str,
        hook: &mut dyn AgentHook,
    ) -> Result<String> {
        self.response_phase(prompt, tool_context, hook).await
    }

    /// Internal: call the response model with accumulated tool context.
    async fn response_phase(
        &self,
        user_prompt: &str,
        tool_context: &str,
        hook: &mut dyn AgentHook,
    ) -> Result<String> {
        let prompt = if tool_context.is_empty() {
            user_prompt.to_string()
        } else {
            format!(
                "{user_prompt}\n\n---\n\
                 The following information was gathered from tools:\n\n\
                 {tool_context}"
            )
        };

        let system = self.response_system.unwrap_or(self.system);

        let mut additional = serde_json::json!({
            "max_tokens": self.response_max_tokens,
            "response_format": {"type": "json_object"},
            "plugins": [{"id": "response-healing"}],
        });

        // Add reasoning config
        if self.response_effort == Effort::None {
            additional["reasoning"] = serde_json::json!({
                "effort": "none",
                "max_tokens": 0,
            });
        } else {
            additional["reasoning"] = serde_json::json!({
                "effort": self.response_effort.as_str(),
            });
        }

        let request = CompletionRequest {
            model: Some(self.response_model.to_string()),
            preamble: Some(system.to_string()),
            chat_history: OneOrMany::one(Message::user(prompt.as_str())),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(additional),
            output_schema: None,
        };

        hook.on_response_phase_request(&request);

        let model = openrouter::CompletionModel::new(self.client.clone(), self.response_model);
        let response = send_with_retry!(model, request)?;

        let answer = extract_text(&response.choice);
        hook.on_response_phase_done(&answer);
        Ok(answer)
    }
}

/// Extract text content from a completion response choice.
fn extract_text(choice: &OneOrMany<AssistantContent>) -> String {
    let mut parts = Vec::new();
    for content in choice.iter() {
        if let AssistantContent::Text(Text { text }) = content {
            parts.push(text.as_str());
        }
    }
    parts.join("")
}

/// Effort levels ordered from lowest to highest.
pub const EFFORT_LEVELS: [Effort; 5] = [
    Effort::None,
    Effort::Minimal,
    Effort::Low,
    Effort::Medium,
    Effort::High,
];

/// Result of probing a model's supported effort levels.
pub struct EffortRange {
    pub min: Effort,
    pub max: Effort,
    pub supported: Vec<Effort>,
}

/// Probe a model to discover which reasoning effort levels it supports.
/// Tries each level from None to High, returns the range and full breakdown.
pub async fn probe_effort_range(client: &openrouter::Client, model: &str) -> Result<EffortRange> {
    let cm = openrouter::CompletionModel::new(client.clone(), model);
    let mut supported = Vec::new();

    for effort in &EFFORT_LEVELS {
        let mut additional = serde_json::json!({
            "max_tokens": 4,
            "reasoning": {"effort": effort.as_str()},
        });
        if *effort == Effort::None {
            additional["reasoning"]["max_tokens"] = serde_json::json!(0);
        }

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("Say OK.")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(additional),
            output_schema: None,
        };

        if cm.completion(request).await.is_ok() {
            supported.push(effort.clone());
        }
    }

    if supported.is_empty() {
        bail!("model rejected all effort levels");
    }

    let min = supported.first().unwrap().clone();
    let max = supported.last().unwrap().clone();

    Ok(EffortRange {
        min,
        max,
        supported,
    })
}

/// Format tool events into a human-readable context block.
pub fn format_tool_context(events: &[ToolEvent]) -> String {
    events
        .iter()
        .map(|e| format!("[{}] {}\n-> {}", e.tool, e.args, e.result))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// Serializable wrapper for CompletionRequest (for prompt logging)
// ---------------------------------------------------------------------------

/// A serializable summary of a CompletionRequest for logging purposes.
/// rig-core's CompletionRequest does not implement Serialize.
#[derive(Serialize)]
pub struct RequestSummary {
    pub model: Option<String>,
    pub preamble_len: Option<usize>,
    pub message_count: usize,
    pub tool_count: usize,
    pub additional_params: Option<serde_json::Value>,
}

impl From<&CompletionRequest> for RequestSummary {
    fn from(req: &CompletionRequest) -> Self {
        Self {
            model: req.model.clone(),
            preamble_len: req.preamble.as_ref().map(|p| p.len()),
            message_count: req.chat_history.iter().count(),
            tool_count: req.tools.len(),
            additional_params: req.additional_params.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build openrouter::Client from config
// ---------------------------------------------------------------------------

/// Create a rig openrouter client from config.
pub fn build_client(config: &Config) -> Result<openrouter::Client> {
    let crate::config::ProviderConfig::OpenRouter { ref api_key, .. } = config.provider;
    Ok(openrouter::Client::new(api_key)?)
}
