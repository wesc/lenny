use anyhow::{Result, bail};
use openrouter_rs::{
    OpenRouterClient,
    api::chat::{ChatCompletionRequest, Message},
    types::{Effort, Role},
};

use crate::config::Config;
use crate::tools::{ToolDef, dispatch};

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
    pub messages: Vec<Message>,
    pub tool_events: Vec<ToolEvent>,
    pub iterations: usize,
    pub done: bool,
}

impl AgentState {
    pub fn new(system: &str, prompt: &str) -> Self {
        Self {
            messages: vec![
                Message::new(Role::System, system),
                Message::new(Role::User, prompt),
            ],
            tool_events: Vec::new(),
            iterations: 0,
            done: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Hook trait
// ---------------------------------------------------------------------------

/// Observability hook for the agent loop. All methods default to no-ops.
#[allow(unused_variables)]
pub trait AgentHook {
    fn on_request(&mut self, iteration: usize, request: &ChatCompletionRequest) {}
    fn on_response(&mut self, iteration: usize, content: Option<&str>, tool_calls: usize) {}
    fn on_tool_call(&mut self, iteration: usize, name: &str, args: &str, result: &str) {}
    fn on_reasoning_done(&mut self, state: &AgentState, outcome: &IterationOutcome) {}
    fn on_response_phase_request(&mut self, request: &ChatCompletionRequest) {}
    fn on_response_phase_done(&mut self, answer: &str) {}
}

/// No-op hook implementation.
pub struct NoopHook;
impl AgentHook for NoopHook {}

/// Eval hook that logs to stderr. `EvalHook::verbose()` dumps full request JSON.
#[allow(dead_code)]
pub struct EvalHook {
    verbose: bool,
}

#[allow(dead_code)]
impl EvalHook {
    pub fn new() -> Self {
        Self { verbose: false }
    }

    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

impl AgentHook for EvalHook {
    fn on_request(&mut self, iteration: usize, request: &ChatCompletionRequest) {
        if self.verbose {
            if let Ok(json) = serde_json::to_string_pretty(request) {
                eprintln!("[iter {iteration}] request:\n{json}");
            }
        }
    }

    fn on_response(&mut self, iteration: usize, content: Option<&str>, tool_calls: usize) {
        let content_len = content.map(|c| c.len()).unwrap_or(0);
        eprintln!("[iter {iteration}] response: {content_len} chars, {tool_calls} tool calls");
    }

    fn on_tool_call(&mut self, iteration: usize, name: &str, args: &str, result: &str) {
        eprintln!("[iter {iteration}] tool: {name}({args}) -> {result}");
    }

    fn on_reasoning_done(&mut self, _state: &AgentState, outcome: &IterationOutcome) {
        match outcome {
            IterationOutcome::ToolCalls { .. } => {}
            IterationOutcome::Done { text } => {
                let len = text.as_ref().map(|t| t.len()).unwrap_or(0);
                eprintln!("[reasoning done] text={len} chars");
            }
            IterationOutcome::MaxIterations => {
                eprintln!("[reasoning done] max iterations reached");
            }
        }
    }

    fn on_response_phase_request(&mut self, request: &ChatCompletionRequest) {
        if self.verbose {
            if let Ok(json) = serde_json::to_string_pretty(request) {
                eprintln!("[response phase] request:\n{json}");
            }
        }
    }

    fn on_response_phase_done(&mut self, answer: &str) {
        eprintln!("[response] {} chars", answer.len());
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Immutable agent config. Short-lived, one per eval.
pub struct Agent<'a> {
    client: &'a OpenRouterClient,
    reasoning_model: &'a str,
    response_model: &'a str,
    system: &'a str,
    tool_defs: &'a [ToolDef],
    max_iterations: usize,
    reasoning_max_tokens: u32,
    response_max_tokens: u32,
    response_effort: Effort,
}

/// Builder for Agent.
pub struct AgentBuilder<'a> {
    client: &'a OpenRouterClient,
    reasoning_model: &'a str,
    response_model: &'a str,
    system: &'a str,
    tool_defs: &'a [ToolDef],
    max_iterations: usize,
    reasoning_max_tokens: u32,
    response_max_tokens: u32,
    response_effort: Effort,
}

impl<'a> AgentBuilder<'a> {
    pub fn system(mut self, system: &'a str) -> Self {
        self.system = system;
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

    pub fn build(self) -> Agent<'a> {
        Agent {
            client: self.client,
            reasoning_model: self.reasoning_model,
            response_model: self.response_model,
            system: self.system,
            tool_defs: self.tool_defs,
            max_iterations: self.max_iterations,
            reasoning_max_tokens: self.reasoning_max_tokens,
            response_max_tokens: self.response_max_tokens,
            response_effort: self.response_effort,
        }
    }
}

impl<'a> Agent<'a> {
    pub fn builder(client: &'a OpenRouterClient, config: &'a Config) -> AgentBuilder<'a> {
        AgentBuilder {
            client,
            reasoning_model: &config.reasoning_model,
            response_model: &config.response_model,
            system: "",
            tool_defs: &[],
            max_iterations: config.max_turns,
            reasoning_max_tokens: 2048,
            response_max_tokens: 1024,
            response_effort: config.response_effort.clone(),
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

        let tools: Vec<openrouter_rs::types::Tool> =
            self.tool_defs.iter().map(|td| td.tool.clone()).collect();

        let request = ChatCompletionRequest::builder()
            .model(self.reasoning_model)
            .messages(state.messages.clone())
            .tools(tools)
            .tool_choice_auto()
            .max_tokens(self.reasoning_max_tokens)
            .reasoning_effort(Effort::High)
            .build()?;

        hook.on_request(state.iterations, &request);

        let response = self.client.send_chat_completion(&request).await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("no choices in response"))?;

        // Model wants to call tools — execute and loop.
        if let Some(tool_calls) = choice.tool_calls() {
            let content = choice.content().unwrap_or("");
            let tool_count = tool_calls.len();
            hook.on_response(state.iterations, Some(content), tool_count);

            state.messages.push(Message::assistant_with_tool_calls(
                content,
                tool_calls.to_vec(),
            ));

            for call in tool_calls {
                let result = dispatch(self.tool_defs, call)?;
                hook.on_tool_call(
                    state.iterations,
                    &call.function.name,
                    &call.function.arguments,
                    &result,
                );
                state.tool_events.push(ToolEvent {
                    tool: call.function.name.clone(),
                    args: call.function.arguments.clone(),
                    result: result.clone(),
                });
                state
                    .messages
                    .push(Message::tool_response(&call.id, result));
            }

            return Ok(IterationOutcome::ToolCalls { tool_count });
        }

        // No tool calls — model is done reasoning.
        let text = choice.content().map(|s| s.to_string());
        hook.on_response(state.iterations, text.as_deref(), 0);

        if let Some(ref t) = text {
            state
                .messages
                .push(Message::new(Role::Assistant, t.as_str()));
        }

        state.done = true;
        let outcome = IterationOutcome::Done { text };
        hook.on_reasoning_done(state, &outcome);
        Ok(outcome)
    }

    /// Run full reasoning loop + response phase.
    pub async fn run(&self, prompt: &str, hook: &mut dyn AgentHook) -> Result<RunResult> {
        let mut state = AgentState::new(self.system, prompt);

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
            .filter(|m| m.role == Role::Assistant)
            .count();

        let tool_context = format_tool_context(&state.tool_events);
        let answer = self.response_phase(prompt, &tool_context, hook).await?;

        Ok(RunResult {
            answer,
            tool_events: state.tool_events,
            tool_context,
            reasoning_turns,
        })
    }

    /// Response phase only (for hallucination evals).
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

        let mut builder = ChatCompletionRequest::builder();
        builder
            .model(self.response_model)
            .messages(vec![
                Message::new(Role::System, self.system),
                Message::new(Role::User, prompt.as_str()),
            ])
            .max_tokens(self.response_max_tokens)
            .reasoning_effort(self.response_effort.clone());
        if matches!(self.response_effort, Effort::None) {
            builder.reasoning_max_tokens(0u32);
        }
        let request = builder.build()?;

        hook.on_response_phase_request(&request);

        let response = self.client.send_chat_completion(&request).await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("no choices in response"))?;

        let answer = choice.content().unwrap_or("").to_string();
        hook.on_response_phase_done(&answer);
        Ok(answer)
    }
}

/// Effort levels ordered from lowest to highest.
const EFFORT_LEVELS: [Effort; 5] = [
    Effort::None,
    Effort::Minimal,
    Effort::Low,
    Effort::Medium,
    Effort::High,
];

/// Probe the response model to find the lowest accepted reasoning effort.
/// Sends a trivial prompt at each level starting from None; returns the first
/// that succeeds.
pub async fn probe_min_effort(client: &OpenRouterClient, model: &str) -> Result<Effort> {
    for effort in &EFFORT_LEVELS {
        let mut builder = ChatCompletionRequest::builder();
        builder
            .model(model)
            .messages(vec![Message::new(Role::User, "Say OK.")])
            .max_tokens(4u32)
            .reasoning_effort(effort.clone());
        if matches!(effort, Effort::None) {
            builder.reasoning_max_tokens(0u32);
        }
        let request = builder.build()?;

        match client.send_chat_completion(&request).await {
            Ok(_) => {
                eprintln!("response model min effort: {effort}");
                return Ok(effort.clone());
            }
            Err(e) => {
                eprintln!("  effort={effort} rejected: {e}");
            }
        }
    }
    bail!("response model rejected all effort levels")
}

/// Format tool events into a human-readable context block.
pub fn format_tool_context(events: &[ToolEvent]) -> String {
    events
        .iter()
        .map(|e| format!("[{}] {}\n-> {}", e.tool, e.args, e.result))
        .collect::<Vec<_>>()
        .join("\n\n")
}
