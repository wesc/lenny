# loop-eval

Eval harness for a two-phase reasoning loop + response model architecture.

## Architecture

### Phase 1: Reasoning loop

The reasoning model runs in a tool-calling loop (`Agent::once()` repeated until
done or max iterations). It receives the system prompt, user prompt, and tool
definitions, then calls tools as needed.

**Reasoning effort**: `Effort::High` — the reasoning model benefits from deep
chain-of-thought.

### Phase 2: Response generation

A separate response model receives the original prompt plus formatted tool
context and produces the final answer.

**Reasoning effort**: Probed at startup via `probe_min_effort()` — tries effort
levels from `None` up to `High` and caches the lowest level the model accepts.
Some models (e.g. `gpt-oss-120b`) require reasoning and reject `Effort::None`.

## Known gaps

### Thinking tokens are discarded

The reasoning model's thinking/chain-of-thought tokens (`choice.reasoning()`)
are never extracted or passed to the response model. Only tool call results
reach the response phase.

### Reasoning model's final text is discarded

When the reasoning model finishes (no more tool calls), its final assistant
text is stored in `AgentState.messages` but is **not** included in the context
passed to the response model. Only `format_tool_context()` output (tool name,
args, result pairs) is forwarded.

This means any synthesis or summary the reasoning model produces after its last
tool call is thrown away.

## Usage

```sh
# Run a specific dataset
cargo run -- tool_use
cargo run -- hallucination

# Run all datasets
cargo run -- all

# Override models
cargo run -- tool_use --reasoning-model deepseek/deepseek-r1 --response-model openai/gpt-4o
```
