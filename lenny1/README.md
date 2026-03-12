# Lenny1

The Lenny1 agent is an initial exploration of CAA, but focused on a [file over app](https://stephango.com/file-over-app) structure.

## Directory Hierarchy

Three top-level directories, each of which can have an arbitrary number of subdirectories:

- `system/`: immutable preambles and identity context, inserted on every call to the LLM
- `dynamic/`: mutable working memory inserted on every call to the LLM — bots write directly here with namespaced paths (e.g. `dynamic/cli-bot/`, `dynamic/matrix/`)
- `knowledge/`: persistent knowledge store:
  - `knowledge/memory.db`: sqlite-vec database containing extracted facts and their embeddings
  - `knowledge/references/`: archive of digested files and immutable context accessed via tool calls (e.g. `knowledge/references/matrix/`, `knowledge/references/chats/`, `knowledge/references/turns/`)

"Immutable" in this context means that the agent is not allowed to manipulate it, and "mutable" means the agent or one of its background processes can rewrite it at will.

Context assembly order: `system/` → `dynamic/`, each sorted by relative path. This means preambles like `system/cli-bot/00-preamble.txt` always appear before working memory. The LLM connects them via their shared path namespace.

## Fact Memory System

The fact system gives Lenny1 long-term memory. As working memory (`dynamic/`) grows, the oldest files are digested: an LLM extracts discrete, self-contained facts from each file, those facts are embedded into 384-dimensional vectors and stored in a sqlite-vec database (`knowledge/memory.db`), and the original files are archived to `knowledge/references/`. At query time the agent searches the database by vector similarity to recall relevant facts without needing the original files in its context window. The net effect is a bounded working memory with unbounded recall.

### Example

Suppose `dynamic/` contains two files totaling 5000 tokens (above the 4000-token threshold):

```
dynamic/
  2025-11-01_10-00-00_preferences.txt   (1800 tokens)
    "Your favorite color is chartreuse. You drive a Toyota Pineapple."
  2025-11-15_09-00-00_meeting.txt       (3200 tokens)
    "Sprint planning is every Monday. The deadline is March 15..."
```

Digest pops the oldest file first. The LLM extracts facts from `preferences.txt`:

```json
{"facts": [
  "The assistant's favorite color is chartreuse.",
  "The assistant drives a Toyota Pineapple."
]}
```

Each fact is embedded and stored in `memory.db` with `created_at: "2025-11-01T10:00:00Z"` and `file_reference: "2025-11-01_10-00-00_preferences.txt"`. The original file moves to `knowledge/references/`. Now `dynamic/` has 3200 tokens (within the 4000 limit) and the two facts are retrievable by semantic search.

Later, when the agent receives "what car do you like?", it calls `context_search` which returns:

```
[2025-11-01_10-00-00_preferences.txt @ 2025-11-01T10:00:00Z] The assistant drives a Toyota Pineapple.
```

### Digest Algorithm

**Filename convention.** All files in `dynamic/` eligible for digest must match `YYYY-MM-DD_HH-MM-SS_<id>.ext`. The timestamp is parsed from the filename and becomes `created_at`. Files not matching this format are silently skipped.

**Inputs.** The set of files in `dynamic/`, plus two thresholds: `max_context_tokens` (default 4000) and `min_context_tokens` (default 2000).

**Trigger.** Let `T` = sum of token counts across all eligible files. If `T <= max_context_tokens`, do nothing.

**Selection.** Sort files oldest-first by `created_at`. Walk from oldest, accumulating files to pop, until the remaining token total `<= min_context_tokens`.

**Extraction.** For each popped file `f`:
  1. Send `f.content` to the comprehension model with the fact extraction prompt.
  2. The model returns `facts[]`, an array of third-person factual statements.

**Storage.** For each extracted fact:
  1. Compute a 384-dim embedding via fastembed (`AllMiniLML6V2`).
  2. Insert into `facts` table: `(created_at, fact, file_reference, token_count)`.
  3. Insert the embedding into `facts_vec` virtual table (sqlite-vec).

**Archival.** Move each popped file from `dynamic/` to `knowledge/references/`, preserving subdirectory structure.

### Retrieval Algorithm

**Input.** A natural language query string and `top_k` (default 5).

**Process.**
  1. Embed the query into a 384-dim vector via fastembed.
  2. Query `facts_vec` for the `top_k` nearest rows by cosine distance.
  3. Join with `facts` table to get `fact`, `created_at`, `file_reference`.

**Output.** A newline-separated string of `[file_reference @ created_at] fact_text` entries, ordered by similarity.

## Commands

- `once <prompt>` — Run a single prompt through the agent and print the answer.
- `dream` — Watch `system/`, `dynamic/`, and `knowledge/` for changes, run background actions, and run fact digest on `dynamic/` when it exceeds the token threshold. Use `--force-digest` to run immediately.
- `matrix bot` — Connect to Matrix via sliding sync and respond to mentions. Logs all room messages/reactions as NDJSON.
- `cli bot` — Interactive CLI chat loop. Persists chat history as NDJSON directly to `dynamic/cli-bot/`.
- `fact digest` — Run one pass of fact extraction over `dynamic/` files. Use `--force` to skip the token threshold check.
- `fact search <query>` — Search facts by semantic similarity.
- `fact dump` — Dump all facts as JSON to stdout.
- `eval basic` — Run basic eval battery (arithmetic, tool use, knowledge retrieval).
- `eval contextual-chats` — Run contextual eval suite against chat fixture data.
- `eval contextual-texts` — Run contextual eval suite against text fixture data.
- `eval contextual-all` — Run both contextual eval suites.
- `eval fact` — Run fact extraction + retrieval eval suite.
- `eval all` — Run all eval suites (basic, contextual, fact).
- `estimate-tokens <path>` — Estimate token counts for files in a directory or a single file.
- `discover-reasoning-levels` — Probe model candidates for supported reasoning effort levels.

## Loop Agent

The processing loop goes like this:

1. Assemble context by appending files in `system/` and `dynamic/`, sorted by lexicographical filename ordering.
2. Call out to LLM
3. Execute tool calls
4. Loop to step 2 until the LLM reaches max iteration or ends processing via a `final_answer(answer, 1-5 word summary slug)` tool call. Question: what should we do if hit max iteration?
5. The full final context of the loop is recorded into `references/turns/{timestamp}-{slug}` with a timestamp in the filename to ensure sort order and uniqueness.

## Dreaming

The dreamer watches for file changes in `system/`, `dynamic/`, and `knowledge/` (debounced). When `dynamic/` exceeds the token threshold, it runs fact digest to extract and store facts, then moves the processed files to `knowledge/references/`.

## Implementation Notes

### Model Reasoning Effort Levels

Discovered via `cargo run -- discover-reasoning-levels` (probes each model with a trivial prompt at every effort level):

| Model | Min Effort | Supported Levels |
|---|---|---|
| `openai/gpt-oss-120b:nitro` | `minimal` | minimal, low, medium, high |
| `openai/gpt-oss-20b:nitro` | `minimal` | minimal, low, medium, high |
| `meta-llama/llama-3.1-8b-instruct:nitro` | `none` | none, minimal, low, medium, high |
| `google/gemini-2.5-flash-lite:nitro` | `none` | none, minimal, low, medium, high |
| `qwen/qwen3-32b:nitro` | `none` | none, minimal, low, medium, high |
| `anthropic/claude-haiku-4.5:nitro` | `none` | none, minimal, low, medium, high |

The gpt-oss models reject `Effort::None` and require at least `minimal`. All other candidate models accept every effort level.

### Dual-Model Architecture

The agent uses a two-phase approach: a reasoning model handles tool calling (e.g. `context_search`), then a response model produces the final JSON answer. This allows using a fast but hallucination-prone reasoner (like gpt-oss) paired with a more reliable response model. The response phase gets a separate system prompt without tool-calling instructions, since it has no tools available.

An optional `comprehension_model` can be configured for fact extraction, falling back to `response_model` if not set.

### JSON Parsing

LLM responses are parsed using the `llm-json` crate, which handles common LLM output issues: markdown fences, leading/trailing prose, missing quotes, trailing commas, and other malformed JSON. This is especially important for weaker models like gpt-oss-20b that produce inconsistent JSON formatting.
