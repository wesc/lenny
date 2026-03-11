# Lenny2, the Interrobot.

Lenny2 implements an Interrobot. It explores and _interrogates_ knowledge and beliefs.

## Rumination

Rumination is split into two phases: **download** and **cluster**. This allows re-clustering during development without re-downloading pages.

### Phase 1: Download

1. Start with a link to a webpage and scrape it via Firecrawl.
2. Use an LLM (via Rig Agent with structured output) to generate 10 exploratory questions as chain-of-thought scaffolding — these help the model think broadly about the topic. From this reasoning, the model produces 3-5 concrete search queries.
3. Perform a web search (Firecrawl search API) on each query and add the top result for each into a FIFO queue (skipping already-seen URLs).
4. Recurse to step 1 with the links in the FIFO queue.

After n pages are scraped (default 100, parameterizable), we terminate. Pages are stored as numbered Markdown files (`0000.md`, `0001.md`, ...) in `data/pages/<RUMINATION-ID>/` where RUMINATION-ID is the current datetime (e.g. `data/pages/20260310-143022/`). Each file includes a `<!-- url: ... -->` header. This page cache persists in perpetuity.

### Phase 2: Cluster

Given a RUMINATION-ID, cluster all downloaded pages:

1. Embed each page using fastembed.
2. Cluster embeddings using k-means with k = sqrt(n) (overridable).
3. For each cluster, compute the centroid (mean of member embeddings).
4. For each cluster, generate a detailed summary (~5k tokens) via LLM with structured output. The summary addresses:
   - What are the 5 core mental models that describe this topic?
   - 3 strong statements on the key ideas, with reasoning.
   - 3 strong statements *disagreeing* on the key ideas, with reasoning.
   - A summary paying attention to key entities and concepts a user might inquire about.
5. Store the centroid, summary, and page references in an sqlite-vec database.

Target: ~5k tokens per cluster summary. At ~10 clusters for 100 pages, that's ~50k tokens total, allowing ~10 summaries to fit within a 50k-token context budget where LLM performance remains reliable.

This ends one rumination pass. We can then start a new pass from scratch, or from any of these topics.

## Query

An agent loop (Rig Agent with OpenRouter) starts with a user prompt and chat history. We look up via the summary database the 3 closest topic summaries to the prompt and recent chat history. This happens on every run through the loop. As the conversation progresses, the expectation is that the target *slides* over topic areas.

## CLI Commands

- `download <url> [--max-pages N]` — Run the download phase. Creates `data/pages/<RUMINATION-ID>/`. Default max-pages: 100.
- `cluster <rumination-id> [--k K]` — Run clustering on a downloaded page set. Default k: sqrt(n).
- `query <prompt>` — One-shot query against the knowledge base.
- `chat` — Interactive chat loop against the knowledge base.
- `build-context -n <N> <prompt>` — Return the N closest cluster summaries to the prompt. Default n: 3.

## Configuration

Lenny2 loads configuration from `config.yaml` if present, otherwise falls back to environment variables:

- `OPENROUTER_API_KEY` (required)
- `FIRECRAWL_API_KEY` (required)
- `LENNY2_MODEL` (optional, defaults to `anthropic/claude-haiku-4.5`)

The data directory defaults to `data/`.

## Implementation Notes

- **Scraping**: Firecrawl (Markdown format). Page content truncated to ~6000 words for LLM input.
- **Search**: Firecrawl search API.
- **LLM**: OpenRouter via Rig's OpenRouter provider. Default model: `anthropic/claude-haiku-4.5` (configurable).
- **Structured output**: Rig's extractor with serde + schemars-derived schemas. No manual parsing.
- **Embeddings**: fastembed `AllMiniLML6V2` (384-dimensional). Input truncated to ~500 words.
- **Token counting**: tiktoken `cl100k_base` tokenizer.
- **Storage**: Single sqlite-vec database (`data/lenny2.db`) for all runs, with WAL journaling.
- **Clustering**: linfa k-means on fastembed embeddings. Centroid = mean embedding vector. Fixed seed (42) for reproducibility.
