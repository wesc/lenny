# Comprehensible Agent Architecture

Comprehensions are small semantically embedded notes of information derived from reference materials. Reference materials are information that either should be filled directly into content, such as chat history, PDFs, knowledge bases, or can be operated on by a subagent or subfunction. CAA defines a pipeline by which we assemble a context for an agent from a database of comprehensions. Don't worry, this will all be made clear soon.

## Pipeline

On receipt of a prompt, the agent constructs a model context like so:

1. Runs through a series comprehension lookup strategies and collect a set of comprehensions, S. A comprehension consists of a short natural language description, from which an embedding may be calculated, and metadata such as creation date. It also holds a reference type string and an opaque reference object. The reference type string identifies the type of reference, eg a PDF filename and page, a URL, a range of messages in a chat history, and the opaque reference object contains reference specific fields.

2. The comprehensions, S, are passed through a reranking phase, reordering the comprehensions in order of relevance.

3. S is then handed to the language model, M, along with a set of tools that can be applied to reference types contained within S. For example, if S contains a reference to an image, then a possible tool might be a subagent query to a vision model. The model, M, makes tool calls as appropriate.

4. Tool call results are stored into a memory log, and a comprehension is generated from the memory log. In other words, the tool call results become a reference object for a comprehension.

5. If M believes it has all the necessary information to answer the prompt then it does so. Otherwise it loops back to 1 with a clean context.

That final step is unusual -- we rely on step 1 to write a usable comprehension for the next iteration of the tool. We might think of this as automatic compaction on every turn.


## Comprehension

More precisely, a comprehension contains 4 fields:

- summary: short approximately 150 token natural language text that describes the referenced data
- create_time: RFC2822 formatted creation datetime
- reference_type: string that identifies the schema of the reference_data, type should be hierarchical and globally unique, org.caa.Pdf
- reference_data: opaque json encoded data according to the reference type

A comprehension lookup strategy takes a prompt, a database of comprehensions, and returns a list of comprehensions. Three basic strategies:

1. Time based lookup: return the most recently created comprehensions (this approximates a short term memory or chat history)
2. Keyword based lookup: rewrite the prompt as a set of entities then perform keyword lookup on the comprehensions
3. Semantic lookup: compute an embedding for the comprehension and look up from a vector database

Each lookup strategy is accompanied by a corresponding indexer. Indexers receive a reference and output comprehensions.

The CAA architecture allows for pluggable reference types, lookup, and indexing strategies. Thus much of the innovation around the system takes place at these entrypoints.


## Dreaming

Over time, a user's agent will receive new lookup and indexing strategies which should improve its recall of old information. To faciliate an index update, agents periodically enter a dream state:

1. For every piece of reference material, construct new comprehensions from the current set of indexers, wiping out old associated comprehensions.
2. Aggregation... open question: we want to be able to collect up a bunch of comprehensions and cluster them, sort of hierarchically. So maybe we define a special kind of recursive comprehension, called a topic, that can always operate on any comprehension regardless of the reference type? As in, perhaps it always operates in vector space on the embedded summary.


## Analysis of the Comprehensible Agent Architecture (CAA)

### Core Insight

The central idea here is treating comprehensions as a universal interface layer between heterogeneous data sources and the language model. Rather than stuffing raw references into context, you create lightweight semantic summaries that point back to the originals. This is a genuinely clean abstraction — it means the LLM only ever reasons over a uniform structure (short text + metadata + opaque reference), and detail retrieval is deferred to tool calls. It's essentially a two-phase approach: find what's relevant, then pull what's needed.

### Strengths

The clean context loop (Step 5 → Step 1) is the most interesting design choice. By forcing every iteration to start fresh and rely on comprehension generation to carry state forward, you get automatic context compaction for free. The LLM never accumulates a bloated context window across iterations — it only ever sees what the comprehension layer thinks is relevant. This sidesteps one of the most common failure modes in agentic loops (context window exhaustion).

The pluggable reference type system is well-conceived. Making reference_type a hierarchical namespace (org.caa.Pdf) gives you an extensibility model that's similar to MIME types or Java package naming. This means third parties can define new reference types and the corresponding lookup strategies, indexers, and tools without colliding.

The triple lookup strategy (recency, keyword, semantic) covers the main retrieval failure modes nicely. Semantic search alone is notoriously unreliable for entity-heavy or temporal queries; keyword search compensates for that, and time-based lookup gives you a working memory analogue.

### Concerns and Weaknesses

**Information loss at the compaction boundary.** The clean-context restart in step 5 is elegant but risky. The comprehension generated in step 4 is the sole carrier of state between iterations. If the comprehension generator drops a nuance — a conditional, a negation, a numeric detail — it's gone permanently. You're betting heavily on the quality of a single summarization step. In practice, multi-hop reasoning chains are exactly where summaries tend to lose fidelity. You might want a mechanism for the model to "pin" specific facts into the next iteration's context verbatim, bypassing summarization.

**Reranking without the query context.** Step 2 reranks comprehensions for relevance, but it's not clear what signal drives the reranking. If it's just the original user prompt, that may be insufficient in later loop iterations where the real information need has evolved based on tool call results. The reranker likely needs access to the accumulated comprehension from prior iterations, not just the original prompt.

**Tool availability is coupled to comprehension content.** Step 3 says the tool set is determined by the reference types present in S. This means if a relevant comprehension wasn't retrieved, the corresponding tool isn't even offered to the model. That's a chicken-and-egg problem: the model can't request data it doesn't know exists. You may need a small set of "discovery" tools that are always available (e.g., a general search tool that can surface new comprehensions).

**The 150-token summary budget is tight.** For simple references (a chat message, a single fact), 150 tokens is plenty. For complex references (a 40-page PDF, a long conversation thread), the summary may be too lossy to support accurate retrieval. This is especially true for keyword-based lookup — if a critical entity doesn't make it into the 150-token summary, that comprehension becomes invisible to that strategy.

**Loop termination is underspecified.** "If M believes it has all the necessary information" is doing a lot of work. In practice, LLMs are poor judges of their own completeness, and you risk either premature termination (the model confidently answers with incomplete info) or infinite loops. A maximum iteration count is obvious, but you might also want an explicit "confidence delta" check — if two consecutive iterations yield essentially the same comprehensions, halt.

### On Dreaming

The dreaming concept is the right instinct — reindexing with improved strategies is analogous to memory consolidation. The open question about hierarchical topic clustering is worth pursuing. A natural approach: define a org.caa.Topic reference type whose reference_data contains a list of child comprehension IDs and whose summary is generated by clustering embedded summaries (e.g., via HDBSCAN or agglomerative clustering in embedding space, then prompting an LLM to name and summarize each cluster). This gives you a tree structure where the model can retrieve at the right level of granularity — a broad topic comprehension first, then drill into children via tool call.

The risk with dreaming is semantic drift: if you wipe old comprehensions and regenerate, you may lose retrieval pathways that previously worked. Consider versioning comprehensions rather than replacing them, at least until you can verify recall quality hasn't degraded.

### Summary

CAA is a thoughtful architecture that gets the core abstraction right: comprehensions as a uniform, lightweight, pointer-rich representation layer. The main risks cluster around the fidelity of the compaction step and the completeness of retrieval. The system would benefit from mechanisms that let the model override summarization when precision matters, and from always-available discovery tools that break the coupling between "what was retrieved" and "what can be explored."

### Human Response

> 1. On wiping comprehensions: perhaps we should have a pluggable architecture. A basic wipe would just do it all the time, but future researchers could implement better wiping. Or perhaps the pluggable unit should actually be dreaming, which includes pruning.
> 2. The 150 token budget is for one comprehension. In reality, a 40 page PDF will emit many comprehensions, and possibly hierarchical comprehensions as well. The 150 token budget is to make the semantic indexing meaningful. We could generate larger comprehensions as well, and perhaps we should, if we were to use late interaction retrieval like ColBERT.
> 3. Should reranking happen on the comprehensions or on the results of tool calls on comprehensions? The reason why I specified on comprehensions is because they have a determined common structure, so a single reranker can operate on all comprehensions agnostic of the reference type. We might also require that tool calls return documents with a summary field, in which case we can rerank on the tool results rather than comprehensions. What do you think?

### Response

These are good refinements. Let me take them in turn.

### 1. Pluggable Dreaming

Making dreaming itself the pluggable unit rather than just the pruning step is the stronger choice. Pruning is really just one concern within dreaming — you also have reindexing, clustering, and potentially quality validation. If you make the pluggable boundary "dreaming," a researcher can implement a strategy that does all of these in a coordinated way. A pruning-only plugin point forces an assumption that reindexing and pruning are independent, which they probably aren't — you'd want to prune *after* seeing what the new indexers produce, for instance.

A reasonable interface might be: a dream strategy receives the full set of reference materials and the current comprehension database, and returns a new comprehension database. That's maximally flexible. The "basic wipe and regenerate" strategy is then just the trivial implementation.

One consideration: dreaming strategies will likely need access to the embedding model and possibly the LLM itself (for re-summarization, clustering, topic generation). So the interface should probably accept those as dependencies rather than baking in assumptions about which models are available.

### 2. Comprehension Granularity

That clarifies things significantly — a 40-page PDF emitting dozens of comprehensions, each covering a coherent chunk, is a much more workable model. The 150-token budget makes good sense as a target for embedding quality; you're right that shorter, focused text embeds more meaningfully than long passages.

On the ColBERT point: the tension is real. Late interaction models like ColBERT benefit from richer token-level representations, so a longer comprehension gives the retriever more to work with. But longer comprehensions also mean fewer fit in the context window after retrieval, and the summaries become harder to scan during reranking. One possible resolution is to maintain *two representations per comprehension*: the short ~150-token summary for context assembly and reranking, and a longer ~500-token version used only as the retrieval document for ColBERT. The short summary is what the LLM sees; the long version is what the retriever indexes. This keeps the context window efficient while giving late interaction models the richer signal they need. It's more storage and more indexing work, but comprehensions are cheap relative to the reference materials they point to.

### 3. Where to Rerank

This is the most interesting question. I think the answer is **both, at different stages**, and here's why.

Reranking on comprehensions serves a *selection* purpose: given a large candidate set from the lookup strategies, narrow it down to what's worth spending tool calls on. This is where the uniform structure argument is strongest. You have a homogeneous set of ~150-token summaries, a single cross-encoder can score them all, and the cost is low. This is essentially deciding what to *look at*.

But there's a second decision point after tool calls return: now you have detailed content, and the model needs to decide what to *use*. If tool results also carry a summary field, you could run a second reranking pass that operates on richer information. This second pass answers a different question — not "is this reference relevant?" but "does this retrieved content actually contain what I need?"

The practical benefit of two-stage reranking is that it lets you be aggressive with the first stage (retrieve broadly, keep more candidates) without overwhelming the context window, because the second stage filters the expanded content before the model reasons over it.

The cost is complexity and latency. If you want to keep it simple, reranking only on comprehensions is defensible — it's where you get the most leverage per unit of complexity. But I'd design the architecture so that tool call results *do* carry a summary field regardless, because even if you don't rerank on them immediately, that summary is what gets written into the memory log comprehension in step 4. If tool results are just raw unstructured content, the comprehension generated from them will be lower quality.

So my recommendation: require that tool call results return a structured response with at least a summary field. Rerank on comprehensions as the default pipeline. Make post-tool reranking an optional pipeline stage that implementations can enable. This keeps the core architecture clean while leaving the door open.

One subtle implication worth noting: if you rerank after tool calls, you're implicitly allowing the system to *discard* information it spent compute retrieving. That feels wasteful, but it's actually the right behavior — it means retrieval errors are recoverable within a single iteration rather than polluting the context.
