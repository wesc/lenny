# Lenny1

The Lenny1 agent is an initial exploration of CAA, but focused on a [file over app](https://stephango.com/file-over-app) structure.

## Directory Hierarchy

We start with three top level directories, both of which can have an arbitrary number of subdirectories:

- `system/`: immutable context inserted on every call to the LLM
- `dynamic/`: mutable context inserted on every call to the LLM
- `references/`: immutable context accessed via tool calls and background processing

"Immutable" in this context means that the agent is not allowed to manipulate it, and "mutable" means the agent or one of its background processes can rewrite it at will.

## Loop Agent

The processing loop goes like this:

1. Assemble context by appending files in `system/` and `dynamic/`, sorted by lexicographical filename ordering.
2. Call out to LLM
3. Execute tool calls
4. Loop to step 2 until the LLM reaches max iteration or ends processing via a `final_answer(answer, 1-5 word summary slug)` tool call. Question: what should we do if hit max iteration?
5. The full final context of the loop is recorded into `references/turns/{timestamp}-{slug}` with a timestamp in the filename to ensure sort order and uniqueness.

## Dreaming

On insertion of new content, a background process starts up that summarizes the new content and inserts it into `dynamic/comprehensions/`. The comprehension, according to the CAA, should include reference information to the location in `references/`. The loop agent is presented with a tool that can lookup content in references.
