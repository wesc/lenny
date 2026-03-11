# Lenny Project

## After code changes

Always run these before considering work complete:

1. `cargo fmt` — format all Rust code
2. `cargo clippy` — fix any warnings
3. `cargo test` — ensure all tests pass

## Token estimation

Use lenny1's built-in command for token counts instead of writing Python scripts:

```
cargo run -- estimate-tokens <path>
```

This uses tiktoken's cl100k_base tokenizer (same as GPT-4/Claude) for accurate counts.
