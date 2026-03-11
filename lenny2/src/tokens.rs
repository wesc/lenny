use tiktoken_rs::cl100k_base_singleton;

/// Count tokens in a string using the cl100k_base tokenizer.
pub fn count_tokens(text: &str) -> usize {
    let bpe = cl100k_base_singleton();
    bpe.encode_ordinary(text).len()
}
