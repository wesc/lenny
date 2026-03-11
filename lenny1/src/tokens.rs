use tiktoken_rs::cl100k_base_singleton;

/// Count tokens in a string using the cl100k_base tokenizer.
/// This is the standard encoding for GPT-4, Claude, and most modern models.
/// While not exact for every model, it's a reliable approximation.
pub fn count_tokens(text: &str) -> usize {
    let bpe = cl100k_base_singleton();
    bpe.encode_ordinary(text).len()
}
