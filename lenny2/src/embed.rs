use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Mutex, OnceLock};

static EMBEDDING_MODEL: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

fn get_model() -> &'static Mutex<TextEmbedding> {
    EMBEDDING_MODEL.get_or_init(|| {
        Mutex::new(
            TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
                .expect("failed to initialize fastembed model"),
        )
    })
}

/// Embed a batch of texts, returning one 384-dim f32 vector per text.
pub fn embed_batch(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    let mut model = get_model().lock().unwrap();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let vecs = model.embed(refs, None)?;
    Ok(vecs)
}

/// Embed a single query string.
pub fn embed_query(text: &str) -> Result<Vec<f32>> {
    let mut model = get_model().lock().unwrap();
    let mut vecs = model.embed(vec![text], None)?;
    Ok(vecs.remove(0))
}

/// Reinterpret a `&[f32]` slice as raw bytes for sqlite-vec.
pub fn f32_to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
