//! Host-injected embedding. The library never opens an HTTP client.

use async_trait::async_trait;

use crate::error::{Error, Result};

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Deterministic bag-of-words style embedder for tests (not for production).
#[cfg(any(test, feature = "test-utils"))]
#[derive(Clone, Debug)]
pub struct FakeEmbedder {
    dim: usize,
}

#[cfg(any(test, feature = "test-utils"))]
impl FakeEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Default for FakeEmbedder {
    fn default() -> Self {
        Self::new(32)
    }
}

#[cfg(any(test, feature = "test-utils"))]
#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0_f32; self.dim];
        for token in text.split_whitespace() {
            let mut h: u64 = 0xcbf29ce484222325;
            for b in token.bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            let idx = (h as usize) % self.dim;
            if let Some(slot) = v.get_mut(idx) {
                *slot += 1.0;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

/// Cosine similarity in \[0, 1\] for non-negative aligned vectors; otherwise clamped.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err(Error::Invalid(format!(
            "embedding length mismatch or empty: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok((dot / denom).clamp(-1.0, 1.0))
}
