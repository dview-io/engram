//! Library errors. Checks and policy never panic; they return these.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("embedder failed: {0}")]
    Embedder(String),

    #[error("store failed: {0}")]
    Store(String),

    #[error("invalid input: {0}")]
    Invalid(String),

    #[error("record not found: {0}")]
    NotFound(String),

    /// pgvector extension missing or unusable — host should disable engram and keep eval.
    #[error("pgvector unavailable: {0}")]
    VectorUnavailable(String),
}
