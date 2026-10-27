//! Graded text-to-SQL memory: verified reuse, few-shot examples, or miss.
//!
//! Phase 1: domain types, policy, SQL inspection, in-memory store, and unit tests.
//! The host injects an [`Embedder`] and a [`MemoryStore`].

#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)
)]

pub mod embedder;
pub mod error;
pub mod memory;
pub mod policy;
pub mod sql_inspect;
pub mod store;
pub mod types;

#[cfg(any(test, feature = "test-utils"))]
pub use embedder::FakeEmbedder;
pub use embedder::{cosine_similarity, Embedder};
pub use error::{Error, Result};
pub use memory::Memory;
pub use policy::Policy;
pub use sql_inspect::sql_has_absolute_dates;
pub use store::{InMemoryStore, MemoryStore};
pub use types::*;
