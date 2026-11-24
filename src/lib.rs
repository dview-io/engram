//! Graded text-to-SQL memory: verified reuse, few-shot examples, or miss.
//!
//! The host injects an [`Embedder`] and a [`MemoryStore`]. Phase 1: in-memory.
//! Phase 2: optional Postgres + pgvector store (`feature = "postgres"`).

#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)
)]

pub mod embedder;
pub mod error;
pub mod memory;
pub mod policy;
pub mod schema;
pub mod sql_inspect;
pub mod store;
pub mod trace;
pub mod types;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(any(test, feature = "test-utils"))]
pub use embedder::FakeEmbedder;
pub use embedder::{cosine_similarity, Embedder};
pub use error::{Error, Result};
pub use memory::Memory;
pub use policy::Policy;
#[cfg(feature = "postgres")]
pub use postgres::{vector_literal, PostgresStore};
#[cfg(feature = "postgres")]
pub use schema::apply_schema;
pub use schema::{
    ask_table_def, repair_table_def, trust_table_def, EngramColumn, EngramIndex, EngramSchemaSpec,
    EngramSqlType, EngramTableDef, EngramUnique, ASK_TABLE, DEFAULT_EMBEDDING_DIM, REPAIR_TABLE,
    TRUST_TABLE,
};
pub use sql_inspect::{
    filter_tables_to_sql, normalize_sql, normalized_sql_hash, sql_has_absolute_dates,
    sql_referenced_tables,
};
pub use store::{InMemoryStore, MemoryStore};
pub use trace::{
    reject_name, CandidateTrace, FieldDiff, FingerprintDiff, RejectReason, RetrieveTrace, SetDiff,
};
pub use types::{
    now, now_ms, AskQuery, AskRecord, AskSignature, DecayReason, ErrorQuery, GradedAsk, KpiRef,
    MissReason, Outcome, PathTaken, RecordAskResult, RecordId, Recorded, RepairInput, RepairRecord,
    SchemaFingerprint, Scope, Scores, SignatureFilter, SignatureWindow, TableRef, Timestamp,
    TimestampMs, TrustOutcome, Vote,
};
