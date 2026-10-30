//! Text-to-SQL memory domain types.

use chrono::{NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Tenant / org isolation key. Required on every call.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope(String);

impl Scope {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Scope {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Scope {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Fully-qualified table reference, e.g. `catalog.schema.table`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableRef(String);

impl TableRef {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TableRef {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TableRef {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KpiRef(String);

impl KpiRef {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for KpiRef {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Hash of a table's column names and types (or sync generation). Opaque to matching.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaFingerprint(String);

impl SchemaFingerprint {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SchemaFingerprint {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Structured intent extracted before SQL generation.
///
/// Hosts should normalize dimensions (sorted, lowercased) and window phrasing before
/// calling retrieve / record. Matching is exact on the values provided.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AskSignature {
    pub metric: Option<String>,
    pub dimensions: Vec<String>,
    pub grain: Option<String>,
    /// Time window as stated, e.g. `last 6 months`.
    pub window: Option<String>,
    pub filters: serde_json::Map<String, serde_json::Value>,
    pub top_n: Option<u32>,
}

impl AskSignature {
    /// Enough structure to safely gate verified reuse (not an empty default).
    pub fn is_reuse_ready(&self) -> bool {
        self.metric.as_ref().is_some_and(|m| !m.trim().is_empty())
            || self.window.as_ref().is_some_and(|w| !w.trim().is_empty())
    }

    pub fn matches(&self, other: &AskSignature) -> bool {
        self.metric == other.metric
            && self.dimensions == other.dimensions
            && self.grain == other.grain
            && self.window == other.window
            && self.filters == other.filters
            && self.top_n == other.top_n
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PathTaken {
    Cold,
    FewShot,
    VerifiedReuse,
    Expert,
}

impl PathTaken {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::FewShot => "few_shot",
            Self::VerifiedReuse => "verified_reuse",
            Self::Expert => "expert",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "cold" => Some(Self::Cold),
            "few_shot" => Some(Self::FewShot),
            "verified_reuse" => Some(Self::VerifiedReuse),
            "expert" => Some(Self::Expert),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DecayReason {
    FailedReuse,
    NegativeFeedback,
    TableInvalidated,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MissReason {
    NoCandidates,
    /// Hard filters removed every eligible row (tables / KPIs).
    HardFilter,
    BelowSimilarity,
    EmptyStore,
    EmbedderFailed,
    /// Host retrieve timeout (fail-open).
    Timeout,
    /// Store / unexpected error (fail-open).
    Error,
    /// Ask-memory not enabled on the host.
    Disabled,
}

/// Outcome of grading a verified-reuse path, keyed by `(scope, eval_id)` for idempotency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TrustOutcome {
    Reinforce,
    DecayFailedReuse,
}

impl TrustOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reinforce => "reinforce",
            Self::DecayFailedReuse => "decay_failed_reuse",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "reinforce" => Some(Self::Reinforce),
            "decay_failed_reuse" => Some(Self::DecayFailedReuse),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecordId(Uuid);

impl RecordId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RecordId {
    fn default() -> Self {
        Self::new()
    }
}

/// Wall-clock time stored as Postgres `timestamp` (naive UTC).
pub type Timestamp = NaiveDateTime;

/// Prefer [`Timestamp`] / [`now`]. Kept as an alias for older call sites.
pub type TimestampMs = Timestamp;

pub fn now() -> Timestamp {
    Utc::now().naive_utc()
}

/// Prefer [`now`].
pub fn now_ms() -> Timestamp {
    now()
}

/// Graded ask stored for retrieve / few-shot / verified reuse.
///
/// Uniqueness: `(scope, eval_id)` — the same eval id may exist under different scopes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AskRecord {
    pub id: RecordId,
    pub scope: Scope,
    pub eval_id: String,
    pub conversation_id: Option<String>,
    pub question: String,
    pub embedding: Vec<f32>,
    pub tables: Vec<TableRef>,
    pub kpis: Vec<KpiRef>,
    pub ask_signature: AskSignature,
    /// Fingerprint per table.
    pub schema_fingerprints: Vec<(TableRef, SchemaFingerprint)>,
    pub sql: String,
    pub sql_has_absolute_dates: bool,
    pub pandas: Option<String>,
    pub scores: Scores,
    pub path_taken: PathTaken,
    pub expert: bool,
    /// Successful verified-reuse confirmations (bumped by [`crate::Memory::reinforce`] only).
    pub uses: u32,
    pub successes: u32,
    pub failures: u32,
    pub thumbs_up: u32,
    pub thumbs_down: u32,
    pub strikes: u32,
    pub eligible: bool,
    pub ineligible_reason: Option<String>,
    pub created_at: TimestampMs,
    pub updated_at: TimestampMs,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub final_score: Option<f32>,
    pub sql_score: Option<f32>,
}

/// Input for recording a newly graded ask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradedAsk {
    pub eval_id: String,
    pub conversation_id: Option<String>,
    pub question: String,
    pub tables: Vec<TableRef>,
    pub kpis: Vec<KpiRef>,
    pub ask_signature: AskSignature,
    pub schema_fingerprints: Vec<(TableRef, SchemaFingerprint)>,
    pub sql: String,
    pub pandas: Option<String>,
    pub scores: Scores,
    pub path_taken: PathTaken,
    pub expert: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepairRecord {
    pub id: RecordId,
    pub scope: Scope,
    pub table_ids: Vec<TableRef>,
    pub error_class: String,
    pub failed_sql_fragment: String,
    pub fixed_sql_fragment: String,
    pub schema_fingerprint: SchemaFingerprint,
    pub times_seen: u32,
    pub times_fixed: u32,
    pub eligible: bool,
    pub created_at: TimestampMs,
    pub updated_at: TimestampMs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepairInput {
    pub table_ids: Vec<TableRef>,
    pub error_class: String,
    pub failed_sql_fragment: String,
    pub fixed_sql_fragment: String,
    pub schema_fingerprint: SchemaFingerprint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AskQuery {
    pub question: String,
    pub tables: Vec<TableRef>,
    pub kpis: Vec<KpiRef>,
    pub signature: AskSignature,
    pub fingerprints: Vec<(TableRef, SchemaFingerprint)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorQuery {
    pub table_ids: Vec<TableRef>,
    pub error_class: String,
    pub schema_fingerprint: SchemaFingerprint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Outcome {
    VerifiedReuse { record: Box<AskRecord> },
    FewShot { examples: Vec<AskRecord> },
    Miss { reason: MissReason },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recorded {
    pub id: RecordId,
    pub created: bool,
}

/// Result of attempting to store a graded ask.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RecordAskResult {
    Stored(Recorded),
    SkippedBelowThreshold,
}
