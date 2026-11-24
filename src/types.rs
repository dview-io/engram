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

/// One structured filter predicate in [`AskSignature`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SignatureFilter {
    #[serde(default)]
    pub column: String,
    #[serde(default)]
    pub op: String,
    #[serde(default)]
    pub value: serde_json::Value,
}

/// Structured relative/anchored time window in [`AskSignature`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignatureWindow {
    /// Canonical singular unit: `hour`, `day`, `week`, `month`, `quarter`, `year`.
    #[serde(default)]
    pub unit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// Optional anchor, e.g. `now`, `current_date`, `mtd`, `ytd`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

/// Structured intent extracted before SQL generation (Phase 4y.2–4y.3).
///
/// When structured fields (`aggregation`, `columns`, `group_by`, `structured_filters`,
/// `structured_window`) are populated, verified-reuse comparison uses **only** those
/// answer-changing fields plus `top_n` (after [`Self::normalize`]); free-text `metric` prose
/// is ignored so phrasing drift does not block reuse.
///
/// Legacy flat fields (`metric`, `dimensions`, `grain`, `window`, `filters`) remain with
/// `#[serde(default)]` so existing DB rows deserialize cleanly. A structured query never
/// matches a legacy-only row.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AskSignature {
    /// Structured aggregation token (`sum`, `count`, `count_distinct`, `avg`, `min`, `max`, `list`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregation: Option<String>,
    /// Target / measure columns referenced by the ask (sorted, lowercased).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<String>,
    /// Grouping columns (sorted, lowercased).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_by: Vec<String>,
    /// Structured filter predicates `{column, op, value}` (normalized + sorted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub structured_filters: Vec<SignatureFilter>,
    /// Structured time window `{unit, count, anchor}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_window: Option<SignatureWindow>,
    /// Optional explicit top-N limit. Compared exact (`None` != `Some(10)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,

    // Legacy free-text fields (kept for backward-compat JSONB deserialization; ignored when
    // structured fields are present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain: Option<String>,
    /// Legacy time window as stated, e.g. `last 6 months`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub filters: serde_json::Map<String, serde_json::Value>,
}

impl AskSignature {
    /// True when any Phase 4y.2 structured field is populated.
    pub fn has_structured_fields(&self) -> bool {
        self.aggregation.as_ref().is_some_and(|s| !s.trim().is_empty())
            || self.columns.iter().any(|c| !c.trim().is_empty())
            || self.group_by.iter().any(|g| !g.trim().is_empty())
            || self.structured_filters.iter().any(|f| !f.column.trim().is_empty())
            || self.structured_window.as_ref().is_some_and(|w| !w.unit.trim().is_empty())
    }

    /// Normalize casing, synonyms, and sort order in place.
    pub fn normalize(&mut self) {
        self.aggregation = self.aggregation.as_deref().and_then(normalize_aggregation);
        normalize_ident_list(&mut self.columns);
        normalize_ident_list(&mut self.group_by);
        self.structured_filters = std::mem::take(&mut self.structured_filters)
            .into_iter()
            .filter_map(normalize_filter)
            .collect();
        self.structured_filters.sort_by(|a, b| {
            a.column
                .cmp(&b.column)
                .then_with(|| a.op.cmp(&b.op))
                .then_with(|| json_sort_key(&a.value).cmp(&json_sort_key(&b.value)))
        });
        self.structured_filters.dedup();
        self.structured_window = self.structured_window.take().and_then(normalize_window);
        if self.top_n == Some(0) {
            self.top_n = None;
        }

        // Legacy fields
        self.metric = normalize_opt_lower(self.metric.as_deref());
        normalize_ident_list(&mut self.dimensions);
        self.grain = normalize_opt_lower(self.grain.as_deref());
        self.window = normalize_opt_lower(self.window.as_deref());
    }

    /// Return a normalized copy without mutating `self`.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Enough structure to safely gate verified reuse (not an empty default).
    pub fn is_reuse_ready(&self) -> bool {
        if self.has_structured_fields() {
            return self.aggregation.as_ref().is_some_and(|s| !s.trim().is_empty())
                || self.columns.iter().any(|c| !c.trim().is_empty())
                || self.structured_window.as_ref().is_some_and(|w| !w.unit.trim().is_empty());
        }
        self.metric.as_ref().is_some_and(|m| !m.trim().is_empty())
            || self.window.as_ref().is_some_and(|w| !w.trim().is_empty())
    }

    /// Reuse comparison (Phase 4y.3):
    /// - Both structured → compare only answer-changing structured fields + `top_n` (ignore prose `metric`).
    /// - Mixed structured vs legacy → never match.
    /// - Both legacy → exact match on legacy fields.
    pub fn matches(&self, other: &AskSignature) -> bool {
        let a = self.normalized();
        let b = other.normalized();
        match (a.has_structured_fields(), b.has_structured_fields()) {
            (true, true) => {
                a.aggregation == b.aggregation
                    && a.columns == b.columns
                    && a.group_by == b.group_by
                    && a.structured_filters == b.structured_filters
                    && a.structured_window == b.structured_window
                    && a.top_n == b.top_n
            }
            (false, false) => {
                a.metric == b.metric
                    && a.dimensions == b.dimensions
                    && a.grain == b.grain
                    && a.window == b.window
                    && a.filters == b.filters
                    && a.top_n == b.top_n
            }
            _ => false,
        }
    }
}

fn normalize_opt_lower(v: Option<&str>) -> Option<String> {
    v.map(str::trim).filter(|s| !s.is_empty()).map(str::to_lowercase)
}

fn normalize_ident_list(items: &mut Vec<String>) {
    *items = items.iter().map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
    items.sort();
    items.dedup();
}

fn normalize_aggregation(raw: &str) -> Option<String> {
    let s = raw.trim().to_lowercase();
    if s.is_empty() {
        return None;
    }
    let canon = match s.as_str() {
        "total" | "sum" => "sum",
        "count" => "count",
        "count_distinct" | "count distinct" | "distinct_count" | "distinct count" => {
            "count_distinct"
        }
        "avg" | "average" | "mean" => "avg",
        "min" | "minimum" => "min",
        "max" | "maximum" => "max",
        "list" | "select" | "detail" | "rows" => "list",
        other => return Some(other.to_string()),
    };
    Some(canon.to_string())
}

fn normalize_op(raw: &str) -> String {
    match raw.trim().to_lowercase().as_str() {
        "==" | "eq" => "=".to_string(),
        "<>" | "neq" | "!=" => "!=".to_string(),
        "gt" => ">".to_string(),
        "gte" | ">=" => ">=".to_string(),
        "lt" => "<".to_string(),
        "lte" | "<=" => "<=".to_string(),
        "not in" => "not_in".to_string(),
        other => other.to_string(),
    }
}

fn normalize_filter_value(val: serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::String(s) => serde_json::Value::String(s.trim().to_lowercase()),
        serde_json::Value::Array(arr) => {
            let mut norm: Vec<serde_json::Value> =
                arr.into_iter().map(normalize_filter_value).collect();
            norm.sort_by_key(json_sort_key);
            serde_json::Value::Array(norm)
        }
        other => other,
    }
}

fn normalize_filter(f: SignatureFilter) -> Option<SignatureFilter> {
    let column = f.column.trim().to_lowercase();
    if column.is_empty() {
        return None;
    }
    let op = normalize_op(&f.op);
    let value = normalize_filter_value(f.value);
    Some(SignatureFilter { column, op, value })
}

fn normalize_window(w: SignatureWindow) -> Option<SignatureWindow> {
    let unit = match w.unit.trim().to_lowercase().as_str() {
        "" => return None,
        "hour" | "hours" | "hr" | "hrs" => "hour".to_string(),
        "day" | "days" => "day".to_string(),
        "week" | "weeks" | "wk" | "wks" => "week".to_string(),
        "month" | "months" | "mo" | "mos" => "month".to_string(),
        "quarter" | "quarters" | "qtr" | "qtrs" => "quarter".to_string(),
        "year" | "years" | "yr" | "yrs" => "year".to_string(),
        other => other.to_string(),
    };
    let anchor = normalize_opt_lower(w.anchor.as_deref());
    Some(SignatureWindow { unit, count: w.count, anchor })
}

fn json_sort_key(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
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
    /// Ask was re-graded below the store threshold; keep the row but stop reuse.
    RegradedBelowThreshold,
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
    /// An existing row for this eval was marked ineligible after a below-threshold regrade.
    Retracted {
        id: RecordId,
    },
    /// Skipped because the ask was already served via `VerifiedReuse` (4y.10).
    SkippedVerifiedReuse,
    /// Skipped because an eligible row with the same normalized SQL + tables + signature already exists (4y.10).
    Deduplicated {
        id: RecordId,
    },
}
