//! Retrieve decision trace (Phase 4y.1).
//!
//! Diagnostic only: building a trace never changes the [`Outcome`] returned by retrieve.
//! Each candidate row records why it was not used for few-shot or verified reuse, plus the
//! signature / table / KPI / fingerprint differences against the ask. Hosts log the trace so a
//! miss or unexpected path can be explained without reading `engram_ask_records` by hand.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::types::{
    AskQuery, AskRecord, AskSignature, MissReason, PathTaken, RecordId, SchemaFingerprint, TableRef,
};

/// Why a candidate was not used for a given path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RejectReason {
    /// Row is not eligible (strikes, thumbs-down, decay, regrade retract).
    Ineligible,
    /// Exact table set or KPI set differs from the ask (hard filter).
    TableOrKpiSet,
    /// Cosine similarity could not be computed (e.g. dimension mismatch).
    ScoreFailed,
    /// Similarity is below `few_shot_similarity_min`.
    BelowFewShotSimilarity,
    /// Similarity is below `verified_reuse_similarity_min`.
    BelowReuseSimilarity,
    /// `verified_reuse_enabled` is false in policy.
    ReuseDisabled,
    /// Passed the reuse gates but another candidate ranked higher for reuse.
    LowerRankedReuse,
    /// Few-shot candidate passed filters but fell outside `few_shot_limit`.
    TruncatedFromFewShot,
    /// Verified reuse was chosen, so few-shot examples were not built.
    ReusedInstead,
    /// SQL references absolute calendar dates; never replayed.
    AbsoluteDates,
    /// Query or stored signature has no metric / window (not reuse-ready).
    SignatureNotReady,
    /// Query or stored row has no tables.
    TablesEmpty,
    /// Query or stored row has no schema fingerprints.
    FingerprintsMissing,
    /// Fingerprints do not cover every table in the table set.
    FingerprintsNotCovering,
    /// Lifetime failures exceed successes after strikes; replay blocked.
    FailedReuseBias,
    /// Signature fields differ (see `signature_diff`).
    SignatureMismatch,
    /// Stored SQL tables are not a subset of the ask's available tables (4y.5).
    TableNotSubset,
    /// Table fingerprints differ (see `fingerprint_diff`).
    FingerprintMismatch,
    /// Stored row did not meet both `store_final_min` and `store_sql_min` for reuse (4y.9).
    BelowReuseScore,
}

/// One differing signature field, rendered as JSON for logs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDiff {
    pub field: String,
    pub ask: String,
    pub stored: String,
}

/// Set difference between the ask and a stored row (tables or KPIs).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetDiff {
    /// In the ask, missing from the stored row.
    pub ask_only: Vec<String>,
    /// In the stored row, missing from the ask.
    pub stored_only: Vec<String>,
}

impl SetDiff {
    pub fn is_empty(&self) -> bool {
        self.ask_only.is_empty() && self.stored_only.is_empty()
    }
}

/// Fingerprint for one table on the ask vs the stored row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintDiff {
    pub table: String,
    pub ask: Option<String>,
    pub stored: Option<String>,
}

/// Per-candidate decision record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateTrace {
    pub id: RecordId,
    pub eval_id: String,
    pub expert: bool,
    pub eligible: bool,
    /// `None` when the row never reached similarity scoring.
    pub similarity: Option<f32>,
    /// Why the row was not used for few-shot. `None` means it was eligible for few-shot.
    pub few_shot_reject: Option<RejectReason>,
    /// Why the row was not used for verified reuse. `None` means it passed every reuse gate.
    pub reuse_reject: Option<RejectReason>,
    /// Path that actually used this row, if any.
    pub selected: Option<PathTaken>,
    pub table_diff: SetDiff,
    pub kpi_diff: SetDiff,
    pub signature_diff: Vec<FieldDiff>,
    pub fingerprint_diff: Vec<FingerprintDiff>,
}

/// Whole-retrieve decision trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrieveTrace {
    pub ask_tables: Vec<String>,
    pub ask_kpis: Vec<String>,
    pub ask_signature: AskSignature,
    /// Whether the scope had any eligible row before table/KPI pushdown.
    pub store_has_eligible: bool,
    /// Rows returned by the store after table/KPI pushdown (`candidate_fetch_limit` cap).
    pub store_rows: usize,
    pub candidate_fetch_limit: usize,
    /// `Some` for VerifiedReuse / FewShot, `None` for Miss.
    pub path: Option<PathTaken>,
    pub miss_reason: Option<MissReason>,
    pub candidates: Vec<CandidateTrace>,
}

impl RetrieveTrace {
    pub(crate) fn new(ask: &AskQuery, candidate_fetch_limit: usize) -> Self {
        Self {
            ask_tables: sorted_strings(ask.tables.iter().map(TableRef::as_str)),
            ask_kpis: sorted_strings(ask.kpis.iter().map(crate::types::KpiRef::as_str)),
            ask_signature: ask.signature.clone(),
            store_has_eligible: false,
            store_rows: 0,
            candidate_fetch_limit,
            path: None,
            miss_reason: None,
            candidates: Vec::new(),
        }
    }

    pub(crate) fn miss(&mut self, reason: MissReason) {
        self.path = None;
        self.miss_reason = Some(reason);
    }

    /// Count of candidates per `few_shot_reject` / `reuse_reject` reason (for log summaries).
    pub fn reject_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut out = std::collections::BTreeMap::new();
        for c in &self.candidates {
            if let Some(r) = c.few_shot_reject {
                *out.entry(format!("few_shot:{}", reject_name(r))).or_insert(0) += 1;
            }
            if let Some(r) = c.reuse_reject {
                *out.entry(format!("reuse:{}", reject_name(r))).or_insert(0) += 1;
            }
        }
        out
    }
}

impl CandidateTrace {
    pub(crate) fn new(ask: &AskQuery, record: &AskRecord) -> Self {
        Self {
            id: record.id.clone(),
            eval_id: record.eval_id.clone(),
            expert: record.expert,
            eligible: record.eligible,
            similarity: None,
            few_shot_reject: None,
            reuse_reject: None,
            selected: None,
            table_diff: set_diff(
                ask.tables.iter().map(TableRef::as_str),
                record.tables.iter().map(TableRef::as_str),
            ),
            kpi_diff: set_diff(
                ask.kpis.iter().map(crate::types::KpiRef::as_str),
                record.kpis.iter().map(crate::types::KpiRef::as_str),
            ),
            signature_diff: signature_diff(&ask.signature, &record.ask_signature),
            fingerprint_diff: fingerprint_diff(&ask.fingerprints, &record.schema_fingerprints),
        }
    }
}

/// Stable snake_case name for a reject reason (matches serde).
pub fn reject_name(reason: RejectReason) -> &'static str {
    match reason {
        RejectReason::Ineligible => "ineligible",
        RejectReason::TableOrKpiSet => "table_or_kpi_set",
        RejectReason::ScoreFailed => "score_failed",
        RejectReason::BelowFewShotSimilarity => "below_few_shot_similarity",
        RejectReason::BelowReuseSimilarity => "below_reuse_similarity",
        RejectReason::ReuseDisabled => "reuse_disabled",
        RejectReason::LowerRankedReuse => "lower_ranked_reuse",
        RejectReason::TruncatedFromFewShot => "truncated_from_few_shot",
        RejectReason::ReusedInstead => "reused_instead",
        RejectReason::AbsoluteDates => "absolute_dates",
        RejectReason::SignatureNotReady => "signature_not_ready",
        RejectReason::TablesEmpty => "tables_empty",
        RejectReason::FingerprintsMissing => "fingerprints_missing",
        RejectReason::FingerprintsNotCovering => "fingerprints_not_covering",
        RejectReason::FailedReuseBias => "failed_reuse_bias",
        RejectReason::SignatureMismatch => "signature_mismatch",
        RejectReason::TableNotSubset => "table_not_subset",
        RejectReason::FingerprintMismatch => "fingerprint_mismatch",
        RejectReason::BelowReuseScore => "below_reuse_score",
    }
}

fn sorted_strings<'a>(items: impl Iterator<Item = &'a str>) -> Vec<String> {
    let set: BTreeSet<&str> = items.collect();
    set.into_iter().map(str::to_string).collect()
}

/// Set difference on the distinct values of both sides.
pub(crate) fn set_diff<'a>(
    ask: impl Iterator<Item = &'a str>,
    stored: impl Iterator<Item = &'a str>,
) -> SetDiff {
    let ask: BTreeSet<&str> = ask.collect();
    let stored: BTreeSet<&str> = stored.collect();
    SetDiff {
        ask_only: ask.difference(&stored).map(|s| (*s).to_string()).collect(),
        stored_only: stored.difference(&ask).map(|s| (*s).to_string()).collect(),
    }
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn push_field<T: PartialEq + Serialize>(out: &mut Vec<FieldDiff>, field: &str, a: &T, b: &T) {
    if a != b {
        out.push(FieldDiff { field: field.to_string(), ask: json(a), stored: json(b) });
    }
}

/// Signature fields that differ. Mirrors [`AskSignature::matches`] field-for-field.
pub(crate) fn signature_diff(ask: &AskSignature, stored: &AskSignature) -> Vec<FieldDiff> {
    let a = ask.normalized();
    let b = stored.normalized();
    let mut out = Vec::new();
    if a.has_structured_fields() || b.has_structured_fields() {
        push_field(&mut out, "aggregation", &a.aggregation, &b.aggregation);
        push_field(&mut out, "columns", &a.columns, &b.columns);
        push_field(&mut out, "group_by", &a.group_by, &b.group_by);
        push_field(&mut out, "structured_filters", &a.structured_filters, &b.structured_filters);
        push_field(&mut out, "structured_window", &a.structured_window, &b.structured_window);
        push_field(&mut out, "top_n", &a.top_n, &b.top_n);
    } else {
        push_field(&mut out, "metric", &a.metric, &b.metric);
        push_field(&mut out, "dimensions", &a.dimensions, &b.dimensions);
        push_field(&mut out, "grain", &a.grain, &b.grain);
        push_field(&mut out, "window", &a.window, &b.window);
        push_field(&mut out, "filters", &a.filters, &b.filters);
        push_field(&mut out, "top_n", &a.top_n, &b.top_n);
    }
    out
}

/// Per-table fingerprint comparison over the union of table ids.
pub(crate) fn fingerprint_diff(
    ask: &[(TableRef, SchemaFingerprint)],
    stored: &[(TableRef, SchemaFingerprint)],
) -> Vec<FingerprintDiff> {
    let tables: BTreeSet<&str> =
        ask.iter().map(|(t, _)| t.as_str()).chain(stored.iter().map(|(t, _)| t.as_str())).collect();
    tables
        .into_iter()
        .filter_map(|table| {
            let a = ask.iter().find(|(t, _)| t.as_str() == table).map(|(_, f)| f.as_str());
            let s = stored.iter().find(|(t, _)| t.as_str() == table).map(|(_, f)| f.as_str());
            (a != s).then(|| FingerprintDiff {
                table: table.to_string(),
                ask: a.map(str::to_string),
                stored: s.map(str::to_string),
            })
        })
        .collect()
}
