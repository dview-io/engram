//! Matching, reuse gates, ranking, and trust thresholds.

use crate::embedder::cosine_similarity;
use crate::error::Result;
use crate::trace::RejectReason;
use crate::types::{AskQuery, AskRecord, AskSignature, SchemaFingerprint, TableRef};

/// Tunable thresholds supplied by the host configuration.
#[derive(Clone, Debug)]
pub struct Policy {
    pub store_final_min: f32,
    pub store_sql_min: f32,
    pub few_shot_similarity_min: f32,
    pub verified_reuse_similarity_min: f32,
    pub few_shot_limit: usize,
    /// Max eligible rows fetched from the store for ranking (pgvector nearest-N).
    pub candidate_fetch_limit: usize,
    pub max_strikes: u32,
    /// Thumbs-down count that marks an ask ineligible.
    pub max_thumbs_down: u32,
    /// When false, never return VerifiedReuse (still may FewShot).
    pub verified_reuse_enabled: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            store_final_min: 70.0,
            store_sql_min: 7.0,
            few_shot_similarity_min: 0.75,
            verified_reuse_similarity_min: 0.92,
            few_shot_limit: 3,
            candidate_fetch_limit: 64,
            max_strikes: 2,
            max_thumbs_down: 2,
            verified_reuse_enabled: true,
        }
    }
}

impl Policy {
    pub fn is_storable(&self, final_score: Option<f32>, sql_score: Option<f32>) -> bool {
        if let Some(f) = final_score {
            if f >= self.store_final_min {
                return true;
            }
        }
        if let Some(s) = sql_score {
            if s >= self.store_sql_min {
                return true;
            }
        }
        false
    }

    /// Strict reuse-tier score gate (Phase 4y.9): requires BOTH `final_score >= store_final_min`
    /// AND `sql_score >= store_sql_min` (expert rows bypass). Rows passing only `is_storable`
    /// remain eligible for few-shot only.
    pub fn is_reuse_score_eligible(&self, expert: bool, scores: &crate::types::Scores) -> bool {
        if expert {
            return true;
        }
        scores.final_score.is_some_and(|f| f >= self.store_final_min)
            && scores.sql_score.is_some_and(|s| s >= self.store_sql_min)
    }

    pub fn should_ineligible_after_strikes(&self, strikes: u32) -> bool {
        strikes >= self.max_strikes
    }

    pub fn should_ineligible_after_thumbs_down(&self, thumbs_down: u32) -> bool {
        thumbs_down >= self.max_thumbs_down
    }
}

#[derive(Clone, Debug)]
pub struct RankedCandidate {
    pub record: AskRecord,
    pub similarity: f32,
    /// Jaccard table overlap ratio in `[0.0, 1.0]` between query and stored tables (4y.5).
    pub table_overlap: f32,
}

/// Few-shot hard filter (4y.5): eligible; non-empty table overlap; same KPI set.
pub fn hard_filter(query: &AskQuery, record: &AskRecord) -> bool {
    if !record.eligible {
        return false;
    }
    tables_overlap(&query.tables, &record.tables) && same_kpi_set(&query.kpis, &record.kpis)
}

/// True when `a` and `b` share at least one table (non-empty overlap).
pub fn tables_overlap(a: &[TableRef], b: &[TableRef]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a.iter().any(|ta| b.iter().any(|tb| ta == tb))
}

/// True when every table in `required` is present in `available` (and `required` is non-empty).
pub fn tables_is_subset(required: &[TableRef], available: &[TableRef]) -> bool {
    !required.is_empty() && required.iter().all(|r| available.iter().any(|a| a == r))
}

/// Jaccard overlap ratio `|A ∩ B| / |A ∪ B|` in `[0.0, 1.0]`.
pub fn table_overlap_ratio(a: &[TableRef], b: &[TableRef]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<&str> = a.iter().map(TableRef::as_str).collect();
    let sb: HashSet<&str> = b.iter().map(TableRef::as_str).collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count();
    inter as f32 / union as f32
}

fn same_kpi_set(a: &[crate::types::KpiRef], b: &[crate::types::KpiRef]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut left: Vec<&str> = a.iter().map(|k| k.as_str()).collect();
    let mut right: Vec<&str> = b.iter().map(|k| k.as_str()).collect();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

/// All reuse gates for VerifiedReuse.
pub fn reuse_gates_pass(query: &AskQuery, record: &AskRecord) -> bool {
    reuse_gate_reject(query, record).is_none()
}

/// First reuse gate that blocks `record` for `query`, or `None` when every gate passes.
///
/// Same order and semantics as [`reuse_gates_pass`]; the reason is for diagnostics (4y.1).
pub fn reuse_gate_reject(query: &AskQuery, record: &AskRecord) -> Option<RejectReason> {
    if record.sql_has_absolute_dates {
        return Some(RejectReason::AbsoluteDates);
    }
    if !query.signature.is_reuse_ready() || !record.ask_signature.is_reuse_ready() {
        return Some(RejectReason::SignatureNotReady);
    }
    // Empty tables / fingerprints must not verify — signature alone is not enough.
    if query.tables.is_empty() || record.tables.is_empty() {
        return Some(RejectReason::TablesEmpty);
    }
    if query.fingerprints.is_empty() || record.schema_fingerprints.is_empty() {
        return Some(RejectReason::FingerprintsMissing);
    }
    // Verified reuse requires every table used by the stored SQL to be available in the ask.
    if !tables_is_subset(&record.tables, &query.tables) {
        return Some(RejectReason::TableNotSubset);
    }
    if !fingerprints_cover_tables(&record.tables, &record.schema_fingerprints)
        || !fingerprints_cover_tables(&record.tables, &query.fingerprints)
    {
        return Some(RejectReason::FingerprintsNotCovering);
    }
    // Lifetime failed-reuse bias: more failures than successes blocks replay.
    if record.strikes > 0 && record.failures > record.successes {
        return Some(RejectReason::FailedReuseBias);
    }
    if !signatures_match(&query.signature, &record.ask_signature) {
        return Some(RejectReason::SignatureMismatch);
    }
    if !fingerprints_match_subset(&record.tables, &query.fingerprints, &record.schema_fingerprints)
    {
        return Some(RejectReason::FingerprintMismatch);
    }
    None
}

/// Every table in `tables` has a fingerprint entry.
pub fn fingerprints_cover_tables(
    tables: &[TableRef],
    fingerprints: &[(TableRef, SchemaFingerprint)],
) -> bool {
    !tables.is_empty() && tables.iter().all(|t| fingerprints.iter().any(|(ft, _)| ft == t))
}

fn signatures_match(a: &AskSignature, b: &AskSignature) -> bool {
    a.matches(b)
}

/// Every table in `required_tables` (stored SQL tables) has matching fingerprint in `query` and `stored`.
fn fingerprints_match_subset(
    required_tables: &[TableRef],
    query: &[(TableRef, SchemaFingerprint)],
    stored: &[(TableRef, SchemaFingerprint)],
) -> bool {
    if required_tables.is_empty() || query.is_empty() || stored.is_empty() {
        return false;
    }
    for table in required_tables {
        let Some((_, qf)) = query.iter().find(|(t, _)| t == table) else {
            return false;
        };
        let Some((_, sf)) = stored.iter().find(|(t, _)| t == table) else {
            return false;
        };
        if qf != sf {
            return false;
        }
    }
    true
}

/// Few-shot ranking: expert first, then similarity + small table-overlap bonus, net feedback, score, successes.
pub fn few_shot_rank_key(c: &RankedCandidate) -> (bool, ordered::OrderedF32, i64, u32, u32) {
    let net_feedback = i64::from(c.record.thumbs_up) - i64::from(c.record.thumbs_down);
    let blended = c.similarity + 0.05 * c.table_overlap;
    (
        c.record.expert,
        ordered::OrderedF32(blended),
        net_feedback,
        ordered::score_bits(c.record.scores.final_score),
        c.record.successes,
    )
}

/// Sort candidates for few-shot presentation (expert first).
pub fn sort_for_few_shot(mut candidates: Vec<RankedCandidate>) -> Vec<RankedCandidate> {
    candidates.sort_by_key(|b| std::cmp::Reverse(few_shot_rank_key(b)));
    candidates
}

/// Among candidates that meet the verified-reuse similarity floor and pass gates,
/// pick the best by similarity, then expert, then score / feedback / successes.
pub fn pick_verified_reuse<'a>(
    candidates: &'a [RankedCandidate],
    query: &AskQuery,
    policy: &Policy,
) -> Option<&'a RankedCandidate> {
    if !policy.verified_reuse_enabled {
        return None;
    }
    candidates
        .iter()
        .filter(|c| {
            c.similarity >= policy.verified_reuse_similarity_min
                && policy.is_reuse_score_eligible(c.record.expert, &c.record.scores)
                && reuse_gates_pass(query, &c.record)
        })
        .max_by_key(|c| reuse_pick_key(c))
}

fn reuse_pick_key(c: &RankedCandidate) -> (ordered::OrderedF32, bool, i64, u32, u32) {
    let net_feedback = i64::from(c.record.thumbs_up) - i64::from(c.record.thumbs_down);
    (
        ordered::OrderedF32(c.similarity),
        c.record.expert,
        net_feedback,
        ordered::score_bits(c.record.scores.final_score),
        c.record.successes,
    )
}

pub fn score_similarity(query_emb: &[f32], record: &AskRecord) -> Result<f32> {
    cosine_similarity(query_emb, &record.embedding)
}

mod ordered {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrderedF32(pub f32);

    impl Eq for OrderedF32 {}

    impl Ord for OrderedF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    impl PartialOrd for OrderedF32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    pub fn score_bits(score: Option<f32>) -> u32 {
        score.map(|s| (s * 1000.0) as u32).unwrap_or(0)
    }
}
