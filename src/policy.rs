//! Matching, reuse gates, ranking, and trust thresholds.

use crate::embedder::cosine_similarity;
use crate::error::Result;
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
}

/// Hard filters: eligible; same table set; same KPI set (exact equality, order-insensitive).
pub fn hard_filter(query: &AskQuery, record: &AskRecord) -> bool {
    if !record.eligible {
        return false;
    }
    same_table_set(&query.tables, &record.tables) && same_kpi_set(&query.kpis, &record.kpis)
}

fn same_table_set(a: &[TableRef], b: &[TableRef]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut left: Vec<&str> = a.iter().map(TableRef::as_str).collect();
    let mut right: Vec<&str> = b.iter().map(TableRef::as_str).collect();
    left.sort_unstable();
    right.sort_unstable();
    left == right
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
    if record.sql_has_absolute_dates {
        return false;
    }
    if !query.signature.is_reuse_ready() || !record.ask_signature.is_reuse_ready() {
        return false;
    }
    // Empty tables / fingerprints must not verify — signature alone is not enough.
    if query.tables.is_empty() || record.tables.is_empty() {
        return false;
    }
    if query.fingerprints.is_empty() || record.schema_fingerprints.is_empty() {
        return false;
    }
    if !fingerprints_cover_tables(&query.tables, &query.fingerprints)
        || !fingerprints_cover_tables(&record.tables, &record.schema_fingerprints)
    {
        return false;
    }
    // Lifetime failed-reuse bias: more failures than successes blocks replay.
    if record.strikes > 0 && record.failures > record.successes {
        return false;
    }
    signatures_match(&query.signature, &record.ask_signature)
        && fingerprints_match(&query.fingerprints, &record.schema_fingerprints)
}

/// Every table in `tables` has a fingerprint entry.
pub fn fingerprints_cover_tables(
    tables: &[TableRef],
    fingerprints: &[(TableRef, SchemaFingerprint)],
) -> bool {
    !tables.is_empty()
        && tables.iter().all(|t| fingerprints.iter().any(|(ft, _)| ft == t))
}

fn signatures_match(a: &AskSignature, b: &AskSignature) -> bool {
    a.matches(b)
}

fn fingerprints_match(
    query: &[(TableRef, SchemaFingerprint)],
    stored: &[(TableRef, SchemaFingerprint)],
) -> bool {
    // Empty-empty must not count as a match (gated earlier, but keep the invariant here).
    if query.is_empty() || stored.is_empty() {
        return false;
    }
    if query.len() != stored.len() {
        return false;
    }
    for (qt, qf) in query {
        let Some((_, sf)) = stored.iter().find(|(t, _)| t == qt) else {
            return false;
        };
        if sf != qf {
            return false;
        }
    }
    true
}

/// Few-shot ranking: expert first, then similarity, net feedback, score, successes.
pub fn few_shot_rank_key(c: &RankedCandidate) -> (bool, ordered::OrderedF32, i64, u32, u32) {
    let net_feedback = i64::from(c.record.thumbs_up) - i64::from(c.record.thumbs_down);
    (
        c.record.expert,
        ordered::OrderedF32(c.similarity),
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
