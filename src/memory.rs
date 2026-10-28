//! Public `Memory` facade: retrieve, record, decay, repairs.

use std::sync::Arc;

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::policy::{
    hard_filter, pick_verified_reuse, score_similarity, sort_for_few_shot, Policy, RankedCandidate,
};
use crate::sql_inspect::sql_has_absolute_dates;
use crate::store::MemoryStore;
use crate::types::{
    now_ms, AskQuery, AskRecord, DecayReason, ErrorQuery, GradedAsk, MissReason, Outcome,
    RecordAskResult, RecordId, Recorded, RepairInput, RepairRecord, Scope, TableRef, TrustOutcome,
    Vote,
};

pub struct Memory<S, E> {
    store: Arc<S>,
    embedder: Arc<E>,
    policy: Policy,
}

impl<S, E> Memory<S, E>
where
    S: MemoryStore,
    E: Embedder,
{
    pub fn new(store: Arc<S>, embedder: Arc<E>, policy: Policy) -> Self {
        Self { store, embedder, policy }
    }

    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    pub async fn retrieve(&self, scope: &Scope, ask: &AskQuery) -> Result<Outcome> {
        if ask.question.trim().is_empty() {
            return Err(Error::Invalid("question must not be empty".into()));
        }
        let embedding = match self.embedder.embed(&ask.question).await {
            Ok(v) => v,
            Err(_) => {
                return Ok(Outcome::Miss { reason: MissReason::EmbedderFailed });
            }
        };

        if !self.store.has_eligible_asks(scope).await? {
            return Ok(Outcome::Miss { reason: MissReason::EmptyStore });
        }

        // Push table/KPI filters into the store so LIMIT isn't wasted on other tables.
        let candidates = self
            .store
            .list_eligible_asks_near(
                scope,
                &embedding,
                self.policy.candidate_fetch_limit,
                Some(&ask.tables),
                Some(&ask.kpis),
            )
            .await?;
        if candidates.is_empty() {
            return Ok(Outcome::Miss { reason: MissReason::HardFilter });
        }

        let mut ranked = Vec::new();
        for record in candidates {
            // Store already filtered tables/KPIs; keep hard_filter as a safety net.
            if !hard_filter(ask, &record) {
                continue;
            }
            let Ok(similarity) = score_similarity(&embedding, &record) else {
                continue;
            };
            if similarity < self.policy.few_shot_similarity_min {
                continue;
            }
            ranked.push(RankedCandidate { record, similarity });
        }

        if ranked.is_empty() {
            return Ok(Outcome::Miss { reason: MissReason::BelowSimilarity });
        }

        if let Some(reuse) = pick_verified_reuse(&ranked, ask, &self.policy) {
            return Ok(Outcome::VerifiedReuse { record: Box::new(reuse.record.clone()) });
        }

        let ranked = sort_for_few_shot(ranked);
        let limit = self.policy.few_shot_limit.max(1);
        let examples: Vec<AskRecord> = ranked.into_iter().take(limit).map(|c| c.record).collect();
        if examples.is_empty() {
            return Ok(Outcome::Miss { reason: MissReason::NoCandidates });
        }
        Ok(Outcome::FewShot { examples })
    }

    pub async fn repairs(&self, scope: &Scope, err: &ErrorQuery) -> Result<Vec<RepairRecord>> {
        let rows = self.store.list_eligible_repairs(scope, &err.error_class).await?;
        Ok(rows
            .into_iter()
            .filter(|r| {
                r.schema_fingerprint == err.schema_fingerprint
                    && same_table_multiset(&r.table_ids, &err.table_ids)
            })
            .collect())
    }

    pub async fn record_ask(&self, scope: &Scope, ask: GradedAsk) -> Result<RecordAskResult> {
        if !self.policy.is_storable(ask.scores.final_score, ask.scores.sql_score) {
            return Ok(RecordAskResult::SkippedBelowThreshold);
        }

        let sql_has_absolute_dates = sql_has_absolute_dates(&ask.sql);
        // Re-embed only when the question changed; trust is preserved by store upsert.
        let embedding =
            if let Some(existing) = self.store.get_ask_by_eval(scope, &ask.eval_id).await? {
                if existing.question == ask.question {
                    existing.embedding
                } else {
                    self.embedder.embed(&ask.question).await?
                }
            } else {
                self.embedder.embed(&ask.question).await?
            };
        let ts = now_ms();
        let record = AskRecord {
            id: RecordId::new(),
            scope: scope.clone(),
            eval_id: ask.eval_id,
            conversation_id: ask.conversation_id,
            question: ask.question,
            embedding,
            tables: ask.tables,
            kpis: ask.kpis,
            ask_signature: ask.ask_signature,
            schema_fingerprints: ask.schema_fingerprints,
            sql: ask.sql,
            sql_has_absolute_dates,
            pandas: ask.pandas,
            scores: ask.scores,
            path_taken: ask.path_taken,
            expert: ask.expert,
            uses: 0,
            successes: 0,
            failures: 0,
            thumbs_up: 0,
            thumbs_down: 0,
            strikes: 0,
            eligible: true,
            ineligible_reason: None,
            created_at: ts,
            updated_at: ts,
        };
        let (id, created) = self.store.upsert_ask(record).await?;
        Ok(RecordAskResult::Stored(Recorded { id, created }))
    }

    pub async fn record_repair(&self, scope: &Scope, repair: RepairInput) -> Result<Recorded> {
        let ts = now_ms();
        let record = RepairRecord {
            id: RecordId::new(),
            scope: scope.clone(),
            table_ids: repair.table_ids,
            error_class: repair.error_class,
            failed_sql_fragment: repair.failed_sql_fragment,
            fixed_sql_fragment: repair.fixed_sql_fragment,
            schema_fingerprint: repair.schema_fingerprint,
            times_seen: 1,
            times_fixed: 1,
            eligible: true,
            created_at: ts,
            updated_at: ts,
        };
        let (id, created) = self.store.upsert_repair(record).await?;
        Ok(Recorded { id, created })
    }

    /// Record a successful verified reuse: bump `uses` and `successes`.
    ///
    /// Failed reuse should call [`Self::decay`] with [`DecayReason::FailedReuse`] instead;
    /// `uses` counts successful uses only.
    ///
    /// Prefer [`Self::reinforce_for_eval`] when the bump is tied to a graded eval (idempotent).
    pub async fn reinforce(&self, scope: &Scope, id: &RecordId) -> Result<()> {
        self.store.bump_uses(scope, id).await?;
        self.store.bump_successes(scope, id).await
    }

    pub async fn decay(&self, scope: &Scope, id: &RecordId, reason: DecayReason) -> Result<()> {
        self.store.apply_decay(scope, id, reason, &self.policy).await
    }

    /// Idempotent reinforce keyed by `(scope, eval_id)`. Returns whether counters were bumped.
    ///
    /// **First verdict wins:** a later rescore that would flip reinforce↔decay is ignored once
    /// an event exists for this eval. Claim and counter updates are atomic in the store.
    pub async fn reinforce_for_eval(
        &self,
        scope: &Scope,
        source_id: &RecordId,
        eval_id: &str,
    ) -> Result<bool> {
        self.store
            .apply_trust_for_eval(
                scope,
                eval_id,
                source_id,
                TrustOutcome::Reinforce,
                &self.policy,
            )
            .await
    }

    /// Idempotent failed-reuse decay keyed by `(scope, eval_id)`.
    ///
    /// Same first-verdict-wins rule as [`Self::reinforce_for_eval`].
    pub async fn decay_failed_reuse_for_eval(
        &self,
        scope: &Scope,
        source_id: &RecordId,
        eval_id: &str,
    ) -> Result<bool> {
        self.store
            .apply_trust_for_eval(
                scope,
                eval_id,
                source_id,
                TrustOutcome::DecayFailedReuse,
                &self.policy,
            )
            .await
    }

    pub async fn feedback(&self, scope: &Scope, id: &RecordId, vote: Vote) -> Result<()> {
        self.store.apply_vote(scope, id, vote, &self.policy).await
    }

    pub async fn invalidate_table(&self, scope: &Scope, table: &TableRef) -> Result<u64> {
        self.store.invalidate_table(scope, table).await
    }
}

fn same_table_multiset(a: &[TableRef], b: &[TableRef]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut left: Vec<&str> = a.iter().map(TableRef::as_str).collect();
    let mut right: Vec<&str> = b.iter().map(TableRef::as_str).collect();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}
