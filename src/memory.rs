//! Public `Memory` facade: retrieve, record, decay, repairs.

use std::sync::Arc;

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::policy::{
    hard_filter, pick_verified_reuse, reuse_gate_reject, score_similarity, sort_for_few_shot,
    table_overlap_ratio, Policy, RankedCandidate,
};
use crate::sql_inspect::{filter_tables_to_sql, normalized_sql_hash, sql_has_absolute_dates};
use crate::store::MemoryStore;
use crate::trace::{CandidateTrace, RejectReason, RetrieveTrace};
use crate::types::{
    now_ms, AskQuery, AskRecord, DecayReason, ErrorQuery, GradedAsk, MissReason, Outcome,
    PathTaken, RecordAskResult, RecordId, Recorded, RepairInput, RepairRecord, Scope, TableRef,
    TrustOutcome, Vote,
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
        self.retrieve_traced(scope, ask).await.map(|(outcome, _)| outcome)
    }

    /// Same decision as [`Self::retrieve`], plus a per-candidate [`RetrieveTrace`] (4y.1).
    ///
    /// The trace is diagnostic: it never changes which outcome is returned.
    pub async fn retrieve_traced(
        &self,
        scope: &Scope,
        ask: &AskQuery,
    ) -> Result<(Outcome, RetrieveTrace)> {
        let mut trace = RetrieveTrace::new(ask, self.policy.candidate_fetch_limit);
        if ask.question.trim().is_empty() {
            return Err(Error::Invalid("question must not be empty".into()));
        }
        let embedding = match self.embedder.embed(&ask.question).await {
            Ok(v) => v,
            Err(_) => {
                trace.miss(MissReason::EmbedderFailed);
                return Ok((Outcome::Miss { reason: MissReason::EmbedderFailed }, trace));
            }
        };

        if !self.store.has_eligible_asks(scope).await? {
            trace.miss(MissReason::EmptyStore);
            return Ok((Outcome::Miss { reason: MissReason::EmptyStore }, trace));
        }
        trace.store_has_eligible = true;

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
        trace.store_rows = candidates.len();
        if candidates.is_empty() {
            // Store-side table/KPI pushdown removed every row that was eligible in scope.
            trace.miss(MissReason::HardFilter);
            return Ok((Outcome::Miss { reason: MissReason::HardFilter }, trace));
        }

        let mut ranked = Vec::new();
        for record in candidates {
            let mut candidate = CandidateTrace::new(ask, &record);
            // Store already filtered tables/KPIs; keep hard_filter as a safety net.
            if !hard_filter(ask, &record) {
                let reason = if record.eligible {
                    RejectReason::TableOrKpiSet
                } else {
                    RejectReason::Ineligible
                };
                candidate.few_shot_reject = Some(reason);
                candidate.reuse_reject = Some(reason);
                trace.candidates.push(candidate);
                continue;
            }
            let Ok(similarity) = score_similarity(&embedding, &record) else {
                candidate.few_shot_reject = Some(RejectReason::ScoreFailed);
                candidate.reuse_reject = Some(RejectReason::ScoreFailed);
                trace.candidates.push(candidate);
                continue;
            };
            candidate.similarity = Some(similarity);
            if similarity < self.policy.few_shot_similarity_min {
                candidate.few_shot_reject = Some(RejectReason::BelowFewShotSimilarity);
                candidate.reuse_reject = Some(RejectReason::BelowFewShotSimilarity);
                trace.candidates.push(candidate);
                continue;
            }
            candidate.reuse_reject = if !self.policy.verified_reuse_enabled {
                Some(RejectReason::ReuseDisabled)
            } else if similarity < self.policy.verified_reuse_similarity_min {
                Some(RejectReason::BelowReuseSimilarity)
            } else if !self.policy.is_reuse_score_eligible(record.expert, &record.scores) {
                Some(RejectReason::BelowReuseScore)
            } else {
                reuse_gate_reject(ask, &record)
            };
            let table_overlap = table_overlap_ratio(&ask.tables, &record.tables);
            trace.candidates.push(candidate);
            ranked.push(RankedCandidate { record, similarity, table_overlap });
        }

        if ranked.is_empty() {
            trace.miss(MissReason::BelowSimilarity);
            return Ok((Outcome::Miss { reason: MissReason::BelowSimilarity }, trace));
        }

        if let Some(reuse) = pick_verified_reuse(&ranked, ask, &self.policy) {
            let winner = reuse.record.id.clone();
            let outcome = Outcome::VerifiedReuse { record: Box::new(reuse.record.clone()) };
            for candidate in &mut trace.candidates {
                if candidate.id == winner {
                    candidate.selected = Some(PathTaken::VerifiedReuse);
                } else if candidate.reuse_reject.is_none() {
                    candidate.reuse_reject = Some(RejectReason::LowerRankedReuse);
                }
                if candidate.few_shot_reject.is_none() {
                    candidate.few_shot_reject = Some(RejectReason::ReusedInstead);
                }
            }
            trace.path = Some(PathTaken::VerifiedReuse);
            return Ok((outcome, trace));
        }

        let ranked = sort_for_few_shot(ranked);
        let limit = self.policy.few_shot_limit.max(1);
        let examples: Vec<AskRecord> = ranked
            .into_iter()
            .enumerate()
            .filter_map(|(rank, c)| {
                if rank < limit {
                    Some(c.record)
                } else {
                    // Mark overflow rows; the record itself is dropped here.
                    if let Some(t) = trace.candidates.iter_mut().find(|t| t.id == c.record.id) {
                        t.few_shot_reject = Some(RejectReason::TruncatedFromFewShot);
                    }
                    None
                }
            })
            .collect();
        if examples.is_empty() {
            trace.miss(MissReason::NoCandidates);
            return Ok((Outcome::Miss { reason: MissReason::NoCandidates }, trace));
        }
        for example in &examples {
            if let Some(t) = trace.candidates.iter_mut().find(|t| t.id == example.id) {
                t.selected = Some(PathTaken::FewShot);
            }
        }
        trace.path = Some(PathTaken::FewShot);
        Ok((Outcome::FewShot { examples }, trace))
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
            // Re-grade below threshold must not leave an older good row eligible for reuse.
            return self.retract_ask_by_eval(scope, &ask.eval_id).await.map(|id| match id {
                Some(id) => RecordAskResult::Retracted { id },
                None => RecordAskResult::SkippedBelowThreshold,
            });
        }
        // Phase 4y.10: replaying an existing verified memory must not create a duplicate row.
        if ask.path_taken == PathTaken::VerifiedReuse {
            return Ok(RecordAskResult::SkippedVerifiedReuse);
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
        let mut ask_signature = ask.ask_signature;
        ask_signature.normalize();
        // Phase 4y.4: key stored rows by tables actually referenced in the final SQL,
        // and keep only schema fingerprints for those tables.
        let tables = filter_tables_to_sql(&ask.tables, &ask.sql);
        let schema_fingerprints: Vec<_> = ask
            .schema_fingerprints
            .into_iter()
            .filter(|(t, _)| tables.iter().any(|kept| kept == t))
            .collect();
        // Phase 4y.10: deduplicate by normalized SQL hash + tables + signature beyond eval_id upsert.
        // Never discard an expert row over a non-expert row, or a reuse-eligible row over a few-shot-only row.
        let new_sql_hash = normalized_sql_hash(&ask.sql);
        let new_reuse_eligible = self.policy.is_reuse_score_eligible(ask.expert, &ask.scores);
        let near = self
            .store
            .list_eligible_asks_near(scope, &embedding, 16, Some(&tables), Some(&ask.kpis))
            .await?;
        if let Some(dup) = near.iter().find(|existing| {
            let existing_reuse_eligible =
                self.policy.is_reuse_score_eligible(existing.expert, &existing.scores);
            existing.eval_id != ask.eval_id
                && normalized_sql_hash(&existing.sql) == new_sql_hash
                && existing.tables == tables
                && ask_signature.matches(&existing.ask_signature)
                && (!ask.expert || existing.expert)
                && (!new_reuse_eligible || existing_reuse_eligible)
        }) {
            return Ok(RecordAskResult::Deduplicated { id: dup.id.clone() });
        }
        let record = AskRecord {
            id: RecordId::new(),
            scope: scope.clone(),
            eval_id: ask.eval_id,
            conversation_id: ask.conversation_id,
            question: ask.question,
            embedding,
            tables,
            kpis: ask.kpis,
            ask_signature,
            schema_fingerprints,
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
            .apply_trust_for_eval(scope, eval_id, source_id, TrustOutcome::Reinforce, &self.policy)
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

    /// Mark the ask for `(scope, eval_id)` ineligible when a re-grade falls below store threshold.
    ///
    /// Returns the record id when a row was found and updated; `None` if none existed.
    pub async fn retract_ask_by_eval(
        &self,
        scope: &Scope,
        eval_id: &str,
    ) -> Result<Option<RecordId>> {
        let Some(existing) = self.store.get_ask_by_eval(scope, eval_id).await? else {
            return Ok(None);
        };
        if !existing.eligible
            && existing.ineligible_reason.as_deref() == Some("regraded_below_threshold")
        {
            return Ok(Some(existing.id));
        }
        self.store
            .apply_decay(scope, &existing.id, DecayReason::RegradedBelowThreshold, &self.policy)
            .await?;
        Ok(Some(existing.id))
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
