//! Persistence port. Phase 1: in-memory. Phase 2: Postgres.

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::policy::Policy;
use crate::types::{
    now_ms, AskRecord, DecayReason, KpiRef, RecordId, RepairRecord, Scope, TableRef, TrustOutcome,
    Vote,
};

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn upsert_ask(&self, record: AskRecord) -> Result<(RecordId, bool)>;
    async fn get_ask(&self, scope: &Scope, id: &RecordId) -> Result<Option<AskRecord>>;
    async fn get_ask_by_eval(&self, scope: &Scope, eval_id: &str) -> Result<Option<AskRecord>>;
    async fn list_eligible_asks(&self, scope: &Scope) -> Result<Vec<AskRecord>>;

    /// Whether any eligible ask exists for `scope` (for EmptyStore vs filtered miss).
    async fn has_eligible_asks(&self, scope: &Scope) -> Result<bool> {
        Ok(!self.list_eligible_asks(scope).await?.is_empty())
    }

    /// Eligible asks nearest to `embedding`, optionally restricted to tables/KPIs.
    /// Postgres does an exact scan within `scope` (no ANN) so tenant filters are correct.
    async fn list_eligible_asks_near(
        &self,
        scope: &Scope,
        _embedding: &[f32],
        _limit: usize,
        _tables: Option<&[TableRef]>,
        _kpis: Option<&[KpiRef]>,
    ) -> Result<Vec<AskRecord>> {
        self.list_eligible_asks(scope).await
    }

    /// Atomic: `uses += 1`.
    async fn bump_uses(&self, scope: &Scope, id: &RecordId) -> Result<()>;
    /// Atomic: `successes += 1`.
    async fn bump_successes(&self, scope: &Scope, id: &RecordId) -> Result<()>;
    async fn apply_decay(
        &self,
        scope: &Scope,
        id: &RecordId,
        reason: DecayReason,
        policy: &Policy,
    ) -> Result<()>;
    async fn apply_vote(
        &self,
        scope: &Scope,
        id: &RecordId,
        vote: Vote,
        policy: &Policy,
    ) -> Result<()>;

    async fn upsert_repair(&self, record: RepairRecord) -> Result<(RecordId, bool)>;
    async fn list_eligible_repairs(
        &self,
        scope: &Scope,
        error_class: &str,
    ) -> Result<Vec<RepairRecord>>;

    async fn invalidate_table(&self, scope: &Scope, table: &TableRef) -> Result<u64>;

    /// Atomically claim `(scope, eval_id)` and apply reinforce / failed-reuse decay.
    ///
    /// Returns `true` if this call applied the update. Returns `false` if an event already
    /// exists (first verdict wins — later rescored flips are ignored). Claim and counter
    /// bump must succeed or fail together (no orphan trust rows).
    async fn apply_trust_for_eval(
        &self,
        scope: &Scope,
        eval_id: &str,
        source_id: &RecordId,
        outcome: TrustOutcome,
        policy: &Policy,
    ) -> Result<bool>;
}

/// Process-local store for unit tests and Phase 1.
pub struct InMemoryStore {
    inner: std::sync::Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    asks: Vec<AskRecord>,
    repairs: Vec<RepairRecord>,
    /// `(scope, eval_id)` → outcome already applied.
    trust_events: std::collections::HashMap<(String, String), TrustOutcome>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self { inner: std::sync::Mutex::new(Inner::default()) }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

fn lock_err() -> Error {
    Error::Store("in-memory store lock poisoned".into())
}

fn find_ask_mut<'a>(
    asks: &'a mut [AskRecord],
    scope: &Scope,
    id: &RecordId,
) -> Result<&'a mut AskRecord> {
    asks.iter_mut()
        .find(|a| &a.scope == scope && &a.id == id)
        .ok_or_else(|| Error::NotFound(format!("ask {}", id.as_uuid())))
}

fn same_sorted_strs(a: &[String], b: &[String]) -> bool {
    a == b
}

fn sorted_tables(tables: &[TableRef]) -> Vec<String> {
    let mut v: Vec<String> = tables.iter().map(|t| t.as_str().to_string()).collect();
    v.sort_unstable();
    v
}

fn sorted_kpis(kpis: &[KpiRef]) -> Vec<String> {
    let mut v: Vec<String> = kpis.iter().map(|k| k.as_str().to_string()).collect();
    v.sort_unstable();
    v
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn upsert_ask(&self, record: AskRecord) -> Result<(RecordId, bool)> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        if let Some(existing) =
            g.asks.iter_mut().find(|a| a.scope == record.scope && a.eval_id == record.eval_id)
        {
            let id = existing.id.clone();
            existing.conversation_id = record.conversation_id;
            existing.question = record.question;
            existing.embedding = record.embedding;
            existing.tables = record.tables;
            existing.kpis = record.kpis;
            existing.ask_signature = record.ask_signature;
            existing.schema_fingerprints = record.schema_fingerprints;
            existing.sql = record.sql;
            existing.sql_has_absolute_dates = record.sql_has_absolute_dates;
            existing.pandas = record.pandas;
            existing.scores = record.scores;
            existing.path_taken = record.path_taken;
            existing.expert = record.expert;
            existing.updated_at = now_ms();
            return Ok((id, false));
        }
        let id = record.id.clone();
        g.asks.push(record);
        Ok((id, true))
    }

    async fn get_ask(&self, scope: &Scope, id: &RecordId) -> Result<Option<AskRecord>> {
        let g = self.inner.lock().map_err(|_| lock_err())?;
        Ok(g.asks.iter().find(|a| &a.scope == scope && &a.id == id).cloned())
    }

    async fn get_ask_by_eval(&self, scope: &Scope, eval_id: &str) -> Result<Option<AskRecord>> {
        let g = self.inner.lock().map_err(|_| lock_err())?;
        Ok(g.asks.iter().find(|a| &a.scope == scope && a.eval_id == eval_id).cloned())
    }

    async fn list_eligible_asks(&self, scope: &Scope) -> Result<Vec<AskRecord>> {
        let g = self.inner.lock().map_err(|_| lock_err())?;
        Ok(g.asks.iter().filter(|a| &a.scope == scope && a.eligible).cloned().collect())
    }

    async fn list_eligible_asks_near(
        &self,
        scope: &Scope,
        _embedding: &[f32],
        limit: usize,
        tables: Option<&[TableRef]>,
        kpis: Option<&[KpiRef]>,
    ) -> Result<Vec<AskRecord>> {
        let mut rows = self.list_eligible_asks(scope).await?;
        if let Some(t) = tables {
            let want = sorted_tables(t);
            rows.retain(|r| same_sorted_strs(&sorted_tables(&r.tables), &want));
        }
        if let Some(k) = kpis {
            let want = sorted_kpis(k);
            rows.retain(|r| same_sorted_strs(&sorted_kpis(&r.kpis), &want));
        }
        rows.truncate(limit.max(1));
        Ok(rows)
    }

    async fn bump_uses(&self, scope: &Scope, id: &RecordId) -> Result<()> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        let ask = find_ask_mut(&mut g.asks, scope, id)?;
        ask.uses = ask.uses.saturating_add(1);
        ask.updated_at = now_ms();
        Ok(())
    }

    async fn bump_successes(&self, scope: &Scope, id: &RecordId) -> Result<()> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        let ask = find_ask_mut(&mut g.asks, scope, id)?;
        ask.successes = ask.successes.saturating_add(1);
        ask.updated_at = now_ms();
        Ok(())
    }

    async fn apply_decay(
        &self,
        scope: &Scope,
        id: &RecordId,
        reason: DecayReason,
        policy: &Policy,
    ) -> Result<()> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        let ask = find_ask_mut(&mut g.asks, scope, id)?;
        match reason {
            DecayReason::FailedReuse => {
                ask.failures = ask.failures.saturating_add(1);
                ask.strikes = ask.strikes.saturating_add(1);
                if policy.should_ineligible_after_strikes(ask.strikes) {
                    ask.eligible = false;
                    ask.ineligible_reason = Some("failed_reuse".into());
                }
            }
            DecayReason::NegativeFeedback => {
                ask.thumbs_down = ask.thumbs_down.saturating_add(1);
                if policy.should_ineligible_after_thumbs_down(ask.thumbs_down) {
                    ask.eligible = false;
                    ask.ineligible_reason = Some("negative_feedback".into());
                }
            }
            DecayReason::TableInvalidated => {
                ask.eligible = false;
                ask.ineligible_reason = Some("table_invalidated".into());
            }
            DecayReason::Manual => {
                ask.eligible = false;
                ask.ineligible_reason = Some("manual".into());
            }
        }
        ask.updated_at = now_ms();
        Ok(())
    }

    async fn apply_vote(
        &self,
        scope: &Scope,
        id: &RecordId,
        vote: Vote,
        policy: &Policy,
    ) -> Result<()> {
        match vote {
            Vote::Down => self.apply_decay(scope, id, DecayReason::NegativeFeedback, policy).await,
            Vote::Up => {
                let mut g = self.inner.lock().map_err(|_| lock_err())?;
                let ask = find_ask_mut(&mut g.asks, scope, id)?;
                ask.thumbs_up = ask.thumbs_up.saturating_add(1);
                ask.updated_at = now_ms();
                Ok(())
            }
        }
    }

    async fn upsert_repair(&self, record: RepairRecord) -> Result<(RecordId, bool)> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        if let Some(existing) = g.repairs.iter_mut().find(|r| {
            r.scope == record.scope
                && r.error_class == record.error_class
                && r.schema_fingerprint == record.schema_fingerprint
                && same_sorted_strs(&sorted_tables(&r.table_ids), &sorted_tables(&record.table_ids))
        }) {
            let id = existing.id.clone();
            let fixed_changed = existing.fixed_sql_fragment != record.fixed_sql_fragment;
            existing.failed_sql_fragment = record.failed_sql_fragment;
            existing.fixed_sql_fragment = record.fixed_sql_fragment;
            existing.times_seen = existing.times_seen.saturating_add(1);
            if fixed_changed {
                existing.times_fixed = existing.times_fixed.saturating_add(1);
            }
            existing.updated_at = now_ms();
            return Ok((id, false));
        }
        let id = record.id.clone();
        g.repairs.push(record);
        Ok((id, true))
    }

    async fn list_eligible_repairs(
        &self,
        scope: &Scope,
        error_class: &str,
    ) -> Result<Vec<RepairRecord>> {
        let g = self.inner.lock().map_err(|_| lock_err())?;
        Ok(g.repairs
            .iter()
            .filter(|r| &r.scope == scope && r.eligible && r.error_class == error_class)
            .cloned()
            .collect())
    }

    async fn invalidate_table(&self, scope: &Scope, table: &TableRef) -> Result<u64> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        let mut n = 0_u64;
        let ts = now_ms();
        for ask in g.asks.iter_mut() {
            if &ask.scope == scope && ask.tables.iter().any(|t| t == table) && ask.eligible {
                ask.eligible = false;
                ask.ineligible_reason = Some("table_invalidated".into());
                ask.updated_at = ts;
                n = n.saturating_add(1);
            }
        }
        for repair in g.repairs.iter_mut() {
            if &repair.scope == scope
                && repair.table_ids.iter().any(|t| t == table)
                && repair.eligible
            {
                repair.eligible = false;
                repair.updated_at = ts;
                n = n.saturating_add(1);
            }
        }
        Ok(n)
    }

    async fn apply_trust_for_eval(
        &self,
        scope: &Scope,
        eval_id: &str,
        source_id: &RecordId,
        outcome: TrustOutcome,
        policy: &Policy,
    ) -> Result<bool> {
        let mut g = self.inner.lock().map_err(|_| lock_err())?;
        let key = (scope.as_str().to_string(), eval_id.to_string());
        if g.trust_events.contains_key(&key) {
            return Ok(false);
        }
        // Apply counters first; only record the event if the ask exists.
        let ask = find_ask_mut(&mut g.asks, scope, source_id)?;
        match outcome {
            TrustOutcome::Reinforce => {
                ask.uses = ask.uses.saturating_add(1);
                ask.successes = ask.successes.saturating_add(1);
            }
            TrustOutcome::DecayFailedReuse => {
                ask.failures = ask.failures.saturating_add(1);
                ask.strikes = ask.strikes.saturating_add(1);
                if policy.should_ineligible_after_strikes(ask.strikes) {
                    ask.eligible = false;
                    ask.ineligible_reason = Some("failed_reuse".into());
                }
            }
        }
        ask.updated_at = now_ms();
        g.trust_events.insert(key, outcome);
        Ok(true)
    }
}
