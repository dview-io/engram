//! Postgres + pgvector [`MemoryStore`]. Host injects [`sqlx::PgPool`].
//!
//! Exact nearest-neighbor within `scope` (no HNSW) so tenant filters are correct.
//! `tables` / `kpis` / `table_ids` are JSONB (host-reconcile friendly).

use async_trait::async_trait;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::policy::Policy;
use crate::schema::{self, ASK_TABLE, REPAIR_TABLE, TRUST_TABLE};
use crate::store::MemoryStore;
use crate::types::{
    now_ms, AskRecord, AskSignature, DecayReason, KpiRef, PathTaken, RecordId, RepairRecord,
    SchemaFingerprint, Scope, Scores, TableRef, TrustOutcome, Vote,
};

/// Format an embedding for binding as `$n::vector`.
pub fn vector_literal(embedding: &[f32]) -> Result<String> {
    for v in embedding {
        if !v.is_finite() {
            return Err(Error::Invalid("embedding contains NaN or Inf".into()));
        }
    }
    let mut out = String::from("[");
    for (i, v) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push(']');
    Ok(out)
}

fn parse_vector_text(text: &str) -> Result<Vec<f32>> {
    let trimmed = text.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<f32>()
                .map_err(|e| Error::Store(format!("bad vector element `{p}`: {e}")))
        })
        .collect()
}

fn map_db(err: sqlx::Error) -> Error {
    // Prefer SQLSTATE over message text (locale / version stable).
    // 42704 = undefined_object (missing type/extension), 42883 = undefined_function.
    if let sqlx::Error::Database(db) = &err {
        match db.code().as_deref() {
            Some("42704") | Some("42883") => {
                let msg = err.to_string();
                if msg.to_ascii_lowercase().contains("vector") {
                    return Error::VectorUnavailable(msg);
                }
            }
            _ => {}
        }
    }
    Error::Store(err.to_string())
}

fn as_u32(v: i32, field: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Store(format!("{field} out of u32 range: {v}")))
}

fn q(schema: &str, table: &str) -> String {
    schema::qualified(schema, table)
}

fn sorted_table_strings(tables: &[TableRef]) -> Vec<String> {
    let mut v: Vec<String> = tables.iter().map(|t| t.as_str().to_string()).collect();
    v.sort_unstable();
    v
}

fn sorted_kpi_strings(kpis: &[KpiRef]) -> Vec<String> {
    let mut v: Vec<String> = kpis.iter().map(|k| k.as_str().to_string()).collect();
    v.sort_unstable();
    v
}

fn strings_json(values: &[String]) -> Result<serde_json::Value> {
    Ok(serde_json::Value::Array(
        values.iter().map(|s| serde_json::Value::String(s.clone())).collect(),
    ))
}

fn strings_from_json(value: serde_json::Value) -> Result<Vec<String>> {
    let Some(arr) = value.as_array() else {
        return Err(Error::Store("expected JSON array of strings".into()));
    };
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::Store("expected string in JSON array".into()))
        })
        .collect()
}

fn fingerprints_json(fps: &[(TableRef, SchemaFingerprint)]) -> Result<serde_json::Value> {
    Ok(serde_json::Value::Array(
        fps.iter().map(|(t, f)| serde_json::json!([t.as_str(), f.as_str()])).collect(),
    ))
}

fn fingerprints_from_json(value: serde_json::Value) -> Result<Vec<(TableRef, SchemaFingerprint)>> {
    let Some(arr) = value.as_array() else {
        return Err(Error::Store("schema_fingerprints must be a JSON array".into()));
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let pair = item
            .as_array()
            .ok_or_else(|| Error::Store("fingerprint entry must be [table, fp]".into()))?;
        let table = pair
            .first()
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Store("fingerprint table missing".into()))?;
        let fp = pair
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Store("fingerprint value missing".into()))?;
        out.push((TableRef::new(table), SchemaFingerprint::new(fp)));
    }
    Ok(out)
}

/// Postgres-backed store. Tables live in `schema` (typically the host schema).
pub struct PostgresStore {
    pool: PgPool,
    schema: String,
    embedding_dim: usize,
}

impl PostgresStore {
    pub fn new(pool: PgPool, schema: impl Into<String>, embedding_dim: usize) -> Result<Self> {
        let schema = schema.into();
        schema::validate_ident(&schema)?;
        if embedding_dim == 0 {
            return Err(Error::Invalid("embedding_dim must be > 0".into()));
        }
        Ok(Self { pool, schema, embedding_dim })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Verify live column width matches `embedding_dim` (call after reconcile / apply).
    pub async fn check_embedding_dim(&self) -> Result<()> {
        let sql = "SELECT format_type(a.atttypid, a.atttypmod) AS ty
             FROM pg_attribute a
             JOIN pg_class c ON c.oid = a.attrelid
             JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = 'embedding' AND NOT a.attisdropped";
        let ty: Option<String> = sqlx::query_scalar(sql)
            .bind(&self.schema)
            .bind(ASK_TABLE)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_db)?;
        let Some(ty) = ty else {
            return Err(Error::Store(format!(
                "{}.{} missing embedding column",
                self.schema, ASK_TABLE
            )));
        };
        let expected = format!("vector({})", self.embedding_dim);
        if ty != expected {
            return Err(Error::Invalid(format!(
                "embedding dimension mismatch: column is {ty}, store expects {expected}"
            )));
        }
        Ok(())
    }

    fn ask_table(&self) -> String {
        q(&self.schema, ASK_TABLE)
    }

    fn repair_table(&self) -> String {
        q(&self.schema, REPAIR_TABLE)
    }

    fn trust_table(&self) -> String {
        q(&self.schema, TRUST_TABLE)
    }

    fn ensure_dim(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::Invalid(format!(
                "embedding length {} != configured {}",
                embedding.len(),
                self.embedding_dim
            )));
        }
        Ok(())
    }

    fn row_to_ask(&self, row: &sqlx::postgres::PgRow) -> Result<AskRecord> {
        let id: Uuid = row.try_get("id").map_err(map_db)?;
        let path: String = row.try_get("path_taken").map_err(map_db)?;
        let path_taken = PathTaken::parse(&path)
            .ok_or_else(|| Error::Store(format!("unknown path_taken `{path}`")))?;
        let emb_text: String = row.try_get("embedding_text").map_err(map_db)?;
        let tables = strings_from_json(row.try_get("tables").map_err(map_db)?)?;
        let kpis = strings_from_json(row.try_get("kpis").map_err(map_db)?)?;
        let sig: serde_json::Value = row.try_get("ask_signature").map_err(map_db)?;
        let fps: serde_json::Value = row.try_get("schema_fingerprints").map_err(map_db)?;
        let scores: serde_json::Value = row.try_get("scores").map_err(map_db)?;
        let ask_signature: AskSignature =
            serde_json::from_value(sig).map_err(|e| Error::Store(e.to_string()))?;
        let scores: Scores =
            serde_json::from_value(scores).map_err(|e| Error::Store(e.to_string()))?;

        Ok(AskRecord {
            id: RecordId::from_uuid(id),
            scope: Scope::new(row.try_get::<String, _>("scope").map_err(map_db)?),
            eval_id: row.try_get("eval_id").map_err(map_db)?,
            conversation_id: row.try_get("conversation_id").map_err(map_db)?,
            question: row.try_get("question").map_err(map_db)?,
            embedding: parse_vector_text(&emb_text)?,
            tables: tables.into_iter().map(TableRef::new).collect(),
            kpis: kpis.into_iter().map(KpiRef::new).collect(),
            ask_signature,
            schema_fingerprints: fingerprints_from_json(fps)?,
            sql: row.try_get("sql").map_err(map_db)?,
            sql_has_absolute_dates: row.try_get("sql_has_absolute_dates").map_err(map_db)?,
            pandas: row.try_get("pandas").map_err(map_db)?,
            scores,
            path_taken,
            expert: row.try_get("expert").map_err(map_db)?,
            uses: as_u32(row.try_get("uses").map_err(map_db)?, "uses")?,
            successes: as_u32(row.try_get("successes").map_err(map_db)?, "successes")?,
            failures: as_u32(row.try_get("failures").map_err(map_db)?, "failures")?,
            thumbs_up: as_u32(row.try_get("thumbs_up").map_err(map_db)?, "thumbs_up")?,
            thumbs_down: as_u32(row.try_get("thumbs_down").map_err(map_db)?, "thumbs_down")?,
            strikes: as_u32(row.try_get("strikes").map_err(map_db)?, "strikes")?,
            eligible: row.try_get("eligible").map_err(map_db)?,
            ineligible_reason: row.try_get("ineligible_reason").map_err(map_db)?,
            created_at: row.try_get("created_at").map_err(map_db)?,
            updated_at: row.try_get("updated_at").map_err(map_db)?,
        })
    }

    fn ask_select_cols() -> &'static str {
        "id, scope, eval_id, conversation_id, question, embedding::text AS embedding_text, \
         tables, kpis, ask_signature, schema_fingerprints, sql, sql_has_absolute_dates, \
         pandas, scores, path_taken, expert, uses, successes, failures, thumbs_up, \
         thumbs_down, strikes, eligible, ineligible_reason, created_at, updated_at"
    }

    fn row_to_repair(&self, row: &sqlx::postgres::PgRow) -> Result<RepairRecord> {
        let id: Uuid = row.try_get("id").map_err(map_db)?;
        let tables = strings_from_json(row.try_get("table_ids").map_err(map_db)?)?;
        Ok(RepairRecord {
            id: RecordId::from_uuid(id),
            scope: Scope::new(row.try_get::<String, _>("scope").map_err(map_db)?),
            table_ids: tables.into_iter().map(TableRef::new).collect(),
            error_class: row.try_get("error_class").map_err(map_db)?,
            failed_sql_fragment: row.try_get("failed_sql_fragment").map_err(map_db)?,
            fixed_sql_fragment: row.try_get("fixed_sql_fragment").map_err(map_db)?,
            schema_fingerprint: SchemaFingerprint::new(
                row.try_get::<String, _>("schema_fingerprint").map_err(map_db)?,
            ),
            times_seen: as_u32(row.try_get("times_seen").map_err(map_db)?, "times_seen")?,
            times_fixed: as_u32(row.try_get("times_fixed").map_err(map_db)?, "times_fixed")?,
            eligible: row.try_get("eligible").map_err(map_db)?,
            created_at: row.try_get("created_at").map_err(map_db)?,
            updated_at: row.try_get("updated_at").map_err(map_db)?,
        })
    }
}

#[async_trait]
impl MemoryStore for PostgresStore {
    async fn upsert_ask(&self, record: AskRecord) -> Result<(RecordId, bool)> {
        self.ensure_dim(&record.embedding)?;
        let table = self.ask_table();
        let vector = vector_literal(&record.embedding)?;
        let ts = now_ms();
        let sql = format!(
            "INSERT INTO {table} (
                id, scope, eval_id, conversation_id, question, embedding,
                tables, kpis, ask_signature, schema_fingerprints, sql,
                sql_has_absolute_dates, pandas, scores, path_taken, expert,
                uses, successes, failures, thumbs_up, thumbs_down, strikes,
                eligible, ineligible_reason, created_at, updated_at
             ) VALUES (
                $1,$2,$3,$4,$5,$6::vector,
                $7,$8,$9,$10,$11,
                $12,$13,$14,$15,$16,
                $17,$18,$19,$20,$21,$22,
                $23,$24,$25,$26
             )
             ON CONFLICT (scope, eval_id) DO UPDATE SET
                conversation_id = EXCLUDED.conversation_id,
                question = EXCLUDED.question,
                embedding = EXCLUDED.embedding,
                tables = EXCLUDED.tables,
                kpis = EXCLUDED.kpis,
                ask_signature = EXCLUDED.ask_signature,
                schema_fingerprints = EXCLUDED.schema_fingerprints,
                sql = EXCLUDED.sql,
                sql_has_absolute_dates = EXCLUDED.sql_has_absolute_dates,
                pandas = EXCLUDED.pandas,
                scores = EXCLUDED.scores,
                path_taken = EXCLUDED.path_taken,
                expert = EXCLUDED.expert,
                updated_at = EXCLUDED.updated_at
             RETURNING id, (xmax = 0) AS created"
        );
        let row = sqlx::query(&sql)
            .bind(record.id.as_uuid())
            .bind(record.scope.as_str())
            .bind(&record.eval_id)
            .bind(&record.conversation_id)
            .bind(&record.question)
            .bind(&vector)
            .bind(strings_json(&sorted_table_strings(&record.tables))?)
            .bind(strings_json(&sorted_kpi_strings(&record.kpis))?)
            .bind(
                serde_json::to_value(&record.ask_signature)
                    .map_err(|e| Error::Store(e.to_string()))?,
            )
            .bind(fingerprints_json(&record.schema_fingerprints)?)
            .bind(&record.sql)
            .bind(record.sql_has_absolute_dates)
            .bind(&record.pandas)
            .bind(serde_json::to_value(&record.scores).map_err(|e| Error::Store(e.to_string()))?)
            .bind(record.path_taken.as_str())
            .bind(record.expert)
            .bind(record.uses as i32)
            .bind(record.successes as i32)
            .bind(record.failures as i32)
            .bind(record.thumbs_up as i32)
            .bind(record.thumbs_down as i32)
            .bind(record.strikes as i32)
            .bind(record.eligible)
            .bind(&record.ineligible_reason)
            .bind(record.created_at.max(ts))
            .bind(ts)
            .fetch_one(&self.pool)
            .await
            .map_err(map_db)?;
        let id: Uuid = row.try_get("id").map_err(map_db)?;
        let created: bool = row.try_get("created").map_err(map_db)?;
        Ok((RecordId::from_uuid(id), created))
    }

    async fn get_ask(&self, scope: &Scope, id: &RecordId) -> Result<Option<AskRecord>> {
        let table = self.ask_table();
        let cols = Self::ask_select_cols();
        let sql = format!("SELECT {cols} FROM {table} WHERE scope = $1 AND id = $2");
        let row = sqlx::query(&sql)
            .bind(scope.as_str())
            .bind(id.as_uuid())
            .fetch_optional(&self.pool)
            .await
            .map_err(map_db)?;
        row.map(|r| self.row_to_ask(&r)).transpose()
    }

    async fn get_ask_by_eval(&self, scope: &Scope, eval_id: &str) -> Result<Option<AskRecord>> {
        let table = self.ask_table();
        let cols = Self::ask_select_cols();
        let sql = format!("SELECT {cols} FROM {table} WHERE scope = $1 AND eval_id = $2");
        let row = sqlx::query(&sql)
            .bind(scope.as_str())
            .bind(eval_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_db)?;
        row.map(|r| self.row_to_ask(&r)).transpose()
    }

    async fn list_eligible_asks(&self, scope: &Scope) -> Result<Vec<AskRecord>> {
        let table = self.ask_table();
        let cols = Self::ask_select_cols();
        let sql = format!("SELECT {cols} FROM {table} WHERE scope = $1 AND eligible = TRUE");
        let rows =
            sqlx::query(&sql).bind(scope.as_str()).fetch_all(&self.pool).await.map_err(map_db)?;
        rows.iter().map(|r| self.row_to_ask(r)).collect()
    }

    async fn has_eligible_asks(&self, scope: &Scope) -> Result<bool> {
        let table = self.ask_table();
        let sql = format!(
            "SELECT EXISTS(
                SELECT 1 FROM {table} WHERE scope = $1 AND eligible = TRUE
             )"
        );
        sqlx::query_scalar(&sql).bind(scope.as_str()).fetch_one(&self.pool).await.map_err(map_db)
    }

    async fn list_eligible_asks_near(
        &self,
        scope: &Scope,
        embedding: &[f32],
        limit: usize,
        tables: Option<&[TableRef]>,
        kpis: Option<&[KpiRef]>,
    ) -> Result<Vec<AskRecord>> {
        self.ensure_dim(embedding)?;
        let table = self.ask_table();
        let cols = Self::ask_select_cols();
        let vector = vector_literal(embedding)?;
        // Exact scan within scope first — no ANN index (avoids cross-tenant HNSW misses).
        let sql = format!(
            "SELECT {cols} FROM {table}
             WHERE scope = $1 AND eligible = TRUE
               AND ($3::jsonb IS NULL OR tables = $3)
               AND ($4::jsonb IS NULL OR kpis = $4)
             ORDER BY embedding <=> $2::vector
             LIMIT $5"
        );
        let tables_json = match tables {
            Some(t) => Some(strings_json(&sorted_table_strings(t))?),
            None => None,
        };
        let kpis_json = match kpis {
            Some(k) => Some(strings_json(&sorted_kpi_strings(k))?),
            None => None,
        };
        let rows = sqlx::query(&sql)
            .bind(scope.as_str())
            .bind(&vector)
            .bind(tables_json)
            .bind(kpis_json)
            .bind(limit.max(1) as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(map_db)?;
        rows.iter().map(|r| self.row_to_ask(r)).collect()
    }

    async fn bump_uses(&self, scope: &Scope, id: &RecordId) -> Result<()> {
        let table = self.ask_table();
        let sql = format!(
            "UPDATE {table} SET uses = uses + 1, updated_at = $1
             WHERE scope = $2 AND id = $3"
        );
        let done = sqlx::query(&sql)
            .bind(now_ms())
            .bind(scope.as_str())
            .bind(id.as_uuid())
            .execute(&self.pool)
            .await
            .map_err(map_db)?;
        if done.rows_affected() == 0 {
            return Err(Error::NotFound(format!("ask {}", id.as_uuid())));
        }
        Ok(())
    }

    async fn bump_successes(&self, scope: &Scope, id: &RecordId) -> Result<()> {
        let table = self.ask_table();
        let sql = format!(
            "UPDATE {table} SET successes = successes + 1, updated_at = $1
             WHERE scope = $2 AND id = $3"
        );
        let done = sqlx::query(&sql)
            .bind(now_ms())
            .bind(scope.as_str())
            .bind(id.as_uuid())
            .execute(&self.pool)
            .await
            .map_err(map_db)?;
        if done.rows_affected() == 0 {
            return Err(Error::NotFound(format!("ask {}", id.as_uuid())));
        }
        Ok(())
    }

    async fn apply_decay(
        &self,
        scope: &Scope,
        id: &RecordId,
        reason: DecayReason,
        policy: &Policy,
    ) -> Result<()> {
        let table = self.ask_table();
        let ts = now_ms();
        let done = match reason {
            DecayReason::FailedReuse => {
                let sql = format!(
                    "UPDATE {table} SET
                        failures = failures + 1,
                        strikes = strikes + 1,
                        eligible = CASE WHEN strikes + 1 >= $1 THEN FALSE ELSE eligible END,
                        ineligible_reason = CASE WHEN strikes + 1 >= $1 THEN 'failed_reuse' ELSE ineligible_reason END,
                        updated_at = $2
                     WHERE scope = $3 AND id = $4"
                );
                sqlx::query(&sql)
                    .bind(policy.max_strikes as i32)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(id.as_uuid())
                    .execute(&self.pool)
                    .await
                    .map_err(map_db)?
            }
            DecayReason::NegativeFeedback => {
                let sql = format!(
                    "UPDATE {table} SET
                        thumbs_down = thumbs_down + 1,
                        eligible = CASE WHEN thumbs_down + 1 >= $1 THEN FALSE ELSE eligible END,
                        ineligible_reason = CASE WHEN thumbs_down + 1 >= $1 THEN 'negative_feedback' ELSE ineligible_reason END,
                        updated_at = $2
                     WHERE scope = $3 AND id = $4"
                );
                sqlx::query(&sql)
                    .bind(policy.max_thumbs_down as i32)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(id.as_uuid())
                    .execute(&self.pool)
                    .await
                    .map_err(map_db)?
            }
            DecayReason::TableInvalidated => {
                let sql = format!(
                    "UPDATE {table} SET eligible = FALSE, ineligible_reason = 'table_invalidated', updated_at = $1
                     WHERE scope = $2 AND id = $3"
                );
                sqlx::query(&sql)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(id.as_uuid())
                    .execute(&self.pool)
                    .await
                    .map_err(map_db)?
            }
            DecayReason::Manual => {
                let sql = format!(
                    "UPDATE {table} SET eligible = FALSE, ineligible_reason = 'manual', updated_at = $1
                     WHERE scope = $2 AND id = $3"
                );
                sqlx::query(&sql)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(id.as_uuid())
                    .execute(&self.pool)
                    .await
                    .map_err(map_db)?
            }
        };
        if done.rows_affected() == 0 {
            return Err(Error::NotFound(format!("ask {}", id.as_uuid())));
        }
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
                let table = self.ask_table();
                let sql = format!(
                    "UPDATE {table} SET thumbs_up = thumbs_up + 1, updated_at = $1
                     WHERE scope = $2 AND id = $3"
                );
                let done = sqlx::query(&sql)
                    .bind(now_ms())
                    .bind(scope.as_str())
                    .bind(id.as_uuid())
                    .execute(&self.pool)
                    .await
                    .map_err(map_db)?;
                if done.rows_affected() == 0 {
                    return Err(Error::NotFound(format!("ask {}", id.as_uuid())));
                }
                Ok(())
            }
        }
    }

    async fn upsert_repair(&self, record: RepairRecord) -> Result<(RecordId, bool)> {
        let table = self.repair_table();
        let tables_json = strings_json(&sorted_table_strings(&record.table_ids))?;
        let ts = now_ms();
        let sql = format!(
            "INSERT INTO {table} (
                id, scope, table_ids, error_class, failed_sql_fragment, fixed_sql_fragment,
                schema_fingerprint, times_seen, times_fixed, eligible, created_at, updated_at
             ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
             ON CONFLICT (scope, error_class, schema_fingerprint, table_ids) DO UPDATE SET
                failed_sql_fragment = EXCLUDED.failed_sql_fragment,
                fixed_sql_fragment = EXCLUDED.fixed_sql_fragment,
                times_seen = {table}.times_seen + 1,
                times_fixed = {table}.times_fixed
                    + CASE WHEN {table}.fixed_sql_fragment IS DISTINCT FROM EXCLUDED.fixed_sql_fragment
                           THEN 1 ELSE 0 END,
                updated_at = EXCLUDED.updated_at
             RETURNING id, (xmax = 0) AS created"
        );
        let row = sqlx::query(&sql)
            .bind(record.id.as_uuid())
            .bind(record.scope.as_str())
            .bind(tables_json)
            .bind(&record.error_class)
            .bind(&record.failed_sql_fragment)
            .bind(&record.fixed_sql_fragment)
            .bind(record.schema_fingerprint.as_str())
            .bind(record.times_seen as i32)
            .bind(record.times_fixed as i32)
            .bind(record.eligible)
            .bind(record.created_at.max(ts))
            .bind(ts)
            .fetch_one(&self.pool)
            .await
            .map_err(map_db)?;
        let id: Uuid = row.try_get("id").map_err(map_db)?;
        let created: bool = row.try_get("created").map_err(map_db)?;
        Ok((RecordId::from_uuid(id), created))
    }

    async fn list_eligible_repairs(
        &self,
        scope: &Scope,
        error_class: &str,
    ) -> Result<Vec<RepairRecord>> {
        let table = self.repair_table();
        let sql = format!(
            "SELECT id, scope, table_ids, error_class, failed_sql_fragment, fixed_sql_fragment,
                    schema_fingerprint, times_seen, times_fixed, eligible, created_at, updated_at
             FROM {table}
             WHERE scope = $1 AND eligible = TRUE AND error_class = $2"
        );
        let rows = sqlx::query(&sql)
            .bind(scope.as_str())
            .bind(error_class)
            .fetch_all(&self.pool)
            .await
            .map_err(map_db)?;
        rows.iter().map(|r| self.row_to_repair(r)).collect()
    }

    async fn invalidate_table(&self, scope: &Scope, table_ref: &TableRef) -> Result<u64> {
        let ask_t = self.ask_table();
        let repair_t = self.repair_table();
        let ts = now_ms();
        // JSONB containment: tables @> '["sales.public.orders"]'
        let needle = strings_json(&[table_ref.as_str().to_string()])?;
        let ask_sql = format!(
            "UPDATE {ask_t} SET eligible = FALSE, ineligible_reason = 'table_invalidated', updated_at = $1
             WHERE scope = $2 AND eligible = TRUE AND tables @> $3"
        );
        let a = sqlx::query(&ask_sql)
            .bind(ts)
            .bind(scope.as_str())
            .bind(&needle)
            .execute(&self.pool)
            .await
            .map_err(map_db)?;
        let repair_sql = format!(
            "UPDATE {repair_t} SET eligible = FALSE, updated_at = $1
             WHERE scope = $2 AND eligible = TRUE AND table_ids @> $3"
        );
        let r = sqlx::query(&repair_sql)
            .bind(ts)
            .bind(scope.as_str())
            .bind(&needle)
            .execute(&self.pool)
            .await
            .map_err(map_db)?;
        Ok(a.rows_affected().saturating_add(r.rows_affected()))
    }

    async fn apply_trust_for_eval(
        &self,
        scope: &Scope,
        eval_id: &str,
        source_id: &RecordId,
        outcome: TrustOutcome,
        policy: &Policy,
    ) -> Result<bool> {
        let trust_t = self.trust_table();
        let ask_t = self.ask_table();
        let ts = now_ms();
        let mut tx = self.pool.begin().await.map_err(map_db)?;

        let claim_sql = format!(
            "INSERT INTO {trust_t} (scope, eval_id, source_id, outcome, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (scope, eval_id) DO NOTHING
             RETURNING eval_id"
        );
        let claimed: Option<String> = sqlx::query_scalar(&claim_sql)
            .bind(scope.as_str())
            .bind(eval_id)
            .bind(source_id.as_uuid())
            .bind(outcome.as_str())
            .bind(ts)
            .fetch_optional(&mut *tx)
            .await
            .map_err(map_db)?;
        if claimed.is_none() {
            tx.commit().await.map_err(map_db)?;
            return Ok(false);
        }

        let done = match outcome {
            TrustOutcome::Reinforce => {
                let sql = format!(
                    "UPDATE {ask_t} SET uses = uses + 1, successes = successes + 1, updated_at = $1
                     WHERE scope = $2 AND id = $3"
                );
                sqlx::query(&sql)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(source_id.as_uuid())
                    .execute(&mut *tx)
                    .await
                    .map_err(map_db)?
            }
            TrustOutcome::DecayFailedReuse => {
                let sql = format!(
                    "UPDATE {ask_t} SET
                        failures = failures + 1,
                        strikes = strikes + 1,
                        eligible = CASE WHEN strikes + 1 >= $1 THEN FALSE ELSE eligible END,
                        ineligible_reason = CASE WHEN strikes + 1 >= $1 THEN 'failed_reuse' ELSE ineligible_reason END,
                        updated_at = $2
                     WHERE scope = $3 AND id = $4"
                );
                sqlx::query(&sql)
                    .bind(policy.max_strikes as i32)
                    .bind(ts)
                    .bind(scope.as_str())
                    .bind(source_id.as_uuid())
                    .execute(&mut *tx)
                    .await
                    .map_err(map_db)?
            }
        };
        if done.rows_affected() == 0 {
            // Roll back the claim so a retry can succeed once the source exists.
            tx.rollback().await.map_err(map_db)?;
            return Err(Error::NotFound(format!("ask {}", source_id.as_uuid())));
        }
        tx.commit().await.map_err(map_db)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_literal_format() {
        assert_eq!(vector_literal(&[1.0, 2.5, -3.0]).unwrap(), "[1,2.5,-3]");
        assert_eq!(vector_literal(&[]).unwrap(), "[]");
        assert!(vector_literal(&[f32::NAN]).is_err());
    }
}
