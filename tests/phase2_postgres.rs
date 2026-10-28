//! Phase 2 Postgres + pgvector integration tests.
//!
//! Requires `ENGRAM_DATABASE_URL` (sqlx form or `postgresql+psycopg2://…`).
//! Each test uses an isolated schema `engram_test_<uuid>` and drops it afterward.
//! Set `ENGRAM_REQUIRE_DB=1` to fail (not skip) when the URL is missing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::field_reassign_with_default
)]

use std::sync::Arc;

use engram_dview::{
    apply_schema, AskQuery, AskSignature, DecayReason, Embedder, ErrorQuery, FakeEmbedder,
    GradedAsk, KpiRef, Memory, MemoryStore, MissReason, Outcome, PathTaken, Policy, PostgresStore,
    RecordAskResult, RepairInput, SchemaFingerprint, Scope, Scores, TableRef, Vote,
    DEFAULT_EMBEDDING_DIM,
};
use sqlx::PgPool;
use uuid::Uuid;

fn load_env() {
    let _ = dotenvy::dotenv();
}

fn normalize_pg_url(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("postgresql+") {
        if let Some((_, after_driver)) = rest.split_once("://") {
            return format!("postgresql://{after_driver}");
        }
    }
    if let Some(rest) = trimmed.strip_prefix("postgres+") {
        if let Some((_, after_driver)) = rest.split_once("://") {
            return format!("postgres://{after_driver}");
        }
    }
    trimmed.to_string()
}

fn database_url() -> Option<String> {
    load_env();
    // Only ENGRAM_DATABASE_URL — never DATABASE_URL / product env vars (DROP SCHEMA safety).
    std::env::var("ENGRAM_DATABASE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .map(|u| normalize_pg_url(&u))
}

fn require_db() -> bool {
    load_env();
    matches!(std::env::var("ENGRAM_REQUIRE_DB").as_deref(), Ok("1") | Ok("true") | Ok("TRUE"))
}

fn embedding_dim() -> usize {
    load_env();
    std::env::var("ENGRAM_EMBEDDING_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EMBEDDING_DIM)
}

struct TestDb {
    pool: PgPool,
    schema: String,
}

impl TestDb {
    async fn setup() -> Option<Self> {
        let Some(url) = database_url() else {
            if require_db() {
                panic!("ENGRAM_REQUIRE_DB=1 but ENGRAM_DATABASE_URL is unset");
            }
            return None;
        };

        let pool = PgPool::connect(&url).await.expect("connect postgres");
        let has_vector: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')",
        )
        .fetch_one(&pool)
        .await
        .expect("check vector extension");
        if !has_vector {
            if require_db() {
                panic!("ENGRAM_REQUIRE_DB=1 but pgvector is not available");
            }
            eprintln!("skipping: pgvector (vector) extension not available");
            return None;
        }

        let schema = format!("engram_test_{}", Uuid::new_v4().simple());
        assert!(
            schema.starts_with("engram_test"),
            "refusing to touch schema that does not start with engram_test"
        );

        apply_schema(&pool, &schema, embedding_dim()).await.expect("apply engram schema");
        Some(Self { pool, schema })
    }

    fn memory(&self) -> Memory<PostgresStore, FakeEmbedder> {
        let store =
            PostgresStore::new(self.pool.clone(), self.schema.clone(), embedding_dim()).unwrap();
        Memory::new(
            Arc::new(store),
            Arc::new(FakeEmbedder::new(embedding_dim())),
            Policy::default(),
        )
    }

    fn store(&self) -> Arc<PostgresStore> {
        Arc::new(
            PostgresStore::new(self.pool.clone(), self.schema.clone(), embedding_dim()).unwrap(),
        )
    }
}

impl Drop for TestDb {
    fn drop(&mut self) {
        let schema = self.schema.clone();
        let Some(url) = database_url() else {
            return;
        };
        // Fresh connection on a dedicated thread — joining on the test pool deadlocks
        // the current-thread tokio runtime.
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
                return;
            };
            rt.block_on(async {
                if let Ok(pool) = PgPool::connect(&url).await {
                    let _ = sqlx::query(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
                        .execute(&pool)
                        .await;
                    pool.close().await;
                }
            });
        })
        .join();
    }
}

fn scope_a() -> Scope {
    Scope::new("tenant-a")
}

fn scope_b() -> Scope {
    Scope::new("tenant-b")
}

fn table_orders() -> TableRef {
    TableRef::new("sales.public.orders")
}

fn fp(v: &str) -> SchemaFingerprint {
    SchemaFingerprint::new(v)
}

fn signature_window(months: &str) -> AskSignature {
    AskSignature {
        metric: Some("revenue".into()),
        dimensions: vec!["region".into()],
        grain: Some("month".into()),
        window: Some(format!("last {months} months")),
        filters: Default::default(),
        top_n: None,
    }
}

fn graded(eval_id: &str, question: &str, window: &str, sql: &str, fingerprint: &str) -> GradedAsk {
    GradedAsk {
        eval_id: eval_id.into(),
        conversation_id: Some("c1".into()),
        question: question.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        ask_signature: signature_window(window),
        schema_fingerprints: vec![(table_orders(), fp(fingerprint))],
        sql: sql.into(),
        pandas: None,
        scores: Scores { final_score: Some(85.0), sql_score: Some(8.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    }
}

fn query(question: &str, window: &str, fingerprint: &str) -> AskQuery {
    AskQuery {
        question: question.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        signature: signature_window(window),
        fingerprints: vec![(table_orders(), fp(fingerprint))],
    }
}

const RELATIVE_SQL: &str = "SELECT region, sum(amount) FROM sales.public.orders WHERE order_date >= current_date - INTERVAL '6' MONTH GROUP BY 1";

async fn store_ask(
    mem: &Memory<PostgresStore, FakeEmbedder>,
    ask: GradedAsk,
) -> engram_dview::Recorded {
    match mem.record_ask(&scope_a(), ask).await.unwrap() {
        RecordAskResult::Stored(r) => r,
        RecordAskResult::SkippedBelowThreshold => panic!("unexpected skip"),
        _ => panic!("unexpected"),
    }
}

#[tokio::test]
async fn postgres_round_trip_record_and_retrieve() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let mem = db.memory();
    let q = "monthly revenue by region last 6 months";
    store_ask(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::VerifiedReuse { record } => {
            assert_eq!(record.eval_id, "e1");
            assert_eq!(record.sql, RELATIVE_SQL);
        }
        other => panic!("expected VerifiedReuse, got {other:?}"),
    }
}

#[tokio::test]
async fn postgres_eval_id_upsert_is_idempotent_and_preserves_trust() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let mem = db.memory();
    let q = "monthly revenue by region last 6 months";
    let first = store_ask(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
    assert!(first.created);

    mem.decay(&scope_a(), &first.id, DecayReason::FailedReuse).await.unwrap();
    mem.decay(&scope_a(), &first.id, DecayReason::FailedReuse).await.unwrap();

    let second = store_ask(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
    assert!(!second.created);
    assert_eq!(first.id.as_uuid(), second.id.as_uuid());

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));

    let n: i64 = sqlx::query_scalar(&format!(
        "SELECT count(*) FROM {}.engram_ask_records WHERE scope = $1 AND eval_id = $2",
        db.schema
    ))
    .bind(scope_a().as_str())
    .bind("e1")
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert_eq!(n, 1);
}

#[tokio::test]
async fn postgres_repair_upsert_and_fingerprint_gate() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let mem = db.memory();

    mem.record_repair(
        &scope_a(),
        RepairInput {
            table_ids: vec![table_orders()],
            error_class: "COLUMN_NOT_FOUND:amount".into(),
            failed_sql_fragment: "SUM(amount)".into(),
            fixed_sql_fragment: "SUM(net_amount)".into(),
            schema_fingerprint: fp("schema-v1"),
        },
    )
    .await
    .unwrap();

    // Same key again — times_seen bumps, stays one row.
    mem.record_repair(
        &scope_a(),
        RepairInput {
            table_ids: vec![table_orders()],
            error_class: "COLUMN_NOT_FOUND:amount".into(),
            failed_sql_fragment: "SUM(amount)".into(),
            fixed_sql_fragment: "SUM(net_amount)".into(),
            schema_fingerprint: fp("schema-v1"),
        },
    )
    .await
    .unwrap();

    let hit = mem
        .repairs(
            &scope_a(),
            &ErrorQuery {
                table_ids: vec![table_orders()],
                error_class: "COLUMN_NOT_FOUND:amount".into(),
                schema_fingerprint: fp("schema-v1"),
            },
        )
        .await
        .unwrap();
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].times_seen, 2);
    assert_eq!(hit[0].times_fixed, 1);

    let miss = mem
        .repairs(
            &scope_a(),
            &ErrorQuery {
                table_ids: vec![table_orders()],
                error_class: "COLUMN_NOT_FOUND:amount".into(),
                schema_fingerprint: fp("schema-v2"),
            },
        )
        .await
        .unwrap();
    assert!(miss.is_empty());
}

#[tokio::test]
async fn postgres_vector_search_nearest_first_and_scope() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let store = db.store();
    let mem = Memory::new(
        Arc::clone(&store),
        Arc::new(FakeEmbedder::new(embedding_dim())),
        Policy::default(),
    );
    let q = "monthly revenue by region last 6 months";
    store_ask(&mem, graded("near", q, "6", RELATIVE_SQL, "schema-v1")).await;
    store_ask(
        &mem,
        graded("far", "totally unrelated kitchen sink query", "6", RELATIVE_SQL, "schema-v1"),
    )
    .await;

    let out = mem.retrieve(&scope_b(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));

    let emb = FakeEmbedder::new(embedding_dim()).embed(q).await.unwrap();
    let near = store
        .list_eligible_asks_near(&scope_a(), &emb, 10, Some(&[table_orders()]), None)
        .await
        .unwrap();
    assert!(near.len() >= 2);
    assert_eq!(near[0].eval_id, "near", "nearest neighbor first");
}

#[tokio::test]
async fn postgres_schema_apply_is_idempotent() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    apply_schema(&db.pool, &db.schema, embedding_dim()).await.expect("second apply");
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM information_schema.tables
         WHERE table_schema = $1 AND table_name LIKE 'engram_%'
         ORDER BY 1",
    )
    .bind(&db.schema)
    .fetch_all(&db.pool)
    .await
    .unwrap();
    assert!(tables.iter().any(|t| t == "engram_ask_records"));
    assert!(tables.iter().any(|t| t == "engram_repair_records"));

    let store = db.store();
    store.check_embedding_dim().await.unwrap();

    // Primary key exists on both tables.
    let pk: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM information_schema.table_constraints
         WHERE table_schema = $1 AND constraint_type = 'PRIMARY KEY'
           AND table_name IN ('engram_ask_records', 'engram_repair_records')",
    )
    .bind(&db.schema)
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert_eq!(pk, 2);
}

#[tokio::test]
async fn postgres_feedback_reinforce_invalidate() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let mem = db.memory();
    let q = "monthly revenue by region last 6 months";
    let recorded = store_ask(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    mem.reinforce(&scope_a(), &recorded.id).await.unwrap();
    mem.feedback(&scope_a(), &recorded.id, Vote::Up).await.unwrap();

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::VerifiedReuse { record } => {
            assert_eq!(record.uses, 1);
            assert_eq!(record.successes, 1);
            assert_eq!(record.thumbs_up, 1);
        }
        other => panic!("expected VerifiedReuse, got {other:?}"),
    }

    mem.record_repair(
        &scope_a(),
        RepairInput {
            table_ids: vec![table_orders()],
            error_class: "COLUMN_NOT_FOUND:amount".into(),
            failed_sql_fragment: "SUM(amount)".into(),
            fixed_sql_fragment: "SUM(net_amount)".into(),
            schema_fingerprint: fp("schema-v1"),
        },
    )
    .await
    .unwrap();

    let n = mem.invalidate_table(&scope_a(), &table_orders()).await.unwrap();
    assert!(n >= 2);
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
}

#[tokio::test]
async fn postgres_thumbs_down_reaches_ineligible() {
    let Some(db) = TestDb::setup().await else {
        return;
    };
    let mem = db.memory();
    let q = "monthly revenue by region last 6 months";
    let recorded = store_ask(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    mem.feedback(&scope_a(), &recorded.id, Vote::Down).await.unwrap();
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));

    mem.feedback(&scope_a(), &recorded.id, Vote::Down).await.unwrap();
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
}
