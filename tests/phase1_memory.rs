//! Phase 1 unit tests (in-memory store + fake embedder).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::field_reassign_with_default
)]

use std::sync::Arc;

use engram_dview::{
    AskQuery, AskSignature, DecayReason, ErrorQuery, FakeEmbedder, GradedAsk, InMemoryStore,
    KpiRef, Memory, MissReason, Outcome, PathTaken, Policy, RecordAskResult, RepairInput,
    SchemaFingerprint, Scope, Scores, TableRef, Vote,
};

fn memory(policy: Policy) -> Memory<InMemoryStore, FakeEmbedder> {
    Memory::new(Arc::new(InMemoryStore::new()), Arc::new(FakeEmbedder::default()), policy)
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

async fn store(
    mem: &Memory<InMemoryStore, FakeEmbedder>,
    ask: GradedAsk,
) -> engram_dview::Recorded {
    match mem.record_ask(&scope_a(), ask).await.unwrap() {
        RecordAskResult::Stored(r) => r,
        RecordAskResult::SkippedBelowThreshold => panic!("unexpected skip"),
        _ => panic!("unexpected record result"),
    }
}

const RELATIVE_SQL: &str = "SELECT region, sum(amount) FROM sales.public.orders WHERE order_date >= current_date - INTERVAL '6' MONTH GROUP BY 1";

const ABSOLUTE_SQL: &str = "SELECT region, sum(amount) FROM sales.public.orders WHERE order_date >= '2024-01-01' GROUP BY 1";

#[tokio::test]
async fn empty_store_is_miss() {
    let mem = memory(Policy::default());
    let out = mem
        .retrieve(&scope_a(), &query("monthly revenue by region last 6 months", "6", "schema-v1"))
        .await
        .unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
}

#[tokio::test]
async fn verified_reuse_when_gates_pass() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::VerifiedReuse { record } => {
            assert_eq!(record.eval_id, "e1");
            assert_eq!(record.sql, RELATIVE_SQL);
            assert_eq!(record.uses, 0, "retrieve must not bump uses");
        }
        other => panic!("expected VerifiedReuse, got {other:?}"),
    }
}

#[tokio::test]
async fn signature_mismatch_six_vs_three_months_is_few_shot_not_reuse() {
    let mem = memory(Policy::default());
    let stored_q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", stored_q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem
        .retrieve(&scope_a(), &query("monthly revenue by region last 3 months", "3", "schema-v1"))
        .await
        .unwrap();
    match out {
        Outcome::FewShot { examples } => {
            assert!(!examples.is_empty());
            assert_eq!(examples[0].eval_id, "e1");
        }
        Outcome::VerifiedReuse { .. } => panic!("must not verified-reuse on window mismatch"),
        other => panic!("expected FewShot, got {other:?}"),
    }
}

#[tokio::test]
async fn schema_fingerprint_mismatch_blocks_verified_reuse() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v2")).await.unwrap();
    assert!(
        !matches!(out, Outcome::VerifiedReuse { .. }),
        "same table name, new schema must not replay SQL"
    );
    assert!(matches!(out, Outcome::FewShot { .. } | Outcome::Miss { .. }));
}

#[tokio::test]
async fn absolute_date_sql_blocks_verified_reuse() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", ABSOLUTE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(
        !matches!(out, Outcome::VerifiedReuse { .. }),
        "absolute dates must not verified-reuse"
    );
}

#[tokio::test]
async fn unparseable_sql_still_stores_but_blocks_reuse() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let mut ask = graded("e1", q, "6", "SELECT * FROM t WHERE x @@ nonsense", "schema-v1");
    ask.sql = "NOT VALID SQL ;;; <<<".into();
    store(&mem, ask).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(!matches!(out, Outcome::VerifiedReuse { .. }));
}

#[tokio::test]
async fn empty_signature_blocks_verified_reuse() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let mut ask = graded("e1", q, "6", RELATIVE_SQL, "schema-v1");
    ask.ask_signature = AskSignature::default();
    store(&mem, ask).await;

    let mut qy = query(q, "6", "schema-v1");
    qy.signature = AskSignature::default();
    let out = mem.retrieve(&scope_a(), &qy).await.unwrap();
    assert!(
        !matches!(out, Outcome::VerifiedReuse { .. }),
        "empty signatures must not verified-reuse"
    );
}

#[tokio::test]
async fn scope_isolation() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_b(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
}

#[tokio::test]
async fn decay_after_strikes_makes_ineligible() {
    let mut policy = Policy::default();
    policy.max_strikes = 2;
    let mem = memory(policy);
    let q = "monthly revenue by region last 6 months";
    let recorded = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    mem.decay(&scope_a(), &recorded.id, DecayReason::FailedReuse).await.unwrap();
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::FewShot { .. }), "one strike blocks reuse but still few-shot");

    mem.decay(&scope_a(), &recorded.id, DecayReason::FailedReuse).await.unwrap();
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { .. }));
}

#[tokio::test]
async fn feedback_down_eventually_ineligible() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let recorded = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    mem.feedback(&scope_a(), &recorded.id, Vote::Down).await.unwrap();
    // One thumbs-down must not count as failed reuse (still VerifiedReuse).
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));

    mem.feedback(&scope_a(), &recorded.id, Vote::Down).await.unwrap();
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { .. }));
}

#[tokio::test]
async fn feedback_up_raises_few_shot_ranking() {
    let store_arc = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(FakeEmbedder::default());
    let mut policy = Policy::default();
    policy.verified_reuse_enabled = false;
    let mem = Memory::new(store_arc, embedder, policy);

    let q = "monthly revenue by region last 6 months";
    let low = store(&mem, graded("low", q, "6", RELATIVE_SQL, "schema-v1")).await;
    let mut high_score = graded("high", q, "6", RELATIVE_SQL, "schema-v1");
    high_score.scores.final_score = Some(99.0);
    store(&mem, high_score).await;

    // Without feedback, higher final score ranks first among non-experts.
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::FewShot { examples } => assert_eq!(examples[0].eval_id, "high"),
        other => panic!("expected FewShot, got {other:?}"),
    }

    // Thumbs-up on the lower-score ask should outrank after net feedback.
    mem.feedback(&scope_a(), &low.id, Vote::Up).await.unwrap();
    mem.feedback(&scope_a(), &low.id, Vote::Up).await.unwrap();
    mem.feedback(&scope_a(), &low.id, Vote::Up).await.unwrap();

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::FewShot { examples } => {
            assert_eq!(examples[0].eval_id, "low", "thumbs-up must raise few-shot trust");
        }
        other => panic!("expected FewShot, got {other:?}"),
    }
}

#[tokio::test]
async fn invalidate_table_marks_asks_and_repairs() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
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
    assert!(matches!(out, Outcome::Miss { .. }));

    let repairs = mem
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
    assert!(repairs.is_empty());
}

#[tokio::test]
async fn repair_upsert_does_not_reenable_invalidated() {
    let mem = memory(Policy::default());
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
    mem.invalidate_table(&scope_a(), &table_orders()).await.unwrap();

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

    let repairs = mem
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
    assert!(repairs.is_empty(), "invalidated repair must stay ineligible");
}

#[tokio::test]
async fn eval_id_upsert_preserves_trust() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let first = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
    assert!(first.created);

    mem.decay(&scope_a(), &first.id, DecayReason::FailedReuse).await.unwrap();
    mem.decay(&scope_a(), &first.id, DecayReason::FailedReuse).await.unwrap();

    let second = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
    assert!(!second.created);
    assert_eq!(first.id.as_uuid(), second.id.as_uuid());

    // Still ineligible after re-record.
    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
}

#[tokio::test]
async fn verified_reuse_can_be_disabled_by_policy() {
    let mut policy = Policy::default();
    policy.verified_reuse_enabled = false;
    let mem = memory(policy);
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    assert!(matches!(out, Outcome::FewShot { .. }));
}

#[tokio::test]
async fn different_tables_hard_filter_miss() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    let mut ask = query(q, "6", "schema-v1");
    ask.tables = vec![TableRef::new("sales.public.customers")];
    let out = mem.retrieve(&scope_a(), &ask).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::HardFilter }));
}

#[tokio::test]
async fn below_threshold_is_skipped_not_error() {
    let mem = memory(Policy::default());
    let mut ask = graded("e1", "q", "6", RELATIVE_SQL, "schema-v1");
    ask.scores = Scores { final_score: Some(10.0), sql_score: Some(1.0) };
    let res = mem.record_ask(&scope_a(), ask).await.unwrap();
    assert!(matches!(res, RecordAskResult::SkippedBelowThreshold));
}

#[tokio::test]
async fn reinforce_bumps_uses_and_successes() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let recorded = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;

    mem.reinforce(&scope_a(), &recorded.id).await.unwrap();

    let out = mem.retrieve(&scope_a(), &query(q, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::VerifiedReuse { record } => {
            assert_eq!(record.uses, 1);
            assert_eq!(record.successes, 1);
        }
        other => panic!("expected VerifiedReuse, got {other:?}"),
    }
}

#[tokio::test]
async fn decay_other_scope_is_not_found() {
    let mem = memory(Policy::default());
    let q = "monthly revenue by region last 6 months";
    let recorded = store(&mem, graded("e1", q, "6", RELATIVE_SQL, "schema-v1")).await;
    let err = mem.decay(&scope_b(), &recorded.id, DecayReason::FailedReuse).await;
    assert!(err.is_err());
}

#[tokio::test]
async fn expert_lower_sim_does_not_block_higher_sim_reuse() {
    // Expert question is crafted so FakeEmbedder similarity to the query sits in
    // [few_shot_min, verified_reuse_min) ≈ [0.75, 0.92). That keeps the expert in
    // the ranked pool (old bug: expert-first could steal reuse) without clearing
    // the verified-reuse floor.
    let store_arc = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(FakeEmbedder::default());
    let mem = Memory::new(Arc::clone(&store_arc), Arc::clone(&embedder), Policy::default());

    let q_exact = "monthly revenue by region last 6 months";
    let q_expert = "monthly revenue by region last year"; // ~0.77 vs query

    let mut expert = graded("expert", q_expert, "6", RELATIVE_SQL, "schema-v1");
    expert.expert = true;
    store(&mem, expert).await;
    store(&mem, graded("normal", q_exact, "6", RELATIVE_SQL, "schema-v1")).await;

    let out = mem.retrieve(&scope_a(), &query(q_exact, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::VerifiedReuse { record } => {
            assert_eq!(record.eval_id, "normal", "reuse picks best gate-passer by similarity");
        }
        other => panic!("expected VerifiedReuse of normal, got {other:?}"),
    }

    // With reuse disabled, few-shot still lists the expert first.
    let mut policy = Policy::default();
    policy.verified_reuse_enabled = false;
    let mem_fs = Memory::new(store_arc, embedder, policy);
    let out = mem_fs.retrieve(&scope_a(), &query(q_exact, "6", "schema-v1")).await.unwrap();
    match out {
        Outcome::FewShot { examples } => {
            assert_eq!(examples[0].eval_id, "expert", "few-shot ranks expert first");
            assert!(examples.iter().any(|e| e.eval_id == "normal"));
        }
        other => panic!("expected FewShot with expert first, got {other:?}"),
    }
}

#[tokio::test]
async fn repair_lookup_requires_matching_fingerprint() {
    let mem = memory(Policy::default());
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
