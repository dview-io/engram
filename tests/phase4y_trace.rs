//! Phase 4y.1 — retrieve decision trace (in-memory store + fake embedder).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::field_reassign_with_default
)]

use std::sync::Arc;

use engram_dview::{
    AskQuery, AskSignature, FakeEmbedder, GradedAsk, InMemoryStore, KpiRef, Memory, MissReason,
    Outcome, PathTaken, Policy, RecordAskResult, RejectReason, RetrieveTrace, SchemaFingerprint,
    Scope, Scores, TableRef,
};

const SQL: &str = "SELECT region, sum(amount) FROM sales.public.orders WHERE order_date >= current_date - INTERVAL '6' MONTH GROUP BY 1";
const QUESTION: &str = "monthly revenue by region last 6 months";

fn memory(policy: Policy) -> Memory<InMemoryStore, FakeEmbedder> {
    Memory::new(Arc::new(InMemoryStore::new()), Arc::new(FakeEmbedder::default()), policy)
}

fn scope() -> Scope {
    Scope::new("tenant-a")
}

fn table_orders() -> TableRef {
    TableRef::new("sales.public.orders")
}

fn signature(window: &str) -> AskSignature {
    AskSignature {
        metric: Some("revenue".into()),
        dimensions: vec!["region".into()],
        grain: Some("month".into()),
        window: Some(format!("last {window} months")),
        filters: Default::default(),
        top_n: None,
        ..Default::default()
    }
}

fn graded(eval_id: &str, window: &str, fingerprint: &str) -> GradedAsk {
    GradedAsk {
        eval_id: eval_id.into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        ask_signature: signature(window),
        schema_fingerprints: vec![(table_orders(), SchemaFingerprint::new(fingerprint))],
        sql: SQL.into(),
        pandas: None,
        scores: Scores { final_score: Some(85.0), sql_score: Some(8.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    }
}

fn query(window: &str, fingerprint: &str) -> AskQuery {
    AskQuery {
        question: QUESTION.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        signature: signature(window),
        fingerprints: vec![(table_orders(), SchemaFingerprint::new(fingerprint))],
    }
}

async fn store(mem: &Memory<InMemoryStore, FakeEmbedder>, ask: GradedAsk) {
    match mem.record_ask(&scope(), ask).await.unwrap() {
        RecordAskResult::Stored(_) => {}
        other => panic!("expected Stored, got {other:?}"),
    }
}

async fn traced(
    mem: &Memory<InMemoryStore, FakeEmbedder>,
    q: &AskQuery,
) -> (Outcome, RetrieveTrace) {
    mem.retrieve_traced(&scope(), q).await.unwrap()
}

#[tokio::test]
async fn traced_outcome_matches_plain_retrieve() {
    let mem = memory(Policy::default());
    store(&mem, graded("e1", "6", "schema-v1")).await;
    let q = query("6", "schema-v1");

    let plain = mem.retrieve(&scope(), &q).await.unwrap();
    let (out, trace) = traced(&mem, &q).await;

    assert!(matches!(plain, Outcome::VerifiedReuse { .. }));
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));
    assert_eq!(trace.path, Some(PathTaken::VerifiedReuse));
    assert_eq!(trace.miss_reason, None);
    assert_eq!(trace.store_rows, 1);
    assert!(trace.store_has_eligible);

    let c = &trace.candidates[0];
    assert_eq!(c.eval_id, "e1");
    assert_eq!(c.selected, Some(PathTaken::VerifiedReuse));
    assert_eq!(c.reuse_reject, None);
    assert!(c.similarity.is_some_and(|s| s >= 0.92));
    assert!(c.signature_diff.is_empty());
    assert!(c.fingerprint_diff.is_empty());
    assert!(c.table_diff.is_empty());
}

#[tokio::test]
async fn empty_store_trace_reports_miss_without_candidates() {
    let mem = memory(Policy::default());
    let (out, trace) = traced(&mem, &query("6", "schema-v1")).await;

    assert!(matches!(out, Outcome::Miss { reason: MissReason::EmptyStore }));
    assert_eq!(trace.miss_reason, Some(MissReason::EmptyStore));
    assert!(!trace.store_has_eligible);
    assert!(trace.candidates.is_empty());
    assert_eq!(trace.path, None);
}

#[tokio::test]
async fn window_mismatch_reports_signature_field_diff() {
    let mem = memory(Policy::default());
    store(&mem, graded("e1", "6", "schema-v1")).await;

    let (out, trace) = traced(&mem, &query("3", "schema-v1")).await;
    assert!(matches!(out, Outcome::FewShot { .. }));
    assert_eq!(trace.path, Some(PathTaken::FewShot));

    let c = &trace.candidates[0];
    assert_eq!(c.reuse_reject, Some(RejectReason::SignatureMismatch));
    assert_eq!(c.few_shot_reject, None);
    assert_eq!(c.selected, Some(PathTaken::FewShot));

    let window = c.signature_diff.iter().find(|d| d.field == "window").expect("window diff");
    assert_eq!(window.ask, "\"last 3 months\"");
    assert_eq!(window.stored, "\"last 6 months\"");
    assert!(c.signature_diff.iter().all(|d| d.field == "window"));
}

#[tokio::test]
async fn fingerprint_mismatch_reports_per_table_diff() {
    let mem = memory(Policy::default());
    store(&mem, graded("e1", "6", "schema-v1")).await;

    let (out, trace) = traced(&mem, &query("6", "schema-v2")).await;
    assert!(matches!(out, Outcome::FewShot { .. }));

    let c = &trace.candidates[0];
    assert_eq!(c.reuse_reject, Some(RejectReason::FingerprintMismatch));
    assert_eq!(c.fingerprint_diff.len(), 1);
    let d = &c.fingerprint_diff[0];
    assert_eq!(d.table, "sales.public.orders");
    assert_eq!(d.ask.as_deref(), Some("schema-v2"));
    assert_eq!(d.stored.as_deref(), Some("schema-v1"));
}

#[tokio::test]
async fn few_shot_overflow_is_marked_truncated() {
    let policy = Policy { few_shot_limit: 2, ..Policy::default() };
    let mem = memory(policy);
    for i in 0..4 {
        let mut g = graded(&format!("e{i}"), "6", "schema-v1");
        g.sql = format!("{} LIMIT {}", g.sql, i + 10);
        store(&mem, g).await;
    }

    // Window mismatch blocks reuse for every row, so all four rank for few-shot.
    let (out, trace) = traced(&mem, &query("3", "schema-v1")).await;
    let Outcome::FewShot { examples } = out else {
        panic!("expected FewShot");
    };
    assert_eq!(examples.len(), 2);

    let selected =
        trace.candidates.iter().filter(|c| c.selected == Some(PathTaken::FewShot)).count();
    let truncated = trace
        .candidates
        .iter()
        .filter(|c| c.few_shot_reject == Some(RejectReason::TruncatedFromFewShot))
        .count();
    assert_eq!(selected, 2);
    assert_eq!(truncated, 2);
}

#[tokio::test]
async fn second_reuse_passer_is_marked_lower_ranked() {
    let mem = memory(Policy::default());
    store(&mem, graded("e1", "6", "schema-v1")).await;
    let mut e2 = graded("e2", "6", "schema-v1");
    e2.sql = format!("{} ORDER BY 1", e2.sql);
    store(&mem, e2).await;

    let (out, trace) = traced(&mem, &query("6", "schema-v1")).await;
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));
    assert_eq!(trace.path, Some(PathTaken::VerifiedReuse));

    let winners =
        trace.candidates.iter().filter(|c| c.selected == Some(PathTaken::VerifiedReuse)).count();
    let lower = trace
        .candidates
        .iter()
        .filter(|c| c.reuse_reject == Some(RejectReason::LowerRankedReuse))
        .count();
    assert_eq!(winners, 1);
    assert_eq!(lower, 1);
    assert!(trace
        .candidates
        .iter()
        .all(|c| c.few_shot_reject == Some(RejectReason::ReusedInstead)));
}

#[tokio::test]
async fn reuse_disabled_reports_reason_and_falls_back_to_few_shot() {
    let policy = Policy { verified_reuse_enabled: false, ..Policy::default() };
    let mem = memory(policy);
    store(&mem, graded("e1", "6", "schema-v1")).await;

    let (out, trace) = traced(&mem, &query("6", "schema-v1")).await;
    assert!(matches!(out, Outcome::FewShot { .. }));
    let c = &trace.candidates[0];
    assert_eq!(c.reuse_reject, Some(RejectReason::ReuseDisabled));
    assert_eq!(c.selected, Some(PathTaken::FewShot));
}

#[tokio::test]
async fn trace_serializes_to_json_for_logs() {
    let mem = memory(Policy::default());
    store(&mem, graded("e1", "6", "schema-v1")).await;
    let (_, trace) = traced(&mem, &query("3", "schema-v1")).await;

    let json = serde_json::to_value(&trace).unwrap();
    assert_eq!(json["path"], "FewShot");
    assert_eq!(json["candidates"][0]["reuse_reject"], "signature_mismatch");
    assert_eq!(json["candidates"][0]["signature_diff"][0]["field"], "window");

    let counts = trace.reject_counts();
    assert_eq!(counts.get("reuse:signature_mismatch"), Some(&1));
}
