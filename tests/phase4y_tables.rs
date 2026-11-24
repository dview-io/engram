//! Phase 4y.4 & 4y.5 — SQL-derived table keys and asymmetric table filters (overlap vs subset).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::field_reassign_with_default
)]

use std::sync::Arc;

use engram_dview::{
    filter_tables_to_sql, sql_referenced_tables, AskQuery, AskSignature, FakeEmbedder, GradedAsk,
    InMemoryStore, Memory, MemoryStore, MissReason, Outcome, PathTaken, Policy, RecordAskResult,
    RejectReason, SchemaFingerprint, Scope, Scores, TableRef,
};

const QUESTION: &str = "total customer transactions by segment";

fn scope() -> Scope {
    Scope::new("tenant-a")
}

fn tbl(name: &str) -> TableRef {
    TableRef::new(name)
}

fn fp(v: &str) -> SchemaFingerprint {
    SchemaFingerprint::new(v)
}

fn sig() -> AskSignature {
    AskSignature {
        aggregation: Some("sum".into()),
        columns: vec!["amount".into()],
        group_by: vec!["segment".into()],
        ..Default::default()
    }
}

fn memory_with_store() -> (Arc<InMemoryStore>, Memory<InMemoryStore, FakeEmbedder>) {
    let store = Arc::new(InMemoryStore::new());
    let mem = Memory::new(Arc::clone(&store), Arc::new(FakeEmbedder::default()), Policy::default());
    (store, mem)
}

#[test]
fn sql_referenced_tables_extracts_from_join_and_ignores_ctes() {
    let sql = r#"
        WITH active_cust AS (
            SELECT id, segment FROM "adhoc"."uploads"."customers" WHERE active = true
        )
        SELECT c.segment, SUM(t.amount)
        FROM active_cust c
        JOIN adhoc.uploads.transactions t ON c.id = t.customer_id
        GROUP BY 1
    "#;
    assert_eq!(
        sql_referenced_tables(sql),
        vec!["adhoc.uploads.customers", "adhoc.uploads.transactions"]
    );
}

#[test]
fn filter_tables_to_sql_drops_unused_search_hits_and_falls_back_on_bad_sql() {
    let candidates = vec![
        tbl("adhoc.uploads.customers"),
        tbl("adhoc.uploads.transactions"),
        tbl("adhoc.uploads.employee"),
    ];
    let sql = "SELECT segment, COUNT(*) FROM customers GROUP BY 1";
    assert_eq!(filter_tables_to_sql(&candidates, sql), vec![tbl("adhoc.uploads.customers")]);

    // Unparseable SQL falls back to candidates unchanged.
    assert_eq!(filter_tables_to_sql(&candidates, "NOT VALID SQL ;;;"), candidates);
}

#[tokio::test]
async fn record_ask_stores_only_sql_referenced_tables_and_fingerprints() {
    let (store, mem) = memory_with_store();
    let ask = GradedAsk {
        eval_id: "e1".into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![
            tbl("adhoc.uploads.customers"),
            tbl("adhoc.uploads.transactions"),
            tbl("adhoc.uploads.employee"),
        ],
        kpis: vec![],
        ask_signature: sig(),
        schema_fingerprints: vec![
            (tbl("adhoc.uploads.customers"), fp("fp-c")),
            (tbl("adhoc.uploads.transactions"), fp("fp-t")),
            (tbl("adhoc.uploads.employee"), fp("fp-e")),
        ],
        sql: "SELECT segment, COUNT(*) FROM adhoc.uploads.customers GROUP BY 1".into(),
        pandas: None,
        scores: Scores { final_score: Some(90.0), sql_score: Some(9.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    };
    assert!(matches!(mem.record_ask(&scope(), ask).await.unwrap(), RecordAskResult::Stored(_)));

    let stored = store.get_ask_by_eval(&scope(), "e1").await.unwrap().unwrap();
    assert_eq!(stored.tables, vec![tbl("adhoc.uploads.customers")]);
    assert_eq!(stored.schema_fingerprints, vec![(tbl("adhoc.uploads.customers"), fp("fp-c"))]);
}

#[tokio::test]
async fn verified_reuse_passes_when_stored_sql_tables_are_subset_of_ask_tables() {
    let (_store, mem) = memory_with_store();
    let ask = GradedAsk {
        eval_id: "e1".into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![tbl("adhoc.uploads.customers")],
        kpis: vec![],
        ask_signature: sig(),
        schema_fingerprints: vec![(tbl("adhoc.uploads.customers"), fp("fp-c"))],
        sql: "SELECT segment, COUNT(*) FROM adhoc.uploads.customers GROUP BY 1".into(),
        pandas: None,
        scores: Scores { final_score: Some(90.0), sql_score: Some(9.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    };
    mem.record_ask(&scope(), ask).await.unwrap();

    // Ask has extra table `transactions` from table search, plus matching `customers` fingerprint.
    let q = AskQuery {
        question: QUESTION.into(),
        tables: vec![tbl("adhoc.uploads.customers"), tbl("adhoc.uploads.transactions")],
        kpis: vec![],
        signature: sig(),
        fingerprints: vec![
            (tbl("adhoc.uploads.customers"), fp("fp-c")),
            (tbl("adhoc.uploads.transactions"), fp("fp-t")),
        ],
    };
    let (out, trace) = mem.retrieve_traced(&scope(), &q).await.unwrap();
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));
    assert_eq!(trace.path, Some(PathTaken::VerifiedReuse));
}

#[tokio::test]
async fn few_shot_overlap_matches_h21_class_while_blocking_verified_reuse() {
    let (_store, mem) = memory_with_store();
    // Stored SQL joins 3 tables (customers + transactions + employee).
    let ask = GradedAsk {
        eval_id: "e1".into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![
            tbl("adhoc.uploads.customers"),
            tbl("adhoc.uploads.transactions"),
            tbl("adhoc.uploads.employee"),
        ],
        kpis: vec![],
        ask_signature: sig(),
        schema_fingerprints: vec![
            (tbl("adhoc.uploads.customers"), fp("fp-c")),
            (tbl("adhoc.uploads.transactions"), fp("fp-t")),
            (tbl("adhoc.uploads.employee"), fp("fp-e")),
        ],
        sql: "SELECT c.segment, SUM(t.amount) FROM adhoc.uploads.customers c \
              JOIN adhoc.uploads.transactions t ON c.id = t.customer_id \
              JOIN adhoc.uploads.employee e ON e.id = t.rep_id GROUP BY 1"
            .into(),
        pandas: None,
        scores: Scores { final_score: Some(90.0), sql_score: Some(9.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    };
    mem.record_ask(&scope(), ask).await.unwrap();

    // Ask only has 2 overlapping tables (missing `employee`).
    let q = AskQuery {
        question: QUESTION.into(),
        tables: vec![tbl("adhoc.uploads.customers"), tbl("adhoc.uploads.transactions")],
        kpis: vec![],
        signature: sig(),
        fingerprints: vec![
            (tbl("adhoc.uploads.customers"), fp("fp-c")),
            (tbl("adhoc.uploads.transactions"), fp("fp-t")),
        ],
    };
    let (out, trace) = mem.retrieve_traced(&scope(), &q).await.unwrap();
    assert!(matches!(out, Outcome::FewShot { .. }));
    let c = &trace.candidates[0];
    assert_eq!(c.reuse_reject, Some(RejectReason::TableNotSubset));
    assert_eq!(c.table_diff.stored_only, vec!["adhoc.uploads.employee"]);
}

#[tokio::test]
async fn zero_table_overlap_is_hard_filter_miss() {
    let (_store, mem) = memory_with_store();
    let ask = GradedAsk {
        eval_id: "e1".into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![tbl("adhoc.uploads.inventory")],
        kpis: vec![],
        ask_signature: sig(),
        schema_fingerprints: vec![(tbl("adhoc.uploads.inventory"), fp("fp-i"))],
        sql: "SELECT sku, COUNT(*) FROM adhoc.uploads.inventory GROUP BY 1".into(),
        pandas: None,
        scores: Scores { final_score: Some(90.0), sql_score: Some(9.0) },
        path_taken: PathTaken::Cold,
        expert: false,
    };
    mem.record_ask(&scope(), ask).await.unwrap();

    let q = AskQuery {
        question: QUESTION.into(),
        tables: vec![tbl("adhoc.uploads.customers")],
        kpis: vec![],
        signature: sig(),
        fingerprints: vec![(tbl("adhoc.uploads.customers"), fp("fp-c"))],
    };
    let (out, _) = mem.retrieve_traced(&scope(), &q).await.unwrap();
    assert!(matches!(out, Outcome::Miss { reason: MissReason::HardFilter }));
}
