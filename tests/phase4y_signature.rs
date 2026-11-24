//! Phase 4y.2 & 4y.3 — structured ask signature normalization, reuse compare, and mismatch matrix.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::field_reassign_with_default
)]

use std::sync::Arc;

use engram_dview::{
    AskQuery, AskSignature, FakeEmbedder, GradedAsk, InMemoryStore, KpiRef, Memory, Outcome,
    PathTaken, Policy, RecordAskResult, RejectReason, SchemaFingerprint, Scope, Scores,
    SignatureFilter, SignatureWindow, TableRef,
};
use serde_json::json;

const SQL: &str = "SELECT region, sum(amount) FROM sales.public.orders WHERE status = 'paid' AND order_date >= current_date - INTERVAL '6' MONTH GROUP BY 1";
const QUESTION: &str = "total paid revenue by region over the last 6 months";

fn memory() -> Memory<InMemoryStore, FakeEmbedder> {
    Memory::new(
        Arc::new(InMemoryStore::new()),
        Arc::new(FakeEmbedder::default()),
        Policy::default(),
    )
}

fn scope() -> Scope {
    Scope::new("tenant-a")
}

fn table_orders() -> TableRef {
    TableRef::new("sales.public.orders")
}

fn structured_sig(window_count: u32) -> AskSignature {
    AskSignature {
        aggregation: Some("sum".into()),
        columns: vec!["amount".into()],
        group_by: vec!["region".into()],
        structured_filters: vec![SignatureFilter {
            column: "status".into(),
            op: "=".into(),
            value: json!("paid"),
        }],
        structured_window: Some(SignatureWindow {
            unit: "month".into(),
            count: Some(window_count),
            anchor: Some("now".into()),
        }),
        top_n: None,
        metric: Some("total paid revenue".into()),
        ..Default::default()
    }
}

fn graded(eval_id: &str, sig: AskSignature) -> GradedAsk {
    GradedAsk {
        eval_id: eval_id.into(),
        conversation_id: Some("c1".into()),
        question: QUESTION.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        ask_signature: sig,
        schema_fingerprints: vec![(table_orders(), SchemaFingerprint::new("fp-v1"))],
        sql: SQL.into(),
        pandas: None,
        scores: Scores { final_score: Some(88.0), sql_score: Some(8.5) },
        path_taken: PathTaken::Cold,
        expert: false,
    }
}

fn query(sig: AskSignature) -> AskQuery {
    AskQuery {
        question: QUESTION.into(),
        tables: vec![table_orders()],
        kpis: vec![KpiRef::new("revenue")],
        signature: sig,
        fingerprints: vec![(table_orders(), SchemaFingerprint::new("fp-v1"))],
    }
}

async fn store(mem: &Memory<InMemoryStore, FakeEmbedder>, ask: GradedAsk) {
    match mem.record_ask(&scope(), ask).await.unwrap() {
        RecordAskResult::Stored(_) => {}
        other => panic!("expected Stored, got {other:?}"),
    }
}

#[test]
fn normalize_sorts_lowercases_and_canonicalizes_aliases() {
    let mut sig = AskSignature {
        aggregation: Some("  Total ".into()),
        columns: vec!["Amount".into(), "  tax ".into(), "amount".into(), "".into()],
        group_by: vec!["State".into(), "Region".into(), "region".into()],
        structured_filters: vec![
            SignatureFilter { column: "Status".into(), op: "==".into(), value: json!("Paid ") },
            SignatureFilter {
                column: "Channel".into(),
                op: "IN".into(),
                value: json!(["Web", "App"]),
            },
        ],
        structured_window: Some(SignatureWindow {
            unit: "Months".into(),
            count: Some(6),
            anchor: Some(" NOW ".into()),
        }),
        top_n: Some(0),
        ..Default::default()
    };
    sig.normalize();

    assert_eq!(sig.aggregation.as_deref(), Some("sum"));
    assert_eq!(sig.columns, vec!["amount", "tax"]);
    assert_eq!(sig.group_by, vec!["region", "state"]);
    assert_eq!(sig.structured_filters.len(), 2);
    assert_eq!(sig.structured_filters[0].column, "channel");
    assert_eq!(sig.structured_filters[0].op, "in");
    assert_eq!(sig.structured_filters[0].value, json!(["app", "web"]));
    assert_eq!(sig.structured_filters[1].column, "status");
    assert_eq!(sig.structured_filters[1].op, "=");
    assert_eq!(sig.structured_filters[1].value, json!("paid"));
    assert_eq!(
        sig.structured_window,
        Some(SignatureWindow { unit: "month".into(), count: Some(6), anchor: Some("now".into()) })
    );
    assert_eq!(sig.top_n, None);
}

#[tokio::test]
async fn prose_metric_drift_still_reuses_when_structured_fields_match() {
    let mem = memory();
    store(&mem, graded("e1", structured_sig(6))).await;

    let mut q_sig = structured_sig(6);
    q_sig.metric = Some("gross revenue from completed orders".into());
    q_sig.aggregation = Some("TOTAL".into());
    q_sig.structured_window =
        Some(SignatureWindow { unit: "months".into(), count: Some(6), anchor: Some("NOW".into()) });

    let (out, trace) = mem.retrieve_traced(&scope(), &query(q_sig)).await.unwrap();
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));
    assert!(trace.candidates[0].signature_diff.is_empty());
}

#[tokio::test]
async fn answer_changing_field_mismatches_block_reuse_and_report_diff() {
    let mem = memory();
    store(&mem, graded("e1", structured_sig(6))).await;

    let cases: Vec<(&str, AskSignature)> = vec![
        ("aggregation", {
            let mut s = structured_sig(6);
            s.aggregation = Some("avg".into());
            s
        }),
        ("columns", {
            let mut s = structured_sig(6);
            s.columns = vec!["net_amount".into()];
            s
        }),
        ("group_by", {
            let mut s = structured_sig(6);
            s.group_by = vec!["country".into()];
            s
        }),
        ("structured_filters", {
            let mut s = structured_sig(6);
            s.structured_filters = vec![SignatureFilter {
                column: "status".into(),
                op: "=".into(),
                value: json!("refunded"),
            }];
            s
        }),
        ("structured_window", structured_sig(3)),
        ("top_n", {
            let mut s = structured_sig(6);
            s.top_n = Some(10);
            s
        }),
    ];

    for (expected_field, sig) in cases {
        let (out, trace) = mem.retrieve_traced(&scope(), &query(sig)).await.unwrap();
        assert!(
            matches!(out, Outcome::FewShot { .. }),
            "expected FewShot (not VerifiedReuse) when `{expected_field}` differs, got {out:?}"
        );
        let c = &trace.candidates[0];
        assert_eq!(c.reuse_reject, Some(RejectReason::SignatureMismatch));
        assert!(
            c.signature_diff.iter().any(|d| d.field == expected_field),
            "expected signature_diff to mention `{expected_field}`, got {:?}",
            c.signature_diff
        );
    }
}

#[test]
fn legacy_json_deserializes_and_never_matches_structured_signature() {
    let legacy_json = json!({
        "metric": "revenue",
        "dimensions": ["region"],
        "grain": "month",
        "window": "last 6 months",
        "filters": {},
        "top_n": null
    });
    let legacy: AskSignature = serde_json::from_value(legacy_json).unwrap();
    assert!(!legacy.has_structured_fields());
    assert!(legacy.is_reuse_ready());

    let structured = structured_sig(6);
    assert!(structured.has_structured_fields());
    assert!(!structured.matches(&legacy));
    assert!(!legacy.matches(&structured));
}
