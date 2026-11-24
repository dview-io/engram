//! Phase 4y.9 (store tiers) and 4y.10 (normalized SQL dedup) tests.

use std::sync::Arc;

use engram_dview::embedder::FakeEmbedder;
use engram_dview::{
    normalized_sql_hash, AskQuery, AskSignature, GradedAsk, InMemoryStore, Memory, Outcome,
    PathTaken, Policy, RecordAskResult, RejectReason, SchemaFingerprint, Scope, Scores, TableRef,
};

fn scope() -> Scope {
    Scope::new("tenant-4y-tiers")
}

fn memory() -> Memory<InMemoryStore, FakeEmbedder> {
    Memory::new(
        Arc::new(InMemoryStore::new()),
        Arc::new(FakeEmbedder::new(16)),
        Policy::default(),
    )
}

fn sig() -> AskSignature {
    AskSignature {
        aggregation: Some("sum".into()),
        columns: vec!["amount".into()],
        group_by: vec!["region".into()],
        ..Default::default()
    }
}

fn graded(eval_id: &str, sql: &str, scores: Scores, expert: bool, path: PathTaken) -> GradedAsk {
    GradedAsk {
        eval_id: eval_id.into(),
        conversation_id: None,
        question: "total amount by region".into(),
        tables: vec![TableRef::new("sales.orders")],
        kpis: vec![],
        ask_signature: sig(),
        schema_fingerprints: vec![(TableRef::new("sales.orders"), SchemaFingerprint::new("fp-v1"))],
        sql: sql.into(),
        pandas: None,
        scores,
        path_taken: path,
        expert,
    }
}

fn query() -> AskQuery {
    AskQuery {
        question: "total amount by region".into(),
        tables: vec![TableRef::new("sales.orders")],
        kpis: vec![],
        signature: sig(),
        fingerprints: vec![(TableRef::new("sales.orders"), SchemaFingerprint::new("fp-v1"))],
    }
}

#[tokio::test]
async fn low_final_high_sql_is_few_shot_only_never_verified_reuse() {
    let mem = memory();
    // final=50.9 (< 70.0) but sql=8.0 (>= 7.0): storable for few-shot, blocked from reuse.
    let res = mem
        .record_ask(
            &scope(),
            graded(
                "e-low-final",
                "SELECT region, SUM(amount) FROM sales.orders GROUP BY region",
                Scores {
                    final_score: Some(50.9),
                    sql_score: Some(8.0),
                },
                false,
                PathTaken::Cold,
            ),
        )
        .await
        .unwrap();
    assert!(matches!(res, RecordAskResult::Stored(_)));

    let (out, trace) = mem.retrieve_traced(&scope(), &query()).await.unwrap();
    assert!(matches!(out, Outcome::FewShot { .. }));
    assert_eq!(trace.candidates.len(), 1);
    assert_eq!(trace.candidates[0].reuse_reject, Some(RejectReason::BelowReuseScore));
}

#[tokio::test]
async fn both_scores_above_threshold_or_expert_allow_verified_reuse() {
    let mem = memory();
    mem.record_ask(
        &scope(),
        graded(
            "e-good",
            "SELECT region, SUM(amount) FROM sales.orders GROUP BY region",
            Scores {
                final_score: Some(85.0),
                sql_score: Some(8.5),
            },
            false,
            PathTaken::Cold,
        ),
    )
    .await
    .unwrap();

    let (out, trace) = mem.retrieve_traced(&scope(), &query()).await.unwrap();
    assert!(matches!(out, Outcome::VerifiedReuse { .. }));
    assert_eq!(trace.candidates[0].reuse_reject, None);

    // Expert bypasses score check.
    let mem_expert = memory();
    mem_expert
        .record_ask(
            &scope(),
            graded(
                "e-expert",
                "SELECT region, SUM(amount) FROM sales.orders GROUP BY region",
                Scores {
                    final_score: Some(55.0),
                    sql_score: Some(8.0),
                },
                true,
                PathTaken::Expert,
            ),
        )
        .await
        .unwrap();
    let (out_exp, _) = mem_expert.retrieve_traced(&scope(), &query()).await.unwrap();
    assert!(matches!(out_exp, Outcome::VerifiedReuse { .. }));
}

#[tokio::test]
async fn normalized_sql_hash_dedups_whitespace_casing_and_comments() {
    let a = "SELECT region, SUM(amount) FROM sales.orders GROUP BY region;";
    let b = "  select   region, SUM(amount) -- total\n from sales.orders /* comment */ group by region  ";
    assert_eq!(normalized_sql_hash(a), normalized_sql_hash(b));

    let mem = memory();
    let first = mem
        .record_ask(
            &scope(),
            graded(
                "e1",
                a,
                Scores {
                    final_score: Some(90.0),
                    sql_score: Some(9.0),
                },
                false,
                PathTaken::Cold,
            ),
        )
        .await
        .unwrap();
    let first_id = match first {
        RecordAskResult::Stored(r) => r.id,
        other => panic!("expected Stored, got {other:?}"),
    };

    // Second eval with equivalent SQL is deduplicated to the first record.
    let second = mem
        .record_ask(
            &scope(),
            graded(
                "e2",
                b,
                Scores {
                    final_score: Some(90.0),
                    sql_score: Some(9.0),
                },
                false,
                PathTaken::FewShot,
            ),
        )
        .await
        .unwrap();
    assert_eq!(second, RecordAskResult::Deduplicated { id: first_id });

    // Replayed VerifiedReuse turn is skipped without storing a duplicate.
    let reused = mem
        .record_ask(
            &scope(),
            graded(
                "e3",
                a,
                Scores {
                    final_score: Some(95.0),
                    sql_score: Some(9.5),
                },
                false,
                PathTaken::VerifiedReuse,
            ),
        )
        .await
        .unwrap();
    assert_eq!(reused, RecordAskResult::SkippedVerifiedReuse);
}

#[tokio::test]
async fn normalize_sql_preserves_string_literal_case_and_comment_markers() {
    // Case-sensitive string literals must NOT hash the same.
    let upper = "SELECT region, SUM(amount) FROM sales.orders WHERE status = 'Paid' GROUP BY region";
    let lower = "SELECT region, SUM(amount) FROM sales.orders WHERE status = 'paid' GROUP BY region";
    assert_ne!(normalized_sql_hash(upper), normalized_sql_hash(lower));

    // `--` and `/*` inside string literals must NOT be stripped as comments.
    let with_dash = "SELECT region, SUM(amount) FROM sales.orders WHERE code = 'a--b' GROUP BY region";
    let truncated = "SELECT region, SUM(amount) FROM sales.orders WHERE code = 'a' GROUP BY region";
    assert_ne!(normalized_sql_hash(with_dash), normalized_sql_hash(truncated));

    let with_block = "SELECT region, SUM(amount) FROM sales.orders WHERE code = 'a/*x*/b' GROUP BY region";
    let without_block = "SELECT region, SUM(amount) FROM sales.orders WHERE code = 'ab' GROUP BY region";
    assert_ne!(normalized_sql_hash(with_block), normalized_sql_hash(without_block));
}

#[tokio::test]
async fn dedup_does_not_discard_expert_or_reuse_tier_upgrade() {
    let sql = "SELECT region, SUM(amount) FROM sales.orders GROUP BY region";
    let mem = memory();

    // First row is few-shot-only (final=50.9 < 70.0, sql=8.0 >= 7.0).
    let low = mem
        .record_ask(
            &scope(),
            graded(
                "e-low",
                sql,
                Scores {
                    final_score: Some(50.9),
                    sql_score: Some(8.0),
                },
                false,
                PathTaken::Cold,
            ),
        )
        .await
        .unwrap();
    assert!(matches!(low, RecordAskResult::Stored(_)));

    // Later reuse-eligible row with the same SQL must NOT be discarded.
    let upgraded = mem
        .record_ask(
            &scope(),
            graded(
                "e-high",
                sql,
                Scores {
                    final_score: Some(90.0),
                    sql_score: Some(9.0),
                },
                false,
                PathTaken::Cold,
            ),
        )
        .await
        .unwrap();
    assert!(matches!(upgraded, RecordAskResult::Stored(_)));

    // Expert edit with the same SQL must NOT be discarded by a non-expert row.
    let expert = mem
        .record_ask(
            &scope(),
            graded(
                "e-expert",
                sql,
                Scores {
                    final_score: Some(90.0),
                    sql_score: Some(9.0),
                },
                true,
                PathTaken::Expert,
            ),
        )
        .await
        .unwrap();
    assert!(matches!(expert, RecordAskResult::Stored(_)));
}
