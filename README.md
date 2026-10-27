# engram_dview

Graded text-to-SQL memory as an embeddable Rust library.

Given a sanitized question, tables/KPIs, an ask signature, and schema fingerprints, `engram_dview` returns:

- **VerifiedReuse** — replay stored SQL when gates pass  
- **FewShot** — up to N close past SQL examples  
- **Miss** — nothing usable  

It also stores graded ask records and SQL repair hints, and decays records after failed reuse, negative feedback, or table invalidation.

## Install

```toml
engram_dview = "0.1"
# optional: FakeEmbedder for out-of-tree tests
# engram_dview = { version = "0.1", features = ["test-utils"] }
```

```rust
use engram_dview::Memory;
```

The host injects an `Embedder` and a `MemoryStore` (Phase 1 ships an in-memory store; Postgres arrives in Phase 2). Ask records are unique on `(scope, eval_id)`.

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

Apache-2.0 — see `LICENSE`.
