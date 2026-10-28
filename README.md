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
# optional features:
# engram_dview = { version = "0.1", features = ["postgres", "test-utils"] }
```

```rust
use engram_dview::Memory;
```

The host injects an `Embedder` and a `MemoryStore`. Ask records are unique on `(scope, eval_id)`.

- **In-memory** store: always available (unit tests).
- **Postgres + pgvector** (`feature = "postgres"`): `PostgresStore`. Host supplies `sqlx::PgPool`.
  - Reconcile specs: `ask_table_def` / `repair_table_def` / `EngramSchemaSpec` (JSONB columns, `vector(N)`, no HNSW in v1).
  - `apply_schema` is a **test / standalone** helper with an advisory lock — production install is host reconcile.

## Postgres integration tests

1. Set in `.env` (gitignored):

```bash
ENGRAM_DATABASE_URL=postgres://root:<password>@localhost:5432/<db_name>
# or: ENGRAM_DATABASE_URL=postgresql+psycopg2://root:<password>@localhost:5432/<db_name>
```

2. Ensure the DB can use the `vector` extension.

3. Run:

```bash
cargo test
# optional: fail instead of skip when URL missing
ENGRAM_REQUIRE_DB=1 cargo test --test phase2_postgres
```

Phase 2 tests reuse a single schema (`engram_test`, override with `ENGRAM_TEST_SCHEMA`): created
only if missing, truncated between tests. Set `ENGRAM_TEST_DROP_SCHEMA=1` to drop it after each test.

## License

Apache-2.0 — see `LICENSE`.
