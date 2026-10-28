//! Table definitions for host reconcile and local test apply.
//!
//! `apply_schema` is a **test / standalone helper**. Production install goes through
//! the host reconcile using [`ask_table_def`] / [`repair_table_def`] (JSONB columns,
//! no HNSW — types the host schema model already understands).

use crate::error::{Error, Result};

pub const ASK_TABLE: &str = "engram_ask_records";
pub const REPAIR_TABLE: &str = "engram_repair_records";
pub const DEFAULT_EMBEDDING_DIM: usize = 32;

/// Neutral SQL type vocabulary for host translation (maps onto dlens `SqlType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngramSqlType {
    Text,
    Integer,
    BigInt,
    Boolean,
    Uuid,
    Jsonb,
    Vector(u32),
}

impl EngramSqlType {
    pub fn to_ddl(self) -> String {
        match self {
            Self::Text => "text".into(),
            Self::Integer => "integer".into(),
            Self::BigInt => "bigint".into(),
            Self::Boolean => "boolean".into(),
            Self::Uuid => "uuid".into(),
            Self::Jsonb => "jsonb".into(),
            Self::Vector(n) => format!("vector({n})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramColumn {
    pub name: &'static str,
    pub ty: EngramSqlType,
    pub nullable: bool,
    pub default_sql: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramUnique {
    pub name: &'static str,
    pub columns: &'static [&'static str],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramIndex {
    pub name: &'static str,
    pub columns: &'static [&'static str],
    pub unique: bool,
    pub predicate: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramTableDef {
    pub name: &'static str,
    pub columns: Vec<EngramColumn>,
    /// Primary key columns (maps to host `Table.primary_key`).
    pub primary_key: &'static [&'static str],
    pub uniques: Vec<EngramUnique>,
    pub indexes: Vec<EngramIndex>,
    /// Host must ensure the `vector` extension exists before creating this table.
    pub requires_vector: bool,
}

impl EngramTableDef {
    pub fn to_create_sql(&self, schema: &str) -> Result<String> {
        validate_ident(schema)?;
        if self.primary_key.is_empty() {
            return Err(Error::Invalid(format!(
                "table `{}` must declare a primary key",
                self.name
            )));
        }
        let table = qualified(schema, self.name);
        let mut cols = Vec::new();
        for c in &self.columns {
            let mut line = format!("    {} {}", c.name, c.ty.to_ddl());
            if !c.nullable {
                line.push_str(" NOT NULL");
            }
            if let Some(d) = c.default_sql {
                line.push_str(" DEFAULT ");
                line.push_str(d);
            }
            cols.push(line);
        }
        cols.push(format!("    PRIMARY KEY ({})", self.primary_key.join(", ")));
        for u in &self.uniques {
            cols.push(format!("    CONSTRAINT {} UNIQUE ({})", u.name, u.columns.join(", ")));
        }
        Ok(format!("CREATE TABLE IF NOT EXISTS {table} (\n{}\n)", cols.join(",\n")))
    }

    pub fn index_sql(&self, schema: &str) -> Result<Vec<String>> {
        validate_ident(schema)?;
        let table = qualified(schema, self.name);
        Ok(self
            .indexes
            .iter()
            .map(|i| {
                let uniq = if i.unique { "UNIQUE " } else { "" };
                let pred = i.predicate.map(|p| format!(" WHERE {p}")).unwrap_or_default();
                format!(
                    "CREATE {uniq}INDEX IF NOT EXISTS {} ON {table} ({}){pred}",
                    i.name,
                    i.columns.join(", ")
                )
            })
            .collect())
    }
}

fn col(
    name: &'static str,
    ty: EngramSqlType,
    nullable: bool,
    default_sql: Option<&'static str>,
) -> EngramColumn {
    EngramColumn { name, ty, nullable, default_sql }
}

/// Ask-record table definition (JSONB for tables/kpis; BigInt epoch ms).
pub fn ask_table_def(embedding_dim: u32) -> EngramTableDef {
    EngramTableDef {
        name: ASK_TABLE,
        requires_vector: true,
        primary_key: &["id"],
        columns: vec![
            col("id", EngramSqlType::Uuid, false, None),
            col("scope", EngramSqlType::Text, false, None),
            col("eval_id", EngramSqlType::Text, false, None),
            col("conversation_id", EngramSqlType::Text, true, None),
            col("question", EngramSqlType::Text, false, None),
            col("embedding", EngramSqlType::Vector(embedding_dim), false, None),
            col("tables", EngramSqlType::Jsonb, false, Some("'[]'::jsonb")),
            col("kpis", EngramSqlType::Jsonb, false, Some("'[]'::jsonb")),
            col("ask_signature", EngramSqlType::Jsonb, false, Some("'{}'::jsonb")),
            col("schema_fingerprints", EngramSqlType::Jsonb, false, Some("'[]'::jsonb")),
            col("sql", EngramSqlType::Text, false, None),
            col("sql_has_absolute_dates", EngramSqlType::Boolean, false, Some("FALSE")),
            col("pandas", EngramSqlType::Text, true, None),
            col("scores", EngramSqlType::Jsonb, false, Some("'{}'::jsonb")),
            col("path_taken", EngramSqlType::Text, false, Some("'cold'")),
            col("expert", EngramSqlType::Boolean, false, Some("FALSE")),
            col("uses", EngramSqlType::Integer, false, Some("0")),
            col("successes", EngramSqlType::Integer, false, Some("0")),
            col("failures", EngramSqlType::Integer, false, Some("0")),
            col("thumbs_up", EngramSqlType::Integer, false, Some("0")),
            col("thumbs_down", EngramSqlType::Integer, false, Some("0")),
            col("strikes", EngramSqlType::Integer, false, Some("0")),
            col("eligible", EngramSqlType::Boolean, false, Some("TRUE")),
            col("ineligible_reason", EngramSqlType::Text, true, None),
            col("created_at", EngramSqlType::BigInt, false, None),
            col("updated_at", EngramSqlType::BigInt, false, None),
        ],
        uniques: vec![EngramUnique {
            name: "engram_ask_records_scope_eval_uidx",
            columns: &["scope", "eval_id"],
        }],
        indexes: vec![EngramIndex {
            name: "engram_ask_records_eligible_idx",
            columns: &["scope"],
            unique: false,
            predicate: Some("eligible"),
        }],
    }
}

/// Repair-record table definition.
pub fn repair_table_def() -> EngramTableDef {
    EngramTableDef {
        name: REPAIR_TABLE,
        requires_vector: false,
        primary_key: &["id"],
        columns: vec![
            col("id", EngramSqlType::Uuid, false, None),
            col("scope", EngramSqlType::Text, false, None),
            col("table_ids", EngramSqlType::Jsonb, false, None),
            col("error_class", EngramSqlType::Text, false, None),
            col("failed_sql_fragment", EngramSqlType::Text, false, None),
            col("fixed_sql_fragment", EngramSqlType::Text, false, None),
            col("schema_fingerprint", EngramSqlType::Text, false, None),
            col("times_seen", EngramSqlType::Integer, false, Some("1")),
            col("times_fixed", EngramSqlType::Integer, false, Some("1")),
            col("eligible", EngramSqlType::Boolean, false, Some("TRUE")),
            col("created_at", EngramSqlType::BigInt, false, None),
            col("updated_at", EngramSqlType::BigInt, false, None),
        ],
        uniques: vec![EngramUnique {
            name: "engram_repair_records_lookup_uidx",
            columns: &["scope", "error_class", "schema_fingerprint", "table_ids"],
        }],
        indexes: vec![EngramIndex {
            name: "engram_repair_records_lookup_idx",
            columns: &["scope", "error_class"],
            unique: false,
            predicate: Some("eligible"),
        }],
    }
}

/// Bundle for host reconcile.
#[derive(Clone, Debug)]
pub struct EngramSchemaSpec {
    pub schema: String,
    pub embedding_dim: usize,
    pub ask: EngramTableDef,
    pub repair: EngramTableDef,
    pub required_extension: &'static str,
}

impl EngramSchemaSpec {
    pub fn new(schema: impl Into<String>, embedding_dim: usize) -> Result<Self> {
        let schema = schema.into();
        validate_ident(&schema)?;
        if embedding_dim == 0 {
            return Err(Error::Invalid("embedding_dim must be > 0".into()));
        }
        let dim = u32::try_from(embedding_dim)
            .map_err(|_| Error::Invalid("embedding_dim too large".into()))?;
        Ok(Self {
            schema,
            embedding_dim,
            ask: ask_table_def(dim),
            repair: repair_table_def(),
            required_extension: "vector",
        })
    }

    /// Table + index DDL only (extension handled separately by [`apply_schema`]).
    pub fn table_ddl(&self) -> Result<Vec<String>> {
        let mut out =
            vec![self.ask.to_create_sql(&self.schema)?, self.repair.to_create_sql(&self.schema)?];
        out.extend(self.ask.index_sql(&self.schema)?);
        out.extend(self.repair.index_sql(&self.schema)?);
        Ok(out)
    }
}

pub fn validate_ident(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(Error::Invalid("identifier must not be empty".into()));
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(Error::Invalid(format!("invalid identifier `{name}`")));
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::Invalid(format!("invalid identifier `{name}`")));
    }
    Ok(())
}

pub fn qualified(schema: &str, table: &str) -> String {
    format!("{schema}.{table}")
}

#[cfg(feature = "postgres")]
async fn vector_extension_installed(
    executor: impl sqlx::Executor<'_, Database = sqlx::Postgres>,
) -> Result<bool> {
    sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        .fetch_one(executor)
        .await
        .map_err(|e| Error::Store(e.to_string()))
}

#[cfg(feature = "postgres")]
pub async fn apply_schema(pool: &sqlx::PgPool, schema: &str, embedding_dim: usize) -> Result<()> {
    let spec = EngramSchemaSpec::new(schema, embedding_dim)?;

    // Serialize concurrent applies (tests / multi-host boot).
    let mut tx = pool.begin().await.map_err(|e| Error::Store(e.to_string()))?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtext('engram_dview_apply_schema'))")
        .execute(&mut *tx)
        .await
        .map_err(|e| Error::Store(e.to_string()))?;

    // Check before CREATE EXTENSION — a failed statement aborts the whole transaction,
    // so we cannot probe pg_extension after a failed CREATE.
    if !vector_extension_installed(&mut *tx).await? {
        match sqlx::query("CREATE EXTENSION IF NOT EXISTS vector").execute(&mut *tx).await {
            Ok(_) => {}
            Err(e) => {
                return Err(Error::VectorUnavailable(e.to_string()));
            }
        }
        if !vector_extension_installed(&mut *tx).await? {
            return Err(Error::VectorUnavailable(
                "CREATE EXTENSION vector reported success but extension is still missing".into(),
            ));
        }
    }

    sqlx::query(&format!("CREATE SCHEMA IF NOT EXISTS {schema}"))
        .execute(&mut *tx)
        .await
        .map_err(|e| Error::Store(e.to_string()))?;

    for stmt in spec.table_ddl()? {
        sqlx::query(&stmt).execute(&mut *tx).await.map_err(|e| Error::Store(e.to_string()))?;
    }

    tx.commit().await.map_err(|e| Error::Store(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_idents() {
        assert!(validate_ident("engram_test").is_ok());
        assert!(validate_ident("engram-test").is_err());
        assert!(validate_ident("engram;drop").is_err());
    }

    #[test]
    fn ask_def_uses_jsonb_and_vector() {
        let def = ask_table_def(32);
        assert_eq!(def.primary_key, &["id"]);
        assert!(def.columns.iter().any(|c| c.name == "tables" && c.ty == EngramSqlType::Jsonb));
        assert!(def
            .columns
            .iter()
            .any(|c| c.name == "embedding" && c.ty == EngramSqlType::Vector(32)));
        let ddl = def.to_create_sql("engram_test").unwrap();
        assert!(ddl.contains("PRIMARY KEY (id)"));
        assert!(ddl.contains("UNIQUE (scope, eval_id)") || ddl.contains("scope, eval_id"));
        assert!(ddl.contains("vector(32)"));
        assert!(!ddl.contains("TEXT[]"));
        assert!(!ddl.to_lowercase().contains("hnsw"));
    }

    #[test]
    fn repair_def_has_primary_key() {
        let def = repair_table_def();
        assert_eq!(def.primary_key, &["id"]);
        let ddl = def.to_create_sql("engram_test").unwrap();
        assert!(ddl.contains("PRIMARY KEY (id)"));
    }
}
