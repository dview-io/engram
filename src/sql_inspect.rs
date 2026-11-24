//! SQL inspection for reuse safety (absolute date literals, etc.).

use std::collections::HashSet;
use std::ops::ControlFlow;

use sqlparser::ast::{Expr, ObjectName, Query, Value as AstValue, Visit, Visitor};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::types::TableRef;

/// Returns true when the SQL contains absolute date / timestamp string literals
/// that would make "last N months" style questions unsafe to replay.
///
/// Unparseable SQL is treated as unsafe (`true`) so the ask can still be stored
/// for few-shot, but never verified-reused.
pub fn sql_has_absolute_dates(sql: &str) -> bool {
    let dialect = GenericDialect {};
    let Ok(statements) = Parser::parse_sql(&dialect, sql) else {
        return true;
    };
    let mut visitor = AbsoluteDateVisitor { found: false };
    for statement in &statements {
        if let ControlFlow::Break(()) = statement.visit(&mut visitor) {
            break;
        }
    }
    visitor.found
}

struct AbsoluteDateVisitor {
    found: bool,
}

impl Visitor for AbsoluteDateVisitor {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
        match expr {
            Expr::Value(v) => {
                if literal_looks_like_absolute_date(&v.value) {
                    self.found = true;
                    return ControlFlow::Break(());
                }
            }
            Expr::TypedString(ts) => {
                if literal_looks_like_absolute_date(&ts.value.value) {
                    self.found = true;
                    return ControlFlow::Break(());
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }
}

fn literal_looks_like_absolute_date(value: &AstValue) -> bool {
    match value {
        AstValue::SingleQuotedString(s)
        | AstValue::DoubleQuotedString(s)
        | AstValue::NationalStringLiteral(s)
        | AstValue::HexStringLiteral(s) => text_looks_like_absolute_date(s),
        AstValue::Number(_, _) => false,
        _ => false,
    }
}

/// Heuristic: ISO-like dates (`2024-01-15`) or timestamps with a calendar date prefix.
fn text_looks_like_absolute_date(text: &str) -> bool {
    let t = text.trim();
    looks_like_ymd(t, b'-') || looks_like_ymd(t, b'/')
}

fn looks_like_ymd(text: &str, sep: u8) -> bool {
    matches!(
        text.as_bytes().get(..10),
        Some(&[y0, y1, y2, y3, s1, m0, m1, s2, d0, d1])
            if y0.is_ascii_digit()
                && y1.is_ascii_digit()
                && y2.is_ascii_digit()
                && y3.is_ascii_digit()
                && s1 == sep
                && m0.is_ascii_digit()
                && m1.is_ascii_digit()
                && s2 == sep
                && d0.is_ascii_digit()
                && d1.is_ascii_digit()
    )
}

/// Extract lowercased, unquoted table names referenced in `FROM`/`JOIN` clauses (4y.4).
///
/// CTE names declared in `WITH` clauses are excluded. Returns an empty `Vec` if the SQL
/// cannot be parsed or references no tables.
pub fn sql_referenced_tables(sql: &str) -> Vec<String> {
    let dialect = GenericDialect {};
    let Ok(statements) = Parser::parse_sql(&dialect, sql) else {
        return Vec::new();
    };
    let mut visitor = TableRefVisitor::default();
    for statement in &statements {
        let _ = statement.visit(&mut visitor);
    }
    let mut out: Vec<String> = visitor
        .relations
        .into_iter()
        .filter(|rel| {
            // Single-part name matching a CTE alias is a CTE, not a base table.
            rel.contains('.') || !visitor.ctes.contains(rel)
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Narrow candidate `tables` (e.g. from table search) to only those referenced by `sql` (4y.4).
///
/// Normalize SQL for deduplication (4y.10): strip `--` and `/* */` comments outside string
/// literals, collapse whitespace outside string literals, lowercase outside `'...'` string
/// literals, and trim trailing `;`.
pub fn normalize_sql(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut prev_space = true; // trim leading whitespace

    while i < n {
        let c = chars[i];
        // Single-quoted string literal: preserve exact casing, whitespace, and `--` / `/*`.
        if c == '\'' {
            out.push('\'');
            i += 1;
            while i < n {
                let sc = chars[i];
                out.push(sc);
                i += 1;
                if sc == '\'' {
                    if i < n && chars[i] == '\'' {
                        // Escaped '' inside string literal.
                        out.push('\'');
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            prev_space = false;
            continue;
        }
        // Line comment: -- ... \n
        if c == '-' && i + 1 < n && chars[i + 1] == '-' {
            i += 2;
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
            continue;
        }
        // Block comment: /* ... */
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(n);
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
            continue;
        }
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
            i += 1;
            continue;
        }
        for lc in c.to_lowercase() {
            out.push(lc);
        }
        prev_space = false;
        i += 1;
    }

    out.trim().trim_end_matches(';').trim().to_string()
}

/// Stable 64-bit FNV-1a hex hash of [`normalize_sql`] (4y.10).
pub fn normalized_sql_hash(sql: &str) -> String {
    let norm = normalize_sql(sql);
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in norm.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Matches exact FQN or dot-separated suffix (`schema.table` / `table`). Falls back to
/// `candidates` unchanged when `sql` cannot be parsed or no candidate matches.
pub fn filter_tables_to_sql(candidates: &[TableRef], sql: &str) -> Vec<TableRef> {
    let referenced = sql_referenced_tables(sql);
    if referenced.is_empty() || candidates.is_empty() {
        return candidates.to_vec();
    }
    let matched: Vec<TableRef> = candidates
        .iter()
        .filter(|cand| {
            let c = cand.as_str().trim().to_lowercase();
            referenced
                .iter()
                .any(|r| c == *r || c.ends_with(&format!(".{r}")) || r.ends_with(&format!(".{c}")))
        })
        .cloned()
        .collect();
    if matched.is_empty() {
        candidates.to_vec()
    } else {
        matched
    }
}

#[derive(Default)]
struct TableRefVisitor {
    ctes: HashSet<String>,
    relations: Vec<String>,
}

impl Visitor for TableRefVisitor {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
        if let Some(with) = &query.with {
            for cte in &with.cte_tables {
                self.ctes.insert(cte.alias.name.value.trim().to_lowercase());
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
        let parts: Vec<String> = relation
            .0
            .iter()
            .filter_map(|p| p.as_ident().map(|i| i.value.trim().to_lowercase()))
            .filter(|s| !s.is_empty())
            .collect();
        if !parts.is_empty() {
            self.relations.push(parts.join("."));
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_iso_date_literal() {
        let sql = "SELECT * FROM t WHERE d >= '2024-01-01'";
        assert!(sql_has_absolute_dates(sql));
    }

    #[test]
    fn relative_current_date_is_ok() {
        let sql = "SELECT * FROM t WHERE d >= current_date - INTERVAL '6' MONTH";
        assert!(!sql_has_absolute_dates(sql));
    }

    #[test]
    fn unparseable_sql_is_treated_as_absolute() {
        let sql = "NOT VALID SQL ;;; <<<";
        assert!(sql_has_absolute_dates(sql));
    }
}
