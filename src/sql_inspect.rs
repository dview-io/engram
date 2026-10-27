//! SQL inspection for reuse safety (absolute date literals, etc.).

use sqlparser::ast::{Expr, Value as AstValue, Visit, Visitor};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::ops::ControlFlow;

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
