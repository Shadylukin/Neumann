// SPDX-License-Identifier: MIT OR Apache-2.0
//! Relational row formatting.

#![allow(clippy::option_if_let_else)]
#![allow(clippy::format_collect)]

use crate::output::TableBuilder;
use crate::style::{styled, Theme};
use relational_engine::Row;

/// Formats relational rows as a styled table.
#[must_use]
pub fn format_rows(rows: &[Row], theme: &Theme) -> String {
    if rows.is_empty() {
        return styled("(0 rows)", theme.muted);
    }

    // Get column names from first row
    let columns: Vec<&String> = rows[0].values.iter().map(|(k, _)| k).collect();
    if columns.is_empty() {
        return styled("(0 rows)", theme.muted);
    }

    let mut builder = TableBuilder::new();

    // Add headers
    let headers: Vec<&str> = columns.iter().map(|c| c.as_str()).collect();
    builder.add_header(headers);

    // Add rows
    for row in rows {
        let row_values: Vec<String> = columns
            .iter()
            .map(|col| format_value(row.get(col), theme))
            .collect();
        builder.add_row(row_values);
    }

    builder.build_with_count(
        theme,
        rows.len(),
        if rows.len() == 1 { "row" } else { "rows" },
    )
}

/// Formats a single value with appropriate styling.
fn format_value(value: Option<&relational_engine::Value>, theme: &Theme) -> String {
    match value {
        None => styled("NULL", theme.null),
        Some(v) => match v {
            relational_engine::Value::Null => styled("NULL", theme.null),
            relational_engine::Value::Int(i) => styled(i, theme.number),
            relational_engine::Value::Float(f) => styled(format!("{f:.4}"), theme.number),
            relational_engine::Value::String(s) => styled(format!("\"{s}\""), theme.string),
            relational_engine::Value::Bool(b) => styled(b, theme.keyword),
            relational_engine::Value::Bytes(bytes) => {
                if bytes.len() <= 16 {
                    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                    styled(format!("0x{hex}"), theme.muted)
                } else {
                    styled(format!("<{} bytes>", bytes.len()), theme.muted)
                }
            },
            relational_engine::Value::Json(j) => styled(j, theme.string),
            _ => styled("?", theme.muted),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_rows_empty() {
        let theme = Theme::plain();
        let result = format_rows(&[], &theme);
        assert!(result.contains("0 rows"));
    }

    #[test]
    fn test_format_value_null() {
        let theme = Theme::plain();
        let result = format_value(None, &theme);
        assert!(result.contains("NULL"));
    }

    #[test]
    fn test_format_value_int() {
        let theme = Theme::plain();
        let result = format_value(Some(&relational_engine::Value::Int(42)), &theme);
        assert!(result.contains("42"));
    }

    #[test]
    fn test_format_value_string() {
        let theme = Theme::plain();
        let result = format_value(
            Some(&relational_engine::Value::String("hello".to_string())),
            &theme,
        );
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_format_value_bool() {
        let theme = Theme::plain();
        let result = format_value(Some(&relational_engine::Value::Bool(true)), &theme);
        assert!(result.contains("true"));
    }

    #[test]
    fn test_format_value_bytes_short() {
        let theme = Theme::plain();
        let result = format_value(
            Some(&relational_engine::Value::Bytes(vec![0xDE, 0xAD])),
            &theme,
        );
        assert!(result.contains("0x") || result.contains("dead"));
    }

    #[test]
    fn test_format_value_bytes_long() {
        let theme = Theme::plain();
        let result = format_value(
            Some(&relational_engine::Value::Bytes(vec![0u8; 32])),
            &theme,
        );
        assert!(result.contains("32 bytes"));
    }
}
