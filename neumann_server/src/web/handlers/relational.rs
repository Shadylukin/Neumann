// SPDX-License-Identifier: MIT OR Apache-2.0
//! Handlers for relational engine browsing with dystopian terminal styling.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use maud::{html, Markup};
use serde::Deserialize;

use relational_engine::{Condition, Value};

use crate::web::templates::layout;
use crate::web::templates::layout::{
    breadcrumb, empty_state, expandable_json, expandable_string, format_number, page_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Query parameters for pagination.
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    /// Page number (zero-indexed).
    #[serde(default)]
    pub page: usize,
    /// Number of items per page.
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

const fn default_page_size() -> usize {
    50
}

/// List all tables.
pub async fn tables_list(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let tables = ctx.relational.list_tables();

    let content = html! {
        (page_header("TABLES", Some("Browse relational data structures")))

        @if tables.is_empty() {
            (empty_state("NO TABLES", "Create a table to initialize storage"))
        } @else {
            div class="terminal-panel" {
                div class="panel-header" { "TABLE REGISTRY" }
                div class="panel-content p-0" {
                    table class="table-rust" {
                        thead {
                            tr {
                                th { "TABLE" }
                                th { "ROWS" }
                                th { "COLUMNS" }
                            }
                        }
                        tbody {
                            @for table in &tables {
                                @let row_count = ctx.relational.row_count(table).unwrap_or(0);
                                @let schema = ctx.relational.get_schema(table);
                                @let col_count = schema.as_ref().map_or(0, |s| s.columns.len());
                                tr {
                                    td {
                                        a href=(format!("/relational/{table}")) class="text-phosphor hover:phosphor-glow-subtle" {
                                            (table)
                                        }
                                    }
                                    td class="font-data text-amber" { (format_number(row_count)) }
                                    td class="font-data" { (col_count) }
                                }
                            }
                        }
                    }
                }
            }

            div class="mt-4 text-sm text-phosphor-dim font-terminal" {
                "[ " (tables.len()) " TABLE(S) REGISTERED ]"
            }
        }
    };

    layout("Tables", NavItem::Relational, content)
}

/// Show table detail with schema.
pub async fn table_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path(table): Path<String>,
) -> Markup {
    let schema = ctx.relational.get_schema(&table);
    let row_count = ctx.relational.row_count(&table).unwrap_or(0);

    let content = html! {
        (breadcrumb(&[("/relational", "TABLES"), ("", &table)]))

        (page_header(&table.to_uppercase(), Some(&format!("{} records", format_number(row_count)))))

        @if let Ok(schema) = schema {
            // Schema section
            div class="mb-6" {
                div class="terminal-panel" {
                    div class="panel-header" { "SCHEMA DEFINITION" }
                    div class="panel-content p-0" {
                        table class="table-rust" {
                            thead {
                                tr {
                                    th { "COLUMN" }
                                    th { "TYPE" }
                                    th { "NULLABLE" }
                                }
                            }
                            tbody {
                                @for col in &schema.columns {
                                    tr {
                                        td class="text-phosphor" { (col.name.clone()) }
                                        td {
                                            span class="text-amber" {
                                                (format!("{:?}", col.column_type))
                                            }
                                        }
                                        td class="text-phosphor-dim" {
                                            @if col.nullable { "YES" } @else { "NO" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Browse rows link
            div {
                a href=(format!("/relational/{}/rows", table)) class="btn-terminal" {
                    "[ BROWSE ROWS ]"
                }
            }
        } @else {
            (empty_state("TABLE NOT FOUND", "The requested table does not exist in storage"))
        }
    };

    layout(&format!("Table: {table}"), NavItem::Relational, content)
}

/// Browse table rows with pagination.
pub async fn table_rows(
    State(ctx): State<Arc<AdminContext>>,
    Path(table): Path<String>,
    Query(params): Query<PaginationParams>,
) -> Markup {
    let schema = ctx.relational.get_schema(&table);
    let total_rows = ctx.relational.row_count(&table).unwrap_or(0);

    let page = params.page;
    let page_size = params.page_size.min(100);
    let offset = page * page_size;

    let content = html! {
        (breadcrumb(&[("/relational", "TABLES"), (&format!("/relational/{table}"), &table), ("", "ROWS")]))

        (page_header(&format!("{} - ROWS", table.to_uppercase()), Some(&format!("{} total records", format_number(total_rows)))))

        @if let Ok(schema) = schema {
            @let rows = ctx.relational.select(&table, Condition::True)
                .map(|r| r.into_iter().skip(offset).take(page_size).collect::<Vec<_>>())
                .unwrap_or_default();

            @if rows.is_empty() && total_rows == 0 {
                (empty_state("NO RECORDS", "This table contains no data"))
            } @else {
                div class="terminal-panel" {
                    div class="panel-header" { "DATA RECORDS" }
                    div class="panel-content p-0 overflow-x-auto" {
                        table class="table-rust min-w-max" {
                            thead {
                                tr {
                                    th { "#" }
                                    @for col in &schema.columns {
                                        th { (col.name.to_uppercase()) }
                                    }
                                }
                            }
                            tbody {
                                @for (idx, row) in rows.iter().enumerate() {
                                    tr {
                                        td class="text-phosphor-dark font-data" { (offset + idx + 1) }
                                        @for col in &schema.columns {
                                            td {
                                                (render_value(row.get(&col.name)))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Pagination
                (pagination(page, page_size, total_rows, &format!("/relational/{table}/rows")))
            }
        } @else {
            (empty_state("TABLE NOT FOUND", "The requested table does not exist"))
        }
    };

    layout(&format!("{table} Rows"), NavItem::Relational, content)
}

fn render_value(value: Option<&Value>) -> Markup {
    match value {
        None => html! { span class="text-phosphor-dark italic" { "null" } },
        Some(Value::Null) => html! { span class="text-phosphor-dark italic" { "null" } },
        Some(Value::Int(v)) => html! { span class="text-phosphor font-data" { (v) } },
        Some(Value::Float(v)) => {
            html! { span class="text-phosphor font-data" { (format!("{v:.4}")) } }
        },
        Some(Value::String(v)) => expandable_string(v, 80),
        Some(Value::Bool(v)) => html! { span class="text-amber" { (v) } },
        Some(Value::Bytes(v)) => {
            html! { span class="text-phosphor-dim" { "[" (v.len()) " bytes]" } }
        },
        Some(Value::Json(v)) => {
            let s = v.to_string();
            expandable_json(&s, 80)
        },
        Some(_) => html! { span class="text-phosphor-dim" { "?" } },
    }
}

fn pagination(page: usize, page_size: usize, total: usize, base_url: &str) -> Markup {
    let total_pages = total.div_ceil(page_size);
    let has_prev = page > 0;
    let has_next = page + 1 < total_pages;

    html! {
        div class="mt-4 flex items-center justify-between font-terminal" {
            div class="text-sm text-phosphor-dim" {
                "SHOWING " (page * page_size + 1) " - " (((page + 1) * page_size).min(total)) " OF " (format_number(total))
            }
            div class="flex items-center gap-2" {
                @if has_prev {
                    a href=(format!("{base_url}?page={}&page_size={page_size}", page - 1))
                      class="btn-terminal text-sm" {
                        "[ PREV ]"
                    }
                }
                span class="px-3 py-1 text-sm text-phosphor-dim" {
                    "PAGE " (page + 1) " / " (total_pages.max(1))
                }
                @if has_next {
                    a href=(format!("{base_url}?page={}&page_size={page_size}", page + 1))
                      class="btn-terminal text-sm" {
                        "[ NEXT ]"
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_page_size() {
        assert_eq!(default_page_size(), 50);
    }

    #[test]
    fn test_pagination_params_default() {
        let params: PaginationParams = serde_json::from_str("{}").unwrap();
        assert_eq!(params.page, 0);
        assert_eq!(params.page_size, 50);
    }

    #[test]
    fn test_pagination_params_custom() {
        let params: PaginationParams =
            serde_json::from_str(r#"{"page": 5, "page_size": 25}"#).unwrap();
        assert_eq!(params.page, 5);
        assert_eq!(params.page_size, 25);
    }

    #[test]
    fn test_render_value_none() {
        let html = render_value(None).into_string();
        assert!(html.contains("null"));
    }

    #[test]
    fn test_render_value_null() {
        let html = render_value(Some(&Value::Null)).into_string();
        assert!(html.contains("null"));
    }

    #[test]
    fn test_render_value_int() {
        let html = render_value(Some(&Value::Int(42))).into_string();
        assert!(html.contains("42"));
    }

    #[test]
    fn test_render_value_negative_int() {
        let html = render_value(Some(&Value::Int(-100))).into_string();
        assert!(html.contains("-100"));
    }

    #[test]
    fn test_render_value_float() {
        let html = render_value(Some(&Value::Float(3.1415))).into_string();
        assert!(html.contains("3.1415"));
    }

    #[test]
    fn test_render_value_string() {
        let html = render_value(Some(&Value::String("hello world".to_string()))).into_string();
        assert!(html.contains("hello world"));
    }

    #[test]
    fn test_render_value_bool_true() {
        let html = render_value(Some(&Value::Bool(true))).into_string();
        assert!(html.contains("true"));
    }

    #[test]
    fn test_render_value_bool_false() {
        let html = render_value(Some(&Value::Bool(false))).into_string();
        assert!(html.contains("false"));
    }

    #[test]
    fn test_render_value_bytes() {
        let html = render_value(Some(&Value::Bytes(vec![1, 2, 3, 4, 5]))).into_string();
        assert!(html.contains("5 bytes"));
    }

    #[test]
    fn test_render_value_json() {
        let json = serde_json::json!({"key": "value"});
        let html = render_value(Some(&Value::Json(json))).into_string();
        assert!(html.contains("key"));
    }

    #[test]
    fn test_pagination_first_page() {
        let html = pagination(0, 10, 100, "/test").into_string();
        assert!(html.contains("PAGE 1"));
        assert!(html.contains("SHOWING 1 - 10"));
        assert!(!html.contains("PREV")); // No prev on first page
        assert!(html.contains("NEXT"));
    }

    #[test]
    fn test_pagination_middle_page() {
        let html = pagination(5, 10, 100, "/test").into_string();
        assert!(html.contains("PAGE 6"));
        assert!(html.contains("PREV"));
        assert!(html.contains("NEXT"));
    }

    #[test]
    fn test_pagination_last_page() {
        let html = pagination(9, 10, 100, "/test").into_string();
        assert!(html.contains("PAGE 10"));
        assert!(html.contains("PREV"));
        assert!(!html.contains("NEXT")); // No next on last page
    }

    #[test]
    fn test_pagination_single_page() {
        let html = pagination(0, 10, 5, "/test").into_string();
        assert!(html.contains("PAGE 1 / 1"));
        assert!(!html.contains("PREV"));
        assert!(!html.contains("NEXT"));
    }

    #[test]
    fn test_pagination_empty() {
        let html = pagination(0, 10, 0, "/test").into_string();
        // Should handle edge case of 0 items
        assert!(html.contains("PAGE 1"));
    }
}
