// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Handlers for relational engine browsing with Memoria design system.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use maud::{html, Markup};
use serde::Deserialize;

use relational_engine::{Condition, Value};

use crate::web::templates::layout;
use crate::web::templates::layout::{
    format_number, m_breadcrumb, m_empty, m_expandable_json, m_expandable_string, m_header,
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
        (m_header("TABLES", Some("Browse relational data structures")))

        @if tables.is_empty() {
            (m_empty("NO TABLES", "Create a table to initialize storage"))
        } @else {
            div class="m-card" {
                div class="m-card-header" { "TABLE REGISTRY" }
                div class="m-card-content p-0" {
                    table class="m-table" {
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
                                        a href=(format!("/relational/{table}")) class="text-white hover:text-neutral-300" {
                                            (table)
                                        }
                                    }
                                    td class="font-mono text-neutral-300" { (format_number(row_count)) }
                                    td class="font-mono" { (col_count) }
                                }
                            }
                        }
                    }
                }
            }

            div class="mt-4 text-sm text-neutral-400" {
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
        (m_breadcrumb(&[("/relational", "TABLES"), ("", &table)]))

        (m_header(&table.to_uppercase(), Some(&format!("{} records", format_number(row_count)))))

        @if let Ok(schema) = schema {
            // Schema section
            div class="mb-6" {
                div class="m-card" {
                    div class="m-card-header" { "SCHEMA DEFINITION" }
                    div class="m-card-content p-0" {
                        table class="m-table" {
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
                                        td class="text-white" { (col.name.clone()) }
                                        td {
                                            span class="text-neutral-300" {
                                                (format!("{:?}", col.column_type))
                                            }
                                        }
                                        td class="text-neutral-400" {
                                            @if col.nullable { "YES" } @else { "NO" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Indexes section
            @let hash_indexes = ctx.relational.get_indexed_columns(&table);
            @let btree_indexes = ctx.relational.get_btree_indexed_columns(&table);
            div class="mb-6" {
                div class="m-card" {
                    div class="m-card-header" { "INDEXES" }
                    div class="m-card-content" {
                        @if hash_indexes.is_empty() && btree_indexes.is_empty() {
                            span class="text-neutral-500 text-sm" { "No indexes defined" }
                        } @else {
                            ul class="space-y-2" {
                                @for col in &hash_indexes {
                                    li class="flex justify-between" {
                                        span class="text-white font-mono" { (col) }
                                        span class="text-neutral-400 text-xs" { "HASH" }
                                    }
                                }
                                @for col in &btree_indexes {
                                    li class="flex justify-between" {
                                        span class="text-white font-mono" { (col) }
                                        span class="text-neutral-400 text-xs" { "BTREE" }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Browse rows link
            div {
                a href=(format!("/relational/{}/rows", table)) class="m-btn" {
                    "[ BROWSE ROWS ]"
                }
            }
        } @else {
            (m_empty("TABLE NOT FOUND", "The requested table does not exist in storage"))
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
        (m_breadcrumb(&[("/relational", "TABLES"), (&format!("/relational/{table}"), &table), ("", "ROWS")]))

        (m_header(&format!("{} - ROWS", table.to_uppercase()), Some(&format!("{} total records", format_number(total_rows)))))

        @if let Ok(schema) = schema {
            @let rows = ctx.relational.select(&table, Condition::True)
                .map(|r| r.into_iter().skip(offset).take(page_size).collect::<Vec<_>>())
                .unwrap_or_default();

            @if rows.is_empty() && total_rows == 0 {
                (m_empty("NO RECORDS", "This table contains no data"))
            } @else {
                div class="m-card" {
                    div class="m-card-header" { "DATA RECORDS" }
                    div class="m-card-content p-0 overflow-x-auto" {
                        table class="m-table min-w-max" {
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
                                        td class="text-neutral-500 font-mono" { (offset + idx + 1) }
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
            (m_empty("TABLE NOT FOUND", "The requested table does not exist"))
        }
    };

    layout(&format!("{table} Rows"), NavItem::Relational, content)
}

fn render_value(value: Option<&Value>) -> Markup {
    match value {
        None => html! { span class="text-neutral-500 italic" { "null" } },
        Some(Value::Null) => html! { span class="text-neutral-500 italic" { "null" } },
        Some(Value::Int(v)) => html! { span class="text-white font-mono" { (v) } },
        Some(Value::Float(v)) => {
            html! { span class="text-white font-mono" { (format!("{v:.4}")) } }
        },
        Some(Value::String(v)) => m_expandable_string(v, 80),
        Some(Value::Bool(v)) => html! { span class="text-neutral-300" { (v) } },
        Some(Value::Bytes(v)) => {
            html! { span class="text-neutral-400" { "[" (v.len()) " bytes]" } }
        },
        Some(Value::Json(v)) => {
            let s = v.to_string();
            m_expandable_json(&s, 80)
        },
        Some(_) => html! { span class="text-neutral-400" { "?" } },
    }
}

fn pagination(page: usize, page_size: usize, total: usize, base_url: &str) -> Markup {
    let total_pages = total.div_ceil(page_size);
    let has_prev = page > 0;
    let has_next = page + 1 < total_pages;

    html! {
        div class="mt-4 flex items-center justify-between" {
            div class="text-sm text-neutral-400" {
                "SHOWING " (page * page_size + 1) " - " (((page + 1) * page_size).min(total)) " OF " (format_number(total))
            }
            div class="flex items-center gap-2" {
                @if has_prev {
                    a href=(format!("{base_url}?page={}&page_size={page_size}", page - 1))
                      class="m-btn text-sm" {
                        "[ PREV ]"
                    }
                }
                span class="px-3 py-1 text-sm text-neutral-400" {
                    "PAGE " (page + 1) " / " (total_pages.max(1))
                }
                @if has_next {
                    a href=(format!("{base_url}?page={}&page_size={page_size}", page + 1))
                      class="m-btn text-sm" {
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

    // ========== Handler integration tests ==========

    fn create_populated_relational_context() -> Arc<AdminContext> {
        use relational_engine::{Column, ColumnType, Schema};
        use std::collections::HashMap;

        let relational = Arc::new(relational_engine::RelationalEngine::new());
        let vector = Arc::new(vector_engine::VectorEngine::new());
        let graph = Arc::new(graph_engine::GraphEngine::new());

        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
            Column::new("email", ColumnType::String),
        ]);
        relational.create_table("users", schema).unwrap();

        for i in 1..=5 {
            let mut row = HashMap::new();
            row.insert("id".to_string(), Value::Int(i));
            row.insert("name".to_string(), Value::String(format!("user_{i}")));
            row.insert(
                "email".to_string(),
                Value::String(format!("user_{i}@test.com")),
            );
            relational.insert("users", row).unwrap();
        }

        // Second table
        let schema2 = Schema::new(vec![
            Column::new("key", ColumnType::String),
            Column::new("value", ColumnType::Float),
        ]);
        relational.create_table("metrics", schema2).unwrap();

        Arc::new(AdminContext::new(relational, vector, graph))
    }

    #[tokio::test]
    async fn test_tables_list_with_data() {
        let ctx = create_populated_relational_context();
        let result = tables_list(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("TABLE REGISTRY"));
        assert!(html.contains("users"));
        assert!(html.contains("metrics"));
    }

    #[tokio::test]
    async fn test_tables_list_empty() {
        let ctx = Arc::new(AdminContext::new(
            Arc::new(relational_engine::RelationalEngine::new()),
            Arc::new(vector_engine::VectorEngine::new()),
            Arc::new(graph_engine::GraphEngine::new()),
        ));
        let result = tables_list(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("NO TABLES"));
    }

    #[tokio::test]
    async fn test_table_detail_with_schema() {
        let ctx = create_populated_relational_context();
        let result = table_detail(State(ctx), Path("users".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("SCHEMA DEFINITION"));
        assert!(html.contains("COLUMN"));
        assert!(html.contains("TYPE"));
        assert!(html.contains("NULLABLE"));
        assert!(html.contains("id"));
        assert!(html.contains("name"));
        assert!(html.contains("email"));
        assert!(html.contains("BROWSE ROWS"));
    }

    #[tokio::test]
    async fn test_table_detail_not_found() {
        let ctx = create_populated_relational_context();
        let result = table_detail(State(ctx), Path("nonexistent".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("TABLE NOT FOUND"));
    }

    #[tokio::test]
    async fn test_table_rows_with_data() {
        let ctx = create_populated_relational_context();
        let params = Query(PaginationParams {
            page: 0,
            page_size: 50,
        });
        let result = table_rows(State(ctx), Path("users".to_string()), params).await;
        let html = result.into_string();
        assert!(html.contains("ROWS"));
        assert!(html.contains("user_1"));
        assert!(html.contains("user_5"));
    }

    #[tokio::test]
    async fn test_table_rows_pagination() {
        let ctx = create_populated_relational_context();
        let params = Query(PaginationParams {
            page: 0,
            page_size: 2,
        });
        let result = table_rows(State(ctx), Path("users".to_string()), params).await;
        let html = result.into_string();
        assert!(html.contains("ROWS"));
        assert!(html.contains("NEXT"));
    }

    #[tokio::test]
    async fn test_table_rows_empty_table() {
        let ctx = create_populated_relational_context();
        let params = Query(PaginationParams {
            page: 0,
            page_size: 50,
        });
        let result = table_rows(State(ctx), Path("metrics".to_string()), params).await;
        let html = result.into_string();
        assert!(html.contains("NO RECORDS"));
    }

    #[tokio::test]
    async fn test_table_detail_no_indexes() {
        let ctx = create_populated_relational_context();
        let result = table_detail(State(ctx), Path("users".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("INDEXES"));
        assert!(html.contains("No indexes defined"));
    }

    #[tokio::test]
    async fn test_table_detail_with_indexes() {
        use relational_engine::{Column, ColumnType, Schema};
        use std::collections::HashMap;

        let relational = Arc::new(relational_engine::RelationalEngine::new());
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        relational
            .create_table("indexed_tbl", schema)
            .expect("create");

        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(1));
        row.insert("name".to_string(), Value::String("alice".into()));
        relational.insert("indexed_tbl", row).expect("insert");

        relational
            .create_index("indexed_tbl", "name")
            .expect("create index");

        let mut ctx = AdminContext::new(
            relational,
            Arc::new(vector_engine::VectorEngine::new()),
            Arc::new(graph_engine::GraphEngine::new()),
        );
        ctx.blob = None;
        ctx.checkpoint = None;
        ctx.store = None;
        let ctx = Arc::new(ctx);

        let result = table_detail(State(ctx), Path("indexed_tbl".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("INDEXES"));
        assert!(html.contains("name"));
        assert!(html.contains("HASH"));
    }
}
