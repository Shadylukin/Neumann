// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Request handlers for the dystopian terminal admin UI.

use std::sync::Arc;

use axum::extract::State;
use maud::{html, Markup, PreEscaped};

use crate::web::templates::{engine_section, format_number, layout, page_header, stat_card};
use crate::web::AdminContext;
use crate::web::NavItem;

pub mod achievements;
pub mod graph;
pub mod graph_algorithms;
pub mod metrics;
pub mod relational;
pub mod vector;

/// Dashboard stats gathered from all engines.
pub struct DashboardStats {
    /// Number of relational tables.
    pub table_count: usize,
    /// Total rows across all tables.
    pub total_rows: usize,
    /// Total vector embeddings.
    pub vector_count: usize,
    /// Number of vector collections.
    pub collection_count: usize,
    /// Total graph nodes.
    pub node_count: usize,
    /// Total graph edges.
    pub edge_count: usize,
    /// Top tables with row counts.
    pub top_tables: Vec<(String, String)>,
    /// Collections with vector counts.
    pub collections: Vec<(String, String)>,
    /// Graph statistics summary.
    pub graph_summary: Vec<(String, String)>,
}

impl DashboardStats {
    fn gather(ctx: &AdminContext) -> Self {
        // Gather relational stats
        let tables = ctx.relational.list_tables();
        let table_count = tables.len();
        let mut total_rows = 0;
        let mut top_tables = Vec::new();

        for table in tables.iter().take(5) {
            let count = ctx.relational.row_count(table).unwrap_or(0);
            total_rows += count;
            top_tables.push((table.clone(), format_number(count)));
        }

        // Gather vector stats
        let collections_list = ctx.vector.list_collections();
        let collection_count = collections_list.len();
        let mut vector_count = 0;
        let mut collections = Vec::new();

        for coll in collections_list.iter().take(5) {
            let count = ctx.vector.collection_count(coll);
            vector_count += count;
            collections.push((coll.clone(), format_number(count)));
        }

        // Add default embeddings count if any exist
        let default_count = ctx.vector.count();
        if default_count > 0 {
            vector_count += default_count;
            if collections.len() < 5 {
                collections.push(("(default)".to_string(), format_number(default_count)));
            }
        }

        // Gather graph stats
        let node_count = ctx.graph.node_count();
        let edge_count = ctx.graph.edge_count();
        let graph_summary = vec![
            ("Nodes".to_string(), format_number(node_count)),
            ("Edges".to_string(), format_number(edge_count)),
        ];

        Self {
            table_count,
            total_rows,
            vector_count,
            collection_count,
            node_count,
            edge_count,
            top_tables,
            collections,
            graph_summary,
        }
    }
}

/// Dashboard handler - shows overview of all engines with terminal aesthetic.
#[allow(clippy::too_many_lines)]
pub async fn dashboard(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let stats = DashboardStats::gather(&ctx);

    let content = html! {
        (page_header("SYSTEM DASHBOARD", Some("Overview of all storage engines")))

        // System status display
        div class="terminal-panel mb-6" {
            div class="panel-header" { "SYSTEM STATUS" }
            div class="panel-content" {
                div class="grid grid-cols-2 md:grid-cols-4 gap-4 font-terminal text-sm" {
                    div {
                        span class="text-phosphor-dim" { "UPTIME: " }
                        span class="text-phosphor" { "ONLINE" }
                    }
                    div {
                        span class="text-phosphor-dim" { "MODE: " }
                        span class="text-phosphor" { "OPERATIONAL" }
                    }
                    div {
                        span class="text-phosphor-dim" { "MEMORY: " }
                        span class="text-phosphor" { "NOMINAL" }
                    }
                    div {
                        span class="text-phosphor-dim" { "DISK: " }
                        span class="text-phosphor" { "AVAILABLE" }
                    }
                }
            }
        }

        // Stats grid with terminal styling
        div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6" {
            (stat_card("TABLES", &stats.table_count.to_string(), "relational_engine", "relational"))
            (stat_card("VECTORS", &format_number(stats.vector_count), "vector_engine", "vector"))
            (stat_card("NODES", &format_number(stats.node_count), "graph_engine", "graph"))
            (stat_card("ROWS", &format_number(stats.total_rows), "total records", "relational"))
            (stat_card("COLLECTIONS", &stats.collection_count.to_string(), "configured", "vector"))
            (stat_card("EDGES", &format_number(stats.edge_count), "relationships", "graph"))
        }

        // Quick navigation shortcuts
        div class="terminal-panel mb-6" {
            div class="panel-header" { "QUICK ACCESS" }
            div class="panel-content" {
                div class="flex flex-wrap gap-2" {
                    a href="/graph" class="btn-terminal" {
                        span class="kbd-hint" { "G" }
                        " GRAPH ENGINE"
                    }
                    a href="/vector" class="btn-terminal" {
                        span class="kbd-hint" { "V" }
                        " VECTOR ENGINE"
                    }
                    a href="/relational" class="btn-terminal" {
                        span class="kbd-hint" { "R" }
                        " RELATIONAL ENGINE"
                    }
                }
            }
        }

        // Engine sections
        div class="grid grid-cols-1 lg:grid-cols-3 gap-6" {
            (engine_section("RELATIONAL", "relational", &stats.top_tables))
            (engine_section("VECTOR", "vector", &stats.collections))
            (engine_section("GRAPH", "graph", &stats.graph_summary))
        }

        // Interactive Query Terminal
        div class="terminal-panel mt-6" {
            div class="panel-header" { "QUERY TERMINAL" }
            div class="panel-content" {
                // Output area
                div id="terminal-output" class="terminal-output mb-3" {
                    div class="terminal-output-line success" { "> System initialized" }
                    div class="terminal-output-line success" { "> All engines operational" }
                    div class="terminal-output-line" { "> Type a query (Ctrl+Enter to execute)" }
                }
                // Input area - multi-line textarea
                form id="terminal-form" class="terminal-input-line" {
                    textarea
                        id="terminal-input"
                        class="terminal-input-field terminal-textarea"
                        placeholder="SELECT * FROM documents LIMIT 5"
                        autocomplete="off"
                        rows="3"
                        spellcheck="false" {}
                }
            }
            div class="panel-footer flex justify-between items-center" {
                span { "Ctrl+Enter to execute | Esc to clear | Up/Down for history" }
                span class="text-phosphor-dim" { "Multi-line supported" }
            }
        }

        // Terminal script for interactive queries
        script { (PreEscaped(r"
            const form = document.getElementById('terminal-form');
            const input = document.getElementById('terminal-input');
            const output = document.getElementById('terminal-output');

            // Command history
            let history = [];
            let historyIndex = -1;

            // Auto-resize textarea
            function autoResize() {
                input.style.height = 'auto';
                input.style.height = Math.min(input.scrollHeight, 200) + 'px';
            }
            input.addEventListener('input', autoResize);

            // Execute query function
            async function executeQuery() {
                const query = input.value.trim();
                if (!query) return;

                // Add to history
                history.push(query);
                historyIndex = history.length;

                // Show command (handle multi-line)
                const lines = query.split('\n');
                lines.forEach((line, i) => {
                    addLine((i === 0 ? '> ' : '  ') + line, 'command');
                });

                // Execute query
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query })
                    });

                    if (response.ok) {
                        const result = await response.json();
                        if (result.error) {
                            addLine('ERROR: ' + result.error, 'error');
                        } else if (result.message) {
                            addLine('OK: ' + result.message, 'success');
                        } else if (result.rows) {
                            addLine('OK: ' + result.rows.length + ' row(s)', 'success');
                            result.rows.slice(0, 10).forEach(row => {
                                addLine('  ' + JSON.stringify(row), '');
                            });
                            if (result.rows.length > 10) {
                                addLine('  ... and ' + (result.rows.length - 10) + ' more', '');
                            }
                        } else {
                            addLine('OK', 'success');
                        }
                    } else {
                        addLine('ERROR: ' + response.statusText, 'error');
                    }
                } catch (err) {
                    addLine('ERROR: ' + err.message, 'error');
                }

                input.value = '';
                autoResize();
            }

            form.addEventListener('submit', (e) => {
                e.preventDefault();
                executeQuery();
            });

            // Keyboard handling
            input.addEventListener('keydown', (e) => {
                // Ctrl+Enter or Cmd+Enter to execute
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    executeQuery();
                }
                // Up arrow at start of input for history
                else if (e.key === 'ArrowUp' && input.selectionStart === 0) {
                    e.preventDefault();
                    if (historyIndex > 0) {
                        historyIndex--;
                        input.value = history[historyIndex];
                        autoResize();
                    }
                }
                // Down arrow at end of input for history
                else if (e.key === 'ArrowDown' && input.selectionStart === input.value.length) {
                    e.preventDefault();
                    if (historyIndex < history.length - 1) {
                        historyIndex++;
                        input.value = history[historyIndex];
                    } else {
                        historyIndex = history.length;
                        input.value = '';
                    }
                    autoResize();
                }
                // Escape to clear
                else if (e.key === 'Escape') {
                    input.value = '';
                    autoResize();
                }
            });

            function addLine(text, type) {
                const line = document.createElement('div');
                line.className = 'terminal-output-line ' + (type || '');
                line.textContent = text;
                output.appendChild(line);
                output.scrollTop = output.scrollHeight;
            }
        ")) }
    };

    layout("Dashboard", NavItem::Dashboard, content)
}

/// Query request payload.
#[derive(Debug, serde::Deserialize)]
pub struct QueryRequest {
    query: String,
}

/// Query response payload.
#[derive(Debug, serde::Serialize)]
pub struct QueryResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    rows: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// Execute a query against the engines.
#[allow(clippy::too_many_lines)]
pub async fn api_query(
    State(ctx): State<Arc<AdminContext>>,
    axum::Json(req): axum::Json<QueryRequest>,
) -> axum::Json<QueryResponse> {
    let query = req.query.trim();

    // Parse and route the query
    if query.is_empty() {
        return axum::Json(QueryResponse {
            rows: None,
            error: Some("Empty query".to_string()),
            message: None,
        });
    }

    // Simple SQL-like query routing
    let query_upper = query.to_uppercase();

    if query_upper.starts_with("SELECT") {
        // Try to parse table name from "SELECT ... FROM table_name"
        if let Some(from_idx) = query_upper.find("FROM") {
            let after_from = &query[from_idx + 4..].trim_start();
            let table_name: String = after_from
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();

            if table_name.is_empty() {
                return axum::Json(QueryResponse {
                    rows: None,
                    error: Some("Could not parse table name".to_string()),
                    message: None,
                });
            }

            // Check if table exists
            let tables = ctx.relational.list_tables();
            if !tables.contains(&table_name) {
                return axum::Json(QueryResponse {
                    rows: None,
                    error: Some(format!(
                        "Table '{table_name}' not found. Available: {tables:?}"
                    )),
                    message: None,
                });
            }

            // Parse LIMIT if present
            let limit = query_upper.find("LIMIT").map_or(100, |limit_idx| {
                let after_limit = &query[limit_idx + 5..].trim_start();
                after_limit
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse::<usize>()
                    .unwrap_or(100)
            });

            // Fetch rows using select with Condition::True
            match ctx
                .relational
                .select(&table_name, relational_engine::Condition::True)
            {
                Ok(rows) => {
                    let json_rows: Vec<serde_json::Value> = rows
                        .into_iter()
                        .take(limit)
                        .map(|row| {
                            let mut obj = serde_json::Map::new();
                            obj.insert("_id".to_string(), serde_json::Value::Number(row.id.into()));
                            for (key, value) in row.values {
                                obj.insert(key, value_to_json(&value));
                            }
                            serde_json::Value::Object(obj)
                        })
                        .collect();

                    axum::Json(QueryResponse {
                        rows: Some(json_rows),
                        error: None,
                        message: None,
                    })
                },
                Err(e) => axum::Json(QueryResponse {
                    rows: None,
                    error: Some(format!("Query error: {e}")),
                    message: None,
                }),
            }
        } else {
            axum::Json(QueryResponse {
                rows: None,
                error: Some("SELECT requires FROM clause".to_string()),
                message: None,
            })
        }
    } else if query_upper.starts_with("SHOW TABLES") {
        let tables = ctx.relational.list_tables();
        let json_rows: Vec<serde_json::Value> = tables
            .into_iter()
            .map(|t| serde_json::json!({ "table_name": t }))
            .collect();

        axum::Json(QueryResponse {
            rows: Some(json_rows),
            error: None,
            message: None,
        })
    } else if query_upper.starts_with("SHOW COLLECTIONS") {
        let collections = ctx.vector.list_collections();
        let json_rows: Vec<serde_json::Value> = collections
            .into_iter()
            .map(|c| serde_json::json!({ "collection_name": c }))
            .collect();

        axum::Json(QueryResponse {
            rows: Some(json_rows),
            error: None,
            message: None,
        })
    } else if query_upper.starts_with("SHOW NODES") {
        let count = ctx.graph.node_count();
        axum::Json(QueryResponse {
            rows: None,
            error: None,
            message: Some(format!("Graph contains {count} nodes")),
        })
    } else if query_upper.starts_with("SHOW EDGES") {
        let count = ctx.graph.edge_count();
        axum::Json(QueryResponse {
            rows: None,
            error: None,
            message: Some(format!("Graph contains {count} edges")),
        })
    } else {
        axum::Json(QueryResponse {
            rows: None,
            error: Some(
                "Unsupported query. Try: SELECT * FROM <table> LIMIT n, SHOW TABLES, SHOW COLLECTIONS, SHOW NODES, SHOW EDGES".to_string()
            ),
            message: None,
        })
    }
}

/// Convert a relational Value to JSON.
fn value_to_json(value: &relational_engine::Value) -> serde_json::Value {
    match value {
        relational_engine::Value::Null => serde_json::Value::Null,
        relational_engine::Value::Bool(b) => serde_json::Value::Bool(*b),
        relational_engine::Value::Int(i) => serde_json::Value::Number((*i).into()),
        relational_engine::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        relational_engine::Value::String(s) => serde_json::Value::String(s.clone()),
        relational_engine::Value::Bytes(b) => {
            serde_json::Value::String(format!("<{} bytes>", b.len()))
        },
        relational_engine::Value::Json(j) => j.clone(),
        _ => serde_json::Value::String("<unknown>".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            auth_config: None,
            metrics: None,
        })
    }

    // === Value Conversion Tests ===

    #[test]
    fn test_value_to_json_null() {
        let result = value_to_json(&relational_engine::Value::Null);
        assert!(result.is_null());
    }

    #[test]
    fn test_value_to_json_bool_true() {
        let result = value_to_json(&relational_engine::Value::Bool(true));
        assert_eq!(result, serde_json::Value::Bool(true));
    }

    #[test]
    fn test_value_to_json_bool_false() {
        let result = value_to_json(&relational_engine::Value::Bool(false));
        assert_eq!(result, serde_json::Value::Bool(false));
    }

    #[test]
    fn test_value_to_json_int() {
        let result = value_to_json(&relational_engine::Value::Int(42));
        assert_eq!(result, serde_json::json!(42));
    }

    #[test]
    fn test_value_to_json_negative_int() {
        let result = value_to_json(&relational_engine::Value::Int(-100));
        assert_eq!(result, serde_json::json!(-100));
    }

    #[test]
    fn test_value_to_json_float() {
        let result = value_to_json(&relational_engine::Value::Float(3.15));
        if let serde_json::Value::Number(n) = result {
            assert!((n.as_f64().unwrap() - 3.15).abs() < 0.001);
        } else {
            panic!("Expected number");
        }
    }

    #[test]
    fn test_value_to_json_float_nan() {
        let result = value_to_json(&relational_engine::Value::Float(f64::NAN));
        // NaN cannot be represented in JSON, should be null
        assert!(result.is_null());
    }

    #[test]
    fn test_value_to_json_string() {
        let result = value_to_json(&relational_engine::Value::String("hello".to_string()));
        assert_eq!(result, serde_json::json!("hello"));
    }

    #[test]
    fn test_value_to_json_empty_string() {
        let result = value_to_json(&relational_engine::Value::String(String::new()));
        assert_eq!(result, serde_json::json!(""));
    }

    #[test]
    fn test_value_to_json_bytes() {
        let result = value_to_json(&relational_engine::Value::Bytes(vec![1, 2, 3, 4, 5]));
        assert_eq!(result, serde_json::json!("<5 bytes>"));
    }

    #[test]
    fn test_value_to_json_empty_bytes() {
        let result = value_to_json(&relational_engine::Value::Bytes(vec![]));
        assert_eq!(result, serde_json::json!("<0 bytes>"));
    }

    #[test]
    fn test_value_to_json_json() {
        let json_value = serde_json::json!({"key": "value", "num": 42});
        let result = value_to_json(&relational_engine::Value::Json(json_value.clone()));
        assert_eq!(result, json_value);
    }

    // === DashboardStats Tests ===

    #[test]
    fn test_dashboard_stats_gather_empty_engines() {
        let ctx = create_test_context();
        let stats = DashboardStats::gather(&ctx);

        assert_eq!(stats.table_count, 0);
        assert_eq!(stats.total_rows, 0);
        assert_eq!(stats.collection_count, 0);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert!(stats.top_tables.is_empty());
    }

    #[test]
    fn test_dashboard_stats_gather_with_tables() {
        use relational_engine::{Column, ColumnType, Schema, Value};

        let relational = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![
            Column::new("id".to_string(), ColumnType::Int),
            Column::new("name".to_string(), ColumnType::String),
        ]);
        relational.create_table("users", schema).unwrap();

        // Insert a few rows
        for i in 0..5 {
            let values = vec![
                ("id".to_string(), Value::Int(i)),
                ("name".to_string(), Value::String(format!("user{}", i))),
            ];
            relational
                .insert("users", values.into_iter().collect())
                .unwrap();
        }

        let ctx = Arc::new(AdminContext {
            relational,
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            auth_config: None,
            metrics: None,
        });

        let stats = DashboardStats::gather(&ctx);

        assert_eq!(stats.table_count, 1);
        assert_eq!(stats.total_rows, 5);
        assert_eq!(stats.top_tables.len(), 1);
        assert_eq!(stats.top_tables[0].0, "users");
    }

    #[test]
    fn test_dashboard_stats_gather_with_vectors() {
        let vector = Arc::new(VectorEngine::new());
        vector
            .create_collection("embeddings", Default::default())
            .unwrap();
        vector
            .store_in_collection("embeddings", "v1", vec![1.0, 0.0, 0.0])
            .unwrap();
        vector
            .store_in_collection("embeddings", "v2", vec![0.0, 1.0, 0.0])
            .unwrap();

        let ctx = Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector,
            graph: Arc::new(GraphEngine::new()),
            auth_config: None,
            metrics: None,
        });

        let stats = DashboardStats::gather(&ctx);

        assert_eq!(stats.collection_count, 1);
        assert_eq!(stats.vector_count, 2);
        assert_eq!(stats.collections.len(), 1);
    }

    #[test]
    fn test_dashboard_stats_gather_with_graph() {
        let graph = Arc::new(GraphEngine::new());
        let n1 = graph.create_node("Person", Default::default()).unwrap();
        let n2 = graph.create_node("Person", Default::default()).unwrap();
        graph
            .create_edge(n1, n2, "KNOWS", Default::default(), true)
            .unwrap();

        let ctx = Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph,
            auth_config: None,
            metrics: None,
        });

        let stats = DashboardStats::gather(&ctx);

        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
        assert_eq!(stats.graph_summary.len(), 2);
    }

    // === QueryRequest/Response Tests ===

    #[test]
    fn test_query_request_deserialize() {
        let json = r#"{"query": "SELECT * FROM users"}"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "SELECT * FROM users");
    }

    #[test]
    fn test_query_response_serialize_rows() {
        let response = QueryResponse {
            rows: Some(vec![
                serde_json::json!({"id": 1}),
                serde_json::json!({"id": 2}),
            ]),
            error: None,
            message: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("rows"));
        assert!(!json.contains("error"));
        assert!(!json.contains("message"));
    }

    #[test]
    fn test_query_response_serialize_error() {
        let response = QueryResponse {
            rows: None,
            error: Some("Table not found".to_string()),
            message: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("rows"));
        assert!(json.contains("Table not found"));
    }

    #[test]
    fn test_query_response_serialize_message() {
        let response = QueryResponse {
            rows: None,
            error: None,
            message: Some("Operation completed".to_string()),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Operation completed"));
    }

    // === API Query Handler Tests ===

    #[tokio::test]
    async fn test_api_query_empty() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.error.is_some());
        assert!(response.0.error.unwrap().contains("Empty"));
    }

    #[tokio::test]
    async fn test_api_query_whitespace_only() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "   \n\t  ".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.error.is_some());
    }

    #[tokio::test]
    async fn test_api_query_show_tables_empty() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SHOW TABLES".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.rows.is_some());
        assert!(response.0.rows.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_api_query_show_collections_empty() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SHOW COLLECTIONS".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.rows.is_some());
        assert!(response.0.rows.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_api_query_show_nodes() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SHOW NODES".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.message.is_some());
        assert!(response.0.message.unwrap().contains("0 nodes"));
    }

    #[tokio::test]
    async fn test_api_query_show_edges() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SHOW EDGES".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.message.is_some());
        assert!(response.0.message.unwrap().contains("0 edges"));
    }

    #[tokio::test]
    async fn test_api_query_unsupported() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "DROP TABLE users".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.error.is_some());
        assert!(response.0.error.unwrap().contains("Unsupported"));
    }

    #[tokio::test]
    async fn test_api_query_select_no_from() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SELECT *".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.error.is_some());
        assert!(response.0.error.unwrap().contains("FROM"));
    }

    #[tokio::test]
    async fn test_api_query_select_table_not_found() {
        let ctx = create_test_context();
        let req = QueryRequest {
            query: "SELECT * FROM nonexistent".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.error.is_some());
        assert!(response.0.error.unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn test_api_query_select_success() {
        use relational_engine::{Column, ColumnType, Schema, Value};

        let relational = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![
            Column::new("id".to_string(), ColumnType::Int),
            Column::new("name".to_string(), ColumnType::String),
        ]);
        relational.create_table("test_table", schema).unwrap();
        relational
            .insert(
                "test_table",
                [
                    ("id".to_string(), Value::Int(1)),
                    ("name".to_string(), Value::String("test".to_string())),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let ctx = Arc::new(AdminContext {
            relational,
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            auth_config: None,
            metrics: None,
        });

        let req = QueryRequest {
            query: "SELECT * FROM test_table".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.rows.is_some());
        let rows = response.0.rows.unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn test_api_query_select_with_limit() {
        use relational_engine::{Column, ColumnType, Schema, Value};

        let relational = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("id".to_string(), ColumnType::Int)]);
        relational.create_table("numbers", schema).unwrap();

        for i in 0..20 {
            relational
                .insert(
                    "numbers",
                    [("id".to_string(), Value::Int(i))].into_iter().collect(),
                )
                .unwrap();
        }

        let ctx = Arc::new(AdminContext {
            relational,
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            auth_config: None,
            metrics: None,
        });

        let req = QueryRequest {
            query: "SELECT * FROM numbers LIMIT 5".to_string(),
        };

        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.rows.is_some());
        let rows = response.0.rows.unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[tokio::test]
    async fn test_api_query_case_insensitive() {
        let ctx = create_test_context();

        // Test lowercase
        let req = QueryRequest {
            query: "show tables".to_string(),
        };
        let response = api_query(State(ctx.clone()), axum::Json(req)).await;
        assert!(response.0.rows.is_some());

        // Test mixed case
        let req = QueryRequest {
            query: "Show Tables".to_string(),
        };
        let response = api_query(State(ctx), axum::Json(req)).await;
        assert!(response.0.rows.is_some());
    }
}
