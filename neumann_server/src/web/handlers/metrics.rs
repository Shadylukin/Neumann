// SPDX-License-Identifier: MIT OR Apache-2.0
//! Metrics dashboard handlers for the admin UI.
//!
//! Provides real-time metrics visualization with engine health,
//! request statistics, and performance histograms.

use std::sync::Arc;
use std::time::SystemTime;

use axum::extract::State;
use maud::{html, Markup, PreEscaped};
use serde::{Deserialize, Serialize};

use crate::web::templates::{layout, m_header};
use crate::web::{AdminContext, NavItem};

/// Snapshot of current metrics for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Timestamp when snapshot was taken.
    pub timestamp: u64,
    /// Relational engine statistics.
    pub relational: EngineStats,
    /// Vector engine statistics.
    pub vector: EngineStats,
    /// Graph engine statistics.
    pub graph: EngineStats,
    /// Overall system health status.
    pub health: HealthStatus,
}

/// Statistics for a single engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineStats {
    /// Number of entities/items.
    pub count: usize,
    /// Engine-specific status.
    pub status: EngineStatus,
    /// Operations per second (estimated).
    pub ops_per_sec: f64,
}

/// Engine operational status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineStatus {
    /// Engine is healthy and operational.
    #[default]
    Healthy,
    /// Engine is degraded but functional.
    Degraded,
    /// Engine is not responding.
    Down,
}

impl EngineStatus {
    /// Returns CSS class for status indicator.
    #[must_use]
    pub const fn css_class(&self) -> &'static str {
        match self {
            Self::Healthy => "opacity-100",
            Self::Degraded => "opacity-60",
            Self::Down => "opacity-40",
        }
    }

    /// Returns display label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Degraded => "DEGRADED",
            Self::Down => "DOWN",
        }
    }
}

/// Overall system health.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// All systems operational.
    #[default]
    Operational,
    /// Some systems degraded.
    Degraded,
    /// Critical systems down.
    Critical,
}

impl HealthStatus {
    /// Returns CSS class for health indicator.
    #[must_use]
    pub const fn css_class(&self) -> &'static str {
        match self {
            Self::Operational => "text-white",
            Self::Degraded => "text-neutral-400 opacity-60",
            Self::Critical => "text-neutral-500 opacity-40",
        }
    }

    /// Returns display label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Operational => "OPERATIONAL",
            Self::Degraded => "DEGRADED",
            Self::Critical => "CRITICAL",
        }
    }
}

impl MetricsSnapshot {
    /// Gather current metrics from the admin context.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn gather(ctx: &AdminContext) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        // Gather relational stats
        let table_count = ctx.relational.list_tables().len();
        let total_rows: usize = ctx
            .relational
            .list_tables()
            .iter()
            .map(|t| ctx.relational.row_count(t).unwrap_or(0))
            .sum();

        let relational = EngineStats {
            count: total_rows,
            status: EngineStatus::Healthy,
            ops_per_sec: 0.0,
        };

        // Gather vector stats
        let vector_count = ctx.vector.count()
            + ctx
                .vector
                .list_collections()
                .iter()
                .map(|c| ctx.vector.collection_count(c))
                .sum::<usize>();

        let vector = EngineStats {
            count: vector_count,
            status: EngineStatus::Healthy,
            ops_per_sec: 0.0,
        };

        // Gather graph stats
        let graph_count = ctx.graph.node_count() + ctx.graph.edge_count();
        let graph = EngineStats {
            count: graph_count,
            status: EngineStatus::Healthy,
            ops_per_sec: 0.0,
        };

        // Determine overall health
        let health = if table_count == 0 && vector_count == 0 && graph_count == 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Operational
        };

        Self {
            timestamp,
            relational,
            vector,
            graph,
            health,
        }
    }
}

/// Metrics dashboard page handler.
pub async fn dashboard(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let snapshot = MetricsSnapshot::gather(&ctx);

    let content = html! {
        (m_header("SYSTEM METRICS", Some("Real-time performance monitoring")))

        // System health banner
        div class="m-card mb-6" {
            div class="m-card-header" { "SYSTEM HEALTH" }
            div class="m-card-content" {
                div class="flex items-center justify-between" {
                    div class="flex items-center gap-4" {
                        div class=(format!("m-dot {}", match snapshot.health {
                            HealthStatus::Operational => "opacity-100",
                            HealthStatus::Degraded => "opacity-60",
                            HealthStatus::Critical => "opacity-40",
                        })) {}
                        span class=(snapshot.health.css_class()) {
                            (snapshot.health.label())
                        }
                    }
                    span class="text-neutral-400 text-sm" {
                        "Last updated: " (format_timestamp(snapshot.timestamp))
                    }
                }
            }
        }

        // Engine status grid
        div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6" {
            (engine_status_card("RELATIONAL ENGINE", &snapshot.relational, "relational"))
            (engine_status_card("VECTOR ENGINE", &snapshot.vector, "vector"))
            (engine_status_card("GRAPH ENGINE", &snapshot.graph, "graph"))
        }

        // Metrics panels
        div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6" {
            // Request metrics
            (request_metrics_card(&ctx))

            // Latency metrics
            div class="m-card" {
                div class="m-card-header" { "LATENCY DISTRIBUTION" }
                div class="m-card-content" {
                    div class="space-y-3" {
                        (latency_bar("Query", 0.0, 100.0))
                        (latency_bar("Blob", 0.0, 100.0))
                        (latency_bar("Vector", 0.0, 100.0))
                    }
                    div class="mt-4 flex justify-between text-xs text-neutral-400" {
                        span { "0ms" }
                        span { "50ms" }
                        span { "100ms+" }
                    }
                }
            }
        }

        // Quick actions
        div class="m-card" {
            div class="m-card-header" { "QUICK ACTIONS" }
            div class="m-card-content" {
                div class="flex flex-wrap gap-2" {
                    a href="/" class="m-btn" {
                        "DASHBOARD"
                    }
                    a href="/graph/algorithms" class="m-btn" {
                        "ALGORITHMS"
                    }
                    button class="m-btn" disabled {
                        "EXPORT METRICS"
                    }
                }
            }
        }

        // Live update script placeholder
        script { (PreEscaped(r"
            // Metrics dashboard would connect to SSE endpoint for live updates
            console.log('[Metrics] Dashboard loaded');

            // Placeholder for future SSE connection
            // const evtSource = new EventSource('/api/metrics/stream');
            // evtSource.onmessage = (event) => {
            //     const data = JSON.parse(event.data);
            //     updateDashboard(data);
            // };
        ")) }
    };

    layout("Metrics", NavItem::Dashboard, content)
}

/// Render request metrics card with real data from shadow counters.
fn request_metrics_card(ctx: &AdminContext) -> Markup {
    ctx.metrics.as_ref().map_or_else(
        || {
            html! {
                div class="m-card" {
                    div class="m-card-header" { "REQUEST METRICS" }
                    div class="m-card-content" {
                        div class="space-y-3" {
                            (metric_row("Total Requests", "--", "No metrics configured"))
                            (metric_row("Success Rate", "--", ""))
                            (metric_row("Error Rate", "--", ""))
                            (metric_row("Rate Limited", "--", ""))
                        }
                        p class="text-neutral-400 text-xs mt-4" {
                            "Enable ServerMetrics for live request data"
                        }
                    }
                }
            }
        },
        |metrics| {
            let snap = metrics.counter_snapshot();
            html! {
                div class="m-card" {
                    div class="m-card-header" { "REQUEST METRICS" }
                    div class="m-card-content" {
                        div class="space-y-3" {
                            (metric_row(
                                "Total Requests",
                                &format_number(usize::try_from(snap.total).unwrap_or(usize::MAX)),
                                "all services",
                            ))
                            (metric_row(
                                "Success Rate",
                                &format!("{:.1}%", snap.success_rate()),
                                &format!("{} succeeded", snap.success),
                            ))
                            (metric_row(
                                "Error Rate",
                                &format!("{:.1}%", snap.error_rate()),
                                &format!("{} failed", snap.errors),
                            ))
                            (metric_row(
                                "Auth Failures",
                                &snap.auth_failures.to_string(),
                                "rejected",
                            ))
                            (metric_row(
                                "Rate Limited",
                                &snap.rate_limited.to_string(),
                                "throttled",
                            ))
                        }
                    }
                }
            }
        },
    )
}

/// Render an engine status card.
fn engine_status_card(name: &str, stats: &EngineStats, _engine_type: &str) -> Markup {
    html! {
        div class="m-card" {
            div class="m-card-content" {
                div class="flex items-center justify-between mb-2" {
                    span class="text-sm text-neutral-400" { (name) }
                    div class="flex items-center gap-2" {
                        div class=(format!("m-dot {}", stats.status.css_class())) {}
                        span class="text-xs" { (stats.status.label()) }
                    }
                }
                div class="text-2xl font-mono text-white" {
                    (format_number(stats.count))
                }
                div class="text-xs text-neutral-400 mt-1" {
                    "items"
                }
            }
        }
    }
}

/// Render a single metric row.
fn metric_row(label: &str, value: &str, subtitle: &str) -> Markup {
    html! {
        div class="flex justify-between items-center" {
            div {
                span class="text-sm text-neutral-400" { (label) }
            }
            div class="text-right" {
                span class="font-mono text-lg text-white" { (value) }
                @if !subtitle.is_empty() {
                    br;
                    span class="text-xs text-neutral-500" { (subtitle) }
                }
            }
        }
    }
}

/// Render a latency bar visualization.
fn latency_bar(label: &str, value: f64, max: f64) -> Markup {
    let percentage = if max > 0.0 {
        ((value / max) * 100.0).min(100.0)
    } else {
        0.0
    };

    html! {
        div {
            div class="flex justify-between mb-1" {
                span class="text-sm text-neutral-400" { (label) }
                span class="font-mono text-sm text-white" {
                    @if value > 0.0 {
                        (format!("{value:.1}ms"))
                    } @else {
                        "--"
                    }
                }
            }
            div class="h-[2px] bg-neutral-800 rounded" {
                div class="h-full bg-white transition-all duration-300 rounded"
                    style=(format!("width: {percentage}%")) {}
            }
        }
    }
}

/// Format a number with thousand separators.
fn format_number(n: usize) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format a Unix timestamp for display.
fn format_timestamp(ts: u64) -> String {
    if ts == 0 {
        return "N/A".to_string();
    }
    // Simple relative time for now
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let diff = now.saturating_sub(ts);
    if diff < 60 {
        "just now".to_string()
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else {
        format!("{}h ago", diff / 3600)
    }
}

/// API endpoint for metrics JSON snapshot.
pub async fn api_snapshot(State(ctx): State<Arc<AdminContext>>) -> axum::Json<MetricsSnapshot> {
    axum::Json(MetricsSnapshot::gather(&ctx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_status_css_class() {
        assert_eq!(EngineStatus::Healthy.css_class(), "opacity-100");
        assert_eq!(EngineStatus::Degraded.css_class(), "opacity-60");
        assert_eq!(EngineStatus::Down.css_class(), "opacity-40");
    }

    #[test]
    fn test_engine_status_label() {
        assert_eq!(EngineStatus::Healthy.label(), "HEALTHY");
        assert_eq!(EngineStatus::Degraded.label(), "DEGRADED");
        assert_eq!(EngineStatus::Down.label(), "DOWN");
    }

    #[test]
    fn test_health_status_css_class() {
        assert_eq!(HealthStatus::Operational.css_class(), "text-white");
        assert_eq!(
            HealthStatus::Degraded.css_class(),
            "text-neutral-400 opacity-60"
        );
        assert_eq!(
            HealthStatus::Critical.css_class(),
            "text-neutral-500 opacity-40"
        );
    }

    #[test]
    fn test_health_status_label() {
        assert_eq!(HealthStatus::Operational.label(), "OPERATIONAL");
        assert_eq!(HealthStatus::Degraded.label(), "DEGRADED");
        assert_eq!(HealthStatus::Critical.label(), "CRITICAL");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "N/A");
    }

    #[test]
    fn test_format_timestamp_recent() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert_eq!(format_timestamp(now), "just now");
    }

    #[test]
    fn test_engine_stats_default() {
        let stats = EngineStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.status, EngineStatus::Healthy);
        assert!((stats.ops_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_status_default() {
        let status = EngineStatus::default();
        assert_eq!(status, EngineStatus::Healthy);
    }

    #[test]
    fn test_health_status_default() {
        let health = HealthStatus::default();
        assert_eq!(health, HealthStatus::Operational);
    }

    #[test]
    fn test_metric_row_rendering() {
        let html = metric_row("Test", "42", "subtitle").into_string();
        assert!(html.contains("Test"));
        assert!(html.contains("42"));
        assert!(html.contains("subtitle"));
    }

    #[test]
    fn test_latency_bar_rendering() {
        let html = latency_bar("Query", 50.0, 100.0).into_string();
        assert!(html.contains("Query"));
        assert!(html.contains("50.0ms"));
        assert!(html.contains("50%"));
    }

    #[test]
    fn test_latency_bar_zero() {
        let html = latency_bar("Query", 0.0, 100.0).into_string();
        assert!(html.contains("--"));
        assert!(html.contains("0%"));
    }

    #[test]
    fn test_engine_status_card_rendering() {
        let stats = EngineStats {
            count: 1234,
            status: EngineStatus::Healthy,
            ops_per_sec: 100.0,
        };
        let html = engine_status_card("TEST ENGINE", &stats, "graph").into_string();
        assert!(html.contains("TEST ENGINE"));
        assert!(html.contains("1,234"));
        assert!(html.contains("HEALTHY"));
    }

    #[test]
    fn test_metrics_snapshot_serialization() {
        let snapshot = MetricsSnapshot {
            timestamp: 12345,
            relational: EngineStats::default(),
            vector: EngineStats::default(),
            graph: EngineStats::default(),
            health: HealthStatus::Operational,
        };

        let json = serde_json::to_string(&snapshot).expect("serialization failed");
        assert!(json.contains("timestamp"));
        assert!(json.contains("relational"));
        assert!(json.contains("operational"));

        let decoded: MetricsSnapshot = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.timestamp, 12345);
    }

    // ========== Handler integration tests ==========

    fn create_populated_metrics_context() -> Arc<AdminContext> {
        use relational_engine::{Column, ColumnType, Schema};
        use std::collections::HashMap;

        let relational = Arc::new(relational_engine::RelationalEngine::new());
        let vector = Arc::new(vector_engine::VectorEngine::new());
        let graph = Arc::new(graph_engine::GraphEngine::new());

        // Seed relational
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        relational.create_table("users", schema).unwrap();
        let mut row = HashMap::new();
        row.insert("id".to_string(), relational_engine::Value::Int(1));
        row.insert(
            "name".to_string(),
            relational_engine::Value::String("alice".into()),
        );
        relational.insert("users", row).unwrap();

        // Seed vector
        vector.store_embedding("v1", vec![1.0, 0.0]).unwrap();
        vector.store_embedding("v2", vec![0.0, 1.0]).unwrap();

        // Seed graph
        let n1 = graph.create_node("Person", Default::default()).unwrap();
        let n2 = graph.create_node("Person", Default::default()).unwrap();
        graph
            .create_edge(n1, n2, "KNOWS", Default::default(), true)
            .unwrap();

        Arc::new(AdminContext::new(relational, vector, graph))
    }

    #[tokio::test]
    async fn test_dashboard_with_populated_engines() {
        let ctx = create_populated_metrics_context();
        let result = dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("SYSTEM METRICS"));
        assert!(html.contains("SYSTEM HEALTH"));
        assert!(html.contains("OPERATIONAL"));
        assert!(html.contains("RELATIONAL"));
        assert!(html.contains("VECTOR"));
        assert!(html.contains("GRAPH"));
        assert!(html.contains("HEALTHY"));
    }

    #[tokio::test]
    async fn test_dashboard_empty_engines_degraded() {
        let ctx = Arc::new(AdminContext::new(
            Arc::new(relational_engine::RelationalEngine::new()),
            Arc::new(vector_engine::VectorEngine::new()),
            Arc::new(graph_engine::GraphEngine::new()),
        ));
        let result = dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("DEGRADED"));
    }

    #[tokio::test]
    async fn test_api_snapshot_with_data() {
        let ctx = create_populated_metrics_context();
        let result = api_snapshot(State(ctx)).await;
        let snapshot = result.0;
        assert!(snapshot.timestamp > 0);
        assert_eq!(snapshot.health, HealthStatus::Operational);
        assert!(snapshot.relational.count > 0);
        assert!(snapshot.vector.count > 0);
        assert!(snapshot.graph.count > 0);
    }

    #[tokio::test]
    async fn test_api_snapshot_empty() {
        let ctx = Arc::new(AdminContext::new(
            Arc::new(relational_engine::RelationalEngine::new()),
            Arc::new(vector_engine::VectorEngine::new()),
            Arc::new(graph_engine::GraphEngine::new()),
        ));
        let result = api_snapshot(State(ctx)).await;
        assert_eq!(result.0.health, HealthStatus::Degraded);
    }

    #[test]
    fn test_gather_with_populated_engines() {
        use relational_engine::{Column, ColumnType, Schema};
        use std::collections::HashMap;

        let relational = Arc::new(relational_engine::RelationalEngine::new());
        let vector = Arc::new(vector_engine::VectorEngine::new());
        let graph = Arc::new(graph_engine::GraphEngine::new());

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        relational.create_table("t", schema).unwrap();
        let mut row = HashMap::new();
        row.insert("id".to_string(), relational_engine::Value::Int(1));
        relational.insert("t", row).unwrap();

        vector.store_embedding("v1", vec![1.0]).unwrap();

        let ctx = AdminContext::new(relational, vector, graph);
        let snapshot = MetricsSnapshot::gather(&ctx);
        assert_eq!(snapshot.health, HealthStatus::Operational);
        assert_eq!(snapshot.relational.count, 1);
        assert_eq!(snapshot.vector.count, 1);
    }

    #[test]
    fn test_format_timestamp_minutes() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let result = format_timestamp(now - 120);
        assert!(result.contains("m ago"));
    }

    #[test]
    fn test_format_timestamp_hours() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let result = format_timestamp(now - 7200);
        assert!(result.contains("h ago"));
    }

    #[tokio::test]
    async fn test_dashboard_with_metrics() {
        use crate::metrics::{init_metrics, MetricsConfig};

        let config = MetricsConfig::new().with_enabled(false);
        let handle = init_metrics(&config).expect("metrics init");
        let metrics = Arc::clone(handle.metrics());

        metrics.record_request("query", "execute", true, 5.0);
        metrics.record_request("query", "execute", false, 10.0);
        metrics.record_auth_failure("bad_key");
        metrics.record_rate_limited("user:x", "query");

        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(relational_engine::RelationalEngine::new()),
                Arc::new(vector_engine::VectorEngine::new()),
                Arc::new(graph_engine::GraphEngine::new()),
            )
            .with_metrics(Some(metrics)),
        );

        let result = dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("REQUEST METRICS"));
        assert!(html.contains("Total Requests"));
        assert!(html.contains("50.0%")); // success rate: 1/2
        assert!(html.contains("Auth Failures"));
        assert!(html.contains("Rate Limited"));
    }

    #[tokio::test]
    async fn test_dashboard_no_metrics() {
        let ctx = Arc::new(AdminContext::new(
            Arc::new(relational_engine::RelationalEngine::new()),
            Arc::new(vector_engine::VectorEngine::new()),
            Arc::new(graph_engine::GraphEngine::new()),
        ));
        let result = dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("No metrics configured"));
    }
}
