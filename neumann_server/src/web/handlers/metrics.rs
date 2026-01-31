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

use crate::web::templates::{layout, page_header};
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
            Self::Healthy => "status-indicator-connected",
            Self::Degraded => "status-indicator-warning",
            Self::Down => "status-indicator-error",
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
            Self::Operational => "text-phosphor",
            Self::Degraded => "text-amber",
            Self::Critical => "text-rust",
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
            .map(|d| d.as_secs())
            .unwrap_or(0);

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
        (page_header("SYSTEM METRICS", Some("Real-time performance monitoring")))

        // System health banner
        div class="terminal-panel mb-6" {
            div class="panel-header" { "SYSTEM HEALTH" }
            div class="panel-content" {
                div class="flex items-center justify-between" {
                    div class="flex items-center gap-4" {
                        div class=(format!("status-indicator {}", match snapshot.health {
                            HealthStatus::Operational => "status-indicator-connected",
                            HealthStatus::Degraded => "status-indicator-warning",
                            HealthStatus::Critical => "status-indicator-error",
                        })) {}
                        span class=(snapshot.health.css_class()) {
                            (snapshot.health.label())
                        }
                    }
                    span class="text-phosphor-dim font-terminal text-sm" {
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
            div class="terminal-panel" {
                div class="panel-header" { "REQUEST METRICS" }
                div class="panel-content" {
                    div class="space-y-3" {
                        (metric_row("Total Requests", "--", "Waiting for data"))
                        (metric_row("Success Rate", "--", "Waiting for data"))
                        (metric_row("Error Rate", "--", "Waiting for data"))
                        (metric_row("Rate Limited", "--", "Waiting for data"))
                    }
                    p class="text-phosphor-dim text-xs mt-4 font-terminal" {
                        "Connect to OTLP endpoint for live data"
                    }
                }
            }

            // Latency metrics
            div class="terminal-panel" {
                div class="panel-header" { "LATENCY DISTRIBUTION" }
                div class="panel-content" {
                    div class="space-y-3" {
                        (latency_bar("Query", 0.0, 100.0))
                        (latency_bar("Blob", 0.0, 100.0))
                        (latency_bar("Vector", 0.0, 100.0))
                    }
                    div class="mt-4 flex justify-between text-xs font-terminal text-phosphor-dim" {
                        span { "0ms" }
                        span { "50ms" }
                        span { "100ms+" }
                    }
                }
            }
        }

        // Quick actions
        div class="terminal-panel" {
            div class="panel-header" { "QUICK ACTIONS" }
            div class="panel-content" {
                div class="flex flex-wrap gap-2" {
                    a href="/" class="btn-terminal" {
                        "DASHBOARD"
                    }
                    a href="/graph/algorithms" class="btn-terminal" {
                        "ALGORITHMS"
                    }
                    button class="btn-terminal" disabled {
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

/// Render an engine status card.
fn engine_status_card(name: &str, stats: &EngineStats, engine_type: &str) -> Markup {
    let border_class = match engine_type {
        "relational" => "border-l-4 border-l-amber-glow",
        "vector" => "border-l-4 border-l-blood-rust",
        "graph" => "border-l-4 border-l-phosphor",
        _ => "",
    };

    html! {
        div class=(format!("terminal-panel {border_class}")) {
            div class="panel-content" {
                div class="flex items-center justify-between mb-2" {
                    span class="font-terminal text-sm text-phosphor-dim" { (name) }
                    div class="flex items-center gap-2" {
                        div class=(format!("status-indicator {}", stats.status.css_class())) {}
                        span class="text-xs font-terminal" { (stats.status.label()) }
                    }
                }
                div class="text-2xl font-data text-phosphor glow-phosphor" {
                    (format_number(stats.count))
                }
                div class="text-xs font-terminal text-phosphor-dim mt-1" {
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
                span class="font-terminal text-sm text-phosphor-dim" { (label) }
            }
            div class="text-right" {
                span class="font-data text-lg text-phosphor" { (value) }
                @if !subtitle.is_empty() {
                    br;
                    span class="text-xs font-terminal text-phosphor-dark" { (subtitle) }
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
                span class="font-terminal text-sm text-phosphor-dim" { (label) }
                span class="font-data text-sm text-phosphor" {
                    @if value > 0.0 {
                        (format!("{value:.1}ms"))
                    } @else {
                        "--"
                    }
                }
            }
            div class="h-2 bg-soot-gray border border-phosphor-dark" {
                div class="h-full bg-phosphor transition-all duration-300"
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
        .map(|d| d.as_secs())
        .unwrap_or(0);

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
        assert_eq!(EngineStatus::Healthy.css_class(), "status-indicator-connected");
        assert_eq!(EngineStatus::Degraded.css_class(), "status-indicator-warning");
        assert_eq!(EngineStatus::Down.css_class(), "status-indicator-error");
    }

    #[test]
    fn test_engine_status_label() {
        assert_eq!(EngineStatus::Healthy.label(), "HEALTHY");
        assert_eq!(EngineStatus::Degraded.label(), "DEGRADED");
        assert_eq!(EngineStatus::Down.label(), "DOWN");
    }

    #[test]
    fn test_health_status_css_class() {
        assert_eq!(HealthStatus::Operational.css_class(), "text-phosphor");
        assert_eq!(HealthStatus::Degraded.css_class(), "text-amber");
        assert_eq!(HealthStatus::Critical.css_class(), "text-rust");
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

        let decoded: MetricsSnapshot =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.timestamp, 12345);
    }
}
