// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Storage internals handlers for the admin UI.
//!
//! Provides views for inspecting `TensorStore` internals: shard access
//! heatmaps, WAL status, and overall storage statistics.

use std::sync::Arc;

use axum::extract::State;
use maud::{html, Markup};

use crate::web::templates::{
    format_bytes, layout, m_breadcrumb, m_card, m_empty, m_header, m_stat,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Storage overview page.
pub async fn overview(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(store) = ctx.store.as_ref() else {
        return layout(
            "Storage",
            NavItem::Storage,
            m_empty(
                "Storage Not Configured",
                "Pass a TensorStore to enable storage internals.",
            ),
        );
    };

    let key_count = store.len();
    let has_instrumentation = store.has_instrumentation();
    let has_wal = store.wal_status().is_some();

    let content = html! {
        (m_header("STORAGE INTERNALS", Some("TensorStore monitoring")))

        div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8 stagger-container" {
            (m_stat("KEYS", &key_count.to_string(), "total stored", "storage"))
            (m_stat(
                "INSTRUMENTATION",
                if has_instrumentation { "ON" } else { "OFF" },
                "shard tracking",
                "storage",
            ))
            (m_stat(
                "WAL",
                if has_wal { "ENABLED" } else { "DISABLED" },
                "write-ahead log",
                "storage",
            ))
        }

        @if has_instrumentation {
            @if let Some(snapshot) = store.access_snapshot() {
                div class="grid grid-cols-2 gap-4 mb-6" {
                    (m_stat("TOTAL READS", &snapshot.total_reads().to_string(), "operations", "storage"))
                    (m_stat("TOTAL WRITES", &snapshot.total_writes().to_string(), "operations", "storage"))
                }
            }
        }

        div class="flex gap-4" {
            a href="/storage/shards" class="m-btn inline-block" { "SHARD HEATMAP" }
            a href="/storage/wal" class="m-btn inline-block" { "WAL STATUS" }
        }
    };

    layout("Storage Internals", NavItem::Storage, content)
}

/// Shard access heatmap.
pub async fn shard_heatmap(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(store) = ctx.store.as_ref() else {
        return layout(
            "Shards",
            NavItem::Storage,
            m_empty(
                "Storage Not Configured",
                "Pass a TensorStore to enable storage internals.",
            ),
        );
    };

    let content = match store.access_snapshot() {
        Some(snapshot) => {
            let stats = &snapshot.shard_stats;
            let hot = store.hot_shards(10).unwrap_or_default();
            let max_access = stats.iter().map(|s| s.reads + s.writes).max().unwrap_or(1);

            html! {
                (m_breadcrumb(&[("/storage", "STORAGE"), ("", "SHARDS")]))
                (m_header("SHARD HEATMAP", Some(&format!("{} shards tracked", stats.len()))))

                // Heat grid
                div class="m-card mb-6" {
                    div class="m-card-header" { "ACCESS DISTRIBUTION" }
                    div class="m-card-content" {
                        div class="grid gap-1" style="grid-template-columns: repeat(auto-fill, minmax(2rem, 1fr));" {
                            @for shard in stats {
                                @let total = shard.reads + shard.writes;
                                @let opacity = if max_access > 0 {
                                    #[allow(clippy::cast_precision_loss)]
                                    let v = total as f64 / max_access as f64;
                                    format!("{:.2}", v.max(0.05))
                                } else {
                                    "0.05".to_string()
                                };
                                div
                                    class="rounded aspect-square"
                                    style=(format!("background: rgba(255,255,255,{opacity});"))
                                    title=(format!("Shard {}: R={} W={}", shard.shard_id, shard.reads, shard.writes))
                                {}
                            }
                        }
                    }
                }

                // Hot shards list
                (m_card("HOT SHARDS (top 10)", html! {
                    @if hot.is_empty() {
                        span class="text-neutral-500 text-sm" { "No hot shards detected" }
                    } @else {
                        ul class="space-y-2" {
                            @for (shard_id, count) in &hot {
                                li class="flex justify-between" {
                                    span class="text-white font-mono" { "Shard " (shard_id) }
                                    span class="text-neutral-400 font-mono" { (count) " accesses" }
                                }
                            }
                        }
                    }
                }))

                // Distribution summary
                (m_card("DISTRIBUTION", html! {
                    @let dist = snapshot.distribution();
                    @if dist.is_empty() {
                        span class="text-neutral-500 text-sm" { "No distribution data" }
                    } @else {
                        dl class="space-y-2" {
                            @for (shard_id, proportion) in &dist {
                                div class="flex justify-between" {
                                    dt class="text-neutral-400" { "Shard " (shard_id) }
                                    dd class="text-white font-mono" { (format!("{:.1}%", proportion * 100.0)) }
                                }
                            }
                        }
                    }
                }))
            }
        },
        None => {
            html! {
                (m_breadcrumb(&[("/storage", "STORAGE"), ("", "SHARDS")]))
                (m_header("SHARD HEATMAP", None))
                (m_empty(
                    "Instrumentation Not Enabled",
                    "Enable instrumentation on TensorStore for shard analytics.",
                ))
            }
        },
    };

    layout("Shard Heatmap", NavItem::Storage, content)
}

/// WAL status page.
pub async fn wal_status(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(store) = ctx.store.as_ref() else {
        return layout(
            "WAL Status",
            NavItem::Storage,
            m_empty(
                "Storage Not Configured",
                "Pass a TensorStore to enable storage internals.",
            ),
        );
    };

    let content = match store.wal_status() {
        Some(status) => {
            html! {
                (m_breadcrumb(&[("/storage", "STORAGE"), ("", "WAL")]))
                (m_header("WRITE-AHEAD LOG", Some("WAL status and configuration")))

                div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6 stagger-container" {
                    (m_stat("ENTRIES", &status.entry_count.to_string(), "log entries", "storage"))
                    @let size = usize::try_from(status.size_bytes).unwrap_or(usize::MAX);
                    (m_stat("SIZE", &format_bytes(size), "on disk", "storage"))
                    (m_stat(
                        "CHECKSUMS",
                        if status.checksums_enabled { "ON" } else { "OFF" },
                        "integrity checks",
                        "storage",
                    ))
                }

                (m_card("WAL PATH", html! {
                    code class="text-white font-mono text-sm break-all" {
                        (status.path.display())
                    }
                }))
            }
        },
        None => {
            html! {
                (m_breadcrumb(&[("/storage", "STORAGE"), ("", "WAL")]))
                (m_header("WRITE-AHEAD LOG", None))
                (m_empty(
                    "WAL Not Enabled",
                    "The write-ahead log is not configured for this store.",
                ))
            }
        },
    };

    layout("WAL Status", NavItem::Storage, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use tensor_store::TensorStore;
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        ))
    }

    fn create_test_context_with_store() -> Arc<AdminContext> {
        let store = TensorStore::new();
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.store = Some(store);
        Arc::new(ctx)
    }

    fn create_test_context_with_instrumented_store() -> Arc<AdminContext> {
        let store = TensorStore::with_instrumentation(1);
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.store = Some(store);
        Arc::new(ctx)
    }

    #[tokio::test]
    async fn test_overview_no_store() {
        let ctx = create_test_context();
        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_overview_with_store() {
        let ctx = create_test_context_with_store();
        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("STORAGE INTERNALS"));
        assert!(html.contains("KEYS"));
    }

    #[tokio::test]
    async fn test_overview_with_instrumented_store() {
        let ctx = create_test_context_with_instrumented_store();
        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("STORAGE INTERNALS"));
        assert!(html.contains("ON")); // instrumentation ON
    }

    #[tokio::test]
    async fn test_shard_heatmap_no_store() {
        let ctx = create_test_context();
        let result = shard_heatmap(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_shard_heatmap_no_instrumentation() {
        let ctx = create_test_context_with_store();
        let result = shard_heatmap(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Instrumentation Not Enabled"));
    }

    #[tokio::test]
    async fn test_shard_heatmap_with_instrumentation() {
        let ctx = create_test_context_with_instrumented_store();
        let result = shard_heatmap(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("SHARD HEATMAP"));
    }

    #[tokio::test]
    async fn test_overview_with_instrumented_data() {
        let store = TensorStore::with_instrumentation(1);
        store
            .put("key1", tensor_store::TensorData::new())
            .expect("put key1");
        store
            .put("key2", tensor_store::TensorData::new())
            .expect("put key2");
        let _ = store.get("key1");
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.store = Some(store);
        let ctx = Arc::new(ctx);

        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("TOTAL READS"));
        assert!(html.contains("TOTAL WRITES"));
    }

    #[tokio::test]
    async fn test_shard_heatmap_with_data() {
        let store = TensorStore::with_instrumentation(1);
        store
            .put("key1", tensor_store::TensorData::new())
            .expect("put key1");
        store
            .put("key2", tensor_store::TensorData::new())
            .expect("put key2");
        // Trigger reads for hot shard tracking
        let _ = store.get("key1");
        let _ = store.get("key1");

        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.store = Some(store);
        let ctx = Arc::new(ctx);

        let result = shard_heatmap(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("SHARD HEATMAP"));
        assert!(html.contains("HOT SHARDS"));
        assert!(html.contains("DISTRIBUTION"));
        assert!(html.contains("accesses"));
    }

    #[tokio::test]
    async fn test_wal_status_no_store() {
        let ctx = create_test_context();
        let result = wal_status(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_wal_status_no_wal() {
        let ctx = create_test_context_with_store();
        let result = wal_status(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("WAL Not Enabled"));
    }

    #[tokio::test]
    async fn test_wal_status_with_wal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wal_path = dir.path().join("test.wal");
        let store = TensorStore::open_durable(&wal_path, tensor_store::WalConfig::default())
            .expect("open durable");
        store
            .put("wal_key", tensor_store::TensorData::new())
            .expect("put wal key");
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.store = Some(store);
        let ctx = Arc::new(ctx);

        let result = wal_status(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("WRITE-AHEAD LOG"));
        assert!(html.contains("ENTRIES"));
        assert!(html.contains("SIZE"));
        assert!(html.contains("CHECKSUMS"));
        assert!(html.contains("WAL PATH"));
    }
}
