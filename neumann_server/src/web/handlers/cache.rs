// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cache view handlers for the admin UI.
//!
//! Provides three views for browsing cache statistics: stats dashboard,
//! configuration viewer, and per-layer breakdown.

use std::sync::Arc;

use axum::extract::State;
use maud::{html, Markup};
use tensor_cache::Cache;

use crate::web::icons::icon_cache;
use crate::web::templates::{
    layout, m_breadcrumb, m_card, m_card_interactive, m_empty, m_header, m_progress, m_stat,
    m_table_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Format a duration in seconds into a human-readable string.
fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

/// Format a dollar amount for display.
fn format_dollars(amount: f64) -> String {
    if amount < 0.01 {
        format!("${amount:.4}")
    } else {
        format!("${amount:.2}")
    }
}

/// Format a hit rate as a percentage.
fn format_hit_rate(rate: f64) -> String {
    if rate.is_nan() || rate.is_infinite() {
        "0.0%".to_string()
    } else {
        format!("{:.1}%", rate * 100.0)
    }
}

/// Compute hit rate for a single layer.
fn layer_hit_rate(hits: u64, misses: u64) -> f64 {
    let total = hits + misses;
    if total == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let rate = hits as f64 / total as f64;
        rate
    }
}

/// Convert a 0.0..1.0 rate to a 0..100 percentage for progress bar.
fn to_percent(rate: f64) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let p = (rate * 100.0).round() as u8;
    p
}

/// Format a large number compactly.
fn format_number_compact(n: u64) -> String {
    if n >= 1_000_000 {
        #[allow(clippy::cast_precision_loss)]
        let m = n as f64 / 1_000_000.0;
        format!("{m:.1}M")
    } else if n >= 1_000 {
        #[allow(clippy::cast_precision_loss)]
        let k = n as f64 / 1_000.0;
        format!("{k:.1}K")
    } else {
        n.to_string()
    }
}

/// Compute an overall hit rate from all layers.
fn overall_hit_rate(
    exact_hits: u64,
    exact_misses: u64,
    semantic_hits: u64,
    semantic_misses: u64,
    embedding_hits: u64,
    embedding_misses: u64,
) -> f64 {
    let total_hits = exact_hits + semantic_hits + embedding_hits;
    let total = total_hits + exact_misses + semantic_misses + embedding_misses;
    if total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = total_hits as f64 / total as f64;
    rate
}

/// Render the hero stats and hit rate breakdown for the dashboard.
fn render_stats_content(cache: &Cache) -> Markup {
    let snap = cache.stats_snapshot();
    let rate = overall_hit_rate(
        snap.exact_hits,
        snap.exact_misses,
        snap.semantic_hits,
        snap.semantic_misses,
        snap.embedding_hits,
        snap.embedding_misses,
    );
    let total_tokens = snap.tokens_saved_in + snap.tokens_saved_out;
    let exact_rate = layer_hit_rate(snap.exact_hits, snap.exact_misses);
    let sem_rate = layer_hit_rate(snap.semantic_hits, snap.semantic_misses);
    let emb_rate = layer_hit_rate(snap.embedding_hits, snap.embedding_misses);

    html! {
        (m_header("CACHE DASHBOARD", Some("LLM response cache performance and statistics")))

        div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6" {
            (m_stat("TOTAL ENTRIES", &snap.total_entries().to_string(), &format!("across {} layers", 3), "cache"))
            (m_stat("HIT RATE", &format_hit_rate(rate), "overall", "cache"))
            (m_stat(
                "TOKENS SAVED",
                &format_number_compact(total_tokens),
                &format!("{} in / {} out", format_number_compact(snap.tokens_saved_in), format_number_compact(snap.tokens_saved_out)),
                "cache",
            ))
            (m_stat("COST SAVED", &format_dollars(snap.cost_saved_dollars), "estimated savings", "cache"))
        }

        (m_card("HIT RATE BREAKDOWN", html! {
            div class="space-y-4" {
                (render_hit_rate_row("EXACT", exact_rate))
                (render_hit_rate_row("SEMANTIC", sem_rate))
                (render_hit_rate_row("EMBEDDING", emb_rate))
            }
        }))

        div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6" {
            (m_stat("EVICTIONS", &snap.evictions.to_string(), "entries removed", "cache"))
            (m_stat("EXPIRATIONS", &snap.expirations.to_string(), "ttl expired", "cache"))
            (m_stat("UPTIME", &format_duration(snap.uptime_secs), "since start", "cache"))
            (m_stat("LAYERS", "3", "exact + semantic + embedding", "cache"))
        }
    }
}

/// Render a single hit rate row for the breakdown.
fn render_hit_rate_row(label: &str, rate: f64) -> Markup {
    html! {
        div class="flex items-center gap-4" {
            span class="w-24 text-sm text-neutral-400 uppercase tracking-wider" { (label) }
            div class="flex-1" { (m_progress(to_percent(rate))) }
            span class="w-16 text-right text-sm text-neutral-300 font-mono" {
                (format_hit_rate(rate))
            }
        }
    }
}

/// Cache stats dashboard view.
pub async fn stats_dashboard(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.cache.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "DASHBOARD")]))
                (m_header("CACHE DASHBOARD", Some("LLM response cache statistics")))
                (m_empty("Cache not configured", "No cache engine is attached to this server"))
            }
        },
        |cache| {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "DASHBOARD")]))
                (render_stats_content(cache))
            }
        },
    );

    layout("Cache", NavItem::Cache, content)
}

/// Render the config tables for cache configuration.
fn render_config_content(cache: &Cache) -> Markup {
    let cfg = cache.config();

    html! {
        (m_header("CACHE CONFIG", Some("Current cache engine configuration")))

        (m_card("CAPACITY", html! {
            table class="m-table w-full" {
                (m_table_header(&["PARAMETER", "VALUE"]))
                tbody {
                    tr { td class="text-neutral-400" { "Exact Capacity" } td class="font-mono" { (cfg.exact_capacity) } }
                    tr { td class="text-neutral-400" { "Semantic Capacity" } td class="font-mono" { (cfg.semantic_capacity) } }
                    tr { td class="text-neutral-400" { "Embedding Capacity" } td class="font-mono" { (cfg.embedding_capacity) } }
                }
            }
        }))

        (m_card("TTL", html! {
            table class="m-table w-full" {
                (m_table_header(&["PARAMETER", "VALUE"]))
                tbody {
                    tr { td class="text-neutral-400" { "Default TTL" } td class="font-mono" { (format_duration(cfg.default_ttl.as_secs())) } }
                    tr { td class="text-neutral-400" { "Max TTL" } td class="font-mono" { (format_duration(cfg.max_ttl.as_secs())) } }
                }
            }
        }))

        (m_card("SEMANTIC", html! {
            table class="m-table w-full" {
                (m_table_header(&["PARAMETER", "VALUE"]))
                tbody {
                    tr { td class="text-neutral-400" { "Similarity Threshold" } td class="font-mono" { (format!("{:.2}", cfg.semantic_threshold)) } }
                    tr { td class="text-neutral-400" { "Distance Metric" } td class="font-mono" { (format!("{:?}", cfg.distance_metric)) } }
                    tr {
                        td class="text-neutral-400" { "Auto Select Metric" }
                        td class="font-mono" { @if cfg.auto_select_metric { "true" } @else { "false" } }
                    }
                    tr { td class="text-neutral-400" { "Embedding Dimensions" } td class="font-mono" { (cfg.embedding_dim) } }
                }
            }
        }))

        (m_card("EVICTION", html! {
            table class="m-table w-full" {
                (m_table_header(&["PARAMETER", "VALUE"]))
                tbody {
                    tr { td class="text-neutral-400" { "Strategy" } td class="font-mono" { (format!("{:?}", cfg.eviction_strategy)) } }
                    tr { td class="text-neutral-400" { "Eviction Interval" } td class="font-mono" { (format_duration(cfg.eviction_interval.as_secs())) } }
                    tr { td class="text-neutral-400" { "Eviction Batch Size" } td class="font-mono" { (cfg.eviction_batch_size) } }
                }
            }
        }))

        (m_card("COST TRACKING", html! {
            table class="m-table w-full" {
                (m_table_header(&["PARAMETER", "VALUE"]))
                tbody {
                    tr { td class="text-neutral-400" { "Input Cost per 1K Tokens" } td class="font-mono" { (format_dollars(cfg.input_cost_per_1k)) } }
                    tr { td class="text-neutral-400" { "Output Cost per 1K Tokens" } td class="font-mono" { (format_dollars(cfg.output_cost_per_1k)) } }
                }
            }
        }))
    }
}

/// Cache configuration viewer.
pub async fn config_viewer(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.cache.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "CONFIG")]))
                (m_header("CACHE CONFIG", Some("Cache engine configuration")))
                (m_empty("Cache not configured", "No cache engine is attached to this server"))
            }
        },
        |cache| {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "CONFIG")]))
                (render_config_content(cache))
            }
        },
    );

    layout("Cache Config", NavItem::Cache, content)
}

/// Render per-layer breakdown content.
fn render_layers_content(cache: &Cache) -> Markup {
    let snap = cache.stats_snapshot();

    let layers = [
        (
            "EXACT",
            "O(1) hash-based exact match",
            snap.exact_size,
            snap.exact_hits,
            snap.exact_misses,
        ),
        (
            "SEMANTIC",
            "O(log n) HNSW similarity search",
            snap.semantic_size,
            snap.semantic_hits,
            snap.semantic_misses,
        ),
        (
            "EMBEDDING",
            "O(1) cached embeddings",
            snap.embedding_size,
            snap.embedding_hits,
            snap.embedding_misses,
        ),
    ];

    let max_size = layers
        .iter()
        .map(|(_, _, size, _, _)| *size)
        .max()
        .unwrap_or(1)
        .max(1);

    html! {
        (m_header("CACHE LAYERS", Some("Per-layer size and hit rate breakdown")))

        div class="grid grid-cols-1 lg:grid-cols-3 gap-4" {
            @for (name, desc, size, hits, misses) in &layers {
                (m_card_interactive(name, html! {
                    div class="space-y-4" {
                        p class="text-sm text-neutral-500" { (desc) }
                        div class="w-8 h-8 opacity-30" { (icon_cache()) }

                        div {
                            span class="text-2xl font-light text-white" { (size) }
                            span class="text-sm text-neutral-500 ml-2" { "entries" }
                        }

                        div class="w-full bg-neutral-800 rounded-sm overflow-hidden" style="height: 4px;" {
                            @let width_pct = (size * 100).checked_div(max_size).unwrap_or(0);
                            div class="bg-neutral-400 h-full transition-all" style=(format!("width: {width_pct}%")) {}
                        }

                        div class="grid grid-cols-2 gap-2 mt-2" {
                            div {
                                span class="text-xs text-neutral-500 uppercase" { "HITS" }
                                div class="text-lg font-light text-white font-mono" { (hits) }
                            }
                            div {
                                span class="text-xs text-neutral-500 uppercase" { "MISSES" }
                                div class="text-lg font-light text-white font-mono" { (misses) }
                            }
                        }

                        @let rate = layer_hit_rate(*hits, *misses);
                        div class="flex items-center gap-2" {
                            span class="text-xs text-neutral-500 uppercase w-16" { "HIT RATE" }
                            div class="flex-1" { (m_progress(to_percent(rate))) }
                        }
                    }
                }))
            }
        }
    }
}

/// Per-layer breakdown view.
pub async fn layers_breakdown(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.cache.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "LAYERS")]))
                (m_header("CACHE LAYERS", Some("Per-layer breakdown")))
                (m_empty("Cache not configured", "No cache engine is attached to this server"))
            }
        },
        |cache| {
            html! {
                (m_breadcrumb(&[("/cache", "CACHE"), ("", "LAYERS")]))
                (render_layers_content(cache))
            }
        },
    );

    layout("Cache Layers", NavItem::Cache, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use axum::extract::State;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use tensor_cache::Cache;
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            unified: None,
            vault: None,
            cache: None,
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    fn create_cache_context() -> Arc<AdminContext> {
        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            unified: None,
            vault: None,
            cache: Some(Arc::new(Cache::new())),
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    fn create_seeded_cache_context() -> Arc<AdminContext> {
        let cache = Arc::new(Cache::new());
        cache.put_simple("greeting", "Hello!").ok();
        cache.put_simple("farewell", "Goodbye!").ok();
        cache.put_simple("thanks", "Thank you!").ok();

        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            unified: None,
            vault: None,
            cache: Some(cache),
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    #[tokio::test]
    async fn test_stats_dashboard_no_cache() {
        let ctx = create_test_context();
        let html = stats_dashboard(State(ctx)).await.into_string();
        assert!(html.contains("Cache not configured"));
    }

    #[tokio::test]
    async fn test_stats_dashboard_empty_cache() {
        let ctx = create_cache_context();
        let html = stats_dashboard(State(ctx)).await.into_string();
        assert!(html.contains("CACHE DASHBOARD"));
        assert!(html.contains("TOTAL ENTRIES"));
        assert!(html.contains("HIT RATE"));
        assert!(html.contains("TOKENS SAVED"));
        assert!(html.contains("COST SAVED"));
    }

    #[tokio::test]
    async fn test_stats_dashboard_with_data() {
        let ctx = create_seeded_cache_context();
        let html = stats_dashboard(State(ctx)).await.into_string();
        assert!(html.contains("CACHE DASHBOARD"));
        assert!(html.contains("HIT RATE BREAKDOWN"));
        assert!(html.contains("EXACT"));
        assert!(html.contains("SEMANTIC"));
        assert!(html.contains("EMBEDDING"));
    }

    #[tokio::test]
    async fn test_stats_dashboard_shows_eviction_counters() {
        let ctx = create_cache_context();
        let html = stats_dashboard(State(ctx)).await.into_string();
        assert!(html.contains("EVICTIONS"));
        assert!(html.contains("EXPIRATIONS"));
        assert!(html.contains("UPTIME"));
    }

    #[tokio::test]
    async fn test_config_viewer_no_cache() {
        let ctx = create_test_context();
        let html = config_viewer(State(ctx)).await.into_string();
        assert!(html.contains("Cache not configured"));
    }

    #[tokio::test]
    async fn test_config_viewer_shows_all_groups() {
        let ctx = create_cache_context();
        let html = config_viewer(State(ctx)).await.into_string();
        assert!(html.contains("CACHE CONFIG"));
        assert!(html.contains("CAPACITY"));
        assert!(html.contains("TTL"));
        assert!(html.contains("SEMANTIC"));
        assert!(html.contains("EVICTION"));
        assert!(html.contains("COST TRACKING"));
    }

    #[tokio::test]
    async fn test_config_viewer_displays_values() {
        let ctx = create_cache_context();
        let html = config_viewer(State(ctx)).await.into_string();
        assert!(html.contains("Exact Capacity"));
        assert!(html.contains("Semantic Capacity"));
        assert!(html.contains("Embedding Capacity"));
        assert!(html.contains("Similarity Threshold"));
        assert!(html.contains("Distance Metric"));
        assert!(html.contains("Eviction Batch Size"));
    }

    #[tokio::test]
    async fn test_layers_breakdown_no_cache() {
        let ctx = create_test_context();
        let html = layers_breakdown(State(ctx)).await.into_string();
        assert!(html.contains("Cache not configured"));
    }

    #[tokio::test]
    async fn test_layers_breakdown_shows_all_layers() {
        let ctx = create_cache_context();
        let html = layers_breakdown(State(ctx)).await.into_string();
        assert!(html.contains("CACHE LAYERS"));
        assert!(html.contains("EXACT"));
        assert!(html.contains("SEMANTIC"));
        assert!(html.contains("EMBEDDING"));
    }

    #[tokio::test]
    async fn test_layers_breakdown_with_data() {
        let ctx = create_seeded_cache_context();
        let html = layers_breakdown(State(ctx)).await.into_string();
        assert!(html.contains("CACHE LAYERS"));
        assert!(html.contains("entries"));
        assert!(html.contains("HITS"));
        assert!(html.contains("MISSES"));
        assert!(html.contains("HIT RATE"));
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(30), "30s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(90), "1m 30s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3661), "1h 1m");
    }

    #[test]
    fn test_format_duration_days() {
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_format_dollars_small() {
        assert_eq!(format_dollars(0.001), "$0.0010");
    }

    #[test]
    fn test_format_dollars_normal() {
        assert_eq!(format_dollars(1.23), "$1.23");
    }

    #[test]
    fn test_format_hit_rate_zero() {
        assert_eq!(format_hit_rate(0.0), "0.0%");
    }

    #[test]
    fn test_format_hit_rate_full() {
        assert_eq!(format_hit_rate(1.0), "100.0%");
    }

    #[test]
    fn test_format_hit_rate_nan() {
        assert_eq!(format_hit_rate(f64::NAN), "0.0%");
    }

    #[test]
    fn test_layer_hit_rate_zero_total() {
        assert_eq!(layer_hit_rate(0, 0), 0.0);
    }

    #[test]
    fn test_layer_hit_rate_all_hits() {
        assert!((layer_hit_rate(10, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_hit_rate_half() {
        assert!((layer_hit_rate(5, 5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_percent() {
        assert_eq!(to_percent(0.0), 0);
        assert_eq!(to_percent(0.5), 50);
        assert_eq!(to_percent(1.0), 100);
    }

    #[test]
    fn test_overall_hit_rate_empty() {
        assert_eq!(overall_hit_rate(0, 0, 0, 0, 0, 0), 0.0);
    }

    #[test]
    fn test_overall_hit_rate_mixed() {
        let rate = overall_hit_rate(5, 5, 3, 7, 2, 8);
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_format_number_compact_small() {
        assert_eq!(format_number_compact(42), "42");
    }

    #[test]
    fn test_format_number_compact_thousands() {
        assert_eq!(format_number_compact(1500), "1.5K");
    }

    #[test]
    fn test_format_number_compact_millions() {
        assert_eq!(format_number_compact(2_500_000), "2.5M");
    }
}
