// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contraction view handlers for the admin UI.
//!
//! Provides four views exposing the cross-modal tensor contraction engine:
//! explainability, multi-lens ranking, sensitivity lab, and counterfactual.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::{Query, State};
use maud::{html, Markup};

use crate::web::templates::{
    layout, m_badge, m_breadcrumb, m_card, m_card_interactive, m_empty, m_header, m_tabs,
};
use crate::web::AdminContext;
use crate::web::NavItem;
use tensor_unified::{ContractionConfig, GraphDirection, Normalization};

/// Query parameters shared across contraction views.
#[derive(Debug, serde::Deserialize)]
pub struct ContractionParams {
    /// Source entity key.
    source_key: Option<String>,
    /// Interaction table name.
    table: Option<String>,
    /// Source column in the interaction table.
    source_col: Option<String>,
    /// Target column in the interaction table.
    target_col: Option<String>,
    /// Graph direction: symmetric, outgoing, incoming.
    direction: Option<String>,
    /// Normalization: none, `total_weight`, `per_item`.
    normalization: Option<String>,
    /// Edge type filter.
    edge_type: Option<String>,
    /// Whether to exclude owned items.
    exclude_owned: Option<bool>,
    /// Top-k results.
    top_k: Option<usize>,
}

impl ContractionParams {
    /// Build a contraction config from the query parameters.
    fn to_config(&self) -> ContractionConfig {
        ContractionConfig {
            direction: match self.direction.as_deref() {
                Some("outgoing") => GraphDirection::Outgoing,
                Some("incoming") => GraphDirection::Incoming,
                _ => GraphDirection::Symmetric,
            },
            normalization: match self.normalization.as_deref() {
                Some("total_weight") => Normalization::TotalWeight,
                Some("per_item") => Normalization::PerItem,
                _ => Normalization::None,
            },
            edge_type: self.edge_type.as_ref().filter(|s| !s.is_empty()).cloned(),
            exclude_owned: self.exclude_owned.unwrap_or(false),
            top_k: self.top_k.unwrap_or(20),
        }
    }

    /// Whether the user has submitted the form (source key is present).
    fn is_submitted(&self) -> bool {
        self.source_key
            .as_ref()
            .is_some_and(|k| !k.trim().is_empty())
    }
}

/// Render the contraction form shared across views.
#[allow(clippy::too_many_lines)]
fn contraction_form(params: &ContractionParams, ctx: &AdminContext, action: &str) -> Markup {
    let tables = ctx.relational.list_tables();

    html! {
        form method="get" action=(action) class="m-card mb-6" {
            div class="m-card-header" { "CONTRACTION PARAMETERS" }
            div class="m-card-content" {
                div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" {
                    // Source key
                    div {
                        label class="text-label block mb-1" { "SOURCE ENTITY KEY" }
                        input
                            type="text"
                            name="source_key"
                            class="m-input w-full"
                            placeholder="e.g. alice"
                            value=(params.source_key.as_deref().unwrap_or(""));
                    }
                    // Interaction table
                    div {
                        label class="text-label block mb-1" { "INTERACTION TABLE" }
                        select name="table" class="m-input w-full" {
                            option value="" { "-- select --" }
                            @for t in &tables {
                                option
                                    value=(t)
                                    selected[params.table.as_deref() == Some(t.as_str())]
                                { (t) }
                            }
                        }
                    }
                    // Source column
                    div {
                        label class="text-label block mb-1" { "SOURCE COLUMN" }
                        input
                            type="text"
                            name="source_col"
                            class="m-input w-full"
                            placeholder="user_id"
                            value=(params.source_col.as_deref().unwrap_or(""));
                    }
                    // Target column
                    div {
                        label class="text-label block mb-1" { "TARGET COLUMN" }
                        input
                            type="text"
                            name="target_col"
                            class="m-input w-full"
                            placeholder="item_id"
                            value=(params.target_col.as_deref().unwrap_or(""));
                    }
                    // Direction
                    div {
                        label class="text-label block mb-1" { "DIRECTION" }
                        select name="direction" class="m-input w-full" {
                            option value="symmetric"
                                selected[params.direction.as_deref() != Some("outgoing") && params.direction.as_deref() != Some("incoming")]
                            { "Symmetric" }
                            option value="outgoing"
                                selected[params.direction.as_deref() == Some("outgoing")]
                            { "Outgoing" }
                            option value="incoming"
                                selected[params.direction.as_deref() == Some("incoming")]
                            { "Incoming" }
                        }
                    }
                    // Normalization
                    div {
                        label class="text-label block mb-1" { "NORMALIZATION" }
                        select name="normalization" class="m-input w-full" {
                            option value="none"
                                selected[params.normalization.as_deref() != Some("total_weight") && params.normalization.as_deref() != Some("per_item")]
                            { "None" }
                            option value="total_weight"
                                selected[params.normalization.as_deref() == Some("total_weight")]
                            { "Total Weight" }
                            option value="per_item"
                                selected[params.normalization.as_deref() == Some("per_item")]
                            { "Per Item" }
                        }
                    }
                    // Edge type
                    div {
                        label class="text-label block mb-1" { "EDGE TYPE FILTER" }
                        input
                            type="text"
                            name="edge_type"
                            class="m-input w-full"
                            placeholder="(all types)"
                            value=(params.edge_type.as_deref().unwrap_or(""));
                    }
                    // Exclude owned
                    div class="flex items-end gap-2" {
                        label class="flex items-center gap-2 cursor-pointer" {
                            input
                                type="checkbox"
                                name="exclude_owned"
                                value="true"
                                checked[params.exclude_owned.unwrap_or(false)];
                            span class="text-sm text-neutral-300" { "Exclude owned items" }
                        }
                    }
                    // Top-k
                    div {
                        label class="text-label block mb-1" { "TOP-K" }
                        input
                            type="number"
                            name="top_k"
                            class="m-input w-full"
                            min="1"
                            max="1000"
                            value=(params.top_k.unwrap_or(20));
                    }
                }
            }
            div class="m-card-footer flex justify-end" {
                button type="submit" class="m-btn" { "RUN CONTRACTION" }
            }
        }
    }
}

/// View 1: Explainability -- full contraction breakdown.
pub async fn explainability(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<ContractionParams>,
) -> Markup {
    let tabs = contraction_tabs("/contraction");

    let result_html = if ctx.unified.is_some() && params.is_submitted() {
        run_explainability(&ctx, &params).await
    } else {
        html! {}
    };

    let content = html! {
        (m_breadcrumb(&[("/contraction", "CONTRACTION"), ("", "EXPLAINABILITY")]))
        (m_header("TENSOR CONTRACTION", Some("Cross-modal explainability")))
        (tabs)

        @if ctx.unified.is_none() {
            (m_empty(
                "Unified Engine Not Configured",
                "Start the server with a UnifiedEngine to enable contraction views."
            ))
        } @else {
            (contraction_form(&params, &ctx, "/contraction"))
            (result_html)
        }
    };

    layout("Contraction", NavItem::Contraction, content)
}

/// Execute explainability and render results.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn run_explainability(ctx: &AdminContext, params: &ContractionParams) -> Markup {
    let Some(unified) = ctx.unified.as_ref() else {
        return html! {};
    };

    let source_key = params.source_key.as_deref().unwrap_or("");
    let config = params.to_config();

    // Step 1: Adjacency
    let adjacency = unified.explain_adjacency(source_key, &config);
    if adjacency.is_empty() {
        return m_card(
            "RESULT",
            html! {
                div class="text-neutral-400 py-4 text-center" {
                    "No graph neighbors found for \"" (source_key) "\""
                }
            },
        );
    }

    // Step 2: Similarity
    let similarity = unified.explain_similarity(source_key, &adjacency);

    // Step 3: Interactions (if table specified)
    let table = params.table.as_deref().unwrap_or("");
    let source_col = params.source_col.as_deref().unwrap_or("");
    let target_col = params.target_col.as_deref().unwrap_or("");

    let has_table = !table.is_empty() && !source_col.is_empty() && !target_col.is_empty();
    let interactions_result = if has_table {
        let intermediary_keys: HashSet<&str> = adjacency.keys().map(String::as_str).collect();
        let owned_key = if config.exclude_owned {
            Some(source_key)
        } else {
            None
        };
        unified
            .explain_interactions(table, source_col, target_col, &intermediary_keys, owned_key)
            .ok()
    } else {
        None
    };

    // Step 4: Full contraction
    let contraction_result = if has_table {
        unified
            .cross_modal_contraction(source_key, table, source_col, target_col, &config, None)
            .await
            .ok()
    } else {
        None
    };

    // Render adjacency breakdown
    let mut adj_items: Vec<_> = adjacency.iter().collect();
    adj_items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    html! {
        // Adjacency
        div class="m-card mb-4" {
            div class="m-card-header" {
                "ADJACENCY "
                (m_badge(&format!("{} neighbors", adj_items.len())))
            }
            div class="m-card-content" {
                table class="m-table" {
                    thead {
                        tr {
                            th { "NEIGHBOR" }
                            th { "WEIGHT" }
                            th { "SIMILARITY" }
                            th { "FUSED" }
                        }
                    }
                    tbody {
                        @for (key, weight) in &adj_items {
                            @let sim = similarity.get(*key).copied().unwrap_or(0.0);
                            @let fused = *weight * sim;
                            tr {
                                td class="text-white" { (key) }
                                td class="font-mono text-neutral-400" { (format!("{weight:.4}")) }
                                td class="font-mono text-neutral-400" { (format!("{sim:.4}")) }
                                td class="font-mono text-white" { (format!("{fused:.4}")) }
                            }
                        }
                    }
                }
            }
        }

        // Interactions
        @if let Some((ref interactions, ref owned)) = interactions_result {
            div class="m-card mb-4" {
                div class="m-card-header" {
                    "INTERACTIONS "
                    (m_badge(&format!("{} intermediaries", interactions.len())))
                }
                div class="m-card-content" {
                    @if interactions.is_empty() {
                        div class="text-neutral-400 text-center py-4" {
                            "No interactions found in table"
                        }
                    } @else {
                        @for (intermediary, items) in interactions {
                            div class="mb-3" {
                                span class="text-white text-sm" { (intermediary) }
                                span class="text-neutral-500 text-xs ml-2" {
                                    "interacted with:"
                                }
                                div class="flex flex-wrap gap-1 mt-1" {
                                    @for item in items {
                                        (m_badge(item))
                                    }
                                }
                            }
                        }
                    }
                    @if !owned.is_empty() {
                        div class="mt-4 pt-3 border-t border-neutral-800" {
                            span class="text-label" { "OWNED BY SOURCE" }
                            div class="flex flex-wrap gap-1 mt-1" {
                                @for item in owned {
                                    (m_badge(item))
                                }
                            }
                        }
                    }
                }
            }
        }

        // Final contraction results
        @if let Some(ref result) = contraction_result {
            div class="m-card" {
                div class="m-card-header" {
                    "CONTRACTION RESULT "
                    (m_badge(&format!("{} items", result.items.len())))
                    @if result.excluded_count > 0 {
                        " "
                        (m_badge(&format!("{} excluded", result.excluded_count)))
                    }
                }
                div class="m-card-content" {
                    @if result.items.is_empty() {
                        div class="text-neutral-400 text-center py-4" {
                            "No results"
                        }
                    } @else {
                        table class="m-table" {
                            thead {
                                tr {
                                    th { "#" }
                                    th { "ITEM" }
                                    th { "SCORE" }
                                    th { "CONTRIBUTORS" }
                                    th { "BAR" }
                                }
                            }
                            tbody {
                                @let max_score = result.items.first().map_or(1.0, |i| i.score.max(0.001));
                                @for (i, item) in result.items.iter().enumerate() {
                                    @let pct = (item.score / max_score * 100.0) as u32;
                                    tr {
                                        td class="font-mono text-neutral-500" { (i + 1) }
                                        td class="text-white" { (&item.item_key) }
                                        td class="font-mono text-neutral-400" {
                                            (format!("{:.4}", item.score))
                                        }
                                        td class="font-mono text-neutral-500 text-center" {
                                            (item.contributors)
                                        }
                                        td {
                                            div class="w-full bg-neutral-800 rounded-sm overflow-hidden"
                                                style="height:4px" {
                                                div class="bg-white h-full"
                                                    style=(format!("width:{pct}%")) {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                div class="m-card-footer" {
                    "Weight norm: " (format!("{:.4}", result.weight_norm))
                }
            }
        }
    }
}

/// View 2: Multi-lens ranking.
pub async fn ranking(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<ContractionParams>,
) -> Markup {
    let tabs = contraction_tabs("/contraction/ranking");

    let result_html = if ctx.unified.is_some() && params.is_submitted() {
        run_ranking(&ctx, &params)
    } else {
        html! {}
    };

    let content = html! {
        (m_breadcrumb(&[("/contraction", "CONTRACTION"), ("", "RANKING")]))
        (m_header("MULTI-LENS RANKING", Some("Compare vector, graph, and fused rankings")))
        (tabs)

        @if ctx.unified.is_none() {
            (m_empty(
                "Unified Engine Not Configured",
                "Start the server with a UnifiedEngine to enable contraction views."
            ))
        } @else {
            (contraction_form(&params, &ctx, "/contraction/ranking"))
            (result_html)
        }
    };

    layout("Ranking", NavItem::Contraction, content)
}

/// Execute multi-lens ranking.
fn run_ranking(ctx: &AdminContext, params: &ContractionParams) -> Markup {
    let Some(unified) = ctx.unified.as_ref() else {
        return html! {};
    };

    let source_key = params.source_key.as_deref().unwrap_or("");
    let config = params.to_config();
    let adjacency = unified.explain_adjacency(source_key, &config);
    let similarity = unified.explain_similarity(source_key, &adjacency);

    // Vector-only ranking (by similarity)
    let mut sim_items: Vec<_> = similarity.iter().collect();
    sim_items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Graph-weighted ranking (by adjacency)
    let mut adj_items: Vec<_> = adjacency.iter().collect();
    adj_items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Fused ranking (adjacency * similarity)
    let mut fused: Vec<(String, f64)> = adjacency
        .iter()
        .map(|(k, w)| {
            let s = similarity.get(k).copied().unwrap_or(0.0);
            (k.clone(), w * s)
        })
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if adjacency.is_empty() {
        return m_card(
            "RESULT",
            html! {
                div class="text-neutral-400 py-4 text-center" {
                    "No neighbors found for \"" (source_key) "\""
                }
            },
        );
    }

    html! {
        div class="grid grid-cols-1 lg:grid-cols-3 gap-4" {
            // Vector-only
            (m_card("VECTOR (SIMILARITY)", html! {
                @if sim_items.is_empty() {
                    div class="text-neutral-400 text-sm" { "No embeddings" }
                } @else {
                    @for (i, (key, score)) in sim_items.iter().enumerate() {
                        div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                            span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                            span class="text-white text-sm flex-1" { (key) }
                            span class="font-mono text-neutral-400 text-xs" { (format!("{score:.3}")) }
                        }
                    }
                }
            }))

            // Graph-weighted
            (m_card("GRAPH (ADJACENCY)", html! {
                @for (i, (key, weight)) in adj_items.iter().enumerate() {
                    div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                        span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                        span class="text-white text-sm flex-1" { (key) }
                        span class="font-mono text-neutral-400 text-xs" { (format!("{weight:.3}")) }
                    }
                }
            }))

            // Full contraction
            (m_card("FUSED (ADJ x SIM)", html! {
                @for (i, (key, score)) in fused.iter().enumerate() {
                    div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                        span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                        span class="text-white text-sm flex-1" { (key) }
                        span class="font-mono text-neutral-400 text-xs" { (format!("{score:.4}")) }
                    }
                }
            }))
        }
    }
}

/// View 3: Sensitivity lab.
pub async fn sensitivity(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<ContractionParams>,
) -> Markup {
    let tabs = contraction_tabs("/contraction/sensitivity");

    let result_html = if ctx.unified.is_some() && params.is_submitted() {
        run_sensitivity(&ctx, &params).await
    } else {
        html! {}
    };

    let content = html! {
        (m_breadcrumb(&[("/contraction", "CONTRACTION"), ("", "SENSITIVITY")]))
        (m_header("SENSITIVITY LAB", Some("Explore how configuration changes affect rankings")))
        (tabs)

        @if ctx.unified.is_none() {
            (m_empty(
                "Unified Engine Not Configured",
                "Start the server with a UnifiedEngine to enable contraction views."
            ))
        } @else {
            (contraction_form(&params, &ctx, "/contraction/sensitivity"))
            (result_html)
        }
    };

    layout("Sensitivity", NavItem::Contraction, content)
}

/// Execute sensitivity comparison.
#[allow(clippy::cast_possible_wrap)]
async fn run_sensitivity(ctx: &AdminContext, params: &ContractionParams) -> Markup {
    let Some(unified) = ctx.unified.as_ref() else {
        return html! {};
    };

    let source_key = params.source_key.as_deref().unwrap_or("");
    let table = params.table.as_deref().unwrap_or("");
    let source_col = params.source_col.as_deref().unwrap_or("");
    let target_col = params.target_col.as_deref().unwrap_or("");

    if table.is_empty() || source_col.is_empty() || target_col.is_empty() {
        return m_card(
            "SENSITIVITY",
            html! {
                div class="text-neutral-400 py-4 text-center" {
                    "Specify interaction table and columns to run sensitivity analysis"
                }
            },
        );
    }

    // Default config
    let default_config = ContractionConfig {
        direction: GraphDirection::Symmetric,
        normalization: Normalization::None,
        edge_type: None,
        exclude_owned: false,
        top_k: params.top_k.unwrap_or(20),
    };

    // User config
    let user_config = params.to_config();

    let default_result = unified
        .cross_modal_contraction(
            source_key,
            table,
            source_col,
            target_col,
            &default_config,
            None,
        )
        .await
        .ok();

    let user_result = unified
        .cross_modal_contraction(
            source_key,
            table,
            source_col,
            target_col,
            &user_config,
            None,
        )
        .await
        .ok();

    html! {
        div class="grid grid-cols-1 lg:grid-cols-2 gap-4" {
            (m_card_interactive("DEFAULT CONFIG", html! {
                @if let Some(ref result) = default_result {
                    @for (i, item) in result.items.iter().enumerate() {
                        div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                            span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                            span class="text-white text-sm flex-1" { (&item.item_key) }
                            span class="font-mono text-neutral-400 text-xs" {
                                (format!("{:.4}", item.score))
                            }
                        }
                    }
                } @else {
                    div class="text-neutral-400 text-sm" { "No results with default config" }
                }
            }))

            (m_card_interactive("CURRENT CONFIG", html! {
                @if let Some(ref result) = user_result {
                    @for (i, item) in result.items.iter().enumerate() {
                        // Find rank in default
                        @let default_rank = default_result.as_ref().and_then(|dr| {
                            dr.items.iter().position(|di| di.item_key == item.item_key)
                        });
                        div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                            span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                            span class="text-white text-sm flex-1" { (&item.item_key) }
                            span class="font-mono text-neutral-400 text-xs" {
                                (format!("{:.4}", item.score))
                            }
                            @if let Some(dr) = default_rank {
                                @let delta = dr as i64 - i as i64;
                                @if delta > 0 {
                                    span class="text-xs text-neutral-300 ml-1" {
                                        "+" (delta)
                                    }
                                } @else if delta < 0 {
                                    span class="text-xs text-neutral-500 ml-1" {
                                        (delta)
                                    }
                                }
                            } @else {
                                span class="text-xs ml-1" { (m_badge("NEW")) }
                            }
                        }
                    }
                } @else {
                    div class="text-neutral-400 text-sm" { "No results with current config" }
                }
            }))
        }
    }
}

/// View 4: Counterfactual analysis.
pub async fn counterfactual(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<ContractionParams>,
) -> Markup {
    let tabs = contraction_tabs("/contraction/counterfactual");

    let result_html = if ctx.unified.is_some() && params.is_submitted() {
        run_counterfactual(&ctx, &params).await
    } else {
        html! {}
    };

    let content = html! {
        (m_breadcrumb(&[("/contraction", "CONTRACTION"), ("", "COUNTERFACTUAL")]))
        (m_header("COUNTERFACTUAL", Some("What if an edge were removed?")))
        (tabs)

        @if ctx.unified.is_none() {
            (m_empty(
                "Unified Engine Not Configured",
                "Start the server with a UnifiedEngine to enable contraction views."
            ))
        } @else {
            (contraction_form(&params, &ctx, "/contraction/counterfactual"))
            (result_html)
        }
    };

    layout("Counterfactual", NavItem::Contraction, content)
}

/// Execute counterfactual comparison.
///
/// Compares the full contraction result with edge type filter vs without,
/// showing how removing a class of edges changes scores.
async fn run_counterfactual(ctx: &AdminContext, params: &ContractionParams) -> Markup {
    let Some(unified) = ctx.unified.as_ref() else {
        return html! {};
    };

    let source_key = params.source_key.as_deref().unwrap_or("");
    let table = params.table.as_deref().unwrap_or("");
    let source_col = params.source_col.as_deref().unwrap_or("");
    let target_col = params.target_col.as_deref().unwrap_or("");

    if table.is_empty() || source_col.is_empty() || target_col.is_empty() {
        return m_card(
            "COUNTERFACTUAL",
            html! {
                div class="text-neutral-400 py-4 text-center" {
                    "Specify interaction table, columns, and edge type to compare"
                }
            },
        );
    }

    let user_config = params.to_config();

    // Baseline: no edge type filter
    let baseline_config = ContractionConfig {
        edge_type: None,
        ..user_config
    };

    let baseline = unified
        .cross_modal_contraction(
            source_key,
            table,
            source_col,
            target_col,
            &baseline_config,
            None,
        )
        .await
        .ok();

    let filtered = unified
        .cross_modal_contraction(
            source_key,
            table,
            source_col,
            target_col,
            &user_config,
            None,
        )
        .await
        .ok();

    html! {
        div class="grid grid-cols-1 lg:grid-cols-2 gap-4" {
            (m_card("BASELINE (ALL EDGES)", html! {
                @if let Some(ref result) = baseline {
                    @if result.items.is_empty() {
                        div class="text-neutral-400 text-sm text-center py-4" { "No results" }
                    } @else {
                        @for (i, item) in result.items.iter().enumerate() {
                            div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                                span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                                span class="text-white text-sm flex-1" { (&item.item_key) }
                                span class="font-mono text-neutral-400 text-xs" {
                                    (format!("{:.4}", item.score))
                                }
                            }
                        }
                    }
                } @else {
                    div class="text-neutral-400 text-sm" { "Error running baseline" }
                }
            }))

            (m_card("FILTERED", html! {
                @if let Some(ref result) = filtered {
                    @if result.items.is_empty() {
                        div class="text-neutral-400 text-sm text-center py-4" { "No results" }
                    } @else {
                        @for (i, item) in result.items.iter().enumerate() {
                            @let baseline_score = baseline.as_ref().and_then(|b| {
                                b.items.iter().find(|bi| bi.item_key == item.item_key).map(|bi| bi.score)
                            });
                            div class="flex items-center justify-between py-1 border-b border-neutral-800" {
                                span class="text-neutral-500 font-mono text-xs w-6" { (i + 1) }
                                span class="text-white text-sm flex-1" { (&item.item_key) }
                                span class="font-mono text-neutral-400 text-xs" {
                                    (format!("{:.4}", item.score))
                                }
                                @if let Some(bs) = baseline_score {
                                    @let delta = item.score - bs;
                                    @if delta.abs() > 0.0001 {
                                        (m_badge(&format!("{delta:+.3}")))
                                    }
                                } @else {
                                    (m_badge("NEW"))
                                }
                            }
                        }
                    }
                } @else {
                    div class="text-neutral-400 text-sm" { "Error running filtered" }
                }
            }))
        }
    }
}

/// Render the contraction view tabs.
fn contraction_tabs(active_path: &str) -> Markup {
    let items = [
        (
            "Explainability",
            "/contraction",
            active_path == "/contraction",
        ),
        (
            "Ranking",
            "/contraction/ranking",
            active_path == "/contraction/ranking",
        ),
        (
            "Sensitivity",
            "/contraction/sensitivity",
            active_path == "/contraction/sensitivity",
        ),
        (
            "Counterfactual",
            "/contraction/counterfactual",
            active_path == "/contraction/counterfactual",
        ),
    ];
    m_tabs(&items.map(|(l, h, a)| (l, h, a)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use graph_engine::GraphEngine;
    use relational_engine::{ColumnType, RelationalEngine, Value};
    use tensor_store::TensorStore;
    use tensor_unified::UnifiedEngine;
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

    fn create_test_context_with_unified() -> Arc<AdminContext> {
        let unified = Arc::new(UnifiedEngine::new());
        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            unified: Some(unified),
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

    /// Create a context with seeded graph, vector, and relational data.
    /// Uses shared engine instances so both `AdminContext` and `UnifiedEngine`
    /// see the same data.
    async fn create_populated_context() -> Arc<AdminContext> {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let store = TensorStore::new();

        let unified =
            UnifiedEngine::with_engines(store, relational.clone(), graph.clone(), vector.clone());

        // Create entities (graph nodes with entity_key property)
        for key in &["alice", "bob", "carol", "dave"] {
            unified
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }

        // Graph edges
        unified
            .connect_entities("alice", "bob", "FRIEND")
            .await
            .unwrap();
        unified
            .connect_entities("alice", "carol", "FRIEND")
            .await
            .unwrap();
        unified
            .connect_entities("alice", "dave", "COWORKER")
            .await
            .unwrap();

        // Embeddings
        vector
            .set_entity_embedding("alice", vec![1.0, 0.0, 0.0])
            .unwrap();
        vector
            .set_entity_embedding("bob", vec![0.9, 0.1, 0.0])
            .unwrap();
        vector
            .set_entity_embedding("carol", vec![0.5, 0.5, 0.0])
            .unwrap();
        vector
            .set_entity_embedding("dave", vec![0.0, 1.0, 0.0])
            .unwrap();

        // Interaction table
        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "buyer".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "item".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        relational.create_table("purchases", schema).unwrap();

        for (buyer, item) in &[
            ("bob", "book"),
            ("bob", "pen"),
            ("carol", "pen"),
            ("carol", "laptop"),
            ("dave", "phone"),
            ("alice", "book"),
        ] {
            relational
                .insert(
                    "purchases",
                    HashMap::from([
                        ("buyer".into(), Value::String((*buyer).into())),
                        ("item".into(), Value::String((*item).into())),
                    ]),
                )
                .unwrap();
        }

        Arc::new(AdminContext {
            relational,
            vector,
            graph,
            unified: Some(Arc::new(unified)),
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

    fn populated_params() -> ContractionParams {
        ContractionParams {
            source_key: Some("alice".to_string()),
            table: Some("purchases".to_string()),
            source_col: Some("buyer".to_string()),
            target_col: Some("item".to_string()),
            direction: None,
            normalization: None,
            edge_type: None,
            exclude_owned: None,
            top_k: Some(10),
        }
    }

    fn empty_params() -> ContractionParams {
        ContractionParams {
            source_key: None,
            table: None,
            source_col: None,
            target_col: None,
            direction: None,
            normalization: None,
            edge_type: None,
            exclude_owned: None,
            top_k: None,
        }
    }

    fn submitted_params() -> ContractionParams {
        ContractionParams {
            source_key: Some("alice".to_string()),
            table: Some("purchases".to_string()),
            source_col: Some("user_id".to_string()),
            target_col: Some("item_id".to_string()),
            direction: None,
            normalization: None,
            edge_type: None,
            exclude_owned: None,
            top_k: Some(10),
        }
    }

    #[test]
    fn test_contraction_params_defaults() {
        let params = empty_params();
        let config = params.to_config();
        assert_eq!(config.direction, GraphDirection::Symmetric);
        assert_eq!(config.normalization, Normalization::None);
        assert!(config.edge_type.is_none());
        assert!(!config.exclude_owned);
        assert_eq!(config.top_k, 20);
    }

    #[test]
    fn test_contraction_params_outgoing() {
        let params = ContractionParams {
            direction: Some("outgoing".to_string()),
            ..empty_params()
        };
        assert_eq!(params.to_config().direction, GraphDirection::Outgoing);
    }

    #[test]
    fn test_contraction_params_incoming() {
        let params = ContractionParams {
            direction: Some("incoming".to_string()),
            ..empty_params()
        };
        assert_eq!(params.to_config().direction, GraphDirection::Incoming);
    }

    #[test]
    fn test_contraction_params_normalization_total_weight() {
        let params = ContractionParams {
            normalization: Some("total_weight".to_string()),
            ..empty_params()
        };
        assert_eq!(params.to_config().normalization, Normalization::TotalWeight);
    }

    #[test]
    fn test_contraction_params_normalization_per_item() {
        let params = ContractionParams {
            normalization: Some("per_item".to_string()),
            ..empty_params()
        };
        assert_eq!(params.to_config().normalization, Normalization::PerItem);
    }

    #[test]
    fn test_contraction_params_edge_type() {
        let params = ContractionParams {
            edge_type: Some("FRIEND".to_string()),
            ..empty_params()
        };
        assert_eq!(params.to_config().edge_type, Some("FRIEND".to_string()));
    }

    #[test]
    fn test_contraction_params_empty_edge_type_is_none() {
        let params = ContractionParams {
            edge_type: Some(String::new()),
            ..empty_params()
        };
        assert!(params.to_config().edge_type.is_none());
    }

    #[test]
    fn test_contraction_params_exclude_owned() {
        let params = ContractionParams {
            exclude_owned: Some(true),
            ..empty_params()
        };
        assert!(params.to_config().exclude_owned);
    }

    #[test]
    fn test_contraction_params_is_submitted_false() {
        assert!(!empty_params().is_submitted());
    }

    #[test]
    fn test_contraction_params_is_submitted_true() {
        assert!(submitted_params().is_submitted());
    }

    #[test]
    fn test_contraction_params_is_submitted_whitespace_only() {
        let params = ContractionParams {
            source_key: Some("  ".to_string()),
            ..empty_params()
        };
        assert!(!params.is_submitted());
    }

    #[test]
    fn test_contraction_tabs_explainability_active() {
        let tabs = contraction_tabs("/contraction");
        let html = tabs.into_string();
        assert!(html.contains("m-tab active"));
        assert!(html.contains("Explainability"));
    }

    #[test]
    fn test_contraction_tabs_ranking_active() {
        let tabs = contraction_tabs("/contraction/ranking");
        let html = tabs.into_string();
        assert!(html.contains("Ranking"));
    }

    #[test]
    fn test_contraction_form_renders() {
        let ctx = create_test_context();
        let params = empty_params();
        let form = contraction_form(&params, &ctx, "/contraction");
        let html = form.into_string();
        assert!(html.contains("SOURCE ENTITY KEY"));
        assert!(html.contains("INTERACTION TABLE"));
        assert!(html.contains("RUN CONTRACTION"));
    }

    #[test]
    fn test_contraction_form_preserves_values() {
        let ctx = create_test_context();
        let params = submitted_params();
        let form = contraction_form(&params, &ctx, "/contraction");
        let html = form.into_string();
        assert!(html.contains("alice"));
    }

    #[tokio::test]
    async fn test_explainability_no_unified() {
        let ctx = create_test_context();
        let params = Query(empty_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Unified Engine Not Configured"));
    }

    #[tokio::test]
    async fn test_explainability_with_unified_no_submit() {
        let ctx = create_test_context_with_unified();
        let params = Query(empty_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION PARAMETERS"));
        assert!(!html.contains("ADJACENCY"));
    }

    #[tokio::test]
    async fn test_explainability_with_submit_no_neighbors() {
        let ctx = create_test_context_with_unified();
        let params = Query(submitted_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("No graph neighbors found"));
    }

    #[tokio::test]
    async fn test_ranking_no_unified() {
        let ctx = create_test_context();
        let params = Query(empty_params());
        let result = ranking(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Unified Engine Not Configured"));
    }

    #[tokio::test]
    async fn test_ranking_with_submit_no_neighbors() {
        let ctx = create_test_context_with_unified();
        let params = Query(submitted_params());
        let result = ranking(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("No neighbors found"));
    }

    #[tokio::test]
    async fn test_sensitivity_no_unified() {
        let ctx = create_test_context();
        let params = Query(empty_params());
        let result = sensitivity(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Unified Engine Not Configured"));
    }

    #[tokio::test]
    async fn test_sensitivity_missing_table() {
        let ctx = create_test_context_with_unified();
        let params = Query(ContractionParams {
            source_key: Some("alice".to_string()),
            ..empty_params()
        });
        let result = sensitivity(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Specify interaction table"));
    }

    #[tokio::test]
    async fn test_counterfactual_no_unified() {
        let ctx = create_test_context();
        let params = Query(empty_params());
        let result = counterfactual(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Unified Engine Not Configured"));
    }

    #[tokio::test]
    async fn test_counterfactual_missing_table() {
        let ctx = create_test_context_with_unified();
        let params = Query(ContractionParams {
            source_key: Some("alice".to_string()),
            ..empty_params()
        });
        let result = counterfactual(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("Specify interaction table"));
    }

    #[tokio::test]
    async fn test_run_explainability_no_unified() {
        let ctx = create_test_context();
        let params = submitted_params();
        let result = run_explainability(&ctx, &params).await;
        let html = result.into_string();
        assert!(html.is_empty());
    }

    // -- Tests with populated engine data --

    #[tokio::test]
    async fn test_explainability_with_data_renders_adjacency() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("ADJACENCY"));
        assert!(html.contains("neighbors"));
        assert!(html.contains("NEIGHBOR"));
        assert!(html.contains("WEIGHT"));
        assert!(html.contains("SIMILARITY"));
        assert!(html.contains("FUSED"));
    }

    #[tokio::test]
    async fn test_explainability_with_data_shows_interactions() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("INTERACTIONS"));
        assert!(html.contains("intermediaries"));
        assert!(html.contains("interacted with"));
    }

    #[tokio::test]
    async fn test_explainability_with_data_shows_contraction_result() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION RESULT"));
        assert!(html.contains("ITEM"));
        assert!(html.contains("SCORE"));
        assert!(html.contains("CONTRIBUTORS"));
    }

    #[tokio::test]
    async fn test_explainability_no_table_shows_adjacency_only() {
        let ctx = create_populated_context().await;
        let params = Query(ContractionParams {
            source_key: Some("alice".to_string()),
            table: None,
            source_col: None,
            target_col: None,
            ..empty_params()
        });
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("ADJACENCY"));
        // Without table, no contraction result section
        assert!(!html.contains("CONTRACTION RESULT"));
    }

    #[tokio::test]
    async fn test_explainability_with_exclude_owned() {
        let ctx = create_populated_context().await;
        let params = Query(ContractionParams {
            exclude_owned: Some(true),
            ..populated_params()
        });
        let result = explainability(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION RESULT"));
        assert!(html.contains("excluded"));
    }

    #[tokio::test]
    async fn test_ranking_with_data_shows_three_columns() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = ranking(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("VECTOR (SIMILARITY)"));
        assert!(html.contains("GRAPH (ADJACENCY)"));
        assert!(html.contains("FUSED (ADJ x SIM)"));
    }

    #[tokio::test]
    async fn test_ranking_no_table_still_shows_columns() {
        let ctx = create_populated_context().await;
        let params = Query(ContractionParams {
            source_key: Some("alice".to_string()),
            ..empty_params()
        });
        let result = ranking(State(ctx), params).await;
        let html = result.into_string();
        // Ranking only needs graph + vector, not relational
        assert!(html.contains("VECTOR (SIMILARITY)"));
        assert!(html.contains("GRAPH (ADJACENCY)"));
    }

    #[tokio::test]
    async fn test_sensitivity_with_data_shows_comparison() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = sensitivity(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("DEFAULT CONFIG"));
        assert!(html.contains("CURRENT CONFIG"));
    }

    #[tokio::test]
    async fn test_sensitivity_with_direction_change() {
        let ctx = create_populated_context().await;
        let params = Query(ContractionParams {
            direction: Some("outgoing".to_string()),
            ..populated_params()
        });
        let result = sensitivity(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("DEFAULT CONFIG"));
        assert!(html.contains("CURRENT CONFIG"));
    }

    #[tokio::test]
    async fn test_counterfactual_with_data_shows_comparison() {
        let ctx = create_populated_context().await;
        let params = Query(populated_params());
        let result = counterfactual(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("BASELINE (ALL EDGES)"));
        assert!(html.contains("FILTERED"));
    }

    #[tokio::test]
    async fn test_counterfactual_with_edge_filter() {
        let ctx = create_populated_context().await;
        let params = Query(ContractionParams {
            edge_type: Some("FRIEND".to_string()),
            ..populated_params()
        });
        let result = counterfactual(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("BASELINE (ALL EDGES)"));
        assert!(html.contains("FILTERED"));
    }

    #[tokio::test]
    async fn test_contraction_form_with_tables() {
        let ctx = create_populated_context().await;
        let params = empty_params();
        let form = contraction_form(&params, &ctx, "/contraction");
        let html = form.into_string();
        assert!(html.contains("purchases"));
    }

    #[tokio::test]
    async fn test_contraction_form_direction_selected() {
        let ctx = create_populated_context().await;
        let params = ContractionParams {
            direction: Some("outgoing".to_string()),
            ..empty_params()
        };
        let form = contraction_form(&params, &ctx, "/contraction");
        let html = form.into_string();
        assert!(html.contains("Outgoing"));
    }

    #[tokio::test]
    async fn test_contraction_form_normalization_selected() {
        let ctx = create_populated_context().await;
        let params = ContractionParams {
            normalization: Some("per_item".to_string()),
            ..empty_params()
        };
        let form = contraction_form(&params, &ctx, "/contraction");
        let html = form.into_string();
        assert!(html.contains("Per Item"));
    }

    #[test]
    fn test_run_ranking_no_unified() {
        let ctx = create_test_context();
        let params = submitted_params();
        let result = run_ranking(&ctx, &params);
        let html = result.into_string();
        assert!(html.is_empty());
    }

    #[tokio::test]
    async fn test_run_sensitivity_no_unified() {
        let ctx = create_test_context();
        let params = submitted_params();
        let result = run_sensitivity(&ctx, &params).await;
        let html = result.into_string();
        assert!(html.is_empty());
    }

    #[tokio::test]
    async fn test_run_counterfactual_no_unified() {
        let ctx = create_test_context();
        let params = submitted_params();
        let result = run_counterfactual(&ctx, &params).await;
        let html = result.into_string();
        assert!(html.is_empty());
    }

    #[test]
    fn test_contraction_tabs_sensitivity_active() {
        let tabs = contraction_tabs("/contraction/sensitivity");
        let html = tabs.into_string();
        assert!(html.contains("Sensitivity"));
    }

    #[test]
    fn test_contraction_tabs_counterfactual_active() {
        let tabs = contraction_tabs("/contraction/counterfactual");
        let html = tabs.into_string();
        assert!(html.contains("Counterfactual"));
    }

    #[test]
    fn test_contraction_params_top_k_custom() {
        let params = ContractionParams {
            top_k: Some(50),
            ..empty_params()
        };
        assert_eq!(params.to_config().top_k, 50);
    }

    #[tokio::test]
    async fn test_ranking_with_unified_no_submit() {
        let ctx = create_test_context_with_unified();
        let params = Query(empty_params());
        let result = ranking(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION PARAMETERS"));
    }

    #[tokio::test]
    async fn test_sensitivity_with_unified_no_submit() {
        let ctx = create_test_context_with_unified();
        let params = Query(empty_params());
        let result = sensitivity(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION PARAMETERS"));
    }

    #[tokio::test]
    async fn test_counterfactual_with_unified_no_submit() {
        let ctx = create_test_context_with_unified();
        let params = Query(empty_params());
        let result = counterfactual(State(ctx), params).await;
        let html = result.into_string();
        assert!(html.contains("CONTRACTION PARAMETERS"));
    }
}
