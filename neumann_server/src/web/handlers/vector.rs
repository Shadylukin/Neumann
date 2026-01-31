// SPDX-License-Identifier: MIT OR Apache-2.0
//! Handlers for vector engine browsing with dystopian terminal styling.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Form;
use maud::{html, Markup};
use serde::Deserialize;

use tensor_store::{ScalarValue, TensorValue};

use crate::web::templates::layout;
use crate::web::templates::layout::{
    breadcrumb, empty_state, expandable_payload_preview, expandable_string, expandable_vector,
    format_number, page_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// List all collections.
pub async fn collections_list(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let collections = ctx.vector.list_collections();
    let default_count = ctx.vector.count();

    let content = html! {
        (page_header("VECTOR COLLECTIONS", Some("Browse embedding storage")))

        @if collections.is_empty() && default_count == 0 {
            (empty_state("NO VECTORS", "Store embeddings to initialize vector storage"))
        } @else {
            div class="terminal-panel" {
                div class="panel-header" { "COLLECTION REGISTRY" }
                div class="panel-content p-0" {
                    table class="table-rust" {
                        thead {
                            tr {
                                th { "COLLECTION" }
                                th { "VECTORS" }
                                th { "DIMENSION" }
                                th { "METRIC" }
                            }
                        }
                        tbody {
                            // Default collection
                            @if default_count > 0 {
                                tr {
                                    td {
                                        a href="/vector/_default" class="text-rust-blood hover:glow-rust italic" {
                                            "(default)"
                                        }
                                    }
                                    td class="text-amber font-data" { (format_number(default_count)) }
                                    td class="text-phosphor-dim" { "-" }
                                    td class="text-amber" { "Cosine" }
                                }
                            }

                            // Named collections
                            @for coll in &collections {
                                @let count = ctx.vector.collection_count(coll);
                                @let config = ctx.vector.get_collection_config(coll);
                                tr {
                                    td {
                                        a href=(format!("/vector/{coll}")) class="text-rust-blood hover:glow-rust" {
                                            (coll)
                                        }
                                    }
                                    td class="text-amber font-data" { (format_number(count)) }
                                    td class="text-phosphor-dim" {
                                        @if let Some(ref cfg) = config {
                                            @if let Some(dim) = cfg.dimension {
                                                (dim)
                                            } @else {
                                                "-"
                                            }
                                        } @else {
                                            "-"
                                        }
                                    }
                                    td {
                                        @if let Some(ref cfg) = config {
                                            span class="text-amber" {
                                                (format!("{:?}", cfg.distance_metric))
                                            }
                                        } @else {
                                            "-"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div class="mt-4 text-sm text-phosphor-dim font-terminal" {
                "[ " (collections.len() + usize::from(default_count > 0)) " COLLECTION(S) ]"
            }
        }
    };

    layout("Collections", NavItem::Vector, content)
}

/// Show default collection detail.
pub async fn default_collection_detail(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let count = ctx.vector.count();

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), ("", "(default)")]))

        (page_header("DEFAULT COLLECTION", Some(&format!("{} vectors", format_number(count)))))

        // Configuration section
        div class="terminal-panel mb-6" {
            div class="panel-header" { "CONFIGURATION" }
            div class="panel-content" {
                dl class="grid grid-cols-2 gap-4 font-terminal" {
                    div {
                        dt class="text-sm text-phosphor-dim" { "DIMENSION" }
                        dd class="text-lg text-phosphor-dark italic" { "Auto" }
                    }
                    div {
                        dt class="text-sm text-phosphor-dim" { "DISTANCE METRIC" }
                        dd class="text-lg text-amber" { "Cosine" }
                    }
                    div {
                        dt class="text-sm text-phosphor-dim" { "VECTOR COUNT" }
                        dd class="text-lg text-phosphor font-data" { (format_number(count)) }
                    }
                }
            }
        }

        // Action buttons
        div class="flex gap-3" {
            a href="/vector/_default/points" class="btn-terminal" { "[ BROWSE POINTS ]" }
            a href="/vector/_default/search" class="btn-terminal btn-terminal-rust" { "[ SEARCH VECTORS ]" }
        }
    };

    layout("Default Collection", NavItem::Vector, content)
}

/// Show collection detail.
pub async fn collection_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path(collection): Path<String>,
) -> Markup {
    let config = ctx.vector.get_collection_config(&collection);
    let count = ctx.vector.collection_count(&collection);

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), ("", &collection)]))

        (page_header(&collection.to_uppercase(), Some(&format!("{} vectors", format_number(count)))))

        @if let Some(cfg) = config {
            // Configuration section
            div class="terminal-panel mb-6" {
                div class="panel-header" { "CONFIGURATION" }
                div class="panel-content" {
                    dl class="grid grid-cols-2 gap-4 font-terminal" {
                        div {
                            dt class="text-sm text-phosphor-dim" { "DIMENSION" }
                            dd class="text-lg" {
                                @if let Some(dim) = cfg.dimension {
                                    span class="text-phosphor font-data" { (dim) }
                                } @else {
                                    span class="text-phosphor-dark italic" { "Auto" }
                                }
                            }
                        }
                        div {
                            dt class="text-sm text-phosphor-dim" { "DISTANCE METRIC" }
                            dd class="text-lg text-amber" { (format!("{:?}", cfg.distance_metric)) }
                        }
                        div {
                            dt class="text-sm text-phosphor-dim" { "AUTO INDEX" }
                            dd class="text-lg" {
                                @if cfg.auto_index {
                                    span class="text-phosphor" { "Enabled (threshold: " (cfg.auto_index_threshold) ")" }
                                } @else {
                                    span class="text-phosphor-dim" { "Disabled" }
                                }
                            }
                        }
                        div {
                            dt class="text-sm text-phosphor-dim" { "VECTOR COUNT" }
                            dd class="text-lg text-phosphor font-data" { (format_number(count)) }
                        }
                    }
                }
            }

            // Action buttons
            div class="flex gap-3" {
                a href=(format!("/vector/{}/points", collection)) class="btn-terminal" { "[ BROWSE POINTS ]" }
                a href=(format!("/vector/{}/search", collection)) class="btn-terminal btn-terminal-rust" { "[ SEARCH VECTORS ]" }
            }
        } @else {
            (empty_state("COLLECTION NOT FOUND", "The requested collection does not exist"))
        }
    };

    layout(&format!("Collection: {collection}"), NavItem::Vector, content)
}

/// Search form parameters.
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    /// Query vector as comma-separated floats.
    #[serde(default)]
    pub vector: String,
    /// Number of nearest neighbors to return.
    #[serde(default = "default_k")]
    pub k: usize,
}

const fn default_k() -> usize {
    10
}

/// Show default collection search form.
pub async fn default_search_form(State(ctx): State<Arc<AdminContext>>) -> Markup {
    render_default_search_page(&ctx, None, None)
}

/// Handle default collection search submission.
pub async fn default_search_submit(
    State(ctx): State<Arc<AdminContext>>,
    Form(params): Form<SearchParams>,
) -> Markup {
    let vector: Result<Vec<f32>, _> = params
        .vector
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect();

    match vector {
        Ok(vec) if !vec.is_empty() => {
            let results = ctx.vector.search_similar(&vec, params.k);
            render_default_search_page(&ctx, Some(&params), Some(results))
        },
        Ok(_) => render_default_search_page(
            &ctx,
            Some(&params),
            Some(Err(vector_engine::VectorError::EmptyVector)),
        ),
        Err(_) => render_default_search_page(
            &ctx,
            Some(&params),
            Some(Err(vector_engine::VectorError::ConfigurationError(
                "Invalid vector format".to_string(),
            ))),
        ),
    }
}

fn render_default_search_page(
    ctx: &AdminContext,
    params: Option<&SearchParams>,
    results: Option<Result<Vec<vector_engine::SearchResult>, vector_engine::VectorError>>,
) -> Markup {
    let sample_vector = get_sample_vector_default(ctx);

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), ("/vector/_default", "(default)"), ("", "SEARCH")]))

        (page_header("VECTOR SEARCH", Some("Query default collection")))

        // Search form
        div class="terminal-panel mb-6" {
            div class="panel-header" { "SEARCH PARAMETERS" }
            div class="panel-content" {
                form method="post" action="/vector/_default/search" class="space-y-4" {
                    div {
                        div class="flex items-center justify-between mb-2" {
                            label for="vector" class="text-sm text-phosphor-dim font-terminal" {
                                "QUERY VECTOR"
                            }
                            @if let Some((ref sample_key, ref sample_vec)) = sample_vector {
                                button type="button" onclick=(format!("document.getElementById('vector').value = '{}';", format_vector_compact(sample_vec))) class="expand-btn-terminal" {
                                    "[USE SAMPLE: " (truncate_key(sample_key, 15)) "]"
                                }
                            }
                        }
                        textarea
                            id="vector"
                            name="vector"
                            rows="3"
                            class="input-terminal w-full"
                            placeholder="[0.1, 0.2, 0.3, ...] - Click 'USE SAMPLE' to fill with existing vector"
                        {
                            @if let Some(p) = params {
                                (p.vector.clone())
                            }
                        }
                    }

                    div class="flex items-end gap-4" {
                        div {
                            label for="k" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "RESULTS (K)"
                            }
                            input
                                type="number"
                                id="k"
                                name="k"
                                min="1"
                                max="100"
                                value=(params.map_or(10, |p| p.k))
                                class="input-terminal w-24";
                        }

                        button type="submit" class="btn-terminal btn-terminal-rust" { "[ SEARCH ]" }
                    }
                }
            }
        }

        // Results
        @if let Some(result) = results {
            @match result {
                Ok(hits) => {
                    div class="terminal-panel" {
                        div class="panel-header" { "RESULTS (" (hits.len()) ")" }
                        div class="panel-content p-0" {
                            @if hits.is_empty() {
                                div class="p-4 text-phosphor-dim italic" { "< NO MATCHES FOUND >" }
                            } @else {
                                table class="table-rust" {
                                    thead {
                                        tr {
                                            th class="w-16" { "#" }
                                            th { "KEY" }
                                            th class="text-right w-32" { "SCORE" }
                                        }
                                    }
                                    tbody {
                                        @for (idx, hit) in hits.iter().enumerate() {
                                            tr {
                                                td class="text-phosphor-dim font-data" { (idx + 1) }
                                                td class="text-phosphor" { (hit.key.clone()) }
                                                td class="text-right" {
                                                    span class=(score_color(hit.score)) {
                                                        (format!("{:.4}", hit.score))
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    div class="terminal-panel terminal-panel-rust" {
                        div class="panel-header" { "ERROR" }
                        div class="panel-content text-amber" {
                            (e.to_string())
                        }
                    }
                }
            }
        }
    };

    layout("Vector Search", NavItem::Vector, content)
}

/// Show search form.
pub async fn search_form(
    State(ctx): State<Arc<AdminContext>>,
    Path(collection): Path<String>,
) -> Markup {
    render_search_page(&ctx, &collection, None, None)
}

/// Handle search submission.
pub async fn search_submit(
    State(ctx): State<Arc<AdminContext>>,
    Path(collection): Path<String>,
    Form(params): Form<SearchParams>,
) -> Markup {
    let vector: Result<Vec<f32>, _> = params
        .vector
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect();

    match vector {
        Ok(vec) if !vec.is_empty() => {
            let results = ctx.vector.search_in_collection(&collection, &vec, params.k);
            render_search_page(&ctx, &collection, Some(&params), Some(results))
        },
        Ok(_) => render_search_page(
            &ctx,
            &collection,
            Some(&params),
            Some(Err(vector_engine::VectorError::EmptyVector)),
        ),
        Err(_) => render_search_page(
            &ctx,
            &collection,
            Some(&params),
            Some(Err(vector_engine::VectorError::ConfigurationError(
                "Invalid vector format".to_string(),
            ))),
        ),
    }
}

fn render_search_page(
    ctx: &AdminContext,
    collection: &str,
    params: Option<&SearchParams>,
    results: Option<Result<Vec<vector_engine::SearchResult>, vector_engine::VectorError>>,
) -> Markup {
    let config = ctx.vector.get_collection_config(collection);
    let sample_vector = get_sample_vector(ctx, collection);

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), (&format!("/vector/{collection}"), collection), ("", "SEARCH")]))

        (page_header("VECTOR SEARCH", Some(&format!("Query {collection}"))))

        // Search form
        div class="terminal-panel mb-6" {
            div class="panel-header" { "SEARCH PARAMETERS" }
            div class="panel-content" {
                form method="post" action=(format!("/vector/{collection}/search")) class="space-y-4" {
                    div {
                        div class="flex items-center justify-between mb-2" {
                            label for="vector" class="text-sm text-phosphor-dim font-terminal" {
                                "QUERY VECTOR"
                            }
                            @if let Some((ref sample_key, ref sample_vec)) = sample_vector {
                                button type="button" onclick=(format!("document.getElementById('vector').value = '{}';", format_vector_compact(sample_vec))) class="expand-btn-terminal" {
                                    "[USE SAMPLE: " (truncate_key(sample_key, 15)) "]"
                                }
                            }
                        }
                        textarea
                            id="vector"
                            name="vector"
                            rows="3"
                            class="input-terminal w-full"
                            placeholder="[0.1, 0.2, 0.3, ...]"
                        {
                            @if let Some(p) = params {
                                (p.vector.clone())
                            }
                        }
                        @if let Some(ref cfg) = config {
                            @if let Some(dim) = cfg.dimension {
                                p class="text-xs text-phosphor-dark mt-1 font-terminal" { "Expected dimension: " (dim) }
                            }
                        }
                    }

                    div class="flex items-end gap-4" {
                        div {
                            label for="k" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "RESULTS (K)"
                            }
                            input
                                type="number"
                                id="k"
                                name="k"
                                min="1"
                                max="100"
                                value=(params.map_or(10, |p| p.k))
                                class="input-terminal w-24";
                        }

                        button type="submit" class="btn-terminal btn-terminal-rust" { "[ SEARCH ]" }
                    }
                }
            }
        }

        // Results
        @if let Some(result) = results {
            @match result {
                Ok(hits) => {
                    div class="terminal-panel" {
                        div class="panel-header" { "RESULTS (" (hits.len()) ")" }
                        div class="panel-content p-0" {
                            @if hits.is_empty() {
                                div class="p-4 text-phosphor-dim italic" { "< NO MATCHES FOUND >" }
                            } @else {
                                table class="table-rust" {
                                    thead {
                                        tr {
                                            th class="w-16" { "#" }
                                            th { "KEY" }
                                            th class="text-right w-32" { "SCORE" }
                                        }
                                    }
                                    tbody {
                                        @for (idx, hit) in hits.iter().enumerate() {
                                            tr {
                                                td class="text-phosphor-dim font-data" { (idx + 1) }
                                                td class="text-phosphor" { (hit.key.clone()) }
                                                td class="text-right" {
                                                    span class=(score_color(hit.score)) {
                                                        (format!("{:.4}", hit.score))
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    div class="terminal-panel terminal-panel-rust" {
                        div class="panel-header" { "ERROR" }
                        div class="panel-content text-amber" {
                            (e.to_string())
                        }
                    }
                }
            }
        }
    };

    layout("Vector Search", NavItem::Vector, content)
}

fn score_color(score: f32) -> &'static str {
    if score >= 0.9 {
        "text-phosphor font-data glow-phosphor"
    } else if score >= 0.7 {
        "text-amber font-data"
    } else if score >= 0.5 {
        "text-rust-blood font-data"
    } else {
        "text-phosphor-dim font-data"
    }
}

// ========== Points List Handlers ==========

/// List points in default collection.
pub async fn default_points_list(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let keys = ctx.vector.list_keys();

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), ("/vector/_default", "(default)"), ("", "POINTS")]))

        (page_header("POINTS", Some(&format!("{} vectors in default collection", format_number(keys.len())))))

        @if keys.is_empty() {
            (empty_state("NO POINTS", "This collection is empty"))
        } @else {
            div class="terminal-panel" {
                div class="panel-header" { "POINT REGISTRY" }
                div class="panel-content p-0" {
                    table class="table-rust" {
                        thead {
                            tr {
                                th { "ID" }
                                th { "PAYLOAD PREVIEW" }
                                th class="w-24" { "" }
                            }
                        }
                        tbody {
                            @for key in &keys {
                                @let metadata = ctx.vector.get_metadata(key).unwrap_or_default();
                                tr {
                                    td {
                                        a href=(format!("/vector/_default/points/{}", urlencoding::encode(key))) class="text-rust-blood hover:glow-rust" {
                                            (key)
                                        }
                                    }
                                    td class="text-phosphor-dim text-sm" {
                                        @if metadata.is_empty() {
                                            span class="italic" { "No payload" }
                                        } @else {
                                            (render_payload_preview(&metadata))
                                        }
                                    }
                                    td class="text-right" {
                                        a href=(format!("/vector/_default/points/{}", urlencoding::encode(key))) class="expand-btn-terminal" {
                                            "[VIEW]"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div class="mt-4 text-sm text-phosphor-dim font-terminal" {
                "[ " (keys.len()) " POINT(S) ]"
            }
        }
    };

    layout("Points", NavItem::Vector, content)
}

/// List points in a named collection.
pub async fn points_list(
    State(ctx): State<Arc<AdminContext>>,
    Path(collection): Path<String>,
) -> Markup {
    let keys = ctx.vector.list_collection_keys(&collection);

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), (&format!("/vector/{collection}"), &collection), ("", "POINTS")]))

        (page_header("POINTS", Some(&format!("{} vectors in {}", format_number(keys.len()), collection))))

        @if keys.is_empty() {
            (empty_state("NO POINTS", "This collection is empty"))
        } @else {
            div class="terminal-panel" {
                div class="panel-header" { "POINT REGISTRY" }
                div class="panel-content p-0" {
                    table class="table-rust" {
                        thead {
                            tr {
                                th { "ID" }
                                th { "PAYLOAD PREVIEW" }
                                th class="w-24" { "" }
                            }
                        }
                        tbody {
                            @for key in &keys {
                                @let metadata = ctx.vector.get_collection_metadata(&collection, key).unwrap_or_default();
                                tr {
                                    td {
                                        a href=(format!("/vector/{}/points/{}", collection, urlencoding::encode(key))) class="text-rust-blood hover:glow-rust" {
                                            (key)
                                        }
                                    }
                                    td class="text-phosphor-dim text-sm" {
                                        @if metadata.is_empty() {
                                            span class="italic" { "No payload" }
                                        } @else {
                                            (render_payload_preview(&metadata))
                                        }
                                    }
                                    td class="text-right" {
                                        a href=(format!("/vector/{}/points/{}", collection, urlencoding::encode(key))) class="expand-btn-terminal" {
                                            "[VIEW]"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div class="mt-4 text-sm text-phosphor-dim font-terminal" {
                "[ " (keys.len()) " POINT(S) ]"
            }
        }
    };

    layout("Points", NavItem::Vector, content)
}

// ========== Point Detail Handlers ==========

/// Show point detail in default collection.
pub async fn default_point_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path(point_id): Path<String>,
) -> Markup {
    let vector = ctx.vector.get_embedding(&point_id);
    let metadata = ctx.vector.get_metadata(&point_id).unwrap_or_default();

    let content = html! {
        (breadcrumb(&[("/vector", "COLLECTIONS"), ("/vector/_default", "(default)"), ("/vector/_default/points", "POINTS"), ("", &point_id)]))

        (page_header(&point_id, Some("Point Details")))

        @match vector {
            Ok(vec) => {
                // Payload section
                div class="terminal-panel mb-6" {
                    div class="panel-header" { "PAYLOAD" }
                    div class="panel-content" {
                        @if metadata.is_empty() {
                            div class="text-phosphor-dim italic" { "< NO PAYLOAD DATA >" }
                        } @else {
                            (render_payload_table(&metadata))
                        }
                    }
                }

                // Vector section
                div class="terminal-panel" {
                    div class="panel-header" { "VECTOR (" (vec.len()) " DIMENSIONS)" }
                    div class="panel-content" {
                        (expandable_vector(&vec, 15))
                    }
                }
            }
            Err(_) => {
                (empty_state("POINT NOT FOUND", "The requested point does not exist"))
            }
        }
    };

    layout(&format!("Point: {point_id}"), NavItem::Vector, content)
}

/// Show point detail in a named collection.
pub async fn point_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path((collection, point_id)): Path<(String, String)>,
) -> Markup {
    let vector = ctx.vector.get_from_collection(&collection, &point_id);
    let metadata = ctx.vector.get_collection_metadata(&collection, &point_id).unwrap_or_default();

    let content = html! {
        (breadcrumb(&[
            ("/vector", "COLLECTIONS"),
            (&format!("/vector/{collection}"), &collection),
            (&format!("/vector/{collection}/points"), "POINTS"),
            ("", &point_id)
        ]))

        (page_header(&point_id, Some(&format!("Point in {collection}"))))

        @match vector {
            Ok(vec) => {
                // Payload section
                div class="terminal-panel mb-6" {
                    div class="panel-header" { "PAYLOAD" }
                    div class="panel-content" {
                        @if metadata.is_empty() {
                            div class="text-phosphor-dim italic" { "< NO PAYLOAD DATA >" }
                        } @else {
                            (render_payload_table(&metadata))
                        }
                    }
                }

                // Vector section
                div class="terminal-panel" {
                    div class="panel-header" { "VECTOR (" (vec.len()) " DIMENSIONS)" }
                    div class="panel-content" {
                        (expandable_vector(&vec, 15))
                    }
                }
            }
            Err(_) => {
                (empty_state("POINT NOT FOUND", "The requested point does not exist"))
            }
        }
    };

    layout(&format!("Point: {point_id}"), NavItem::Vector, content)
}

// ========== Helper Functions ==========

fn render_payload_preview(metadata: &HashMap<String, TensorValue>) -> Markup {
    let items: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| (k.clone(), format_tensor_value_short(v)))
        .collect();

    expandable_payload_preview(&items, 3)
}

fn format_tensor_value_short(value: &TensorValue) -> String {
    match value {
        TensorValue::Scalar(ScalarValue::String(s)) => {
            if s.len() > 30 {
                format!("\"{}...\"", &s[..30])
            } else {
                format!("\"{s}\"")
            }
        },
        TensorValue::Scalar(s) => format!("{s:?}"),
        TensorValue::Vector(v) => format!("[{}d vector]", v.len()),
        TensorValue::Sparse(s) => format!("[sparse {}d]", s.dimension()),
        TensorValue::Pointer(p) => format!("-> {p}"),
        TensorValue::Pointers(ps) => format!("[{} pointers]", ps.len()),
    }
}

fn render_payload_table(metadata: &HashMap<String, TensorValue>) -> Markup {
    html! {
        table class="table-rust w-full" {
            thead {
                tr {
                    th class="w-48" { "KEY" }
                    th { "VALUE" }
                }
            }
            tbody {
                @for (key, value) in metadata {
                    tr {
                        td class="text-phosphor" { (key) }
                        td {
                            (render_tensor_value(value))
                        }
                    }
                }
            }
        }
    }
}

fn render_tensor_value(value: &TensorValue) -> Markup {
    match value {
        TensorValue::Scalar(ScalarValue::String(s)) => expandable_string(s, 100),
        TensorValue::Scalar(s) => {
            html! { span class="text-phosphor font-data" { (format!("{s:?}")) } }
        },
        TensorValue::Vector(v) => expandable_vector(v, 10),
        TensorValue::Sparse(s) => {
            html! { span class="text-amber font-data" { "[sparse " (s.dimension()) "d]" } }
        },
        TensorValue::Pointer(p) => {
            html! { span class="text-phosphor-dim" { "-> " (p) } }
        },
        TensorValue::Pointers(ps) => {
            html! { span class="text-phosphor-dim" { "[" (ps.len()) " pointers]" } }
        },
    }
}

fn get_sample_vector_default(ctx: &AdminContext) -> Option<(String, Vec<f32>)> {
    let keys = ctx.vector.list_keys();
    if let Some(first_key) = keys.first() {
        if let Ok(vec) = ctx.vector.get_embedding(first_key) {
            return Some((first_key.clone(), vec));
        }
    }
    None
}

fn get_sample_vector(ctx: &AdminContext, collection: &str) -> Option<(String, Vec<f32>)> {
    let keys = ctx.vector.list_collection_keys(collection);
    if let Some(first_key) = keys.first() {
        if let Ok(vec) = ctx.vector.get_from_collection(collection, first_key) {
            return Some((first_key.clone(), vec));
        }
    }
    None
}

fn format_vector_compact(vec: &[f32]) -> String {
    format!(
        "[{}]",
        vec.iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn truncate_key(key: &str, max_len: usize) -> String {
    if key.len() <= max_len {
        key.to_string()
    } else {
        format!("{}...", &key[..max_len - 3])
    }
}
