// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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

    layout(
        &format!("Collection: {collection}"),
        NavItem::Vector,
        content,
    )
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

#[allow(clippy::too_many_lines)]
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
    let metadata = ctx
        .vector
        .get_collection_metadata(&collection, &point_id)
        .unwrap_or_default();

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

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_store::SparseVector;

    // ========== score_color tests ==========

    #[test]
    fn test_score_color_high_score() {
        assert_eq!(score_color(0.95), "text-phosphor font-data glow-phosphor");
        assert_eq!(score_color(0.9), "text-phosphor font-data glow-phosphor");
        assert_eq!(score_color(1.0), "text-phosphor font-data glow-phosphor");
    }

    #[test]
    fn test_score_color_medium_high_score() {
        assert_eq!(score_color(0.89), "text-amber font-data");
        assert_eq!(score_color(0.7), "text-amber font-data");
        assert_eq!(score_color(0.75), "text-amber font-data");
    }

    #[test]
    fn test_score_color_medium_score() {
        assert_eq!(score_color(0.69), "text-rust-blood font-data");
        assert_eq!(score_color(0.5), "text-rust-blood font-data");
        assert_eq!(score_color(0.55), "text-rust-blood font-data");
    }

    #[test]
    fn test_score_color_low_score() {
        assert_eq!(score_color(0.49), "text-phosphor-dim font-data");
        assert_eq!(score_color(0.0), "text-phosphor-dim font-data");
        assert_eq!(score_color(0.25), "text-phosphor-dim font-data");
    }

    #[test]
    fn test_score_color_boundary_values() {
        // Exact boundaries
        assert_eq!(score_color(0.9), "text-phosphor font-data glow-phosphor");
        assert_eq!(score_color(0.7), "text-amber font-data");
        assert_eq!(score_color(0.5), "text-rust-blood font-data");
    }

    // ========== format_tensor_value_short tests ==========

    #[test]
    fn test_format_tensor_value_short_string_short() {
        let value = TensorValue::Scalar(ScalarValue::String("hello".to_string()));
        assert_eq!(format_tensor_value_short(&value), "\"hello\"");
    }

    #[test]
    fn test_format_tensor_value_short_string_long() {
        let long_string = "a".repeat(50);
        let value = TensorValue::Scalar(ScalarValue::String(long_string));
        let result = format_tensor_value_short(&value);
        assert!(result.starts_with("\"aaaaaa"));
        assert!(result.ends_with("...\""));
        assert!(result.len() < 40); // truncated
    }

    #[test]
    fn test_format_tensor_value_short_string_exact_30() {
        let exact_30 = "a".repeat(30);
        let value = TensorValue::Scalar(ScalarValue::String(exact_30.clone()));
        let result = format_tensor_value_short(&value);
        assert_eq!(result, format!("\"{}\"", exact_30));
    }

    #[test]
    fn test_format_tensor_value_short_int() {
        let value = TensorValue::Scalar(ScalarValue::Int(42));
        let result = format_tensor_value_short(&value);
        assert!(result.contains("42"));
    }

    #[test]
    fn test_format_tensor_value_short_float() {
        let value = TensorValue::Scalar(ScalarValue::Float(3.14));
        let result = format_tensor_value_short(&value);
        assert!(result.contains("3.14"));
    }

    #[test]
    fn test_format_tensor_value_short_bool() {
        let value = TensorValue::Scalar(ScalarValue::Bool(true));
        let result = format_tensor_value_short(&value);
        assert!(result.contains("true"));
    }

    #[test]
    fn test_format_tensor_value_short_null() {
        let value = TensorValue::Scalar(ScalarValue::Null);
        let result = format_tensor_value_short(&value);
        assert!(result.contains("Null"));
    }

    #[test]
    fn test_format_tensor_value_short_vector() {
        let value = TensorValue::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(format_tensor_value_short(&value), "[5d vector]");
    }

    #[test]
    fn test_format_tensor_value_short_vector_empty() {
        let value = TensorValue::Vector(vec![]);
        assert_eq!(format_tensor_value_short(&value), "[0d vector]");
    }

    #[test]
    fn test_format_tensor_value_short_sparse() {
        let mut sparse = SparseVector::new(100);
        sparse.set(0, 1.0);
        sparse.set(50, 2.0);
        sparse.set(99, 3.0);
        let value = TensorValue::Sparse(sparse);
        assert_eq!(format_tensor_value_short(&value), "[sparse 100d]");
    }

    #[test]
    fn test_format_tensor_value_short_pointer() {
        let value = TensorValue::Pointer("entity_123".to_string());
        assert_eq!(format_tensor_value_short(&value), "-> entity_123");
    }

    #[test]
    fn test_format_tensor_value_short_pointers() {
        let value = TensorValue::Pointers(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(format_tensor_value_short(&value), "[3 pointers]");
    }

    #[test]
    fn test_format_tensor_value_short_pointers_empty() {
        let value = TensorValue::Pointers(vec![]);
        assert_eq!(format_tensor_value_short(&value), "[0 pointers]");
    }

    // ========== format_vector_compact tests ==========

    #[test]
    fn test_format_vector_compact_empty() {
        let result = format_vector_compact(&[]);
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_format_vector_compact_single() {
        let result = format_vector_compact(&[1.5]);
        assert_eq!(result, "[1.500000]");
    }

    #[test]
    fn test_format_vector_compact_multiple() {
        let result = format_vector_compact(&[1.0, 2.5, 3.14159]);
        assert_eq!(result, "[1.000000, 2.500000, 3.141590]");
    }

    #[test]
    fn test_format_vector_compact_negative() {
        let result = format_vector_compact(&[-1.0, 0.0, 1.0]);
        assert_eq!(result, "[-1.000000, 0.000000, 1.000000]");
    }

    #[test]
    fn test_format_vector_compact_precision() {
        let result = format_vector_compact(&[0.123456789]);
        assert_eq!(result, "[0.123457]"); // 6 decimal places
    }

    // ========== truncate_key tests ==========

    #[test]
    fn test_truncate_key_short() {
        assert_eq!(truncate_key("short", 10), "short");
    }

    #[test]
    fn test_truncate_key_exact_length() {
        assert_eq!(truncate_key("exactly10!", 10), "exactly10!");
    }

    #[test]
    fn test_truncate_key_long() {
        assert_eq!(truncate_key("this_is_a_very_long_key", 10), "this_is...");
    }

    #[test]
    fn test_truncate_key_unicode() {
        // Note: truncation works on bytes, be careful with unicode
        assert_eq!(truncate_key("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_key_minimum_length() {
        // With max_len=4, we'd have 1 char + "..."
        assert_eq!(truncate_key("abcdef", 4), "a...");
    }

    // ========== render_tensor_value tests ==========

    #[test]
    fn test_render_tensor_value_string() {
        let value = TensorValue::Scalar(ScalarValue::String("test".to_string()));
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("test"));
    }

    #[test]
    fn test_render_tensor_value_int() {
        let value = TensorValue::Scalar(ScalarValue::Int(100));
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("100"));
    }

    #[test]
    fn test_render_tensor_value_vector() {
        let value = TensorValue::Vector(vec![1.0, 2.0]);
        let html = render_tensor_value(&value).into_string();
        // Should contain the expandable vector markup
        assert!(html.contains("1") || html.contains("2"));
    }

    #[test]
    fn test_render_tensor_value_sparse() {
        let mut sparse = SparseVector::new(50);
        sparse.set(10, 1.0);
        let value = TensorValue::Sparse(sparse);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("sparse"));
        assert!(html.contains("50"));
    }

    #[test]
    fn test_render_tensor_value_pointer() {
        let value = TensorValue::Pointer("ref_123".to_string());
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("ref_123"));
        assert!(html.contains("-&gt;")); // HTML escaped ->
    }

    #[test]
    fn test_render_tensor_value_pointers() {
        let value = TensorValue::Pointers(vec!["a".to_string(), "b".to_string()]);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("2 pointers"));
    }

    // ========== render_payload_preview tests ==========

    #[test]
    fn test_render_payload_preview_empty() {
        let metadata = HashMap::new();
        let html = render_payload_preview(&metadata).into_string();
        // Empty map should produce minimal output
        assert!(!html.is_empty());
    }

    #[test]
    fn test_render_payload_preview_single_item() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String("test".to_string())),
        );
        let html = render_payload_preview(&metadata).into_string();
        assert!(html.contains("name"));
    }

    #[test]
    fn test_render_payload_preview_multiple_items() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), TensorValue::Scalar(ScalarValue::Int(1)));
        metadata.insert("key2".to_string(), TensorValue::Scalar(ScalarValue::Int(2)));
        metadata.insert("key3".to_string(), TensorValue::Scalar(ScalarValue::Int(3)));
        metadata.insert("key4".to_string(), TensorValue::Scalar(ScalarValue::Int(4)));
        let html = render_payload_preview(&metadata).into_string();
        // Should show preview (limited items due to expandable_payload_preview limit of 3)
        assert!(!html.is_empty());
    }

    // ========== render_payload_table tests ==========

    #[test]
    fn test_render_payload_table_empty() {
        let metadata = HashMap::new();
        let html = render_payload_table(&metadata).into_string();
        assert!(html.contains("table"));
        assert!(html.contains("KEY"));
        assert!(html.contains("VALUE"));
    }

    #[test]
    fn test_render_payload_table_with_data() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String("Alice".to_string())),
        );
        metadata.insert("age".to_string(), TensorValue::Scalar(ScalarValue::Int(30)));
        let html = render_payload_table(&metadata).into_string();
        assert!(html.contains("name"));
        assert!(html.contains("Alice"));
        assert!(html.contains("age"));
        assert!(html.contains("30"));
    }

    // ========== SearchParams tests ==========

    #[test]
    fn test_search_params_struct() {
        let params = SearchParams {
            vector: "1.0,2.0,3.0".to_string(),
            k: 5,
        };
        assert_eq!(params.vector, "1.0,2.0,3.0");
        assert_eq!(params.k, 5);
    }

    #[test]
    fn test_search_params_empty_vector() {
        let params = SearchParams {
            vector: String::new(),
            k: 10,
        };
        assert!(params.vector.is_empty());
        assert_eq!(params.k, 10);
    }

    #[test]
    fn test_search_params_debug() {
        let params = SearchParams {
            vector: "test".to_string(),
            k: 20,
        };
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("SearchParams"));
        assert!(debug_str.contains("test"));
    }

    // ========== default_k tests ==========

    #[test]
    fn test_default_k_value() {
        assert_eq!(default_k(), 10);
    }

    // ========== format_tensor_value_short bytes tests ==========

    #[test]
    fn test_format_tensor_value_short_bytes_small() {
        let value = TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3, 4, 5]));
        let result = format_tensor_value_short(&value);
        // Uses debug format for ScalarValue
        assert!(result.contains("Bytes"));
    }

    #[test]
    fn test_format_tensor_value_short_bytes_empty() {
        let value = TensorValue::Scalar(ScalarValue::Bytes(vec![]));
        let result = format_tensor_value_short(&value);
        assert!(result.contains("Bytes"));
    }

    // ========== render_tensor_value additional tests ==========

    #[test]
    fn test_render_tensor_value_bool() {
        let value = TensorValue::Scalar(ScalarValue::Bool(false));
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("false"));
    }

    #[test]
    fn test_render_tensor_value_float() {
        let value = TensorValue::Scalar(ScalarValue::Float(3.14159));
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("3.14"));
    }

    #[test]
    fn test_render_tensor_value_null() {
        let value = TensorValue::Scalar(ScalarValue::Null);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("NULL") || html.contains("Null"));
    }

    #[test]
    fn test_render_tensor_value_bytes() {
        let value = TensorValue::Scalar(ScalarValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]));
        let html = render_tensor_value(&value).into_string();
        // Bytes are rendered with debug format
        assert!(html.contains("Bytes") || html.contains("222"));
    }

    #[test]
    fn test_render_tensor_value_pointers_empty() {
        let value = TensorValue::Pointers(vec![]);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("0 pointers"));
    }

    // ========== truncate_key edge cases ==========

    #[test]
    fn test_truncate_key_empty() {
        assert_eq!(truncate_key("", 10), "");
    }

    #[test]
    fn test_truncate_key_max_3() {
        // Very small max length
        assert_eq!(truncate_key("abcdef", 3), "...");
    }

    // ========== format_vector_compact edge cases ==========

    #[test]
    fn test_format_vector_compact_large() {
        let vec: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let result = format_vector_compact(&vec);
        assert!(result.starts_with('['));
        assert!(result.ends_with(']'));
        // Should have 100 elements
        assert_eq!(result.matches(',').count(), 99);
    }

    // ========== render_payload_preview additional tests ==========

    #[test]
    fn test_render_payload_preview_many_items() {
        let mut metadata = HashMap::new();
        for i in 0..10 {
            metadata.insert(
                format!("key{}", i),
                TensorValue::Scalar(ScalarValue::Int(i as i64)),
            );
        }
        let html = render_payload_preview(&metadata).into_string();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_render_payload_preview_nested_types() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "vector".to_string(),
            TensorValue::Vector(vec![1.0, 2.0, 3.0]),
        );
        metadata.insert(
            "pointer".to_string(),
            TensorValue::Pointer("ref_id".to_string()),
        );
        let html = render_payload_preview(&metadata).into_string();
        assert!(!html.is_empty());
    }

    // ========== render_payload_table additional tests ==========

    #[test]
    fn test_render_payload_table_multiple_keys() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "z_key".to_string(),
            TensorValue::Scalar(ScalarValue::Int(1)),
        );
        metadata.insert(
            "a_key".to_string(),
            TensorValue::Scalar(ScalarValue::Int(2)),
        );
        metadata.insert(
            "m_key".to_string(),
            TensorValue::Scalar(ScalarValue::Int(3)),
        );
        let html = render_payload_table(&metadata).into_string();
        // All keys should be present
        assert!(html.contains("a_key"));
        assert!(html.contains("m_key"));
        assert!(html.contains("z_key"));
    }

    #[test]
    fn test_render_payload_table_various_types() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "str".to_string(),
            TensorValue::Scalar(ScalarValue::String("text".to_string())),
        );
        metadata.insert("int".to_string(), TensorValue::Scalar(ScalarValue::Int(42)));
        metadata.insert(
            "float".to_string(),
            TensorValue::Scalar(ScalarValue::Float(3.14)),
        );
        metadata.insert(
            "bool".to_string(),
            TensorValue::Scalar(ScalarValue::Bool(true)),
        );
        metadata.insert("null".to_string(), TensorValue::Scalar(ScalarValue::Null));
        let html = render_payload_table(&metadata).into_string();
        assert!(html.contains("text"));
        assert!(html.contains("42"));
        assert!(html.contains("3.14"));
        assert!(html.contains("true"));
    }

    // ========== Additional score_color edge cases ==========

    #[test]
    fn test_score_color_exact_boundaries() {
        // Test the exact boundary values for each threshold
        assert_eq!(score_color(0.8999999), "text-amber font-data");
        assert_eq!(score_color(0.6999999), "text-rust-blood font-data");
        assert_eq!(score_color(0.4999999), "text-phosphor-dim font-data");
    }

    #[test]
    fn test_score_color_negative() {
        // Edge case: negative scores
        assert_eq!(score_color(-0.1), "text-phosphor-dim font-data");
        assert_eq!(score_color(-1.0), "text-phosphor-dim font-data");
    }

    #[test]
    fn test_score_color_over_one() {
        // Edge case: scores > 1.0
        assert_eq!(score_color(1.1), "text-phosphor font-data glow-phosphor");
        assert_eq!(score_color(2.0), "text-phosphor font-data glow-phosphor");
    }

    // ========== Additional format_tensor_value_short tests ==========

    #[test]
    fn test_format_tensor_value_short_string_exactly_31_chars() {
        let string_31 = "a".repeat(31);
        let value = TensorValue::Scalar(ScalarValue::String(string_31));
        let result = format_tensor_value_short(&value);
        // Should be truncated because > 30
        assert!(result.ends_with("...\""));
    }

    #[test]
    fn test_format_tensor_value_short_vector_high_dimension() {
        let value = TensorValue::Vector(vec![0.0; 1024]);
        assert_eq!(format_tensor_value_short(&value), "[1024d vector]");
    }

    #[test]
    fn test_format_tensor_value_short_sparse_zero_nnz() {
        let sparse = SparseVector::new(100);
        let value = TensorValue::Sparse(sparse);
        assert_eq!(format_tensor_value_short(&value), "[sparse 100d]");
    }

    #[test]
    fn test_format_tensor_value_short_pointers_single() {
        let value = TensorValue::Pointers(vec!["single".to_string()]);
        assert_eq!(format_tensor_value_short(&value), "[1 pointers]");
    }

    // ========== Additional truncate_key tests ==========

    #[test]
    fn test_truncate_key_max_4() {
        // With max_len=4, we'd have 1 char + "..."
        assert_eq!(truncate_key("abcdefgh", 4), "a...");
    }

    #[test]
    fn test_truncate_key_max_5() {
        assert_eq!(truncate_key("abcdefgh", 5), "ab...");
    }

    #[test]
    fn test_truncate_key_max_6() {
        assert_eq!(truncate_key("abcdefgh", 6), "abc...");
    }

    #[test]
    fn test_truncate_key_large_string() {
        let long_string = "x".repeat(1000);
        let result = truncate_key(&long_string, 50);
        assert_eq!(result.len(), 50);
        assert!(result.ends_with("..."));
    }

    // ========== Additional format_vector_compact tests ==========

    #[test]
    fn test_format_vector_compact_special_values() {
        let result = format_vector_compact(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        assert!(result.contains("inf") || result.contains("NaN"));
    }

    #[test]
    fn test_format_vector_compact_very_small_values() {
        let result = format_vector_compact(&[0.0000001, 0.0000002]);
        assert!(result.contains("0."));
    }

    #[test]
    fn test_format_vector_compact_very_large_values() {
        let result = format_vector_compact(&[1000000.0, 2000000.0]);
        assert!(result.contains("1000000"));
    }

    // ========== score_color boundary tests ==========

    #[test]
    fn test_score_color_boundary_at_seven() {
        // Test value just at and below 0.7
        assert_eq!(score_color(0.7), "text-amber font-data");
        assert_eq!(score_color(0.699999), "text-rust-blood font-data");
    }

    #[test]
    fn test_score_color_boundary_at_five() {
        // Test value just at and below 0.5
        assert_eq!(score_color(0.5), "text-rust-blood font-data");
        assert_eq!(score_color(0.499999), "text-phosphor-dim font-data");
    }

    // ========== render_tensor_value additional tests ==========

    #[test]
    fn test_render_tensor_value_string_with_special_chars() {
        let value = TensorValue::Scalar(ScalarValue::String(
            "<script>alert('xss')</script>".to_string(),
        ));
        let html = render_tensor_value(&value).into_string();
        // Should escape HTML
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;") || html.contains("script"));
    }

    #[test]
    fn test_render_tensor_value_vector_empty() {
        let value = TensorValue::Vector(vec![]);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("0d") || html.contains("empty") || html.contains("[]"));
    }

    #[test]
    fn test_render_tensor_value_sparse_empty() {
        let sparse = SparseVector::new(0);
        let value = TensorValue::Sparse(sparse);
        let html = render_tensor_value(&value).into_string();
        assert!(html.contains("sparse") || html.contains("0d"));
    }

    // ========== render_payload_preview additional tests ==========

    #[test]
    fn test_render_payload_preview_with_long_keys() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "this_is_a_very_long_key_name_that_exceeds_normal_limits".to_string(),
            TensorValue::Scalar(ScalarValue::Int(1)),
        );
        let html = render_payload_preview(&metadata).into_string();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_render_payload_preview_with_nested_vectors() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "embedding".to_string(),
            TensorValue::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        metadata.insert(
            "sparse_embedding".to_string(),
            TensorValue::Sparse({
                let mut s = SparseVector::new(100);
                s.set(50, 1.0);
                s
            }),
        );
        let html = render_payload_preview(&metadata).into_string();
        assert!(!html.is_empty());
    }

    // ========== render_payload_table additional tests ==========

    #[test]
    fn test_render_payload_table_all_types() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "string".to_string(),
            TensorValue::Scalar(ScalarValue::String("hello".to_string())),
        );
        metadata.insert("int".to_string(), TensorValue::Scalar(ScalarValue::Int(42)));
        metadata.insert(
            "float".to_string(),
            TensorValue::Scalar(ScalarValue::Float(3.14)),
        );
        metadata.insert(
            "bool".to_string(),
            TensorValue::Scalar(ScalarValue::Bool(true)),
        );
        metadata.insert("null".to_string(), TensorValue::Scalar(ScalarValue::Null));
        metadata.insert(
            "bytes".to_string(),
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
        );
        metadata.insert("vector".to_string(), TensorValue::Vector(vec![1.0, 2.0]));
        metadata.insert(
            "pointer".to_string(),
            TensorValue::Pointer("ref".to_string()),
        );
        metadata.insert(
            "pointers".to_string(),
            TensorValue::Pointers(vec!["a".to_string(), "b".to_string()]),
        );

        let html = render_payload_table(&metadata).into_string();
        assert!(html.contains("table"));
        assert!(html.contains("hello"));
        assert!(html.contains("42"));
    }

    // ========== SearchParams additional tests ==========

    #[test]
    fn test_search_params_with_brackets() {
        let params = SearchParams {
            vector: "[1.0, 2.0, 3.0]".to_string(),
            k: 10,
        };
        // The vector parsing should handle brackets
        let trimmed = params.vector.trim_start_matches('[').trim_end_matches(']');
        assert_eq!(trimmed, "1.0, 2.0, 3.0");
    }

    #[test]
    fn test_search_params_with_whitespace() {
        let params = SearchParams {
            vector: "  [ 1.0 , 2.0 , 3.0 ]  ".to_string(),
            k: 5,
        };
        let trimmed = params
            .vector
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']');
        assert_eq!(trimmed, " 1.0 , 2.0 , 3.0 ");
    }

    #[test]
    fn test_search_params_large_k() {
        let params = SearchParams {
            vector: "1.0, 2.0".to_string(),
            k: 100,
        };
        assert_eq!(params.k, 100);
    }
}
