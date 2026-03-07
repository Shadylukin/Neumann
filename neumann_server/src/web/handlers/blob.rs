// SPDX-License-Identifier: MIT OR Apache-2.0
//! Blob storage browser handlers for the admin UI.
//!
//! Provides three views for browsing the content-addressable blob store:
//! overview stats, artifacts list, and artifact detail.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use maud::{html, Markup};
use tensor_blob::ArtifactMetadata;

use crate::web::templates::{
    format_bytes, layout, m_badge, m_breadcrumb, m_card, m_empty, m_header, m_pagination, m_stat,
    m_table_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Query parameters for the artifacts list.
#[derive(Debug, serde::Deserialize)]
pub struct ArtifactsParams {
    /// Page number (1-indexed).
    page: Option<usize>,
    /// Prefix filter for artifact IDs.
    prefix: Option<String>,
}

/// Blob overview page with hero stats.
pub async fn overview(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(blob) = ctx.blob.as_ref() else {
        return layout(
            "Blob Storage",
            NavItem::Blob,
            m_empty(
                "Blob Store Not Configured",
                "Enable blob storage to browse artifacts.",
            ),
        );
    };

    let stats = blob.lock().await.stats().await;

    let content = match stats {
        Ok(stats) => {
            html! {
                (m_header("BLOB STORAGE", Some("Content-addressable artifact store")))

                div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8 stagger-container" {
                    (m_stat("ARTIFACTS", &stats.artifact_count.to_string(), "stored", "blob"))
                    (m_stat("CHUNKS", &stats.chunk_count.to_string(), "content blocks", "blob"))
                    (m_stat("TOTAL SIZE", &format_bytes(stats.total_bytes), "logical", "blob"))
                    (m_stat("UNIQUE SIZE", &format_bytes(stats.unique_bytes), "after dedup", "blob"))
                    (m_stat("DEDUP RATIO", &format!("{:.1}%", stats.dedup_ratio * 100.0), "space saved", "blob"))
                    (m_stat("ORPHANED", &stats.orphaned_chunks.to_string(), "unreferenced chunks", "blob"))
                }

                div class="flex gap-4" {
                    a href="/blob/artifacts" class="m-btn inline-block" { "BROWSE ARTIFACTS" }
                }
            }
        },
        Err(e) => {
            html! {
                (m_header("BLOB STORAGE", Some("Content-addressable artifact store")))
                (m_empty("Error Loading Stats", &e.to_string()))
            }
        },
    };

    layout("Blob Storage", NavItem::Blob, content)
}

/// Paginated artifacts list.
pub async fn artifacts_list(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<ArtifactsParams>,
) -> Markup {
    let Some(blob) = ctx.blob.as_ref() else {
        return layout(
            "Artifacts",
            NavItem::Blob,
            m_empty(
                "Blob Store Not Configured",
                "Enable blob storage to browse artifacts.",
            ),
        );
    };

    let page = params.page.unwrap_or(1).max(1);
    let per_page = 25;
    let prefix = params.prefix.as_deref();

    let all_ids = blob.lock().await.list(prefix).await;

    let content = match all_ids {
        Ok(ids) => {
            let total = ids.len();
            let total_pages = total.div_ceil(per_page);
            let start = (page - 1) * per_page;
            let page_ids: Vec<_> = ids.into_iter().skip(start).take(per_page).collect();

            // Gather metadata for each artifact
            let mut artifacts = Vec::new();
            {
                let store = blob.lock().await;
                for id in &page_ids {
                    if let Ok(meta) = store.metadata(id).await {
                        artifacts.push(meta);
                    }
                }
            }

            let base_url = params.prefix.as_ref().map_or_else(
                || "/blob/artifacts".to_string(),
                |p| format!("/blob/artifacts?prefix={p}"),
            );

            html! {
                (m_breadcrumb(&[("/blob", "BLOB"), ("", "ARTIFACTS")]))
                (m_header("ARTIFACTS", Some(&format!("{total} artifacts"))))

                // Prefix filter
                form class="mb-4 flex gap-2" method="get" action="/blob/artifacts" {
                    input
                        type="text"
                        name="prefix"
                        placeholder="Filter by prefix..."
                        value=(prefix.unwrap_or(""))
                        class="m-input flex-1";
                    button type="submit" class="m-btn" { "FILTER" }
                }

                @if artifacts.is_empty() {
                    (m_empty("No Artifacts", "No artifacts found matching your criteria."))
                } @else {
                    div class="m-card" {
                        div class="m-card-content overflow-x-auto" {
                            table class="m-table w-full" {
                                (m_table_header(&["FILENAME", "TYPE", "SIZE", "CHUNKS", "CREATED"]))
                                tbody {
                                    @for meta in &artifacts {
                                        tr {
                                            td {
                                                a href=(format!("/blob/artifacts/{}", meta.id))
                                                    class="text-white hover:underline" {
                                                    (meta.filename)
                                                }
                                            }
                                            td class="text-neutral-400" { (meta.content_type) }
                                            td class="text-neutral-400 font-mono" { (format_bytes(meta.size)) }
                                            td class="text-neutral-400 font-mono" { (meta.chunk_count) }
                                            td class="text-neutral-400 font-mono" {
                                                (format_timestamp(meta.created))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    (m_pagination(page, total_pages, &base_url))
                }
            }
        },
        Err(e) => {
            html! {
                (m_breadcrumb(&[("/blob", "BLOB"), ("", "ARTIFACTS")]))
                (m_header("ARTIFACTS", None))
                (m_empty("Error Loading Artifacts", &e.to_string()))
            }
        },
    };

    layout("Artifacts", NavItem::Blob, content)
}

/// Artifact detail view.
pub async fn artifact_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path(artifact_id): Path<String>,
) -> Markup {
    let Some(blob) = ctx.blob.as_ref() else {
        return layout(
            "Artifact",
            NavItem::Blob,
            m_empty(
                "Blob Store Not Configured",
                "Enable blob storage to browse artifacts.",
            ),
        );
    };

    let store = blob.lock().await;
    let meta = store.metadata(&artifact_id).await;
    let links = store.links(&artifact_id).await.unwrap_or_default();
    drop(store);

    let content = match meta {
        Ok(meta) => {
            html! {
                (m_breadcrumb(&[
                    ("/blob", "BLOB"),
                    ("/blob/artifacts", "ARTIFACTS"),
                    ("", &meta.filename),
                ]))

                (m_header(&meta.filename, Some(&format!("Artifact {}", meta.id))))

                (render_artifact_metadata(&meta, &links))
            }
        },
        Err(e) => {
            html! {
                (m_breadcrumb(&[
                    ("/blob", "BLOB"),
                    ("/blob/artifacts", "ARTIFACTS"),
                    ("", &artifact_id),
                ]))
                (m_empty("Artifact Not Found", &e.to_string()))
            }
        },
    };

    layout("Artifact Detail", NavItem::Blob, content)
}

/// Render the identity and storage info cards for an artifact.
fn render_artifact_identity(meta: &ArtifactMetadata) -> Markup {
    html! {
        (m_card("IDENTITY", html! {
            dl class="space-y-3" {
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "ID" }
                    dd class="text-white font-mono text-sm" { (meta.id) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Filename" }
                    dd class="text-white" { (meta.filename) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Content Type" }
                    dd class="text-white" { (meta.content_type) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Size" }
                    dd class="text-white font-mono" { (format_bytes(meta.size)) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Checksum" }
                    dd class="text-white font-mono text-xs break-all" { (meta.checksum) }
                }
            }
        }))

        (m_card("STORAGE", html! {
            dl class="space-y-3" {
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Chunks" }
                    dd class="text-white font-mono" { (meta.chunk_count) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Chunk Size" }
                    dd class="text-white font-mono" { (format_bytes(meta.chunk_size)) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Created By" }
                    dd class="text-white" { (meta.created_by) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Created" }
                    dd class="text-white font-mono" { (format_timestamp(meta.created)) }
                }
                div class="flex justify-between" {
                    dt class="text-neutral-400" { "Modified" }
                    dd class="text-white font-mono" { (format_timestamp(meta.modified)) }
                }
            }
        }))
    }
}

/// Render the metadata card grid for an artifact detail view.
fn render_artifact_metadata(meta: &ArtifactMetadata, links: &[String]) -> Markup {
    html! {
        div class="grid grid-cols-1 lg:grid-cols-2 gap-6" {
            (render_artifact_identity(meta))

            // Tags card
            (m_card("TAGS", html! {
                @if meta.tags.is_empty() {
                    span class="text-neutral-500 text-sm" { "No tags" }
                } @else {
                    div class="flex flex-wrap gap-2" {
                        @for tag in &meta.tags {
                            (m_badge(tag))
                        }
                    }
                }
            }))

            // Custom metadata card
            (m_card("CUSTOM METADATA", html! {
                @if meta.custom.is_empty() {
                    span class="text-neutral-500 text-sm" { "No custom metadata" }
                } @else {
                    dl class="space-y-2" {
                        @for (key, value) in &meta.custom {
                            div class="flex justify-between" {
                                dt class="text-neutral-400 font-mono text-sm" { (key) }
                                dd class="text-white font-mono text-sm" { (value) }
                            }
                        }
                    }
                }
            }))

            // Embedding card
            (m_card("EMBEDDING", html! {
                dl class="space-y-3" {
                    div class="flex justify-between" {
                        dt class="text-neutral-400" { "Has Embedding" }
                        dd class="text-white" {
                            @if meta.has_embedding { "Yes" } @else { "No" }
                        }
                    }
                    @if let Some(ref model) = meta.embedding_model {
                        div class="flex justify-between" {
                            dt class="text-neutral-400" { "Model" }
                            dd class="text-white" { (model) }
                        }
                    }
                }
            }))

            // Links card
            (m_card("LINKED ENTITIES", html! {
                @if links.is_empty() {
                    span class="text-neutral-500 text-sm" { "No linked entities" }
                } @else {
                    ul class="space-y-1" {
                        @for link in links {
                            li class="text-white font-mono text-sm" { (link) }
                        }
                    }
                }
            }))
        }
    }
}

/// Format a Unix timestamp as a human-readable date/time string.
fn format_timestamp(epoch_secs: u64) -> String {
    if epoch_secs == 0 {
        return "--".to_string();
    }
    // Simple date formatting: show as Unix timestamp with annotation
    let days = epoch_secs / 86400;
    let hours = (epoch_secs % 86400) / 3600;
    let mins = (epoch_secs % 3600) / 60;
    if days > 365 {
        #[allow(clippy::cast_precision_loss)]
        let years = days as f64 / 365.25;
        format!("{years:.1}y ago")
    } else if days > 0 {
        format!("{days}d {hours}h ago")
    } else if hours > 0 {
        format!("{hours}h {mins}m ago")
    } else {
        format!("{mins}m ago")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use tensor_blob::{BlobConfig, BlobStore};
    use tensor_store::TensorStore;
    use tokio::sync::Mutex;
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        ))
    }

    async fn create_test_context_with_blob() -> Arc<AdminContext> {
        let store = TensorStore::new();
        let blob = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("blob store");
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.blob = Some(Arc::new(Mutex::new(blob)));
        Arc::new(ctx)
    }

    #[tokio::test]
    async fn test_overview_no_blob() {
        let ctx = create_test_context();
        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_overview_empty_blob() {
        let ctx = create_test_context_with_blob().await;
        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("BLOB STORAGE"));
        assert!(html.contains("ARTIFACTS"));
        assert!(html.contains("CHUNKS"));
    }

    #[tokio::test]
    async fn test_artifacts_list_no_blob() {
        let ctx = create_test_context();
        let params = ArtifactsParams {
            page: None,
            prefix: None,
        };
        let result = artifacts_list(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_artifacts_list_empty() {
        let ctx = create_test_context_with_blob().await;
        let params = ArtifactsParams {
            page: None,
            prefix: None,
        };
        let result = artifacts_list(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("No Artifacts"));
    }

    #[tokio::test]
    async fn test_artifacts_list_with_prefix() {
        let ctx = create_test_context_with_blob().await;
        let params = ArtifactsParams {
            page: Some(1),
            prefix: Some("test/".to_string()),
        };
        let result = artifacts_list(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("ARTIFACTS"));
    }

    #[tokio::test]
    async fn test_artifact_detail_no_blob() {
        let ctx = create_test_context();
        let result = artifact_detail(State(ctx), Path("abc123".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_artifact_detail_not_found() {
        let ctx = create_test_context_with_blob().await;
        let result = artifact_detail(State(ctx), Path("nonexistent".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("Not Found") || html.contains("not found") || html.contains("error"));
    }

    #[tokio::test]
    async fn test_artifact_detail_with_data() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("blob store");

        let opts = tensor_blob::PutOptions::new()
            .with_content_type("text/plain")
            .with_created_by("tester");
        let artifact_id = blob
            .put("test.txt", b"hello world test data", opts)
            .await
            .expect("put");

        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.blob = Some(Arc::new(Mutex::new(blob)));
        let ctx = Arc::new(ctx);

        let result = artifact_detail(State(ctx), Path(artifact_id)).await;
        let html = result.into_string();
        assert!(html.contains("test.txt"));
        assert!(html.contains("text/plain"));
        assert!(html.contains("IDENTITY"));
        assert!(html.contains("STORAGE"));
    }

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "--");
    }

    #[test]
    fn test_format_timestamp_minutes() {
        assert_eq!(format_timestamp(300), "5m ago");
    }

    #[test]
    fn test_format_timestamp_hours() {
        assert_eq!(format_timestamp(7200), "2h 0m ago");
    }

    #[test]
    fn test_format_timestamp_days() {
        assert_eq!(format_timestamp(172_800), "2d 0h ago");
    }

    #[test]
    fn test_format_timestamp_years() {
        let result = format_timestamp(400 * 86400);
        assert!(result.contains("y ago"));
    }

    #[tokio::test]
    async fn test_overview_with_artifact() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("blob store");

        let opts = tensor_blob::PutOptions::new()
            .with_content_type("application/octet-stream")
            .with_created_by("tester");
        blob.put("demo.bin", b"test data", opts).await.expect("put");

        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.blob = Some(Arc::new(Mutex::new(blob)));
        let ctx = Arc::new(ctx);

        let result = overview(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("BLOB STORAGE"));
        assert!(html.contains("BROWSE ARTIFACTS"));
    }

    #[tokio::test]
    async fn test_artifacts_list_with_data() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("blob store");

        let opts = tensor_blob::PutOptions::new()
            .with_content_type("text/plain")
            .with_created_by("tester");
        blob.put("file1.txt", b"content", opts).await.expect("put");

        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.blob = Some(Arc::new(Mutex::new(blob)));
        let ctx = Arc::new(ctx);

        let params = ArtifactsParams {
            page: Some(1),
            prefix: None,
        };
        let result = artifacts_list(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("file1.txt"));
        assert!(html.contains("text/plain"));
    }
}
