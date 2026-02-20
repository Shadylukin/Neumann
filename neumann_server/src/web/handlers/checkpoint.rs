// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Checkpoint manager handlers for the admin UI.
//!
//! Provides read-only views for browsing checkpoints and configuration.
//! No rollback or delete operations are exposed for safety.

use std::sync::Arc;

use axum::extract::{Path, State};
use maud::{html, Markup};

use crate::web::templates::{
    format_bytes, layout, m_breadcrumb, m_card, m_empty, m_header, m_stat, m_table_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Checkpoint list view.
pub async fn list_view(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(checkpoint) = ctx.checkpoint.as_ref() else {
        return layout(
            "Checkpoints",
            NavItem::Checkpoint,
            m_empty(
                "Checkpoint Manager Not Configured",
                "Enable checkpoint management to view backups.",
            ),
        );
    };

    let checkpoints = checkpoint.list(Some(50)).await;

    let content = match checkpoints {
        Ok(list) => {
            let count = list.len();
            html! {
                (m_header("CHECKPOINTS", Some(&format!("{count} checkpoints"))))

                @if list.is_empty() {
                    (m_empty("No Checkpoints", "No checkpoints have been created yet."))
                } @else {
                    div class="m-card" {
                        div class="m-card-content overflow-x-auto" {
                            table class="m-table w-full" {
                                (m_table_header(&["NAME", "SIZE", "TRIGGER", "CREATED"]))
                                tbody {
                                    @for cp in &list {
                                        tr {
                                            td {
                                                a href=(format!("/checkpoint/{}", cp.id))
                                                    class="text-white hover:underline" {
                                                    (cp.name)
                                                }
                                            }
                                            td class="text-neutral-400 font-mono" {
                                                (format_bytes(cp.size))
                                            }
                                            td class="text-neutral-400" {
                                                @match &cp.trigger {
                                                    Some(t) => (t),
                                                    None => "Manual",
                                                }
                                            }
                                            td class="text-neutral-400 font-mono" {
                                                (format_checkpoint_time(cp.created_at))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                div class="mt-4 flex gap-4" {
                    a href="/checkpoint/config" class="m-btn inline-block" { "VIEW CONFIG" }
                }
            }
        },
        Err(e) => {
            html! {
                (m_header("CHECKPOINTS", None))
                (m_empty("Error Loading Checkpoints", &e.to_string()))
            }
        },
    };

    layout("Checkpoints", NavItem::Checkpoint, content)
}

/// Checkpoint detail view.
pub async fn detail_view(State(ctx): State<Arc<AdminContext>>, Path(id): Path<String>) -> Markup {
    let Some(checkpoint) = ctx.checkpoint.as_ref() else {
        return layout(
            "Checkpoint",
            NavItem::Checkpoint,
            m_empty(
                "Checkpoint Manager Not Configured",
                "Enable checkpoint management to view backups.",
            ),
        );
    };

    // Find the checkpoint in the list
    let checkpoints = checkpoint.list(Some(100)).await;
    let found = checkpoints
        .ok()
        .and_then(|list| list.into_iter().find(|cp| cp.id == id));

    let content = match found {
        Some(cp) => {
            html! {
                (m_breadcrumb(&[
                    ("/checkpoint", "CHECKPOINTS"),
                    ("", &cp.name),
                ]))

                (m_header(&cp.name, Some(&format!("Checkpoint {}", cp.id))))

                div class="grid grid-cols-1 lg:grid-cols-2 gap-6" {
                    (m_card("DETAILS", html! {
                        dl class="space-y-3" {
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "ID" }
                                dd class="text-white font-mono text-sm" { (cp.id) }
                            }
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "Name" }
                                dd class="text-white" { (cp.name) }
                            }
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "Size" }
                                dd class="text-white font-mono" { (format_bytes(cp.size)) }
                            }
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "Trigger" }
                                dd class="text-white" {
                                    @match &cp.trigger {
                                        Some(t) => (t),
                                        None => "Manual",
                                    }
                                }
                            }
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "Created" }
                                dd class="text-white font-mono" {
                                    (format_checkpoint_time(cp.created_at))
                                }
                            }
                            div class="flex justify-between" {
                                dt class="text-neutral-400" { "Artifact ID" }
                                dd class="text-white font-mono text-xs break-all" {
                                    (cp.artifact_id)
                                }
                            }
                        }
                    }))

                    (m_card("RESTORE", html! {
                        div class="text-neutral-400 text-sm" {
                            p class="mb-2" {
                                "To restore this checkpoint, use the CLI:"
                            }
                            code class="block bg-neutral-800 p-3 rounded text-white font-mono text-sm" {
                                "ROLLBACK TO " (cp.name)
                            }
                        }
                    }))
                }
            }
        },
        None => {
            html! {
                (m_breadcrumb(&[
                    ("/checkpoint", "CHECKPOINTS"),
                    ("", &id),
                ]))
                (m_empty("Checkpoint Not Found", &format!("No checkpoint with ID '{id}'")))
            }
        },
    };

    layout("Checkpoint Detail", NavItem::Checkpoint, content)
}

/// Checkpoint configuration view.
pub async fn config_view(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let Some(checkpoint) = ctx.checkpoint.as_ref() else {
        return layout(
            "Checkpoint Config",
            NavItem::Checkpoint,
            m_empty(
                "Checkpoint Manager Not Configured",
                "Enable checkpoint management to view configuration.",
            ),
        );
    };

    let config = checkpoint.config();

    let content = html! {
        (m_breadcrumb(&[
            ("/checkpoint", "CHECKPOINTS"),
            ("", "CONFIG"),
        ]))
        (m_header("CHECKPOINT CONFIGURATION", None))

        div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8 stagger-container" {
            (m_stat(
                "AUTO-CHECKPOINT",
                if config.auto_checkpoint { "ON" } else { "OFF" },
                "automatic backups",
                "checkpoint",
            ))
            (m_stat(
                "CONFIRM",
                if config.interactive_confirm { "ON" } else { "OFF" },
                "interactive confirm",
                "checkpoint",
            ))
            (m_stat(
                "MAX RETAINED",
                &config.max_checkpoints.to_string(),
                "checkpoint limit",
                "checkpoint",
            ))
        }
    };

    layout("Checkpoint Config", NavItem::Checkpoint, content)
}

/// Format a checkpoint timestamp for display.
fn format_checkpoint_time(timestamp: u64) -> String {
    if timestamp == 0 {
        return "--".to_string();
    }
    // Format as relative time from epoch
    let days = timestamp / 86400;
    let hours = (timestamp % 86400) / 3600;
    if days > 0 {
        format!("{days}d {hours}h")
    } else {
        let mins = (timestamp % 3600) / 60;
        format!("{hours}h {mins}m")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use tensor_checkpoint::{CheckpointConfig, CheckpointManager};
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        ))
    }

    async fn create_test_context_with_checkpoint() -> Arc<AdminContext> {
        let store = tensor_store::TensorStore::new();
        let blob = tensor_blob::BlobStore::new(store, tensor_blob::BlobConfig::default())
            .await
            .expect("blob store creation");
        let blob = Arc::new(tokio::sync::Mutex::new(blob));
        let mgr = CheckpointManager::new(blob, CheckpointConfig::default());
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.checkpoint = Some(Arc::new(mgr));
        Arc::new(ctx)
    }

    #[tokio::test]
    async fn test_list_view_no_checkpoint() {
        let ctx = create_test_context();
        let result = list_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_list_view_empty() {
        let ctx = create_test_context_with_checkpoint().await;
        let result = list_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("CHECKPOINTS"));
    }

    #[tokio::test]
    async fn test_detail_view_no_checkpoint() {
        let ctx = create_test_context();
        let result = detail_view(State(ctx), Path("abc".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_detail_view_not_found() {
        let ctx = create_test_context_with_checkpoint().await;
        let result = detail_view(State(ctx), Path("nonexistent".to_string())).await;
        let html = result.into_string();
        assert!(html.contains("Not Found"));
    }

    #[tokio::test]
    async fn test_config_view_no_checkpoint() {
        let ctx = create_test_context();
        let result = config_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Not Configured"));
    }

    #[tokio::test]
    async fn test_config_view_with_checkpoint() {
        let ctx = create_test_context_with_checkpoint().await;
        let result = config_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("CHECKPOINT CONFIGURATION"));
        assert!(html.contains("AUTO-CHECKPOINT"));
        assert!(html.contains("CONFIRM"));
        assert!(html.contains("MAX RETAINED"));
    }

    /// Create a test context with an actual checkpoint stored in the manager.
    async fn create_test_context_with_data() -> (Arc<AdminContext>, String) {
        let store = tensor_store::TensorStore::new();
        let blob = tensor_blob::BlobStore::new(store.clone(), tensor_blob::BlobConfig::default())
            .await
            .expect("blob store");
        let blob = Arc::new(tokio::sync::Mutex::new(blob));
        let mgr = CheckpointManager::new(blob, CheckpointConfig::default());
        let cp_id = mgr
            .create(Some("test_backup"), &store)
            .await
            .expect("create checkpoint");
        let mut ctx = AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        );
        ctx.checkpoint = Some(Arc::new(mgr));
        (Arc::new(ctx), cp_id)
    }

    #[tokio::test]
    async fn test_list_view_with_data() {
        let (ctx, _id) = create_test_context_with_data().await;
        let result = list_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("test_backup"));
        assert!(html.contains("Manual"));
    }

    #[tokio::test]
    async fn test_detail_view_with_data() {
        let (ctx, id) = create_test_context_with_data().await;
        let result = detail_view(State(ctx), Path(id)).await;
        let html = result.into_string();
        assert!(html.contains("test_backup"));
        assert!(html.contains("DETAILS"));
        assert!(html.contains("RESTORE"));
        assert!(html.contains("ROLLBACK TO"));
    }

    #[test]
    fn test_format_checkpoint_time_zero() {
        assert_eq!(format_checkpoint_time(0), "--");
    }

    #[test]
    fn test_format_checkpoint_time_hours() {
        assert_eq!(format_checkpoint_time(7200), "2h 0m");
    }

    #[test]
    fn test_format_checkpoint_time_days() {
        assert_eq!(format_checkpoint_time(172_800), "2d 0h");
    }
}
