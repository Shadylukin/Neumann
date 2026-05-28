// SPDX-License-Identifier: MIT OR Apache-2.0
//! `BLOB` and `BLOBS` statement execution, including async variants.

use neumann_parser::{BlobOp, BlobOptions, BlobStmt, BlobsOp, BlobsStmt};
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::result::{ArtifactInfoResult, BlobStatsResult, SimilarResult};
use crate::{protection, QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Execute a `BLOB ...` statement (sync, via blob runtime).
#[allow(clippy::too_many_lines, reason = "covers every BLOB sub-op")]
pub fn exec_blob(router: &QueryRouter, stmt: &BlobStmt) -> Result<QueryResult> {
    let _identity = router.require_identity()?;

    // Handle BLOB INIT specially - doesn't require blob to be initialized
    if matches!(stmt.operation, BlobOp::Init) {
        if router.blob.is_some() {
            return Ok(QueryResult::Value(
                "Blob store already initialized".to_string(),
            ));
        }
        return Err(RouterError::BlobError(
            "Use router.init_blob() to initialize blob storage".to_string(),
        ));
    }

    let blob = router
        .blob
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;
    let runtime = router
        .blob_runtime
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob runtime not initialized".to_string()))?;

    match &stmt.operation {
        BlobOp::Init => unreachable!(),
        BlobOp::Put {
            filename,
            data,
            from_path,
            options,
        } => {
            let filename_str = expr::eval_string_expr(filename)?;
            let put_options = blob_options_to_put_options(options)?;

            let blob_data = if let Some(data_expr) = data {
                expr::expr_to_bytes(data_expr)?
            } else if let Some(path_expr) = from_path {
                let path = expr::eval_string_expr(path_expr)?;
                std::fs::read(&path)
                    .map_err(|e| RouterError::BlobError(format!("Failed to read file: {e}")))?
            } else {
                return Err(RouterError::MissingArgument(
                    "PUT requires either DATA or FROM path".to_string(),
                ));
            };

            let artifact_id = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.put(&filename_str, &blob_data, put_options).await
            })?;
            Ok(QueryResult::Value(artifact_id))
        },
        BlobOp::Get {
            artifact_id,
            to_path,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let data = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.get(&id).await
            })?;

            if let Some(path_expr) = to_path {
                let path = expr::eval_string_expr(path_expr)?;
                std::fs::write(&path, &data)
                    .map_err(|e| RouterError::BlobError(format!("Failed to write file: {e}")))?;
                Ok(QueryResult::Value(format!(
                    "Written {} bytes to {path}",
                    data.len()
                )))
            } else {
                Ok(QueryResult::Blob(data))
            }
        },
        BlobOp::Delete { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;

            let size = runtime
                .block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.metadata(&id).await
                })
                .map_or(0, |m| m.size);

            let op = DestructiveOp::BlobDelete {
                artifact_id: id.clone(),
                size,
            };

            match protection::protect_destructive_op(
                router,
                &format!("BLOB DELETE '{id}'"),
                op,
                vec![format!("artifact: {}, size: {} bytes", id, size)],
            ) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }

            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.delete(&id).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Info { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let meta = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.metadata(&id).await
            })?;

            Ok(QueryResult::ArtifactInfo(ArtifactInfoResult {
                id: meta.id,
                filename: meta.filename,
                content_type: meta.content_type,
                size: meta.size,
                checksum: meta.checksum,
                chunk_count: meta.chunk_count,
                created: meta.created,
                modified: meta.modified,
                created_by: meta.created_by,
                tags: meta.tags,
                linked_to: meta.linked_to,
                custom: meta.custom,
            }))
        },
        BlobOp::Link {
            artifact_id,
            entity,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let entity_str = expr::eval_string_expr(entity)?;
            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.link(&id, &entity_str).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Unlink {
            artifact_id,
            entity,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let entity_str = expr::eval_string_expr(entity)?;
            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.unlink(&id, &entity_str).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Links { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let links = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.links(&id).await
            })?;
            Ok(QueryResult::ArtifactList(links))
        },
        BlobOp::Tag { artifact_id, tag } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let tag_str = expr::eval_string_expr(tag)?;
            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.tag(&id, &tag_str).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Untag { artifact_id, tag } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let tag_str = expr::eval_string_expr(tag)?;
            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.untag(&id, &tag_str).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Verify { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let valid = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.verify(&id)
            })?;
            Ok(QueryResult::Value(if valid {
                "OK".to_string()
            } else {
                "INVALID".to_string()
            }))
        },
        BlobOp::Gc { full } => {
            let stats = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                if *full {
                    blob_guard.full_gc().await
                } else {
                    blob_guard.gc().await
                }
            })?;
            Ok(QueryResult::Value(format!(
                "Deleted {} chunks, freed {} bytes",
                stats.deleted, stats.freed_bytes
            )))
        },
        BlobOp::Repair => {
            let stats = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.repair()
            })?;
            Ok(QueryResult::Value(format!(
                "Fixed {} refs, deleted {} orphans",
                stats.refs_fixed, stats.orphans_deleted
            )))
        },
        BlobOp::Stats => {
            let stats = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.stats().await
            })?;
            Ok(QueryResult::BlobStats(BlobStatsResult {
                artifact_count: stats.artifact_count,
                chunk_count: stats.chunk_count,
                total_bytes: stats.total_bytes,
                unique_bytes: stats.unique_bytes,
                dedup_ratio: stats.dedup_ratio,
                orphaned_chunks: stats.orphaned_chunks,
            }))
        },
        BlobOp::MetaSet {
            artifact_id,
            key,
            value,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let key_str = expr::eval_string_expr(key)?;
            let value_str = expr::eval_string_expr(value)?;
            runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.set_meta(&id, &key_str, &value_str).await
            })?;
            Ok(QueryResult::Empty)
        },
        BlobOp::MetaGet { artifact_id, key } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let key_str = expr::eval_string_expr(key)?;
            let value = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.get_meta(&id, &key_str).await
            })?;
            Ok(QueryResult::Value(
                value.unwrap_or_else(|| "(not found)".to_string()),
            ))
        },
    }
}

/// Execute a `BLOBS ...` listing statement.
pub fn exec_blobs(router: &QueryRouter, stmt: &BlobsStmt) -> Result<QueryResult> {
    let _identity = router.require_identity()?;

    let blob = router
        .blob
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;
    let runtime = router
        .blob_runtime
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob runtime not initialized".to_string()))?;

    match &stmt.operation {
        BlobsOp::List { pattern } => {
            let prefix = pattern.as_ref().map(expr::eval_string_expr).transpose()?;
            let ids = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.list(prefix.as_deref()).await
            })?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::For { entity } => {
            let entity_str = expr::eval_string_expr(entity)?;
            let ids = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.artifacts_for(&entity_str).await
            })?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::ByTag { tag } => {
            let tag_str = expr::eval_string_expr(tag)?;
            let ids = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.by_tag(&tag_str).await
            })?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::ByType { content_type } => {
            let ct = expr::eval_string_expr(content_type)?;
            let ids = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.by_content_type(&ct).await
            })?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::Similar { artifact_id, limit } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let k = limit
                .as_ref()
                .map(expr::expr_to_usize)
                .transpose()?
                .unwrap_or(10);
            let similar = runtime.block_on(async {
                let blob_guard = blob.lock().await;
                blob_guard.similar(&id, k).await
            })?;
            Ok(QueryResult::Similar(
                similar
                    .into_iter()
                    .map(|s| SimilarResult {
                        key: s.id,
                        score: s.similarity,
                    })
                    .collect(),
            ))
        },
    }
}

/// Convert parsed [`BlobOptions`] into [`tensor_blob::PutOptions`].
fn blob_options_to_put_options(options: &BlobOptions) -> Result<tensor_blob::PutOptions> {
    let mut put_options = tensor_blob::PutOptions::new();

    if let Some(ct) = &options.content_type {
        put_options = put_options.with_content_type(&expr::eval_string_expr(ct)?);
    }

    if let Some(cb) = &options.created_by {
        put_options = put_options.with_created_by(&expr::eval_string_expr(cb)?);
    }

    for link_expr in &options.link {
        let link = expr::eval_string_expr(link_expr)?;
        put_options = put_options.with_link(&link);
    }

    for tag_expr in &options.tag {
        let tag = expr::eval_string_expr(tag_expr)?;
        put_options = put_options.with_tag(&tag);
    }

    Ok(put_options)
}

/// Execute a `BLOB ...` statement asynchronously (no runtime blocking).
#[allow(clippy::too_many_lines, reason = "covers every BLOB sub-op")]
#[allow(
    clippy::significant_drop_tightening,
    reason = "Blob guard acquired per match arm"
)]
pub async fn exec_blob_async(router: &QueryRouter, stmt: &BlobStmt) -> Result<QueryResult> {
    if matches!(stmt.operation, BlobOp::Init) {
        if router.blob.is_some() {
            return Ok(QueryResult::Value(
                "Blob store already initialized".to_string(),
            ));
        }
        return Err(RouterError::BlobError(
            "Use router.init_blob() to initialize blob storage".to_string(),
        ));
    }

    let blob = router
        .blob
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;

    match &stmt.operation {
        BlobOp::Init => unreachable!(),
        BlobOp::Put {
            filename,
            data,
            from_path,
            options,
        } => {
            let filename_str = expr::eval_string_expr(filename)?;
            let put_options = blob_options_to_put_options(options)?;

            let blob_data = if let Some(data_expr) = data {
                expr::expr_to_bytes(data_expr)?
            } else if let Some(path_expr) = from_path {
                let path = expr::eval_string_expr(path_expr)?;
                tokio::fs::read(&path)
                    .await
                    .map_err(|e| RouterError::BlobError(format!("Failed to read file: {e}")))?
            } else {
                return Err(RouterError::MissingArgument(
                    "PUT requires either DATA or FROM path".to_string(),
                ));
            };

            let blob_guard = blob.lock().await;
            let artifact_id = blob_guard
                .put(&filename_str, &blob_data, put_options)
                .await?;
            Ok(QueryResult::Value(artifact_id))
        },
        BlobOp::Get {
            artifact_id,
            to_path,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let blob_guard = blob.lock().await;
            let data = blob_guard.get(&id).await?;

            if let Some(path_expr) = to_path {
                let path = expr::eval_string_expr(path_expr)?;
                tokio::fs::write(&path, &data)
                    .await
                    .map_err(|e| RouterError::BlobError(format!("Failed to write file: {e}")))?;
                Ok(QueryResult::Value(format!(
                    "Written {} bytes to {path}",
                    data.len()
                )))
            } else {
                Ok(QueryResult::Blob(data))
            }
        },
        BlobOp::Delete { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let blob_guard = blob.lock().await;
            blob_guard.delete(&id).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Info { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let blob_guard = blob.lock().await;
            let meta = blob_guard.metadata(&id).await?;

            Ok(QueryResult::ArtifactInfo(ArtifactInfoResult {
                id: meta.id,
                filename: meta.filename,
                content_type: meta.content_type,
                size: meta.size,
                checksum: meta.checksum,
                chunk_count: meta.chunk_count,
                created: meta.created,
                modified: meta.modified,
                created_by: meta.created_by,
                tags: meta.tags,
                linked_to: meta.linked_to,
                custom: meta.custom,
            }))
        },
        BlobOp::Link {
            artifact_id,
            entity,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let entity_str = expr::eval_string_expr(entity)?;
            let blob_guard = blob.lock().await;
            blob_guard.link(&id, &entity_str).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Unlink {
            artifact_id,
            entity,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let entity_str = expr::eval_string_expr(entity)?;
            let blob_guard = blob.lock().await;
            blob_guard.unlink(&id, &entity_str).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Links { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let blob_guard = blob.lock().await;
            let links = blob_guard.links(&id).await?;
            Ok(QueryResult::ArtifactList(links))
        },
        BlobOp::Tag { artifact_id, tag } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let tag_str = expr::eval_string_expr(tag)?;
            let blob_guard = blob.lock().await;
            blob_guard.tag(&id, &tag_str).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Untag { artifact_id, tag } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let tag_str = expr::eval_string_expr(tag)?;
            let blob_guard = blob.lock().await;
            blob_guard.untag(&id, &tag_str).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::Verify { artifact_id } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let blob_guard = blob.lock().await;
            let valid = blob_guard.verify(&id)?;
            Ok(QueryResult::Value(if valid {
                "OK".to_string()
            } else {
                "INVALID".to_string()
            }))
        },
        BlobOp::Gc { full } => {
            let blob_guard = blob.lock().await;
            let stats = if *full {
                blob_guard.full_gc().await?
            } else {
                blob_guard.gc().await?
            };
            Ok(QueryResult::Value(format!(
                "Deleted {} chunks, freed {} bytes",
                stats.deleted, stats.freed_bytes
            )))
        },
        BlobOp::Repair => {
            let blob_guard = blob.lock().await;
            let stats = blob_guard.repair()?;
            Ok(QueryResult::Value(format!(
                "Fixed {} refs, deleted {} orphans",
                stats.refs_fixed, stats.orphans_deleted
            )))
        },
        BlobOp::Stats => {
            let blob_guard = blob.lock().await;
            let stats = blob_guard.stats().await?;
            Ok(QueryResult::BlobStats(BlobStatsResult {
                artifact_count: stats.artifact_count,
                chunk_count: stats.chunk_count,
                total_bytes: stats.total_bytes,
                unique_bytes: stats.unique_bytes,
                dedup_ratio: stats.dedup_ratio,
                orphaned_chunks: stats.orphaned_chunks,
            }))
        },
        BlobOp::MetaSet {
            artifact_id,
            key,
            value,
        } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let key_str = expr::eval_string_expr(key)?;
            let value_str = expr::eval_string_expr(value)?;
            let blob_guard = blob.lock().await;
            blob_guard.set_meta(&id, &key_str, &value_str).await?;
            Ok(QueryResult::Empty)
        },
        BlobOp::MetaGet { artifact_id, key } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let key_str = expr::eval_string_expr(key)?;
            let blob_guard = blob.lock().await;
            let value = blob_guard.get_meta(&id, &key_str).await?;
            Ok(QueryResult::Value(
                value.unwrap_or_else(|| "(not found)".to_string()),
            ))
        },
    }
}

/// Execute a `BLOBS ...` listing statement asynchronously.
#[allow(
    clippy::significant_drop_tightening,
    reason = "blob guard held for listing operations"
)]
pub async fn exec_blobs_async(router: &QueryRouter, stmt: &BlobsStmt) -> Result<QueryResult> {
    let blob = router
        .blob
        .as_ref()
        .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;

    let blob_guard = blob.lock().await;
    match &stmt.operation {
        BlobsOp::List { pattern } => {
            let prefix = pattern.as_ref().map(expr::eval_string_expr).transpose()?;
            let ids = blob_guard.list(prefix.as_deref()).await?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::For { entity } => {
            let entity_str = expr::eval_string_expr(entity)?;
            let ids = blob_guard.artifacts_for(&entity_str).await?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::ByTag { tag } => {
            let tag_str = expr::eval_string_expr(tag)?;
            let ids = blob_guard.by_tag(&tag_str).await?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::ByType { content_type } => {
            let ct = expr::eval_string_expr(content_type)?;
            let ids = blob_guard.by_content_type(&ct).await?;
            Ok(QueryResult::ArtifactList(ids))
        },
        BlobsOp::Similar { artifact_id, limit } => {
            let id = expr::eval_string_expr(artifact_id)?;
            let k = limit
                .as_ref()
                .map(expr::expr_to_usize)
                .transpose()?
                .unwrap_or(10);
            let similar = blob_guard.similar(&id, k).await?;
            Ok(QueryResult::Similar(
                similar
                    .into_iter()
                    .map(|s| SimilarResult {
                        key: s.id,
                        score: s.similarity,
                    })
                    .collect(),
            ))
        },
    }
}
