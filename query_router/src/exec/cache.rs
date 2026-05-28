// SPDX-License-Identifier: MIT OR Apache-2.0
//! `CACHE` statement execution and the cache-integration helpers used by the
//! top-level dispatcher.

use neumann_parser::{CacheOp, CacheStmt};
use tensor_cache::CacheLayer;
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::{protection, QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Execute a `CACHE ...` statement.
#[allow(clippy::too_many_lines, reason = "covers every CACHE sub-op")]
pub fn exec_cache(router: &QueryRouter, stmt: &CacheStmt) -> Result<QueryResult> {
    let _identity = router.require_identity()?;

    let cache = router
        .cache
        .as_ref()
        .ok_or_else(|| RouterError::CacheError("Cache not initialized".to_string()))?;

    match &stmt.operation {
        CacheOp::Init => {
            // Cache is already initialized if we got here
            Ok(QueryResult::Value("Cache initialized".to_string()))
        },
        CacheOp::Stats => {
            let stats = cache.stats();
            let (tokens_in, tokens_out) = stats.tokens_saved();
            let output = format!(
                "Cache Statistics:\n\
                 Exact: {} hits, {} misses\n\
                 Semantic: {} hits, {} misses\n\
                 Embedding: {} hits, {} misses\n\
                 Tokens saved: {} in, {} out\n\
                 Evictions: {}",
                stats.hits(CacheLayer::Exact),
                stats.misses(CacheLayer::Exact),
                stats.hits(CacheLayer::Semantic),
                stats.misses(CacheLayer::Semantic),
                stats.hits(CacheLayer::Embedding),
                stats.misses(CacheLayer::Embedding),
                tokens_in,
                tokens_out,
                stats.evictions(),
            );
            Ok(QueryResult::Value(output))
        },
        CacheOp::Clear => {
            let entry_count = cache.stats().total_entries();
            let op = DestructiveOp::CacheClear { entry_count };

            match protection::protect_destructive_op(
                router,
                "CACHE CLEAR",
                op,
                vec![format!("{} cached entries", entry_count)],
            ) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }

            cache.clear();
            Ok(QueryResult::Value("Cache cleared".to_string()))
        },
        CacheOp::Evict { count } => {
            let count_val = match count {
                Some(e) => expr::expr_to_usize(e)?,
                None => 100,
            };
            let evicted = cache.evict(count_val);
            Ok(QueryResult::Value(format!("Evicted {evicted} entries")))
        },
        CacheOp::Get { key } => {
            let key_str = expr::expr_to_string(key)?;
            Ok(QueryResult::Value(
                cache
                    .get_simple(&key_str)
                    .unwrap_or_else(|| "(not found)".to_string()),
            ))
        },
        CacheOp::Put { key, value } => {
            let key_str = expr::expr_to_string(key)?;
            let value_str = expr::expr_to_string(value)?;
            cache
                .put_simple(&key_str, &value_str)
                .map_err(|e| RouterError::CacheError(e.to_string()))?;
            Ok(QueryResult::Value("OK".to_string()))
        },
        CacheOp::SemanticGet { query, threshold } => {
            let query_str = expr::expr_to_string(query)?;
            let embedding = router.vector.get_embedding(&query_str).ok();
            let _threshold = threshold.as_ref().map(expr::expr_to_f32).transpose()?;

            match cache.get(&query_str, embedding.as_deref()) {
                Some(hit) => {
                    let similarity_str = hit
                        .similarity
                        .map(|s| format!(", similarity: {s:.4}"))
                        .unwrap_or_default();
                    Ok(QueryResult::Value(format!(
                        "response: {}, layer: {:?}{}",
                        hit.response, hit.layer, similarity_str
                    )))
                },
                None => Ok(QueryResult::Value("(not found)".to_string())),
            }
        },
        CacheOp::SemanticPut {
            query,
            response,
            embedding,
        } => {
            let query_str = expr::expr_to_string(query)?;
            let response_str = expr::expr_to_string(response)?;
            let emb: Vec<f32> = embedding
                .iter()
                .map(expr::expr_to_f32)
                .collect::<Result<_>>()?;

            cache
                .put(&query_str, &emb, &response_str, "manual", None)
                .map_err(|e| RouterError::CacheError(e.to_string()))?;
            Ok(QueryResult::Value("OK".to_string()))
        },
    }
}

fn cache_key_for_query(command: &str) -> String {
    format!("query:{}", command.trim().to_lowercase())
}

/// Look up a cached query result by command string.
pub fn try_cache_get(router: &QueryRouter, command: &str) -> Option<QueryResult> {
    let cache = router.cache.as_ref()?;
    let key = cache_key_for_query(command);
    let json = cache.get_simple(&key)?;
    serde_json::from_str(&json).ok()
}

/// Best-effort store of a query result keyed by its command string.
pub fn try_cache_put(router: &QueryRouter, command: &str, result: &QueryResult) {
    if let Some(cache) = router.cache.as_ref() {
        let key = cache_key_for_query(command);
        if let Ok(json) = serde_json::to_string(result) {
            let _ = cache.put_simple(&key, &json);
        }
    }
}

/// Clear the cache on writes. A future enhancement could track table-level
/// dependencies and invalidate selectively.
pub fn invalidate_cache_on_write(router: &QueryRouter) {
    if let Some(cache) = router.cache.as_ref() {
        cache.clear();
    }
}
