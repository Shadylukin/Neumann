// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vector statement execution: `EMBED`, `SIMILAR`, `SPATIAL`, plus the public
//! cross-engine helpers (`build_vector_index`, `find_similar_connected`,
//! `find_neighbors_by_similarity`) and their async counterparts.

#![allow(
    clippy::too_many_lines,
    reason = "match dispatchers cover many sub-ops"
)]

use std::sync::atomic::Ordering as AtomicOrdering;

use neumann_parser::{
    DistanceMetric as ParsedDistanceMetric, EmbedOp, EmbedStmt, Expr, SimilarQuery, SimilarStmt,
    SpatialOp, SpatialStmt,
};
use relational_engine::{Condition, Value};
use tensor_checkpoint::DestructiveOp;
use tensor_unified::UnifiedItem;
use vector_engine::{DistanceMetric as VectorDistanceMetric, FilterCondition, FilterValue};

use crate::exec::cache::invalidate_cache_on_write;
use crate::policy::ProtectedOpResult;
use crate::result::{SimilarResult, SpatialResult};
use crate::vector_ops::{SearchOptions, VectorPoint};
use crate::{exec, QueryResult, QueryRouter, Result, RouterError};

impl QueryRouter {
    /// Direct writes to the underlying `VectorEngine` via `vector()` accessor
    /// may cause stale results until the next rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error if index building fails.
    pub fn build_vector_index(&mut self) -> Result<()> {
        let (index, keys) = self.vector.build_hnsw_index_default()?;
        self.hnsw_index = Some((index, keys));
        self.hnsw_generation.store(
            self.vector_generation.load(AtomicOrdering::SeqCst),
            AtomicOrdering::SeqCst,
        );
        Ok(())
    }

    // ========== Cross-Engine Query Methods ==========
    // These methods enable queries that span multiple engines using unified entities.

    /// Find entities similar to a query entity that are also connected via graph edges.
    ///
    /// Returns entities that:
    /// 1. Have similar embeddings to the query entity
    /// 2. Are connected (directly or indirectly) to the specified `connected_to` entity
    ///
    /// Delegates to `UnifiedEngine::find_similar_connected_with_hnsw()`.
    ///
    /// # Errors
    ///
    /// Returns an error if unified engine is not initialized or query fails.
    pub fn find_similar_connected(
        &self,
        query_key: &str,
        connected_to: &str,
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        // Use HNSW if available and fresh for faster search
        let hnsw_results = if let Some((ref index, ref keys)) =
            self.hnsw_index.as_ref().filter(|_| self.hnsw_is_fresh())
        {
            let query_embedding = self
                .vector
                .get_entity_embedding(query_key)
                .map_err(|e| RouterError::VectorError(e.to_string()))?;
            Some(
                self.vector
                    .search_with_hnsw(index, keys, &query_embedding, top_k.saturating_mul(8))
                    .map_err(|e| RouterError::VectorError(e.to_string()))?,
            )
        } else {
            None
        };

        runtime
            .block_on(unified.find_similar_connected_with_hnsw(
                query_key,
                connected_to,
                top_k,
                hnsw_results,
            ))
            .map_err(Into::into)
    }

    /// Find graph neighbors of an entity that have embeddings, sorted by similarity to a query.
    ///
    /// Delegates to `UnifiedEngine::find_neighbors_by_similarity()`.
    ///
    /// # Errors
    ///
    /// Returns an error if unified engine is not initialized or query fails.
    pub fn find_neighbors_by_similarity(
        &self,
        entity_key: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        runtime
            .block_on(unified.find_neighbors_by_similarity(entity_key, query, top_k))
            .map_err(Into::into)
    }

    pub(crate) fn exec_embed(&self, embed: &EmbedStmt) -> Result<QueryResult> {
        let collection = embed.collection.as_deref();

        match &embed.operation {
            EmbedOp::Store { key, vector } => {
                let key_str = self.expr_to_string(key)?;
                let vec: Vec<f32> = vector
                    .iter()
                    .map(|e| self.expr_to_f32(e))
                    .collect::<Result<_>>()?;

                let (result, wrote_any) = self.upsert_points_impl(
                    collection,
                    vec![VectorPoint {
                        id: key_str,
                        vector: vec,
                        metadata: None,
                    }],
                );
                if wrote_any && result.is_err() {
                    // Partial mutation on failure — `execute()` won't invalidate
                    // on Err, so close the gap explicitly here. See vector_ops
                    // module docs.
                    invalidate_cache_on_write(self);
                }
                result?;
                Ok(QueryResult::Empty)
            },
            EmbedOp::Get { key } => {
                let key_str = self.expr_to_string(key)?;
                let vec = self
                    .get_point(collection, &key_str)?
                    .ok_or_else(|| RouterError::VectorError(format!("not found: {key_str}")))?;
                Ok(QueryResult::Value(format!("{vec:?}")))
            },
            EmbedOp::Delete { key } => {
                let key_str = self.expr_to_string(key)?;

                // Single-key checkpoint protection stays here so the
                // `protect_destructive_op` call shape matches today's exec
                // behavior exactly (sample data, command string, op kind).
                let op = DestructiveOp::EmbedDelete {
                    key: key_str.clone(),
                };

                match self.protect_destructive_op(
                    &format!("EMBED DELETE '{key_str}'"),
                    op,
                    vec![format!("embedding key: {}", key_str)],
                ) {
                    ProtectedOpResult::Proceed => {},
                    ProtectedOpResult::Cancelled => {
                        return Err(RouterError::CheckpointError(
                            "Operation cancelled by user".to_string(),
                        ));
                    },
                }

                let ids = [key_str];
                let (result, _mutated) = self.delete_points_impl(collection, &ids);
                let outcome = result?;
                if !outcome.missing.is_empty() {
                    // EMBED DELETE strict semantic: error on missing key.
                    return Err(RouterError::VectorError(format!(
                        "not found: {}",
                        outcome.missing[0]
                    )));
                }
                Ok(QueryResult::Count(outcome.deleted))
            },
            EmbedOp::BuildIndex => {
                // Building the index requires mutable access to the router
                // Check if index already exists
                if self.hnsw_index.is_some() {
                    Ok(QueryResult::Value("HNSW index already built".to_string()))
                } else {
                    Err(RouterError::VectorError(
                        "Use router.build_vector_index() to build HNSW index".to_string(),
                    ))
                }
            },
            EmbedOp::Batch { items } => {
                let mut points: Vec<VectorPoint> = Vec::with_capacity(items.len());
                for (key_expr, vector_exprs) in items {
                    let key_str = self.expr_to_string(key_expr)?;
                    let vec: Vec<f32> = vector_exprs
                        .iter()
                        .map(|e| self.expr_to_f32(e))
                        .collect::<Result<_>>()?;
                    points.push(VectorPoint {
                        id: key_str,
                        vector: vec,
                        metadata: None,
                    });
                }
                let (result, wrote_any) = self.upsert_points_impl(collection, points);
                if wrote_any && result.is_err() {
                    invalidate_cache_on_write(self);
                }
                Ok(QueryResult::Count(result?))
            },
        }
    }

    pub(crate) fn exec_similar(&self, similar: &SimilarStmt) -> Result<QueryResult> {
        let top_k = similar
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?
            .unwrap_or(10);

        let collection = similar.collection.as_deref();

        // SIMILAR...CONNECTED TO is a cross-engine path that combines vector
        // similarity with graph connectivity. It must short-circuit BEFORE the
        // standard delegation below.
        if let Some(ref connected_to_expr) = similar.connected_to {
            let query_key = match &similar.query {
                SimilarQuery::Key(key) => self.expr_to_string(key)?,
                SimilarQuery::Vector(_) => {
                    return Err(RouterError::ParseError(
                        "SIMILAR...CONNECTED TO requires a key, not a vector".to_string(),
                    ));
                },
            };
            let connected_to = self.expr_to_string(connected_to_expr)?;

            let items = self.find_similar_connected(&query_key, &connected_to, top_k)?;

            let results: Vec<SimilarResult> = items
                .into_iter()
                .map(|item| SimilarResult {
                    key: item.id,
                    score: item.score.unwrap_or(0.0),
                })
                .collect();

            return Ok(QueryResult::Similar(results));
        }

        // Standard similarity search: resolve query, filter, metric, then
        // delegate to the typed surface.
        let query_vec = match &similar.query {
            SimilarQuery::Key(key) => {
                let key_str = self.expr_to_string(key)?;
                if let Some(coll) = collection {
                    self.vector.get_from_collection(coll, &key_str)?
                } else {
                    self.vector.get_embedding(&key_str)?
                }
            },
            SimilarQuery::Vector(exprs) => exprs
                .iter()
                .map(|e| self.expr_to_f32(e))
                .collect::<Result<_>>()?,
        };

        let filter = if let Some(ref where_expr) = similar.where_clause {
            Some(self.expr_to_filter_condition(where_expr)?)
        } else {
            None
        };

        let metric = match similar.metric {
            Some(ParsedDistanceMetric::Cosine) | None => VectorDistanceMetric::Cosine,
            Some(ParsedDistanceMetric::Euclidean) => VectorDistanceMetric::Euclidean,
            Some(ParsedDistanceMetric::DotProduct) => VectorDistanceMetric::DotProduct,
            Some(ParsedDistanceMetric::Poincare) => VectorDistanceMetric::Poincare,
        };

        let opts = SearchOptions {
            limit: top_k,
            offset: 0,
            filter: filter.as_ref(),
            metric: Some(metric),
            score_threshold: None,
            with_vector: false,
            with_payload: false,
        };
        let hits = self.search_points(collection, &query_vec, &opts)?;

        let results: Vec<SimilarResult> = hits
            .into_iter()
            .map(|h| SimilarResult {
                key: h.id,
                score: h.score,
            })
            .collect();

        Ok(QueryResult::Similar(results))
    }

    // ========== AST Conversion Helpers ==========

    pub(crate) fn expr_to_condition(&self, expr: &Expr) -> Result<Condition> {
        exec::expr::expr_to_condition(self, expr)
    }

    /// Convert an expression to a vector engine `FilterCondition`.
    ///
    /// This method is public to allow programmatic construction of filters
    /// from parsed expressions.
    ///
    /// # Errors
    ///
    /// Returns an error if the expression cannot be converted to a filter condition.
    pub fn expr_to_filter_condition(&self, expr: &Expr) -> Result<FilterCondition> {
        exec::expr::expr_to_filter_condition(self, expr)
    }

    /// Convert an expression to a vector engine `FilterValue`.
    ///
    /// This method is public to allow programmatic construction of filter values
    /// from parsed expressions.
    ///
    /// # Errors
    ///
    /// Returns an error if the expression cannot be converted to a filter value.
    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub fn expr_to_filter_value(&self, expr: &Expr) -> Result<FilterValue> {
        exec::expr::expr_to_filter_value(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_value(&self, expr: &Expr) -> Result<Value> {
        exec::expr::expr_to_value(expr)
    }

    /// Extract a column name from an expression.
    ///
    /// This method is public to allow programmatic extraction of column names
    /// from parsed expressions.
    ///
    /// # Errors
    ///
    /// Returns an error if the expression is not a column reference.
    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub fn expr_to_column_name(&self, expr: &Expr) -> Result<String> {
        exec::expr::expr_to_column_name(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_u64(&self, expr: &Expr) -> Result<u64> {
        exec::expr::expr_to_u64(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_f32(&self, expr: &Expr) -> Result<f32> {
        exec::expr::expr_to_f32(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_f64(&self, expr: &Expr) -> Result<f64> {
        exec::expr::expr_to_f64(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_usize(&self, expr: &Expr) -> Result<usize> {
        exec::expr::expr_to_usize(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn expr_to_string(&self, expr: &Expr) -> Result<String> {
        exec::expr::expr_to_string(expr)
    }

    /// Executes a spatial statement.
    pub(crate) fn exec_spatial(&self, spatial_stmt: &SpatialStmt) -> Result<QueryResult> {
        match &spatial_stmt.op {
            SpatialOp::Insert {
                key,
                x,
                y,
                width,
                height,
            } => {
                let key_str = self.expr_to_string(key)?;
                let x_val = self.expr_to_f32(x)?;
                let y_val = self.expr_to_f32(y)?;
                let w_val = self.expr_to_f32(width)?;
                let h_val = self.expr_to_f32(height)?;
                self.spatial_insert_impl(key_str, x_val, y_val, w_val, h_val)?;
                Ok(QueryResult::Empty)
            },
            SpatialOp::WithinRadius {
                x,
                y,
                radius,
                limit,
            } => {
                let cx = self.expr_to_f32(x)?;
                let cy = self.expr_to_f32(y)?;
                let r = self.expr_to_f32(radius)?;
                let max_results = limit.as_ref().map(|e| self.expr_to_usize(e)).transpose()?;

                let hits = self.spatial_query_radius(cx, cy, r, max_results)?;
                let results: Vec<SpatialResult> = hits
                    .into_iter()
                    .map(|h| SpatialResult {
                        key: h.id,
                        distance: h.distance,
                        x: h.x,
                        y: h.y,
                        width: h.width,
                        height: h.height,
                    })
                    .collect();
                Ok(QueryResult::Spatial(results))
            },
            SpatialOp::Delete {
                key,
                x,
                y,
                width,
                height,
            } => {
                let key_str = self.expr_to_string(key)?;
                let x_val = self.expr_to_f32(x)?;
                let y_val = self.expr_to_f32(y)?;
                let w_val = self.expr_to_f32(width)?;
                let h_val = self.expr_to_f32(height)?;
                let removed = self.spatial_delete_impl(&key_str, x_val, y_val, w_val, h_val)?;
                if !removed {
                    // Preserve the strict NotFound semantic the string-dispatch
                    // path had before — the typed surface returns Ok(false) when
                    // the entry is missing.
                    return Err(RouterError::NotFound(format!(
                        "spatial entry '{key_str}' not found"
                    )));
                }
                Ok(QueryResult::Empty)
            },
            SpatialOp::Nearest { x, y, limit } => self.exec_spatial_nearest(x, y, limit.as_ref()),
            SpatialOp::Count => {
                let count = self.spatial_count();
                Ok(QueryResult::Count(count))
            },
        }
    }

    /// Executes a `SPATIAL NEAREST` centroid-distance query by delegating to
    /// the typed `spatial_nearest`.
    pub(crate) fn exec_spatial_nearest(
        &self,
        x: &Expr,
        y: &Expr,
        limit: Option<&Expr>,
    ) -> Result<QueryResult> {
        let cx = self.expr_to_f32(x)?;
        let cy = self.expr_to_f32(y)?;
        let k = limit
            .map(|e| self.expr_to_usize(e))
            .transpose()?
            .unwrap_or(1);
        let hits = self.spatial_nearest(cx, cy, k)?;
        let results: Vec<SpatialResult> = hits
            .into_iter()
            .map(|h| SpatialResult {
                key: h.id,
                distance: h.distance,
                x: h.x,
                y: h.y,
                width: h.width,
                height: h.height,
            })
            .collect();
        Ok(QueryResult::Spatial(results))
    }

    /// Store multiple embeddings in parallel using the unified engine.
    ///
    /// # Errors
    ///
    /// Returns an error if unified engine is not initialized or embedding fails.
    pub async fn embed_batch_parallel(&self, items: Vec<(String, Vec<f32>)>) -> Result<usize> {
        let unified = self.require_unified()?;

        let count = unified
            .embed_batch(items)
            .await
            .map(|result| result.count)
            .map_err(|e| RouterError::VectorError(e.to_string()))?;

        if count > 0 {
            self.bump_vector_generation();
        }
        Ok(count)
    }

    /// Find similar entities connected to a target asynchronously.
    ///
    /// Delegates to `UnifiedEngine::find_similar_connected_with_hnsw()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the unified engine is not available or vector operations fail.
    pub async fn find_similar_connected_async(
        &self,
        query_key: &str,
        connected_to: &str,
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;

        // Use HNSW if available and fresh for faster search
        let hnsw_results = if let Some((ref index, ref keys)) =
            self.hnsw_index.as_ref().filter(|_| self.hnsw_is_fresh())
        {
            let query_embedding = self
                .vector
                .get_entity_embedding(query_key)
                .map_err(|e| RouterError::VectorError(e.to_string()))?;
            Some(
                self.vector
                    .search_with_hnsw(index, keys, &query_embedding, top_k.saturating_mul(8))
                    .map_err(|e| RouterError::VectorError(e.to_string()))?,
            )
        } else {
            None
        };

        unified
            .find_similar_connected_with_hnsw(query_key, connected_to, top_k, hnsw_results)
            .await
            .map_err(Into::into)
    }

    /// Find graph neighbors sorted by similarity asynchronously.
    ///
    /// Delegates to `UnifiedEngine::find_neighbors_by_similarity()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the unified engine is not available or similarity search fails.
    pub async fn find_neighbors_by_similarity_async(
        &self,
        entity_key: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;

        unified
            .find_neighbors_by_similarity(entity_key, query, top_k)
            .await
            .map_err(Into::into)
    }

    /// Execute `SHOW EMBEDDINGS [LIMIT n]` — list stored embedding keys.
    pub(crate) fn exec_show_embeddings(&self, limit: Option<&Expr>) -> Result<QueryResult> {
        let limit_val = limit
            .map(|e| self.expr_to_usize(e))
            .transpose()?
            .unwrap_or(100);
        let keys = self.vector.list_keys();
        let limited: Vec<String> = keys.into_iter().take(limit_val).collect();
        Ok(QueryResult::Value(format!("Embeddings: {limited:?}")))
    }

    /// Execute `SHOW VECTOR INDEX` — report HNSW index status.
    pub(crate) fn exec_show_vector_index(&self) -> QueryResult {
        match &self.hnsw_index {
            Some((_, keys)) => {
                let count = keys.len();
                QueryResult::Value(format!("HNSW index: {count} vectors indexed"))
            },
            None => QueryResult::Value("No HNSW index built".to_string()),
        }
    }

    /// Execute `COUNT EMBEDDINGS`.
    pub(crate) fn exec_count_embeddings(&self) -> QueryResult {
        QueryResult::Count(self.vector.list_keys().len())
    }
}
