// SPDX-License-Identifier: MIT OR Apache-2.0
//! Typed surface for vector / collection / spatial operations.
//!
//! These methods are the dispatch primitives that REST handlers, gRPC services,
//! and the string-parser path all eventually reach. Public methods wrap the
//! `pub(crate)` `_impl` siblings with cache invalidation, checkpoint
//! protection, and partial-mutation tracking. The `_impl` siblings own the
//! engine calls plus `bump_vector_generation` for default-namespace writes.
//!
//! Two distinct dispatch paths each manage their own cache-clear:
//!
//! - **REST / gRPC** call public methods. Each public write invalidates the
//!   cache iff the inner tuple reports `mutated == true`, regardless of
//!   overall Ok/Err.
//! - **String-dispatch** (`exec_embed`, `exec_spatial`) call `_impl` directly
//!   and let `QueryRouter::execute()` (lib.rs:336) invalidate on success.
//!   For partial-mutation `Err` paths on that surface, `exec_*` must call
//!   `crate::exec::cache::invalidate_cache_on_write(self)` itself.

use std::collections::HashMap;
use std::sync::Arc;

use tensor_checkpoint::DestructiveOp;
use tensor_spatial::{BoundingBox, BoundingBox3D, SpatialEntry, SpatialEntry3D, SpatialIndex3D};
use tensor_store::TensorValue;
use vector_engine::{FilterCondition, VectorCollectionConfig, VectorError};

pub use vector_engine::DistanceMetric as VectorDistanceMetric;

use crate::exec::cache::invalidate_cache_on_write;
use crate::policy::ProtectedOpResult;
use crate::{QueryRouter, Result, RouterError};

// ============================================================================
// DTOs
// ============================================================================

/// A point to upsert into a vector collection (or the default namespace).
#[derive(Debug, Clone)]
pub struct VectorPoint {
    /// External identifier.
    pub id: String,
    /// Dense vector payload.
    pub vector: Vec<f32>,
    /// Optional metadata fields stored alongside the vector. Only honored
    /// when `collection` is `Some` (the default namespace does not persist
    /// metadata).
    pub metadata: Option<HashMap<String, TensorValue>>,
}

/// Options for typed similarity search.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions<'a> {
    /// Maximum number of results to return after `offset` is applied. Treated
    /// as 1 if zero.
    pub limit: usize,
    /// Skip this many top-scoring hits before collecting results.
    pub offset: usize,
    /// Optional metadata filter applied during search.
    pub filter: Option<&'a FilterCondition>,
    /// Distance metric override (default cosine).
    pub metric: Option<VectorDistanceMetric>,
    /// Drop hits with `score < score_threshold` after enrichment.
    pub score_threshold: Option<f32>,
    /// Populate `SearchHit::vector` on each result.
    pub with_vector: bool,
    /// Populate `SearchHit::metadata` on each result (collection only).
    pub with_payload: bool,
}

/// Single similarity-search result with optional enrichment.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// External identifier.
    pub id: String,
    /// Similarity score.
    pub score: f32,
    /// Vector payload when `SearchOptions::with_vector` is set.
    pub vector: Option<Vec<f32>>,
    /// Metadata when `SearchOptions::with_payload` is set (collection only).
    pub metadata: Option<HashMap<String, TensorValue>>,
}

/// Options for typed paginated scroll across a collection.
#[derive(Debug, Clone, Default)]
pub struct ScrollOptions {
    /// Page size. Treated as 1 if zero.
    pub limit: usize,
    /// Resume cursor: return keys with `key > offset_id`.
    pub offset_id: Option<String>,
    /// Populate `ScrollHit::vector`.
    pub with_vector: bool,
    /// Populate `ScrollHit::metadata`.
    pub with_payload: bool,
}

/// Single scroll result with optional enrichment.
#[derive(Debug, Clone)]
pub struct ScrollHit {
    /// External identifier.
    pub id: String,
    /// Vector payload when `ScrollOptions::with_vector` is set.
    pub vector: Option<Vec<f32>>,
    /// Metadata when `ScrollOptions::with_payload` is set.
    pub metadata: Option<HashMap<String, TensorValue>>,
}

/// Page of scroll results plus the next-page cursor.
#[derive(Debug, Clone, Default)]
pub struct ScrolledPoints {
    /// Returned hits.
    pub hits: Vec<ScrollHit>,
    /// Cursor to pass as `offset_id` for the next page, or `None` at EOF.
    pub next_offset_id: Option<String>,
}

/// Outcome of a typed `delete_points` call.
///
/// Per finding #3 in the unification plan: REST silently skips missing IDs and
/// reports the count of successful deletes; `EMBED DELETE` errors on missing.
/// The structured outcome lets each caller adapt the semantics it needs.
#[derive(Debug, Clone, Default)]
pub struct DeleteOutcome {
    /// IDs that were actually removed from the engine.
    pub deleted: usize,
    /// IDs that were not present in the engine at probe time.
    pub missing: Vec<String>,
}

/// Information about a single collection.
#[derive(Debug, Clone)]
pub struct CollectionInfoResult {
    /// Collection name.
    pub name: String,
    /// Number of points currently in the collection.
    pub points_count: usize,
    /// Configuration (dimension, metric, indexing settings).
    pub config: VectorCollectionConfig,
}

/// 2D spatial hit returned by typed spatial queries.
#[derive(Debug, Clone)]
pub struct SpatialHit {
    /// External identifier.
    pub id: String,
    /// Distance from the query point.
    pub distance: f32,
    /// X coordinate of the lower corner.
    pub x: f32,
    /// Y coordinate of the lower corner.
    pub y: f32,
    /// Width of the bounding box.
    pub width: f32,
    /// Height of the bounding box.
    pub height: f32,
}

/// 3D spatial hit returned by typed spatial queries.
#[derive(Debug, Clone)]
pub struct Spatial3DHit {
    /// External identifier.
    pub id: String,
    /// Distance from the query point.
    pub distance: f32,
    /// X coordinate (centroid).
    pub x: f32,
    /// Y coordinate (centroid).
    pub y: f32,
    /// Z coordinate (centroid).
    pub z: f32,
}

// ============================================================================
// Typed methods
// ============================================================================

impl QueryRouter {
    // -----------------------------------------------------------------------
    // Points
    // -----------------------------------------------------------------------

    /// Upsert one or more vector points into a collection (or the default
    /// namespace when `collection` is `None`).
    ///
    /// Invalidates the query cache when at least one engine write succeeded,
    /// even if a later point in the batch failed. This avoids leaving cached
    /// SIMILAR results stale after a partial mutation.
    ///
    /// # Errors
    ///
    /// Forwards any [`VectorError`] from the engine. The error is returned
    /// after stable partial-mutation accounting (cache invalidation + HNSW
    /// staleness bump) has fired for the writes that did succeed.
    pub fn upsert_points(
        &self,
        collection: Option<&str>,
        points: Vec<VectorPoint>,
    ) -> Result<usize> {
        let (result, wrote_any) = self.upsert_points_impl(collection, points);
        if wrote_any {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Read a single point's vector. Returns `None` when the key is missing
    /// (REST semantic — `EMBED GET` callers adapt by treating `None` as an
    /// engine `NotFound`).
    ///
    /// # Errors
    ///
    /// Returns an error for engine-internal failures other than "not found".
    pub fn get_point(&self, collection: Option<&str>, id: &str) -> Result<Option<Vec<f32>>> {
        let res = collection.map_or_else(
            || self.vector.get_embedding(id),
            |coll| self.vector.get_from_collection(coll, id),
        );
        match res {
            Ok(v) => Ok(Some(v)),
            Err(VectorError::NotFound(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Read a single collection point's metadata.
    ///
    /// # Errors
    ///
    /// Returns a [`VectorError`] when the key does not exist in the
    /// collection (collections own metadata; the default namespace does not).
    pub fn get_point_metadata(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<HashMap<String, TensorValue>> {
        self.vector
            .get_collection_metadata(collection, id)
            .map_err(Into::into)
    }

    /// Delete a batch of IDs. Probes existence first so checkpoint protection
    /// records only the actually-existing subset (preventing misleading
    /// checkpoints on no-op deletes), then fires `protect_destructive_op`
    /// once. Returns a [`DeleteOutcome`] so each caller can decide whether to
    /// surface missing IDs (REST) or treat them as an error (`EMBED DELETE`).
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::CheckpointError`] when an auto-checkpoint is
    /// configured and the confirmation handler cancels the operation;
    /// otherwise forwards engine errors. Cache invalidation fires on any
    /// partial mutation (Err or Ok).
    pub fn delete_points(&self, collection: Option<&str>, ids: &[String]) -> Result<DeleteOutcome> {
        let mut existing: Vec<&str> = Vec::with_capacity(ids.len());
        for id in ids {
            let exists = collection.map_or_else(
                || self.vector.exists(id),
                |coll| self.vector.exists_in_collection(coll, id),
            );
            if exists {
                existing.push(id.as_str());
            }
        }

        if !existing.is_empty() {
            let keys: Vec<String> = existing.iter().copied().map(String::from).collect();
            let cmd = if keys.len() == 1 {
                format!("EMBED DELETE '{}'", keys[0])
            } else {
                format!(
                    "VECTOR DELETE collection={collection:?} ids=[{first} +{rest}]",
                    first = keys[0],
                    rest = keys.len() - 1
                )
            };
            let op = if keys.len() == 1 {
                DestructiveOp::EmbedDelete {
                    key: keys[0].clone(),
                }
            } else {
                DestructiveOp::EmbedDeleteBatch {
                    collection: collection.map(str::to_string),
                    keys: keys.clone(),
                }
            };
            match self.protect_destructive_op(&cmd, op, keys) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }
        }

        let (result, mutated) = self.delete_points_impl(collection, ids);
        if mutated {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Similarity search with optional filter, metric, score threshold,
    /// offset, and per-hit enrichment. Absorbs the search + per-hit
    /// vector/metadata fetches into a single call so handlers don't need
    /// extra round-trips under their lock.
    ///
    /// HNSW fast path only fires when `collection` is `None`, `opts.filter`
    /// is `None`, the cached index is fresh, and `opts.metric` is cosine
    /// (matching the existing `exec_similar` behavior).
    ///
    /// # Errors
    ///
    /// Forwards engine errors. `score_threshold` filtering happens after
    /// enrichment so the threshold compares the raw similarity score.
    pub fn search_points(
        &self,
        collection: Option<&str>,
        query: &[f32],
        opts: &SearchOptions<'_>,
    ) -> Result<Vec<SearchHit>> {
        let metric = opts.metric.unwrap_or_default();
        let take = opts.limit.max(1);
        let total_needed = opts.offset.saturating_add(take);

        let raw = match (collection, opts.filter) {
            (Some(coll), Some(f)) => {
                self.vector
                    .search_filtered_in_collection(coll, query, total_needed, f, None)?
            },
            (Some(coll), None) => self
                .vector
                .search_in_collection(coll, query, total_needed)?,
            (None, Some(f)) => self
                .vector
                .search_similar_filtered(query, total_needed, f, None)?,
            (None, None) => {
                if let Some((index, keys)) =
                    self.hnsw_index.as_ref().filter(|_| self.hnsw_is_fresh())
                {
                    if matches!(metric, VectorDistanceMetric::Cosine) {
                        self.vector
                            .search_with_hnsw(index, keys, query, total_needed)?
                    } else {
                        self.vector
                            .search_similar_with_metric(query, total_needed, metric)?
                    }
                } else {
                    self.vector
                        .search_similar_with_metric(query, total_needed, metric)?
                }
            },
        };

        let mut hits = Vec::with_capacity(take);
        for item in raw.into_iter().skip(opts.offset).take(take) {
            if let Some(threshold) = opts.score_threshold {
                if item.score < threshold {
                    continue;
                }
            }
            let vector = if opts.with_vector {
                match collection {
                    Some(coll) => self.vector.get_from_collection(coll, &item.key).ok(),
                    None => self.vector.get_embedding(&item.key).ok(),
                }
            } else {
                None
            };
            let metadata = if opts.with_payload {
                collection
                    .and_then(|coll| self.vector.get_collection_metadata(coll, &item.key).ok())
            } else {
                None
            };
            hits.push(SearchHit {
                id: item.key,
                score: item.score,
                vector,
                metadata,
            });
        }
        Ok(hits)
    }

    /// Scroll through points in a collection, sorted lexicographically.
    /// `offset_id` resumes after the given key. Returns the next cursor when
    /// more pages are available.
    ///
    /// # Errors
    ///
    /// Returns errors only from optional metadata fetches when the engine
    /// reports an unexpected failure (missing metadata silently yields
    /// `None`).
    pub fn scroll_points(&self, collection: &str, opts: &ScrollOptions) -> Result<ScrolledPoints> {
        let take = opts.limit.max(1);
        let mut keys = self.vector.list_collection_keys(collection);
        keys.sort();

        let start_idx = opts.offset_id.as_ref().map_or(0, |offset_id| {
            keys.iter()
                .position(|k| k > offset_id)
                .unwrap_or(keys.len())
        });

        let mut hits = Vec::new();
        let mut next_offset_id = None;
        for key in keys.iter().skip(start_idx) {
            if hits.len() >= take {
                next_offset_id = Some(key.clone());
                break;
            }
            let vector = if opts.with_vector {
                self.vector.get_from_collection(collection, key).ok()
            } else {
                None
            };
            let metadata = if opts.with_payload {
                self.vector.get_collection_metadata(collection, key).ok()
            } else {
                None
            };
            hits.push(ScrollHit {
                id: key.clone(),
                vector,
                metadata,
            });
        }
        Ok(ScrolledPoints {
            hits,
            next_offset_id,
        })
    }

    // -----------------------------------------------------------------------
    // Points: pub(crate) inner methods (no invalidate, no checkpoint).
    //
    // Called by `exec_embed` so the string-dispatch path goes through the
    // same engine code as REST/gRPC without double-clearing the cache.
    // -----------------------------------------------------------------------

    /// Engine-only upsert. Returns `(result, wrote_any)`. Bumps the HNSW
    /// staleness generation iff at least one default-namespace write
    /// succeeded — matches the behavior at exec/vector.rs:129/199 before
    /// the typed refactor.
    pub(crate) fn upsert_points_impl(
        &self,
        collection: Option<&str>,
        points: Vec<VectorPoint>,
    ) -> (Result<usize>, bool) {
        let mut wrote_any = false;
        let mut count = 0;
        for p in points {
            let res = match (collection, p.metadata) {
                (Some(coll), Some(meta)) => self
                    .vector
                    .store_in_collection_with_metadata(coll, &p.id, p.vector, meta),
                (Some(coll), None) => self.vector.store_in_collection(coll, &p.id, p.vector),
                (None, _) => self.vector.store_embedding(&p.id, p.vector),
            };
            match res {
                Ok(()) => {
                    wrote_any = true;
                    count += 1;
                },
                Err(e) => {
                    if wrote_any && collection.is_none() {
                        self.bump_vector_generation();
                    }
                    return (Err(e.into()), wrote_any);
                },
            }
        }
        if wrote_any && collection.is_none() {
            self.bump_vector_generation();
        }
        (Ok(count), wrote_any)
    }

    /// Engine-only delete. Returns `(result, mutated)`. Bumps the HNSW
    /// staleness generation iff at least one default-namespace delete
    /// succeeded — matches exec/vector.rs:167 behavior.
    pub(crate) fn delete_points_impl(
        &self,
        collection: Option<&str>,
        ids: &[String],
    ) -> (Result<DeleteOutcome>, bool) {
        let mut deleted = 0;
        let mut missing: Vec<String> = Vec::new();
        let mut mutated = false;
        for id in ids {
            let res = collection.map_or_else(
                || self.vector.delete_embedding(id),
                |coll| self.vector.delete_from_collection(coll, id),
            );
            match res {
                Ok(()) => {
                    deleted += 1;
                    mutated = true;
                },
                Err(VectorError::NotFound(_)) => {
                    missing.push(id.clone());
                },
                Err(e) => {
                    if mutated && collection.is_none() {
                        self.bump_vector_generation();
                    }
                    return (Err(e.into()), mutated);
                },
            }
        }
        if mutated && collection.is_none() {
            self.bump_vector_generation();
        }
        (Ok(DeleteOutcome { deleted, missing }), mutated)
    }

    // -----------------------------------------------------------------------
    // Collections
    // -----------------------------------------------------------------------

    /// Create a vector collection.
    ///
    /// # Errors
    ///
    /// Returns a [`VectorError`] when the collection already exists or the
    /// config is invalid. Cache invalidates on success.
    pub fn create_collection_typed(
        &self,
        name: &str,
        config: VectorCollectionConfig,
    ) -> Result<()> {
        let result = self.create_collection_impl(name, config);
        if result.is_ok() {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Drop a vector collection and return the number of points it held.
    ///
    /// Per the unification plan finding #1, this does NOT call
    /// `protect_destructive_op` — the checkpoint snapshot covers only the
    /// vector tensor store, while collection configs live in
    /// `VectorEngine.collections` outside that snapshot. Rollback would lie.
    /// Cache invalidates on success.
    ///
    /// # Errors
    ///
    /// Returns a [`VectorError`] when the collection does not exist.
    pub fn delete_collection_typed(&self, name: &str) -> Result<usize> {
        let result = self.delete_collection_impl(name);
        if result.is_ok() {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Look up collection metadata + point count. Returns `None` when the
    /// collection does not exist.
    ///
    /// # Errors
    ///
    /// Infallible today; signature reserves a slot for future configuration
    /// errors.
    pub fn get_collection_typed(&self, name: &str) -> Result<Option<CollectionInfoResult>> {
        let Some(config) = self.vector.get_collection_config(name) else {
            return Ok(None);
        };
        Ok(Some(CollectionInfoResult {
            name: name.to_string(),
            points_count: self.vector.collection_count(name),
            config,
        }))
    }

    /// List all collection names.
    #[must_use]
    pub fn list_collections_typed(&self) -> Vec<String> {
        self.vector.list_collections()
    }

    pub(crate) fn create_collection_impl(
        &self,
        name: &str,
        config: VectorCollectionConfig,
    ) -> Result<()> {
        self.vector
            .create_collection(name, config)
            .map_err(Into::into)
    }

    pub(crate) fn delete_collection_impl(&self, name: &str) -> Result<usize> {
        let count = self.vector.collection_count(name);
        self.vector
            .delete_collection(name)
            .map_err(RouterError::from)?;
        Ok(count)
    }

    // -----------------------------------------------------------------------
    // Spatial 2D (global; no collection parameter per finding #8).
    // -----------------------------------------------------------------------

    /// Insert a 2D bounding box into the global spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the bounding box is
    /// invalid (negative width/height).
    pub fn spatial_insert(
        &self,
        id: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<()> {
        let result = self.spatial_insert_impl(id, x, y, width, height);
        if result.is_ok() {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Radius query against the global 2D index.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when `radius` is negative
    /// or non-finite.
    pub fn spatial_query_radius(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        limit: Option<usize>,
    ) -> Result<Vec<SpatialHit>> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(RouterError::InvalidArgument(format!(
                "spatial radius must be non-negative and finite, got {radius}"
            )));
        }
        let mut results: Vec<SpatialHit> = {
            let guard = self.spatial.read();
            guard
                .query_within_radius_with_distances(x, y, radius)
                .into_iter()
                .map(|(entry, dist)| SpatialHit {
                    id: entry.data.clone(),
                    distance: dist,
                    x: entry.bounds.x(),
                    y: entry.bounds.y(),
                    width: entry.bounds.width(),
                    height: entry.bounds.height(),
                })
                .collect()
        };
        if let Some(max) = limit {
            results.truncate(max);
        }
        Ok(results)
    }

    /// k-nearest-neighbors against the global 2D index.
    ///
    /// # Errors
    ///
    /// Infallible today; the signature reserves space for future limit
    /// validation. `k == 0` returns an empty vector.
    pub fn spatial_nearest(&self, x: f32, y: f32, k: usize) -> Result<Vec<SpatialHit>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        // `query_nearest_by_centroid` matches the historical SPATIAL NEAREST
        // semantics (compare query against centroid distance). The point-to-box
        // `query_nearest` API would change observable ordering.
        let results: Vec<SpatialHit> = {
            let guard = self.spatial.read();
            guard
                .query_nearest_by_centroid(x, y, k)
                .into_iter()
                .map(|entry| {
                    let (cx, cy) = entry.bounds.center();
                    let dx = x - cx;
                    let dy = y - cy;
                    SpatialHit {
                        id: entry.data.clone(),
                        distance: dx.mul_add(dx, dy * dy).sqrt(),
                        x: entry.bounds.x(),
                        y: entry.bounds.y(),
                        width: entry.bounds.width(),
                        height: entry.bounds.height(),
                    }
                })
                .collect()
        };
        Ok(results)
    }

    /// Remove a 2D entry by ID + bounds. Returns `true` if the entry was
    /// found and removed, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the bounding box is
    /// invalid.
    pub fn spatial_delete(
        &self,
        id: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<bool> {
        let result = self.spatial_delete_impl(id, x, y, width, height);
        if matches!(result, Ok(true)) {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Number of entries in the 2D global index.
    #[must_use]
    pub fn spatial_count(&self) -> usize {
        self.spatial.read().len()
    }

    pub(crate) fn spatial_insert_impl(
        &self,
        id: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<()> {
        let bounds = BoundingBox::new(x, y, width, height)
            .map_err(|e| RouterError::InvalidArgument(format!("spatial bounds: {e}")))?;
        self.spatial
            .write()
            .insert(SpatialEntry { data: id, bounds });
        Ok(())
    }

    pub(crate) fn spatial_delete_impl(
        &self,
        id: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<bool> {
        let bounds = BoundingBox::new(x, y, width, height)
            .map_err(|e| RouterError::InvalidArgument(format!("spatial bounds: {e}")))?;
        let result = self
            .spatial
            .write()
            .remove(bounds, |entry: &SpatialEntry<String>| entry.data == id);
        match result {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    // -----------------------------------------------------------------------
    // Spatial 3D (global; optional — REST returns "not configured" when None)
    // -----------------------------------------------------------------------

    fn require_spatial_3d(&self) -> Result<&Arc<parking_lot::RwLock<SpatialIndex3D<String>>>> {
        self.spatial_3d.as_ref().ok_or_else(|| {
            RouterError::InvalidArgument("3D spatial index not configured".to_string())
        })
    }

    /// Insert a 3D bounding box into the global 3D index.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured or the bounds are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn spatial3d_insert(
        &self,
        id: String,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<()> {
        let result = self.spatial3d_insert_impl(id, x, y, z, width, height, depth);
        if result.is_ok() {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// 3D radius query (the REST `/spatial3d/query` endpoint).
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured or `radius` is negative/non-finite.
    pub fn spatial3d_query_radius(
        &self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        limit: Option<usize>,
    ) -> Result<Vec<Spatial3DHit>> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(RouterError::InvalidArgument(format!(
                "spatial radius must be non-negative and finite, got {radius}"
            )));
        }
        let spatial = self.require_spatial_3d()?;
        let mut results: Vec<Spatial3DHit> = {
            let guard = spatial.read();
            guard
                .query_within_radius_with_distances(x, y, z, radius)
                .into_iter()
                .map(|(entry, dist)| {
                    let (cx, cy, cz) = entry.bounds.center();
                    Spatial3DHit {
                        id: entry.data.clone(),
                        distance: dist,
                        x: cx,
                        y: cy,
                        z: cz,
                    }
                })
                .collect()
        };
        if let Some(max) = limit {
            results.truncate(max);
        }
        Ok(results)
    }

    /// 3D axis-aligned bounding-box query (the REST `/spatial3d/region`
    /// endpoint).
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured or the region is invalid.
    pub fn spatial3d_query_region(
        &self,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
        limit: Option<usize>,
    ) -> Result<Vec<Spatial3DHit>> {
        let width = max.0 - min.0;
        let height = max.1 - min.1;
        let depth = max.2 - min.2;
        let region = BoundingBox3D::new(min.0, min.1, min.2, width, height, depth)
            .map_err(|e| RouterError::InvalidArgument(format!("region: {e}")))?;
        let spatial = self.require_spatial_3d()?;
        let mut results: Vec<Spatial3DHit> = {
            let guard = spatial.read();
            guard
                .query_region(region)
                .into_iter()
                .map(|entry| {
                    let (cx, cy, cz) = entry.bounds.center();
                    Spatial3DHit {
                        id: entry.data.clone(),
                        distance: 0.0,
                        x: cx,
                        y: cy,
                        z: cz,
                    }
                })
                .collect()
        };
        if let Some(max) = limit {
            results.truncate(max);
        }
        Ok(results)
    }

    /// 3D k-nearest query (the REST `/spatial3d/nearest` endpoint).
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured. `k == 0` returns an empty vector.
    pub fn spatial3d_nearest(&self, x: f32, y: f32, z: f32, k: usize) -> Result<Vec<Spatial3DHit>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let spatial = self.require_spatial_3d()?;
        let results: Vec<Spatial3DHit> = {
            let guard = spatial.read();
            guard
                .query_nearest_by_centroid(x, y, z, k)
                .into_iter()
                .map(|entry| {
                    let (cx, cy, cz) = entry.bounds.center();
                    let dx = x - cx;
                    let dy = y - cy;
                    let dz = z - cz;
                    Spatial3DHit {
                        id: entry.data.clone(),
                        distance: dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt(),
                        x: cx,
                        y: cy,
                        z: cz,
                    }
                })
                .collect()
        };
        Ok(results)
    }

    /// Remove a 3D entry by ID + bounds. Returns `true` when the entry was
    /// found and removed.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured or the bounds are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn spatial3d_delete(
        &self,
        id: &str,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<bool> {
        let result = self.spatial3d_delete_impl(id, x, y, z, width, height, depth);
        if matches!(result, Ok(true)) {
            invalidate_cache_on_write(self);
        }
        result
    }

    /// Number of entries in the 3D global index.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::InvalidArgument`] when the 3D index is not
    /// configured.
    pub fn spatial3d_count(&self) -> Result<usize> {
        Ok(self.require_spatial_3d()?.read().len())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spatial3d_insert_impl(
        &self,
        id: String,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<()> {
        let bounds = BoundingBox3D::new(x, y, z, width, height, depth)
            .map_err(|e| RouterError::InvalidArgument(format!("spatial bounds: {e}")))?;
        self.require_spatial_3d()?
            .write()
            .insert(SpatialEntry3D { data: id, bounds });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spatial3d_delete_impl(
        &self,
        id: &str,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<bool> {
        let bounds = BoundingBox3D::new(x, y, z, width, height, depth)
            .map_err(|e| RouterError::InvalidArgument(format!("spatial bounds: {e}")))?;
        let result = self
            .require_spatial_3d()?
            .write()
            .remove(bounds, |entry: &SpatialEntry3D<String>| entry.data == id);
        match result {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}
