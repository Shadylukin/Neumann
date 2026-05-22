// SPDX-License-Identifier: MIT OR Apache-2.0
//! REST API handlers for spatial operations.
//!
//! The spatial index is global (shared across all collections via `QueryRouter`).
//! The `{name}` path parameter is accepted for URL consistency but the index
//! is not per-collection.  Future work can add per-collection spatial indexes.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::rate_limit::Operation;
use crate::rest::auth::{check_rate_limit, validate_auth};
use crate::rest::error::{ApiError, ApiResult};
use crate::rest::VectorApiContext;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request to insert a spatial entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialInsertRequest {
    /// Key identifying the spatial entry.
    pub key: String,
    /// X coordinate of the bounding-box origin.
    pub x: f32,
    /// Y coordinate of the bounding-box origin.
    pub y: f32,
    /// Width of the bounding box (must be non-negative).
    pub width: f32,
    /// Height of the bounding box (must be non-negative).
    pub height: f32,
}

/// Request to query entries within a radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQueryRequest {
    /// X coordinate of the query centre.
    pub x: f32,
    /// Y coordinate of the query centre.
    pub y: f32,
    /// Search radius (must be non-negative and finite).
    pub radius: f32,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
}

/// Request to delete a spatial entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialDeleteRequest {
    /// Key of the entry to remove.
    pub key: String,
    /// X coordinate of the bounding-box origin.
    pub x: f32,
    /// Y coordinate of the bounding-box origin.
    pub y: f32,
    /// Width of the bounding box.
    pub width: f32,
    /// Height of the bounding box.
    pub height: f32,
}

/// A single spatial result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialResultItem {
    /// Key of the spatial entry.
    pub key: String,
    /// Distance from query centre to closest point on the bounding box.
    pub distance: f32,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

/// Response for a spatial radius query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQueryResponse {
    /// Matching entries.
    pub result: Vec<SpatialResultItem>,
    /// Query execution time in milliseconds.
    pub time: f64,
}

/// Response for the spatial count endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialCountResponse {
    /// Number of entries in the spatial index.
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Insert a spatial entry.
///
/// # Errors
///
/// Returns an error if authentication fails, the spatial index is not
/// configured, or the bounding box has invalid dimensions.
pub async fn insert(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<SpatialInsertRequest>,
) -> ApiResult<serde_json::Value> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let spatial = ctx
        .spatial
        .as_ref()
        .ok_or_else(|| ApiError::internal("Spatial index not configured"))?;

    let bounds = tensor_spatial::BoundingBox::new(body.x, body.y, body.width, body.height)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let entry = tensor_spatial::SpatialEntry {
        data: body.key,
        bounds,
    };
    spatial.write().insert(entry);

    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// Query entries within a radius.
///
/// # Errors
///
/// Returns an error if authentication fails or the spatial index is not
/// configured.
pub async fn query(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<SpatialQueryRequest>,
) -> ApiResult<SpatialQueryResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let spatial = ctx
        .spatial
        .as_ref()
        .ok_or_else(|| ApiError::internal("Spatial index not configured"))?;

    let start = Instant::now();
    let guard = spatial.read();
    let mut results: Vec<SpatialResultItem> = guard
        .query_within_radius_with_distances(body.x, body.y, body.radius)
        .into_iter()
        .map(|(e, dist)| SpatialResultItem {
            key: e.data.clone(),
            distance: dist,
            x: e.bounds.x(),
            y: e.bounds.y(),
            width: e.bounds.width(),
            height: e.bounds.height(),
        })
        .collect();
    drop(guard);

    if let Some(max) = body.limit {
        results.truncate(max);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SpatialQueryResponse {
        result: results,
        time: elapsed,
    }))
}

/// Delete a spatial entry.
///
/// # Errors
///
/// Returns an error if authentication fails, the spatial index is not
/// configured, or the entry is not found.
pub async fn delete(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<SpatialDeleteRequest>,
) -> ApiResult<serde_json::Value> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let spatial = ctx
        .spatial
        .as_ref()
        .ok_or_else(|| ApiError::internal("Spatial index not configured"))?;

    let bounds = tensor_spatial::BoundingBox::new(body.x, body.y, body.width, body.height)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let key = body.key;
    spatial
        .write()
        .remove(bounds, |e| e.data == key && e.bounds == bounds)
        .map_err(|e| ApiError::not_found(e.to_string()))?;

    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// Get the number of entries in the spatial index.
///
/// # Errors
///
/// Returns an error if authentication fails or the spatial index is not
/// configured.
pub async fn count(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
) -> ApiResult<SpatialCountResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let spatial = ctx
        .spatial
        .as_ref()
        .ok_or_else(|| ApiError::internal("Spatial index not configured"))?;

    let count = spatial.read().len();
    Ok(Json(SpatialCountResponse { count }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Serde Round-Trip Tests ==========

    #[test]
    fn test_serde_spatial_insert_request() {
        let req = SpatialInsertRequest {
            key: "building".to_string(),
            x: 10.0,
            y: 20.0,
            width: 5.0,
            height: 3.0,
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: SpatialInsertRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.key, "building");
        assert!((decoded.x - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_spatial_query_request() {
        let req = SpatialQueryRequest {
            x: 5.0,
            y: 5.0,
            radius: 10.0,
            limit: Some(50),
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: SpatialQueryRequest = serde_json::from_str(&json).unwrap();
        assert!((decoded.radius - 10.0).abs() < f32::EPSILON);
        assert_eq!(decoded.limit, Some(50));
    }

    #[test]
    fn test_serde_spatial_query_request_no_limit() {
        let json = r#"{"x":1.0,"y":2.0,"radius":3.0}"#;
        let decoded: SpatialQueryRequest = serde_json::from_str(json).unwrap();
        assert!(decoded.limit.is_none());
    }

    #[test]
    fn test_serde_spatial_delete_request() {
        let req = SpatialDeleteRequest {
            key: "park".to_string(),
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: SpatialDeleteRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.key, "park");
    }

    #[test]
    fn test_serde_spatial_result_item() {
        let item = SpatialResultItem {
            key: "a".to_string(),
            distance: 1.5,
            x: 2.0,
            y: 3.0,
            width: 4.0,
            height: 5.0,
        };
        let json = serde_json::to_string(&item).unwrap();
        let decoded: SpatialResultItem = serde_json::from_str(&json).unwrap();
        assert!((decoded.distance - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_spatial_query_response() {
        let resp = SpatialQueryResponse {
            result: vec![SpatialResultItem {
                key: "b".to_string(),
                distance: 0.5,
                x: 1.0,
                y: 1.0,
                width: 2.0,
                height: 2.0,
            }],
            time: 1.234,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: SpatialQueryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.result.len(), 1);
    }

    #[test]
    fn test_serde_spatial_count_response() {
        let resp = SpatialCountResponse { count: 42 };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: SpatialCountResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.count, 42);
    }

    // ========== Handler Unit Tests ==========

    #[tokio::test]
    async fn test_insert_no_spatial_configured() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));

        let body = SpatialInsertRequest {
            key: "test".to_string(),
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };

        let result = insert(
            State(ctx),
            HeaderMap::new(),
            Path("default".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_insert_and_count() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let spatial = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::<
            String,
        >::new()));
        let ctx = Arc::new(VectorApiContext::new(engine).with_spatial(Some(Arc::clone(&spatial))));

        let body = SpatialInsertRequest {
            key: "obj1".to_string(),
            x: 10.0,
            y: 20.0,
            width: 5.0,
            height: 3.0,
        };
        let result = insert(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_ok());

        let count_result = count(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
        )
        .await;
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap().0.count, 1);
    }

    #[tokio::test]
    async fn test_query_within_radius() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let spatial = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::<
            String,
        >::new()));
        let ctx = Arc::new(VectorApiContext::new(engine).with_spatial(Some(Arc::clone(&spatial))));

        // Insert two entries
        for (key, x, y) in [("near", 1.0_f32, 1.0_f32), ("far", 100.0, 100.0)] {
            let body = SpatialInsertRequest {
                key: key.to_string(),
                x,
                y,
                width: 1.0,
                height: 1.0,
            };
            let _ = insert(
                State(Arc::clone(&ctx)),
                HeaderMap::new(),
                Path("col".to_string()),
                Json(body),
            )
            .await
            .unwrap();
        }

        let q = SpatialQueryRequest {
            x: 0.0,
            y: 0.0,
            radius: 10.0,
            limit: None,
        };
        let result = query(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(q),
        )
        .await
        .unwrap();
        assert_eq!(result.0.result.len(), 1);
        assert_eq!(result.0.result[0].key, "near");
    }

    #[tokio::test]
    async fn test_delete_entry() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let spatial = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::<
            String,
        >::new()));
        let ctx = Arc::new(VectorApiContext::new(engine).with_spatial(Some(Arc::clone(&spatial))));

        // Insert
        let body = SpatialInsertRequest {
            key: "temp".to_string(),
            x: 5.0,
            y: 5.0,
            width: 2.0,
            height: 2.0,
        };
        let _ = insert(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await
        .unwrap();
        assert_eq!(spatial.read().len(), 1);

        // Delete
        let del = SpatialDeleteRequest {
            key: "temp".to_string(),
            x: 5.0,
            y: 5.0,
            width: 2.0,
            height: 2.0,
        };
        let _ = delete(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(del),
        )
        .await
        .unwrap();
        assert_eq!(spatial.read().len(), 0);
    }

    #[tokio::test]
    async fn test_insert_invalid_bounds() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let spatial = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::<
            String,
        >::new()));
        let ctx = Arc::new(VectorApiContext::new(engine).with_spatial(Some(spatial)));

        let body = SpatialInsertRequest {
            key: "bad".to_string(),
            x: 0.0,
            y: 0.0,
            width: -1.0,
            height: 5.0,
        };
        let result = insert(
            State(ctx),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_count_no_spatial_configured() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));

        let result = count(State(ctx), HeaderMap::new(), Path("default".to_string())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_no_spatial_configured() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));

        let body = SpatialQueryRequest {
            x: 0.0,
            y: 0.0,
            radius: 10.0,
            limit: None,
        };
        let result = query(
            State(ctx),
            HeaderMap::new(),
            Path("default".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_no_spatial_configured() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));

        let body = SpatialDeleteRequest {
            key: "missing".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        };
        let result = delete(
            State(ctx),
            HeaderMap::new(),
            Path("default".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }
}
