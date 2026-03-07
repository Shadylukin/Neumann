// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! REST API handlers for 3D spatial operations.
//!
//! Provides insert, query (radius), nearest, region, delete, and count
//! endpoints backed by a `SpatialIndex3D<String>` R-tree. The index is
//! global (shared across all collections via the `VectorApiContext`).

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::config::AuthConfig;
use crate::rate_limit::{Operation, RateLimiter};
use crate::rest::error::{ApiError, ApiResult};
use crate::rest::VectorApiContext;

// ---------------------------------------------------------------------------
// Auth helpers (same pattern as spatial.rs)
// ---------------------------------------------------------------------------

fn extract_api_key(headers: &HeaderMap, auth_config: Option<&AuthConfig>) -> Option<String> {
    let header_name = auth_config.map_or("x-api-key", |c| c.api_key_header.as_str());
    headers
        .get(header_name)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

fn validate_auth(
    headers: &HeaderMap,
    auth_config: Option<&AuthConfig>,
) -> Result<Option<String>, ApiError> {
    let api_key = extract_api_key(headers, auth_config);

    match (auth_config, api_key) {
        (None, _) => Ok(None),
        (Some(config), None) => {
            if config.allow_anonymous {
                Ok(None)
            } else {
                Err(ApiError::unauthorized("API key required"))
            }
        },
        (Some(config), Some(key)) => config.validate_key(&key).map_or_else(
            || Err(ApiError::unauthorized("Invalid API key")),
            |identity| Ok(Some(identity.to_string())),
        ),
    }
}

fn check_rate_limit(
    identity: Option<&String>,
    rate_limiter: Option<&Arc<RateLimiter>>,
    operation: &str,
) -> Result<(), ApiError> {
    if let Some(limiter) = rate_limiter {
        if let Some(id) = identity {
            if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                tracing::warn!("Rate limited: {id} for {operation}");
                return Err(ApiError::rate_limited(msg));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request to insert a 3D spatial entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DInsertRequest {
    /// Key identifying the spatial entry.
    pub key: String,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
    /// Bounding box width (defaults to 1.0).
    #[serde(default = "default_extent")]
    pub w: f32,
    /// Bounding box height (defaults to 1.0).
    #[serde(default = "default_extent")]
    pub h: f32,
    /// Bounding box depth (defaults to 1.0).
    #[serde(default = "default_extent")]
    pub d: f32,
}

const fn default_extent() -> f32 {
    1.0
}

/// Request for radius or k-NN queries in 3D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DQueryRequest {
    /// X coordinate of the query point.
    pub x: f32,
    /// Y coordinate of the query point.
    pub y: f32,
    /// Z coordinate of the query point.
    pub z: f32,
    /// Search radius (for radius queries).
    pub radius: Option<f32>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

/// Request for bounding-box region query in 3D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DRegionRequest {
    /// Minimum corner `[x, y, z]`.
    pub min: [f32; 3],
    /// Maximum corner `[x, y, z]`.
    pub max: [f32; 3],
}

/// Request to delete a 3D spatial entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DDeleteRequest {
    /// Key of the entry to remove.
    pub key: String,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
    /// Bounding box width.
    #[serde(default = "default_extent")]
    pub w: f32,
    /// Bounding box height.
    #[serde(default = "default_extent")]
    pub h: f32,
    /// Bounding box depth.
    #[serde(default = "default_extent")]
    pub d: f32,
}

/// A single 3D spatial result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DResultItem {
    /// Key of the spatial entry.
    pub key: String,
    /// Distance from query point.
    pub distance: f32,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
}

/// Response for a 3D spatial query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DQueryResponse {
    /// Matching entries.
    pub results: Vec<Spatial3DResultItem>,
    /// Query execution time in milliseconds.
    pub time: f64,
}

/// Response for the 3D spatial count endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spatial3DCountResponse {
    /// Number of entries in the 3D spatial index.
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Insert a 3D spatial entry.
///
/// # Errors
///
/// Returns an error if authentication fails, the 3D spatial index is not
/// configured, or the bounding box has invalid dimensions.
pub async fn insert_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<Spatial3DInsertRequest>,
) -> ApiResult<serde_json::Value> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_insert",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let bounds = tensor_spatial::BoundingBox3D::new(body.x, body.y, body.z, body.w, body.h, body.d)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let entry = tensor_spatial::SpatialEntry3D {
        data: body.key,
        bounds,
    };
    spatial.write().insert(entry);

    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// Query 3D entries within a radius.
///
/// # Errors
///
/// Returns an error if authentication fails or the 3D spatial index is not
/// configured.
pub async fn query_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<Spatial3DQueryRequest>,
) -> ApiResult<Spatial3DQueryResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_query",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let radius = body.radius.unwrap_or(100.0);
    let start = Instant::now();
    let guard = spatial.read();
    let mut results: Vec<Spatial3DResultItem> = guard
        .query_within_radius_with_distances(body.x, body.y, body.z, radius)
        .into_iter()
        .map(|(e, dist)| {
            let (cx, cy, cz) = e.bounds.center();
            Spatial3DResultItem {
                key: e.data.clone(),
                distance: dist,
                x: cx,
                y: cy,
                z: cz,
            }
        })
        .collect();
    drop(guard);

    if let Some(max) = body.limit {
        results.truncate(max);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(Spatial3DQueryResponse {
        results,
        time: elapsed,
    }))
}

/// Find k nearest neighbors in 3D.
///
/// # Errors
///
/// Returns an error if authentication fails or the 3D spatial index is not
/// configured.
pub async fn nearest_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<Spatial3DQueryRequest>,
) -> ApiResult<Spatial3DQueryResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_nearest",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let k = body.limit.unwrap_or(10);
    let start = Instant::now();
    let guard = spatial.read();
    let results: Vec<Spatial3DResultItem> = guard
        .query_nearest_by_centroid(body.x, body.y, body.z, k)
        .into_iter()
        .map(|e| {
            let (cx, cy, cz) = e.bounds.center();
            let dx = body.x - cx;
            let dy = body.y - cy;
            let dz = body.z - cz;
            Spatial3DResultItem {
                key: e.data.clone(),
                distance: dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt(),
                x: cx,
                y: cy,
                z: cz,
            }
        })
        .collect();
    drop(guard);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(Spatial3DQueryResponse {
        results,
        time: elapsed,
    }))
}

/// Query 3D entries within an axis-aligned bounding box region.
///
/// # Errors
///
/// Returns an error if authentication fails, the 3D spatial index is not
/// configured, or the region bounds are invalid.
pub async fn region_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<Spatial3DRegionRequest>,
) -> ApiResult<Spatial3DQueryResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_region",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let width = body.max[0] - body.min[0];
    let height = body.max[1] - body.min[1];
    let depth = body.max[2] - body.min[2];
    let region = tensor_spatial::BoundingBox3D::new(
        body.min[0],
        body.min[1],
        body.min[2],
        width,
        height,
        depth,
    )
    .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let start = Instant::now();
    let guard = spatial.read();
    let results: Vec<Spatial3DResultItem> = guard
        .query_region(region)
        .into_iter()
        .map(|e| {
            let (cx, cy, cz) = e.bounds.center();
            Spatial3DResultItem {
                key: e.data.clone(),
                distance: 0.0,
                x: cx,
                y: cy,
                z: cz,
            }
        })
        .collect();
    drop(guard);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(Spatial3DQueryResponse {
        results,
        time: elapsed,
    }))
}

/// Delete a 3D spatial entry.
///
/// # Errors
///
/// Returns an error if authentication fails, the 3D spatial index is not
/// configured, or the entry is not found.
pub async fn delete_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
    Json(body): Json<Spatial3DDeleteRequest>,
) -> ApiResult<serde_json::Value> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_delete",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let bounds = tensor_spatial::BoundingBox3D::new(body.x, body.y, body.z, body.w, body.h, body.d)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let key = body.key;
    spatial
        .write()
        .remove(bounds, |e| e.data == key && e.bounds == bounds)
        .map_err(|e| ApiError::not_found(e.to_string()))?;

    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// Get the number of entries in the 3D spatial index.
///
/// # Errors
///
/// Returns an error if authentication fails or the 3D spatial index is not
/// configured.
pub async fn count_3d(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
    Path(_name): Path<String>,
) -> ApiResult<Spatial3DCountResponse> {
    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "spatial3d_count",
    )?;

    let spatial = ctx
        .spatial_3d
        .as_ref()
        .ok_or_else(|| ApiError::internal("3D spatial index not configured"))?;

    let count = spatial.read().len();
    Ok(Json(Spatial3DCountResponse { count }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Serde Round-Trip Tests ==========

    #[test]
    fn test_serde_insert_request() {
        let req = Spatial3DInsertRequest {
            key: "paper:W123".to_string(),
            x: 10.0,
            y: 20.0,
            z: 30.0,
            w: 1.0,
            h: 1.0,
            d: 1.0,
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: Spatial3DInsertRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.key, "paper:W123");
        assert!((decoded.z - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_insert_request_defaults() {
        let json = r#"{"key":"p1","x":1.0,"y":2.0,"z":3.0}"#;
        let decoded: Spatial3DInsertRequest = serde_json::from_str(json).unwrap();
        assert!((decoded.w - 1.0).abs() < f32::EPSILON);
        assert!((decoded.h - 1.0).abs() < f32::EPSILON);
        assert!((decoded.d - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_query_request() {
        let req = Spatial3DQueryRequest {
            x: 5.0,
            y: 5.0,
            z: 5.0,
            radius: Some(10.0),
            limit: Some(50),
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: Spatial3DQueryRequest = serde_json::from_str(&json).unwrap();
        assert!((decoded.z - 5.0).abs() < f32::EPSILON);
        assert_eq!(decoded.limit, Some(50));
    }

    #[test]
    fn test_serde_region_request() {
        let req = Spatial3DRegionRequest {
            min: [-10.0, -10.0, -10.0],
            max: [10.0, 10.0, 10.0],
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: Spatial3DRegionRequest = serde_json::from_str(&json).unwrap();
        assert!((decoded.min[2] - (-10.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_result_item() {
        let item = Spatial3DResultItem {
            key: "a".to_string(),
            distance: 1.5,
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let json = serde_json::to_string(&item).unwrap();
        let decoded: Spatial3DResultItem = serde_json::from_str(&json).unwrap();
        assert!((decoded.distance - 1.5).abs() < f32::EPSILON);
        assert!((decoded.z - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_count_response() {
        let resp = Spatial3DCountResponse { count: 42 };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: Spatial3DCountResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.count, 42);
    }

    // ========== Handler Unit Tests ==========

    fn make_ctx_with_spatial_3d() -> Arc<VectorApiContext> {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let spatial_3d = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex3D::<
            String,
        >::new()));
        Arc::new(VectorApiContext::new(engine).with_spatial_3d(Some(spatial_3d)))
    }

    #[tokio::test]
    async fn test_insert_3d_no_spatial() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));

        let body = Spatial3DInsertRequest {
            key: "test".to_string(),
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 1.0,
            h: 1.0,
            d: 1.0,
        };

        let result = insert_3d(
            State(ctx),
            HeaderMap::new(),
            Path("default".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_insert_3d_and_count() {
        let ctx = make_ctx_with_spatial_3d();

        let body = Spatial3DInsertRequest {
            key: "p1".to_string(),
            x: 10.0,
            y: 20.0,
            z: 30.0,
            w: 1.0,
            h: 1.0,
            d: 1.0,
        };
        let result = insert_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_ok());

        let count_result = count_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
        )
        .await;
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap().0.count, 1);
    }

    #[tokio::test]
    async fn test_query_3d_within_radius() {
        let ctx = make_ctx_with_spatial_3d();

        // Insert two entries: one near origin, one far
        for (key, x, y, z) in [
            ("near", 1.0_f32, 1.0_f32, 1.0_f32),
            ("far", 100.0, 100.0, 100.0),
        ] {
            let body = Spatial3DInsertRequest {
                key: key.to_string(),
                x,
                y,
                z,
                w: 1.0,
                h: 1.0,
                d: 1.0,
            };
            let _ = insert_3d(
                State(Arc::clone(&ctx)),
                HeaderMap::new(),
                Path("col".to_string()),
                Json(body),
            )
            .await
            .unwrap();
        }

        let q = Spatial3DQueryRequest {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            radius: Some(10.0),
            limit: None,
        };
        let result = query_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(q),
        )
        .await
        .unwrap();
        assert_eq!(result.0.results.len(), 1);
        assert_eq!(result.0.results[0].key, "near");
    }

    #[tokio::test]
    async fn test_nearest_3d() {
        let ctx = make_ctx_with_spatial_3d();

        // Insert three points
        for (key, x, y, z) in [
            ("a", 1.0_f32, 0.0_f32, 0.0_f32),
            ("b", 5.0, 0.0, 0.0),
            ("c", 10.0, 0.0, 0.0),
        ] {
            let body = Spatial3DInsertRequest {
                key: key.to_string(),
                x,
                y,
                z,
                w: 1.0,
                h: 1.0,
                d: 1.0,
            };
            let _ = insert_3d(
                State(Arc::clone(&ctx)),
                HeaderMap::new(),
                Path("col".to_string()),
                Json(body),
            )
            .await
            .unwrap();
        }

        let q = Spatial3DQueryRequest {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            radius: None,
            limit: Some(2),
        };
        let result = nearest_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(q),
        )
        .await
        .unwrap();
        assert_eq!(result.0.results.len(), 2);
        // Closest should be "a"
        assert_eq!(result.0.results[0].key, "a");
    }

    #[tokio::test]
    async fn test_region_3d() {
        let ctx = make_ctx_with_spatial_3d();

        // Insert points
        for (key, x, y, z) in [
            ("inside", 5.0_f32, 5.0_f32, 5.0_f32),
            ("outside", 50.0, 50.0, 50.0),
        ] {
            let body = Spatial3DInsertRequest {
                key: key.to_string(),
                x,
                y,
                z,
                w: 1.0,
                h: 1.0,
                d: 1.0,
            };
            let _ = insert_3d(
                State(Arc::clone(&ctx)),
                HeaderMap::new(),
                Path("col".to_string()),
                Json(body),
            )
            .await
            .unwrap();
        }

        let region = Spatial3DRegionRequest {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let result = region_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(region),
        )
        .await
        .unwrap();
        assert_eq!(result.0.results.len(), 1);
        assert_eq!(result.0.results[0].key, "inside");
    }

    #[tokio::test]
    async fn test_delete_3d() {
        let ctx = make_ctx_with_spatial_3d();

        let body = Spatial3DInsertRequest {
            key: "temp".to_string(),
            x: 5.0,
            y: 5.0,
            z: 5.0,
            w: 2.0,
            h: 2.0,
            d: 2.0,
        };
        let _ = insert_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await
        .unwrap();

        let count = count_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
        )
        .await
        .unwrap();
        assert_eq!(count.0.count, 1);

        let del = Spatial3DDeleteRequest {
            key: "temp".to_string(),
            x: 5.0,
            y: 5.0,
            z: 5.0,
            w: 2.0,
            h: 2.0,
            d: 2.0,
        };
        let _ = delete_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(del),
        )
        .await
        .unwrap();

        let count = count_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
        )
        .await
        .unwrap();
        assert_eq!(count.0.count, 0);
    }

    #[tokio::test]
    async fn test_insert_3d_invalid_bounds() {
        let ctx = make_ctx_with_spatial_3d();

        let body = Spatial3DInsertRequest {
            key: "bad".to_string(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: -1.0,
            h: 5.0,
            d: 5.0,
        };
        let result = insert_3d(
            State(ctx),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(body),
        )
        .await;
        assert!(result.is_err());
    }
}
