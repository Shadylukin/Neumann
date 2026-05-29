// SPDX-License-Identifier: MIT OR Apache-2.0
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

use query_router::QueryRouter;

use crate::rate_limit::Operation;
use crate::rest::auth::{check_rate_limit, validate_auth};
use crate::rest::error::{ApiError, ApiResult};
use crate::rest::VectorApiContext;
use crate::router_dispatch::dispatch_with_identity;

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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let key = body.key;
    dispatch_with_identity(&ctx.router, identity.as_deref(), |r| {
        r.spatial3d_insert(key.clone(), body.x, body.y, body.z, body.w, body.h, body.d)
    })
    .map_err(ApiError::from)?;

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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let radius = body.radius.unwrap_or(100.0);
    let start = Instant::now();
    let hits = dispatch_with_identity(&ctx.router, identity.as_deref(), |r| {
        r.spatial3d_query_radius(body.x, body.y, body.z, radius, body.limit)
    })
    .map_err(ApiError::from)?;
    let results: Vec<Spatial3DResultItem> = hits
        .into_iter()
        .map(|h| Spatial3DResultItem {
            key: h.id,
            distance: h.distance,
            x: h.x,
            y: h.y,
            z: h.z,
        })
        .collect();
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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let k = body.limit.unwrap_or(10);
    let start = Instant::now();
    let hits = dispatch_with_identity(&ctx.router, identity.as_deref(), |r| {
        r.spatial3d_nearest(body.x, body.y, body.z, k)
    })
    .map_err(ApiError::from)?;
    let results: Vec<Spatial3DResultItem> = hits
        .into_iter()
        .map(|h| Spatial3DResultItem {
            key: h.id,
            distance: h.distance,
            x: h.x,
            y: h.y,
            z: h.z,
        })
        .collect();
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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let start = Instant::now();
    let hits = dispatch_with_identity(&ctx.router, identity.as_deref(), |r| {
        r.spatial3d_query_region(
            (body.min[0], body.min[1], body.min[2]),
            (body.max[0], body.max[1], body.max[2]),
            None,
        )
    })
    .map_err(ApiError::from)?;
    let results: Vec<Spatial3DResultItem> = hits
        .into_iter()
        .map(|h| Spatial3DResultItem {
            key: h.id,
            distance: h.distance,
            x: h.x,
            y: h.y,
            z: h.z,
        })
        .collect();
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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let removed = dispatch_with_identity(&ctx.router, identity.as_deref(), |r| {
        r.spatial3d_delete(&body.key, body.x, body.y, body.z, body.w, body.h, body.d)
    })
    .map_err(ApiError::from)?;
    if !removed {
        return Err(ApiError::not_found(format!(
            "3D spatial entry '{}' not found",
            body.key
        )));
    }

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
        identity.as_deref(),
        ctx.rate_limiter.as_ref(),
        Operation::VectorOp,
    )?;

    let count = dispatch_with_identity(
        &ctx.router,
        identity.as_deref(),
        QueryRouter::spatial3d_count,
    )
    .map_err(ApiError::from)?;
    Ok(Json(Spatial3DCountResponse { count }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limit::RateLimiter;
    use crate::rest::auth::extract_api_key;

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
        Arc::new(VectorApiContext::from_engine(engine).with_spatial_3d(Some(spatial_3d)))
    }

    #[tokio::test]
    async fn test_insert_3d_no_spatial() {
        let engine = Arc::new(vector_engine::VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::from_engine(engine));

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
    async fn test_query_3d_with_limit() {
        let ctx = make_ctx_with_spatial_3d();

        // Insert three entries
        for (key, x) in [("a", 1.0_f32), ("b", 2.0), ("c", 3.0)] {
            let body = Spatial3DInsertRequest {
                key: key.to_string(),
                x,
                y: 0.0,
                z: 0.0,
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
            radius: Some(100.0),
            limit: Some(2),
        };
        let result = query_3d(
            State(Arc::clone(&ctx)),
            HeaderMap::new(),
            Path("col".to_string()),
            Json(q),
        )
        .await
        .unwrap();
        assert!(result.0.results.len() <= 2);
    }

    #[test]
    fn test_validate_auth_no_config() {
        let result = validate_auth(&HeaderMap::new(), None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_validate_auth_required_but_missing() {
        use crate::config::{ApiKey, AuthConfig};

        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:test".to_string(),
        ));
        let result = validate_auth(&HeaderMap::new(), Some(&auth));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_auth_anonymous_allowed() {
        use crate::config::AuthConfig;

        let auth = AuthConfig::new().with_anonymous(true);
        let result = validate_auth(&HeaderMap::new(), Some(&auth));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_validate_auth_valid_key() {
        use crate::config::{ApiKey, AuthConfig};

        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "test-api-key-12345678".parse().unwrap());
        let result = validate_auth(&headers, Some(&auth));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), "user:alice");
    }

    #[test]
    fn test_validate_auth_invalid_key() {
        use crate::config::{ApiKey, AuthConfig};

        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "wrong-key-value".parse().unwrap());
        let result = validate_auth(&headers, Some(&auth));
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_api_key_default_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "mykey".parse().unwrap());
        let key = extract_api_key(&headers, None);
        assert_eq!(key.unwrap(), "mykey");
    }

    #[test]
    fn test_extract_api_key_custom_header() {
        use crate::config::AuthConfig;

        let auth = AuthConfig::new().with_header("authorization".to_string());
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "bearer-token".parse().unwrap());
        let key = extract_api_key(&headers, Some(&auth));
        assert_eq!(key.unwrap(), "bearer-token");
    }

    #[test]
    fn test_check_rate_limit_no_limiter() {
        let result = check_rate_limit(None, None, Operation::Query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rate_limit_no_identity() {
        let limiter = Arc::new(RateLimiter::default());
        let result = check_rate_limit(None, Some(&limiter), Operation::Query);
        assert!(result.is_ok());
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
