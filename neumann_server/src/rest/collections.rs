// SPDX-License-Identifier: MIT OR Apache-2.0
//! REST API handlers for collection operations.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;

use vector_engine::{DistanceMetric, VectorCollectionConfig};

use crate::audit::AuditEvent;
use crate::config::AuthConfig;
use crate::rate_limit::{Operation, RateLimiter};
use crate::rest::error::{ApiError, ApiResult};
use crate::rest::types::{
    CollectionInfo, CreateCollectionRequest, CreateCollectionResponse, DeleteCollectionResponse,
    ListCollectionsResponse,
};
use crate::rest::VectorApiContext;

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
        (Some(config), Some(key)) => match config.validate_key(&key) {
            Some(identity) => Ok(Some(identity.to_string())),
            None => Err(ApiError::unauthorized("Invalid API key")),
        },
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

fn parse_distance_metric(distance: &str) -> Result<DistanceMetric, ApiError> {
    match distance.to_lowercase().as_str() {
        "cosine" | "" => Ok(DistanceMetric::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "dot" | "dot_product" | "inner_product" => Ok(DistanceMetric::DotProduct),
        _ => Err(ApiError::bad_request(format!(
            "unknown distance metric: {distance}. Expected: cosine, euclidean, or dot"
        ))),
    }
}

fn metric_to_string(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::Euclidean => "euclidean",
        DistanceMetric::DotProduct => "dot",
    }
}

/// Create a new collection.
pub async fn create(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(request): Json<CreateCollectionRequest>,
) -> ApiResult<CreateCollectionResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "create_collection",
    )?;

    let metric = parse_distance_metric(&request.distance)?;

    let config = VectorCollectionConfig::default()
        .with_dimension(request.dimension)
        .with_metric(metric);

    match ctx.engine.create_collection(&name, config) {
        Ok(()) => {
            // Record metrics
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_vector_latency("create_collection", latency_ms);
                m.record_request("vector", "create_collection", true, latency_ms);
            }

            // Audit log
            if let Some(ref logger) = ctx.audit_logger {
                logger.record(
                    AuditEvent::CollectionCreated {
                        identity,
                        collection: name,
                    },
                    None,
                );
            }

            Ok(Json(CreateCollectionResponse { created: true }))
        },
        Err(e) => {
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_request("vector", "create_collection", false, latency_ms);
            }
            Err(ApiError::conflict(e.to_string()))
        },
    }
}

/// Get collection information.
pub async fn get(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> ApiResult<CollectionInfo> {
    let start = Instant::now();

    let _identity = validate_auth(&headers, ctx.auth_config.as_ref())?;

    let config = ctx
        .engine
        .get_collection_config(&name)
        .ok_or_else(|| ApiError::not_found(format!("collection not found: {name}")))?;

    let points_count = ctx.engine.collection_count(&name);

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("get_collection", latency_ms);
        m.record_request("vector", "get_collection", true, latency_ms);
    }

    Ok(Json(CollectionInfo {
        name,
        points_count,
        dimension: config.dimension.unwrap_or(0),
        distance: metric_to_string(config.distance_metric).to_string(),
    }))
}

/// Delete a collection.
pub async fn delete(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> ApiResult<DeleteCollectionResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "delete_collection",
    )?;

    match ctx.engine.delete_collection(&name) {
        Ok(()) => {
            // Record metrics
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_vector_latency("delete_collection", latency_ms);
                m.record_request("vector", "delete_collection", true, latency_ms);
            }

            // Audit log
            if let Some(ref logger) = ctx.audit_logger {
                logger.record(
                    AuditEvent::CollectionDeleted {
                        identity,
                        collection: name,
                    },
                    None,
                );
            }

            Ok(Json(DeleteCollectionResponse { deleted: true }))
        },
        Err(e) => {
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_request("vector", "delete_collection", false, latency_ms);
            }
            Err(ApiError::not_found(e.to_string()))
        },
    }
}

/// List all collections.
pub async fn list(
    State(ctx): State<Arc<VectorApiContext>>,
    headers: HeaderMap,
) -> ApiResult<ListCollectionsResponse> {
    let start = Instant::now();

    let _identity = validate_auth(&headers, ctx.auth_config.as_ref())?;

    let collections = ctx.engine.list_collections();

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("list_collections", latency_ms);
        m.record_request("vector", "list_collections", true, latency_ms);
    }

    Ok(Json(ListCollectionsResponse { collections }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_distance_metric_cosine() {
        assert!(matches!(
            parse_distance_metric("cosine"),
            Ok(DistanceMetric::Cosine)
        ));
        assert!(matches!(
            parse_distance_metric("COSINE"),
            Ok(DistanceMetric::Cosine)
        ));
        assert!(matches!(
            parse_distance_metric(""),
            Ok(DistanceMetric::Cosine)
        ));
    }

    #[test]
    fn test_parse_distance_metric_euclidean() {
        assert!(matches!(
            parse_distance_metric("euclidean"),
            Ok(DistanceMetric::Euclidean)
        ));
        assert!(matches!(
            parse_distance_metric("l2"),
            Ok(DistanceMetric::Euclidean)
        ));
    }

    #[test]
    fn test_parse_distance_metric_dot() {
        assert!(matches!(
            parse_distance_metric("dot"),
            Ok(DistanceMetric::DotProduct)
        ));
        assert!(matches!(
            parse_distance_metric("dot_product"),
            Ok(DistanceMetric::DotProduct)
        ));
        assert!(matches!(
            parse_distance_metric("inner_product"),
            Ok(DistanceMetric::DotProduct)
        ));
    }

    #[test]
    fn test_parse_distance_metric_invalid() {
        let result = parse_distance_metric("unknown");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 400);
    }

    #[test]
    fn test_metric_to_string() {
        assert_eq!(metric_to_string(DistanceMetric::Cosine), "cosine");
        assert_eq!(metric_to_string(DistanceMetric::Euclidean), "euclidean");
        assert_eq!(metric_to_string(DistanceMetric::DotProduct), "dot");
    }

    // === Auth Tests ===

    #[test]
    fn test_extract_api_key_present() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "test-key-12345".parse().unwrap());

        let key = extract_api_key(&headers, None);
        assert_eq!(key, Some("test-key-12345".to_string()));
    }

    #[test]
    fn test_extract_api_key_missing() {
        let headers = HeaderMap::new();
        let key = extract_api_key(&headers, None);
        assert_eq!(key, None);
    }

    #[test]
    fn test_extract_api_key_custom_header() {
        use crate::config::AuthConfig;

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer token123".parse().unwrap());

        let auth_config = AuthConfig::new().with_header("authorization".to_string());
        let key = extract_api_key(&headers, Some(&auth_config));
        assert_eq!(key, Some("Bearer token123".to_string()));
    }

    #[test]
    fn test_validate_auth_no_config() {
        let headers = HeaderMap::new();
        let result = validate_auth(&headers, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_validate_auth_anonymous_allowed() {
        use crate::config::AuthConfig;

        let headers = HeaderMap::new();
        let auth_config = AuthConfig::new().with_anonymous(true);
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_auth_anonymous_not_allowed() {
        use crate::config::{ApiKey, AuthConfig};

        let headers = HeaderMap::new();
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "valid-key-123456".to_string(),
                "user:test".to_string(),
            ))
            .with_anonymous(false);
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 401);
    }

    #[test]
    fn test_validate_auth_valid_key() {
        use crate::config::{ApiKey, AuthConfig};

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "valid-key-123456".parse().unwrap());

        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "valid-key-123456".to_string(),
            "user:test".to_string(),
        ));
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:test".to_string()));
    }

    #[test]
    fn test_validate_auth_invalid_key() {
        use crate::config::{ApiKey, AuthConfig};

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "wrong-key-000000".parse().unwrap());

        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "valid-key-123456".to_string(),
            "user:test".to_string(),
        ));
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 401);
    }

    // === Rate Limiting Tests ===

    #[test]
    fn test_check_rate_limit_no_limiter() {
        let identity = Some("user:test".to_string());
        let result = check_rate_limit(identity.as_ref(), None, "test_op");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rate_limit_no_identity() {
        use crate::rate_limit::RateLimiter;

        let rate_limiter = Arc::new(RateLimiter::default());
        let result = check_rate_limit(None, Some(&rate_limiter), "test_op");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rate_limit_enforced() {
        use crate::rate_limit::{RateLimitConfig, RateLimiter};

        let rate_limiter = Arc::new(RateLimiter::new(
            RateLimitConfig::new().with_max_vector_ops(1),
        ));
        let identity = Some("user:rate_test".to_string());

        // First call should pass
        let result = check_rate_limit(identity.as_ref(), Some(&rate_limiter), "op1");
        assert!(result.is_ok());

        // Second call should be rate limited
        let result = check_rate_limit(identity.as_ref(), Some(&rate_limiter), "op2");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 429);
    }

    // === Handler Integration Tests ===

    #[tokio::test]
    async fn test_create_collection_success() {
        use axum::http::HeaderMap;
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(Arc::clone(&engine)));

        let headers = HeaderMap::new();
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(
            State(ctx),
            Path("test_collection".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.0.created);

        // Verify collection exists
        assert!(engine.get_collection_config("test_collection").is_some());
    }

    #[tokio::test]
    async fn test_create_collection_duplicate() {
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection("existing", VectorCollectionConfig::default())
            .unwrap();

        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(
            State(ctx),
            Path("existing".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 409);
    }

    #[tokio::test]
    async fn test_create_collection_invalid_distance() {
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "invalid_metric".to_string(),
        };

        let result = create(State(ctx), Path("test".to_string()), headers, Json(request)).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 400);
    }

    #[tokio::test]
    async fn test_get_collection_success() {
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection(
                "test_get",
                VectorCollectionConfig::default()
                    .with_dimension(64)
                    .with_metric(DistanceMetric::Euclidean),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();

        let result = get(State(ctx), Path("test_get".to_string()), headers).await;

        assert!(result.is_ok());
        let info = result.unwrap().0;
        assert_eq!(info.name, "test_get");
        assert_eq!(info.dimension, 64);
        assert_eq!(info.distance, "euclidean");
    }

    #[tokio::test]
    async fn test_get_collection_not_found() {
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();

        let result = get(State(ctx), Path("nonexistent".to_string()), headers).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 404);
    }

    #[tokio::test]
    async fn test_delete_collection_success() {
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection("to_delete", VectorCollectionConfig::default())
            .unwrap();

        let ctx = Arc::new(VectorApiContext::new(Arc::clone(&engine)));
        let headers = HeaderMap::new();

        let result = delete(State(ctx), Path("to_delete".to_string()), headers).await;

        assert!(result.is_ok());
        assert!(result.unwrap().0.deleted);

        // Verify deleted
        assert!(engine.get_collection_config("to_delete").is_none());
    }

    #[tokio::test]
    async fn test_delete_collection_not_found() {
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();

        let result = delete(State(ctx), Path("nonexistent".to_string()), headers).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 404);
    }

    #[tokio::test]
    async fn test_list_collections_empty() {
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();

        let result = list(State(ctx), headers).await;

        assert!(result.is_ok());
        assert!(result.unwrap().0.collections.is_empty());
    }

    #[tokio::test]
    async fn test_list_collections_multiple() {
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection("coll_a", VectorCollectionConfig::default())
            .unwrap();
        engine
            .create_collection("coll_b", VectorCollectionConfig::default())
            .unwrap();
        engine
            .create_collection("coll_c", VectorCollectionConfig::default())
            .unwrap();

        let ctx = Arc::new(VectorApiContext::new(engine));
        let headers = HeaderMap::new();

        let result = list(State(ctx), headers).await;

        assert!(result.is_ok());
        let collections = result.unwrap().0.collections;
        assert_eq!(collections.len(), 3);
    }

    // === Auth Required Handler Tests ===

    #[tokio::test]
    async fn test_create_collection_auth_required() {
        use crate::config::{ApiKey, AuthConfig};
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "secret-key-123456".to_string(),
                "user:admin".to_string(),
            ))
            .with_anonymous(false);

        let ctx = Arc::new(VectorApiContext::new(engine).with_auth(Some(auth_config)));
        let headers = HeaderMap::new(); // No API key
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(State(ctx), Path("test".to_string()), headers, Json(request)).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 401);
    }

    #[tokio::test]
    async fn test_create_collection_with_valid_auth() {
        use crate::config::{ApiKey, AuthConfig};
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "secret-key-123456".to_string(),
            "user:admin".to_string(),
        ));

        let ctx = Arc::new(VectorApiContext::new(engine).with_auth(Some(auth_config)));
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "secret-key-123456".parse().unwrap());

        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(
            State(ctx),
            Path("authed_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
    }

    // === Metrics Tests ===

    #[tokio::test]
    async fn test_create_collection_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let ctx = Arc::new(VectorApiContext::new(engine).with_metrics(Some(metrics)));
        let headers = HeaderMap::new();
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(
            State(ctx),
            Path("metrics_test".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_collection_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection("metrics_get", VectorCollectionConfig::default())
            .unwrap();

        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let ctx = Arc::new(VectorApiContext::new(engine).with_metrics(Some(metrics)));
        let headers = HeaderMap::new();

        let result = get(State(ctx), Path("metrics_get".to_string()), headers).await;

        assert!(result.is_ok());
    }

    // === Audit Logging Tests ===

    #[tokio::test]
    async fn test_create_collection_audit_logged() {
        use crate::audit::{AuditConfig, AuditLogger};
        use vector_engine::VectorEngine;

        let engine = Arc::new(VectorEngine::new());
        let audit_logger = Arc::new(AuditLogger::new(AuditConfig::default()));

        let ctx = Arc::new(VectorApiContext::new(engine).with_audit_logger(Some(audit_logger)));
        let headers = HeaderMap::new();
        let request = CreateCollectionRequest {
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let result = create(
            State(ctx),
            Path("audited_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delete_collection_audit_logged() {
        use crate::audit::{AuditConfig, AuditLogger};
        use vector_engine::{VectorCollectionConfig, VectorEngine};

        let engine = Arc::new(VectorEngine::new());
        engine
            .create_collection("audit_delete", VectorCollectionConfig::default())
            .unwrap();

        let audit_logger = Arc::new(AuditLogger::new(AuditConfig::default()));
        let ctx = Arc::new(VectorApiContext::new(engine).with_audit_logger(Some(audit_logger)));
        let headers = HeaderMap::new();

        let result = delete(State(ctx), Path("audit_delete".to_string()), headers).await;

        assert!(result.is_ok());
    }
}
