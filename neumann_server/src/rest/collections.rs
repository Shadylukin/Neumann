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
}
