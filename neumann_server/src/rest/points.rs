// SPDX-License-Identifier: MIT OR Apache-2.0
//! REST API handlers for point operations.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;

use tensor_store::{ScalarValue, TensorValue};

use crate::audit::AuditEvent;
use crate::config::AuthConfig;
use crate::rate_limit::{Operation, RateLimiter};
use crate::rest::error::{ApiError, ApiResult};
use crate::rest::types::{
    DeleteRequest, DeleteResponse, GetRequest, GetResponse, PointStruct, QueryRequest,
    QueryResponse, ScoredPoint, ScrollRequest, ScrollResponse, UpsertRequest, UpsertResponse,
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

fn json_to_tensor_value(value: &serde_json::Value) -> TensorValue {
    match value {
        serde_json::Value::Null => TensorValue::Scalar(ScalarValue::Null),
        serde_json::Value::Bool(b) => TensorValue::Scalar(ScalarValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                TensorValue::Scalar(ScalarValue::Int(i))
            } else if let Some(f) = n.as_f64() {
                TensorValue::Scalar(ScalarValue::Float(f))
            } else {
                TensorValue::Scalar(ScalarValue::Null)
            }
        },
        serde_json::Value::String(s) => TensorValue::Scalar(ScalarValue::String(s.clone())),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            TensorValue::Scalar(ScalarValue::String(value.to_string()))
        },
    }
}

fn convert_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> std::collections::HashMap<String, TensorValue> {
    metadata
        .iter()
        .map(|(k, v)| (k.clone(), json_to_tensor_value(v)))
        .collect()
}

/// Upsert points into a collection.
pub async fn upsert(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(request): Json<UpsertRequest>,
) -> ApiResult<UpsertResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "vector_upsert",
    )?;

    let mut count = 0usize;

    for point in request.points {
        let result = if let Some(ref metadata) = point.payload {
            ctx.engine.store_in_collection_with_metadata(
                &collection,
                &point.id,
                point.vector,
                convert_metadata(metadata),
            )
        } else {
            ctx.engine
                .store_in_collection(&collection, &point.id, point.vector)
        };

        if let Err(e) = result {
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_request("vector", "upsert", false, latency_ms);
            }
            return Err(ApiError::internal(e.to_string()));
        }
        count += 1;
    }

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("upsert", latency_ms);
        m.record_request("vector", "upsert", true, latency_ms);
    }

    // Audit log
    if let Some(ref logger) = ctx.audit_logger {
        logger.record(
            AuditEvent::VectorUpsert {
                identity,
                collection,
                count,
            },
            None,
        );
    }

    Ok(Json(UpsertResponse {
        status: "ok".to_string(),
        upserted: count,
    }))
}

/// Get points by IDs.
pub async fn get(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(request): Json<GetRequest>,
) -> ApiResult<GetResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(identity.as_ref(), ctx.rate_limiter.as_ref(), "vector_get")?;

    let mut points = Vec::with_capacity(request.ids.len());

    for id in &request.ids {
        if let Ok(vector) = ctx.engine.get_from_collection(&collection, id) {
            points.push(PointStruct {
                id: id.clone(),
                vector: if request.with_vector { vector } else { vec![] },
                payload: None,
            });
        }
    }

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("get", latency_ms);
        m.record_request("vector", "get", true, latency_ms);
    }

    Ok(Json(GetResponse { points }))
}

/// Delete points by IDs.
pub async fn delete(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(request): Json<DeleteRequest>,
) -> ApiResult<DeleteResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "vector_delete",
    )?;

    let mut count = 0usize;

    for id in &request.ids {
        if ctx.engine.delete_from_collection(&collection, id).is_ok() {
            count += 1;
        }
    }

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("delete", latency_ms);
        m.record_request("vector", "delete", true, latency_ms);
    }

    // Audit log
    if let Some(ref logger) = ctx.audit_logger {
        logger.record(
            AuditEvent::VectorDelete {
                identity,
                collection,
                count,
            },
            None,
        );
    }

    Ok(Json(DeleteResponse {
        status: "ok".to_string(),
        deleted: count,
    }))
}

/// Query similar points.
pub async fn query(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(request): Json<QueryRequest>,
) -> ApiResult<QueryResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(identity.as_ref(), ctx.rate_limiter.as_ref(), "vector_query")?;

    let limit = request.limit.max(1);
    let search_result =
        ctx.engine
            .search_in_collection(&collection, &request.vector, limit + request.offset);

    let results = match search_result {
        Ok(items) => {
            let mut results = Vec::new();
            for item in items.into_iter().skip(request.offset).take(limit) {
                // Apply score threshold if specified
                if let Some(threshold) = request.score_threshold {
                    if item.score < threshold {
                        continue;
                    }
                }

                let vector = if request.with_vector {
                    ctx.engine.get_from_collection(&collection, &item.key).ok()
                } else {
                    None
                };

                results.push(ScoredPoint {
                    id: item.key,
                    score: item.score,
                    payload: None,
                    vector,
                });
            }
            results
        },
        Err(e) => {
            if let Some(ref m) = ctx.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_request("vector", "query", false, latency_ms);
            }
            return Err(ApiError::internal(e.to_string()));
        },
    };

    let elapsed = start.elapsed().as_secs_f64();

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = elapsed * 1000.0;
        m.record_vector_latency("query", latency_ms);
        m.record_request("vector", "query", true, latency_ms);
    }

    // Audit log
    if let Some(ref logger) = ctx.audit_logger {
        logger.record(
            AuditEvent::VectorQuery {
                identity,
                collection,
                limit,
            },
            None,
        );
    }

    Ok(Json(QueryResponse {
        result: results,
        time: elapsed,
    }))
}

/// Scroll through points in a collection.
pub async fn scroll(
    State(ctx): State<Arc<VectorApiContext>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(request): Json<ScrollRequest>,
) -> ApiResult<ScrollResponse> {
    let start = Instant::now();

    let identity = validate_auth(&headers, ctx.auth_config.as_ref())?;
    check_rate_limit(
        identity.as_ref(),
        ctx.rate_limiter.as_ref(),
        "vector_scroll",
    )?;

    let limit = request.limit.max(1);
    let keys = ctx.engine.list_collection_keys(&collection);

    // Find the starting position
    let start_idx = if let Some(ref offset_id) = request.offset_id {
        keys.iter()
            .position(|k| k > offset_id)
            .unwrap_or(keys.len())
    } else {
        0
    };

    // Get the page of keys
    let page_keys: Vec<_> = keys.iter().skip(start_idx).take(limit + 1).collect();
    let has_more = page_keys.len() > limit;
    let keys_to_fetch: Vec<_> = page_keys.into_iter().take(limit).collect();

    let mut points = Vec::with_capacity(keys_to_fetch.len());
    for key in &keys_to_fetch {
        let vector = if request.with_vector {
            ctx.engine
                .get_from_collection(&collection, key)
                .unwrap_or_default()
        } else {
            vec![]
        };

        points.push(PointStruct {
            id: (*key).clone(),
            vector,
            payload: None,
        });
    }

    let next_offset = if has_more {
        keys_to_fetch.last().copied().cloned()
    } else {
        None
    };

    // Record metrics
    if let Some(ref m) = ctx.metrics {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        m.record_vector_latency("scroll", latency_ms);
        m.record_request("vector", "scroll", true, latency_ms);
    }

    let _ = identity;

    Ok(Json(ScrollResponse {
        points,
        next_offset,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiKey;
    use axum::http::HeaderValue;

    #[test]
    fn test_extract_api_key_default_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("test-key"));

        let key = extract_api_key(&headers, None);
        assert_eq!(key, Some("test-key".to_string()));
    }

    #[test]
    fn test_extract_api_key_custom_header() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("test-key"));

        let auth_config = AuthConfig::new().with_header("authorization".to_string());
        let key = extract_api_key(&headers, Some(&auth_config));
        assert_eq!(key, Some("test-key".to_string()));
    }

    #[test]
    fn test_extract_api_key_missing() {
        let headers = HeaderMap::new();
        let key = extract_api_key(&headers, None);
        assert!(key.is_none());
    }

    #[test]
    fn test_validate_auth_no_config() {
        let headers = HeaderMap::new();
        let result = validate_auth(&headers, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_validate_auth_anonymous_allowed() {
        let headers = HeaderMap::new();
        let auth_config = AuthConfig::new().with_anonymous(true);
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_auth_anonymous_not_allowed() {
        let headers = HeaderMap::new();
        let auth_config = AuthConfig::new().with_anonymous(false);
        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 401);
    }

    #[test]
    fn test_validate_auth_valid_key() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_static("test-api-key-12345678"),
        );

        let auth_config = AuthConfig::new()
            .with_anonymous(false)
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ));

        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:test".to_string()));
    }

    #[test]
    fn test_validate_auth_invalid_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("wrong-key-12345678"));

        let auth_config = AuthConfig::new()
            .with_anonymous(false)
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ));

        let result = validate_auth(&headers, Some(&auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 401);
    }

    #[test]
    fn test_check_rate_limit_no_limiter() {
        let identity = "test".to_string();
        let result = check_rate_limit(Some(&identity), None, "test_op");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rate_limit_no_identity() {
        let limiter = Arc::new(RateLimiter::default());
        let result = check_rate_limit(None, Some(&limiter), "test_op");
        assert!(result.is_ok());
    }
}
