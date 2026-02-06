// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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

    #[test]
    fn test_check_rate_limit_enforced() {
        use crate::rate_limit::RateLimitConfig;
        use std::time::Duration;

        let config = RateLimitConfig::new()
            .with_max_vector_ops(1)
            .with_window(Duration::from_secs(60));
        let limiter = Arc::new(RateLimiter::new(config));
        let identity = "test-user".to_string();

        // First request should succeed
        let result = check_rate_limit(Some(&identity), Some(&limiter), "vector_op");
        assert!(result.is_ok());

        // Second request should fail (rate limited)
        let result = check_rate_limit(Some(&identity), Some(&limiter), "vector_op");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 429);
    }

    // === JSON to TensorValue Tests ===

    #[test]
    fn test_json_to_tensor_value_null() {
        let value = serde_json::Value::Null;
        let result = json_to_tensor_value(&value);
        assert!(matches!(result, TensorValue::Scalar(ScalarValue::Null)));
    }

    #[test]
    fn test_json_to_tensor_value_bool_true() {
        let value = serde_json::Value::Bool(true);
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::Bool(true))
        ));
    }

    #[test]
    fn test_json_to_tensor_value_bool_false() {
        let value = serde_json::Value::Bool(false);
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::Bool(false))
        ));
    }

    #[test]
    fn test_json_to_tensor_value_int() {
        let value = serde_json::json!(42);
        let result = json_to_tensor_value(&value);
        assert!(matches!(result, TensorValue::Scalar(ScalarValue::Int(42))));
    }

    #[test]
    fn test_json_to_tensor_value_negative_int() {
        let value = serde_json::json!(-100);
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::Int(-100))
        ));
    }

    #[test]
    fn test_json_to_tensor_value_float() {
        let value = serde_json::json!(3.15);
        let result = json_to_tensor_value(&value);
        if let TensorValue::Scalar(ScalarValue::Float(f)) = result {
            assert!((f - 3.15).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_json_to_tensor_value_string() {
        let value = serde_json::json!("hello world");
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::String(ref s)) if s == "hello world"
        ));
    }

    #[test]
    fn test_json_to_tensor_value_empty_string() {
        let value = serde_json::json!("");
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::String(ref s)) if s.is_empty()
        ));
    }

    #[test]
    fn test_json_to_tensor_value_array() {
        let value = serde_json::json!([1, 2, 3]);
        let result = json_to_tensor_value(&value);
        // Arrays are converted to string representation
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::String(ref s)) if s.contains("[1,2,3]")
        ));
    }

    #[test]
    fn test_json_to_tensor_value_object() {
        let value = serde_json::json!({"key": "value"});
        let result = json_to_tensor_value(&value);
        // Objects are converted to string representation
        if let TensorValue::Scalar(ScalarValue::String(s)) = result {
            assert!(s.contains("key"));
            assert!(s.contains("value"));
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn test_json_to_tensor_value_large_int() {
        let value = serde_json::json!(i64::MAX);
        let result = json_to_tensor_value(&value);
        assert!(matches!(
            result,
            TensorValue::Scalar(ScalarValue::Int(i64::MAX))
        ));
    }

    // === Convert Metadata Tests ===

    #[test]
    fn test_convert_metadata_empty() {
        let metadata: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        let result = convert_metadata(&metadata);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_metadata_single_field() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), serde_json::json!("test"));
        let result = convert_metadata(&metadata);
        assert_eq!(result.len(), 1);
        assert!(matches!(
            result.get("name"),
            Some(TensorValue::Scalar(ScalarValue::String(ref s))) if s == "test"
        ));
    }

    #[test]
    fn test_convert_metadata_multiple_fields() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), serde_json::json!("item"));
        metadata.insert("count".to_string(), serde_json::json!(42));
        metadata.insert("active".to_string(), serde_json::json!(true));
        metadata.insert("price".to_string(), serde_json::json!(19.99));

        let result = convert_metadata(&metadata);
        assert_eq!(result.len(), 4);

        assert!(matches!(
            result.get("name"),
            Some(TensorValue::Scalar(ScalarValue::String(ref s))) if s == "item"
        ));
        assert!(matches!(
            result.get("count"),
            Some(TensorValue::Scalar(ScalarValue::Int(42)))
        ));
        assert!(matches!(
            result.get("active"),
            Some(TensorValue::Scalar(ScalarValue::Bool(true)))
        ));
        if let Some(TensorValue::Scalar(ScalarValue::Float(f))) = result.get("price") {
            assert!((*f - 19.99).abs() < 0.001);
        } else {
            panic!("Expected float for price");
        }
    }

    #[test]
    fn test_convert_metadata_with_null() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("empty".to_string(), serde_json::Value::Null);
        let result = convert_metadata(&metadata);
        assert!(matches!(
            result.get("empty"),
            Some(TensorValue::Scalar(ScalarValue::Null))
        ));
    }

    #[test]
    fn test_convert_metadata_with_nested_object() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("nested".to_string(), serde_json::json!({"a": 1, "b": 2}));
        let result = convert_metadata(&metadata);
        // Nested objects become string representation
        if let Some(TensorValue::Scalar(ScalarValue::String(s))) = result.get("nested") {
            assert!(s.contains("\"a\"") || s.contains("'a'"));
        } else {
            panic!("Expected string for nested object");
        }
    }

    // === Handler Tests with Mock Context ===

    #[tokio::test]
    async fn test_upsert_success() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![PointStruct {
                id: "point1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: None,
            }],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.status, "ok");
        assert_eq!(response.upserted, 1);
    }

    #[tokio::test]
    async fn test_upsert_with_payload() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let mut payload = std::collections::HashMap::new();
        payload.insert("category".to_string(), serde_json::json!("documents"));
        payload.insert("priority".to_string(), serde_json::json!(5));

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![PointStruct {
                id: "point2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                payload: Some(payload),
            }],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.upserted, 1);
    }

    #[tokio::test]
    async fn test_upsert_multiple_points() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![
                PointStruct {
                    id: "p1".to_string(),
                    vector: vec![1.0, 0.0, 0.0],
                    payload: None,
                },
                PointStruct {
                    id: "p2".to_string(),
                    vector: vec![0.0, 1.0, 0.0],
                    payload: None,
                },
                PointStruct {
                    id: "p3".to_string(),
                    vector: vec![0.0, 0.0, 1.0],
                    payload: None,
                },
            ],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.upserted, 3);
    }

    #[tokio::test]
    async fn test_upsert_auth_required() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let auth_config = AuthConfig::new().with_anonymous(false);

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: Some(auth_config),
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![PointStruct {
                id: "point1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: None,
            }],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.code, 401);
    }

    #[tokio::test]
    async fn test_upsert_with_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: Some(metrics.clone()),
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![PointStruct {
                id: "point1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: None,
            }],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        // Should succeed and metrics should have been recorded
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_upsert_with_audit_logging() {
        use crate::audit::{AuditConfig, AuditLogger};
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let audit_logger = Arc::new(AuditLogger::new(AuditConfig::default()));

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: Some(audit_logger),
        });

        let headers = HeaderMap::new();
        let request = UpsertRequest {
            points: vec![PointStruct {
                id: "point1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: None,
            }],
        };

        let result = upsert(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        // Should succeed - audit logger records the event
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_success() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "point1", vec![1.0, 0.5, 0.25])
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = GetRequest {
            ids: vec!["point1".to_string()],
            with_payload: false,
            with_vector: true,
        };

        let result = get(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.points.len(), 1);
        assert_eq!(response.points[0].id, "point1");
        assert!(!response.points[0].vector.is_empty());
    }

    #[tokio::test]
    async fn test_get_without_vector() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "point1", vec![1.0, 0.5, 0.25])
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = GetRequest {
            ids: vec!["point1".to_string()],
            with_payload: false,
            with_vector: false,
        };

        let result = get(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.points.len(), 1);
        assert!(response.points[0].vector.is_empty());
    }

    #[tokio::test]
    async fn test_get_nonexistent_point() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = GetRequest {
            ids: vec!["nonexistent".to_string()],
            with_payload: false,
            with_vector: true,
        };

        let result = get(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.points.is_empty());
    }

    #[tokio::test]
    async fn test_delete_success() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "point1", vec![1.0, 0.5, 0.25])
            .unwrap();

        let engine = Arc::new(engine);
        let ctx = Arc::new(VectorApiContext {
            engine: engine.clone(),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = DeleteRequest {
            ids: vec!["point1".to_string()],
        };

        let result = delete(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.status, "ok");
        assert_eq!(response.deleted, 1);

        // Verify point is deleted
        assert!(engine.get_from_collection("test_coll", "point1").is_err());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_point() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = DeleteRequest {
            ids: vec!["nonexistent".to_string()],
        };

        let result = delete(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.deleted, 0);
    }

    #[tokio::test]
    async fn test_query_success() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "p1", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .store_in_collection("test_coll", "p2", vec![0.9, 0.1, 0.0])
            .unwrap();
        engine
            .store_in_collection("test_coll", "p3", vec![0.0, 1.0, 0.0])
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = QueryRequest {
            vector: vec![1.0, 0.0, 0.0],
            limit: 2,
            offset: 0,
            with_payload: false,
            with_vector: false,
            score_threshold: None,
        };

        let result = query(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(!response.result.is_empty());
        assert!(response.time > 0.0);
    }

    #[tokio::test]
    async fn test_query_with_score_threshold() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "p1", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .store_in_collection("test_coll", "p2", vec![0.0, 1.0, 0.0])
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = QueryRequest {
            vector: vec![1.0, 0.0, 0.0],
            limit: 10,
            offset: 0,
            with_payload: false,
            with_vector: false,
            score_threshold: Some(0.9),
        };

        let result = query(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        // Only highly similar results should be returned
        for point in &response.result {
            assert!(point.score >= 0.9);
        }
    }

    #[tokio::test]
    async fn test_query_with_offset() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        for i in 0..10 {
            let v = vec![1.0 - (i as f32) * 0.05, 0.0, 0.0];
            engine
                .store_in_collection("test_coll", &format!("p{}", i), v)
                .unwrap();
        }

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = QueryRequest {
            vector: vec![1.0, 0.0, 0.0],
            limit: 3,
            offset: 2,
            with_payload: false,
            with_vector: false,
            score_threshold: None,
        };

        let result = query(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.result.len() <= 3);
    }

    #[tokio::test]
    async fn test_query_with_vector() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_coll", "p1", vec![1.0, 0.0, 0.0])
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = QueryRequest {
            vector: vec![1.0, 0.0, 0.0],
            limit: 1,
            offset: 0,
            with_payload: false,
            with_vector: true,
            score_threshold: None,
        };

        let result = query(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(!response.result.is_empty());
        assert!(response.result[0].vector.is_some());
    }

    #[tokio::test]
    async fn test_scroll_success() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        for i in 0..5 {
            engine
                .store_in_collection("test_coll", &format!("p{}", i), vec![1.0, 0.0, 0.0])
                .unwrap();
        }

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = ScrollRequest {
            limit: 3,
            offset_id: None,
            with_payload: false,
            with_vector: true,
        };

        let result = scroll(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.points.len() <= 3);
    }

    #[tokio::test]
    async fn test_scroll_with_offset() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        for i in 0..10 {
            engine
                .store_in_collection("test_coll", &format!("p{:02}", i), vec![1.0, 0.0, 0.0])
                .unwrap();
        }

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = ScrollRequest {
            limit: 3,
            offset_id: Some("p03".to_string()),
            with_payload: false,
            with_vector: false,
        };

        let result = scroll(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        // Should start after p03
        if !response.points.is_empty() {
            assert!(response.points[0].id.as_str() > "p03");
        }
    }

    #[tokio::test]
    async fn test_scroll_pagination() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        for i in 0..10 {
            engine
                .store_in_collection("test_coll", &format!("p{:02}", i), vec![1.0, 0.0, 0.0])
                .unwrap();
        }

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        // First page
        let headers = HeaderMap::new();
        let request = ScrollRequest {
            limit: 3,
            offset_id: None,
            with_payload: false,
            with_vector: false,
        };

        let result = scroll(
            State(ctx.clone()),
            Path("test_coll".to_string()),
            headers.clone(),
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.points.len(), 3);
        assert!(response.next_offset.is_some());

        // Second page using next_offset
        let request = ScrollRequest {
            limit: 3,
            offset_id: response.next_offset,
            with_payload: false,
            with_vector: false,
        };

        let result = scroll(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scroll_empty_collection() {
        use vector_engine::VectorEngine;

        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_coll",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let ctx = Arc::new(VectorApiContext {
            engine: Arc::new(engine),
            auth_config: None,
            rate_limiter: None,
            metrics: None,
            audit_logger: None,
        });

        let headers = HeaderMap::new();
        let request = ScrollRequest {
            limit: 10,
            offset_id: None,
            with_payload: false,
            with_vector: true,
        };

        let result = scroll(
            State(ctx),
            Path("test_coll".to_string()),
            headers,
            Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.points.is_empty());
        assert!(response.next_offset.is_none());
    }
}
