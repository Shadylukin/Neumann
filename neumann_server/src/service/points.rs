// SPDX-License-Identifier: MIT OR Apache-2.0
//! `PointsService` implementation for vector point operations.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tensor_store::{ScalarValue, TensorValue};
use tonic::{Request, Response, Status};

use query_router::{QueryRouter, ScrollOptions, SearchOptions, VectorPoint};
use vector_engine::VectorEngine;

use crate::error::router_error_to_status;
use crate::router_dispatch::dispatch_with_identity;

/// Converts a `TensorValue` back to a JSON value for payload retrieval.
///
/// This is the inverse of `json_to_tensor_value`. Since `json_to_tensor_value` only
/// produces `Scalar(*)` variants, `meta:` prefixed fields will only contain scalars.
/// Non-scalar variants are handled gracefully for completeness.
fn tensor_value_to_json(value: &TensorValue) -> serde_json::Value {
    match value {
        TensorValue::Scalar(s) => match s {
            ScalarValue::Null => serde_json::Value::Null,
            ScalarValue::Bool(b) => serde_json::Value::Bool(*b),
            ScalarValue::Int(i) => serde_json::json!(*i),
            ScalarValue::Float(f) => serde_json::Number::from_f64(*f)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            ScalarValue::String(s) => serde_json::Value::String(s.clone()),
            ScalarValue::Bytes(b) => serde_json::Value::String(
                String::from_utf8(b.clone()).unwrap_or_else(|e| format!("{:02x?}", e.into_bytes())),
            ),
        },
        TensorValue::Vector(v) => serde_json::json!(v),
        TensorValue::Sparse(_) => serde_json::Value::String("(sparse vector)".to_string()),
        TensorValue::Pointer(p) => serde_json::Value::String(p.clone()),
        TensorValue::Pointers(ps) => serde_json::json!(ps),
    }
}

/// Convert engine-side metadata into the gRPC byte map.
fn metadata_to_proto(
    metadata: std::collections::HashMap<String, TensorValue>,
) -> std::collections::HashMap<String, Vec<u8>> {
    metadata
        .into_iter()
        .filter_map(|(k, v)| {
            serde_json::to_vec(&tensor_value_to_json(&v))
                .ok()
                .map(|bytes| (k, bytes))
        })
        .collect()
}

fn json_to_tensor_value(value: &serde_json::Value) -> TensorValue {
    match value {
        serde_json::Value::Null => TensorValue::Scalar(ScalarValue::Null),
        serde_json::Value::Bool(b) => TensorValue::Scalar(ScalarValue::Bool(*b)),
        serde_json::Value::Number(n) => n.as_i64().map_or_else(
            || {
                n.as_f64()
                    .map_or(TensorValue::Scalar(ScalarValue::Null), |f| {
                        TensorValue::Scalar(ScalarValue::Float(f))
                    })
            },
            |i| TensorValue::Scalar(ScalarValue::Int(i)),
        ),
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

use crate::audit::{AuditEvent, AuditLogger};
use crate::auth;
use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;
use crate::proto::vector::{
    points_service_server::PointsService, DeletePointsRequest, DeletePointsResponse,
    GetPointsRequest, GetPointsResponse, Point, QueryPointsRequest, QueryPointsResponse,
    ScoredPoint, ScrollPointsRequest, ScrollPointsResponse, UpsertPointsRequest,
    UpsertPointsResponse,
};
use crate::rate_limit::{Operation, RateLimiter};
use crate::service::health::HealthState;

/// Threshold for consecutive failures before marking unhealthy.
const FAILURE_THRESHOLD: u32 = 5;

/// Implementation of the `PointsService` gRPC service.
pub struct PointsServiceImpl {
    router: Arc<RwLock<QueryRouter>>,
    auth_config: Option<AuthConfig>,
    health_state: Option<Arc<HealthState>>,
    consecutive_failures: AtomicU32,
    rate_limiter: Option<Arc<RateLimiter>>,
    audit_logger: Option<Arc<AuditLogger>>,
    metrics: Option<Arc<ServerMetrics>>,
}

impl PointsServiceImpl {
    /// Create a new points service.
    #[must_use]
    pub const fn new(router: Arc<RwLock<QueryRouter>>) -> Self {
        Self {
            router,
            auth_config: None,
            health_state: None,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Convenience constructor for tests that already have a [`VectorEngine`].
    /// Wraps it in a fresh [`QueryRouter`] (matching the REST helper of the
    /// same name on [`crate::rest::VectorApiContext`]).
    #[must_use]
    pub fn from_engine(engine: Arc<VectorEngine>) -> Self {
        let mut router = QueryRouter::new();
        router.replace_vector_engine(engine);
        Self::new(Arc::new(RwLock::new(router)))
    }

    /// Create a new points service with authentication.
    #[must_use]
    pub const fn with_auth(router: Arc<RwLock<QueryRouter>>, auth_config: AuthConfig) -> Self {
        Self {
            router,
            auth_config: Some(auth_config),
            health_state: None,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new points service with health state monitoring.
    #[must_use]
    pub const fn with_config(
        router: Arc<RwLock<QueryRouter>>,
        auth_config: Option<AuthConfig>,
        health_state: Arc<HealthState>,
    ) -> Self {
        Self {
            router,
            auth_config,
            health_state: Some(health_state),
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new points service with all options.
    #[must_use]
    pub const fn with_full_config(
        router: Arc<RwLock<QueryRouter>>,
        auth_config: Option<AuthConfig>,
        health_state: Option<Arc<HealthState>>,
        rate_limiter: Option<Arc<RateLimiter>>,
        audit_logger: Option<Arc<AuditLogger>>,
        metrics: Option<Arc<ServerMetrics>>,
    ) -> Self {
        Self {
            router,
            auth_config,
            health_state,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter,
            audit_logger,
            metrics,
        }
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        if let Some(ref health) = self.health_state {
            health.set_vector_service_healthy(true);
        }
    }

    fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= FAILURE_THRESHOLD {
            if let Some(ref health) = self.health_state {
                health.set_vector_service_healthy(false);
                tracing::warn!(
                    "Points service marked unhealthy after {} consecutive failures",
                    failures
                );
            }
        }
    }
}

#[tonic::async_trait]
impl PointsService for PointsServiceImpl {
    async fn upsert(
        &self,
        request: Request<UpsertPointsRequest>,
    ) -> Result<Response<UpsertPointsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    if status.code() == tonic::Code::Unauthenticated {
                        m.record_auth_failure("invalid_key");
                    }
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "upsert", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "vector_upsert".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "vector_upsert");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "upsert", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let collection = req.collection.clone();

        let points: Vec<VectorPoint> = req
            .points
            .into_iter()
            .map(|point| {
                let metadata = if point.payload.is_empty() {
                    None
                } else {
                    let mut map: std::collections::HashMap<String, serde_json::Value> =
                        std::collections::HashMap::new();
                    for (k, v) in point.payload {
                        if let Ok(val) = serde_json::from_slice(&v) {
                            map.insert(k, val);
                        }
                    }
                    Some(convert_metadata(&map))
                };
                VectorPoint {
                    id: point.id,
                    vector: point.vector,
                    metadata,
                }
            })
            .collect();

        let count_result = dispatch_with_identity(&self.router, identity.as_deref(), |r| {
            r.upsert_points(Some(&collection), points)
        });
        let count = match count_result {
            Ok(n) => u64::try_from(n).unwrap_or(u64::MAX),
            Err(e) => {
                self.record_failure();
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "upsert", false, latency_ms);
                }
                return Err(router_error_to_status(e));
            },
        };

        self.record_success();

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("upsert", latency_ms);
            m.record_request("vector", "upsert", true, latency_ms);
        }

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::VectorUpsert {
                    identity,
                    collection,
                    count: usize::try_from(count).unwrap_or(usize::MAX),
                },
                None,
            );
        }

        Ok(Response::new(UpsertPointsResponse { upserted: count }))
    }

    async fn get(
        &self,
        request: Request<GetPointsRequest>,
    ) -> Result<Response<GetPointsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "get", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "vector_get");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "get", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let coll = req.collection;
        let ids = req.ids;
        let with_vector = req.with_vector;
        let with_payload = req.with_payload;
        let points_result = dispatch_with_identity(&self.router, identity.as_deref(), |r| {
            let mut acc = Vec::with_capacity(ids.len());
            for id in &ids {
                let Some(vector) = r.get_point(Some(&coll), id)? else {
                    continue;
                };
                let payload = if with_payload {
                    metadata_to_proto(r.get_point_metadata(&coll, id).unwrap_or_default())
                } else {
                    std::collections::HashMap::new()
                };
                acc.push(Point {
                    id: id.clone(),
                    vector: if with_vector { vector } else { vec![] },
                    payload,
                });
            }
            Ok(acc)
        });
        let points = match points_result {
            Ok(p) => p,
            Err(e) => {
                self.record_failure();
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "get", false, latency_ms);
                }
                return Err(router_error_to_status(e));
            },
        };

        self.record_success();

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("get", latency_ms);
            m.record_request("vector", "get", true, latency_ms);
        }

        Ok(Response::new(GetPointsResponse { points }))
    }

    async fn delete(
        &self,
        request: Request<DeletePointsRequest>,
    ) -> Result<Response<DeletePointsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "delete", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "vector_delete");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "delete", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let collection = req.collection.clone();
        let ids = req.ids;
        let outcome_result = dispatch_with_identity(&self.router, identity.as_deref(), |r| {
            r.delete_points(Some(&collection), &ids)
        });
        let count = match outcome_result {
            Ok(o) => u64::try_from(o.deleted).unwrap_or(u64::MAX),
            Err(e) => {
                self.record_failure();
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "delete", false, latency_ms);
                }
                return Err(router_error_to_status(e));
            },
        };

        self.record_success();

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("delete", latency_ms);
            m.record_request("vector", "delete", true, latency_ms);
        }

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::VectorDelete {
                    identity,
                    collection,
                    count: usize::try_from(count).unwrap_or(usize::MAX),
                },
                None,
            );
        }

        Ok(Response::new(DeletePointsResponse { deleted: count }))
    }

    async fn query(
        &self,
        request: Request<QueryPointsRequest>,
    ) -> Result<Response<QueryPointsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "query", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "vector_query");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "query", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let collection = req.collection.clone();
        let limit = usize::try_from(req.limit.max(1)).unwrap_or(usize::MAX);
        let offset = usize::try_from(req.offset).unwrap_or(0);
        let opts = SearchOptions {
            limit,
            offset,
            filter: None,
            metric: None,
            score_threshold: req.score_threshold,
            with_vector: req.with_vector,
            with_payload: req.with_payload,
        };

        let hits_result = dispatch_with_identity(&self.router, identity.as_deref(), |r| {
            r.search_points(Some(&collection), &req.vector, &opts)
        });
        let results = match hits_result {
            Ok(hits) => {
                self.record_success();
                hits.into_iter()
                    .map(|h| ScoredPoint {
                        id: h.id,
                        score: h.score,
                        payload: h.metadata.map(metadata_to_proto).unwrap_or_default(),
                        vector: h.vector.unwrap_or_default(),
                    })
                    .collect()
            },
            Err(e) => {
                self.record_failure();
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "query", false, latency_ms);
                }
                return Err(router_error_to_status(e));
            },
        };

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("query", latency_ms);
            m.record_request("vector", "query", true, latency_ms);
        }

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::VectorQuery {
                    identity,
                    collection,
                    limit,
                },
                None,
            );
        }

        Ok(Response::new(QueryPointsResponse { results }))
    }

    async fn scroll(
        &self,
        request: Request<ScrollPointsRequest>,
    ) -> Result<Response<ScrollPointsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "scroll", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "vector_scroll");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "scroll", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let opts = ScrollOptions {
            limit: usize::try_from(req.limit.max(1)).unwrap_or(usize::MAX),
            offset_id: req.offset_id,
            with_vector: req.with_vector,
            with_payload: req.with_payload,
        };

        let scrolled_result = dispatch_with_identity(&self.router, identity.as_deref(), |r| {
            r.scroll_points(&req.collection, &opts)
        });
        let (points, next_offset) = match scrolled_result {
            Ok(s) => {
                let pts: Vec<Point> = s
                    .hits
                    .into_iter()
                    .map(|hit| Point {
                        id: hit.id,
                        vector: hit.vector.unwrap_or_default(),
                        payload: hit.metadata.map(metadata_to_proto).unwrap_or_default(),
                    })
                    .collect();
                (pts, s.next_offset_id)
            },
            Err(e) => {
                self.record_failure();
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "scroll", false, latency_ms);
                }
                return Err(router_error_to_status(e));
            },
        };

        self.record_success();

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("scroll", latency_ms);
            m.record_request("vector", "scroll", true, latency_ms);
        }

        let _ = identity;

        Ok(Response::new(ScrollPointsResponse {
            points,
            next_offset,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_points_service_new() {
        let engine = Arc::new(VectorEngine::new());
        let service = PointsServiceImpl::from_engine(engine);
        assert!(service.auth_config.is_none());
        assert!(service.health_state.is_none());
    }

    #[test]
    fn test_points_service_with_auth() {
        use crate::config::ApiKey;

        let engine = Arc::new(VectorEngine::new());
        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:test".to_string(),
        ));
        let svc = PointsServiceImpl::from_engine(engine);
        let service = PointsServiceImpl::with_auth(svc.router, auth_config);
        assert!(service.auth_config.is_some());
    }

    #[test]
    fn test_points_service_with_config() {
        let engine = Arc::new(VectorEngine::new());
        let health_state = Arc::new(HealthState::new());
        let svc = PointsServiceImpl::from_engine(engine);
        let service = PointsServiceImpl::with_config(svc.router, None, health_state);
        assert!(service.health_state.is_some());
    }

    #[test]
    fn test_points_service_with_full_config() {
        let engine = Arc::new(VectorEngine::new());
        let health_state = Arc::new(HealthState::new());
        let rate_limiter = Arc::new(RateLimiter::default());
        let audit_logger = Arc::new(AuditLogger::default());

        let svc = PointsServiceImpl::from_engine(engine);
        let service = PointsServiceImpl::with_full_config(
            svc.router,
            None,
            Some(health_state),
            Some(rate_limiter),
            Some(audit_logger),
            None,
        );

        assert!(service.rate_limiter.is_some());
        assert!(service.audit_logger.is_some());
    }

    #[test]
    fn test_record_success() {
        let engine = Arc::new(VectorEngine::new());
        let service = PointsServiceImpl::from_engine(engine);

        service.consecutive_failures.store(3, Ordering::SeqCst);
        service.record_success();
        assert_eq!(service.consecutive_failures.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_record_failure() {
        let engine = Arc::new(VectorEngine::new());
        let service = PointsServiceImpl::from_engine(engine);

        service.record_failure();
        assert_eq!(service.consecutive_failures.load(Ordering::SeqCst), 1);

        service.record_failure();
        assert_eq!(service.consecutive_failures.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_tensor_value_to_json_scalars() {
        assert_eq!(
            tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Null)),
            serde_json::Value::Null
        );
        assert_eq!(
            tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Bool(true))),
            serde_json::json!(true)
        );
        assert_eq!(
            tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Int(42))),
            serde_json::json!(42)
        );
        assert_eq!(
            tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Float(3.14))),
            serde_json::json!(3.14)
        );
        assert_eq!(
            tensor_value_to_json(&TensorValue::Scalar(ScalarValue::String(
                "hello".to_string()
            ))),
            serde_json::json!("hello")
        );
    }

    #[test]
    fn test_tensor_value_to_json_nan_float() {
        let result = tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Float(f64::NAN)));
        assert_eq!(result, serde_json::Value::Null);
    }

    #[test]
    fn test_tensor_value_to_json_inf_float() {
        let result = tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Float(f64::INFINITY)));
        assert_eq!(result, serde_json::Value::Null);
    }

    #[test]
    fn test_tensor_value_to_json_bytes() {
        let result = tensor_value_to_json(&TensorValue::Scalar(ScalarValue::Bytes(
            b"utf8text".to_vec(),
        )));
        assert_eq!(result, serde_json::json!("utf8text"));
    }

    #[test]
    fn test_tensor_value_to_json_non_scalar() {
        let result = tensor_value_to_json(&TensorValue::Vector(vec![1.0, 2.0, 3.0]));
        assert_eq!(result, serde_json::json!([1.0, 2.0, 3.0]));

        let result = tensor_value_to_json(&TensorValue::Pointer("ptr".to_string()));
        assert_eq!(result, serde_json::json!("ptr"));

        let result = tensor_value_to_json(&TensorValue::Pointers(vec![
            "a".to_string(),
            "b".to_string(),
        ]));
        assert_eq!(result, serde_json::json!(["a", "b"]));
    }

    #[test]
    fn test_retrieve_payload_roundtrip() {
        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_payload",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("docs".to_string())),
        );
        metadata.insert(
            "priority".to_string(),
            TensorValue::Scalar(ScalarValue::Int(5)),
        );

        engine
            .store_in_collection_with_metadata(
                "test_payload",
                "point1",
                vec![1.0, 0.0, 0.0],
                metadata,
            )
            .unwrap();

        let payload = metadata_to_proto(
            engine
                .get_collection_metadata("test_payload", "point1")
                .unwrap_or_default(),
        );
        assert!(!payload.is_empty());

        // Verify category
        let cat_json: serde_json::Value =
            serde_json::from_slice(payload.get("category").unwrap()).unwrap();
        assert_eq!(cat_json, serde_json::json!("docs"));

        // Verify priority
        let pri_json: serde_json::Value =
            serde_json::from_slice(payload.get("priority").unwrap()).unwrap();
        assert_eq!(pri_json, serde_json::json!(5));
    }

    #[test]
    fn test_retrieve_payload_empty() {
        let engine = VectorEngine::new();
        engine
            .create_collection(
                "test_empty",
                vector_engine::VectorCollectionConfig::default().with_dimension(3),
            )
            .unwrap();
        engine
            .store_in_collection("test_empty", "point1", vec![1.0, 0.0, 0.0])
            .unwrap();

        let payload = metadata_to_proto(
            engine
                .get_collection_metadata("test_empty", "point1")
                .unwrap_or_default(),
        );
        assert!(payload.is_empty());
    }
}
