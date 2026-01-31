// SPDX-License-Identifier: MIT OR Apache-2.0
//! CollectionsService implementation for vector collection management.

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use vector_engine::{DistanceMetric, VectorCollectionConfig, VectorEngine};

use crate::audit::{AuditEvent, AuditLogger};
use crate::auth;
use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;
use crate::proto::vector::{
    collections_service_server::CollectionsService, CreateCollectionRequest,
    CreateCollectionResponse, DeleteCollectionRequest, DeleteCollectionResponse,
    GetCollectionRequest, GetCollectionResponse, ListCollectionsRequest, ListCollectionsResponse,
};
use crate::rate_limit::{Operation, RateLimiter};

/// Implementation of the CollectionsService gRPC service.
pub struct CollectionsServiceImpl {
    engine: Arc<VectorEngine>,
    auth_config: Option<AuthConfig>,
    rate_limiter: Option<Arc<RateLimiter>>,
    audit_logger: Option<Arc<AuditLogger>>,
    metrics: Option<Arc<ServerMetrics>>,
}

impl CollectionsServiceImpl {
    /// Create a new collections service.
    #[must_use]
    pub fn new(engine: Arc<VectorEngine>) -> Self {
        Self {
            engine,
            auth_config: None,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new collections service with authentication.
    #[must_use]
    pub fn with_auth(engine: Arc<VectorEngine>, auth_config: AuthConfig) -> Self {
        Self {
            engine,
            auth_config: Some(auth_config),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new collections service with rate limiting.
    #[must_use]
    pub fn with_config(
        engine: Arc<VectorEngine>,
        auth_config: Option<AuthConfig>,
        rate_limiter: Option<Arc<RateLimiter>>,
    ) -> Self {
        Self {
            engine,
            auth_config,
            rate_limiter,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new collections service with all options.
    #[must_use]
    pub fn with_full_config(
        engine: Arc<VectorEngine>,
        auth_config: Option<AuthConfig>,
        rate_limiter: Option<Arc<RateLimiter>>,
        audit_logger: Option<Arc<AuditLogger>>,
        metrics: Option<Arc<ServerMetrics>>,
    ) -> Self {
        Self {
            engine,
            auth_config,
            rate_limiter,
            audit_logger,
            metrics,
        }
    }
}

fn parse_distance_metric(distance: &str) -> Result<DistanceMetric, Status> {
    match distance.to_lowercase().as_str() {
        "cosine" | "" => Ok(DistanceMetric::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "dot" | "dot_product" | "inner_product" => Ok(DistanceMetric::DotProduct),
        _ => Err(Status::invalid_argument(format!(
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

#[tonic::async_trait]
impl CollectionsService for CollectionsServiceImpl {
    async fn create(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
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
                    m.record_request("vector", "create_collection", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "create_collection");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "create_collection", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();
        let metric = parse_distance_metric(&req.distance)?;

        let config = VectorCollectionConfig::default()
            .with_dimension(req.dimension as usize)
            .with_metric(metric);

        match self.engine.create_collection(&req.name, config) {
            Ok(()) => {
                // Record metrics
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_vector_latency("create_collection", latency_ms);
                    m.record_request("vector", "create_collection", true, latency_ms);
                }

                // Audit log
                if let Some(ref logger) = self.audit_logger {
                    logger.record(
                        AuditEvent::CollectionCreated {
                            identity,
                            collection: req.name,
                        },
                        None,
                    );
                }

                Ok(Response::new(CreateCollectionResponse { created: true }))
            },
            Err(e) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "create_collection", false, latency_ms);
                }
                Err(Status::already_exists(e.to_string()))
            },
        }
    }

    async fn get(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let _identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "get_collection", false, latency_ms);
                }
                return Err(status);
            },
        };

        let req = request.into_inner();

        let config = self
            .engine
            .get_collection_config(&req.name)
            .ok_or_else(|| Status::not_found(format!("collection not found: {}", req.name)))?;

        let points_count = self.engine.collection_count(&req.name) as u64;

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("get_collection", latency_ms);
            m.record_request("vector", "get_collection", true, latency_ms);
        }

        Ok(Response::new(GetCollectionResponse {
            name: req.name,
            points_count,
            dimension: u32::try_from(config.dimension.unwrap_or(0)).unwrap_or(u32::MAX),
            distance: metric_to_string(config.distance_metric).to_string(),
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
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
                    m.record_request("vector", "delete_collection", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::VectorOp) {
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "delete_collection");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("vector", "delete_collection", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let req = request.into_inner();

        match self.engine.delete_collection(&req.name) {
            Ok(()) => {
                // Record metrics
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_vector_latency("delete_collection", latency_ms);
                    m.record_request("vector", "delete_collection", true, latency_ms);
                }

                // Audit log
                if let Some(ref logger) = self.audit_logger {
                    logger.record(
                        AuditEvent::CollectionDeleted {
                            identity,
                            collection: req.name,
                        },
                        None,
                    );
                }

                Ok(Response::new(DeleteCollectionResponse { deleted: true }))
            },
            Err(e) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "delete_collection", false, latency_ms);
                }
                Err(Status::not_found(e.to_string()))
            },
        }
    }

    async fn list(
        &self,
        request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let start = Instant::now();

        // Validate authentication
        let _identity = match auth::validate_request_with_audit(
            &request,
            &self.auth_config,
            self.rate_limiter.as_deref(),
            self.audit_logger.as_deref(),
        ) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("vector", "list_collections", false, latency_ms);
                }
                return Err(status);
            },
        };

        let collections = self.engine.list_collections();

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_vector_latency("list_collections", latency_ms);
            m.record_request("vector", "list_collections", true, latency_ms);
        }

        Ok(Response::new(ListCollectionsResponse { collections }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collections_service_new() {
        let engine = Arc::new(VectorEngine::new());
        let service = CollectionsServiceImpl::new(engine);
        assert!(service.auth_config.is_none());
    }

    #[test]
    fn test_collections_service_with_auth() {
        use crate::config::ApiKey;

        let engine = Arc::new(VectorEngine::new());
        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:test".to_string(),
        ));
        let service = CollectionsServiceImpl::with_auth(engine, auth_config);
        assert!(service.auth_config.is_some());
    }

    #[test]
    fn test_collections_service_with_config() {
        let engine = Arc::new(VectorEngine::new());
        let rate_limiter = Arc::new(RateLimiter::default());
        let service = CollectionsServiceImpl::with_config(engine, None, Some(rate_limiter));
        assert!(service.rate_limiter.is_some());
    }

    #[test]
    fn test_collections_service_with_full_config() {
        let engine = Arc::new(VectorEngine::new());
        let rate_limiter = Arc::new(RateLimiter::default());
        let audit_logger = Arc::new(AuditLogger::default());

        let service = CollectionsServiceImpl::with_full_config(
            engine,
            None,
            Some(rate_limiter),
            Some(audit_logger),
            None,
        );

        assert!(service.rate_limiter.is_some());
        assert!(service.audit_logger.is_some());
    }

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
        assert!(parse_distance_metric("unknown").is_err());
    }

    #[test]
    fn test_metric_to_string() {
        assert_eq!(metric_to_string(DistanceMetric::Cosine), "cosine");
        assert_eq!(metric_to_string(DistanceMetric::Euclidean), "euclidean");
        assert_eq!(metric_to_string(DistanceMetric::DotProduct), "dot");
    }
}
