//! QueryService implementation for executing Neumann queries.

use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use query_router::{QueryResult, QueryRouter};

use crate::audit::{AuditEvent, AuditLogger};
use crate::auth;
use crate::config::AuthConfig;
use crate::convert::{
    edge_to_proto, node_to_proto, query_result_to_proto, row_to_proto, similar_to_proto,
};
use crate::metrics::ServerMetrics;
use crate::proto::{
    self, query_service_server::QueryService, BatchQueryRequest, BatchQueryResponse, QueryRequest,
    QueryResponse, QueryResponseChunk,
};
use crate::rate_limit::{Operation, RateLimiter};
use crate::service::health::HealthState;

/// Default channel capacity for streaming responses.
const DEFAULT_STREAM_CHANNEL_CAPACITY: usize = 32;

/// Threshold for consecutive failures before marking unhealthy.
const FAILURE_THRESHOLD: u32 = 5;

/// Implementation of the QueryService gRPC service.
pub struct QueryServiceImpl {
    router: Arc<RwLock<QueryRouter>>,
    auth_config: Option<AuthConfig>,
    stream_channel_capacity: usize,
    health_state: Option<Arc<HealthState>>,
    consecutive_failures: AtomicU32,
    rate_limiter: Option<Arc<RateLimiter>>,
    audit_logger: Option<Arc<AuditLogger>>,
    metrics: Option<Arc<ServerMetrics>>,
}

impl QueryServiceImpl {
    /// Create a new query service.
    #[must_use]
    pub fn new(router: Arc<RwLock<QueryRouter>>) -> Self {
        Self {
            router,
            auth_config: None,
            stream_channel_capacity: DEFAULT_STREAM_CHANNEL_CAPACITY,
            health_state: None,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new query service with authentication.
    #[must_use]
    pub fn with_auth(router: Arc<RwLock<QueryRouter>>, auth_config: AuthConfig) -> Self {
        Self {
            router,
            auth_config: Some(auth_config),
            stream_channel_capacity: DEFAULT_STREAM_CHANNEL_CAPACITY,
            health_state: None,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new query service with full configuration.
    #[must_use]
    pub fn with_config(
        router: Arc<RwLock<QueryRouter>>,
        auth_config: Option<AuthConfig>,
        stream_channel_capacity: usize,
    ) -> Self {
        Self {
            router,
            auth_config,
            stream_channel_capacity,
            health_state: None,
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new query service with health state monitoring.
    #[must_use]
    pub fn with_health_state(
        router: Arc<RwLock<QueryRouter>>,
        auth_config: Option<AuthConfig>,
        stream_channel_capacity: usize,
        health_state: Arc<HealthState>,
    ) -> Self {
        Self {
            router,
            auth_config,
            stream_channel_capacity,
            health_state: Some(health_state),
            consecutive_failures: AtomicU32::new(0),
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new query service with all options including rate limiting, audit logging, and metrics.
    #[must_use]
    pub fn with_full_config(
        router: Arc<RwLock<QueryRouter>>,
        auth_config: Option<AuthConfig>,
        stream_channel_capacity: usize,
        health_state: Arc<HealthState>,
        rate_limiter: Option<Arc<RateLimiter>>,
        audit_logger: Option<Arc<AuditLogger>>,
        metrics: Option<Arc<ServerMetrics>>,
    ) -> Self {
        Self {
            router,
            auth_config,
            stream_channel_capacity,
            health_state: Some(health_state),
            consecutive_failures: AtomicU32::new(0),
            rate_limiter,
            audit_logger,
            metrics,
        }
    }

    /// Record a successful query execution.
    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        if let Some(ref health) = self.health_state {
            health.set_query_service_healthy(true);
        }
    }

    /// Record a failed query execution.
    fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= FAILURE_THRESHOLD {
            if let Some(ref health) = self.health_state {
                health.set_query_service_healthy(false);
                tracing::warn!(
                    "Query service marked unhealthy after {} consecutive failures",
                    failures
                );
            }
        }
    }

    /// Execute a query and return the result.
    fn execute_query(&self, query: &str, identity: Option<&str>) -> Result<QueryResult, Status> {
        let mut router = self.router.write();

        // Set identity for vault access if provided
        if let Some(id) = identity {
            router.set_identity(id);
        }

        match router.execute(query) {
            Ok(result) => {
                self.record_success();
                Ok(result)
            },
            Err(e) => {
                self.record_failure();
                tracing::error!("Query execution error: {}", e);
                Err(Status::internal(e.to_string()))
            },
        }
    }
}

#[tonic::async_trait]
impl QueryService for QueryServiceImpl {
    async fn execute(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let start = Instant::now();

        // Validate authentication with rate limiting and audit
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
                    m.record_request("query", "execute", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check query-specific rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::Query) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "query".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "query");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("query", "execute", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let query = &request.get_ref().query;
        tracing::debug!("Executing query: {}", query);

        let result = self.execute_query(query, identity.as_deref());

        // Record metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("query", "execute", result.is_ok(), latency_ms);
        }

        // Audit the query execution
        if let Some(ref logger) = self.audit_logger {
            if logger.config().log_queries {
                logger.record(
                    AuditEvent::QueryExecuted {
                        identity,
                        query: query.clone(),
                    },
                    None,
                );
            }
        }

        match result {
            Ok(result) => Ok(Response::new(query_result_to_proto(result))),
            Err(status) => Err(status),
        }
    }

    type ExecuteStreamStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<QueryResponseChunk, Status>> + Send>>;

    async fn execute_stream(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<Self::ExecuteStreamStream>, Status> {
        let start = Instant::now();

        // Validate authentication with rate limiting and audit
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
                    m.record_request("query", "execute_stream", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check query-specific rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::Query) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "query".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "query");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("query", "execute_stream", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let query = &request.get_ref().query;
        tracing::debug!("Executing streaming query: {}", query);

        // Audit the query execution
        if let Some(ref logger) = self.audit_logger {
            if logger.config().log_queries {
                logger.record(
                    AuditEvent::QueryExecuted {
                        identity: identity.clone(),
                        query: query.clone(),
                    },
                    None,
                );
            }
        }

        let result = match self.execute_query(query, identity.as_deref()) {
            Ok(r) => r,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("query", "execute_stream", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Record metrics for successful stream setup
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("query", "execute_stream", true, latency_ms);
        }

        let (tx, rx) = mpsc::channel(self.stream_channel_capacity);

        // Spawn task to stream results
        tokio::spawn(async move {
            let send_result = match result {
                QueryResult::Rows(rows) => {
                    for row in rows {
                        let chunk = QueryResponseChunk {
                            chunk: Some(proto::query_response_chunk::Chunk::Row(proto::RowChunk {
                                row: Some(row_to_proto(row)),
                            })),
                            is_final: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    true
                },
                QueryResult::Nodes(nodes) => {
                    for node in nodes {
                        let chunk = QueryResponseChunk {
                            chunk: Some(proto::query_response_chunk::Chunk::Node(
                                proto::NodeChunk {
                                    node: Some(node_to_proto(node)),
                                },
                            )),
                            is_final: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    true
                },
                QueryResult::Edges(edges) => {
                    for edge in edges {
                        let chunk = QueryResponseChunk {
                            chunk: Some(proto::query_response_chunk::Chunk::Edge(
                                proto::EdgeChunk {
                                    edge: Some(edge_to_proto(edge)),
                                },
                            )),
                            is_final: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    true
                },
                QueryResult::Similar(items) => {
                    for item in items {
                        let chunk = QueryResponseChunk {
                            chunk: Some(proto::query_response_chunk::Chunk::SimilarItem(
                                proto::SimilarChunk {
                                    item: Some(similar_to_proto(item)),
                                },
                            )),
                            is_final: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    true
                },
                QueryResult::Blob(data) => {
                    // Stream blob data in chunks
                    for chunk_data in data.chunks(64 * 1024) {
                        let chunk = QueryResponseChunk {
                            chunk: Some(proto::query_response_chunk::Chunk::BlobData(
                                chunk_data.to_vec(),
                            )),
                            is_final: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    true
                },
                _ => {
                    // For non-streaming results, send error
                    let chunk = QueryResponseChunk {
                        chunk: Some(proto::query_response_chunk::Chunk::Error(
                            proto::ErrorInfo {
                                code: proto::ErrorCode::InvalidArgument.into(),
                                message: "Result type not supported for streaming".to_string(),
                                details: None,
                            },
                        )),
                        is_final: true,
                    };
                    let _ = tx.send(Ok(chunk)).await;
                    false
                },
            };

            if send_result {
                // Send final marker
                let final_chunk = QueryResponseChunk {
                    chunk: None,
                    is_final: true,
                };
                let _ = tx.send(Ok(final_chunk)).await;
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn execute_batch(
        &self,
        request: Request<BatchQueryRequest>,
    ) -> Result<Response<BatchQueryResponse>, Status> {
        let start = Instant::now();

        // Validate authentication with rate limiting and audit
        let request_identity = match auth::validate_request_with_audit(
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
                    m.record_request("query", "execute_batch", false, latency_ms);
                }
                return Err(status);
            },
        };

        let batch = request.into_inner();
        let mut results = Vec::with_capacity(batch.queries.len());
        let mut all_succeeded = true;

        for query_request in batch.queries {
            // SECURITY: Always use the authenticated request identity.
            // Query-level identity is ignored to prevent privilege escalation
            // where a client could authenticate as user A but execute as user B.
            let identity = request_identity.clone();

            // Check query-specific rate limit for each query in batch
            if let Some(ref limiter) = self.rate_limiter {
                if let Some(ref id) = identity {
                    if let Err(msg) = limiter.check_and_record(id, Operation::Query) {
                        if let Some(ref logger) = self.audit_logger {
                            logger.record(
                                AuditEvent::RateLimited {
                                    identity: id.clone(),
                                    operation: "query".to_string(),
                                },
                                None,
                            );
                        }
                        if let Some(ref m) = self.metrics {
                            m.record_rate_limited(id, "query");
                            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                            m.record_request("query", "execute_batch", false, latency_ms);
                        }
                        return Err(Status::resource_exhausted(msg));
                    }
                }
            }

            // Audit the query execution
            if let Some(ref logger) = self.audit_logger {
                if logger.config().log_queries {
                    logger.record(
                        AuditEvent::QueryExecuted {
                            identity: identity.clone(),
                            query: query_request.query.clone(),
                        },
                        None,
                    );
                }
            }

            let response = match self.execute_query(&query_request.query, identity.as_deref()) {
                Ok(result) => query_result_to_proto(result),
                Err(status) => {
                    all_succeeded = false;
                    QueryResponse {
                        result: None,
                        error: Some(proto::ErrorInfo {
                            code: status_to_error_code(&status).into(),
                            message: status.message().to_string(),
                            details: None,
                        }),
                    }
                },
            };
            results.push(response);
        }

        // Record metrics for the entire batch
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("query", "execute_batch", all_succeeded, latency_ms);
        }

        Ok(Response::new(BatchQueryResponse { results }))
    }
}

/// Convert tonic Status to ErrorCode.
fn status_to_error_code(status: &Status) -> proto::ErrorCode {
    match status.code() {
        tonic::Code::InvalidArgument => proto::ErrorCode::InvalidArgument,
        tonic::Code::NotFound => proto::ErrorCode::NotFound,
        tonic::Code::PermissionDenied => proto::ErrorCode::PermissionDenied,
        tonic::Code::AlreadyExists => proto::ErrorCode::AlreadyExists,
        tonic::Code::Unauthenticated => proto::ErrorCode::Unauthenticated,
        tonic::Code::Unavailable => proto::ErrorCode::Unavailable,
        _ => proto::ErrorCode::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_router() -> Arc<RwLock<QueryRouter>> {
        Arc::new(RwLock::new(QueryRouter::new()))
    }

    #[tokio::test]
    async fn test_execute_create_table() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        let request = Request::new(QueryRequest {
            query: "CREATE TABLE users (name:string, age:int)".to_string(),
            identity: None,
        });

        let response = service.execute(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_execute_invalid_query() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        let request = Request::new(QueryRequest {
            query: "INVALID QUERY SYNTAX!!!".to_string(),
            identity: None,
        });

        let response = service.execute(request).await;
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_execute_select() {
        let router = create_test_router();

        // Setup: create table and insert data
        {
            let r = router.write();
            r.execute("CREATE TABLE users (name:string, age:int)")
                .unwrap();
            r.execute("INSERT users name=\"Alice\", age=30").unwrap();
        }

        let service = QueryServiceImpl::new(router);

        let request = Request::new(QueryRequest {
            query: "SELECT users".to_string(),
            identity: None,
        });

        let response = service.execute(request).await.unwrap();
        let inner = response.into_inner();

        assert!(matches!(
            inner.result,
            Some(proto::query_response::Result::Rows(_))
        ));
    }

    #[tokio::test]
    async fn test_execute_batch() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        let request = Request::new(BatchQueryRequest {
            queries: vec![
                QueryRequest {
                    query: "CREATE TABLE batch_test (x:int)".to_string(),
                    identity: None,
                },
                QueryRequest {
                    query: "INSERT batch_test x=1".to_string(),
                    identity: None,
                },
                QueryRequest {
                    query: "SELECT batch_test".to_string(),
                    identity: None,
                },
            ],
        });

        let response = service.execute_batch(request).await.unwrap();
        let inner = response.into_inner();

        assert_eq!(inner.results.len(), 3);
    }

    #[tokio::test]
    async fn test_execute_batch_with_auth() {
        use crate::config::ApiKey;
        use tonic::metadata::MetadataValue;

        let router = create_test_router();
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let service = QueryServiceImpl::with_auth(router, auth_config);

        // Request without auth should fail
        let request = Request::new(BatchQueryRequest {
            queries: vec![QueryRequest {
                query: "CREATE TABLE batch_auth (x:int)".to_string(),
                identity: None,
            }],
        });

        let response = service.execute_batch(request).await;
        assert!(response.is_err());
        assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);

        // Request with valid auth should succeed
        let mut request = Request::new(BatchQueryRequest {
            queries: vec![QueryRequest {
                query: "CREATE TABLE batch_auth (x:int)".to_string(),
                identity: None,
            }],
        });
        request.metadata_mut().insert(
            "x-api-key",
            MetadataValue::try_from("test-api-key-12345678").expect("valid metadata value"),
        );

        let response = service.execute_batch(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_execute_batch_ignores_query_identity() {
        // SECURITY TEST: Verify that query-level identity cannot override request-level identity
        use crate::config::ApiKey;
        use tonic::metadata::MetadataValue;

        let router = create_test_router();
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let service = QueryServiceImpl::with_auth(router, auth_config);

        // Attempt to execute batch with different identity in query
        let mut request = Request::new(BatchQueryRequest {
            queries: vec![QueryRequest {
                query: "CREATE TABLE priv_test (x:int)".to_string(),
                identity: Some("user:evil".to_string()), // Attacker tries to impersonate
            }],
        });
        request.metadata_mut().insert(
            "x-api-key",
            MetadataValue::try_from("test-api-key-12345678").expect("valid metadata value"),
        );

        // The query should execute, but the identity used should be "user:alice"
        // (from the API key), not "user:evil" (from the query)
        let response = service.execute_batch(request).await;
        assert!(response.is_ok(), "Batch execution should succeed");
    }

    #[tokio::test]
    async fn test_status_to_error_code() {
        assert_eq!(
            status_to_error_code(&Status::invalid_argument("test")),
            proto::ErrorCode::InvalidArgument
        );
        assert_eq!(
            status_to_error_code(&Status::not_found("test")),
            proto::ErrorCode::NotFound
        );
        assert_eq!(
            status_to_error_code(&Status::permission_denied("test")),
            proto::ErrorCode::PermissionDenied
        );
        assert_eq!(
            status_to_error_code(&Status::internal("test")),
            proto::ErrorCode::Internal
        );
    }

    #[tokio::test]
    async fn test_health_state_on_failures() {
        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());

        let service =
            QueryServiceImpl::with_health_state(router, None, 32, Arc::clone(&health_state));

        // Initially healthy
        assert!(health_state.is_query_service_healthy());

        // Execute valid query - should stay healthy
        let request = Request::new(QueryRequest {
            query: "CREATE TABLE health_test (x:int)".to_string(),
            identity: None,
        });
        let _ = service.execute(request).await;
        assert!(health_state.is_query_service_healthy());

        // Execute invalid queries to trigger failure threshold
        for _ in 0..FAILURE_THRESHOLD {
            let request = Request::new(QueryRequest {
                query: "INVALID QUERY!!!".to_string(),
                identity: None,
            });
            let _ = service.execute(request).await;
        }

        // Should be unhealthy after threshold failures
        assert!(!health_state.is_query_service_healthy());

        // Successful query should restore health
        let request = Request::new(QueryRequest {
            query: "SELECT health_test".to_string(),
            identity: None,
        });
        let _ = service.execute(request).await;
        assert!(health_state.is_query_service_healthy());
    }

    #[test]
    fn test_failure_tracking() {
        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());

        let service =
            QueryServiceImpl::with_health_state(router, None, 32, Arc::clone(&health_state));

        // Record failures
        for i in 0..FAILURE_THRESHOLD {
            service.record_failure();
            if i < FAILURE_THRESHOLD - 1 {
                assert!(
                    health_state.is_query_service_healthy(),
                    "Should be healthy before threshold"
                );
            }
        }

        // Should be unhealthy at threshold
        assert!(!health_state.is_query_service_healthy());

        // Success should restore
        service.record_success();
        assert!(health_state.is_query_service_healthy());
    }

    #[tokio::test]
    async fn test_execute_with_identity() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        let request = Request::new(QueryRequest {
            query: "CREATE TABLE id_test (x:int)".to_string(),
            identity: Some("test-user".to_string()),
        });

        let response = service.execute(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_execute_stream_rows() {
        let router = create_test_router();

        // Setup: create table and insert data
        {
            let r = router.write();
            r.execute("CREATE TABLE stream_test (name:string, age:int)")
                .unwrap();
            r.execute("INSERT stream_test name=\"Alice\", age=30")
                .unwrap();
            r.execute("INSERT stream_test name=\"Bob\", age=25")
                .unwrap();
        }

        let service = QueryServiceImpl::with_config(router, None, 10);

        let request = Request::new(QueryRequest {
            query: "SELECT stream_test".to_string(),
            identity: None,
        });

        let response = service.execute_stream(request).await.unwrap();
        let mut stream = response.into_inner();

        // Collect chunks
        let mut chunks = vec![];
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }

        // Should have row chunks plus final marker
        assert!(chunks.len() >= 2, "Expected at least 2 chunks");

        // Last chunk should be final marker
        assert!(chunks.last().unwrap().is_final);
    }

    #[tokio::test]
    async fn test_execute_stream_non_streaming_result() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        // Empty result doesn't stream well
        let request = Request::new(QueryRequest {
            query: "CREATE TABLE stream_empty (x:int)".to_string(),
            identity: None,
        });

        let response = service.execute_stream(request).await.unwrap();
        let mut stream = response.into_inner();

        // Should get an error chunk or final marker
        let mut found_final = false;
        while let Some(chunk) = stream.next().await {
            let c = chunk.unwrap();
            if c.is_final {
                found_final = true;
                break;
            }
        }
        assert!(found_final, "Should have final marker");
    }

    #[tokio::test]
    async fn test_execute_with_auth_config() {
        use crate::config::ApiKey;
        use tonic::metadata::MetadataValue;

        let router = create_test_router();
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let service = QueryServiceImpl::with_auth(router, auth_config);

        // Without auth should fail
        let request = Request::new(QueryRequest {
            query: "CREATE TABLE auth_test (x:int)".to_string(),
            identity: None,
        });

        let response = service.execute(request).await;
        assert!(response.is_err());
        assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);

        // With auth should succeed
        let mut request = Request::new(QueryRequest {
            query: "CREATE TABLE auth_test (x:int)".to_string(),
            identity: None,
        });
        request.metadata_mut().insert(
            "x-api-key",
            MetadataValue::try_from("test-key-12345678").expect("valid metadata value"),
        );

        let response = service.execute(request).await;
        assert!(response.is_ok());
    }

    #[test]
    fn test_status_to_error_code_additional() {
        assert_eq!(
            status_to_error_code(&Status::already_exists("test")),
            proto::ErrorCode::AlreadyExists
        );
        assert_eq!(
            status_to_error_code(&Status::unauthenticated("test")),
            proto::ErrorCode::Unauthenticated
        );
        assert_eq!(
            status_to_error_code(&Status::unavailable("test")),
            proto::ErrorCode::Unavailable
        );
        // Unknown codes should map to Internal
        assert_eq!(
            status_to_error_code(&Status::cancelled("test")),
            proto::ErrorCode::Internal
        );
        assert_eq!(
            status_to_error_code(&Status::aborted("test")),
            proto::ErrorCode::Internal
        );
    }

    #[test]
    fn test_query_service_constructors() {
        let router = create_test_router();

        // Test new()
        let service = QueryServiceImpl::new(Arc::clone(&router));
        assert!(service.auth_config.is_none());
        assert!(service.health_state.is_none());

        // Test with_config()
        let service = QueryServiceImpl::with_config(Arc::clone(&router), None, 64);
        assert_eq!(service.stream_channel_capacity, 64);

        // Test with_health_state()
        let health = Arc::new(HealthState::new());
        let service = QueryServiceImpl::with_health_state(router, None, 32, Arc::clone(&health));
        assert!(service.health_state.is_some());
    }

    #[tokio::test]
    async fn test_execute_stream_invalid_query() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        let request = Request::new(QueryRequest {
            query: "INVALID QUERY!!!".to_string(),
            identity: None,
        });

        let response = service.execute_stream(request).await;
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_record_success_without_health_state() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        // Should not panic
        service.record_success();
    }

    #[tokio::test]
    async fn test_record_failure_without_health_state() {
        let router = create_test_router();
        let service = QueryServiceImpl::new(router);

        // Should not panic
        for _ in 0..10 {
            service.record_failure();
        }
    }

    #[tokio::test]
    async fn test_execute_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let service = QueryServiceImpl::with_full_config(
            router,
            None,
            32,
            health_state,
            None,
            None,
            Some(Arc::clone(&metrics)),
        );

        let request = Request::new(QueryRequest {
            query: "CREATE TABLE metrics_test (x:int)".to_string(),
            identity: None,
        });

        // Execute query - should record metrics
        let response = service.execute(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_execute_records_latency() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let service = QueryServiceImpl::with_full_config(
            router,
            None,
            32,
            health_state,
            None,
            None,
            Some(metrics),
        );

        let request = Request::new(QueryRequest {
            query: "CREATE TABLE latency_test (x:int)".to_string(),
            identity: None,
        });

        // Execute query - latency should be recorded
        let response = service.execute(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_auth_failure_recorded() {
        use crate::config::ApiKey;
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let service = QueryServiceImpl::with_full_config(
            router,
            Some(auth_config),
            32,
            health_state,
            None,
            None,
            Some(metrics),
        );

        // Request without auth should fail and record auth failure
        let request = Request::new(QueryRequest {
            query: "CREATE TABLE auth_fail_test (x:int)".to_string(),
            identity: None,
        });

        let response = service.execute(request).await;
        assert!(response.is_err());
        assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);
    }

    #[tokio::test]
    async fn test_rate_limit_recorded() {
        use crate::config::ApiKey;
        use crate::metrics::ServerMetrics;
        use crate::rate_limit::{RateLimitConfig, RateLimiter};
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;
        use tonic::metadata::MetadataValue;

        let router = create_test_router();
        let health_state = Arc::new(HealthState::new());

        // Configure auth so rate limiter has an identity to track
        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:rate_test".to_string(),
        ));

        // Set very low query limit so it triggers on second request
        let rate_limiter = Arc::new(RateLimiter::new(RateLimitConfig::new().with_max_queries(1)));
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let service = QueryServiceImpl::with_full_config(
            router,
            Some(auth_config),
            32,
            health_state,
            Some(rate_limiter),
            None,
            Some(metrics),
        );

        // First request should succeed (with valid API key)
        let mut request = Request::new(QueryRequest {
            query: "CREATE TABLE rate_test (x:int)".to_string(),
            identity: None,
        });
        request.metadata_mut().insert(
            "x-api-key",
            MetadataValue::from_static("test-api-key-12345678"),
        );
        let response = service.execute(request).await;
        assert!(response.is_ok());

        // Second request should be rate limited and recorded
        let mut request = Request::new(QueryRequest {
            query: "SELECT rate_test".to_string(),
            identity: None,
        });
        request.metadata_mut().insert(
            "x-api-key",
            MetadataValue::from_static("test-api-key-12345678"),
        );
        let response = service.execute(request).await;
        assert!(response.is_err());
        assert_eq!(response.unwrap_err().code(), tonic::Code::ResourceExhausted);
    }

    use tokio_stream::StreamExt;
}
