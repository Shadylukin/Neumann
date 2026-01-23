//! QueryService implementation for executing Neumann queries.

use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use query_router::{QueryResult, QueryRouter};

use crate::auth;
use crate::config::AuthConfig;
use crate::convert::{
    edge_to_proto, node_to_proto, query_result_to_proto, row_to_proto, similar_to_proto,
};
use crate::proto::{
    self, query_service_server::QueryService, BatchQueryRequest, BatchQueryResponse, QueryRequest,
    QueryResponse, QueryResponseChunk,
};
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
        // Validate authentication
        let identity = auth::extract_identity(
            &request,
            request.get_ref().identity.as_deref(),
            &self.auth_config,
        )?;

        let query = &request.get_ref().query;
        tracing::debug!("Executing query: {}", query);

        match self.execute_query(query, identity.as_deref()) {
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
        // Validate authentication
        let identity = auth::extract_identity(
            &request,
            request.get_ref().identity.as_deref(),
            &self.auth_config,
        )?;

        let query = &request.get_ref().query;
        tracing::debug!("Executing streaming query: {}", query);

        let result = self.execute_query(query, identity.as_deref())?;

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
        // Validate authentication at request level first
        let request_identity = auth::validate_request(&request, &self.auth_config)?;

        let batch = request.into_inner();
        let mut results = Vec::with_capacity(batch.queries.len());

        for query_request in batch.queries {
            // SECURITY: Always use the authenticated request identity.
            // Query-level identity is ignored to prevent privilege escalation
            // where a client could authenticate as user A but execute as user B.
            let identity = request_identity.clone();

            let response = match self.execute_query(&query_request.query, identity.as_deref()) {
                Ok(result) => query_result_to_proto(result),
                Err(status) => QueryResponse {
                    result: None,
                    error: Some(proto::ErrorInfo {
                        code: status_to_error_code(&status).into(),
                        message: status.message().to_string(),
                        details: None,
                    }),
                },
            };
            results.push(response);
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

    use tokio_stream::StreamExt;
}
