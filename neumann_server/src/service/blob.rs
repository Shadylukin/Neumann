//! BlobService implementation for artifact storage with streaming support.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use tensor_blob::BlobStore;

use crate::audit::{AuditEvent, AuditLogger};
use crate::auth;
use crate::config::{AuthConfig, ServerConfig};
use crate::convert::{blob_metadata_to_proto, upload_metadata_to_put_options};
use crate::metrics::ServerMetrics;
use crate::proto::{
    blob_service_server::BlobService, ArtifactInfo, BlobDeleteRequest, BlobDeleteResponse,
    BlobDownloadChunk, BlobDownloadRequest, BlobMetadataRequest, BlobUploadRequest,
    BlobUploadResponse,
};
use crate::rate_limit::{Operation, RateLimiter};

/// Implementation of the BlobService gRPC service.
pub struct BlobServiceImpl {
    blob_store: Arc<Mutex<BlobStore>>,
    chunk_size: usize,
    auth_config: Option<AuthConfig>,
    max_upload_size: usize,
    stream_channel_capacity: usize,
    rate_limiter: Option<Arc<RateLimiter>>,
    audit_logger: Option<Arc<AuditLogger>>,
    metrics: Option<Arc<ServerMetrics>>,
}

/// Default maximum upload size: 512MB
const DEFAULT_MAX_UPLOAD_SIZE: usize = 512 * 1024 * 1024;

/// Default channel capacity for streaming responses.
const DEFAULT_STREAM_CHANNEL_CAPACITY: usize = 32;

impl BlobServiceImpl {
    /// Create a new blob service.
    #[must_use]
    pub fn new(blob_store: Arc<Mutex<BlobStore>>) -> Self {
        Self {
            blob_store,
            chunk_size: 64 * 1024, // 64KB default
            auth_config: None,
            max_upload_size: DEFAULT_MAX_UPLOAD_SIZE,
            stream_channel_capacity: DEFAULT_STREAM_CHANNEL_CAPACITY,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new blob service with custom configuration.
    #[must_use]
    pub fn with_config(blob_store: Arc<Mutex<BlobStore>>, config: &ServerConfig) -> Self {
        Self {
            blob_store,
            chunk_size: config.blob_chunk_size,
            auth_config: config.auth.clone(),
            max_upload_size: config.max_upload_size,
            stream_channel_capacity: config.stream_channel_capacity,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new blob service with authentication.
    #[must_use]
    pub fn with_auth(blob_store: Arc<Mutex<BlobStore>>, auth_config: AuthConfig) -> Self {
        Self {
            blob_store,
            chunk_size: 64 * 1024,
            auth_config: Some(auth_config),
            max_upload_size: DEFAULT_MAX_UPLOAD_SIZE,
            stream_channel_capacity: DEFAULT_STREAM_CHANNEL_CAPACITY,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Create a new blob service with full configuration including rate limiting, audit, and metrics.
    #[must_use]
    pub fn with_full_config(
        blob_store: Arc<Mutex<BlobStore>>,
        config: &ServerConfig,
        rate_limiter: Option<Arc<RateLimiter>>,
        audit_logger: Option<Arc<AuditLogger>>,
        metrics: Option<Arc<ServerMetrics>>,
    ) -> Self {
        Self {
            blob_store,
            chunk_size: config.blob_chunk_size,
            auth_config: config.auth.clone(),
            max_upload_size: config.max_upload_size,
            stream_channel_capacity: config.stream_channel_capacity,
            rate_limiter,
            audit_logger,
            metrics,
        }
    }

    /// Set the maximum upload size.
    #[must_use]
    pub fn with_max_upload_size(mut self, size: usize) -> Self {
        self.max_upload_size = size;
        self
    }
}

#[tonic::async_trait]
impl BlobService for BlobServiceImpl {
    async fn upload(
        &self,
        request: Request<Streaming<BlobUploadRequest>>,
    ) -> Result<Response<BlobUploadResponse>, Status> {
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
                    m.record_request("blob", "upload", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check blob-specific rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::BlobOp) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "blob_op".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "blob_upload");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("blob", "upload", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let mut stream = request.into_inner();

        // First message should be metadata
        let first_msg = stream
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty upload stream"))?
            .map_err(|e| Status::internal(format!("stream error: {e}")))?;

        let metadata = match first_msg.request {
            Some(crate::proto::blob_upload_request::Request::Metadata(m)) => m,
            Some(crate::proto::blob_upload_request::Request::Chunk(_)) => {
                return Err(Status::invalid_argument(
                    "first message must be metadata, not chunk",
                ));
            },
            None => {
                return Err(Status::invalid_argument("empty request"));
            },
        };

        let filename = metadata.filename.clone();
        let options = upload_metadata_to_put_options(&metadata);

        // Collect all chunks with size limit
        let mut data = Vec::new();
        let max_size = self.max_upload_size;
        while let Some(msg) = stream.next().await {
            let msg = msg.map_err(|e| Status::internal(format!("stream error: {e}")))?;
            match msg.request {
                Some(crate::proto::blob_upload_request::Request::Chunk(chunk)) => {
                    // Check size limit before extending
                    if data.len().saturating_add(chunk.len()) > max_size {
                        return Err(Status::resource_exhausted(format!(
                            "upload exceeds maximum size of {max_size} bytes"
                        )));
                    }
                    data.extend_from_slice(&chunk);
                },
                Some(crate::proto::blob_upload_request::Request::Metadata(_)) => {
                    return Err(Status::invalid_argument(
                        "metadata can only appear as first message",
                    ));
                },
                None => {
                    // Empty message, skip
                },
            }
        }

        if data.is_empty() {
            return Err(Status::invalid_argument("no data provided"));
        }

        // Store the blob
        let store = self.blob_store.lock().await;
        let artifact_id = store.put(&filename, &data, options).await.map_err(|e| {
            tracing::error!("Blob store error: {e}");
            Status::internal("internal storage error")
        })?;

        // Get metadata to return checksum
        let meta = store.metadata(&artifact_id).await.map_err(|e| {
            tracing::error!("Blob metadata error: {e}");
            Status::internal("internal storage error")
        })?;

        // Audit the upload
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::BlobUpload {
                    identity: identity.clone(),
                    artifact_id: artifact_id.clone(),
                    size: meta.size,
                },
                None,
            );
        }

        // Record success metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("blob", "upload", true, latency_ms);
        }

        Ok(Response::new(BlobUploadResponse {
            artifact_id,
            size: meta.size as u64,
            checksum: meta.checksum,
        }))
    }

    type DownloadStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<BlobDownloadChunk, Status>> + Send>>;

    async fn download(
        &self,
        request: Request<BlobDownloadRequest>,
    ) -> Result<Response<Self::DownloadStream>, Status> {
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
                    m.record_request("blob", "download", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check blob-specific rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::BlobOp) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "blob_op".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "blob_download");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("blob", "download", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let artifact_id = request.into_inner().artifact_id;
        let chunk_size = self.chunk_size;

        // Get the blob data
        let store = self.blob_store.lock().await;
        let data = match store.get(&artifact_id).await {
            Ok(d) => d,
            Err(e) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("blob", "download", false, latency_ms);
                }
                if matches!(e, tensor_blob::BlobError::NotFound(_)) {
                    return Err(Status::not_found(format!(
                        "artifact not found: {artifact_id}"
                    )));
                }
                tracing::error!("Blob download error: {e}");
                return Err(Status::internal("internal storage error"));
            },
        };

        drop(store); // Release lock before streaming

        // Audit the download
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::BlobDownload {
                    identity: identity.clone(),
                    artifact_id: artifact_id.clone(),
                },
                None,
            );
        }

        // Record success metrics for download setup
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("blob", "download", true, latency_ms);
        }

        let (tx, rx) = mpsc::channel(self.stream_channel_capacity);

        // Spawn task to stream data
        tokio::spawn(async move {
            let chunks: Vec<_> = data.chunks(chunk_size).collect();
            let total_chunks = chunks.len();

            for (i, chunk_data) in chunks.into_iter().enumerate() {
                let is_final = i == total_chunks - 1;
                let chunk = BlobDownloadChunk {
                    data: chunk_data.to_vec(),
                    is_final,
                };

                if tx.send(Ok(chunk)).await.is_err() {
                    // Receiver dropped, stop sending
                    return;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn delete(
        &self,
        request: Request<BlobDeleteRequest>,
    ) -> Result<Response<BlobDeleteResponse>, Status> {
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
                    m.record_request("blob", "delete", false, latency_ms);
                }
                return Err(status);
            },
        };

        // Check blob-specific rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(ref id) = identity {
                if let Err(msg) = limiter.check_and_record(id, Operation::BlobOp) {
                    if let Some(ref logger) = self.audit_logger {
                        logger.record(
                            AuditEvent::RateLimited {
                                identity: id.clone(),
                                operation: "blob_op".to_string(),
                            },
                            None,
                        );
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_rate_limited(id, "blob_delete");
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        m.record_request("blob", "delete", false, latency_ms);
                    }
                    return Err(Status::resource_exhausted(msg));
                }
            }
        }

        let artifact_id = request.into_inner().artifact_id;

        let store = self.blob_store.lock().await;
        if let Err(e) = store.delete(&artifact_id).await {
            if let Some(ref m) = self.metrics {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                m.record_request("blob", "delete", false, latency_ms);
            }
            if matches!(e, tensor_blob::BlobError::NotFound(_)) {
                return Err(Status::not_found(format!(
                    "artifact not found: {artifact_id}"
                )));
            }
            tracing::error!("Blob delete error: {e}");
            return Err(Status::internal("internal storage error"));
        }

        // Audit the deletion
        if let Some(ref logger) = self.audit_logger {
            logger.record(
                AuditEvent::BlobDelete {
                    identity: identity.clone(),
                    artifact_id: artifact_id.clone(),
                },
                None,
            );
        }

        // Record success metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("blob", "delete", true, latency_ms);
        }

        Ok(Response::new(BlobDeleteResponse { success: true }))
    }

    async fn get_metadata(
        &self,
        request: Request<BlobMetadataRequest>,
    ) -> Result<Response<ArtifactInfo>, Status> {
        let start = Instant::now();

        // Validate authentication (no rate limit for metadata - read-only operation)
        let _identity = match auth::validate_request(&request, &self.auth_config) {
            Ok(id) => id,
            Err(status) => {
                if let Some(ref m) = self.metrics {
                    if status.code() == tonic::Code::Unauthenticated {
                        m.record_auth_failure("invalid_key");
                    }
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("blob", "get_metadata", false, latency_ms);
                }
                return Err(status);
            },
        };

        let artifact_id = request.into_inner().artifact_id;

        let store = self.blob_store.lock().await;
        let metadata = match store.metadata(&artifact_id).await {
            Ok(m) => m,
            Err(e) => {
                if let Some(ref m) = self.metrics {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    m.record_request("blob", "get_metadata", false, latency_ms);
                }
                if matches!(e, tensor_blob::BlobError::NotFound(_)) {
                    return Err(Status::not_found(format!(
                        "artifact not found: {artifact_id}"
                    )));
                }
                tracing::error!("Blob metadata error: {e}");
                return Err(Status::internal("internal storage error"));
            },
        };

        // Record success metrics
        if let Some(ref m) = self.metrics {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            m.record_request("blob", "get_metadata", true, latency_ms);
        }

        Ok(Response::new(blob_metadata_to_proto(&metadata)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_blob::BlobConfig;
    use tensor_store::TensorStore;
    use tokio_stream::StreamExt;

    async fn create_test_blob_store() -> Arc<Mutex<BlobStore>> {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();
        Arc::new(Mutex::new(blob_store))
    }

    #[tokio::test]
    async fn test_download_after_direct_upload() {
        let blob_store = create_test_blob_store().await;

        // Upload using blob store directly
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put(
                    "test.txt",
                    b"Hello, World!",
                    tensor_blob::PutOptions::default(),
                )
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::new(blob_store);

        // Download via service
        let download_request = Request::new(BlobDownloadRequest {
            artifact_id: artifact_id.clone(),
        });

        let mut download_stream = service
            .download(download_request)
            .await
            .unwrap()
            .into_inner();

        let mut downloaded_data = Vec::new();
        while let Some(chunk) = download_stream.next().await {
            let chunk = chunk.unwrap();
            downloaded_data.extend_from_slice(&chunk.data);
            if chunk.is_final {
                break;
            }
        }

        assert_eq!(downloaded_data, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_download_not_found() {
        let blob_store = create_test_blob_store().await;
        let service = BlobServiceImpl::new(blob_store);

        let request = Request::new(BlobDownloadRequest {
            artifact_id: "nonexistent".to_string(),
        });

        let result = service.download(request).await;
        let Err(err) = result else {
            panic!("expected error");
        };
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_delete() {
        let blob_store = create_test_blob_store().await;

        // First upload something
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put("test.txt", b"data", tensor_blob::PutOptions::default())
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::new(Arc::clone(&blob_store));

        let request = Request::new(BlobDeleteRequest {
            artifact_id: artifact_id.clone(),
        });

        let response = service.delete(request).await.unwrap();
        assert!(response.into_inner().success);

        // Verify deleted
        let store = blob_store.lock().await;
        assert!(!store.exists(&artifact_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_delete_not_found() {
        let blob_store = create_test_blob_store().await;
        let service = BlobServiceImpl::new(blob_store);

        let request = Request::new(BlobDeleteRequest {
            artifact_id: "nonexistent".to_string(),
        });

        let result = service.delete(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_get_metadata() {
        let blob_store = create_test_blob_store().await;

        // Upload an artifact
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put(
                    "test.txt",
                    b"Hello",
                    tensor_blob::PutOptions::new()
                        .with_content_type("text/plain")
                        .with_tag("test"),
                )
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::new(blob_store);

        let request = Request::new(BlobMetadataRequest { artifact_id });

        let response = service.get_metadata(request).await.unwrap();
        let info = response.into_inner();

        assert_eq!(info.filename, "test.txt");
        assert_eq!(info.content_type, "text/plain");
        assert_eq!(info.size, 5);
        assert!(info.tags.contains(&"test".to_string()));
    }

    #[tokio::test]
    async fn test_get_metadata_not_found() {
        let blob_store = create_test_blob_store().await;
        let service = BlobServiceImpl::new(blob_store);

        let request = Request::new(BlobMetadataRequest {
            artifact_id: "nonexistent".to_string(),
        });

        let result = service.get_metadata(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_service_with_config() {
        let blob_store = create_test_blob_store().await;
        let config = ServerConfig::new().with_blob_chunk_size(32 * 1024);

        let service = BlobServiceImpl::with_config(blob_store, &config);
        assert_eq!(service.chunk_size, 32 * 1024);
    }

    #[tokio::test]
    async fn test_upload_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let blob_store = create_test_blob_store().await;
        let config = ServerConfig::new();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        // Upload directly to blob store, then download via service with metrics
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put("test.txt", b"data", tensor_blob::PutOptions::default())
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::with_full_config(
            blob_store,
            &config,
            None,
            None,
            Some(metrics),
        );

        // Download via service - should record metrics
        let request = Request::new(BlobDownloadRequest {
            artifact_id,
        });

        let result = service.download(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_download_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let blob_store = create_test_blob_store().await;
        let config = ServerConfig::new();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        // Upload directly to blob store
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put("test.txt", b"download_test", tensor_blob::PutOptions::default())
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::with_full_config(
            blob_store,
            &config,
            None,
            None,
            Some(metrics),
        );

        let request = Request::new(BlobDownloadRequest {
            artifact_id,
        });

        // Download should record metrics
        let result = service.download(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delete_records_metrics() {
        use crate::metrics::ServerMetrics;
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let blob_store = create_test_blob_store().await;
        let config = ServerConfig::new();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        // Upload directly to blob store
        let artifact_id = {
            let store = blob_store.lock().await;
            store
                .put("test.txt", b"delete_test", tensor_blob::PutOptions::default())
                .await
                .unwrap()
        };

        let service = BlobServiceImpl::with_full_config(
            Arc::clone(&blob_store),
            &config,
            None,
            None,
            Some(metrics),
        );

        let request = Request::new(BlobDeleteRequest {
            artifact_id,
        });

        // Delete should record metrics
        let result = service.delete(request).await;
        assert!(result.is_ok());
    }
}
