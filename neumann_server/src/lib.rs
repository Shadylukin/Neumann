//! Neumann gRPC Server
//!
//! This crate provides a gRPC server that exposes the Neumann database via
//! the `QueryRouter`. It supports:
//!
//! - Query execution with streaming results
//! - Blob storage with streaming upload/download
//! - Health checks for service monitoring
//! - Optional TLS and API key authentication
//!
//! # Example
//!
//! ```ignore
//! use neumann_server::{NeumannServer, ServerConfig};
//! use query_router::QueryRouter;
//! use std::sync::Arc;
//! use parking_lot::RwLock;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let router = Arc::new(RwLock::new(QueryRouter::new()));
//!     let config = ServerConfig::default();
//!
//!     let server = NeumannServer::new(router, config);
//!     server.serve().await?;
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![deny(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    rustdoc::broken_intra_doc_links
)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::result_large_err)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::future_not_send)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::use_self)]

pub mod audit;
pub mod auth;
pub mod config;
pub mod convert;
pub mod correlation;
pub mod error;
pub mod metrics;
pub mod rate_limit;
pub mod service;
pub mod shutdown;
pub mod signals;
pub mod tls_loader;

/// Generated protobuf types.
#[allow(missing_docs)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
pub mod proto {
    tonic::include_proto!("neumann.v1");

    /// File descriptor set for reflection service.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("neumann_descriptor");
}

use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::Mutex;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic_web::GrpcWebLayer;

use query_router::QueryRouter;
use tensor_blob::{BlobConfig, BlobStore};
use tensor_store::TensorStore;

pub use audit::{AuditConfig, AuditEntry, AuditEvent, AuditLogger};
pub use config::{AuthConfig, ServerConfig, TlsConfig};
pub use correlation::{extract_or_generate, request_span, RequestSpan, TRACE_ID_HEADER};
pub use error::{Result, ServerError};
pub use metrics::{init_metrics, MetricsConfig, MetricsHandle, ServerMetrics};
pub use rate_limit::{Operation, RateLimitConfig, RateLimiter};
pub use service::{BlobServiceImpl, HealthServiceImpl, HealthState, QueryServiceImpl};
pub use shutdown::{ShutdownConfig, ShutdownManager};
pub use tls_loader::TlsLoader;

use proto::blob_service_server::BlobServiceServer;
use proto::health_server::HealthServer;
use proto::query_service_server::QueryServiceServer;

/// The main Neumann gRPC server.
pub struct NeumannServer {
    router: Arc<RwLock<QueryRouter>>,
    blob_store: Option<Arc<Mutex<BlobStore>>>,
    config: ServerConfig,
    rate_limiter: Option<Arc<RateLimiter>>,
    audit_logger: Option<Arc<AuditLogger>>,
    metrics: Option<Arc<ServerMetrics>>,
}

impl NeumannServer {
    /// Create a new server with the given router and configuration.
    #[must_use]
    pub fn new(router: Arc<RwLock<QueryRouter>>, config: ServerConfig) -> Self {
        let rate_limiter = config
            .rate_limit
            .as_ref()
            .map(|c| Arc::new(RateLimiter::new(c.clone())));

        let audit_logger = config
            .audit
            .as_ref()
            .map(|c| Arc::new(AuditLogger::new(c.clone())));

        Self {
            router,
            blob_store: None,
            config,
            rate_limiter,
            audit_logger,
            metrics: None,
        }
    }

    /// Create a new server with shared storage for all engines.
    ///
    /// This creates a `QueryRouter` with shared `TensorStore` and initializes
    /// blob storage using the same store.
    ///
    /// # Errors
    ///
    /// Returns an error if blob store initialization fails.
    pub async fn with_shared_storage(config: ServerConfig) -> Result<Self> {
        let store = TensorStore::new();
        let router = Arc::new(RwLock::new(QueryRouter::with_shared_store(store.clone())));

        let blob_store = BlobStore::new(store, BlobConfig::default())
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;

        let rate_limiter = config
            .rate_limit
            .as_ref()
            .map(|c| Arc::new(RateLimiter::new(c.clone())));

        let audit_logger = config
            .audit
            .as_ref()
            .map(|c| Arc::new(AuditLogger::new(c.clone())));

        Ok(Self {
            router,
            blob_store: Some(Arc::new(Mutex::new(blob_store))),
            config,
            rate_limiter,
            audit_logger,
            metrics: None,
        })
    }

    /// Set the blob store for blob service support.
    #[must_use]
    pub fn with_blob_store(mut self, blob_store: Arc<Mutex<BlobStore>>) -> Self {
        self.blob_store = Some(blob_store);
        self
    }

    /// Set the metrics for the server.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<ServerMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Get a reference to the router.
    #[must_use]
    pub const fn router(&self) -> &Arc<RwLock<QueryRouter>> {
        &self.router
    }

    /// Load TLS configuration from files.
    fn load_tls_config(tls: &TlsConfig) -> Result<ServerTlsConfig> {
        let cert = std::fs::read(&tls.cert_path).map_err(|e| {
            ServerError::Config(format!(
                "failed to read certificate file {}: {e}",
                tls.cert_path.display()
            ))
        })?;
        let key = std::fs::read(&tls.key_path).map_err(|e| {
            ServerError::Config(format!(
                "failed to read key file {}: {e}",
                tls.key_path.display()
            ))
        })?;

        let identity = Identity::from_pem(&cert, &key);

        let mut tls_config = ServerTlsConfig::new().identity(identity);

        if let Some(ref ca_path) = tls.ca_cert_path {
            let ca_cert = std::fs::read(ca_path).map_err(|e| {
                ServerError::Config(format!(
                    "failed to read CA certificate file {}: {e}",
                    ca_path.display()
                ))
            })?;
            let ca = tonic::transport::Certificate::from_pem(ca_cert);
            tls_config = tls_config.client_ca_root(ca);
        }

        Ok(tls_config)
    }

    /// Start the gRPC server.
    ///
    /// This method blocks until the server is shut down.
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start or encounters a runtime error.
    pub async fn serve(self) -> Result<()> {
        self.config.validate()?;

        let addr = self.config.bind_addr;
        let tls_enabled = self.config.tls.is_some();

        if tls_enabled {
            tracing::info!("Starting Neumann gRPC server with TLS on {}", addr);
        } else {
            tracing::info!("Starting Neumann gRPC server on {}", addr);
        }

        // Create shared health state
        let health_state = Arc::new(HealthState::new());

        // Create services with configuration, health monitoring, rate limiting, audit, and metrics
        let query_service = QueryServiceImpl::with_full_config(
            Arc::clone(&self.router),
            self.config.auth.clone(),
            self.config.stream_channel_capacity,
            Arc::clone(&health_state),
            self.rate_limiter.clone(),
            self.audit_logger.clone(),
            self.metrics.clone(),
        );

        let health_service = HealthServiceImpl::with_state(Arc::clone(&health_state));

        // Build services router
        let query_svc = QueryServiceServer::new(query_service);
        let health_svc = HealthServer::new(health_service);

        // Load TLS configuration if enabled
        let tls_config = if let Some(ref tls) = self.config.tls {
            Some(Self::load_tls_config(tls)?)
        } else {
            None
        };

        // Build server with optional TLS
        let mut builder = if let Some(tls_cfg) = tls_config {
            Server::builder().tls_config(tls_cfg)?
        } else {
            Server::builder()
        };

        // Apply gRPC-web layer if enabled
        if self.config.enable_grpc_web {
            // Note: gRPC-web requires HTTP/1 support
            builder = builder.accept_http1(true);
        }

        // Build reflection service if enabled
        let reflection_svc = if self.config.enable_reflection {
            tracing::info!("Reflection service enabled");
            Some(
                tonic_reflection::server::Builder::configure()
                    .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
                    .build_v1()
                    .map_err(|e| {
                        ServerError::Internal(format!("Failed to build reflection service: {e}"))
                    })?,
            )
        } else {
            None
        };

        // Build blob service if store is available
        let blob_svc = self.blob_store.map(|store| {
            let blob_service = BlobServiceImpl::with_full_config(
                store,
                &self.config,
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            tracing::info!("Blob service enabled");
            BlobServiceServer::new(blob_service)
        });

        // Add services and start server
        if self.config.enable_grpc_web {
            let layer = GrpcWebLayer::new();
            let mut router = builder
                .layer(layer)
                .add_service(query_svc)
                .add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            router.serve(addr).await?;
        } else {
            let mut router = builder.add_service(query_svc).add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            router.serve(addr).await?;
        }

        Ok(())
    }

    /// Start the server with graceful shutdown support.
    ///
    /// The server will shut down when the provided future completes. If shutdown
    /// configuration is provided, the server will wait for in-flight requests to
    /// drain before fully shutting down, up to the configured timeout.
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start or encounters a runtime error.
    pub async fn serve_with_shutdown<F>(self, shutdown: F) -> Result<()>
    where
        F: std::future::Future<Output = ()> + Send,
    {
        self.config.validate()?;

        let addr = self.config.bind_addr;
        let tls_enabled = self.config.tls.is_some();

        if tls_enabled {
            tracing::info!("Starting Neumann gRPC server with TLS on {}", addr);
        } else {
            tracing::info!("Starting Neumann gRPC server on {}", addr);
        }

        // Create shared health state
        let health_state = Arc::new(HealthState::new());

        // Create shutdown manager if configured
        let shutdown_manager = self
            .config
            .shutdown
            .as_ref()
            .map(|cfg| Arc::new(ShutdownManager::new(cfg.clone(), Arc::clone(&health_state))));

        // Create services with configuration, health monitoring, rate limiting, audit, and metrics
        let query_service = QueryServiceImpl::with_full_config(
            Arc::clone(&self.router),
            self.config.auth.clone(),
            self.config.stream_channel_capacity,
            Arc::clone(&health_state),
            self.rate_limiter.clone(),
            self.audit_logger.clone(),
            self.metrics.clone(),
        );

        let health_service = HealthServiceImpl::with_state(Arc::clone(&health_state));

        let query_svc = QueryServiceServer::new(query_service);
        let health_svc = HealthServer::new(health_service);

        // Load TLS configuration if enabled and register SIGHUP handler
        let tls_config = if let Some(ref tls) = self.config.tls {
            let tls_loader = Arc::new(TlsLoader::new(tls.clone())?);
            let tls_config = tls_loader.load()?;

            // Register SIGHUP handler for certificate reloading
            signals::register_sighup_handler(tls_loader);

            Some(tls_config)
        } else {
            None
        };

        // Build server with optional TLS
        let mut builder = if let Some(tls_cfg) = tls_config {
            Server::builder().tls_config(tls_cfg)?
        } else {
            Server::builder()
        };

        // Apply gRPC-web layer if enabled
        if self.config.enable_grpc_web {
            builder = builder.accept_http1(true);
        }

        // Build reflection service if enabled
        let reflection_svc = if self.config.enable_reflection {
            tracing::info!("Reflection service enabled");
            Some(
                tonic_reflection::server::Builder::configure()
                    .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
                    .build_v1()
                    .map_err(|e| {
                        ServerError::Internal(format!("Failed to build reflection service: {e}"))
                    })?,
            )
        } else {
            None
        };

        // Build blob service if store is available
        let blob_svc = self.blob_store.map(|store| {
            let blob_service = BlobServiceImpl::with_full_config(
                store,
                &self.config,
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            BlobServiceServer::new(blob_service)
        });

        // Create drain-aware shutdown future
        let shutdown_manager_clone = shutdown_manager.clone();
        let health_state_clone = Arc::clone(&health_state);
        let drain_future = async move {
            shutdown.await;

            if let Some(ref mgr) = shutdown_manager_clone {
                mgr.trigger_shutdown();

                let drained = mgr.wait_for_drain().await;

                if !drained {
                    tracing::warn!(
                        remaining_streams = mgr.active_count(),
                        "Drain timeout reached, forcing shutdown"
                    );
                }

                // Wait grace period before final shutdown
                tokio::time::sleep(mgr.config().grace_period).await;
            } else {
                // No shutdown config, just mark as draining
                health_state_clone.set_draining(true);
            }
        };

        // Add services and start server
        if self.config.enable_grpc_web {
            let layer = GrpcWebLayer::new();
            let mut router = builder
                .layer(layer)
                .add_service(query_svc)
                .add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            router.serve_with_shutdown(addr, drain_future).await?;
        } else {
            let mut router = builder.add_service(query_svc).add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            router.serve_with_shutdown(addr, drain_future).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let server = NeumannServer::new(router, config);

        assert!(server.blob_store.is_none());
    }

    #[tokio::test]
    async fn test_server_with_shared_storage() {
        let config = ServerConfig::default();
        let server = NeumannServer::with_shared_storage(config).await.unwrap();

        assert!(server.blob_store.is_some());
    }

    #[test]
    fn test_server_with_blob_store() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();

        // We can't easily create a BlobStore synchronously, so just test the builder pattern
        let server = NeumannServer::new(router, config);
        assert!(server.blob_store.is_none());
    }

    #[test]
    fn test_router_access() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let server = NeumannServer::new(Arc::clone(&router), config);

        // Should be able to access router through server
        let server_router = server.router();
        assert!(Arc::ptr_eq(&router, server_router));
    }

    #[tokio::test]
    async fn test_server_with_blob_store_builder() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let store = TensorStore::new();

        let blob_store = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("should create blob store");

        let server =
            NeumannServer::new(router, config).with_blob_store(Arc::new(Mutex::new(blob_store)));

        assert!(server.blob_store.is_some());
    }

    #[test]
    fn test_load_tls_config_cert_not_found() {
        let tls = TlsConfig {
            cert_path: std::path::PathBuf::from("/nonexistent/cert.pem"),
            key_path: std::path::PathBuf::from("/nonexistent/key.pem"),
            ca_cert_path: None,
            require_client_cert: false,
        };

        let result = NeumannServer::load_tls_config(&tls);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ServerError::Config(_)));
        assert!(err.to_string().contains("cert"));
    }

    #[test]
    fn test_load_tls_config_key_not_found() {
        use std::io::Write;

        // Create a temp cert file
        let temp_dir = std::env::temp_dir();
        let cert_path = temp_dir.join("test_cert.pem");
        let mut cert_file = std::fs::File::create(&cert_path).unwrap();
        cert_file.write_all(b"dummy cert content").unwrap();

        let tls = TlsConfig {
            cert_path,
            key_path: std::path::PathBuf::from("/nonexistent/key.pem"),
            ca_cert_path: None,
            require_client_cert: false,
        };

        let result = NeumannServer::load_tls_config(&tls);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ServerError::Config(_)));
        assert!(err.to_string().contains("key"));

        // Cleanup
        let _ = std::fs::remove_file(tls.cert_path);
    }

    #[test]
    fn test_load_tls_config_ca_not_found() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let cert_path = temp_dir.join("test_cert2.pem");
        let key_path = temp_dir.join("test_key2.pem");

        let mut cert_file = std::fs::File::create(&cert_path).unwrap();
        cert_file.write_all(b"dummy cert content").unwrap();

        let mut key_file = std::fs::File::create(&key_path).unwrap();
        key_file.write_all(b"dummy key content").unwrap();

        let tls = TlsConfig {
            cert_path: cert_path.clone(),
            key_path: key_path.clone(),
            ca_cert_path: Some(std::path::PathBuf::from("/nonexistent/ca.pem")),
            require_client_cert: false,
        };

        let result = NeumannServer::load_tls_config(&tls);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ServerError::Config(_)));
        assert!(err.to_string().contains("CA"));

        // Cleanup
        let _ = std::fs::remove_file(cert_path);
        let _ = std::fs::remove_file(key_path);
    }

    #[tokio::test]
    async fn test_serve_with_invalid_config() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();

        // Add invalid auth config
        config.auth = Some(AuthConfig::new().with_anonymous(false));
        // No API keys but anonymous disabled - should fail validation

        let server = NeumannServer::new(router, config);
        let result = server.serve().await;

        assert!(result.is_err());
    }

    #[test]
    fn test_server_config_default_values() {
        let config = ServerConfig::default();

        assert_eq!(config.bind_addr.port(), 9200);
        assert!(config.tls.is_none());
        assert!(config.auth.is_none());
        assert!(config.enable_grpc_web);
        assert!(config.enable_reflection);
        assert!(config.stream_channel_capacity > 0);
    }

    #[test]
    fn test_server_with_metrics() {
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let server = NeumannServer::new(router, config).with_metrics(metrics);

        assert!(server.metrics.is_some());
    }

    #[test]
    fn test_metrics_passed_to_services() {
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        // Create server with metrics
        let server = NeumannServer::new(router, config).with_metrics(Arc::clone(&metrics));

        // Verify metrics are set
        assert!(server.metrics.is_some());

        // The metrics Arc should be the same one we passed in
        let stored = server.metrics.as_ref().unwrap();
        assert!(Arc::ptr_eq(stored, &metrics));
    }
}
