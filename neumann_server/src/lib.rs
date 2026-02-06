// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
pub mod gamification;
pub mod memory;
pub mod metrics;
pub mod rate_limit;
pub mod rest;
pub mod service;
pub mod shutdown;
pub mod signals;
pub mod tls_loader;
pub mod web;

/// Generated protobuf types.
#[allow(missing_docs)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
pub mod proto {
    tonic::include_proto!("neumann.v1");

    /// Vector service protobuf types.
    pub mod vector {
        tonic::include_proto!("neumann.vector.v1");
    }

    /// File descriptor set for reflection service.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("neumann_descriptor");
}

use std::sync::Arc;

use parking_lot::RwLock;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic_web::GrpcWebLayer;

use graph_engine::GraphEngine;
use query_router::QueryRouter;
use relational_engine::RelationalEngine;
use tensor_blob::{BlobConfig, BlobStore};
use tensor_store::TensorStore;

pub use audit::{AuditConfig, AuditEntry, AuditEvent, AuditLogger};
pub use config::{AuthConfig, ServerConfig, TlsConfig};
pub use correlation::{extract_or_generate, request_span, RequestSpan, TRACE_ID_HEADER};
pub use error::{sanitize_error, sanitize_internal_error, Result, ServerError};
pub use memory::{MemoryBudgetConfig, MemoryTracker};
pub use metrics::{init_metrics, MetricsConfig, MetricsHandle, ServerMetrics};
pub use rate_limit::{Operation, RateLimitConfig, RateLimiter};
pub use rest::{RestConfig, VectorApiContext};
pub use service::{
    BlobServiceImpl, CollectionsServiceImpl, HealthServiceImpl, HealthState, PointsServiceImpl,
    QueryServiceImpl,
};
pub use shutdown::{ShutdownConfig, ShutdownManager};
pub use tls_loader::TlsLoader;
pub use web::{AdminContext, NavItem};

use proto::blob_service_server::BlobServiceServer;
use proto::health_server::HealthServer;
use proto::query_service_server::QueryServiceServer;
use proto::vector::collections_service_server::CollectionsServiceServer;
use proto::vector::points_service_server::PointsServiceServer;

use vector_engine::VectorEngine;

/// The main Neumann gRPC server.
pub struct NeumannServer {
    router: Arc<RwLock<QueryRouter>>,
    blob_store: Option<Arc<Mutex<BlobStore>>>,
    relational_engine: Option<Arc<RelationalEngine>>,
    vector_engine: Option<Arc<VectorEngine>>,
    graph_engine: Option<Arc<GraphEngine>>,
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
            relational_engine: None,
            vector_engine: None,
            graph_engine: None,
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
            relational_engine: None,
            vector_engine: None,
            graph_engine: None,
            config,
            rate_limiter,
            audit_logger,
            metrics: None,
        })
    }

    /// Set the relational engine for web admin UI.
    #[must_use]
    pub fn with_relational_engine(mut self, relational_engine: Arc<RelationalEngine>) -> Self {
        self.relational_engine = Some(relational_engine);
        self
    }

    /// Set the vector engine for vector services.
    #[must_use]
    pub fn with_vector_engine(mut self, vector_engine: Arc<VectorEngine>) -> Self {
        self.vector_engine = Some(vector_engine);
        self
    }

    /// Set the graph engine for web admin UI.
    #[must_use]
    pub fn with_graph_engine(mut self, graph_engine: Arc<GraphEngine>) -> Self {
        self.graph_engine = Some(graph_engine);
        self
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

            // When CA is set but require_client_cert is false, make client auth optional.
            // When require_client_cert is true (or not set), client cert is required (default).
            if !tls.require_client_cert {
                tls_config = tls_config.client_auth_optional(true);
            }
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

        // Apply HTTP/2 settings
        if let Some(max_streams) = self.config.max_concurrent_streams_per_connection {
            builder = builder.http2_max_pending_accept_reset_streams(Some(max_streams as usize));
        }
        if let Some(window_size) = self.config.initial_window_size {
            builder = builder.initial_stream_window_size(window_size);
        }
        if let Some(conn_window) = self.config.initial_connection_window_size {
            builder = builder.initial_connection_window_size(conn_window);
        }
        if let Some(limit) = self.config.max_concurrent_connections {
            builder = builder.concurrency_limit_per_connection(limit);
        }
        if let Some(timeout) = self.config.request_timeout {
            builder = builder.timeout(timeout);
        }

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

        // Build vector services if engine is available
        let (points_svc, collections_svc) = if let Some(ref engine) = self.vector_engine {
            let points_service = PointsServiceImpl::with_full_config(
                Arc::clone(engine),
                self.config.auth.clone(),
                Some(Arc::clone(&health_state)),
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            let collections_service = CollectionsServiceImpl::with_full_config(
                Arc::clone(engine),
                self.config.auth.clone(),
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            tracing::info!("Vector services enabled");
            (
                Some(PointsServiceServer::new(points_service)),
                Some(CollectionsServiceServer::new(collections_service)),
            )
        } else {
            (None, None)
        };

        // Build REST server if configured
        let rest_handle = if let (Some(rest_addr), Some(ref engine)) =
            (self.config.rest_addr, &self.vector_engine)
        {
            let rest_ctx = Arc::new(
                VectorApiContext::new(Arc::clone(engine))
                    .with_auth(self.config.auth.clone())
                    .with_rate_limiter(self.rate_limiter.clone())
                    .with_audit_logger(self.audit_logger.clone())
                    .with_metrics(self.metrics.clone()),
            );
            let rest_router = rest::router(rest_ctx);
            let listener = TcpListener::bind(rest_addr).await.map_err(|e| {
                ServerError::Internal(format!("failed to bind REST server to {rest_addr}: {e}"))
            })?;
            tracing::info!("REST API enabled on {}", rest_addr);
            Some(tokio::spawn(async move {
                axum::serve(listener, rest_router)
                    .await
                    .map_err(|e| ServerError::Internal(format!("REST server error: {e}")))
            }))
        } else {
            None
        };

        // Build Web admin server if configured and all engines are available
        let web_handle =
            if let (Some(web_addr), Some(ref relational), Some(ref vector), Some(ref graph)) = (
                self.config.web_addr,
                &self.relational_engine,
                &self.vector_engine,
                &self.graph_engine,
            ) {
                let web_ctx = Arc::new(
                    web::AdminContext::new(
                        Arc::clone(relational),
                        Arc::clone(vector),
                        Arc::clone(graph),
                    )
                    .with_auth(self.config.auth.clone())
                    .with_metrics(self.metrics.clone()),
                );
                let web_router = web::router(web_ctx);
                let listener = TcpListener::bind(web_addr).await.map_err(|e| {
                    ServerError::Internal(format!("failed to bind Web server to {web_addr}: {e}"))
                })?;
                tracing::info!("Web admin UI enabled on {}", web_addr);
                Some(tokio::spawn(async move {
                    axum::serve(listener, web_router)
                        .await
                        .map_err(|e| ServerError::Internal(format!("Web server error: {e}")))
                }))
            } else {
                None
            };

        // Add services and start server
        let grpc_future = async {
            if self.config.enable_grpc_web {
                let layer = GrpcWebLayer::new();
                let mut router = builder
                    .layer(layer)
                    .add_service(query_svc)
                    .add_service(health_svc);

                if let Some(blob) = blob_svc {
                    router = router.add_service(blob);
                }
                if let Some(points) = points_svc {
                    router = router.add_service(points);
                }
                if let Some(collections) = collections_svc {
                    router = router.add_service(collections);
                }
                if let Some(refl) = reflection_svc {
                    router = router.add_service(refl);
                }

                router.serve(addr).await
            } else {
                let mut router = builder.add_service(query_svc).add_service(health_svc);

                if let Some(blob) = blob_svc {
                    router = router.add_service(blob);
                }
                if let Some(points) = points_svc {
                    router = router.add_service(points);
                }
                if let Some(collections) = collections_svc {
                    router = router.add_service(collections);
                }
                if let Some(refl) = reflection_svc {
                    router = router.add_service(refl);
                }

                router.serve(addr).await
            }
        };

        // Run gRPC server, REST server, and Web server if configured
        match (rest_handle, web_handle) {
            (Some(rest), Some(web)) => {
                tokio::select! {
                    result = grpc_future => result?,
                    result = rest => result.map_err(|e| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    result = web => result.map_err(|e| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                }
            },
            (Some(rest), None) => {
                tokio::select! {
                    result = grpc_future => result?,
                    result = rest => result.map_err(|e| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                }
            },
            (None, Some(web)) => {
                tokio::select! {
                    result = grpc_future => result?,
                    result = web => result.map_err(|e| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                }
            },
            (None, None) => {
                grpc_future.await?;
            },
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

        // Apply HTTP/2 settings
        if let Some(max_streams) = self.config.max_concurrent_streams_per_connection {
            builder = builder.http2_max_pending_accept_reset_streams(Some(max_streams as usize));
        }
        if let Some(window_size) = self.config.initial_window_size {
            builder = builder.initial_stream_window_size(window_size);
        }
        if let Some(conn_window) = self.config.initial_connection_window_size {
            builder = builder.initial_connection_window_size(conn_window);
        }
        if let Some(limit) = self.config.max_concurrent_connections {
            builder = builder.concurrency_limit_per_connection(limit);
        }
        if let Some(timeout) = self.config.request_timeout {
            builder = builder.timeout(timeout);
        }

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

        // Build vector services if engine is available
        let (points_svc, collections_svc) = if let Some(ref engine) = self.vector_engine {
            let points_service = PointsServiceImpl::with_full_config(
                Arc::clone(engine),
                self.config.auth.clone(),
                Some(Arc::clone(&health_state)),
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            let collections_service = CollectionsServiceImpl::with_full_config(
                Arc::clone(engine),
                self.config.auth.clone(),
                self.rate_limiter.clone(),
                self.audit_logger.clone(),
                self.metrics.clone(),
            );
            tracing::info!("Vector services enabled");
            (
                Some(PointsServiceServer::new(points_service)),
                Some(CollectionsServiceServer::new(collections_service)),
            )
        } else {
            (None, None)
        };

        // Build REST server if configured
        let rest_handle = if let (Some(rest_addr), Some(ref engine)) =
            (self.config.rest_addr, &self.vector_engine)
        {
            let rest_ctx = Arc::new(
                VectorApiContext::new(Arc::clone(engine))
                    .with_auth(self.config.auth.clone())
                    .with_rate_limiter(self.rate_limiter.clone())
                    .with_audit_logger(self.audit_logger.clone())
                    .with_metrics(self.metrics.clone()),
            );
            let rest_router = rest::router(rest_ctx);
            let listener = TcpListener::bind(rest_addr).await.map_err(|e| {
                ServerError::Internal(format!("failed to bind REST server to {rest_addr}: {e}"))
            })?;
            tracing::info!("REST API enabled on {}", rest_addr);
            Some((listener, rest_router))
        } else {
            None
        };

        // Build Web admin server if configured and all engines are available
        let web_handle =
            if let (Some(web_addr), Some(ref relational), Some(ref vector), Some(ref graph)) = (
                self.config.web_addr,
                &self.relational_engine,
                &self.vector_engine,
                &self.graph_engine,
            ) {
                let web_ctx = Arc::new(
                    web::AdminContext::new(
                        Arc::clone(relational),
                        Arc::clone(vector),
                        Arc::clone(graph),
                    )
                    .with_auth(self.config.auth.clone())
                    .with_metrics(self.metrics.clone()),
                );
                let web_router = web::router(web_ctx);
                let listener = TcpListener::bind(web_addr).await.map_err(|e| {
                    ServerError::Internal(format!("failed to bind Web server to {web_addr}: {e}"))
                })?;
                tracing::info!("Web admin UI enabled on {}", web_addr);
                Some((listener, web_router))
            } else {
                None
            };

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

        // Start REST server task if configured
        let rest_task = rest_handle.map(|(listener, rest_router)| {
            tokio::spawn(async move {
                axum::serve(listener, rest_router)
                    .await
                    .map_err(|e| ServerError::Internal(format!("REST server error: {e}")))
            })
        });

        // Start Web server task if configured
        let web_task = web_handle.map(|(listener, web_router)| {
            tokio::spawn(async move {
                axum::serve(listener, web_router)
                    .await
                    .map_err(|e| ServerError::Internal(format!("Web server error: {e}")))
            })
        });

        // Add services and start gRPC server
        if self.config.enable_grpc_web {
            let layer = GrpcWebLayer::new();
            let mut router = builder
                .layer(layer)
                .add_service(query_svc)
                .add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(points) = points_svc {
                router = router.add_service(points);
            }
            if let Some(collections) = collections_svc {
                router = router.add_service(collections);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            // Run gRPC server with optional REST and Web servers
            match (rest_task, web_task) {
                (Some(rest), Some(web)) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = rest => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                        result = web => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (Some(rest), None) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = rest => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (None, Some(web)) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = web => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (None, None) => {
                    router.serve_with_shutdown(addr, drain_future).await?;
                },
            }
        } else {
            let mut router = builder.add_service(query_svc).add_service(health_svc);

            if let Some(blob) = blob_svc {
                router = router.add_service(blob);
            }
            if let Some(points) = points_svc {
                router = router.add_service(points);
            }
            if let Some(collections) = collections_svc {
                router = router.add_service(collections);
            }
            if let Some(refl) = reflection_svc {
                router = router.add_service(refl);
            }

            // Run gRPC server with optional REST and Web servers
            match (rest_task, web_task) {
                (Some(rest), Some(web)) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = rest => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                        result = web => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (Some(rest), None) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = rest => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("REST task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (None, Some(web)) => {
                    tokio::select! {
                        result = router.serve_with_shutdown(addr, drain_future) => result?,
                        result = web => result.map_err(|e: tokio::task::JoinError| ServerError::Internal(format!("Web task panic: {e}")))?.map_err(|e| ServerError::Internal(e.to_string()))?,
                    }
                },
                (None, None) => {
                    router.serve_with_shutdown(addr, drain_future).await?;
                },
            }
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

    #[test]
    fn test_server_with_relational_engine() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let relational = Arc::new(RelationalEngine::new());

        let server =
            NeumannServer::new(router, config).with_relational_engine(Arc::clone(&relational));

        assert!(server.relational_engine.is_some());
        let stored = server.relational_engine.as_ref().unwrap();
        assert!(Arc::ptr_eq(stored, &relational));
    }

    #[test]
    fn test_server_with_vector_engine() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let vector = Arc::new(VectorEngine::new());

        let server = NeumannServer::new(router, config).with_vector_engine(Arc::clone(&vector));

        assert!(server.vector_engine.is_some());
        let stored = server.vector_engine.as_ref().unwrap();
        assert!(Arc::ptr_eq(stored, &vector));
    }

    #[test]
    fn test_server_with_graph_engine() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let graph = Arc::new(GraphEngine::new());

        let server = NeumannServer::new(router, config).with_graph_engine(Arc::clone(&graph));

        assert!(server.graph_engine.is_some());
        let stored = server.graph_engine.as_ref().unwrap();
        assert!(Arc::ptr_eq(stored, &graph));
    }

    #[test]
    fn test_server_full_builder_chain() {
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let server = NeumannServer::new(router, config)
            .with_relational_engine(relational)
            .with_vector_engine(vector)
            .with_graph_engine(graph)
            .with_metrics(metrics);

        assert!(server.relational_engine.is_some());
        assert!(server.vector_engine.is_some());
        assert!(server.graph_engine.is_some());
        assert!(server.metrics.is_some());
    }

    #[test]
    fn test_server_with_rate_limit_config() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();
        config.rate_limit = Some(RateLimitConfig::default());

        let server = NeumannServer::new(router, config);

        assert!(server.rate_limiter.is_some());
    }

    #[test]
    fn test_server_with_audit_config() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();
        config.audit = Some(AuditConfig::default());

        let server = NeumannServer::new(router, config);

        assert!(server.audit_logger.is_some());
    }

    #[test]
    fn test_server_without_rate_limit() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();

        let server = NeumannServer::new(router, config);

        assert!(server.rate_limiter.is_none());
    }

    #[test]
    fn test_server_without_audit() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();

        let server = NeumannServer::new(router, config);

        assert!(server.audit_logger.is_none());
    }

    #[test]
    fn test_server_config_with_all_engines() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let server = NeumannServer::new(router, config)
            .with_relational_engine(Arc::clone(&relational))
            .with_vector_engine(Arc::clone(&vector))
            .with_graph_engine(Arc::clone(&graph));

        assert!(server.relational_engine.is_some());
        assert!(server.vector_engine.is_some());
        assert!(server.graph_engine.is_some());
    }

    #[tokio::test]
    async fn test_server_with_blob_and_engines() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default())
            .await
            .expect("should create blob store");

        let server = NeumannServer::new(router, config)
            .with_blob_store(Arc::new(Mutex::new(blob_store)))
            .with_relational_engine(relational)
            .with_vector_engine(vector)
            .with_graph_engine(graph);

        assert!(server.blob_store.is_some());
        assert!(server.relational_engine.is_some());
        assert!(server.vector_engine.is_some());
        assert!(server.graph_engine.is_some());
    }

    #[test]
    fn test_server_config_validate_ok() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_validate_with_valid_auth() {
        let mut config = ServerConfig::default();
        config.auth = Some(AuthConfig::new().with_anonymous(true));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_validate_with_rate_limit() {
        let mut config = ServerConfig::default();
        config.rate_limit = Some(RateLimitConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_validate_with_audit() {
        let mut config = ServerConfig::default();
        config.audit = Some(AuditConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_with_full_config() {
        use std::time::Duration;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default()
            .with_rate_limit(RateLimitConfig::default())
            .with_audit(AuditConfig::default())
            .with_shutdown(ShutdownConfig::default())
            .with_memory_budget(MemoryBudgetConfig::default())
            .with_max_message_size(128 * 1024 * 1024)
            .with_blob_chunk_size(64 * 1024)
            .with_max_concurrent_connections(500)
            .with_max_concurrent_streams_per_connection(100)
            .with_initial_window_size(65536)
            .with_initial_connection_window_size(1024 * 1024)
            .with_request_timeout(Duration::from_secs(30));

        let server = NeumannServer::new(router, config);

        assert!(server.rate_limiter.is_some());
        assert!(server.audit_logger.is_some());
    }

    #[test]
    fn test_router_accessor_returns_same_instance() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let server = NeumannServer::new(Arc::clone(&router), config);

        assert!(Arc::ptr_eq(&router, server.router()));
    }

    #[test]
    fn test_server_multiple_blob_store_sets() {
        // Test that with_blob_store can override previous blob store
        use tokio::runtime::Runtime;

        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let router = Arc::new(RwLock::new(QueryRouter::new()));
            let config = ServerConfig::default();

            let store1 = TensorStore::new();
            let blob_store1 = BlobStore::new(store1, BlobConfig::default())
                .await
                .expect("should create blob store 1");

            let store2 = TensorStore::new();
            let blob_store2 = BlobStore::new(store2, BlobConfig::default())
                .await
                .expect("should create blob store 2");

            let server = NeumannServer::new(router, config)
                .with_blob_store(Arc::new(Mutex::new(blob_store1)))
                .with_blob_store(Arc::new(Mutex::new(blob_store2)));

            // The second blob store should be the one that's set
            assert!(server.blob_store.is_some());
        });
    }

    #[test]
    fn test_server_builder_order_independence() {
        // Test that builder methods can be called in any order
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default()
            .with_rate_limit(RateLimitConfig::default())
            .with_audit(AuditConfig::default());
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        // Order 1: metrics first
        let server1 = NeumannServer::new(Arc::clone(&router), config.clone())
            .with_metrics(Arc::clone(&metrics))
            .with_relational_engine(Arc::clone(&relational))
            .with_vector_engine(Arc::clone(&vector))
            .with_graph_engine(Arc::clone(&graph));

        // Order 2: metrics last
        let server2 = NeumannServer::new(router, config)
            .with_relational_engine(relational)
            .with_vector_engine(vector)
            .with_graph_engine(graph)
            .with_metrics(metrics);

        // Both should have all components
        assert!(server1.metrics.is_some());
        assert!(server1.relational_engine.is_some());
        assert!(server1.vector_engine.is_some());
        assert!(server1.graph_engine.is_some());
        assert!(server1.rate_limiter.is_some());
        assert!(server1.audit_logger.is_some());

        assert!(server2.metrics.is_some());
        assert!(server2.relational_engine.is_some());
        assert!(server2.vector_engine.is_some());
        assert!(server2.graph_engine.is_some());
        assert!(server2.rate_limiter.is_some());
        assert!(server2.audit_logger.is_some());
    }

    #[test]
    fn test_server_config_streaming_options() {
        use crate::config::StreamingConfig;

        let config = ServerConfig::default().with_streaming(StreamingConfig::default());

        assert!(config.streaming.is_some());
    }

    #[test]
    fn test_server_config_enable_options() {
        let config = ServerConfig::default()
            .with_grpc_web(false)
            .with_reflection(false);

        assert!(!config.enable_grpc_web);
        assert!(!config.enable_reflection);
    }

    #[test]
    fn test_server_config_with_tls() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/tmp/cert.pem"),
            PathBuf::from("/tmp/key.pem"),
        );

        let config = ServerConfig::default().with_tls(tls);

        assert!(config.tls.is_some());
    }

    #[test]
    fn test_server_config_with_rest_addr() {
        use std::net::SocketAddr;

        let rest_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let config = ServerConfig::default().with_rest_addr(rest_addr);

        assert!(config.rest_addr.is_some());
        assert_eq!(config.rest_addr.unwrap().port(), 8080);
    }

    #[test]
    fn test_server_config_with_web_addr() {
        use std::net::SocketAddr;

        let web_addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let config = ServerConfig::default().with_web_addr(web_addr);

        assert!(config.web_addr.is_some());
        assert_eq!(config.web_addr.unwrap().port(), 9000);
    }

    #[test]
    fn test_server_config_channel_capacity() {
        let config = ServerConfig::default().with_stream_channel_capacity(128);

        assert_eq!(config.stream_channel_capacity, 128);
    }

    #[test]
    fn test_server_config_clone() {
        let config = ServerConfig::default()
            .with_grpc_web(false)
            .with_stream_channel_capacity(64);

        let cloned = config.clone();

        assert_eq!(cloned.enable_grpc_web, config.enable_grpc_web);
        assert_eq!(
            cloned.stream_channel_capacity,
            config.stream_channel_capacity
        );
    }

    #[test]
    fn test_server_config_debug() {
        let config = ServerConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("ServerConfig"));
    }

    #[test]
    fn test_tls_config_new() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/path/to/cert.pem"),
            PathBuf::from("/path/to/key.pem"),
        );

        assert_eq!(tls.cert_path, PathBuf::from("/path/to/cert.pem"));
        assert_eq!(tls.key_path, PathBuf::from("/path/to/key.pem"));
        assert!(tls.ca_cert_path.is_none());
        assert!(!tls.require_client_cert);
    }

    #[test]
    fn test_tls_config_with_client_auth() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/path/to/cert.pem"),
            PathBuf::from("/path/to/key.pem"),
        )
        .with_ca_cert(PathBuf::from("/path/to/ca.pem"))
        .with_required_client_cert(true);

        assert!(tls.ca_cert_path.is_some());
        assert!(tls.require_client_cert);
    }

    #[test]
    fn test_tls_config_clone() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/path/to/cert.pem"),
            PathBuf::from("/path/to/key.pem"),
        )
        .with_required_client_cert(true);

        let cloned = tls.clone();

        assert_eq!(cloned.cert_path, tls.cert_path);
        assert_eq!(cloned.require_client_cert, tls.require_client_cert);
    }

    #[test]
    fn test_tls_config_debug() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/path/to/cert.pem"),
            PathBuf::from("/path/to/key.pem"),
        );

        let debug_str = format!("{:?}", tls);
        assert!(debug_str.contains("TlsConfig"));
    }

    #[test]
    fn test_server_new_with_rate_limit_and_audit() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();
        config.rate_limit = Some(RateLimitConfig::default());
        config.audit = Some(AuditConfig::default());

        let server = NeumannServer::new(router, config);

        assert!(server.rate_limiter.is_some());
        assert!(server.audit_logger.is_some());
    }

    #[test]
    fn test_server_engines_none_by_default() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let server = NeumannServer::new(router, config);

        assert!(server.blob_store.is_none());
        assert!(server.relational_engine.is_none());
        assert!(server.vector_engine.is_none());
        assert!(server.graph_engine.is_none());
        assert!(server.metrics.is_none());
    }

    #[test]
    fn test_server_config_validate_anonymous_auth() {
        let mut config = ServerConfig::default();
        config.auth = Some(AuthConfig::new().with_anonymous(true));

        // Anonymous auth should be valid
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_validate_no_auth() {
        let config = ServerConfig::default();

        // No auth should be valid (defaults to anonymous)
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_with_all_addresses() {
        use std::net::SocketAddr;

        let bind: SocketAddr = "0.0.0.0:9200".parse().unwrap();
        let rest: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        let web: SocketAddr = "0.0.0.0:9000".parse().unwrap();

        let config = ServerConfig::new()
            .with_bind_addr(bind)
            .with_rest_addr(rest)
            .with_web_addr(web);

        assert_eq!(config.bind_addr.port(), 9200);
        assert_eq!(config.rest_addr.unwrap().port(), 8080);
        assert_eq!(config.web_addr.unwrap().port(), 9000);
    }

    #[tokio::test]
    async fn test_server_with_shared_storage_creates_all_components() {
        let config = ServerConfig::default();
        let server = NeumannServer::with_shared_storage(config).await.unwrap();

        assert!(server.blob_store.is_some());
        // Router should be accessible
        let router = server.router();
        assert!(Arc::strong_count(router) >= 1);
    }

    #[test]
    fn test_server_config_http2_settings() {
        let config = ServerConfig::default()
            .with_initial_window_size(1024 * 1024)
            .with_initial_connection_window_size(2 * 1024 * 1024)
            .with_max_concurrent_streams_per_connection(200);

        assert_eq!(config.initial_window_size, Some(1024 * 1024));
        assert_eq!(config.initial_connection_window_size, Some(2 * 1024 * 1024));
        assert_eq!(config.max_concurrent_streams_per_connection, Some(200));
    }

    #[test]
    fn test_server_config_blob_settings() {
        let config = ServerConfig::default()
            .with_blob_chunk_size(128 * 1024)
            .with_max_message_size(64 * 1024 * 1024);

        assert_eq!(config.blob_chunk_size, 128 * 1024);
        assert_eq!(config.max_message_size, 64 * 1024 * 1024);
    }

    #[test]
    fn test_server_multiple_engine_sets() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();

        let relational1 = Arc::new(RelationalEngine::new());
        let relational2 = Arc::new(RelationalEngine::new());

        // Setting engine multiple times should use the last one
        let server = NeumannServer::new(router, config)
            .with_relational_engine(relational1)
            .with_relational_engine(Arc::clone(&relational2));

        assert!(server.relational_engine.is_some());
        assert!(Arc::ptr_eq(
            server.relational_engine.as_ref().unwrap(),
            &relational2
        ));
    }

    #[test]
    fn test_server_with_vector_engine_only() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let vector = Arc::new(VectorEngine::new());

        let server = NeumannServer::new(router, config).with_vector_engine(vector);

        assert!(server.vector_engine.is_some());
        assert!(server.relational_engine.is_none());
        assert!(server.graph_engine.is_none());
    }

    #[test]
    fn test_server_with_graph_engine_only() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let graph = Arc::new(GraphEngine::new());

        let server = NeumannServer::new(router, config).with_graph_engine(graph);

        assert!(server.graph_engine.is_some());
        assert!(server.relational_engine.is_none());
        assert!(server.vector_engine.is_none());
    }

    #[test]
    fn test_server_config_default_bind_addr() {
        let config = ServerConfig::default();
        // Default binds to localhost
        assert_eq!(config.bind_addr.port(), 9200);
    }

    #[test]
    fn test_server_config_with_custom_bind_addr() {
        use std::net::SocketAddr;

        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let config = ServerConfig::default().with_bind_addr(addr);
        assert_eq!(config.bind_addr.ip().to_string(), "127.0.0.1");
        assert_eq!(config.bind_addr.port(), 8080);
    }

    #[test]
    fn test_server_config_max_upload_size() {
        let config = ServerConfig::default().with_max_upload_size(100 * 1024 * 1024);
        assert_eq!(config.max_upload_size, 100 * 1024 * 1024);
    }

    #[test]
    fn test_server_config_request_timeout() {
        use std::time::Duration;

        let config = ServerConfig::default().with_request_timeout(Duration::from_secs(60));
        assert!(config.request_timeout.is_some());
        assert_eq!(config.request_timeout.unwrap(), Duration::from_secs(60));
    }

    #[test]
    fn test_server_config_max_concurrent_connections() {
        let config = ServerConfig::default().with_max_concurrent_connections(1000);
        assert_eq!(config.max_concurrent_connections, Some(1000));
    }

    #[test]
    fn test_tls_config_optional_client_cert() {
        use std::path::PathBuf;

        let tls = TlsConfig::new(
            PathBuf::from("/path/cert.pem"),
            PathBuf::from("/path/key.pem"),
        )
        .with_ca_cert(PathBuf::from("/path/ca.pem"))
        .with_required_client_cert(false);

        assert!(tls.ca_cert_path.is_some());
        assert!(!tls.require_client_cert);
    }

    #[test]
    fn test_server_metrics_field_access() {
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let config = ServerConfig::default();
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));

        let server = NeumannServer::new(router, config).with_metrics(Arc::clone(&metrics));

        // Verify the metrics reference
        assert!(server.metrics.is_some());
        let stored_metrics = server.metrics.as_ref().unwrap();
        assert!(Arc::ptr_eq(stored_metrics, &metrics));
    }

    #[test]
    fn test_server_rate_limiter_field_access() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();
        config.rate_limit = Some(RateLimitConfig::default());

        let server = NeumannServer::new(router, config);
        assert!(server.rate_limiter.is_some());
    }

    #[test]
    fn test_server_audit_logger_field_access() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();
        config.audit = Some(AuditConfig::default());

        let server = NeumannServer::new(router, config);
        assert!(server.audit_logger.is_some());
    }

    #[tokio::test]
    async fn test_serve_validation_error() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let mut config = ServerConfig::default();

        // Create an auth config that requires authentication but has no API keys
        config.auth = Some(AuthConfig::new().with_anonymous(false));

        let server = NeumannServer::new(router, config);
        let result = server.serve().await;

        // Should fail validation
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_config_path_accessors() {
        use std::path::PathBuf;

        let cert_path = PathBuf::from("/test/cert.pem");
        let key_path = PathBuf::from("/test/key.pem");

        let tls = TlsConfig::new(cert_path.clone(), key_path.clone());

        assert_eq!(tls.cert_path, cert_path);
        assert_eq!(tls.key_path, key_path);
    }

    #[test]
    fn test_server_config_chained_builder() {
        use std::net::SocketAddr;
        use std::time::Duration;

        let addr: SocketAddr = "0.0.0.0:9300".parse().unwrap();
        let rest_addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();

        let config = ServerConfig::new()
            .with_bind_addr(addr)
            .with_rest_addr(rest_addr)
            .with_max_message_size(32 * 1024 * 1024)
            .with_max_upload_size(16 * 1024 * 1024)
            .with_blob_chunk_size(64 * 1024)
            .with_stream_channel_capacity(64)
            .with_grpc_web(true)
            .with_reflection(true)
            .with_rate_limit(RateLimitConfig::default())
            .with_audit(AuditConfig::default())
            .with_request_timeout(Duration::from_secs(30));

        assert_eq!(config.bind_addr.port(), 9300);
        assert_eq!(config.rest_addr.unwrap().port(), 8080);
        assert_eq!(config.max_message_size, 32 * 1024 * 1024);
        assert_eq!(config.max_upload_size, 16 * 1024 * 1024);
        assert_eq!(config.blob_chunk_size, 64 * 1024);
        assert_eq!(config.stream_channel_capacity, 64);
        assert!(config.enable_grpc_web);
        assert!(config.enable_reflection);
        assert!(config.rate_limit.is_some());
        assert!(config.audit.is_some());
        assert!(config.request_timeout.is_some());
    }
}
