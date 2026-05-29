// SPDX-License-Identifier: MIT OR Apache-2.0
//! REST API for vector operations.
//!
//! Provides Qdrant-style REST endpoints for vector point and collection operations.

use std::sync::Arc;

use axum::http::{header, HeaderName, HeaderValue, Method};
use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use vector_engine::VectorEngine;

use query_router::QueryRouter;

use crate::audit::AuditLogger;
use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;
use crate::rate_limit::RateLimiter;

pub mod auth;
pub mod collections;
pub mod error;
pub mod points;
pub mod spatial;
pub mod spatial3d;
pub mod types;

pub use error::{ApiError, ApiResult};
pub use types::*;

/// Default maximum request body size (16MB).
const DEFAULT_MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

/// Context shared across REST handlers.
///
/// Carries an [`Arc<parking_lot::RwLock<QueryRouter>>`] (the post-unification
/// dispatch entry point) plus the auth/rate-limit/audit/metrics state that
/// each handler may consult before delegating to the router. The spatial
/// indexes live on the router itself (`QueryRouter::spatial()` /
/// `QueryRouter::spatial_3d()`); the handlers read them through the router
/// rather than through this context.
pub struct VectorApiContext {
    /// Query router for typed vector dispatch.
    pub router: Arc<parking_lot::RwLock<QueryRouter>>,
    /// Authentication configuration.
    pub auth_config: Option<AuthConfig>,
    /// Rate limiter.
    pub rate_limiter: Option<Arc<RateLimiter>>,
    /// Audit logger.
    pub audit_logger: Option<Arc<AuditLogger>>,
    /// Server metrics.
    pub metrics: Option<Arc<ServerMetrics>>,
}

impl VectorApiContext {
    /// Create a new context with a query router.
    #[must_use]
    pub const fn new(router: Arc<parking_lot::RwLock<QueryRouter>>) -> Self {
        Self {
            router,
            auth_config: None,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
    }

    /// Convenience constructor for tests and callers that have only a
    /// [`VectorEngine`] (the pre-unification handle). Builds an isolated
    /// [`QueryRouter`], installs the engine via
    /// [`QueryRouter::replace_vector_engine`], and wraps it in the standard
    /// `Arc<RwLock<...>>` shape.
    #[must_use]
    pub fn from_engine(engine: Arc<VectorEngine>) -> Self {
        let mut router = QueryRouter::new();
        router.replace_vector_engine(engine);
        Self::new(Arc::new(parking_lot::RwLock::new(router)))
    }

    /// Add authentication configuration.
    #[must_use]
    pub fn with_auth(mut self, auth_config: Option<AuthConfig>) -> Self {
        self.auth_config = auth_config;
        self
    }

    /// Add rate limiter.
    #[must_use]
    pub fn with_rate_limiter(mut self, rate_limiter: Option<Arc<RateLimiter>>) -> Self {
        self.rate_limiter = rate_limiter;
        self
    }

    /// Add audit logger.
    #[must_use]
    pub fn with_audit_logger(mut self, audit_logger: Option<Arc<AuditLogger>>) -> Self {
        self.audit_logger = audit_logger;
        self
    }

    /// Add server metrics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Option<Arc<ServerMetrics>>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Install (or clear) the 3D spatial index on the underlying router.
    ///
    /// 3D spatial is `Option` on [`QueryRouter`] because it remains
    /// optionally-configured (REST returns "not configured" when `None`). 2D
    /// spatial is always present on the router and does not need a setter.
    #[must_use]
    pub fn with_spatial_3d(
        self,
        spatial_3d: Option<Arc<parking_lot::RwLock<tensor_spatial::SpatialIndex3D<String>>>>,
    ) -> Self {
        self.router.write().set_spatial_3d(spatial_3d);
        self
    }
}

/// REST API configuration.
#[derive(Debug, Clone)]
pub struct RestConfig {
    /// Maximum request body size in bytes.
    pub max_body_size: usize,
    /// Enable CORS.
    pub cors_enabled: bool,
    /// CORS allowed origins.
    pub cors_origins: Vec<String>,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            max_body_size: DEFAULT_MAX_BODY_SIZE,
            cors_enabled: false,
            cors_origins: Vec::new(),
        }
    }
}

impl RestConfig {
    /// Create a new REST configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum body size.
    #[must_use]
    pub const fn with_max_body_size(mut self, size: usize) -> Self {
        self.max_body_size = size;
        self
    }

    /// Enable CORS.
    #[must_use]
    pub const fn with_cors(mut self, enabled: bool) -> Self {
        self.cors_enabled = enabled;
        self
    }

    /// Set CORS allowed origins.
    #[must_use]
    pub fn with_cors_origins(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }
}

/// Create the REST API router.
pub fn router(ctx: Arc<VectorApiContext>) -> Router {
    router_with_config(ctx, &RestConfig::default())
}

/// Create the REST API router with configuration.
pub fn router_with_config(ctx: Arc<VectorApiContext>, config: &RestConfig) -> Router {
    let mut router = Router::new()
        // Points endpoints
        .route(
            "/collections/{name}/points",
            put(points::upsert).post(points::upsert),
        )
        .route("/collections/{name}/points/get", post(points::get))
        .route("/collections/{name}/points/delete", post(points::delete))
        .route("/collections/{name}/points/query", post(points::query))
        .route("/collections/{name}/points/scroll", post(points::scroll))
        // Collections endpoints
        .route("/collections/{name}", put(collections::create))
        .route("/collections/{name}", get(collections::get))
        .route("/collections/{name}", delete(collections::delete))
        .route("/collections", get(collections::list))
        // Spatial endpoints (2D)
        .route(
            "/collections/{name}/spatial/insert",
            post(spatial::insert),
        )
        .route("/collections/{name}/spatial/query", post(spatial::query))
        .route(
            "/collections/{name}/spatial/delete",
            post(spatial::delete),
        )
        .route("/collections/{name}/spatial/count", get(spatial::count))
        // Spatial 3D endpoints
        .route(
            "/collections/{name}/spatial3d/insert",
            post(spatial3d::insert_3d),
        )
        .route(
            "/collections/{name}/spatial3d/query",
            post(spatial3d::query_3d),
        )
        .route(
            "/collections/{name}/spatial3d/nearest",
            post(spatial3d::nearest_3d),
        )
        .route(
            "/collections/{name}/spatial3d/region",
            post(spatial3d::region_3d),
        )
        .route(
            "/collections/{name}/spatial3d/delete",
            post(spatial3d::delete_3d),
        )
        .route(
            "/collections/{name}/spatial3d/count",
            get(spatial3d::count_3d),
        )
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .with_state(ctx);

    // Apply CORS if configured
    if config.cors_enabled {
        let origins: Vec<HeaderValue> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        let cors = CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, HeaderName::from_static("x-api-key")]);
        router = router.layer(cors);
    }

    router
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_api_context_new() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = VectorApiContext::from_engine(engine);

        assert!(ctx.auth_config.is_none());
        assert!(ctx.rate_limiter.is_none());
        assert!(ctx.audit_logger.is_none());
        assert!(ctx.metrics.is_none());
    }

    #[test]
    fn test_vector_api_context_with_auth() {
        use crate::config::ApiKey;

        let engine = Arc::new(VectorEngine::new());
        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:test".to_string(),
        ));
        let ctx = VectorApiContext::from_engine(engine).with_auth(Some(auth_config));

        assert!(ctx.auth_config.is_some());
    }

    #[test]
    fn test_vector_api_context_with_rate_limiter() {
        let engine = Arc::new(VectorEngine::new());
        let rate_limiter = Arc::new(RateLimiter::default());
        let ctx = VectorApiContext::from_engine(engine).with_rate_limiter(Some(rate_limiter));

        assert!(ctx.rate_limiter.is_some());
    }

    #[test]
    fn test_vector_api_context_with_audit_logger() {
        let engine = Arc::new(VectorEngine::new());
        let audit_logger = Arc::new(AuditLogger::default());
        let ctx = VectorApiContext::from_engine(engine).with_audit_logger(Some(audit_logger));

        assert!(ctx.audit_logger.is_some());
    }

    #[test]
    fn test_rest_config_default() {
        let config = RestConfig::default();

        assert_eq!(config.max_body_size, DEFAULT_MAX_BODY_SIZE);
        assert!(!config.cors_enabled);
        assert!(config.cors_origins.is_empty());
    }

    #[test]
    fn test_rest_config_builder() {
        let config = RestConfig::new()
            .with_max_body_size(32 * 1024 * 1024)
            .with_cors(true)
            .with_cors_origins(vec!["http://localhost:3000".to_string()]);

        assert_eq!(config.max_body_size, 32 * 1024 * 1024);
        assert!(config.cors_enabled);
        assert_eq!(config.cors_origins.len(), 1);
    }

    #[test]
    fn test_router_creation() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::from_engine(engine));
        let _router = router(ctx);
    }

    #[test]
    fn test_router_with_config_creation() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::from_engine(engine));
        let config = RestConfig::new().with_max_body_size(8 * 1024 * 1024);
        let _router = router_with_config(ctx, &config);
    }

    #[test]
    fn test_vector_api_context_with_spatial_3d() {
        let engine = Arc::new(VectorEngine::new());
        let spatial_3d = Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex3D::<
            String,
        >::new()));
        let ctx =
            VectorApiContext::from_engine(engine).with_spatial_3d(Some(Arc::clone(&spatial_3d)));

        // `with_spatial_3d` proxies the index to the underlying router; verify
        // the router observes the same Arc.
        let installed = {
            let guard = ctx.router.read();
            guard.spatial_3d().map(Arc::clone)
        };
        assert!(installed.is_some());
        assert!(Arc::ptr_eq(&installed.unwrap(), &spatial_3d));
    }

    #[test]
    fn test_vector_api_context_with_metrics() {
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        let engine = Arc::new(VectorEngine::new());
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = Arc::new(ServerMetrics::new(meter));
        let ctx = VectorApiContext::from_engine(engine).with_metrics(Some(metrics));

        assert!(ctx.metrics.is_some());
    }

    #[test]
    fn test_router_with_cors_enabled() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::from_engine(engine));
        let config = RestConfig::new()
            .with_cors(true)
            .with_cors_origins(vec!["http://localhost:3000".to_string()]);
        let _router = router_with_config(ctx, &config);
    }
}
