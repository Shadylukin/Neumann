// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! REST API for vector operations.
//!
//! Provides Qdrant-style REST endpoints for vector point and collection operations.

use std::sync::Arc;

use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use vector_engine::VectorEngine;

use crate::audit::AuditLogger;
use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;
use crate::rate_limit::RateLimiter;

pub mod collections;
pub mod error;
pub mod points;
pub mod types;

pub use error::{ApiError, ApiResult};
pub use types::*;

/// Default maximum request body size (16MB).
const DEFAULT_MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

/// Context shared across REST handlers.
pub struct VectorApiContext {
    /// Vector engine for operations.
    pub engine: Arc<VectorEngine>,
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
    /// Create a new context with a vector engine.
    #[must_use]
    pub const fn new(engine: Arc<VectorEngine>) -> Self {
        Self {
            engine,
            auth_config: None,
            rate_limiter: None,
            audit_logger: None,
            metrics: None,
        }
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
    Router::new()
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
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_api_context_new() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = VectorApiContext::new(engine);

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
        let ctx = VectorApiContext::new(engine).with_auth(Some(auth_config));

        assert!(ctx.auth_config.is_some());
    }

    #[test]
    fn test_vector_api_context_with_rate_limiter() {
        let engine = Arc::new(VectorEngine::new());
        let rate_limiter = Arc::new(RateLimiter::default());
        let ctx = VectorApiContext::new(engine).with_rate_limiter(Some(rate_limiter));

        assert!(ctx.rate_limiter.is_some());
    }

    #[test]
    fn test_vector_api_context_with_audit_logger() {
        let engine = Arc::new(VectorEngine::new());
        let audit_logger = Arc::new(AuditLogger::default());
        let ctx = VectorApiContext::new(engine).with_audit_logger(Some(audit_logger));

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
        let ctx = Arc::new(VectorApiContext::new(engine));
        let _router = router(ctx);
    }

    #[test]
    fn test_router_with_config_creation() {
        let engine = Arc::new(VectorEngine::new());
        let ctx = Arc::new(VectorApiContext::new(engine));
        let config = RestConfig::new().with_max_body_size(8 * 1024 * 1024);
        let _router = router_with_config(ctx, &config);
    }
}
