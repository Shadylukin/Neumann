// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Web UI for Neumann Server administration.
//!
//! Provides a modern, dark-mode admin interface for browsing and managing
//! data across all three engines: relational, vector, and graph.

use std::sync::Arc;

use axum::routing::get;
use axum::Router;

use graph_engine::GraphEngine;
use relational_engine::RelationalEngine;
use vector_engine::VectorEngine;

use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;

mod assets;
pub mod handlers;
pub mod templates;
pub mod tro;

pub use assets::{ADMIN_CSS, AUDIO_SCRIPT, TRO_CSS, TRO_SCRIPT};

/// Context shared across web handlers.
pub struct AdminContext {
    /// Relational engine for table operations.
    pub relational: Arc<RelationalEngine>,
    /// Vector engine for embedding operations.
    pub vector: Arc<VectorEngine>,
    /// Graph engine for node/edge operations.
    pub graph: Arc<GraphEngine>,
    /// Authentication configuration (optional).
    pub auth_config: Option<AuthConfig>,
    /// Server metrics (optional).
    pub metrics: Option<Arc<ServerMetrics>>,
}

impl AdminContext {
    /// Create a new admin context with all three engines.
    #[must_use]
    pub fn new(
        relational: Arc<RelationalEngine>,
        vector: Arc<VectorEngine>,
        graph: Arc<GraphEngine>,
    ) -> Self {
        Self {
            relational,
            vector,
            graph,
            auth_config: None,
            metrics: None,
        }
    }

    /// Add authentication configuration.
    #[must_use]
    pub fn with_auth(mut self, config: Option<AuthConfig>) -> Self {
        self.auth_config = config;
        self
    }

    /// Add server metrics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Option<Arc<ServerMetrics>>) -> Self {
        self.metrics = metrics;
        self
    }
}

/// Navigation item for sidebar highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavItem {
    /// Dashboard overview page.
    Dashboard,
    /// Relational engine browser.
    Relational,
    /// Vector engine browser.
    Vector,
    /// Graph engine browser.
    Graph,
}

/// Create the admin web UI router.
pub fn router(ctx: Arc<AdminContext>) -> Router {
    Router::new()
        // Dashboard
        .route("/", get(handlers::dashboard))
        // Relational engine routes
        .route("/relational", get(handlers::relational::tables_list))
        .route("/relational/{table}", get(handlers::relational::table_detail))
        .route(
            "/relational/{table}/rows",
            get(handlers::relational::table_rows),
        )
        // Vector engine routes
        .route("/vector", get(handlers::vector::collections_list))
        // Default collection routes (must be before :collection to match first)
        .route(
            "/vector/_default",
            get(handlers::vector::default_collection_detail),
        )
        .route(
            "/vector/_default/points",
            get(handlers::vector::default_points_list),
        )
        .route(
            "/vector/_default/points/{point_id}",
            get(handlers::vector::default_point_detail),
        )
        .route(
            "/vector/_default/search",
            get(handlers::vector::default_search_form)
                .post(handlers::vector::default_search_submit),
        )
        // Named collection routes
        .route(
            "/vector/{collection}",
            get(handlers::vector::collection_detail),
        )
        .route(
            "/vector/{collection}/points",
            get(handlers::vector::points_list),
        )
        .route(
            "/vector/{collection}/points/{point_id}",
            get(handlers::vector::point_detail),
        )
        .route(
            "/vector/{collection}/search",
            get(handlers::vector::search_form).post(handlers::vector::search_submit),
        )
        // Graph engine routes
        .route("/graph", get(handlers::graph::overview))
        .route("/graph/nodes", get(handlers::graph::nodes_list))
        .route("/graph/edges", get(handlers::graph::edges_list))
        .route(
            "/graph/path",
            get(handlers::graph::path_finder).post(handlers::graph::path_finder_submit),
        )
        .route(
            "/graph/algorithms",
            get(handlers::graph::algorithms).post(handlers::graph::algorithms_submit),
        )
        // Algorithm dashboard routes
        .route(
            "/graph/algorithms/dashboard",
            get(handlers::graph_algorithms::dashboard),
        )
        .route(
            "/graph/algorithms/execute",
            get(handlers::graph_algorithms::execute_form)
                .post(handlers::graph_algorithms::execute_submit),
        )
        // Metrics routes
        .route("/metrics", get(handlers::metrics::dashboard))
        .route("/api/metrics", get(handlers::metrics::api_snapshot))
        // Achievements routes
        .route("/achievements", get(handlers::achievements::dashboard))
        // API routes for HTMX
        .route("/api/graph/subgraph", get(handlers::graph::api_subgraph))
        // Query API for terminal
        .route("/api/query", axum::routing::post(handlers::api_query))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_context_new() {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let ctx = AdminContext::new(relational, vector, graph);

        assert!(ctx.auth_config.is_none());
        assert!(ctx.metrics.is_none());
    }

    #[test]
    fn test_admin_context_with_auth() {
        use crate::config::ApiKey;

        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let auth_config = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:test".to_string(),
        ));
        let ctx = AdminContext::new(relational, vector, graph).with_auth(Some(auth_config));

        assert!(ctx.auth_config.is_some());
    }

    #[test]
    fn test_router_creation() {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let ctx = Arc::new(AdminContext::new(relational, vector, graph));
        let _router = router(ctx);
    }
}
