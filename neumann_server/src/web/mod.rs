// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Web UI for Neumann Server administration.
//!
//! Provides a modern, dark-mode admin interface for browsing and managing
//! data across all engines: relational, vector, graph, vault, and cache.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use parking_lot::RwLock;
use tower_http::cors::CorsLayer;

use graph_engine::GraphEngine;
use query_router::QueryRouter;
use relational_engine::RelationalEngine;
use tensor_blob::BlobStore;
use tensor_cache::Cache;
use tensor_checkpoint::CheckpointManager;
use tensor_store::TensorStore;
use tensor_unified::UnifiedEngine;
use tensor_vault::Vault;
use vector_engine::VectorEngine;

use crate::config::AuthConfig;
use crate::metrics::ServerMetrics;

mod assets;
pub mod handlers;
pub mod icons;
pub mod templates;

pub use assets::ADMIN_CSS;

/// Point-in-time snapshot of consensus and distributed transaction state.
///
/// Uses only primitive types so no dependency on `tensor_chain` is needed.
/// The server populates this from a running `RaftNode` / `DistributedTxCoordinator`
/// / `DeadlockDetector` if consensus is active.
pub struct ChainStatus {
    // -- Raft consensus --
    /// Current Raft role ("Leader", "Follower", "Candidate").
    pub raft_state: String,
    /// Current term number.
    pub current_term: u64,
    /// Highest committed log index.
    pub commit_index: u64,
    /// Total log entries.
    pub log_length: usize,
    /// Current leader node ID (if known).
    pub leader_id: Option<String>,
    /// Fast-path acceptance rate (0.0..1.0).
    pub fast_path_rate: f32,
    /// Heartbeat success rate (0.0..1.0).
    pub heartbeat_success_rate: f32,
    /// Successful heartbeats sent.
    pub heartbeat_successes: u64,
    /// Failed heartbeats.
    pub heartbeat_failures: u64,
    /// Quorum health checks performed.
    pub quorum_checks: u64,
    /// Times quorum was lost.
    pub quorum_lost_events: u64,
    /// Leadership changes (step-downs).
    pub leader_step_downs: u64,

    // -- Distributed transactions --
    /// Total transactions started.
    pub tx_started: u64,
    /// Total committed.
    pub tx_committed: u64,
    /// Total aborted.
    pub tx_aborted: u64,
    /// Total timed out.
    pub tx_timed_out: u64,
    /// Conflicts detected.
    pub tx_conflicts: u64,
    /// Commit rate (committed / started).
    pub tx_commit_rate: f32,
    /// Conflict rate (conflicts / started).
    pub tx_conflict_rate: f32,
    /// Currently pending transactions.
    pub tx_pending: usize,

    // -- Deadlock detection --
    /// Total deadlock cycles found.
    pub deadlocks_detected: u64,
    /// Victims aborted to break cycles.
    pub victims_aborted: u64,
    /// Detection cycles executed.
    pub detection_cycles: u64,
    /// Longest cycle ever seen.
    pub max_cycle_length: u64,
    /// Whether detection is enabled.
    pub deadlock_enabled: bool,
}

/// Context shared across web handlers.
pub struct AdminContext {
    /// Relational engine for table operations.
    pub relational: Arc<RelationalEngine>,
    /// Vector engine for embedding operations.
    pub vector: Arc<VectorEngine>,
    /// Graph engine for node/edge operations.
    pub graph: Arc<GraphEngine>,
    /// Unified engine for cross-modal contraction (optional).
    pub unified: Option<Arc<UnifiedEngine>>,
    /// Vault for secret management (optional).
    pub vault: Option<Arc<Vault>>,
    /// Cache for LLM response caching (optional).
    pub cache: Option<Arc<Cache>>,
    /// Blob store for artifact management (optional).
    pub blob: Option<Arc<tokio::sync::Mutex<BlobStore>>>,
    /// Checkpoint manager for backup/restore (optional).
    pub checkpoint: Option<Arc<CheckpointManager>>,
    /// Tensor store for storage internals (optional).
    pub store: Option<TensorStore>,
    /// Consensus / chain status snapshot (optional).
    pub chain: Option<Arc<ChainStatus>>,
    /// Authentication configuration (optional).
    pub auth_config: Option<AuthConfig>,
    /// Server metrics (optional).
    pub metrics: Option<Arc<ServerMetrics>>,
    /// Query router for executing parsed statements (optional).
    pub query_router: Option<Arc<RwLock<QueryRouter>>>,
}

impl AdminContext {
    /// Create a new admin context with all three engines.
    #[must_use]
    pub const fn new(
        relational: Arc<RelationalEngine>,
        vector: Arc<VectorEngine>,
        graph: Arc<GraphEngine>,
    ) -> Self {
        Self {
            relational,
            vector,
            graph,
            unified: None,
            vault: None,
            cache: None,
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
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

    /// Add unified engine for contraction views.
    #[must_use]
    pub fn with_unified(mut self, unified: Option<Arc<UnifiedEngine>>) -> Self {
        self.unified = unified;
        self
    }

    /// Add vault for secret management views.
    #[must_use]
    pub fn with_vault(mut self, vault: Option<Arc<Vault>>) -> Self {
        self.vault = vault;
        self
    }

    /// Add cache for cache stats views.
    #[must_use]
    pub fn with_cache(mut self, cache: Option<Arc<Cache>>) -> Self {
        self.cache = cache;
        self
    }

    /// Add blob store for artifact management views.
    #[must_use]
    pub fn with_blob(mut self, blob: Option<Arc<tokio::sync::Mutex<BlobStore>>>) -> Self {
        self.blob = blob;
        self
    }

    /// Add checkpoint manager for backup/restore views.
    #[must_use]
    pub fn with_checkpoint(mut self, checkpoint: Option<Arc<CheckpointManager>>) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    /// Add tensor store for storage internals views.
    #[must_use]
    pub fn with_store(mut self, store: Option<TensorStore>) -> Self {
        self.store = store;
        self
    }

    /// Add consensus / chain status snapshot.
    #[must_use]
    pub fn with_chain(mut self, chain: Option<Arc<ChainStatus>>) -> Self {
        self.chain = chain;
        self
    }

    /// Add query router for executing parsed statements.
    #[must_use]
    pub fn with_query_router(mut self, router: Option<Arc<RwLock<QueryRouter>>>) -> Self {
        self.query_router = router;
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
    /// Contraction explainability views.
    Contraction,
    /// Vault secret management.
    Vault,
    /// Cache stats dashboard.
    Cache,
    /// Blob storage browser.
    Blob,
    /// Checkpoint manager.
    Checkpoint,
    /// Storage internals dashboard.
    Storage,
    /// Consensus / chain dashboard.
    Chain,
}

/// Vector engine routes (default and named collections).
fn vector_routes() -> Router<Arc<AdminContext>> {
    Router::new()
        .route("/vector", get(handlers::vector::collections_list))
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
}

/// Graph engine and algorithm routes.
fn graph_routes() -> Router<Arc<AdminContext>> {
    Router::new()
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
        .route(
            "/graph/algorithms/dashboard",
            get(handlers::graph_algorithms::dashboard),
        )
        .route(
            "/graph/algorithms/execute",
            get(handlers::graph_algorithms::execute_form)
                .post(handlers::graph_algorithms::execute_submit),
        )
}

/// Storage subsystem routes (blob, checkpoint, storage internals, cache).
fn storage_routes() -> Router<Arc<AdminContext>> {
    Router::new()
        .route("/blob", get(handlers::blob::overview))
        .route("/blob/artifacts", get(handlers::blob::artifacts_list))
        .route(
            "/blob/artifacts/{artifact_id}",
            get(handlers::blob::artifact_detail),
        )
        .route("/checkpoint", get(handlers::checkpoint::list_view))
        .route("/checkpoint/config", get(handlers::checkpoint::config_view))
        .route("/checkpoint/{id}", get(handlers::checkpoint::detail_view))
        .route("/storage", get(handlers::storage::overview))
        .route("/storage/shards", get(handlers::storage::shard_heatmap))
        .route("/storage/wal", get(handlers::storage::wal_status))
        .route("/cache", get(handlers::cache::stats_dashboard))
        .route("/cache/config", get(handlers::cache::config_viewer))
        .route("/cache/layers", get(handlers::cache::layers_breakdown))
}

/// Create the admin web UI router.
pub fn router(ctx: Arc<AdminContext>) -> Router {
    // CORS layer for API endpoints accessed by external frontends
    let cors = CorsLayer::new()
        .allow_origin([axum::http::HeaderValue::from_static(
            "http://localhost:5173",
        )])
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    Router::new()
        .route("/", get(handlers::dashboard))
        .route("/relational", get(handlers::relational::tables_list))
        .route(
            "/relational/{table}",
            get(handlers::relational::table_detail),
        )
        .route(
            "/relational/{table}/rows",
            get(handlers::relational::table_rows),
        )
        .merge(vector_routes())
        .merge(graph_routes())
        .route("/contraction", get(handlers::contraction::explainability))
        .route("/contraction/ranking", get(handlers::contraction::ranking))
        .route(
            "/contraction/sensitivity",
            get(handlers::contraction::sensitivity),
        )
        .route(
            "/contraction/counterfactual",
            get(handlers::contraction::counterfactual),
        )
        .route("/vault", get(handlers::vault::secrets_list))
        .route("/vault/audit", get(handlers::vault::audit_log))
        .route("/vault/status", get(handlers::vault::vault_status))
        .route("/vault/security", get(handlers::vault::security_dashboard))
        .route(
            "/vault/blast-radius",
            get(handlers::vault::blast_radius_view),
        )
        .route(
            "/vault/critical",
            get(handlers::vault::critical_entities_view),
        )
        .route("/vault/reveal", post(handlers::vault::reveal_value))
        .route("/vault/{*key}", get(handlers::vault::secret_detail))
        .merge(storage_routes())
        .route("/chain", get(handlers::chain::consensus))
        .route("/chain/transactions", get(handlers::chain::transactions))
        .route("/chain/deadlocks", get(handlers::chain::deadlocks))
        .route("/metrics", get(handlers::metrics::dashboard))
        .route("/api/metrics", get(handlers::metrics::api_snapshot))
        .route("/api/graph/subgraph", get(handlers::graph::api_subgraph))
        .route("/api/query", axum::routing::post(handlers::api_query))
        .route("/api/galaxy", post(handlers::api_galaxy))
        .route("/api/execute", post(handlers::api_execute))
        .layer(cors)
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
        assert!(ctx.unified.is_none());
        assert!(ctx.vault.is_none());
        assert!(ctx.cache.is_none());
        assert!(ctx.blob.is_none());
        assert!(ctx.checkpoint.is_none());
        assert!(ctx.store.is_none());
        assert!(ctx.chain.is_none());
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
    fn test_admin_context_with_unified() {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let unified = Arc::new(UnifiedEngine::new());

        let ctx = AdminContext::new(relational, vector, graph).with_unified(Some(unified));

        assert!(ctx.unified.is_some());
    }

    #[test]
    fn test_admin_context_with_vault() {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());

        let ctx = AdminContext::new(relational, vector, graph).with_vault(None);

        assert!(ctx.vault.is_none());
    }

    #[test]
    fn test_admin_context_with_cache() {
        let relational = Arc::new(RelationalEngine::new());
        let vector = Arc::new(VectorEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let cache = Arc::new(Cache::new());

        let ctx = AdminContext::new(relational, vector, graph).with_cache(Some(cache));

        assert!(ctx.cache.is_some());
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
