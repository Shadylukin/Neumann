// SPDX-License-Identifier: MIT OR Apache-2.0
//! Subsystem lifecycle: `init`/`ensure`/`shutdown` and identity for the
//! [`QueryRouter`].
//!
//! These methods are kept as `impl QueryRouter` (not free functions) because
//! every callsite is on a router instance (`router.init_blob()`) and they all
//! mutate router state. Splitting them across files using `impl QueryRouter`
//! blocks preserves the public API exactly.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tensor_blob::{BlobConfig, BlobStore};
use tensor_cache::{Cache, CacheConfig};
use tensor_chain::{
    ClusterNodeConfig, ClusterOrchestrator, ClusterPeerConfig, OrchestratorConfig, TensorChain,
};

/// When the `query_router` crate is built in `cfg(test)` mode, default the
/// [`OrchestratorConfig`] to development security so unit tests don't need to
/// provision TLS certs. Matches `tensor_chain`'s own internal pattern in
/// `tensor_chain/src/cluster.rs` (`cfg!(test)` fallback at line 289).
#[cfg(test)]
const fn maybe_dev_security(cfg: OrchestratorConfig) -> OrchestratorConfig {
    cfg.with_security_mode(tensor_chain::SecurityMode::Development)
}
/// Production no-op (cluster security mode is configured via the caller).
#[cfg(not(test))]
const fn maybe_dev_security(cfg: OrchestratorConfig) -> OrchestratorConfig {
    cfg
}
use tensor_checkpoint::{
    CheckpointConfig, CheckpointManager, ConfirmationHandler, FileCheckpointStore,
};
use tensor_store::{ConsistentHashConfig, ConsistentHashPartitioner};
use tensor_vault::{Vault, VaultConfig};
use tokio::runtime::Runtime;

use crate::distributed::QueryPlanner;
use crate::{QueryRouter, Result, RouterError};

impl QueryRouter {
    /// Get reference to vault (if initialized).
    pub fn vault(&self) -> Option<&Vault> {
        self.vault.as_deref()
    }

    /// Initialize the vault with a master key.
    ///
    /// # Errors
    ///
    /// Returns an error if the vault fails to initialize with the provided key.
    pub fn init_vault(&mut self, master_key: &[u8]) -> Result<()> {
        let vault = Vault::new(
            master_key,
            Arc::clone(&self.graph),
            self.vector.store().clone(),
            VaultConfig::default(),
        )?;
        self.vault = Some(Arc::new(vault));
        Ok(())
    }

    /// Get reference to cache (if initialized).
    pub fn cache(&self) -> Option<&Cache> {
        self.cache.as_deref()
    }

    /// Initialize the LLM response cache with default configuration.
    pub fn init_cache(&mut self) {
        self.cache = Some(Arc::new(Cache::new()));
    }

    /// Initialize the LLM response cache with default configuration (returns Result).
    ///
    /// # Errors
    ///
    /// This method currently always succeeds but returns `Result` for API consistency.
    pub fn init_cache_default(&mut self) -> Result<()> {
        self.cache = Some(Arc::new(Cache::new()));
        Ok(())
    }

    /// Initialize the LLM response cache with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache configuration is invalid.
    pub fn init_cache_with_config(&mut self, config: CacheConfig) -> Result<()> {
        let cache =
            Cache::with_config(config).map_err(|e| RouterError::CacheError(e.to_string()))?;
        self.cache = Some(Arc::new(cache));
        Ok(())
    }

    /// Get reference to blob store (if initialized).
    pub const fn blob(&self) -> Option<&Arc<tokio::sync::Mutex<BlobStore>>> {
        self.blob.as_ref()
    }

    // ========== Auto-Initialization Methods ==========

    /// Ensure vault is initialized, auto-initializing from `NEUMANN_VAULT_KEY` if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if vault cannot be initialized (no key available or init fails).
    ///
    /// # Panics
    ///
    /// Panics if vault is `None` after successful initialization (should never happen).
    pub fn ensure_vault(&mut self) -> Result<&Vault> {
        if self.vault.is_none() {
            if let Ok(key) = std::env::var("NEUMANN_VAULT_KEY") {
                self.init_vault(key.as_bytes())?;
            } else {
                return Err(RouterError::VaultError(
                    "Vault not initialized. Set NEUMANN_VAULT_KEY env var or call init_vault()"
                        .to_string(),
                ));
            }
        }
        Ok(self.vault.as_deref().unwrap())
    }

    /// Ensure cache is initialized, auto-initializing with defaults if needed.
    ///
    /// # Panics
    ///
    /// Panics if cache is `None` after initialization (should never happen).
    pub fn ensure_cache(&mut self) -> &Cache {
        if self.cache.is_none() {
            self.init_cache();
        }
        self.cache.as_deref().unwrap()
    }

    /// Ensure blob store is initialized, auto-initializing with defaults if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if blob store initialization fails.
    ///
    /// # Panics
    ///
    /// Panics if blob store is `None` after successful initialization (should never happen).
    pub fn ensure_blob(&mut self) -> Result<&Arc<tokio::sync::Mutex<BlobStore>>> {
        if self.blob.is_none() {
            self.init_blob()?;
        }
        Ok(self.blob.as_ref().unwrap())
    }

    /// Initialize the blob store with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the blob store fails to initialize.
    pub fn init_blob(&mut self) -> Result<()> {
        self.init_blob_with_config(BlobConfig::default())
    }

    /// Initialize the blob store with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the blob store fails to initialize or runtime creation fails.
    pub fn init_blob_with_config(&mut self, config: BlobConfig) -> Result<()> {
        // Use a multi-threaded runtime with a bounded worker pool so spawned background
        // tasks (e.g. GC) can run concurrently with block_on calls. The dedicated worker
        // count keeps test parallelism from exhausting OS thread limits when many routers
        // each spin up their own blob runtime.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| RouterError::BlobError(format!("Failed to create runtime: {e}")))?;

        // Create blob store
        let store = self.vector.store().clone();
        let blob_store = runtime
            .block_on(async { BlobStore::new(store, config).await })
            .map_err(|e| RouterError::BlobError(e.to_string()))?;

        self.blob = Some(Arc::new(tokio::sync::Mutex::new(blob_store)));
        self.blob_runtime = Some(Arc::new(runtime));
        Ok(())
    }

    /// Start the blob store background tasks (GC).
    ///
    /// # Errors
    ///
    /// Returns an error if blob store is not initialized or GC start fails.
    pub fn start_blob(&mut self) -> Result<()> {
        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;
        let runtime = self
            .blob_runtime
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob runtime not initialized".to_string()))?;

        runtime.block_on(async {
            let mut blob_guard = blob.lock().await;
            blob_guard.start().await
        })?;
        Ok(())
    }

    /// Shutdown the blob store gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails.
    pub fn shutdown_blob(&mut self) -> Result<()> {
        if let (Some(blob), Some(runtime)) = (self.blob.as_ref(), self.blob_runtime.as_ref()) {
            runtime.block_on(async {
                let mut blob_guard = blob.lock().await;
                blob_guard.shutdown().await
            })?;
        }
        Ok(())
    }

    /// Initialize cluster mode and connect to peers.
    ///
    /// This starts the cluster orchestrator with Raft consensus, TCP transport,
    /// and membership management.
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `bind_addr` - Address to bind for incoming connections
    /// * `peers` - List of (`node_id`, address) tuples for peer nodes
    ///
    /// # Errors
    ///
    /// Returns an error if cluster initialization fails.
    pub fn init_cluster(
        &mut self,
        node_id: &str,
        bind_addr: SocketAddr,
        peers: &[(String, SocketAddr)],
    ) -> Result<()> {
        self.init_cluster_with_executor(node_id, bind_addr, peers, None)
    }

    /// Initialize cluster mode with WAL-based persistence for crash recovery.
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `bind_addr` - Address to bind for incoming connections
    /// * `peers` - List of (`node_id`, address) tuples for peer nodes
    /// * `wal_dir` - Directory for WAL files
    ///
    /// # Errors
    ///
    /// Returns an error if cluster initialization fails.
    pub fn init_cluster_with_wal(
        &mut self,
        node_id: &str,
        bind_addr: SocketAddr,
        peers: &[(String, SocketAddr)],
        wal_dir: &std::path::Path,
    ) -> Result<()> {
        if self.cluster.is_some() {
            return Err(RouterError::InvalidArgument(
                "Cluster already initialized".to_string(),
            ));
        }

        let runtime = Runtime::new()
            .map_err(|e| RouterError::ChainError(format!("Failed to create runtime: {e}")))?;

        let local_config = ClusterNodeConfig::new(node_id, bind_addr);
        let peer_configs: Vec<ClusterPeerConfig> = peers
            .iter()
            .map(|(id, addr)| ClusterPeerConfig::new(id.clone(), *addr))
            .collect();

        let config = maybe_dev_security(
            OrchestratorConfig::new(local_config, peer_configs).with_wal_dir(wal_dir.to_path_buf()),
        );

        let orchestrator = runtime
            .block_on(ClusterOrchestrator::start(config))
            .map_err(|e| RouterError::ChainError(e.to_string()))?;

        let all_nodes: Vec<String> = std::iter::once(node_id.to_string())
            .chain(peers.iter().map(|(id, _)| id.clone()))
            .collect();

        let hash_config = ConsistentHashConfig::new(node_id);
        let partitioner = ConsistentHashPartitioner::with_nodes(hash_config, all_nodes.clone());

        let local_shard = all_nodes.iter().position(|n| n == node_id).unwrap_or(0);

        let planner = QueryPlanner::new(Arc::new(partitioner), local_shard);

        self.cluster = Some(Arc::new(orchestrator));
        self.cluster_runtime = Some(Arc::new(runtime));
        self.distributed_planner = Some(Arc::new(planner));
        self.local_shard_id = local_shard;
        Ok(())
    }

    /// Initialize cluster mode with an optional query executor.
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `bind_addr` - Address to bind for incoming connections
    /// * `peers` - List of (`node_id`, address) tuples for peer nodes
    /// * `executor` - Optional query executor for handling distributed queries
    ///
    /// # Errors
    ///
    /// Returns an error if cluster is already initialized or startup fails.
    pub fn init_cluster_with_executor(
        &mut self,
        node_id: &str,
        bind_addr: SocketAddr,
        peers: &[(String, SocketAddr)],
        executor: Option<Arc<dyn tensor_chain::QueryExecutor>>,
    ) -> Result<()> {
        if self.cluster.is_some() {
            return Err(RouterError::InvalidArgument(
                "Cluster already initialized".to_string(),
            ));
        }

        // Create runtime for async cluster operations
        let runtime = Runtime::new()
            .map_err(|e| RouterError::ChainError(format!("Failed to create runtime: {e}")))?;

        // Build cluster configuration
        let local_config = ClusterNodeConfig::new(node_id, bind_addr);
        let peer_configs: Vec<ClusterPeerConfig> = peers
            .iter()
            .map(|(id, addr)| ClusterPeerConfig::new(id.clone(), *addr))
            .collect();

        let config = maybe_dev_security(OrchestratorConfig::new(local_config, peer_configs));

        // Start cluster orchestrator
        let orchestrator = runtime
            .block_on(ClusterOrchestrator::start(config))
            .map_err(|e| RouterError::ChainError(e.to_string()))?;

        // Register query executor if provided
        if let Some(exec) = executor {
            orchestrator.register_query_executor(exec);
        }

        // Create consistent hash partitioner with all nodes
        let all_nodes: Vec<String> = std::iter::once(node_id.to_string())
            .chain(peers.iter().map(|(id, _)| id.clone()))
            .collect();

        let hash_config = ConsistentHashConfig::new(node_id);
        let partitioner = ConsistentHashPartitioner::with_nodes(hash_config, all_nodes.clone());

        // Determine local shard ID (index of this node in the sorted list)
        let local_shard = all_nodes.iter().position(|n| n == node_id).unwrap_or(0);

        // Create query planner
        let planner = QueryPlanner::new(Arc::new(partitioner), local_shard);

        self.cluster = Some(Arc::new(orchestrator));
        self.cluster_runtime = Some(Arc::new(runtime));
        self.distributed_planner = Some(Arc::new(planner));
        self.local_shard_id = local_shard;
        Ok(())
    }

    /// Shutdown the cluster gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if cluster shutdown fails.
    pub fn shutdown_cluster(&mut self) -> Result<()> {
        if let (Some(cluster), Some(runtime)) =
            (self.cluster.as_ref(), self.cluster_runtime.as_ref())
        {
            runtime
                .block_on(cluster.shutdown())
                .map_err(|e| RouterError::ChainError(e.to_string()))?;
        }
        self.cluster = None;
        self.cluster_runtime = None;
        Ok(())
    }

    /// Check if cluster mode is active.
    pub const fn is_cluster_active(&self) -> bool {
        self.cluster.is_some()
    }

    /// Get reference to cluster orchestrator (if initialized).
    pub const fn cluster(&self) -> Option<&Arc<ClusterOrchestrator>> {
        self.cluster.as_ref()
    }

    /// Get reference to checkpoint manager (if initialized).
    pub const fn checkpoint(&self) -> Option<&Arc<CheckpointManager>> {
        self.checkpoint.as_ref()
    }

    /// Set the directory used for checkpoint file storage.
    pub fn set_checkpoint_dir(&mut self, dir: PathBuf) {
        self.checkpoint_dir = Some(dir);
    }

    /// Get the configured checkpoint directory, if any.
    pub fn checkpoint_dir(&self) -> Option<&Path> {
        self.checkpoint_dir.as_deref()
    }

    /// Initialize the checkpoint manager with default configuration.
    ///
    /// Requires checkpoint directory to be set first.
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint directory is not set.
    pub fn init_checkpoint(&mut self) -> Result<()> {
        self.init_checkpoint_with_config(CheckpointConfig::default())
    }

    /// Initialize the checkpoint manager with custom configuration.
    ///
    /// Requires checkpoint directory to be set first via [`Self::set_checkpoint_dir`].
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint directory is not set.
    pub fn init_checkpoint_with_config(&mut self, config: CheckpointConfig) -> Result<()> {
        let dir = self.checkpoint_dir.as_ref().ok_or_else(|| {
            RouterError::CheckpointError(
                "Checkpoint directory must be set before initializing checkpoint manager"
                    .to_string(),
            )
        })?;

        let store = Arc::new(
            FileCheckpointStore::new(dir)
                .map_err(|e| RouterError::CheckpointError(e.to_string()))?,
        );

        let manager = CheckpointManager::new(store, config);
        self.checkpoint = Some(Arc::new(manager));
        Ok(())
    }

    /// Ensure checkpoint manager is initialized, auto-initializing with defaults if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint directory is not set or initialization fails.
    ///
    /// # Panics
    ///
    /// Panics if checkpoint is `None` after successful initialization (should never happen).
    pub fn ensure_checkpoint(&mut self) -> Result<&Arc<CheckpointManager>> {
        if self.checkpoint.is_none() {
            self.init_checkpoint()?;
        }
        Ok(self.checkpoint.as_ref().unwrap())
    }

    /// Check if checkpoint manager is initialized.
    pub const fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// Check if an HNSW index has been built.
    pub const fn has_hnsw_index(&self) -> bool {
        self.hnsw_index.is_some()
    }

    /// Get TLS certificate path from cluster (if connected and TLS configured).
    pub fn tls_cert_path(&self) -> Option<std::path::PathBuf> {
        self.cluster.as_ref().and_then(|c| c.tls_cert_path())
    }

    /// Set a confirmation handler for destructive operations.
    ///
    /// The handler will be called to confirm operations before they execute.
    /// Requires checkpoint manager to be initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint manager is not initialized.
    pub fn set_confirmation_handler(&self, handler: Arc<dyn ConfirmationHandler>) -> Result<()> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;

        checkpoint.set_confirmation_handler(handler);
        Ok(())
    }

    /// Initialize the tensor chain with a node ID.
    ///
    /// # Errors
    ///
    /// Returns an error if chain initialization fails.
    pub fn init_chain(&mut self, node_id: &str) -> Result<()> {
        let store = self.vector.store().clone();
        let chain = TensorChain::new(store, node_id);
        chain.initialize()?;
        self.chain = Some(Arc::new(chain));
        Ok(())
    }

    /// Get a reference to the chain if initialized.
    pub const fn chain(&self) -> Option<&Arc<TensorChain>> {
        self.chain.as_ref()
    }

    /// Ensure chain is initialized, auto-initializing with default node ID if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if chain initialization fails.
    ///
    /// # Panics
    ///
    /// Panics if chain is None after successful initialization (should never happen).
    pub fn ensure_chain(&mut self) -> Result<&Arc<TensorChain>> {
        if self.chain.is_none() {
            self.init_chain("default_node")?;
        }
        Ok(self.chain.as_ref().unwrap())
    }

    /// Set the current identity for vault access control.
    /// This authenticates the session for vault operations.
    pub fn set_identity(&mut self, identity: &str) {
        self.current_identity = Some(identity.to_string());
    }

    /// Clear the current identity.
    ///
    /// Must be called after authenticated execution to prevent identity from
    /// bleeding into subsequent anonymous requests on the shared router.
    pub fn clear_identity(&mut self) {
        self.current_identity = None;
    }

    /// Get the current identity.
    /// Returns `None` if not authenticated.
    pub fn current_identity(&self) -> Option<&str> {
        self.current_identity.as_deref()
    }

    /// Check if the router is authenticated.
    pub const fn is_authenticated(&self) -> bool {
        self.current_identity.is_some()
    }
}
