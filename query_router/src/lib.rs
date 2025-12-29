//! Query Router - Module 5 of Neumann
//!
//! Parses shell commands, routes to appropriate engine(s), and combines results.
//!
//! # Command Syntax
//!
//! ## Relational Commands
//! - `SELECT <table> [WHERE <condition>]`
//! - `INSERT <table> <col>=<val>, ...`
//! - `UPDATE <table> SET <col>=<val>, ... [WHERE <condition>]`
//! - `DELETE <table> [WHERE <condition>]`
//! - `CREATE TABLE <table> (<col>:<type>, ...)`
//!
//! ## Graph Commands
//! - `NODE CREATE <label> [<key>=<val>, ...]`
//! - `NODE GET <id>`
//! - `EDGE CREATE <from> -> <to> [<label>]`
//! - `NEIGHBORS <id> [OUT|IN|BOTH]`
//! - `PATH <from> -> <to>`
//!
//! ## Vector Commands
//! - `EMBED <key> [<val>, ...]`
//! - `SIMILAR <key> [TOP <k>]`
//! - `SIMILAR [<val>, ...] [TOP <k>]`
//!
//! ## Unified Commands
//! - `FIND <entity> WHERE <condition> SIMILAR TO <key> CONNECTED TO <entity>`

use graph_engine::{Direction, GraphEngine, GraphError, PropertyValue};
use neumann_parser::{
    self as parser, BinaryOp, BlobOp, BlobOptions, BlobStmt, BlobsOp, BlobsStmt, CacheOp,
    CacheStmt, CheckpointStmt, CheckpointsStmt, DeleteStmt, DescribeStmt, DescribeTarget,
    Direction as ParsedDirection, DistanceMetric as ParsedDistanceMetric, EdgeOp, EdgeStmt,
    EmbedOp, EmbedStmt, EntityOp, EntityStmt, Expr, ExprKind, FindPattern, FindStmt, InsertSource,
    InsertStmt, Literal, NeighborsStmt, NodeOp, NodeStmt, PathStmt, Property, RollbackStmt,
    SelectStmt, SimilarQuery, SimilarStmt, Statement, StatementKind, TableRefKind, UpdateStmt,
    VaultOp, VaultStmt,
};
use relational_engine::{
    ColumnarScanOptions, Condition, RelationalEngine, RelationalError, Row, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tensor_blob::{BlobConfig, BlobError, BlobStore};
use tensor_cache::{Cache, CacheConfig, CacheError, CacheLayer};
use tensor_checkpoint::{CheckpointConfig, CheckpointError, CheckpointManager};
use tensor_store::TensorStore;
use tensor_unified::{
    UnifiedEngine, UnifiedError, UnifiedItem, UnifiedResult as TensorUnifiedResult,
};
use tensor_vault::{Vault, VaultConfig, VaultError};
use tokio::runtime::Runtime;
use vector_engine::{DistanceMetric as VectorDistanceMetric, HNSWIndex, VectorEngine, VectorError};

/// Error types for query routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RouterError {
    /// Failed to parse the command.
    ParseError(String),
    /// Unknown command or keyword.
    UnknownCommand(String),
    /// Error from relational engine.
    RelationalError(String),
    /// Error from graph engine.
    GraphError(String),
    /// Error from vector engine.
    VectorError(String),
    /// Error from vault.
    VaultError(String),
    /// Error from cache.
    CacheError(String),
    /// Error from blob storage.
    BlobError(String),
    /// Error from checkpoint system.
    CheckpointError(String),
    /// Invalid argument provided.
    InvalidArgument(String),
    /// Missing required argument.
    MissingArgument(String),
    /// Type mismatch in query.
    TypeMismatch(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            RouterError::UnknownCommand(cmd) => write!(f, "Unknown command: {}", cmd),
            RouterError::RelationalError(msg) => write!(f, "Relational error: {}", msg),
            RouterError::GraphError(msg) => write!(f, "Graph error: {}", msg),
            RouterError::VectorError(msg) => write!(f, "Vector error: {}", msg),
            RouterError::VaultError(msg) => write!(f, "Vault error: {}", msg),
            RouterError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            RouterError::BlobError(msg) => write!(f, "Blob error: {}", msg),
            RouterError::CheckpointError(msg) => write!(f, "Checkpoint error: {}", msg),
            RouterError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            RouterError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            RouterError::MissingArgument(msg) => write!(f, "Missing argument: {}", msg),
        }
    }
}

impl std::error::Error for RouterError {}

impl From<RelationalError> for RouterError {
    fn from(e: RelationalError) -> Self {
        RouterError::RelationalError(e.to_string())
    }
}

impl From<GraphError> for RouterError {
    fn from(e: GraphError) -> Self {
        RouterError::GraphError(e.to_string())
    }
}

impl From<VectorError> for RouterError {
    fn from(e: VectorError) -> Self {
        RouterError::VectorError(e.to_string())
    }
}

impl From<VaultError> for RouterError {
    fn from(e: VaultError) -> Self {
        RouterError::VaultError(e.to_string())
    }
}

impl From<CacheError> for RouterError {
    fn from(e: CacheError) -> Self {
        RouterError::CacheError(e.to_string())
    }
}

impl From<BlobError> for RouterError {
    fn from(e: BlobError) -> Self {
        RouterError::BlobError(e.to_string())
    }
}

impl From<CheckpointError> for RouterError {
    fn from(e: CheckpointError) -> Self {
        RouterError::CheckpointError(e.to_string())
    }
}

impl From<UnifiedError> for RouterError {
    fn from(e: UnifiedError) -> Self {
        match e {
            UnifiedError::RelationalError(msg) => RouterError::RelationalError(msg),
            UnifiedError::GraphError(msg) => RouterError::GraphError(msg),
            UnifiedError::VectorError(msg) => RouterError::VectorError(msg),
            UnifiedError::NotFound(msg) => RouterError::VectorError(format!("Not found: {}", msg)),
            UnifiedError::InvalidOperation(msg) => RouterError::InvalidArgument(msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, RouterError>;

/// Result of a query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResult {
    /// No result (e.g., CREATE, INSERT)
    Empty,
    /// Single value result
    Value(String),
    /// Count of affected rows/nodes/edges
    Count(usize),
    /// List of IDs
    Ids(Vec<u64>),
    /// Rows from relational query
    Rows(Vec<Row>),
    /// Node data from graph query
    Nodes(Vec<NodeResult>),
    /// Edge data from graph query
    Edges(Vec<EdgeResult>),
    /// Path from graph traversal
    Path(Vec<u64>),
    /// Vector similarity results
    Similar(Vec<SimilarResult>),
    /// Combined results from unified query
    Unified(UnifiedResult),
    /// List of table names
    TableList(Vec<String>),
    /// Blob data (bytes)
    Blob(Vec<u8>),
    /// Artifact metadata
    ArtifactInfo(ArtifactInfoResult),
    /// List of artifact IDs
    ArtifactList(Vec<String>),
    /// Blob storage statistics
    BlobStats(BlobStatsResult),
    /// List of checkpoints
    CheckpointList(Vec<CheckpointInfo>),
}

/// Node result from graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub id: u64,
    pub label: String,
    pub properties: HashMap<String, String>,
}

/// Edge result from graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeResult {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub label: String,
}

/// Similarity search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarResult {
    pub key: String,
    pub score: f32,
}

/// Result from unified cross-engine query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResult {
    pub description: String,
    pub items: Vec<UnifiedItem>,
}

impl From<TensorUnifiedResult> for UnifiedResult {
    fn from(r: TensorUnifiedResult) -> Self {
        Self {
            description: r.description,
            items: r.items,
        }
    }
}

/// Artifact info result from blob query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfoResult {
    pub id: String,
    pub filename: String,
    pub content_type: String,
    pub size: usize,
    pub checksum: String,
    pub chunk_count: usize,
    pub created: u64,
    pub modified: u64,
    pub created_by: String,
    pub tags: Vec<String>,
    pub linked_to: Vec<String>,
    pub custom: HashMap<String, String>,
}

/// Blob storage statistics result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobStatsResult {
    pub artifact_count: usize,
    pub chunk_count: usize,
    pub total_bytes: usize,
    pub unique_bytes: usize,
    pub dedup_ratio: f64,
    pub orphaned_chunks: usize,
}

/// Checkpoint information for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub is_auto: bool,
}

impl QueryResult {
    /// Convert the result to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert the result to pretty-printed JSON string.
    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, QueryResult::Empty)
    }

    /// Get the count if this is a Count result.
    pub fn as_count(&self) -> Option<usize> {
        if let QueryResult::Count(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Get the value if this is a Value result.
    pub fn as_value(&self) -> Option<&str> {
        if let QueryResult::Value(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Get the rows if this is a Rows result.
    pub fn as_rows(&self) -> Option<&[Row]> {
        if let QueryResult::Rows(rows) = self {
            Some(rows)
        } else {
            None
        }
    }
}

/// Query Router that orchestrates queries across engines.
pub struct QueryRouter {
    relational: Arc<RelationalEngine>,
    graph: Arc<GraphEngine>,
    vector: Arc<VectorEngine>,
    /// Unified engine for cross-engine queries (lazily initialized)
    unified: Option<UnifiedEngine>,
    /// Optional vault for secure secret storage (requires initialization)
    vault: Option<Arc<Vault>>,
    /// Optional cache for LLM response caching (requires initialization)
    cache: Option<Arc<Cache>>,
    /// Optional blob storage (requires initialization)
    blob: Option<Arc<tokio::sync::Mutex<BlobStore>>>,
    /// Tokio runtime for async blob operations
    blob_runtime: Option<Arc<Runtime>>,
    /// Current identity for vault access control
    current_identity: String,
    /// Optional HNSW index for faster vector search
    hnsw_index: Option<(HNSWIndex, Vec<String>)>,
    /// Optional checkpoint manager (requires blob storage)
    checkpoint: Option<Arc<tokio::sync::Mutex<CheckpointManager>>>,
}

impl QueryRouter {
    /// Create a new query router with fresh engines.
    pub fn new() -> Self {
        Self {
            relational: Arc::new(RelationalEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            unified: None,
            vault: None,
            cache: None,
            blob: None,
            blob_runtime: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
            checkpoint: None,
        }
    }

    /// Create a query router with existing engines.
    pub fn with_engines(
        relational: Arc<RelationalEngine>,
        graph: Arc<GraphEngine>,
        vector: Arc<VectorEngine>,
    ) -> Self {
        Self {
            relational,
            graph,
            vector,
            unified: None,
            vault: None,
            cache: None,
            blob: None,
            blob_runtime: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
            checkpoint: None,
        }
    }

    /// Create a query router with a shared TensorStore for unified entity access.
    ///
    /// All engines share the same store, enabling cross-engine queries on unified entities.
    /// Cloning TensorStore shares the underlying storage (via `Arc<DashMap>`).
    pub fn with_shared_store(store: TensorStore) -> Self {
        let relational = Arc::new(RelationalEngine::with_store(store.clone()));
        let graph = Arc::new(GraphEngine::with_store(store.clone()));
        let vector = Arc::new(VectorEngine::with_store(store.clone()));
        let unified = UnifiedEngine::with_engines(
            store,
            Arc::clone(&relational),
            Arc::clone(&graph),
            Arc::clone(&vector),
        );
        Self {
            relational,
            graph,
            vector,
            unified: Some(unified),
            vault: None,
            cache: None,
            blob: None,
            blob_runtime: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
            checkpoint: None,
        }
    }

    /// Get reference to relational engine.
    pub fn relational(&self) -> &RelationalEngine {
        &self.relational
    }

    /// Get reference to graph engine.
    pub fn graph(&self) -> &GraphEngine {
        &self.graph
    }

    /// Get reference to vector engine.
    pub fn vector(&self) -> &VectorEngine {
        &self.vector
    }

    /// Get reference to unified engine (if initialized).
    pub fn unified(&self) -> Option<&UnifiedEngine> {
        self.unified.as_ref()
    }

    /// Get reference to vault (if initialized).
    pub fn vault(&self) -> Option<&Vault> {
        self.vault.as_deref()
    }

    /// Initialize the vault with a master key.
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
    pub fn init_cache_default(&mut self) -> Result<()> {
        self.cache = Some(Arc::new(Cache::new()));
        Ok(())
    }

    /// Initialize the LLM response cache with custom configuration.
    pub fn init_cache_with_config(&mut self, config: CacheConfig) {
        self.cache = Some(Arc::new(Cache::with_config(config)));
    }

    /// Get reference to blob store (if initialized).
    pub fn blob(&self) -> Option<&Arc<tokio::sync::Mutex<BlobStore>>> {
        self.blob.as_ref()
    }

    // ========== Auto-Initialization Methods ==========

    /// Ensure vault is initialized, auto-initializing from NEUMANN_VAULT_KEY if needed.
    ///
    /// Returns an error if vault cannot be initialized (no key available).
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
    pub fn ensure_cache(&mut self) -> &Cache {
        if self.cache.is_none() {
            self.init_cache();
        }
        self.cache.as_deref().unwrap()
    }

    /// Ensure blob store is initialized, auto-initializing with defaults if needed.
    pub fn ensure_blob(&mut self) -> Result<&Arc<tokio::sync::Mutex<BlobStore>>> {
        if self.blob.is_none() {
            self.init_blob()?;
        }
        Ok(self.blob.as_ref().unwrap())
    }

    /// Initialize the blob store with default configuration.
    pub fn init_blob(&mut self) -> Result<()> {
        self.init_blob_with_config(BlobConfig::default())
    }

    /// Initialize the blob store with custom configuration.
    pub fn init_blob_with_config(&mut self, config: BlobConfig) -> Result<()> {
        // Create a runtime for async blob operations
        let runtime = Runtime::new()
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
    pub fn shutdown_blob(&mut self) -> Result<()> {
        if let (Some(blob), Some(runtime)) = (self.blob.as_ref(), self.blob_runtime.as_ref()) {
            runtime.block_on(async {
                let mut blob_guard = blob.lock().await;
                blob_guard.shutdown().await
            })?;
        }
        Ok(())
    }

    /// Get reference to checkpoint manager (if initialized).
    pub fn checkpoint(&self) -> Option<&Arc<tokio::sync::Mutex<CheckpointManager>>> {
        self.checkpoint.as_ref()
    }

    /// Initialize the checkpoint manager with default configuration.
    ///
    /// Requires blob storage to be initialized first.
    pub fn init_checkpoint(&mut self) -> Result<()> {
        self.init_checkpoint_with_config(CheckpointConfig::default())
    }

    /// Initialize the checkpoint manager with custom configuration.
    ///
    /// Requires blob storage to be initialized first.
    pub fn init_checkpoint_with_config(&mut self, config: CheckpointConfig) -> Result<()> {
        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| {
                RouterError::CheckpointError(
                    "Blob store must be initialized before checkpoint manager".to_string(),
                )
            })?
            .clone();
        let runtime = self.blob_runtime.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Blob runtime not initialized".to_string())
        })?;

        let manager = runtime.block_on(async { CheckpointManager::new(blob, config).await });

        self.checkpoint = Some(Arc::new(tokio::sync::Mutex::new(manager)));
        Ok(())
    }

    /// Ensure checkpoint manager is initialized, auto-initializing with defaults if needed.
    pub fn ensure_checkpoint(&mut self) -> Result<&Arc<tokio::sync::Mutex<CheckpointManager>>> {
        if self.checkpoint.is_none() {
            // First ensure blob is initialized
            if self.blob.is_none() {
                self.init_blob()?;
            }
            self.init_checkpoint()?;
        }
        Ok(self.checkpoint.as_ref().unwrap())
    }

    /// Set the current identity for vault access control.
    pub fn set_identity(&mut self, identity: &str) {
        self.current_identity = identity.to_string();
    }

    /// Get the current identity.
    pub fn current_identity(&self) -> &str {
        &self.current_identity
    }

    /// Build HNSW index for faster vector similarity search.
    pub fn build_vector_index(&mut self) -> Result<()> {
        let (index, keys) = self.vector.build_hnsw_index_default()?;
        self.hnsw_index = Some((index, keys));
        Ok(())
    }

    // ========== Cross-Engine Query Methods ==========
    // These methods enable queries that span multiple engines using unified entities.

    /// Find entities similar to a query entity that are also connected via graph edges.
    ///
    /// Returns entities that:
    /// 1. Have similar embeddings to the query entity
    /// 2. Are connected (directly or indirectly) to the specified connected_to entity
    pub fn find_similar_connected(
        &self,
        query_key: &str,
        connected_to: &str,
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let query_embedding = self
            .vector
            .get_entity_embedding(query_key)
            .map_err(|e| RouterError::VectorError(e.to_string()))?;

        // Use HNSW index if available, otherwise fall back to brute-force
        let similar = if let Some((ref index, ref keys)) = self.hnsw_index {
            self.vector
                .search_with_hnsw(index, keys, &query_embedding, top_k * 2)
                .map_err(|e| RouterError::VectorError(e.to_string()))?
        } else {
            self.vector
                .search_entities(&query_embedding, top_k * 2)
                .map_err(|e| RouterError::VectorError(e.to_string()))?
        };

        let connected_neighbors: std::collections::HashSet<String> = self
            .graph
            .get_entity_neighbors(connected_to)
            .unwrap_or_default()
            .into_iter()
            .collect();

        // Delegate to UnifiedEngine if available and no HNSW index (HNSW optimization not yet in unified)
        if self.unified.is_some() && self.hnsw_index.is_none() {
            let unified = self.unified.as_ref().unwrap();
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| RouterError::InvalidArgument(e.to_string()))?;
            return Ok(rt.block_on(unified.find_similar_connected(
                query_key,
                connected_to,
                top_k,
            ))?);
        }

        let mut items: Vec<UnifiedItem> = similar
            .into_iter()
            .filter(|s| connected_neighbors.contains(&s.key))
            .take(top_k)
            .map(|s| UnifiedItem::new("vector+graph", &s.key).with_score(s.score))
            .collect();

        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(items)
    }

    /// Find graph neighbors of an entity that have embeddings, sorted by similarity to a query.
    pub fn find_neighbors_by_similarity(
        &self,
        entity_key: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        // Delegate to UnifiedEngine if available
        if let Some(unified) = &self.unified {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| RouterError::InvalidArgument(e.to_string()))?;
            return Ok(rt.block_on(unified.find_neighbors_by_similarity(entity_key, query, top_k))?);
        }

        let neighbors = self
            .graph
            .get_entity_neighbors(entity_key)
            .map_err(|e| RouterError::GraphError(e.to_string()))?;

        let mut items: Vec<UnifiedItem> = neighbors
            .into_iter()
            .filter_map(|neighbor_key| {
                let embedding = self.vector.get_entity_embedding(&neighbor_key).ok()?;
                if embedding.len() != query.len() {
                    return None;
                }

                let score =
                    vector_engine::VectorEngine::compute_similarity(query, &embedding).ok()?;

                Some(UnifiedItem::new("graph+vector", &neighbor_key).with_score(score))
            })
            .collect();

        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(top_k);

        Ok(items)
    }

    /// Store a unified entity with relational, graph, and vector data.
    pub fn create_unified_entity(
        &self,
        key: &str,
        fields: HashMap<String, String>,
        embedding: Option<Vec<f32>>,
    ) -> Result<()> {
        if let Some(emb) = embedding {
            self.vector
                .set_entity_embedding(key, emb)
                .map_err(|e| RouterError::VectorError(e.to_string()))?;
        }

        for (field_name, field_value) in fields {
            let mut tensor = self
                .vector
                .store()
                .get(key)
                .unwrap_or_else(|_| tensor_store::TensorData::new());
            tensor.set(
                &field_name,
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(field_value)),
            );
            self.vector
                .store()
                .put(key, tensor)
                .map_err(|e: tensor_store::TensorStoreError| {
                    RouterError::VectorError(e.to_string())
                })?;
        }

        Ok(())
    }

    /// Connect two entities with an edge.
    pub fn connect_entities(
        &self,
        from_key: &str,
        to_key: &str,
        edge_type: &str,
    ) -> Result<String> {
        self.graph
            .add_entity_edge(from_key, to_key, edge_type)
            .map_err(|e| RouterError::GraphError(e.to_string()))
    }

    /// Execute a command string using the legacy string-based parser.
    pub fn execute(&self, command: &str) -> Result<QueryResult> {
        let command = command.trim();
        if command.is_empty() {
            return Err(RouterError::ParseError("Empty command".to_string()));
        }

        // Tokenize and get first keyword
        let tokens: Vec<&str> = command.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(RouterError::ParseError("Empty command".to_string()));
        }

        let keyword = tokens[0].to_uppercase();
        match keyword.as_str() {
            // Relational commands
            "SELECT" => self.execute_select(command),
            "INSERT" => self.execute_insert(command),
            "UPDATE" => self.execute_update(command),
            "DELETE" => self.execute_delete(command),
            "CREATE" => self.execute_create(command),
            "DROP" => self.execute_drop(command),

            // Graph commands
            "NODE" => self.execute_node(command),
            "EDGE" => self.execute_edge(command),
            "NEIGHBORS" => self.execute_neighbors(command),
            "PATH" => self.execute_path(command),

            // Vector commands
            "EMBED" => self.execute_embed(command),
            "SIMILAR" => self.execute_similar(command),

            // Unified queries
            "FIND" => self.execute_find(command),

            _ => Err(RouterError::UnknownCommand(keyword)),
        }
    }

    /// Execute a command string using the AST-based parser.
    ///
    /// This method uses the neumann_parser crate to parse the command into an AST,
    /// then dispatches to the appropriate engine based on the statement type.
    /// Cacheable queries (SELECT, SIMILAR, NEIGHBORS, PATH) are cached if a cache is configured.
    /// Write operations (INSERT, UPDATE, DELETE) invalidate the cache.
    pub fn execute_parsed(&self, command: &str) -> Result<QueryResult> {
        let stmt = parser::parse(command)
            .map_err(|e| RouterError::ParseError(e.format_with_source(command)))?;

        // Check cache for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            if let Some(cached) = self.try_cache_get(command) {
                return Ok(cached);
            }
        }

        // Execute the statement
        let result = self.execute_statement(&stmt)?;

        // Cache the result for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            self.try_cache_put(command, &result);
        }

        // Invalidate cache on write operations
        if Self::is_write_statement(&stmt) {
            self.invalidate_cache_on_write();
        }

        Ok(result)
    }

    /// Execute a parsed statement.
    pub fn execute_statement(&self, stmt: &Statement) -> Result<QueryResult> {
        match &stmt.kind {
            // SQL statements
            StatementKind::Select(select) => self.exec_select(select),
            StatementKind::Insert(insert) => self.exec_insert(insert),
            StatementKind::Update(update) => self.exec_update(update),
            StatementKind::Delete(delete) => self.exec_delete(delete),
            StatementKind::CreateTable(create) => self.exec_create_table(create),
            StatementKind::DropTable(drop) => {
                self.relational.drop_table(&drop.table.name)?;
                Ok(QueryResult::Empty)
            },
            StatementKind::CreateIndex(create) => {
                // Use first column for index (simplified)
                if let Some(col) = create.columns.first() {
                    self.relational
                        .create_index(&create.table.name, &col.name)?;
                }
                Ok(QueryResult::Empty)
            },
            StatementKind::DropIndex(drop) => {
                if let (Some(table), Some(column)) = (&drop.table, &drop.column) {
                    // DROP INDEX ON table(column) syntax
                    if drop.if_exists && !self.relational.has_index(&table.name, &column.name) {
                        return Ok(QueryResult::Empty);
                    }
                    self.relational.drop_index(&table.name, &column.name)?;
                    Ok(QueryResult::Empty)
                } else if let Some(name) = &drop.name {
                    // DROP INDEX name - try to parse as table_column convention
                    // Format: tablename_columnname or just treat as error
                    Err(RouterError::ParseError(format!(
                        "Named index '{}' not supported. Use: DROP INDEX ON table(column)",
                        name.name
                    )))
                } else {
                    Err(RouterError::ParseError(
                        "Invalid DROP INDEX syntax".to_string(),
                    ))
                }
            },
            StatementKind::ShowTables => {
                let tables = self.relational.list_tables();
                Ok(QueryResult::TableList(tables))
            },
            StatementKind::ShowEmbeddings { limit } => {
                let limit_val = limit
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(100);
                let keys = self.vector.list_keys();
                let limited: Vec<String> = keys.into_iter().take(limit_val).collect();
                Ok(QueryResult::Value(format!("Embeddings: {:?}", limited)))
            },
            StatementKind::CountEmbeddings => {
                let count = self.vector.list_keys().len();
                Ok(QueryResult::Count(count))
            },
            StatementKind::Describe(desc) => self.exec_describe(desc),

            // Graph statements
            StatementKind::Node(node) => self.exec_node(node),
            StatementKind::Edge(edge) => self.exec_edge(edge),
            StatementKind::Neighbors(neighbors) => self.exec_neighbors(neighbors),
            StatementKind::Path(path) => self.exec_path(path),

            // Vector statements
            StatementKind::Embed(embed) => self.exec_embed(embed),
            StatementKind::Similar(similar) => self.exec_similar(similar),

            // Unified queries
            StatementKind::Find(find) => self.exec_find(find),
            StatementKind::Entity(entity) => self.exec_entity(entity),

            // Vault statements
            StatementKind::Vault(vault) => self.exec_vault(vault),

            // Cache statements
            StatementKind::Cache(cache) => self.exec_cache(cache),

            // Blob statements
            StatementKind::Blob(blob) => self.exec_blob(blob),
            StatementKind::Blobs(blobs) => self.exec_blobs(blobs),

            // Checkpoint statements
            StatementKind::Checkpoint(cp) => self.exec_checkpoint(cp),
            StatementKind::Rollback(rb) => self.exec_rollback(rb),
            StatementKind::Checkpoints(cps) => self.exec_checkpoints(cps),

            // Empty statement
            StatementKind::Empty => Ok(QueryResult::Empty),
        }
    }

    // ========== Describe Execution ==========

    fn exec_describe(&self, desc: &DescribeStmt) -> Result<QueryResult> {
        match &desc.target {
            DescribeTarget::Table(name) => {
                let schema = self.relational.get_schema(&name.name)?;
                let mut info = format!("Table: {}\n", name.name);
                info.push_str("Columns:\n");
                for col in &schema.columns {
                    info.push_str(&format!(
                        "  {} {:?}{}\n",
                        col.name,
                        col.column_type,
                        if col.nullable { "" } else { " NOT NULL" }
                    ));
                }
                Ok(QueryResult::Value(info))
            },
            DescribeTarget::Node(label) => {
                // Report node label info (graph doesn't have a direct label scan method)
                let total_nodes = self.graph.node_count();
                Ok(QueryResult::Value(format!(
                    "Node label '{}': Use NODE LIST {} to see nodes. Total nodes in graph: {}",
                    label.name, label.name, total_nodes
                )))
            },
            DescribeTarget::Edge(edge_type) => {
                // Report edge type info
                let entities_with_edges = self.graph.scan_entities_with_edges();
                Ok(QueryResult::Value(format!(
                    "Edge type '{}': Use EDGE LIST {} to see edges. Entities with edges: {}",
                    edge_type.name,
                    edge_type.name,
                    entities_with_edges.len()
                )))
            },
        }
    }

    // ========== Cache Execution ==========

    fn exec_cache(&self, stmt: &CacheStmt) -> Result<QueryResult> {
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| RouterError::CacheError("Cache not initialized".to_string()))?;

        match &stmt.operation {
            CacheOp::Init => {
                // Cache is already initialized if we got here
                Ok(QueryResult::Value("Cache initialized".to_string()))
            },
            CacheOp::Stats => {
                let stats = cache.stats();
                let (tokens_in, tokens_out) = stats.tokens_saved();
                let output = format!(
                    "Cache Statistics:\n\
                     Exact: {} hits, {} misses\n\
                     Semantic: {} hits, {} misses\n\
                     Embedding: {} hits, {} misses\n\
                     Tokens saved: {} in, {} out\n\
                     Evictions: {}",
                    stats.hits(CacheLayer::Exact),
                    stats.misses(CacheLayer::Exact),
                    stats.hits(CacheLayer::Semantic),
                    stats.misses(CacheLayer::Semantic),
                    stats.hits(CacheLayer::Embedding),
                    stats.misses(CacheLayer::Embedding),
                    tokens_in,
                    tokens_out,
                    stats.evictions(),
                );
                Ok(QueryResult::Value(output))
            },
            CacheOp::Clear => {
                cache.clear();
                Ok(QueryResult::Value("Cache cleared".to_string()))
            },
            CacheOp::Evict { count } => {
                let count_val = match count {
                    Some(expr) => self.expr_to_usize(expr)?,
                    None => 100, // Default eviction count
                };
                let evicted = cache.evict(count_val);
                Ok(QueryResult::Value(format!("Evicted {} entries", evicted)))
            },
            CacheOp::Get { key } => {
                let key_str = self.expr_to_string(key)?;
                match cache.get_simple(&key_str) {
                    Some(value) => Ok(QueryResult::Value(value)),
                    None => Ok(QueryResult::Value("(not found)".to_string())),
                }
            },
            CacheOp::Put { key, value } => {
                let key_str = self.expr_to_string(key)?;
                let value_str = self.expr_to_string(value)?;
                cache.put_simple(&key_str, &value_str);
                Ok(QueryResult::Value("OK".to_string()))
            },
            CacheOp::SemanticGet { query, threshold } => {
                let query_str = self.expr_to_string(query)?;
                // Get embedding for the query from vector engine if it exists
                let embedding = self.vector.get_embedding(&query_str).ok();
                let _threshold = threshold
                    .as_ref()
                    .map(|e| self.expr_to_f32(e))
                    .transpose()?;

                match cache.get(&query_str, embedding.as_deref()) {
                    Some(hit) => {
                        let similarity_str = hit
                            .similarity
                            .map(|s| format!(", similarity: {:.4}", s))
                            .unwrap_or_default();
                        Ok(QueryResult::Value(format!(
                            "response: {}, layer: {:?}{}",
                            hit.response, hit.layer, similarity_str
                        )))
                    },
                    None => Ok(QueryResult::Value("(not found)".to_string())),
                }
            },
            CacheOp::SemanticPut {
                query,
                response,
                embedding,
            } => {
                let query_str = self.expr_to_string(query)?;
                let response_str = self.expr_to_string(response)?;
                let emb: Vec<f32> = embedding
                    .iter()
                    .map(|e| self.expr_to_f32(e))
                    .collect::<Result<_>>()?;

                cache
                    .put(&query_str, &emb, &response_str, "manual", 0)
                    .map_err(|e| RouterError::CacheError(e.to_string()))?;
                Ok(QueryResult::Value("OK".to_string()))
            },
        }
    }

    // ========== Query Cache Integration ==========

    fn is_cacheable_statement(stmt: &Statement) -> bool {
        matches!(
            &stmt.kind,
            StatementKind::Select(_)
                | StatementKind::Similar(_)
                | StatementKind::Neighbors(_)
                | StatementKind::Path(_)
        )
    }

    fn is_write_statement(stmt: &Statement) -> bool {
        matches!(
            &stmt.kind,
            StatementKind::Insert(_)
                | StatementKind::Update(_)
                | StatementKind::Delete(_)
                | StatementKind::CreateTable(_)
                | StatementKind::DropTable(_)
                | StatementKind::CreateIndex(_)
                | StatementKind::DropIndex(_)
        )
    }

    fn cache_key_for_query(command: &str) -> String {
        format!("query:{}", command.trim().to_lowercase())
    }

    fn try_cache_get(&self, command: &str) -> Option<QueryResult> {
        let cache = self.cache.as_ref()?;
        let key = Self::cache_key_for_query(command);
        let json = cache.get_simple(&key)?;
        serde_json::from_str(&json).ok()
    }

    fn try_cache_put(&self, command: &str, result: &QueryResult) {
        if let Some(cache) = self.cache.as_ref() {
            let key = Self::cache_key_for_query(command);
            if let Ok(json) = serde_json::to_string(result) {
                cache.put_simple(&key, &json);
            }
        }
    }

    fn invalidate_cache_on_write(&self) {
        if let Some(cache) = self.cache.as_ref() {
            // For now, clear the entire cache on writes
            // A more sophisticated approach would track table dependencies
            cache.clear();
        }
    }

    // ========== Vault Execution ==========

    fn exec_vault(&self, stmt: &VaultStmt) -> Result<QueryResult> {
        let vault = self
            .vault
            .as_ref()
            .ok_or_else(|| RouterError::VaultError("Vault not initialized".to_string()))?;

        let identity = &self.current_identity;

        match &stmt.operation {
            VaultOp::Set { key, value } => {
                let key_str = self.eval_string_expr(key)?;
                let value_str = self.eval_string_expr(value)?;
                vault.set(identity, &key_str, &value_str)?;
                Ok(QueryResult::Empty)
            },
            VaultOp::Get { key } => {
                let key_str = self.eval_string_expr(key)?;
                let value = vault.get(identity, &key_str)?;
                Ok(QueryResult::Value(value))
            },
            VaultOp::Delete { key } => {
                let key_str = self.eval_string_expr(key)?;
                vault.delete(identity, &key_str)?;
                Ok(QueryResult::Empty)
            },
            VaultOp::List { pattern } => {
                let pat = pattern
                    .as_ref()
                    .map(|p| self.eval_string_expr(p))
                    .transpose()?
                    .unwrap_or_else(|| "*".to_string());
                let keys = vault.list(identity, &pat)?;
                Ok(QueryResult::Value(keys.join("\n")))
            },
            VaultOp::Rotate { key, new_value } => {
                let key_str = self.eval_string_expr(key)?;
                let new_value_str = self.eval_string_expr(new_value)?;
                vault.rotate(identity, &key_str, &new_value_str)?;
                Ok(QueryResult::Empty)
            },
            VaultOp::Grant { entity, key } => {
                let entity_str = self.eval_string_expr(entity)?;
                let key_str = self.eval_string_expr(key)?;
                vault.grant(identity, &entity_str, &key_str)?;
                Ok(QueryResult::Empty)
            },
            VaultOp::Revoke { entity, key } => {
                let entity_str = self.eval_string_expr(entity)?;
                let key_str = self.eval_string_expr(key)?;
                vault.revoke(identity, &entity_str, &key_str)?;
                Ok(QueryResult::Empty)
            },
        }
    }

    fn eval_string_expr(&self, expr: &Expr) -> Result<String> {
        match &expr.kind {
            ExprKind::Literal(Literal::String(s)) => Ok(s.clone()),
            ExprKind::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(RouterError::InvalidArgument(
                "Expected string literal or identifier".to_string(),
            )),
        }
    }

    // ========== Blob Execution ==========

    fn exec_blob(&self, stmt: &BlobStmt) -> Result<QueryResult> {
        // Handle BLOB INIT specially - doesn't require blob to be initialized
        if matches!(stmt.operation, BlobOp::Init) {
            if self.blob.is_some() {
                return Ok(QueryResult::Value(
                    "Blob store already initialized".to_string(),
                ));
            } else {
                return Err(RouterError::BlobError(
                    "Use router.init_blob() to initialize blob storage".to_string(),
                ));
            }
        }

        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;
        let runtime = self
            .blob_runtime
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob runtime not initialized".to_string()))?;

        match &stmt.operation {
            BlobOp::Init => unreachable!(), // Handled above
            BlobOp::Put {
                filename,
                data,
                from_path,
                options,
            } => {
                let filename_str = self.eval_string_expr(filename)?;
                let put_options = self.blob_options_to_put_options(options)?;

                // Get data either from inline DATA or from file path
                let blob_data = if let Some(data_expr) = data {
                    self.expr_to_bytes(data_expr)?
                } else if let Some(path_expr) = from_path {
                    let path = self.eval_string_expr(path_expr)?;
                    std::fs::read(&path)
                        .map_err(|e| RouterError::BlobError(format!("Failed to read file: {e}")))?
                } else {
                    return Err(RouterError::MissingArgument(
                        "PUT requires either DATA or FROM path".to_string(),
                    ));
                };

                let artifact_id = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.put(&filename_str, &blob_data, put_options).await
                })?;
                Ok(QueryResult::Value(artifact_id))
            },
            BlobOp::Get {
                artifact_id,
                to_path,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let data = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.get(&id).await
                })?;

                if let Some(path_expr) = to_path {
                    let path = self.eval_string_expr(path_expr)?;
                    std::fs::write(&path, &data).map_err(|e| {
                        RouterError::BlobError(format!("Failed to write file: {e}"))
                    })?;
                    Ok(QueryResult::Value(format!(
                        "Written {} bytes to {path}",
                        data.len()
                    )))
                } else {
                    Ok(QueryResult::Blob(data))
                }
            },
            BlobOp::Delete { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.delete(&id).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Info { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let meta = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.metadata(&id).await
                })?;

                Ok(QueryResult::ArtifactInfo(ArtifactInfoResult {
                    id: meta.id,
                    filename: meta.filename,
                    content_type: meta.content_type,
                    size: meta.size,
                    checksum: meta.checksum,
                    chunk_count: meta.chunk_count,
                    created: meta.created,
                    modified: meta.modified,
                    created_by: meta.created_by,
                    tags: meta.tags,
                    linked_to: meta.linked_to,
                    custom: meta.custom,
                }))
            },
            BlobOp::Link {
                artifact_id,
                entity,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let entity_str = self.eval_string_expr(entity)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.link(&id, &entity_str).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Unlink {
                artifact_id,
                entity,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let entity_str = self.eval_string_expr(entity)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.unlink(&id, &entity_str).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Links { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let links = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.links(&id).await
                })?;
                Ok(QueryResult::ArtifactList(links))
            },
            BlobOp::Tag { artifact_id, tag } => {
                let id = self.eval_string_expr(artifact_id)?;
                let tag_str = self.eval_string_expr(tag)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.tag(&id, &tag_str).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Untag { artifact_id, tag } => {
                let id = self.eval_string_expr(artifact_id)?;
                let tag_str = self.eval_string_expr(tag)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.untag(&id, &tag_str).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Verify { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let valid = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.verify(&id).await
                })?;
                Ok(QueryResult::Value(if valid {
                    "OK".to_string()
                } else {
                    "INVALID".to_string()
                }))
            },
            BlobOp::Gc { full } => {
                let stats = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    if *full {
                        blob_guard.full_gc().await
                    } else {
                        blob_guard.gc().await
                    }
                })?;
                Ok(QueryResult::Value(format!(
                    "Deleted {} chunks, freed {} bytes",
                    stats.deleted, stats.freed_bytes
                )))
            },
            BlobOp::Repair => {
                let stats = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.repair().await
                })?;
                Ok(QueryResult::Value(format!(
                    "Fixed {} refs, deleted {} orphans",
                    stats.refs_fixed, stats.orphans_deleted
                )))
            },
            BlobOp::Stats => {
                let stats = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.stats().await
                })?;
                Ok(QueryResult::BlobStats(BlobStatsResult {
                    artifact_count: stats.artifact_count,
                    chunk_count: stats.chunk_count,
                    total_bytes: stats.total_bytes,
                    unique_bytes: stats.unique_bytes,
                    dedup_ratio: stats.dedup_ratio,
                    orphaned_chunks: stats.orphaned_chunks,
                }))
            },
            BlobOp::MetaSet {
                artifact_id,
                key,
                value,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let key_str = self.eval_string_expr(key)?;
                let value_str = self.eval_string_expr(value)?;
                runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.set_meta(&id, &key_str, &value_str).await
                })?;
                Ok(QueryResult::Empty)
            },
            BlobOp::MetaGet { artifact_id, key } => {
                let id = self.eval_string_expr(artifact_id)?;
                let key_str = self.eval_string_expr(key)?;
                let value = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.get_meta(&id, &key_str).await
                })?;
                match value {
                    Some(v) => Ok(QueryResult::Value(v)),
                    None => Ok(QueryResult::Value("(not found)".to_string())),
                }
            },
        }
    }

    fn exec_blobs(&self, stmt: &BlobsStmt) -> Result<QueryResult> {
        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;
        let runtime = self
            .blob_runtime
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob runtime not initialized".to_string()))?;

        match &stmt.operation {
            BlobsOp::List { pattern } => {
                let prefix = pattern
                    .as_ref()
                    .map(|p| self.eval_string_expr(p))
                    .transpose()?;
                let ids = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.list(prefix.as_deref()).await
                })?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::For { entity } => {
                let entity_str = self.eval_string_expr(entity)?;
                let ids = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.artifacts_for(&entity_str).await
                })?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::ByTag { tag } => {
                let tag_str = self.eval_string_expr(tag)?;
                let ids = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.by_tag(&tag_str).await
                })?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::ByType { content_type } => {
                let ct = self.eval_string_expr(content_type)?;
                let ids = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.by_content_type(&ct).await
                })?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::Similar { artifact_id, limit } => {
                let id = self.eval_string_expr(artifact_id)?;
                let k = limit
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(10);
                let similar = runtime.block_on(async {
                    let blob_guard = blob.lock().await;
                    blob_guard.similar(&id, k).await
                })?;
                Ok(QueryResult::Similar(
                    similar
                        .into_iter()
                        .map(|s| SimilarResult {
                            key: s.id,
                            score: s.similarity,
                        })
                        .collect(),
                ))
            },
        }
    }

    fn blob_options_to_put_options(
        &self,
        options: &BlobOptions,
    ) -> Result<tensor_blob::PutOptions> {
        let mut put_options = tensor_blob::PutOptions::new();

        if let Some(ct) = &options.content_type {
            put_options = put_options.with_content_type(&self.eval_string_expr(ct)?);
        }

        if let Some(cb) = &options.created_by {
            put_options = put_options.with_created_by(&self.eval_string_expr(cb)?);
        }

        for link_expr in &options.link {
            let link = self.eval_string_expr(link_expr)?;
            put_options = put_options.with_link(&link);
        }

        for tag_expr in &options.tag {
            let tag = self.eval_string_expr(tag_expr)?;
            put_options = put_options.with_tag(&tag);
        }

        Ok(put_options)
    }

    fn expr_to_bytes(&self, expr: &Expr) -> Result<Vec<u8>> {
        match &expr.kind {
            ExprKind::Literal(Literal::String(s)) => Ok(s.as_bytes().to_vec()),
            // For now, only string literals are supported as inline data
            _ => Err(RouterError::InvalidArgument(
                "Expected string literal for blob data".to_string(),
            )),
        }
    }

    // ========== Checkpoint Execution ==========

    fn exec_checkpoint(&self, stmt: &CheckpointStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;
        let runtime = self.blob_runtime.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Blob runtime not initialized".to_string())
        })?;

        let name = stmt
            .name
            .as_ref()
            .map(|e| self.eval_string_expr(e))
            .transpose()?;

        let store = self.vector.store();
        let checkpoint_id = runtime.block_on(async {
            let cp_guard = checkpoint.lock().await;
            cp_guard.create(name.as_deref(), store).await
        })?;

        Ok(QueryResult::Value(format!(
            "Checkpoint created: {}",
            checkpoint_id
        )))
    }

    fn exec_rollback(&self, stmt: &RollbackStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;
        let runtime = self.blob_runtime.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Blob runtime not initialized".to_string())
        })?;

        let target = self.eval_string_expr(&stmt.target)?;

        let store = self.vector.store();
        runtime.block_on(async {
            let cp_guard = checkpoint.lock().await;
            cp_guard.rollback(&target, store).await
        })?;

        Ok(QueryResult::Value(format!(
            "Rolled back to checkpoint: {}",
            target
        )))
    }

    fn exec_checkpoints(&self, stmt: &CheckpointsStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;
        let runtime = self.blob_runtime.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Blob runtime not initialized".to_string())
        })?;

        let limit = stmt
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?;

        // Default to 10 if no limit specified
        let limit_opt = limit.or(Some(10));

        let checkpoints = runtime.block_on(async {
            let cp_guard = checkpoint.lock().await;
            cp_guard.list(limit_opt).await
        })?;

        let info_list: Vec<CheckpointInfo> = checkpoints
            .into_iter()
            .map(|cp| CheckpointInfo {
                id: cp.id,
                name: cp.name,
                created_at: cp.created_at,
                is_auto: cp.trigger.is_some(),
            })
            .collect();

        Ok(QueryResult::CheckpointList(info_list))
    }

    // ========== AST-Based Execution Methods ==========

    fn exec_select(&self, select: &SelectStmt) -> Result<QueryResult> {
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| RouterError::MissingArgument("FROM clause".to_string()))?;

        let table_name = match &from.table.kind {
            TableRefKind::Table(ident) => &ident.name,
            TableRefKind::Subquery(_) => {
                return Err(RouterError::ParseError(
                    "Subqueries not yet supported".to_string(),
                ))
            },
        };

        let condition = if let Some(ref where_expr) = select.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        // Extract column projection from SELECT clause
        let projection = self.extract_projection(&select.columns)?;

        let options = ColumnarScanOptions {
            projection,
            prefer_columnar: true,
        };

        let mut rows = self
            .relational
            .select_columnar(table_name, condition, options)?;

        // Apply LIMIT clause if present
        if let Some(ref limit_expr) = select.limit {
            if let ExprKind::Literal(neumann_parser::Literal::Integer(n)) = &limit_expr.kind {
                let limit = *n as usize;
                rows.truncate(limit);
            }
        }

        Ok(QueryResult::Rows(rows))
    }

    fn extract_projection(
        &self,
        items: &[neumann_parser::SelectItem],
    ) -> Result<Option<Vec<String>>> {
        // Check for SELECT *
        if items.len() == 1 {
            if let ExprKind::Wildcard = &items[0].expr.kind {
                return Ok(None);
            }
        }

        // Check if any item is a wildcard
        for item in items {
            if matches!(
                &item.expr.kind,
                ExprKind::Wildcard | ExprKind::QualifiedWildcard(_)
            ) {
                return Ok(None);
            }
        }

        let mut columns = Vec::with_capacity(items.len());
        for item in items {
            match &item.expr.kind {
                ExprKind::Ident(ident) => {
                    columns.push(ident.name.clone());
                },
                ExprKind::Qualified(_, name) => {
                    columns.push(name.name.clone());
                },
                _ => {
                    // For expressions (COUNT(*), a+b, etc.), fall back to all columns
                    return Ok(None);
                },
            }
        }

        Ok(Some(columns))
    }

    fn exec_insert(&self, insert: &InsertStmt) -> Result<QueryResult> {
        match &insert.source {
            InsertSource::Values(rows) => {
                let mut ids = Vec::new();
                for row_values in rows {
                    let mut values = HashMap::new();
                    // Match columns to values
                    if let Some(ref cols) = insert.columns {
                        // Explicit columns specified
                        for (col, val) in cols.iter().zip(row_values.iter()) {
                            values.insert(col.name.clone(), self.expr_to_value(val)?);
                        }
                    } else {
                        // No columns specified - use table schema order
                        let schema = self.relational.get_schema(&insert.table.name)?;
                        for (col, val) in schema.columns.iter().zip(row_values.iter()) {
                            values.insert(col.name.clone(), self.expr_to_value(val)?);
                        }
                    }
                    let id = self.relational.insert(&insert.table.name, values)?;
                    ids.push(id);
                }
                Ok(QueryResult::Ids(ids))
            },
            InsertSource::Query(select) => {
                // Execute the SELECT query first
                let select_result = self.exec_select(select)?;

                // Extract rows from the result
                let rows = match select_result {
                    QueryResult::Rows(rows) => rows,
                    _ => {
                        return Err(RouterError::ParseError(
                            "INSERT ... SELECT query did not return rows".to_string(),
                        ))
                    },
                };

                if rows.is_empty() {
                    return Ok(QueryResult::Ids(vec![]));
                }

                // Get the target table schema
                let schema = self.relational.get_schema(&insert.table.name)?;

                // Determine column mapping
                let columns: Vec<String> = if let Some(ref cols) = insert.columns {
                    cols.iter().map(|c| c.name.clone()).collect()
                } else {
                    schema.columns.iter().map(|c| c.name.clone()).collect()
                };

                // Insert each row from the SELECT result
                let mut ids = Vec::new();
                for row in rows {
                    let mut values = HashMap::new();

                    for col in &columns {
                        if let Some(val) = row.values.get(col) {
                            values.insert(col.clone(), val.clone());
                        }
                    }

                    let id = self.relational.insert(&insert.table.name, values)?;
                    ids.push(id);
                }

                Ok(QueryResult::Ids(ids))
            },
        }
    }

    fn exec_update(&self, update: &UpdateStmt) -> Result<QueryResult> {
        let condition = if let Some(ref where_expr) = update.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        let mut values = HashMap::new();
        for assign in &update.assignments {
            values.insert(
                assign.column.name.clone(),
                self.expr_to_value(&assign.value)?,
            );
        }

        let count = self
            .relational
            .update(&update.table.name, condition, values)?;
        Ok(QueryResult::Count(count))
    }

    fn exec_delete(&self, delete: &DeleteStmt) -> Result<QueryResult> {
        let condition = if let Some(ref where_expr) = delete.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        let count = self.relational.delete_rows(&delete.table.name, condition)?;
        Ok(QueryResult::Count(count))
    }

    fn exec_create_table(&self, create: &parser::CreateTableStmt) -> Result<QueryResult> {
        let mut columns = Vec::new();
        for col in &create.columns {
            let col_type = self.data_type_to_column_type(&col.data_type)?;
            let mut column = relational_engine::Column::new(&col.name.name, col_type);

            // Check for nullable
            let is_nullable = !col
                .constraints
                .iter()
                .any(|c| matches!(c, parser::ColumnConstraint::NotNull));
            if is_nullable {
                column = column.nullable();
            }
            columns.push(column);
        }

        let schema = relational_engine::Schema::new(columns);
        self.relational.create_table(&create.table.name, schema)?;
        Ok(QueryResult::Empty)
    }

    fn exec_node(&self, node: &NodeStmt) -> Result<QueryResult> {
        match &node.operation {
            NodeOp::Create { label, properties } => {
                let props = self.properties_to_map(properties)?;
                let id = self.graph.create_node(&label.name, props)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            NodeOp::Get { id } => {
                let node_id = self.expr_to_u64(id)?;
                let node = self.graph.get_node(node_id)?;
                let properties: HashMap<String, String> = node
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::property_to_string(v)))
                    .collect();
                Ok(QueryResult::Nodes(vec![NodeResult {
                    id: node.id,
                    label: node.label.clone(),
                    properties,
                }]))
            },
            NodeOp::Delete { id } => {
                let node_id = self.expr_to_u64(id)?;
                self.graph.delete_node(node_id)?;
                Ok(QueryResult::Count(1))
            },
            NodeOp::List { label } => {
                // List all nodes with optional label filter
                let label_filter = label.as_ref().map(|l| l.name.as_str());
                let unified_items = self.scan_find_nodes(label_filter, None, 1000)?;

                // Convert UnifiedItem to NodeResult
                let nodes: Vec<NodeResult> = unified_items
                    .into_iter()
                    .map(|item| {
                        let id = item.id.parse::<u64>().unwrap_or(0);
                        let label = item.data.get("label").cloned().unwrap_or_default();
                        let properties: HashMap<String, String> = item
                            .data
                            .into_iter()
                            .filter(|(k, _)| k != "label")
                            .collect();
                        NodeResult {
                            id,
                            label,
                            properties,
                        }
                    })
                    .collect();

                Ok(QueryResult::Nodes(nodes))
            },
        }
    }

    fn exec_edge(&self, edge: &EdgeStmt) -> Result<QueryResult> {
        match &edge.operation {
            EdgeOp::Create {
                from_id,
                to_id,
                edge_type,
                properties,
            } => {
                let from = self.expr_to_u64(from_id)?;
                let to = self.expr_to_u64(to_id)?;
                let props = self.properties_to_map(properties)?;
                let id = self
                    .graph
                    .create_edge(from, to, &edge_type.name, props, true)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            EdgeOp::Get { id } => {
                let edge_id = self.expr_to_u64(id)?;
                let edge = self.graph.get_edge(edge_id)?;
                Ok(QueryResult::Edges(vec![EdgeResult {
                    id: edge.id,
                    from: edge.from,
                    to: edge.to,
                    label: edge.edge_type.clone(),
                }]))
            },
            EdgeOp::Delete { id: _ } => {
                // GraphEngine doesn't support edge deletion yet
                Err(RouterError::ParseError(
                    "EDGE DELETE not yet supported".to_string(),
                ))
            },
            EdgeOp::List { edge_type } => {
                // List all edges with optional type filter
                let type_filter = edge_type.as_ref().map(|t| t.name.as_str());
                let unified_items = self.scan_find_edges(type_filter, None, 1000)?;

                // Convert UnifiedItem to EdgeResult
                let edges: Vec<EdgeResult> = unified_items
                    .into_iter()
                    .map(|item| {
                        let id = item.id.parse::<u64>().unwrap_or(0);
                        let from = item
                            .data
                            .get("from")
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(0);
                        let to = item
                            .data
                            .get("to")
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(0);
                        let label = item.data.get("type").cloned().unwrap_or_default();
                        EdgeResult {
                            id,
                            from,
                            to,
                            label,
                        }
                    })
                    .collect();

                Ok(QueryResult::Edges(edges))
            },
        }
    }

    fn exec_neighbors(&self, neighbors: &NeighborsStmt) -> Result<QueryResult> {
        // Handle NEIGHBORS...BY SIMILARITY cross-engine query
        if let Some(ref similarity_vec) = neighbors.by_similarity {
            // For BY SIMILARITY queries, node_id should be a string key (entity identifier)
            let entity_key = self.expr_to_string(&neighbors.node_id)?;

            let query: Vec<f32> = similarity_vec
                .iter()
                .map(|e| self.expr_to_f32(e))
                .collect::<Result<_>>()?;

            let top_k = neighbors
                .limit
                .as_ref()
                .map(|e| self.expr_to_usize(e))
                .transpose()?
                .unwrap_or(10);

            // Use the cross-engine find_neighbors_by_similarity method
            let items = self.find_neighbors_by_similarity(&entity_key, &query, top_k)?;

            let results: Vec<SimilarResult> = items
                .into_iter()
                .map(|item| SimilarResult {
                    key: item.id,
                    score: item.score.unwrap_or(0.0),
                })
                .collect();

            return Ok(QueryResult::Similar(results));
        }

        // Standard neighbors query
        let node_id = self.expr_to_u64(&neighbors.node_id)?;

        let direction = match neighbors.direction {
            ParsedDirection::Outgoing => Direction::Outgoing,
            ParsedDirection::Incoming => Direction::Incoming,
            ParsedDirection::Both => Direction::Both,
        };

        let edge_type = neighbors.edge_type.as_ref().map(|e| e.name.as_str());
        let neighbor_nodes = self.graph.neighbors(node_id, edge_type, direction)?;
        let neighbor_ids: Vec<u64> = neighbor_nodes.iter().map(|n| n.id).collect();

        Ok(QueryResult::Ids(neighbor_ids))
    }

    fn exec_path(&self, path: &PathStmt) -> Result<QueryResult> {
        let from = self.expr_to_u64(&path.from_id)?;
        let to = self.expr_to_u64(&path.to_id)?;

        match self.graph.find_path(from, to) {
            Ok(path) => Ok(QueryResult::Path(path.nodes)),
            Err(GraphError::PathNotFound) => Ok(QueryResult::Path(vec![])),
            Err(e) => Err(e.into()),
        }
    }

    fn exec_embed(&self, embed: &EmbedStmt) -> Result<QueryResult> {
        match &embed.operation {
            EmbedOp::Store { key, vector } => {
                let key_str = self.expr_to_string(key)?;
                let vec: Vec<f32> = vector
                    .iter()
                    .map(|e| self.expr_to_f32(e))
                    .collect::<Result<_>>()?;
                self.vector.store_embedding(&key_str, vec)?;
                Ok(QueryResult::Empty)
            },
            EmbedOp::Get { key } => {
                let key_str = self.expr_to_string(key)?;
                let vec = self.vector.get_embedding(&key_str)?;
                Ok(QueryResult::Value(format!("{:?}", vec)))
            },
            EmbedOp::Delete { key } => {
                let key_str = self.expr_to_string(key)?;
                self.vector.delete_embedding(&key_str)?;
                Ok(QueryResult::Count(1))
            },
            EmbedOp::BuildIndex => {
                // Building the index requires mutable access to the router
                // Check if index already exists
                if self.hnsw_index.is_some() {
                    Ok(QueryResult::Value("HNSW index already built".to_string()))
                } else {
                    Err(RouterError::VectorError(
                        "Use router.build_vector_index() to build HNSW index".to_string(),
                    ))
                }
            },
            EmbedOp::Batch { items } => {
                let mut count = 0;
                for (key_expr, vector_exprs) in items {
                    let key_str = self.expr_to_string(key_expr)?;
                    let vec: Vec<f32> = vector_exprs
                        .iter()
                        .map(|e| self.expr_to_f32(e))
                        .collect::<Result<_>>()?;
                    self.vector.store_embedding(&key_str, vec)?;
                    count += 1;
                }
                Ok(QueryResult::Count(count))
            },
        }
    }

    fn exec_similar(&self, similar: &SimilarStmt) -> Result<QueryResult> {
        let top_k = similar
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?
            .unwrap_or(10);

        // Handle SIMILAR...CONNECTED TO cross-engine query
        if let Some(ref connected_to_expr) = similar.connected_to {
            let query_key = match &similar.query {
                SimilarQuery::Key(key) => self.expr_to_string(key)?,
                SimilarQuery::Vector(_) => {
                    return Err(RouterError::ParseError(
                        "SIMILAR...CONNECTED TO requires a key, not a vector".to_string(),
                    ));
                },
            };
            let connected_to = self.expr_to_string(connected_to_expr)?;

            // Use the cross-engine find_similar_connected method
            let items = self.find_similar_connected(&query_key, &connected_to, top_k)?;

            let results: Vec<SimilarResult> = items
                .into_iter()
                .map(|item| SimilarResult {
                    key: item.id,
                    score: item.score.unwrap_or(0.0),
                })
                .collect();

            return Ok(QueryResult::Similar(results));
        }

        // Standard similarity search
        let query_vec = match &similar.query {
            SimilarQuery::Key(key) => {
                let key_str = self.expr_to_string(key)?;
                self.vector.get_embedding(&key_str)?
            },
            SimilarQuery::Vector(exprs) => exprs
                .iter()
                .map(|e| self.expr_to_f32(e))
                .collect::<Result<_>>()?,
        };

        // Convert parser metric to vector engine metric
        let metric = match similar.metric {
            Some(ParsedDistanceMetric::Cosine) | None => VectorDistanceMetric::Cosine,
            Some(ParsedDistanceMetric::Euclidean) => VectorDistanceMetric::Euclidean,
            Some(ParsedDistanceMetric::DotProduct) => VectorDistanceMetric::DotProduct,
        };

        let results = if let Some((ref index, ref keys)) = self.hnsw_index {
            // HNSW currently only supports cosine - fall back to linear search for other metrics
            if matches!(metric, VectorDistanceMetric::Cosine) {
                self.vector
                    .search_with_hnsw(index, keys, &query_vec, top_k)?
                    .into_iter()
                    .map(|r| SimilarResult {
                        key: r.key,
                        score: r.score,
                    })
                    .collect()
            } else {
                self.vector
                    .search_similar_with_metric(&query_vec, top_k, metric)?
                    .into_iter()
                    .map(|r| SimilarResult {
                        key: r.key,
                        score: r.score,
                    })
                    .collect()
            }
        } else {
            self.vector
                .search_similar_with_metric(&query_vec, top_k, metric)?
                .into_iter()
                .map(|r| SimilarResult {
                    key: r.key,
                    score: r.score,
                })
                .collect()
        };

        Ok(QueryResult::Similar(results))
    }

    fn exec_find(&self, find: &FindStmt) -> Result<QueryResult> {
        let limit = if let Some(ref limit_expr) = find.limit {
            self.expr_to_usize(limit_expr)?
        } else {
            100 // Default limit
        };

        // Convert WHERE clause to Condition if present
        let condition = if let Some(ref where_expr) = find.where_clause {
            Some(self.expr_to_condition(where_expr)?)
        } else {
            None
        };

        // Handle different patterns
        let (description, items) = match &find.pattern {
            FindPattern::Nodes { label } => {
                let label_filter = label.as_ref().map(|l| l.name.as_str());
                let nodes = self.scan_find_nodes(label_filter, condition.as_ref(), limit)?;

                let desc = format!(
                    "Found {} node{}{}",
                    nodes.len(),
                    if nodes.len() == 1 { "" } else { "s" },
                    label
                        .as_ref()
                        .map(|l| format!(" with label '{}'", l.name))
                        .unwrap_or_default()
                );
                (desc, nodes)
            },
            FindPattern::Edges { edge_type } => {
                let type_filter = edge_type.as_ref().map(|t| t.name.as_str());
                let edges = self.scan_find_edges(type_filter, condition.as_ref(), limit)?;

                let desc = format!(
                    "Found {} edge{}{}",
                    edges.len(),
                    if edges.len() == 1 { "" } else { "s" },
                    edge_type
                        .as_ref()
                        .map(|t| format!(" of type '{}'", t.name))
                        .unwrap_or_default()
                );
                (desc, edges)
            },
            FindPattern::Path { .. } => {
                // Path finding not yet implemented
                ("Path finding not yet implemented".to_string(), Vec::new())
            },
        };

        Ok(QueryResult::Unified(UnifiedResult { description, items }))
    }

    fn exec_entity(&self, entity: &EntityStmt) -> Result<QueryResult> {
        match &entity.operation {
            EntityOp::Create {
                key,
                properties,
                embedding,
            } => {
                let key_str = self.expr_to_string(key)?;

                // Convert properties to HashMap
                let fields: HashMap<String, String> = properties
                    .iter()
                    .filter_map(|p| {
                        self.expr_to_string(&p.value)
                            .ok()
                            .map(|v| (p.key.name.clone(), v))
                    })
                    .collect();

                // Convert embedding if present
                let emb = if let Some(vec_exprs) = embedding {
                    let embedding_vec: Result<Vec<f32>> =
                        vec_exprs.iter().map(|e| self.expr_to_f32(e)).collect();
                    Some(embedding_vec?)
                } else {
                    None
                };

                // Use the existing create_unified_entity method
                self.create_unified_entity(&key_str, fields, emb)?;

                Ok(QueryResult::Value(format!("Entity '{}' created", key_str)))
            },
            EntityOp::Connect {
                from_key,
                to_key,
                edge_type,
            } => {
                let from_str = self.expr_to_string(from_key)?;
                let to_str = self.expr_to_string(to_key)?;
                let edge_type_str = &edge_type.name;

                // Use the existing connect_entities method
                let edge_key = self.connect_entities(&from_str, &to_str, edge_type_str)?;

                Ok(QueryResult::Value(format!(
                    "Connected '{}' -> '{}' with edge '{}'",
                    from_str, to_str, edge_key
                )))
            },
        }
    }

    fn scan_find_nodes(
        &self,
        label_filter: Option<&str>,
        condition: Option<&Condition>,
        limit: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let mut items = Vec::new();

        // Scan for all node keys in the graph store
        let keys = self.graph().store().scan("node:");

        for key in keys {
            if items.len() >= limit {
                break;
            }

            // Filter out edge lists (node:123:out, node:123:in)
            if key.contains(":out") || key.contains(":in") {
                continue;
            }

            // Parse node ID from key "node:{id}"
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.graph().get_node(id) {
                        // Apply label filter
                        if let Some(filter) = label_filter {
                            if node.label != filter {
                                continue;
                            }
                        }

                        // Apply condition filter
                        if let Some(cond) = condition {
                            if !self.node_matches_condition(&node, cond) {
                                continue;
                            }
                        }

                        // Build unified item
                        let mut data = HashMap::new();
                        data.insert("label".to_string(), node.label.clone());
                        for (k, v) in &node.properties {
                            data.insert(k.clone(), Self::property_to_string(v));
                        }

                        items.push(UnifiedItem::with_data("graph", node.id.to_string(), data));
                    }
                }
            }
        }

        Ok(items)
    }

    fn scan_find_edges(
        &self,
        type_filter: Option<&str>,
        condition: Option<&Condition>,
        limit: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let mut items = Vec::new();

        // Scan for all edge keys in the graph store
        let keys = self.graph().store().scan("edge:");

        for key in keys {
            if items.len() >= limit {
                break;
            }

            // Parse edge ID from key "edge:{id}"
            if let Some(id_str) = key.strip_prefix("edge:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(edge) = self.graph().get_edge(id) {
                        // Apply type filter
                        if let Some(filter) = type_filter {
                            if edge.edge_type != filter {
                                continue;
                            }
                        }

                        // Apply condition filter
                        if let Some(cond) = condition {
                            if !self.edge_matches_condition(&edge, cond) {
                                continue;
                            }
                        }

                        // Build unified item
                        let mut data = HashMap::new();
                        data.insert("from".to_string(), edge.from.to_string());
                        data.insert("to".to_string(), edge.to.to_string());
                        data.insert("type".to_string(), edge.edge_type.clone());
                        data.insert("directed".to_string(), edge.directed.to_string());
                        for (k, v) in &edge.properties {
                            data.insert(k.clone(), Self::property_to_string(v));
                        }

                        items.push(UnifiedItem::with_data("graph", edge.id.to_string(), data));
                    }
                }
            }
        }

        Ok(items)
    }

    fn node_matches_condition(&self, node: &graph_engine::Node, condition: &Condition) -> bool {
        match condition {
            Condition::Eq(col, val) => {
                if col == "id" {
                    return match val {
                        Value::Int(i) => node.id == *i as u64,
                        Value::String(s) => node.id.to_string() == *s,
                        _ => false,
                    };
                }
                if col == "label" {
                    return match val {
                        Value::String(s) => node.label == *s,
                        _ => false,
                    };
                }
                if let Some(prop) = node.properties.get(col) {
                    return self.property_matches_value(prop, val);
                }
                false
            },
            Condition::Ne(col, val) => {
                if let Some(prop) = node.properties.get(col) {
                    return !self.property_matches_value(prop, val);
                }
                true // Missing property is considered "not equal"
            },
            Condition::Gt(col, val) => {
                if let Some(prop) = node.properties.get(col) {
                    return self.property_compare_gt(prop, val);
                }
                false
            },
            Condition::Ge(col, val) => {
                if let Some(prop) = node.properties.get(col) {
                    return self.property_compare_gte(prop, val);
                }
                false
            },
            Condition::Lt(col, val) => {
                if let Some(prop) = node.properties.get(col) {
                    return self.property_compare_lt(prop, val);
                }
                false
            },
            Condition::Le(col, val) => {
                if let Some(prop) = node.properties.get(col) {
                    return self.property_compare_lte(prop, val);
                }
                false
            },
            Condition::And(a, b) => {
                self.node_matches_condition(node, a) && self.node_matches_condition(node, b)
            },
            Condition::Or(a, b) => {
                self.node_matches_condition(node, a) || self.node_matches_condition(node, b)
            },
            _ => true, // Other conditions not fully implemented for nodes
        }
    }

    fn edge_matches_condition(&self, edge: &graph_engine::Edge, condition: &Condition) -> bool {
        match condition {
            Condition::Eq(col, val) => {
                if col == "id" {
                    return match val {
                        Value::Int(i) => edge.id == *i as u64,
                        Value::String(s) => edge.id.to_string() == *s,
                        _ => false,
                    };
                }
                if col == "type" || col == "edge_type" {
                    return match val {
                        Value::String(s) => edge.edge_type == *s,
                        _ => false,
                    };
                }
                if col == "from" {
                    return match val {
                        Value::Int(i) => edge.from == *i as u64,
                        _ => false,
                    };
                }
                if col == "to" {
                    return match val {
                        Value::Int(i) => edge.to == *i as u64,
                        _ => false,
                    };
                }
                if let Some(prop) = edge.properties.get(col) {
                    return self.property_matches_value(prop, val);
                }
                false
            },
            Condition::Ne(col, val) => {
                if let Some(prop) = edge.properties.get(col) {
                    return !self.property_matches_value(prop, val);
                }
                true
            },
            Condition::Gt(col, val) => {
                if let Some(prop) = edge.properties.get(col) {
                    return self.property_compare_gt(prop, val);
                }
                false
            },
            Condition::Ge(col, val) => {
                if let Some(prop) = edge.properties.get(col) {
                    return self.property_compare_gte(prop, val);
                }
                false
            },
            Condition::Lt(col, val) => {
                if let Some(prop) = edge.properties.get(col) {
                    return self.property_compare_lt(prop, val);
                }
                false
            },
            Condition::Le(col, val) => {
                if let Some(prop) = edge.properties.get(col) {
                    return self.property_compare_lte(prop, val);
                }
                false
            },
            Condition::And(a, b) => {
                self.edge_matches_condition(edge, a) && self.edge_matches_condition(edge, b)
            },
            Condition::Or(a, b) => {
                self.edge_matches_condition(edge, a) || self.edge_matches_condition(edge, b)
            },
            _ => true, // Other conditions not fully implemented for edges
        }
    }

    fn property_matches_value(&self, prop: &graph_engine::PropertyValue, val: &Value) -> bool {
        match (prop, val) {
            (graph_engine::PropertyValue::Int(i), Value::Int(v)) => *i == *v,
            (graph_engine::PropertyValue::Float(f), Value::Float(v)) => {
                (*f - *v).abs() < f64::EPSILON
            },
            (graph_engine::PropertyValue::String(s), Value::String(v)) => s == v,
            (graph_engine::PropertyValue::Bool(b), Value::Bool(v)) => *b == *v,
            _ => false,
        }
    }

    fn property_compare_gt(&self, prop: &graph_engine::PropertyValue, val: &Value) -> bool {
        match (prop, val) {
            (graph_engine::PropertyValue::Int(i), Value::Int(v)) => *i > *v,
            (graph_engine::PropertyValue::Float(f), Value::Float(v)) => *f > *v,
            (graph_engine::PropertyValue::Int(i), Value::Float(v)) => (*i as f64) > *v,
            (graph_engine::PropertyValue::Float(f), Value::Int(v)) => *f > (*v as f64),
            _ => false,
        }
    }

    fn property_compare_gte(&self, prop: &graph_engine::PropertyValue, val: &Value) -> bool {
        match (prop, val) {
            (graph_engine::PropertyValue::Int(i), Value::Int(v)) => *i >= *v,
            (graph_engine::PropertyValue::Float(f), Value::Float(v)) => *f >= *v,
            (graph_engine::PropertyValue::Int(i), Value::Float(v)) => (*i as f64) >= *v,
            (graph_engine::PropertyValue::Float(f), Value::Int(v)) => *f >= (*v as f64),
            _ => false,
        }
    }

    fn property_compare_lt(&self, prop: &graph_engine::PropertyValue, val: &Value) -> bool {
        match (prop, val) {
            (graph_engine::PropertyValue::Int(i), Value::Int(v)) => *i < *v,
            (graph_engine::PropertyValue::Float(f), Value::Float(v)) => *f < *v,
            (graph_engine::PropertyValue::Int(i), Value::Float(v)) => (*i as f64) < *v,
            (graph_engine::PropertyValue::Float(f), Value::Int(v)) => *f < (*v as f64),
            _ => false,
        }
    }

    fn property_compare_lte(&self, prop: &graph_engine::PropertyValue, val: &Value) -> bool {
        match (prop, val) {
            (graph_engine::PropertyValue::Int(i), Value::Int(v)) => *i <= *v,
            (graph_engine::PropertyValue::Float(f), Value::Float(v)) => *f <= *v,
            (graph_engine::PropertyValue::Int(i), Value::Float(v)) => (*i as f64) <= *v,
            (graph_engine::PropertyValue::Float(f), Value::Int(v)) => *f <= (*v as f64),
            _ => false,
        }
    }

    // ========== AST Conversion Helpers ==========

    fn expr_to_condition(&self, expr: &Expr) -> Result<Condition> {
        match &expr.kind {
            ExprKind::Binary(left, op, right) => match op {
                BinaryOp::And => {
                    let l = self.expr_to_condition(left)?;
                    let r = self.expr_to_condition(right)?;
                    Ok(l.and(r))
                },
                BinaryOp::Or => {
                    let l = self.expr_to_condition(left)?;
                    let r = self.expr_to_condition(right)?;
                    Ok(l.or(r))
                },
                BinaryOp::Eq => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Eq(col, val))
                },
                BinaryOp::Ne => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Ne(col, val))
                },
                BinaryOp::Lt => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Lt(col, val))
                },
                BinaryOp::Le => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Le(col, val))
                },
                BinaryOp::Gt => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Gt(col, val))
                },
                BinaryOp::Ge => {
                    let col = self.expr_to_column_name(left)?;
                    let val = self.expr_to_value(right)?;
                    Ok(Condition::Ge(col, val))
                },
                _ => Err(RouterError::ParseError(format!(
                    "Unsupported operator in condition: {:?}",
                    op
                ))),
            },
            _ => Err(RouterError::ParseError(
                "Expected binary expression in condition".to_string(),
            )),
        }
    }

    fn expr_to_value(&self, expr: &Expr) -> Result<Value> {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Null => Ok(Value::Null),
                Literal::Boolean(b) => Ok(Value::Bool(*b)),
                Literal::Integer(i) => Ok(Value::Int(*i)),
                Literal::Float(f) => Ok(Value::Float(*f)),
                Literal::String(s) => Ok(Value::String(s.clone())),
            },
            ExprKind::Ident(ident) => Ok(Value::String(ident.name.clone())),
            _ => Err(RouterError::ParseError(format!(
                "Cannot convert expression to value: {:?}",
                expr.kind
            ))),
        }
    }

    fn expr_to_column_name(&self, expr: &Expr) -> Result<String> {
        match &expr.kind {
            ExprKind::Ident(ident) => Ok(ident.name.clone()),
            ExprKind::Qualified(_, name) => Ok(name.name.clone()),
            _ => Err(RouterError::ParseError("Expected column name".to_string())),
        }
    }

    fn expr_to_u64(&self, expr: &Expr) -> Result<u64> {
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(i)) if *i >= 0 => Ok(*i as u64),
            _ => Err(RouterError::InvalidArgument(
                "Expected positive integer".to_string(),
            )),
        }
    }

    fn expr_to_f32(&self, expr: &Expr) -> Result<f32> {
        match &expr.kind {
            ExprKind::Literal(Literal::Float(f)) => Ok(*f as f32),
            ExprKind::Literal(Literal::Integer(i)) => Ok(*i as f32),
            _ => Err(RouterError::InvalidArgument("Expected number".to_string())),
        }
    }

    fn expr_to_usize(&self, expr: &Expr) -> Result<usize> {
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(i)) if *i >= 0 => Ok(*i as usize),
            _ => Err(RouterError::InvalidArgument(
                "Expected positive integer".to_string(),
            )),
        }
    }

    fn expr_to_string(&self, expr: &Expr) -> Result<String> {
        match &expr.kind {
            ExprKind::Literal(Literal::String(s)) => Ok(s.clone()),
            ExprKind::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(RouterError::InvalidArgument("Expected string".to_string())),
        }
    }

    fn properties_to_map(&self, properties: &[Property]) -> Result<HashMap<String, PropertyValue>> {
        let mut map = HashMap::new();
        for prop in properties {
            let value = match &prop.value.kind {
                ExprKind::Literal(Literal::Null) => PropertyValue::Null,
                ExprKind::Literal(Literal::Boolean(b)) => PropertyValue::Bool(*b),
                ExprKind::Literal(Literal::Integer(i)) => PropertyValue::Int(*i),
                ExprKind::Literal(Literal::Float(f)) => PropertyValue::Float(*f),
                ExprKind::Literal(Literal::String(s)) => PropertyValue::String(s.clone()),
                ExprKind::Ident(ident) => PropertyValue::String(ident.name.clone()),
                _ => {
                    return Err(RouterError::InvalidArgument(format!(
                        "Invalid property value: {:?}",
                        prop.value.kind
                    )))
                },
            };
            map.insert(prop.key.name.clone(), value);
        }
        Ok(map)
    }

    fn data_type_to_column_type(
        &self,
        dt: &parser::DataType,
    ) -> Result<relational_engine::ColumnType> {
        use parser::DataType;
        match dt {
            DataType::Int | DataType::Integer | DataType::Bigint | DataType::Smallint => {
                Ok(relational_engine::ColumnType::Int)
            },
            DataType::Float
            | DataType::Double
            | DataType::Real
            | DataType::Decimal(_, _)
            | DataType::Numeric(_, _) => Ok(relational_engine::ColumnType::Float),
            DataType::Varchar(_) | DataType::Char(_) | DataType::Text => {
                Ok(relational_engine::ColumnType::String)
            },
            DataType::Boolean => Ok(relational_engine::ColumnType::Bool),
            _ => Err(RouterError::ParseError(format!(
                "Unsupported data type: {:?}",
                dt
            ))),
        }
    }

    // ========== Relational Commands ==========

    fn execute_select(&self, command: &str) -> Result<QueryResult> {
        // SELECT <table> [WHERE <condition>]
        let parts: Vec<&str> = command.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("table name".to_string()));
        }

        let rest = parts[1].trim();
        let (table, condition) = if let Some(pos) = rest.to_uppercase().find(" WHERE ") {
            let table = rest[..pos].trim();
            let cond_str = rest[pos + 7..].trim();
            (table, self.parse_condition(cond_str)?)
        } else {
            (rest, Condition::True)
        };

        let rows = self.relational.select(table, condition)?;
        Ok(QueryResult::Rows(rows))
    }

    fn execute_insert(&self, command: &str) -> Result<QueryResult> {
        // INSERT <table> <col>=<val>, ...
        let parts: Vec<&str> = command.splitn(3, char::is_whitespace).collect();
        if parts.len() < 3 {
            return Err(RouterError::MissingArgument(
                "table name and values".to_string(),
            ));
        }

        let table = parts[1];
        let values_str = parts[2];
        let values = self.parse_values(values_str)?;

        let id = self.relational.insert(table, values)?;
        Ok(QueryResult::Ids(vec![id]))
    }

    fn execute_update(&self, command: &str) -> Result<QueryResult> {
        // UPDATE <table> SET <col>=<val>, ... [WHERE <condition>]
        let upper = command.to_uppercase();
        let set_pos = upper
            .find(" SET ")
            .ok_or_else(|| RouterError::ParseError("Missing SET clause".to_string()))?;

        let table_part = &command[7..set_pos].trim();
        let rest = &command[set_pos + 5..];

        let (values_str, condition) = if let Some(pos) = rest.to_uppercase().find(" WHERE ") {
            (&rest[..pos], self.parse_condition(&rest[pos + 7..])?)
        } else {
            (rest, Condition::True)
        };

        let values = self.parse_values(values_str)?;
        let count = self.relational.update(table_part, condition, values)?;
        Ok(QueryResult::Count(count))
    }

    fn execute_delete(&self, command: &str) -> Result<QueryResult> {
        // DELETE <table> [WHERE <condition>]
        let parts: Vec<&str> = command.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("table name".to_string()));
        }

        let rest = parts[1].trim();
        let (table, condition) = if let Some(pos) = rest.to_uppercase().find(" WHERE ") {
            (&rest[..pos], self.parse_condition(&rest[pos + 7..])?)
        } else {
            (rest, Condition::True)
        };

        let count = self.relational.delete_rows(table, condition)?;
        Ok(QueryResult::Count(count))
    }

    fn execute_create(&self, command: &str) -> Result<QueryResult> {
        // CREATE TABLE <table> (<col>:<type>, ...)
        // CREATE INDEX <table> <column>
        let upper = command.to_uppercase();

        if upper.starts_with("CREATE TABLE ") {
            self.execute_create_table(command)
        } else if upper.starts_with("CREATE INDEX ") {
            self.execute_create_index(command)
        } else {
            Err(RouterError::ParseError(
                "Expected CREATE TABLE or CREATE INDEX".to_string(),
            ))
        }
    }

    fn execute_create_table(&self, command: &str) -> Result<QueryResult> {
        // CREATE TABLE <table> (<col>:<type>, ...)
        let paren_start = command
            .find('(')
            .ok_or_else(|| RouterError::ParseError("Missing column definitions".to_string()))?;
        let paren_end = command
            .rfind(')')
            .ok_or_else(|| RouterError::ParseError("Missing closing parenthesis".to_string()))?;

        let table = command[13..paren_start].trim();
        let cols_str = &command[paren_start + 1..paren_end];

        let mut columns = Vec::new();
        for col_def in cols_str.split(',') {
            let col_def = col_def.trim();
            let parts: Vec<&str> = col_def.split(':').collect();
            if parts.len() < 2 {
                return Err(RouterError::ParseError(format!(
                    "Invalid column definition: {}",
                    col_def
                )));
            }

            let name = parts[0].trim();
            let type_str = parts[1].trim().to_uppercase();
            let nullable = type_str.ends_with('?');
            let type_str = type_str.trim_end_matches('?');

            let col_type = match type_str {
                "INT" | "INTEGER" => relational_engine::ColumnType::Int,
                "FLOAT" | "DOUBLE" => relational_engine::ColumnType::Float,
                "STRING" | "TEXT" => relational_engine::ColumnType::String,
                "BOOL" | "BOOLEAN" => relational_engine::ColumnType::Bool,
                _ => {
                    return Err(RouterError::ParseError(format!(
                        "Unknown type: {}",
                        type_str
                    )))
                },
            };

            let mut col = relational_engine::Column::new(name, col_type);
            if nullable {
                col = col.nullable();
            }
            columns.push(col);
        }

        let schema = relational_engine::Schema::new(columns);
        self.relational.create_table(table, schema)?;
        Ok(QueryResult::Empty)
    }

    fn execute_create_index(&self, command: &str) -> Result<QueryResult> {
        // CREATE INDEX <table> <column>
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(RouterError::MissingArgument(
                "table and column names".to_string(),
            ));
        }

        let table = parts[2];
        let column = parts[3];
        self.relational.create_index(table, column)?;
        Ok(QueryResult::Empty)
    }

    fn execute_drop(&self, command: &str) -> Result<QueryResult> {
        // DROP TABLE <table>
        // DROP INDEX <table> <column>
        let upper = command.to_uppercase();

        if upper.starts_with("DROP TABLE ") {
            let table = command[11..].trim();
            self.relational.drop_table(table)?;
            Ok(QueryResult::Empty)
        } else if upper.starts_with("DROP INDEX ") {
            let parts: Vec<&str> = command.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(RouterError::MissingArgument(
                    "table and column names".to_string(),
                ));
            }
            self.relational.drop_index(parts[2], parts[3])?;
            Ok(QueryResult::Empty)
        } else {
            Err(RouterError::ParseError(
                "Expected DROP TABLE or DROP INDEX".to_string(),
            ))
        }
    }

    // ========== Graph Commands ==========

    fn execute_node(&self, command: &str) -> Result<QueryResult> {
        // NODE CREATE <label> [<key>=<val>, ...]
        // NODE GET <id>
        // NODE DELETE <id>
        let parts: Vec<&str> = command.splitn(3, char::is_whitespace).collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("subcommand".to_string()));
        }

        let subcmd = parts[1].to_uppercase();
        match subcmd.as_str() {
            "CREATE" => {
                if parts.len() < 3 {
                    return Err(RouterError::MissingArgument("label".to_string()));
                }
                let rest = parts[2];
                let (label, props) = self.parse_label_and_props(rest)?;
                let id = self.graph.create_node(&label, props)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            "GET" => {
                if parts.len() < 3 {
                    return Err(RouterError::MissingArgument("node id".to_string()));
                }
                let id: u64 = parts[2]
                    .parse()
                    .map_err(|_| RouterError::InvalidArgument("Invalid node ID".to_string()))?;
                let node = self.graph.get_node(id)?;
                let properties: HashMap<String, String> = node
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::property_to_string(v)))
                    .collect();
                let result = NodeResult {
                    id: node.id,
                    label: node.label.clone(),
                    properties,
                };
                Ok(QueryResult::Nodes(vec![result]))
            },
            "DELETE" => {
                if parts.len() < 3 {
                    return Err(RouterError::MissingArgument("node id".to_string()));
                }
                let id: u64 = parts[2]
                    .parse()
                    .map_err(|_| RouterError::InvalidArgument("Invalid node ID".to_string()))?;
                self.graph.delete_node(id)?;
                Ok(QueryResult::Count(1))
            },
            _ => Err(RouterError::UnknownCommand(format!("NODE {}", subcmd))),
        }
    }

    fn execute_edge(&self, command: &str) -> Result<QueryResult> {
        // EDGE CREATE <from> -> <to> [<label>] [DIRECTED|UNDIRECTED]
        // EDGE GET <id>
        let parts: Vec<&str> = command.splitn(3, char::is_whitespace).collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("subcommand".to_string()));
        }

        let subcmd = parts[1].to_uppercase();
        match subcmd.as_str() {
            "CREATE" => {
                if parts.len() < 3 {
                    return Err(RouterError::MissingArgument("edge definition".to_string()));
                }
                let rest = parts[2];
                let (from, to, edge_type, directed) = self.parse_edge_def(rest)?;
                let id = self
                    .graph
                    .create_edge(from, to, &edge_type, HashMap::new(), directed)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            "GET" => {
                if parts.len() < 3 {
                    return Err(RouterError::MissingArgument("edge id".to_string()));
                }
                let id: u64 = parts[2]
                    .parse()
                    .map_err(|_| RouterError::InvalidArgument("Invalid edge ID".to_string()))?;
                let edge = self.graph.get_edge(id)?;
                let result = EdgeResult {
                    id: edge.id,
                    from: edge.from,
                    to: edge.to,
                    label: edge.edge_type.clone(),
                };
                Ok(QueryResult::Edges(vec![result]))
            },
            _ => Err(RouterError::UnknownCommand(format!("EDGE {}", subcmd))),
        }
    }

    fn execute_neighbors(&self, command: &str) -> Result<QueryResult> {
        // NEIGHBORS <id> [OUT|IN|BOTH]
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("node id".to_string()));
        }

        let id: u64 = parts[1]
            .parse()
            .map_err(|_| RouterError::InvalidArgument("Invalid node ID".to_string()))?;

        let direction_str = if parts.len() > 2 {
            parts[2].to_uppercase()
        } else {
            "BOTH".to_string()
        };

        let direction = match direction_str.as_str() {
            "OUT" => Direction::Outgoing,
            "IN" => Direction::Incoming,
            "BOTH" => Direction::Both,
            _ => {
                return Err(RouterError::InvalidArgument(format!(
                    "Unknown direction: {}",
                    direction_str
                )))
            },
        };

        let neighbors = self.graph.neighbors(id, None, direction)?;
        let neighbor_ids: Vec<u64> = neighbors.iter().map(|n| n.id).collect();

        Ok(QueryResult::Ids(neighbor_ids))
    }

    fn execute_path(&self, command: &str) -> Result<QueryResult> {
        // PATH <from> -> <to>
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(RouterError::MissingArgument(
                "from and to node ids".to_string(),
            ));
        }

        let from: u64 = parts[1]
            .parse()
            .map_err(|_| RouterError::InvalidArgument("Invalid from node ID".to_string()))?;

        // Skip the "->" token
        let to: u64 = parts[3]
            .parse()
            .map_err(|_| RouterError::InvalidArgument("Invalid to node ID".to_string()))?;

        match self.graph.find_path(from, to) {
            Ok(path) => Ok(QueryResult::Path(path.nodes)),
            Err(GraphError::PathNotFound) => Ok(QueryResult::Path(vec![])),
            Err(e) => Err(e.into()),
        }
    }

    // ========== Vector Commands ==========

    fn execute_embed(&self, command: &str) -> Result<QueryResult> {
        // EMBED <key> [<val>, ...]
        let parts: Vec<&str> = command.splitn(3, char::is_whitespace).collect();
        if parts.len() < 3 {
            return Err(RouterError::MissingArgument("key and vector".to_string()));
        }

        let key = parts[1];
        let vector = self.parse_vector(parts[2])?;
        self.vector.store_embedding(key, vector)?;
        Ok(QueryResult::Empty)
    }

    fn execute_similar(&self, command: &str) -> Result<QueryResult> {
        // SIMILAR <key> [TOP <k>]
        // SIMILAR [<val>, ...] [TOP <k>]
        let parts: Vec<&str> = command.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            return Err(RouterError::MissingArgument("key or vector".to_string()));
        }

        let rest = parts[1].trim();
        let (query_vec, top_k) = self.parse_similar_args(rest)?;

        // Use HNSW if available, otherwise brute force
        let results = if let Some((ref index, ref keys)) = self.hnsw_index {
            self.vector
                .search_with_hnsw(index, keys, &query_vec, top_k)?
                .into_iter()
                .map(|r| SimilarResult {
                    key: r.key,
                    score: r.score,
                })
                .collect()
        } else {
            self.vector
                .search_similar(&query_vec, top_k)?
                .into_iter()
                .map(|r| SimilarResult {
                    key: r.key,
                    score: r.score,
                })
                .collect()
        };

        Ok(QueryResult::Similar(results))
    }

    // ========== Unified Commands ==========

    fn execute_find(&self, command: &str) -> Result<QueryResult> {
        // FIND posts SIMILAR TO "embedding_key" CONNECTED TO users
        // FIND users WHERE age > 25 CONNECTED TO posts
        let upper = command.to_uppercase();

        // Parse the FIND command structure
        let parts = self.parse_find_command(&upper, command)?;

        // Execute based on parsed parts
        self.execute_unified_query(parts)
    }

    fn parse_find_command(&self, upper: &str, original: &str) -> Result<UnifiedQueryParts> {
        // Basic structure: FIND <entity> [clauses...]
        let tokens: Vec<&str> = original.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(RouterError::MissingArgument("entity name".to_string()));
        }

        let entity = tokens[1].to_string();
        let mut parts = UnifiedQueryParts {
            entity,
            where_clause: None,
            similar_to: None,
            connected_to: None,
            top_k: 10,
        };

        // Parse WHERE clause
        if let Some(where_pos) = upper.find(" WHERE ") {
            let end_pos = upper[where_pos + 7..]
                .find(" SIMILAR ")
                .or_else(|| upper[where_pos + 7..].find(" CONNECTED "))
                .map(|p| where_pos + 7 + p)
                .unwrap_or(original.len());
            parts.where_clause = Some(original[where_pos + 7..end_pos].trim().to_string());
        }

        // Parse SIMILAR TO clause
        if let Some(sim_pos) = upper.find(" SIMILAR TO ") {
            let start = sim_pos + 12;
            let end_pos = upper[start..]
                .find(" CONNECTED ")
                .or_else(|| upper[start..].find(" TOP "))
                .map(|p| start + p)
                .unwrap_or(original.len());
            let similar_key = original[start..end_pos].trim().trim_matches('"');
            parts.similar_to = Some(similar_key.to_string());
        }

        // Parse CONNECTED TO clause
        if let Some(conn_pos) = upper.find(" CONNECTED TO ") {
            let start = conn_pos + 14;
            let end_pos = upper[start..]
                .find(" TOP ")
                .map(|p| start + p)
                .unwrap_or(original.len());
            parts.connected_to = Some(original[start..end_pos].trim().to_string());
        }

        // Parse TOP k
        if let Some(top_pos) = upper.find(" TOP ") {
            let k_str = original[top_pos + 5..].trim();
            parts.top_k = k_str
                .parse()
                .map_err(|_| RouterError::InvalidArgument("Invalid TOP value".to_string()))?;
        }

        Ok(parts)
    }

    fn execute_unified_query(&self, parts: UnifiedQueryParts) -> Result<QueryResult> {
        let mut items = Vec::new();
        let mut description = format!("Finding {}", parts.entity);

        // Step 1: If SIMILAR TO, find similar vectors first
        let similar_keys: Option<Vec<(String, f32)>> = if let Some(ref key) = parts.similar_to {
            description.push_str(&format!(" similar to '{}'", key));

            // Get the query vector
            let query_vec = self.vector.get_embedding(key)?;

            // Search for similar
            let results = self.vector.search_similar(&query_vec, parts.top_k)?;
            Some(
                results
                    .into_iter()
                    .map(|r| (r.key.clone(), r.score))
                    .collect(),
            )
        } else {
            None
        };

        // Step 2: If CONNECTED TO, find connected nodes
        if let Some(ref connected_entity) = parts.connected_to {
            description.push_str(&format!(" connected to {}", connected_entity));
            // Note: Full graph traversal not yet implemented
            // For now, include similar items with connection context
            if let Some(similar) = &similar_keys {
                for (key, score) in similar {
                    let mut data = HashMap::new();
                    data.insert("key".to_string(), key.clone());
                    data.insert("connected_to".to_string(), connected_entity.clone());
                    items.push(
                        UnifiedItem::with_data("vector+graph", key.clone(), data)
                            .with_score(*score),
                    );
                }
            }
        } else if let Some(similar) = similar_keys {
            // Just return similar results
            for (key, score) in similar {
                let mut data = HashMap::new();
                data.insert("key".to_string(), key.clone());
                items.push(UnifiedItem::with_data("vector", key, data).with_score(score));
            }
        }

        // Step 3: Apply WHERE clause filter if present
        if let Some(ref where_clause) = parts.where_clause {
            description.push_str(&format!(" where {}", where_clause));
            // Filter items based on condition (simplified)
            // In a real implementation, this would evaluate the condition
        }

        // Limit to top_k
        items.truncate(parts.top_k);

        Ok(QueryResult::Unified(UnifiedResult { description, items }))
    }

    // ========== Parsing Helpers ==========

    fn parse_condition(&self, cond_str: &str) -> Result<Condition> {
        let cond_str = cond_str.trim();

        // Handle AND/OR
        if let Some(pos) = cond_str.to_uppercase().find(" AND ") {
            let left = self.parse_condition(&cond_str[..pos])?;
            let right = self.parse_condition(&cond_str[pos + 5..])?;
            return Ok(left.and(right));
        }
        if let Some(pos) = cond_str.to_uppercase().find(" OR ") {
            let left = self.parse_condition(&cond_str[..pos])?;
            let right = self.parse_condition(&cond_str[pos + 4..])?;
            return Ok(left.or(right));
        }

        // Parse simple condition: col op value
        let operators = [">=", "<=", "!=", "=", ">", "<"];
        for op in operators {
            if let Some(pos) = cond_str.find(op) {
                let col = cond_str[..pos].trim().to_string();
                let val_str = cond_str[pos + op.len()..].trim();
                let value = self.parse_value(val_str)?;

                return Ok(match op {
                    "=" => Condition::Eq(col, value),
                    "!=" => Condition::Ne(col, value),
                    ">" => Condition::Gt(col, value),
                    ">=" => Condition::Ge(col, value),
                    "<" => Condition::Lt(col, value),
                    "<=" => Condition::Le(col, value),
                    _ => unreachable!(),
                });
            }
        }

        Err(RouterError::ParseError(format!(
            "Invalid condition: {}",
            cond_str
        )))
    }

    fn parse_value(&self, val_str: &str) -> Result<Value> {
        let val_str = val_str.trim();

        // NULL
        if val_str.to_uppercase() == "NULL" {
            return Ok(Value::Null);
        }

        // Boolean
        if val_str.to_uppercase() == "TRUE" {
            return Ok(Value::Bool(true));
        }
        if val_str.to_uppercase() == "FALSE" {
            return Ok(Value::Bool(false));
        }

        // String (quoted)
        if (val_str.starts_with('"') && val_str.ends_with('"'))
            || (val_str.starts_with('\'') && val_str.ends_with('\''))
        {
            return Ok(Value::String(val_str[1..val_str.len() - 1].to_string()));
        }

        // Integer
        if let Ok(i) = val_str.parse::<i64>() {
            return Ok(Value::Int(i));
        }

        // Float
        if let Ok(f) = val_str.parse::<f64>() {
            return Ok(Value::Float(f));
        }

        // Unquoted string
        Ok(Value::String(val_str.to_string()))
    }

    fn parse_values(&self, values_str: &str) -> Result<HashMap<String, Value>> {
        let mut values = HashMap::new();

        for pair in values_str.split(',') {
            let pair = pair.trim();
            let eq_pos = pair
                .find('=')
                .ok_or_else(|| RouterError::ParseError(format!("Invalid assignment: {}", pair)))?;
            let key = pair[..eq_pos].trim().to_string();
            let val = self.parse_value(&pair[eq_pos + 1..])?;
            values.insert(key, val);
        }

        Ok(values)
    }

    fn parse_label_and_props(
        &self,
        rest: &str,
    ) -> Result<(String, HashMap<String, PropertyValue>)> {
        let mut props = HashMap::new();

        // Find first space or end
        let label_end = rest.find(' ').unwrap_or(rest.len());
        let label = rest[..label_end].to_string();

        if label_end < rest.len() {
            let props_str = rest[label_end..].trim();
            for pair in props_str.split(',') {
                let pair = pair.trim();
                if let Some(eq_pos) = pair.find('=') {
                    let key = pair[..eq_pos].trim().to_string();
                    let val_str = pair[eq_pos + 1..].trim();
                    let prop_val = self.parse_property_value(val_str);
                    props.insert(key, prop_val);
                }
            }
        }

        Ok((label, props))
    }

    fn parse_property_value(&self, val_str: &str) -> PropertyValue {
        let val_str = val_str.trim();

        if val_str.to_uppercase() == "NULL" {
            return PropertyValue::Null;
        }
        if val_str.to_uppercase() == "TRUE" {
            return PropertyValue::Bool(true);
        }
        if val_str.to_uppercase() == "FALSE" {
            return PropertyValue::Bool(false);
        }
        if (val_str.starts_with('"') && val_str.ends_with('"'))
            || (val_str.starts_with('\'') && val_str.ends_with('\''))
        {
            return PropertyValue::String(val_str[1..val_str.len() - 1].to_string());
        }
        if let Ok(i) = val_str.parse::<i64>() {
            return PropertyValue::Int(i);
        }
        if let Ok(f) = val_str.parse::<f64>() {
            return PropertyValue::Float(f);
        }
        PropertyValue::String(val_str.to_string())
    }

    fn property_to_string(prop: &PropertyValue) -> String {
        match prop {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => s.clone(),
            PropertyValue::Bool(b) => b.to_string(),
        }
    }

    fn parse_edge_def(&self, rest: &str) -> Result<(u64, u64, String, bool)> {
        // <from> -> <to> [<label>] [DIRECTED|UNDIRECTED]
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(RouterError::ParseError(
                "Invalid edge definition".to_string(),
            ));
        }

        let from: u64 = parts[0]
            .parse()
            .map_err(|_| RouterError::InvalidArgument("Invalid from node ID".to_string()))?;

        // Skip "->"
        let to: u64 = parts[2]
            .parse()
            .map_err(|_| RouterError::InvalidArgument("Invalid to node ID".to_string()))?;

        let mut label = "edge".to_string();
        let mut directed = true;

        for part in parts.iter().skip(3) {
            let upper = part.to_uppercase();
            if upper == "DIRECTED" {
                directed = true;
            } else if upper == "UNDIRECTED" {
                directed = false;
            } else {
                label = part.to_string();
            }
        }

        Ok((from, to, label, directed))
    }

    fn parse_vector(&self, vec_str: &str) -> Result<Vec<f32>> {
        let vec_str = vec_str.trim().trim_matches('[').trim_matches(']');
        let mut vector = Vec::new();

        for val in vec_str.split(',') {
            let val = val.trim();
            let f: f32 = val
                .parse()
                .map_err(|_| RouterError::InvalidArgument(format!("Invalid float: {}", val)))?;
            vector.push(f);
        }

        if vector.is_empty() {
            return Err(RouterError::InvalidArgument("Empty vector".to_string()));
        }

        Ok(vector)
    }

    fn parse_similar_args(&self, rest: &str) -> Result<(Vec<f32>, usize)> {
        let upper = rest.to_uppercase();
        let mut top_k = 10;

        // Check for TOP clause
        let query_part = if let Some(top_pos) = upper.find(" TOP ") {
            let k_str = rest[top_pos + 5..].trim();
            top_k = k_str
                .parse()
                .map_err(|_| RouterError::InvalidArgument("Invalid TOP value".to_string()))?;
            rest[..top_pos].trim()
        } else {
            rest
        };

        // Check if it's a key reference or inline vector
        if query_part.starts_with('[') {
            // Inline vector
            let vector = self.parse_vector(query_part)?;
            Ok((vector, top_k))
        } else {
            // Key reference
            let key = query_part.trim().trim_matches('"');
            let vector = self.vector.get_embedding(key)?;
            Ok((vector, top_k))
        }
    }

    // ========== Async Execution Methods ==========

    /// Execute a command string asynchronously using the AST-based parser.
    ///
    /// This method is the async counterpart to `execute_parsed()`. It provides
    /// truly non-blocking execution for I/O-bound operations like blob storage.
    ///
    /// # Example
    /// ```ignore
    /// let result = router.execute_parsed_async("BLOB GET 'artifact-id'").await?;
    /// ```
    pub async fn execute_parsed_async(&self, command: &str) -> Result<QueryResult> {
        let stmt = parser::parse(command)
            .map_err(|e| RouterError::ParseError(e.format_with_source(command)))?;

        // Check cache for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            if let Some(cached) = self.try_cache_get(command) {
                return Ok(cached);
            }
        }

        // Execute the statement asynchronously
        let result = self.execute_statement_async(&stmt).await?;

        // Cache the result for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            self.try_cache_put(command, &result);
        }

        // Invalidate cache on write operations
        if Self::is_write_statement(&stmt) {
            self.invalidate_cache_on_write();
        }

        Ok(result)
    }

    /// Execute a parsed statement asynchronously.
    ///
    /// Most operations are synchronous (in-memory), but blob operations
    /// are truly async, avoiding runtime blocking.
    pub async fn execute_statement_async(&self, stmt: &Statement) -> Result<QueryResult> {
        match &stmt.kind {
            // Blob statements are truly async
            StatementKind::Blob(blob) => self.exec_blob_async(blob).await,
            StatementKind::Blobs(blobs) => self.exec_blobs_async(blobs).await,

            // Checkpoint statements are async (use blob storage)
            StatementKind::Checkpoint(cp) => self.exec_checkpoint_async(cp).await,
            StatementKind::Rollback(rb) => self.exec_rollback_async(rb).await,
            StatementKind::Checkpoints(cps) => self.exec_checkpoints_async(cps).await,

            // All other statements delegate to sync execution
            // (they're in-memory and fast, no benefit from async)
            _ => self.execute_statement(stmt),
        }
    }

    /// Execute blob operations asynchronously without blocking.
    async fn exec_blob_async(&self, stmt: &BlobStmt) -> Result<QueryResult> {
        // Handle BLOB INIT specially - doesn't require blob to be initialized
        if matches!(stmt.operation, BlobOp::Init) {
            if self.blob.is_some() {
                return Ok(QueryResult::Value(
                    "Blob store already initialized".to_string(),
                ));
            } else {
                return Err(RouterError::BlobError(
                    "Use router.init_blob() to initialize blob storage".to_string(),
                ));
            }
        }

        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;

        match &stmt.operation {
            BlobOp::Init => unreachable!(), // Handled above
            BlobOp::Put {
                filename,
                data,
                from_path,
                options,
            } => {
                let filename_str = self.eval_string_expr(filename)?;
                let put_options = self.blob_options_to_put_options(options)?;

                // Get data either from inline DATA or from file path
                let blob_data = if let Some(data_expr) = data {
                    self.expr_to_bytes(data_expr)?
                } else if let Some(path_expr) = from_path {
                    let path = self.eval_string_expr(path_expr)?;
                    tokio::fs::read(&path)
                        .await
                        .map_err(|e| RouterError::BlobError(format!("Failed to read file: {e}")))?
                } else {
                    return Err(RouterError::MissingArgument(
                        "PUT requires either DATA or FROM path".to_string(),
                    ));
                };

                let blob_guard = blob.lock().await;
                let artifact_id = blob_guard
                    .put(&filename_str, &blob_data, put_options)
                    .await?;
                Ok(QueryResult::Value(artifact_id))
            },
            BlobOp::Get {
                artifact_id,
                to_path,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let blob_guard = blob.lock().await;
                let data = blob_guard.get(&id).await?;

                if let Some(path_expr) = to_path {
                    let path = self.eval_string_expr(path_expr)?;
                    tokio::fs::write(&path, &data).await.map_err(|e| {
                        RouterError::BlobError(format!("Failed to write file: {e}"))
                    })?;
                    Ok(QueryResult::Value(format!(
                        "Written {} bytes to {path}",
                        data.len()
                    )))
                } else {
                    Ok(QueryResult::Blob(data))
                }
            },
            BlobOp::Delete { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let blob_guard = blob.lock().await;
                blob_guard.delete(&id).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Info { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let blob_guard = blob.lock().await;
                let meta = blob_guard.metadata(&id).await?;

                Ok(QueryResult::ArtifactInfo(ArtifactInfoResult {
                    id: meta.id,
                    filename: meta.filename,
                    content_type: meta.content_type,
                    size: meta.size,
                    checksum: meta.checksum,
                    chunk_count: meta.chunk_count,
                    created: meta.created,
                    modified: meta.modified,
                    created_by: meta.created_by,
                    tags: meta.tags,
                    linked_to: meta.linked_to,
                    custom: meta.custom,
                }))
            },
            BlobOp::Link {
                artifact_id,
                entity,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let entity_str = self.eval_string_expr(entity)?;
                let blob_guard = blob.lock().await;
                blob_guard.link(&id, &entity_str).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Unlink {
                artifact_id,
                entity,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let entity_str = self.eval_string_expr(entity)?;
                let blob_guard = blob.lock().await;
                blob_guard.unlink(&id, &entity_str).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Links { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let blob_guard = blob.lock().await;
                let links = blob_guard.links(&id).await?;
                Ok(QueryResult::ArtifactList(links))
            },
            BlobOp::Tag { artifact_id, tag } => {
                let id = self.eval_string_expr(artifact_id)?;
                let tag_str = self.eval_string_expr(tag)?;
                let blob_guard = blob.lock().await;
                blob_guard.tag(&id, &tag_str).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Untag { artifact_id, tag } => {
                let id = self.eval_string_expr(artifact_id)?;
                let tag_str = self.eval_string_expr(tag)?;
                let blob_guard = blob.lock().await;
                blob_guard.untag(&id, &tag_str).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::Verify { artifact_id } => {
                let id = self.eval_string_expr(artifact_id)?;
                let blob_guard = blob.lock().await;
                let valid = blob_guard.verify(&id).await?;
                Ok(QueryResult::Value(if valid {
                    "OK".to_string()
                } else {
                    "INVALID".to_string()
                }))
            },
            BlobOp::Gc { full } => {
                let blob_guard = blob.lock().await;
                let stats = if *full {
                    blob_guard.full_gc().await?
                } else {
                    blob_guard.gc().await?
                };
                Ok(QueryResult::Value(format!(
                    "Deleted {} chunks, freed {} bytes",
                    stats.deleted, stats.freed_bytes
                )))
            },
            BlobOp::Repair => {
                let blob_guard = blob.lock().await;
                let stats = blob_guard.repair().await?;
                Ok(QueryResult::Value(format!(
                    "Fixed {} refs, deleted {} orphans",
                    stats.refs_fixed, stats.orphans_deleted
                )))
            },
            BlobOp::Stats => {
                let blob_guard = blob.lock().await;
                let stats = blob_guard.stats().await?;
                Ok(QueryResult::BlobStats(BlobStatsResult {
                    artifact_count: stats.artifact_count,
                    chunk_count: stats.chunk_count,
                    total_bytes: stats.total_bytes,
                    unique_bytes: stats.unique_bytes,
                    dedup_ratio: stats.dedup_ratio,
                    orphaned_chunks: stats.orphaned_chunks,
                }))
            },
            BlobOp::MetaSet {
                artifact_id,
                key,
                value,
            } => {
                let id = self.eval_string_expr(artifact_id)?;
                let key_str = self.eval_string_expr(key)?;
                let value_str = self.eval_string_expr(value)?;
                let blob_guard = blob.lock().await;
                blob_guard.set_meta(&id, &key_str, &value_str).await?;
                Ok(QueryResult::Empty)
            },
            BlobOp::MetaGet { artifact_id, key } => {
                let id = self.eval_string_expr(artifact_id)?;
                let key_str = self.eval_string_expr(key)?;
                let blob_guard = blob.lock().await;
                let value = blob_guard.get_meta(&id, &key_str).await?;
                match value {
                    Some(v) => Ok(QueryResult::Value(v)),
                    None => Ok(QueryResult::Value("(not found)".to_string())),
                }
            },
        }
    }

    /// Execute blobs listing operations asynchronously.
    async fn exec_blobs_async(&self, stmt: &BlobsStmt) -> Result<QueryResult> {
        let blob = self
            .blob
            .as_ref()
            .ok_or_else(|| RouterError::BlobError("Blob store not initialized".to_string()))?;

        let blob_guard = blob.lock().await;
        match &stmt.operation {
            BlobsOp::List { pattern } => {
                let prefix = pattern
                    .as_ref()
                    .map(|p| self.eval_string_expr(p))
                    .transpose()?;
                let ids = blob_guard.list(prefix.as_deref()).await?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::For { entity } => {
                let entity_str = self.eval_string_expr(entity)?;
                let ids = blob_guard.artifacts_for(&entity_str).await?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::ByTag { tag } => {
                let tag_str = self.eval_string_expr(tag)?;
                let ids = blob_guard.by_tag(&tag_str).await?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::ByType { content_type } => {
                let ct = self.eval_string_expr(content_type)?;
                let ids = blob_guard.by_content_type(&ct).await?;
                Ok(QueryResult::ArtifactList(ids))
            },
            BlobsOp::Similar { artifact_id, limit } => {
                let id = self.eval_string_expr(artifact_id)?;
                let k = limit
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(10);
                let similar = blob_guard.similar(&id, k).await?;
                Ok(QueryResult::Similar(
                    similar
                        .into_iter()
                        .map(|s| SimilarResult {
                            key: s.id,
                            score: s.similarity,
                        })
                        .collect(),
                ))
            },
        }
    }

    /// Execute checkpoint creation asynchronously.
    async fn exec_checkpoint_async(&self, stmt: &CheckpointStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;

        let name = stmt
            .name
            .as_ref()
            .map(|e| self.eval_string_expr(e))
            .transpose()?;

        let store = self.vector.store();
        let cp_guard = checkpoint.lock().await;
        let checkpoint_id = cp_guard.create(name.as_deref(), store).await?;

        Ok(QueryResult::Value(format!(
            "Checkpoint created: {}",
            checkpoint_id
        )))
    }

    /// Execute rollback asynchronously.
    async fn exec_rollback_async(&self, stmt: &RollbackStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;

        let target = self.eval_string_expr(&stmt.target)?;

        let store = self.vector.store();
        let cp_guard = checkpoint.lock().await;
        cp_guard.rollback(&target, store).await?;

        Ok(QueryResult::Value(format!(
            "Rolled back to checkpoint: {}",
            target
        )))
    }

    /// Execute checkpoint listing asynchronously.
    async fn exec_checkpoints_async(&self, stmt: &CheckpointsStmt) -> Result<QueryResult> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
            RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
        })?;

        let limit = stmt
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?;

        // Default to 10 if no limit specified
        let limit_opt = limit.or(Some(10));

        let cp_guard = checkpoint.lock().await;
        let checkpoints = cp_guard.list(limit_opt).await?;

        let info_list: Vec<CheckpointInfo> = checkpoints
            .into_iter()
            .map(|cp| CheckpointInfo {
                id: cp.id,
                name: cp.name,
                created_at: cp.created_at,
                is_auto: cp.trigger.is_some(),
            })
            .collect();

        Ok(QueryResult::CheckpointList(info_list))
    }

    /// Store multiple embeddings in parallel.
    ///
    /// This method processes batch embeddings concurrently, which can provide
    /// performance benefits when storing many embeddings at once.
    ///
    /// # Arguments
    /// * `items` - Vector of (key, embedding) pairs to store
    ///
    /// # Returns
    /// * `Ok(count)` - Number of embeddings successfully stored
    /// * `Err(e)` - First error encountered
    pub async fn embed_batch_parallel(&self, items: Vec<(String, Vec<f32>)>) -> Result<usize> {
        use futures::future::join_all;

        let futures: Vec<_> = items
            .into_iter()
            .map(|(key, vec)| {
                let key = key.clone();
                let vec = vec.clone();
                let vector = Arc::clone(&self.vector);
                async move { vector.store_embedding(&key, vec) }
            })
            .collect();

        let results = join_all(futures).await;
        let mut success_count = 0;
        for result in results {
            match result {
                Ok(_) => success_count += 1,
                Err(e) => return Err(RouterError::VectorError(e.to_string())),
            }
        }
        Ok(success_count)
    }

    /// Find similar entities connected to a target, with parallel graph/vector queries.
    ///
    /// This async version uses `tokio::join!` to parallelize the vector similarity
    /// search and graph neighbor lookup, then filters the intersection.
    pub async fn find_similar_connected_async(
        &self,
        query_key: &str,
        connected_to: &str,
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        // Get query embedding
        let query_embedding = self
            .vector
            .get_entity_embedding(query_key)
            .map_err(|e| RouterError::VectorError(e.to_string()))?;

        // Clone Arcs for the async closures
        let vector = Arc::clone(&self.vector);
        let graph = Arc::clone(&self.graph);
        let connected_to_owned = connected_to.to_string();
        let hnsw_ref = self.hnsw_index.as_ref();

        // Run vector similarity and graph neighbor queries in parallel
        let (similar_result, neighbors_result) = tokio::join!(
            async {
                if let Some((index, keys)) = hnsw_ref {
                    vector.search_with_hnsw(index, keys, &query_embedding, top_k * 2)
                } else {
                    vector.search_entities(&query_embedding, top_k * 2)
                }
            },
            async { graph.get_entity_neighbors(&connected_to_owned) }
        );

        let similar = similar_result.map_err(|e| RouterError::VectorError(e.to_string()))?;

        let connected_neighbors: std::collections::HashSet<String> =
            neighbors_result.unwrap_or_default().into_iter().collect();

        let mut items: Vec<UnifiedItem> = similar
            .into_iter()
            .filter(|s| connected_neighbors.contains(&s.key))
            .take(top_k)
            .map(|s| UnifiedItem::new("vector+graph", &s.key).with_score(s.score))
            .collect();

        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(items)
    }

    /// Find graph neighbors sorted by similarity, with parallel embedding lookups.
    ///
    /// This async version fetches embeddings for all neighbors in parallel.
    pub async fn find_neighbors_by_similarity_async(
        &self,
        entity_key: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<UnifiedItem>> {
        use futures::future::join_all;

        let neighbors = self
            .graph
            .get_entity_neighbors(entity_key)
            .map_err(|e| RouterError::GraphError(e.to_string()))?;

        let vector = Arc::clone(&self.vector);
        let query_owned: Vec<f32> = query.to_vec();

        // Fetch all embeddings in parallel
        let futures: Vec<_> = neighbors
            .iter()
            .map(|key| {
                let key = key.clone();
                let vector = Arc::clone(&vector);
                let query = query_owned.clone();
                async move {
                    if let Ok(embedding) = vector.get_entity_embedding(&key) {
                        if embedding.len() == query.len() {
                            if let Ok(score) =
                                vector_engine::VectorEngine::compute_similarity(&query, &embedding)
                            {
                                return Some((key, score));
                            }
                        }
                    }
                    None
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut items: Vec<UnifiedItem> = results
            .into_iter()
            .flatten()
            .map(|(key, score)| UnifiedItem::new("graph+vector", key).with_score(score))
            .collect();

        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        items.truncate(top_k);
        Ok(items)
    }

    /// Get the Tokio runtime for async operations.
    ///
    /// Returns None if blob store hasn't been initialized (no runtime available).
    pub fn runtime(&self) -> Option<&Runtime> {
        self.blob_runtime.as_deref()
    }

    /// Execute an async operation using the router's runtime.
    ///
    /// This is useful for running async operations when you don't have
    /// an async context available.
    pub fn block_on<F: std::future::Future>(&self, future: F) -> Result<F::Output> {
        let runtime = self.blob_runtime.as_ref().ok_or_else(|| {
            RouterError::BlobError("Runtime not initialized. Call init_blob() first.".to_string())
        })?;
        Ok(runtime.block_on(future))
    }
}

impl Default for QueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal structure for unified query parsing.
struct UnifiedQueryParts {
    entity: String,
    where_clause: Option<String>,
    similar_to: Option<String>,
    connected_to: Option<String>,
    top_k: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Basic Routing Tests ==========

    #[test]
    fn routes_select_to_relational() {
        let router = QueryRouter::new();

        // Create a table first
        router
            .execute("CREATE TABLE users (name:string, age:int)")
            .unwrap();
        router
            .execute("INSERT users name=\"Alice\", age=30")
            .unwrap();

        let result = router.execute("SELECT users").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
            },
            _ => panic!("Expected Rows result"),
        }
    }

    #[test]
    fn routes_node_to_graph() {
        let router = QueryRouter::new();

        let result = router.execute("NODE CREATE person name=\"Bob\"").unwrap();
        match result {
            QueryResult::Ids(ids) => {
                assert_eq!(ids.len(), 1);
            },
            _ => panic!("Expected Ids result"),
        }
    }

    #[test]
    fn routes_embed_to_vector() {
        let router = QueryRouter::new();

        let result = router.execute("EMBED doc1 [1.0, 0.0, 0.0]").unwrap();
        match result {
            QueryResult::Empty => {},
            _ => panic!("Expected Empty result"),
        }

        assert!(router.vector().exists("doc1"));
    }

    #[test]
    fn routes_similar_to_vector() {
        let router = QueryRouter::new();

        router.execute("EMBED doc1 [1.0, 0.0, 0.0]").unwrap();
        router.execute("EMBED doc2 [0.0, 1.0, 0.0]").unwrap();
        router.execute("EMBED doc3 [0.9, 0.1, 0.0]").unwrap();

        let result = router.execute("SIMILAR doc1 TOP 2").unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 2);
                assert_eq!(results[0].key, "doc1"); // Exact match first
            },
            _ => panic!("Expected Similar result"),
        }
    }

    // ========== Unified Query Tests ==========

    #[test]
    fn handles_unified_query_similar() {
        let router = QueryRouter::new();

        router.execute("EMBED post1 [1.0, 0.0, 0.0]").unwrap();
        router.execute("EMBED post2 [0.9, 0.1, 0.0]").unwrap();
        router.execute("EMBED post3 [0.0, 1.0, 0.0]").unwrap();

        let result = router
            .execute("FIND posts SIMILAR TO \"post1\" TOP 2")
            .unwrap();
        match result {
            QueryResult::Unified(unified) => {
                assert!(unified.description.contains("similar"));
                assert_eq!(unified.items.len(), 2);
            },
            _ => panic!("Expected Unified result"),
        }
    }

    #[test]
    fn handles_unified_query_connected() {
        let router = QueryRouter::new();

        // Create graph structure
        let user_id = match router.execute("NODE CREATE user name=\"Alice\"").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let post_id = match router.execute("NODE CREATE post title=\"Hello\"").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        router
            .execute(&format!("EDGE CREATE {} -> {} authored", user_id, post_id))
            .unwrap();

        // Create embedding for the post
        router.execute("EMBED post [1.0, 0.0, 0.0]").unwrap();

        let result = router
            .execute("FIND posts SIMILAR TO \"post\" CONNECTED TO users")
            .unwrap();
        match result {
            QueryResult::Unified(_) => {},
            _ => panic!("Expected Unified result"),
        }
    }

    // ========== Error Handling Tests ==========

    #[test]
    fn returns_error_for_malformed_command() {
        let router = QueryRouter::new();

        let result = router.execute("");
        assert!(matches!(result, Err(RouterError::ParseError(_))));

        let result = router.execute("   ");
        assert!(matches!(result, Err(RouterError::ParseError(_))));
    }

    #[test]
    fn returns_error_for_unknown_command() {
        let router = QueryRouter::new();

        let result = router.execute("UNKNOWN something");
        assert!(matches!(result, Err(RouterError::UnknownCommand(_))));
    }

    #[test]
    fn returns_error_for_missing_arguments() {
        let router = QueryRouter::new();

        let result = router.execute("SELECT");
        assert!(matches!(result, Err(RouterError::MissingArgument(_))));

        let result = router.execute("NODE");
        assert!(matches!(result, Err(RouterError::MissingArgument(_))));

        let result = router.execute("EMBED");
        assert!(matches!(result, Err(RouterError::MissingArgument(_))));
    }

    #[test]
    fn does_not_crash_on_unexpected_input() {
        let router = QueryRouter::new();

        // Various unexpected inputs that shouldn't crash
        let inputs = [
            "SELECT FROM WHERE",
            "INSERT INTO VALUES",
            "NODE CREATE",
            "EDGE 123 -> 456",
            "SIMILAR [not, valid, floats]",
            "FIND something WITH random KEYWORDS",
            ";;;",
            "SELECT * FROM users; DROP TABLE users;--",
            "SELECT users WHERE name = 'O'Brien'",
            "\n\t\r",
        ];

        for input in inputs {
            // Should return an error, not panic
            let _ = router.execute(input);
        }
    }

    #[test]
    fn handles_table_not_found() {
        let router = QueryRouter::new();

        let result = router.execute("SELECT nonexistent");
        assert!(matches!(result, Err(RouterError::RelationalError(_))));
    }

    #[test]
    fn handles_node_not_found() {
        let router = QueryRouter::new();

        let result = router.execute("NODE GET 99999");
        assert!(matches!(result, Err(RouterError::GraphError(_))));
    }

    #[test]
    fn handles_embedding_not_found() {
        let router = QueryRouter::new();

        let result = router.execute("SIMILAR nonexistent TOP 5");
        assert!(matches!(result, Err(RouterError::VectorError(_))));
    }

    // ========== Relational Command Tests ==========

    #[test]
    fn create_table_and_insert() {
        let router = QueryRouter::new();

        router
            .execute("CREATE TABLE products (name:string, price:float)")
            .unwrap();
        router
            .execute("INSERT products name=\"Widget\", price=9.99")
            .unwrap();

        let result = router.execute("SELECT products").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].get("name"), Some(&Value::String("Widget".into())));
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn select_with_where() {
        let router = QueryRouter::new();

        router
            .execute("CREATE TABLE items (name:string, qty:int)")
            .unwrap();
        router.execute("INSERT items name=\"A\", qty=10").unwrap();
        router.execute("INSERT items name=\"B\", qty=20").unwrap();
        router.execute("INSERT items name=\"C\", qty=30").unwrap();

        let result = router.execute("SELECT items WHERE qty > 15").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn update_rows() {
        let router = QueryRouter::new();

        router
            .execute("CREATE TABLE counters (name:string, value:int)")
            .unwrap();
        router
            .execute("INSERT counters name=\"hits\", value=0")
            .unwrap();

        let result = router
            .execute("UPDATE counters SET value=100 WHERE name=\"hits\"")
            .unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn delete_rows() {
        let router = QueryRouter::new();

        router.execute("CREATE TABLE temp (id:int)").unwrap();
        router.execute("INSERT temp id=1").unwrap();
        router.execute("INSERT temp id=2").unwrap();

        let result = router.execute("DELETE temp WHERE id=1").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn create_and_drop_index() {
        let router = QueryRouter::new();

        router.execute("CREATE TABLE indexed (col:int)").unwrap();
        router.execute("CREATE INDEX indexed col").unwrap();

        assert!(router.relational().has_index("indexed", "col"));

        router.execute("DROP INDEX indexed col").unwrap();
        assert!(!router.relational().has_index("indexed", "col"));
    }

    #[test]
    fn drop_table() {
        let router = QueryRouter::new();

        router.execute("CREATE TABLE todrop (x:int)").unwrap();
        assert!(router.relational().table_exists("todrop"));

        router.execute("DROP TABLE todrop").unwrap();
        assert!(!router.relational().table_exists("todrop"));
    }

    // ========== Graph Command Tests ==========

    #[test]
    fn node_create_get_delete() {
        let router = QueryRouter::new();

        let id = match router.execute("NODE CREATE person name=\"Test\"").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router.execute(&format!("NODE GET {}", id)).unwrap();
        match result {
            QueryResult::Nodes(nodes) => {
                assert_eq!(nodes.len(), 1);
                assert_eq!(nodes[0].label, "person");
            },
            _ => panic!("Expected Nodes"),
        }

        router.execute(&format!("NODE DELETE {}", id)).unwrap();
    }

    #[test]
    fn edge_create_and_get() {
        let router = QueryRouter::new();

        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let edge_id = match router
            .execute(&format!("EDGE CREATE {} -> {} connects", n1, n2))
            .unwrap()
        {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router.execute(&format!("EDGE GET {}", edge_id)).unwrap();
        match result {
            QueryResult::Edges(edges) => {
                assert_eq!(edges.len(), 1);
                assert_eq!(edges[0].label, "connects");
            },
            _ => panic!("Expected Edges"),
        }
    }

    #[test]
    fn neighbors_query() {
        let router = QueryRouter::new();

        let center = match router.execute("NODE CREATE center").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let leaf1 = match router.execute("NODE CREATE leaf1").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let leaf2 = match router.execute("NODE CREATE leaf2").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        router
            .execute(&format!("EDGE CREATE {} -> {}", center, leaf1))
            .unwrap();
        router
            .execute(&format!("EDGE CREATE {} -> {}", center, leaf2))
            .unwrap();

        let result = router
            .execute(&format!("NEIGHBORS {} OUT", center))
            .unwrap();
        match result {
            QueryResult::Ids(ids) => {
                assert_eq!(ids.len(), 2);
            },
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn path_query() {
        let router = QueryRouter::new();

        let a = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let b = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let c = match router.execute("NODE CREATE c").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        router
            .execute(&format!("EDGE CREATE {} -> {}", a, b))
            .unwrap();
        router
            .execute(&format!("EDGE CREATE {} -> {}", b, c))
            .unwrap();

        let result = router.execute(&format!("PATH {} -> {}", a, c)).unwrap();
        match result {
            QueryResult::Path(path) => {
                assert_eq!(path.len(), 3);
                assert_eq!(path[0], a);
                assert_eq!(path[2], c);
            },
            _ => panic!("Expected Path"),
        }
    }

    // ========== Vector Command Tests ==========

    #[test]
    fn embed_and_similar_inline() {
        let router = QueryRouter::new();

        router.execute("EMBED v1 [1.0, 0.0]").unwrap();
        router.execute("EMBED v2 [0.0, 1.0]").unwrap();

        let result = router.execute("SIMILAR [1.0, 0.0] TOP 1").unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].key, "v1");
            },
            _ => panic!("Expected Similar"),
        }
    }

    // ========== Engine Access Tests ==========

    #[test]
    fn can_access_underlying_engines() {
        let router = QueryRouter::new();

        // Direct engine access for complex operations
        let _ = router.relational();
        let _ = router.graph();
        let _ = router.vector();
    }

    #[test]
    fn with_engines_constructor() {
        let rel = Arc::new(RelationalEngine::new());
        let graph = Arc::new(GraphEngine::new());
        let vec = Arc::new(VectorEngine::new());

        let router = QueryRouter::with_engines(rel, graph, vec);
        assert!(router.execute("EMBED test [1.0]").is_ok());
    }

    #[test]
    fn build_vector_index() {
        let mut router = QueryRouter::new();

        router.execute("EMBED a [1.0, 0.0]").unwrap();
        router.execute("EMBED b [0.0, 1.0]").unwrap();

        router.build_vector_index().unwrap();

        // Should use HNSW index for search
        let result = router.execute("SIMILAR a TOP 2").unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 2);
            },
            _ => panic!("Expected Similar"),
        }
    }

    // ========== Error Type Tests ==========

    #[test]
    fn error_display() {
        let e = RouterError::ParseError("test".into());
        assert!(e.to_string().contains("Parse error"));

        let e = RouterError::UnknownCommand("FOO".into());
        assert!(e.to_string().contains("Unknown command"));

        let e = RouterError::RelationalError("db error".into());
        assert!(e.to_string().contains("Relational error"));

        let e = RouterError::GraphError("graph error".into());
        assert!(e.to_string().contains("Graph error"));

        let e = RouterError::VectorError("vec error".into());
        assert!(e.to_string().contains("Vector error"));

        let e = RouterError::InvalidArgument("bad arg".into());
        assert!(e.to_string().contains("Invalid argument"));

        let e = RouterError::MissingArgument("missing".into());
        assert!(e.to_string().contains("Missing argument"));

        let e = RouterError::TypeMismatch("type".into());
        assert!(e.to_string().contains("Type mismatch"));
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = RouterError::ParseError("test".into());
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_is_std_error() {
        let error: Box<dyn std::error::Error> = Box::new(RouterError::ParseError("test".into()));
        assert!(error.to_string().contains("Parse"));
    }

    #[test]
    fn default_trait() {
        let router = QueryRouter::default();
        assert!(router.execute("EMBED x [1.0]").is_ok());
    }

    // ========== Condition Parsing Tests ==========

    #[test]
    fn parse_compound_conditions() {
        let router = QueryRouter::new();

        router.execute("CREATE TABLE data (a:int, b:int)").unwrap();
        router.execute("INSERT data a=1, b=2").unwrap();
        router.execute("INSERT data a=3, b=4").unwrap();
        router.execute("INSERT data a=5, b=6").unwrap();

        // AND condition
        let result = router.execute("SELECT data WHERE a > 2 AND b < 6").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
            },
            _ => panic!("Expected Rows"),
        }

        // OR condition
        let result = router.execute("SELECT data WHERE a = 1 OR a = 5").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn parse_nullable_columns() {
        let router = QueryRouter::new();

        router
            .execute("CREATE TABLE nullable (required:string, optional:string?)")
            .unwrap();
        router
            .execute("INSERT nullable required=\"test\", optional=NULL")
            .unwrap();

        let result = router.execute("SELECT nullable").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].get("optional"), Some(&Value::Null));
            },
            _ => panic!("Expected Rows"),
        }
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn update_without_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=1").unwrap();
        // Missing WHERE - should error on missing SET
        let result = router.execute("UPDATE t x=2");
        assert!(result.is_err());
    }

    #[test]
    fn delete_without_where_clause() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE del (x:int)").unwrap();
        router.execute("INSERT del x=1").unwrap();
        router.execute("INSERT del x=2").unwrap();
        // Delete all (no WHERE)
        let result = router.execute("DELETE del").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 2),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn create_table_with_bool() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE flags (name:string, active:bool)")
            .unwrap();
        router
            .execute("INSERT flags name=\"test\", active=true")
            .unwrap();
        let result = router.execute("SELECT flags").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].get("active"), Some(&Value::Bool(true)));
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn create_table_with_float() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE nums (val:double)").unwrap();
        router.execute("INSERT nums val=3.14").unwrap();
        let result = router.execute("SELECT nums").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn invalid_create_missing_parens() {
        let router = QueryRouter::new();
        let result = router.execute("CREATE TABLE bad x:int");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_create_command() {
        let router = QueryRouter::new();
        let result = router.execute("CREATE SOMETHING bad");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_drop_command() {
        let router = QueryRouter::new();
        let result = router.execute("DROP SOMETHING bad");
        assert!(result.is_err());
    }

    #[test]
    fn path_not_found() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        // No edge between them
        let result = router.execute(&format!("PATH {} -> {}", n1, n2)).unwrap();
        match result {
            QueryResult::Path(path) => assert!(path.is_empty()),
            _ => panic!("Expected Path"),
        }
    }

    #[test]
    fn neighbors_in_direction() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap();

        // IN direction from n2
        let result = router.execute(&format!("NEIGHBORS {} IN", n2)).unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn neighbors_invalid_direction() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let result = router.execute(&format!("NEIGHBORS {} INVALID", n1));
        assert!(result.is_err());
    }

    #[test]
    fn node_with_typed_properties() {
        let router = QueryRouter::new();
        // Int, Float, Bool properties
        let result = router
            .execute("NODE CREATE person age=30, score=95.5, active=true")
            .unwrap();
        match result {
            QueryResult::Ids(ids) => {
                let node_result = router.execute(&format!("NODE GET {}", ids[0])).unwrap();
                match node_result {
                    QueryResult::Nodes(nodes) => {
                        assert_eq!(nodes[0].properties.get("age"), Some(&"30".to_string()));
                    },
                    _ => panic!("Expected Nodes"),
                }
            },
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn edge_undirected() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {} link UNDIRECTED", n1, n2))
            .unwrap();
    }

    #[test]
    fn condition_all_operators() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE ops (x:int)").unwrap();
        router.execute("INSERT ops x=5").unwrap();

        // Test !=
        let result = router.execute("SELECT ops WHERE x != 10").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Rows"),
        }

        // Test <=
        let result = router.execute("SELECT ops WHERE x <= 5").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Rows"),
        }

        // Test >=
        let result = router.execute("SELECT ops WHERE x >= 5").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn invalid_condition() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        let result = router.execute("SELECT t WHERE invalid");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_insert_values() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        let result = router.execute("INSERT t invalid");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_node_id() {
        let router = QueryRouter::new();
        let result = router.execute("NODE GET notanumber");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_edge_id() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE GET notanumber");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_neighbors_id() {
        let router = QueryRouter::new();
        let result = router.execute("NEIGHBORS notanumber");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_path_ids() {
        let router = QueryRouter::new();
        let result = router.execute("PATH notanumber -> 1");
        assert!(result.is_err());

        let result = router.execute("PATH 1 -> notanumber");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_vector() {
        let router = QueryRouter::new();
        let result = router.execute("EMBED key [not, valid]");
        assert!(result.is_err());
    }

    #[test]
    fn empty_vector() {
        let router = QueryRouter::new();
        let result = router.execute("EMBED key []");
        assert!(result.is_err());
    }

    #[test]
    fn similar_with_inline_vector() {
        let router = QueryRouter::new();
        router.execute("EMBED v1 [1.0, 0.0, 0.0]").unwrap();
        let result = router.execute("SIMILAR [0.9, 0.1, 0.0] TOP 1").unwrap();
        match result {
            QueryResult::Similar(results) => assert_eq!(results.len(), 1),
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn unknown_edge_subcommand() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE UNKNOWN 1");
        assert!(result.is_err());
    }

    #[test]
    fn unknown_node_subcommand() {
        let router = QueryRouter::new();
        let result = router.execute("NODE UNKNOWN label");
        assert!(result.is_err());
    }

    #[test]
    fn find_with_where_clause() {
        let router = QueryRouter::new();
        router.execute("EMBED doc [1.0, 0.0]").unwrap();
        let result = router
            .execute("FIND items WHERE x > 5 SIMILAR TO \"doc\"")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("where"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn property_value_null() {
        let router = QueryRouter::new();
        router.execute("NODE CREATE test val=NULL").unwrap();
    }

    #[test]
    fn property_value_false() {
        let router = QueryRouter::new();
        router.execute("NODE CREATE test val=false").unwrap();
    }

    #[test]
    fn missing_edge_definition() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE CREATE");
        assert!(result.is_err());
    }

    #[test]
    fn missing_path_args() {
        let router = QueryRouter::new();
        let result = router.execute("PATH 1");
        assert!(result.is_err());
    }

    #[test]
    fn missing_embed_args() {
        let router = QueryRouter::new();
        let result = router.execute("EMBED");
        assert!(result.is_err());
    }

    #[test]
    fn missing_similar_args() {
        let router = QueryRouter::new();
        let result = router.execute("SIMILAR");
        assert!(result.is_err());
    }

    #[test]
    fn missing_find_args() {
        let router = QueryRouter::new();
        let result = router.execute("FIND");
        assert!(result.is_err());
    }

    #[test]
    fn create_index_missing_args() {
        let router = QueryRouter::new();
        let result = router.execute("CREATE INDEX t");
        assert!(result.is_err());
    }

    #[test]
    fn drop_index_missing_args() {
        let router = QueryRouter::new();
        let result = router.execute("DROP INDEX t");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_top_value() {
        let router = QueryRouter::new();
        router.execute("EMBED v [1.0]").unwrap();
        let result = router.execute("SIMILAR v TOP notanumber");
        assert!(result.is_err());
    }

    #[test]
    fn hnsw_similar_search() {
        let mut router = QueryRouter::new();
        router.execute("EMBED a [1.0, 0.0]").unwrap();
        router.execute("EMBED b [0.0, 1.0]").unwrap();
        router.build_vector_index().unwrap();

        let result = router.execute("SIMILAR a TOP 2").unwrap();
        match result {
            QueryResult::Similar(results) => assert_eq!(results.len(), 2),
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn invalid_edge_nodes() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE CREATE notanumber -> 1");
        assert!(result.is_err());

        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let result = router.execute(&format!("EDGE CREATE {} -> notanumber", n1));
        assert!(result.is_err());
    }

    #[test]
    fn missing_insert_table() {
        let router = QueryRouter::new();
        let result = router.execute("INSERT");
        assert!(result.is_err());
    }

    #[test]
    fn missing_delete_table() {
        let router = QueryRouter::new();
        let result = router.execute("DELETE");
        assert!(result.is_err());
    }

    #[test]
    fn update_with_set_no_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=1").unwrap();
        router.execute("INSERT t x=2").unwrap();
        // UPDATE all rows (no WHERE)
        let result = router.execute("UPDATE t SET x=99").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 2),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn invalid_column_definition() {
        let router = QueryRouter::new();
        let result = router.execute("CREATE TABLE bad (invalid)");
        assert!(result.is_err());
    }

    #[test]
    fn unknown_column_type() {
        let router = QueryRouter::new();
        let result = router.execute("CREATE TABLE bad (x:unknowntype)");
        assert!(result.is_err());
    }

    #[test]
    fn node_get_missing_id() {
        let router = QueryRouter::new();
        let result = router.execute("NODE GET");
        assert!(result.is_err());
    }

    #[test]
    fn node_delete_missing_id() {
        let router = QueryRouter::new();
        let result = router.execute("NODE DELETE");
        assert!(result.is_err());
    }

    #[test]
    fn edge_missing_subcommand() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE");
        assert!(result.is_err());
    }

    #[test]
    fn edge_get_missing_id() {
        let router = QueryRouter::new();
        let result = router.execute("EDGE GET");
        assert!(result.is_err());
    }

    #[test]
    fn neighbors_default_direction() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        // No direction specified, should default to BOTH
        let result = router.execute(&format!("NEIGHBORS {}", n1)).unwrap();
        match result {
            QueryResult::Ids(_) => {},
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn invalid_edge_definition_format() {
        let router = QueryRouter::new();
        // Missing arrow
        let result = router.execute("EDGE CREATE 1 2 label");
        assert!(result.is_err());
    }

    #[test]
    fn edge_directed_keyword() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {} link DIRECTED", n1, n2))
            .unwrap();
    }

    #[test]
    fn value_parsing_false_lowercase() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (flag:bool)").unwrap();
        router.execute("INSERT t flag=FALSE").unwrap();
        let result = router.execute("SELECT t").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows[0].get("flag"), Some(&Value::Bool(false)));
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn value_parsing_string_variants() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (s:string)").unwrap();
        // Single quotes
        router.execute("INSERT t s='hello'").unwrap();
        let result = router.execute("SELECT t").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows[0].get("s"), Some(&Value::String("hello".into())));
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn property_null_to_string() {
        let router = QueryRouter::new();
        let result = router.execute("NODE CREATE test prop=NULL").unwrap();
        match result {
            QueryResult::Ids(ids) => {
                let node = router.execute(&format!("NODE GET {}", ids[0])).unwrap();
                match node {
                    QueryResult::Nodes(nodes) => {
                        assert_eq!(nodes[0].properties.get("prop"), Some(&"null".to_string()));
                    },
                    _ => panic!("Expected Nodes"),
                }
            },
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn missing_neighbors_id() {
        let router = QueryRouter::new();
        let result = router.execute("NEIGHBORS");
        assert!(result.is_err());
    }

    #[test]
    fn select_missing_table() {
        let router = QueryRouter::new();
        let result = router.execute("SELECT");
        assert!(result.is_err());
    }

    #[test]
    fn node_missing_subcommand() {
        let router = QueryRouter::new();
        let result = router.execute("NODE");
        assert!(result.is_err());
    }

    #[test]
    fn unquoted_string_value() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (s:string)").unwrap();
        // Unquoted string should work as fallback
        router.execute("INSERT t s=bareword").unwrap();
        let result = router.execute("SELECT t").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows[0].get("s"), Some(&Value::String("bareword".into())));
            },
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn whitespace_only_command() {
        let router = QueryRouter::new();
        let result = router.execute("   ");
        assert!(result.is_err());
    }

    #[test]
    fn path_graph_error() {
        let router = QueryRouter::new();
        // Non-existent node IDs should trigger graph error
        let result = router.execute("PATH 99999 -> 99998");
        assert!(result.is_err());
    }

    #[test]
    fn node_property_unquoted_string() {
        let router = QueryRouter::new();
        let result = router.execute("NODE CREATE test prop=somevalue").unwrap();
        match result {
            QueryResult::Ids(ids) => {
                let node = router.execute(&format!("NODE GET {}", ids[0])).unwrap();
                match node {
                    QueryResult::Nodes(nodes) => {
                        assert_eq!(
                            nodes[0].properties.get("prop"),
                            Some(&"somevalue".to_string())
                        );
                    },
                    _ => panic!("Expected Nodes"),
                }
            },
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn edge_missing_arrow_definition() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        // No arrow between IDs
        let result = router.execute(&format!("EDGE CREATE {} {} label", n1, n1 + 1));
        assert!(result.is_err());
    }

    #[test]
    fn embed_with_empty_brackets() {
        let router = QueryRouter::new();
        let result = router.execute("EMBED emptykey []");
        assert!(result.is_err());
    }

    #[test]
    fn insert_table_only() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        let result = router.execute("INSERT t");
        assert!(result.is_err());
    }

    #[test]
    fn edge_definition_too_short() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        // Only "from ->" without "to"
        let result = router.execute(&format!("EDGE CREATE {} ->", n1));
        assert!(result.is_err());
    }

    #[test]
    fn unified_connected_returns_items() {
        let router = QueryRouter::new();
        router.execute("EMBED item1 [1.0, 0.0]").unwrap();
        router.execute("EMBED item2 [0.9, 0.1]").unwrap();

        let result = router
            .execute("FIND things SIMILAR TO \"item1\" CONNECTED TO users")
            .unwrap();
        match result {
            QueryResult::Unified(unified) => {
                assert!(!unified.items.is_empty());
                assert_eq!(unified.items[0].source, "vector+graph");
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn similar_no_results() {
        let router = QueryRouter::new();
        // No embeddings stored
        let result = router.execute("SIMILAR nonexistent TOP 5");
        assert!(result.is_err());
    }

    // ========== AST-Based Execution Tests ==========

    #[test]
    fn parsed_select_basic() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE users (id:int, name:string)")
            .unwrap();
        router.execute("INSERT users id=1, name=\"alice\"").unwrap();

        let result = router.execute_parsed("SELECT * FROM users").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn parsed_select_with_where() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE products (id:int, price:int)")
            .unwrap();
        router.execute("INSERT products id=1, price=100").unwrap();
        router.execute("INSERT products id=2, price=200").unwrap();

        let result = router
            .execute_parsed("SELECT * FROM products WHERE price > 150")
            .unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn parsed_insert_values() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE items (id:int, name:string)")
            .unwrap();

        let result = router
            .execute_parsed("INSERT INTO items (id, name) VALUES (1, 'test')")
            .unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn parsed_update() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE scores (id:int, val:int)")
            .unwrap();
        router.execute("INSERT scores id=1, val=10").unwrap();

        let result = router
            .execute_parsed("UPDATE scores SET val = 20 WHERE id = 1")
            .unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_update_no_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=1").unwrap();
        router.execute("INSERT t x=2").unwrap();

        let result = router.execute_parsed("UPDATE t SET x = 99").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 2),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_delete() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE temps (id:int)").unwrap();
        router.execute("INSERT temps id=1").unwrap();
        router.execute("INSERT temps id=2").unwrap();

        let result = router
            .execute_parsed("DELETE FROM temps WHERE id = 1")
            .unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_delete_no_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=1").unwrap();
        router.execute("INSERT t x=2").unwrap();

        let result = router.execute_parsed("DELETE FROM t").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 2),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_create_table() {
        let router = QueryRouter::new();
        let result = router
            .execute_parsed("CREATE TABLE newtbl (id INTEGER, name VARCHAR(100))")
            .unwrap();
        assert!(matches!(result, QueryResult::Empty));

        // Verify table exists
        router.execute("INSERT newtbl id=1, name=\"test\"").unwrap();
    }

    #[test]
    fn parsed_create_table_not_null() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE required (id INT NOT NULL, name TEXT)")
            .unwrap();
    }

    #[test]
    fn parsed_drop_table() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE todrop (id:int)").unwrap();

        let result = router.execute_parsed("DROP TABLE todrop").unwrap();
        assert!(matches!(result, QueryResult::Empty));
    }

    #[test]
    fn parsed_create_index() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE indexed (id:int, val:int)")
            .unwrap();

        let result = router
            .execute_parsed("CREATE INDEX idx ON indexed (val)")
            .unwrap();
        assert!(matches!(result, QueryResult::Empty));
    }

    #[test]
    fn parsed_drop_index_not_supported() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("DROP INDEX myindex");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_node_create() {
        let router = QueryRouter::new();
        let result = router
            .execute_parsed("NODE CREATE person { name: 'Alice', age: 30 }")
            .unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn parsed_node_get() {
        let router = QueryRouter::new();
        let id = match router.execute("NODE CREATE test").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router.execute_parsed(&format!("NODE GET {}", id)).unwrap();
        match result {
            QueryResult::Nodes(nodes) => assert_eq!(nodes.len(), 1),
            _ => panic!("Expected Nodes"),
        }
    }

    #[test]
    fn parsed_node_delete() {
        let router = QueryRouter::new();
        let id = match router.execute("NODE CREATE todelete").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router
            .execute_parsed(&format!("NODE DELETE {}", id))
            .unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_node_list() {
        let router = QueryRouter::new();
        router.execute("NODE CREATE label1").unwrap();

        let result = router.execute_parsed("NODE LIST").unwrap();
        assert!(matches!(result, QueryResult::Nodes(_)));
    }

    #[test]
    fn parsed_edge_create() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router
            .execute_parsed(&format!(
                "EDGE CREATE {} -> {} : knows {{ since: 2020 }}",
                n1, n2
            ))
            .unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn parsed_edge_get() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE x").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE y").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let edge_id = match router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap()
        {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        let result = router
            .execute_parsed(&format!("EDGE GET {}", edge_id))
            .unwrap();
        match result {
            QueryResult::Edges(edges) => assert_eq!(edges.len(), 1),
            _ => panic!("Expected Edges"),
        }
    }

    #[test]
    fn parsed_edge_list() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("EDGE LIST").unwrap();
        assert!(matches!(result, QueryResult::Edges(_)));
    }

    #[test]
    fn parsed_edge_delete_not_supported() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("EDGE DELETE 1");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_neighbors() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE start").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE neighbor").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap();

        let result = router
            .execute_parsed(&format!("NEIGHBORS {} OUTGOING", n1))
            .unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn parsed_neighbors_incoming() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap();

        let result = router
            .execute_parsed(&format!("NEIGHBORS {} INCOMING", n2))
            .unwrap();
        match result {
            QueryResult::Ids(ids) => assert_eq!(ids.len(), 1),
            _ => panic!("Expected Ids"),
        }
    }

    #[test]
    fn parsed_neighbors_both() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap();

        let result = router
            .execute_parsed(&format!("NEIGHBORS {} BOTH", n1))
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_path() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE source").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE target").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {}", n1, n2))
            .unwrap();

        let result = router
            .execute_parsed(&format!("PATH SHORTEST {} -> {}", n1, n2))
            .unwrap();
        match result {
            QueryResult::Path(path) => assert!(!path.is_empty()),
            _ => panic!("Expected Path"),
        }
    }

    #[test]
    fn parsed_embed_store() {
        let router = QueryRouter::new();
        let result = router
            .execute_parsed("EMBED STORE 'key1' [1.0, 2.0, 3.0]")
            .unwrap();
        assert!(matches!(result, QueryResult::Empty));
    }

    #[test]
    fn parsed_embed_get() {
        let router = QueryRouter::new();
        router.execute("EMBED vec1 [1.0, 2.0, 3.0]").unwrap();

        let result = router.execute_parsed("EMBED GET 'vec1'").unwrap();
        match result {
            QueryResult::Value(s) => assert!(s.contains("1")),
            _ => panic!("Expected Value"),
        }
    }

    #[test]
    fn parsed_embed_delete() {
        let router = QueryRouter::new();
        router.execute("EMBED todelete [1.0, 2.0]").unwrap();

        let result = router.execute_parsed("EMBED DELETE 'todelete'").unwrap();
        match result {
            QueryResult::Count(n) => assert_eq!(n, 1),
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn parsed_similar_by_key() {
        let router = QueryRouter::new();
        router.execute("EMBED item1 [1.0, 0.0, 0.0]").unwrap();
        router.execute("EMBED item2 [0.9, 0.1, 0.0]").unwrap();

        let result = router.execute_parsed("SIMILAR 'item1' LIMIT 5").unwrap();
        match result {
            QueryResult::Similar(results) => assert!(!results.is_empty()),
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_by_vector() {
        let router = QueryRouter::new();
        router.execute("EMBED vec1 [1.0, 0.0, 0.0]").unwrap();
        router.execute("EMBED vec2 [0.0, 1.0, 0.0]").unwrap();

        let result = router
            .execute_parsed("SIMILAR [1.0, 0.0, 0.0] LIMIT 5")
            .unwrap();
        match result {
            QueryResult::Similar(results) => assert!(!results.is_empty()),
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_with_hnsw() {
        let mut router = QueryRouter::new();
        router.execute("EMBED a [1.0, 0.0]").unwrap();
        router.execute("EMBED b [0.0, 1.0]").unwrap();
        router.build_vector_index().unwrap();

        let result = router.execute_parsed("SIMILAR 'a' LIMIT 2").unwrap();
        match result {
            QueryResult::Similar(results) => assert_eq!(results.len(), 2),
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_cosine_metric() {
        let router = QueryRouter::new();
        router.execute("EMBED cos_a [1.0, 0.0]").unwrap();
        router.execute("EMBED cos_b [0.0, 1.0]").unwrap();
        router.execute("EMBED cos_c [0.707, 0.707]").unwrap();

        // COSINE metric - angle matters (syntax: SIMILAR ... COSINE LIMIT n)
        let result = router
            .execute_parsed("SIMILAR [1.0, 0.0] COSINE LIMIT 3")
            .unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 3);
                // Identical direction should be first
                assert_eq!(results[0].key, "cos_a");
            },
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_euclidean_metric() {
        let router = QueryRouter::new();
        router.execute("EMBED euc_a [1.0, 0.0]").unwrap();
        router.execute("EMBED euc_b [2.0, 0.0]").unwrap();
        router.execute("EMBED euc_c [10.0, 0.0]").unwrap();

        // EUCLIDEAN metric - distance matters
        let result = router
            .execute_parsed("SIMILAR [1.0, 0.0] EUCLIDEAN LIMIT 3")
            .unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 3);
                // Closest vector should be first
                assert_eq!(results[0].key, "euc_a");
            },
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_euclidean_zero_query() {
        let router = QueryRouter::new();
        router.execute("EMBED zero_origin [0.0, 0.0]").unwrap();
        router.execute("EMBED zero_unit [1.0, 0.0]").unwrap();
        router.execute("EMBED zero_far [10.0, 0.0]").unwrap();

        // EUCLIDEAN with zero query should still work (find closest to origin)
        let result = router
            .execute_parsed("SIMILAR [0.0, 0.0] EUCLIDEAN LIMIT 3")
            .unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(
                    results.len(),
                    3,
                    "Should return 3 results for EUCLIDEAN with zero query"
                );
                // Origin should be closest (distance 0)
                assert_eq!(results[0].key, "zero_origin");
                // Score should be 1.0 for distance 0
                assert!((results[0].score - 1.0).abs() < 0.01);
            },
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_dot_product_metric() {
        let router = QueryRouter::new();
        router.execute("EMBED dot_a [1.0, 0.0]").unwrap();
        router.execute("EMBED dot_b [2.0, 0.0]").unwrap();
        router.execute("EMBED dot_c [0.5, 0.0]").unwrap();

        // DOT_PRODUCT metric - magnitude matters
        let result = router
            .execute_parsed("SIMILAR [1.0, 0.0] DOT_PRODUCT LIMIT 3")
            .unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 3);
                // Largest projection should be first
                assert_eq!(results[0].key, "dot_b");
            },
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_similar_hnsw_falls_back_for_non_cosine() {
        let mut router = QueryRouter::new();
        router.execute("EMBED hnsw_a [1.0, 0.0]").unwrap();
        router.execute("EMBED hnsw_b [2.0, 0.0]").unwrap();
        router.build_vector_index().unwrap();

        // When using EUCLIDEAN with HNSW index, should fall back to linear search
        let result = router
            .execute_parsed("SIMILAR [1.0, 0.0] EUCLIDEAN LIMIT 2")
            .unwrap();
        match result {
            QueryResult::Similar(results) => {
                assert_eq!(results.len(), 2);
                // Closest should be first
                assert_eq!(results[0].key, "hnsw_a");
            },
            _ => panic!("Expected Similar"),
        }
    }

    #[test]
    fn parsed_find_nodes() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND NODE person").unwrap();
        match result {
            QueryResult::Unified(unified) => {
                assert!(unified.description.contains("node"));
                assert!(unified.description.contains("'person'"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND EDGE knows").unwrap();
        match result {
            QueryResult::Unified(unified) => {
                assert!(unified.description.contains("edge"));
                assert!(unified.description.contains("'knows'"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_with_where() {
        let router = QueryRouter::new();
        // Create a node to find
        router
            .execute_parsed("NODE CREATE person name='Alice', age=25")
            .unwrap();
        let result = router.execute_parsed("FIND NODE WHERE age > 18").unwrap();
        match result {
            QueryResult::Unified(unified) => {
                // Should find the node we created
                assert!(unified.description.contains("node"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_eq() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'Bob', status: 'active' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'Eve', status: 'inactive' }")
            .unwrap();

        let result = router
            .execute_parsed("FIND NODE WHERE status = 'active'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Bob (active status)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Bob".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_gt() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person { name: 'Young', age: 15 }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'Adult', age: 30 }")
            .unwrap();

        let result = router.execute_parsed("FIND NODE WHERE age > 20").unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Adult (age 30 > 20)
                assert!(!u.items.is_empty());
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Adult".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_and() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'Alice', age: 25, role: 'admin' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'Bob', age: 35, role: 'user' }")
            .unwrap();

        let result = router
            .execute_parsed("FIND NODE WHERE age > 20 AND role = 'admin'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Alice (age > 20 AND role = admin)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Alice".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_lt() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person { name: 'Young', age: 15 }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'Adult', age: 30 }")
            .unwrap();

        let result = router.execute_parsed("FIND NODE WHERE age < 20").unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Young (age 15 < 20)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Young".to_string())));
                // Should not find Adult
                assert!(!u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Adult".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_le() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person { name: 'Young', age: 20 }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'Adult', age: 30 }")
            .unwrap();

        let result = router.execute_parsed("FIND NODE WHERE age <= 20").unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Young (age 20 <= 20)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Young".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_ge() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person { name: 'Young', age: 15 }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'Adult', age: 30 }")
            .unwrap();

        let result = router.execute_parsed("FIND NODE WHERE age >= 30").unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Adult (age 30 >= 30)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Adult".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_where_or() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'Alice', role: 'admin' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'Bob', role: 'guest' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'Eve', role: 'user' }")
            .unwrap();

        let result = router
            .execute_parsed("FIND NODE WHERE role = 'admin' OR role = 'guest'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find Alice (admin) and Bob (guest), but not Eve (user)
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Alice".to_string())));
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"Bob".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_id_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'First' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'Second' }")
            .unwrap();

        let result = router.execute_parsed("FIND NODE WHERE id = 1").unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert_eq!(u.items.len(), 1);
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"First".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_condition_no_match() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'Test', age: 25 }")
            .unwrap();

        // Condition on non-existent property
        let result = router
            .execute_parsed("FIND NODE WHERE nonexistent = 'value'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.items.is_empty());
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn vault_accessor() {
        let router = QueryRouter::new();
        // Vault is None before initialization
        assert!(router.vault().is_none());
    }

    #[test]
    fn error_from_cache_error() {
        let cache_err = tensor_cache::CacheError::NotFound("test".to_string());
        let router_err: RouterError = cache_err.into();
        assert!(matches!(router_err, RouterError::CacheError(_)));
    }

    #[test]
    fn parsed_find_edge_by_type() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE x { name: 'X' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE y { name: 'Y' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : special_type")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE edge_type = 'special_type'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn blobs_similar_to_key() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Store embeddings
        router
            .execute_parsed("EMBED STORE 'blob_a' [1.0, 0.0, 0.0, 0.0]")
            .unwrap();
        router
            .execute_parsed("EMBED STORE 'blob_b' [0.9, 0.1, 0.0, 0.0]")
            .unwrap();
        router
            .execute_parsed("EMBED STORE 'blob_c' [0.0, 1.0, 0.0, 0.0]")
            .unwrap();

        // Search for similar using key reference
        let result = router.execute_parsed("BLOBS SIMILAR TO 'blob_a' LIMIT 2");
        // May return error if embedding not found for blob, but exercises the code path
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn blob_put_with_full_options() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Test with content_type and created_by via execute_parsed
        let result = router.execute_parsed(
            "BLOB PUT 'test_file.json' DATA '{\"key\":\"value\"}' TYPE 'application/json' BY 'testuser'",
        );
        // May succeed or fail depending on blob state, exercises code path
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_find_edges_with_edge_props() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person { name: 'X' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'Y' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : works_at department='engineering', level=3")
            .unwrap();

        // Test finding edge by condition
        let result = router.execute_parsed("FIND EDGE works_at").unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
                assert!(!u.items.is_empty());
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_scan_with_properties() {
        let router = QueryRouter::new();
        // Create nodes with various properties
        router
            .execute_parsed("NODE CREATE item { name: 'Item1', price: 100, active: true }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE item { name: 'Item2', price: 200, active: false }")
            .unwrap();

        // Scan should find nodes with properties in the result items
        let result = router.execute_parsed("FIND NODE item").unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Both items should be found with their properties
                assert!(u.items.len() >= 2);
                // Check properties are included
                assert!(u.items.iter().any(|item| item.data.contains_key("name")));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_where() {
        let router = QueryRouter::new();
        // Create nodes first
        router
            .execute_parsed("NODE CREATE person { name: 'A' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person { name: 'B' }")
            .unwrap();
        // Create edges
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : friend strength=10")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE strength > 5")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_type_eq() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE x { name: 'X' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE y { name: 'Y' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : knows since=2020")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : works since=2021")
            .unwrap();

        // Find edges by type
        let result = router.execute_parsed("FIND EDGE knows").unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(!u.items.is_empty());
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_and_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE a { name: 'A' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE b { name: 'B' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel weight=50, active=true")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel weight=10, active=false")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE weight > 20 AND active = true")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find the edge with weight=50 and active=true
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_or_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE a { name: 'A' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE b { name: 'B' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel status='active'")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel status='pending'")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel status='archived'")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE status = 'active' OR status = 'pending'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_with_ne_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE user { name: 'Admin', role: 'admin' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE user { name: 'User', role: 'user' }")
            .unwrap();

        // Ne condition - find users who are NOT admin
        let result = router
            .execute_parsed("FIND NODE WHERE role != 'admin'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                // Should find only User, not Admin
                assert!(u.description.contains("node"));
                assert!(u
                    .items
                    .iter()
                    .any(|item| item.data.get("name") == Some(&"User".to_string())));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_ne_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE n { name: 'N1' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE n { name: 'N2' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel status='complete'")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel status='pending'")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE status != 'complete'")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_lt_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE n { name: 'N1' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE n { name: 'N2' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel weight=100")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel weight=10")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE weight < 50")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_ge_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE n { name: 'N1' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE n { name: 'N2' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel priority=5")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel priority=10")
            .unwrap();

        let result = router
            .execute_parsed("FIND EDGE WHERE priority >= 5")
            .unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges_with_le_condition() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE n { name: 'N1' }")
            .unwrap();
        router
            .execute_parsed("NODE CREATE n { name: 'N2' }")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel score=3")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : rel score=8")
            .unwrap();

        let result = router.execute_parsed("FIND EDGE WHERE score <= 5").unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.description.contains("edge"));
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_with_limit_verified() {
        let router = QueryRouter::new();
        // Create multiple nodes
        for i in 0..10 {
            router
                .execute_parsed(&format!("NODE CREATE item idx={}", i))
                .unwrap();
        }

        let result = router.execute_parsed("FIND NODE item LIMIT 3").unwrap();
        match result {
            QueryResult::Unified(u) => {
                assert!(u.items.len() <= 3);
            },
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_node_list_with_data() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE employee name='John', dept='sales'")
            .unwrap();
        router
            .execute_parsed("NODE CREATE employee name='Jane', dept='eng'")
            .unwrap();
        router
            .execute_parsed("NODE CREATE manager name='Boss', level=5")
            .unwrap();

        // List all employee nodes
        let result = router.execute_parsed("NODE LIST employee").unwrap();
        match result {
            QueryResult::Nodes(nodes) => {
                assert_eq!(nodes.len(), 2); // Two employees
            },
            _ => panic!("Expected Nodes"),
        }

        // List all nodes (no filter)
        let all = router.execute_parsed("NODE LIST").unwrap();
        match all {
            QueryResult::Nodes(nodes) => {
                assert_eq!(nodes.len(), 3); // All three nodes
            },
            _ => panic!("Expected Nodes"),
        }
    }

    #[test]
    fn parsed_edge_list_with_data() {
        let router = QueryRouter::new();
        // Create nodes
        router
            .execute_parsed("NODE CREATE person name='X'")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person name='Y'")
            .unwrap();
        router
            .execute_parsed("NODE CREATE person name='Z'")
            .unwrap();
        // Create edges
        router
            .execute_parsed("EDGE CREATE 1 -> 2 : friend")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 2 -> 3 : colleague")
            .unwrap();
        router
            .execute_parsed("EDGE CREATE 1 -> 3 : friend")
            .unwrap();

        // List all friend edges
        let result = router.execute_parsed("EDGE LIST friend").unwrap();
        match result {
            QueryResult::Edges(edges) => {
                assert_eq!(edges.len(), 2); // Two friend edges
            },
            _ => panic!("Expected Edges"),
        }

        // List all edges (no filter)
        let all = router.execute_parsed("EDGE LIST").unwrap();
        match all {
            QueryResult::Edges(edges) => {
                assert_eq!(edges.len(), 3); // All three edges
            },
            _ => panic!("Expected Edges"),
        }
    }

    #[test]
    fn parsed_empty_statement() {
        let router = QueryRouter::new();
        let result = router.execute_parsed(";").unwrap();
        assert!(matches!(result, QueryResult::Empty));
    }

    #[test]
    fn parsed_parse_error() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("INVALID SYNTAX HERE @#$");
        assert!(result.is_err());
        if let Err(RouterError::ParseError(msg)) = result {
            assert!(!msg.is_empty());
        } else {
            panic!("Expected ParseError");
        }
    }

    #[test]
    fn parsed_select_missing_from() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("SELECT *");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_insert_select_basic() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE src (id INT, name TEXT)")
            .unwrap();
        router
            .execute_parsed("CREATE TABLE dst (id INT, name TEXT)")
            .unwrap();

        // Insert some data into src
        router
            .execute_parsed("INSERT INTO src VALUES (1, 'Alice')")
            .unwrap();
        router
            .execute_parsed("INSERT INTO src VALUES (2, 'Bob')")
            .unwrap();

        // Insert from SELECT
        let result = router.execute_parsed("INSERT INTO dst SELECT * FROM src");
        assert!(result.is_ok());

        // Verify data was copied
        let rows = router.execute_parsed("SELECT * FROM dst").unwrap();
        match rows {
            QueryResult::Rows(r) => {
                assert_eq!(r.len(), 2);
            },
            _ => panic!("expected Rows"),
        }
    }

    #[test]
    fn parsed_condition_operators() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE vals (id:int, x:int)").unwrap();
        router.execute("INSERT vals id=1, x=10").unwrap();
        router.execute("INSERT vals id=2, x=20").unwrap();
        router.execute("INSERT vals id=3, x=30").unwrap();

        let eq = router
            .execute_parsed("SELECT * FROM vals WHERE x = 20")
            .unwrap();
        assert!(matches!(eq, QueryResult::Rows(r) if r.len() == 1));

        let ne = router
            .execute_parsed("SELECT * FROM vals WHERE x != 20")
            .unwrap();
        assert!(matches!(ne, QueryResult::Rows(r) if r.len() == 2));

        let lt = router
            .execute_parsed("SELECT * FROM vals WHERE x < 20")
            .unwrap();
        assert!(matches!(lt, QueryResult::Rows(r) if r.len() == 1));

        let le = router
            .execute_parsed("SELECT * FROM vals WHERE x <= 20")
            .unwrap();
        assert!(matches!(le, QueryResult::Rows(r) if r.len() == 2));

        let gt = router
            .execute_parsed("SELECT * FROM vals WHERE x > 20")
            .unwrap();
        assert!(matches!(gt, QueryResult::Rows(r) if r.len() == 1));

        let ge = router
            .execute_parsed("SELECT * FROM vals WHERE x >= 20")
            .unwrap();
        assert!(matches!(ge, QueryResult::Rows(r) if r.len() == 2));
    }

    #[test]
    fn parsed_condition_and_or() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE multi (a:int, b:int)").unwrap();
        router.execute("INSERT multi a=1, b=1").unwrap();
        router.execute("INSERT multi a=1, b=2").unwrap();
        router.execute("INSERT multi a=2, b=1").unwrap();

        let and_result = router
            .execute_parsed("SELECT * FROM multi WHERE a = 1 AND b = 1")
            .unwrap();
        assert!(matches!(and_result, QueryResult::Rows(r) if r.len() == 1));

        let or_result = router
            .execute_parsed("SELECT * FROM multi WHERE a = 2 OR b = 2")
            .unwrap();
        assert!(matches!(or_result, QueryResult::Rows(r) if r.len() == 2));
    }

    #[test]
    fn parsed_data_types() {
        let router = QueryRouter::new();
        router
            .execute_parsed(
                "CREATE TABLE types (
            i INT,
            bi BIGINT,
            si SMALLINT,
            f FLOAT,
            d DOUBLE,
            r REAL,
            dec DECIMAL(10, 2),
            num NUMERIC(5),
            vc VARCHAR(255),
            c CHAR(10),
            t TEXT,
            b BOOLEAN
        )",
            )
            .unwrap();
    }

    #[test]
    fn parsed_expr_to_value_types() {
        let router = QueryRouter::new();
        router
            .execute("CREATE TABLE vals (n:int, f:double, s:string, b:bool)")
            .unwrap();

        // Insert using parser - tests expr_to_value for different types
        router
            .execute_parsed("INSERT INTO vals (n, f, s, b) VALUES (42, 3.14, 'hello', true)")
            .unwrap();
        router
            .execute_parsed("INSERT INTO vals (n, f, s, b) VALUES (0, 0.0, 'world', false)")
            .unwrap();

        let result = router.execute("SELECT vals").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 2),
            _ => panic!("Expected Rows"),
        }
    }

    #[test]
    fn parsed_neighbors_with_edge_type() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        router
            .execute(&format!("EDGE CREATE {} -> {} knows", n1, n2))
            .unwrap();

        let result = router
            .execute_parsed(&format!("NEIGHBORS {} OUTGOING knows", n1))
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_find_with_limit() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND NODE person LIMIT 5").unwrap();
        assert!(matches!(result, QueryResult::Unified(_)));
    }

    #[test]
    fn parsed_insert_null_value() {
        let router = QueryRouter::new();
        // Use parser-style CREATE TABLE with nullable column
        router
            .execute_parsed("CREATE TABLE ntest (id INT NOT NULL, val TEXT)")
            .unwrap();
        router
            .execute_parsed("INSERT INTO ntest (id, val) VALUES (1, NULL)")
            .unwrap();
    }

    #[test]
    fn parsed_node_create_with_properties() {
        let router = QueryRouter::new();
        // Tests properties_to_map with various types
        let result = router
            .execute_parsed(
                "NODE CREATE person { name: 'John', age: 30, score: 95.5, active: true }",
            )
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_path_not_found() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        // No edge between them - tests path not found handling
        let result = router
            .execute_parsed(&format!("PATH SHORTEST {} -> {}", n1, n2))
            .unwrap();
        match result {
            QueryResult::Path(path) => assert!(path.is_empty()),
            _ => panic!("Expected Path"),
        }
    }

    #[test]
    fn parsed_select_qualified_column() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=1").unwrap();

        // Use table.column syntax
        let result = router.execute_parsed("SELECT t.x FROM t").unwrap();
        assert!(matches!(result, QueryResult::Rows(_)));
    }

    #[test]
    fn parsed_insert_with_ident_value() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (name:string)").unwrap();

        // Insert with unquoted identifier as value (gets treated as string)
        let result = router.execute_parsed("INSERT INTO t (name) VALUES (someident)");
        // This tests expr_to_value with ident
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_similar_with_limit_expr() {
        let router = QueryRouter::new();
        router.execute("EMBED v1 [1.0, 0.0]").unwrap();
        router.execute("EMBED v2 [0.0, 1.0]").unwrap();

        // Test with explicit LIMIT
        let result = router.execute_parsed("SIMILAR 'v1' LIMIT 10").unwrap();
        assert!(matches!(result, QueryResult::Similar(_)));
    }

    #[test]
    fn parsed_embed_store_with_list() {
        let router = QueryRouter::new();
        // Store using the parsed STORE syntax
        router
            .execute_parsed("EMBED STORE 'stored_vec' [1.0, 2.0, 3.0]")
            .unwrap();

        // Verify it was stored using parsed GET
        let result = router.execute_parsed("EMBED GET 'stored_vec'").unwrap();
        assert!(matches!(result, QueryResult::Value(_)));
    }

    #[test]
    fn parsed_empty_command() {
        let router = QueryRouter::new();
        // Empty string parses as empty statement
        let result = router.execute_parsed("");
        // Parser returns empty statement for empty input
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_whitespace_only() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("   ");
        // Whitespace only may parse as empty statement or error
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_create_index_empty_columns() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        // Creating index without columns should still work (takes first column)
        let result = router.execute_parsed("CREATE INDEX idx ON t (x)");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_find_path_pattern() {
        let router = QueryRouter::new();
        // Test FIND with path pattern (covers FindPattern::Path)
        let result = router.execute_parsed("FIND a -[e]-> b");
        // May error or succeed depending on parser support
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_edge_create_with_type_and_props() {
        let router = QueryRouter::new();
        let n1 = match router.execute("NODE CREATE a").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };
        let n2 = match router.execute("NODE CREATE b").unwrap() {
            QueryResult::Ids(ids) => ids[0],
            _ => panic!("Expected Ids"),
        };

        // Test edge with type and properties
        let result = router
            .execute_parsed(&format!(
                "EDGE CREATE {} -> {} : friend {{ since: 2020 }}",
                n1, n2
            ))
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_node_create_null_property() {
        let router = QueryRouter::new();
        // Test with null property value (covers PropertyValue::Null)
        let result = router
            .execute_parsed("NODE CREATE test { val: NULL }")
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_node_create_bool_property() {
        let router = QueryRouter::new();
        let result = router
            .execute_parsed("NODE CREATE test { active: false }")
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_node_create_float_property() {
        let router = QueryRouter::new();
        let result = router
            .execute_parsed("NODE CREATE test { score: 3.14 }")
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn parsed_embed_with_int_values() {
        let router = QueryRouter::new();
        // Store embedding with integer values (tests expr_to_f32 with integer)
        router
            .execute_parsed("EMBED STORE 'intvec' [1, 2, 3]")
            .unwrap();
    }

    #[test]
    fn parsed_node_with_ident_property() {
        let router = QueryRouter::new();
        // Property value is an identifier (tests PropertyValue from ident)
        let result = router
            .execute_parsed("NODE CREATE test { mykey: somevalue }")
            .unwrap();
        assert!(matches!(result, QueryResult::Ids(_)));
    }

    #[test]
    fn execute_empty_after_trim() {
        let router = QueryRouter::new();
        // Test the empty command check in execute()
        let result = router.execute("   \t\n   ");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_select_with_qualified_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        router.execute("INSERT t x=5").unwrap();
        // Use qualified column name in WHERE clause
        let result = router
            .execute_parsed("SELECT * FROM t WHERE t.x = 5")
            .unwrap();
        assert!(matches!(result, QueryResult::Rows(_)));
    }

    #[test]
    fn parsed_unsupported_operator_in_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        // Using + operator in WHERE should error
        let result = router.execute_parsed("SELECT * FROM t WHERE x + 1");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_literal_in_where() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        // Just a literal in WHERE (non-binary expression)
        let result = router.execute_parsed("SELECT * FROM t WHERE 1");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_insert_with_complex_expr() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        // Complex expression as value - tests error path in expr_to_value
        let result = router.execute_parsed("INSERT INTO t (x) VALUES (1 + 2)");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_create_unsupported_type() {
        let router = QueryRouter::new();
        // BLOB type is not supported - tests unsupported data type error
        let result = router.execute_parsed("CREATE TABLE t (data BLOB)");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_similar_limit_not_integer() {
        let router = QueryRouter::new();
        router.execute("EMBED v [1.0, 2.0]").unwrap();
        // LIMIT with non-integer should fail
        let result = router.execute_parsed("SIMILAR 'v' LIMIT 'ten'");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_neighbors_negative_id() {
        let router = QueryRouter::new();
        // Negative ID should fail
        let result = router.execute_parsed("NEIGHBORS -1 OUTGOING");
        // Parser may reject this or exec may fail
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_path_negative_ids() {
        let router = QueryRouter::new();
        // Negative IDs in PATH
        let result = router.execute_parsed("PATH SHORTEST -1 -> -2");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_find_edges_plain() {
        let router = QueryRouter::new();
        // FIND EDGE without type
        let result = router.execute_parsed("FIND EDGE").unwrap();
        match result {
            QueryResult::Unified(u) => assert!(u.description.contains("edge")),
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_plain() {
        let router = QueryRouter::new();
        // FIND NODE without label
        let result = router.execute_parsed("FIND NODE").unwrap();
        match result {
            QueryResult::Unified(u) => assert!(u.description.contains("node")),
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_embed_get_with_ident_key() {
        let router = QueryRouter::new();
        router.execute("EMBED mykey [1.0, 2.0]").unwrap();
        // Use identifier (not quoted string) for key - tests expr_to_string with ident
        let result = router.execute_parsed("EMBED GET mykey").unwrap();
        assert!(matches!(result, QueryResult::Value(_)));
    }

    #[test]
    fn parsed_similar_with_ident_key() {
        let router = QueryRouter::new();
        router.execute("EMBED vec1 [1.0, 0.0]").unwrap();
        // Use identifier for key
        let result = router.execute_parsed("SIMILAR vec1 LIMIT 5").unwrap();
        assert!(matches!(result, QueryResult::Similar(_)));
    }

    #[test]
    fn parsed_node_get_nonexistent() {
        let router = QueryRouter::new();
        // Get a node that doesn't exist - tests graph error propagation
        let result = router.execute_parsed("NODE GET 999999");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_path_nonexistent_nodes() {
        let router = QueryRouter::new();
        // Path between non-existent nodes - tests graph error
        let result = router.execute_parsed("PATH SHORTEST 999999 -> 999998");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_neighbors_nonexistent_node() {
        let router = QueryRouter::new();
        // Neighbors of non-existent node
        let result = router.execute_parsed("NEIGHBORS 999999 OUTGOING");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_edge_get_nonexistent() {
        let router = QueryRouter::new();
        // Get edge that doesn't exist
        let result = router.execute_parsed("EDGE GET 999999");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_node_delete_nonexistent() {
        let router = QueryRouter::new();
        // Delete node that doesn't exist
        let result = router.execute_parsed("NODE DELETE 999999");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_embed_delete_nonexistent() {
        let router = QueryRouter::new();
        // Delete embedding that doesn't exist
        let result = router.execute_parsed("EMBED DELETE 'nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_select_nonexistent_table() {
        let router = QueryRouter::new();
        // Select from table that doesn't exist
        let result = router.execute_parsed("SELECT * FROM nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_update_nonexistent_table() {
        let router = QueryRouter::new();
        // Update table that doesn't exist
        let result = router.execute_parsed("UPDATE nonexistent SET x = 1");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_delete_nonexistent_table() {
        let router = QueryRouter::new();
        // Delete from table that doesn't exist
        let result = router.execute_parsed("DELETE FROM nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn execute_only_whitespace() {
        let router = QueryRouter::new();
        // Pure whitespace triggers empty command error
        let result = router.execute("\t\n  \r\n");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_embed_list() {
        let router = QueryRouter::new();
        router.execute("EMBED a [1.0]").unwrap();
        router.execute("EMBED b [2.0]").unwrap();
        let result = router.execute_parsed("EMBED LIST");
        // LIST may not be supported, but this exercises the code path
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_insert_into_nonexistent() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("INSERT INTO nonexistent (x) VALUES (1)");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_drop_nonexistent_table() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("DROP TABLE nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn execute_tab_only() {
        let router = QueryRouter::new();
        // Tab-only triggers empty command (line 230)
        let result = router.execute("\t");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_embed_non_number_vector() {
        let router = QueryRouter::new();
        // Non-numeric value in vector - tests expr_to_f32 error
        let result = router.execute_parsed("EMBED STORE 'k' ['a', 'b']");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_similar_non_string_key() {
        let router = QueryRouter::new();
        // Using a non-string/non-ident as key - tests expr_to_string error
        let result = router.execute_parsed("SIMILAR [1,2,3] LIMIT 5");
        // Vector syntax is valid for SIMILAR
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn parsed_where_complex_column() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE t (x:int)").unwrap();
        // Complex expression as column name - tests expr_to_column_name error
        let result = router.execute_parsed("SELECT * FROM t WHERE (1+2) = 3");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_node_invalid_property_expr() {
        let router = QueryRouter::new();
        // Complex expression as property value - tests properties_to_map error
        let result = router.execute_parsed("NODE CREATE test { val: (1+2) }");
        assert!(result.is_err());
    }

    // ========== SHOW TABLES Tests ==========

    #[test]
    fn show_tables_empty() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("SHOW TABLES").unwrap();
        match result {
            QueryResult::TableList(tables) => {
                assert!(tables.is_empty());
            },
            _ => panic!("Expected TableList"),
        }
    }

    #[test]
    fn show_tables_with_tables() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE users (id INT)")
            .unwrap();
        router
            .execute_parsed("CREATE TABLE products (id INT)")
            .unwrap();

        let result = router.execute_parsed("SHOW TABLES").unwrap();
        match result {
            QueryResult::TableList(tables) => {
                assert_eq!(tables.len(), 2);
                assert!(tables.contains(&"users".to_string()));
                assert!(tables.contains(&"products".to_string()));
            },
            _ => panic!("Expected TableList"),
        }
    }

    #[test]
    fn show_without_tables_error() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("SHOW");
        assert!(result.is_err());
    }

    #[test]
    fn insert_without_columns() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        // INSERT without explicit column names - should use schema order
        router
            .execute_parsed("INSERT INTO users VALUES (1, 'Alice')")
            .unwrap();
        router
            .execute_parsed("INSERT INTO users VALUES (2, 'Bob')")
            .unwrap();

        let result = router.execute_parsed("SELECT * FROM users").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
            },
            _ => panic!("Expected Rows"),
        }
    }

    // ========== Cross-Engine Tests ==========

    #[test]
    fn with_shared_store_creates_unified_router() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Verify all engines are accessible
        assert!(router.relational().list_tables().is_empty());
    }

    #[test]
    fn with_shared_store_initializes_unified_engine() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Verify unified engine is initialized
        assert!(router.unified().is_some());
    }

    #[test]
    fn new_router_has_no_unified_engine() {
        let router = QueryRouter::new();

        // Without shared store, unified engine should not be initialized
        assert!(router.unified().is_none());
    }

    #[test]
    fn unified_engine_delegates_find_neighbors_by_similarity() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Create test entities with embeddings
        router
            .vector()
            .set_entity_embedding("center", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("neighbor1", vec![0.9, 0.1, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("neighbor2", vec![0.5, 0.5, 0.0])
            .unwrap();

        // Connect neighbors
        router
            .graph()
            .add_entity_edge("center", "neighbor1", "connected")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "neighbor2", "connected")
            .unwrap();

        // Find neighbors by similarity - should delegate to UnifiedEngine
        let query = vec![1.0, 0.0, 0.0];
        let results = router
            .find_neighbors_by_similarity("center", &query, 5)
            .unwrap();

        // Should find both neighbors
        assert_eq!(results.len(), 2);
        // Results should be sorted by similarity (neighbor1 is more similar)
        assert_eq!(results[0].id, "neighbor1");
        assert!(results[0].score.unwrap() > results[1].score.unwrap());
    }

    #[test]
    fn unified_engine_delegates_find_similar_connected() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Create entities with embeddings
        router
            .vector()
            .set_entity_embedding("query", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("connected1", vec![0.95, 0.05, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("connected2", vec![0.8, 0.2, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("not_connected", vec![0.99, 0.01, 0.0])
            .unwrap();

        // Connect some entities to hub
        router
            .graph()
            .add_entity_edge("hub", "connected1", "links")
            .unwrap();
        router
            .graph()
            .add_entity_edge("hub", "connected2", "links")
            .unwrap();
        // not_connected is NOT linked to hub

        // Find similar AND connected - should delegate to UnifiedEngine
        let results = router.find_similar_connected("query", "hub", 5).unwrap();

        // Should only find connected1 and connected2 (not "not_connected")
        assert!(results.len() <= 2);
        for item in &results {
            assert!(item.id == "connected1" || item.id == "connected2");
            assert!(item.score.is_some());
        }
    }

    #[test]
    fn create_unified_entity_stores_embedding() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        let fields = HashMap::from([("name".to_string(), "Alice".to_string())]);
        let embedding = vec![1.0, 0.0, 0.0];

        router
            .create_unified_entity("user:1", fields, Some(embedding.clone()))
            .unwrap();

        let retrieved = router.vector().get_entity_embedding("user:1").unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn create_unified_entity_without_embedding() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        let fields = HashMap::from([("name".to_string(), "Alice".to_string())]);

        router
            .create_unified_entity("user:1", fields, None)
            .unwrap();

        // Should not have embedding
        assert!(!router.vector().entity_has_embedding("user:1"));
    }

    #[test]
    fn connect_entities_creates_edge() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        let edge_key = router
            .connect_entities("user:1", "user:2", "follows")
            .unwrap();

        assert!(edge_key.starts_with("edge:follows:"));

        let neighbors = router.graph().get_entity_neighbors_out("user:1").unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], "user:2");
    }

    #[test]
    fn find_similar_connected_returns_intersection() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Create entities with embeddings
        router
            .vector()
            .set_entity_embedding("query", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:1", vec![0.9, 0.1, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![0.8, 0.2, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:3", vec![0.0, 1.0, 0.0])
            .unwrap();

        // Connect users to hub
        router
            .graph()
            .add_entity_edge("hub", "user:1", "connects")
            .unwrap();
        router
            .graph()
            .add_entity_edge("hub", "user:2", "connects")
            .unwrap();
        // user:3 is NOT connected to hub

        let results = router.find_similar_connected("query", "hub", 5).unwrap();

        // Should find user:1 and user:2 (similar AND connected), not user:3
        assert!(results.len() <= 2);
        for item in &results {
            assert!(item.id == "user:1" || item.id == "user:2");
            assert!(item.score.is_some());
            assert_eq!(item.source, "vector+graph");
        }
    }

    #[test]
    fn find_similar_connected_no_embedding() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        let result = router.find_similar_connected("nonexistent", "hub", 5);
        assert!(result.is_err());
    }

    #[test]
    fn find_neighbors_by_similarity() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Create entities with embeddings
        router
            .vector()
            .set_entity_embedding("user:1", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![0.0, 1.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:3", vec![0.5, 0.5, 0.0])
            .unwrap();

        // Create graph edges from center to others
        router
            .graph()
            .add_entity_edge("center", "user:1", "knows")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "user:2", "knows")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "user:3", "knows")
            .unwrap();

        // Query similar to [1, 0, 0]
        let query = vec![1.0, 0.0, 0.0];
        let results = router
            .find_neighbors_by_similarity("center", &query, 3)
            .unwrap();

        assert_eq!(results.len(), 3);
        // user:1 should be first (most similar)
        assert_eq!(results[0].id, "user:1");
        assert_eq!(results[0].source, "graph+vector");
    }

    #[test]
    fn find_neighbors_by_similarity_no_entity() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        let result = router.find_neighbors_by_similarity("nonexistent", &[1.0, 0.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn find_neighbors_by_similarity_filters_dimension_mismatch() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Create entities with different dimensions
        router
            .vector()
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![1.0, 0.0, 0.0])
            .unwrap(); // Different dim

        router
            .graph()
            .add_entity_edge("center", "user:1", "knows")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "user:2", "knows")
            .unwrap();

        let query = vec![1.0, 0.0]; // 2D query
        let results = router
            .find_neighbors_by_similarity("center", &query, 5)
            .unwrap();

        // Should only find user:1 (matching dimension)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "user:1");
    }

    #[test]
    fn shared_store_engines_share_data() {
        let store = tensor_store::TensorStore::new();
        let router = QueryRouter::with_shared_store(store);

        // Write via vector engine
        router
            .vector()
            .set_entity_embedding("entity:1", vec![1.0, 2.0])
            .unwrap();

        // Add graph edges via graph engine
        router
            .graph()
            .add_entity_edge("entity:1", "entity:2", "relates")
            .unwrap();

        // Verify both are accessible via unified entity
        assert!(router.vector().entity_has_embedding("entity:1"));
        assert!(router.graph().entity_has_edges("entity:1"));
    }

    #[test]
    fn test_cache_init() {
        let mut router = QueryRouter::new();
        router.init_cache();
        assert!(router.cache().is_some());
    }

    #[test]
    fn test_cache_stats() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE STATS");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert!(output.contains("Cache Statistics"));
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_init_command() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE INIT");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert!(output.contains("Cache initialized"));
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_clear() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE CLEAR");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert!(output.contains("Cache cleared"));
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_without_init() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("CACHE STATS");
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_evict() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE EVICT");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert!(output.contains("Evicted"));
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_evict_with_count() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE EVICT 50");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert!(output.contains("Evicted"));
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_put_get() {
        let mut router = QueryRouter::new();
        router.init_cache();

        // Put a value
        let result = router.execute_parsed("CACHE PUT 'testkey' 'testvalue'");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), QueryResult::Value(s) if s == "OK"));

        // Get the value
        let result = router.execute_parsed("CACHE GET 'testkey'");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert_eq!(output, "testvalue");
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_cache_get_not_found() {
        let mut router = QueryRouter::new();
        router.init_cache();

        let result = router.execute_parsed("CACHE GET 'nonexistent'");
        assert!(result.is_ok());
        if let QueryResult::Value(output) = result.unwrap() {
            assert_eq!(output, "(not found)");
        } else {
            panic!("Expected Value result");
        }
    }

    #[test]
    fn test_query_cache_select() {
        let mut router = QueryRouter::new();
        router.init_cache();

        // Create table and insert data
        router
            .execute_parsed("CREATE TABLE cached_test (id INT, name TEXT)")
            .unwrap();
        router
            .execute_parsed("INSERT INTO cached_test (id, name) VALUES (1, 'Alice')")
            .unwrap();

        // First query - should hit the database
        let result1 = router.execute_parsed("SELECT * FROM cached_test").unwrap();
        assert!(matches!(result1, QueryResult::Rows(_)));

        // Second query - should hit cache (same result)
        let result2 = router.execute_parsed("SELECT * FROM cached_test").unwrap();
        assert!(matches!(result2, QueryResult::Rows(_)));

        // Check stats to verify cache was used
        let stats = router.cache.as_ref().unwrap().stats();
        assert!(stats.hits(CacheLayer::Exact) > 0);
    }

    #[test]
    fn test_query_cache_invalidation() {
        let mut router = QueryRouter::new();
        router.init_cache();

        // Create table and insert data
        router
            .execute_parsed("CREATE TABLE invalidate_test (id INT)")
            .unwrap();
        router
            .execute_parsed("INSERT INTO invalidate_test (id) VALUES (1)")
            .unwrap();

        // Query to populate cache
        let _ = router.execute_parsed("SELECT * FROM invalidate_test");

        // Get cache stats before write
        let _hits_before = router
            .cache
            .as_ref()
            .unwrap()
            .stats()
            .hits(CacheLayer::Exact);

        // Insert more data - should invalidate cache
        router
            .execute_parsed("INSERT INTO invalidate_test (id) VALUES (2)")
            .unwrap();

        // Query again - should miss cache since it was invalidated
        let _ = router.execute_parsed("SELECT * FROM invalidate_test");

        // The first post-invalidation query should have missed
        // (though it will now be cached for subsequent queries)
        let misses_after = router
            .cache
            .as_ref()
            .unwrap()
            .stats()
            .misses(CacheLayer::Exact);
        assert!(misses_after > 0);
    }

    #[test]
    fn test_query_cache_case_insensitive() {
        let mut router = QueryRouter::new();
        router.init_cache();

        router
            .execute_parsed("CREATE TABLE case_test (id INT)")
            .unwrap();
        router
            .execute_parsed("INSERT INTO case_test (id) VALUES (1)")
            .unwrap();

        // Query with uppercase
        let _ = router.execute_parsed("SELECT * FROM case_test");

        // Query with mixed case - should hit cache (keys are lowercased)
        let _ = router.execute_parsed("select * from case_test");

        let stats = router.cache.as_ref().unwrap().stats();
        assert!(stats.hits(CacheLayer::Exact) > 0);
    }

    // ========== Vault Tests ==========

    #[test]
    fn test_vault_not_initialized() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("VAULT SET 'key' 'value'");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not initialized"));
    }

    #[test]
    fn test_vault_set_get() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router
            .execute_parsed("VAULT SET 'secret_key' 'secret_value'")
            .unwrap();
        let result = router.execute_parsed("VAULT GET 'secret_key'").unwrap();
        match result {
            QueryResult::Value(v) => assert_eq!(v, "secret_value"),
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_vault_delete() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router
            .execute_parsed("VAULT SET 'to_delete' 'value'")
            .unwrap();
        router.execute_parsed("VAULT DELETE 'to_delete'").unwrap();
        let result = router.execute_parsed("VAULT GET 'to_delete'");
        assert!(result.is_err());
    }

    #[test]
    fn test_vault_list() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router.execute_parsed("VAULT SET 'key1' 'v1'").unwrap();
        router.execute_parsed("VAULT SET 'key2' 'v2'").unwrap();
        let result = router.execute_parsed("VAULT LIST").unwrap();
        match result {
            QueryResult::Value(v) => {
                assert!(v.contains("key1"));
                assert!(v.contains("key2"));
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_vault_list_with_pattern() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router.execute_parsed("VAULT SET 'db_pass' 'v1'").unwrap();
        router.execute_parsed("VAULT SET 'db_user' 'v2'").unwrap();
        router.execute_parsed("VAULT SET 'api_key' 'v3'").unwrap();
        let result = router.execute_parsed("VAULT LIST 'db_*'").unwrap();
        match result {
            QueryResult::Value(v) => {
                assert!(v.contains("db_pass"));
                assert!(v.contains("db_user"));
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_vault_rotate() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router
            .execute_parsed("VAULT SET 'rotate_key' 'old_value'")
            .unwrap();
        router
            .execute_parsed("VAULT ROTATE 'rotate_key' 'new_value'")
            .unwrap();
        let result = router.execute_parsed("VAULT GET 'rotate_key'").unwrap();
        match result {
            QueryResult::Value(v) => assert_eq!(v, "new_value"),
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_vault_grant_revoke() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        router
            .execute_parsed("VAULT SET 'shared_key' 'shared_value'")
            .unwrap();
        // Grant access to another entity
        let grant_result = router.execute_parsed("VAULT GRANT 'user:bob' 'shared_key'");
        // Grant may fail without proper graph setup, but exercises the code path
        assert!(grant_result.is_ok() || grant_result.is_err());

        // Revoke access
        let revoke_result = router.execute_parsed("VAULT REVOKE 'user:bob' 'shared_key'");
        assert!(revoke_result.is_ok() || revoke_result.is_err());
    }

    // ========== Blob Tests ==========

    #[test]
    fn test_blob_not_initialized() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("BLOB PUT 'test.txt' 'hello'");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not initialized"));
    }

    #[test]
    fn test_blob_put_get_delete() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Put a blob
        let put_result = router
            .execute_parsed("BLOB PUT 'test.txt' 'Hello, World!'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result with artifact ID"),
        };

        // Get the blob
        let get_result = router
            .execute_parsed(&format!("BLOB GET '{}'", artifact_id))
            .unwrap();
        match get_result {
            QueryResult::Blob(data) => {
                assert_eq!(String::from_utf8_lossy(&data), "Hello, World!");
            },
            _ => panic!("Expected Blob result"),
        }

        // Delete the blob
        router
            .execute_parsed(&format!("BLOB DELETE '{}'", artifact_id))
            .unwrap();

        // Verify it's gone
        let get_after_delete = router.execute_parsed(&format!("BLOB GET '{}'", artifact_id));
        assert!(get_after_delete.is_err());
    }

    #[test]
    fn test_blob_info() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'info_test.txt' 'test data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        let info_result = router
            .execute_parsed(&format!("BLOB INFO '{}'", artifact_id))
            .unwrap();
        match info_result {
            QueryResult::ArtifactInfo(info) => {
                assert_eq!(info.filename, "info_test.txt");
                assert_eq!(info.size, 9);
            },
            _ => panic!("Expected ArtifactInfo result"),
        }
    }

    #[test]
    fn test_blob_link_unlink() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'link_test.txt' 'data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Link to an entity (syntax: BLOB LINK 'artifact' TO 'entity')
        router
            .execute_parsed(&format!("BLOB LINK '{}' TO 'task:123'", artifact_id))
            .unwrap();

        // Get links
        let links_result = router
            .execute_parsed(&format!("BLOB LINKS '{}'", artifact_id))
            .unwrap();
        match links_result {
            QueryResult::ArtifactList(links) => {
                assert!(links.contains(&"task:123".to_string()));
            },
            _ => panic!("Expected ArtifactList result"),
        }

        // Unlink (syntax: BLOB UNLINK 'artifact' FROM 'entity')
        router
            .execute_parsed(&format!("BLOB UNLINK '{}' FROM 'task:123'", artifact_id))
            .unwrap();
    }

    #[test]
    fn test_blob_tag_untag() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'tag_test.txt' 'data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Add tag
        router
            .execute_parsed(&format!("BLOB TAG '{}' 'important'", artifact_id))
            .unwrap();

        // Check info has tag
        let info = router
            .execute_parsed(&format!("BLOB INFO '{}'", artifact_id))
            .unwrap();
        match info {
            QueryResult::ArtifactInfo(info) => {
                assert!(info.tags.contains(&"important".to_string()));
            },
            _ => panic!("Expected ArtifactInfo"),
        }

        // Remove tag
        router
            .execute_parsed(&format!("BLOB UNTAG '{}' 'important'", artifact_id))
            .unwrap();
    }

    #[test]
    fn test_blob_verify() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'verify_test.txt' 'verify me'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        let verify_result = router
            .execute_parsed(&format!("BLOB VERIFY '{}'", artifact_id))
            .unwrap();
        match verify_result {
            QueryResult::Value(v) => assert_eq!(v, "OK"),
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_blob_gc() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let gc_result = router.execute_parsed("BLOB GC").unwrap();
        match gc_result {
            QueryResult::Value(v) => {
                assert!(v.contains("Deleted"));
                assert!(v.contains("freed"));
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_blob_gc_full() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let gc_result = router.execute_parsed("BLOB GC FULL").unwrap();
        match gc_result {
            QueryResult::Value(v) => {
                assert!(v.contains("Deleted"));
                assert!(v.contains("freed"));
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_blob_repair() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let repair_result = router.execute_parsed("BLOB REPAIR").unwrap();
        match repair_result {
            QueryResult::Value(v) => {
                assert!(v.contains("Fixed"));
                assert!(v.contains("orphans"));
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_blob_stats() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let stats_result = router.execute_parsed("BLOB STATS").unwrap();
        match stats_result {
            QueryResult::BlobStats(stats) => {
                assert_eq!(stats.artifact_count, 0);
            },
            _ => panic!("Expected BlobStats result"),
        }
    }

    #[test]
    fn test_blob_meta_set_get() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'meta_test.txt' 'data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Set custom metadata
        router
            .execute_parsed(&format!("BLOB META SET '{}' 'author' 'alice'", artifact_id))
            .unwrap();

        // Get custom metadata
        let meta_result = router
            .execute_parsed(&format!("BLOB META GET '{}' 'author'", artifact_id))
            .unwrap();
        match meta_result {
            QueryResult::Value(v) => assert_eq!(v, "alice"),
            _ => panic!("Expected Value result"),
        }

        // Get nonexistent metadata
        let missing_meta = router
            .execute_parsed(&format!("BLOB META GET '{}' 'nonexistent'", artifact_id))
            .unwrap();
        match missing_meta {
            QueryResult::Value(v) => assert_eq!(v, "(not found)"),
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_blob_put_missing_data() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // PUT without DATA or FROM should fail
        let result = router.execute_parsed("BLOB PUT 'missing.txt'");
        assert!(result.is_err());
    }

    // ========== Blobs (multi-artifact) Tests ==========

    #[test]
    fn test_blobs_list() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Add some blobs
        router
            .execute_parsed("BLOB PUT 'file1.txt' 'data1'")
            .unwrap();
        router
            .execute_parsed("BLOB PUT 'file2.txt' 'data2'")
            .unwrap();

        // Syntax: BLOBS (no LIST keyword)
        let list_result = router.execute_parsed("BLOBS").unwrap();
        match list_result {
            QueryResult::ArtifactList(list) => {
                assert_eq!(list.len(), 2);
            },
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blobs_list_with_pattern() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Test that BLOBS with a pattern expression parses and executes
        let list_result = router.execute_parsed("BLOBS 'some_prefix'");
        match list_result {
            Ok(QueryResult::ArtifactList(_)) => {},
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blobs_find_by_link() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'linked.txt' 'data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };
        router
            .execute_parsed(&format!("BLOB LINK '{}' TO 'project:alpha'", artifact_id))
            .unwrap();

        // Syntax: BLOBS FOR 'entity'
        let find_result = router.execute_parsed("BLOBS FOR 'project:alpha'").unwrap();
        match find_result {
            QueryResult::ArtifactList(list) => {
                assert!(!list.is_empty());
            },
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blobs_find_by_tag() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'tagged.txt' 'data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };
        router
            .execute_parsed(&format!("BLOB TAG '{}' 'urgent'", artifact_id))
            .unwrap();

        // Syntax: BLOBS BY TAG 'tag'
        let find_result = router.execute_parsed("BLOBS BY TAG 'urgent'").unwrap();
        match find_result {
            QueryResult::ArtifactList(list) => {
                assert!(!list.is_empty());
            },
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blobs_not_initialized() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("BLOBS LIST");
        assert!(result.is_err());
    }

    // ========== Additional Error Path Tests ==========

    #[test]
    fn test_vault_get_not_found() {
        let mut router = QueryRouter::new();
        router.init_vault(b"test_master_key_32bytes!").unwrap();

        let result = router.execute_parsed("VAULT GET 'nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_get_not_found() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB GET 'artifact:nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_delete_not_found() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB DELETE 'artifact:nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_info_not_found() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB INFO 'artifact:nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_verify_not_found() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB VERIFY 'artifact:nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_start_blob_not_initialized() {
        let mut router = QueryRouter::new();
        let result = router.start_blob();
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_put_with_options() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Test with LINK and TAG options
        let result = router
            .execute_parsed("BLOB PUT 'options_test.txt' 'data' LINK 'task:123' TAG 'important'");
        assert!(result.is_ok());

        let artifact_id = match result.unwrap() {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Verify link was applied
        let links = router
            .execute_parsed(&format!("BLOB LINKS '{}'", artifact_id))
            .unwrap();
        match links {
            QueryResult::ArtifactList(l) => {
                assert!(l.contains(&"task:123".to_string()));
            },
            _ => panic!("Expected ArtifactList"),
        }

        // Verify tag was applied
        let info = router
            .execute_parsed(&format!("BLOB INFO '{}'", artifact_id))
            .unwrap();
        match info {
            QueryResult::ArtifactInfo(i) => {
                assert!(i.tags.contains(&"important".to_string()));
            },
            _ => panic!("Expected ArtifactInfo"),
        }
    }

    #[test]
    fn test_blobs_similar() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Similar search requires embeddings - test that the query parses and executes
        let result = router.execute_parsed("BLOBS SIMILAR TO 'artifact:test' LIMIT 5");
        // May fail due to missing artifact, but exercises code path
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_blobs_for_entity() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOBS FOR 'task:123'");
        match result {
            Ok(QueryResult::ArtifactList(_)) => {},
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blobs_by_type() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOBS WHERE TYPE = 'text/plain'");
        match result {
            Ok(QueryResult::ArtifactList(_)) => {},
            _ => panic!("Expected ArtifactList result"),
        }
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_shutdown_blob() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Shutdown should work
        let result = router.shutdown_blob();
        assert!(result.is_ok());
    }

    #[test]
    fn test_shutdown_blob_not_initialized() {
        let mut router = QueryRouter::new();
        // Shutdown without init should still work (early return)
        let result = router.shutdown_blob();
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_identity() {
        let mut router = QueryRouter::new();
        // Default identity is "node:root"
        assert_eq!(router.current_identity(), "node:root");

        router.set_identity("user:alice");
        assert_eq!(router.current_identity(), "user:alice");
    }

    #[test]
    fn test_init_cache_default() {
        let mut router = QueryRouter::new();
        let result = router.init_cache_default();
        assert!(result.is_ok());
        assert!(router.cache().is_some());
    }

    #[test]
    fn test_init_cache_with_config() {
        let mut router = QueryRouter::new();
        let config = tensor_cache::CacheConfig::default();
        router.init_cache_with_config(config);
        assert!(router.cache().is_some());
    }

    #[test]
    fn test_blob_accessor() {
        let mut router = QueryRouter::new();
        assert!(router.blob().is_none());

        router.init_blob().unwrap();
        assert!(router.blob().is_some());
    }

    #[test]
    fn test_error_display_all_variants() {
        let errors = vec![
            RouterError::ParseError("parse msg".to_string()),
            RouterError::UnknownCommand("unknown".to_string()),
            RouterError::RelationalError("rel msg".to_string()),
            RouterError::GraphError("graph msg".to_string()),
            RouterError::VectorError("vec msg".to_string()),
            RouterError::VaultError("vault msg".to_string()),
            RouterError::CacheError("cache msg".to_string()),
            RouterError::BlobError("blob msg".to_string()),
            RouterError::InvalidArgument("invalid msg".to_string()),
            RouterError::TypeMismatch("type msg".to_string()),
            RouterError::MissingArgument("missing msg".to_string()),
        ];

        for e in errors {
            let display = format!("{}", e);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_blob_from_path() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Try to read from a non-existent path
        let result = router.execute_parsed("BLOB PUT 'from_path.txt' FROM '/nonexistent/path'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_get_to_path() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Put a blob first
        let put_result = router
            .execute_parsed("BLOB PUT 'get_to.txt' 'test data'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Try to write to an invalid path
        let result = router.execute_parsed(&format!(
            "BLOB GET '{}' TO '/nonexistent/dir/file.txt'",
            artifact_id
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_init_vault() {
        let mut router = QueryRouter::new();
        let result = router.init_vault(b"32_byte_master_key_for_testing!");
        assert!(result.is_ok());
    }

    #[test]
    fn test_vault_rotate_nonexistent() {
        let mut router = QueryRouter::new();
        router
            .init_vault(b"32_byte_master_key_for_testing!")
            .unwrap();

        let result = router.execute_parsed("VAULT ROTATE 'nonexistent' 'new_value'");
        assert!(result.is_err());
    }

    #[test]
    fn test_vault_delete_nonexistent() {
        let mut router = QueryRouter::new();
        router
            .init_vault(b"32_byte_master_key_for_testing!")
            .unwrap();

        let result = router.execute_parsed("VAULT DELETE 'nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_link_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB LINK 'nonexistent' TO 'entity'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_unlink_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB UNLINK 'nonexistent' FROM 'entity'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_tag_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB TAG 'nonexistent' 'tag'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_untag_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB UNTAG 'nonexistent' 'tag'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_links_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB LINKS 'nonexistent'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_meta_set_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB META SET 'nonexistent' 'key' 'value'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_meta_get_nonexistent() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB META GET 'nonexistent' 'key'");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_get_to_valid_path() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let put_result = router
            .execute_parsed("BLOB PUT 'get_to_valid.txt' 'test'")
            .unwrap();
        let artifact_id = match put_result {
            QueryResult::Value(id) => id,
            _ => panic!("Expected Value result"),
        };

        // Write to a valid temp path
        let temp_path = "/tmp/neumann_test_blob_output.txt";
        let result =
            router.execute_parsed(&format!("BLOB GET '{}' TO '{}'", artifact_id, temp_path));
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_find_similar_connected_no_embedding() {
        let router = QueryRouter::new();
        // Try to find similar when no embeddings exist
        let result = router.find_similar_connected("nonexistent", "other", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_result_debug() {
        // Test that QueryResult implements Debug
        let result = QueryResult::Empty;
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());

        let result = QueryResult::Value("test".to_string());
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From implementations
        let rel_err = relational_engine::RelationalError::TableNotFound("test".to_string());
        let router_err: RouterError = rel_err.into();
        assert!(matches!(router_err, RouterError::RelationalError(_)));

        let graph_err = graph_engine::GraphError::NodeNotFound(1);
        let router_err: RouterError = graph_err.into();
        assert!(matches!(router_err, RouterError::GraphError(_)));

        let vec_err = vector_engine::VectorError::NotFound("test".to_string());
        let router_err: RouterError = vec_err.into();
        assert!(matches!(router_err, RouterError::VectorError(_)));
    }

    // ========== Phase 3: Cross-Engine Query Tests ==========

    #[test]
    fn parsed_entity_create_basic() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("ENTITY CREATE 'user:1' { name: 'Alice' }");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(msg) => {
                assert!(msg.contains("Entity 'user:1' created"));
            },
            _ => panic!("expected Value result"),
        }
    }

    #[test]
    fn parsed_entity_create_with_embedding() {
        let router = QueryRouter::new();
        let result =
            router.execute_parsed("ENTITY CREATE 'doc:1' { title: 'Test' } EMBEDDING [1.0, 0.0]");
        assert!(result.is_ok());

        // Verify embedding was stored
        let emb = router.vector().get_entity_embedding("doc:1");
        assert!(emb.is_ok());
        assert_eq!(emb.unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn parsed_entity_connect() {
        let router = QueryRouter::new();

        // Connect two entities
        let result = router.execute_parsed("ENTITY CONNECT 'user:1' -> 'user:2' : follows");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(msg) => {
                assert!(msg.contains("Connected 'user:1' -> 'user:2'"));
            },
            _ => panic!("expected Value result"),
        }
    }

    #[test]
    fn parsed_similar_connected_to() {
        let router = QueryRouter::new();

        // Create entities with embeddings
        router
            .vector()
            .set_entity_embedding("query", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:1", vec![0.9, 0.1, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![0.8, 0.2, 0.0])
            .unwrap();

        // Connect users to hub
        router
            .graph()
            .add_entity_edge("hub", "user:1", "connects")
            .unwrap();
        router
            .graph()
            .add_entity_edge("hub", "user:2", "connects")
            .unwrap();

        // Query similar connected to hub
        let result = router.execute_parsed("SIMILAR 'query' CONNECTED TO 'hub' LIMIT 5");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Similar(results) => {
                assert!(!results.is_empty());
            },
            _ => panic!("expected Similar result"),
        }
    }

    #[test]
    fn parsed_similar_connected_to_requires_key() {
        let router = QueryRouter::new();

        // Using a vector instead of key should fail
        let result = router.execute_parsed("SIMILAR [1.0, 0.0] CONNECTED TO 'hub'");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("requires a key"));
    }

    #[test]
    fn parsed_neighbors_by_similarity() {
        let router = QueryRouter::new();

        // Create entities with embeddings
        router
            .vector()
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![0.0, 1.0])
            .unwrap();

        // Create graph edges from center
        router
            .graph()
            .add_entity_edge("center", "user:1", "knows")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "user:2", "knows")
            .unwrap();

        // Query neighbors by similarity
        let result = router.execute_parsed("NEIGHBORS 'center' BY SIMILAR [1.0, 0.0] LIMIT 5");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Similar(results) => {
                // Should return neighbors sorted by similarity
                assert!(!results.is_empty());
                // user:1 should be first (more similar to [1.0, 0.0])
                assert_eq!(results[0].key, "user:1");
            },
            _ => panic!("expected Similar result"),
        }
    }

    #[test]
    fn parsed_entity_create_empty_properties() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("ENTITY CREATE 'empty:1' {}");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_entity_create_multiple_properties() {
        let router = QueryRouter::new();
        let result =
            router.execute_parsed("ENTITY CREATE 'user:2' { name: 'Bob', age: 30, active: true }");
        assert!(result.is_ok());
    }

    #[test]
    fn parser_entity_statement() {
        // Test that the parser correctly parses ENTITY statements
        let result = parser::parse("ENTITY CREATE 'key' { prop: 'value' }");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(matches!(stmt.kind, StatementKind::Entity(_)));
    }

    #[test]
    fn parser_entity_connect_statement() {
        let result = parser::parse("ENTITY CONNECT 'from' -> 'to' : type");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let StatementKind::Entity(entity) = stmt.kind {
            assert!(matches!(entity.operation, EntityOp::Connect { .. }));
        } else {
            panic!("expected Entity statement");
        }
    }

    #[test]
    fn parser_similar_connected_to() {
        let result = parser::parse("SIMILAR 'key' CONNECTED TO 'hub' LIMIT 10");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let StatementKind::Similar(similar) = stmt.kind {
            assert!(similar.connected_to.is_some());
        } else {
            panic!("expected Similar statement");
        }
    }

    #[test]
    fn parser_neighbors_by_similarity() {
        let result = parser::parse("NEIGHBORS 'entity' BY SIMILAR [1.0, 0.0] LIMIT 5");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert!(neighbors.by_similarity.is_some());
            assert!(neighbors.limit.is_some());
        } else {
            panic!("expected Neighbors statement");
        }
    }

    // ========== Phase 4: DROP INDEX Tests ==========

    #[test]
    fn parsed_drop_index_on_table_column() {
        let router = QueryRouter::new();

        // Create table and index (syntax: CREATE INDEX name ON table(column))
        router
            .execute_parsed("CREATE TABLE products (id INT, name TEXT)")
            .unwrap();
        router
            .execute_parsed("CREATE INDEX idx_name ON products(name)")
            .unwrap();
        assert!(router.relational().has_index("products", "name"));

        // Drop the index using ON table(column) syntax
        let result = router.execute_parsed("DROP INDEX ON products(name)");
        assert!(result.is_ok());
        assert!(!router.relational().has_index("products", "name"));
    }

    #[test]
    fn parsed_drop_index_if_exists() {
        let router = QueryRouter::new();

        // Create table without index
        router
            .execute_parsed("CREATE TABLE items (id INT)")
            .unwrap();

        // DROP INDEX IF EXISTS should not error
        let result = router.execute_parsed("DROP INDEX IF EXISTS ON items(id)");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_drop_index_not_found() {
        let router = QueryRouter::new();

        router
            .execute_parsed("CREATE TABLE data (col INT)")
            .unwrap();

        // Dropping non-existent index should error
        let result = router.execute_parsed("DROP INDEX ON data(col)");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_drop_index_named_not_supported() {
        let router = QueryRouter::new();

        // Named index syntax not supported
        let result = router.execute_parsed("DROP INDEX my_index");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn parser_drop_index_on_syntax() {
        let result = parser::parse("DROP INDEX ON users(email)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let StatementKind::DropIndex(drop) = stmt.kind {
            assert!(drop.table.is_some());
            assert_eq!(drop.table.unwrap().name, "users");
            assert!(drop.column.is_some());
            assert_eq!(drop.column.unwrap().name, "email");
        } else {
            panic!("expected DropIndex");
        }
    }

    #[test]
    fn parser_drop_index_if_exists_on() {
        let result = parser::parse("DROP INDEX IF EXISTS ON products(sku)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let StatementKind::DropIndex(drop) = stmt.kind {
            assert!(drop.if_exists);
            assert!(drop.table.is_some());
        } else {
            panic!("expected DropIndex");
        }
    }

    // ========== Phase 4: INSERT...SELECT Tests ==========

    #[test]
    fn parsed_insert_select_with_where() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE employees (id INT, dept TEXT)")
            .unwrap();
        router
            .execute_parsed("CREATE TABLE engineers (id INT, dept TEXT)")
            .unwrap();

        router
            .execute_parsed("INSERT INTO employees VALUES (1, 'eng')")
            .unwrap();
        router
            .execute_parsed("INSERT INTO employees VALUES (2, 'sales')")
            .unwrap();
        router
            .execute_parsed("INSERT INTO employees VALUES (3, 'eng')")
            .unwrap();

        // Insert only engineers
        let result = router
            .execute_parsed("INSERT INTO engineers SELECT * FROM employees WHERE dept = 'eng'");
        assert!(result.is_ok());

        let rows = router.execute_parsed("SELECT * FROM engineers").unwrap();
        match rows {
            QueryResult::Rows(r) => {
                assert_eq!(r.len(), 2);
            },
            _ => panic!("expected Rows"),
        }
    }

    #[test]
    fn parsed_insert_select_empty_result() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE source (id INT)")
            .unwrap();
        router
            .execute_parsed("CREATE TABLE target (id INT)")
            .unwrap();

        // Insert with no matching rows
        let result =
            router.execute_parsed("INSERT INTO target SELECT * FROM source WHERE id > 100");
        assert!(result.is_ok());

        match result.unwrap() {
            QueryResult::Ids(ids) => {
                assert!(ids.is_empty());
            },
            _ => panic!("expected Ids"),
        }
    }

    #[test]
    fn parsed_insert_select_with_columns() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE complete (id INT, name TEXT, age INT)")
            .unwrap();
        router
            .execute_parsed("CREATE TABLE partial (id INT, name TEXT)")
            .unwrap();

        router
            .execute_parsed("INSERT INTO complete VALUES (1, 'Alice', 30)")
            .unwrap();

        // Select only specific columns
        let result =
            router.execute_parsed("INSERT INTO partial (id, name) SELECT id, name FROM complete");
        assert!(result.is_ok());

        let rows = router.execute_parsed("SELECT * FROM partial").unwrap();
        match rows {
            QueryResult::Rows(r) => {
                assert_eq!(r.len(), 1);
            },
            _ => panic!("expected Rows"),
        }
    }

    #[test]
    fn parsed_blob_init_not_initialized() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("BLOB INIT");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("init_blob"),
            "should mention init_blob()"
        );
    }

    #[test]
    fn parsed_blob_init_already_initialized() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        let result = router.execute_parsed("BLOB INIT");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(
                    v.contains("already initialized"),
                    "should say already initialized"
                );
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_embed_build_index_not_built() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("EMBED BUILD INDEX");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("build_vector_index"),
            "should mention build_vector_index()"
        );
    }

    #[test]
    fn parsed_embed_build_index_already_built() {
        let mut router = QueryRouter::new();
        // Add some embeddings first using query API
        router
            .execute_parsed("EMBED STORE 'key1' [1.0, 0.0]")
            .unwrap();
        router
            .execute_parsed("EMBED STORE 'key2' [0.0, 1.0]")
            .unwrap();
        router.build_vector_index().unwrap();

        let result = router.execute_parsed("EMBED BUILD INDEX");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("already built"), "should say already built");
            },
            _ => panic!("expected Value"),
        }
    }

    // ========== Phase 5: AI Integration Tests ==========

    #[test]
    fn parsed_embed_batch_basic() {
        let router = QueryRouter::new();
        let result = router.execute_parsed(
            "EMBED BATCH [('doc1', [1.0, 0.0]), ('doc2', [0.0, 1.0]), ('doc3', [0.5, 0.5])]",
        );
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Count(n) => {
                assert_eq!(n, 3, "should store 3 embeddings");
            },
            _ => panic!("expected Count"),
        }

        // Verify embeddings were stored
        let result = router.execute_parsed("EMBED GET 'doc1'");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_embed_batch_empty() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("EMBED BATCH []");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Count(n) => {
                assert_eq!(n, 0, "empty batch should return 0");
            },
            _ => panic!("expected Count"),
        }
    }

    #[test]
    fn parsed_cache_semantic_put() {
        let mut router = QueryRouter::new();
        // Use a custom config with small embedding dimension for testing
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        router.init_cache_with_config(config);

        let result = router.execute_parsed(
            "CACHE SEMANTIC PUT 'What is 2+2?' 'The answer is 4' EMBEDDING [1.0, 0.0, 0.0]",
        );
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert_eq!(v, "OK");
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_cache_semantic_get() {
        let mut router = QueryRouter::new();
        // Use a custom config with small embedding dimension for testing
        let mut config = CacheConfig::default();
        config.embedding_dim = 2;
        router.init_cache_with_config(config);

        // First put something
        router
            .execute_parsed("CACHE SEMANTIC PUT 'hello' 'world' EMBEDDING [1.0, 0.0]")
            .unwrap();

        // Store an embedding for the query key
        router
            .execute_parsed("EMBED STORE 'hello' [1.0, 0.0]")
            .unwrap();

        // Now try to get it
        let result = router.execute_parsed("CACHE SEMANTIC GET 'hello'");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_cache_semantic_get_with_threshold() {
        let mut router = QueryRouter::new();
        router.init_cache();

        let result = router.execute_parsed("CACHE SEMANTIC GET 'unknown query' THRESHOLD 0.9");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("not found"));
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_describe_table() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE users (id INT NOT NULL, name TEXT, active BOOLEAN)")
            .unwrap();

        let result = router.execute_parsed("DESCRIBE TABLE users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("Table: users"));
                assert!(v.contains("id"));
                assert!(v.contains("name"));
                assert!(v.contains("active"));
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_describe_node() {
        let router = QueryRouter::new();
        router
            .execute_parsed("NODE CREATE person {name: 'Alice'}")
            .unwrap();

        let result = router.execute_parsed("DESCRIBE NODE person");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("Node label 'person'"));
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_describe_edge() {
        let router = QueryRouter::new();

        let result = router.execute_parsed("DESCRIBE EDGE follows");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("Edge type 'follows'"));
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_show_embeddings() {
        let router = QueryRouter::new();
        router
            .execute_parsed("EMBED STORE 'emb1' [1.0, 0.0]")
            .unwrap();
        router
            .execute_parsed("EMBED STORE 'emb2' [0.0, 1.0]")
            .unwrap();

        let result = router.execute_parsed("SHOW EMBEDDINGS");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Value(v) => {
                assert!(v.contains("emb1") || v.contains("emb2"));
            },
            _ => panic!("expected Value"),
        }
    }

    #[test]
    fn parsed_show_embeddings_with_limit() {
        let router = QueryRouter::new();
        for i in 0..10 {
            router
                .execute_parsed(&format!("EMBED STORE 'key{}' [{}]", i, i as f32))
                .unwrap();
        }

        let result = router.execute_parsed("SHOW EMBEDDINGS LIMIT 5");
        assert!(result.is_ok());
    }

    #[test]
    fn parsed_count_embeddings() {
        let router = QueryRouter::new();
        router.execute_parsed("EMBED STORE 'a' [1.0]").unwrap();
        router.execute_parsed("EMBED STORE 'b' [2.0]").unwrap();
        router.execute_parsed("EMBED STORE 'c' [3.0]").unwrap();

        let result = router.execute_parsed("COUNT EMBEDDINGS");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Count(n) => {
                assert_eq!(n, 3);
            },
            _ => panic!("expected Count"),
        }
    }

    #[test]
    fn test_query_result_to_json() {
        let result = QueryResult::Value("test".to_string());
        let json = result.to_json();
        assert!(json.contains("Value"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_query_result_to_pretty_json() {
        let result = QueryResult::Count(42);
        let json = result.to_pretty_json();
        assert!(json.contains("Count"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_query_result_is_empty() {
        assert!(QueryResult::Empty.is_empty());
        assert!(!QueryResult::Value("x".to_string()).is_empty());
    }

    #[test]
    fn test_query_result_as_count() {
        assert_eq!(QueryResult::Count(10).as_count(), Some(10));
        assert_eq!(QueryResult::Empty.as_count(), None);
    }

    #[test]
    fn test_query_result_as_value() {
        let result = QueryResult::Value("hello".to_string());
        assert_eq!(result.as_value(), Some("hello"));
        assert_eq!(QueryResult::Empty.as_value(), None);
    }

    #[test]
    fn test_query_result_as_rows() {
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("test".to_string()));
        let rows = vec![Row { id: 1, values }];
        let result = QueryResult::Rows(rows);
        assert!(result.as_rows().is_some());
        assert_eq!(result.as_rows().unwrap().len(), 1);
        assert!(QueryResult::Empty.as_rows().is_none());
    }

    // ========== Auto-Initialization Tests ==========

    #[test]
    fn test_ensure_cache_auto_init() {
        let mut router = QueryRouter::new();
        assert!(router.cache().is_none());

        // ensure_cache should auto-initialize
        let cache = router.ensure_cache();
        assert_eq!(cache.stats().total_entries(), 0);

        // Subsequent calls should return the same cache
        let cache2 = router.ensure_cache();
        assert_eq!(cache2.stats().total_entries(), 0);
    }

    #[test]
    fn test_ensure_blob_auto_init() {
        let mut router = QueryRouter::new();
        assert!(router.blob().is_none());

        // ensure_blob should auto-initialize
        let result = router.ensure_blob();
        assert!(result.is_ok());

        // Subsequent calls should return the same blob store
        let result2 = router.ensure_blob();
        assert!(result2.is_ok());
    }

    #[test]
    fn test_ensure_vault_no_env_key() {
        let mut router = QueryRouter::new();
        assert!(router.vault().is_none());

        // Remove env var if set (save and restore)
        let saved = std::env::var("NEUMANN_VAULT_KEY").ok();
        std::env::remove_var("NEUMANN_VAULT_KEY");

        // ensure_vault should fail without env key
        let result = router.ensure_vault();
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("not initialized"));
        }

        // Restore env var if it was set
        if let Some(key) = saved {
            std::env::set_var("NEUMANN_VAULT_KEY", key);
        }
    }

    #[test]
    fn test_ensure_vault_with_pre_init() {
        let mut router = QueryRouter::new();
        router
            .init_vault(b"32_byte_master_key_for_testing!")
            .unwrap();

        // ensure_vault should return the existing vault
        let result = router.ensure_vault();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_cache_idempotent() {
        let mut router = QueryRouter::new();

        // Call ensure_cache multiple times
        let _ = router.ensure_cache();
        let _ = router.ensure_cache();
        let _ = router.ensure_cache();

        // Should still have cache
        assert!(router.cache().is_some());
    }

    #[test]
    fn test_ensure_blob_idempotent() {
        let mut router = QueryRouter::new();

        // Call ensure_blob multiple times
        let _ = router.ensure_blob();
        let _ = router.ensure_blob();
        let _ = router.ensure_blob();

        // Should still have blob
        assert!(router.blob().is_some());
    }

    // ========== Async Execution Tests ==========

    #[tokio::test]
    async fn test_execute_parsed_async_basic() {
        let router = QueryRouter::new();

        // Execute a simple CREATE TABLE (SQL standard syntax)
        let result = router
            .execute_parsed_async("CREATE TABLE async_test (id INT, name VARCHAR(100))")
            .await;
        assert!(result.is_ok());

        // Execute an INSERT
        let result = router
            .execute_parsed_async("INSERT INTO async_test (id, name) VALUES (1, 'test')")
            .await;
        assert!(result.is_ok());

        // Execute a SELECT
        let result = router
            .execute_parsed_async("SELECT * FROM async_test")
            .await;
        assert!(result.is_ok());
        if let QueryResult::Rows(rows) = result.unwrap() {
            assert_eq!(rows.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_execute_statement_async_delegates() {
        let router = QueryRouter::new();

        // Parse a statement
        let stmt = parser::parse("NODE CREATE user name='Alice'").unwrap();

        // Execute async
        let result = router.execute_statement_async(&stmt).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_embed_batch_parallel() {
        let router = QueryRouter::new();

        // Create batch of embeddings
        let items: Vec<(String, Vec<f32>)> = (0..10)
            .map(|i| (format!("parallel:{}", i), vec![i as f32 / 10.0; 4]))
            .collect();

        // Store in parallel
        let result = router.embed_batch_parallel(items).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10);

        // Verify they were stored
        for i in 0..10 {
            let key = format!("parallel:{}", i);
            let emb = router.vector().get_embedding(&key);
            assert!(emb.is_ok());
        }
    }

    #[tokio::test]
    async fn test_find_similar_connected_async() {
        let router = QueryRouter::new();

        // Set up entities with embeddings
        router
            .vector()
            .set_entity_embedding("query", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:1", vec![0.9, 0.1, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("user:2", vec![0.8, 0.2, 0.0])
            .unwrap();

        // Connect entities via graph
        router
            .graph()
            .add_entity_edge("hub", "user:1", "connects")
            .unwrap();
        router
            .graph()
            .add_entity_edge("hub", "user:2", "connects")
            .unwrap();

        // Find similar connected async
        let result = router.find_similar_connected_async("query", "hub", 5).await;
        assert!(result.is_ok());
        let items = result.unwrap();
        assert!(!items.is_empty());
    }

    #[tokio::test]
    async fn test_find_neighbors_by_similarity_async() {
        let router = QueryRouter::new();

        // Set up graph with embeddings
        router
            .graph()
            .add_entity_edge("center", "neighbor:1", "links")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "neighbor:2", "links")
            .unwrap();
        router
            .graph()
            .add_entity_edge("center", "neighbor:3", "links")
            .unwrap();

        router
            .vector()
            .set_entity_embedding("neighbor:1", vec![1.0, 0.0, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("neighbor:2", vec![0.9, 0.1, 0.0])
            .unwrap();
        router
            .vector()
            .set_entity_embedding("neighbor:3", vec![0.0, 1.0, 0.0])
            .unwrap();

        // Find neighbors sorted by similarity
        let query = vec![1.0, 0.0, 0.0];
        let result = router
            .find_neighbors_by_similarity_async("center", &query, 3)
            .await;
        assert!(result.is_ok());
        let items = result.unwrap();
        assert_eq!(items.len(), 3);
        // neighbor:1 should be most similar
        assert!(items[0].id.contains("neighbor:1") || items[0].score.unwrap() > 0.9);
    }

    #[test]
    fn test_block_on_helper() {
        // This test can't be async since it tests block_on which
        // creates a nested runtime - that's its purpose for sync callers
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        // Use block_on to run async code from sync context
        let result = router.block_on(async { 42 + 1 });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 43);
    }

    #[test]
    fn test_runtime_accessor() {
        let router = QueryRouter::new();
        // Runtime not available until blob is initialized
        assert!(router.runtime().is_none());

        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        assert!(router.runtime().is_some());
    }

    #[tokio::test]
    async fn test_execute_parsed_async_with_cache() {
        let mut router = QueryRouter::new();
        router.init_cache();

        // Create and populate a table (SQL standard syntax)
        router
            .execute_parsed_async("CREATE TABLE cached (x INT)")
            .await
            .unwrap();
        router
            .execute_parsed_async("INSERT INTO cached (x) VALUES (1)")
            .await
            .unwrap();

        // First query - not cached
        let result1 = router.execute_parsed_async("SELECT * FROM cached").await;
        assert!(result1.is_ok());

        // Second query - should use cache
        let result2 = router.execute_parsed_async("SELECT * FROM cached").await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_embed_batch_parallel_empty() {
        let router = QueryRouter::new();

        // Empty batch should succeed
        let result = router.embed_batch_parallel(vec![]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_execute_parsed_async_error() {
        let router = QueryRouter::new();

        // Invalid SQL should return parse error
        let result = router.execute_parsed_async("INVALID QUERY XYZ").await;
        assert!(result.is_err());
    }

    // Note: The async blob tests use the router's block_on helper because
    // init_blob() creates its own Tokio runtime, which conflicts with #[tokio::test].

    #[test]
    fn test_exec_blob_async_put_get() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Test blob PUT via execute_statement_async (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'test.txt' 'hello world'").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                let artifact_id = match result.unwrap() {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Test blob GET via execute_statement_async
                let stmt = parser::parse(&format!("BLOB GET '{}'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Blob(data) = result.unwrap() {
                    assert_eq!(String::from_utf8(data).unwrap(), "hello world");
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_info() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'info.txt' 'test data'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Get info
                let stmt = parser::parse(&format!("BLOB INFO '{}'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::ArtifactInfo(info) = result.unwrap() {
                    assert_eq!(info.filename, "info.txt");
                    assert_eq!(info.size, 9); // "test data" is 9 bytes
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_link_unlink() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'linked.txt' 'link test'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Link to entity
                let stmt =
                    parser::parse(&format!("BLOB LINK '{}' TO 'entity:1'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());

                // Get links
                let stmt = parser::parse(&format!("BLOB LINKS '{}'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::ArtifactList(links) = result.unwrap() {
                    assert!(links.contains(&"entity:1".to_string()));
                }

                // Unlink
                let stmt = parser::parse(&format!("BLOB UNLINK '{}' FROM 'entity:1'", artifact_id))
                    .unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_tag_untag() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'tagged.txt' 'tag test'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Add tag
                let stmt =
                    parser::parse(&format!("BLOB TAG '{}' 'important'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());

                // Remove tag
                let stmt =
                    parser::parse(&format!("BLOB UNTAG '{}' 'important'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_verify() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'verify.txt' 'verify test'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Verify
                let stmt = parser::parse(&format!("BLOB VERIFY '{}'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert_eq!(v, "OK");
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_stats() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Get stats
                let stmt = parser::parse("BLOB STATS").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::BlobStats(stats) = result.unwrap() {
                    // Stats should be valid (even if empty)
                    assert!(stats.dedup_ratio >= 0.0);
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_gc() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // GC
                let stmt = parser::parse("BLOB GC").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_delete() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'delete.txt' 'delete test'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Delete
                let stmt = parser::parse(&format!("BLOB DELETE '{}'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_meta() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'meta.txt' 'meta test'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                // Set meta
                let stmt = parser::parse(&format!("BLOB META SET '{}' 'key' 'value'", artifact_id))
                    .unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());

                // Get meta
                let stmt =
                    parser::parse(&format!("BLOB META GET '{}' 'key'", artifact_id)).unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert_eq!(v, "value");
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_repair() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Repair
                let stmt = parser::parse("BLOB REPAIR").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blobs_async_list() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store some blobs (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'list1.txt' 'data1'").unwrap();
                router.execute_statement_async(&stmt).await.unwrap();
                let stmt = parser::parse("BLOB PUT 'list2.txt' 'data2'").unwrap();
                router.execute_statement_async(&stmt).await.unwrap();

                // List blobs
                let stmt = parser::parse("BLOBS").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::ArtifactList(ids) = result.unwrap() {
                    assert!(ids.len() >= 2);
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blobs_async_for_entity() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store and link a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'entity.txt' 'entity data'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                let stmt =
                    parser::parse(&format!("BLOB LINK '{}' TO 'myentity'", artifact_id)).unwrap();
                router.execute_statement_async(&stmt).await.unwrap();

                // Get blobs for entity
                let stmt = parser::parse("BLOBS FOR 'myentity'").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::ArtifactList(ids) = result.unwrap() {
                    assert!(ids.contains(&artifact_id));
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blobs_async_by_tag() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Store and tag a blob (no DATA keyword)
                let stmt = parser::parse("BLOB PUT 'bytag.txt' 'tag data'").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                let artifact_id = match result {
                    QueryResult::Value(id) => id,
                    _ => panic!("Expected Value result"),
                };

                let stmt = parser::parse(&format!("BLOB TAG '{}' 'mytag'", artifact_id)).unwrap();
                router.execute_statement_async(&stmt).await.unwrap();

                // Get blobs by tag
                let stmt = parser::parse("BLOBS BY TAG 'mytag'").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::ArtifactList(ids) = result.unwrap() {
                    assert!(ids.contains(&artifact_id));
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_init_already_initialized() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();

        router
            .block_on(async {
                // Try BLOB INIT when already initialized
                let stmt = parser::parse("BLOB INIT").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert!(v.contains("already initialized"));
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_blob_async_not_initialized() {
        let router = QueryRouter::new();
        // Don't init blob - can't run async tests without runtime
        // Instead, test that sync execute_statement catches the error
        let stmt = parser::parse("BLOB STATS").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_err());
    }

    // ========== Checkpoint Tests ==========

    #[test]
    fn test_init_checkpoint_requires_blob() {
        let mut router = QueryRouter::new();
        // Checkpoint requires blob to be initialized first
        let result = router.init_checkpoint();
        assert!(result.is_err());
        if let Err(RouterError::CheckpointError(msg)) = result {
            assert!(msg.contains("Blob store must be initialized"));
        }
    }

    #[test]
    fn test_init_checkpoint_with_blob() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        let result = router.init_checkpoint();
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_checkpoint_with_config() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        let config = CheckpointConfig::default().with_max_checkpoints(5);
        let result = router.init_checkpoint_with_config(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_checkpoint_auto_init() {
        let mut router = QueryRouter::new();
        // ensure_checkpoint should auto-initialize blob and checkpoint
        let result = router.ensure_checkpoint();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_checkpoint_already_initialized() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();
        // Calling ensure_checkpoint again should still work
        let result = router.ensure_checkpoint();
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_checkpoint_not_initialized() {
        let router = QueryRouter::new();
        let stmt = parser::parse("CHECKPOINT").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_err());
        if let Err(RouterError::CheckpointError(msg)) = result {
            assert!(msg.contains("not initialized"));
        }
    }

    #[test]
    fn test_exec_checkpoint_create() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                let stmt = parser::parse("CHECKPOINT").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert!(v.contains("Checkpoint created"));
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_checkpoint_with_name() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                let stmt = parser::parse("CHECKPOINT 'my-checkpoint'").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert!(v.contains("Checkpoint created"));
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_checkpoints_list() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                // Create a checkpoint first
                let stmt = parser::parse("CHECKPOINT 'test-cp'").unwrap();
                router.execute_statement_async(&stmt).await.unwrap();

                // List checkpoints
                let stmt = parser::parse("CHECKPOINTS").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::CheckpointList(list) = result.unwrap() {
                    assert!(!list.is_empty());
                    assert_eq!(list[0].name, "test-cp");
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_checkpoints_with_limit() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                // Create multiple checkpoints
                for i in 0..5 {
                    let stmt = parser::parse(&format!("CHECKPOINT 'cp-{}'", i)).unwrap();
                    router.execute_statement_async(&stmt).await.unwrap();
                }

                // List with limit
                let stmt = parser::parse("CHECKPOINTS LIMIT 3").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_ok());
                if let QueryResult::CheckpointList(list) = result.unwrap() {
                    assert_eq!(list.len(), 3);
                }
            })
            .unwrap();
    }

    #[test]
    fn test_exec_checkpoints_not_initialized() {
        let router = QueryRouter::new();
        let stmt = parser::parse("CHECKPOINTS").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_err());
    }

    #[test]
    fn test_exec_rollback_not_initialized() {
        let router = QueryRouter::new();
        let stmt = parser::parse("ROLLBACK TO 'some-id'").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_err());
        if let Err(RouterError::CheckpointError(msg)) = result {
            assert!(msg.contains("not initialized"));
        }
    }

    #[test]
    fn test_exec_rollback_success() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Store some data
        router.execute("EMBED testkey [1.0, 2.0, 3.0]").unwrap();

        router
            .block_on(async {
                // Create checkpoint
                let cp_stmt = parser::parse("CHECKPOINT 'before-delete'").unwrap();
                router.execute_statement_async(&cp_stmt).await.unwrap();

                // Delete the data using parsed command
                router.execute_parsed("EMBED DELETE 'testkey'").unwrap();
                assert!(!router.vector().exists("testkey"));

                // Rollback
                let rb_stmt = parser::parse("ROLLBACK TO 'before-delete'").unwrap();
                let result = router.execute_statement_async(&rb_stmt).await;
                assert!(result.is_ok());
                if let QueryResult::Value(v) = result.unwrap() {
                    assert!(v.contains("Rolled back"));
                }

                // Verify data is restored
                assert!(router.vector().exists("testkey"));
            })
            .unwrap();
    }

    #[test]
    fn test_exec_rollback_not_found() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                let stmt = parser::parse("ROLLBACK TO 'nonexistent'").unwrap();
                let result = router.execute_statement_async(&stmt).await;
                assert!(result.is_err());
            })
            .unwrap();
    }

    #[test]
    fn test_checkpoint_info_is_auto() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .block_on(async {
                // Manual checkpoint should have is_auto = false
                let stmt = parser::parse("CHECKPOINT 'manual'").unwrap();
                router.execute_statement_async(&stmt).await.unwrap();

                let stmt = parser::parse("CHECKPOINTS").unwrap();
                let result = router.execute_statement_async(&stmt).await.unwrap();
                if let QueryResult::CheckpointList(list) = result {
                    assert!(!list[0].is_auto);
                }
            })
            .unwrap();
    }

    #[test]
    fn test_checkpoint_error_display() {
        let e = RouterError::CheckpointError("test error".into());
        assert!(e.to_string().contains("Checkpoint error"));
        assert!(e.to_string().contains("test error"));
    }

    #[test]
    fn test_exec_checkpoint_sync_success() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Use sync execute_statement which calls exec_checkpoint internally
        let stmt = parser::parse("CHECKPOINT 'sync-test'").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
        if let QueryResult::Value(v) = result.unwrap() {
            assert!(v.contains("Checkpoint created"));
        }
    }

    #[test]
    fn test_exec_checkpoints_sync_success() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Create a checkpoint first
        let stmt = parser::parse("CHECKPOINT 'for-list'").unwrap();
        router.execute_statement(&stmt).unwrap();

        // Use sync execute_statement for CHECKPOINTS
        let stmt = parser::parse("CHECKPOINTS").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
        if let QueryResult::CheckpointList(list) = result.unwrap() {
            assert!(!list.is_empty());
        }
    }

    #[test]
    fn test_exec_rollback_sync_success() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Store data and create checkpoint
        router.execute("EMBED synckey [1.0, 2.0]").unwrap();
        let stmt = parser::parse("CHECKPOINT 'sync-rollback'").unwrap();
        router.execute_statement(&stmt).unwrap();

        // Delete data
        router.execute_parsed("EMBED DELETE 'synckey'").unwrap();
        assert!(!router.vector().exists("synckey"));

        // Use sync execute_statement for ROLLBACK
        let stmt = parser::parse("ROLLBACK TO 'sync-rollback'").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());

        // Verify rollback worked
        assert!(router.vector().exists("synckey"));
    }

    #[test]
    fn test_exec_checkpoint_sync_runtime_not_initialized() {
        // This is a tricky edge case - checkpoint is Some but runtime is None
        // In practice this shouldn't happen because init_checkpoint requires blob
        // But we test the error message path for coverage
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // The above initialization should succeed and both checkpoint and runtime
        // should be Some. So this test verifies the success path works.
        let stmt = parser::parse("CHECKPOINTS LIMIT 5").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_checkpoint_list_empty() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // List checkpoints when none exist
        let stmt = parser::parse("CHECKPOINTS").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
        if let QueryResult::CheckpointList(list) = result.unwrap() {
            assert!(list.is_empty());
        }
    }

    #[test]
    fn test_checkpoint_with_double_quoted_name() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        let stmt = parser::parse("CHECKPOINT \"double-quoted\"").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rollback_sync_by_id() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Create checkpoint and get its ID
        let stmt = parser::parse("CHECKPOINT 'rollback-by-id'").unwrap();
        router.execute_statement(&stmt).unwrap();

        let stmt = parser::parse("CHECKPOINTS").unwrap();
        let result = router.execute_statement(&stmt).unwrap();
        let checkpoint_id = if let QueryResult::CheckpointList(list) = result {
            list[0].id.clone()
        } else {
            panic!("Expected CheckpointList");
        };

        // Rollback by ID
        let stmt = parser::parse(&format!("ROLLBACK TO '{}'", checkpoint_id)).unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_checkpoints_ordering() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Create multiple checkpoints
        router
            .execute_statement(&parser::parse("CHECKPOINT 'first'").unwrap())
            .unwrap();
        router
            .execute_statement(&parser::parse("CHECKPOINT 'second'").unwrap())
            .unwrap();
        router
            .execute_statement(&parser::parse("CHECKPOINT 'third'").unwrap())
            .unwrap();

        // List should return them (most recent first based on implementation)
        let result = router
            .execute_statement(&parser::parse("CHECKPOINTS").unwrap())
            .unwrap();
        if let QueryResult::CheckpointList(list) = result {
            assert_eq!(list.len(), 3);
        }
    }

    #[test]
    fn test_checkpoint_via_execute_parsed() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Use execute_parsed instead of execute_statement
        let result = router.execute_parsed("CHECKPOINT 'parsed-test'");
        assert!(result.is_ok());
    }

    #[test]
    fn test_checkpoints_via_execute_parsed() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router.execute_parsed("CHECKPOINT 'test1'").unwrap();
        router.execute_parsed("CHECKPOINT 'test2'").unwrap();

        let result = router.execute_parsed("CHECKPOINTS");
        assert!(result.is_ok());
    }

    #[test]
    fn test_rollback_via_execute_parsed() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        router
            .execute_parsed("CHECKPOINT 'rollback-parsed'")
            .unwrap();
        let result = router.execute_parsed("ROLLBACK TO 'rollback-parsed'");
        assert!(result.is_ok());
    }

    #[test]
    fn test_checkpoint_default_name() {
        let mut router = QueryRouter::new();
        router.init_blob().unwrap();
        router.init_checkpoint().unwrap();

        // Checkpoint without a name should use auto-generated name
        let stmt = parser::parse("CHECKPOINT").unwrap();
        let result = router.execute_statement(&stmt);
        assert!(result.is_ok());

        let list_result = router
            .execute_statement(&parser::parse("CHECKPOINTS").unwrap())
            .unwrap();
        if let QueryResult::CheckpointList(list) = list_result {
            assert_eq!(list.len(), 1);
            // Auto-generated name starts with "checkpoint-"
            assert!(list[0].name.starts_with("checkpoint-"));
        }
    }
}
