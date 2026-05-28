// SPDX-License-Identifier: MIT OR Apache-2.0
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

pub mod cursor;
pub mod cursor_store;
pub mod cypher;
pub mod distributed;
mod error;
mod exec;
mod init;
mod policy;
mod protection;
mod result;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

pub use cursor::{CursorError, CursorId, CursorResultType, CursorState};
pub use cursor_store::{CursorStore, CursorStoreConfig};
pub use distributed::{
    DistributedQueryConfig, MergeStrategy, QueryPlan, QueryPlanner, ResultMerger, ShardId,
    ShardResult,
};
#[cfg(test)]
use graph_engine::Direction;
use graph_engine::{GraphEngine, PropertyValue};
use neumann_parser::{
    self as parser, error::ParseErrorKind, Expr, Property, Statement, StatementKind,
};
#[cfg(test)]
use neumann_parser::{EntityOp, ExprKind, GraphPatternOp, GraphPatternStmt};
#[cfg(test)]
use relational_engine::Value;
use relational_engine::{RelationalEngine, Row};
use std::path::PathBuf;
use tensor_blob::BlobStore;
use tensor_cache::Cache;
#[cfg(test)]
use tensor_cache::{CacheConfig, CacheLayer};
use tensor_chain::{ClusterOrchestrator, QueryExecutor, TensorChain};
use tensor_checkpoint::CheckpointManager;
#[cfg(test)]
use tensor_checkpoint::DestructiveOp;
#[cfg(test)]
use tensor_checkpoint::{CheckpointConfig, ConfirmationHandler};
use tensor_store::TensorStore;
use tensor_unified::UnifiedEngine;
#[cfg(test)]
use tensor_unified::UnifiedError;
use tensor_vault::Vault;
use tokio::runtime::Runtime;
use tracing::instrument;
use vector_engine::{HNSWIndex, VectorEngine};

// Re-export filter types for programmatic use by external consumers.
pub use vector_engine::{FilterCondition, FilterStrategy, FilterValue, FilteredSearchConfig};

pub use error::{Result, RouterError};
pub use policy::{classify_statement, StatementSafety};
pub use result::{
    AggregateResultValue, ArtifactInfoResult, BatchOperationResult, BindingValue, BlobStatsResult,
    CentralityItem, CentralityResult, CentralityType, ChainBlockInfo, ChainCodebookInfo,
    ChainDriftResult, ChainHistoryEntry, ChainResult, ChainSimilarResult, ChainTransitionAnalysis,
    CheckpointInfo, CommunityItem, CommunityResult, ConstraintInfo, EdgeResult, NodeResult,
    PageRankItem, PageRankResult, PagedQueryResult, PaginationOptions, PatternMatchBinding,
    PatternMatchResultValue, PatternMatchStatsValue, QueryResult, SimilarResult, SpatialResult,
    UnifiedResult,
};

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
    /// Current identity for vault access control (None = not authenticated)
    current_identity: Option<String>,
    /// Optional HNSW index for faster vector search
    hnsw_index: Option<(HNSWIndex, Vec<String>)>,
    /// Generation counter incremented on vector writes to the default namespace.
    ///
    /// HNSW freshness tracking only covers writes through `QueryRouter` methods.
    /// Direct writes to the underlying `VectorEngine` may cause stale results.
    vector_generation: AtomicU64,
    /// Generation at which the current HNSW index was built.
    hnsw_generation: AtomicU64,
    /// Directory for checkpoint storage files.
    checkpoint_dir: Option<PathBuf>,
    /// Optional checkpoint manager (requires checkpoint directory).
    checkpoint: Option<Arc<CheckpointManager>>,
    /// Optional tensor chain (requires initialization)
    chain: Option<Arc<TensorChain>>,
    /// Optional cluster orchestrator for distributed mode
    cluster: Option<Arc<ClusterOrchestrator>>,
    /// Tokio runtime for async cluster operations (shared with blob)
    cluster_runtime: Option<Arc<Runtime>>,
    /// Query planner for distributed execution
    distributed_planner: Option<Arc<QueryPlanner>>,
    /// Distributed query configuration
    distributed_config: DistributedQueryConfig,
    /// Local shard ID in the cluster
    local_shard_id: ShardId,
    /// Cursor store for paginated queries
    cursor_store: Arc<CursorStore>,
    /// Spatial index for 2D range queries (always initialized, zero cost when empty)
    spatial: Arc<parking_lot::RwLock<tensor_spatial::SpatialIndex<String>>>,
}

impl QueryRouter {
    /// Create a new query router with fresh engines sharing a common store.
    #[must_use]
    pub fn new() -> Self {
        Self::with_shared_store(TensorStore::new())
    }

    /// Create a query router with existing engines.
    ///
    /// The unified engine is initialized using the vector engine's store.
    pub fn with_engines(
        relational: Arc<RelationalEngine>,
        graph: Arc<GraphEngine>,
        vector: Arc<VectorEngine>,
    ) -> Self {
        let store = vector.store().clone();
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
            current_identity: None,
            hnsw_index: None,
            vector_generation: AtomicU64::new(0),
            hnsw_generation: AtomicU64::new(0),
            checkpoint_dir: None,
            checkpoint: None,
            chain: None,
            cluster: None,
            cluster_runtime: None,
            distributed_planner: None,
            distributed_config: DistributedQueryConfig::default(),
            local_shard_id: 0,
            cursor_store: Arc::new(CursorStore::new()),
            spatial: Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::new())),
        }
    }

    /// Create a query router with a shared `TensorStore` for unified entity access.
    ///
    /// All engines share the same store, enabling cross-engine queries on unified entities.
    /// Cloning `TensorStore` shares the underlying storage (via `Arc<DashMap>`).
    #[must_use]
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
            current_identity: None,
            hnsw_index: None,
            vector_generation: AtomicU64::new(0),
            hnsw_generation: AtomicU64::new(0),
            checkpoint_dir: None,
            checkpoint: None,
            chain: None,
            cluster: None,
            cluster_runtime: None,
            distributed_planner: None,
            distributed_config: DistributedQueryConfig::default(),
            local_shard_id: 0,
            cursor_store: Arc::new(CursorStore::new()),
            spatial: Arc::new(parking_lot::RwLock::new(tensor_spatial::SpatialIndex::new())),
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

    /// Get reference to the spatial index.
    pub const fn spatial(&self) -> &Arc<parking_lot::RwLock<tensor_spatial::SpatialIndex<String>>> {
        &self.spatial
    }

    /// Get reference to unified engine (if initialized).
    pub const fn unified(&self) -> Option<&UnifiedEngine> {
        self.unified.as_ref()
    }

    /// Returns a reference to the unified engine or an error if not initialized.
    fn require_unified(&self) -> Result<&UnifiedEngine> {
        self.unified.as_ref().ok_or_else(|| {
            RouterError::InvalidArgument("Unified engine not initialized".to_string())
        })
    }

    /// Creates a new Tokio runtime for sync-to-async bridging.
    fn create_runtime() -> Result<Runtime> {
        Runtime::new()
            .map_err(|e| RouterError::InvalidArgument(format!("Failed to create runtime: {e}")))
    }

    /// Increment the vector generation counter to signal that the HNSW index
    /// may be stale. Called after writes to the default vector namespace.
    fn bump_vector_generation(&self) {
        self.vector_generation.fetch_add(1, AtomicOrdering::SeqCst);
    }

    /// Returns true if the HNSW index matches the current vector generation.
    fn hnsw_is_fresh(&self) -> bool {
        self.hnsw_generation.load(AtomicOrdering::SeqCst)
            == self.vector_generation.load(AtomicOrdering::SeqCst)
    }

    /// Get the current identity or return an authentication error.
    /// Use this for operations that require authentication.
    fn require_identity(&self) -> Result<&str> {
        self.current_identity
            .as_deref()
            .ok_or(RouterError::AuthenticationRequired)
    }

    /// Execute a command string, trying the parser-first path with legacy fallback.
    ///
    /// Queries are first parsed into an AST. If parsing succeeds, the statement is
    /// executed through the unified path with caching and distributed execution.
    /// If parsing fails for one of the 12 legacy keywords, the legacy string-based
    /// handler is used as a fallback during migration.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or the command execution fails.
    #[instrument(skip(self))]
    #[allow(clippy::too_many_lines)]
    pub fn execute(&self, command: &str) -> Result<QueryResult> {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            return Err(RouterError::ParseError("Empty command".to_string()));
        }

        // Distributed execution check FIRST (operates on raw string)
        if let Some(result) = exec::cluster::try_execute_distributed(self, trimmed) {
            return result;
        }

        // Parse and execute
        let stmt = parser::parse(trimmed).map_err(|parse_err| {
            let upper = trimmed.to_ascii_uppercase();
            let first_word = upper.split_whitespace().next().unwrap_or("");
            if matches!(&parse_err.kind, ParseErrorKind::UnknownCommand(_)) {
                RouterError::UnknownCommand(first_word.to_string())
            } else {
                RouterError::ParseError(parse_err.format_with_source(trimmed))
            }
        })?;

        // Cache check for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            if let Some(cached) = exec::cache::try_cache_get(self, trimmed) {
                return Ok(cached);
            }
        }
        let result = self.execute_statement(&stmt)?;
        if Self::is_cacheable_statement(&stmt) {
            exec::cache::try_cache_put(self, trimmed, &result);
        }
        if Self::is_write_statement(&stmt) {
            exec::cache::invalidate_cache_on_write(self);
        }
        Ok(result)
    }

    /// Execute a paginated query.
    ///
    /// If a cursor is provided, resumes from that position. Otherwise, executes
    /// the query and creates a new cursor for subsequent pages.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails, cursor is invalid/expired, or the
    /// result type doesn't support pagination.
    #[instrument(skip(self))]
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn execute_paginated(
        &self,
        command: &str,
        options: PaginationOptions,
    ) -> Result<PagedQueryResult> {
        let page_size = options.page_size.unwrap_or(CursorState::DEFAULT_PAGE_SIZE);
        #[allow(clippy::cast_possible_truncation)] // TTL seconds won't exceed u32::MAX
        let ttl_secs = options
            .cursor_ttl
            .map_or(CursorState::DEFAULT_TTL_SECS, |d| d.as_secs() as u32)
            .min(CursorState::MAX_TTL_SECS);

        // If resuming from cursor, decode and validate
        let cursor_state = if let Some(ref token) = options.cursor {
            let state = CursorState::decode(token)?;

            // Verify query matches
            if state.query != command {
                return Err(RouterError::CursorError(
                    "Cursor query does not match request".to_string(),
                ));
            }

            Some(state)
        } else {
            None
        };

        // Execute the full query
        let full_result = self.execute(command)?;

        // Determine result type and apply pagination
        let (result_type, total_count, paged_result) = self.apply_pagination(
            &full_result,
            cursor_state.as_ref(),
            page_size,
            options.count_total,
        )?;

        let offset = cursor_state.as_ref().map_or(0, |s| s.offset);

        // Create cursor state for next page
        let has_more = total_count.is_some_and(|total| offset + page_size < total);

        let next_cursor = if has_more {
            let cursor_id = uuid::Uuid::new_v4().to_string();
            let mut next_state = CursorState::new(
                cursor_id,
                command.to_string(),
                result_type,
                page_size,
                total_count,
                ttl_secs,
            );
            next_state.offset = offset + page_size;

            // Store cursor
            self.cursor_store.insert(next_state.clone())?;

            Some(next_state.encode()?)
        } else {
            None
        };

        let prev_cursor = if offset > 0 {
            let cursor_id = uuid::Uuid::new_v4().to_string();
            let mut prev_state = CursorState::new(
                cursor_id,
                command.to_string(),
                result_type,
                page_size,
                total_count,
                ttl_secs,
            );
            prev_state.offset = offset.saturating_sub(page_size);

            self.cursor_store.insert(prev_state.clone())?;

            Some(prev_state.encode()?)
        } else {
            None
        };

        Ok(PagedQueryResult {
            result: paged_result,
            next_cursor,
            prev_cursor,
            total_count,
            has_more,
            page_size,
        })
    }

    /// Apply pagination to a query result.
    #[allow(clippy::unused_self)] // Method signature for API consistency
    fn apply_pagination(
        &self,
        result: &QueryResult,
        cursor_state: Option<&CursorState>,
        page_size: usize,
        count_total: bool,
    ) -> Result<(CursorResultType, Option<usize>, QueryResult)> {
        let offset = cursor_state.map_or(0, |s| s.offset);

        match result {
            QueryResult::Rows(rows) => {
                let total = if count_total { Some(rows.len()) } else { None };
                let paged: Vec<_> = rows.iter().skip(offset).take(page_size).cloned().collect();
                Ok((CursorResultType::Rows, total, QueryResult::Rows(paged)))
            },
            QueryResult::Nodes(nodes) => {
                let total = if count_total { Some(nodes.len()) } else { None };
                let paged: Vec<_> = nodes.iter().skip(offset).take(page_size).cloned().collect();
                Ok((CursorResultType::Nodes, total, QueryResult::Nodes(paged)))
            },
            QueryResult::Edges(edges) => {
                let total = if count_total { Some(edges.len()) } else { None };
                let paged: Vec<_> = edges.iter().skip(offset).take(page_size).cloned().collect();
                Ok((CursorResultType::Edges, total, QueryResult::Edges(paged)))
            },
            QueryResult::Similar(items) => {
                let total = if count_total { Some(items.len()) } else { None };
                let paged: Vec<_> = items.iter().skip(offset).take(page_size).cloned().collect();
                Ok((
                    CursorResultType::Similar,
                    total,
                    QueryResult::Similar(paged),
                ))
            },
            QueryResult::Unified(unified) => {
                let total = if count_total {
                    Some(unified.items.len())
                } else {
                    None
                };
                let paged_items: Vec<_> = unified
                    .items
                    .iter()
                    .skip(offset)
                    .take(page_size)
                    .cloned()
                    .collect();
                Ok((
                    CursorResultType::Unified,
                    total,
                    QueryResult::Unified(UnifiedResult {
                        description: unified.description.clone(),
                        items: paged_items,
                    }),
                ))
            },
            QueryResult::PatternMatch(pattern) => {
                let total = if count_total {
                    Some(pattern.matches.len())
                } else {
                    None
                };
                let paged_matches: Vec<_> = pattern
                    .matches
                    .iter()
                    .skip(offset)
                    .take(page_size)
                    .cloned()
                    .collect();
                Ok((
                    CursorResultType::PatternMatch,
                    total,
                    QueryResult::PatternMatch(PatternMatchResultValue {
                        matches: paged_matches,
                        stats: pattern.stats.clone(),
                    }),
                ))
            },
            // Non-paginatable result types return as-is
            _ => Err(RouterError::InvalidArgument(
                "Result type does not support pagination".to_string(),
            )),
        }
    }

    /// Close a cursor, freeing its resources.
    ///
    /// Returns `true` if the cursor was found and closed, `false` if not found.
    ///
    /// # Errors
    ///
    /// Returns an error if the cursor token is invalid or cannot be decoded.
    pub fn close_cursor(&self, cursor_token: &str) -> Result<bool> {
        let state = CursorState::decode(cursor_token)?;
        Ok(self.cursor_store.remove(&state.id))
    }

    /// Get a reference to the cursor store.
    #[must_use]
    pub const fn cursor_store(&self) -> &Arc<CursorStore> {
        &self.cursor_store
    }

    /// Execute a command string using the AST-based parser.
    ///
    /// This method uses the `neumann_parser` crate to parse the command into an AST,
    /// then dispatches to the appropriate engine based on the statement type.
    /// Cacheable queries (SELECT, SIMILAR, NEIGHBORS, PATH) are cached if a cache is configured.
    /// Write operations (INSERT, UPDATE, DELETE) invalidate the cache.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or statement execution fails.
    pub fn execute_parsed(&self, command: &str) -> Result<QueryResult> {
        // Try distributed execution first if cluster is active
        if let Some(result) = exec::cluster::try_execute_distributed(self, command) {
            return result;
        }

        let stmt = parser::parse(command)
            .map_err(|e| RouterError::ParseError(e.format_with_source(command)))?;

        // Check cache for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            if let Some(cached) = exec::cache::try_cache_get(self, command) {
                return Ok(cached);
            }
        }

        // Execute the statement
        let result = self.execute_statement(&stmt)?;

        // Cache the result for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            exec::cache::try_cache_put(self, command, &result);
        }

        // Invalidate cache on write operations
        if Self::is_write_statement(&stmt) {
            exec::cache::invalidate_cache_on_write(self);
        }

        Ok(result)
    }

    /// Execute a parsed statement.
    ///
    /// # Errors
    ///
    /// Returns an error if statement execution fails.
    #[instrument(skip(self, stmt))]
    #[allow(clippy::too_many_lines)] // Large match dispatch over all statement kinds
    pub fn execute_statement(&self, stmt: &Statement) -> Result<QueryResult> {
        match &stmt.kind {
            // SQL statements
            StatementKind::Select(select) => self.exec_select(select),
            StatementKind::Insert(insert) => self.exec_insert(insert),
            StatementKind::Update(update) => self.exec_update(update),
            StatementKind::Delete(delete) => self.exec_delete(delete),
            StatementKind::CreateTable(create) => self.exec_create_table(create),
            StatementKind::DropTable(drop) => self.exec_drop_table(drop),
            StatementKind::CreateIndex(create) => self.exec_create_index(create),
            StatementKind::DropIndex(drop) => self.exec_drop_index(drop),
            StatementKind::ShowTables => Ok(self.exec_show_tables()),
            StatementKind::ShowEmbeddings { limit } => self.exec_show_embeddings(limit.as_ref()),
            StatementKind::ShowVectorIndex => Ok(self.exec_show_vector_index()),
            StatementKind::CountEmbeddings => Ok(self.exec_count_embeddings()),
            StatementKind::Describe(desc) => exec::describe::exec_describe(self, desc),

            // Graph statements
            StatementKind::Node(node) => self.exec_node(node),
            StatementKind::Edge(edge) => self.exec_edge(edge),
            StatementKind::Neighbors(neighbors) => self.exec_neighbors(neighbors),
            StatementKind::Path(path) => self.exec_path(path),

            // Vector statements
            StatementKind::Embed(embed) => self.exec_embed(embed),
            StatementKind::Similar(similar) => self.exec_similar(similar),

            // Spatial statements
            StatementKind::Spatial(spatial) => self.exec_spatial(spatial),

            // Unified queries
            StatementKind::Find(find) => self.exec_find(find),
            StatementKind::Entity(entity) => self.exec_entity(entity),

            // Vault statements
            StatementKind::Vault(vault) => exec::vault::exec_vault(self, vault),

            // Cache statements
            StatementKind::Cache(cache) => exec::cache::exec_cache(self, cache),

            // Blob statements
            StatementKind::Blob(blob) => exec::blob::exec_blob(self, blob),
            StatementKind::Blobs(blobs) => exec::blob::exec_blobs(self, blobs),

            // Checkpoint statements
            StatementKind::Checkpoint(cp) => exec::checkpoint::exec_checkpoint(self, cp),
            StatementKind::Rollback(rb) => exec::checkpoint::exec_rollback(self, rb),
            StatementKind::Checkpoints(cps) => exec::checkpoint::exec_checkpoints(self, cps),

            // Chain statements
            StatementKind::Chain(chain) => exec::chain::exec_chain(self, chain),

            // Cluster statements
            StatementKind::Cluster(cluster) => exec::cluster::exec_cluster(self, cluster),

            // Extended graph statements
            StatementKind::GraphAlgorithm(algo) => self.exec_graph_algorithm(algo),
            StatementKind::GraphConstraint(constraint) => self.exec_graph_constraint(constraint),
            StatementKind::GraphIndex(idx) => self.exec_graph_index(idx),
            StatementKind::GraphAggregate(agg) => self.exec_graph_aggregate(agg),
            StatementKind::GraphPattern(pattern) => self.exec_graph_pattern(pattern),
            StatementKind::GraphBatch(batch) => self.exec_graph_batch(batch),

            // Cypher statements
            StatementKind::CypherMatch(stmt) => cypher::exec_cypher_match(&self.graph, stmt),
            StatementKind::CypherCreate(stmt) => cypher::exec_cypher_create(&self.graph, stmt),
            StatementKind::CypherDelete(stmt) => cypher::exec_cypher_delete(&self.graph, stmt),
            StatementKind::CypherMerge(stmt) => cypher::exec_cypher_merge(&self.graph, stmt),

            // Empty statement
            StatementKind::Empty => Ok(QueryResult::Empty),
        }
    }

    // ========== Query Cache Integration ==========

    /// Whether the statement's result may be cached. Delegates to [`policy`].
    pub(crate) const fn is_cacheable_statement(stmt: &Statement) -> bool {
        policy::is_cacheable_statement(stmt)
    }

    /// Whether the statement should invalidate cached query results.
    /// Delegates to [`policy`].
    pub(crate) const fn is_write_statement(stmt: &Statement) -> bool {
        policy::is_write_statement(stmt)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn expr_to_property_value(&self, expr: &Expr) -> Result<PropertyValue> {
        exec::expr::expr_to_property_value(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    #[allow(clippy::needless_pass_by_value, reason = "matches original signature")]
    fn property_value_to_f64(&self, value: Option<PropertyValue>) -> Option<f64> {
        exec::expr::property_value_to_f64(value)
    }

    // ========== AST-Based Execution Methods ==========

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn merge_rows(
        &self,
        row_a: Option<&Row>,
        row_b: Option<&Row>,
        table_a: &str,
        table_b: &str,
    ) -> Row {
        exec::expr::merge_rows(row_a, row_b, table_a, table_b)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn properties_to_map(&self, properties: &[Property]) -> Result<HashMap<String, PropertyValue>> {
        exec::expr::properties_to_map(properties)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn data_type_to_column_type(
        &self,
        dt: &parser::DataType,
    ) -> Result<relational_engine::ColumnType> {
        exec::expr::data_type_to_column_type(dt)
    }

    fn property_to_string(prop: &PropertyValue) -> String {
        exec::expr::property_to_string(prop)
    }

    // ========== Async Execution Methods ==========

    /// Execute a command string asynchronously using the AST-based parser.
    ///
    /// This method is the async counterpart to `execute_parsed()`. It provides
    /// truly non-blocking execution for I/O-bound operations like blob storage.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or statement execution fails.
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
            if let Some(cached) = exec::cache::try_cache_get(self, command) {
                return Ok(cached);
            }
        }

        // Execute the statement asynchronously
        let result = self.execute_statement_async(&stmt).await?;

        // Cache the result for cacheable statements
        if Self::is_cacheable_statement(&stmt) {
            exec::cache::try_cache_put(self, command, &result);
        }

        // Invalidate cache on write operations
        if Self::is_write_statement(&stmt) {
            exec::cache::invalidate_cache_on_write(self);
        }

        Ok(result)
    }

    /// Execute a parsed statement asynchronously.
    ///
    /// Most operations are synchronous (in-memory), but blob operations
    /// are truly async, avoiding runtime blocking.
    ///
    /// # Errors
    ///
    /// Returns an error if statement execution fails.
    pub async fn execute_statement_async(&self, stmt: &Statement) -> Result<QueryResult> {
        match &stmt.kind {
            // Blob statements are truly async
            StatementKind::Blob(blob) => exec::blob::exec_blob_async(self, blob).await,
            StatementKind::Blobs(blobs) => exec::blob::exec_blobs_async(self, blobs).await,

            // Checkpoint statements use sync file-based storage
            StatementKind::Checkpoint(cp) => exec::checkpoint::exec_checkpoint(self, cp),
            StatementKind::Rollback(rb) => exec::checkpoint::exec_rollback(self, rb),
            StatementKind::Checkpoints(cps) => exec::checkpoint::exec_checkpoints(self, cps),

            // All other statements delegate to sync execution
            // (they're in-memory and fast, no benefit from async)
            _ => self.execute_statement(stmt),
        }
    }

    /// Store multiple embeddings in parallel.
    ///
    /// Delegates to `UnifiedEngine::embed_batch()`.
    ///
    /// # Arguments
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
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has not been initialized.
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

impl QueryRouter {
    /// Execute a query for cluster distribution, returning serialized result.
    ///
    /// This is the same as `QueryExecutor::execute` but as a regular method
    /// for use when the router is behind a lock.
    ///
    /// # Errors
    ///
    /// Returns an error string if query parsing, execution, or serialization fails.
    pub fn execute_for_cluster(&self, query: &str) -> std::result::Result<Vec<u8>, String> {
        exec::cluster::execute_for_cluster(self, query)
    }
}

/// Implementation of `QueryExecutor` for distributed query handling.
///
/// This enables the `QueryRouter` to receive remote queries from the cluster
/// and execute them locally, returning serialized results.
impl QueryExecutor for QueryRouter {
    fn execute(&self, query: &str) -> std::result::Result<Vec<u8>, String> {
        exec::cluster::execute_for_cluster(self, query)
    }
}

#[cfg(test)]
#[allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::needless_collect,
    clippy::single_match,
    clippy::significant_drop_tightening,
    clippy::unnecessary_get_then_check,
    reason = "test code uses idiomatic match-panic, exact float compares on deterministic data, \
              and short-lived guards that don't merit pedantic rewrites"
)]
mod tests;
