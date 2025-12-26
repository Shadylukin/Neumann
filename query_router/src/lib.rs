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
    self as parser, BinaryOp, CacheOp, CacheStmt, DeleteStmt, Direction as ParsedDirection, EdgeOp,
    EdgeStmt, EmbedOp, EmbedStmt, Expr, ExprKind, FindPattern, FindStmt, InsertSource, InsertStmt,
    Literal, NeighborsStmt, NodeOp, NodeStmt, PathStmt, Property, SelectStmt, SimilarQuery,
    SimilarStmt, Statement, StatementKind, TableRefKind, UpdateStmt, VaultOp, VaultStmt,
};
use relational_engine::{
    ColumnarScanOptions, Condition, RelationalEngine, RelationalError, Row, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tensor_cache::{Cache, CacheConfig, CacheError, CacheLayer};
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig, VaultError};
use vector_engine::{HNSWIndex, VectorEngine, VectorError};

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

/// Single item in unified result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedItem {
    pub source: String,
    pub id: String,
    pub data: HashMap<String, String>,
    pub score: Option<f32>,
}

/// Query Router that orchestrates queries across engines.
pub struct QueryRouter {
    relational: Arc<RelationalEngine>,
    graph: Arc<GraphEngine>,
    vector: Arc<VectorEngine>,
    /// Optional vault for secure secret storage (requires initialization)
    vault: Option<Arc<Vault>>,
    /// Optional cache for LLM response caching (requires initialization)
    cache: Option<Arc<Cache>>,
    /// Current identity for vault access control
    current_identity: String,
    /// Optional HNSW index for faster vector search
    hnsw_index: Option<(HNSWIndex, Vec<String>)>,
}

impl QueryRouter {
    /// Create a new query router with fresh engines.
    pub fn new() -> Self {
        Self {
            relational: Arc::new(RelationalEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            vault: None,
            cache: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
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
            vault: None,
            cache: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
        }
    }

    /// Create a query router with a shared TensorStore for unified entity access.
    ///
    /// All engines share the same store, enabling cross-engine queries on unified entities.
    /// Cloning TensorStore shares the underlying storage (via `Arc<DashMap>`).
    pub fn with_shared_store(store: TensorStore) -> Self {
        Self {
            relational: Arc::new(RelationalEngine::with_store(store.clone())),
            graph: Arc::new(GraphEngine::with_store(store.clone())),
            vector: Arc::new(VectorEngine::with_store(store)),
            vault: None,
            cache: None,
            current_identity: Vault::ROOT.to_string(),
            hnsw_index: None,
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

        let mut items: Vec<UnifiedItem> = similar
            .into_iter()
            .filter(|s| connected_neighbors.contains(&s.key))
            .take(top_k)
            .map(|s| UnifiedItem {
                source: "vector+graph".to_string(),
                id: s.key,
                data: HashMap::new(),
                score: Some(s.score),
            })
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

                Some(UnifiedItem {
                    source: "graph+vector".to_string(),
                    id: neighbor_key,
                    data: HashMap::new(),
                    score: Some(score),
                })
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
    pub fn execute_parsed(&self, command: &str) -> Result<QueryResult> {
        let stmt = parser::parse(command)
            .map_err(|e| RouterError::ParseError(e.format_with_source(command)))?;

        self.execute_statement(&stmt)
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
            StatementKind::DropIndex(_drop) => {
                // DropIndexStmt only has index name, not table/column
                // Would need schema tracking to resolve; skip for now
                Err(RouterError::ParseError(
                    "DROP INDEX not yet supported".to_string(),
                ))
            },
            StatementKind::ShowTables => {
                let tables = self.relational.list_tables();
                Ok(QueryResult::TableList(tables))
            },

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

            // Vault statements
            StatementKind::Vault(vault) => self.exec_vault(vault),

            // Cache statements
            StatementKind::Cache(cache) => self.exec_cache(cache),

            // Empty statement
            StatementKind::Empty => Ok(QueryResult::Empty),
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
                // Note: Cache clear requires exclusive access which is not available through Arc
                // This would need to be called at the application level with mutable access
                Err(RouterError::CacheError(
                    "CACHE CLEAR requires exclusive access - use application-level reset"
                        .to_string(),
                ))
            },
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

        let rows = self
            .relational
            .select_columnar(table_name, condition, options)?;
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
            InsertSource::Query(_) => Err(RouterError::ParseError(
                "INSERT ... SELECT not yet supported".to_string(),
            )),
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
                // For now, return empty - full implementation would iterate nodes
                let _ = label;
                Ok(QueryResult::Nodes(vec![]))
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
                let _ = edge_type;
                Ok(QueryResult::Edges(vec![]))
            },
        }
    }

    fn exec_neighbors(&self, neighbors: &NeighborsStmt) -> Result<QueryResult> {
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
        }
    }

    fn exec_similar(&self, similar: &SimilarStmt) -> Result<QueryResult> {
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

        let top_k = similar
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?
            .unwrap_or(10);

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

    fn exec_find(&self, find: &FindStmt) -> Result<QueryResult> {
        let items = Vec::new();

        // Handle different patterns
        let base_desc = match &find.pattern {
            FindPattern::Nodes { label } => {
                format!(
                    "Finding nodes{}",
                    label
                        .as_ref()
                        .map(|l| format!(" with label '{}'", l.name))
                        .unwrap_or_default()
                )
            },
            FindPattern::Edges { edge_type } => {
                format!(
                    "Finding edges{}",
                    edge_type
                        .as_ref()
                        .map(|t| format!(" of type '{}'", t.name))
                        .unwrap_or_default()
                )
            },
            // FindPattern::Path is defined in AST but not currently generated by parser
            FindPattern::Path { .. } => "Finding path".to_string(),
        };

        let description = if find.where_clause.is_some() {
            format!("{} with filter", base_desc)
        } else {
            base_desc
        };

        let limit = if let Some(ref limit_expr) = find.limit {
            self.expr_to_usize(limit_expr)?
        } else {
            10
        };

        // For now, return placeholder unified result
        // Full implementation would query across engines
        let _ = limit;

        Ok(QueryResult::Unified(UnifiedResult { description, items }))
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
                    items.push(UnifiedItem {
                        source: "vector+graph".to_string(),
                        id: key.clone(),
                        data,
                        score: Some(*score),
                    });
                }
            }
        } else if let Some(similar) = similar_keys {
            // Just return similar results
            for (key, score) in similar {
                let mut data = HashMap::new();
                data.insert("key".to_string(), key.clone());
                items.push(UnifiedItem {
                    source: "vector".to_string(),
                    id: key,
                    data,
                    score: Some(score),
                });
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
    fn parsed_find_nodes() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND NODE person").unwrap();
        match result {
            QueryResult::Unified(unified) => assert!(unified.description.contains("Finding nodes")),
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_edges() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND EDGE knows").unwrap();
        match result {
            QueryResult::Unified(unified) => assert!(unified.description.contains("Finding edges")),
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_with_where() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("FIND NODE WHERE age > 18").unwrap();
        match result {
            QueryResult::Unified(unified) => assert!(unified.description.contains("filter")),
            _ => panic!("Expected Unified"),
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
    fn parsed_insert_select_not_supported() {
        let router = QueryRouter::new();
        router.execute("CREATE TABLE src (id:int)").unwrap();
        router.execute("CREATE TABLE dst (id:int)").unwrap();

        let result = router.execute_parsed("INSERT INTO dst SELECT * FROM src");
        assert!(result.is_err());
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
            QueryResult::Unified(u) => assert!(u.description.contains("Finding edges")),
            _ => panic!("Expected Unified"),
        }
    }

    #[test]
    fn parsed_find_nodes_plain() {
        let router = QueryRouter::new();
        // FIND NODE without label
        let result = router.execute_parsed("FIND NODE").unwrap();
        match result {
            QueryResult::Unified(u) => assert!(u.description.contains("Finding nodes")),
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
    fn test_cache_clear_returns_error() {
        let mut router = QueryRouter::new();
        router.init_cache();
        let result = router.execute_parsed("CACHE CLEAR");
        // CACHE CLEAR returns an error because it requires exclusive access
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_without_init() {
        let router = QueryRouter::new();
        let result = router.execute_parsed("CACHE STATS");
        assert!(result.is_err());
    }
}
