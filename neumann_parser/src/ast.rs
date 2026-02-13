// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Abstract Syntax Tree (AST) nodes for the Neumann query language.
//!
//! Defines all AST node types for:
//! - SQL statements (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP)
//! - Graph commands (NODE, EDGE, NEIGHBORS, PATH)
//! - Vector commands (EMBED, SIMILAR)
//! - Expressions (binary, unary, literals, function calls)

use std::fmt;

use crate::span::Span;

/// A complete statement.
#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    /// The kind of statement.
    pub kind: StatementKind,
    /// Source location of this statement.
    pub span: Span,
}

impl Statement {
    /// Creates a new statement.
    #[must_use]
    pub const fn new(kind: StatementKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Statement variants.
#[derive(Clone, Debug, PartialEq)]
pub enum StatementKind {
    // === SQL Statements ===
    /// SELECT query
    Select(SelectStmt),
    /// INSERT statement
    Insert(InsertStmt),
    /// UPDATE statement
    Update(UpdateStmt),
    /// DELETE statement
    Delete(DeleteStmt),
    /// CREATE TABLE statement
    CreateTable(CreateTableStmt),
    /// DROP TABLE statement
    DropTable(DropTableStmt),
    /// CREATE INDEX statement
    CreateIndex(CreateIndexStmt),
    /// DROP INDEX statement
    DropIndex(DropIndexStmt),
    /// SHOW TABLES statement
    ShowTables,
    /// SHOW EMBEDDINGS statement
    ShowEmbeddings {
        /// Maximum number of embeddings to show.
        limit: Option<Expr>,
    },
    /// SHOW VECTOR INDEX statement
    ShowVectorIndex,
    /// COUNT EMBEDDINGS statement
    CountEmbeddings,
    /// DESCRIBE statement
    Describe(DescribeStmt),

    // === Graph Statements ===
    /// NODE command
    Node(NodeStmt),
    /// EDGE command
    Edge(EdgeStmt),
    /// NEIGHBORS command
    Neighbors(NeighborsStmt),
    /// PATH command
    Path(PathStmt),

    // === Vector Statements ===
    /// EMBED command
    Embed(EmbedStmt),
    /// SIMILAR search
    Similar(SimilarStmt),

    // === Unified Statements ===
    /// FIND unified query
    Find(FindStmt),
    /// ENTITY command
    Entity(EntityStmt),

    // === Vault Statements ===
    /// VAULT command
    Vault(VaultStmt),

    // === Cache Statements ===
    /// CACHE command
    Cache(CacheStmt),

    // === Blob Storage Statements ===
    /// BLOB command
    Blob(BlobStmt),
    /// BLOBS command (list/query blobs)
    Blobs(BlobsStmt),

    // === Checkpoint Statements ===
    /// CHECKPOINT command - create a named checkpoint
    Checkpoint(CheckpointStmt),
    /// ROLLBACK command - restore to a checkpoint
    Rollback(RollbackStmt),
    /// CHECKPOINTS command - list checkpoints
    Checkpoints(CheckpointsStmt),

    // === Chain Statements ===
    /// CHAIN command
    Chain(ChainStmt),

    // === Cluster Statements ===
    /// CLUSTER command
    Cluster(ClusterStmt),

    // === Extended Graph Statements ===
    /// GRAPH ALGORITHM command (`PageRank`, centrality, etc.)
    GraphAlgorithm(GraphAlgorithmStmt),
    /// GRAPH CONSTRAINT command
    GraphConstraint(GraphConstraintStmt),
    /// GRAPH INDEX command
    GraphIndex(GraphIndexStmt),
    /// GRAPH AGGREGATE command (COUNT NODES, etc.)
    GraphAggregate(GraphAggregateStmt),
    /// GRAPH PATTERN command (MATCH PATTERN)
    GraphPattern(GraphPatternStmt),
    /// GRAPH BATCH command
    GraphBatch(GraphBatchStmt),

    // === Cypher Graph Statements ===
    /// Cypher MATCH statement
    CypherMatch(crate::cypher::CypherMatchStmt),
    /// Cypher CREATE statement
    CypherCreate(crate::cypher::CypherCreateStmt),
    /// Cypher DELETE statement
    CypherDelete(crate::cypher::CypherDeleteStmt),
    /// Cypher MERGE statement
    CypherMerge(crate::cypher::CypherMergeStmt),

    /// Empty statement (just semicolons)
    Empty,
}

// =============================================================================
// SQL Statements
// =============================================================================

/// SELECT statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectStmt {
    /// DISTINCT modifier
    pub distinct: bool,
    /// Selected columns/expressions
    pub columns: Vec<SelectItem>,
    /// FROM clause
    pub from: Option<FromClause>,
    /// WHERE clause
    pub where_clause: Option<Box<Expr>>,
    /// GROUP BY clause
    pub group_by: Vec<Expr>,
    /// HAVING clause
    pub having: Option<Box<Expr>>,
    /// ORDER BY clause
    pub order_by: Vec<OrderByItem>,
    /// LIMIT
    pub limit: Option<Box<Expr>>,
    /// OFFSET
    pub offset: Option<Box<Expr>>,
}

/// A selected column or expression.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectItem {
    /// The selected expression.
    pub expr: Expr,
    /// Optional alias (`AS name`).
    pub alias: Option<Ident>,
}

/// FROM clause with optional joins.
#[derive(Clone, Debug, PartialEq)]
pub struct FromClause {
    /// The primary table reference.
    pub table: TableRef,
    /// Join clauses.
    pub joins: Vec<Join>,
}

/// A table reference.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRef {
    /// The table reference kind.
    pub kind: TableRefKind,
    /// Optional table alias.
    pub alias: Option<Ident>,
    /// Source location.
    pub span: Span,
}

/// Table reference variants.
#[derive(Clone, Debug, PartialEq)]
pub enum TableRefKind {
    /// Simple table name
    Table(Ident),
    /// Subquery
    Subquery(Box<SelectStmt>),
}

/// A JOIN clause.
#[derive(Clone, Debug, PartialEq)]
pub struct Join {
    /// The type of join.
    pub kind: JoinKind,
    /// The joined table.
    pub table: TableRef,
    /// Join condition (ON or USING).
    pub condition: Option<JoinCondition>,
    /// Source location.
    pub span: Span,
}

/// JOIN types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinKind {
    /// Inner join.
    Inner,
    /// Left outer join.
    Left,
    /// Right outer join.
    Right,
    /// Full outer join.
    Full,
    /// Cross join.
    Cross,
    /// Natural join.
    Natural,
}

/// JOIN conditions.
#[derive(Clone, Debug, PartialEq)]
pub enum JoinCondition {
    /// ON condition
    On(Box<Expr>),
    /// USING columns
    Using(Vec<Ident>),
}

/// ORDER BY item.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderByItem {
    /// The expression to sort by.
    pub expr: Expr,
    /// Sort direction (ASC or DESC).
    pub direction: SortDirection,
    /// NULLS FIRST or NULLS LAST.
    pub nulls: Option<NullsOrder>,
}

/// Sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SortDirection {
    /// Ascending order.
    #[default]
    Asc,
    /// Descending order.
    Desc,
}

/// NULLS FIRST/LAST.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullsOrder {
    /// Nulls sort before non-null values.
    First,
    /// Nulls sort after non-null values.
    Last,
}

/// INSERT statement.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertStmt {
    /// Target table.
    pub table: Ident,
    /// Optional column list.
    pub columns: Option<Vec<Ident>>,
    /// Source of values.
    pub source: InsertSource,
}

/// Source of values for INSERT.
#[derive(Clone, Debug, PartialEq)]
pub enum InsertSource {
    /// VALUES clause
    Values(Vec<Vec<Expr>>),
    /// SELECT subquery
    Query(Box<SelectStmt>),
}

/// UPDATE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateStmt {
    /// Target table.
    pub table: Ident,
    /// Column assignments.
    pub assignments: Vec<Assignment>,
    /// Optional WHERE filter.
    pub where_clause: Option<Box<Expr>>,
}

/// Column assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    /// Target column.
    pub column: Ident,
    /// New value expression.
    pub value: Expr,
}

/// DELETE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct DeleteStmt {
    /// Target table.
    pub table: Ident,
    /// Optional WHERE filter.
    pub where_clause: Option<Box<Expr>>,
}

/// CREATE TABLE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct CreateTableStmt {
    /// Whether IF NOT EXISTS was specified.
    pub if_not_exists: bool,
    /// Table name.
    pub table: Ident,
    /// Column definitions.
    pub columns: Vec<ColumnDef>,
    /// Table-level constraints.
    pub constraints: Vec<TableConstraint>,
}

/// Column definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnDef {
    /// Column name.
    pub name: Ident,
    /// Column data type.
    pub data_type: DataType,
    /// Column-level constraints.
    pub constraints: Vec<ColumnConstraint>,
}

/// Data types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    /// SQL `INT` type.
    Int,
    /// SQL `INTEGER` type.
    Integer,
    /// SQL `BIGINT` type.
    Bigint,
    /// SQL `SMALLINT` type.
    Smallint,
    /// SQL `FLOAT` type.
    Float,
    /// SQL `DOUBLE` type.
    Double,
    /// SQL `REAL` type.
    Real,
    /// SQL `DECIMAL(precision, scale)` type.
    Decimal(Option<u32>, Option<u32>),
    /// SQL `NUMERIC(precision, scale)` type.
    Numeric(Option<u32>, Option<u32>),
    /// SQL `VARCHAR(length)` type.
    Varchar(Option<u32>),
    /// SQL `CHAR(length)` type.
    Char(Option<u32>),
    /// SQL `TEXT` type.
    Text,
    /// SQL `BOOLEAN` type.
    Boolean,
    /// SQL `DATE` type.
    Date,
    /// SQL `TIME` type.
    Time,
    /// SQL `TIMESTAMP` type.
    Timestamp,
    /// SQL `BLOB` type.
    Blob,
    /// Custom type name
    Custom(String),
}

/// Column constraints.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnConstraint {
    /// Column must not be null.
    NotNull,
    /// Column allows nulls (explicit).
    Null,
    /// Column values must be unique.
    Unique,
    /// Column is the primary key.
    PrimaryKey,
    /// Default value expression.
    Default(Expr),
    /// Check constraint expression.
    Check(Expr),
    /// Foreign key reference.
    References(ForeignKeyRef),
}

/// Foreign key reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignKeyRef {
    /// Referenced table.
    pub table: Ident,
    /// Referenced column.
    pub column: Option<Ident>,
    /// Action on delete.
    pub on_delete: Option<ReferentialAction>,
    /// Action on update.
    pub on_update: Option<ReferentialAction>,
}

/// Referential actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferentialAction {
    /// Cascade the operation to referencing rows.
    Cascade,
    /// Restrict the operation if references exist.
    Restrict,
    /// Set referencing columns to null.
    SetNull,
    /// Set referencing columns to their default.
    SetDefault,
    /// Take no action.
    NoAction,
}

/// Table-level constraints.
#[derive(Clone, Debug, PartialEq)]
pub enum TableConstraint {
    /// Primary key on columns.
    PrimaryKey(Vec<Ident>),
    /// Unique constraint on columns.
    Unique(Vec<Ident>),
    /// Foreign key constraint.
    ForeignKey {
        /// Local columns.
        columns: Vec<Ident>,
        /// Foreign key reference.
        reference: ForeignKeyRef,
    },
    /// Check constraint.
    Check(Expr),
}

/// DROP TABLE statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DropTableStmt {
    /// Whether IF EXISTS was specified.
    pub if_exists: bool,
    /// Table name.
    pub table: Ident,
    /// Whether CASCADE was specified.
    pub cascade: bool,
}

/// CREATE INDEX statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CreateIndexStmt {
    /// Whether this is a UNIQUE index.
    pub unique: bool,
    /// Whether IF NOT EXISTS was specified.
    pub if_not_exists: bool,
    /// Index name.
    pub name: Ident,
    /// Table to index.
    pub table: Ident,
    /// Columns to index.
    pub columns: Vec<Ident>,
}

/// DROP INDEX statement.
/// Supports both `DROP INDEX name` and `DROP INDEX ON table(column)` syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DropIndexStmt {
    /// Whether IF EXISTS was specified.
    pub if_exists: bool,
    /// Index name (for named indexes)
    pub name: Option<Ident>,
    /// Table name (for ON table(column) syntax)
    pub table: Option<Ident>,
    /// Column name (for ON table(column) syntax)
    pub column: Option<Ident>,
}

/// DESCRIBE statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescribeStmt {
    /// What to describe.
    pub target: DescribeTarget,
}

/// Target of DESCRIBE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DescribeTarget {
    /// DESCRIBE TABLE name
    Table(Ident),
    /// DESCRIBE NODE label
    Node(Ident),
    /// DESCRIBE EDGE type
    Edge(Ident),
}

// =============================================================================
// Graph Statements
// =============================================================================

/// NODE command.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeStmt {
    /// The node operation to perform.
    pub operation: NodeOp,
}

/// NODE operations.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeOp {
    /// Create a node: `NODE label { properties }`
    Create {
        /// Node label.
        label: Ident,
        /// Node properties.
        properties: Vec<Property>,
    },
    /// Get a node: `NODE GET id`
    Get {
        /// Node ID.
        id: Expr,
    },
    /// Delete a node: `NODE DELETE id`
    Delete {
        /// Node ID.
        id: Expr,
    },
    /// List nodes: `NODE LIST [label] [LIMIT n] [OFFSET m]`
    List {
        /// Optional label filter.
        label: Option<Ident>,
        /// Maximum number of results.
        limit: Option<Box<Expr>>,
        /// Number of results to skip.
        offset: Option<Box<Expr>>,
    },
}

/// EDGE command.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeStmt {
    /// The edge operation to perform.
    pub operation: EdgeOp,
}

/// EDGE operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EdgeOp {
    /// Create an edge: `EDGE from_ref edge_type to_ref { properties }`
    Create {
        /// Source node ID.
        from_id: Expr,
        /// Target node ID.
        to_id: Expr,
        /// Edge type label.
        edge_type: Ident,
        /// Edge properties.
        properties: Vec<Property>,
    },
    /// Get an edge: `EDGE GET id`
    Get {
        /// Edge ID.
        id: Expr,
    },
    /// Delete an edge: `EDGE DELETE id`
    Delete {
        /// Edge ID.
        id: Expr,
    },
    /// List edges: `EDGE LIST [type] [LIMIT n] [OFFSET m]`
    List {
        /// Optional edge type filter.
        edge_type: Option<Ident>,
        /// Maximum number of results.
        limit: Option<Box<Expr>>,
        /// Number of results to skip.
        offset: Option<Box<Expr>>,
    },
}

/// A property key-value pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    /// Property name.
    pub key: Ident,
    /// Property value expression.
    pub value: Expr,
}

/// NEIGHBORS command.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborsStmt {
    /// ID of the node to query neighbors for.
    pub node_id: Expr,
    /// Traversal direction.
    pub direction: Direction,
    /// Optional edge type filter.
    pub edge_type: Option<Ident>,
    /// Optional BY SIMILARITY constraint for cross-engine queries
    pub by_similarity: Option<Vec<Expr>>,
    /// Optional LIMIT for BY SIMILARITY queries
    pub limit: Option<Expr>,
}

/// Direction for graph traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Direction {
    /// Outgoing edges only.
    #[default]
    Outgoing,
    /// Incoming edges only.
    Incoming,
    /// Both directions.
    Both,
}

/// PATH command.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStmt {
    /// Path-finding algorithm to use.
    pub algorithm: PathAlgorithm,
    /// Source node ID.
    pub from_id: Expr,
    /// Destination node ID.
    pub to_id: Expr,
    /// Maximum traversal depth.
    pub max_depth: Option<Expr>,
    /// Minimum traversal depth.
    pub min_depth: Option<Expr>,
    /// Edge property to use as weight.
    pub weight_property: Option<Ident>,
}

/// Path-finding algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PathAlgorithm {
    /// Shortest unweighted path (BFS).
    #[default]
    Shortest,
    /// All paths up to max depth.
    All,
    /// Shortest weighted path (Dijkstra).
    Weighted,
    /// All weighted paths.
    AllWeighted,
    /// Variable-length path pattern.
    Variable,
}

// =============================================================================
// Vector Statements
// =============================================================================

/// EMBED command.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedStmt {
    /// The embed operation to perform.
    pub operation: EmbedOp,
    /// Optional collection name (e.g., `IN my_collection`).
    pub collection: Option<String>,
}

/// EMBED operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EmbedOp {
    /// Store embedding: `EMBED STORE 'key' [vector]`
    Store {
        /// Embedding key.
        key: Expr,
        /// Embedding vector values.
        vector: Vec<Expr>,
    },
    /// Get embedding: `EMBED GET 'key'`
    Get {
        /// Embedding key.
        key: Expr,
    },
    /// Delete embedding: `EMBED DELETE 'key'`
    Delete {
        /// Embedding key.
        key: Expr,
    },
    /// Build HNSW index: `EMBED BUILD INDEX`
    BuildIndex,
    /// Batch store embeddings: `EMBED BATCH [('key1', [v1, v2]), ('key2', [v1, v2])]`
    Batch {
        /// Key-vector pairs to store.
        items: Vec<(Expr, Vec<Expr>)>,
    },
}

/// SIMILAR search.
#[derive(Clone, Debug, PartialEq)]
pub struct SimilarStmt {
    /// The similarity query (by key or vector).
    pub query: SimilarQuery,
    /// Maximum number of results.
    pub limit: Option<Expr>,
    /// Distance metric to use.
    pub metric: Option<DistanceMetric>,
    /// Optional CONNECTED TO constraint for cross-engine queries
    pub connected_to: Option<Expr>,
    /// Optional collection name (e.g., `IN my_collection`).
    pub collection: Option<String>,
    /// Optional WHERE clause for filtered search.
    pub where_clause: Option<Box<Expr>>,
}

/// Query for SIMILAR search.
#[derive(Clone, Debug, PartialEq)]
pub enum SimilarQuery {
    /// Search by key
    Key(Expr),
    /// Search by vector
    Vector(Vec<Expr>),
}

/// Distance metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine similarity.
    #[default]
    Cosine,
    /// Euclidean distance.
    Euclidean,
    /// Dot product similarity.
    DotProduct,
}

// =============================================================================
// Unified Statements
// =============================================================================

/// FIND unified query.
#[derive(Clone, Debug, PartialEq)]
pub struct FindStmt {
    /// The find pattern.
    pub pattern: FindPattern,
    /// Optional WHERE filter.
    pub where_clause: Option<Box<Expr>>,
    /// Items to return.
    pub return_items: Vec<SelectItem>,
    /// Maximum number of results.
    pub limit: Option<Box<Expr>>,
}

/// FIND pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FindPattern {
    /// Match nodes by label
    Nodes {
        /// Optional label filter.
        label: Option<Ident>,
    },
    /// Match edges by type
    Edges {
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Match rows in a relational table: `FIND ROWS FROM table`
    Rows {
        /// Table name.
        table: Ident,
    },
    /// Match path pattern
    Path {
        /// Source node label.
        from: Option<Ident>,
        /// Edge type.
        edge: Option<Ident>,
        /// Target node label.
        to: Option<Ident>,
    },
}

/// ENTITY command.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityStmt {
    /// The entity operation to perform.
    pub operation: EntityOp,
}

/// ENTITY operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EntityOp {
    /// Create an entity: `ENTITY CREATE 'key' { properties } [EMBEDDING [vector]]`
    Create {
        /// Entity key.
        key: Expr,
        /// Entity properties.
        properties: Vec<Property>,
        /// Optional embedding vector.
        embedding: Option<Vec<Expr>>,
    },
    /// Get an entity: `ENTITY GET 'key'`
    Get {
        /// Entity key.
        key: Expr,
    },
    /// Update an entity: `ENTITY UPDATE 'key' { properties } [EMBEDDING [vector]]`
    Update {
        /// Entity key.
        key: Expr,
        /// Entity properties.
        properties: Vec<Property>,
        /// Optional embedding vector.
        embedding: Option<Vec<Expr>>,
    },
    /// Delete an entity: `ENTITY DELETE 'key'`
    Delete {
        /// Entity key.
        key: Expr,
    },
    /// Connect entities: `ENTITY CONNECT 'from' -> 'to' : type`
    Connect {
        /// Source entity key.
        from_key: Expr,
        /// Target entity key.
        to_key: Expr,
        /// Edge type label.
        edge_type: Ident,
    },
    /// Batch create entities: `ENTITY BATCH CREATE [{key: 'k1', props...}, ...]`
    Batch {
        /// Entity definitions.
        entities: Vec<BatchEntityDef>,
    },
}

/// Batch entity definition for ENTITY BATCH CREATE.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchEntityDef {
    /// Entity key.
    pub key: Expr,
    /// Entity properties.
    pub properties: Vec<Property>,
    /// Optional embedding vector.
    pub embedding: Option<Vec<Expr>>,
}

// =============================================================================
// Vault Statements
// =============================================================================

/// VAULT command.
#[derive(Clone, Debug, PartialEq)]
pub struct VaultStmt {
    /// The vault operation to perform.
    pub operation: VaultOp,
}

/// VAULT operations.
#[derive(Clone, Debug, PartialEq)]
pub enum VaultOp {
    /// Set a secret: `VAULT SET 'key' 'value'`
    Set {
        /// Secret key.
        key: Expr,
        /// Secret value.
        value: Expr,
    },
    /// Get a secret: `VAULT GET 'key'`
    Get {
        /// Secret key.
        key: Expr,
    },
    /// Delete a secret: `VAULT DELETE 'key'`
    Delete {
        /// Secret key.
        key: Expr,
    },
    /// List secrets: `VAULT LIST 'pattern'`
    List {
        /// Optional name pattern.
        pattern: Option<Expr>,
    },
    /// Rotate a secret: `VAULT ROTATE 'key' 'new_value'`
    Rotate {
        /// Secret key.
        key: Expr,
        /// New secret value.
        new_value: Expr,
    },
    /// Grant access: `VAULT GRANT 'entity' ON 'key'`
    Grant {
        /// Entity to grant access to.
        entity: Expr,
        /// Secret key.
        key: Expr,
    },
    /// Revoke access: `VAULT REVOKE 'entity' ON 'key'`
    Revoke {
        /// Entity to revoke access from.
        entity: Expr,
        /// Secret key.
        key: Expr,
    },
}

// =============================================================================
// Cache Statements
// =============================================================================

/// CACHE command.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheStmt {
    /// The cache operation to perform.
    pub operation: CacheOp,
}

/// CACHE operations.
#[derive(Clone, Debug, PartialEq)]
pub enum CacheOp {
    /// Initialize cache: `CACHE INIT`
    Init,
    /// Show cache statistics: `CACHE STATS`
    Stats,
    /// Clear all cache entries: `CACHE CLEAR`
    Clear,
    /// Evict entries: `CACHE EVICT [n]`
    Evict {
        /// Number of entries to evict.
        count: Option<Expr>,
    },
    /// Get cached response: `CACHE GET 'key'`
    Get {
        /// Cache key.
        key: Expr,
    },
    /// Store cache entry: `CACHE PUT 'key' 'value'`
    Put {
        /// Cache key.
        key: Expr,
        /// Value to cache.
        value: Expr,
    },
    /// Semantic cache lookup: `CACHE SEMANTIC GET 'query' [THRESHOLD n]`
    SemanticGet {
        /// Query string.
        query: Expr,
        /// Similarity threshold.
        threshold: Option<Expr>,
    },
    /// Semantic cache store: `CACHE SEMANTIC PUT 'query' 'response' EMBEDDING [vector]`
    SemanticPut {
        /// Query string.
        query: Expr,
        /// Cached response.
        response: Expr,
        /// Embedding vector.
        embedding: Vec<Expr>,
    },
}

// =============================================================================
// Cluster Statements
// =============================================================================

/// CLUSTER command.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterStmt {
    /// The cluster operation to perform.
    pub operation: ClusterOp,
}

/// CLUSTER operations.
#[derive(Clone, Debug, PartialEq)]
pub enum ClusterOp {
    /// Connect to cluster: `CLUSTER CONNECT 'address'`
    Connect {
        /// Cluster address(es).
        addresses: Expr,
    },
    /// Disconnect from cluster: `CLUSTER DISCONNECT`
    Disconnect,
    /// Show cluster status: `CLUSTER STATUS`
    Status,
    /// List cluster nodes: `CLUSTER NODES`
    Nodes,
    /// Show current leader: `CLUSTER LEADER`
    Leader,
}

// =============================================================================
// Blob Storage Statements
// =============================================================================

/// BLOB command.
#[derive(Clone, Debug, PartialEq)]
pub struct BlobStmt {
    /// The blob operation to perform.
    pub operation: BlobOp,
}

/// BLOB operations.
#[derive(Clone, Debug, PartialEq)]
pub enum BlobOp {
    /// Initialize blob store: `BLOB INIT`
    Init,
    /// Store blob: `BLOB PUT 'filename' DATA` or `BLOB PUT 'filename' FROM 'path'`
    Put {
        /// File name for the blob.
        filename: Expr,
        /// Inline data content.
        data: Option<Expr>,
        /// Path to read data from.
        from_path: Option<Expr>,
        /// Additional options.
        options: BlobOptions,
    },
    /// Get blob: `BLOB GET 'artifact_id'` or `BLOB GET 'artifact_id' TO 'path'`
    Get {
        /// Artifact ID.
        artifact_id: Expr,
        /// Optional file path to write to.
        to_path: Option<Expr>,
    },
    /// Delete blob: `BLOB DELETE 'artifact_id'`
    Delete {
        /// Artifact ID.
        artifact_id: Expr,
    },
    /// Show blob info: `BLOB INFO 'artifact_id'`
    Info {
        /// Artifact ID.
        artifact_id: Expr,
    },
    /// Link blob to entity: `BLOB LINK 'artifact_id' TO entity`
    Link {
        /// Artifact ID.
        artifact_id: Expr,
        /// Entity to link to.
        entity: Expr,
    },
    /// Unlink blob from entity: `BLOB UNLINK 'artifact_id' FROM entity`
    Unlink {
        /// Artifact ID.
        artifact_id: Expr,
        /// Entity to unlink from.
        entity: Expr,
    },
    /// Get links: `BLOB LINKS 'artifact_id'`
    Links {
        /// Artifact ID.
        artifact_id: Expr,
    },
    /// Add tag: `BLOB TAG 'artifact_id' 'tag'`
    Tag {
        /// Artifact ID.
        artifact_id: Expr,
        /// Tag to add.
        tag: Expr,
    },
    /// Remove tag: `BLOB UNTAG 'artifact_id' 'tag'`
    Untag {
        /// Artifact ID.
        artifact_id: Expr,
        /// Tag to remove.
        tag: Expr,
    },
    /// Verify integrity: `BLOB VERIFY 'artifact_id'`
    Verify {
        /// Artifact ID.
        artifact_id: Expr,
    },
    /// Run garbage collection: `BLOB GC` or `BLOB GC FULL`
    Gc {
        /// Whether to run full GC.
        full: bool,
    },
    /// Repair blob storage: `BLOB REPAIR`
    Repair,
    /// Show blob statistics: `BLOB STATS`
    Stats,
    /// Set metadata: `BLOB META SET 'artifact_id' 'key' 'value'`
    MetaSet {
        /// Artifact ID.
        artifact_id: Expr,
        /// Metadata key.
        key: Expr,
        /// Metadata value.
        value: Expr,
    },
    /// Get metadata: `BLOB META GET 'artifact_id' 'key'`
    MetaGet {
        /// Artifact ID.
        artifact_id: Expr,
        /// Metadata key.
        key: Expr,
    },
}

/// Options for BLOB PUT.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BlobOptions {
    /// Content type
    pub content_type: Option<Expr>,
    /// Creator
    pub created_by: Option<Expr>,
    /// Entities to link
    pub link: Vec<Expr>,
    /// Tags to apply
    pub tag: Vec<Expr>,
}

/// BLOBS command (list/query blobs).
#[derive(Clone, Debug, PartialEq)]
pub struct BlobsStmt {
    /// The blobs query operation.
    pub operation: BlobsOp,
}

/// BLOBS operations.
#[derive(Clone, Debug, PartialEq)]
pub enum BlobsOp {
    /// List all blobs: `BLOBS`
    List {
        /// Optional filename pattern.
        pattern: Option<Expr>,
    },
    /// Find blobs for entity: `BLOBS FOR entity`
    For {
        /// Entity to find blobs for.
        entity: Expr,
    },
    /// Find blobs by tag: `BLOBS BY TAG 'tag'`
    ByTag {
        /// Tag to filter by.
        tag: Expr,
    },
    /// Find blobs by content type: `BLOBS WHERE TYPE = 'type'`
    ByType {
        /// Content type to filter by.
        content_type: Expr,
    },
    /// Find similar blobs: `BLOBS SIMILAR TO 'artifact_id' LIMIT n`
    Similar {
        /// Artifact ID to compare against.
        artifact_id: Expr,
        /// Maximum number of results.
        limit: Option<Expr>,
    },
}

// =============================================================================
// Checkpoint Statements
// =============================================================================

/// CHECKPOINT statement: `CHECKPOINT` or `CHECKPOINT 'name'`
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointStmt {
    /// Optional checkpoint name.
    pub name: Option<Expr>,
}

/// ROLLBACK statement: `ROLLBACK TO 'checkpoint_id'`
#[derive(Clone, Debug, PartialEq)]
pub struct RollbackStmt {
    /// Checkpoint ID or name to roll back to.
    pub target: Expr,
}

/// CHECKPOINTS statement: `CHECKPOINTS` or `CHECKPOINTS LIMIT n`
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointsStmt {
    /// Maximum number of checkpoints to list.
    pub limit: Option<Expr>,
}

// =============================================================================
// Chain Statements
// =============================================================================

/// CHAIN command.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainStmt {
    /// The chain operation to perform.
    pub operation: ChainOp,
}

/// CHAIN operations.
#[derive(Clone, Debug, PartialEq)]
pub enum ChainOp {
    /// Begin a chain transaction: `BEGIN CHAIN TRANSACTION`
    Begin,
    /// Commit a chain transaction: `COMMIT CHAIN`
    Commit,
    /// Rollback chain to height: `ROLLBACK CHAIN TO height`
    Rollback {
        /// Block height to roll back to.
        height: Expr,
    },
    /// Get chain history for key: `CHAIN HISTORY 'key'`
    History {
        /// Key to get history for.
        key: Expr,
    },
    /// Search chain by similarity: `CHAIN SIMILAR [embedding] LIMIT n`
    Similar {
        /// Embedding vector to search by.
        embedding: Vec<Expr>,
        /// Maximum number of results.
        limit: Option<Expr>,
    },
    /// Get chain drift metrics: `CHAIN DRIFT FROM height TO height`
    Drift {
        /// Starting block height.
        from_height: Expr,
        /// Ending block height.
        to_height: Expr,
    },
    /// Show global codebook: `SHOW CODEBOOK GLOBAL`
    ShowCodebookGlobal,
    /// Show local codebook: `SHOW CODEBOOK LOCAL 'domain'`
    ShowCodebookLocal {
        /// Codebook domain.
        domain: Expr,
    },
    /// Analyze codebook transitions: `ANALYZE CODEBOOK TRANSITIONS`
    AnalyzeTransitions,
    /// Get chain height: `CHAIN HEIGHT`
    Height,
    /// Get chain tip: `CHAIN TIP`
    Tip,
    /// Get block at height: `CHAIN BLOCK height`
    Block {
        /// Block height.
        height: Expr,
    },
    /// Verify chain integrity: `CHAIN VERIFY`
    Verify,
}

// =============================================================================
// Extended Graph Statements
// =============================================================================

/// GRAPH ALGORITHM command.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphAlgorithmStmt {
    /// The graph algorithm to run.
    pub operation: GraphAlgorithmOp,
}

/// Graph algorithm operations.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphAlgorithmOp {
    /// Run `PageRank` centrality.
    PageRank {
        /// Damping factor (default 0.85).
        damping: Option<Expr>,
        /// Convergence tolerance.
        tolerance: Option<Expr>,
        /// Maximum number of iterations.
        max_iterations: Option<Expr>,
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Run betweenness centrality.
    BetweennessCentrality {
        /// Fraction of nodes to sample.
        sampling_ratio: Option<Expr>,
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Run closeness centrality.
    ClosenessCentrality {
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Run eigenvector centrality.
    EigenvectorCentrality {
        /// Maximum number of iterations.
        max_iterations: Option<Expr>,
        /// Convergence tolerance.
        tolerance: Option<Expr>,
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Detect communities using the Louvain method.
    LouvainCommunities {
        /// Resolution parameter for community granularity.
        resolution: Option<Expr>,
        /// Maximum number of passes.
        max_passes: Option<Expr>,
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Detect communities via label propagation.
    LabelPropagation {
        /// Maximum number of iterations.
        max_iterations: Option<Expr>,
        /// Traversal direction.
        direction: Option<Direction>,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
}

/// GRAPH CONSTRAINT command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphConstraintStmt {
    /// The constraint operation.
    pub operation: GraphConstraintOp,
}

/// Graph constraint operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphConstraintOp {
    /// Create a new constraint.
    Create {
        /// Constraint name.
        name: Ident,
        /// Constraint target (node or edge).
        target: ConstraintTarget,
        /// Property to constrain.
        property: Ident,
        /// Type of constraint.
        constraint_type: ConstraintType,
    },
    /// Drop a constraint by name.
    Drop {
        /// Constraint name.
        name: Ident,
    },
    /// List all constraints.
    List,
    /// Get a constraint by name.
    Get {
        /// Constraint name.
        name: Ident,
    },
}

/// Constraint target (NODE or EDGE).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintTarget {
    /// Constraint on nodes.
    Node {
        /// Optional node label filter.
        label: Option<Ident>,
    },
    /// Constraint on edges.
    Edge {
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
}

/// Constraint types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Unique property values.
    Unique,
    /// Property must exist.
    Exists,
    /// Property must have a specific type.
    Type(String),
}

/// GRAPH INDEX command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphIndexStmt {
    /// The index operation.
    pub operation: GraphIndexOp,
}

/// Graph index operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphIndexOp {
    /// Create index on a node property.
    CreateNodeProperty {
        /// Property to index.
        property: Ident,
    },
    /// Create index on an edge property.
    CreateEdgeProperty {
        /// Property to index.
        property: Ident,
    },
    /// Create index on node labels.
    CreateLabel,
    /// Create index on edge types.
    CreateEdgeType,
    /// Drop a node property index.
    DropNode {
        /// Property whose index to drop.
        property: Ident,
    },
    /// Drop an edge property index.
    DropEdge {
        /// Property whose index to drop.
        property: Ident,
    },
    /// Show node indexes.
    ShowNodeIndexes,
    /// Show edge indexes.
    ShowEdgeIndexes,
}

/// GRAPH AGGREGATE command.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphAggregateStmt {
    /// The aggregate operation.
    pub operation: GraphAggregateOp,
}

/// Graph aggregate operations.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphAggregateOp {
    /// Count nodes, optionally filtered by label.
    CountNodes {
        /// Optional label filter.
        label: Option<Ident>,
    },
    /// Count edges, optionally filtered by type.
    CountEdges {
        /// Optional edge type filter.
        edge_type: Option<Ident>,
    },
    /// Aggregate a node property.
    AggregateNodeProperty {
        /// Aggregate function to apply.
        function: AggregateFunction,
        /// Property to aggregate.
        property: Ident,
        /// Optional label filter.
        label: Option<Ident>,
        /// Optional WHERE filter.
        filter: Option<Box<Expr>>,
    },
    /// Aggregate an edge property.
    AggregateEdgeProperty {
        /// Aggregate function to apply.
        function: AggregateFunction,
        /// Property to aggregate.
        property: Ident,
        /// Optional edge type filter.
        edge_type: Option<Ident>,
        /// Optional WHERE filter.
        filter: Option<Box<Expr>>,
    },
}

/// Aggregate functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Sum of values.
    Sum,
    /// Average of values.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Count of values.
    Count,
}

/// GRAPH PATTERN command.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphPatternStmt {
    /// The pattern operation.
    pub operation: GraphPatternOp,
}

/// Graph pattern operations.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphPatternOp {
    /// Match a pattern and return results.
    Match {
        /// The pattern to match.
        pattern: PatternSpec,
        /// Maximum number of results.
        limit: Option<Expr>,
    },
    /// Count matches of a pattern.
    Count {
        /// The pattern to count.
        pattern: PatternSpec,
    },
    /// Check if a pattern exists.
    Exists {
        /// The pattern to check.
        pattern: PatternSpec,
    },
}

/// Pattern specification for graph matching.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternSpec {
    /// Node patterns in this match.
    pub nodes: Vec<NodePatternSpec>,
    /// Edge patterns in this match.
    pub edges: Vec<EdgePatternSpec>,
}

/// Node pattern in a match.
#[derive(Clone, Debug, PartialEq)]
pub struct NodePatternSpec {
    /// Optional binding alias.
    pub alias: Option<Ident>,
    /// Optional node label filter.
    pub label: Option<Ident>,
    /// Property filters.
    pub properties: Vec<Property>,
}

/// Edge pattern in a match.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgePatternSpec {
    /// Optional binding alias.
    pub alias: Option<Ident>,
    /// Optional edge type filter.
    pub edge_type: Option<Ident>,
    /// Traversal direction.
    pub direction: Direction,
    /// Index of the source node in the pattern.
    pub from_node: usize,
    /// Index of the target node in the pattern.
    pub to_node: usize,
    /// Property filters.
    pub properties: Vec<Property>,
}

/// GRAPH BATCH command.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphBatchStmt {
    /// The batch operation.
    pub operation: GraphBatchOp,
}

/// Graph batch operations.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphBatchOp {
    /// Batch create nodes.
    CreateNodes {
        /// Node definitions.
        nodes: Vec<BatchNodeDef>,
    },
    /// Batch create edges.
    CreateEdges {
        /// Edge definitions.
        edges: Vec<BatchEdgeDef>,
    },
    /// Batch delete nodes by ID.
    DeleteNodes {
        /// Node IDs to delete.
        ids: Vec<Expr>,
    },
    /// Batch delete edges by ID.
    DeleteEdges {
        /// Edge IDs to delete.
        ids: Vec<Expr>,
    },
    /// Batch update node properties.
    UpdateNodes {
        /// Node updates.
        updates: Vec<BatchNodeUpdate>,
    },
}

/// Batch node definition for creation.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchNodeDef {
    /// Node labels.
    pub labels: Vec<Ident>,
    /// Node properties.
    pub properties: Vec<Property>,
}

/// Batch edge definition for creation.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchEdgeDef {
    /// Source node ID.
    pub from_id: Expr,
    /// Target node ID.
    pub to_id: Expr,
    /// Edge type label.
    pub edge_type: Ident,
    /// Edge properties.
    pub properties: Vec<Property>,
}

/// Batch node update definition.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchNodeUpdate {
    /// Node ID to update.
    pub id: Expr,
    /// New properties to set.
    pub properties: Vec<Property>,
}

/// Pagination options for paginated queries.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PaginationOpts {
    /// Number of results to skip.
    pub skip: Option<Expr>,
    /// Maximum number of results.
    pub limit: Option<Expr>,
    /// Whether to include total count.
    pub count_total: bool,
}

// =============================================================================
// Expressions
// =============================================================================

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    /// The expression kind.
    pub kind: ExprKind,
    /// Source location.
    pub span: Span,
}

impl Expr {
    /// Creates a new expression.
    #[must_use]
    pub const fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Creates a boxed expression.
    #[must_use]
    pub fn boxed(kind: ExprKind, span: Span) -> Box<Self> {
        Box::new(Self::new(kind, span))
    }
}

/// Expression variants.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// Literal value
    Literal(Literal),
    /// Identifier (column name, variable)
    Ident(Ident),
    /// Qualified name (table.column)
    Qualified(Box<Expr>, Ident),
    /// Binary operation
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    /// Unary operation
    Unary(UnaryOp, Box<Expr>),
    /// Function call
    Call(FunctionCall),
    /// CASE expression
    Case(CaseExpr),
    /// Subquery
    Subquery(Box<SelectStmt>),
    /// EXISTS subquery
    Exists(Box<SelectStmt>),
    /// IN / NOT IN expression
    In {
        /// Expression to test.
        expr: Box<Expr>,
        /// List of values or subquery.
        list: InList,
        /// Whether this is NOT IN.
        negated: bool,
    },
    /// BETWEEN expression
    Between {
        /// Expression to test.
        expr: Box<Expr>,
        /// Lower bound.
        low: Box<Expr>,
        /// Upper bound.
        high: Box<Expr>,
        /// Whether this is NOT BETWEEN.
        negated: bool,
    },
    /// LIKE expression
    Like {
        /// Expression to test.
        expr: Box<Expr>,
        /// LIKE pattern.
        pattern: Box<Expr>,
        /// Whether this is NOT LIKE.
        negated: bool,
    },
    /// IS NULL / IS NOT NULL
    IsNull {
        /// Expression to test.
        expr: Box<Expr>,
        /// Whether this is IS NOT NULL.
        negated: bool,
    },
    /// Array literal
    Array(Vec<Expr>),
    /// Tuple/row literal
    Tuple(Vec<Expr>),
    /// Cast expression
    Cast(Box<Expr>, DataType),
    /// Wildcard (*)
    Wildcard,
    /// Qualified wildcard (table.*)
    QualifiedWildcard(Ident),
}

/// Literal values.
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    /// SQL `NULL` literal.
    Null,
    /// Boolean literal (`TRUE` or `FALSE`).
    Boolean(bool),
    /// Integer literal.
    Integer(i64),
    /// Floating-point literal.
    Float(f64),
    /// String literal.
    String(String),
}

/// An identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    /// The identifier text.
    pub name: String,
    /// Source location.
    pub span: Span,
}

impl Ident {
    /// Creates a new identifier.
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    /// Creates an identifier with a dummy span.
    pub fn unspanned(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            span: Span::dummy(),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Modulo (`%`).
    Mod,
    /// Equality (`=`).
    Eq,
    /// Inequality (`!=`).
    Ne,
    /// Less than (`<`).
    Lt,
    /// Less than or equal (`<=`).
    Le,
    /// Greater than (`>`).
    Gt,
    /// Greater than or equal (`>=`).
    Ge,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// String concatenation (`||`).
    Concat,
    /// Bitwise AND (`&`).
    BitAnd,
    /// Bitwise OR (`|`).
    BitOr,
    /// Bitwise XOR (`^`).
    BitXor,
    /// Left shift (`<<`).
    Shl,
    /// Right shift (`>>`).
    Shr,
}

impl BinaryOp {
    /// Returns the precedence of this operator (higher = binds tighter).
    #[must_use]
    pub const fn precedence(self) -> u8 {
        match self {
            Self::Or => 1,
            Self::And => 2,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge => 3,
            Self::BitOr => 4,
            Self::BitXor => 5,
            Self::BitAnd => 6,
            Self::Shl | Self::Shr => 7,
            Self::Add | Self::Sub | Self::Concat => 8,
            Self::Mul | Self::Div | Self::Mod => 9,
        }
    }

    /// Returns true if this operator is left-associative.
    #[must_use]
    pub const fn is_left_assoc(self) -> bool {
        true
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Concat => "||",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
        };
        write!(f, "{s}")
    }
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// Logical NOT.
    Not,
    /// Arithmetic negation (`-`).
    Neg,
    /// Bitwise NOT (`~`).
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Not => "NOT",
            Self::Neg => "-",
            Self::BitNot => "~",
        };
        write!(f, "{s}")
    }
}

/// Function call.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall {
    /// Function name.
    pub name: Ident,
    /// Function arguments.
    pub args: Vec<Expr>,
    /// Whether DISTINCT was specified.
    pub distinct: bool,
}

/// CASE expression.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseExpr {
    /// Optional CASE operand for simple CASE.
    pub operand: Option<Box<Expr>>,
    /// WHEN...THEN clauses.
    pub when_clauses: Vec<WhenClause>,
    /// Optional ELSE clause.
    pub else_clause: Option<Box<Expr>>,
}

/// WHEN clause in CASE expression.
#[derive(Clone, Debug, PartialEq)]
pub struct WhenClause {
    /// The WHEN condition.
    pub condition: Expr,
    /// The THEN result expression.
    pub result: Expr,
}

/// IN list (values or subquery).
#[derive(Clone, Debug, PartialEq)]
pub enum InList {
    /// List of value expressions.
    Values(Vec<Expr>),
    /// Subquery returning a set.
    Subquery(Box<SelectStmt>),
}

// =============================================================================
// Display implementations
// =============================================================================

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "INT"),
            Self::Integer => write!(f, "INTEGER"),
            Self::Bigint => write!(f, "BIGINT"),
            Self::Smallint => write!(f, "SMALLINT"),
            Self::Float => write!(f, "FLOAT"),
            Self::Double => write!(f, "DOUBLE"),
            Self::Real => write!(f, "REAL"),
            Self::Decimal(p, s) => match (p, s) {
                (Some(p), Some(s)) => write!(f, "DECIMAL({p}, {s})"),
                (Some(p), None) => write!(f, "DECIMAL({p})"),
                _ => write!(f, "DECIMAL"),
            },
            Self::Numeric(p, s) => match (p, s) {
                (Some(p), Some(s)) => write!(f, "NUMERIC({p}, {s})"),
                (Some(p), None) => write!(f, "NUMERIC({p})"),
                _ => write!(f, "NUMERIC"),
            },
            Self::Varchar(n) => match n {
                Some(n) => write!(f, "VARCHAR({n})"),
                None => write!(f, "VARCHAR"),
            },
            Self::Char(n) => match n {
                Some(n) => write!(f, "CHAR({n})"),
                None => write!(f, "CHAR"),
            },
            Self::Text => write!(f, "TEXT"),
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Blob => write!(f, "BLOB"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl fmt::Display for JoinKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Inner => "INNER JOIN",
            Self::Left => "LEFT JOIN",
            Self::Right => "RIGHT JOIN",
            Self::Full => "FULL JOIN",
            Self::Cross => "CROSS JOIN",
            Self::Natural => "NATURAL JOIN",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for SortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asc => write!(f, "ASC"),
            Self::Desc => write!(f, "DESC"),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Outgoing => write!(f, "OUTGOING"),
            Self::Incoming => write!(f, "INCOMING"),
            Self::Both => write!(f, "BOTH"),
        }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cosine => write!(f, "COSINE"),
            Self::Euclidean => write!(f, "EUCLIDEAN"),
            Self::DotProduct => write!(f, "DOT_PRODUCT"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Boolean(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::String(s) => {
                let escaped = s.replace('\'', "''");
                write!(f, "'{escaped}'")
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ident() {
        let ident = Ident::new("users", Span::from_offsets(0, 5));
        assert_eq!(ident.name, "users");
        assert_eq!(format!("{}", ident), "users");

        let unspanned = Ident::unspanned("column");
        assert!(unspanned.span.is_dummy());
    }

    #[test]
    fn test_literal_display() {
        assert_eq!(format!("{}", Literal::Null), "NULL");
        assert_eq!(format!("{}", Literal::Boolean(true)), "TRUE");
        assert_eq!(format!("{}", Literal::Boolean(false)), "FALSE");
        assert_eq!(format!("{}", Literal::Integer(42)), "42");
        assert_eq!(format!("{}", Literal::Float(3.15)), "3.15");
        assert_eq!(
            format!("{}", Literal::String("hello".to_string())),
            "'hello'"
        );
        assert_eq!(
            format!("{}", Literal::String("it's".to_string())),
            "'it''s'"
        );
    }

    #[test]
    fn test_binary_op_precedence() {
        assert!(BinaryOp::Mul.precedence() > BinaryOp::Add.precedence());
        assert!(BinaryOp::Add.precedence() > BinaryOp::Eq.precedence());
        assert!(BinaryOp::Eq.precedence() > BinaryOp::And.precedence());
        assert!(BinaryOp::And.precedence() > BinaryOp::Or.precedence());
    }

    #[test]
    fn test_binary_op_display() {
        assert_eq!(format!("{}", BinaryOp::Add), "+");
        assert_eq!(format!("{}", BinaryOp::Sub), "-");
        assert_eq!(format!("{}", BinaryOp::Mul), "*");
        assert_eq!(format!("{}", BinaryOp::Eq), "=");
        assert_eq!(format!("{}", BinaryOp::And), "AND");
        assert_eq!(format!("{}", BinaryOp::Or), "OR");
    }

    #[test]
    fn test_unary_op_display() {
        assert_eq!(format!("{}", UnaryOp::Not), "NOT");
        assert_eq!(format!("{}", UnaryOp::Neg), "-");
        assert_eq!(format!("{}", UnaryOp::BitNot), "~");
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(format!("{}", DataType::Int), "INT");
        assert_eq!(format!("{}", DataType::Varchar(Some(255))), "VARCHAR(255)");
        assert_eq!(format!("{}", DataType::Varchar(None)), "VARCHAR");
        assert_eq!(
            format!("{}", DataType::Decimal(Some(10), Some(2))),
            "DECIMAL(10, 2)"
        );
        assert_eq!(
            format!("{}", DataType::Decimal(Some(10), None)),
            "DECIMAL(10)"
        );
        assert_eq!(format!("{}", DataType::Custom("UUID".to_string())), "UUID");
    }

    #[test]
    fn test_join_kind_display() {
        assert_eq!(format!("{}", JoinKind::Inner), "INNER JOIN");
        assert_eq!(format!("{}", JoinKind::Left), "LEFT JOIN");
        assert_eq!(format!("{}", JoinKind::Right), "RIGHT JOIN");
        assert_eq!(format!("{}", JoinKind::Full), "FULL JOIN");
        assert_eq!(format!("{}", JoinKind::Cross), "CROSS JOIN");
    }

    #[test]
    fn test_direction_display() {
        assert_eq!(format!("{}", Direction::Outgoing), "OUTGOING");
        assert_eq!(format!("{}", Direction::Incoming), "INCOMING");
        assert_eq!(format!("{}", Direction::Both), "BOTH");
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(format!("{}", DistanceMetric::Cosine), "COSINE");
        assert_eq!(format!("{}", DistanceMetric::Euclidean), "EUCLIDEAN");
        assert_eq!(format!("{}", DistanceMetric::DotProduct), "DOT_PRODUCT");
    }

    #[test]
    fn test_sort_direction_default() {
        assert_eq!(SortDirection::default(), SortDirection::Asc);
    }

    #[test]
    fn test_direction_default() {
        assert_eq!(Direction::default(), Direction::Outgoing);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_path_algorithm_default() {
        assert_eq!(PathAlgorithm::default(), PathAlgorithm::Shortest);
    }

    #[test]
    fn test_expr_boxed() {
        let expr = Expr::boxed(
            ExprKind::Literal(Literal::Integer(42)),
            Span::from_offsets(0, 2),
        );
        assert!(matches!(expr.kind, ExprKind::Literal(Literal::Integer(42))));
    }

    #[test]
    fn test_statement_new() {
        let stmt = Statement::new(StatementKind::Empty, Span::from_offsets(0, 1));
        assert!(matches!(stmt.kind, StatementKind::Empty));
    }

    #[test]
    fn test_binary_op_left_assoc() {
        assert!(BinaryOp::Add.is_left_assoc());
        assert!(BinaryOp::Mul.is_left_assoc());
        assert!(BinaryOp::And.is_left_assoc());
    }

    #[test]
    fn test_binary_op_display_comprehensive() {
        assert_eq!(format!("{}", BinaryOp::Div), "/");
        assert_eq!(format!("{}", BinaryOp::Mod), "%");
        assert_eq!(format!("{}", BinaryOp::Ne), "!=");
        assert_eq!(format!("{}", BinaryOp::Lt), "<");
        assert_eq!(format!("{}", BinaryOp::Le), "<=");
        assert_eq!(format!("{}", BinaryOp::Gt), ">");
        assert_eq!(format!("{}", BinaryOp::Ge), ">=");
        assert_eq!(format!("{}", BinaryOp::Concat), "||");
        assert_eq!(format!("{}", BinaryOp::BitAnd), "&");
        assert_eq!(format!("{}", BinaryOp::BitOr), "|");
        assert_eq!(format!("{}", BinaryOp::BitXor), "^");
        assert_eq!(format!("{}", BinaryOp::Shl), "<<");
        assert_eq!(format!("{}", BinaryOp::Shr), ">>");
    }

    #[test]
    fn test_data_type_display_comprehensive() {
        assert_eq!(format!("{}", DataType::Integer), "INTEGER");
        assert_eq!(format!("{}", DataType::Bigint), "BIGINT");
        assert_eq!(format!("{}", DataType::Smallint), "SMALLINT");
        assert_eq!(format!("{}", DataType::Float), "FLOAT");
        assert_eq!(format!("{}", DataType::Double), "DOUBLE");
        assert_eq!(format!("{}", DataType::Real), "REAL");
        assert_eq!(format!("{}", DataType::Text), "TEXT");
        assert_eq!(format!("{}", DataType::Boolean), "BOOLEAN");
        assert_eq!(format!("{}", DataType::Date), "DATE");
        assert_eq!(format!("{}", DataType::Time), "TIME");
        assert_eq!(format!("{}", DataType::Timestamp), "TIMESTAMP");
        assert_eq!(format!("{}", DataType::Blob), "BLOB");
        assert_eq!(format!("{}", DataType::Char(Some(10))), "CHAR(10)");
        assert_eq!(format!("{}", DataType::Char(None)), "CHAR");
        assert_eq!(format!("{}", DataType::Decimal(None, None)), "DECIMAL");
        assert_eq!(
            format!("{}", DataType::Numeric(Some(5), Some(2))),
            "NUMERIC(5, 2)"
        );
        assert_eq!(
            format!("{}", DataType::Numeric(Some(5), None)),
            "NUMERIC(5)"
        );
        assert_eq!(format!("{}", DataType::Numeric(None, None)), "NUMERIC");
    }

    #[test]
    fn test_binary_op_precedence_comprehensive() {
        // Test precedence relationships
        assert!(BinaryOp::Or.precedence() < BinaryOp::And.precedence());
        assert!(BinaryOp::And.precedence() < BinaryOp::Eq.precedence());
        // Eq and Lt have the same precedence (both comparison)
        assert_eq!(BinaryOp::Eq.precedence(), BinaryOp::Lt.precedence());
        assert!(BinaryOp::Lt.precedence() < BinaryOp::Add.precedence());
        assert!(BinaryOp::Add.precedence() < BinaryOp::Mul.precedence());
        assert!(BinaryOp::Concat.precedence() > 0);
    }

    #[test]
    fn test_nulls_order_variants() {
        // Test that NullsOrder enum works
        let first = NullsOrder::First;
        let last = NullsOrder::Last;
        assert_ne!(first, last);
    }

    #[test]
    fn test_column_constraint_variants() {
        let pk = ColumnConstraint::PrimaryKey;
        let nn = ColumnConstraint::NotNull;
        let u = ColumnConstraint::Unique;
        assert_ne!(pk, nn);
        assert_ne!(nn, u);
    }

    #[test]
    fn test_join_kind_display_all() {
        assert!(format!("{}", JoinKind::Cross).contains("CROSS"));
        assert!(format!("{}", JoinKind::Natural).contains("NATURAL"));
        assert!(format!("{}", JoinKind::Full).contains("FULL"));
    }

    #[test]
    fn test_literal_display_all() {
        assert_eq!(format!("{}", Literal::Float(3.15)), "3.15");
        assert_eq!(format!("{}", Literal::Boolean(false)), "FALSE");
    }

    #[test]
    fn test_sort_direction_display() {
        assert_eq!(format!("{}", SortDirection::Asc), "ASC");
        assert_eq!(format!("{}", SortDirection::Desc), "DESC");
    }

    #[test]
    fn test_binary_op_precedence_bitwise() {
        // Bitwise operators have specific precedence
        assert_eq!(BinaryOp::BitOr.precedence(), 4);
        assert_eq!(BinaryOp::BitXor.precedence(), 5);
        assert_eq!(BinaryOp::BitAnd.precedence(), 6);
        assert_eq!(BinaryOp::Shl.precedence(), 7);
        assert_eq!(BinaryOp::Shr.precedence(), 7);
        // Bitwise ops come between comparison and arithmetic
        assert!(BinaryOp::Eq.precedence() < BinaryOp::BitOr.precedence());
        assert!(BinaryOp::BitAnd.precedence() < BinaryOp::Shl.precedence());
        assert!(BinaryOp::Shr.precedence() < BinaryOp::Add.precedence());
    }
}
