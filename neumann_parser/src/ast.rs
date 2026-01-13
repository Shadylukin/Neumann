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
    pub kind: StatementKind,
    pub span: Span,
}

impl Statement {
    /// Creates a new statement.
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
    ShowEmbeddings { limit: Option<Expr> },
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
    pub expr: Expr,
    pub alias: Option<Ident>,
}

/// FROM clause with optional joins.
#[derive(Clone, Debug, PartialEq)]
pub struct FromClause {
    pub table: TableRef,
    pub joins: Vec<Join>,
}

/// A table reference.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRef {
    pub kind: TableRefKind,
    pub alias: Option<Ident>,
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
    pub kind: JoinKind,
    pub table: TableRef,
    pub condition: Option<JoinCondition>,
    pub span: Span,
}

/// JOIN types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
    Cross,
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
    pub expr: Expr,
    pub direction: SortDirection,
    pub nulls: Option<NullsOrder>,
}

/// Sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SortDirection {
    #[default]
    Asc,
    Desc,
}

/// NULLS FIRST/LAST.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullsOrder {
    First,
    Last,
}

/// INSERT statement.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertStmt {
    pub table: Ident,
    pub columns: Option<Vec<Ident>>,
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
    pub table: Ident,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Box<Expr>>,
}

/// Column assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub column: Ident,
    pub value: Expr,
}

/// DELETE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct DeleteStmt {
    pub table: Ident,
    pub where_clause: Option<Box<Expr>>,
}

/// CREATE TABLE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct CreateTableStmt {
    pub if_not_exists: bool,
    pub table: Ident,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
}

/// Column definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnDef {
    pub name: Ident,
    pub data_type: DataType,
    pub constraints: Vec<ColumnConstraint>,
}

/// Data types.
#[derive(Clone, Debug, PartialEq)]
pub enum DataType {
    Int,
    Integer,
    Bigint,
    Smallint,
    Float,
    Double,
    Real,
    Decimal(Option<u32>, Option<u32>),
    Numeric(Option<u32>, Option<u32>),
    Varchar(Option<u32>),
    Char(Option<u32>),
    Text,
    Boolean,
    Date,
    Time,
    Timestamp,
    Blob,
    /// Custom type name
    Custom(String),
}

/// Column constraints.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnConstraint {
    NotNull,
    Null,
    Unique,
    PrimaryKey,
    Default(Expr),
    Check(Expr),
    References(ForeignKeyRef),
}

/// Foreign key reference.
#[derive(Clone, Debug, PartialEq)]
pub struct ForeignKeyRef {
    pub table: Ident,
    pub column: Option<Ident>,
    pub on_delete: Option<ReferentialAction>,
    pub on_update: Option<ReferentialAction>,
}

/// Referential actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferentialAction {
    Cascade,
    Restrict,
    SetNull,
    SetDefault,
    NoAction,
}

/// Table-level constraints.
#[derive(Clone, Debug, PartialEq)]
pub enum TableConstraint {
    PrimaryKey(Vec<Ident>),
    Unique(Vec<Ident>),
    ForeignKey {
        columns: Vec<Ident>,
        reference: ForeignKeyRef,
    },
    Check(Expr),
}

/// DROP TABLE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct DropTableStmt {
    pub if_exists: bool,
    pub table: Ident,
    pub cascade: bool,
}

/// CREATE INDEX statement.
#[derive(Clone, Debug, PartialEq)]
pub struct CreateIndexStmt {
    pub unique: bool,
    pub if_not_exists: bool,
    pub name: Ident,
    pub table: Ident,
    pub columns: Vec<Ident>,
}

/// DROP INDEX statement.
/// Supports both `DROP INDEX name` and `DROP INDEX ON table(column)` syntax.
#[derive(Clone, Debug, PartialEq)]
pub struct DropIndexStmt {
    pub if_exists: bool,
    /// Index name (for named indexes)
    pub name: Option<Ident>,
    /// Table name (for ON table(column) syntax)
    pub table: Option<Ident>,
    /// Column name (for ON table(column) syntax)
    pub column: Option<Ident>,
}

/// DESCRIBE statement.
#[derive(Clone, Debug, PartialEq)]
pub struct DescribeStmt {
    pub target: DescribeTarget,
}

/// Target of DESCRIBE.
#[derive(Clone, Debug, PartialEq)]
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
    pub operation: NodeOp,
}

/// NODE operations.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeOp {
    /// Create a node: `NODE label { properties }`
    Create {
        label: Ident,
        properties: Vec<Property>,
    },
    /// Get a node: `NODE GET id`
    Get { id: Expr },
    /// Delete a node: `NODE DELETE id`
    Delete { id: Expr },
    /// List nodes: `NODE LIST`
    List { label: Option<Ident> },
}

/// EDGE command.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeStmt {
    pub operation: EdgeOp,
}

/// EDGE operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EdgeOp {
    /// Create an edge: `EDGE from_ref edge_type to_ref { properties }`
    Create {
        from_id: Expr,
        to_id: Expr,
        edge_type: Ident,
        properties: Vec<Property>,
    },
    /// Get an edge: `EDGE GET id`
    Get { id: Expr },
    /// Delete an edge: `EDGE DELETE id`
    Delete { id: Expr },
    /// List edges: `EDGE LIST`
    List { edge_type: Option<Ident> },
}

/// A property key-value pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub key: Ident,
    pub value: Expr,
}

/// NEIGHBORS command.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborsStmt {
    pub node_id: Expr,
    pub direction: Direction,
    pub edge_type: Option<Ident>,
    /// Optional BY SIMILARITY constraint for cross-engine queries
    pub by_similarity: Option<Vec<Expr>>,
    /// Optional LIMIT for BY SIMILARITY queries
    pub limit: Option<Expr>,
}

/// Direction for graph traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Direction {
    #[default]
    Outgoing,
    Incoming,
    Both,
}

/// PATH command.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStmt {
    pub algorithm: PathAlgorithm,
    pub from_id: Expr,
    pub to_id: Expr,
    pub max_depth: Option<Expr>,
}

/// Path-finding algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PathAlgorithm {
    #[default]
    Shortest,
    All,
}

// =============================================================================
// Vector Statements
// =============================================================================

/// EMBED command.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedStmt {
    pub operation: EmbedOp,
}

/// EMBED operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EmbedOp {
    /// Store embedding: `EMBED STORE 'key' [vector]`
    Store { key: Expr, vector: Vec<Expr> },
    /// Get embedding: `EMBED GET 'key'`
    Get { key: Expr },
    /// Delete embedding: `EMBED DELETE 'key'`
    Delete { key: Expr },
    /// Build HNSW index: `EMBED BUILD INDEX`
    BuildIndex,
    /// Batch store embeddings: `EMBED BATCH [('key1', [v1, v2]), ('key2', [v1, v2])]`
    Batch { items: Vec<(Expr, Vec<Expr>)> },
}

/// SIMILAR search.
#[derive(Clone, Debug, PartialEq)]
pub struct SimilarStmt {
    pub query: SimilarQuery,
    pub limit: Option<Expr>,
    pub metric: Option<DistanceMetric>,
    /// Optional CONNECTED TO constraint for cross-engine queries
    pub connected_to: Option<Expr>,
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
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

// =============================================================================
// Unified Statements
// =============================================================================

/// FIND unified query.
#[derive(Clone, Debug, PartialEq)]
pub struct FindStmt {
    pub pattern: FindPattern,
    pub where_clause: Option<Box<Expr>>,
    pub return_items: Vec<SelectItem>,
    pub limit: Option<Box<Expr>>,
}

/// FIND pattern.
#[derive(Clone, Debug, PartialEq)]
pub enum FindPattern {
    /// Match nodes by label
    Nodes { label: Option<Ident> },
    /// Match edges by type
    Edges { edge_type: Option<Ident> },
    /// Match path pattern
    Path {
        from: Option<Ident>,
        edge: Option<Ident>,
        to: Option<Ident>,
    },
}

/// ENTITY command.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityStmt {
    pub operation: EntityOp,
}

/// ENTITY operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EntityOp {
    /// Create an entity: `ENTITY CREATE 'key' { properties } [EMBEDDING [vector]]`
    Create {
        key: Expr,
        properties: Vec<Property>,
        embedding: Option<Vec<Expr>>,
    },
    /// Connect entities: `ENTITY CONNECT 'from' -> 'to' : type`
    Connect {
        from_key: Expr,
        to_key: Expr,
        edge_type: Ident,
    },
}

// =============================================================================
// Vault Statements
// =============================================================================

/// VAULT command.
#[derive(Clone, Debug, PartialEq)]
pub struct VaultStmt {
    pub operation: VaultOp,
}

/// VAULT operations.
#[derive(Clone, Debug, PartialEq)]
pub enum VaultOp {
    /// Set a secret: `VAULT SET 'key' 'value'`
    Set { key: Expr, value: Expr },
    /// Get a secret: `VAULT GET 'key'`
    Get { key: Expr },
    /// Delete a secret: `VAULT DELETE 'key'`
    Delete { key: Expr },
    /// List secrets: `VAULT LIST 'pattern'`
    List { pattern: Option<Expr> },
    /// Rotate a secret: `VAULT ROTATE 'key' 'new_value'`
    Rotate { key: Expr, new_value: Expr },
    /// Grant access: `VAULT GRANT 'entity' ON 'key'`
    Grant { entity: Expr, key: Expr },
    /// Revoke access: `VAULT REVOKE 'entity' ON 'key'`
    Revoke { entity: Expr, key: Expr },
}

// =============================================================================
// Cache Statements
// =============================================================================

/// CACHE command.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheStmt {
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
    Evict { count: Option<Expr> },
    /// Get cached response: `CACHE GET 'key'`
    Get { key: Expr },
    /// Store cache entry: `CACHE PUT 'key' 'value'`
    Put { key: Expr, value: Expr },
    /// Semantic cache lookup: `CACHE SEMANTIC GET 'query' [THRESHOLD n]`
    SemanticGet {
        query: Expr,
        threshold: Option<Expr>,
    },
    /// Semantic cache store: `CACHE SEMANTIC PUT 'query' 'response' EMBEDDING [vector]`
    SemanticPut {
        query: Expr,
        response: Expr,
        embedding: Vec<Expr>,
    },
}

// =============================================================================
// Cluster Statements
// =============================================================================

/// CLUSTER command.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterStmt {
    pub operation: ClusterOp,
}

/// CLUSTER operations.
#[derive(Clone, Debug, PartialEq)]
pub enum ClusterOp {
    /// Connect to cluster: `CLUSTER CONNECT 'address'`
    Connect { addresses: Expr },
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
    pub operation: BlobOp,
}

/// BLOB operations.
#[derive(Clone, Debug, PartialEq)]
pub enum BlobOp {
    /// Initialize blob store: `BLOB INIT`
    Init,
    /// Store blob: `BLOB PUT 'filename' DATA` or `BLOB PUT 'filename' FROM 'path'`
    Put {
        filename: Expr,
        data: Option<Expr>,
        from_path: Option<Expr>,
        options: BlobOptions,
    },
    /// Get blob: `BLOB GET 'artifact_id'` or `BLOB GET 'artifact_id' TO 'path'`
    Get {
        artifact_id: Expr,
        to_path: Option<Expr>,
    },
    /// Delete blob: `BLOB DELETE 'artifact_id'`
    Delete { artifact_id: Expr },
    /// Show blob info: `BLOB INFO 'artifact_id'`
    Info { artifact_id: Expr },
    /// Link blob to entity: `BLOB LINK 'artifact_id' TO entity`
    Link { artifact_id: Expr, entity: Expr },
    /// Unlink blob from entity: `BLOB UNLINK 'artifact_id' FROM entity`
    Unlink { artifact_id: Expr, entity: Expr },
    /// Get links: `BLOB LINKS 'artifact_id'`
    Links { artifact_id: Expr },
    /// Add tag: `BLOB TAG 'artifact_id' 'tag'`
    Tag { artifact_id: Expr, tag: Expr },
    /// Remove tag: `BLOB UNTAG 'artifact_id' 'tag'`
    Untag { artifact_id: Expr, tag: Expr },
    /// Verify integrity: `BLOB VERIFY 'artifact_id'`
    Verify { artifact_id: Expr },
    /// Run garbage collection: `BLOB GC` or `BLOB GC FULL`
    Gc { full: bool },
    /// Repair blob storage: `BLOB REPAIR`
    Repair,
    /// Show blob statistics: `BLOB STATS`
    Stats,
    /// Set metadata: `BLOB META SET 'artifact_id' 'key' 'value'`
    MetaSet {
        artifact_id: Expr,
        key: Expr,
        value: Expr,
    },
    /// Get metadata: `BLOB META GET 'artifact_id' 'key'`
    MetaGet { artifact_id: Expr, key: Expr },
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
    pub operation: BlobsOp,
}

/// BLOBS operations.
#[derive(Clone, Debug, PartialEq)]
pub enum BlobsOp {
    /// List all blobs: `BLOBS`
    List { pattern: Option<Expr> },
    /// Find blobs for entity: `BLOBS FOR entity`
    For { entity: Expr },
    /// Find blobs by tag: `BLOBS BY TAG 'tag'`
    ByTag { tag: Expr },
    /// Find blobs by content type: `BLOBS WHERE TYPE = 'type'`
    ByType { content_type: Expr },
    /// Find similar blobs: `BLOBS SIMILAR TO 'artifact_id' LIMIT n`
    Similar {
        artifact_id: Expr,
        limit: Option<Expr>,
    },
}

// =============================================================================
// Checkpoint Statements
// =============================================================================

/// CHECKPOINT statement: `CHECKPOINT` or `CHECKPOINT 'name'`
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointStmt {
    pub name: Option<Expr>,
}

/// ROLLBACK statement: `ROLLBACK TO 'checkpoint_id'`
#[derive(Clone, Debug, PartialEq)]
pub struct RollbackStmt {
    pub target: Expr,
}

/// CHECKPOINTS statement: `CHECKPOINTS` or `CHECKPOINTS LIMIT n`
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointsStmt {
    pub limit: Option<Expr>,
}

// =============================================================================
// Chain Statements
// =============================================================================

/// CHAIN command.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainStmt {
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
    Rollback { height: Expr },
    /// Get chain history for key: `CHAIN HISTORY 'key'`
    History { key: Expr },
    /// Search chain by similarity: `CHAIN SIMILAR [embedding] LIMIT n`
    Similar {
        embedding: Vec<Expr>,
        limit: Option<Expr>,
    },
    /// Get chain drift metrics: `CHAIN DRIFT FROM height TO height`
    Drift { from_height: Expr, to_height: Expr },
    /// Show global codebook: `SHOW CODEBOOK GLOBAL`
    ShowCodebookGlobal,
    /// Show local codebook: `SHOW CODEBOOK LOCAL 'domain'`
    ShowCodebookLocal { domain: Expr },
    /// Analyze codebook transitions: `ANALYZE CODEBOOK TRANSITIONS`
    AnalyzeTransitions,
    /// Get chain height: `CHAIN HEIGHT`
    Height,
    /// Get chain tip: `CHAIN TIP`
    Tip,
    /// Get block at height: `CHAIN BLOCK height`
    Block { height: Expr },
    /// Verify chain integrity: `CHAIN VERIFY`
    Verify,
}

// =============================================================================
// Expressions
// =============================================================================

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    /// Creates a new expression.
    pub const fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Creates a boxed expression.
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
        expr: Box<Expr>,
        list: InList,
        negated: bool,
    },
    /// BETWEEN expression
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    /// LIKE expression
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    /// IS NULL / IS NOT NULL
    IsNull { expr: Box<Expr>, negated: bool },
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
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

/// An identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
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
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // String
    Concat,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinaryOp {
    /// Returns the precedence of this operator (higher = binds tighter).
    pub const fn precedence(self) -> u8 {
        use BinaryOp::*;
        match self {
            Or => 1,
            And => 2,
            Eq | Ne | Lt | Le | Gt | Ge => 3,
            BitOr => 4,
            BitXor => 5,
            BitAnd => 6,
            Shl | Shr => 7,
            Add | Sub | Concat => 8,
            Mul | Div | Mod => 9,
        }
    }

    /// Returns true if this operator is left-associative.
    pub const fn is_left_assoc(self) -> bool {
        true
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinaryOp::*;
        let s = match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Eq => "=",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "AND",
            Or => "OR",
            Concat => "||",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            Shl => "<<",
            Shr => ">>",
        };
        write!(f, "{}", s)
    }
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnaryOp::Not => "NOT",
            UnaryOp::Neg => "-",
            UnaryOp::BitNot => "~",
        };
        write!(f, "{}", s)
    }
}

/// Function call.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall {
    pub name: Ident,
    pub args: Vec<Expr>,
    pub distinct: bool,
}

/// CASE expression.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseExpr {
    pub operand: Option<Box<Expr>>,
    pub when_clauses: Vec<WhenClause>,
    pub else_clause: Option<Box<Expr>>,
}

/// WHEN clause in CASE expression.
#[derive(Clone, Debug, PartialEq)]
pub struct WhenClause {
    pub condition: Expr,
    pub result: Expr,
}

/// IN list (values or subquery).
#[derive(Clone, Debug, PartialEq)]
pub enum InList {
    Values(Vec<Expr>),
    Subquery(Box<SelectStmt>),
}

// =============================================================================
// Display implementations
// =============================================================================

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Int => write!(f, "INT"),
            DataType::Integer => write!(f, "INTEGER"),
            DataType::Bigint => write!(f, "BIGINT"),
            DataType::Smallint => write!(f, "SMALLINT"),
            DataType::Float => write!(f, "FLOAT"),
            DataType::Double => write!(f, "DOUBLE"),
            DataType::Real => write!(f, "REAL"),
            DataType::Decimal(p, s) => match (p, s) {
                (Some(p), Some(s)) => write!(f, "DECIMAL({}, {})", p, s),
                (Some(p), None) => write!(f, "DECIMAL({})", p),
                _ => write!(f, "DECIMAL"),
            },
            DataType::Numeric(p, s) => match (p, s) {
                (Some(p), Some(s)) => write!(f, "NUMERIC({}, {})", p, s),
                (Some(p), None) => write!(f, "NUMERIC({})", p),
                _ => write!(f, "NUMERIC"),
            },
            DataType::Varchar(n) => match n {
                Some(n) => write!(f, "VARCHAR({})", n),
                None => write!(f, "VARCHAR"),
            },
            DataType::Char(n) => match n {
                Some(n) => write!(f, "CHAR({})", n),
                None => write!(f, "CHAR"),
            },
            DataType::Text => write!(f, "TEXT"),
            DataType::Boolean => write!(f, "BOOLEAN"),
            DataType::Date => write!(f, "DATE"),
            DataType::Time => write!(f, "TIME"),
            DataType::Timestamp => write!(f, "TIMESTAMP"),
            DataType::Blob => write!(f, "BLOB"),
            DataType::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for JoinKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            JoinKind::Inner => "INNER JOIN",
            JoinKind::Left => "LEFT JOIN",
            JoinKind::Right => "RIGHT JOIN",
            JoinKind::Full => "FULL JOIN",
            JoinKind::Cross => "CROSS JOIN",
            JoinKind::Natural => "NATURAL JOIN",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for SortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortDirection::Asc => write!(f, "ASC"),
            SortDirection::Desc => write!(f, "DESC"),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Outgoing => write!(f, "OUTGOING"),
            Direction::Incoming => write!(f, "INCOMING"),
            Direction::Both => write!(f, "BOTH"),
        }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "COSINE"),
            DistanceMetric::Euclidean => write!(f, "EUCLIDEAN"),
            DistanceMetric::DotProduct => write!(f, "DOT_PRODUCT"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => write!(f, "NULL"),
            Literal::Boolean(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Literal::Integer(n) => write!(f, "{}", n),
            Literal::Float(n) => write!(f, "{}", n),
            Literal::String(s) => write!(f, "'{}'", s.replace('\'', "''")),
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
        assert_eq!(format!("{}", Literal::Float(3.14)), "3.14");
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
        assert_eq!(format!("{}", Literal::Float(3.14)), "3.14");
        assert_eq!(format!("{}", Literal::Boolean(false)), "FALSE");
    }
}
