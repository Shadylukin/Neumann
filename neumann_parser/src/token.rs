// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Token types for the Neumann query language.
//!
//! Defines all tokens produced by the lexer, including:
//! - SQL keywords (SELECT, FROM, WHERE, etc.)
//! - Graph keywords (NODE, EDGE, NEIGHBORS, PATH)
//! - Vector keywords (EMBED, SIMILAR)
//! - Operators and punctuation
//! - Literals (strings, numbers, identifiers)

#![allow(clippy::enum_glob_use)]
#![allow(clippy::uninlined_format_args)]

use std::fmt;

use crate::span::Span;

/// A token with its span.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    /// Creates a new token.
    #[inline]
    #[must_use]
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Returns true if this is an EOF token.
    #[inline]
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.kind == TokenKind::Eof
    }

    /// Returns true if this token is a keyword.
    #[inline]
    #[must_use]
    pub const fn is_keyword(&self) -> bool {
        self.kind.is_keyword()
    }
}

/// Token kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // === Literals ===
    /// An identifier (table name, column name, etc.)
    Ident(String),
    /// An integer literal
    Integer(i64),
    /// A floating-point literal
    Float(f64),
    /// A string literal (content without quotes)
    String(String),
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// NULL value
    Null,

    // === SQL Keywords ===
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    In,
    Is,
    Like,
    Between,
    Case,
    When,
    Then,
    Else,
    End,
    As,
    On,
    Join,
    Left,
    Right,
    Inner,
    Outer,
    Full,
    Cross,
    Natural,
    Using,
    Group,
    By,
    Having,
    Order,
    Asc,
    Desc,
    Nulls,
    First,
    Last,
    Limit,
    Offset,
    Distinct,
    All,
    Union,
    Intersect,
    Except,
    Exists,
    Cast,
    Any,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,
    Create,
    Table,
    Index,
    Drop,
    Alter,
    Add,
    Column,
    Primary,
    Key,
    Foreign,
    References,
    Unique,
    Check,
    Default,
    Constraint,
    Cascade,
    Restrict,
    If,
    Show,
    Tables,
    Describe,
    Embeddings,

    // === Type Keywords ===
    Int,
    Integer_,
    Bigint,
    Smallint,
    Float_,
    Double,
    Real,
    Decimal,
    Numeric,
    Varchar,
    Char,
    Text,
    Boolean,
    Date,
    Time,
    Timestamp,
    Blob,

    // === Aggregate Functions ===
    Count,
    Sum,
    Avg,
    Min,
    Max,

    // === Graph Keywords ===
    Node,
    Edge,
    Neighbors,
    Path,
    Get,
    List,
    Store,
    Outgoing,
    Incoming,
    Both,
    Shortest,
    Properties,
    Label,
    Vertex,
    Vertices,
    Edges,

    // === Vector Keywords ===
    Embed,
    Similar,
    Vector,
    Embedding,
    Dimension,
    Distance,
    Cosine,
    Euclidean,
    DotProduct,
    Build,
    Batch,

    // === Unified Query Keywords ===
    Find,
    With,
    Return,
    Match,
    Entity,
    Connected,
    Rows,

    // === Vault Keywords ===
    Vault,
    Grant,
    Revoke,
    Rotate,

    // === Cache Keywords ===
    Cache,
    Init,
    Stats,
    Clear,
    Evict,
    Put,
    Semantic,
    Threshold,

    // === Checkpoint Keywords ===
    Checkpoint,
    Checkpoints,
    Rollback,

    // === Chain Keywords ===
    Chain,
    Begin,
    Commit,
    Transaction,
    History,
    Drift,
    Codebook,
    Global,
    Local,
    Analyze,
    Height,
    Transitions,
    Tip,
    Block,

    // === Cluster Keywords ===
    Cluster,
    Connect,
    Disconnect,
    Status,
    Nodes,
    Leader,

    // === Blob Storage Keywords ===
    Blobs,
    Info,
    Link,
    Unlink,
    Links,
    Tag,
    Untag,
    Verify,
    Gc,
    Repair,
    To,
    For,
    Meta,
    Artifacts,

    // === Graph Algorithm Keywords ===
    PageRank,
    Betweenness,
    Closeness,
    Eigenvector,
    Centrality,
    Louvain,
    Communities,
    Propagation,
    Damping,
    Tolerance,
    Iterations,
    Sampling,
    Resolution,
    Passes,

    // === Graph Extended Keywords ===
    Weighted,
    Variable,
    Hops,
    Depth,
    Skip,
    Total,
    Pattern,
    Aggregate,
    Property,
    Type,
    Graph,

    // === Operators ===
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `=`
    Eq,
    /// `!=` or `<>`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `||` (string concatenation)
    Concat,
    /// `&&` (logical and, alternative)
    AmpAmp,
    /// `!`
    Bang,
    /// `~`
    Tilde,
    /// `^`
    Caret,
    /// `&`
    Amp,
    /// `|`
    Pipe,
    /// `<<`
    Shl,
    /// `>>`
    Shr,

    // === Punctuation ===
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `::`
    ColonColon,
    /// `->`
    Arrow,
    /// `=>`
    FatArrow,
    /// `?`
    Question,
    /// `@`
    At,
    /// `#`
    Hash,
    /// `$`
    Dollar,
    /// `_` (placeholder/wildcard)
    Underscore,

    // === Special ===
    /// End of file
    Eof,
    /// Invalid/unknown token
    Error(String),
}

impl TokenKind {
    /// Returns true if this is a keyword token.
    #[must_use]
    #[allow(clippy::too_many_lines)] // Exhaustive keyword list
    pub const fn is_keyword(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Select
                | From
                | Where
                | And
                | Or
                | Not
                | In
                | Is
                | Like
                | Between
                | Case
                | When
                | Then
                | Else
                | End
                | As
                | On
                | Join
                | Left
                | Right
                | Inner
                | Outer
                | Full
                | Cross
                | Natural
                | Using
                | Group
                | By
                | Having
                | Order
                | Asc
                | Desc
                | Nulls
                | First
                | Last
                | Limit
                | Offset
                | Distinct
                | All
                | Union
                | Intersect
                | Except
                | Exists
                | Cast
                | Any
                | Insert
                | Into
                | Values
                | Update
                | Set
                | Delete
                | Create
                | Table
                | Index
                | Drop
                | Alter
                | Add
                | Column
                | Primary
                | Key
                | Foreign
                | References
                | Unique
                | Check
                | Default
                | Constraint
                | Cascade
                | Restrict
                | If
                | Show
                | Tables
                | Describe
                | Embeddings
                | True
                | False
                | Null
                | Int
                | Integer_
                | Bigint
                | Smallint
                | Float_
                | Double
                | Real
                | Decimal
                | Numeric
                | Varchar
                | Char
                | Text
                | Boolean
                | Date
                | Time
                | Timestamp
                | Blob
                | Count
                | Sum
                | Avg
                | Min
                | Max
                | Node
                | Edge
                | Neighbors
                | Path
                | Get
                | List
                | Store
                | Outgoing
                | Incoming
                | Both
                | Shortest
                | Properties
                | Label
                | Vertex
                | Vertices
                | Edges
                | Embed
                | Similar
                | Vector
                | Embedding
                | Dimension
                | Distance
                | Cosine
                | Euclidean
                | DotProduct
                | Build
                | Batch
                | Find
                | With
                | Return
                | Match
                | Entity
                | Connected
                | Rows
                | Vault
                | Grant
                | Revoke
                | Rotate
                | Cache
                | Init
                | Stats
                | Clear
                | Evict
                | Put
                | Semantic
                | Threshold
                | Checkpoint
                | Checkpoints
                | Rollback
                | Chain
                | Begin
                | Commit
                | Transaction
                | History
                | Drift
                | Codebook
                | Global
                | Local
                | Analyze
                | Height
                | Transitions
                | Tip
                | Block
                | Cluster
                | Connect
                | Disconnect
                | Status
                | Nodes
                | Leader
                | Blobs
                | Info
                | Link
                | Unlink
                | Links
                | Tag
                | Untag
                | Verify
                | Gc
                | Repair
                | To
                | For
                | Meta
                | Artifacts
                // Graph algorithm keywords
                | PageRank
                | Betweenness
                | Closeness
                | Eigenvector
                | Centrality
                | Louvain
                | Communities
                | Propagation
                | Damping
                | Tolerance
                | Iterations
                | Sampling
                | Resolution
                | Passes
                // Graph extended keywords
                | Weighted
                | Variable
                | Hops
                | Depth
                | Skip
                | Total
                | Pattern
                | Aggregate
                | Property
                | Type
                | Graph
        )
    }

    /// Returns true if this is a comparison operator.
    #[must_use]
    pub const fn is_comparison(&self) -> bool {
        use TokenKind::*;
        matches!(self, Eq | Ne | Lt | Le | Gt | Ge)
    }

    /// Returns true if this is an arithmetic operator.
    #[must_use]
    pub const fn is_arithmetic(&self) -> bool {
        use TokenKind::*;
        matches!(self, Plus | Minus | Star | Slash | Percent)
    }

    /// Returns true if this is a logical operator.
    #[must_use]
    pub const fn is_logical(&self) -> bool {
        use TokenKind::*;
        matches!(self, And | Or | Not)
    }

    /// Returns true if this is a literal.
    #[must_use]
    pub const fn is_literal(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Integer(_) | Float(_) | String(_) | True | False | Null
        )
    }

    /// Returns true if this keyword can be used as an identifier in expression contexts.
    /// These are domain-specific keywords that don't conflict with SQL syntax.
    #[must_use]
    pub const fn is_contextual_keyword(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            // Cluster/network related
            Status | Nodes | Leader | Connect | Disconnect | Cluster
            // Blob related
            | Blobs | Info | Link | Unlink | Links | Tag | Untag | Verify | Gc | Repair | Meta | Artifacts
            // Chain related
            | Height | Transitions | Tip | Block | Codebook | Global | Local | Drift | Analyze | History
            // Transaction related
            | Begin | Commit | Transaction
            // Graph algorithm related (can be used as identifiers)
            | PageRank | Betweenness | Closeness | Eigenvector | Centrality | Louvain | Communities | Propagation
            | Damping | Tolerance | Iterations | Sampling | Resolution | Passes
            // Graph extended (can be used as identifiers)
            | Weighted | Variable | Hops | Depth | Skip | Total | Pattern | Aggregate | Property | Type | Graph
            // Data types (can be used as identifiers)
            | Int | Float_ | Boolean | Text
        )
    }

    /// Returns the keyword for a string, if it matches.
    #[must_use]
    #[allow(clippy::too_many_lines)] // Exhaustive keyword mapping
    pub fn keyword_from_str(s: &str) -> Option<Self> {
        let upper = s.to_uppercase();
        Some(match upper.as_str() {
            // SQL keywords
            "SELECT" => Self::Select,
            "FROM" => Self::From,
            "WHERE" => Self::Where,
            "AND" => Self::And,
            "OR" => Self::Or,
            "NOT" => Self::Not,
            "IN" => Self::In,
            "IS" => Self::Is,
            "LIKE" => Self::Like,
            "BETWEEN" => Self::Between,
            "CASE" => Self::Case,
            "WHEN" => Self::When,
            "THEN" => Self::Then,
            "ELSE" => Self::Else,
            "END" => Self::End,
            "AS" => Self::As,
            "ON" => Self::On,
            "JOIN" => Self::Join,
            "LEFT" => Self::Left,
            "RIGHT" => Self::Right,
            "INNER" => Self::Inner,
            "OUTER" => Self::Outer,
            "FULL" => Self::Full,
            "CROSS" => Self::Cross,
            "NATURAL" => Self::Natural,
            "USING" => Self::Using,
            "GROUP" => Self::Group,
            "BY" => Self::By,
            "HAVING" => Self::Having,
            "ORDER" => Self::Order,
            "ASC" => Self::Asc,
            "DESC" => Self::Desc,
            "NULLS" => Self::Nulls,
            "FIRST" => Self::First,
            "LAST" => Self::Last,
            "LIMIT" => Self::Limit,
            "OFFSET" => Self::Offset,
            "DISTINCT" => Self::Distinct,
            "ALL" => Self::All,
            "UNION" => Self::Union,
            "INTERSECT" => Self::Intersect,
            "EXCEPT" => Self::Except,
            "EXISTS" => Self::Exists,
            "CAST" => Self::Cast,
            "ANY" => Self::Any,
            "INSERT" => Self::Insert,
            "INTO" => Self::Into,
            "VALUES" => Self::Values,
            "UPDATE" => Self::Update,
            "SET" => Self::Set,
            "DELETE" => Self::Delete,
            "CREATE" => Self::Create,
            "TABLE" => Self::Table,
            "INDEX" => Self::Index,
            "DROP" => Self::Drop,
            "ALTER" => Self::Alter,
            "ADD" => Self::Add,
            "COLUMN" => Self::Column,
            "PRIMARY" => Self::Primary,
            "KEY" => Self::Key,
            "FOREIGN" => Self::Foreign,
            "REFERENCES" => Self::References,
            "UNIQUE" => Self::Unique,
            "CHECK" => Self::Check,
            "DEFAULT" => Self::Default,
            "CONSTRAINT" => Self::Constraint,
            "CASCADE" => Self::Cascade,
            "RESTRICT" => Self::Restrict,
            "IF" => Self::If,
            "SHOW" => Self::Show,
            "TABLES" => Self::Tables,
            "DESCRIBE" => Self::Describe,
            "EMBEDDINGS" => Self::Embeddings,
            "TRUE" => Self::True,
            "FALSE" => Self::False,
            "NULL" => Self::Null,

            // Type keywords
            "INT" => Self::Int,
            "INTEGER" => Self::Integer_,
            "BIGINT" => Self::Bigint,
            "SMALLINT" => Self::Smallint,
            "FLOAT" => Self::Float_,
            "DOUBLE" => Self::Double,
            "REAL" => Self::Real,
            "DECIMAL" => Self::Decimal,
            "NUMERIC" => Self::Numeric,
            "VARCHAR" => Self::Varchar,
            "CHAR" => Self::Char,
            "TEXT" => Self::Text,
            "BOOLEAN" => Self::Boolean,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "TIMESTAMP" => Self::Timestamp,
            "BLOB" => Self::Blob,

            // Aggregates
            "COUNT" => Self::Count,
            "SUM" => Self::Sum,
            "AVG" => Self::Avg,
            "MIN" => Self::Min,
            "MAX" => Self::Max,

            // Graph keywords
            "NODE" => Self::Node,
            "EDGE" => Self::Edge,
            "NEIGHBORS" => Self::Neighbors,
            "PATH" => Self::Path,
            "GET" => Self::Get,
            "LIST" => Self::List,
            "STORE" => Self::Store,
            "OUTGOING" => Self::Outgoing,
            "INCOMING" => Self::Incoming,
            "BOTH" => Self::Both,
            "SHORTEST" => Self::Shortest,
            "PROPERTIES" => Self::Properties,
            "LABEL" => Self::Label,
            "VERTEX" => Self::Vertex,
            "VERTICES" => Self::Vertices,
            "EDGES" => Self::Edges,

            // Vector keywords
            "EMBED" => Self::Embed,
            "SIMILAR" => Self::Similar,
            "VECTOR" => Self::Vector,
            "EMBEDDING" => Self::Embedding,
            "DIMENSION" => Self::Dimension,
            "DISTANCE" => Self::Distance,
            "COSINE" => Self::Cosine,
            "EUCLIDEAN" => Self::Euclidean,
            "DOT_PRODUCT" | "DOTPRODUCT" => Self::DotProduct,
            "BUILD" => Self::Build,
            "BATCH" => Self::Batch,

            // Unified keywords
            "FIND" => Self::Find,
            "WITH" => Self::With,
            "RETURN" => Self::Return,
            "MATCH" => Self::Match,
            "ENTITY" => Self::Entity,
            "CONNECTED" => Self::Connected,
            "ROWS" => Self::Rows,

            // Vault keywords
            "VAULT" => Self::Vault,
            "GRANT" => Self::Grant,
            "REVOKE" => Self::Revoke,
            "ROTATE" => Self::Rotate,

            // Cache keywords
            "CACHE" => Self::Cache,
            "INIT" => Self::Init,
            "STATS" => Self::Stats,
            "CLEAR" => Self::Clear,
            "EVICT" => Self::Evict,
            "PUT" => Self::Put,
            "SEMANTIC" => Self::Semantic,
            "THRESHOLD" => Self::Threshold,

            // Checkpoint keywords
            "CHECKPOINT" => Self::Checkpoint,
            "CHECKPOINTS" => Self::Checkpoints,
            "ROLLBACK" => Self::Rollback,

            // Chain keywords
            "CHAIN" => Self::Chain,
            "BEGIN" => Self::Begin,
            "COMMIT" => Self::Commit,
            "TRANSACTION" => Self::Transaction,
            "HISTORY" => Self::History,
            "DRIFT" => Self::Drift,
            "CODEBOOK" => Self::Codebook,
            "GLOBAL" => Self::Global,
            "LOCAL" => Self::Local,
            "ANALYZE" => Self::Analyze,
            "HEIGHT" => Self::Height,
            "TRANSITIONS" => Self::Transitions,
            "TIP" => Self::Tip,
            "BLOCK" => Self::Block,

            // Cluster keywords
            "CLUSTER" => Self::Cluster,
            "CONNECT" => Self::Connect,
            "DISCONNECT" => Self::Disconnect,
            "STATUS" => Self::Status,
            "NODES" => Self::Nodes,
            "LEADER" => Self::Leader,

            // Blob storage keywords
            "BLOBS" => Self::Blobs,
            "INFO" => Self::Info,
            "LINK" => Self::Link,
            "UNLINK" => Self::Unlink,
            "LINKS" => Self::Links,
            "TAG" => Self::Tag,
            "UNTAG" => Self::Untag,
            "VERIFY" => Self::Verify,
            "GC" => Self::Gc,
            "REPAIR" => Self::Repair,
            "TO" => Self::To,
            "FOR" => Self::For,
            "META" => Self::Meta,
            "ARTIFACTS" => Self::Artifacts,

            // Graph algorithm keywords
            "PAGERANK" => Self::PageRank,
            "BETWEENNESS" => Self::Betweenness,
            "CLOSENESS" => Self::Closeness,
            "EIGENVECTOR" => Self::Eigenvector,
            "CENTRALITY" => Self::Centrality,
            "LOUVAIN" => Self::Louvain,
            "COMMUNITIES" => Self::Communities,
            "PROPAGATION" => Self::Propagation,
            "DAMPING" => Self::Damping,
            "TOLERANCE" => Self::Tolerance,
            "ITERATIONS" => Self::Iterations,
            "SAMPLING" => Self::Sampling,
            "RESOLUTION" => Self::Resolution,
            "PASSES" => Self::Passes,

            // Graph extended keywords
            "WEIGHTED" => Self::Weighted,
            "VARIABLE" => Self::Variable,
            "HOPS" => Self::Hops,
            "DEPTH" => Self::Depth,
            "SKIP" => Self::Skip,
            "TOTAL" => Self::Total,
            "PATTERN" => Self::Pattern,
            "AGGREGATE" => Self::Aggregate,
            "PROPERTY" => Self::Property,
            "TYPE" => Self::Type,
            "GRAPH" => Self::Graph,

            _ => return None,
        })
    }

    /// Returns a string representation of the token kind.
    #[must_use]
    #[allow(clippy::too_many_lines)] // Exhaustive token mapping
    pub const fn as_str(&self) -> &'static str {
        use TokenKind::*;
        match self {
            Ident(_) => "identifier",
            Integer(_) => "integer",
            Float(_) => "float",
            String(_) => "string",
            True => "TRUE",
            False => "FALSE",
            Null => "NULL",
            Select => "SELECT",
            From => "FROM",
            Where => "WHERE",
            And => "AND",
            Or => "OR",
            Not => "NOT",
            In => "IN",
            Is => "IS",
            Like => "LIKE",
            Between => "BETWEEN",
            Case => "CASE",
            When => "WHEN",
            Then => "THEN",
            Else => "ELSE",
            End => "END",
            As => "AS",
            On => "ON",
            Join => "JOIN",
            Left => "LEFT",
            Right => "RIGHT",
            Inner => "INNER",
            Outer => "OUTER",
            Full => "FULL",
            Cross => "CROSS",
            Natural => "NATURAL",
            Using => "USING",
            Group => "GROUP",
            By => "BY",
            Having => "HAVING",
            Order => "ORDER",
            Asc => "ASC",
            Desc => "DESC",
            Nulls => "NULLS",
            First => "FIRST",
            Last => "LAST",
            Limit => "LIMIT",
            Offset => "OFFSET",
            Distinct => "DISTINCT",
            All => "ALL",
            Union => "UNION",
            Intersect => "INTERSECT",
            Except => "EXCEPT",
            Exists => "EXISTS",
            Cast => "CAST",
            Any => "ANY",
            Insert => "INSERT",
            Into => "INTO",
            Values => "VALUES",
            Update => "UPDATE",
            Set => "SET",
            Delete => "DELETE",
            Create => "CREATE",
            Table => "TABLE",
            Index => "INDEX",
            Drop => "DROP",
            Alter => "ALTER",
            Add => "ADD",
            Column => "COLUMN",
            Primary => "PRIMARY",
            Key => "KEY",
            Foreign => "FOREIGN",
            References => "REFERENCES",
            Unique => "UNIQUE",
            Check => "CHECK",
            Default => "DEFAULT",
            Constraint => "CONSTRAINT",
            Cascade => "CASCADE",
            Restrict => "RESTRICT",
            If => "IF",
            Show => "SHOW",
            Tables => "TABLES",
            Describe => "DESCRIBE",
            Embeddings => "EMBEDDINGS",
            Int => "INT",
            Integer_ => "INTEGER",
            Bigint => "BIGINT",
            Smallint => "SMALLINT",
            Float_ => "FLOAT",
            Double => "DOUBLE",
            Real => "REAL",
            Decimal => "DECIMAL",
            Numeric => "NUMERIC",
            Varchar => "VARCHAR",
            Char => "CHAR",
            Text => "TEXT",
            Boolean => "BOOLEAN",
            Date => "DATE",
            Time => "TIME",
            Timestamp => "TIMESTAMP",
            Blob => "BLOB",
            Count => "COUNT",
            Sum => "SUM",
            Avg => "AVG",
            Min => "MIN",
            Max => "MAX",
            Node => "NODE",
            Edge => "EDGE",
            Neighbors => "NEIGHBORS",
            Path => "PATH",
            Get => "GET",
            List => "LIST",
            Store => "STORE",
            Outgoing => "OUTGOING",
            Incoming => "INCOMING",
            Both => "BOTH",
            Shortest => "SHORTEST",
            Properties => "PROPERTIES",
            Label => "LABEL",
            Vertex => "VERTEX",
            Vertices => "VERTICES",
            Edges => "EDGES",
            Embed => "EMBED",
            Similar => "SIMILAR",
            Vector => "VECTOR",
            Embedding => "EMBEDDING",
            Dimension => "DIMENSION",
            Distance => "DISTANCE",
            Cosine => "COSINE",
            Euclidean => "EUCLIDEAN",
            DotProduct => "DOT_PRODUCT",
            Build => "BUILD",
            Batch => "BATCH",
            Find => "FIND",
            With => "WITH",
            Return => "RETURN",
            Match => "MATCH",
            Entity => "ENTITY",
            Connected => "CONNECTED",
            Rows => "ROWS",
            Vault => "VAULT",
            Grant => "GRANT",
            Revoke => "REVOKE",
            Rotate => "ROTATE",
            Cache => "CACHE",
            Init => "INIT",
            Stats => "STATS",
            Clear => "CLEAR",
            Evict => "EVICT",
            Put => "PUT",
            Semantic => "SEMANTIC",
            Threshold => "THRESHOLD",
            Checkpoint => "CHECKPOINT",
            Checkpoints => "CHECKPOINTS",
            Rollback => "ROLLBACK",
            Chain => "CHAIN",
            Begin => "BEGIN",
            Commit => "COMMIT",
            Transaction => "TRANSACTION",
            History => "HISTORY",
            Drift => "DRIFT",
            Codebook => "CODEBOOK",
            Global => "GLOBAL",
            Local => "LOCAL",
            Analyze => "ANALYZE",
            Height => "HEIGHT",
            Transitions => "TRANSITIONS",
            Tip => "TIP",
            Block => "BLOCK",
            Cluster => "CLUSTER",
            Connect => "CONNECT",
            Disconnect => "DISCONNECT",
            Status => "STATUS",
            Nodes => "NODES",
            Leader => "LEADER",
            Blobs => "BLOBS",
            Info => "INFO",
            Link => "LINK",
            Unlink => "UNLINK",
            Links => "LINKS",
            Tag => "TAG",
            Untag => "UNTAG",
            Verify => "VERIFY",
            Gc => "GC",
            Repair => "REPAIR",
            To => "TO",
            For => "FOR",
            Meta => "META",
            Artifacts => "ARTIFACTS",
            // Graph algorithm keywords
            PageRank => "PAGERANK",
            Betweenness => "BETWEENNESS",
            Closeness => "CLOSENESS",
            Eigenvector => "EIGENVECTOR",
            Centrality => "CENTRALITY",
            Louvain => "LOUVAIN",
            Communities => "COMMUNITIES",
            Propagation => "PROPAGATION",
            Damping => "DAMPING",
            Tolerance => "TOLERANCE",
            Iterations => "ITERATIONS",
            Sampling => "SAMPLING",
            Resolution => "RESOLUTION",
            Passes => "PASSES",
            // Graph extended keywords
            Weighted => "WEIGHTED",
            Variable => "VARIABLE",
            Hops => "HOPS",
            Depth => "DEPTH",
            Skip => "SKIP",
            Total => "TOTAL",
            Pattern => "PATTERN",
            Aggregate => "AGGREGATE",
            Property => "PROPERTY",
            Type => "TYPE",
            Graph => "GRAPH",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Eq => "=",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            Concat => "||",
            AmpAmp => "&&",
            Bang => "!",
            Tilde => "~",
            Caret => "^",
            Amp => "&",
            Pipe => "|",
            Shl => "<<",
            Shr => ">>",
            LParen => "(",
            RParen => ")",
            LBracket => "[",
            RBracket => "]",
            LBrace => "{",
            RBrace => "}",
            Comma => ",",
            Dot => ".",
            Semicolon => ";",
            Colon => ":",
            ColonColon => "::",
            Arrow => "->",
            FatArrow => "=>",
            Question => "?",
            At => "@",
            Hash => "#",
            Dollar => "$",
            Underscore => "_",
            Eof => "EOF",
            Error(_) => "error",
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident(s) => write!(f, "{}", s),
            Self::Integer(n) => write!(f, "{}", n),
            Self::Float(n) => write!(f, "{}", n),
            Self::String(s) => write!(f, "'{}'", s),
            Self::Error(e) => write!(f, "error: {}", e),
            _ => write!(f, "{}", self.as_str()),
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.kind, self.span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::BytePos;

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenKind::Select, Span::from_offsets(0, 6));
        assert_eq!(token.kind, TokenKind::Select);
        assert_eq!(token.span.len(), 6);
        assert!(!token.is_eof());
        assert!(token.is_keyword());
    }

    #[test]
    fn test_token_eof() {
        let token = Token::new(TokenKind::Eof, Span::point(BytePos(100)));
        assert!(token.is_eof());
        assert!(!token.is_keyword());
    }

    #[test]
    fn test_keyword_from_str() {
        assert_eq!(
            TokenKind::keyword_from_str("select"),
            Some(TokenKind::Select)
        );
        assert_eq!(
            TokenKind::keyword_from_str("SELECT"),
            Some(TokenKind::Select)
        );
        assert_eq!(
            TokenKind::keyword_from_str("SeLeCt"),
            Some(TokenKind::Select)
        );
        assert_eq!(TokenKind::keyword_from_str("from"), Some(TokenKind::From));
        assert_eq!(TokenKind::keyword_from_str("WHERE"), Some(TokenKind::Where));
        assert_eq!(TokenKind::keyword_from_str("NODE"), Some(TokenKind::Node));
        assert_eq!(TokenKind::keyword_from_str("EMBED"), Some(TokenKind::Embed));
        assert_eq!(TokenKind::keyword_from_str("foobar"), None);
    }

    #[test]
    fn test_is_comparison() {
        assert!(TokenKind::Eq.is_comparison());
        assert!(TokenKind::Ne.is_comparison());
        assert!(TokenKind::Lt.is_comparison());
        assert!(TokenKind::Le.is_comparison());
        assert!(TokenKind::Gt.is_comparison());
        assert!(TokenKind::Ge.is_comparison());
        assert!(!TokenKind::Plus.is_comparison());
        assert!(!TokenKind::And.is_comparison());
    }

    #[test]
    fn test_is_arithmetic() {
        assert!(TokenKind::Plus.is_arithmetic());
        assert!(TokenKind::Minus.is_arithmetic());
        assert!(TokenKind::Star.is_arithmetic());
        assert!(TokenKind::Slash.is_arithmetic());
        assert!(TokenKind::Percent.is_arithmetic());
        assert!(!TokenKind::Eq.is_arithmetic());
        assert!(!TokenKind::And.is_arithmetic());
    }

    #[test]
    fn test_is_logical() {
        assert!(TokenKind::And.is_logical());
        assert!(TokenKind::Or.is_logical());
        assert!(TokenKind::Not.is_logical());
        assert!(!TokenKind::Plus.is_logical());
        assert!(!TokenKind::Eq.is_logical());
    }

    #[test]
    fn test_is_literal() {
        assert!(TokenKind::Integer(42).is_literal());
        assert!(TokenKind::Float(3.15).is_literal());
        assert!(TokenKind::String("hello".to_string()).is_literal());
        assert!(TokenKind::True.is_literal());
        assert!(TokenKind::False.is_literal());
        assert!(TokenKind::Null.is_literal());
        assert!(!TokenKind::Select.is_literal());
        assert!(!TokenKind::Ident("foo".to_string()).is_literal());
    }

    #[test]
    fn test_is_keyword() {
        assert!(TokenKind::Select.is_keyword());
        assert!(TokenKind::From.is_keyword());
        assert!(TokenKind::Node.is_keyword());
        assert!(TokenKind::Embed.is_keyword());
        assert!(TokenKind::True.is_keyword());
        assert!(TokenKind::Null.is_keyword());
        assert!(!TokenKind::Ident("foo".to_string()).is_keyword());
        assert!(!TokenKind::Integer(42).is_keyword());
        assert!(!TokenKind::Plus.is_keyword());
    }

    #[test]
    fn test_as_str() {
        assert_eq!(TokenKind::Select.as_str(), "SELECT");
        assert_eq!(TokenKind::From.as_str(), "FROM");
        assert_eq!(TokenKind::Plus.as_str(), "+");
        assert_eq!(TokenKind::Eq.as_str(), "=");
        assert_eq!(TokenKind::LParen.as_str(), "(");
        assert_eq!(TokenKind::Eof.as_str(), "EOF");
    }

    #[test]
    fn test_as_str_comprehensive() {
        // Literals
        assert_eq!(TokenKind::Ident("x".into()).as_str(), "identifier");
        assert_eq!(TokenKind::Integer(1).as_str(), "integer");
        assert_eq!(TokenKind::Float(1.0).as_str(), "float");
        assert_eq!(TokenKind::String("s".into()).as_str(), "string");
        assert_eq!(TokenKind::True.as_str(), "TRUE");
        assert_eq!(TokenKind::False.as_str(), "FALSE");
        assert_eq!(TokenKind::Null.as_str(), "NULL");

        // SQL Keywords
        assert_eq!(TokenKind::Where.as_str(), "WHERE");
        assert_eq!(TokenKind::And.as_str(), "AND");
        assert_eq!(TokenKind::Or.as_str(), "OR");
        assert_eq!(TokenKind::Not.as_str(), "NOT");
        assert_eq!(TokenKind::In.as_str(), "IN");
        assert_eq!(TokenKind::Is.as_str(), "IS");
        assert_eq!(TokenKind::Like.as_str(), "LIKE");
        assert_eq!(TokenKind::Between.as_str(), "BETWEEN");
        assert_eq!(TokenKind::Case.as_str(), "CASE");
        assert_eq!(TokenKind::When.as_str(), "WHEN");
        assert_eq!(TokenKind::Then.as_str(), "THEN");
        assert_eq!(TokenKind::Else.as_str(), "ELSE");
        assert_eq!(TokenKind::End.as_str(), "END");
        assert_eq!(TokenKind::As.as_str(), "AS");
        assert_eq!(TokenKind::On.as_str(), "ON");
        assert_eq!(TokenKind::Join.as_str(), "JOIN");
        assert_eq!(TokenKind::Left.as_str(), "LEFT");
        assert_eq!(TokenKind::Right.as_str(), "RIGHT");
        assert_eq!(TokenKind::Inner.as_str(), "INNER");
        assert_eq!(TokenKind::Outer.as_str(), "OUTER");
        assert_eq!(TokenKind::Full.as_str(), "FULL");
        assert_eq!(TokenKind::Cross.as_str(), "CROSS");
        assert_eq!(TokenKind::Natural.as_str(), "NATURAL");
        assert_eq!(TokenKind::Using.as_str(), "USING");
        assert_eq!(TokenKind::Group.as_str(), "GROUP");
        assert_eq!(TokenKind::By.as_str(), "BY");
        assert_eq!(TokenKind::Having.as_str(), "HAVING");
        assert_eq!(TokenKind::Order.as_str(), "ORDER");
        assert_eq!(TokenKind::Asc.as_str(), "ASC");
        assert_eq!(TokenKind::Desc.as_str(), "DESC");
        assert_eq!(TokenKind::Nulls.as_str(), "NULLS");
        assert_eq!(TokenKind::First.as_str(), "FIRST");
        assert_eq!(TokenKind::Last.as_str(), "LAST");
        assert_eq!(TokenKind::Limit.as_str(), "LIMIT");
        assert_eq!(TokenKind::Offset.as_str(), "OFFSET");
        assert_eq!(TokenKind::Distinct.as_str(), "DISTINCT");
        assert_eq!(TokenKind::All.as_str(), "ALL");
        assert_eq!(TokenKind::Union.as_str(), "UNION");
        assert_eq!(TokenKind::Intersect.as_str(), "INTERSECT");
        assert_eq!(TokenKind::Except.as_str(), "EXCEPT");
        assert_eq!(TokenKind::Exists.as_str(), "EXISTS");
        assert_eq!(TokenKind::Any.as_str(), "ANY");
        assert_eq!(TokenKind::Insert.as_str(), "INSERT");
        assert_eq!(TokenKind::Into.as_str(), "INTO");
        assert_eq!(TokenKind::Values.as_str(), "VALUES");
        assert_eq!(TokenKind::Update.as_str(), "UPDATE");
        assert_eq!(TokenKind::Set.as_str(), "SET");
        assert_eq!(TokenKind::Delete.as_str(), "DELETE");
        assert_eq!(TokenKind::Create.as_str(), "CREATE");
        assert_eq!(TokenKind::Table.as_str(), "TABLE");
        assert_eq!(TokenKind::Index.as_str(), "INDEX");
        assert_eq!(TokenKind::Drop.as_str(), "DROP");
        assert_eq!(TokenKind::Alter.as_str(), "ALTER");
        assert_eq!(TokenKind::Add.as_str(), "ADD");
        assert_eq!(TokenKind::Column.as_str(), "COLUMN");
        assert_eq!(TokenKind::Primary.as_str(), "PRIMARY");
        assert_eq!(TokenKind::Key.as_str(), "KEY");
        assert_eq!(TokenKind::Foreign.as_str(), "FOREIGN");
        assert_eq!(TokenKind::References.as_str(), "REFERENCES");
        assert_eq!(TokenKind::Unique.as_str(), "UNIQUE");
        assert_eq!(TokenKind::Check.as_str(), "CHECK");
        assert_eq!(TokenKind::Default.as_str(), "DEFAULT");
        assert_eq!(TokenKind::Constraint.as_str(), "CONSTRAINT");
        assert_eq!(TokenKind::Cascade.as_str(), "CASCADE");
        assert_eq!(TokenKind::Restrict.as_str(), "RESTRICT");
        assert_eq!(TokenKind::If.as_str(), "IF");

        // Types
        assert_eq!(TokenKind::Int.as_str(), "INT");
        assert_eq!(TokenKind::Integer_.as_str(), "INTEGER");
        assert_eq!(TokenKind::Bigint.as_str(), "BIGINT");
        assert_eq!(TokenKind::Smallint.as_str(), "SMALLINT");
        assert_eq!(TokenKind::Float_.as_str(), "FLOAT");
        assert_eq!(TokenKind::Double.as_str(), "DOUBLE");
        assert_eq!(TokenKind::Real.as_str(), "REAL");
        assert_eq!(TokenKind::Decimal.as_str(), "DECIMAL");
        assert_eq!(TokenKind::Numeric.as_str(), "NUMERIC");
        assert_eq!(TokenKind::Varchar.as_str(), "VARCHAR");
        assert_eq!(TokenKind::Char.as_str(), "CHAR");
        assert_eq!(TokenKind::Text.as_str(), "TEXT");
        assert_eq!(TokenKind::Boolean.as_str(), "BOOLEAN");
        assert_eq!(TokenKind::Date.as_str(), "DATE");
        assert_eq!(TokenKind::Time.as_str(), "TIME");
        assert_eq!(TokenKind::Timestamp.as_str(), "TIMESTAMP");
        assert_eq!(TokenKind::Blob.as_str(), "BLOB");

        // Aggregates
        assert_eq!(TokenKind::Count.as_str(), "COUNT");
        assert_eq!(TokenKind::Sum.as_str(), "SUM");
        assert_eq!(TokenKind::Avg.as_str(), "AVG");
        assert_eq!(TokenKind::Min.as_str(), "MIN");
        assert_eq!(TokenKind::Max.as_str(), "MAX");

        // Graph
        assert_eq!(TokenKind::Node.as_str(), "NODE");
        assert_eq!(TokenKind::Edge.as_str(), "EDGE");
        assert_eq!(TokenKind::Neighbors.as_str(), "NEIGHBORS");
        assert_eq!(TokenKind::Path.as_str(), "PATH");
        assert_eq!(TokenKind::Outgoing.as_str(), "OUTGOING");
        assert_eq!(TokenKind::Incoming.as_str(), "INCOMING");
        assert_eq!(TokenKind::Both.as_str(), "BOTH");
        assert_eq!(TokenKind::Shortest.as_str(), "SHORTEST");
        assert_eq!(TokenKind::Properties.as_str(), "PROPERTIES");
        assert_eq!(TokenKind::Label.as_str(), "LABEL");
        assert_eq!(TokenKind::Vertex.as_str(), "VERTEX");
        assert_eq!(TokenKind::Vertices.as_str(), "VERTICES");
        assert_eq!(TokenKind::Edges.as_str(), "EDGES");

        // Vector
        assert_eq!(TokenKind::Embed.as_str(), "EMBED");
        assert_eq!(TokenKind::Similar.as_str(), "SIMILAR");
        assert_eq!(TokenKind::Vector.as_str(), "VECTOR");
        assert_eq!(TokenKind::Embedding.as_str(), "EMBEDDING");
        assert_eq!(TokenKind::Dimension.as_str(), "DIMENSION");
        assert_eq!(TokenKind::Distance.as_str(), "DISTANCE");
        assert_eq!(TokenKind::Cosine.as_str(), "COSINE");
        assert_eq!(TokenKind::Euclidean.as_str(), "EUCLIDEAN");
        assert_eq!(TokenKind::DotProduct.as_str(), "DOT_PRODUCT");

        // Unified
        assert_eq!(TokenKind::Find.as_str(), "FIND");
        assert_eq!(TokenKind::With.as_str(), "WITH");
        assert_eq!(TokenKind::Return.as_str(), "RETURN");
        assert_eq!(TokenKind::Match.as_str(), "MATCH");

        // Cache
        assert_eq!(TokenKind::Cache.as_str(), "CACHE");
        assert_eq!(TokenKind::Init.as_str(), "INIT");
        assert_eq!(TokenKind::Stats.as_str(), "STATS");
        assert_eq!(TokenKind::Clear.as_str(), "CLEAR");
        assert_eq!(TokenKind::Evict.as_str(), "EVICT");
        assert_eq!(TokenKind::Put.as_str(), "PUT");

        // Operators
        assert_eq!(TokenKind::Minus.as_str(), "-");
        assert_eq!(TokenKind::Star.as_str(), "*");
        assert_eq!(TokenKind::Slash.as_str(), "/");
        assert_eq!(TokenKind::Percent.as_str(), "%");
        assert_eq!(TokenKind::Ne.as_str(), "!=");
        assert_eq!(TokenKind::Lt.as_str(), "<");
        assert_eq!(TokenKind::Le.as_str(), "<=");
        assert_eq!(TokenKind::Gt.as_str(), ">");
        assert_eq!(TokenKind::Ge.as_str(), ">=");
        assert_eq!(TokenKind::Concat.as_str(), "||");
        assert_eq!(TokenKind::AmpAmp.as_str(), "&&");
        assert_eq!(TokenKind::Bang.as_str(), "!");
        assert_eq!(TokenKind::Tilde.as_str(), "~");
        assert_eq!(TokenKind::Caret.as_str(), "^");
        assert_eq!(TokenKind::Amp.as_str(), "&");
        assert_eq!(TokenKind::Pipe.as_str(), "|");
        assert_eq!(TokenKind::Shl.as_str(), "<<");
        assert_eq!(TokenKind::Shr.as_str(), ">>");

        // Punctuation
        assert_eq!(TokenKind::RParen.as_str(), ")");
        assert_eq!(TokenKind::LBracket.as_str(), "[");
        assert_eq!(TokenKind::RBracket.as_str(), "]");
        assert_eq!(TokenKind::LBrace.as_str(), "{");
        assert_eq!(TokenKind::RBrace.as_str(), "}");
        assert_eq!(TokenKind::Comma.as_str(), ",");
        assert_eq!(TokenKind::Dot.as_str(), ".");
        assert_eq!(TokenKind::Semicolon.as_str(), ";");
        assert_eq!(TokenKind::Colon.as_str(), ":");
        assert_eq!(TokenKind::ColonColon.as_str(), "::");
        assert_eq!(TokenKind::Arrow.as_str(), "->");
        assert_eq!(TokenKind::FatArrow.as_str(), "=>");
        assert_eq!(TokenKind::Question.as_str(), "?");
        assert_eq!(TokenKind::At.as_str(), "@");
        assert_eq!(TokenKind::Hash.as_str(), "#");
        assert_eq!(TokenKind::Dollar.as_str(), "$");
        assert_eq!(TokenKind::Underscore.as_str(), "_");

        // Special
        assert_eq!(TokenKind::Error("x".into()).as_str(), "error");

        // Additional keywords not covered above
        assert_eq!(TokenKind::Cast.as_str(), "CAST");
        assert_eq!(TokenKind::Show.as_str(), "SHOW");
        assert_eq!(TokenKind::Tables.as_str(), "TABLES");
        assert_eq!(TokenKind::Describe.as_str(), "DESCRIBE");
        assert_eq!(TokenKind::Embeddings.as_str(), "EMBEDDINGS");
        assert_eq!(TokenKind::Build.as_str(), "BUILD");
        assert_eq!(TokenKind::Batch.as_str(), "BATCH");
        assert_eq!(TokenKind::Entity.as_str(), "ENTITY");
        assert_eq!(TokenKind::Connected.as_str(), "CONNECTED");
        assert_eq!(TokenKind::Rows.as_str(), "ROWS");
        assert_eq!(TokenKind::Vault.as_str(), "VAULT");
        assert_eq!(TokenKind::Grant.as_str(), "GRANT");
        assert_eq!(TokenKind::Revoke.as_str(), "REVOKE");
        assert_eq!(TokenKind::Rotate.as_str(), "ROTATE");
        assert_eq!(TokenKind::Semantic.as_str(), "SEMANTIC");
        assert_eq!(TokenKind::Threshold.as_str(), "THRESHOLD");
        assert_eq!(TokenKind::Checkpoint.as_str(), "CHECKPOINT");
        assert_eq!(TokenKind::Checkpoints.as_str(), "CHECKPOINTS");
        assert_eq!(TokenKind::Rollback.as_str(), "ROLLBACK");
        assert_eq!(TokenKind::Chain.as_str(), "CHAIN");
        assert_eq!(TokenKind::Begin.as_str(), "BEGIN");
        assert_eq!(TokenKind::Commit.as_str(), "COMMIT");
        assert_eq!(TokenKind::Transaction.as_str(), "TRANSACTION");
        assert_eq!(TokenKind::History.as_str(), "HISTORY");
        assert_eq!(TokenKind::Drift.as_str(), "DRIFT");
        assert_eq!(TokenKind::Codebook.as_str(), "CODEBOOK");
        assert_eq!(TokenKind::Global.as_str(), "GLOBAL");
        assert_eq!(TokenKind::Local.as_str(), "LOCAL");
        assert_eq!(TokenKind::Analyze.as_str(), "ANALYZE");
        assert_eq!(TokenKind::Height.as_str(), "HEIGHT");
        assert_eq!(TokenKind::Transitions.as_str(), "TRANSITIONS");
        assert_eq!(TokenKind::Tip.as_str(), "TIP");
        assert_eq!(TokenKind::Block.as_str(), "BLOCK");
        assert_eq!(TokenKind::Cluster.as_str(), "CLUSTER");
        assert_eq!(TokenKind::Connect.as_str(), "CONNECT");
        assert_eq!(TokenKind::Disconnect.as_str(), "DISCONNECT");
        assert_eq!(TokenKind::Status.as_str(), "STATUS");
        assert_eq!(TokenKind::Nodes.as_str(), "NODES");
        assert_eq!(TokenKind::Leader.as_str(), "LEADER");
        assert_eq!(TokenKind::Blobs.as_str(), "BLOBS");
        assert_eq!(TokenKind::Info.as_str(), "INFO");
        assert_eq!(TokenKind::Link.as_str(), "LINK");
        assert_eq!(TokenKind::Unlink.as_str(), "UNLINK");
        assert_eq!(TokenKind::Links.as_str(), "LINKS");
        assert_eq!(TokenKind::Tag.as_str(), "TAG");
        assert_eq!(TokenKind::Untag.as_str(), "UNTAG");
        assert_eq!(TokenKind::Verify.as_str(), "VERIFY");
        assert_eq!(TokenKind::Gc.as_str(), "GC");
        assert_eq!(TokenKind::Repair.as_str(), "REPAIR");
        assert_eq!(TokenKind::To.as_str(), "TO");
        assert_eq!(TokenKind::For.as_str(), "FOR");
        assert_eq!(TokenKind::Meta.as_str(), "META");
        assert_eq!(TokenKind::Artifacts.as_str(), "ARTIFACTS");

        // Graph algorithm keywords
        assert_eq!(TokenKind::PageRank.as_str(), "PAGERANK");
        assert_eq!(TokenKind::Betweenness.as_str(), "BETWEENNESS");
        assert_eq!(TokenKind::Closeness.as_str(), "CLOSENESS");
        assert_eq!(TokenKind::Eigenvector.as_str(), "EIGENVECTOR");
        assert_eq!(TokenKind::Centrality.as_str(), "CENTRALITY");
        assert_eq!(TokenKind::Louvain.as_str(), "LOUVAIN");
        assert_eq!(TokenKind::Communities.as_str(), "COMMUNITIES");
        assert_eq!(TokenKind::Propagation.as_str(), "PROPAGATION");
        assert_eq!(TokenKind::Damping.as_str(), "DAMPING");
        assert_eq!(TokenKind::Tolerance.as_str(), "TOLERANCE");
        assert_eq!(TokenKind::Iterations.as_str(), "ITERATIONS");
        assert_eq!(TokenKind::Sampling.as_str(), "SAMPLING");
        assert_eq!(TokenKind::Resolution.as_str(), "RESOLUTION");
        assert_eq!(TokenKind::Passes.as_str(), "PASSES");
        assert_eq!(TokenKind::Property.as_str(), "PROPERTY");
        assert_eq!(TokenKind::Type.as_str(), "TYPE");
        assert_eq!(TokenKind::Aggregate.as_str(), "AGGREGATE");
        assert_eq!(TokenKind::Graph.as_str(), "GRAPH");

        // Additional tokens for full coverage
        assert_eq!(TokenKind::Get.as_str(), "GET");
        assert_eq!(TokenKind::List.as_str(), "LIST");
        assert_eq!(TokenKind::Store.as_str(), "STORE");
        assert_eq!(TokenKind::Weighted.as_str(), "WEIGHTED");
        assert_eq!(TokenKind::Variable.as_str(), "VARIABLE");
        assert_eq!(TokenKind::Hops.as_str(), "HOPS");
        assert_eq!(TokenKind::Depth.as_str(), "DEPTH");
        assert_eq!(TokenKind::Skip.as_str(), "SKIP");
        assert_eq!(TokenKind::Total.as_str(), "TOTAL");
        assert_eq!(TokenKind::Pattern.as_str(), "PATTERN");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TokenKind::Select), "SELECT");
        assert_eq!(format!("{}", TokenKind::Ident("foo".to_string())), "foo");
        assert_eq!(format!("{}", TokenKind::Integer(42)), "42");
        assert_eq!(format!("{}", TokenKind::Float(3.15)), "3.15");
        assert_eq!(
            format!("{}", TokenKind::String("hello".to_string())),
            "'hello'"
        );
        assert_eq!(
            format!("{}", TokenKind::Error("bad".to_string())),
            "error: bad"
        );
    }

    #[test]
    fn test_token_display() {
        let token = Token::new(TokenKind::Select, Span::from_offsets(0, 6));
        assert_eq!(format!("{}", token), "SELECT at 0..6");
    }

    #[test]
    fn test_all_keywords() {
        // Test a sampling of all keyword categories
        let keywords = [
            // SQL
            ("SELECT", TokenKind::Select),
            ("INSERT", TokenKind::Insert),
            ("UPDATE", TokenKind::Update),
            ("DELETE", TokenKind::Delete),
            ("CREATE", TokenKind::Create),
            ("DROP", TokenKind::Drop),
            ("JOIN", TokenKind::Join),
            ("UNION", TokenKind::Union),
            // Types
            ("INT", TokenKind::Int),
            ("VARCHAR", TokenKind::Varchar),
            ("BOOLEAN", TokenKind::Boolean),
            // Aggregates
            ("COUNT", TokenKind::Count),
            ("SUM", TokenKind::Sum),
            ("AVG", TokenKind::Avg),
            // Graph
            ("NODE", TokenKind::Node),
            ("EDGE", TokenKind::Edge),
            ("NEIGHBORS", TokenKind::Neighbors),
            ("PATH", TokenKind::Path),
            // Vector
            ("EMBED", TokenKind::Embed),
            ("SIMILAR", TokenKind::Similar),
            ("COSINE", TokenKind::Cosine),
            // Unified
            ("FIND", TokenKind::Find),
            ("MATCH", TokenKind::Match),
            ("RETURN", TokenKind::Return),
            // Cache
            ("CACHE", TokenKind::Cache),
            ("INIT", TokenKind::Init),
            ("STATS", TokenKind::Stats),
            ("CLEAR", TokenKind::Clear),
            ("EVICT", TokenKind::Evict),
            ("PUT", TokenKind::Put),
        ];

        for (s, expected) in keywords {
            assert_eq!(
                TokenKind::keyword_from_str(s),
                Some(expected),
                "Failed for keyword: {}",
                s
            );
        }
    }

    #[test]
    fn test_dot_product_variants() {
        assert_eq!(
            TokenKind::keyword_from_str("DOT_PRODUCT"),
            Some(TokenKind::DotProduct)
        );
        assert_eq!(
            TokenKind::keyword_from_str("DOTPRODUCT"),
            Some(TokenKind::DotProduct)
        );
    }

    #[test]
    fn test_keyword_from_str_comprehensive() {
        // SQL Keywords
        assert!(TokenKind::keyword_from_str("FROM").is_some());
        assert!(TokenKind::keyword_from_str("WHERE").is_some());
        assert!(TokenKind::keyword_from_str("AND").is_some());
        assert!(TokenKind::keyword_from_str("OR").is_some());
        assert!(TokenKind::keyword_from_str("NOT").is_some());
        assert!(TokenKind::keyword_from_str("IN").is_some());
        assert!(TokenKind::keyword_from_str("IS").is_some());
        assert!(TokenKind::keyword_from_str("LIKE").is_some());
        assert!(TokenKind::keyword_from_str("BETWEEN").is_some());
        assert!(TokenKind::keyword_from_str("CASE").is_some());
        assert!(TokenKind::keyword_from_str("WHEN").is_some());
        assert!(TokenKind::keyword_from_str("THEN").is_some());
        assert!(TokenKind::keyword_from_str("ELSE").is_some());
        assert!(TokenKind::keyword_from_str("END").is_some());
        assert!(TokenKind::keyword_from_str("AS").is_some());
        assert!(TokenKind::keyword_from_str("ON").is_some());
        assert!(TokenKind::keyword_from_str("LEFT").is_some());
        assert!(TokenKind::keyword_from_str("RIGHT").is_some());
        assert!(TokenKind::keyword_from_str("INNER").is_some());
        assert!(TokenKind::keyword_from_str("OUTER").is_some());
        assert!(TokenKind::keyword_from_str("FULL").is_some());
        assert!(TokenKind::keyword_from_str("CROSS").is_some());
        assert!(TokenKind::keyword_from_str("NATURAL").is_some());
        assert!(TokenKind::keyword_from_str("USING").is_some());
        assert!(TokenKind::keyword_from_str("GROUP").is_some());
        assert!(TokenKind::keyword_from_str("BY").is_some());
        assert!(TokenKind::keyword_from_str("HAVING").is_some());
        assert!(TokenKind::keyword_from_str("ORDER").is_some());
        assert!(TokenKind::keyword_from_str("ASC").is_some());
        assert!(TokenKind::keyword_from_str("DESC").is_some());
        assert!(TokenKind::keyword_from_str("NULLS").is_some());
        assert!(TokenKind::keyword_from_str("FIRST").is_some());
        assert!(TokenKind::keyword_from_str("LAST").is_some());
        assert!(TokenKind::keyword_from_str("LIMIT").is_some());
        assert!(TokenKind::keyword_from_str("OFFSET").is_some());
        assert!(TokenKind::keyword_from_str("DISTINCT").is_some());
        assert!(TokenKind::keyword_from_str("ALL").is_some());
        assert!(TokenKind::keyword_from_str("INTERSECT").is_some());
        assert!(TokenKind::keyword_from_str("EXCEPT").is_some());
        assert!(TokenKind::keyword_from_str("EXISTS").is_some());
        assert!(TokenKind::keyword_from_str("CAST").is_some());
        assert!(TokenKind::keyword_from_str("ANY").is_some());
        assert!(TokenKind::keyword_from_str("INTO").is_some());
        assert!(TokenKind::keyword_from_str("VALUES").is_some());
        assert!(TokenKind::keyword_from_str("SET").is_some());
        assert!(TokenKind::keyword_from_str("TABLE").is_some());
        assert!(TokenKind::keyword_from_str("INDEX").is_some());
        assert!(TokenKind::keyword_from_str("ALTER").is_some());
        assert!(TokenKind::keyword_from_str("ADD").is_some());
        assert!(TokenKind::keyword_from_str("COLUMN").is_some());
        assert!(TokenKind::keyword_from_str("PRIMARY").is_some());
        assert!(TokenKind::keyword_from_str("KEY").is_some());
        assert!(TokenKind::keyword_from_str("FOREIGN").is_some());
        assert!(TokenKind::keyword_from_str("REFERENCES").is_some());
        assert!(TokenKind::keyword_from_str("UNIQUE").is_some());
        assert!(TokenKind::keyword_from_str("CHECK").is_some());
        assert!(TokenKind::keyword_from_str("DEFAULT").is_some());
        assert!(TokenKind::keyword_from_str("CONSTRAINT").is_some());
        assert!(TokenKind::keyword_from_str("CASCADE").is_some());
        assert!(TokenKind::keyword_from_str("RESTRICT").is_some());
        assert!(TokenKind::keyword_from_str("IF").is_some());
        assert!(TokenKind::keyword_from_str("SHOW").is_some());
        assert!(TokenKind::keyword_from_str("TABLES").is_some());
        assert!(TokenKind::keyword_from_str("DESCRIBE").is_some());
        assert!(TokenKind::keyword_from_str("EMBEDDINGS").is_some());
        assert!(TokenKind::keyword_from_str("TRUE").is_some());
        assert!(TokenKind::keyword_from_str("FALSE").is_some());
        assert!(TokenKind::keyword_from_str("NULL").is_some());
        // Type keywords
        assert!(TokenKind::keyword_from_str("INTEGER").is_some());
        assert!(TokenKind::keyword_from_str("BIGINT").is_some());
        assert!(TokenKind::keyword_from_str("SMALLINT").is_some());
        assert!(TokenKind::keyword_from_str("FLOAT").is_some());
        assert!(TokenKind::keyword_from_str("DOUBLE").is_some());
        assert!(TokenKind::keyword_from_str("REAL").is_some());
        assert!(TokenKind::keyword_from_str("DECIMAL").is_some());
        assert!(TokenKind::keyword_from_str("NUMERIC").is_some());
        assert!(TokenKind::keyword_from_str("CHAR").is_some());
        assert!(TokenKind::keyword_from_str("TEXT").is_some());
        assert!(TokenKind::keyword_from_str("DATE").is_some());
        assert!(TokenKind::keyword_from_str("TIME").is_some());
        assert!(TokenKind::keyword_from_str("TIMESTAMP").is_some());
        assert!(TokenKind::keyword_from_str("BLOB").is_some());
        // Graph keywords
        assert!(TokenKind::keyword_from_str("GET").is_some());
        assert!(TokenKind::keyword_from_str("LIST").is_some());
        assert!(TokenKind::keyword_from_str("STORE").is_some());
        assert!(TokenKind::keyword_from_str("OUTGOING").is_some());
        assert!(TokenKind::keyword_from_str("INCOMING").is_some());
        assert!(TokenKind::keyword_from_str("BOTH").is_some());
        assert!(TokenKind::keyword_from_str("SHORTEST").is_some());
        assert!(TokenKind::keyword_from_str("PROPERTIES").is_some());
        assert!(TokenKind::keyword_from_str("LABEL").is_some());
        assert!(TokenKind::keyword_from_str("VERTEX").is_some());
        assert!(TokenKind::keyword_from_str("VERTICES").is_some());
        assert!(TokenKind::keyword_from_str("EDGES").is_some());
        // Vector keywords
        assert!(TokenKind::keyword_from_str("VECTOR").is_some());
        assert!(TokenKind::keyword_from_str("EMBEDDING").is_some());
        assert!(TokenKind::keyword_from_str("DIMENSION").is_some());
        assert!(TokenKind::keyword_from_str("DISTANCE").is_some());
        assert!(TokenKind::keyword_from_str("EUCLIDEAN").is_some());
        assert!(TokenKind::keyword_from_str("BUILD").is_some());
        assert!(TokenKind::keyword_from_str("BATCH").is_some());
        // Aggregates
        assert!(TokenKind::keyword_from_str("MIN").is_some());
        assert!(TokenKind::keyword_from_str("MAX").is_some());
        // Unified keywords
        assert!(TokenKind::keyword_from_str("WITH").is_some());
        // Chain keywords
        assert!(TokenKind::keyword_from_str("BEGIN").is_some());
        assert!(TokenKind::keyword_from_str("COMMIT").is_some());
        assert!(TokenKind::keyword_from_str("ROLLBACK").is_some());
        assert!(TokenKind::keyword_from_str("CHAIN").is_some());
        assert!(TokenKind::keyword_from_str("TRANSACTION").is_some());
        assert!(TokenKind::keyword_from_str("HISTORY").is_some());
        assert!(TokenKind::keyword_from_str("DRIFT").is_some());
        assert!(TokenKind::keyword_from_str("HEIGHT").is_some());
        assert!(TokenKind::keyword_from_str("TIP").is_some());
        assert!(TokenKind::keyword_from_str("BLOCK").is_some());
        assert!(TokenKind::keyword_from_str("VERIFY").is_some());
        assert!(TokenKind::keyword_from_str("CODEBOOK").is_some());
        assert!(TokenKind::keyword_from_str("GLOBAL").is_some());
        assert!(TokenKind::keyword_from_str("LOCAL").is_some());
        assert!(TokenKind::keyword_from_str("TRANSITIONS").is_some());
        assert!(TokenKind::keyword_from_str("ANALYZE").is_some());
        // Entity keywords
        assert!(TokenKind::keyword_from_str("ENTITY").is_some());
        assert!(TokenKind::keyword_from_str("CONNECT").is_some());
        assert!(TokenKind::keyword_from_str("CONNECTED").is_some());
        assert!(TokenKind::keyword_from_str("TO").is_some());
        // Blob keywords
        assert!(TokenKind::keyword_from_str("BLOBS").is_some());
        assert!(TokenKind::keyword_from_str("LINK").is_some());
        assert!(TokenKind::keyword_from_str("UNLINK").is_some());
        assert!(TokenKind::keyword_from_str("LINKS").is_some());
        assert!(TokenKind::keyword_from_str("TAG").is_some());
        assert!(TokenKind::keyword_from_str("UNTAG").is_some());
        assert!(TokenKind::keyword_from_str("INFO").is_some());
        assert!(TokenKind::keyword_from_str("GC").is_some());
        assert!(TokenKind::keyword_from_str("REPAIR").is_some());
        assert!(TokenKind::keyword_from_str("META").is_some());
        assert!(TokenKind::keyword_from_str("ARTIFACTS").is_some());
        // Checkpoint keywords
        assert!(TokenKind::keyword_from_str("CHECKPOINT").is_some());
        assert!(TokenKind::keyword_from_str("CHECKPOINTS").is_some());
        // Vault keywords
        assert!(TokenKind::keyword_from_str("VAULT").is_some());
        assert!(TokenKind::keyword_from_str("GRANT").is_some());
        assert!(TokenKind::keyword_from_str("REVOKE").is_some());
        assert!(TokenKind::keyword_from_str("ROTATE").is_some());
        // Cluster keywords
        assert!(TokenKind::keyword_from_str("CLUSTER").is_some());
        assert!(TokenKind::keyword_from_str("DISCONNECT").is_some());
        assert!(TokenKind::keyword_from_str("STATUS").is_some());
        assert!(TokenKind::keyword_from_str("NODES").is_some());
        assert!(TokenKind::keyword_from_str("LEADER").is_some());
        // Graph algorithm keywords
        assert!(TokenKind::keyword_from_str("PAGERANK").is_some());
        assert!(TokenKind::keyword_from_str("LOUVAIN").is_some());
        assert!(TokenKind::keyword_from_str("COMMUNITIES").is_some());
        assert!(TokenKind::keyword_from_str("PROPAGATION").is_some());
        assert!(TokenKind::keyword_from_str("CENTRALITY").is_some());
        assert!(TokenKind::keyword_from_str("BETWEENNESS").is_some());
        assert!(TokenKind::keyword_from_str("CLOSENESS").is_some());
        assert!(TokenKind::keyword_from_str("EIGENVECTOR").is_some());
        assert!(TokenKind::keyword_from_str("DAMPING").is_some());
        assert!(TokenKind::keyword_from_str("ITERATIONS").is_some());
        assert!(TokenKind::keyword_from_str("TOLERANCE").is_some());
        assert!(TokenKind::keyword_from_str("SAMPLING").is_some());
        assert!(TokenKind::keyword_from_str("RESOLUTION").is_some());
        assert!(TokenKind::keyword_from_str("PASSES").is_some());
        assert!(TokenKind::keyword_from_str("WEIGHTED").is_some());
        // Graph extended keywords
        assert!(TokenKind::keyword_from_str("VARIABLE").is_some());
        assert!(TokenKind::keyword_from_str("HOPS").is_some());
        assert!(TokenKind::keyword_from_str("DEPTH").is_some());
        assert!(TokenKind::keyword_from_str("SKIP").is_some());
        assert!(TokenKind::keyword_from_str("TOTAL").is_some());
        assert!(TokenKind::keyword_from_str("PATTERN").is_some());
        assert!(TokenKind::keyword_from_str("AGGREGATE").is_some());
        assert!(TokenKind::keyword_from_str("PROPERTY").is_some());
        assert!(TokenKind::keyword_from_str("TYPE").is_some());
        assert!(TokenKind::keyword_from_str("GRAPH").is_some());
        assert!(TokenKind::keyword_from_str("FOR").is_some());
        // Cache keywords
        assert!(TokenKind::keyword_from_str("SEMANTIC").is_some());
        assert!(TokenKind::keyword_from_str("THRESHOLD").is_some());
        // Other keywords
        assert!(TokenKind::keyword_from_str("ROWS").is_some());
    }
}
