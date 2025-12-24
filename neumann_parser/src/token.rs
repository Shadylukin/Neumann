//! Token types for the Neumann query language.
//!
//! Defines all tokens produced by the lexer, including:
//! - SQL keywords (SELECT, FROM, WHERE, etc.)
//! - Graph keywords (NODE, EDGE, NEIGHBORS, PATH)
//! - Vector keywords (EMBED, SIMILAR)
//! - Operators and punctuation
//! - Literals (strings, numbers, identifiers)

use crate::span::Span;
use std::fmt;

/// A token with its span.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    /// Creates a new token.
    #[inline]
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Returns true if this is an EOF token.
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.kind == TokenKind::Eof
    }

    /// Returns true if this token is a keyword.
    #[inline]
    pub fn is_keyword(&self) -> bool {
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

    // === Unified Query Keywords ===
    Find,
    With,
    Return,
    Match,

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
    pub fn is_keyword(&self) -> bool {
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
                | Find
                | With
                | Return
                | Match
        )
    }

    /// Returns true if this is a comparison operator.
    pub fn is_comparison(&self) -> bool {
        use TokenKind::*;
        matches!(self, Eq | Ne | Lt | Le | Gt | Ge)
    }

    /// Returns true if this is an arithmetic operator.
    pub fn is_arithmetic(&self) -> bool {
        use TokenKind::*;
        matches!(self, Plus | Minus | Star | Slash | Percent)
    }

    /// Returns true if this is a logical operator.
    pub fn is_logical(&self) -> bool {
        use TokenKind::*;
        matches!(self, And | Or | Not)
    }

    /// Returns true if this is a literal.
    pub fn is_literal(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Integer(_) | Float(_) | String(_) | True | False | Null
        )
    }

    /// Returns the keyword for a string, if it matches.
    pub fn keyword_from_str(s: &str) -> Option<TokenKind> {
        let upper = s.to_uppercase();
        Some(match upper.as_str() {
            // SQL keywords
            "SELECT" => TokenKind::Select,
            "FROM" => TokenKind::From,
            "WHERE" => TokenKind::Where,
            "AND" => TokenKind::And,
            "OR" => TokenKind::Or,
            "NOT" => TokenKind::Not,
            "IN" => TokenKind::In,
            "IS" => TokenKind::Is,
            "LIKE" => TokenKind::Like,
            "BETWEEN" => TokenKind::Between,
            "CASE" => TokenKind::Case,
            "WHEN" => TokenKind::When,
            "THEN" => TokenKind::Then,
            "ELSE" => TokenKind::Else,
            "END" => TokenKind::End,
            "AS" => TokenKind::As,
            "ON" => TokenKind::On,
            "JOIN" => TokenKind::Join,
            "LEFT" => TokenKind::Left,
            "RIGHT" => TokenKind::Right,
            "INNER" => TokenKind::Inner,
            "OUTER" => TokenKind::Outer,
            "FULL" => TokenKind::Full,
            "CROSS" => TokenKind::Cross,
            "NATURAL" => TokenKind::Natural,
            "USING" => TokenKind::Using,
            "GROUP" => TokenKind::Group,
            "BY" => TokenKind::By,
            "HAVING" => TokenKind::Having,
            "ORDER" => TokenKind::Order,
            "ASC" => TokenKind::Asc,
            "DESC" => TokenKind::Desc,
            "NULLS" => TokenKind::Nulls,
            "FIRST" => TokenKind::First,
            "LAST" => TokenKind::Last,
            "LIMIT" => TokenKind::Limit,
            "OFFSET" => TokenKind::Offset,
            "DISTINCT" => TokenKind::Distinct,
            "ALL" => TokenKind::All,
            "UNION" => TokenKind::Union,
            "INTERSECT" => TokenKind::Intersect,
            "EXCEPT" => TokenKind::Except,
            "EXISTS" => TokenKind::Exists,
            "CAST" => TokenKind::Cast,
            "ANY" => TokenKind::Any,
            "INSERT" => TokenKind::Insert,
            "INTO" => TokenKind::Into,
            "VALUES" => TokenKind::Values,
            "UPDATE" => TokenKind::Update,
            "SET" => TokenKind::Set,
            "DELETE" => TokenKind::Delete,
            "CREATE" => TokenKind::Create,
            "TABLE" => TokenKind::Table,
            "INDEX" => TokenKind::Index,
            "DROP" => TokenKind::Drop,
            "ALTER" => TokenKind::Alter,
            "ADD" => TokenKind::Add,
            "COLUMN" => TokenKind::Column,
            "PRIMARY" => TokenKind::Primary,
            "KEY" => TokenKind::Key,
            "FOREIGN" => TokenKind::Foreign,
            "REFERENCES" => TokenKind::References,
            "UNIQUE" => TokenKind::Unique,
            "CHECK" => TokenKind::Check,
            "DEFAULT" => TokenKind::Default,
            "CONSTRAINT" => TokenKind::Constraint,
            "CASCADE" => TokenKind::Cascade,
            "RESTRICT" => TokenKind::Restrict,
            "IF" => TokenKind::If,
            "SHOW" => TokenKind::Show,
            "TABLES" => TokenKind::Tables,
            "TRUE" => TokenKind::True,
            "FALSE" => TokenKind::False,
            "NULL" => TokenKind::Null,

            // Type keywords
            "INT" => TokenKind::Int,
            "INTEGER" => TokenKind::Integer_,
            "BIGINT" => TokenKind::Bigint,
            "SMALLINT" => TokenKind::Smallint,
            "FLOAT" => TokenKind::Float_,
            "DOUBLE" => TokenKind::Double,
            "REAL" => TokenKind::Real,
            "DECIMAL" => TokenKind::Decimal,
            "NUMERIC" => TokenKind::Numeric,
            "VARCHAR" => TokenKind::Varchar,
            "CHAR" => TokenKind::Char,
            "TEXT" => TokenKind::Text,
            "BOOLEAN" => TokenKind::Boolean,
            "DATE" => TokenKind::Date,
            "TIME" => TokenKind::Time,
            "TIMESTAMP" => TokenKind::Timestamp,
            "BLOB" => TokenKind::Blob,

            // Aggregates
            "COUNT" => TokenKind::Count,
            "SUM" => TokenKind::Sum,
            "AVG" => TokenKind::Avg,
            "MIN" => TokenKind::Min,
            "MAX" => TokenKind::Max,

            // Graph keywords
            "NODE" => TokenKind::Node,
            "EDGE" => TokenKind::Edge,
            "NEIGHBORS" => TokenKind::Neighbors,
            "PATH" => TokenKind::Path,
            "GET" => TokenKind::Get,
            "LIST" => TokenKind::List,
            "STORE" => TokenKind::Store,
            "OUTGOING" => TokenKind::Outgoing,
            "INCOMING" => TokenKind::Incoming,
            "BOTH" => TokenKind::Both,
            "SHORTEST" => TokenKind::Shortest,
            "PROPERTIES" => TokenKind::Properties,
            "LABEL" => TokenKind::Label,
            "VERTEX" => TokenKind::Vertex,
            "VERTICES" => TokenKind::Vertices,
            "EDGES" => TokenKind::Edges,

            // Vector keywords
            "EMBED" => TokenKind::Embed,
            "SIMILAR" => TokenKind::Similar,
            "VECTOR" => TokenKind::Vector,
            "EMBEDDING" => TokenKind::Embedding,
            "DIMENSION" => TokenKind::Dimension,
            "DISTANCE" => TokenKind::Distance,
            "COSINE" => TokenKind::Cosine,
            "EUCLIDEAN" => TokenKind::Euclidean,
            "DOT_PRODUCT" | "DOTPRODUCT" => TokenKind::DotProduct,

            // Unified keywords
            "FIND" => TokenKind::Find,
            "WITH" => TokenKind::With,
            "RETURN" => TokenKind::Return,
            "MATCH" => TokenKind::Match,

            _ => return None,
        })
    }

    /// Returns a string representation of the token kind.
    pub fn as_str(&self) -> &'static str {
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
            Find => "FIND",
            With => "WITH",
            Return => "RETURN",
            Match => "MATCH",
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
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::Integer(n) => write!(f, "{}", n),
            TokenKind::Float(n) => write!(f, "{}", n),
            TokenKind::String(s) => write!(f, "'{}'", s),
            TokenKind::Error(e) => write!(f, "error: {}", e),
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
        assert!(TokenKind::Float(3.14).is_literal());
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
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TokenKind::Select), "SELECT");
        assert_eq!(format!("{}", TokenKind::Ident("foo".to_string())), "foo");
        assert_eq!(format!("{}", TokenKind::Integer(42)), "42");
        assert_eq!(format!("{}", TokenKind::Float(3.14)), "3.14");
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
}
