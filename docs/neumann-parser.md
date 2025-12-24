# Neumann Parser

Module 6 of Neumann. Hand-written recursive descent parser for the unified query language.

## Design Principles

1. **Zero Dependencies**: No external parsing libraries
2. **Hand-Written**: Full control over error messages and parsing behavior
3. **Span Tracking**: Every AST node tracks source location for error reporting
4. **Pratt Parsing**: Precedence-based expression parsing for correct operator handling

## Architecture

```
Input String
    |
    v
+----------+     +--------+     +--------+
|  Lexer   | --> | Tokens | --> | Parser |
+----------+     +--------+     +--------+
                                    |
                                    v
                               +--------+
                               |  AST   |
                               +--------+
```

### Components

| File | Purpose |
|------|---------|
| `lexer.rs` | Tokenization - converts source to tokens |
| `token.rs` | Token definitions and keywords |
| `parser.rs` | Statement parsing - recursive descent |
| `expr.rs` | Expression parsing - Pratt algorithm |
| `ast.rs` | AST node definitions |
| `span.rs` | Source location tracking |
| `error.rs` | Error types with source context |
| `lib.rs` | Public API |

## API Reference

### Parsing

```rust
use neumann_parser::parser;

// Parse a complete statement
let stmt = parser::parse("SELECT * FROM users WHERE id = 1")?;

// Access statement kind
match &stmt.kind {
    StatementKind::Select(select) => {
        // Handle SELECT
    }
    StatementKind::Insert(insert) => {
        // Handle INSERT
    }
    // ... other kinds
}
```

### Tokenization

```rust
use neumann_parser::lexer::tokenize;

let tokens = tokenize("SELECT * FROM users")?;
for token in tokens {
    println!("{:?} at {:?}", token.kind, token.span);
}
```

## Supported Syntax

### SQL Statements

```sql
-- SELECT
SELECT * FROM table
SELECT col1, col2 FROM table WHERE condition
SELECT * FROM table ORDER BY col ASC
SELECT * FROM table LIMIT 10 OFFSET 5

-- INSERT
INSERT INTO table VALUES (1, 'text', 3.14)
INSERT INTO table (col1, col2) VALUES (1, 2)

-- UPDATE
UPDATE table SET col = value WHERE condition

-- DELETE
DELETE FROM table WHERE condition

-- CREATE TABLE
CREATE TABLE name (col1, col2, col3)
```

### Graph Statements

```sql
-- NODE
NODE label {prop1: value1, prop2: value2}

-- EDGE
EDGE label1:id1 type label2:id2
EDGE label1:id1 type label2:id2 {properties}
EDGE label1:id1 type label2:id2 DIRECTED

-- NEIGHBORS
NEIGHBORS label:id
NEIGHBORS label:id edge_type DIRECTION

-- PATH
PATH label1:id1 TO label2:id2
PATH label1:id1 TO label2:id2 VIA edge_type

-- FIND
FIND NODE WHERE condition
FIND EDGE WHERE condition
```

### Vector Statements

```sql
-- EMBED
EMBED STORE 'key' [0.1, 0.2, 0.3]
EMBED GET 'key'
EMBED DELETE 'key'

-- SIMILAR
SIMILAR 'key' LIMIT k
```

### Expressions

```sql
-- Literals
42                    -- integer
3.14                  -- float
'hello'               -- string
true, false           -- boolean
[1, 2, 3]             -- array

-- Comparisons
a = b
a != b
a < b, a > b
a <= b, a >= b

-- Logical
a AND b
a OR b
NOT a

-- Arithmetic
a + b, a - b
a * b, a / b

-- Special
a IS NULL
a IS NOT NULL
a IN (1, 2, 3)
a LIKE 'pattern%'
a BETWEEN 1 AND 10
```

## AST Structure

### Statement

```rust
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

pub enum StatementKind {
    // Relational
    Select(SelectStmt),
    Insert(InsertStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateTable(CreateTableStmt),

    // Graph
    Node(NodeStmt),
    Edge(EdgeStmt),
    Neighbors(NeighborsStmt),
    Path(PathStmt),
    Find(FindStmt),

    // Vector
    Embed(EmbedStmt),
    Similar(SimilarStmt),
}
```

### Expressions

```rust
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

pub enum ExprKind {
    Literal(Literal),
    Identifier(String),
    Column(ColumnRef),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    IsNull(Box<Expr>, bool),
    Between(Box<Expr>, Box<Expr>, Box<Expr>),
    In(Box<Expr>, Vec<Expr>),
    Like(Box<Expr>, String),
    Array(Vec<Expr>),
}
```

## Error Handling

### Parse Errors

```rust
pub struct ParseError {
    pub message: String,
    pub span: Span,
    pub expected: Option<String>,
    pub found: Option<String>,
}
```

### Rich Error Messages

```rust
let result = parser::parse("SELECT FROM users");
if let Err(e) = result {
    println!("{}", e.format_with_source(input));
}
// Output:
// error: Expected expression or '*' after SELECT
//   --> input:1:8
//   |
// 1 | SELECT FROM users
//   |        ^^^^ unexpected token
```

## Operator Precedence

| Precedence | Operators |
|------------|-----------|
| 1 (lowest) | OR |
| 2 | AND |
| 3 | NOT |
| 4 | =, !=, <, >, <=, >= |
| 5 | IS, IN, LIKE, BETWEEN |
| 6 | +, - |
| 7 (highest) | *, / |

## Reserved Keywords

```
SELECT, FROM, WHERE, INSERT, INTO, VALUES, UPDATE, SET,
DELETE, CREATE, TABLE, AND, OR, NOT, NULL, IS, IN, LIKE,
BETWEEN, ORDER, BY, ASC, DESC, LIMIT, OFFSET, NODE, EDGE,
PATH, TO, VIA, NEIGHBORS, FIND, DIRECTED, OUTGOING, INCOMING,
BOTH, EMBED, STORE, GET, SIMILAR, TRUE, FALSE
```

## Performance

| Operation | Complexity |
|-----------|------------|
| Tokenize | O(n) |
| Parse | O(n) |
| Total | O(n) |

Where n = input length. Both lexer and parser are single-pass.

## Test Coverage

| Category | Tests |
|----------|-------|
| Lexer | Token types, keywords, operators, literals |
| Parser | All statement types, edge cases, errors |
| Expressions | All operators, precedence, associativity |
| Errors | Missing tokens, invalid syntax, unexpected EOF |

## Usage with Query Router

```rust
use neumann_parser::parser;
use query_router::QueryRouter;

// Parse statement
let stmt = parser::parse("SELECT * FROM users")?;

// Execute via router (uses execute_parsed internally)
let router = QueryRouter::new();
let result = router.execute_parsed("SELECT * FROM users")?;
```

## Future Considerations

Not implemented:

- **Subqueries**: Nested SELECT statements
- **Joins**: JOIN syntax (currently handled differently)
- **Window functions**: OVER, PARTITION BY
- **CTEs**: WITH clauses
- **Aggregations**: GROUP BY, HAVING, COUNT, SUM, etc.
