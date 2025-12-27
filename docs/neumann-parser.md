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
SELECT DISTINCT col1, col2 FROM table
SELECT col1, col2 FROM table WHERE condition
SELECT * FROM table ORDER BY col ASC NULLS LAST
SELECT * FROM table LIMIT 10 OFFSET 5

-- SELECT with GROUP BY and HAVING
SELECT col, COUNT(*) FROM table GROUP BY col
SELECT col, COUNT(*) AS cnt FROM table GROUP BY col HAVING COUNT(*) > 5

-- SELECT with JOINs
SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id
SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id
SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id
SELECT * FROM t1 FULL JOIN t2 ON t1.id = t2.id
SELECT * FROM t1 CROSS JOIN t2
SELECT * FROM t1 NATURAL JOIN t2
SELECT * FROM t1 JOIN t2 USING (id)

-- Subqueries
SELECT * FROM (SELECT id FROM users) AS u
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)
SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)

-- INSERT
INSERT INTO table VALUES (1, 'text', 3.14)
INSERT INTO table (col1, col2) VALUES (1, 2)

-- UPDATE
UPDATE table SET col = value WHERE condition

-- DELETE
DELETE FROM table WHERE condition

-- CREATE TABLE with constraints
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    age INT CHECK (age >= 18),
    status VARCHAR DEFAULT 'active',
    dept_id INT REFERENCES departments(id)
)

-- DROP TABLE
DROP TABLE IF EXISTS users CASCADE

-- CREATE/DROP INDEX
CREATE UNIQUE INDEX idx_email ON users(email)
CREATE INDEX IF NOT EXISTS idx_name ON users(name)
DROP INDEX IF EXISTS idx_name
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
SIMILAR [0.1, 0.2, 0.3] LIMIT k
```

### Vault Statements

```sql
-- VAULT SET
VAULT SET 'key' 'value'

-- VAULT GET
VAULT GET 'key'

-- VAULT DELETE
VAULT DELETE 'key'

-- VAULT LIST
VAULT LIST
VAULT LIST 'pattern'

-- VAULT ROTATE
VAULT ROTATE 'key' 'new_value'

-- VAULT GRANT
VAULT GRANT 'entity' ON 'key'

-- VAULT REVOKE
VAULT REVOKE 'entity' ON 'key'
```

### Cache Statements

```sql
-- CACHE INIT
CACHE INIT

-- CACHE STATS
CACHE STATS

-- CACHE CLEAR
CACHE CLEAR

-- CACHE EVICT
CACHE EVICT
CACHE EVICT 100

-- CACHE GET/PUT
CACHE GET 'key'
CACHE PUT 'key' 'value'
```

### Blob Statements

```sql
-- BLOB PUT
BLOB PUT 'filename' 'data'
BLOB PUT 'filename' FROM 'path'
BLOB PUT 'filename' 'data' LINK entity TAG tag

-- BLOB GET
BLOB GET 'artifact_id'
BLOB GET 'artifact_id' TO 'path'

-- BLOB DELETE
BLOB DELETE 'artifact_id'

-- BLOB INFO
BLOB INFO 'artifact_id'

-- BLOB LINK/UNLINK
BLOB LINK 'artifact_id' TO entity
BLOB UNLINK 'artifact_id' FROM entity
BLOB LINKS 'artifact_id'

-- BLOB TAG/UNTAG
BLOB TAG 'artifact_id' 'tag'
BLOB UNTAG 'artifact_id' 'tag'

-- BLOB VERIFY
BLOB VERIFY 'artifact_id'

-- BLOB GC
BLOB GC
BLOB GC FULL

-- BLOB REPAIR
BLOB REPAIR

-- BLOB STATS
BLOB STATS

-- BLOB META
BLOB META SET 'artifact_id' 'key' 'value'
BLOB META GET 'artifact_id' 'key'

-- BLOBS (Query/List)
BLOBS
BLOBS 'prefix'
BLOBS FOR entity
BLOBS BY TAG 'tag'
BLOBS WHERE TYPE = 'content_type'
BLOBS SIMILAR TO 'artifact_id' LIMIT n
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
    DropTable(DropTableStmt),
    CreateIndex(CreateIndexStmt),
    DropIndex(DropIndexStmt),
    ShowTables,

    // Graph
    Node(NodeStmt),
    Edge(EdgeStmt),
    Neighbors(NeighborsStmt),
    Path(PathStmt),
    Find(FindStmt),

    // Vector
    Embed(EmbedStmt),
    Similar(SimilarStmt),

    // Vault
    Vault(VaultStmt),

    // Cache
    Cache(CacheStmt),

    // Blob
    Blob(BlobStmt),
    Blobs(BlobsStmt),
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
-- SQL
SELECT, DISTINCT, FROM, WHERE, INSERT, INTO, VALUES, UPDATE, SET,
DELETE, CREATE, DROP, TABLE, INDEX, AND, OR, NOT, NULL, IS, IN,
LIKE, BETWEEN, ORDER, BY, ASC, DESC, NULLS, FIRST, LAST, LIMIT,
OFFSET, GROUP, HAVING, JOIN, INNER, LEFT, RIGHT, FULL, OUTER,
CROSS, NATURAL, ON, USING, AS, PRIMARY, KEY, UNIQUE, REFERENCES,
FOREIGN, CHECK, DEFAULT, CASCADE, IF, EXISTS, SHOW, TABLES

-- Graph
NODE, EDGE, PATH, TO, VIA, NEIGHBORS, FIND, DIRECTED, OUTGOING,
INCOMING, BOTH, RETURN

-- Vector
EMBED, STORE, GET, SIMILAR, COSINE, EUCLIDEAN, DOT_PRODUCT

-- Vault
VAULT, GRANT, REVOKE, ROTATE, LIST

-- Cache
CACHE, INIT, STATS, CLEAR, EVICT, PUT

-- Blob
BLOB, BLOBS, INFO, LINK, UNLINK, LINKS, TAG, UNTAG, VERIFY,
GC, REPAIR, META, TYPE, FOR

-- Literals
TRUE, FALSE
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

Not yet implemented:

- **Window functions**: OVER, PARTITION BY, ROW_NUMBER, RANK
- **CTEs**: WITH clauses (Common Table Expressions)
- **UNION/INTERSECT/EXCEPT**: Set operations
- **Stored procedures**: User-defined functions
- **Triggers**: Automatic actions on data changes
