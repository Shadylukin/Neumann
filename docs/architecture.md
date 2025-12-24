# Neumann Architecture

## System Overview

Neumann is a unified runtime that stores relational data, graph relationships, and vector embeddings in a single tensor structure.

```
+--------------------------------------------------+
|                    Shell (CLI)                    |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|              Neumann Parser [DONE]               |
|   - Tokenization (lexer)                         |
|   - Recursive descent parsing                    |
|   - AST generation                               |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|               Query Router [DONE]                |
|   - Unified query execution                      |
|   - Engine dispatch                              |
|   - Result aggregation                           |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|                  Query Engines                    |
|  +------------+ +------------+ +------------+    |
|  | Relational | |   Graph    | |   Vector   |    |
|  |  [DONE]    | |   [DONE]   | |   [DONE]   |    |
|  +------------+ +------------+ +------------+    |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Tensor Store (Module 1) [DONE]          |
|   - Key-value storage                            |
|   - Scalars, vectors, pointers                   |
|   - Thread-safe, in-memory                       |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Persistence Layer (Future)              |
|   - File backend                                 |
|   - S3/cloud storage                             |
|   - WAL for durability                           |
+--------------------------------------------------+
```

## Module Boundaries

### Module 1: Tensor Store (Complete)

**Responsibility**: Hold data. Know nothing about queries.

**Interface**:
- `put(key, tensor) -> Result<()>`
- `get(key) -> Result<TensorData>`
- `delete(key) -> Result<()>`
- `exists(key) -> Result<bool>`
- `scan(prefix) -> Result<Vec<String>>`
- `scan_count(prefix) -> Result<usize>`
- `clear() -> Result<()>`
- `len() -> Result<usize>`

**Does Not**:
- Parse queries
- Validate schemas
- Enforce relationships
- Handle persistence

### Module 2: Relational Engine (Complete)

**Responsibility**: SQL-like table operations on Tensor Store.

**Interface**:
- `create_table(name, schema) -> Result<()>`
- `insert(table, values) -> Result<row_id>`
- `select(table, condition) -> Result<Vec<Row>>`
- `update(table, condition, values) -> Result<count>`
- `delete_rows(table, condition) -> Result<count>`
- `join(table_a, table_b, on_a, on_b) -> Result<Vec<(Row, Row)>>`

**Does Not**:
- Store data directly (uses Tensor Store)
- Implement indexes (full table scans)
- Support transactions

### Module 3: Graph Engine (Complete)

**Responsibility**: Graph traversals and path queries.

**Interface**:
- `create_node(label, properties) -> Result<node_id>`
- `create_edge(from, to, type, properties, directed) -> Result<edge_id>`
- `traverse(start, direction, depth, edge_type) -> Result<Vec<Node>>`
- `find_path(from, to) -> Result<Path>`
- `neighbors(node_id, edge_type, direction) -> Result<Vec<Node>>`

**Does Not**:
- Store data directly (uses Tensor Store)
- Implement weighted paths (Dijkstra)
- Support pattern matching (Cypher-style)

### Module 4: Vector Engine (Complete)

**Responsibility**: Similarity search on embeddings.

**Interface**:
- `store_embedding(key, vector) -> Result<()>`
- `get_embedding(key) -> Result<Vec<f64>>`
- `delete_embedding(key) -> Result<()>`
- `search_similar(query, top_k) -> Result<Vec<SearchResult>>`
- `compute_similarity(a, b) -> Result<f64>`
- `exists(key) -> bool`
- `count() -> usize`
- `list_keys() -> Vec<String>`

**Does Not**:
- Store data directly (uses Tensor Store)
- Implement approximate nearest neighbor (brute-force search)
- Support metadata filtering

### Module 5: Query Router (Complete)

**Responsibility**: Unified query execution across all engines.

**Interface**:
- `execute(command) -> Result<QueryResult>` - String-based execution
- `execute_parsed(command) -> Result<QueryResult>` - AST-based execution

**Supports**:
- Relational: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE
- Graph: NODE, EDGE, PATH, NEIGHBORS, FIND
- Vector: EMBED, SIMILAR

**Does Not**:
- Parse queries directly (delegates to Neumann Parser)
- Implement cross-engine joins (planned)

### Module 6: Neumann Parser (Complete)

**Responsibility**: Parse unified query language into AST.

**Interface**:
- `parse(input) -> Result<Statement>` - Parse full statement
- `tokenize(input) -> Result<Vec<Token>>` - Tokenize only

**Features**:
- Hand-written recursive descent parser
- Pratt parsing for expression precedence
- Span tracking for error messages
- Zero external dependencies

**Supported Syntax**:
- SQL: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE
- Graph: NODE, EDGE, PATH, NEIGHBORS, FIND NODE/EDGE
- Vector: EMBED STORE/GET/DELETE, SIMILAR

### Module 7: Shell (Planned)

**Responsibility**: User interface.

**Will Provide**:
- CLI commands
- REPL
- Script execution

## Data Flow

### Write Path

```
User Command
    |
    v
Shell parses command
    |
    v
Query Layer validates and transforms
    |
    v
Tensor Store.put(key, tensor)
    |
    v
DashMap shard lock acquired (only this shard)
    |
    v
HashMap insert
    |
    v
Lock released, Ok(()) returned
```

### Read Path

```
User Query
    |
    v
Shell parses query
    |
    v
Query Layer plans execution
    |
    v
Tensor Store.get(key) / scan(prefix)
    |
    v
DashMap lookup (lock-free for reads)
    |
    v
Data cloned and returned
    |
    v
Query Layer processes results
    |
    v
Shell formats output
```

## Key Design Decisions

### 1. In-Memory First

The README states: "Single-node, in-memory first. Durability and clustering come later."

This simplifies Module 1 and allows focus on correctness. Persistence is a separate concern.

### 2. Clone on Read

`get()` returns cloned data rather than references. This:
- Prevents holding locks during processing
- Enables concurrent reads
- Trades memory for parallelism

For future optimization, consider copy-on-write or arena allocation.

### 3. String Keys

Keys are strings with convention `type:id`. This:
- Enables prefix scanning
- Is human-readable
- Avoids complex key encoding

### 4. Thread Safety via DashMap

Uses DashMap (sharded concurrent HashMap) for thread safety:
- Reads are lock-free
- Writes only block other writes to the same shard (~16 shards)
- No lock poisoning (unlike RwLock)
- Simple API without manual lock handling

This provides significantly better concurrent write throughput compared to a single RwLock.

### 5. No Schema Enforcement

Tensor Store accepts any `TensorData`. Schema validation belongs in Query Layer.

## File Structure

```
Neumann/
  README.md                  # Project overview
  CLAUDE.md                  # AI coding guidelines
  .gitignore
  .github/
    workflows/
      ci.yml                 # GitHub Actions CI
  scripts/
    pre-commit               # Quality gate hook
    setup-hooks.sh           # Hook installer
  docs/
    architecture.md          # This file
    tensor-store.md          # Module 1 documentation
    relational-engine.md     # Module 2 documentation
    graph-engine.md          # Module 3 documentation
    vector-engine.md         # Module 4 documentation
    query-router.md          # Module 5 documentation
    neumann-parser.md        # Module 6 documentation
  tensor_store/
    Cargo.toml
    src/lib.rs               # Module 1: Storage layer
  relational_engine/
    Cargo.toml
    src/lib.rs               # Module 2: SQL-like operations
  graph_engine/
    Cargo.toml
    src/lib.rs               # Module 3: Graph operations
  vector_engine/
    Cargo.toml
    src/lib.rs               # Module 4: Similarity search
  query_router/
    Cargo.toml
    src/lib.rs               # Module 5: Query execution
  neumann_parser/
    Cargo.toml
    src/
      lib.rs                 # Module 6: Public API
      lexer.rs               # Tokenization
      token.rs               # Token definitions
      ast.rs                 # AST node types
      parser.rs              # Statement parsing
      expr.rs                # Expression parsing
      span.rs                # Source locations
      error.rs               # Error types
```

## Quality Gates

### Pre-commit Hook

Runs before every commit for all crates:
1. `cargo fmt --check` - Code formatting
2. `cargo clippy -- -D warnings` - Lints
3. `cargo test --quiet` - Unit tests
4. `cargo doc --no-deps --quiet` - Documentation
5. `cargo llvm-cov` - Coverage check (minimum 95%)

### CI Pipeline

Runs on every PR:
1. Check - Compilation verification
2. Format check - Code style
3. Clippy lints - Static analysis
4. Tests - Unit and integration tests
5. Coverage - Minimum 95% per crate
6. Documentation build - Doc generation
7. Security audit - Dependency vulnerabilities
8. Miri - Undefined behavior detection (tensor_store)

## Versioning Strategy

Not yet implemented. Future considerations:
- Semantic versioning for crate
- API stability guarantees
- Migration paths for data format changes
