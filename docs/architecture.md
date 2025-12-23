# Neumann Architecture

## System Overview

Neumann is a unified runtime that stores relational data, graph relationships, and vector embeddings in a single tensor structure.

```
+--------------------------------------------------+
|                    Shell (CLI)                    |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|                  Query Engines                    |
|  +------------+ +------------+ +------------+    |
|  | Relational | |   Graph    | |   Vector   |    |
|  |  [DONE]    | |  (Future)  | |  (Future)  |    |
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

### Module 3: Graph Engine (Planned)

**Responsibility**: Graph traversals and path queries.

**Will Provide**:
- `TRAVERSE`, `PATH`, `NEIGHBORS`
- Relationship queries

### Module 4: Vector Engine (Planned)

**Responsibility**: Similarity search on embeddings.

**Will Provide**:
- `SIMILAR`, `NEAREST`
- Distance calculations

### Module 5: Shell (Planned)

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
RwLock write lock acquired
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
RwLock read lock acquired
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

### 4. Thread Safety via RwLock

Single `RwLock` over entire HashMap. Simple and correct, but:
- All writes serialize
- Long scans block writes

For high-throughput scenarios, consider:
- Sharded locks by key prefix
- Lock-free data structures
- Read-optimized snapshots

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
  tensor_store/
    Cargo.toml
    src/lib.rs               # Module 1: Storage layer
  relational_engine/
    Cargo.toml
    src/lib.rs               # Module 2: SQL-like operations
```

## Quality Gates

### Pre-commit Hook

Runs before every commit:
1. `cargo fmt --check`
2. `cargo clippy -- -D warnings`
3. `cargo test --quiet`
4. `cargo doc --no-deps --quiet`

### CI Pipeline

Runs on every PR:
1. Format check
2. Clippy lints
3. Tests
4. Documentation build
5. Miri (undefined behavior detection)

## Versioning Strategy

Not yet implemented. Future considerations:
- Semantic versioning for crate
- API stability guarantees
- Migration paths for data format changes
