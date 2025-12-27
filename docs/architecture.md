# Neumann Architecture

## System Overview

Neumann is a unified runtime that stores relational data, graph relationships, and vector embeddings in a single tensor structure.

```
+--------------------------------------------------+
|                 Shell (CLI) [DONE]                |
|   - Interactive REPL                             |
|   - Command history                              |
|   - Formatted output                             |
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
|   - LLM cache integration                        |
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
|            Tensor Cache (Module 10) [DONE]        |
|   - LLM response caching (exact + semantic)      |
|   - Token counting (tiktoken)                    |
|   - Cost tracking and savings estimation         |
|   - Background eviction (LRU/LFU/Cost/Hybrid)    |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|            Tensor Vault (Module 9) [DONE]         |
|   - AES-256-GCM encrypted secrets                |
|   - Argon2id key derivation                      |
|   - Graph-based access control                   |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Tensor Store (Module 1) [DONE]          |
|   - Key-value storage                            |
|   - Scalars, vectors, pointers                   |
|   - Thread-safe, in-memory                       |
|   - Snapshot persistence (bincode)               |
|   - HNSW index for similarity search             |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|         Tensor Compress (Module 8) [DONE]         |
|   - Vector quantization (int8, binary)           |
|   - Delta encoding for sorted IDs                |
|   - Run-length encoding                          |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|         Sparse/Delta Vectors [DONE]               |
|   - SparseVector for high-sparsity embeddings    |
|   - DeltaVector for archetype-based encoding     |
|   - K-means clustering for archetype discovery   |
|   - Unified HNSW index (Dense/Sparse/Delta)      |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Persistence Layer [DONE]                |
|   - Snapshot save/load (bincode)                 |
|   - Compressed snapshots (tensor_compress)       |
|   - Atomic writes (temp file + rename)           |
|   - Write-ahead log (WAL) for crash recovery     |
|   - Bloom filter rebuild on load                 |
|   [Future: incremental, streaming]               |
+--------------------------------------------------+
```

## Module Boundaries

### Module 1: Tensor Store (Complete)

**Responsibility**: Hold data. Know nothing about queries. Provide persistence.

**Interface**:
- `put(key, tensor) -> Result<()>`
- `get(key) -> Result<TensorData>`
- `delete(key) -> Result<()>`
- `exists(key) -> bool`
- `scan(prefix) -> Vec<String>`
- `scan_count(prefix) -> usize`
- `clear()`
- `len() -> usize`
- `save_snapshot(path) -> Result<(), SnapshotError>`
- `load_snapshot(path) -> Result<TensorStore, SnapshotError>`
- `save_snapshot_compressed(path, config) -> Result<(), SnapshotError>`
- `load_snapshot_compressed(path) -> Result<TensorStore, SnapshotError>`

**Serialization**: All core types (`TensorData`, `TensorValue`, `ScalarValue`) implement `serde::Serialize` and `serde::Deserialize`.

**Does Not**:
- Parse queries
- Validate schemas
- Enforce relationships
- Provide incremental persistence (future)

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

### Module 7: Shell (Complete)

**Responsibility**: Interactive command-line interface.

**Interface**:
- `Shell::new()` - Create shell with default config
- `Shell::with_config(config)` - Create with custom config
- `shell.execute(command)` - Execute single command
- `shell.run()` - Start interactive REPL loop

**Features**:
- Readline support (history, arrow keys, Ctrl+C)
- ASCII table output formatting
- Built-in commands (help, exit, tables, clear)
- Persistent command history

**Does Not**:
- Parse queries directly (delegates to Query Router)
- Implement syntax highlighting
- Support multi-line queries

### Module 8: Tensor Compress (Complete)

**Responsibility**: Compression algorithms for tensor data.

**Interface**:
- `quantize_int8(vector) -> QuantizedInt8` - Compress f32 to int8 (4x)
- `dequantize_int8(quantized) -> Vec<f32>` - Restore from int8
- `quantize_binary(vector) -> QuantizedBinary` - Compress to binary (32x)
- `dequantize_binary(quantized, len) -> Vec<f32>` - Restore from binary
- `compress_ids(ids) -> Vec<u8>` - Delta + varint encoding
- `decompress_ids(bytes) -> Vec<u64>` - Restore IDs
- `rle_encode(data) -> RleEncoded<T>` - Run-length encoding
- `rle_decode(encoded) -> Vec<T>` - Restore from RLE

**Features**:
- Vector quantization with ~1% error bound (int8)
- Lossless delta encoding for sorted sequences
- Lossless RLE for repeated values
- Snapshot format v2 with magic bytes "NEUM"

**Does Not**:
- Handle persistence directly (used by TensorStore)
- Implement product quantization (future)
- Support streaming compression (future)

### Sparse/Delta Vectors (Complete)

**Responsibility**: Memory-efficient vector storage that exploits sparsity and clustering.

**Interface** (SparseVector):
- `from_dense(vector) -> SparseVector` - Convert dense to sparse
- `from_dense_with_threshold(vector, threshold) -> SparseVector` - Sparsify with threshold
- `to_dense() -> Vec<f32>` - Reconstruct dense
- `dot(other) -> f32` - Sparse-sparse dot product O(nnz)
- `dot_dense(other) -> f32` - Sparse-dense dot product O(nnz)
- `cosine_similarity(other) -> f32` - Cosine similarity

**Interface** (DeltaVector):
- `from_dense_with_reference(vector, archetype, id, threshold) -> DeltaVector` - Delta encode
- `to_dense(archetype) -> Vec<f32>` - Reconstruct via archetype + delta
- `dot_dense_with_precomputed(query, arch_dot_query) -> f32` - O(nnz) with precomputed

**Interface** (ArchetypeRegistry + K-means):
- `discover_archetypes(vectors, k, config) -> usize` - K-means clustering
- `encode_batch(vectors, threshold) -> Vec<(DeltaVector, ratio)>` - Bulk encoding
- `analyze_coverage(vectors, threshold) -> CoverageStats` - Quality metrics

**Features**:
- SparseVector: 2.8-33x memory reduction at 90-99% sparsity
- SparseVector: 10-152x dot product speedup at 99% sparsity
- DeltaVector: Stores only differences from reference archetypes
- K-means: Automatic archetype discovery with k-means++ initialization
- HNSW: Unified index supporting Dense, Sparse, and Delta storage

**Philosophy**: Replace explicit zeros with geometric structure. Instead of storing thousands of zero values, we store only the non-zero positions (sparse) or the delta from a learned centroid (delta). This is not just compression - it's recognizing that zeros carry no information and discarding them.

### Module 9: Tensor Vault (Complete)

**Responsibility**: Encrypted secret storage with graph-based access control.

**Interface**:
- `store_secret(key, value, master_password) -> Result<()>`
- `retrieve_secret(key, master_password) -> Result<String>`
- `grant_access(secret_key, entity_id) -> Result<()>`
- `revoke_access(secret_key, entity_id) -> Result<()>`
- `can_access(secret_key, entity_id) -> bool`

**Features**:
- AES-256-GCM encryption for secrets
- Argon2id key derivation with configurable parameters
- Graph-based access control via edges
- Secure key zeroization on drop

**Does Not**:
- Implement key rotation (planned)
- Support secret versioning (planned)
- Provide audit logging (planned)

### Module 10: Tensor Cache (Complete)

**Responsibility**: Semantic caching for LLM responses.

**Interface**:
- `get(prompt, embedding) -> Option<CacheHit>` - Look up cached response
- `put(prompt, embedding, response, model, params) -> Result<()>` - Store response
- `get_embedding(source, content) -> Option<Vec<f32>>` - Get cached embedding
- `put_embedding(source, content, embedding, model) -> Result<()>` - Store embedding
- `invalidate(prompt, model, params) -> bool` - Remove entry
- `evict(count) -> usize` - Manually evict entries
- `stats() -> &CacheStats` - Get statistics

**Features**:
- Three-layer caching: Exact O(1), Semantic O(log n), Embedding O(1)
- Token counting via tiktoken (cl100k_base encoding)
- Cost tracking and savings estimation
- Background eviction with LRU/LFU/Cost/Hybrid strategies
- TTL-based expiration with min-heap tracking

**Does Not**:
- Persist cache to disk (in-memory only)
- Implement distributed caching (single-node)
- Support cache warming (manual only)

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

### 1. In-Memory First with Snapshot Persistence

The system is designed for in-memory operation with snapshot persistence and WAL:

- **Primary storage**: DashMap in memory for fast concurrent access
- **Persistence**: Snapshot-based save/load using bincode serialization
- **Atomicity**: Snapshots write to temp file, then atomic rename
- **Durability**: Write-ahead log (WAL) for crash recovery between snapshots
- **Future**: Incremental persistence planned for large datasets

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

### 6. Zero-Aware Storage (Sparse/Delta Vectors)

Neumann uses geometric structure to eliminate zero storage overhead:

```
Dense Vector (768d):        3,072 bytes (768 x 4 bytes)
  [0.1, 0.0, 0.0, ..., 0.2, 0.0, ...]

Sparse Vector (99% zeros):  ~60 bytes (indices + values)
  positions: [0, 500]
  values: [0.1, 0.2]

Delta Vector (5% diff):     ~120 bytes (archetype_id + sparse delta)
  archetype_id: 3
  delta_positions: [12, 45, 89]
  delta_values: [0.01, -0.02, 0.03]
```

The philosophy:
1. **Zeros carry no information** - don't store what doesn't exist
2. **Similarity implies shared structure** - clustered vectors share archetypes
3. **Geometry replaces data** - k-means centroids capture cluster structure
4. **O(nnz) operations** - dot products scale with non-zeros, not dimension

This yields 10-150x speedups and 3-33x memory savings depending on sparsity.

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
    neumann-shell.md         # Module 7 documentation
    tensor-compress.md       # Module 8 documentation
    tensor-vault.md          # Module 9 documentation
    tensor-cache.md          # Module 10 documentation
    benchmarks.md            # Performance benchmarks
  tensor_store/
    Cargo.toml
    src/
      lib.rs                 # Module 1: Storage layer + SparseVector
      hnsw.rs                # HNSW index with Dense/Sparse/Delta support
      delta_vector.rs        # Delta encoding, archetypes, k-means clustering
    benches/
      tensor_store_bench.rs  # Storage, sparse, delta, k-means benchmarks
  relational_engine/
    Cargo.toml
    src/lib.rs               # Module 2: SQL-like operations
  graph_engine/
    Cargo.toml
    src/lib.rs               # Module 3: Graph operations
  vector_engine/
    Cargo.toml
    src/lib.rs               # Module 4: Similarity search
  tensor_compress/
    Cargo.toml
    src/
      lib.rs                 # Module 8: Public API
      quantize.rs            # Vector quantization
      delta.rs               # Delta + varint encoding
      rle.rs                 # Run-length encoding
      format.rs              # Snapshot format v2
    benches/
      tensor_compress_bench.rs # Compression benchmarks
  tensor_vault/
    Cargo.toml
    src/
      lib.rs                 # Module 9: Vault API
      encryption.rs          # AES-256-GCM encryption
      key.rs                 # Argon2id key derivation
      access.rs              # Graph-based access control
    benches/
      tensor_vault_bench.rs  # Vault benchmarks
  tensor_cache/
    Cargo.toml
    src/
      lib.rs                 # Module 10: Cache public API
      config.rs              # Configuration and presets
      error.rs               # Error types
      stats.rs               # Statistics tracking
      tokenizer.rs           # tiktoken wrapper
      exact.rs               # Exact cache (O(1))
      semantic.rs            # Semantic cache (O(log n))
      embedding.rs           # Embedding cache (O(1))
      index.rs               # HNSW index wrapper
      eviction.rs            # Background eviction
      ttl.rs                 # TTL tracking
    benches/
      cache_bench.rs         # Cache benchmarks
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
  neumann_shell/
    Cargo.toml
    src/
      lib.rs                 # Module 7: Shell implementation
      main.rs                # CLI entry point
    benches/
      neumann_shell_bench.rs # Performance benchmarks
```

## Quality Gates

### Pre-commit Hook

Runs before every commit for all crates:
1. `cargo fmt --check` - Code formatting
2. `cargo clippy -- -D warnings` - Lints
3. `cargo test --quiet` - Unit tests
4. `cargo doc --no-deps --quiet` - Documentation
5. `cargo llvm-cov` - Coverage check (95% minimum, 92% for shell)

### CI Pipeline

Runs on every PR:
1. Check - Compilation verification
2. Format check - Code style
3. Clippy lints - Static analysis
4. Tests - Unit and integration tests
5. Coverage - Minimum 95% per crate (92% for shell)
6. Documentation build - Doc generation
7. Security audit - Dependency vulnerabilities
8. Miri - Undefined behavior detection (tensor_store)

## Versioning Strategy

Not yet implemented. Future considerations:
- Semantic versioning for crate
- API stability guarantees
- Migration paths for data format changes
