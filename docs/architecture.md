# Neumann Architecture

## System Overview

Neumann is a unified runtime that stores relational data, graph relationships,
and vector embeddings in a single tensor structure.

```text
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
|          Tensor Unified (Module 12) [DONE]        |
|   - Cross-engine operations                      |
|   - Unified entity management                    |
|   - find_similar_connected, find_neighbors_by_similarity |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|            Tensor Blob (Module 11) [DONE]         |
|   - S3-style chunked object storage             |
|   - Content-addressable SHA-256 deduplication   |
|   - Streaming upload/download (BlobWriter/Reader)|
|   - Entity linking, tagging, metadata           |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|        Tensor Checkpoint (Module 13) [DONE]       |
|   - Manual and auto checkpoints                 |
|   - Rollback to previous state                  |
|   - Count-based retention (purge oldest)        |
|   - Uses tensor_blob for storage                |
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
|          Tensor Chain (Module 14) [DONE]          |
|   - Tensor-native blockchain with Raft consensus |
|   - Semantic conflict detection (delta vectors)  |
|   - Hierarchical codebook validation             |
|   - 2PC distributed transactions                 |
|   - TCP transport with membership management     |
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
|         Tiered Memory Storage [DONE]              |
|   - Hot tier: DashMap in-memory                  |
|   - Cold tier: mmap-backed file storage          |
|   - Access instrumentation for hot/cold tracking |
|   - Automatic cold migration based on access     |
|   - Cold-to-hot promotion on access              |
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

**Serialization**: All core types (`TensorData`, `TensorValue`, `ScalarValue`)
implement `serde::Serialize` and `serde::Deserialize`.

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
- `search_similar_with_metric(query, top_k, metric) ->
  Result<Vec<SearchResult>>`
- `compute_similarity(a, b) -> Result<f64>`
- `exists(key) -> bool`
- `count() -> usize`
- `list_keys() -> Vec<String>`

**Distance Metrics**:

- `DistanceMetric::Cosine` - Cosine similarity (default)
- `DistanceMetric::Euclidean` - Euclidean distance (L2)
- `DistanceMetric::DotProduct` - Raw dot product

**Does Not**:

- Store data directly (uses Tensor Store)
- Support metadata filtering
- HNSW with non-cosine metrics (falls back to brute-force)

### Module 5: Query Router (Complete)

**Responsibility**: Unified query execution across all engines.

**Interface**:

- `execute(command) -> Result<QueryResult>` - String-based execution
- `execute_parsed(command) -> Result<QueryResult>` - AST-based execution
- `execute_async(command) -> Result<QueryResult>` - Async string-based
- `execute_parsed_async(command) -> Result<QueryResult>` - Async AST-based

**Supports**:

- Relational: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE
- Graph: NODE, EDGE, PATH, NEIGHBORS, FIND
- Vector: EMBED, SIMILAR (with COSINE, EUCLIDEAN, DOT_PRODUCT metrics)
- Unified: ENTITY CREATE, ENTITY CONNECT, SIMILAR...CONNECTED TO

**Does Not**:

- Parse queries directly (delegates to Neumann Parser)
- Implement cross-engine SQL joins

### Module 12: Tensor Unified (Complete)

**Responsibility**: Cross-engine operations and unified entity management.

**Interface**:

- `create_entity(key, fields, embedding) -> Result<()>`
- `connect_entities(from, to, edge_type) -> Result<String>`
- `find_similar_connected(query, connected_to, k) -> Result<Vec<UnifiedItem>>`
- `find_neighbors_by_similarity(entity, query, k) -> Result<Vec<UnifiedItem>>`

**Features**:

- Entities with relational fields, graph connections, and embeddings
- Cross-engine queries combining vector similarity with graph connectivity
- Async-first design for concurrent operations

**Does Not**:

- Store data directly (delegates to individual engines)
- Implement transactions across engines

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

**Responsibility**: Memory-efficient vector storage that exploits sparsity and
clustering.

**Interface** (SparseVector):

- `from_dense(vector) -> SparseVector` - Convert dense to sparse
- `from_dense_with_threshold(vector, threshold) -> SparseVector` - Sparsify with
  threshold
- `from_diff(before, after, threshold) -> SparseVector` - Compute sparse delta
  directly
- `to_dense() -> Vec<f32>` - Reconstruct dense
- `dot(other) -> f32` - Sparse-sparse dot product O(nnz)
- `dot_dense(other) -> f32` - Sparse-dense dot product O(nnz)
- `cosine_similarity(other) -> f32` - Cosine similarity
- `sub(other) -> SparseVector` - Sparse subtraction
- `weighted_average(other, w1, w2) -> SparseVector` - Weighted combination
- `project_orthogonal(direction) -> SparseVector` - Orthogonal projection

**Geometric Distance Metrics**:

- `angular_distance(other) -> f32` - Angle-based distance
- `geodesic_distance(other) -> f32` - Distance on unit sphere
- `jaccard_index(other) -> f32` - Structural overlap (shared non-zero positions)
- `overlap_coefficient(other) -> f32` - Normalized overlap
- `weighted_jaccard(other) -> f32` - Value-weighted Jaccard
- `euclidean_distance(other) -> f32` - L2 distance

**DistanceMetric Enum**:

- `Cosine` - Angular similarity (default)
- `Angular` - Angular distance
- `Geodesic` - Spherical distance
- `Jaccard` - Structural overlap
- `Overlap` - Normalized overlap
- `Euclidean` - L2 distance
- `Composite(GeometricConfig)` - Weighted combination of metrics

**Interface** (DeltaVector - Consensus):

- `new(delta: SparseVector, keys, tx_id) -> DeltaVector` - Create from sparse
  delta
- `from_sparse(delta: SparseVector, keys, tx_id) -> DeltaVector` - Explicit
  sparse constructor
- `from_dense(delta: Vec<f32>, keys, tx_id) -> DeltaVector` - Convert dense to
  sparse
- `magnitude() -> f32` - Delta magnitude
- `cosine_similarity(other) -> f32` - Similarity between deltas
- `add(other) -> DeltaVector` - Orthogonal merge via vector addition
- `weighted_average(other, w1, w2) -> DeltaVector` - Low-conflict merge
- `project_non_conflicting(direction) -> DeltaVector` - Remove conflicting
  component
- `structural_similarity(other) -> f32` - Jaccard index of affected positions

**Interface** (DeltaVector - Storage):

- `from_dense_with_reference(vector, archetype, id, threshold) -> DeltaVector` -
  Delta encode
- `to_dense(archetype) -> Vec<f32>` - Reconstruct via archetype + delta
- `dot_dense_with_precomputed(query, arch_dot_query) -> f32` - O(nnz) with
  precomputed

**EmbeddingState** (Type-safe embedding lifecycle):

- `Initial { before: SparseVector }` - Transaction started, before-state
  captured
- `Computed { before, after, delta: SparseVector }` - Ready to commit, delta
  computed
- `new(before) -> Initial` - Create initial state
- `from_dense(before) -> Initial` - Create from dense vector
- `compute(after) -> Result<Computed>` - Transition to computed state
- `compute_with_threshold(after, threshold) -> Result<Computed>` - Sparse delta
  computation
- `before() -> &SparseVector` - Always available
- `delta() -> Option<&SparseVector>` - Only in Computed state
- `is_computed() -> bool` - Check state

**Interface** (ArchetypeRegistry + K-means):

- `discover_archetypes(vectors, k, config) -> usize` - K-means clustering
- `encode_batch(vectors, threshold) -> Vec<(DeltaVector, ratio)>` - Bulk
  encoding
- `analyze_coverage(vectors, threshold) -> CoverageStats` - Quality metrics

**Features**:

- SparseVector: 2.8-33x memory reduction at 90-99% sparsity
- SparseVector: 10-152x dot product speedup at 99% sparsity
- DeltaVector: Stores only differences from reference archetypes
- K-means: Automatic archetype discovery with k-means++ initialization
- HNSW: Unified index supporting Dense, Sparse, and Delta storage

**Philosophy**: Replace explicit zeros with geometric structure. Instead of
storing thousands of zero values, we store only the non-zero positions (sparse)
or the delta from a learned centroid (delta). This is not just compression -
it's recognizing that zeros carry no information and discarding them.

### Tiered Memory Storage (Complete)

**Responsibility**: Enable datasets larger than RAM by automatically tiering
data between hot (in-memory) and cold (mmap) storage based on access patterns.

**Interface** (TieredStore):

- `put(key, tensor)` - Insert into hot tier
- `get(key) -> Result<TensorData>` - Get from hot or cold (promotes if cold)
- `exists(key) -> bool` - Check either tier
- `delete(key) -> bool` - Remove from both tiers
- `migrate_cold(threshold_ms) -> Result<usize>` - Move cold shards to mmap
- `preload(keys) -> Result<usize>` - Warm specific keys from cold
- `hot_shards(limit) -> Vec<(usize, u64)>` - Most accessed shards
- `cold_shards(threshold_ms) -> Vec<usize>` - Least accessed shards
- `stats() -> TieredStats` - Hot/cold counts, lookups, migrations

**Interface** (MmapStore):

- `MmapStoreBuilder::create(path)` - Create new mmap file
- `MmapStore::open(path)` - Open read-only mmap
- `MmapStoreMut::create(path, capacity)` - Create mutable mmap
- `get(key) -> Result<TensorData>` - Deserialize on demand
- `compact(output_path) -> Result<u64>` - Remove garbage, reclaim space

**Features**:

- Hot tier: DashMap with ~16 shards for concurrent access
- Cold tier: Memory-mapped files with bincode serialization
- Access instrumentation: Per-shard read/write tracking with sampling
- Auto-grow: Cold storage files expand as needed
- Promotion: Cold data moves to hot on access
- Compaction: Remove old versions of updated keys

**Performance** (128-dim embeddings, 10K entries):

- Hot put: 1.4-1.5 M ops/sec
- Hot get: 2.9-3.4 M ops/sec
- Cold migration: 1.0-1.4 M entries/sec
- Cold promotion: 0.5-0.6 M entries/sec
- Write overhead vs pure in-memory: 5-7%

**Use Cases**:

- Datasets exceeding available RAM (e.g., 52GB+ vector indices)
- Working sets smaller than total data (hot 10%, cold 90%)
- Development machines with limited memory
- Graceful degradation under memory pressure

### Module 9: Tensor Vault (Complete)

**Responsibility**: Encrypted secret storage with graph-based access control for
multi-agent environments.

**Interface**:

- `set(requester, key, value) -> Result<()>`
- `get(requester, key) -> Result<String>`
- `delete(requester, key) -> Result<()>`
- `rotate(requester, key, new_value) -> Result<()>`
- `list(requester, pattern) -> Result<Vec<String>>`
- `grant(requester, entity, key) -> Result<()>`
- `grant_with_permission(requester, entity, key, permission) -> Result<()>`
- `grant_with_ttl(requester, entity, key, ttl) -> Result<()>`
- `revoke(requester, entity, key) -> Result<()>`
- `get_version(requester, key, version) -> Result<String>`
- `list_versions(requester, key) -> Result<Vec<VersionInfo>>`
- `rollback(requester, key, version) -> Result<()>`
- `audit_log(key) -> Vec<AuditEntry>`
- `audit_by_entity(entity) -> Vec<AuditEntry>`
- `namespace(namespace, identity) -> NamespacedVault`

**Features**:

- AES-256-GCM encryption for secrets
- Argon2id key derivation with configurable parameters
- Graph-based access control via edges
- Permission levels (Read/Write/Admin)
- TTL grants with automatic expiration
- Rate limiting per entity/operation
- Namespace isolation for multi-tenant systems
- Audit logging for all operations
- Secret versioning with rollback
- Secure key zeroization on drop

**Does Not**:

- Persist audit logs to disk (in-memory only)
- Implement distributed rate limiting (single-node)

### Module 10: Tensor Cache (Complete)

**Responsibility**: Semantic caching for LLM responses.

**Interface**:

- `get(prompt, embedding) -> Option<CacheHit>` - Look up cached response
- `put(prompt, embedding, response, model, params) -> Result<()>` - Store
  response
- `get_embedding(source, content) -> Option<Vec<f32>>` - Get cached embedding
- `put_embedding(source, content, embedding, model) -> Result<()>` - Store
  embedding
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

### Module 11: Tensor Blob (Complete)

**Responsibility**: S3-style object storage for large artifacts using
content-addressable chunked storage.

**Interface**:

- `put(filename, data, options) -> Result<artifact_id>` - Store artifact
- `get(artifact_id) -> Result<Vec<u8>>` - Retrieve artifact
- `delete(artifact_id) -> Result<()>` - Delete artifact
- `writer(filename, options) -> BlobWriter` - Streaming upload
- `reader(artifact_id) -> BlobReader` - Streaming download
- `metadata(artifact_id) -> ArtifactMetadata` - Get metadata
- `link(artifact_id, entity) -> Result<()>` - Link to entity
- `tag(artifact_id, tag) -> Result<()>` - Add tag
- `by_tag(tag) -> Vec<ArtifactMetadata>` - Find by tag
- `artifacts_for(entity) -> Vec<ArtifactMetadata>` - Find by entity
- `gc() -> GcStats` - Garbage collect orphaned chunks
- `verify(artifact_id) -> Result<bool>` - Verify integrity
- `repair() -> RepairStats` - Repair broken references

**Features**:

- Content-addressable chunking with SHA-256 hashes
- Automatic deduplication via reference counting
- Streaming API for large files (never fully in memory)
- Entity linking via graph edges
- Tagging for organization
- Custom metadata fields
- Semantic search via embeddings
- Background garbage collection
- Integrity verification and repair

**Does Not**:

- Implement S3-compatible HTTP API (future)
- Support multi-part uploads (single stream only)
- Implement versioning (future)

### Module 13: Tensor Checkpoint (Complete)

**Responsibility**: Checkpoint and rollback system for database state recovery.

**Interface**:

- `create(name, store) -> Result<checkpoint_id>` - Create named checkpoint
- `create_auto(command, op, preview, store) -> Result<checkpoint_id>` -
  Auto-checkpoint before destructive op
- `rollback(id_or_name, store) -> Result<()>` - Restore to checkpoint
- `list(limit) -> Result<Vec<CheckpointInfo>>` - List checkpoints
- `delete(id_or_name) -> Result<()>` - Delete checkpoint
- `generate_preview(op, sample_data) -> OperationPreview` - Generate destructive
  op preview

**Features**:

- Manual checkpoints via CHECKPOINT command
- Auto-checkpoints before destructive operations (DELETE, DROP, etc.)
- Rollback to any checkpoint by ID or name
- Count-based retention (configurable max checkpoints)
- Uses tensor_blob for durable storage
- Interactive confirmation with data preview
- Bincode-serialized TensorStore snapshots

**Configuration** (CheckpointConfig):

- `max_checkpoints: usize` - Maximum checkpoints to retain (default: 10)
- `auto_checkpoint: bool` - Enable auto-checkpoints (default: true)
- `interactive_confirm: bool` - Require confirmation for destructive ops
  (default: true)
- `preview_sample_size: usize` - Sample rows to show in preview (default: 5)

**Destructive Operations** (DestructiveOp):

- `Delete` - DELETE FROM table
- `DropTable` - DROP TABLE
- `DropIndex` - DROP INDEX
- `NodeDelete` - NODE DELETE
- `EmbedDelete` - EMBED DELETE
- `VaultDelete` - VAULT DELETE
- `BlobDelete` - BLOB DELETE
- `CacheClear` - CACHE CLEAR

**Does Not**:

- Implement incremental checkpoints (full snapshots only)
- Support distributed checkpoints (single-node)
- Auto-checkpoint on every write (only destructive ops)

### Module 14: Tensor Chain (Complete)

**Responsibility**: Tensor-native blockchain with semantic conflict detection,
hierarchical codebook validation, and distributed consensus.

**Interface**:

- `new(store, node_id) -> TensorChain` - Create chain
- `begin() -> Result<TransactionWorkspace>` - Start transaction
- `commit(workspace) -> Result<BlockHash>` - Commit transaction to block
- `rollback(workspace) -> Result<()>` - Abort transaction
- `height() -> u64` - Current chain height
- `get_block(height) -> Result<Block>` - Get block by height
- `history(key) -> Result<Vec<(u64, Transaction)>>` - Key change history
- `verify() -> Result<()>` - Verify chain integrity

**Features**:

- Semantic transactions with delta embedding tracking
- Hierarchical codebooks (Global static + Local EMA-adaptive)
- Cosine similarity-based conflict detection with configurable metrics
- Orthogonal transaction auto-merge via vector addition
- Tensor-Raft consensus with similarity fast-path
- Two-phase finality (Raft quorum -> checkpointed)
- TCP transport with persistent connections
- Static cluster membership with health checking
- Delta-compressed replication (4-10x bandwidth reduction)
- 2PC distributed transactions with delta-based conflict resolution
- EmbeddingState machine for type-safe transaction embedding lifecycle
- Sparse network messages (8-10x bandwidth reduction for typical deltas)

**Test Coverage**: >95% across all modules (validation.rs: 100%, block.rs:
99.44%, codebook.rs: 99.36%, etc.)

**Does Not**:

- Implement dynamic cluster membership (static config only)
- Support sharding (single chain, future work)
- Persist audit logs (in-memory only)

## Data Flow

### Write Path

```text
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

```text
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

The system is designed for in-memory operation with snapshot persistence and
WAL:

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

This provides significantly better concurrent write throughput compared to a
single RwLock.

### 5. No Schema Enforcement

Tensor Store accepts any `TensorData`. Schema validation belongs in Query Layer.

### 6. Zero-Aware Storage (Sparse/Delta Vectors)

Neumann uses geometric structure to eliminate zero storage overhead:

```text
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

```text
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
    tensor-blob.md           # Module 11 documentation
    tensor-unified.md        # Module 12 documentation
    tensor-checkpoint.md     # Module 13 documentation
    tensor-chain.md          # Module 14 documentation
    benchmarks.md            # Performance benchmarks
  tensor_store/
    Cargo.toml
    src/
      lib.rs                 # Module 1: Storage layer + SparseVector
      hnsw.rs                # HNSW index with Dense/Sparse/Delta support
      delta_vector.rs        # Delta encoding, archetypes, k-means clustering
      instrumentation.rs     # Shard access tracking for hot/cold detection
      mmap.rs                # Memory-mapped cold storage
      tiered.rs              # Two-tier hot/cold store
    benches/
      tensor_store_bench.rs  # Storage, sparse, delta, k-means, tiered benchmarks
  relational_engine/
    Cargo.toml
    src/lib.rs               # Module 2: SQL-like operations
  graph_engine/
    Cargo.toml
    src/lib.rs               # Module 3: Graph operations
  vector_engine/
    Cargo.toml
    src/lib.rs               # Module 4: Similarity search
  tensor_unified/
    Cargo.toml
    src/lib.rs               # Module 12: Cross-engine operations
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
      lib.rs                 # Module 9: Vault API, versioning, namespaces
      encryption.rs          # AES-256-GCM encryption
      key.rs                 # Argon2id key derivation
      access.rs              # Graph-based access control with permissions
      audit.rs               # Audit logging and query API
      rate_limit.rs          # Per-entity rate limiting
      ttl.rs                 # Grant TTL tracking
      obfuscation.rs         # Key obfuscation and padding
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
  tensor_blob/
    Cargo.toml
    src/
      lib.rs                 # Module 11: BlobStore API
      chunker.rs             # SHA-256 content-addressable chunking
      streaming.rs           # BlobWriter, BlobReader
      gc.rs                  # Background garbage collection
      integrity.rs           # Checksum verification, repair
    benches/
      blob_bench.rs          # Blob benchmarks
  tensor_checkpoint/
    Cargo.toml
    src/
      lib.rs                 # Module 13: CheckpointManager API
      state.rs               # CheckpointState, DestructiveOp, metadata
      storage.rs             # Integration with tensor_blob
      retention.rs           # Count-based purge logic
      preview.rs             # Operation preview generation
      error.rs               # Error types
  tensor_chain/
    Cargo.toml
    src/
      lib.rs                 # Module 14: TensorChain API, ChainConfig
      block.rs               # Block, BlockHeader, Transaction types
      chain.rs               # Chain linked via graph edges
      transaction.rs         # Workspace isolation, delta tracking
      embedding.rs           # EmbeddingState machine (Initial, Computed)
      codebook.rs            # GlobalCodebook, LocalCodebook, CodebookManager
      validation.rs          # TransitionValidator, FastPathValidator
      consensus.rs           # Semantic conflict detection, auto-merge (sparse DeltaVector)
      raft.rs                # Tensor-Raft consensus state machine
      network.rs             # Transport trait, MemoryTransport (sparse messages)
      membership.rs          # Cluster membership and health checking
      delta_replication.rs   # Delta-compressed state replication
      distributed_tx.rs      # 2PC coordinator, LockManager
      error.rs               # ChainError types
      tcp/
        mod.rs               # TCP module exports
        config.rs            # TCP transport configuration
        transport.rs         # TcpTransport implementation
        connection.rs        # Connection and ConnectionPool
        framing.rs           # Length-delimited wire protocol
        listener.rs          # Accept loop
        reconnect.rs         # Exponential backoff
        error.rs             # TCP-specific errors
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
5. `cargo llvm-cov` - Coverage check (95% minimum, per-crate thresholds: shell
   88%, parser 91%, blob 91%, router 92%, chain 95%)

### CI Pipeline

Runs on every PR:

1. Check - Compilation verification
2. Format check - Code style
3. Clippy lints - Static analysis
4. Tests - Unit and integration tests
5. Coverage - Minimum 95% per crate (per-crate thresholds for complex modules)
6. Documentation build - Doc generation
7. Security audit - Dependency vulnerabilities
8. Miri - Undefined behavior detection (tensor_store)
9. Fuzz testing - 23 targets including distributed components

## Distributed System Status

### Complete Infrastructure (>95% tested)

| Component | Status | Test Coverage |
| --- | --- | --- |
| Raft consensus | Complete | 98.47% |
| 2PC distributed transactions | Complete | 98.78% |
| TCP transport | Complete | 94.89% |
| TLS encryption | Complete | Enabled by default |
| Cluster membership | Complete | 95.0% |
| Health checking | Complete | 95.0% |
| Delta replication | Complete | 95.28% |
| Semantic conflict detection | Complete | 99.01% |
| Lock manager | Complete | 98.78% |
| Sparse network messages | Complete | 94.51% |
| Shell cluster commands | Complete | Parser + router |

### Integration Points (Wired but not battle-tested)

| Component | Status | Notes |
| --- | --- | --- |
| QueryPlanner with SemanticPartitioner | Wired | Routes by embedding similarity |
| ClusterOrchestrator startup | Implemented | Ties all components together |
| TensorStateMachine | Implemented | Applies Raft entries to chain |

### Recently Completed (Production Infrastructure)

| Component | Status | Notes |
| --- | --- | --- |
| Dynamic cluster membership | Complete | Joint consensus via Raft, learner promotion |
| Gossip protocol | Complete | SWIM-style with LWW CRDT, failure detection |
| Partition detection and merge | Complete | 6-phase protocol with semantic reconciliation |
| Automatic log compaction | Complete | Threshold-based with cooldown |
| Snapshot persistence | Complete | SHA-256 validation, streaming transfer |
| WAL for Raft/2PC | Complete | fsync-durable, crash recovery |
| Deadlock detection | Complete | Wait-for graph with cycle detection |
| Automatic heartbeat | Complete | Background task on leader election |

### Remaining for Production Deployment

| Component | Status | Priority |
| --- | --- | --- |
| Multi-machine validation | Tests use MemoryTransport | High |
| Cross-shard query execution | Infrastructure only | Medium |
| Chaos engineering | Not started | High |
| Performance profiling | Not started | Medium |

### Formal Verification (TLA+)

The three core distributed protocols are formally specified in TLA+
and exhaustively model-checked with the TLC model checker:

| Spec | Distinct States | Depth | Properties |
|------|----------------|-------|------------|
| Raft.tla | 18,268,659 | 54 | ElectionSafety, LogMatching, StateMachineSafety, LeaderCompleteness, VoteIntegrity, TermMonotonicity |
| TwoPhaseCommit.tla | 2,264,939 | 21 | Atomicity, NoOrphanedLocks, ConsistentDecision, VoteIrrevocability, DecisionStability |
| Membership.tla | 54,148 | 17 | NoFalsePositivesSafety, MonotonicEpochs, MonotonicIncarnations |

All three pass with zero errors. TLC output saved in
`specs/tla/tlc-results/`. Model checking found and led to fixes for
protocol-level bugs in Raft log replication (out-of-order message
handling, self-message processing) and the SWIM gossip fairness
specification. See `specs/tla/README.md` for details.

### Honest Assessment

Neumann has **production-ready distributed infrastructure** with all critical
gaps addressed:

- **Raft consensus**: Pre-vote, log compaction, snapshot persistence, WAL
  durability, leadership transfer, automatic heartbeat
- **2PC transactions**: Coordinator/participant, abort broadcast with retry,
  deadlock detection
- **Membership**: Gossip protocol, dynamic membership, partition detection and
  merge
- **Network**: TCP with TLS, LZ4 compression, rate limiting, I/O timeouts

However, it is **not yet battle-tested in production** for these reasons:

1. **All tests use MemoryTransport**: TCP transport is complete but hasn't been
   stress-tested across physical machines.

2. **Query execution is single-node**: The QueryPlanner and SemanticPartitioner
   exist, but queries don't actually scatter to remote nodes and merge results.

3. **No chaos engineering**: Network partition injection, node crash scenarios
   not yet validated.

To become production-ready, Neumann needs:

- Multi-machine integration tests with real TCP
- Chaos engineering (network failures, node crashes, partition scenarios)
- Query scatter-gather wired end-to-end
- Performance testing under sustained load

## Versioning Strategy

Not yet implemented. Future considerations:

- Semantic versioning for crate
- API stability guarantees
- Migration paths for data format changes
