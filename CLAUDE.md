# CLAUDE.md

This file provides guidance for Claude Code when working on this project.

## Project Overview

Neumann is a unified tensor-based runtime that stores relational data, graph
relationships, and vector embeddings in a single mathematical structure.

## Quality Standards

This project maintains **production-grade code quality**. All code must pass:

```bash
cargo fmt --check                              # Formatting
cargo clippy -- -D warnings                    # Lints as errors
cargo clippy -- -W clippy::pedantic            # Pedantic lints (advisory)
cargo test                                     # All tests pass
cargo doc --no-deps                            # Documentation builds
```

**Coverage requirements** (enforced per-crate):

- Default: 95% minimum line coverage
- neumann_shell: 88%
- neumann_parser: 91%
- tensor_blob: 91%
- query_router: 92%
- tensor_chain: 95%

**Code expectations:**

- Clean, idiomatic Rust with no hacks or workarounds
- Proper error handling using `Result` and `?` propagation
- Thread-safe designs using sharded data structures (DashMap, parking_lot)
- Comprehensive tests including unit, integration, concurrency, and fuzz tests
- No `unsafe` code unless absolutely necessary and well-justified

## Workspace Structure

The project consists of 19 crates organized in dependency tiers:

### Foundation Layer (no workspace dependencies)

| Crate | Purpose |
| ----- | ------- |
| `tensor_store` | Core key-value storage with HNSW, sparse vectors, tiered storage |
| `tensor_compress` | Tensor Train decomposition, delta encoding, RLE compression |
| `neumann_parser` | Hand-written recursive descent parser for query language |

### Engine Layer (depends on tensor_store)

| Crate | Purpose |
| ----- | ------- |
| `relational_engine` | SQL-like tables with SIMD filtering, indexes, columnar scans |
| `graph_engine` | Directed graphs with BFS traversal, shortest path, properties |
| `vector_engine` | k-NN similarity search via HNSW with multiple distance metrics |

### Specialized Storage Layer

| Crate | Purpose | Dependencies |
| ----- | ------- | ------------ |
| `tensor_vault` | AES-256-GCM encrypted secrets with graph-based access control | tensor_store, graph_engine |
| `tensor_cache` | Multi-layer LLM response cache (exact + semantic + embedding) | tensor_store |
| `tensor_blob` | S3-style content-addressable blob storage with streaming | tensor_store |
| `tensor_checkpoint` | Atomic snapshot/restore with retention and confirmation | tensor_store, tensor_blob |
| `tensor_unified` | Cross-engine unified entity operations | all engines |

### Distributed Layer

| Crate | Purpose | Dependencies |
| ----- | ------- | ------------ |
| `tensor_chain` | Tensor-native blockchain with Raft consensus and 2PC | tensor_store, tensor_compress, graph_engine |

### Query Execution Layer

| Crate | Purpose | Dependencies |
| ----- | ------- | ------------ |
| `query_router` | Unified query routing across all engines | all crates |
| `neumann_shell` | Interactive CLI with readline, WAL, snapshots | query_router |
| `neumann_server` | gRPC server exposing QueryRouter | query_router, tensor_blob |

### Testing and Utilities

| Crate | Purpose |
| ----- | ------- |
| `integration_tests` | Cross-crate integration tests (267+ tests) |
| `stress_tests` | Performance and concurrency stress tests |
| `experiments` | Research and experimental features |
| `seed_model` | Geometric intelligence model implementation |

## Code Style

- No emojis in code, comments, or commit messages
- Use Rust idioms: prefer iterators over loops, use `?` for error propagation
- Keep functions small and focused
- Prefer composition over inheritance patterns

## Comments Policy

Doc comments (`///`) are for rustdoc generation. Use them sparingly:

**DO document:**

- Types (structs, enums) - explain purpose and invariants
- Non-obvious behavior - when a method does something unexpected
- Complex algorithms - when the "why" isn't clear from code

**DO NOT document:**

- Methods with self-explanatory names (`get`, `set`, `new`, `len`, `is_empty`)
- Trivial implementations
- Anything where the doc would just repeat the function name

**Examples:**

```rust
// BAD - restates the obvious
/// Get a field value
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - no comment needed, name is clear
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - explains non-obvious behavior
/// Returns cloned data to ensure thread safety. For zero-copy access, use get_ref().
pub fn get(&self, key: &str) -> Result<TensorData>
```

Inline comments (`//`) should explain "why", never "what".

## Testing Philosophy

- Unit tests live in the same file as the code (`#[cfg(test)]` module)
- Test the public API, not implementation details
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include edge cases: empty inputs, boundaries, error conditions
- Performance tests for operations that must scale (10k+ entities)
- Concurrent tests for thread-safe code

## Fuzzing

The project uses cargo-fuzz (libFuzzer-based) for coverage-guided fuzzing.

### Fuzz Targets

| Target | Module | What it tests |
| ------ | ------ | ------------- |
| `parser_parse` | neumann_parser | Statement parsing |
| `parser_parse_all` | neumann_parser | Multi-statement parsing |
| `parser_parse_expr` | neumann_parser | Expression parsing |
| `parser_tokenize` | neumann_parser | Lexer/tokenization |
| `compress_ids` | tensor_compress | Varint ID compression |
| `compress_rle` | tensor_compress | RLE encode/decode |
| `compress_snapshot` | tensor_compress | Snapshot serialization |
| `vault_cipher` | tensor_vault | AES-256-GCM roundtrip |
| `checkpoint_state` | tensor_checkpoint | Checkpoint bincode |
| `storage_sparse_vector` | tensor_store | Sparse vector roundtrip |
| `slab_entity_index` | tensor_store | EntityIndex operations |
| `consistent_hash` | tensor_store | Consistent hash partitioner |
| `tcp_framing` | tensor_chain | TCP wire protocol codec |
| `membership` | tensor_chain | Cluster config serialization |
| `relational_condition` | relational_engine | Condition evaluation |
| `relational_engine_ops` | relational_engine | Engine CRUD operations |
| `cache_eviction_scorer` | tensor_cache | Eviction strategy scoring |
| `cache_semantic_search` | tensor_cache | Semantic search with metrics |
| `cache_metric_roundtrip` | tensor_cache | Metric consistency |
| `archetype_registry` | tensor_store | ArchetypeRegistry bincode |
| `dtx_state_cleanup` | tensor_chain | Distributed tx cleanup |
| `error_hierarchy` | integration_tests | Error type hierarchy |

### Running Locally

```bash
# Install cargo-fuzz (requires nightly)
cargo install cargo-fuzz

# List available targets
cd fuzz && cargo +nightly fuzz list

# Run a specific target for 60 seconds
cargo +nightly fuzz run parser_parse -- -max_total_time=60

# Run without sanitizer (2x faster for safe Rust)
cargo +nightly fuzz run parser_parse --sanitizer none

# Reproduce a crash
cargo +nightly fuzz run parser_parse artifacts/parser_parse/crash-xxx
```

### Adding New Fuzz Targets

1. Create target file in `fuzz/fuzz_targets/<name>.rs`
2. Add `[[bin]]` entry to `fuzz/Cargo.toml`
3. Add seed corpus files to `fuzz/corpus/<name>/`
4. Update CI matrix in `.github/workflows/fuzz.yml`

## Architecture

See `docs/architecture.md` for full system design.

### tensor_store (23,625 lines, 22 modules)

Core storage layer with specialized slabs and partitioning.

```text
src/lib.rs              # TensorStore, TensorData, TensorValue, BloomFilter, EntityStore
src/slab_router.rs      # BTreeMap-based multi-slab routing (~3.2M PUT, ~5M GET ops/sec)
src/hnsw.rs             # HNSW index (Dense/Sparse/Delta/TensorTrain support)
src/sparse_vector.rs    # Memory-efficient sparse vectors with 15+ distance metrics
src/delta_vector.rs     # Delta encoding, ArchetypeRegistry, k-means clustering
src/relational_slab.rs  # Column-oriented table storage with indexing
src/graph_tensor.rs     # Graph nodes/edges with BFS and shortest path
src/embedding_slab.rs   # Entity-to-embedding mapping with compression
src/entity_index.rs     # Bidirectional String <-> EntityId mapping
src/metadata_slab.rs    # Arbitrary key-value metadata storage
src/blob_log.rs         # Content-addressable chunk storage with GC
src/cache_ring.rs       # LRU/LFU/Cost/Hybrid eviction cache
src/tiered.rs           # Two-tier hot/cold storage with auto-migration
src/mmap.rs             # Memory-mapped cold storage (builder + mutable)
src/instrumentation.rs  # Shard access tracking for hot/cold detection
src/consistent_hash.rs  # Consistent hashing with virtual nodes
src/voronoi.rs          # Voronoi diagram-based vector partitioning
src/semantic_partitioner.rs  # K-means semantic partitioning
src/partitioned.rs      # Partition-aware store wrapper
src/partitioner.rs      # Pluggable partitioning trait
src/distance.rs         # DistanceMetric enum + GeometricConfig
src/snapshot.rs         # V2/V3 format detection and migration
```

### tensor_compress

```text
src/lib.rs        # TTVector, TTConfig, CompressionConfig
src/tt.rs         # Tensor Train decomposition (10-20x compression)
src/delta.rs      # Delta + varint encoding for sorted IDs
src/rle.rs        # Run-length encoding
src/format.rs     # Snapshot format versioning
```

### relational_engine

```text
src/lib.rs        # RelationalEngine, Schema, Column, Condition
src/simd.rs       # SIMD-accelerated filtering (wide crate)
```

### graph_engine

```text
src/lib.rs        # GraphEngine, Node, Edge, Path, Direction
```

### vector_engine

```text
src/lib.rs        # VectorEngine, SearchResult, DistanceMetric
```

### tensor_vault

```text
src/lib.rs          # Vault API with versioning and namespaces
src/encryption.rs   # AES-256-GCM authenticated encryption
src/key.rs          # Argon2id key derivation (GPU/ASIC resistant)
src/access.rs       # Graph-based access control with permissions
src/audit.rs        # Audit logging
src/rate_limit.rs   # Per-entity rate limiting
src/ttl.rs          # Grant TTL tracking
src/obfuscation.rs  # Key obfuscation via HMAC
```

### tensor_cache

```text
src/lib.rs        # Cache API with multi-layer lookup
src/config.rs     # Configuration and presets
src/exact.rs      # O(1) hash-based exact cache
src/semantic.rs   # O(log n) HNSW-based semantic cache
src/embedding.rs  # Embedding cache with content hashing
src/eviction.rs   # Background eviction (LRU/LFU/Cost/Hybrid)
src/ttl.rs        # TTL tracking
src/tokenizer.rs  # tiktoken token counting
src/stats.rs      # Hit rates and cost tracking
```

### tensor_blob

```text
src/lib.rs        # BlobStore API (async-first)
src/chunker.rs    # SHA-256 content-addressable chunking
src/streaming.rs  # BlobWriter, BlobReader for streaming I/O
src/metadata.rs   # ArtifactMetadata with tags and links
src/gc.rs         # Background garbage collection
src/integrity.rs  # Checksum verification and repair
```

### tensor_checkpoint

```text
src/lib.rs        # CheckpointManager with confirmation workflow
src/storage.rs    # Checkpoint storage via BlobStore
src/retention.rs  # Automatic old checkpoint cleanup
src/preview.rs    # Operation preview generation
```

### tensor_unified

```text
src/lib.rs        # UnifiedEngine for cross-engine operations
```

### tensor_chain (51,977 lines, 42 modules)

Production-ready distributed consensus with semantic transactions.

```text
src/lib.rs              # TensorChain, ChainConfig, ChainMetrics
src/raft.rs             # Tensor-Raft consensus (7,684 lines)
src/distributed_tx.rs   # 2PC coordinator with deadlock detection
src/network.rs          # Transport trait, message types (9 categories)
src/tcp/                # TCP transport with TLS, compression, rate limiting
src/membership.rs       # Health checking, cluster config, partition detection
src/gossip.rs           # SWIM-based gossip protocol with signed messages
src/consensus.rs        # Semantic conflict detection (6-way classification)
src/codebook.rs         # Global/local codebooks, hierarchical quantization
src/validation.rs       # State transition validation
src/delta_replication.rs  # Delta-compressed replication (4-6x compression)
src/partition_merge.rs  # Partition healing and reconciliation
src/raft_wal.rs         # Persistent Raft state
src/tx_wal.rs           # Persistent 2PC state
src/snapshot_streaming.rs  # Chunked snapshot transfer
src/signing.rs          # Ed25519 signatures, validator registry
src/deadlock.rs         # Wait-for graph deadlock detection
src/hlc.rs              # Hybrid logical clocks
```

### neumann_parser

```text
src/lib.rs        # Public API (parse, parse_all, parse_expr, tokenize)
src/lexer.rs      # Tokenization
src/token.rs      # Token definitions
src/ast.rs        # AST node types (Statement, Expr)
src/parser.rs     # Statement parsing
src/expr.rs       # Expression parsing (Pratt precedence)
src/span.rs       # Source locations
src/error.rs      # Parse errors
```

### query_router

```text
src/lib.rs        # QueryRouter, QueryResult, distributed query support
```

### neumann_shell

```text
src/lib.rs        # Shell, ShellConfig, CommandResult
src/main.rs       # Binary entry point
src/wal.rs        # Write-ahead log for crash recovery
```

### neumann_server

```text
src/lib.rs        # gRPC server via tonic
src/main.rs       # Server binary
proto/            # Protocol buffer definitions
```

## Key Types

### Tensor Store

- `TensorValue`: Scalar | Vector | Sparse | Pointer | Pointers
- `TensorData`: HashMap-based entity with field accessors
- `ScalarValue`: Null | Bool | Int | Float | String | Bytes
- `TensorStore`: Thread-safe key-value store with SlabRouter
- `EntityStore`: High-level entity abstraction with type-aware scanning
- `BloomFilter`: Thread-safe probabilistic set with configurable FPR
- `SparseVector`: Memory-efficient sparse embedding (15+ distance metrics)
- `DeltaVector`: Archetype-based delta encoding with k-means
- `HNSWIndex`: Hierarchical navigable small world graph (Dense/Sparse/Delta/TT)
- `DistanceMetric`: Cosine, Angular, Geodesic, Jaccard, Overlap, Euclidean, Composite
- `TieredStore`: Hot/cold storage with mmap backing
- `CacheRing`: Multi-strategy eviction cache (LRU/LFU/Cost/Hybrid)
- `ConsistentHashPartitioner`: Consistent hashing with virtual nodes
- `VoronoiPartitioner`: Vector-based Voronoi region partitioning
- `SemanticPartitioner`: K-means semantic partitioning with routing

### Relational Engine

- `Schema`, `Column`, `ColumnType`: Table structure (Int, Float, String, Bool,
  Bytes, Json)
- `Value`: Typed values for queries
- `Condition`, `RangeOp`: Composable predicates for filtering
- `RelationalEngine`: Table CRUD with SIMD-accelerated scans

### Graph Engine

- `Node`, `Edge`: Graph elements with properties
- `Direction`: Outgoing, Incoming, Both
- `Path`: Sequence of nodes and edges
- `GraphEngine`: Node/edge CRUD, BFS, shortest path

### Vector Engine

- `SearchResult`: Key and similarity score
- `DistanceMetric`: Cosine, Euclidean, DotProduct
- `VectorEngine`: Embedding storage and k-NN search via HNSW

### Tensor Vault

- `Vault`: Encrypted secret storage with graph-based access
- `VaultConfig`: Argon2id parameters, rate limits, max versions
- `Permission`: Read, Write, Admin
- `Cipher`: AES-256-GCM wrapper
- `MasterKey`: Derived key with zeroize on drop
- `Obfuscator`: HMAC-based key obfuscation

### Tensor Cache

- `Cache`: Multi-layer LLM response cache
- `CacheConfig`: Embedding dimension, capacity, TTL, eviction strategy
- `CacheHit`: Successful lookup (layer, response, tokens, cost)
- `CacheStats`: Hit rates, token counts, cost savings
- `EvictionStrategy`: LRU, LFU, Cost, Hybrid

### Tensor Blob

- `BlobStore`: Content-addressable artifact storage (async)
- `BlobConfig`: Chunk size, GC settings
- `ArtifactMetadata`: Filename, size, checksum, links, tags
- `BlobWriter`, `BlobReader`: Streaming upload/download
- `GarbageCollector`: Background cleanup

### Tensor Checkpoint

- `CheckpointManager`: Snapshot/restore coordinator
- `CheckpointConfig`: Max checkpoints, auto-checkpoint, interactive confirm
- `DestructiveOp`: Delete, Update operations requiring confirmation
- `OperationPreview`: Preview of affected data
- `ConfirmationHandler`: Trait for custom confirmation logic

### Tensor Unified

- `UnifiedEngine`: Cross-engine query execution
- `UnifiedResult`: Combined results from multiple engines
- `UnifiedItem`: Entity with source, data, embedding, score
- `FindPattern`: Nodes(label) | Edges(type)

### Neumann Parser

- `Statement`, `StatementKind`: Top-level parsed statements
- `Expr`, `ExprKind`: Expression AST nodes
- `Token`, `TokenKind`: Lexer tokens
- `ParseError`: Error with span information

### Query Router

- `QueryRouter`: Unified query execution across all engines
- `QueryResult`: Result variants (Rows, Nodes, Edges, Similar, Chain, etc.)
- `QueryPlanner`: Distributed query optimization
- `ResultMerger`: Combine results from multiple shards

### Tensor Chain

- `TensorChain`: Tensor-native blockchain API
- `Chain`, `Block`, `BlockHeader`: Block structure with Ed25519 signatures
- `Transaction`: Put, Delete, Update operations
- `RaftNode`: Tensor-Raft consensus state machine
- `RaftConfig`: Election timeout, heartbeat, similarity threshold, fast-path
- `DistributedTxCoordinator`: 2PC with lock manager
- `LockManager`: Key-level locking with deadlock detection
- `ConsensusManager`: Semantic conflict detection (6-way classification)
- `DeltaVector`: Sparse delta for conflict detection
- `GlobalCodebook`, `LocalCodebook`: Quantization codebooks
- `MembershipManager`: Health checking, partition detection
- `DeltaReplicationManager`: Delta-compressed state sync

### Neumann Shell

- `Shell`: Interactive REPL with query execution
- `ShellConfig`: History file, size, prompt
- `CommandResult`: Output, Exit, Help, Empty, Error
- `Wal`: Write-ahead log for crash recovery

## Concurrency Design

All engines inherit thread safety from TensorStore's SlabRouter:

- **Why**: Better concurrent write performance, no lock poisoning
- **How**: SlabRouter uses sharded BTreeMaps, writes only block same-shard writes
- **Trade-off**: Adds dashmap/parking_lot dependencies, but eliminates failure modes

When adding new concurrent data structures:

1. Prefer sharded/partitioned designs over single locks
2. Avoid lock poisoning by using parking_lot or dashmap
3. Always add concurrent tests (`store_concurrent_*`)
4. Document the concurrency model in doc comments

## Commit Guidelines

- Write clear, imperative commit messages
- No emoji in commits
- Reference issue numbers when applicable
- Keep commits atomic - one logical change per commit
