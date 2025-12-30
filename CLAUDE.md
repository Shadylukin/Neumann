# CLAUDE.md

This file provides guidance for Claude Code when working on this project.

## Project Overview

Neumann is a unified tensor-based runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure.

## Modules

| Module | Purpose | Depends On |
|--------|---------|------------|
| `tensor_store` | Key-value storage layer | tensor_compress |
| `relational_engine` | SQL-like tables with indexes | tensor_store |
| `graph_engine` | Graph nodes and edges | tensor_store |
| `vector_engine` | Embeddings and similarity search | tensor_store |
| `tensor_compress` | Compression algorithms | - |
| `tensor_vault` | Encrypted secret storage | tensor_store, graph_engine |
| `tensor_cache` | Semantic LLM response caching | tensor_store |
| `tensor_blob` | S3-style chunked blob storage | tensor_store |
| `tensor_checkpoint` | Atomic snapshot/restore | tensor_store |
| `tensor_unified` | Multi-engine unified storage | all engines |
| `tensor_chain` | Tensor-native blockchain | tensor_store, graph_engine, tensor_checkpoint |
| `neumann_parser` | Query tokenization and parsing | - |
| `query_router` | Unified query execution | all engines, parser, vault, cache, blob, chain |
| `neumann_shell` | Interactive CLI interface | query_router |

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

## Quality Standards

All code must pass before commit:
- `cargo fmt --check` - formatting
- `cargo clippy -- -D warnings` - lints as errors
- `cargo test` - all tests pass
- `cargo doc --no-deps` - documentation builds
- 95% minimum line coverage (per-crate thresholds: shell 88%, parser 91%, blob 91%, router 92%, chain 95%)

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
|--------|--------|---------------|
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
| `relational_condition` | relational_engine | Condition evaluate() vs evaluate_tensor() |
| `relational_engine_ops` | relational_engine | Engine CRUD operations |
| `cache_eviction_scorer` | tensor_cache | Eviction strategy scoring |

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

```
tensor_store/           # Module 1: Storage layer
  src/lib.rs            # Core types, TensorStore, SparseVector, BloomFilter
  src/hnsw.rs           # HNSW index (Dense/Sparse/Delta support)
  src/delta_vector.rs   # Delta encoding, ArchetypeRegistry, k-means
  src/instrumentation.rs # Shard access tracking for hot/cold detection
  src/mmap.rs           # Memory-mapped cold storage
  src/tiered.rs         # Two-tier hot/cold TieredStore
relational_engine/      # Module 2: Relational operations
  src/lib.rs            # Tables, schemas, conditions, indexes
graph_engine/           # Module 3: Graph operations
  src/lib.rs            # Nodes, edges, traversals
vector_engine/          # Module 4: Vector operations
  src/lib.rs            # Embeddings, similarity search
tensor_compress/        # Module 8: Compression algorithms
  src/lib.rs            # Public API
  src/quantize.rs       # Int8/binary vector quantization
  src/delta.rs          # Delta + varint encoding
  src/rle.rs            # Run-length encoding
  src/format.rs         # Snapshot format v2
tensor_vault/           # Module 9: Secret storage
  src/lib.rs            # Vault API, versioning, namespaces
  src/encryption.rs     # AES-256-GCM encryption
  src/key.rs            # Argon2id key derivation
  src/access.rs         # Graph-based access control with permissions
  src/audit.rs          # Audit logging
  src/rate_limit.rs     # Per-entity rate limiting
  src/ttl.rs            # Grant TTL tracking
  src/obfuscation.rs    # Key obfuscation via HMAC
tensor_cache/           # Module 10: LLM response cache
  src/lib.rs            # Cache API, multi-layer lookup
  src/config.rs         # Configuration and presets
  src/error.rs          # Error types
  src/stats.rs          # Statistics tracking
  src/exact.rs          # O(1) hash-based cache
  src/semantic.rs       # O(log n) HNSW-based cache
  src/embedding.rs      # Embedding cache
  src/index.rs          # HNSW index wrapper
  src/eviction.rs       # Background eviction
  src/ttl.rs            # TTL tracking
  src/tokenizer.rs      # tiktoken token counting
tensor_blob/            # Module 11: Blob storage
  src/lib.rs            # BlobStore API
  src/config.rs         # Configuration
  src/error.rs          # Error types
  src/metadata.rs       # Artifact metadata
  src/chunker.rs        # SHA-256 content-addressable chunking
  src/streaming.rs      # BlobWriter, BlobReader
  src/gc.rs             # Background garbage collection
  src/integrity.rs      # Checksum verification, repair
tensor_chain/           # Module 12: Tensor blockchain
  src/lib.rs            # TensorChain API, ChainConfig
  src/block.rs          # Block, BlockHeader, Transaction
  src/chain.rs          # Chain linked via graph edges
  src/transaction.rs    # Workspace isolation, delta tracking
  src/embedding.rs      # EmbeddingState machine (Initial, Computed)
  src/codebook.rs       # GlobalCodebook, LocalCodebook, CodebookManager
  src/validation.rs     # TransitionValidator
  src/consensus.rs      # Semantic conflict detection, auto-merge (sparse DeltaVector)
  src/raft.rs           # Tensor-Raft consensus
  src/network.rs        # Transport trait, MemoryTransport (sparse messages)
  src/distributed_tx.rs # 2PC coordinator, LockManager
  src/membership.rs     # Cluster membership and health checking
  src/delta_replication.rs # Delta-compressed state replication
  src/error.rs          # ChainError types
neumann_parser/         # Module 5: Query parsing
  src/lib.rs            # Public API
  src/lexer.rs          # Tokenization
  src/token.rs          # Token definitions
  src/ast.rs            # AST node types
  src/parser.rs         # Statement parsing
  src/expr.rs           # Expression parsing (Pratt)
  src/span.rs           # Source locations
  src/error.rs          # Error types
query_router/           # Module 6: Query execution
  src/lib.rs            # Unified query routing
neumann_shell/          # Module 7: CLI interface
  src/lib.rs            # Shell implementation, WAL
  src/main.rs           # Binary entry point
docs/
  architecture.md       # System architecture overview
  tensor-store.md       # Module 1 API documentation
  relational-engine.md  # Module 2 API documentation
  graph-engine.md       # Module 3 API documentation
  vector-engine.md      # Module 4 API documentation
  query-router.md       # Module 5 API documentation
  neumann-parser.md     # Module 6 API documentation
  neumann-shell.md      # Module 7 API documentation
  tensor-compress.md    # Module 8 API documentation
  tensor-vault.md       # Module 9 API documentation
  tensor-cache.md       # Module 10 API documentation
  tensor-blob.md        # Module 11 API documentation
  tensor-chain.md       # Module 12 API documentation
  benchmarks.md         # Performance benchmarks
```

## Key Types

### Tensor Store
- `TensorValue`: Scalar, Vector, Pointer, or Pointers
- `TensorData`: A map of field names to TensorValues
- `ScalarValue`: Int, Float, String, Bool, Bytes, Null
- `TensorStore`: Thread-safe key-value store using DashMap
- `BloomFilter`: Probabilistic set for fast negative lookups
- `SparseVector`: Memory-efficient sparse embedding with geometric operations
  - Arithmetic: `sub`, `weighted_average`, `project_orthogonal`, `from_diff`
  - Metrics: `cosine_similarity`, `jaccard_index`, `euclidean_distance`, `angular_distance`
- `DistanceMetric`: Cosine, Angular, Geodesic, Jaccard, Overlap, Euclidean, Composite
- `HNSWIndex`: Hierarchical navigable small world graph
- `TieredStore`: Hot/cold storage with mmap backing

### Relational Engine
- `Schema`, `Column`, `ColumnType`: Table structure
- `Value`: Typed values (Int, Float, String, Bool, Null)
- `Condition`: Composable predicates for filtering
- `RelationalEngine`: Table operations with index support

### Graph Engine
- `Node`, `Edge`: Graph elements with id, label/type, properties
- `Direction`: Edge traversal direction (Outgoing, Incoming, Both)
- `Path`: Sequence of nodes and edges
- `GraphEngine`: Node/edge CRUD and traversals

### Vector Engine
- `SearchResult`: Key and similarity score
- `VectorEngine`: Embedding storage and k-NN search

### Neumann Shell
- `Shell`: Interactive REPL with query execution
- `ShellConfig`: Configuration (history, prompt)
- `CommandResult`: Output, Exit, Help, Empty, Error

### Tensor Vault
- `Vault`: Encrypted secret storage with graph-based access
- `VaultConfig`: Argon2id parameters for key derivation
- `VaultError`: Access denied, not found, crypto errors
- `Permission`: Access levels (Read, Write, Admin)
- `Cipher`: AES-256-GCM encryption wrapper
- `MasterKey`: Derived key with zeroize on drop

### Tensor Cache
- `Cache`: Multi-layer LLM response cache
- `CacheConfig`: Configuration and presets
- `CacheHit`: Successful cache lookup result
- `CacheStats`: Hit rates, token counts, cost savings
- `EvictionStrategy`: LRU, LFU, Cost, Hybrid

### Tensor Blob
- `BlobStore`: Content-addressable artifact storage
- `BlobConfig`: Chunk size, GC settings
- `ArtifactMetadata`: Filename, size, checksum, links, tags
- `PutOptions`: Upload options (links, tags, metadata)
- `BlobWriter`, `BlobReader`: Streaming upload/download

### Neumann Parser
- `Statement`: Top-level parsed statement
- `StatementKind`: Select, Insert, Node, Edge, Blob, Vault, Cache, etc.
- `Expr`, `ExprKind`: Expression AST nodes
- `Token`, `TokenKind`: Lexer tokens
- `ParseError`: Error with span information

### Query Router
- `QueryRouter`: Unified query execution across all engines
- `QueryResult`: Result variants (Rows, Nodes, Edges, Count, etc.)
- `RouterError`: Query execution errors

### Tensor Chain
- `TensorChain`: Tensor-native blockchain with semantic transactions
- `Block`, `BlockHeader`: Block structure with delta embeddings
- `Transaction`: Put, Delete, Update operations
- `GlobalCodebook`: Static codebook for consensus validation
- `LocalCodebook`: EMA-adaptive codebook per domain
- `ConsensusManager`: Semantic conflict detection and auto-merge
- `DeltaVector`: Sparse delta for conflict detection (uses SparseVector internally)
- `EmbeddingState`: Type-safe embedding lifecycle (Initial, Computed)
- `WorkspaceEmbedding`: Transaction embedding tracking via EmbeddingState
- `RaftNode`: Tensor-Raft consensus state machine
- `ChainError`: Chain operation errors
- Network messages (RequestVote, AppendEntries, TxPrepareMsg, TxVote): Use SparseVector

## Concurrency Design

All engines inherit thread safety from TensorStore's DashMap:

- **Why**: Better concurrent write performance, no lock poisoning
- **How**: DashMap uses ~16 shards, writes only block same-shard writes
- **Trade-off**: Adds dashmap dependency, but eliminates failure modes

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
