# Changelog

All notable changes to Neumann are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Note**: All 0.x versions are considered unstable.
> Breaking changes may occur between minor versions.

## [Unreleased]

## [0.1.0] - 2024-12-31

### Added

#### Core Storage (tensor_store)

- Thread-safe key-value storage using DashMap
- TensorValue types: Scalar, Vector, Pointer, Pointers
- Bloom filter for fast negative lookups
- HNSW index for O(log n) approximate nearest neighbor search
- SparseVector for memory-efficient high-sparsity embeddings (3-33x compression)
- DeltaVector with archetype-based encoding and k-means clustering
- Tiered storage with hot (in-memory) and cold (mmap) tiers
- Snapshot persistence with bincode serialization
- Compressed snapshots (int8 quantization, delta encoding, RLE)
- Write-ahead log (WAL) for crash recovery

#### Query Engines

- **Relational Engine**: Tables, schemas, SQL-like operations, B-tree indexes
- **Graph Engine**: Nodes, edges, traversals, path finding, BFS
- **Vector Engine**: Embedding storage, k-NN search, distance metrics

#### Extended Modules

- **Tensor Compress**: Int8/binary quantization, delta encoding, RLE
- **Tensor Vault**: AES-256-GCM encryption, Argon2id key derivation,
  graph-based access control, TTL grants, rate limiting, audit logging
- **Tensor Cache**: Multi-layer LLM response caching (exact + semantic),
  tiktoken integration, cost tracking, background eviction
- **Tensor Blob**: S3-style chunked blob storage, SHA-256 content addressing,
  streaming upload/download, entity linking, tagging, garbage collection

#### Query Language (neumann_parser)

- Hand-written recursive descent parser
- SQL: SELECT, INSERT, UPDATE, DELETE, CREATE/DROP TABLE, JOINs, GROUP BY
- Graph: NODE, EDGE, NEIGHBORS, PATH, FIND
- Vector: EMBED, SIMILAR
- Vault: SET, GET, DELETE, LIST, ROTATE, GRANT, REVOKE
- Cache: INIT, STATS, CLEAR, EVICT
- Blob: PUT, GET, DELETE, INFO, LINK, TAG, GC, REPAIR

#### Shell (neumann_shell)

- Interactive REPL with readline support
- Command history persistence
- ASCII table output formatting
- Built-in commands: help, tables, save, load, wal status

### Security

- AES-256-GCM for secret encryption
- Argon2id for key derivation
- Secure key zeroization on drop
- Graph-based access control with permission levels

### Performance

- DashMap sharded storage for concurrent access
- HNSW index: O(log n) similarity search
- Sparse vectors: 10-150x dot product speedup at 99% sparsity
- Tiered storage: 5-7% overhead, ~1M entries/sec migration
- 95%+ test coverage across all crates

### Known Limitations

- Single-node only (no distributed queries)
- In-memory first (snapshots for persistence)
- API unstable (0.x version)

---

## Version History

- **0.1.0** - Initial public release

[Unreleased]: https://github.com/Shadylukin/Neumann/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shadylukin/Neumann/releases/tag/v0.1.0
