# Changelog

All notable changes to Neumann are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Note**: All 0.x versions are considered unstable.
> Breaking changes may occur between minor versions.

## [Unreleased]

## [0.3.0] - 2026-02-04

### Changed

- **License**: Migrated from dual MIT/Apache-2.0 to Business Source License 1.1
  - Free for personal use, education, evaluation, non-commercial OSS, and small businesses under $5M revenue
  - Automatically converts to Apache-2.0 after 4 years
  - Production use by companies with $5M+ revenue requires commercial license
  - Contact: licensing@tensortech.dev

## [0.1.0] - 2026-01-31

### Added

#### Core Storage (tensor_store)

- Thread-safe key-value storage using DashMap sharded concurrency
- TensorValue types: Scalar, Vector, Sparse, Pointer, Pointers
- Bloom filter for fast negative lookups with configurable FPR
- HNSW index for O(log n) approximate nearest neighbor search
- SparseVector for memory-efficient high-sparsity embeddings (3-33x compression)
- DeltaVector with archetype-based encoding and k-means clustering
- Tiered storage with hot (in-memory) and cold (mmap) tiers
- Snapshot persistence with bincode serialization
- Compressed snapshots (Tensor Train decomposition, delta encoding, RLE)
- Write-ahead log (WAL) for crash recovery
- Consistent hashing partitioner with virtual nodes
- Voronoi and k-means semantic partitioners

#### Query Engines

- **Relational Engine**: Tables, schemas, SQL-like operations, B-tree indexes, SIMD filtering
- **Graph Engine**: Nodes, edges, traversals, path finding, BFS, shortest path
- **Vector Engine**: Embedding storage, k-NN search, 15+ distance metrics

#### Extended Modules

- **Tensor Compress**: Tensor Train decomposition (10-20x compression), int8/binary quantization, delta encoding, RLE
- **Tensor Vault**: AES-256-GCM encryption, Argon2id key derivation, graph-based access control, TTL grants, rate limiting, audit logging
- **Tensor Cache**: Multi-layer LLM response caching (exact + semantic + embedding), tiktoken integration, cost tracking, background eviction
- **Tensor Blob**: S3-style chunked blob storage, SHA-256 content addressing, streaming upload/download, entity linking, tagging, garbage collection
- **Tensor Checkpoint**: Atomic snapshot/restore with retention policies and interactive confirmation
- **Tensor Unified**: Cross-engine unified entity operations

#### Distributed Consensus (tensor_chain)

- Tensor-Raft consensus with semantic similarity fast-path
- 6-way geometric conflict detection
- 2PC coordinator with deadlock detection via wait-for graph
- SWIM gossip protocol with signed messages
- Delta-compressed replication (4-6x compression)
- Partition detection and healing
- Ed25519 block signatures

#### Query Language (neumann_parser)

- Hand-written recursive descent parser
- SQL: SELECT, INSERT, UPDATE, DELETE, CREATE/DROP TABLE, JOINs, GROUP BY
- Graph: NODE, EDGE, NEIGHBORS, PATH, FIND
- Vector: EMBED, SIMILAR
- Vault: SET, GET, DELETE, LIST, ROTATE, GRANT, REVOKE
- Cache: INIT, STATS, CLEAR, EVICT
- Blob: PUT, GET, DELETE, INFO, LINK, TAG, GC, REPAIR
- Cluster: CONNECT, DISCONNECT, STATUS

#### Shell (neumann_shell)

- Interactive REPL with readline support
- TRO (Tensor Rust Organism) animated border
- Phosphor-green retro terminal aesthetic
- Command history persistence
- ASCII table output formatting
- Built-in commands: help, tables, save, load, wal status
- **CLI argument parsing**:
  - `-c/--command`: Execute single query and exit
  - `-f/--file`: Execute queries from file
  - `-o/--output`: Output format (table, json, csv)
  - `--no-color`: Disable colored output
  - `--no-boot`: Skip boot sequence animation
  - `-q/--quiet`: Suppress non-essential output
- **Shell completions**: Bash, Zsh, Fish, PowerShell
- **Man page**: `man neumann`

#### Distribution

- Cross-platform binaries: Linux x64, macOS x64/ARM64, Windows x64
- Shell install script with auto-detection
- Homebrew formula

### Security

- AES-256-GCM for secret encryption
- Argon2id for key derivation (GPU/ASIC resistant)
- Ed25519 signatures for distributed consensus
- Secure key zeroization on drop
- Graph-based access control with permission levels
- Rate limiting per entity
- `forbid(unsafe_code)` in most crates

### Performance

- DashMap sharded storage for concurrent access
- HNSW index: O(log n) similarity search
- Sparse vectors: 10-150x dot product speedup at 99% sparsity
- Tiered storage: 5-7% overhead, ~1M entries/sec migration
- SIMD-accelerated filtering in relational engine
- 95%+ test coverage across all crates

### Known Limitations

- Single-node only (distributed mode experimental)
- In-memory first (snapshots for persistence)
- API unstable (0.x version)

---

## Version History

- **0.1.0** - Initial public release

[Unreleased]: https://github.com/Shadylukin/Neumann/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Shadylukin/Neumann/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/Shadylukin/Neumann/releases/tag/v0.1.0
