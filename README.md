# Neumann

[![CI](https://github.com/Shadylukin/Neumann/workflows/CI/badge.svg)](https://github.com/Shadylukin/Neumann/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/uN3KbAyKvw)

A unified runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure — the tensor.

Instead of spinning up Postgres, Neo4j, Qdrant, and Redis separately, you spin up Neumann.

> Code and data live together. Files are exports. The tensor is truth.

## What Problem Does It Solve?

Modern development requires too many moving pieces. A simple project might need:

- A relational database for structured data
- A graph database for relationships
- A vector database for semantic search
- A cache layer for performance
- Version control for code

Each has its own query language, its own mental model, its own operational overhead. Teams spend more time on infrastructure than on their actual problem.

**Neumann collapses this.** One runtime. One shell. One way of thinking about information.

For vibe coders and small teams, this means getting started in minutes instead of hours. For AI-native applications, it means code, data, and embeddings are naturally co-located — no glue code, no sync problems.

## Core Concepts

### Tensor

The underlying mathematical structure. Just as von Neumann showed that code and data are both patterns in memory, a tensor can represent a scalar, a vector, a matrix, a table, a graph, or a higher-dimensional structure. The storage format is unified even when the query patterns differ.

### Shell

The interface. Simple CLI commands to create, query, and manipulate data. Designed for humans first.

### Benchmarks on M1 Max

| Operation | Dimensionality | Time | Throughput | improvement |
|-----------|----------------|------|------------|-------------|
| **Write** | 1536d | 133ns | 7.5M/sec | - |
| **Read** | 1536d | 89ns | 11.2M/sec | - |
| **TT-Decompose** | 1024d | 33.5µs | 29.8k/sec | **62% faster** (Randomized SVD) |
| **TT-Decompose** | 4096d | 79.7µs | 12.5k/sec | **32% faster** (Randomized SVD) |
| **TT-Reconstruct** | 4096d | 886µs | 1.1k/sec | **28% faster** |

> **Note**: Throughput numbers are for single-threaded execution. Neumann scales linearly with cores.

### Three Query Patterns, One Substrate

| Pattern | Style | Examples |
|---------|-------|----------|
| **Relational** | Postgres-style | Tables, rows, columns, joins |
| **Graph** | Neo4j-style | Nodes, edges, traversals, path-finding |
| **Vector** | Qdrant-style | Embeddings, similarity search, nearest neighbors |

All three operate on the same underlying tensor. A node in your graph can have relational properties and a vector embedding. A row in your table can have graph relationships. The boundaries dissolve.

### Unified Entities

The core insight: **one key, one entity, three perspectives**.

```
user:1 = {
    // Relational fields
    name: "Alice",
    email: "alice@example.com",

    // Graph connections
    _out: ["edge:follows:1", "edge:follows:2"],
    _in: ["edge:follows:3"],

    // Vector embedding
    _embedding: [0.1, 0.2, 0.3, ...]
}
```

This enables cross-engine queries:
- "Find users similar to Alice who are also friends with Bob" (vector + graph)
- "Get all products in the 'electronics' category that are similar to this description" (relational + vector)
- "Find the shortest path between users who have similar embeddings" (graph + vector)

### Code as Data

When you point Neumann at a codebase, it doesn't just store files — it understands structure. Functions, dependencies, call graphs. This enables queries like "what would break if I changed this?" without leaving the same system that holds your application data.

### Zero-Aware Storage

Neumann treats zeros as the absence of information rather than data to store:

```
Traditional: Store 768 floats → 3,072 bytes per embedding
Sparse (99% zeros): Store 8 positions + values → ~64 bytes (48x smaller)
Delta (clustered): Store archetype_id + small delta → ~120 bytes (25x smaller)
```

When vectors cluster around archetypes (common in embeddings), k-means discovers these patterns automatically. Similarity operations scale with actual information (non-zeros), not dimension size — yielding 10-150x speedups.

### Tiered Memory

When datasets exceed RAM, Neumann automatically tiers between hot and cold storage:

```
Hot Tier (DashMap):     Fast, in-memory, concurrent access
Cold Tier (mmap):       Disk-backed, loaded on demand

TieredStore tracks access patterns:
- Frequently accessed data stays hot
- Cold data migrates to memory-mapped files
- Accessing cold data promotes it back to hot
```

This enables 50GB+ vector indices on machines with 8GB RAM — hot data (working set) stays fast while cold data lives on disk. Benchmarks show 5-7% overhead for hot operations, with ~1M entries/sec migration throughput.

## Installation

```bash
# Clone and build
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann
cargo build --release

# Run the shell
./target/release/neumann_shell
```

See [Installation Guide](docs/installation.md) for detailed instructions.

## Quick Start

```bash
# Start the interactive shell
$ ./target/release/neumann_shell

# Create a table (relational)
> CREATE TABLE users (id INT, name TEXT, role TEXT)
> INSERT INTO users VALUES (1, 'Alice', 'engineer')
> SELECT * FROM users WHERE role = 'engineer'

# Create nodes and edges (graph)
> NODE CREATE person {name: 'Alice'}
> NODE CREATE project {name: 'Neumann'}
> EDGE CREATE node:1 -> node:2 : works_on {}
> NEIGHBORS node:2 INCOMING

# Store and search embeddings (vector)
> EMBED STORE 'doc:1' [0.1, 0.2, 0.3, 0.4]
> EMBED STORE 'doc:2' [0.15, 0.25, 0.35, 0.45]
> SIMILAR 'doc:1' LIMIT 5

# Save your work
> SAVE 'mydata.bin'
```

See [Getting Started](docs/getting-started.md) for a full tutorial.

### Using the Unified Entity API (Rust)

```rust
use query_router::QueryRouter;
use tensor_store::TensorStore;

// Create a router with shared storage
let store = TensorStore::new();
let router = QueryRouter::with_shared_store(store);

// Create entities with embeddings
router.vector().set_entity_embedding("user:1", vec![0.1, 0.2, 0.3])?;
router.vector().set_entity_embedding("user:2", vec![0.15, 0.25, 0.35])?;

// Connect entities via graph edges
router.connect_entities("user:1", "user:2", "follows")?;

// Cross-engine query: find neighbors sorted by similarity
let results = router.find_neighbors_by_similarity("user:1", &query_vec, 10)?;
```

### Persistence

Save and load the entire store to disk:

```rust
use tensor_store::TensorStore;

// Create and populate store
let store = TensorStore::new();
store.put("user:1", user_data)?;
store.put("user:2", user_data)?;

// Save snapshot to file (atomic write)
store.save_snapshot("data.bin")?;

// Later: load from snapshot
let store = TensorStore::load_snapshot("data.bin")?;
assert!(store.exists("user:1"));
```

Snapshots use bincode for compact binary serialization. All core types (`TensorData`, `TensorValue`, `ScalarValue`) are serializable.

## Project Status

| Module | Status | Description |
|--------|--------|-------------|
| Tensor Store | Complete | Key-value storage with shared entity support, HNSW index |
| Tiered Storage | Complete | Hot/cold memory tiering with mmap for datasets larger than RAM |
| Sparse Vectors | Complete | Memory-efficient storage for high-sparsity embeddings (3-33x compression) |
| Delta Vectors | Complete | Archetype-based encoding with k-means clustering (auto-discovery) |
| Relational Engine | Complete | Tables, schemas, SQL-like operations |
| Graph Engine | Complete | Nodes, edges, traversals, unified entity edges |
| Vector Engine | Complete | Embeddings, similarity search, unified entity embeddings |
| Tensor Compress | Complete | Int8/binary quantization, delta encoding, RLE |
| Tensor Vault | Complete | AES-256-GCM encrypted secrets with graph-based access, permission levels, TTL grants, rate limiting, namespace isolation, audit logging, and secret versioning |
| Tensor Cache | Complete | LLM response caching with semantic similarity, cost tracking |
| Tensor Blob | Complete | S3-style chunked blob storage with deduplication, streaming, entity linking |
| Query Router | Complete | Cross-engine queries on unified entities, cache integration |
| Neumann Parser | Complete | Hand-written recursive descent SQL/Graph/Vector parser |
| Shell | Complete | Interactive CLI with readline, history, formatted output |
| Tensor Chain | Complete | Tensor-native blockchain with production-ready distributed systems (see below) |
| Persistence | Basic | Snapshot-based save/load with bincode serialization |

### Distributed Systems (Tensor Chain)

Tensor Chain provides **production-ready distributed infrastructure** with >95% test coverage:

| Component | Status | Features |
|-----------|--------|----------|
| **Raft Consensus** | Production | Pre-vote protocol, automatic log compaction, snapshot persistence, WAL durability, leadership transfer, automatic heartbeat |
| **2PC Transactions** | Production | Coordinator/participant model, abort broadcast with retry, deadlock detection, orthogonal delta optimization |
| **Membership** | Production | Gossip protocol (SWIM), dynamic membership via joint consensus, partition detection and automatic merge |
| **Network** | Production | TCP transport with TLS, LZ4 compression, per-peer rate limiting, I/O timeouts, connection pooling |
| **Replication** | Production | Delta-compressed state replication, archetype persistence, streaming snapshots with memory bounds |

All components have integration tests and fuzz targets. See [Tensor Chain docs](docs/tensor-chain.md) for details.

## What Neumann Is Not (For Now)

- **Not battle-tested in production yet.** Distributed infrastructure is complete with >95% test coverage, but real-world multi-node deployments need validation.
- **Not a replacement for production Postgres at scale.** It's for development, prototyping, small-to-medium workloads, and AI-native applications.
- **Not a full IDE or code editor.** It stores and queries code structure, but you still write code elsewhere.

## Why "Neumann"?

John von Neumann unified code and data in the stored-program architecture. Sixty years later, we've re-fragmented them into separate systems.

Neumann finishes the thought.

## Documentation

**Getting Started**
- [Installation](docs/installation.md)
- [Getting Started Tutorial](docs/getting-started.md)
- [Architecture Overview](docs/architecture.md)

**Core Modules**
- [Tensor Store API](docs/tensor-store.md)
- [Relational Engine API](docs/relational-engine.md)
- [Graph Engine API](docs/graph-engine.md)
- [Vector Engine API](docs/vector-engine.md)

**Extended Modules**
- [Tensor Compress](docs/tensor-compress.md)
- [Tensor Vault](docs/tensor-vault.md)
- [Tensor Cache](docs/tensor-cache.md)
- [Tensor Blob](docs/tensor-blob.md)
- [Tensor Chain](docs/tensor-chain.md)

**Query Language**
- [Query Router API](docs/query-router.md)
- [Neumann Parser](docs/neumann-parser.md)
- [Shell](docs/neumann-shell.md)

**Reference**
- [Benchmarks](docs/benchmarks.md)
- [Changelog](CHANGELOG.md)

## Getting Help

- **Discord**: [Join our community](https://discord.gg/uN3KbAyKvw)
- **GitHub Issues**: [Report bugs](https://github.com/Shadylukin/Neumann/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Shadylukin/Neumann/discussions)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

## Author

Created by [Lukin Ackroyd](https://scrunchee.ai) in Auckland, New Zealand.

Neumann is the infrastructure layer for [Scrunchee](https://scrunchee.ai), a code intelligence platform for the AI era.
