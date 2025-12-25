# Neumann

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

## Quick Start

```bash
$ neumann init myproject
$ neumann ingest ./src                              # code becomes queryable
$ neumann create table users                        # relational
$ neumann link user:1 -> post:5                     # graph
$ neumann embed user:1 "semantic description"       # vector
$ neumann query "users connected to posts similar to X"  # unified
```

The shell is the primary interface. Everything is a command. State lives in the tensor.

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
| Tensor Store | Complete | Key-value storage with shared entity support |
| Relational Engine | Complete | Tables, schemas, SQL-like operations |
| Graph Engine | Complete | Nodes, edges, traversals, unified entity edges |
| Vector Engine | Complete | Embeddings, similarity search, unified entity embeddings |
| Query Router | Complete | Cross-engine queries on unified entities |
| Neumann Parser | Complete | Hand-written recursive descent SQL/Graph/Vector parser |
| Shell | Complete | Interactive CLI with readline, history, formatted output |
| Persistence | Basic | Snapshot-based save/load with bincode serialization |

## What Neumann Is Not (For Now)

- **Not a distributed system.** Single-node, in-memory first. Durability and clustering come later.
- **Not a replacement for production Postgres at scale.** It's for development, prototyping, small-to-medium workloads, and AI-native applications.
- **Not a full IDE or code editor.** It stores and queries code structure, but you still write code elsewhere.

## Why "Neumann"?

John von Neumann unified code and data in the stored-program architecture. Sixty years later, we've re-fragmented them into separate systems.

Neumann finishes the thought.

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Tensor Store API](docs/tensor-store.md)
- [Relational Engine API](docs/relational-engine.md)
- [Graph Engine API](docs/graph-engine.md)
- [Vector Engine API](docs/vector-engine.md)
- [Query Router API](docs/query-router.md)
- [Neumann Parser](docs/neumann-parser.md)
- [Shell](docs/neumann-shell.md)
- [Benchmarks](docs/benchmarks.md)

## License

TBD
