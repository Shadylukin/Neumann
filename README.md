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

## Project Status

| Module | Status | Description |
|--------|--------|-------------|
| Tensor Store | Complete | Key-value storage for tensor data |
| Relational Engine | Complete | Tables, schemas, SQL-like operations |
| Graph Engine | Complete | Nodes, edges, traversals, path-finding |
| Vector Engine | Planned | Embeddings, similarity search |
| Shell | Planned | CLI interface |
| Persistence | Planned | Durability and backup |

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

## License

TBD
