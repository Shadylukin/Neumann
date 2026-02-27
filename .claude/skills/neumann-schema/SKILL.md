---
name: neumann-schema
description: Design Neumann data models using relational tables, graph nodes/edges, and vector embeddings. Use when planning a database schema or deciding which Neumann engines to use for a feature.
---

# Neumann Schema Design

## When to Use Which Engine

Neumann has multiple engines. Choose based on data shape and query patterns.

| Engine | Best For | Query Style |
|--------|----------|-------------|
| **Relational** | Structured records, joins, aggregations | `SELECT`, `INSERT`, `WHERE`, `JOIN` |
| **Graph** | Relationships, traversals, influence | `NODE CREATE`, `EDGE CREATE`, `NEIGHBORS`, `PATH` |
| **Vector** | Similarity, semantic search, embeddings | `EMBED STORE`, `SIMILAR` |
| **Unified Entity** | Objects spanning multiple engines | `ENTITY CREATE`, `ENTITY GET` |
| **Vault** | Secrets, encrypted credentials | `VAULT STORE`, `VAULT GET` |
| **Cache** | LLM response caching | `CACHE GET`, `CACHE PUT`, `CACHE SEMANTIC GET` |
| **Blob** | Files, images, large binary objects | `ARTIFACT UPLOAD`, `ARTIFACT DOWNLOAD` |
| **Chain** | Immutable audit trail, versioned tensors | `CHAIN BEGIN`, `CHAIN COMMIT` |

### Decision Guide

- **Need columns and types?** Use relational tables (`CREATE TABLE`).
- **Need to traverse connections?** Use graph nodes and edges.
- **Need "find similar" or "nearest neighbor"?** Use vector embeddings.
- **Need all three for one entity?** Use unified entities or link by key.
- **Need encryption at rest?** Use vault for secrets.
- **Need to cache expensive LLM calls?** Use cache engine.

Most real applications use 2-3 engines together. A user profile might be a
relational row (structured fields), a graph node (connections), and a vector
embedding (semantic search) -- all linked by the same user ID.

## Schema Design Rules

### Relational Tables

Use for structured data with known columns and types. Tables support
indexes, constraints, joins, and aggregations.

```sql
CREATE TABLE users (name STRING, email STRING, age INT, active BOOL)
CREATE INDEX idx_email ON users (email)
```

### Graph Nodes and Edges

Use for entities with dynamic properties and typed relationships. Edges
are always directed (from -> to). Use labels to categorize nodes and
type edges.

```sql
NODE CREATE user { name: 'Alice', role: 'engineer' }
NODE CREATE user { name: 'Bob', role: 'manager' }
EDGE CREATE 1 -> 2 : reports_to
```

### Vector Embeddings

Use for data that needs similarity search. Key embeddings by entity ID
for cross-engine linking. Choose the right distance metric:

- `cosine` -- text embeddings, normalized vectors (most common)
- `euclidean` -- spatial data, coordinate distances
- `dot` -- when vectors are already normalized and you want speed

```sql
EMBED STORE 'user:1' [0.12, -0.34, 0.56, ...]
SIMILAR TO [0.12, -0.34, 0.56, ...] LIMIT 10
```

### Vault, Cache, Blob

- **Vault**: Store secrets with identity-based access control.
  `VAULT STORE 'db-password' 'secret123' AS admin`
- **Cache**: Cache LLM responses by exact query or semantic similarity.
  `CACHE PUT 'prompt-hash' 'response-text'`
- **Blob**: Store files with content-addressable deduplication.
  `ARTIFACT UPLOAD 'report.pdf' FROM '/path/to/file'`

## Cross-Engine Linking Patterns

The key principle: **use the same identifier across engines**.

### Pattern 1: Shared Key

Store the same ID in a table column, as a node property, and as the
embedding key.

```sql
-- Relational: row with id
INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@co.com')

-- Graph: node with same id
NODE CREATE user { user_id: '1', name: 'Alice' }

-- Vector: embedding keyed by same id
EMBED STORE 'user:1' [0.12, -0.34, ...]
```

### Pattern 2: Unified Entity

ENTITY CREATE writes to all relevant engines atomically.

```sql
ENTITY CREATE user {
  name: 'Alice',
  email: 'alice@co.com',
  embedding: [0.12, -0.34, ...]
}
```

### Pattern 3: Combined Queries

Chain vector search with graph traversal.

```sql
SIMILAR TO [0.12, ...] LIMIT 10
-- Then use those keys to traverse the graph
NEIGHBORS 42 OUTGOING DEPTH 2
```

## Anti-Patterns

**Do not store graphs in relational tables.** Using foreign keys and
self-joins to model a graph is slow and awkward. Use `NODE CREATE` and
`EDGE CREATE` instead -- the graph engine handles traversals efficiently.

**Do not store structured records as node properties.** If you have 20
typed columns with constraints and indexes, that is a table. Node
properties are schemaless key-value pairs.

**Do not embed everything.** Vector embeddings cost storage and compute.
Only embed data that needs similarity search or semantic matching. Exact
lookups should use relational SELECT or graph GET.

**Do not use vector search for exact lookups.** If you know the exact
key, use `SELECT ... WHERE id = X` or `NODE GET id`. Vector search is
approximate and slower for exact matches.

**Do not duplicate data across engines without a linking key.** If a
user exists in the table and in the graph, both must share an identifier.
Otherwise updates diverge silently.

## Example Schemas

See the `examples/` directory for complete schema designs:

- `examples/rag-app.md` -- RAG application with documents, chunks, and
  semantic retrieval
- `examples/agent-memory.md` -- AI agent memory with structured recall,
  graph associations, and semantic search
- `examples/knowledge-graph.md` -- Knowledge graph with entity
  resolution, typed relationships, and similarity

## Reference

- See `neumann-query` skill for complete query syntax
- See `neumann-vector` skill for HNSW index configuration
- See `neumann-graph` skill for traversal patterns
- See `neumann-client` skill for SDK integration
