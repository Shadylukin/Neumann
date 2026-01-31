# Neumann

[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/uN3KbAyKvw)

A distributed tensor runtime that unifies relational, graph, and vector
storage with semantic consensus.

```text
One database. Three query patterns. Geometric conflict resolution.
```

## The Problem

Building AI applications requires juggling multiple databases:

| Need            | Typical Solution | Overhead                            |
| --------------- | ---------------- | ----------------------------------- |
| Structured data | PostgreSQL       | Connection pooling, migrations, ORM |
| Relationships   | Neo4j            | Separate query language, sync logic |
| Embeddings      | Pinecone/Qdrant  | API calls, eventual consistency     |
| Caching         | Redis            | Another connection, invalidation    |
| Secrets         | Vault            | Yet another service                 |

Each system has its own failure modes, its own scaling story, its own
operational burden. Your "simple" AI application now depends on five
services that must stay synchronized.

## The Solution

Neumann stores everything in tensors. A user can have relational properties,
graph edges, and vector embeddings in the same entity. Queries cross these
boundaries naturally:

```sql
-- Find users similar to Alice who are connected to Bob
FIND NODE user
  WHERE role = 'engineer'
  SIMILAR TO 'user:alice'
  CONNECTED TO 'user:bob'
```

One runtime. One query language. One consistency model.

<p align="center">
  <img src="images/dash.png" alt="Neumann Dashboard" width="800">
  <br>
  <em>Web dashboard with system status and query terminal</em>
</p>

<p align="center">
  <img src="images/graph.png" alt="Graph Visualization" width="800">
  <br>
  <em>Interactive graph visualization with force-directed layout</em>
</p>

## Performance

Benchmarked on Apple M-series silicon:

| Operation              | Performance               | Notes                         |
| ---------------------- | ------------------------- | ----------------------------- |
| **Storage throughput** | 3.2M PUT, 5M GET ops/sec  | BTreeMap-based, no stalls     |
| **Concurrent writes**  | 7.5M ops/sec @ 1M entries | 8 threads, CV <7%             |
| **Vector similarity**  | 150us @ 10K vectors       | HNSW index, 13x vs brute      |
| **Conflict detection** | 52M pairs/sec             | 99% sparse deltas             |
| **Query parsing**      | 1.9M queries/sec          | Hand-written recursive descent|

Sub-microsecond operations for most workloads. Linear scaling with cores.

## Distributed Consensus

Neumann includes a production-ready distributed layer built on Raft
consensus with semantic extensions:

### Tensor-Native Raft

Standard Raft treats data as opaque bytes. Neumann embeds state semantically:

- **Similarity fast-path**: Blocks with >95% cosine similarity to current
  state bypass full validation (40-60% latency reduction)
- **Geometric tie-breaking**: Leader elections prefer candidates semantically
  closer to cluster state
- **Two-phase finality**: Optimistic reads while guaranteeing durability

### 6-Way Conflict Detection

Instead of binary commit/abort, conflicts are classified geometrically:

| Class        | Cosine  | Jaccard | Action                       |
| ------------ | ------- | ------- | ---------------------------- |
| Orthogonal   | <0.1    | <0.5    | Auto-merge via vector add    |
| Low Conflict | 0.1-0.7 | <0.5    | Weighted average             |
| Identical    | 1.0     | -       | Deduplicate                  |
| Opposite     | <-0.95  | -       | Cancel (no-op)               |
| Conflicting  | >0.7    | -       | Reject                       |
| Ambiguous    | 0.1-0.7 | >0.5    | Reject                       |

This catches subtle conflicts that pure voting misses. Two transactions
modifying different fields of the same entity can commit in parallel
if they're orthogonal.

### Delta Replication

State replication uses archetype-based compression:

- Embeddings cluster around learned archetypes
- Replicate `(archetype_id, sparse_delta)` instead of full vectors
- **4-6x bandwidth reduction** with BLAKE2b integrity checks
- 100 updates batched per network round-trip

### Cluster Management

- SWIM gossip protocol with LWW-CRDT membership state
- Partition detection with quorum enforcement
- DFS-based deadlock detection in 2PC coordinator
- Ed25519 block signatures via validator registry

## Quick Start

```bash
# Clone and build
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann
cargo build --release

# Start the shell
./target/release/neumann
```

### Relational

```sql
> CREATE TABLE users (id INT, name TEXT, role TEXT)
> INSERT users id=1, name='Alice', role='engineer'
> SELECT * FROM users WHERE role = 'engineer'
```

### Graph

```sql
> NODE CREATE person name='Alice'
> NODE CREATE project name='Neumann'
> EDGE CREATE node:1 -> node:2 works_on
> PATH node:1 -> node:2
```

### Vector

```sql
> EMBED 'doc:readme' [0.1, 0.2, 0.3, 0.4, ...]
> SIMILAR 'doc:readme' TOP 5
```

### Unified

```sql
> FIND NODE person
    WHERE role = 'engineer'
    SIMILAR TO 'user:alice'
```

## Architecture

20 crates organized in dependency tiers:

```text
                    +---------------------------------------------+
                    |              tensor_chain                   |
                    |   Raft consensus, 2PC, gossip, delta        |
                    |         replication, codebooks              |
                    +---------------------------------------------+
                                        |
        +-------------------------------+-------------------------------+
        v                               v                               v
+---------------+               +---------------+               +---------------+
| tensor_vault  |               | tensor_cache  |               |  tensor_blob  |
|   AES-256     |               |  LLM cache    |               |    S3-style   |
| graph access  |               |   semantic    |               |   chunked     |
+---------------+               +---------------+               +---------------+
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        v
                    +---------------------------------------------+
                    |             query_router                    |
                    |     Unified cross-engine queries            |
                    +---------------------------------------------+
                                        |
        +-------------------------------+-------------------------------+
        v                               v                               v
+---------------+               +---------------+               +---------------+
|   relational  |               |    graph      |               |    vector     |
|    engine     |               |    engine     |               |    engine     |
|  SIMD filter  |               |  BFS, paths   |               |  HNSW, sparse |
+---------------+               +---------------+               +---------------+
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        v
                    +---------------------------------------------+
                    |              tensor_store                   |
                    |   SlabRouter, HNSW, sparse vectors,         |
                    |   tiered storage, bloom filters             |
                    +---------------------------------------------+
```

### Core Storage (tensor_store)

- **SlabRouter**: Routes keys to specialized slabs by prefix
- **7 embedding formats**: Dense, Sparse, Delta, TensorTrain, Quantized, ProductQuantized, Binary
- **15+ distance metrics**: Cosine, Angular, Geodesic, Jaccard, Overlap,
  Euclidean, Manhattan, Composite
- **Hot/cold tiering**: Automatic migration based on access patterns
- **Zero-allocation sparse ops**: Similarity scales with non-zeros

### Engines

| Engine            | Key Features                                  |
| ----------------- | --------------------------------------------- |
| relational_engine | SIMD filtering, B-tree indexes, transactions  |
| graph_engine      | BFS traversal, shortest path, Dijkstra        |
| vector_engine     | HNSW O(log n) search, sparse detection, batch |

### Specialized Storage

| Crate             | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| tensor_vault      | AES-256-GCM encryption with graph-based access       |
| tensor_cache      | Multi-layer LLM caching: exact + semantic + embedding|
| tensor_blob       | Content-addressable chunks with SHA-256 deduplication|
| tensor_checkpoint | Pre-destructive-op snapshots with confirmation       |

### Distributed (tensor_chain)

- Tensor-Raft with similarity fast-path and geometric tie-breaking
- 2PC coordinator with DFS deadlock detection
- SWIM gossip with LWW-CRDT membership
- Delta-compressed replication with archetype codebooks
- Hierarchical vector quantization for state validation

## Client SDKs

### Python

```python
from neumann import NeumannClient

# Embedded (in-process)
client = NeumannClient.embedded()

# Remote (gRPC)
client = NeumannClient.connect("localhost:9200", api_key="...")

result = client.execute("SELECT * FROM users")
for row in result.rows:
    print(row.to_dict())

# Transactions
with client.transaction() as tx:
    tx.execute("INSERT users name='Alice'")
    tx.execute("INSERT users name='Bob'")
```

### TypeScript

```typescript
const client = await NeumannClient.connect("localhost:9200");
const result = await client.execute("SELECT * FROM users");

// Streaming for large results
for await (const chunk of client.executeStream("SELECT * FROM large_table")) {
  console.log(chunk.rows);
}
```

### gRPC

Full service definitions in `neumann_server/proto/neumann.proto`:

- QueryService: Execute, ExecuteStream, ExecuteBatch
- BlobService: Upload, Download, Delete, GetMetadata
- Health: Check

## Testing

```bash
# All tests (267+ integration tests)
cargo test

# Quality gates
cargo fmt --check
cargo clippy -- -D warnings -D clippy::pedantic

# Fuzzing (22 targets)
cargo +nightly fuzz run parser_parse -- -max_total_time=60
```

Coverage requirements enforced per-crate (78-95% minimum depending on crate complexity).

## What This Is

- A unified runtime for AI-native applications
- A research platform for semantic consensus
- A foundation for building distributed tensor systems

## What This Isn't (Yet)

- Not battle-tested in production multi-node deployments
- Not a drop-in replacement for PostgreSQL at enterprise scale
- Not optimized for petabyte-scale cold storage

The distributed layer has >95% test coverage and comprehensive fuzz testing,
but real-world validation across diverse failure modes is ongoing.

## Why "Neumann"

John von Neumann unified code and data in the stored-program architecture.
Sixty years later, we've re-fragmented them across separate systems for
structure, relationships, and semantics.

Neumann finishes the thought: one mathematical substrate for all your data.

## Documentation

- [API Documentation](https://shadylukin.github.io/Neumann/) - Full rustdoc API reference
- [Installation](docs/book/src/getting-started/installation.md)
- [Quick Start](docs/book/src/getting-started/quick-start.md)
- [Architecture](docs/architecture.md)
- [Benchmarks](docs/book/src/benchmarks/index.md)
- [Tensor Chain](docs/book/src/architecture/tensor-chain.md)
- [API Reference](docs/book/src/api-reference.md)

## License

Dual licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

## Author

Built by [Lukin Ackroyd](https://scrunchee.ai) in Auckland, New Zealand.

Neumann is the infrastructure layer for [Scrunchee](https://scrunchee.ai).
