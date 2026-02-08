# Benchmarks

This section provides performance benchmarks for all Neumann crates, measured
using Criterion.rs.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run benchmarks for a specific crate
cargo bench --package tensor_store
cargo bench --package relational_engine
cargo bench --package graph_engine
cargo bench --package vector_engine
cargo bench --package neumann_parser
cargo bench --package query_router
cargo bench --package neumann_shell
cargo bench --package tensor_compress
cargo bench --package tensor_vault
cargo bench --package tensor_cache
cargo bench --package tensor_chain
```

Benchmark reports are generated in `target/criterion/` with HTML visualizations.

## Performance Summary

<!-- BENCH:START -->
### In-Memory Operations

| Component | Key Metric | Performance |
| --- | --- | --- |
| [tensor_store](tensor-store.md) | Concurrent writes | 7.5M/sec @ 1M entities |
| [relational_engine](relational-engine.md) | Indexed lookup | 2.9us (1,604x vs scan) |
| [graph_engine](graph-engine.md) | BFS traversal | 3us/node |
| [vector_engine](vector-engine.md) | HNSW search | 150us @ 10K vectors |
| [tensor_compress](tensor-compress.md) | TT decompose | 10-20x compression |
| [tensor_vault](tensor-vault.md) | AES-256-GCM | 24us get, 29us set |
| [tensor_cache](tensor-cache.md) | Exact lookup | 208ns hit |
| [tensor_chain](tensor-chain.md) | Conflict detection | 52M pairs/sec @ 99% sparse |
| [neumann_parser](neumann-parser.md) | Query parsing | 1.9M queries/sec |
| [query_router](query-router.md) | Mixed workload | 455 queries/sec |

### Durable Storage (WAL)

| Operation | Key Metric | Performance |
| --- | --- | --- |
| WAL writes | Durable PUT (128d embeddings) | 1.4M ops/sec |
| WAL recovery | Replay 10K records | ~400us (25M records/sec) |
<!-- BENCH:END -->

All engines (`RelationalEngine`, `GraphEngine`, `VectorEngine`) support
optional durability via `open_durable()` with full crash consistency.

## Hardware Notes

Benchmarks run on:

- Apple M-series (ARM64) or Intel x86_64
- Results may vary based on CPU cache sizes, memory bandwidth, and core count

For consistent benchmarking:

```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Run with minimal background activity
cargo bench -- --noplot  # Skip HTML report generation for faster runs
```

## Benchmark Categories

### Storage Layer

- [tensor_store](tensor-store.md) - DashMap concurrent storage, Bloom filters,
  snapshots
- [tensor_compress](tensor-compress.md) - Tensor Train, delta encoding, RLE

### Engines

- [relational_engine](relational-engine.md) - SQL operations, indexes, JOINs,
  aggregates
- [graph_engine](graph-engine.md) - Node/edge operations, traversals, path
  finding
- [vector_engine](vector-engine.md) - Embeddings, SIMD similarity, HNSW index

### Extended Modules

- [tensor_vault](tensor-vault.md) - Encrypted storage, access control
- [tensor_cache](tensor-cache.md) - LLM response caching, semantic search
- [tensor_blob](tensor-blob.md) - Blob storage operations

### Distributed Systems

- [tensor_chain](tensor-chain.md) - Consensus, 2PC, gossip, sparse vectors

### Query Layer

- [neumann_parser](neumann-parser.md) - Tokenization, parsing, expressions
- [query_router](query-router.md) - Cross-engine query routing
