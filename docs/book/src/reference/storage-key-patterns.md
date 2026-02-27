# Storage Key Patterns

All Neumann crates store data in TensorStore using key prefixes to
namespace their data. This reference consolidates the key patterns used
by each engine.

## Key Format

Keys are UTF-8 strings. Engines use colon-delimited prefixes to partition
the key space.

## tensor_store

| Pattern | Description |
| --- | --- |
| `{key}` | Raw key-value pairs |
| `_meta:{key}` | Entity metadata |
| `_hnsw:{index}:{layer}:{id}` | HNSW graph structure |
| `_wal:{sequence}` | Write-ahead log entries |
| `_snap:{id}` | Snapshot metadata |
| `_tier:{level}:{key}` | Tiered storage entries |

## relational_engine

| Pattern | Description |
| --- | --- |
| `rel:{table}:row:{id}` | Row data |
| `rel:{table}:meta` | Table schema and metadata |
| `rel:{table}:idx:{index}:{value}` | Index entries |
| `rel:{table}:seq` | Auto-increment sequence |
| `rel:{table}:col:{column}:{id}` | Columnar storage entries |

## graph_engine

| Pattern | Description |
| --- | --- |
| `graph:node:{id}` | Node data |
| `graph:edge:{id}` | Edge data |
| `graph:edges:{node_id}:out` | Outgoing edge list |
| `graph:edges:{node_id}:in` | Incoming edge list |
| `graph:node_count` | Global node counter |
| `graph:edge_count` | Global edge counter |
| `entity:{key}._edges` | Unified entity edge list |

## vector_engine

| Pattern | Description |
| --- | --- |
| `emb:{key}` | Embedding vector |
| `entity:{key}._embedding` | Unified entity embedding |
| `vec:col:{collection}:{key}` | Collection-scoped embedding |
| `vec:col:{collection}:meta` | Collection metadata |

## tensor_vault

| Pattern | Description |
| --- | --- |
| `vault:{path}` | Encrypted secret data |
| `vault:meta:{path}` | Secret metadata (created, updated) |
| `vault:acl:{path}` | Access control list |

## tensor_cache

| Pattern | Description |
| --- | --- |
| `cache:exact:{key}` | Exact-match cache entry |
| `cache:semantic:{key}` | Semantic cache entry with embedding |
| `cache:meta:{key}` | Cache entry metadata (TTL, hits) |

## tensor_blob

| Pattern | Description |
| --- | --- |
| `blob:{hash}` | Content-addressed blob data |
| `blob:meta:{hash}` | Blob metadata (size, type, created) |
| `blob:ref:{name}` | Named reference to blob hash |

## tensor_checkpoint

| Pattern | Description |
| --- | --- |
| `checkpoint:{id}` | Checkpoint snapshot data |
| `checkpoint:meta:{id}` | Checkpoint metadata |
| `checkpoint:latest` | Pointer to most recent checkpoint |

## tensor_chain

| Pattern | Description |
| --- | --- |
| `chain:block:{height}` | Block at given height |
| `chain:tx:{id}` | Transaction record |
| `chain:raft:log:{index}` | Raft log entry |
| `chain:raft:meta` | Raft state (term, voted_for) |
| `chain:lock:{key}` | Transaction lock |

## See Also

- Per-crate API reference pages for detailed key documentation
- [Tensor Data Model](../explanation/tensor-data-model.md) -- how data
  is structured
