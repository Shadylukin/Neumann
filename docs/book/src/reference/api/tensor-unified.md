# Tensor Unified API Reference

## See Also

- **Explanation**: [Unified Entities](../../explanation/unified-entities.md)
- **How-to**: [Unified Entities](../../how-to/unified-entities.md)

---

## Core Types

| Type | Description |
| --- | --- |
| `UnifiedEngine` | Main entry point for cross-engine operations |
| `UnifiedResult` | Query result containing description and items |
| `UnifiedItem` | Single item with source, id, data, embedding, and score |
| `UnifiedError` | Error type wrapping engine-specific errors |
| `FindPattern` | Pattern for FIND queries (Nodes or Edges) |
| `DistanceMetric` | Similarity metric (Cosine, Euclidean, DotProduct) |
| `EntityInput` | Tuple type for batch operations: (key, fields, embedding) |
| `Unified` | Trait for converting engine types to UnifiedItem |
| `FilterCondition` | Re-exported from vector_engine for filtered search |
| `FilterValue` | Re-exported from vector_engine for filter values |
| `VectorCollectionConfig` | Re-exported from vector_engine for collection config |

### UnifiedEngine

```rust
pub struct UnifiedEngine {
    store: TensorStore,
    relational: Arc<RelationalEngine>,
    graph: Arc<GraphEngine>,
    vector: Arc<VectorEngine>,
}
```

The `Arc` wrappers enable thread-safe sharing across async tasks, zero-copy
cloning of the engine, and independent engine access when needed.

### UnifiedItem

```rust
pub struct UnifiedItem {
    pub source: String,                    // "relational", "graph", "vector", or combined
    pub id: String,                        // Entity key
    pub data: HashMap<String, String>,     // Entity fields
    pub embedding: Option<Vec<f32>>,       // Optional embedding
    pub score: Option<f32>,                // Similarity score if applicable
}
```

The `source` field indicates which engine(s) produced the result:

- `"graph"` - Result from graph operations (nodes, edges)
- `"vector"` - Result from vector similarity search
- `"unified"` - Result from cross-engine entity retrieval
- `"vector+graph"` - Result from `find_similar_connected` (similarity +
  connectivity)
- `"graph+vector"` - Result from `find_neighbors_by_similarity` (connectivity +
  similarity)

### UnifiedError

| Variant | Cause |
| --- | --- |
| `RelationalError` | Error from relational engine |
| `GraphError` | Error from graph engine |
| `VectorError` | Error from vector engine |
| `NotFound` | Entity not found |
| `InvalidOperation` | Invalid operation attempted |

Error conversion is automatic via `From` implementations:

```rust
impl From<graph_engine::GraphError> for UnifiedError {
    fn from(e: graph_engine::GraphError) -> Self {
        UnifiedError::GraphError(e.to_string())
    }
}

impl From<vector_engine::VectorError> for UnifiedError {
    fn from(e: vector_engine::VectorError) -> Self {
        UnifiedError::VectorError(e.to_string())
    }
}

impl From<relational_engine::RelationalError> for UnifiedError {
    fn from(e: relational_engine::RelationalError) -> Self {
        UnifiedError::RelationalError(e.to_string())
    }
}
```

### Unified Trait

Types implementing the `Unified` trait can be converted to `UnifiedItem`:

```rust
pub trait Unified {
    fn as_unified(&self) -> UnifiedItem;
    fn source_engine(&self) -> &'static str;
    fn unified_id(&self) -> String;
}
```

Implemented for:

- `graph_engine::Node` - Converts label and properties to data fields
- `graph_engine::Edge` - Converts from, to, type, and properties to data fields
- `vector_engine::SearchResult` - Converts key and score

## Entity Storage Format

Unified entities use reserved field prefixes in `TensorData` to store
cross-engine data within a single key-value entry:

| Field | Type | Description |
| --- | --- | --- |
| `_out` | `Pointers(Vec<String>)` | Outgoing edge keys |
| `_in` | `Pointers(Vec<String>)` | Incoming edge keys |
| `_embedding` | `Vector(Vec<f32>)` or `Sparse(SparseVector)` | Embedding vector |
| `_label` | `Scalar(String)` | Entity type/label |
| `_type` | `Scalar(String)` | Discriminator ("node", "edge", "row") |
| `_id` | `Scalar(Int)` | Numeric entity ID |
| `_from` | `Scalar(String)` | Edge source key |
| `_to` | `Scalar(String)` | Edge target key |
| `_edge_type` | `Scalar(String)` | Edge type |
| `_directed` | `Scalar(Bool)` | Whether edge is directed |
| `_table` | `Scalar(String)` | Table name for relational rows |

### Entity Storage Example

```text
Key: "user:alice"
TensorData:
  _embedding: Vector([0.1, 0.2, 0.3, 0.4])
  _out: Pointers(["edge:follows:1", "edge:likes:2"])
  _in: Pointers(["edge:follows:3"])
  name: Scalar(String("Alice"))
  role: Scalar(String("admin"))

Key: "edge:follows:1"
TensorData:
  _type: Scalar(String("edge"))
  _from: Scalar(String("user:alice"))
  _to: Scalar(String("user:bob"))
  _edge_type: Scalar(String("follows"))
  _directed: Scalar(Bool(true))
```

### Sparse Vector Auto-Detection

Embeddings are automatically stored in sparse format when >50% of values are
zero:

```rust
fn should_use_sparse(vector: &[f32]) -> bool {
    if vector.is_empty() {
        return false;
    }
    let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
    // Sparse if nnz <= len/2
    nnz * 2 <= vector.len()
}
```

## Cross-Engine Operation Table

| Method | Description | Complexity | Notes |
| --- | --- | --- | --- |
| `create_entity` | Create entity with fields and optional embedding | O(1) | Single store put + optional embedding |
| `connect_entities` | Connect entities via graph edge | O(1) | Three store operations (edge + 2 entity updates) |
| `get_entity` | Retrieve entity with all data and embedding | O(1) | Single store get + optional embedding lookup |
| `create_entity_unified` | Store entity with fields as vector metadata | O(1) | Single store with metadata |
| `get_entity_unified` | Retrieve with metadata fallback | O(1) | Metadata lookup with fallback |
| `create_entity_in_collection` | Store entity in a collection | O(1) | Collection-scoped store |
| `create_entity_collection` | Create a named collection | O(1) | Enforces dimension and metric |
| `delete_entity_collection` | Delete a collection and all entities | O(c) | c = collection size |
| `list_entity_collections` | List all collections | O(1) | Returns collection names |
| `find_similar_connected` | Similar AND connected entities | O(k log n) | HNSW search + graph intersection |
| `find_similar_connected` (brute) | Same, without HNSW index | O(n) | Linear scan fallback |
| `find_similar_connected_filtered` | Similar + connected + metadata filter | O(m) | Pre-filter search, m = matching keys |
| `find_similar_in_collection` | Similarity search within a collection | O(c) | c = collection size |
| `find_neighbors_by_similarity` | Neighbors sorted by vector similarity | O(d * k) | d = avg degree, k = top-k |
| `find_nodes` | Scan for matching nodes | O(n) | Full scan with prefix filter |
| `find_edges` | Scan for matching edges | O(e) | Full scan with prefix filter |
| `embed_batch` | Batch embedding storage | O(b) | Sequential, b = batch size |
| `create_entities_batch` | Batch entity creation | O(b) | Sequential, failures counted |
| `cross_modal_contraction` | Fuse graph + vector + relational scoring | O(table) | Materializes full table |

Where: n = entities with embeddings, d = average degree, k = top-k, e = edges,
b = batch size, c = collection size, m = matching keys.

### Benchmarks

From `tensor_unified_bench.rs`:

| Operation | 10 items | 100 items | 1000 items |
| --- | --- | --- | --- |
| `create_entity` | ~50us | ~500us | ~5ms |
| `embed_batch` | ~30us | ~300us | ~3ms |
| `find_nodes` | ~10us | ~100us | ~1ms |
| `UnifiedItem::new` | ~50ns | --- | --- |
| `UnifiedItem::with_data` | ~200ns | --- | --- |

## Condition Matching

Conditions are evaluated against node/edge properties in FIND queries:

| Condition | Node Fields | Edge Fields |
| --- | --- | --- |
| `Eq("id", ...)` | Matches `node.id` | Matches `edge.id` |
| `Eq("label", ...)` | Matches `node.label` | N/A |
| `Eq("type", ...)` | N/A | Matches `edge.edge_type` |
| `Eq("edge_type", ...)` | N/A | Matches `edge.edge_type` (alias) |
| `Eq("from", ...)` | N/A | Matches `edge.from` |
| `Eq("to", ...)` | N/A | Matches `edge.to` |
| `Eq(property, ...)` | Matches `node.properties[property]` | Matches `edge.properties[property]` |
| `And(a, b)` | Both must match | Both must match |
| `Or(a, b)` | Either must match | Either must match |
| Other conditions | Returns `true` (pass-through) | Returns `true` (pass-through) |

**Gotcha:** Conditions other than `Eq`, `And`, `Or` return `true` (not yet
implemented for graph entities).

## Cross-Modal Tensor Contraction Types

| Type | Description |
| --- | --- |
| `AdjacencyVec` | `HashMap<String, f64>` -- neighbor key to edge weight |
| `SimilarityVec` | `HashMap<String, f64>` -- neighbor key to cosine similarity |
| `InteractionMap` | `HashMap<String, HashSet<String>>` -- intermediary to item set |
| `ContractionConfig` | Direction, normalization, edge type filter, top-k |
| `ScoredItem` | Item key, score, and contributor count |
| `ContractionResult` | Sorted items, weight norm, and excluded count |

### ContractionConfig

```rust
let config = ContractionConfig {
    direction: GraphDirection::Symmetric,       // or Outgoing / Incoming
    normalization: Normalization::TotalWeight,  // or None / PerItem
    edge_type: Some("FRIEND".into()),           // None = all edge types
    exclude_owned: true,                        // remove source's own items
    top_k: 10,
};
```

**Normalization strategies**:

- `None` -- raw scores, no normalization.
- `TotalWeight` -- divide each weight by the L1 norm. Prevents sources
  with many high-similarity neighbors from dominating.
- `PerItem` -- divide each item's final score by its contributor count. Reduces
  popularity bias for widely-consumed items.

## Configuration

UnifiedEngine uses the configuration of its underlying engines:

- `TensorStore`: Storage configuration
- `VectorEngine`: HNSW index parameters, similarity metrics
- `GraphEngine`: Graph traversal settings
- `RelationalEngine`: Table and index configuration

## Thread Safety

UnifiedEngine is thread-safe via:

- `Arc<VectorEngine>`, `Arc<GraphEngine>`, `Arc<RelationalEngine>`
- All underlying engines share thread-safe TensorStore (DashMap)
- No lock poisoning (parking_lot semantics)

**Safe concurrent patterns:**

- Multiple readers on same entity
- Multiple writers on different entities
- Mixed reads/writes (DashMap shard locking)

**Gotcha:** Concurrent writes to the same entity may interleave fields. Use
transactions for atomicity.

## Dependencies

- `tensor_store`: Core storage
- `relational_engine`: Table operations
- `graph_engine`: Graph operations
- `vector_engine`: Vector search
- `tokio`: Async runtime (multi-threaded)
- `futures`: Async utilities
- `serde`: Serialization for results and items
- `serde_json`: JSON output for `UnifiedResult`

## Related Modules

| Module | Relationship |
| --- | --- |
| `tensor_store` | Shared storage backend, provides `TensorData` and `fields` constants |
| `relational_engine` | Relational data, conditions for filtering |
| `graph_engine` | Graph connectivity, entity edges, neighbor queries |
| `vector_engine` | Embeddings, similarity search, HNSW index, `FilterCondition`, `FilterValue`, `VectorCollectionConfig` |
| `query_router` | Query execution, language integration, HNSW optimization, re-exports filter types |
