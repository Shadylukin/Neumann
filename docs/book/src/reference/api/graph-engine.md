# Graph Engine API Reference

> **See also:** [Graph Storage Internals](../../explanation/graph-storage.md) |
> [Graph CRUD How-To](../../how-to/graph-crud.md) |
> [Graph Traversals How-To](../../how-to/graph-traversals.md) |
> [Graph Property Indexes How-To](../../how-to/graph-property-indexes.md)

## Core Types

| Type | Description |
| --- | --- |
| `GraphEngine` | Main entry point for graph operations |
| `Node` | Graph node with id, label, and properties |
| `Edge` | Graph edge with from/to nodes, type, properties, and direction flag |
| `Path` | Result of path finding containing node and edge sequences |
| `Direction` | Edge traversal direction (Outgoing, Incoming, Both) |
| `PropertyValue` | Node/edge property values (Null, Int, Float, String, Bool) |
| `GraphError` | Error types for graph operations |

## PropertyValue Variants

| Variant | Rust Type | Description |
| --- | --- | --- |
| `Null` | --- | NULL value |
| `Int` | `i64` | 64-bit signed integer |
| `Float` | `f64` | 64-bit floating point |
| `String` | `String` | UTF-8 string |
| `Bool` | `bool` | Boolean |

`ScalarValue::Bytes` converts to `PropertyValue::Null` since `PropertyValue`
does not support binary data.

## Error Types

| Error | Cause |
| --- | --- |
| `NodeNotFound(u64)` | Node with given ID does not exist |
| `EdgeNotFound(u64)` | Edge with given ID does not exist |
| `PathNotFound` | No path exists between the specified nodes |
| `StorageError(String)` | Underlying Tensor Store error |

## Direction Enum

| Direction | Behavior |
| --- | --- |
| `Outgoing` | Follow edges away from the node |
| `Incoming` | Follow edges toward the node |
| `Both` | Follow edges in either direction |

## Storage Key Patterns

Nodes and edges are stored in Tensor Store using the following key patterns:

| Key Pattern | Content | TensorData Fields |
| --- | --- | --- |
| `node:{id}` | Node data | `_id`, `_type="node"`, `_label`, user properties |
| `node:{id}:out` | List of outgoing edge IDs | `e{edge_id}` fields |
| `node:{id}:in` | List of incoming edge IDs | `e{edge_id}` fields |
| `edge:{id}` | Edge data | `_id`, `_type="edge"`, `_from`, `_to`, `_edge_type`, `_directed`, user properties |

### Edge List Storage Format

Edge lists are stored as TensorData with dynamically named fields:

```rust
// Each edge ID stored as: "e{edge_id}" -> edge_id
tensor.set("e1", TensorValue::Scalar(ScalarValue::Int(1)));
tensor.set("e5", TensorValue::Scalar(ScalarValue::Int(5)));
```

This format allows O(1) edge addition but O(n) edge listing.

### Entity Edge Key Format

Entity edges use a different key format from node-based edges:

```text
edge:{edge_type}:{edge_id}
```

For example: `edge:follows:42`

### Reserved Entity Fields

| Field | Type | Purpose |
| --- | --- | --- |
| `_out` | `Vec<String>` | Outgoing edge keys |
| `_in` | `Vec<String>` | Incoming edge keys |
| `_embedding` | `Vec<f32>` | Vector embedding |
| `_type` | `String` | Entity type |
| `_id` | `i64` | Entity numeric ID |
| `_label` | `String` | Entity label |

## Engine Construction

```rust
// Create new engine with internal store
let engine = GraphEngine::new();

// Create engine with shared store (for cross-engine queries)
let store = TensorStore::new();
let engine = GraphEngine::with_store(store.clone());

// Access underlying store
let store = engine.store();
```

## Node Operations

| Method | Signature | Description |
| --- | --- | --- |
| `create_node` | `(&self, label: &str, props: HashMap<String, PropertyValue>) -> Result<u64>` | Create a node and return its ID |
| `get_node` | `(&self, id: u64) -> Result<Node>` | Retrieve a node by ID |
| `node_exists` | `(&self, id: u64) -> bool` | Check whether a node exists |
| `delete_node` | `(&self, id: u64) -> Result<()>` | Delete a node and cascade to connected edges |
| `node_count` | `(&self) -> usize` | Count nodes in the graph (scan-based) |

## Edge Operations

| Method | Signature | Description |
| --- | --- | --- |
| `create_edge` | `(&self, from: u64, to: u64, edge_type: &str, props: HashMap<String, PropertyValue>, directed: bool) -> Result<u64>` | Create an edge between two nodes |
| `get_edge` | `(&self, id: u64) -> Result<Edge>` | Retrieve an edge by ID |

## Traversal Operations

| Method | Signature | Description |
| --- | --- | --- |
| `neighbors` | `(&self, id: u64, edge_type: Option<&str>, direction: Direction) -> Result<Vec<Node>>` | Get immediate neighbors of a node |
| `traverse` | `(&self, start: u64, direction: Direction, max_depth: usize, edge_type: Option<&str>) -> Result<Vec<Node>>` | BFS traversal with depth limit |
| `find_path` | `(&self, from: u64, to: u64) -> Result<Path>` | Find shortest path (BFS, minimum hops) |

## Unified Entity API

| Method | Signature | Description |
| --- | --- | --- |
| `add_entity_edge` | `(&self, from: &str, to: &str, edge_type: &str) -> Result<String>` | Add directed edge between entities |
| `add_entity_edge_undirected` | `(&self, key1: &str, key2: &str, edge_type: &str) -> Result<String>` | Add undirected edge between entities |
| `get_entity_neighbors` | `(&self, key: &str) -> Result<Vec<String>>` | Get all neighbors of an entity |
| `get_entity_neighbors_out` | `(&self, key: &str) -> Result<Vec<String>>` | Get outgoing neighbors of an entity |
| `get_entity_neighbors_in` | `(&self, key: &str) -> Result<Vec<String>>` | Get incoming neighbors of an entity |
| `get_entity_outgoing` | `(&self, key: &str) -> Result<Vec<String>>` | Get outgoing edge keys |
| `get_entity_incoming` | `(&self, key: &str) -> Result<Vec<String>>` | Get incoming edge keys |
| `get_entity_edge` | `(&self, key: &str) -> Result<(String, String, String, bool)>` | Get edge details (from, to, type, directed) |
| `entity_has_edges` | `(&self, key: &str) -> bool` | Check if entity has any edges |
| `delete_entity_edge` | `(&self, key: &str) -> Result<()>` | Delete an entity edge |
| `scan_entities_with_edges` | `(&self) -> Vec<String>` | Scan for entities that have edges |

## Cross-Engine Integration

### Query Router Integration

The Query Router provides unified queries combining graph traversal with
vector similarity:

```rust
// Find entities similar to query that are connected to a specific entity
let items = router.find_similar_connected("query:entity", "connected_to:entity", top_k)?;

// Find graph neighbors sorted by embedding similarity
let items = router.find_neighbors_by_similarity("entity:key", &query_vector, top_k)?;
```

### Tensor Vault Integration

Tensor Vault uses `GraphEngine` for access control relationships. Access
control edges connect principals to secrets with permission metadata.

### Tensor Chain Integration

Tensor Chain uses `GraphEngine` for block linking. Blocks are stored with
`chain:block:{height}` keys and linked via graph edges with type `chain_next`.

## Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| `create_node` | O(1) | Store put |
| `create_edge` | O(1) | Store put + edge list updates |
| `get_node` | O(1) | Store get |
| `get_edge` | O(1) | Store get |
| `neighbors` | O(e) | e = edges from node |
| `traverse` | O(n + e) | BFS over reachable nodes |
| `find_path` | O(n + e) | BFS shortest path |
| `delete_node` | O(e) | Parallel for e >= 100 |
| `node_count` | O(k) | k = total keys (scan-based) |
| `get_edge_list` | O(k) | k = keys in edge list |

### Memory Characteristics

| Data | Storage |
| --- | --- |
| Node | ~50-200 bytes + properties |
| Edge | ~50-150 bytes + properties |
| Edge list entry | ~10 bytes per edge |

## Configuration

The Graph Engine has minimal configuration as it inherits behavior from
TensorStore:

| Setting | Value | Description |
| --- | --- | --- |
| Parallel threshold | 100 | Edge count triggering parallel deletion |
| ID ordering | SeqCst | Atomic ordering for ID generation |

## Dependencies

| Crate | Purpose |
| --- | --- |
| `tensor_store` | Underlying key-value storage |
| `rayon` | Parallel iteration for high-degree node deletion |
| `serde` | Serialization of graph types |

## Related Modules

| Module | Relationship |
| --- | --- |
| [Tensor Store](./tensor-store.md) | Storage backend |
| [Tensor Vault](./tensor-vault.md) | Uses graph for access control |
| [Tensor Chain](./tensor-chain.md) | Uses graph for block linking |
| [Query Router](./query-router.md) | Executes graph queries |
