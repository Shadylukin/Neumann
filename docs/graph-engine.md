# Graph Engine

Module 3 of Neumann. Provides graph operations on top of the Tensor Store.

## Design Principles

1. **Layered Architecture**: Depends only on Tensor Store for persistence
2. **Direction-Aware**: Supports both directed and undirected edges
3. **BFS Traversal**: Breadth-first search for shortest paths
4. **Cycle-Safe**: Handles cyclic graphs without infinite loops
5. **Unified Entities**: Edges can connect shared entities across engines
6. **Thread Safety**: Inherits from Tensor Store

## Data Model

### Nodes

Nodes have a label and properties:

```rust
let mut props = HashMap::new();
props.insert("name".to_string(), PropertyValue::String("Alice".into()));
props.insert("age".to_string(), PropertyValue::Int(30));

let node_id = engine.create_node("Person", props)?;
```

### Edges

Edges connect nodes with a type, properties, and direction:

```rust
// Directed edge: Alice -> Bob
engine.create_edge(alice_id, bob_id, "KNOWS", HashMap::new(), true)?;

// Undirected edge: Alice -- Bob (friendship)
engine.create_edge(alice_id, bob_id, "FRIENDS", HashMap::new(), false)?;
```

### Property Types

| Type | Rust Type | Description |
|------|-----------|-------------|
| `Null` | - | NULL value |
| `Int` | `i64` | 64-bit signed integer |
| `Float` | `f64` | 64-bit floating point |
| `String` | `String` | UTF-8 string |
| `Bool` | `bool` | Boolean |

## API Reference

### Node Operations

```rust
let engine = GraphEngine::new();

// Create node
let id = engine.create_node("Person", properties)?;

// Get node
let node = engine.get_node(id)?;

// Check existence
engine.node_exists(id)?;  // -> bool

// Delete node (also deletes connected edges)
engine.delete_node(id)?;
```

### Edge Operations

```rust
// Create directed edge
let edge_id = engine.create_edge(from, to, "KNOWS", properties, true)?;

// Create undirected edge
let edge_id = engine.create_edge(from, to, "FRIENDS", properties, false)?;

// Get edge
let edge = engine.get_edge(edge_id)?;
```

### Traversal Operations

```rust
// Get neighbors
let neighbors = engine.neighbors(node_id, None, Direction::Both)?;

// Get neighbors by edge type
let friends = engine.neighbors(node_id, Some("FRIENDS"), Direction::Both)?;

// Traverse graph (BFS)
let nodes = engine.traverse(start_id, Direction::Outgoing, max_depth, None)?;

// Find shortest path
let path = engine.find_path(from_id, to_id)?;
```

### Direction

| Direction | Description |
|-----------|-------------|
| `Outgoing` | Follow edges away from the node |
| `Incoming` | Follow edges toward the node |
| `Both` | Follow edges in either direction |

### Unified Entity API

Connect any shared entities (not just graph nodes) for cross-engine queries:

```rust
// Create engine with shared store
let store = TensorStore::new();
let engine = GraphEngine::with_store(store.clone());

// Add an edge between any entities (e.g., user:1 and user:2)
engine.add_entity_edge("user:1", "user:2", "follows")?;

// Get neighbors of an entity
let neighbors = engine.get_entity_neighbors("user:1")?;
// Returns ["user:2"]
```

Entity edges use the `_out` and `_in` reserved fields in TensorData. This enables the same entity key to have relational fields, graph connections, and a vector embedding.

## Storage Model

Nodes and edges are stored in Tensor Store:

| Key Pattern | Content |
|-------------|---------|
| `node:{id}` | Node data (label, properties) |
| `node:{id}:out` | Outgoing edge IDs |
| `node:{id}:in` | Incoming edge IDs |
| `edge:{id}` | Edge data (from, to, type, properties, directed) |

## Error Handling

| Error | Cause |
|-------|-------|
| `NodeNotFound` | Node doesn't exist |
| `EdgeNotFound` | Edge doesn't exist |
| `PathNotFound` | No path between nodes |
| `StorageError` | Underlying Tensor Store error |

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `create_node` | O(1) | Store put |
| `create_edge` | O(1) | Store put + edge list updates |
| `get_node` | O(1) | Store get |
| `neighbors` | O(e) | e = edges from node |
| `traverse` | O(n + e) | BFS over reachable nodes |
| `find_path` | O(n + e) | BFS shortest path |

## Test Coverage

| Test | What It Verifies |
|------|------------------|
| `create_1000_nodes_with_edges_traverse` | Scale + traversal correctness |
| `traverse_handles_cycles` | No infinite loops |
| `find_path_shortest` | BFS finds shortest path |
| `directed_vs_undirected_edges` | Edge direction behavior |
| `neighbors_by_edge_type` | Edge type filtering |
| `self_loop_edge` | Self-referential edges |
| `delete_node` | Cascade edge deletion |

## Usage Examples

### Social Network

```rust
let engine = GraphEngine::new();

// Create users
let alice = engine.create_node("User", user_props("Alice"))?;
let bob = engine.create_node("User", user_props("Bob"))?;
let charlie = engine.create_node("User", user_props("Charlie"))?;

// Create friendships (undirected)
engine.create_edge(alice, bob, "FRIENDS", HashMap::new(), false)?;
engine.create_edge(bob, charlie, "FRIENDS", HashMap::new(), false)?;

// Find path from Alice to Charlie
let path = engine.find_path(alice, charlie)?;
// path.nodes = [alice, bob, charlie]

// Get Alice's friends
let friends = engine.neighbors(alice, Some("FRIENDS"), Direction::Both)?;
```

### Dependency Graph

```rust
let engine = GraphEngine::new();

// Create packages
let app = engine.create_node("Package", package_props("app"))?;
let lib_a = engine.create_node("Package", package_props("lib-a"))?;
let lib_b = engine.create_node("Package", package_props("lib-b"))?;

// Create dependencies (directed)
engine.create_edge(app, lib_a, "DEPENDS_ON", HashMap::new(), true)?;
engine.create_edge(app, lib_b, "DEPENDS_ON", HashMap::new(), true)?;
engine.create_edge(lib_a, lib_b, "DEPENDS_ON", HashMap::new(), true)?;

// Find all dependencies of app
let deps = engine.traverse(app, Direction::Outgoing, 10, Some("DEPENDS_ON"))?;
```

## Future Considerations

Not implemented (out of scope for Module 3):

- **Weighted Paths**: Dijkstra's algorithm for weighted edges
- **Pattern Matching**: Cypher-style graph patterns
- **Indexes**: Edge type or property indexes
- **Transactions**: ACID guarantees
- **Batch Operations**: Bulk node/edge creation
