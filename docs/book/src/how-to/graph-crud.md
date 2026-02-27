# How To: Graph CRUD Operations

> **See also:** [Graph Engine API Reference](../reference/api/graph-engine.md) |
> [Graph Storage Internals](../explanation/graph-storage.md) |
> [Graph Traversals How-To](graph-traversals.md)

This guide covers creating, reading, updating, and deleting nodes and edges
in the Graph Engine.

## Create an Engine

```rust
// Standalone engine with its own store
let engine = GraphEngine::new();

// Engine sharing a store with other engines (recommended for cross-engine queries)
let store = TensorStore::new();
let engine = GraphEngine::with_store(store.clone());
```

Use `with_store` when entities need to combine relational, graph, and vector
data through the Unified Entity API.

## Create Nodes

```rust
let mut props = HashMap::new();
props.insert("name".to_string(), PropertyValue::String("Alice".into()));
props.insert("age".to_string(), PropertyValue::Int(30));
let id = engine.create_node("Person", props)?;
```

Each node receives a unique monotonically increasing ID.

## Read Nodes

```rust
// Get a node by ID
let node = engine.get_node(id)?;

// Check existence without fetching
let exists = engine.node_exists(id);

// Count all nodes in the graph
let count = engine.node_count();
```

## Delete Nodes

Deleting a node cascades to all connected edges. For high-degree nodes
(>= 100 edges), deletion runs in parallel using rayon.

```rust
engine.delete_node(id)?;
```

## Create Edges

### Directed Edges

```rust
let edge_id = engine.create_edge(from, to, "KNOWS", properties, true)?;
```

A directed edge appears in the source's outgoing list and the target's
incoming list.

### Undirected Edges

```rust
let edge_id = engine.create_edge(from, to, "FRIENDS", properties, false)?;
```

An undirected edge is added to both nodes' outgoing **and** incoming lists,
enabling traversal from either endpoint regardless of the direction filter.

## Read Edges

```rust
let edge = engine.get_edge(edge_id)?;
```

## Get Neighbors

```rust
// All neighbors, both directions
let neighbors = engine.neighbors(node_id, None, Direction::Both)?;

// Filter by edge type
let friends = engine.neighbors(node_id, Some("FRIENDS"), Direction::Both)?;
```

## Unified Entity API -- Cross-Engine CRUD

When your entities span multiple engines, use the entity API instead of the
node API. Entity edges are stored in `_out` and `_in` fields on `TensorData`,
keeping all engines in sync through a shared store.

### Create Entity Edges

```rust
let store = TensorStore::new();
let engine = GraphEngine::with_store(store.clone());

// Directed entity edge
let edge_key = engine.add_entity_edge("user:1", "user:2", "follows")?;

// Undirected entity edge
let edge_key = engine.add_entity_edge_undirected("user:1", "user:2", "friend")?;
```

### Read Entity Neighbors

```rust
let all_neighbors = engine.get_entity_neighbors("user:1")?;
let out_neighbors = engine.get_entity_neighbors_out("user:1")?;
let in_neighbors  = engine.get_entity_neighbors_in("user:1")?;
```

### Read Entity Edge Details

```rust
let (from, to, edge_type, directed) = engine.get_entity_edge(&edge_key)?;
```

### Check for Edges

```rust
let has_edges = engine.entity_has_edges("user:1");
```

### Delete Entity Edges

```rust
engine.delete_entity_edge(&edge_key)?;
```

### Scan Entities with Edges

```rust
let entities = engine.scan_entities_with_edges();
```

## Example: Social Network

```rust
let engine = GraphEngine::new();

// Create users
let alice = engine.create_node("User", user_props("Alice"))?;
let bob = engine.create_node("User", user_props("Bob"))?;
let charlie = engine.create_node("User", user_props("Charlie"))?;

// Create friendships (undirected)
engine.create_edge(alice, bob, "FRIENDS", HashMap::new(), false)?;
engine.create_edge(bob, charlie, "FRIENDS", HashMap::new(), false)?;

// Get Alice's friends
let friends = engine.neighbors(alice, Some("FRIENDS"), Direction::Both)?;
```

## Example: Cross-Engine Unified Entities

```rust
let store = TensorStore::new();
let graph = GraphEngine::with_store(store.clone());

// Add graph edges between entities
graph.add_entity_edge("user:1", "post:1", "created")?;
graph.add_entity_edge("user:2", "post:1", "liked")?;

// Query relationships
let creators = graph.get_entity_neighbors_in("post:1")?;
```

## Best Practices

- **Use shared store for cross-engine queries.** Create a `TensorStore` first
  and pass it to all engines with `with_store`.
- **Prefer the Entity API for cross-engine data.** It preserves all fields
  across relational, graph, and vector engines.
- **Batch edge creation carefully.** Each `create_edge` call performs multiple
  store operations; consider the overhead when creating many edges.
