# How To: Graph Traversals

> **See also:** [Graph Engine API Reference](../reference/api/graph-engine.md) |
> [Graph Storage Internals](../explanation/graph-storage.md) |
> [Graph CRUD How-To](graph-crud.md)

This guide covers BFS traversals, shortest path queries, and direction-aware
navigation in the Graph Engine.

## BFS Traversal

Use `traverse` for breadth-first search with a depth limit. The starting node
is always included in the result at depth 0.

```rust
// Traverse outgoing edges up to 5 hops
let nodes = engine.traverse(start_id, Direction::Outgoing, 5, None)?;
```

### Filter by Edge Type

```rust
let deps = engine.traverse(start_id, Direction::Outgoing, 10, Some("DEPENDS_ON"))?;
```

### Choose Direction

| Direction | Use When |
| --- | --- |
| `Direction::Outgoing` | Forward-only traversals (dependency graphs, following links) |
| `Direction::Incoming` | Reverse lookups (finding predecessors, who depends on me) |
| `Direction::Both` | Symmetric relationships (social graphs, undirected links) |

```rust
// Forward only
let downstream = engine.traverse(start, Direction::Outgoing, 4, None)?;

// Reverse only
let upstream = engine.traverse(start, Direction::Incoming, 4, None)?;

// Both directions
let reachable = engine.traverse(start, Direction::Both, 4, None)?;
```

### Set Appropriate Depth

BFS traversal can be expensive on dense graphs. Set `max_depth` based on the
expected graph diameter:

```rust
// For typical social networks, 3-6 hops is usually sufficient
let reachable = engine.traverse(start, Direction::Both, 4, None)?;
```

The traversal is cycle-safe -- it uses a visited set internally so cyclic
graphs will not cause infinite loops.

## Shortest Path

Use `find_path` to find the shortest (minimum hop) path between two nodes.
It uses BFS internally.

```rust
let path = engine.find_path(from_id, to_id)?;

// path.nodes contains the node IDs in order
// path.edges contains the edge IDs connecting them
```

### Same-Node Path

Finding a path from a node to itself returns immediately with a single-node
path:

```rust
let path = engine.find_path(n1, n1)?;
assert_eq!(path.nodes, vec![n1]);
assert!(path.edges.is_empty());
```

### No Path Found

If no path exists between the two nodes, `find_path` returns
`GraphError::PathNotFound`.

## Example: Dependency Graph

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

// Find all transitive dependencies of app
let deps = engine.traverse(app, Direction::Outgoing, 10, Some("DEPENDS_ON"))?;
```

## Example: Social Network Path

```rust
let engine = GraphEngine::new();

let alice = engine.create_node("User", user_props("Alice"))?;
let bob = engine.create_node("User", user_props("Bob"))?;
let charlie = engine.create_node("User", user_props("Charlie"))?;

engine.create_edge(alice, bob, "FRIENDS", HashMap::new(), false)?;
engine.create_edge(bob, charlie, "FRIENDS", HashMap::new(), false)?;

// Find shortest path from Alice to Charlie
let path = engine.find_path(alice, charlie)?;
// path.nodes = [alice, bob, charlie]
```

## Example: High-Degree Hub

When a hub node has many connections, deletion automatically uses parallel
processing for edges above the threshold (100):

```rust
let engine = GraphEngine::new();

let hub = engine.create_node("Hub", HashMap::new())?;
for i in 0..150 {
    let leaf = engine.create_node("Leaf", HashMap::new())?;
    engine.create_edge(hub, leaf, "CONNECTS", HashMap::new(), true)?;
}

// Deletion uses parallel processing (150 >= 100 threshold)
engine.delete_node(hub)?;
```

## Best Practices

- **Bound your depth.** Always set `max_depth` to a reasonable value to avoid
  traversing the entire graph on dense datasets.
- **Filter by edge type.** Use the `edge_type` parameter to restrict traversal
  to relevant relationship types.
- **Choose direction carefully.** `Direction::Both` explores more of the graph
  than `Outgoing` or `Incoming` alone, which increases the number of nodes
  visited.
