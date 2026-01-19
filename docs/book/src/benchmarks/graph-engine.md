# graph_engine Benchmarks

The graph engine stores nodes and edges as tensors, using adjacency lists for neighbor lookups.

## Node Creation

| Count | Time | Per Node |
|-------|------|----------|
| 100 | 107us | 1.07us |
| 1,000 | 1.67ms | 1.67us |
| 5,000 | 9.4ms | 1.88us |

## Edge Creation (1,000 edges)

| Type | Time | Per Edge |
|------|------|----------|
| Directed | 2.4ms | 2.4us |
| Undirected | 3.6ms | 3.6us |

## Neighbor Lookup (star graph)

| Fan-out | Time | Per Neighbor |
|---------|------|--------------|
| 10 | 16us | 1.6us |
| 50 | 79us | 1.6us |
| 100 | 178us | 1.8us |

## BFS Traversal (binary tree)

| Depth | Nodes | Time | Per Node |
|-------|-------|------|----------|
| 5 | 31 | 110us | 3.5us |
| 7 | 127 | 442us | 3.5us |
| 9 | 511 | 1.5ms | 2.9us |

## Shortest Path (BFS)

| Graph Type | Size | Time |
|------------|------|------|
| Chain | 10 nodes | 8.2us |
| Chain | 50 nodes | 44us |
| Chain | 100 nodes | 96us |
| Grid | 5x5 | 55us |
| Grid | 10x10 | 265us |

## Analysis

- **Undirected edges**: ~50% slower than directed (stores reverse edge internally)
- **Traversal**: Consistent ~3us per node visited, good BFS implementation
- **Path finding**: Near-linear with path length in chains; grid explores more nodes
- **Parallel delete_node**: Uses rayon for high-degree nodes (>100 edges)
- **Memory overhead**: Each node/edge is a full TensorData (~5-10 allocations)

## Storage Model

graph_engine stores each node and edge as a separate tensor:

```
node:{id} -> TensorData { label, properties... }
edge:{id} -> TensorData { from, to, label, directed, properties... }
adj:{node_id}:out -> TensorData { edge_ids: [...] }
adj:{node_id}:in -> TensorData { edge_ids: [...] }
```

### Trade-offs

- **Pro**: Flexible property storage, consistent with tensor model
- **Con**: More key lookups than traditional adjacency list
- **Pro**: Each component independently updatable

## Complexity

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| create_node | O(1) | Hash insert |
| create_edge | O(1) | Hash insert + adjacency update |
| get_neighbors | O(degree) | Adjacency list lookup |
| bfs | O(V + E) | Standard BFS |
| shortest_path | O(V + E) | BFS-based |
| delete_node | O(degree) | Removes all edges |
