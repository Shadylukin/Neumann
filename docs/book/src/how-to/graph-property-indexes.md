# How To: Graph Property Indexes

> **See also:** [Graph Engine API Reference](../reference/api/graph-engine.md) |
> [Graph CRUD How-To](graph-crud.md) |
> [Graph Storage Internals](../explanation/graph-storage.md)

This guide covers using property values on graph nodes and edges, and
working with property-based filtering through the Query Router.

## Property Types

Graph nodes and edges support properties via the `PropertyValue` enum:

| Variant | Rust Type | Example |
| --- | --- | --- |
| `Null` | --- | `PropertyValue::Null` |
| `Int` | `i64` | `PropertyValue::Int(42)` |
| `Float` | `f64` | `PropertyValue::Float(3.14)` |
| `String` | `String` | `PropertyValue::String("Alice".into())` |
| `Bool` | `bool` | `PropertyValue::Bool(true)` |

Note: `ScalarValue::Bytes` maps to `PropertyValue::Null` -- binary data is
not supported as a graph property.

## Setting Properties on Nodes

Properties are set at creation time via a `HashMap`:

```rust
let mut props = HashMap::new();
props.insert("name".to_string(), PropertyValue::String("Alice".into()));
props.insert("age".to_string(), PropertyValue::Int(30));
props.insert("active".to_string(), PropertyValue::Bool(true));
let id = engine.create_node("Person", props)?;
```

## Setting Properties on Edges

Edge properties work the same way:

```rust
let mut edge_props = HashMap::new();
edge_props.insert("since".to_string(), PropertyValue::Int(2020));
edge_props.insert("weight".to_string(), PropertyValue::Float(0.85));
let edge_id = engine.create_edge(from, to, "KNOWS", edge_props, true)?;
```

## Reading Properties

Properties are available on the `Node` and `Edge` structs returned by
`get_node` and `get_edge`:

```rust
let node = engine.get_node(id)?;
// Access node.label, node.properties, etc.

let edge = engine.get_edge(edge_id)?;
// Access edge.edge_type, edge.properties, etc.
```

## Filtering Neighbors by Edge Type

While the Graph Engine does not expose standalone property indexes, it
supports filtering by edge type during neighbor and traversal queries:

```rust
// Only follow "KNOWS" edges
let contacts = engine.neighbors(node_id, Some("KNOWS"), Direction::Outgoing)?;

// Traverse only "DEPENDS_ON" edges
let deps = engine.traverse(start, Direction::Outgoing, 10, Some("DEPENDS_ON"))?;
```

## Cross-Engine Property Queries

For richer property filtering, use the Query Router which combines graph
traversal with relational-style filtering and vector similarity:

```rust
// Find graph neighbors sorted by embedding similarity
let items = router.find_neighbors_by_similarity("entity:key", &query_vector, top_k)?;
```

The Unified Entity API stores properties alongside graph edges in `TensorData`,
meaning the same entity can be queried by relational filters, graph traversal,
and vector similarity through a single shared store.

## Best Practices

- **Set properties at creation time.** Node and edge properties are defined
  in the `HashMap` passed to `create_node` and `create_edge`.
- **Use edge type filtering** to narrow traversals when your graph has
  multiple relationship types.
- **Use the Query Router** for complex property-based queries that combine
  graph structure with relational or vector criteria.
- **Avoid binary properties.** `ScalarValue::Bytes` converts to
  `PropertyValue::Null`; store binary data in Tensor Blob instead.
