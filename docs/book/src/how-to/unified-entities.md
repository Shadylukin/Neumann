# Unified Entities

## Goal

Create unified entities that span relational, graph, and vector engines, link
them together, and query across modalities.

## Initialize UnifiedEngine

```rust
use tensor_unified::UnifiedEngine;
use tensor_store::TensorStore;

// Create with a new store
let engine = UnifiedEngine::new();

// Create with a shared store (recommended for cross-engine use)
let store = TensorStore::new();
let engine = UnifiedEngine::with_store(store);

// Create with existing engines
let engine = UnifiedEngine::with_engines(store, relational, graph, vector);
```

## Create an Entity with Fields and Embedding

```rust
use std::collections::HashMap;

let mut fields = HashMap::new();
fields.insert("name".to_string(), "Alice".to_string());
fields.insert("role".to_string(), "admin".to_string());

engine.create_entity(
    "user:1",
    fields,
    Some(vec![0.1, 0.2, 0.3, 0.4])  // Optional embedding
).await?;
```

Use prefixed keys to distinguish entity types:

```text
"user:123"       -- User entities
"doc:456"        -- Document entities
"hub:main"       -- Hub/aggregate entities
"edge:follows:1" -- Edge entities (auto-generated)
```

## Create an Entity with Vector Metadata

Use `create_entity_unified` when you plan to run filtered similarity searches.
This stores fields as vector metadata alongside the embedding, eliminating a
separate lookup:

```rust
let mut fields = HashMap::new();
fields.insert("title".to_string(), "Introduction to Rust".to_string());
fields.insert("author".to_string(), "Alice".to_string());

engine.create_entity_unified(
    "doc:1",
    fields,
    Some(vec![0.1, 0.2, 0.3, 0.4])
).await?;
```

## Connect Entities via Graph Edges

```rust
let edge_id = engine.connect_entities("user:1", "user:2", "follows").await?;
```

This creates three store entries: the edge itself, plus updates to the source
entity's `_out` list and the target entity's `_in` list.

## Retrieve an Entity

```rust
let item = engine.get_entity("user:1").await?;
println!("Fields: {:?}", item.data);
println!("Embedding: {:?}", item.embedding);
```

Returns `UnifiedError::NotFound` if the entity has neither fields nor embedding.

## Find Similar AND Connected Entities

```rust
let results = engine.find_similar_connected(
    "user:1",      // Query entity (uses its embedding)
    "hub:main",    // Find entities connected to this
    10             // Top-k results
).await?;
```

The query entity must have an embedding. The `connected_to` entity must have
graph edges. Results are returned with source `"vector+graph"`.

## Find Similar Connected with Metadata Filter

```rust
use vector_engine::{FilterCondition, FilterValue};

let filter = FilterCondition::Eq(
    "category".to_string(),
    FilterValue::String("article".to_string())
);

let results = engine.find_similar_connected_filtered(
    "user:1",
    "hub:main",
    Some(&filter),
    10
).await?;
```

## Find Neighbors Sorted by Similarity

```rust
let results = engine.find_neighbors_by_similarity(
    "user:1",                    // Entity to get neighbors of
    &[0.1, 0.2, 0.3, 0.4],      // Query vector
    10                           // Top-k results
).await?;
```

Neighbors without embeddings are silently skipped. Ensure the query vector has
the same dimensions as stored embeddings.

## Use Collections for Scoped Search

```rust
use vector_engine::VectorCollectionConfig;

// Create a collection
let config = VectorCollectionConfig::default()
    .with_dimension(768)
    .with_metric(DistanceMetric::Cosine);

engine.create_entity_collection("documents", config)?;

// Store an entity in the collection
let mut fields = HashMap::new();
fields.insert("title".to_string(), "ML Paper".to_string());

engine.create_entity_in_collection(
    "documents",
    "paper:1",
    fields,
    vec![0.1; 768]
).await?;

// Search within the collection
let results = engine.find_similar_in_collection(
    "documents",
    &query_embedding,
    None,  // No filter
    10
).await?;

// Search with metadata filter
let filter = FilterCondition::Eq("author".to_string(), "Alice".into());
let results = engine.find_similar_in_collection(
    "documents",
    &query_embedding,
    Some(&filter),
    10
).await?;

// List and delete collections
let collections = engine.list_entity_collections();
engine.delete_entity_collection("documents")?;
```

## Find Nodes and Edges

```rust
// Find all nodes with optional label filter
let nodes = engine.find_nodes(Some("person"), None).await?;

// Find all edges with optional type filter
let edges = engine.find_edges(Some("follows"), None).await?;

// Find with pattern and limit
let pattern = FindPattern::Nodes { label: Some("document".to_string()) };
let result = engine.find(&pattern, Some(10)).await?;
```

## Batch Operations

```rust
// Store multiple embeddings
let items = vec![
    ("doc1".to_string(), vec![0.1, 0.2, 0.3]),
    ("doc2".to_string(), vec![0.4, 0.5, 0.6]),
];
let count = engine.embed_batch(items).await?;

// Create multiple entities
let entities: Vec<EntityInput> = vec![
    ("e1".to_string(), HashMap::from([("name".to_string(), "A".to_string())]), None),
    ("e2".to_string(), HashMap::from([("name".to_string(), "B".to_string())]), Some(vec![0.1, 0.2])),
];
let count = engine.create_entities_batch(entities).await?;
```

Batch operations process sequentially. Failed individual operations are counted
as failures but do not abort the batch.

## Run Cross-Modal Tensor Contraction

```rust
let config = ContractionConfig {
    direction: GraphDirection::Symmetric,
    normalization: Normalization::TotalWeight,
    edge_type: Some("FRIEND".into()),
    exclude_owned: true,
    top_k: 10,
};

let result = engine.cross_modal_contraction(
    "alice",         // source entity key
    "purchases",     // relational table
    "buyer",         // column identifying the intermediary
    "item",          // column identifying the item
    &config,
    None,            // optional category mask
).await?;

for item in &result.items {
    println!("{}: {:.3} ({} contributors)",
        item.item_key, item.score, item.contributors);
}
```

## Use from QueryRouter

```rust
use query_router::QueryRouter;
use tensor_store::TensorStore;

let store = TensorStore::new();
let router = QueryRouter::with_shared_store(store);

// Query language commands
router.execute_parsed("ENTITY CREATE 'doc:1' {title: 'Hello'} EMBEDDING [0.1, 0.2]")?;
router.execute_parsed("ENTITY CONNECT 'user:1' -> 'doc:1' : authored")?;
router.execute_parsed("SIMILAR 'query:doc' CONNECTED TO 'user:1' LIMIT 5")?;

// Rust API methods delegate to UnifiedEngine
let results = router.find_neighbors_by_similarity("entity:1", &[0.1, 0.2], 10)?;
let results = router.find_similar_connected("query:1", "hub:1", 5)?;
```

## Query Language Syntax

```sql
-- Create entity with fields and embedding
ENTITY CREATE 'user:1' {name: 'Alice', role: 'admin'} EMBEDDING [0.1, 0.2, 0.3]

-- Create entity with fields only
ENTITY CREATE 'user:2' {name: 'Bob'}

-- Connect entities
ENTITY CONNECT 'user:1' -> 'user:2' : follows

-- Find similar entities that are also connected to a hub
SIMILAR 'query:key' CONNECTED TO 'hub:entity' LIMIT 10

-- Find neighbors sorted by similarity
NEIGHBORS 'entity:key' BY SIMILAR [0.1, 0.2, 0.3] LIMIT 10
```

## Tips

- Keep embedding dimensions consistent across entities. Dimension mismatches
  cause neighbors to be silently skipped in similarity queries.
- Build an HNSW index via `QueryRouter::build_vector_index()` for large vector
  sets (>5000 entities) to get O(log n) similarity search instead of O(n).
- Ensure the `connected_to` entity actually has edges; empty neighbors produce
  empty results from `find_similar_connected`.
- For high-degree nodes, `find_neighbors_by_similarity` computes one cosine
  similarity per neighbor -- consider the degree distribution before querying.

## See Also

- **Reference**: [Tensor Unified API](../reference/api/tensor-unified.md)
- **Explanation**: [Unified Entities](../explanation/unified-entities.md)
- **Architecture**: [Tensor Unified](../reference/api/tensor-unified.md)
