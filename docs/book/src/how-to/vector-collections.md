# Manage Vector Collections

Step-by-step guide for creating and using named vector collections to organize
embeddings into isolated namespaces.

> **See Also:**
> [Vector Engine API Reference](../reference/api/vector-engine.md) |
> [Embeddings Search How-To](embeddings-search.md) |
> [Filtered Search How-To](filtered-search.md)

## What Collections Provide

Collections are isolated namespaces within a single Vector Engine instance. Each
collection has its own key space, optional dimension enforcement, and distance
metric configuration. Use collections to separate embeddings by type or purpose
(e.g., "documents", "images", "user-profiles") without key collisions.

Collections use prefixed storage keys for isolation:

| Scope | Storage Key Pattern |
| --- | --- |
| Default (no collection) | `emb:{key}` |
| Named collection | `coll:{collection}:emb:{key}` |
| Entity embeddings | `{entity_key}._embedding` |

## Create a Collection

```rust
use vector_engine::{VectorEngine, VectorCollectionConfig, DistanceMetric};

let engine = VectorEngine::new();

// Create with default config
engine.create_collection("documents", VectorCollectionConfig::default())?;

// Create with custom config
let config = VectorCollectionConfig::default()
    .with_dimension(768)              // Enforce 768-dim vectors
    .with_metric(DistanceMetric::Cosine)
    .with_auto_index(5000);           // Auto-build HNSW at 5000 vectors

engine.create_collection("images", config)?;
```

Creating a collection that already exists returns `VectorError::CollectionExists`.

## List and Inspect Collections

```rust
// List all collection names
let collections = engine.list_collections();

// Check if a collection exists
if engine.collection_exists("documents") {
    // Get the collection's configuration
    let config = engine.get_collection_config("documents");
}
```

## Store Embeddings in a Collection

```rust
// Store a vector in a collection
engine.store_in_collection("documents", "doc1", vec![0.1, 0.2, 0.3])?;

// Store with metadata
use std::collections::HashMap;
use tensor_store::TensorValue;

let mut metadata = HashMap::new();
metadata.insert("title".to_string(), TensorValue::from("Introduction to Rust"));
metadata.insert("author".to_string(), TensorValue::from("Alice"));

engine.store_in_collection_with_metadata(
    "documents",
    "doc1",
    vec![0.1, 0.2, 0.3],
    metadata,
)?;
```

If the collection has a configured dimension, storing a vector with a different
dimension returns `VectorError::DimensionMismatch`.

## Search Within a Collection

```rust
let query = vec![0.1, 0.2, 0.3];

// Basic similarity search in a collection
let results = engine.search_in_collection("documents", &query, 10)?;

for result in &results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

### Filtered Search in a Collection

```rust
use vector_engine::{FilterCondition, FilterValue};

let filter = FilterCondition::Eq(
    "author".to_string(),
    FilterValue::String("Alice".to_string()),
);

let results = engine.search_filtered_in_collection(
    "documents",
    &query,
    10,
    &filter,
    None,  // auto strategy
)?;
```

See [Filtered Search How-To](filtered-search.md) for more filter examples.

## Delete a Collection

Deleting a collection removes all vectors stored in it:

```rust
engine.delete_collection("documents")?;
```

Operating on a deleted or nonexistent collection returns
`VectorError::CollectionNotFound`.

## Persist and Restore Collection Indices

### Save All Collections

```rust
use std::path::Path;

// Save all collections to a directory (one JSON file per collection)
let saved = engine.save_all_indices(Path::new("./vector_index"))?;
println!("Saved {} collections", saved);

// Load all collections from a directory
let loaded = engine.load_all_indices(Path::new("./vector_index"))?;
println!("Loaded {} collections", loaded);
```

### Save a Single Collection

```rust
// JSON format (human-readable)
engine.save_index("documents", Path::new("./documents.json"))?;

// Binary format (compact)
engine.save_index_binary("documents", Path::new("./documents.bin"))?;
```

### Load a Single Collection

```rust
// Load from JSON (returns collection name)
let collection = engine.load_index(Path::new("./documents.json"))?;

// Load from binary
let collection = engine.load_index_binary(Path::new("./documents.bin"))?;
```

### Snapshot for Manual Serialization

```rust
use vector_engine::PersistentVectorIndex;

let index: PersistentVectorIndex = engine.snapshot_collection("documents");
// Serialize `index` with serde as needed
```

## VectorCollectionConfig Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dimension` | `Option<usize>` | `None` | Enforced dimension (rejects mismatches) |
| `distance_metric` | `DistanceMetric` | `Cosine` | Default metric for this collection |
| `auto_index` | `bool` | `false` | Auto-build HNSW on threshold |
| `auto_index_threshold` | `usize` | `1000` | Vector count to trigger auto-index |

## Related Pages

- [Embeddings Search How-To](embeddings-search.md) -- basic storage and search operations
- [Filtered Search How-To](filtered-search.md) -- metadata-based search filtering
- [Vector Engine API Reference](../reference/api/vector-engine.md) -- complete type and config tables
- [Vector Search Modes](../explanation/vector-search-modes.md) -- brute-force vs HNSW architecture
