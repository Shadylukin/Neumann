# Tensor Unified

Module 12 of Neumann. Provides cross-engine operations and unified entity management.

## Design Principles

1. **Cross-Engine Abstraction**: Single interface for operations spanning multiple engines
2. **Unified Entities**: Entities can have relational fields, graph connections, and embeddings
3. **Composable Queries**: Combine vector similarity with graph connectivity
4. **Async-First**: All cross-engine operations support async execution
5. **Thread Safety**: Inherits from underlying engines via TensorStore

## API Reference

### Initialization

```rust
use tensor_unified::UnifiedEngine;
use tensor_store::TensorStore;

// Create with new store
let engine = UnifiedEngine::new();

// Create with shared store
let store = TensorStore::new();
let engine = UnifiedEngine::with_store(store);
```

### Unified Entity Operations

```rust
use std::collections::HashMap;

// Create an entity with fields and optional embedding
let mut fields = HashMap::new();
fields.insert("name".to_string(), "Alice".to_string());
fields.insert("role".to_string(), "admin".to_string());

engine.create_entity(
    "user:1",
    fields,
    Some(vec![0.1, 0.2, 0.3, 0.4])  // Optional embedding
)?;

// Connect entities via graph edge
let edge_id = engine.connect_entities("user:1", "user:2", "follows")?;
```

### Cross-Engine Queries

```rust
// Find entities similar to query AND connected to target
let results = engine.find_similar_connected(
    "user:1",      // Query entity (uses its embedding)
    "hub:main",    // Find entities connected to this
    10             // Top-k results
)?;

// Find neighbors of an entity sorted by similarity to a vector
let results = engine.find_neighbors_by_similarity(
    "user:1",                    // Entity to get neighbors of
    &[0.1, 0.2, 0.3, 0.4],      // Query vector
    10                           // Top-k results
)?;
```

### UnifiedItem Result

All cross-engine queries return `Vec<UnifiedItem>`:

```rust
pub struct UnifiedItem {
    pub source: String,                    // "relational", "graph", "vector", or combined
    pub id: String,                        // Entity key
    pub data: HashMap<String, String>,     // Entity fields
    pub embedding: Option<Vec<f32>>,       // Optional embedding
    pub score: Option<f32>,                // Similarity score if applicable
}
```

## Query Language

Cross-engine operations are exposed via the query language:

### Entity Creation

```sql
-- Create entity with fields and embedding
ENTITY CREATE 'user:1' {name: 'Alice', role: 'admin'} EMBEDDING [0.1, 0.2, 0.3]

-- Create entity with fields only
ENTITY CREATE 'user:2' {name: 'Bob'}

-- Connect entities
ENTITY CONNECT 'user:1' -> 'user:2' : follows
```

### Cross-Engine Similarity

```sql
-- Find similar entities that are also connected to a hub
SIMILAR 'query:key' CONNECTED TO 'hub:entity' LIMIT 10

-- Find neighbors sorted by similarity
NEIGHBORS 'entity:key' BY SIMILAR [0.1, 0.2, 0.3] LIMIT 10
```

## Integration with QueryRouter

The QueryRouter delegates cross-engine operations to UnifiedEngine:

```rust
use query_router::QueryRouter;

let router = QueryRouter::new();

// These commands use UnifiedEngine internally
router.execute_parsed("ENTITY CREATE 'doc:1' {title: 'Hello'} EMBEDDING [0.1, 0.2]")?;
router.execute_parsed("ENTITY CONNECT 'user:1' -> 'doc:1' : authored")?;
router.execute_parsed("SIMILAR 'query:doc' CONNECTED TO 'user:1' LIMIT 5")?;
```

## Error Handling

| Error | Cause |
|-------|-------|
| `EntityNotFound` | Referenced entity doesn't exist |
| `EmbeddingNotFound` | Entity has no embedding |
| `ConnectionFailed` | Could not create graph edge |
| `InvalidQuery` | Malformed cross-engine query |

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `create_entity` | O(1) | Single store put + optional embedding |
| `connect_entities` | O(1) | Single edge creation |
| `find_similar_connected` | O(k log n) | HNSW search + graph intersection |
| `find_neighbors_by_similarity` | O(d * k) | Neighbor fetch + k similarity computations |

## Thread Safety

UnifiedEngine is thread-safe via:
- `Arc<VectorEngine>`, `Arc<GraphEngine>`, `Arc<RelationalEngine>`
- All underlying engines share thread-safe TensorStore (DashMap)
- No lock poisoning (parking_lot semantics)

## Test Coverage

| Test | What It Verifies |
|------|------------------|
| `create_entity_with_embedding` | Entity creation with all components |
| `connect_entities` | Graph edge creation between entities |
| `find_similar_connected` | Cross-engine query correctness |
| `concurrent_entity_creation` | Thread safety under parallel load |
