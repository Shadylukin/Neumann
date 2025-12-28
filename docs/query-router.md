# Query Router

Module 5 of Neumann. Provides unified query execution across all engines.

## Design Principles

1. **Unified Interface**: Single entry point for all query types
2. **Engine Dispatch**: Routes queries to appropriate engine based on statement type
3. **AST-Based Execution**: Uses Neumann Parser for structured query handling
4. **Result Aggregation**: Consistent QueryResult type across all operations
5. **Shared Storage**: All engines share the same TensorStore for unified entities
6. **Cross-Engine Queries**: Combine graph connections with vector similarity
7. **Serializable Results**: All result types implement `serde::Serialize`/`Deserialize`

## API Reference

### Initialization

```rust
use query_router::QueryRouter;
use tensor_store::TensorStore;

// Create with independent engines
let router = QueryRouter::new();

// Create with shared storage (enables unified entities)
let store = TensorStore::new();
let router = QueryRouter::with_shared_store(store);
```

### Query Execution

Multiple execution methods are available:

```rust
// String-based execution (legacy, uses regex parsing)
let result = router.execute("SELECT * FROM users")?;

// AST-based execution (recommended, uses neumann_parser)
let result = router.execute_parsed("SELECT * FROM users")?;

// Async execution (for concurrent operations)
let result = router.execute_async("SELECT * FROM users").await?;
let result = router.execute_parsed_async("SELECT * FROM users").await?;
```

### Async Execution

For I/O-bound or cross-engine operations, async methods provide better concurrency:

```rust
use query_router::QueryRouter;

let router = QueryRouter::new();

// Concurrent queries
let (users, posts, similar) = tokio::join!(
    router.execute_parsed_async("SELECT * FROM users"),
    router.execute_parsed_async("SELECT * FROM posts"),
    router.execute_parsed_async("SIMILAR 'doc:1' LIMIT 10"),
);
```

### Query Result

All queries return `QueryResult`:

```rust
pub enum QueryResult {
    // Relational results
    Rows(Vec<Row>),
    RowsAffected(usize),
    Created,

    // Graph results
    NodeCreated(u64),
    EdgeCreated(u64),
    Nodes(Vec<GraphNode>),
    Edges(Vec<GraphEdge>),
    Path(Vec<u64>),
    NoPath,

    // Vector results
    Stored,
    Embedding(Vec<f64>),
    Deleted,
    Similar(Vec<SimilarResult>),
}
```

## Supported Queries

### Relational Operations

```sql
-- Create a table
CREATE TABLE users (id, name, email)

-- Insert data
INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')

-- Query data
SELECT * FROM users
SELECT * FROM users WHERE id = 1

-- Update data
UPDATE users SET name = 'Bob' WHERE id = 1

-- Delete data
DELETE FROM users WHERE id = 1
```

### Graph Operations

```sql
-- Create a node
NODE person {name: 'Alice', age: 30}

-- Create an edge
EDGE person:1 friend person:2 {since: 2020}

-- Find neighbors
NEIGHBORS person:1
NEIGHBORS person:1 friend OUTGOING

-- Find path
PATH person:1 TO person:5
PATH person:1 TO person:5 VIA friend

-- Find nodes/edges
FIND NODE WHERE label = 'person'
FIND EDGE WHERE type = 'friend'
```

### Vector Operations

```sql
-- Store an embedding
EMBED doc1 0.1, 0.2, 0.3, 0.4

-- Store with bracket syntax
EMBED doc2 [0.1, 0.2, 0.3, 0.4]

-- Find similar embeddings (default: COSINE)
SIMILAR 'doc1' LIMIT 5

-- Find similar with explicit distance metric
SIMILAR 'doc1' LIMIT 5 COSINE
SIMILAR 'doc1' LIMIT 5 EUCLIDEAN
SIMILAR 'doc1' LIMIT 5 DOT_PRODUCT

-- Search by vector literal
SIMILAR [0.1, 0.2, 0.3, 0.4] LIMIT 10 EUCLIDEAN

-- Batch embedding storage
EMBED BATCH [('key1', [0.1, 0.2]), ('key2', [0.3, 0.4])]

-- Introspection
SHOW EMBEDDINGS LIMIT 100
COUNT EMBEDDINGS
```

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `COSINE` | Cosine similarity (default) | Semantic similarity, normalized vectors |
| `EUCLIDEAN` | Euclidean distance (L2) | Spatial distance, absolute positions |
| `DOT_PRODUCT` | Dot product | Magnitude-aware similarity |

**Note**: EUCLIDEAN scores are transformed as `1/(1+distance)` so higher is always better.

### Unified Entity Operations

```sql
-- Create entity with fields and embedding
ENTITY CREATE 'user:1' {name: 'Alice', role: 'admin'} EMBEDDING [0.1, 0.2, 0.3]

-- Connect entities
ENTITY CONNECT 'user:1' -> 'doc:1' : authored

-- Cross-engine similarity search
SIMILAR 'query:key' CONNECTED TO 'hub:entity' LIMIT 10
```

### Cross-Engine Queries (Rust API)

The QueryRouter provides methods for queries that span multiple engines:

```rust
use query_router::QueryRouter;
use tensor_store::TensorStore;

let store = TensorStore::new();
let mut router = QueryRouter::with_shared_store(store);

// Set up entities with embeddings
router.vector().set_entity_embedding("user:1", vec![0.1, 0.2, 0.3])?;
router.vector().set_entity_embedding("user:2", vec![0.15, 0.25, 0.35])?;
router.vector().set_entity_embedding("user:3", vec![0.9, 0.8, 0.7])?;

// Connect entities via graph edges
router.connect_entities("user:1", "user:2", "follows")?;

// Build HNSW index for fast similarity search (O(log n) instead of O(n))
router.build_vector_index()?;

// Cross-engine query: find neighbors of an entity sorted by similarity
let query_vec = vec![0.1, 0.2, 0.3];
let results = router.find_neighbors_by_similarity("user:1", &query_vec, 10)?;
// Returns neighbors of user:1 ranked by cosine similarity to query_vec

// Cross-engine query: find entities similar to query_key AND connected to connected_to
let results = router.find_similar_connected(
    "user:1",           // query entity (use its embedding)
    "user:2",           // find entities connected to this
    5,                  // top_k
)?;
// Returns intersection of similar entities and graph neighbors
```

| Method | Description |
|--------|-------------|
| `build_vector_index()` | Build HNSW index for O(log n) similarity search |
| `connect_entities(from, to, edge_type)` | Add a graph edge between entities |
| `find_neighbors_by_similarity(key, query, k)` | Get neighbors sorted by vector similarity |
| `find_similar_connected(query_key, connected_to, k)` | Find entities similar to query AND connected to target |

## Error Handling

| Error | Cause |
|-------|-------|
| `ParseError` | Invalid query syntax |
| `TableNotFound` | Referenced table doesn't exist |
| `ColumnNotFound` | Referenced column doesn't exist |
| `NodeNotFound` | Referenced node doesn't exist |
| `EdgeNotFound` | Referenced edge doesn't exist |
| `EmbeddingNotFound` | Referenced embedding doesn't exist |
| `InvalidArgument` | Invalid argument value |
| `EngineError` | Underlying engine error |

## Storage Model

Query Router doesn't store data directly. It delegates to:

| Engine | Handles |
|--------|---------|
| Relational Engine | Tables, rows, columns |
| Graph Engine | Nodes, edges, paths |
| Vector Engine | Embeddings, similarity search |

All engines share the same Tensor Store instance.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Parse | O(n) | n = query length |
| SELECT | O(m) | m = rows in table |
| INSERT | O(1) | Single row insert |
| NODE | O(1) | Single node create |
| EDGE | O(1) | Single edge create |
| PATH | O(V+E) | BFS traversal |
| SIMILAR (brute-force) | O(n*d) | n = embeddings, d = dimensions |
| SIMILAR (HNSW) | O(log n * d) | After `build_vector_index()` |
| `find_similar_connected` | O(log n) or O(n) | Uses HNSW if index built |

### HNSW Index Performance

Building the HNSW index with `build_vector_index()` provides dramatic speedups:

| Entities | Brute-force | With HNSW | Speedup |
|----------|-------------|-----------|---------|
| 200 | 4.17s | 9.3us | 448,000x |

## Test Coverage

| Test Category | What It Verifies |
|---------------|------------------|
| Relational | CREATE, INSERT, SELECT, UPDATE, DELETE |
| Graph | NODE, EDGE, PATH, NEIGHBORS, FIND |
| Vector | EMBED STORE/GET/DELETE, SIMILAR |
| Error handling | Parse errors, not found errors |
| Edge cases | Empty results, invalid syntax |

## Usage Examples

### Complete Workflow

```rust
use query_router::QueryRouter;

let router = QueryRouter::new();

// Create table and insert data
router.execute_parsed("CREATE TABLE users (id, name, email)")?;
router.execute_parsed("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")?;

// Create graph relationships
router.execute_parsed("NODE person {name: 'Alice'}")?;
router.execute_parsed("NODE person {name: 'Bob'}")?;
router.execute_parsed("EDGE person:1 friend person:2")?;

// Store embeddings
router.execute_parsed("EMBED STORE 'alice' [0.1, 0.2, 0.3]")?;
router.execute_parsed("EMBED STORE 'bob' [0.15, 0.25, 0.35]")?;

// Query across all engines
let users = router.execute_parsed("SELECT * FROM users")?;
let neighbors = router.execute_parsed("NEIGHBORS person:1")?;
let similar = router.execute_parsed("SIMILAR 'alice' LIMIT 1")?;
```

## Future Considerations

Not implemented (out of scope):

- **Cross-engine SQL joins**: SQL syntax for joining relational rows with graph/vector results
- **Transactions**: Multi-statement atomic operations
- **Query planning**: Cost-based optimization

## Known Issues and Limitations

### Distance Metric Syntax
The distance metric keyword (COSINE, EUCLIDEAN, DOT_PRODUCT) must appear **after** the LIMIT clause:
```sql
-- Correct
SIMILAR 'doc:1' LIMIT 10 EUCLIDEAN

-- Incorrect (will not parse)
SIMILAR 'doc:1' METRIC EUCLIDEAN LIMIT 10
```

### Zero-Magnitude Vectors
- COSINE and DOT_PRODUCT return empty results for zero-magnitude query vectors
- EUCLIDEAN correctly handles zero vectors (finds vectors closest to origin)
