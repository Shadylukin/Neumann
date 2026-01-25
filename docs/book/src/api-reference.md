# API Reference

This document provides detailed public API documentation for all Neumann crates.
For auto-generated rustdoc, see [Building Locally](#building-locally).

## Table of Contents

- [tensor_store](#tensor_store) - Core storage layer
- [relational_engine](#relational_engine) - SQL-like tables
- [graph_engine](#graph_engine) - Graph operations
- [vector_engine](#vector_engine) - Embeddings and similarity
- [tensor_chain](#tensor_chain) - Distributed consensus
- [neumann_parser](#neumann_parser) - Query parsing
- [query_router](#query_router) - Query execution
- [tensor_cache](#tensor_cache) - LLM response caching
- [tensor_vault](#tensor_vault) - Encrypted storage
- [tensor_blob](#tensor_blob) - Blob storage
- [tensor_checkpoint](#tensor_checkpoint) - Snapshots

---

## tensor_store

Core key-value storage with HNSW indexing, sparse vectors, and tiered storage.

### Core Types

| Type | Description |
| --- | --- |
| `TensorStore` | Thread-safe key-value store with slab routing |
| `TensorData` | HashMap-based entity with typed fields |
| `TensorValue` | Field value: Scalar, Vector, Sparse, Pointer(s) |
| `ScalarValue` | Null, Bool, Int, Float, String, Bytes |

### TensorStore

```rust
use tensor_store::{TensorStore, TensorData, TensorValue, ScalarValue};

let store = TensorStore::new();

// Store entity
let mut data = TensorData::new();
data.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
data.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
store.put("user:1", data)?;

// Retrieve
let entity = store.get("user:1")?;
assert!(entity.has("name"));

// Check existence
store.exists("user:1");  // -> bool

// Delete
store.delete("user:1")?;

// Scan by prefix
let count = store.scan_count("user:");
```

### TensorData

```rust
let mut data = TensorData::new();

// Set fields
data.set("field", TensorValue::Scalar(ScalarValue::Int(42)));

// Get fields
let value = data.get("field");  // -> Option<&TensorValue>

// Check field existence
data.has("field");  // -> bool

// Field names
let fields: Vec<&str> = data.keys().collect();
```

### HNSW Index

Hierarchical Navigable Small World graph for approximate nearest neighbor search.

```rust
use tensor_store::{HNSWIndex, HNSWConfig, DistanceMetric};

// Create with config
let config = HNSWConfig {
    m: 16,              // Connections per node
    ef_construction: 200,
    ef_search: 50,
    max_elements: 10000,
    distance_metric: DistanceMetric::Cosine,
    ..Default::default()
};
let index = HNSWIndex::new(128, config);  // 128 dimensions

// Insert vector
index.insert("doc:1", &embedding)?;

// Search
let results = index.search(&query_vector, 10)?;
for (key, distance) in results {
    println!("{}: {}", key, distance);
}
```

### Sparse Vectors

Memory-efficient sparse embeddings with 15+ distance metrics.

```rust
use tensor_store::SparseVector;

// Create from dense (auto-detects sparsity)
let sparse = SparseVector::from_dense(&[0.0, 0.5, 0.0, 0.3, 0.0]);

// Create from indices and values
let sparse = SparseVector::new(vec![1, 3], vec![0.5, 0.3], 5)?;

// Operations
let dense = sparse.to_dense();
let dot = sparse.dot(&other_sparse);
let cosine = sparse.cosine_similarity(&other_sparse);
```

### Tiered Storage

Automatic hot/cold storage with mmap backing.

```rust
use tensor_store::{TieredStore, TieredConfig};
use std::path::Path;

let config = TieredConfig {
    hot_capacity: 10000,
    cold_path: Path::new("/data/cold").to_path_buf(),
    migration_threshold: 0.8,
    ..Default::default()
};
let store = TieredStore::new(config)?;

// Automatic migration based on access patterns
store.put("key", data)?;
let value = store.get("key")?;
```

### Cache Ring

Fixed-size eviction cache with multiple strategies.

```rust
use tensor_store::{CacheRing, EvictionStrategy};

let cache = CacheRing::new(1000, EvictionStrategy::LRU);

cache.put("key", value);
let hit = cache.get("key");  // -> Option<V>

// Statistics
let stats = cache.stats();
println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
```

### Consistent Hash Partitioner

Partition routing with virtual nodes.

```rust
use tensor_store::{ConsistentHashPartitioner, ConsistentHashConfig};

let config = ConsistentHashConfig {
    virtual_nodes: 150,
    replication_factor: 3,
};
let partitioner = ConsistentHashPartitioner::new(config);

partitioner.add_node("node1");
partitioner.add_node("node2");

let partition = partitioner.partition("user:123");
let replicas = partitioner.replicas("user:123");
```

---

## relational_engine

SQL-like table operations with SIMD-accelerated filtering.

### Core Types

| Type | Description |
| --- | --- |
| `RelationalEngine` | Main engine with TensorStore backend |
| `Schema` | Table schema with column definitions |
| `Column` | Column name, type, nullability |
| `ColumnType` | Int, Float, String, Bool, Bytes, Json |
| `Value` | Typed query value |
| `Condition` | Composable filter predicate |
| `Row` | Row with ID and values |

### Table Operations

```rust
use relational_engine::{RelationalEngine, Schema, Column, ColumnType};

let engine = RelationalEngine::new();

// Create table
let schema = Schema::new(vec![
    Column::new("name", ColumnType::String),
    Column::new("age", ColumnType::Int),
    Column::new("email", ColumnType::String).nullable(),
]);
engine.create_table("users", schema)?;

// Check existence
engine.table_exists("users")?;  // -> bool

// List tables
let tables = engine.list_tables();  // -> Vec<String>

// Get schema
let schema = engine.get_schema("users")?;

// Row count
engine.row_count("users")?;  // -> usize

// Drop table
engine.drop_table("users")?;
```

### CRUD Operations

```rust
use relational_engine::{Condition, Value};
use std::collections::HashMap;

// INSERT
let mut values = HashMap::new();
values.insert("name".to_string(), Value::String("Alice".into()));
values.insert("age".to_string(), Value::Int(30));
let row_id = engine.insert("users", values)?;

// BATCH INSERT (59x faster)
let rows: Vec<HashMap<String, Value>> = vec![/* ... */];
let row_ids = engine.batch_insert("users", rows)?;

// SELECT with condition
let rows = engine.select("users",
    Condition::Ge("age".into(), Value::Int(18)))?;

// UPDATE
let mut updates = HashMap::new();
updates.insert("age".to_string(), Value::Int(31));
let count = engine.update("users",
    Condition::Eq("name".into(), Value::String("Alice".into())),
    updates)?;

// DELETE
let count = engine.delete_rows("users",
    Condition::Lt("age".into(), Value::Int(18)))?;
```

### Conditions

```rust
use relational_engine::{Condition, Value};

// Simple conditions
Condition::True                           // Match all
Condition::Eq("col".into(), Value::Int(1))  // col = 1
Condition::Ne("col".into(), Value::Int(1))  // col != 1
Condition::Lt("col".into(), Value::Int(10)) // col < 10
Condition::Le("col".into(), Value::Int(10)) // col <= 10
Condition::Gt("col".into(), Value::Int(0))  // col > 0
Condition::Ge("col".into(), Value::Int(0))  // col >= 0

// Compound conditions
let cond = Condition::Ge("age".into(), Value::Int(18))
    .and(Condition::Lt("age".into(), Value::Int(65)));

let cond = Condition::Eq("status".into(), Value::String("active".into()))
    .or(Condition::Gt("priority".into(), Value::Int(5)));
```

### Indexes

```rust
// Hash index (O(1) equality)
engine.create_index("users", "email")?;
engine.has_index("users", "email");  // -> bool
engine.drop_index("users", "email")?;

// B-tree index (O(log n) range)
engine.create_btree_index("users", "age")?;
engine.has_btree_index("users", "age");  // -> bool
engine.drop_btree_index("users", "age")?;

// List indexed columns
engine.get_indexed_columns("users");        // -> Vec<String>
engine.get_btree_indexed_columns("users");  // -> Vec<String>
```

### Joins

```rust
// INNER JOIN
let joined = engine.join("users", "posts", "_id", "user_id")?;
// -> Vec<(Row, Row)>

// LEFT JOIN
let joined = engine.left_join("users", "posts", "_id", "user_id")?;
// -> Vec<(Row, Option<Row>)>

// RIGHT JOIN
let joined = engine.right_join("users", "posts", "_id", "user_id")?;
// -> Vec<(Option<Row>, Row)>

// FULL JOIN
let joined = engine.full_join("users", "posts", "_id", "user_id")?;
// -> Vec<(Option<Row>, Option<Row>)>

// CROSS JOIN
let joined = engine.cross_join("users", "posts")?;
// -> Vec<(Row, Row)>

// NATURAL JOIN
let joined = engine.natural_join("users", "profiles")?;
// -> Vec<(Row, Row)>
```

### Aggregates

```rust
// COUNT
let count = engine.count("users", Condition::True)?;
let count = engine.count_column("users", "email", Condition::True)?;

// SUM
let total = engine.sum("orders", "amount", Condition::True)?;

// AVG
let avg = engine.avg("orders", "amount", Condition::True)?;  // Option<f64>

// MIN/MAX
let min = engine.min("products", "price", Condition::True)?;  // Option<Value>
let max = engine.max("products", "price", Condition::True)?;
```

### Transactions

```rust
use relational_engine::{TransactionManager, TxPhase};

let tx_manager = TransactionManager::new();

// Begin transaction
let tx_id = tx_manager.begin();

// Check state
tx_manager.is_active(tx_id);  // -> bool
tx_manager.get(tx_id);        // -> Option<TxPhase>

// Acquire row locks
tx_manager.lock_manager().try_lock(tx_id, &[
    ("users".to_string(), 1),
    ("users".to_string(), 2),
])?;

// Commit or rollback
tx_manager.set_phase(tx_id, TxPhase::Committed);
tx_manager.release_locks(tx_id);
tx_manager.remove(tx_id);
```

---

## graph_engine

Directed graph operations with BFS traversal and shortest path.

### Core Types

| Type | Description |
| --- | --- |
| `GraphEngine` | Main engine with TensorStore backend |
| `Node` | Graph node with label and properties |
| `Edge` | Directed edge with type and properties |
| `Direction` | Outgoing, Incoming, Both |
| `PropertyValue` | Null, Int, Float, String, Bool |
| `Path` | Sequence of nodes and edges |

### Node Operations

```rust
use graph_engine::{GraphEngine, PropertyValue};
use std::collections::HashMap;

let engine = GraphEngine::new();

// Create node
let mut props = HashMap::new();
props.insert("name".to_string(), PropertyValue::String("Alice".into()));
let node_id = engine.create_node("person", props)?;

// Get node
let node = engine.get_node(node_id)?;
println!("{}: {:?}", node.label, node.properties);

// Update node
let mut updates = HashMap::new();
updates.insert("age".to_string(), PropertyValue::Int(30));
engine.update_node(node_id, updates)?;

// Delete node
engine.delete_node(node_id)?;

// Find nodes by label
let people = engine.find_nodes_by_label("person")?;
```

### Edge Operations

```rust
use graph_engine::Direction;

// Create edge
let edge_id = engine.create_edge(from_id, to_id, "follows", HashMap::new())?;

// Get edge
let edge = engine.get_edge(edge_id)?;

// Get neighbors
let neighbors = engine.neighbors(node_id, Direction::Outgoing)?;
let neighbors = engine.neighbors(node_id, Direction::Incoming)?;
let neighbors = engine.neighbors(node_id, Direction::Both)?;

// Get edges
let edges = engine.edges(node_id, Direction::Outgoing)?;

// Delete edge
engine.delete_edge(edge_id)?;
```

### Traversal

```rust
// BFS traversal
let visited = engine.bfs(start_id, |node| {
    // Return true to continue traversal
    true
})?;

// Shortest path (Dijkstra)
let path = engine.shortest_path(from_id, to_id)?;
if let Some(path) = path {
    for node_id in path.nodes {
        println!("-> {}", node_id);
    }
}
```

### Property Indexes

```rust
use graph_engine::{IndexTarget, RangeOp};

// Create index on node property
engine.create_property_index(IndexTarget::Node, "age")?;

// Create index on edge property
engine.create_property_index(IndexTarget::Edge, "weight")?;

// Range query using index
let results = engine.find_by_range(
    IndexTarget::Node,
    "age",
    &PropertyValue::Int(18),
    RangeOp::Ge,
)?;
```

---

## vector_engine

Embedding storage and similarity search.

### Core Types

| Type | Description |
| --- | --- |
| `VectorEngine` | Main engine for embedding operations |
| `SearchResult` | Key and similarity score |
| `DistanceMetric` | Cosine, Euclidean, DotProduct |

### Operations

```rust
use vector_engine::{VectorEngine, DistanceMetric};

let engine = VectorEngine::new();

// Store embedding (auto-detects sparse)
engine.store_embedding("doc:1", vec![0.1, 0.2, 0.3])?;

// Get embedding
let vector = engine.get_embedding("doc:1")?;

// Check existence
engine.exists("doc:1");  // -> bool

// Delete
engine.delete_embedding("doc:1")?;

// Count embeddings
engine.count();  // -> usize
```

### Similarity Search

```rust
// Search similar embeddings
let query = vec![0.1, 0.2, 0.3];
let results = engine.search_similar(&query, 10)?;

for result in results {
    println!("{}: {:.4}", result.key, result.score);
}

// Search with metric
let results = engine.search_similar_with_metric(
    &query,
    10,
    DistanceMetric::Euclidean,
)?;
```

---

## tensor_chain

Distributed consensus with Raft and 2PC transactions.

### Core Types

| Type | Description |
| --- | --- |
| `Chain` | Block chain with graph-based linking |
| `Block` | Block with header and transactions |
| `Transaction` | Put, Delete, Update operations |
| `RaftNode` | Raft consensus state machine |
| `DistributedTxCoordinator` | 2PC transaction coordinator |

### Chain Operations

```rust
use tensor_chain::{Chain, Transaction, Block};
use graph_engine::GraphEngine;
use std::sync::Arc;

let graph = Arc::new(GraphEngine::new());
let chain = Chain::new(graph, "node1".to_string());
chain.initialize()?;

// Create block
let builder = chain.new_block()
    .add_transaction(Transaction::Put {
        key: "user:1".into(),
        data: vec![1, 2, 3],
    })
    .add_transaction(Transaction::Delete {
        key: "user:0".into(),
    });

let block = builder.build();
chain.append(block)?;

// Query chain
let height = chain.height();
let block = chain.get_block(1)?;
```

### Raft Consensus

```rust
use tensor_chain::{RaftNode, RaftConfig, RaftState};

let config = RaftConfig {
    election_timeout_min: 150,
    election_timeout_max: 300,
    heartbeat_interval: 50,
    ..Default::default()
};

let raft = RaftNode::new("node1".into(), config);

// State queries
raft.is_leader();     // -> bool
raft.current_term();  // -> u64
raft.state();         // -> RaftState

// Statistics
let stats = raft.stats();
```

### Distributed Transactions

```rust
use tensor_chain::{DistributedTxCoordinator, DistributedTxConfig};

let config = DistributedTxConfig {
    prepare_timeout_ms: 5000,
    commit_timeout_ms: 5000,
    max_retries: 3,
    ..Default::default()
};

let coordinator = DistributedTxCoordinator::new(config);

// Begin distributed transaction
let tx_id = coordinator.begin()?;

// Prepare phase
coordinator.prepare(tx_id, keys, participants).await?;

// Commit phase
coordinator.commit(tx_id).await?;

// Or abort
coordinator.abort(tx_id).await?;
```

### Membership Management

```rust
use tensor_chain::{MembershipManager, ClusterConfig, HealthConfig};

let config = ClusterConfig {
    local: LocalNodeConfig { id: "node1".into(), addr: "127.0.0.1:9000".parse()? },
    peers: vec![],
    health: HealthConfig::default(),
};

let membership = MembershipManager::new(config);

// Add/remove nodes
membership.add_node("node2", "127.0.0.1:9001".parse()?)?;
membership.remove_node("node2")?;

// Health status
let health = membership.node_health("node2");
let status = membership.partition_status();
```

---

## neumann_parser

Hand-written recursive descent parser for the Neumann query language.

### Core Types

| Type | Description |
| --- | --- |
| `Statement` | Parsed statement with span |
| `StatementKind` | Select, Insert, Update, Delete, Node, Edge, etc. |
| `Expr` | Expression AST node |
| `Token` | Lexer token with span |
| `ParseError` | Error with source location |

### Parsing

```rust
use neumann_parser::{parse, parse_all, parse_expr, tokenize};

// Parse single statement
let stmt = parse("SELECT * FROM users WHERE id = 1")?;

// Parse multiple statements
let stmts = parse_all("SELECT 1; SELECT 2")?;

// Parse expression only
let expr = parse_expr("1 + 2 * 3")?;

// Tokenize
let tokens = tokenize("SELECT id, name FROM users");
```

### Error Handling

```rust
let result = parse("SELCT * FROM users");
if let Err(err) = result {
    // Format with source context
    let formatted = err.format_with_source("SELCT * FROM users");
    eprintln!("{}", formatted);

    // Access error details
    println!("Line: {}", err.line());
    println!("Column: {}", err.column());
}
```

### Span Utilities

```rust
use neumann_parser::{line_number, line_col, get_line, BytePos};

let source = "SELECT\nFROM\nWHERE";

// Get line number (1-indexed)
let line = line_number(source, BytePos(7));  // -> 2

// Get line and column
let (line, col) = line_col(source, BytePos(7));  // -> (2, 1)

// Get line text
let text = get_line(source, BytePos(7));  // -> "FROM"
```

---

## query_router

Unified query routing across all engines.

### Core Types

| Type | Description |
| --- | --- |
| `QueryRouter` | Main router handling all query types |
| `QueryResult` | Result variants for different query types |
| `RouterError` | Error types from all engines |

### Query Execution

```rust
use query_router::{QueryRouter, QueryResult};

let router = QueryRouter::new();

// Execute query
let result = router.execute("SELECT * FROM users")?;

match result {
    QueryResult::Rows(rows) => { /* relational result */ }
    QueryResult::Nodes(nodes) => { /* graph result */ }
    QueryResult::Similar(results) => { /* vector result */ }
    QueryResult::Success(msg) => { /* command result */ }
    _ => {}
}
```

### Distributed Queries

```rust
use query_router::{QueryPlanner, MergeStrategy, ResultMerger};

let planner = QueryPlanner::new(partitioner);

// Plan distributed query
let plan = planner.plan("SELECT * FROM users WHERE region = 'us'")?;

// Execute on shards
let shard_results = execute_on_shards(&plan).await?;

// Merge results
let merger = ResultMerger::new(MergeStrategy::Union);
let final_result = merger.merge(shard_results)?;
```

---

## tensor_cache

LLM response cache with exact and semantic matching.

### Core Types

| Type | Description |
| --- | --- |
| `Cache` | Multi-layer LLM response cache |
| `CacheConfig` | Configuration for cache behavior |
| `CacheHit` | Successful lookup result |
| `CacheLayer` | Exact, Semantic, Embedding |
| `EvictionStrategy` | LRU, LFU, CostBased, Hybrid |

### Operations

```rust
use tensor_cache::{Cache, CacheConfig, EvictionStrategy};

let mut config = CacheConfig::default();
config.embedding_dim = 384;
config.eviction_strategy = EvictionStrategy::Hybrid;
let cache = Cache::with_config(config)?;

// Store response
let embedding = vec![0.1, 0.2, /* ... */];
cache.put(
    "What is 2+2?",
    &embedding,
    "The answer is 4.",
    "gpt-4",
    None,  // version
)?;

// Lookup (tries exact, then semantic)
if let Some(hit) = cache.get("What is 2+2?", Some(&embedding)) {
    println!("Response: {}", hit.response);
    println!("Layer: {:?}", hit.layer);
    println!("Cost saved: ${:.4}", hit.cost_saved);
}

// Statistics
let stats = cache.stats();
println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
```

---

## tensor_vault

Encrypted secret storage with graph-based access control.

### Core Types

| Type | Description |
| --- | --- |
| `Vault` | Main vault API |
| `VaultConfig` | Configuration for security settings |
| `Permission` | Read, Write, Admin |
| `MasterKey` | Derived encryption key |

### Operations

```rust
use tensor_vault::{Vault, VaultConfig, Permission};

let config = VaultConfig::default();
let vault = Vault::new(config)?;

// Store secret
vault.set("requester", "db/password", b"secret123", Permission::Admin)?;

// Get secret
let secret = vault.get("requester", "db/password")?;

// Grant access
vault.grant("admin", "user", "db/password", Permission::Read)?;

// Revoke access
vault.revoke("admin", "user", "db/password")?;

// List secrets
let secrets = vault.list("requester", "db/")?;

// Delete secret
vault.delete("admin", "db/password")?;
```

---

## tensor_blob

S3-style object storage with content-addressable chunks.

### Core Types

| Type | Description |
| --- | --- |
| `BlobStore` | Main blob storage API |
| `BlobConfig` | Configuration for chunk size, GC |
| `PutOptions` | Options for storing artifacts |
| `ArtifactMetadata` | Metadata for stored artifacts |
| `BlobWriter` | Streaming upload |
| `BlobReader` | Streaming download |

### Operations

```rust
use tensor_blob::{BlobStore, BlobConfig, PutOptions};

let config = BlobConfig::default();
let store = BlobStore::new(tensor_store, config).await?;

// Store artifact
let artifact_id = store.put(
    "report.pdf",
    &file_bytes,
    PutOptions::new()
        .with_created_by("user:alice")
        .with_tag("quarterly"),
).await?;

// Get artifact
let data = store.get(&artifact_id).await?;

// Streaming upload
let mut writer = store.writer("large-file.bin", PutOptions::new()).await?;
writer.write(&chunk1).await?;
writer.write(&chunk2).await?;
let artifact_id = writer.finish().await?;

// Streaming download
let mut reader = store.reader(&artifact_id).await?;
let chunk = reader.read(1024).await?;

// Delete
store.delete(&artifact_id).await?;

// Metadata
let metadata = store.metadata(&artifact_id).await?;
```

---

## tensor_checkpoint

Snapshot and rollback system.

### Core Types

| Type | Description |
| --- | --- |
| `CheckpointManager` | Main checkpoint API |
| `CheckpointConfig` | Configuration for checkpoints |
| `DestructiveOp` | Delete, Update operations |
| `OperationPreview` | Preview of affected data |
| `ConfirmationHandler` | Custom confirmation logic |

### Operations

```rust
use tensor_checkpoint::{CheckpointManager, CheckpointConfig, AutoConfirm};
use std::sync::Arc;

let config = CheckpointConfig::new()
    .with_max_checkpoints(10)
    .with_auto_checkpoint(true);

let manager = CheckpointManager::new(blob_store, config).await;
manager.set_confirmation_handler(Arc::new(AutoConfirm));

// Create checkpoint
let checkpoint_id = manager.create(Some("before-migration"), &store).await?;

// List checkpoints
let checkpoints = manager.list().await?;

// Restore from checkpoint
manager.restore(&checkpoint_id, &mut store).await?;

// Delete checkpoint
manager.delete(&checkpoint_id).await?;
```

---

## Common Patterns

### Error Handling

All crates use the `Result` type with crate-specific error enums:

```rust
use relational_engine::{RelationalEngine, RelationalError};

let result = engine.create_table("users", schema);
match result {
    Ok(()) => println!("Table created"),
    Err(RelationalError::TableAlreadyExists) => println!("Already exists"),
    Err(e) => eprintln!("Error: {}", e),
}
```

### Thread Safety

All engines use `parking_lot` and `DashMap` for concurrent access:

```rust
use std::sync::Arc;
use std::thread;

let engine = Arc::new(RelationalEngine::new());

let handles: Vec<_> = (0..4).map(|i| {
    let engine = Arc::clone(&engine);
    thread::spawn(move || {
        engine.insert("users", values).unwrap();
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

### Async Operations

`tensor_blob`, `tensor_cache`, and `tensor_checkpoint` use async APIs:

```rust
use tokio::runtime::Runtime;

let rt = Runtime::new()?;
rt.block_on(async {
    let store = BlobStore::new(tensor_store, config).await?;
    store.put("file.txt", &data, options).await?;
    Ok(())
})?;
```

---

## Building Locally

Generate documentation from source:

```bash
# Basic documentation
cargo doc --workspace --no-deps --open

# With all features and private items
cargo doc --workspace --no-deps --all-features --document-private-items

# With scraped examples (nightly)
RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc \
  -Zunstable-options \
  -Zrustdoc-scrape-examples \
  --all-features
```

## Online Documentation

When deployed, the API reference is available at:

- [tensor_store](api/tensor_store/index.html)
- [relational_engine](api/relational_engine/index.html)
- [graph_engine](api/graph_engine/index.html)
- [vector_engine](api/vector_engine/index.html)
- [tensor_chain](api/tensor_chain/index.html)
- [neumann_parser](api/neumann_parser/index.html)
- [query_router](api/query_router/index.html)
- [tensor_cache](api/tensor_cache/index.html)
- [tensor_vault](api/tensor_vault/index.html)
- [tensor_blob](api/tensor_blob/index.html)
- [tensor_checkpoint](api/tensor_checkpoint/index.html)
