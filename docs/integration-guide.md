# Neumann Integration Guide

A complete reference for installing and using Neumann in your projects.

Neumann is a unified tensor-based database that stores relational tables,
graph relationships, and vector embeddings in a single runtime. Version 0.4.0,
MIT/Apache-2.0, Rust 1.75+.

**Repository:** <https://github.com/Shadylukin/Neumann>

---

## Table of Contents

- [Installation](#installation)
- [Integration Patterns](#integration-patterns)
- [Crate Reference](#crate-reference)
- [Query Language](#query-language)
- [Client SDKs](#client-sdks)
- [Server Deployment](#server-deployment)
- [Recipes](#recipes)

---

## Installation

### Binary (CLI)

```bash
# Homebrew
brew tap Shadylukin/tap
brew install neumann

# Cargo (installs the `neumann` binary)
cargo install neumann-db

# Install script
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
```

### As a Rust Dependency

Add crates to your project's `Cargo.toml`. All crates follow workspace version
`0.4.0` and are published under MIT OR Apache-2.0.

```toml
[dependencies]
# Client SDK -- the recommended starting point
neumann_client = { version = "0.4", features = ["embedded"] }

# Or use individual engines directly
query_router   = "0.4"   # Unified query execution across all engines
tensor_store   = "0.4"   # Core key-value storage with HNSW indexing
relational_engine = "0.4"   # SQL-like tables with SIMD filtering
graph_engine   = "0.4"   # Directed graphs, traversal, pattern matching
vector_engine  = "0.4"   # k-NN similarity search via HNSW

# Specialized storage
tensor_vault      = "0.4"   # AES-256-GCM encrypted secrets
tensor_cache      = "0.4"   # Multi-layer LLM response cache
tensor_blob       = "0.4"   # Content-addressable blob storage (async)
tensor_checkpoint = "0.4"   # Atomic snapshot/restore
tensor_unified    = "0.4"   # Cross-engine unified entity operations
tensor_chain      = "0.4"   # Tensor-native blockchain with Raft consensus

# Utilities
neumann_parser   = "0.4"   # Query language parser
tensor_compress  = "0.4"   # Tensor Train decomposition, delta/RLE encoding
tensor_spatial   = "0.4"   # R-tree spatial indexing
tensor_learn     = "0.4"   # Geometric intelligence / hyperbolic codebooks
```

### Python SDK

```bash
pip install neumann-db

# With embedded (in-process) support
pip install neumann-db[native]
```

### TypeScript SDK

```bash
npm install neumann-db
```

---

## Integration Patterns

There are three ways to use Neumann from Rust, depending on your needs.

### Pattern 1: Embedded Client (Recommended)

The simplest path. The `neumann_client` crate with the `embedded` feature gives
you an in-process database with no server required.

```rust
use neumann_client::NeumannClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::embedded()?;

    // Create a table
    client.execute_sync("CREATE TABLE users (id INT, name TEXT, age INT)")?;

    // Insert rows
    client.execute_sync("INSERT INTO users VALUES (1, 'Alice', 30)")?;
    client.execute_sync("INSERT INTO users VALUES (2, 'Bob', 25)")?;

    // Query
    let result = client.execute_sync("SELECT * FROM users WHERE age > 20")?;
    println!("{result:?}");

    // Graph operations work too
    client.execute_sync("NODE CREATE person { name: 'Alice' }")?;

    // Vector operations
    client.execute_sync("EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]")?;
    let similar = client.execute_sync("SIMILAR 'doc1' LIMIT 5")?;

    Ok(())
}
```

**Cargo.toml:**

```toml
[dependencies]
neumann_client = { version = "0.4", features = ["embedded"] }
```

### Pattern 2: Remote Client (gRPC)

Connect to a running Neumann server. Requires async runtime.

```rust
use neumann_client::NeumannClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::connect("localhost:9200")
        .api_key("your-api-key")
        .with_tls()
        .timeout_ms(5000)
        .build()
        .await?;

    let result = client.execute("SELECT * FROM users").await?;
    println!("{result:?}");

    // Batch queries
    let results = client.execute_batch(&[
        "INSERT INTO users VALUES (1, 'Alice', 30)",
        "INSERT INTO users VALUES (2, 'Bob', 25)",
    ]).await?;

    // Streaming for large result sets
    let mut stream = client.execute_stream("SELECT * FROM large_table").await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        // Process each QueryChunk
    }

    Ok(())
}
```

**Cargo.toml:**

```toml
[dependencies]
neumann_client = "0.4"   # remote feature is on by default
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Pattern 3: Direct Engine Access

Use the engines directly for maximum control. All engines share a `TensorStore`
so data is accessible across engines without copying.

```rust
use tensor_store::TensorStore;
use relational_engine::RelationalEngine;
use graph_engine::GraphEngine;
use vector_engine::VectorEngine;
use query_router::QueryRouter;

fn main() {
    // Option A: Use QueryRouter for unified access
    let mut router = QueryRouter::new();
    let _ = router.execute("CREATE TABLE users (id INT, name TEXT)");
    let _ = router.execute("NODE CREATE person { name: 'Alice' }");
    let _ = router.execute("EMBED STORE 'doc1' [0.1, 0.2, 0.3]");

    // Option B: Use engines directly for typed APIs
    let store = TensorStore::new();
    let relational = RelationalEngine::with_store(store.clone());
    let graph = GraphEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store);
}
```

---

## Crate Reference

### Foundation Layer

#### `tensor_store` -- Core Storage

The shared storage layer that all engines build on. Provides thread-safe
key-value storage with sharded BTrees, HNSW vector indexing, WAL, and snapshots.

**Key types:**

| Type | Purpose |
|------|---------|
| `TensorStore` | Thread-safe key-value store with `SlabRouter` |
| `TensorData` | HashMap-based entity with field accessors |
| `TensorValue` | `Scalar \| Vector \| Sparse \| Pointer \| Pointers` |
| `ScalarValue` | `Null \| Bool \| Int \| Float \| String \| Bytes` |
| `HNSWIndex` | Hierarchical navigable small world graph index |
| `HNSWConfig` | HNSW tuning (M, ef_construction, ef_search) |
| `SparseVector` | Memory-efficient sparse embedding (15+ metrics) |
| `TensorWal` | Write-ahead log for durability |
| `CacheRing` | Multi-layer eviction cache |
| `TieredStore` | Hot/warm/cold tiered storage |

```rust
use tensor_store::{TensorStore, TensorData, TensorValue, ScalarValue};

let store = TensorStore::new();

// Store a key-value pair
let mut data = TensorData::new();
data.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
data.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
store.put("user:1", data);

// Retrieve
if let Some(data) = store.get("user:1") {
    println!("{:?}", data.get("name"));
}
```

#### `neumann_parser` -- Query Language Parser

Hand-written recursive descent parser. Tokenizes and parses the Neumann query
language into an AST.

**Key types:**

| Type | Purpose |
|------|---------|
| `parse(input)` | Parse a single statement into AST |
| `parse_all(input)` | Parse multiple semicolon-separated statements |
| `tokenize(input)` | Tokenize input into `Token` stream |
| `Statement` | Top-level AST enum (Select, Insert, NodeCreate, etc.) |
| `Expr` | Expression AST (literals, operators, functions) |
| `ParseError` | Error with span and message |

```rust
use neumann_parser::{parse, ast::Statement};

let stmt = parse("SELECT name, age FROM users WHERE age > 25")?;
match stmt {
    Statement::Select { columns, table, condition, .. } => {
        // Process AST
    }
    _ => {}
}
```

#### `tensor_compress` -- Compression

Tensor Train decomposition, delta encoding, and RLE compression.

```rust
use tensor_compress::{rle_encode, rle_decode, CompressionConfig};

// RLE compression
let data = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
let encoded = rle_encode(&data);
let decoded = rle_decode(&encoded);
assert_eq!(data, decoded);
```

---

### Engine Layer

#### `relational_engine` -- SQL Tables

SQL-like relational engine with SIMD-accelerated filtering, B-tree and hash
indexes, columnar scans, transactions, and joins.

**Key types:**

| Type | Purpose |
|------|---------|
| `RelationalEngine` | Main engine API |
| `Schema` / `Column` / `ColumnType` | Table schema definition |
| `Value` / `Row` | Row data |
| `Condition` | WHERE clause filters |
| `AggregateExpr` | SUM, AVG, COUNT, MIN, MAX |
| `Transaction` / `TransactionManager` | ACID transactions |
| `RelationalConfig` | Engine tuning |

**Via QueryRouter:**

```sql
CREATE TABLE products (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT,
    category TEXT,
    in_stock BOOLEAN DEFAULT true
);

INSERT INTO products VALUES (1, 'Widget', 9.99, 'tools', true);
INSERT INTO products VALUES (2, 'Gadget', 24.99, 'electronics', true);

SELECT name, price FROM products WHERE price > 10.0 ORDER BY price DESC;

SELECT category, COUNT(*) as cnt, AVG(price) as avg_price
FROM products GROUP BY category HAVING COUNT(*) > 1;

UPDATE products SET price = 19.99 WHERE name = 'Widget';
DELETE FROM products WHERE in_stock = false;

CREATE INDEX idx_category ON products(category);

-- Joins
SELECT p.name, o.quantity
FROM products p
INNER JOIN orders o ON p.id = o.product_id
WHERE o.quantity > 5;
```

**Direct Rust API:**

```rust
use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
use std::collections::HashMap;

let engine = RelationalEngine::new();

let schema = Schema::new(vec![
    Column::new("id", ColumnType::Int),
    Column::new("name", ColumnType::String),
    Column::new("price", ColumnType::Float),
]);
engine.create_table("products", schema)?;

let mut row = HashMap::new();
row.insert("id".to_string(), Value::Int(1));
row.insert("name".to_string(), Value::String("Widget".into()));
row.insert("price".to_string(), Value::Float(9.99));
engine.insert("products", row)?;

let results = engine.select(
    "products",
    Condition::Gt("price".to_string(), Value::Float(5.0)),
)?;
```

#### `graph_engine` -- Graph Database

Directed property graph with BFS/DFS traversal, shortest path, PageRank,
community detection, pattern matching, full-text search, and geo indexing.

**Key types:**

| Type | Purpose |
|------|---------|
| `GraphEngine` | Main engine API |
| `Node` / `Edge` / `Path` | Graph primitives |
| `Direction` | `Outgoing \| Incoming \| Both` |
| `PropertyValue` | Node/edge property values |
| `Pattern` / `PatternMatch` | Graph pattern matching |
| `PageRankConfig` | PageRank algorithm config |
| `CommunityConfig` | Louvain community detection config |

**Via QueryRouter:**

```sql
-- Nodes
NODE CREATE person { name: 'Alice', role: 'engineer' }
NODE CREATE person { name: 'Bob', role: 'manager' }
NODE CREATE project { name: 'Neumann' }
NODE GET 'node-id'
NODE LIST person LIMIT 20

-- Edges (direction keywords are OUTGOING/INCOMING/BOTH)
EDGE CREATE 'alice-id' -> 'bob-id' : reports_to
EDGE CREATE 'alice-id' -> 'project-id' : works_on { since: '2024-01' }
EDGE LIST reports_to LIMIT 10

-- Traversal
NEIGHBORS 'alice-id' OUTGOING : reports_to
PATH SHORTEST 'alice-id' TO 'bob-id'
PATH SHORTEST 'alice-id' TO 'bob-id' MAX_DEPTH 5
PATH ALL 'alice-id' TO 'bob-id' MAX_DEPTH 3

-- Algorithms
PAGERANK DAMPING 0.85 MAX_ITERATIONS 100
BETWEENNESS SAMPLING_RATIO 0.5
CLOSENESS
LOUVAIN RESOLUTION 1.0
LABEL_PROPAGATION MAX_ITERATIONS 20
```

**Direct Rust API:**

```rust
use graph_engine::{GraphEngine, PropertyValue, Direction};
use std::collections::HashMap;

let engine = GraphEngine::new();

let mut props = HashMap::new();
props.insert("name".into(), PropertyValue::String("Alice".into()));
let alice = engine.create_node("person", props)?;

let mut bob_props = HashMap::new();
bob_props.insert("name".into(), PropertyValue::String("Bob".into()));
let bob = engine.create_node("person", bob_props)?;

engine.create_edge(alice, bob, "REPORTS_TO", HashMap::new(), true)?;

let neighbors = engine.neighbors(alice, Some("REPORTS_TO"), Direction::Outgoing)?;
let path = engine.find_path(alice, bob, None)?;
```

#### `vector_engine` -- Similarity Search

k-NN similarity search with HNSW indexing, multiple distance metrics, filtered
search, batch operations, and metadata.

**Key types:**

| Type | Purpose |
|------|---------|
| `VectorEngine` | Main engine API |
| `VectorEngineConfig` | Dimension, metric, HNSW tuning |
| `DistanceMetric` | `Cosine \| Euclidean \| DotProduct \| Manhattan \| ...` |
| `SearchResult` | Key + similarity score |
| `FilterCondition` / `FilterValue` | Metadata filters for search |
| `EmbeddingInput` | Batch embedding input |

**Via QueryRouter:**

```sql
-- Store embeddings
EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]
EMBED STORE 'doc2' [0.15, 0.25, 0.35, 0.45]
EMBED STORE 'doc3' [0.9, 0.8, 0.7, 0.6]

-- Search by key (find items similar to doc1)
SIMILAR 'doc1' LIMIT 5

-- Search by vector
SIMILAR [0.12, 0.22, 0.32, 0.42] LIMIT 5 METRIC COSINE

-- Available metrics: COSINE, EUCLIDEAN, DOT_PRODUCT

-- Index management
EMBED BUILD INDEX
SHOW EMBEDDINGS
COUNT EMBEDDINGS
SHOW VECTOR INDEX
```

**Direct Rust API:**

```rust
use vector_engine::{VectorEngine, VectorEngineConfig, DistanceMetric, EmbeddingInput};

let config = VectorEngineConfig {
    default_dimension: Some(4),
    default_metric: DistanceMetric::Cosine,
    ..Default::default()
};
let engine = VectorEngine::with_config(config)?;

engine.store_embedding("doc1", vec![0.1, 0.2, 0.3, 0.4])?;

let results = engine.search_similar(&[0.12, 0.22, 0.32, 0.42], 5)?;
for r in &results {
    println!("{} (score: {:.4})", r.key, r.score);
}

// Batch storage
let inputs = vec![
    EmbeddingInput::new("doc1", vec![0.1, 0.2, 0.3, 0.4]),
    EmbeddingInput::new("doc2", vec![0.2, 0.3, 0.4, 0.5]),
];
engine.batch_store_embeddings(inputs)?;
```

---

### Specialized Storage Layer

#### `tensor_vault` -- Encrypted Secrets

AES-256-GCM encrypted secret storage with graph-based access control, audit
logging, key rotation, PKI, rate limiting, and anomaly detection.

**Key types:**

| Type | Purpose |
|------|---------|
| `Vault` / `VaultConfig` | Main vault API |
| `AccessController` / `Permission` | RBAC access control |
| `Cipher` / `MasterKey` | Encryption primitives |
| `PolicyManager` / `PolicyTemplate` | Access policies |
| `RotationGenerator` / `RotationPolicy` | Key rotation |
| `AuditLog` / `AuditEntry` | Audit trail |

**Via QueryRouter:**

```sql
VAULT SET 'db_password' 's3cret'
VAULT GET 'db_password'
VAULT DELETE 'db_password'
VAULT ROTATE 'db_password' 'n3w_s3cret'
VAULT LIST

-- Access control
VAULT GRANT 'service-a' ON 'db_password' READ
VAULT REVOKE 'service-a' ON 'db_password'
```

#### `tensor_cache` -- LLM Response Cache

Multi-layer cache for LLM responses: exact match, semantic similarity, and
embedding-based lookup. Tracks token usage and cost savings.

**Key types:**

| Type | Purpose |
|------|---------|
| `Cache` / `CacheConfig` | Main cache API |
| `CacheHit` | Hit result with layer info |
| `EvictionStrategy` | LRU, LFU, TTL, etc. |
| `TokenCounter` / `ModelPricing` | Cost tracking |

**Via QueryRouter:**

```sql
-- Exact cache
CACHE PUT 'greeting' 'Hello, how can I help you?'
CACHE GET 'greeting'

-- Semantic cache (matches similar queries)
CACHE SEMANTIC PUT 'what is gravity' 'Gravity is...' EMBEDDING [0.9, 0.1, 0.2]
CACHE SEMANTIC GET 'explain gravity' THRESHOLD 0.85

-- Management
CACHE INIT
CACHE STATS
CACHE CLEAR
CACHE EVICT 100
```

#### `tensor_blob` -- Blob Storage

Content-addressable blob storage with chunking, streaming I/O, garbage
collection, and integrity verification. Async-first design.

**Key types:**

| Type | Purpose |
|------|---------|
| `BlobStore` / `BlobConfig` | Main blob API |
| `BlobReader` / `BlobWriter` | Streaming I/O |
| `Chunker` / `compute_hash` | Content-addressable chunking |
| `GarbageCollector` / `GcConfig` | Cleanup |

**Features:** `graph` (graph integration), `vector` (vector integration), `full` (all).

**Via QueryRouter:**

```sql
BLOB PUT 'report.pdf' DATA 'base64...' TYPE 'application/pdf' BY 'alice'
BLOB GET 'artifact-id'
BLOB INFO 'artifact-id'
BLOB DELETE 'artifact-id'

-- Link blobs to entities
BLOB LINK 'artifact-id' TO 'project:neumann'
BLOB TAG 'artifact-id' 'important'

-- Query blobs
BLOBS
BLOBS FOR 'project:neumann'
BLOBS BY TAG 'important'
```

#### `tensor_checkpoint` -- Snapshots

Atomic checkpoint/restore with retention policies and confirmation prompts.

**Key types:**

| Type | Purpose |
|------|---------|
| `CheckpointManager` / `CheckpointConfig` | Main API |
| `ConfirmationHandler` (trait) | Custom confirm/reject |
| `AutoConfirm` / `AutoReject` | Built-in handlers |
| `FileCheckpointStore` | Persistent storage |
| `RetentionManager` | Retention policies |

**Via QueryRouter:**

```sql
CHECKPOINT 'before-migration'
CHECKPOINTS LIMIT 10
ROLLBACK TO 'checkpoint-id'

-- Shell persistence
SAVE 'backup.bin'
LOAD 'backup.bin'
```

#### `tensor_unified` -- Cross-Engine Entities

Unified entity abstraction that spans relational, graph, and vector engines.
Create entities with properties, embeddings, and graph connections in one
operation.

**Key types:**

| Type | Purpose |
|------|---------|
| `UnifiedEngine` / `Unified` (trait) | Main API |
| `UnifiedResult` / `UnifiedItem` | Query results |
| `FindPattern` | Cross-engine query patterns |
| `EntityInput` | `(key, properties, optional_embedding)` |

**Via QueryRouter:**

```sql
-- Create entity with properties and embedding
ENTITY CREATE 'user:alice' { name: 'Alice', role: 'engineer' } EMBEDDING [0.1, 0.2, 0.3]

-- Query
ENTITY GET 'user:alice'
ENTITY UPDATE 'user:alice' { role: 'senior-engineer' }
ENTITY DELETE 'user:alice'

-- Relationships
ENTITY CONNECT 'user:alice' -> 'user:bob' : reports_to

-- Batch
ENTITY BATCH CREATE [
    { key: 'user:1', name: 'Alice' },
    { key: 'user:2', name: 'Bob' }
]

-- Cross-engine search
FIND NODE person WHERE name = 'Alice' LIMIT 5
FIND EDGE reports_to WHERE since > '2024-01' LIMIT 10
FIND ROWS FROM users WHERE age > 25 LIMIT 10
```

---

### Distributed Layer

#### `tensor_chain` -- Blockchain + Consensus

Tensor-native blockchain with Raft consensus, 2PC distributed transactions,
semantic conflict detection, auto-merge, and hybrid logical clocks.

**Key types:**

| Type | Purpose |
|------|---------|
| `TensorChain` / `ChainConfig` | Main chain API |
| `Block` / `BlockHeader` / `Transaction` | Block structure |
| `Raft` / `RaftHandle` | Consensus state machine |
| `HybridLogicalClock` / `HLCTimestamp` | Distributed time |
| `DistributedTxCoordinator` | 2PC with deadlock detection |
| `TcpServer` | Network transport |
| `DeltaReplication` | Efficient delta sync |

**Features:** `tls` (default), `loom` (testing).

**Via QueryRouter:**

```sql
BEGIN CHAIN TRANSACTION
COMMIT CHAIN
CHAIN HEIGHT
CHAIN TIP
CHAIN BLOCK 42
CHAIN HISTORY 'key'
CHAIN VERIFY
CHAIN DRIFT FROM 10 TO 50
```

---

### Query Execution Layer

#### `query_router` -- Unified Query Router

Routes queries to the appropriate engine based on the parsed AST. This is the
main entry point if you want to execute Neumann queries programmatically without
using the client SDK.

**Key types:**

| Type | Purpose |
|------|---------|
| `QueryRouter` | Main API -- `execute(&mut self, query) -> QueryResult` |
| `QueryResult` | Enum: `Rows`, `Nodes`, `Edges`, `Similar`, `Chain`, etc. |
| `RouterError` | Error type |
| `CursorStore` / `CursorState` | Server-side cursor pagination |
| `StatementSafety` | Read-only vs mutating classification |

```rust
use query_router::{QueryRouter, QueryResult};

let mut router = QueryRouter::new();

match router.execute("SELECT * FROM users")? {
    QueryResult::Rows { columns, rows } => {
        for row in rows {
            println!("{row:?}");
        }
    }
    QueryResult::Nodes(nodes) => { /* graph nodes */ }
    QueryResult::Similar(results) => { /* vector results */ }
    _ => {}
}
```

#### `neumann_shell` -- CLI Shell

Interactive REPL with readline editing, syntax highlighting, WAL-based command
history, and snapshot management. This is what `cargo install neumann-db`
installs.

**Key types:** `Shell`, `ShellConfig`, `Wal`.

#### `neumann_server` -- gRPC Server

Production server exposing `QueryRouter` via gRPC and REST. Includes TLS,
API key auth, rate limiting, audit logging, memory budgets, and metrics.

**Key types:**

| Type | Purpose |
|------|---------|
| `NeumannServer` / `ServerConfig` | Main server API |
| `TlsConfig` | TLS certificate configuration |
| `AuthConfig` | API key authentication |
| `RateLimiter` / `RateLimitConfig` | Request rate limiting |
| `AuditLogger` / `AuditConfig` | Audit trail |
| `MetricsHandle` / `MetricsConfig` | Prometheus metrics |
| `MemoryTracker` / `MemoryBudgetConfig` | Memory management |
| `ShutdownManager` | Graceful shutdown |

#### `neumann_client` -- Client SDK

See [Integration Patterns](#integration-patterns) above.

**Features:**

- `remote` (default) -- gRPC client
- `embedded` -- In-process database via `QueryRouter`
- `full` -- Both modes

---

### Utility Crates

#### `tensor_spatial` -- Spatial Indexing

R-tree spatial index for 2D, 3D, and N-dimensional region and nearest-neighbor
queries.

**Key types:**

| Type | Purpose |
|------|---------|
| `SpatialIndex<T>` / `SpatialIndex3D<T>` | 2D/3D R-tree |
| `SpatialIndexN` | N-dimensional R-tree |
| `BoundingBox` / `BoundingBox3D` / `BoundingBoxN` | Bounding regions |
| `SpatialConfig` / `SplitStrategy` | Tree configuration |

```rust
use tensor_spatial::{SpatialIndex, BoundingBox, SpatialConfig};

let mut index = SpatialIndex::new(SpatialConfig::default());
index.insert(BoundingBox::new(0.0, 0.0, 1.0, 1.0), "region-a");

let hits = index.query(&BoundingBox::new(0.5, 0.5, 2.0, 2.0));
```

#### `tensor_learn` -- Geometric Intelligence

Hyperbolic geometry learning with Poincare ball embeddings, codebook training,
and grokking experiments.

**Key types:**

| Type | Purpose |
|------|---------|
| `Codebook` / `TrainingSession` / `TrainingConfig` | Codebook learning |
| `GrokSession` / `GrokConfig` / `GrokStats` | Grokking experiments |
| `PoincarePoint` | Hyperbolic embedding |
| `Matrix8x8` | Small dense matrix operations |

---

## Query Language

All queries go through `neumann_parser::parse()`. The language supports
SQL-style and domain-specific syntax. A few important parser behaviors:

- **Colon in keys:** The parser splits `doc:1` into 3 tokens. Quote keys
  containing colons: `'doc:1'`.
- **Keywords as identifiers:** Use quoted identifiers or rename labels
  that collide with keywords (e.g., `Node` -> `nd`).
- **Direction keywords:** Use `OUTGOING`, `INCOMING`, `BOTH` (not `OUT`/`IN`).

### Quick Reference

| Domain | Commands |
|--------|----------|
| **DDL** | `CREATE TABLE`, `DROP TABLE`, `ALTER TABLE`, `CREATE INDEX`, `DROP INDEX` |
| **DML** | `INSERT INTO`, `SELECT`, `UPDATE`, `DELETE` |
| **Joins** | `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL JOIN`, `CROSS JOIN` |
| **Aggregates** | `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `GROUP BY`, `HAVING` |
| **Nodes** | `NODE CREATE`, `NODE GET`, `NODE LIST`, `NODE DELETE` |
| **Edges** | `EDGE CREATE from -> to : type`, `EDGE LIST`, `EDGE DELETE` |
| **Traversal** | `NEIGHBORS`, `PATH SHORTEST`, `PATH ALL`, `PATH WEIGHTED` |
| **Algorithms** | `PAGERANK`, `BETWEENNESS`, `CLOSENESS`, `EIGENVECTOR`, `LOUVAIN`, `LABEL_PROPAGATION` |
| **Vectors** | `EMBED STORE`, `EMBED GET`, `EMBED DELETE`, `EMBED BUILD INDEX` |
| **Search** | `SIMILAR key\|[vec] LIMIT n METRIC m` |
| **Vault** | `VAULT SET`, `VAULT GET`, `VAULT DELETE`, `VAULT ROTATE`, `VAULT GRANT`, `VAULT REVOKE` |
| **Cache** | `CACHE PUT`, `CACHE GET`, `CACHE SEMANTIC PUT`, `CACHE SEMANTIC GET`, `CACHE STATS` |
| **Blob** | `BLOB PUT`, `BLOB GET`, `BLOB DELETE`, `BLOB INFO`, `BLOB LINK`, `BLOB TAG` |
| **Checkpoint** | `CHECKPOINT`, `CHECKPOINTS`, `ROLLBACK TO` |
| **Chain** | `BEGIN CHAIN TRANSACTION`, `COMMIT CHAIN`, `CHAIN HEIGHT`, `CHAIN VERIFY` |
| **Entity** | `ENTITY CREATE`, `ENTITY GET`, `ENTITY UPDATE`, `ENTITY DELETE`, `ENTITY CONNECT` |
| **Find** | `FIND NODE`, `FIND EDGE`, `FIND ROWS`, `FIND PATH` |

---

## Client SDKs

### Rust

See [Integration Patterns](#integration-patterns). The `neumann_client` crate
supports embedded and remote modes with streaming, batching, and identity-based
access.

**Error handling:**

```rust
use neumann_client::ClientError;

match client.execute("...").await {
    Ok(result) => { /* handle result */ }
    Err(ClientError::Connection(msg)) => { /* retryable */ }
    Err(ClientError::Query(msg)) => { /* query error */ }
    Err(ClientError::Authentication(msg)) => { /* auth failure */ }
    Err(ClientError::InvalidArgument(msg)) => { /* bad input */ }
    Err(ClientError::Internal(msg)) => { /* internal error */ }
    Err(ClientError::Timeout) => { /* retryable */ }
}
```

### Python

```python
from neumann import NeumannClient

# Remote
client = NeumannClient.connect("localhost:9200", api_key="your-key")
result = client.query("SELECT * FROM users WHERE age > 25")

# Embedded (requires neumann-db[native])
client = NeumannClient.embedded(path="/tmp/neumann-data")
client.execute("CREATE TABLE users (name TEXT, age INT)")
result = client.query("SELECT * FROM users")

# Async
from neumann.aio import AsyncNeumannClient
async with await AsyncNeumannClient.connect("localhost:9200") as client:
    result = await client.query("SELECT * FROM users")
```

### TypeScript

```typescript
import { NeumannClient } from 'neumann-db';

// Node.js (gRPC)
const client = await NeumannClient.connect('localhost:9200', {
  apiKey: 'your-api-key',
  tls: false,
});

const result = await client.query('SELECT * FROM users');

// Browser (gRPC-Web)
const client = await NeumannClient.connectWeb('http://localhost:9200');
```

---

## Server Deployment

### Quick Start

```bash
# Install the server binary
cargo install neumann_server

# Start with defaults (localhost:9200)
neumann-server

# With configuration
neumann-server --config /etc/neumann/config.toml
```

### Docker

```bash
# Interactive CLI
docker run -it shadylukinack/neumann:latest

# Server mode with persistent storage
docker run -d -p 9200:9200 -v neumann-data:/var/lib/neumann shadylukinack/neumann:server

# Docker Compose
git clone https://github.com/Shadylukin/Neumann.git && cd Neumann
docker compose up -d neumann-server
```

### Configuration

Configuration is via TOML file or environment variables:

```toml
[node]
id = "node1"
data_dir = "/var/lib/neumann"
bind_address = "0.0.0.0:9200"

[storage]
max_memory_mb = 1024
wal_sync_mode = "fsync"

[cluster]
peers = ["node2:9200", "node3:9200"]

[raft]
election_timeout_min_ms = 150
election_timeout_max_ms = 300
heartbeat_interval_ms = 50

[metrics]
enabled = true
bind_address = "0.0.0.0:9090"
```

**Environment variables:**

```bash
NEUMANN_BIND_ADDR=0.0.0.0:9200
NEUMANN_DATA_DIR=/var/lib/neumann
NEUMANN_CLUSTER_NODE_ID=node1
NEUMANN_CLUSTER_PEERS=node2=10.0.0.2:9200,node3=10.0.0.3:9200
RUST_LOG=info
```

---

## Recipes

### RAG (Retrieval-Augmented Generation)

```rust
use neumann_client::NeumannClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::embedded()?;

    // Store document metadata
    client.execute_sync("CREATE TABLE docs (id INT, title TEXT, source TEXT)")?;
    client.execute_sync("INSERT INTO docs VALUES (1, 'ML Intro', 'textbook')")?;

    // Store document embeddings
    client.execute_sync("EMBED STORE 'doc:1' [0.8, 0.7, 0.1, 0.2]")?;
    client.execute_sync("EMBED STORE 'doc:2' [0.1, 0.9, 0.3, 0.1]")?;
    client.execute_sync("EMBED BUILD INDEX")?;

    // Retrieve relevant docs for a user query
    let similar = client.execute_sync(
        "SIMILAR [0.75, 0.65, 0.15, 0.25] LIMIT 5 METRIC COSINE"
    )?;

    Ok(())
}
```

### Knowledge Graph with Semantic Search

```rust
use neumann_client::NeumannClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::embedded()?;

    // Build knowledge graph
    client.execute_sync("NODE CREATE concept { name: 'Machine Learning' }")?;
    client.execute_sync("NODE CREATE concept { name: 'Neural Networks' }")?;
    // Use returned node IDs to create edges
    // EDGE CREATE 'ml-id' -> 'nn-id' : includes

    // Add embeddings for semantic search
    client.execute_sync("EMBED STORE 'concept:ml' [0.8, 0.2, 0.1]")?;
    client.execute_sync("EMBED STORE 'concept:nn' [0.7, 0.3, 0.2]")?;

    // Find semantically related concepts
    let results = client.execute_sync(
        "SIMILAR [0.75, 0.25, 0.15] LIMIT 10 METRIC COSINE"
    )?;

    Ok(())
}
```

### LLM Response Caching

```rust
use neumann_client::NeumannClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::embedded()?;
    client.execute_sync("CACHE INIT")?;

    // Cache an LLM response with its embedding
    client.execute_sync(
        "CACHE SEMANTIC PUT 'explain quantum computing' \
         'Quantum computing uses qubits...' \
         EMBEDDING [0.9, 0.1, 0.05, 0.3]"
    )?;

    // Later, check cache before calling LLM
    let hit = client.execute_sync(
        "CACHE SEMANTIC GET 'what is quantum computing' THRESHOLD 0.85"
    )?;
    // If hit, use cached response; otherwise call LLM and cache result

    Ok(())
}
```

### Multi-Engine Entity

```rust
use neumann_client::NeumannClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NeumannClient::embedded()?;

    // Create entities that span all engines at once
    client.execute_sync(
        "ENTITY CREATE 'user:alice' { name: 'Alice', role: 'engineer' } \
         EMBEDDING [0.1, 0.2, 0.3, 0.4]"
    )?;
    client.execute_sync(
        "ENTITY CREATE 'user:bob' { name: 'Bob', role: 'manager' } \
         EMBEDDING [0.5, 0.6, 0.7, 0.8]"
    )?;

    // Connect entities in the graph
    client.execute_sync("ENTITY CONNECT 'user:alice' -> 'user:bob' : reports_to")?;

    // Find similar users
    client.execute_sync("SIMILAR 'user:alice' LIMIT 5")?;

    // Query via graph
    client.execute_sync("NEIGHBORS 'user:alice' OUTGOING : reports_to")?;

    Ok(())
}
```

---

## Architecture Summary

```text
                    +------------------+
                    |  neumann_client  |  Embedded or gRPC
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    |   query_router    |         |  neumann_server   |  gRPC + REST
    +---------+---------+         +---------+---------+
              |                             |
              +-----------------------------+
              |
    +---------v---------+
    |  neumann_parser   |  Query -> AST
    +-------------------+
              |
    +---------v-----------------------------------------+
    |                    Engines                         |
    |  +----------------+  +-------------+  +---------+ |
    |  | relational_eng |  | graph_engine|  | vector  | |
    |  +----------------+  +-------------+  | engine  | |
    |                                       +---------+ |
    +---------------------------------------------------+
              |
    +---------v---------+
    |   tensor_store    |  Shared storage layer
    +-------------------+
              |
    +---------v-----------------------------------------+
    |              Specialized Storage                   |
    |  +-------+ +-------+ +------+ +----------+ +---+ |
    |  | vault | | cache | | blob | | checkpoint| |   | |
    |  +-------+ +-------+ +------+ +----------+ |   | |
    |  +--------+ +---------+ +----------+  | chain | |
    |  | unified| | spatial | |  learn   |  +-------+ |
    |  +--------+ +---------+ +----------+            |
    +---------------------------------------------------+
```

All engines share the same `TensorStore`, so data written by one engine is
accessible to others without copying or ETL.
