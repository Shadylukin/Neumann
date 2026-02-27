# Query Router API Reference

## See Also

- **Explanation**: [Query Routing](../../explanation/query-routing.md)

---

## Core Types

| Type | Description |
| --- | --- |
| `QueryRouter` | Main router orchestrating queries across all engines |
| `QueryResult` | Unified result enum for all query types |
| `RouterError` | Error types for query routing failures |
| `NodeResult` | Graph node result with id, label, properties |
| `EdgeResult` | Graph edge result with id, from, to, label |
| `SimilarResult` | Vector similarity result with key and score |
| `UnifiedResult` | Cross-engine query result with description and items |
| `ChainResult` | Blockchain operation results |
| `QueryPlanner` | Plans distributed query execution across shards |
| `ResultMerger` | Merges results from multiple shards |
| `ShardResult` | Result from a single shard with timing and error info |
| `DistributedQueryConfig` | Configuration for distributed execution |
| `DistributedQueryStats` | Statistics tracking for distributed queries |
| `FilterCondition` | Re-exported from vector_engine for programmatic filter building |
| `FilterValue` | Re-exported from vector_engine for filter values |
| `FilterStrategy` | Re-exported from vector_engine for search strategy |
| `FilteredSearchConfig` | Re-exported from vector_engine for filtered search config |

### QueryRouter

```rust
pub struct QueryRouter {
    // Core engines (always initialized)
    relational: Arc<RelationalEngine>,
    graph: Arc<GraphEngine>,
    vector: Arc<VectorEngine>,

    // Unified engine for cross-engine queries (lazily initialized)
    unified: Option<UnifiedEngine>,

    // Optional services (require explicit initialization)
    vault: Option<Arc<Vault>>,
    cache: Option<Arc<Cache>>,
    blob: Option<Arc<tokio::sync::Mutex<BlobStore>>>,
    blob_runtime: Option<Arc<Runtime>>,
    checkpoint: Option<Arc<tokio::sync::Mutex<CheckpointManager>>>,
    chain: Option<Arc<TensorChain>>,

    // Cluster mode
    cluster: Option<Arc<ClusterOrchestrator>>,
    cluster_runtime: Option<Arc<Runtime>>,
    distributed_planner: Option<Arc<QueryPlanner>>,
    distributed_config: DistributedQueryConfig,
    local_shard_id: ShardId,

    // Authentication state
    current_identity: Option<String>,

    // Vector index for fast similarity search
    hnsw_index: Option<(HNSWIndex, Vec<String>)>,
}
```

### QueryResult Variants

| Variant | Description | Typical Source |
| --- | --- | --- |
| `Empty` | No result (CREATE, INSERT) | DDL, writes |
| `Value(String)` | Single value result | Scalar queries, DESCRIBE |
| `Count(usize)` | Count of affected rows/nodes/edges | UPDATE, DELETE |
| `Ids(Vec<u64>)` | List of IDs | INSERT |
| `Rows(Vec<Row>)` | Relational query results | SELECT |
| `Nodes(Vec<NodeResult>)` | Graph node results | NODE queries |
| `Edges(Vec<EdgeResult>)` | Graph edge results | EDGE queries |
| `Path(Vec<u64>)` | Graph traversal path | PATH queries |
| `Similar(Vec<SimilarResult>)` | Vector similarity results | SIMILAR queries |
| `Unified(UnifiedResult)` | Cross-engine query results | FIND queries |
| `TableList(Vec<String>)` | List of table names | SHOW TABLES |
| `Blob(Vec<u8>)` | Blob data bytes | BLOB GET |
| `ArtifactInfo(ArtifactInfoResult)` | Blob artifact metadata | BLOB INFO |
| `ArtifactList(Vec<String>)` | List of artifact IDs | BLOBS LIST |
| `BlobStats(BlobStatsResult)` | Blob storage statistics | BLOB STATS |
| `CheckpointList(Vec<CheckpointInfo>)` | List of checkpoints | CHECKPOINTS |
| `Chain(ChainResult)` | Chain operation result | CHAIN queries |

### RouterError Types

| Error | Cause | Recovery |
| --- | --- | --- |
| `ParseError` | Invalid query syntax | Fix query syntax |
| `UnknownCommand` | Unknown command or keyword | Check command spelling |
| `RelationalError` | Error from relational engine | Check table/column names |
| `GraphError` | Error from graph engine | Verify node/edge IDs |
| `VectorError` | Error from vector engine | Check embedding dimensions |
| `VaultError` | Error from vault | Verify permissions |
| `CacheError` | Error from cache | Check cache configuration |
| `BlobError` | Error from blob storage | Verify artifact exists |
| `CheckpointError` | Error from checkpoint system | Check blob store initialized |
| `ChainError` | Error from chain system | Verify chain initialized |
| `InvalidArgument` | Invalid argument value | Check argument types |
| `MissingArgument` | Missing required argument | Provide required args |
| `TypeMismatch` | Type mismatch in query | Check value types |
| `AuthenticationRequired` | Vault operations require identity | Call `SET IDENTITY` first |

### Error Propagation

The router implements `From` traits to convert engine-specific errors:

```rust
impl From<RelationalError> for RouterError { ... }
impl From<GraphError> for RouterError { ... }
impl From<VectorError> for RouterError { ... }
impl From<VaultError> for RouterError { ... }
impl From<CacheError> for RouterError { ... }
impl From<BlobError> for RouterError { ... }
impl From<CheckpointError> for RouterError { ... }
impl From<ChainError> for RouterError { ... }
impl From<UnifiedError> for RouterError { ... }
```

This allows using the `?` operator throughout execution methods.

## Constructors

| Constructor | UnifiedEngine | Use Case |
| --- | --- | --- |
| `new()` | No | Simple single-engine queries |
| `with_engines(...)` | No | Custom engine configuration |
| `with_shared_store(...)` | Yes | Cross-engine unified queries |

```rust
use query_router::QueryRouter;
use tensor_store::TensorStore;

let router = QueryRouter::new();
let router = QueryRouter::with_engines(relational, graph, vector);

let store = TensorStore::new();
let router = QueryRouter::with_shared_store(store);
```

## Execution Methods

| Method | Parser | Async | Distributed | Cache |
| --- | --- | --- | --- | --- |
| `execute(command)` | AST | No | Yes | Yes |
| `execute_parsed(command)` | AST | No | Yes | Yes |
| `execute_parsed_async(command)` | AST | Yes | No | Yes |
| `execute_statement(stmt)` | Pre-parsed | No | No | No |
| `execute_statement_async(stmt)` | Pre-parsed | Yes | No | No |

`execute()` and `execute_parsed()` are functionally equivalent -- both parse via
`neumann_parser::parse()`, check for distributed execution, apply caching, and
dispatch to `execute_statement()`. There is no legacy regex-based parsing path.

## Statement Routing Table

| Statement Type | Engine | Handler Method | Operations |
| --- | --- | --- | --- |
| `Select` | Relational | `exec_select` | Table queries with WHERE, JOIN, GROUP BY, ORDER BY |
| `Insert` | Relational | `exec_insert` | Single/multi-row insert, INSERT...SELECT |
| `Update` | Relational | `exec_update` | Row updates with conditions |
| `Delete` | Relational | `exec_delete` | Row deletion with protection |
| `CreateTable` | Relational | `exec_create_table` | Table DDL |
| `DropTable` | Relational | inline | Table removal with protection |
| `CreateIndex` | Relational | inline | Index creation |
| `DropIndex` | Relational | inline | Index removal with protection |
| `ShowTables` | Relational | inline | List tables |
| `Describe` | Multiple | `exec_describe` | Schema/node/edge info |
| `Node` | Graph | `exec_node` | CREATE/GET/DELETE/LIST/UPDATE |
| `Edge` | Graph | `exec_edge` | CREATE/GET/DELETE/LIST/UPDATE |
| `Neighbors` | Graph | `exec_neighbors` | Neighbor traversal |
| `Path` | Graph | `exec_path` | Path finding |
| `Embed` | Vector | `exec_embed` | Embedding storage, batch, delete |
| `Similar` | Vector | `exec_similar` | k-NN search |
| `ShowEmbeddings` | Vector | inline | List embedding keys |
| `CountEmbeddings` | Vector | inline | Count embeddings |
| `Find` | Unified | `exec_find` | Cross-engine queries |
| `Entity` | Unified | `exec_entity` | Entity CRUD |
| `Vault` | Vault | `exec_vault` | Secret management |
| `Cache` | Cache | `exec_cache` | LLM response cache |
| `Blob` | BlobStore | `exec_blob` | Artifact operations |
| `Blobs` | BlobStore | `exec_blobs` | Artifact listing |
| `Checkpoint` | Checkpoint | `exec_checkpoint` | Create snapshot |
| `Rollback` | Checkpoint | `exec_rollback` | Restore snapshot |
| `Checkpoints` | Checkpoint | `exec_checkpoints` | List snapshots |
| `Chain` | TensorChain | `exec_chain` | Blockchain operations |
| `Cluster` | Orchestrator | `exec_cluster` | Cluster management |
| `Empty` | --- | inline | No-op |

## Supported Query Types

### Relational Operations

```sql
CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(255))
DROP TABLE users

INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')
INSERT INTO users SELECT * FROM temp_users
UPDATE users SET name = 'Bob' WHERE id = 1
DELETE FROM users WHERE id = 1

SELECT * FROM users WHERE id = 1
SELECT id, name FROM users ORDER BY name ASC LIMIT 10 OFFSET 5
SELECT COUNT(*), AVG(age) FROM users WHERE active = true GROUP BY dept HAVING COUNT(*) > 5

SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id
SELECT * FROM users u LEFT JOIN profiles p ON u.id = p.user_id
SELECT * FROM a CROSS JOIN b
SELECT * FROM a NATURAL JOIN b
```

### Aggregate Functions

| Function | Description | Null Handling |
| --- | --- | --- |
| `COUNT(*)` | Count all rows | Counts nulls |
| `COUNT(col)` | Count non-null values | Excludes nulls |
| `SUM(col)` | Sum numeric values | Skips nulls |
| `AVG(col)` | Average numeric values | Skips nulls, returns NULL if no values |
| `MIN(col)` | Minimum value | Skips nulls |
| `MAX(col)` | Maximum value | Skips nulls |

### Graph Operations

```sql
NODE CREATE person { name: 'Alice', age: 30 }
NODE GET 123
NODE DELETE 123
NODE LIST person LIMIT 100
NODE UPDATE 123 { name: 'Alice Smith' }

EDGE CREATE 1 -> 2 : friend
EDGE GET 456
EDGE DELETE 456
EDGE LIST friend LIMIT 50

NEIGHBORS 1 friend OUTGOING
NEIGHBORS 123
PATH 1 -> 5 VIA friend
```

### Vector Operations

```sql
EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]
EMBED DELETE 'doc1'
EMBED BATCH [('key1', [0.1, 0.2]), ('key2', [0.3, 0.4])]

SIMILAR 'doc1' LIMIT 5
SIMILAR 'doc1' LIMIT 5 EUCLIDEAN
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 COSINE

EMBED STORE 'doc1' [0.1, 0.2] INTO my_collection

SHOW EMBEDDINGS LIMIT 100
COUNT EMBEDDINGS
```

### Distance Metrics

| Metric | Description | Use Case | Formula |
| --- | --- | --- | --- |
| `COSINE` | Cosine similarity (default) | Semantic similarity | `1 - (a.b) / (||a|| * ||b||)` |
| `EUCLIDEAN` | Euclidean distance (L2) | Spatial distance | `sqrt(sum((a_i - b_i)^2))` |
| `DOT_PRODUCT` | Dot product | Magnitude-aware similarity | `sum(a_i * b_i)` |

### Unified Entity Operations

```sql
ENTITY CREATE 'user:1' {name: 'Alice'} EMBEDDING [0.1, 0.2, 0.3]
ENTITY CONNECT 'user:1' -> 'doc:1' : authored
SIMILAR 'query:key' CONNECTED TO 'hub:entity' LIMIT 10
```

### Vault Operations

```sql
VAULT SET 'secret' 'value'
VAULT GET 'secret'
VAULT DELETE 'secret'
VAULT LIST
VAULT GRANT 'user:bob' 'secret'
VAULT REVOKE 'user:bob' 'secret'
VAULT ROTATE 'secret'
```

### Cache Operations

```sql
CACHE INIT
CACHE STATS
CACHE CLEAR
CACHE EVICT 100
CACHE GET 'key'
CACHE PUT 'key' 'value'
CACHE SEMANTIC GET 'query' THRESHOLD 0.9
CACHE SEMANTIC PUT 'query' 'response' [0.1, 0.2, 0.3]
```

### Chain Operations

```sql
CHAIN BEGIN
CHAIN COMMIT
CHAIN ROLLBACK 100
CHAIN HISTORY 'key'
CHAIN HEIGHT
CHAIN TIP
CHAIN BLOCK 42
CHAIN VERIFY
CHAIN SHOW CODEBOOK GLOBAL
CHAIN SHOW CODEBOOK LOCAL 'domain'
CHAIN ANALYZE TRANSITIONS
```

## Cross-Engine Methods

| Method | Description | Complexity |
| --- | --- | --- |
| `build_vector_index()` | Build HNSW index for O(log n) search | O(n log n) |
| `connect_entities(from, to, type)` | Add graph edge between entities | O(1) |
| `find_neighbors_by_similarity(key, query, k)` | Neighbors sorted by vector similarity | O(k * log n) with HNSW |
| `find_similar_connected(query, connected_to, k)` | Similar AND connected entities | O(k * log n) + O(neighbors) |
| `create_unified_entity(key, fields, embedding)` | Create entity with all modalities | O(1) |

## Distributed Query Types

### Query Plans

| Plan | When Used | Example | Shards Contacted |
| --- | --- | --- | --- |
| `Local` | Point lookups on local shard | `GET user:1` (local key) | 1 |
| `Remote` | Point lookups on remote shard | `GET user:2` (remote key) | 1 |
| `ScatterGather` | Full scans, aggregates, similarity | `SELECT *`, `SIMILAR`, `COUNT` | All |

### Merge Strategies

| Strategy | Description | Use Case | Algorithm |
| --- | --- | --- | --- |
| `Union` | Combine all results | SELECT, NODE queries | Concatenate rows/nodes/edges |
| `TopK(k)` | Keep top K by score | SIMILAR queries | Sort by score desc, truncate |
| `Aggregate(func)` | SUM, COUNT, AVG, MAX, MIN | Aggregate queries | Combine partial aggregates |
| `FirstNonEmpty` | First result found | Point lookups | Short-circuit on first result |
| `Concat` | Concatenate in order | Ordered results | Same as Union |

### DistributedQueryConfig

```rust
pub struct DistributedQueryConfig {
    /// Maximum concurrent shard queries (default: 10)
    pub max_concurrent: usize,
    /// Query timeout per shard in milliseconds (default: 5000)
    pub shard_timeout_ms: u64,
    /// Retry count for failed shards (default: 2)
    pub retry_count: usize,
    /// Whether to fail fast on first shard error (default: false)
    pub fail_fast: bool,
}
```

## Query Caching

Cacheable statements are automatically cached when a cache is configured:

- **Cacheable**: `SELECT`, `SIMILAR`, `NEIGHBORS`, `PATH`
- **Write operations** invalidate cache: `INSERT`, `UPDATE`, `DELETE`, DDL

### Cache Key Generation

```rust
fn cache_key_for_query(command: &str) -> String {
    format!("query:{}", command.trim().to_lowercase())
}
```

### Cache Gotchas

1. **Full cache invalidation**: Any write operation clears the entire cache. No
   table-level tracking.
2. **Case sensitivity**: Cache keys are lowercased, so `SELECT` and `select` hit
   the same entry.
3. **Whitespace normalization**: Queries are trimmed but not fully normalized.
4. **No TTL**: Cached entries persist until invalidated by writes or explicit
   `CACHE CLEAR`.

## Protected Operations

Operations that receive automatic checkpoint protection:

- `DELETE` (relational rows)
- `DROP TABLE`
- `DROP INDEX`
- `NODE DELETE`
- `EMBED DELETE`
- `VAULT DELETE`
- `BLOB DELETE`
- `CACHE CLEAR`

## Performance

| Operation | Complexity | Notes |
| --- | --- | --- |
| Parse | O(n) | n = query length |
| SELECT | O(m) | m = rows in table |
| SELECT with index | O(log m + k) | k = matching rows |
| INSERT | O(1) | Single row insert |
| NODE | O(1) | Single node create |
| EDGE | O(1) | Single edge create |
| PATH | O(V+E) | BFS traversal |
| SIMILAR (brute-force) | O(n*d) | n = embeddings, d = dimensions |
| SIMILAR (HNSW) | O(log n * d) | After `build_vector_index()` |
| `find_similar_connected` | O(log n) or O(n) | Uses HNSW if index built |
| Distributed query | O(query) / shards | Parallelized across shards |
| Result merge (Union) | O(total results) | Linear in combined size |
| Result merge (TopK) | O(n log k) | Sort + truncate |

### HNSW Index Performance

| Entities | Brute-force | With HNSW | Speedup |
| --- | --- | --- | --- |
| 200 | 4.17s | 9.3us | 448,000x |

### Distributed Query Overhead

| Operation | Overhead |
| --- | --- |
| Query planning | ~1-5 us |
| Network round-trip | ~1-10 ms (depends on network) |
| Result serialization | ~10-100 us (depends on result size) |
| Result merging | ~1-10 us (TopK), O(n) for Union |

## Related Modules

| Module | Relationship |
| --- | --- |
| [Tensor Store](./tensor-store.md) | Underlying storage layer |
| [Relational Engine](./relational-engine.md) | Table operations |
| [Graph Engine](./graph-engine.md) | Node/edge operations |
| [Vector Engine](./vector-engine.md) | Embedding operations |
| [Tensor Unified](./tensor-unified.md) | Cross-engine queries |
| [Neumann Parser](./neumann-parser.md) | Query parsing |
| [Tensor Vault](./tensor-vault.md) | Secret storage |
| [Tensor Cache](./tensor-cache.md) | LLM response caching |
| [Tensor Blob](./tensor-blob.md) | Artifact storage |
| [Tensor Checkpoint](./tensor-checkpoint.md) | Snapshots |
| [Tensor Chain](./tensor-chain.md) | Blockchain |
| [Neumann Shell](./neumann-shell.md) | CLI interface |
