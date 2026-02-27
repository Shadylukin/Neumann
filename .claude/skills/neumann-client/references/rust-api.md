# Rust SDK Reference

## NeumannClient

```rust
use neumann_client::{NeumannClient, ClientBuilder, ClientConfig, ClientMode, ClientError};

// Remote (async) -- requires "remote" feature (default)
let client = NeumannClient::connect("localhost:9200")
    .api_key("key")
    .with_tls()
    .timeout_ms(30_000)
    .build()        // async
    .await?;

// Remote (blocking)
let client = NeumannClient::connect("host:9200")
    .build_blocking()?;

// Embedded (sync) -- requires "embedded" feature
let client = NeumannClient::embedded()?;

// Embedded with custom router
let client = NeumannClient::with_router(router_arc);
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `remote` | Yes | gRPC client via tonic |
| `embedded` | No | In-process QueryRouter |
| `full` | No | Both remote and embedded |

### Methods

| Method | Mode | Returns |
|--------|------|---------|
| `execute(query).await` | Remote | `Result<RemoteQueryResult>` |
| `execute_sync(query)` | Embedded | `Result<QueryResult>` |
| `execute_batch(queries).await` | Remote | `Result<Vec<RemoteQueryResult>>` |
| `execute_stream(query).await` | Remote | `Result<StreamingQueryResult>` |
| `close().await` | Remote | `Result<()>` |
| `is_connected()` | Both | `bool` |
| `mode()` | Both | `ClientMode` |

### RemoteQueryResult Accessors

Each returns `Option<T>` or empty collection. Check before use.

| Accessor | Returns |
|----------|---------|
| `is_empty()` | `bool` |
| `has_error()` | `bool` |
| `error_message()` | `Option<&str>` |
| `count()` | `Option<u64>` |
| `rows()` | `Option<&[Row]>` |
| `nodes()` | `Option<&[Node]>` |
| `edges()` | `Option<&[Edge]>` |
| `similar()` | `Option<&[SimilarItem]>` |

### Embedded QueryResult (Enum)

```rust
match result {
    QueryResult::Empty => {},
    QueryResult::Value(s) => {},         // String
    QueryResult::Count(n) => {},         // usize
    QueryResult::Ids(ids) => {},         // Vec<u64>
    QueryResult::Rows(rows) => {},       // Vec<Row>
    QueryResult::Nodes(nodes) => {},     // Vec<NodeResult>
    QueryResult::Edges(edges) => {},     // Vec<EdgeResult>
    QueryResult::Path(ids) => {},        // Vec<u64>
    QueryResult::Similar(items) => {},   // Vec<SimilarResult>
    QueryResult::TableList(names) => {}, // Vec<String>
    QueryResult::Unified(u) => {},       // UnifiedResult
    QueryResult::Spatial(items) => {},   // Vec<SpatialResult>
    _ => {} // Chain, Blob, PageRank, Centrality, Communities, etc.
}
```

## Error Types

```rust
use neumann_client::ClientError;

match err {
    ClientError::Connection(msg) => {},      // code 6, retryable
    ClientError::Timeout(msg) => {},         // code 6, retryable
    ClientError::Unavailable(msg) => {},     // code 6, retryable
    ClientError::Authentication(msg) => {},  // code 5
    ClientError::PermissionDenied(msg) => {},// code 3
    ClientError::NotFound(msg) => {},        // code 2
    ClientError::InvalidArgument(msg) => {}, // code 1
    ClientError::Parse(msg) => {},           // code 8
    ClientError::Query(msg) => {},           // code 9
    ClientError::Internal(msg) => {},        // code 7
}

err.code();          // u32
err.is_retryable();  // bool -- true for Connection, Timeout, Unavailable
```

## Streaming

```rust
let mut stream = client.execute_stream("SELECT users").await?;
while let Some(chunk_result) = stream.next().await {
    match chunk_result? {
        QueryChunk::Row(row) => {},
        QueryChunk::Node(node) => {},
        QueryChunk::Edge(edge) => {},
        QueryChunk::SimilarItem(item) => {},
        QueryChunk::BlobData(bytes) => {},
        QueryChunk::CursorInfo(info) => {},
    }
}
```
