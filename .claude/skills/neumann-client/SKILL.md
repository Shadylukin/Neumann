---
name: neumann-client
description: Integrate Neumann database using official client SDKs. Use when writing Rust, Python, or TypeScript code that connects to Neumann, executes queries, or handles results.
---

# Neumann Client SDK

## SDK Overview

Three official SDKs connect to Neumann:

| SDK | Package | Transport | Embedded Mode |
|-----|---------|-----------|---------------|
| **Rust** | `neumann_client` crate | gRPC (tonic) | Yes (in-process `QueryRouter`) |
| **Python** | `neumann` package | gRPC (grpcio) | Yes (PyO3 native bindings) |
| **TypeScript** | `@neumann/client` | gRPC / gRPC-Web | No |

All SDKs share the same gRPC protocol (`neumann.v1.QueryService`) on the default
port **9200**. The Rust SDK additionally supports an embedded mode that runs the
full query engine in-process without a server.

The query language is identical across all SDKs -- you send the same Neumann
query strings regardless of which client you use. Result types map to
language-appropriate discriminated unions (Rust enums, Python dataclass with
`QueryResultType`, TypeScript tagged unions with type guards).

## Connection Patterns

```rust
// Rust -- remote (async)
let client = NeumannClient::connect("localhost:9200")
    .api_key("key").with_tls().timeout_ms(30_000).build().await?;

// Rust -- embedded (sync, no server needed; feature = "embedded")
let client = NeumannClient::embedded()?;
let result = client.execute_sync("SELECT users")?;
```

```python
# Python -- sync
with NeumannClient.connect("localhost:9200", api_key="...") as client:
    result = client.query("SELECT users")

# Python -- async
async with AsyncNeumannClient("localhost:9200") as client:
    result = await client.query("SELECT users")
```

```typescript
// TypeScript -- Node.js (gRPC) or browser (gRPC-Web via connectWeb)
const client = await NeumannClient.connect('localhost:9200', { apiKey: 'key' });
const result = await client.query('SELECT users');
client.close();
```

## Query Execution

All SDKs support single query, batch, and streaming execution.

| Operation | Rust | Python | TypeScript |
|-----------|------|--------|------------|
| Single | `execute(q).await` | `query(q)` | `query(q)` |
| Batch | `execute_batch(qs).await` | `execute_batch(qs)` | `executeBatch(qs)` |
| Stream | `execute_stream(q).await` | `execute_stream(q)` | `executeStream(q)` |
| Paginated | N/A | N/A | `executePaginated(q, opts)` |

## Result Handling

Results are discriminated unions. You MUST check the result type before
accessing data. Accessing the wrong accessor returns `None`/`null`/empty.

### Rust

```rust
use query_router::QueryResult;

match result {
    QueryResult::Rows(rows) => { /* Vec<Row> */ }
    QueryResult::Nodes(nodes) => { /* Vec<NodeResult> */ }
    QueryResult::Edges(edges) => { /* Vec<EdgeResult> */ }
    QueryResult::Similar(items) => { /* Vec<SimilarResult> */ }
    QueryResult::Count(n) => { /* usize */ }
    QueryResult::Value(s) => { /* String */ }
    QueryResult::Path(ids) => { /* Vec<u64> */ }
    QueryResult::TableList(names) => { /* Vec<String> */ }
    QueryResult::Empty => { /* DDL/DML success */ }
    _ => { /* handle other variants */ }
}
```

### Python

```python
from neumann.types import QueryResultType

if result.type == QueryResultType.ROWS:
    for row in result.rows:
        print(row.get_string("name"), row.get_int("age"))
        print(row.to_dict())  # {"name": "Alice", "age": 30}
elif result.type == QueryResultType.NODES:
    for node in result.nodes:
        print(node.label, node.properties)
elif result.type == QueryResultType.SIMILAR:
    for item in result.similar_items:
        print(item.key, item.score)
elif result.type == QueryResultType.COUNT:
    print(result.count)
```

### TypeScript

```typescript
import { isRowsResult, isNodesResult, isSimilarResult, rowToObject } from '@neumann/client';

if (isRowsResult(result)) {
  for (const row of result.rows) {
    console.log(rowToObject(row));
  }
} else if (isNodesResult(result)) {
  for (const node of result.nodes) {
    console.log(node.label, node.properties);
  }
} else if (isSimilarResult(result)) {
  for (const item of result.items) {
    console.log(item.key, item.score);
  }
}
```

## Error Handling

All SDKs use the same error code hierarchy. Retryable errors (Connection,
Timeout, Unavailable) should be retried with backoff. Non-retryable errors
(Parse, Query, NotFound, Authentication) indicate a logic or data problem.

| Error | Code | Retryable | When |
|-------|------|-----------|------|
| Connection/Unavailable | 6 | Yes | Server down or unreachable |
| Timeout | 6 | Yes | Request exceeded deadline |
| Authentication | 5 | No | Bad or missing API key |
| PermissionDenied | 3 | No | Insufficient privileges |
| NotFound | 2 | No | Table, node, or key missing |
| InvalidArgument | 1 | No | Bad parameter values |
| Parse | 8 | No | Query syntax error |
| Query | 9 | No | Execution failure |
| Internal | 7 | No | Server bug |

Check retryability: Rust `err.is_retryable()`, Python `ClientConfig.fast_fail()`
or `ClientConfig.high_latency()`, TypeScript `isRetryable(err)`.

## Vector Client

Each SDK provides a dedicated `VectorClient` for collection and point
operations without writing raw queries.

```python
from neumann import VectorClient

vc = VectorClient.connect("localhost:9200")
vc.create_collection("docs", dimension=384, distance="cosine")
vc.upsert_points("docs", [VectorPoint(id="d1", vector=[...], payload={"title": "Hello"})])
results = vc.query_points("docs", query_vector=[...], limit=10)
```

```typescript
const vc = await VectorClient.connect('localhost:9200');
await vc.createCollection('docs', 384, 'cosine');
await vc.upsertPoints('docs', [{ id: 'd1', vector: [...], payload: { title: 'Hello' } }]);
const results = await vc.queryPoints('docs', queryVector, { limit: 10 });
```

## Configuration

All SDKs support timeout, retry, and keepalive configuration with presets.

| Preset | Timeouts | Retries | Use Case |
|--------|----------|---------|----------|
| `default` | 30s connect, 30s query | 3 attempts | General use |
| `fast_fail` | 2s connect, 5s query | 1 attempt | Health checks, probes |
| `no_retry` | 30s connect, 30s query | 1 attempt | Non-idempotent writes |
| `high_latency` | 30s connect, 120s query | 5 attempts | Cross-region, slow networks |

```python
config = ClientConfig.high_latency()
client = NeumannClient.connect("remote-host:9200", config=config)
```

```typescript
import { highLatencyConfig } from '@neumann/client';
const client = await NeumannClient.connect('remote-host:9200', highLatencyConfig());
```

## Reference

- `references/rust-api.md` -- Rust SDK type reference
- `references/python-api.md` -- Python SDK type reference
- `references/typescript-api.md` -- TypeScript SDK type reference
- `references/grpc-api.md` -- gRPC protocol and grpcurl examples
- `references/rest-api.md` -- REST/HTTP gateway reference
