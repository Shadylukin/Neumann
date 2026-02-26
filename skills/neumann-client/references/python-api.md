# Python SDK Reference

## Installation

```bash
pip install neumann
```

## NeumannClient (Sync)

```python
from neumann import NeumannClient, ClientConfig

# Remote
with NeumannClient.connect("localhost:9200", api_key="...") as client:
    result = client.query("SELECT users")

# Embedded (requires native module)
client = NeumannClient.embedded(path="/tmp/mydb")  # or None for in-memory
```

### Methods

| Method | Returns |
|--------|---------|
| `query(sql)` | `QueryResult` |
| `execute(sql)` | `QueryResult` |
| `execute_batch(queries)` | `list[QueryResult]` |
| `execute_stream(sql)` | `Iterator[QueryResult]` |
| `close()` | `None` |

## AsyncNeumannClient

```python
from neumann.aio import AsyncNeumannClient

async with AsyncNeumannClient("localhost:9200") as client:
    result = await client.query("SELECT users")
# Or: async with await AsyncNeumannClient.connect(...) as client:
```

## QueryResult

```python
from neumann.types import QueryResultType

result = client.query("SELECT users")
result.type        # QueryResultType enum
result.is_empty    # bool
result.is_error    # bool
```

### QueryResultType Values

`EMPTY`, `VALUE`, `COUNT`, `ROWS`, `NODES`, `EDGES`, `PATHS`, `SIMILAR`,
`IDS`, `TABLE_LIST`, `BLOB`, `BLOB_INFO`, `BLOB_STATS`, `ARTIFACT_LIST`,
`CHECKPOINT_LIST`, `UNIFIED`, `PAGE_RANK`, `CENTRALITY`, `COMMUNITIES`,
`CONSTRAINTS`, `AGGREGATE`, `BATCH_OPERATION`, `GRAPH_INDEXES`,
`PATTERN_MATCH`, `ERROR`, `CHAIN_*` variants.

### Row Access

```python
for row in result.rows:
    row.get_string("name")  # str | None
    row.get_int("age")      # int | None
    row.get_float("score")  # float | None
    row.get_bool("active")  # bool | None
    row.to_dict()           # {"name": "Alice", "age": 30}
```

## VectorClient

```python
from neumann import VectorClient, VectorPoint

vc = VectorClient.connect("localhost:9200")
vc.create_collection("docs", dimension=384, distance="cosine")
vc.upsert_points("docs", [
    VectorPoint(id="d1", vector=[0.1, 0.2, ...], payload={"title": "Doc"})
])
results = vc.query_points("docs", query_vector=[0.1, ...], limit=10)
# results: list[ScoredVectorPoint] with .id, .score, .payload
collections = vc.list_collections()
# list[CollectionInfo] with .name, .points_count, .dimension, .distance
```

## Pandas Integration

```python
from neumann.integrations.pandas import result_to_dataframe, dataframe_to_inserts

df = result_to_dataframe(result)  # QueryResult -> DataFrame
queries = dataframe_to_inserts(df, "users")  # DataFrame -> INSERT queries
```

## NumPy Integration

```python
from neumann.integrations.numpy import vector_to_insert, vectors_to_inserts, cosine_similarity

query = vector_to_insert("doc:1", np_array, normalize=True)
queries = vectors_to_inserts("collection", {"k1": vec1, "k2": vec2})
sim = cosine_similarity(vec_a, vec_b)
```

## Error Types

```python
from neumann.errors import (
    NeumannError,        # Base (code, message)
    ConnectionError,     # code=UNAVAILABLE
    AuthenticationError, # code=UNAUTHENTICATED
    PermissionError,     # code=PERMISSION_DENIED
    NotFoundError,       # code=NOT_FOUND
    InvalidArgumentError,# code=INVALID_ARGUMENT
    ParseError,          # code=PARSE_ERROR
    QueryError,          # code=QUERY_ERROR
    InternalError,       # code=INTERNAL
    ErrorCode,           # Enum: UNKNOWN(0)..QUERY_ERROR(9)
    error_from_code,     # int|ErrorCode + message -> specific error
)
```

## Configuration

```python
from neumann import ClientConfig, TimeoutConfig, RetryConfig

config = ClientConfig.default()        # 30s timeout, 3 retries
config = ClientConfig.fast_fail()      # 5s timeout, no retry
config = ClientConfig.no_retry()       # 30s timeout, no retry
config = ClientConfig.high_latency()   # 120s timeout, 5 retries

# Custom
config = ClientConfig(
    timeout=TimeoutConfig(default_timeout_s=60.0, connect_timeout_s=10.0),
    retry=RetryConfig(max_attempts=5, initial_backoff_ms=200),
)
```
