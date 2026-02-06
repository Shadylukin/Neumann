# Python SDK Quickstart

The Neumann Python SDK provides both synchronous and asynchronous clients for
querying a Neumann server, with optional embedded mode via PyO3 bindings.

## Installation

```bash
pip install neumann
```

## Connect

### Remote (gRPC)

```python
from neumann import NeumannClient

client = NeumannClient.connect("localhost:50051", api_key="your-api-key")
```

### Embedded (no server needed)

```python
client = NeumannClient.embedded(path="/tmp/neumann-data")
```

### Async

```python
from neumann.aio import AsyncNeumannClient

async with await AsyncNeumannClient.connect("localhost:50051") as client:
    result = await client.query("SELECT * FROM users")
```

## Execute Queries

```python
# Single query
result = client.query("SELECT * FROM users WHERE age > 25")

# Batch queries
results = client.execute_batch([
    "INSERT INTO users VALUES (1, 'Alice', 30)",
    "INSERT INTO users VALUES (2, 'Bob', 25)",
])

# Streaming results
for chunk in client.execute_stream("SELECT * FROM large_table"):
    process(chunk)
```

## Handle Results

Results are typed by `QueryResultType`. Check the type before accessing data:

```python
from neumann import QueryResultType

result = client.query("SELECT * FROM users")

if result.type == QueryResultType.ROWS:
    for row in result.rows:
        print(row.get_string("name"), row.get_int("age"))
        # Or convert to dict:
        print(row.to_dict())

elif result.type == QueryResultType.COUNT:
    print(f"Count: {result.count}")

elif result.type == QueryResultType.NODES:
    for node in result.nodes:
        print(node.id, node.label, node.properties)

elif result.type == QueryResultType.SIMILAR:
    for item in result.similar_items:
        print(f"{item.key}: {item.score:.4f}")
```

### Result types

| Type | Field | Description |
|------|-------|-------------|
| `ROWS` | `result.rows` | Relational query results |
| `NODES` | `result.nodes` | Graph nodes |
| `EDGES` | `result.edges` | Graph edges |
| `PATHS` | `result.paths` | Graph paths |
| `SIMILAR` | `result.similar_items` | Vector similarity results |
| `COUNT` | `result.count` | Integer count |
| `VALUE` | `result.value` | Single scalar value |
| `TABLE_LIST` | `result.tables` | Available tables |
| `EMPTY` | -- | No result |

## Vector Operations

For dedicated vector operations, use `VectorClient`:

```python
from neumann import VectorClient, VectorPoint

vectors = VectorClient.connect("localhost:50051", api_key="your-key")

# Create a collection
vectors.create_collection("documents", dimension=384, distance="cosine")

# Upsert points
vectors.upsert_points("documents", [
    VectorPoint(id="doc1", vector=[0.1, 0.2, ...], payload={"title": "Hello"}),
    VectorPoint(id="doc2", vector=[0.3, 0.4, ...], payload={"title": "World"}),
])

# Query similar points
results = vectors.query_points(
    "documents",
    query_vector=[0.15, 0.25, ...],
    limit=10,
    score_threshold=0.8,
    with_payload=True,
)

for point in results:
    print(f"{point.id}: {point.score:.4f} - {point.payload}")

# Manage collections
names = vectors.list_collections()
info = vectors.get_collection("documents")
count = vectors.count_points("documents")

vectors.close()
```

## Pandas Integration

Convert query results to DataFrames:

```python
from neumann.integrations.pandas import result_to_dataframe, dataframe_to_inserts

# Query to DataFrame
result = client.query("SELECT * FROM users")
df = result_to_dataframe(result)
print(df.head())

# DataFrame to INSERT queries
queries = dataframe_to_inserts(df, "users_backup")
client.execute_batch(queries)
```

## NumPy Integration

Work with vectors as NumPy arrays:

```python
import numpy as np
from neumann.integrations.numpy import (
    vector_to_insert,
    vectors_to_inserts,
    cosine_similarity,
    normalize_vectors,
)

# Single vector insert
query = vector_to_insert("doc1", np.array([0.1, 0.2, 0.3]), normalize=True)
client.query(query)

# Batch insert
vectors_dict = {"doc1": np.array([0.1, 0.2]), "doc2": np.array([0.3, 0.4])}
queries = vectors_to_inserts(vectors_dict)
client.execute_batch(queries)

# Compute similarity locally
sim = cosine_similarity(vec1, vec2)
```

## Error Handling

```python
from neumann import (
    NeumannError,
    ConnectionError,
    AuthenticationError,
    NotFoundError,
    QueryError,
    ParseError,
)

try:
    result = client.query("SELECT * FROM nonexistent")
except NotFoundError as e:
    print(f"Not found: {e.message}")
except ParseError as e:
    print(f"Syntax error: {e.message}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except NeumannError as e:
    print(f"Error [{e.code.name}]: {e.message}")
```

## Configuration

Fine-tune timeouts, retries, and keepalive:

```python
from neumann import ClientConfig, TimeoutConfig, RetryConfig

config = ClientConfig(
    timeout=TimeoutConfig(
        default_timeout_s=30.0,
        connect_timeout_s=10.0,
    ),
    retry=RetryConfig(
        max_attempts=3,
        initial_backoff_ms=100,
        max_backoff_ms=10000,
        backoff_multiplier=2.0,
    ),
)

client = NeumannClient.connect("localhost:50051", config=config)

# Preset configurations
config = ClientConfig.fast_fail()       # 5s timeout, 1 attempt
config = ClientConfig.no_retry()        # Default timeout, 1 attempt
config = ClientConfig.high_latency()    # 120s timeout, 5 attempts
```

## Next Steps

- [Query Language Reference](../reference/query-language.md) -- All commands
- [TypeScript SDK](typescript-quickstart.md) -- TypeScript alternative
- [Architecture](../architecture/neumann-py.md) -- SDK internals
