# Neumann Python SDK

Python client library for the Neumann database - a unified tensor-based
runtime for relational, graph, and vector data.

## Installation

```bash
pip install neumann-db
```

For embedded mode with native bindings:

```bash
pip install neumann-db[native]
```

For integration support:

```bash
pip install neumann-db[pandas,numpy]
```

## Quick Start

### Embedded Mode (In-Process)

```python
from neumann import NeumannClient

# Create an in-memory embedded client
with NeumannClient.embedded() as client:
    # Create a table
    client.execute("CREATE TABLE users (id INT, name STRING, email STRING)")

    # Insert data
    client.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
    client.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")

    # Query data
    result = client.execute("SELECT * FROM users")
    for row in result.rows():
        print(f"{row['id']}: {row['name']}")
```

### Remote Mode (gRPC)

```python
from neumann import NeumannClient

# Connect to a remote server
with NeumannClient.connect("localhost:50051", api_key="your-api-key") as client:
    result = client.execute("SELECT * FROM users")
    for row in result.rows():
        print(row)
```

### Async Client

```python
import asyncio
from neumann.aio import AsyncNeumannClient

async def main():
    async with await AsyncNeumannClient.connect("localhost:50051") as client:
        result = await client.execute("SELECT * FROM users")
        print(result)

asyncio.run(main())
```

## Features

### Relational Queries

```python
# Create tables with schemas
client.execute("""
    CREATE TABLE products (
        id INT PRIMARY KEY,
        name STRING,
        price FLOAT,
        in_stock BOOL
    )
""")

# Insert, update, delete
client.execute("INSERT INTO products VALUES (1, 'Widget', 29.99, true)")
client.execute("UPDATE products SET price = 24.99 WHERE id = 1")
client.execute("DELETE FROM products WHERE in_stock = false")

# Query with conditions
result = client.execute("SELECT * FROM products WHERE price < 50.00")
```

### Graph Queries

```python
# Create nodes
client.execute("CREATE NODE Person { name: 'Alice', age: 30 }")
client.execute("CREATE NODE Person { name: 'Bob', age: 25 }")

# Create edges
client.execute("CREATE EDGE KNOWS FROM (Person WHERE name = 'Alice') TO (Person WHERE name = 'Bob')")

# Traverse graph
result = client.execute("MATCH (p:Person)-[:KNOWS]->(friend) WHERE p.name = 'Alice' RETURN friend")
```

### Vector Search

```python
# Create vector index
client.execute("CREATE VECTOR INDEX embeddings DIMENSION 384 METRIC cosine")

# Insert embeddings
client.execute("INSERT INTO embeddings VALUES ('doc1', [0.1, 0.2, ...])")

# Similarity search
result = client.execute("SEARCH embeddings SIMILAR TO [0.1, 0.2, ...] LIMIT 10")
for item in result.similar():
    print(f"{item.key}: {item.score}")
```

### Vault (Encrypted Secrets)

```python
# Store encrypted secrets with access control
client.execute("VAULT SET api_key 'secret-value'", identity="admin")

# Retrieve secrets
result = client.execute("VAULT GET api_key", identity="admin")

# Grant access
client.execute("VAULT GRANT READ ON api_key TO 'service-account'", identity="admin")
```

## Query Result Types

The `QueryResult` object provides typed access to results:

```python
result = client.execute("SELECT * FROM users")

# Check result type
if result.result_type == QueryResultType.ROWS:
    for row in result.rows():
        print(row['name'])

elif result.result_type == QueryResultType.NODES:
    for node in result.nodes():
        print(f"{node.label}: {node.properties}")

elif result.result_type == QueryResultType.SIMILAR:
    for item in result.similar():
        print(f"{item.key}: {item.score}")

elif result.result_type == QueryResultType.COUNT:
    print(f"Count: {result.count()}")
```

## Pandas Integration

```python
from neumann.integrations import result_to_dataframe, dataframe_to_inserts

# Query to DataFrame
result = client.execute("SELECT * FROM users")
df = result_to_dataframe(result)

# DataFrame to INSERT statements
inserts = dataframe_to_inserts("users", df)
for stmt in inserts:
    client.execute(stmt)
```

## NumPy Integration

```python
import numpy as np
from neumann.integrations import (
    vector_to_insert,
    cosine_similarity,
    normalize_vectors,
)

# Insert vector embedding
embedding = np.array([0.1, 0.2, 0.3, ...])
client.execute(vector_to_insert("embeddings", "doc1", embedding))

# Calculate similarity
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.707, 0.707, 0.0])
sim = cosine_similarity(a, b)

# Normalize vectors
vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
normalized = normalize_vectors(vectors)
```

## Batch Operations

```python
# Execute multiple queries in a batch
results = client.execute_batch([
    "INSERT INTO users VALUES (1, 'Alice')",
    "INSERT INTO users VALUES (2, 'Bob')",
    "SELECT COUNT(*) FROM users",
])

count = results[2].count()  # 2
```

## Streaming Results

For large result sets, use streaming:

```python
for chunk in client.execute_stream("SELECT * FROM large_table"):
    for row in chunk.rows():
        process(row)
```

## Error Handling

```python
from neumann.errors import (
    NeumannError,
    ConnectionError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    ParseError,
    QueryError,
)

try:
    client.execute("INVALID QUERY")
except ParseError as e:
    print(f"Parse error: {e}")
except QueryError as e:
    print(f"Query error: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except NeumannError as e:
    print(f"General error: {e}")
```

## Configuration

### TLS Connection

```python
client = NeumannClient.connect(
    "localhost:50051",
    tls=True,
    api_key="your-api-key",
)
```

### Persistent Storage (Embedded)

```python
client = NeumannClient.embedded(path="/path/to/data")
```

## Type Hints

The SDK is fully typed with Python type hints. Use with mypy or your IDE
for better development experience:

```python
from neumann import NeumannClient
from neumann.types import QueryResult, Row, Node, Edge, Value

def get_user_names(client: NeumannClient) -> list[str]:
    result: QueryResult = client.execute("SELECT name FROM users")
    return [row["name"].as_string() for row in result.rows()]
```

## API Reference

### NeumannClient

| Method | Description |
| ------ | ----------- |
| `embedded(path=None)` | Create embedded client |
| `connect(address, api_key=None, tls=False)` | Connect to remote server |
| `execute(query, identity=None)` | Execute single query |
| `execute_batch(queries, identity=None)` | Execute batch of queries |
| `execute_stream(query, identity=None)` | Execute streaming query |
| `close()` | Close connection |

### QueryResult

| Property/Method | Description |
| --------------- | ----------- |
| `result_type` | Type of result (ROWS, NODES, etc.) |
| `data` | Raw result data |
| `rows()` | Get rows (for ROWS type) |
| `nodes()` | Get nodes (for NODES type) |
| `edges()` | Get edges (for EDGES type) |
| `similar()` | Get similar items (for SIMILAR type) |
| `count()` | Get count (for COUNT type) |

## License

MIT License - see LICENSE file for details.
