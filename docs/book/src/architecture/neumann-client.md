# Neumann Client Architecture

The Neumann Client (`neumann_client`) provides a Rust SDK for interacting with
the Neumann database. It supports two modes: embedded mode for in-process
database access via the Query Router, and remote mode for network access via
gRPC to a Neumann Server.

The client follows four design principles: dual-mode flexibility (same API for
embedded and remote), security-first (API keys are zeroized on drop),
async-native (built on tokio for remote operations), and zero-copy where
possible (streaming results for large datasets).

## Architecture Overview

```mermaid
flowchart TD
    subgraph Application
        App[User Application]
    end

    subgraph NeumannClient
        Client[NeumannClient]
        Builder[ClientBuilder]
        Config[ClientConfig]
    end

    subgraph EmbeddedMode
        Router[QueryRouter]
        Store[TensorStore]
    end

    subgraph RemoteMode
        gRPC[gRPC Channel]
        TLS[TLS Layer]
    end

    subgraph NeumannServer
        Server[NeumannServer]
    end

    App --> Builder
    Builder --> Client
    Client -->|embedded| Router
    Router --> Store
    Client -->|remote| gRPC
    gRPC --> TLS
    TLS --> Server
```

## Key Types

| Type | Description |
| --- | --- |
| `NeumannClient` | Main client struct supporting both embedded and remote modes |
| `ClientBuilder` | Fluent builder for remote client connections |
| `ClientConfig` | Configuration for remote connections (address, API key, TLS) |
| `ClientMode` | Enum: `Embedded` or `Remote` |
| `ClientError` | Error type for client operations |
| `RemoteQueryResult` | Wrapper for proto query response with typed accessors |
| `QueryResult` | Re-export of query_router result type (embedded mode) |

## Client Modes

| Mode | Feature Flag | Use Case |
| --- | --- | --- |
| Embedded | `embedded` | In-process database, unit testing, CLI tools |
| Remote | `remote` (default) | Production gRPC connections to server |
| Full | `full` | Both modes available |

### Feature Flags

```toml
[dependencies]
# Remote only (default)
neumann_client = "0.1"

# Embedded only
neumann_client = { version = "0.1", default-features = false, features = ["embedded"] }

# Both modes
neumann_client = { version = "0.1", features = ["full"] }
```

## Client Configuration

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `address` | `String` | `localhost:9200` | Server address (host:port) |
| `api_key` | `Option<String>` | `None` | API key for authentication |
| `tls` | `bool` | `false` | Enable TLS encryption |
| `timeout_ms` | `u64` | `30000` | Request timeout in milliseconds |

### Security: API Key Zeroization

API keys are automatically zeroed from memory when the configuration is dropped
to prevent credential leakage:

```rust
impl Drop for ClientConfig {
    fn drop(&mut self) {
        if let Some(ref mut key) = self.api_key {
            key.zeroize();  // Overwrites memory with zeros
        }
    }
}
```

## Remote Mode

### Connection Builder

The `ClientBuilder` provides a fluent API for configuring remote connections:

```rust
use neumann_client::NeumannClient;
use std::time::Duration;

// Minimal connection
let client = NeumannClient::connect("localhost:9200")
    .build()
    .await?;

// Full configuration
let client = NeumannClient::connect("db.example.com:9443")
    .api_key("sk-production-key")
    .with_tls()
    .timeout_ms(60_000)
    .build()
    .await?;
```

### Connection Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Builder as ClientBuilder
    participant Client as NeumannClient
    participant Channel as gRPC Channel
    participant Server as NeumannServer

    App->>Builder: connect("address")
    App->>Builder: api_key("key")
    App->>Builder: with_tls()
    App->>Builder: build().await
    Builder->>Channel: Create endpoint
    Channel->>Server: TCP/TLS handshake
    Server-->>Channel: Connection established
    Channel-->>Builder: Channel ready
    Builder-->>Client: NeumannClient created
    Client-->>App: Ready for queries
```

### Query Execution

```rust
// Single query
let result = client.execute("SELECT * FROM users").await?;

// With identity (for vault access)
let result = client
    .execute_with_identity("VAULT GET 'secret'", Some("service:backend"))
    .await?;

// Batch queries
let results = client
    .execute_batch(&[
        "CREATE TABLE orders (id:int, total:float)",
        "INSERT orders id=1, total=99.99",
        "SELECT orders",
    ])
    .await?;
```

### RemoteQueryResult Accessors

The `RemoteQueryResult` wrapper provides typed access to query results:

```rust
let result = client.execute("SELECT * FROM users").await?;

// Check for errors
if result.has_error() {
    eprintln!("Error: {}", result.error_message().unwrap());
    return Err(...);
}

// Check result type
if result.is_empty() {
    println!("No results");
}

// Access typed data
if let Some(count) = result.count() {
    println!("Count: {}", count);
}

if let Some(rows) = result.rows() {
    for row in rows {
        println!("Row ID: {}", row.id);
    }
}

if let Some(nodes) = result.nodes() {
    for node in nodes {
        println!("Node: {} ({})", node.id, node.label);
    }
}

if let Some(edges) = result.edges() {
    for edge in edges {
        println!("Edge: {} -> {}", edge.from, edge.to);
    }
}

if let Some(similar) = result.similar() {
    for item in similar {
        println!("{}: {:.4}", item.key, item.score);
    }
}

// Access raw proto response
let proto = result.into_inner();
```

### Blocking Connection

For synchronous contexts, use the blocking builder:

```rust
let client = NeumannClient::connect("localhost:9200")
    .api_key("test-key")
    .build_blocking()?;  // Creates temporary tokio runtime
```

## Embedded Mode

### Creating an Embedded Client

```rust
use neumann_client::NeumannClient;

// New embedded database
let client = NeumannClient::embedded()?;

// With custom router (for shared state)
use query_router::QueryRouter;
use std::sync::Arc;
use parking_lot::RwLock;

let router = Arc::new(RwLock::new(QueryRouter::new()));
let client = NeumannClient::with_router(router);
```

### Synchronous Query Execution

Embedded mode provides synchronous execution for simpler code flow:

```rust
use neumann_client::QueryResult;

// Create table
let result = client.execute_sync("CREATE TABLE users (name:string, age:int)")?;
assert!(matches!(result, QueryResult::Empty));

// Insert data
let result = client.execute_sync("INSERT users name=\"Alice\", age=30")?;

// Query data
let result = client.execute_sync("SELECT users")?;
match result {
    QueryResult::Rows(rows) => {
        for row in rows {
            println!("{:?}", row);
        }
    }
    _ => {}
}
```

### With Identity

```rust
// Set identity for vault access control
let result = client.execute_sync_with_identity(
    "VAULT GET 'api_secret'",
    Some("service:backend"),
)?;
```

## Error Handling

### Error Types

| Error | Code | Retryable | Description |
| --- | --- | --- | --- |
| `Connection` | 6 | Yes | Failed to connect to server |
| `Query` | 9 | No | Query execution failed |
| `Authentication` | 5 | No | Invalid API key |
| `PermissionDenied` | 3 | No | Access denied |
| `NotFound` | 2 | No | Resource not found |
| `InvalidArgument` | 1 | No | Bad request data |
| `Parse` | 8 | No | Query parse error |
| `Internal` | 7 | No | Server internal error |
| `Timeout` | 6 | Yes | Request timed out |
| `Unavailable` | 6 | Yes | Server unavailable |

### Error Methods

```rust
let err = ClientError::Connection("connection refused".to_string());

// Get error code
let code = err.code();  // 6

// Check if retryable
if err.is_retryable() {
    // Retry with exponential backoff
}

// Display error
eprintln!("Error: {}", err);  // "connection error: connection refused"
```

### Error Handling Pattern

```rust
use neumann_client::ClientError;

match client.execute("SELECT * FROM users").await {
    Ok(result) => {
        if result.has_error() {
            // Query-level error (e.g., table not found)
            eprintln!("Query error: {}", result.error_message().unwrap());
        } else {
            // Process results
        }
    }
    Err(ClientError::Connection(msg)) => {
        // Network error - maybe retry
        eprintln!("Connection failed: {}", msg);
    }
    Err(ClientError::Authentication(msg)) => {
        // Bad credentials - check API key
        eprintln!("Auth failed: {}", msg);
    }
    Err(ClientError::Timeout(msg)) => {
        // Request too slow - maybe retry with longer timeout
        eprintln!("Timeout: {}", msg);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

### Conversion from gRPC Status

Remote errors are automatically converted from tonic Status:

```rust
impl From<tonic::Status> for ClientError {
    fn from(status: tonic::Status) -> Self {
        match status.code() {
            Code::InvalidArgument => Self::InvalidArgument(status.message().to_string()),
            Code::NotFound => Self::NotFound(status.message().to_string()),
            Code::PermissionDenied => Self::PermissionDenied(status.message().to_string()),
            Code::Unauthenticated => Self::Authentication(status.message().to_string()),
            Code::Unavailable => Self::Unavailable(status.message().to_string()),
            Code::DeadlineExceeded => Self::Timeout(status.message().to_string()),
            _ => Self::Internal(status.message().to_string()),
        }
    }
}
```

## Connection Management

### Connection State

```rust
let client = NeumannClient::connect("localhost:9200")
    .build()
    .await?;

// Check mode
match client.mode() {
    ClientMode::Embedded => println!("In-process"),
    ClientMode::Remote => println!("Connected to server"),
}

// Check connection status
if client.is_connected() {
    // Ready for queries
}
```

### Closing Connections

```rust
let mut client = NeumannClient::connect("localhost:9200")
    .build()
    .await?;

// Explicit close
client.close();

// Or automatic on drop
drop(client);  // Connection closed, API key zeroized
```

## Usage Examples

### Complete Remote Example

```rust
use neumann_client::{NeumannClient, ClientError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to server
    let client = NeumannClient::connect("localhost:9200")
        .api_key(std::env::var("NEUMANN_API_KEY")?)
        .with_tls()
        .timeout_ms(30_000)
        .build()
        .await?;

    // Create schema
    client.execute("CREATE TABLE products (name:string, price:float)").await?;

    // Insert data
    client.execute("INSERT products name=\"Widget\", price=9.99").await?;
    client.execute("INSERT products name=\"Gadget\", price=19.99").await?;

    // Query data
    let result = client.execute("SELECT products WHERE price > 10").await?;

    if let Some(rows) = result.rows() {
        for row in rows {
            println!("Product: {:?}", row);
        }
    }

    Ok(())
}
```

### Complete Embedded Example

```rust
use neumann_client::{NeumannClient, QueryResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedded client
    let client = NeumannClient::embedded()?;

    // Create schema
    client.execute_sync("CREATE TABLE events (name:string, timestamp:int)")?;

    // Insert data
    client.execute_sync("INSERT events name=\"login\", timestamp=1700000000")?;

    // Query data
    match client.execute_sync("SELECT events")? {
        QueryResult::Rows(rows) => {
            println!("Found {} events", rows.len());
            for row in rows {
                println!("  {:?}", row);
            }
        }
        _ => println!("Unexpected result type"),
    }

    Ok(())
}
```

### Testing with Embedded Mode

```rust
#[cfg(test)]
mod tests {
    use neumann_client::{NeumannClient, QueryResult};

    #[test]
    fn test_user_creation() {
        let client = NeumannClient::embedded().unwrap();

        // Setup
        client
            .execute_sync("CREATE TABLE users (email:string, active:bool)")
            .unwrap();

        // Test
        client
            .execute_sync("INSERT users email=\"test@example.com\", active=true")
            .unwrap();

        // Verify
        let result = client.execute_sync("SELECT users").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
            }
            _ => panic!("Expected rows"),
        }
    }
}
```

### Shared Router Between Clients

```rust
use neumann_client::NeumannClient;
use query_router::QueryRouter;
use std::sync::Arc;
use parking_lot::RwLock;

// Create shared router
let router = Arc::new(RwLock::new(QueryRouter::new()));

// Create multiple clients sharing same state
let client1 = NeumannClient::with_router(Arc::clone(&router));
let client2 = NeumannClient::with_router(Arc::clone(&router));

// Changes from client1 visible to client2
client1.execute_sync("CREATE TABLE shared (x:int)")?;
let result = client2.execute_sync("SELECT shared")?;  // Works!
```

## Best Practices

### Connection Reuse

Create one client and reuse it for multiple queries:

```rust
// Good: Reuse client
let client = NeumannClient::connect("localhost:9200").build().await?;
for query in queries {
    client.execute(&query).await?;
}

// Bad: New connection per query
for query in queries {
    let client = NeumannClient::connect("localhost:9200").build().await?;
    client.execute(&query).await?;
}  // Connection overhead for each query
```

### Timeout Configuration

Set appropriate timeouts based on query complexity:

```rust
// Quick queries
let client = NeumannClient::connect("localhost:9200")
    .timeout_ms(5_000)  // 5 seconds
    .build()
    .await?;

// Complex analytics
let client = NeumannClient::connect("localhost:9200")
    .timeout_ms(300_000)  // 5 minutes
    .build()
    .await?;
```

### API Key Security

Never hardcode API keys:

```rust
// Good: Environment variable
let api_key = std::env::var("NEUMANN_API_KEY")?;
let client = NeumannClient::connect("localhost:9200")
    .api_key(api_key)
    .build()
    .await?;

// Bad: Hardcoded key
let client = NeumannClient::connect("localhost:9200")
    .api_key("sk-secret-key-12345")  // Will be in binary!
    .build()
    .await?;
```

## Dependencies

| Crate | Purpose | Feature |
| --- | --- | --- |
| `query_router` | Embedded mode query execution | `embedded` |
| `tonic` | gRPC client | `remote` |
| `tokio` | Async runtime | `remote` |
| `parking_lot` | Thread-safe router access | `embedded` |
| `zeroize` | Secure memory clearing | Always |
| `thiserror` | Error type derivation | Always |
| `tracing` | Structured logging | Always |

## Related Modules

| Module | Relationship |
| --- | --- |
| `neumann_server` | Server counterpart for remote mode |
| `query_router` | Query execution backend for embedded mode |
| `neumann_shell` | Alternative interactive interface |
