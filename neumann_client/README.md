# neumann_client

Rust client SDK for the Neumann database.

## Features

- **Dual-mode operation**: Embedded (in-process) and Remote (gRPC)
- **Async-first**: Built on Tokio for remote connections
- **Type-safe**: Strong typing with comprehensive error handling
- **Secure**: Automatic API key zeroization on drop

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
neumann_client = "0.1"
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `remote` | gRPC client for remote connections | Yes |
| `embedded` | In-process database via QueryRouter | No |
| `full` | Both embedded and remote modes | No |

## Quick Start

### Remote Connection

```rust
use neumann_client::{NeumannClient, ClientError};

#[tokio::main]
async fn main() -> Result<(), ClientError> {
    // Connect to server
    let client = NeumannClient::connect("localhost:9200")
        .api_key("your-api-key")
        .with_tls()
        .build()
        .await?;

    // Execute queries
    let result = client.execute("SELECT users").await?;

    if let Some(rows) = result.rows() {
        println!("Found {} rows", rows.len());
    }

    Ok(())
}
```

### Embedded Mode

Enable the `embedded` feature:

```toml
[dependencies]
neumann_client = { version = "0.1", features = ["embedded"] }
```

```rust
use neumann_client::{NeumannClient, ClientError};

fn main() -> Result<(), ClientError> {
    let client = NeumannClient::embedded()?;

    client.execute_sync("CREATE TABLE users (name:string, age:int)")?;
    client.execute_sync("INSERT users name=\"Alice\", age=30")?;

    let result = client.execute_sync("SELECT users")?;
    println!("{:?}", result);

    Ok(())
}
```

## API Reference

### NeumannClient

| Method | Description |
|--------|-------------|
| `connect(address)` | Create builder for remote connection |
| `embedded()` | Create in-process client |
| `with_router(router)` | Create client with custom QueryRouter |
| `execute(query)` | Execute async query (remote) |
| `execute_sync(query)` | Execute sync query (embedded) |
| `execute_batch(queries)` | Execute multiple queries |
| `close()` | Close the connection |
| `is_connected()` | Check connection status |
| `mode()` | Get client mode |

### ClientBuilder

| Method | Description |
|--------|-------------|
| `api_key(key)` | Set API key for authentication |
| `with_tls()` | Enable TLS encryption |
| `timeout_ms(ms)` | Set connection timeout |
| `build()` | Connect and create client |
| `build_blocking()` | Blocking version of build |

### RemoteQueryResult

| Method | Description |
|--------|-------------|
| `is_empty()` | Check if result is empty |
| `has_error()` | Check for error |
| `error_message()` | Get error message |
| `count()` | Get count result |
| `rows()` | Get rows result |
| `nodes()` | Get graph nodes |
| `edges()` | Get graph edges |
| `similar()` | Get similarity search results |

## Error Handling

```rust
use neumann_client::{NeumannClient, ClientError};

async fn example() {
    let result = NeumannClient::connect("localhost:9200")
        .build()
        .await;

    match result {
        Ok(client) => { /* use client */ }
        Err(ClientError::Connection(msg)) => {
            eprintln!("Connection failed: {}", msg);
        }
        Err(ClientError::Authentication(msg)) => {
            eprintln!("Auth failed: {}", msg);
        }
        Err(e) if e.is_retryable() => {
            eprintln!("Retryable error: {}", e);
        }
        Err(e) => {
            eprintln!("Error (code {}): {}", e.code(), e);
        }
    }
}
```

### Error Types

| Error | Code | Retryable |
|-------|------|-----------|
| `Connection` | 6 | Yes |
| `Timeout` | 6 | Yes |
| `Unavailable` | 6 | Yes |
| `Authentication` | 5 | No |
| `PermissionDenied` | 3 | No |
| `NotFound` | 2 | No |
| `InvalidArgument` | 1 | No |
| `Parse` | 8 | No |
| `Query` | 9 | No |
| `Internal` | 7 | No |

## Security

- API keys are automatically zeroized when `ClientConfig` is dropped
- TLS support for encrypted connections
- Identity context for multi-tenant scenarios

## Requirements

- Rust 1.75.0 or later
- Tokio runtime (for remote mode)

## License

MIT OR Apache-2.0
