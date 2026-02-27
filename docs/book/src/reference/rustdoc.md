# Rustdoc

Auto-generated API documentation from source code.

## Building Locally

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

## Crate Documentation

After generating docs locally with `cargo doc`, you can browse
documentation for each crate. See the per-crate API reference pages
for type tables and usage examples:

| Crate | API Reference |
| --- | --- |
| `tensor_store` | [tensor_store API](api/tensor-store.md) |
| `relational_engine` | [relational_engine API](api/relational-engine.md) |
| `graph_engine` | [graph_engine API](api/graph-engine.md) |
| `vector_engine` | [vector_engine API](api/vector-engine.md) |
| `tensor_chain` | [tensor_chain API](api/tensor-chain.md) |
| `neumann_parser` | [neumann_parser API](api/neumann-parser.md) |
| `query_router` | [query_router API](api/query-router.md) |
| `tensor_cache` | [tensor_cache API](api/tensor-cache.md) |
| `tensor_vault` | [tensor_vault API](api/tensor-vault.md) |
| `tensor_blob` | [tensor_blob API](api/tensor-blob.md) |
| `tensor_checkpoint` | [tensor_checkpoint API](api/tensor-checkpoint.md) |
| `tensor_unified` | [tensor_unified API](api/tensor-unified.md) |
| `tensor_compress` | [tensor_compress API](api/tensor-compress.md) |
| `tensor_spatial` | [tensor_spatial API](api/tensor-spatial.md) |
| `neumann_shell` | [neumann_shell API](api/neumann-shell.md) |
| `neumann_server` | [neumann_server API](api/neumann-server.md) |
| `neumann_client` | [neumann_client API](api/neumann-client.md) |
| `neumann-ts` | [TypeScript SDK API](api/neumann-ts.md) |
| `neumann-py` | [Python SDK API](api/neumann-py.md) |

## Thread Safety

All engines inherit thread safety from TensorStore's SlabRouter:

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

## Async Operations

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
