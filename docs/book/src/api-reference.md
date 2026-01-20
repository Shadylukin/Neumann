# API Reference

Full API documentation is generated from source code using rustdoc.

## Online Documentation

When deployed, the API reference is available at:

- [tensor_store](api/tensor_store/index.html) - Core storage layer
- [relational_engine](api/relational_engine/index.html) - SQL-like tables
- [graph_engine](api/graph_engine/index.html) - Graph operations
- [vector_engine](api/vector_engine/index.html) - Embeddings and similarity
- [tensor_chain](api/tensor_chain/index.html) - Distributed consensus
- [neumann_parser](api/neumann_parser/index.html) - Query parsing
- [query_router](api/query_router/index.html) - Query execution
- [neumann_shell](api/neumann_shell/index.html) - Interactive CLI
- [tensor_compress](api/tensor_compress/index.html) - Compression algorithms
- [tensor_vault](api/tensor_vault/index.html) - Encrypted storage
- [tensor_cache](api/tensor_cache/index.html) - LLM response caching
- [tensor_blob](api/tensor_blob/index.html) - Blob storage
- [tensor_checkpoint](api/tensor_checkpoint/index.html) - Snapshots
- [tensor_unified](api/tensor_unified/index.html) - Multi-engine facade

## Building Locally

Generate documentation locally:

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

## Documentation Standards

All public items should be documented following these guidelines:

### Module Documentation

```rust
//! # Module Name
//!
//! Brief one-line description.
//!
//! ## Overview
//!
//! Longer explanation of purpose and design.
//!
//! ## Example
//!
//! \`\`\`rust
//! // Example code
//! \`\`\`
```

### Type Documentation

```rust
/// Brief description of the type.
///
/// Longer explanation if needed.
///
/// # Example
///
/// \`\`\`rust
/// let value = MyType::new();
/// \`\`\`
pub struct MyType { ... }
```

### Method Documentation

Self-explanatory methods (`get`, `set`, `new`, `len`) don't need docs. Complex
methods should explain:

- What the method does
- Parameters and return values
- Panics or errors
- Examples for non-obvious usage
