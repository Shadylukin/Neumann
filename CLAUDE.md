# CLAUDE.md

This file provides guidance for Claude Code when working on this project.

## Project Overview

Neumann is a unified tensor-based runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure.

## Modules

| Module | Purpose | Depends On |
|--------|---------|------------|
| `tensor_store` | Key-value storage layer | - |
| `relational_engine` | SQL-like tables with indexes | tensor_store |
| `graph_engine` | Graph nodes and edges | tensor_store |
| `vector_engine` | Embeddings and similarity search | tensor_store |
| `neumann_parser` | Query tokenization and parsing | - |
| `query_router` | Unified query execution | all engines, parser |
| `neumann_shell` | Interactive CLI interface | query_router |

## Code Style

- No emojis in code, comments, or commit messages
- Use Rust idioms: prefer iterators over loops, use `?` for error propagation
- Keep functions small and focused
- Prefer composition over inheritance patterns

## Comments Policy

Doc comments (`///`) are for rustdoc generation. Use them sparingly:

**DO document:**
- Types (structs, enums) - explain purpose and invariants
- Non-obvious behavior - when a method does something unexpected
- Complex algorithms - when the "why" isn't clear from code

**DO NOT document:**
- Methods with self-explanatory names (`get`, `set`, `new`, `len`, `is_empty`)
- Trivial implementations
- Anything where the doc would just repeat the function name

**Examples:**
```rust
// BAD - restates the obvious
/// Get a field value
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - no comment needed, name is clear
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - explains non-obvious behavior
/// Returns cloned data to ensure thread safety. For zero-copy access, use get_ref().
pub fn get(&self, key: &str) -> Result<TensorData>
```

Inline comments (`//`) should explain "why", never "what".

## Quality Standards

All code must pass before commit:
- `cargo fmt --check` - formatting
- `cargo clippy -- -D warnings` - lints as errors
- `cargo test` - all tests pass
- `cargo doc --no-deps` - documentation builds
- 95% minimum line coverage per crate (94% for shell due to interactive REPL)

## Testing Philosophy

- Unit tests live in the same file as the code (`#[cfg(test)]` module)
- Test the public API, not implementation details
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include edge cases: empty inputs, boundaries, error conditions
- Performance tests for operations that must scale (10k+ entities)
- Concurrent tests for thread-safe code

## Architecture

See `docs/architecture.md` for full system design.

```
tensor_store/           # Module 1: Storage layer
  src/lib.rs            # Core types and TensorStore implementation
relational_engine/      # Module 2: Relational operations
  src/lib.rs            # Tables, schemas, conditions, indexes
graph_engine/           # Module 3: Graph operations
  src/lib.rs            # Nodes, edges, traversals
vector_engine/          # Module 4: Vector operations
  src/lib.rs            # Embeddings, similarity search
neumann_parser/         # Module 5: Query parsing
  src/lib.rs            # Tokenization, parsing, AST
query_router/           # Module 6: Query execution
  src/lib.rs            # Unified query routing
neumann_shell/          # Module 7: CLI interface
  src/lib.rs            # Shell implementation
  src/main.rs           # Binary entry point
docs/
  architecture.md       # System architecture overview
  tensor-store.md       # Module 1 API documentation
  relational-engine.md  # Module 2 API documentation
  graph-engine.md       # Module 3 API documentation
  vector-engine.md      # Module 4 API documentation
  query-router.md       # Module 5 API documentation
  neumann-parser.md     # Module 6 API documentation
  neumann-shell.md      # Module 7 API documentation
  benchmarks.md         # Performance benchmarks
```

## Key Types

### Tensor Store
- `TensorValue`: Scalar, Vector, Pointer, or Pointers
- `TensorData`: A map of field names to TensorValues
- `TensorStore`: Thread-safe key-value store using DashMap

### Relational Engine
- `Schema`, `Column`, `ColumnType`: Table structure
- `Value`: Typed values (Int, Float, String, Bool, Null)
- `Condition`: Composable predicates for filtering
- `RelationalEngine`: Table operations with index support

### Graph Engine
- `NodeData`, `EdgeData`: Graph element properties
- `Direction`: Edge traversal direction
- `GraphEngine`: Node/edge CRUD and traversals

### Vector Engine
- `SearchResult`: Key and similarity score
- `VectorEngine`: Embedding storage and k-NN search

### Neumann Shell
- `Shell`: Interactive REPL with query execution
- `ShellConfig`: Configuration (history, prompt)
- `CommandResult`: Output, Exit, Help, Empty, Error

## Concurrency Design

All engines inherit thread safety from TensorStore's DashMap:

- **Why**: Better concurrent write performance, no lock poisoning
- **How**: DashMap uses ~16 shards, writes only block same-shard writes
- **Trade-off**: Adds dashmap dependency, but eliminates failure modes

When adding new concurrent data structures:
1. Prefer sharded/partitioned designs over single locks
2. Avoid lock poisoning by using parking_lot or dashmap
3. Always add concurrent tests (`store_concurrent_*`)
4. Document the concurrency model in doc comments

## Commit Guidelines

- Write clear, imperative commit messages
- No emoji in commits
- Reference issue numbers when applicable
- Keep commits atomic - one logical change per commit
