# CLAUDE.md

This file provides guidance for Claude Code when working on this project.

## Project Overview

Neumann is a unified tensor-based runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure. This is the tensor_store module - the foundational storage layer.

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
docs/
  architecture.md       # System architecture overview
  tensor-store.md       # Module 1 API documentation
```

The store knows nothing about queries - it only stores and retrieves tensors by key.

## Key Types

- `TensorValue`: Scalar, Vector, Pointer, or Pointers
- `TensorData`: A map of field names to TensorValues
- `TensorStore`: Thread-safe key-value store using DashMap (sharded concurrent HashMap)

## Concurrency Design

TensorStore uses DashMap instead of RwLock<HashMap>:

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
