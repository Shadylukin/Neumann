# Code Style

This guide covers the coding standards for Neumann. All contributions must
follow these guidelines.

## Rust Idioms

- Prefer iterators over loops
- Use `?` for error propagation
- Keep functions small and focused
- Prefer composition over inheritance patterns

## Formatting

All code must pass `cargo fmt`:

```bash
cargo fmt --check
```

## Lints

All code must pass clippy with warnings as errors:

```bash
cargo clippy -- -D warnings
```

## Comments Policy

Doc comments (`///`) are for rustdoc generation. Use them sparingly.

### DO Document

- Types (structs, enums) - explain purpose and invariants
- Non-obvious behavior - when a method does something unexpected
- Complex algorithms - when the "why" isn't clear from code

### DO NOT Document

- Methods with self-explanatory names (`get`, `set`, `new`, `len`, `is_empty`)
- Trivial implementations
- Anything where the doc would just repeat the function name

### Examples

```rust
// BAD - restates the obvious
/// Get a field value
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - no comment needed, name is clear
pub fn get(&self, key: &str) -> Option<&TensorValue>

// GOOD - explains non-obvious behavior
/// Returns cloned data to ensure thread safety.
/// For zero-copy access, use get_ref().
pub fn get(&self, key: &str) -> Result<TensorData>
```

Inline comments (`//`) should explain "why", never "what".

## Naming

- Types: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`

## Error Handling

- Use `Result` for fallible operations
- Define error types with `thiserror`
- Provide context with error messages

```rust
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("failed to parse config: {0}")]
    ConfigParse(String),

    #[error("connection failed: {source}")]
    Connection {
        #[from]
        source: std::io::Error,
    },
}
```

## Concurrency

- Use `DashMap` for concurrent hash maps
- Avoid `Mutex` where possible (use `parking_lot` if needed)
- Document thread-safety in type docs

## Testing

- Unit tests in the same file as code (`#[cfg(test)]` module)
- Test the public API, not implementation details
- Use descriptive names: `test_<function>_<scenario>_<expected>`

## Commits

- Write clear, imperative commit messages
- No emoji in commits
- Reference issue numbers when applicable
- Keep commits atomic - one logical change per commit
