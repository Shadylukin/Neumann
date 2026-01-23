# Contributing to Neumann

Thank you for your interest in contributing to Neumann! This project aims to
provide unified tensor-based storage for the AI era, and we welcome
contributions of all kinds.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Cargo (comes with Rust)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Install the pre-commit hook
./scripts/setup-hooks.sh

# Build all crates
cargo build

# Run tests
cargo test

# Verify everything passes
./scripts/pre-commit
```

### Project Structure

Neumann is organized as a Cargo workspace with 11 crates:

| Crate              | Purpose                      |
| ------------------ | ---------------------------- |
| `tensor_store`     | Core key-value storage layer |
| `relational_engine`| SQL-like table operations    |
| `graph_engine`     | Node/edge graph operations   |
| `vector_engine`    | Embedding similarity search  |
| `tensor_compress`  | Compression algorithms       |
| `tensor_vault`     | Encrypted secret storage     |
| `tensor_cache`     | LLM response caching         |
| `tensor_blob`      | Chunked blob storage         |
| `neumann_parser`   | Query language parser        |
| `query_router`     | Unified query execution      |
| `neumann_shell`    | Interactive CLI              |

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/Shadylukin/Neumann/issues)
   to avoid duplicates
2. Use the bug report template
3. Include:
   - Rust version (`rustc --version`)
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features

1. Open an issue with the feature request template
2. Describe the use case and motivation
3. Be open to discussion about implementation approaches

### Submitting Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure all checks pass:

   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   cargo doc --no-deps
   ```

5. Commit with a clear message (see below)
6. Push and open a pull request

## Code Standards

### Style

- Run `cargo fmt` before committing
- No warnings from `cargo clippy -- -D warnings`
- No emojis in code, comments, or commit messages
- Prefer iterators over loops
- Use `?` for error propagation
- Keep functions small and focused

### Testing

- Minimum 95% line coverage per crate (some exceptions for interactive code)
- Test the public API, not implementation details
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include edge cases: empty inputs, boundaries, error conditions

### Documentation

- Doc comments (`///`) for public types and non-obvious behavior
- Don't document self-explanatory methods (`get`, `set`, `new`)
- Inline comments (`//`) explain "why", never "what"

### Commit Messages

Write clear, imperative commit messages:

```text
Add vector similarity search with cosine metric

- Implement cosine, euclidean, and dot product metrics
- Add HNSW index for O(log n) approximate search
- Include benchmarks showing 10x speedup over brute force
```

## Pull Request Process

1. Update documentation if you changed public APIs
2. Add tests for new functionality
3. Ensure CI passes (all checks green)
4. Request review from maintainers
5. Address feedback promptly
6. Squash commits if requested

## Community

- [Discord](https://discord.gg/uN3KbAyKvw) - Chat with the community
- [GitHub Issues](https://github.com/Shadylukin/Neumann/issues) - Bug reports
- [GitHub Discussions](https://github.com/Shadylukin/Neumann/discussions) -
  Questions and ideas

## License

By contributing, you agree that your contributions will be licensed under the
same terms as the project: MIT OR Apache-2.0.
