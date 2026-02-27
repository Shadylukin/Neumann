# Building from Source

## Development Requirements

- Rust 1.75+ (stable)
- Rust nightly (for fuzzing)
- Git

## Clone and Build

```bash
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p tensor_chain

# Run with output
cargo test -- --nocapture
```

## Quality Checks

All code must pass before commit:

```bash
# Formatting
cargo fmt --check

# Lints (warnings as errors)
cargo clippy -- -D warnings

# Documentation builds
cargo doc --no-deps
```

## Fuzzing

Requires nightly Rust:

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run a fuzz target
cd fuzz
cargo +nightly fuzz run parser_parse -- -max_total_time=60
```

## Running the Shell

```bash
# Debug mode
cargo run -p neumann_shell

# Release mode
cargo run --release -p neumann_shell
```

## IDE Setup

### VS Code

Install the rust-analyzer extension. Recommended settings:

```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.features": "all"
}
```

### IntelliJ/CLion

Install the Rust plugin. Enable clippy in settings.

## Project Structure

```text
Neumann/
├── tensor_store/       # Core storage layer
├── relational_engine/  # SQL-like tables
├── graph_engine/       # Graph operations
├── vector_engine/      # Embeddings
├── tensor_chain/       # Distributed consensus
├── neumann_parser/     # Query parsing
├── query_router/       # Query execution
├── neumann_shell/      # CLI interface
├── tensor_compress/    # Compression
├── tensor_vault/       # Encrypted storage
├── tensor_cache/       # LLM caching
├── tensor_blob/        # Blob storage
├── tensor_checkpoint/  # Snapshots
├── tensor_unified/     # Multi-engine facade
├── fuzz/               # Fuzz targets
└── docs/               # Documentation
```
