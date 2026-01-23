# Agent Guide

## Purpose

Neumann is a unified tensor-based runtime for relational data, graph
relationships, and vector embeddings in one system.

## Docs First Map

- README.md for the high-level overview and CLI examples.
- docs/book for the detailed system guide (architecture, concepts,
  operations, and contributing).
- docs/architecture.md for module boundaries and interfaces.
- docs/DOCUMENTATION_STANDARDS.md for Markdown rules and style.

## Architecture Snapshot

- neumann_shell provides the CLI and REPL interface.
- query_router dispatches parsed queries to engines.
- neumann_parser tokenizes and parses the query language.
- Core engines: relational_engine, graph_engine, vector_engine.
- Storage base: tensor_store with tensor_compress.
- Extended modules: tensor_vault, tensor_cache, tensor_blob,
  tensor_checkpoint, tensor_unified.
- Distributed layer: tensor_chain (Raft, 2PC, SWIM gossip, HLC time,
  deadlock detection) with TCP transport.

## Data Model and Semantics

- TensorValue covers scalars, vectors, pointers, and pointer lists.
- TensorData maps field names to TensorValue items.
- SparseVector and DeltaVector reduce storage and enable fast similarity.
- Similarity metrics include cosine, euclidean, and dot product.

## Repository Map

- tensor_store, tensor_compress, relational_engine, graph_engine,
  vector_engine.
- tensor_vault, tensor_cache, tensor_blob, tensor_checkpoint,
  tensor_unified.
- tensor_chain for consensus, transactions, gossip, and transport.
- neumann_parser, query_router, neumann_shell.
- neumann_server and neumann_client for gRPC access.
- integration_tests, stress_tests, fuzz, docs.

## Build and Run

```bash
cargo build
cargo build --release
cargo run -p neumann_shell
```

## Tests and Quality Gates

```bash
cargo test
cargo test -p tensor_chain
cargo test -- --nocapture
cargo test -- --ignored
cargo test -p integration_tests
```

```bash
cargo fmt --check
cargo clippy -- -D warnings -W clippy::pedantic -W clippy::nursery \
  -W clippy::cargo
cargo doc --no-deps
./scripts/pre-commit
```

## Documentation Workflow

```bash
./scripts/validate-docs.sh
cd docs/book && mdbook build
```

## Fuzzing

```bash
cargo install cargo-fuzz
cd fuzz && cargo +nightly fuzz run parser_parse -- -max_total_time=60
```

## Coverage Targets

- scripts/pre-commit enforces 90 percent line coverage for all crates.
- Project docs may list higher targets; align policy if needed.

## Style and Conventions

- Keep functions small, use iterators, and prefer ? for errors.
- Avoid unsafe unless required and well-justified.
- Use DashMap for concurrency; avoid Mutex where possible.
- Doc comments only for non-obvious behavior and public types.
- Inline comments explain why, not what.
- No emojis in code, comments, commits, or docs.

## Documentation Standards

- 80 character line length (headings and code blocks excluded).
- ATX headings with blank lines around them.
- Use dashes for lists and asterisks for emphasis.
- Fenced code blocks with language tags.
- Mermaid uses flowchart, not graph.

## Operational Notes

- Config precedence: defaults, config file, env, CLI flags.
- Metrics at <http://node:9090/metrics>; health at <http://node:9090/health>.
- Cluster ports: 7878 for Raft, 7879 for gossip, 9090 for metrics.
