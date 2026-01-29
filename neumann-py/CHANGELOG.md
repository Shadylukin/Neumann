# Changelog

All notable changes to the Neumann Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Comprehensive documentation
- CI/CD pipeline

## [0.1.0] - 2026-01-27

### Added
- **Dual-mode client**: Embedded (in-process) and Remote (gRPC) operation
- **Sync client** (`NeumannClient`): Full-featured synchronous client
  - `execute()` - single query execution
  - `execute_batch()` - batch query execution
  - `execute_stream()` - streaming results for large datasets
  - Context manager support for clean resource management
- **Async client** (`AsyncNeumannClient`): Async/await support
  - Async versions of all client methods
  - Async iterator for streaming results
- **Type-safe results**: 20+ typed result variants
  - Rows, Nodes, Edges, Paths for structured data
  - SimilarItem for vector search results
  - Transaction and checkpoint results
  - Chain/blockchain operation results
- **Error handling**: Typed exception hierarchy
  - ConnectionError, AuthenticationError, PermissionError
  - NotFoundError, InvalidArgumentError, ParseError
  - QueryError, InternalError
- **Retry logic**: Automatic retry with exponential backoff
  - Configurable max attempts and delays
  - Jitter to prevent thundering herd
  - gRPC status code awareness
- **Transaction support**: ACID transactions
  - Context manager for automatic commit/rollback
  - Explicit begin/commit/rollback methods
- **Configuration**: Flexible configuration system
  - RetryConfig with presets (default, no_retry, fast_fail, high_latency)
  - TimeoutConfig for operation-specific timeouts
  - KeepaliveConfig for gRPC connection tuning
- **Integrations**:
  - Pandas: `result_to_dataframe()`, `dataframe_to_inserts()`
  - NumPy: Vector operations, similarity metrics
- **Native bindings**: PyO3/Rust bindings for embedded mode
  - Direct access to Neumann engines
  - Zero-copy where possible
  - Snapshot save/load

### Security
- TLS support for encrypted connections
- API key authentication
- Identity-based access control for Vault operations

### Documentation
- Comprehensive README with examples
- Full API reference
- Type hints for IDE support

[Unreleased]: https://github.com/Shadylukin/Neumann/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shadylukin/Neumann/releases/tag/v0.1.0
