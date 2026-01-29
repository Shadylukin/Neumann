# Changelog

All notable changes to the Neumann TypeScript SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Comprehensive documentation
- CI/CD pipeline

## [0.1.0] - 2026-01-27

### Added
- **Dual environment client**: Node.js (gRPC) and browser (gRPC-Web) support
- **NeumannClient class**: Full-featured client implementation
  - `connect()` - gRPC connection for Node.js
  - `connectWeb()` - gRPC-Web connection for browsers
  - `execute()` - single query execution
  - `executeStream()` - streaming query results
  - `executeBatch()` - batch query execution
- **Type-safe values**: Discriminated union value types
  - `intValue()`, `floatValue()`, `stringValue()`, `boolValue()`, `bytesValue()`, `nullValue()`
  - `valueToNative()`, `valueFromNative()` conversion utilities
- **Query result types**: 12 typed result variants
  - Rows, Nodes, Edges, Paths for structured data
  - SimilarItem for vector search results
  - Type guard functions for result discrimination
- **Error handling**: Typed error hierarchy
  - `NeumannError` base class with error codes
  - `ConnectionError`, `AuthenticationError`, `PermissionDeniedError`
  - `NotFoundError`, `InvalidArgumentError`, `ParseError`
  - `QueryError`, `InternalError`
  - `errorFromCode()` factory function
- **Proto conversion**: Utility functions for proto-to-SDK type conversion
  - `convertProtoValue()`, `convertProtoRow()`, `convertProtoNode()`
  - `convertProtoEdge()`, `convertProtoPath()`, `convertProtoSimilarItem()`
  - `convertProtoArtifactInfo()`
- **Object conversion**: Convenience functions for plain objects
  - `rowToObject()`, `nodeToObject()`, `edgeToObject()`

### Configuration
- ESM and CommonJS dual build support
- TypeScript strict mode enabled
- Vitest test framework with 95%+ coverage thresholds
- ESLint with TypeScript recommended rules

[Unreleased]: https://github.com/neumann-db/neumann/compare/ts-v0.1.0...HEAD
[0.1.0]: https://github.com/neumann-db/neumann/releases/tag/ts-v0.1.0
