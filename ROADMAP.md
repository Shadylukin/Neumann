# Neumann Roadmap

## Vision

Neumann started as a unified storage engine -- tables, graphs, and vectors
in one system. The next phase is making that data *intelligent*: native
embedding models, natural language queries, and AI-native analytics that
replace the traditional BI stack (warehouse + transform + dashboard) with
a single runtime that understands what your data means.

## Current Status: v0.3.1

Workspace version: 0.3.1 (pre-1.0, APIs may change between minor versions).

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Where We Are

### Foundation (v0.1.0 -- shipped January 2026)

The core runtime: 19 Rust crates, ~400K lines, three query engines behind
one parser.

- **Storage**: Sharded B-trees, HNSW index, sparse/delta vectors, tiered
  hot/cold storage, Bloom filters, WAL, compressed snapshots
- **Relational engine**: SQL-like tables, B-tree indexes, SIMD filtering,
  columnar scans, hash joins
- **Graph engine**: Nodes, edges, BFS/DFS traversal, shortest path,
  property indexes
- **Vector engine**: k-NN similarity search, 15+ distance metrics,
  filtered search, collection management
- **Distributed layer**: Tensor-Raft consensus, 2PC with deadlock
  detection, SWIM gossip, geometric conflict resolution
- **Specialized storage**: Encrypted vault (AES-256-GCM), LLM response
  cache, S3-style blob store, atomic checkpoints, cross-engine unified
  entities, tensor compression
- **Tooling**: Interactive CLI, gRPC server, Python and TypeScript SDKs,
  Docker images, Homebrew formula

### Quality (v0.3.1 -- current)

Production-grade testing and documentation.

- 95%+ test coverage across all crates (per-crate thresholds enforced)
- 139 fuzz targets for parsers, serialization, crypto, and storage
- Correctness testing: loom concurrency, proptest properties,
  deterministic simulation, mutation testing
- R-tree spatial indexing (tensor_spatial)
- Parser-first execution (no legacy string-based query fallback)
- Documentation restructured to Divio system (90+ pages)

## Where We're Going

### v0.4.0 -- Intelligence

This is the current focus. Neumann becomes AI-native: it understands
your data semantically, not just structurally.

**Native embeddings.** Bundle a small embedding model (ONNX Runtime) so
Neumann can compute embeddings at insert time. Store text, get vectors
for free -- no external API calls, no pre-processing pipeline.

```sql
-- Today: user provides embeddings
EMBED STORE 'doc:1' [0.1, 0.2, 0.3, ...]

-- v0.4.0: Neumann computes them
INSERT docs id=1, content='Introduction to machine learning'
-- embedding generated automatically
```

**Semantic cache overhaul.** Extend tensor_cache beyond LLM response
caching to *insight caching* -- derived analytical results with
provenance, invalidation when source data changes, and reuse across
users asking similar questions.

**Natural language queries.** Ask questions in plain English.
Neumann translates to cross-engine queries using schema-aware
structured generation with local models (Ollama integration).

```text
> "Find customers similar to our best accounts who churned last quarter"

Translated to:
  FIND NODE customer
    WHERE status = 'churned' AND churn_date > '2025-10-01'
    SIMILAR TO 'segment:best'
    LIMIT 20
```

This is hard. It requires the embedding model for schema matching,
constrained output generation for valid AST nodes, and graceful
fallback when translation fails.

### v0.5.0 -- API Stabilization

Lock down the public surface before 1.0.

- Public API review and documentation
- Deprecation warnings for unstable APIs
- Client SDK stabilization (Python, TypeScript)
- gRPC API versioning (v1 namespace)
- Storage format migration tooling
- Error message sanitization for production

### v1.0.0 -- Stable Release (Target: Q4 2026)

**Stability guarantees:**

- No breaking changes without major version bump
- Minimum 12 months support for each major version
- Security patches backported to supported versions

**What stable means:**

- Query language syntax frozen
- gRPC service definitions frozen
- Storage format versioned with forward migration
- Distributed consensus production-validated
- Monitoring and observability built in

### v1.x -- Insights

Neumann becomes an analytical platform. The traditional BI stack --
Snowflake for storage, dbt for transforms, Qlik/Tableau for
visualization -- collapses into one system that understands
relationships, similarity, and meaning natively.

**Materialized semantic views.** Transforms that combine relational
aggregation with vector clustering and graph-aware rollups.

```sql
-- dbt-style transforms, but tensor-native
CREATE VIEW churn_risk AS
  FIND NODE customer
    SIMILAR TO 'segment:churned'
    WHERE last_active < '2026-01-01'
    LIMIT 100
```

**Connectors.** Ingest from Snowflake, BigQuery, Postgres. Neumann
doesn't replace the warehouse -- it becomes the AI-native analytical
layer that sits alongside it.

**Streaming queries.** Live dashboards powered by continuous query
evaluation. Results update as source data changes.

**Enterprise features.** Multi-tenancy, RBAC, audit log retention,
cluster management UI.

### v2.0 -- Next Generation

Planning phase.

- GPU-accelerated tensor operations
- Federated learning across Neumann clusters
- Native Python/TypeScript in-process embedding
- Real-time collaborative analytics

## Version Policy

Neumann follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

## Breaking Change Policy

### Before 1.0.0

- Breaking changes may occur in minor versions
- Changes are documented in release notes
- Migration guides provided when feasible

### After 1.0.0

1. **Deprecation**: Feature marked deprecated with warning
2. **Grace Period**: Minimum 2 minor versions before removal
3. **Removal**: Only in next major version
4. **Migration**: Guide and tooling provided

Security vulnerabilities may require immediate breaking changes without
the standard deprecation period.

## Support Matrix

| Version | Status       | Security Fixes | Bug Fixes |
|---------|--------------|----------------|-----------|
| 1.x     | Planned      | Yes            | Yes       |
| 0.5.x   | Planned      | Yes            | Yes       |
| 0.4.x   | Planned      | Yes            | Yes       |
| 0.3.x   | Active       | Yes            | Yes       |
| 0.1.x   | EOL          | No             | No        |

## Release Schedule

- **Minor releases**: Monthly during active development
- **Patch releases**: As needed for critical fixes
- **Major releases**: As needed for breaking changes

## Feature Requests

Feature requests are tracked in
[GitHub Issues](https://github.com/Shadylukin/Neumann/issues).
For major features, open an issue with the `rfc` label for community
discussion (minimum 2 weeks) before implementation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.
