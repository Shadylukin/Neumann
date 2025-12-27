# Neumann Documentation Roadmap

This document outlines the current state of documentation and the plan for user-facing docs.

## Current State Assessment

### Documentation Quality by Module

| Module | Rating | Status | Key Gaps |
|--------|--------|--------|----------|
| tensor-store.md | 4.5/5 | Up-to-date | None |
| relational-engine.md | 4/5 | Good | SIMD optimizations undocumented |
| graph-engine.md | 4/5 | Good | None |
| vector-engine.md | 4/5 | Good | `search_entity_neighbors_by_similarity` missing |
| query-router.md | 4/5 | Good | Blob/Cache init methods incomplete |
| neumann-parser.md | 4/5 | Needs update | BLOB, CACHE, VAULT syntax missing |
| neumann-shell.md | 4/5 | Needs update | BLOB, CACHE, VAULT commands missing |
| tensor-compress.md | 4.5/5 | Up-to-date | None |
| tensor-vault.md | 4.5/5 | Excellent | None |
| tensor-cache.md | 4/5 | Good | TTL tracking details |
| tensor-blob.md | 4/5 | Good | Vector feature examples |

**Overall: 4.2/5 average**

### Critical Gaps

1. **Undocumented Query Syntax** (in neumann-parser.md):
   - VAULT statements (7 operations)
   - CACHE statements (6 operations)
   - BLOB statements (18 operations)
   - BLOBS queries (5 modes)
   - JOIN operations (6 types)
   - Advanced SQL (GROUP BY, HAVING, subqueries)

2. **Missing User-Facing Documentation**:
   - No getting started guide
   - No query syntax quick reference
   - No examples/tutorials
   - No production deployment guide

## Proposed Documentation Structure

```
docs/
  # Existing (Developer Reference)
  architecture.md
  tensor-store.md
  relational-engine.md
  graph-engine.md
  vector-engine.md
  query-router.md
  neumann-parser.md
  neumann-shell.md
  tensor-compress.md
  tensor-vault.md
  tensor-cache.md
  tensor-blob.md
  benchmarks.md

  # New (User-Facing)
  getting-started.md        # 10-minute quick start
  query-reference.md        # Complete syntax reference
  examples/
    basic-crud.md           # Tables, inserts, selects
    graph-modeling.md       # Nodes, edges, traversals
    vector-search.md        # Embeddings, similarity
    blob-storage.md         # File storage patterns
    secret-management.md    # Vault usage
    llm-caching.md          # Cache integration
  production/
    persistence.md          # SAVE, LOAD, WAL
    recovery.md             # Crash recovery
    performance.md          # Tuning guide
```

## Priority 1: Query Reference (docs/query-reference.md)

Complete syntax reference covering all 19 statement types:

### SQL Statements
- SELECT with JOIN, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET
- INSERT with VALUES and subqueries
- UPDATE with WHERE
- DELETE with WHERE
- CREATE/DROP TABLE with constraints
- CREATE/DROP INDEX

### Graph Statements
- NODE CREATE/GET/DELETE/LIST
- EDGE CREATE/GET/DELETE/LIST
- NEIGHBORS with direction and edge type
- PATH with LIMIT
- FIND with WHERE and RETURN

### Vector Statements
- EMBED STORE/GET/DELETE
- SIMILAR with LIMIT and distance metrics

### Vault Statements
- VAULT SET/GET/DELETE/LIST
- VAULT ROTATE
- VAULT GRANT/REVOKE

### Cache Statements
- CACHE INIT/STATS/CLEAR/EVICT
- CACHE GET/PUT

### Blob Statements
- BLOB PUT (inline and FROM file)
- BLOB GET (inline and TO file)
- BLOB DELETE/INFO/VERIFY
- BLOB LINK/UNLINK/LINKS
- BLOB TAG/UNTAG
- BLOB GC/REPAIR/STATS
- BLOB META SET/GET
- BLOBS (list, FOR entity, BY TAG, WHERE TYPE, SIMILAR TO)

## Priority 2: Getting Started Guide (docs/getting-started.md)

### Structure
1. Installation (cargo build)
2. Starting the shell
3. First table (CREATE, INSERT, SELECT)
4. First graph (NODE, EDGE, NEIGHBORS)
5. First embedding (EMBED, SIMILAR)
6. Saving your work (SAVE)
7. Next steps

### Estimated length: 500-800 words

## Priority 3: Examples Directory

### basic-crud.md
- Creating tables with various column types
- Inserting data
- Querying with WHERE conditions
- Updating and deleting
- Using indexes

### graph-modeling.md
- When to use nodes vs tables
- Creating node hierarchies
- Bidirectional relationships
- Path finding examples
- FIND queries

### vector-search.md
- Storing embeddings
- K-NN search
- Distance metrics (cosine, euclidean, dot product)
- Combining with graph traversals

### blob-storage.md
- Storing files
- Content-addressable deduplication
- Linking to entities
- Tagging and querying
- Garbage collection

### secret-management.md
- Initializing vault
- Storing secrets
- Access control with identities
- Grant/revoke permissions
- Secret rotation

### llm-caching.md
- Initializing cache
- Semantic vs exact matching
- Token counting
- Cost savings
- Eviction strategies

## Priority 4: Production Guide

### persistence.md
- Snapshot save/load
- Compressed snapshots
- Write-ahead log (WAL)
- Backup strategies

### recovery.md
- Crash recovery with WAL
- Snapshot + WAL workflow
- Troubleshooting failed loads

### performance.md
- Tiered storage for large datasets
- Compression trade-offs
- Index usage
- Concurrent access patterns

## Immediate Actions

1. **Update neumann-parser.md** with VAULT, CACHE, BLOB syntax
2. **Update neumann-shell.md** with VAULT, CACHE, BLOB commands
3. **Create docs/query-reference.md** as a comprehensive cheat sheet
4. **Create docs/getting-started.md** for new users

## Timeline Estimate

| Task | Effort |
|------|--------|
| Update parser docs | 1-2 hours |
| Update shell docs | 1-2 hours |
| Query reference | 2-3 hours |
| Getting started | 1-2 hours |
| Examples (6 files) | 4-6 hours |
| Production guide | 2-3 hours |

**Total: ~15-20 hours of documentation work**
