---
name: neumann-vector
description: Build vector search and embedding pipelines with Neumann. Use when implementing similarity search, RAG retrieval, semantic caching, or working with HNSW indexes.
---

# Neumann Vector Search Guide

## Vector Storage Model

Neumann stores embeddings as key-vector pairs in a dedicated vector engine backed
by an HNSW (Hierarchical Navigable Small World) index for fast approximate nearest
neighbor search.

Key concepts:

- **Embedding**: A string key mapped to a float vector of any dimensionality.
- **Collection**: An optional namespace for grouping related embeddings (e.g., per-tenant, per-domain).
- **HNSW Index**: Must be explicitly built after inserts for fast similarity queries.
- **Distance Metrics**: `COSINE` (default), `EUCLIDEAN`, `DOT_PRODUCT`.

All vectors within a query must have the same dimensionality. Mixed dimensions
cause runtime errors, not parse errors.

Embeddings are stored in `tensor_store`'s embedding slab, which uses sharded
locking -- concurrent writes to different keys do not block each other.

## Embedding Lifecycle

The standard workflow: store embeddings, build the index, then query.

### Store a single embedding

```sql
EMBED STORE 'doc-1' [0.12, -0.34, 0.56, 0.78]
```

Keys must be quoted strings. Unquoted keys with hyphens or colons will fail to parse.

### Store into a collection

```sql
EMBED STORE 'doc-1' [0.12, -0.34, 0.56] IN my_collection
```

### Batch store multiple embeddings

```sql
EMBED BATCH [('doc-1', [0.1, 0.2, 0.3]), ('doc-2', [0.4, 0.5, 0.6]), ('doc-3', [0.7, 0.8, 0.9])]
```

Always prefer `EMBED BATCH` over multiple `EMBED STORE` calls for bulk inserts.

### Build the HNSW index

```sql
EMBED BUILD INDEX
```

This must be called after bulk inserts and before similarity queries. Building
the index is an O(n log n) operation. Do it once after loading, not after
each individual insert.

### Retrieve an embedding by key

```sql
EMBED GET 'doc-1'
```

Returns the vector associated with the key, or an error if not found.

### Delete an embedding

```sql
EMBED DELETE 'doc-1'
```

### List stored embeddings

```sql
SHOW EMBEDDINGS
SHOW EMBEDDINGS LIMIT 50
```

### Count embeddings

```sql
COUNT EMBEDDINGS
```

### Check index status

```sql
SHOW VECTOR INDEX
```

## Similarity Search

The `SIMILAR` command performs approximate nearest neighbor search. It returns
key-score pairs ranked by similarity.

### Search by vector

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10
```

### Search by existing key

```sql
SIMILAR 'doc-1' LIMIT 5
```

Uses the vector already stored under that key as the query vector.

### Specify distance metric

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC COSINE
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC EUCLIDEAN
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC DOT_PRODUCT
```

Default is `COSINE` if omitted.

### Search within a collection

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 IN my_collection
```

### Filtered search

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 WHERE category = 'science'
```

The WHERE clause filters results after the ANN search.

### Cross-engine: vector + graph

Find similar embeddings that are also connected to a specific graph node:

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 CONNECTED TO 'user-42'
```

This intersects ANN results with the graph neighborhood of `user-42`.

### Full syntax summary

```sql
SIMILAR <key-or-vector> [LIMIT n] [METRIC COSINE|EUCLIDEAN|DOT_PRODUCT] [IN collection] [CONNECTED TO 'node-id'] [WHERE expr]
```

Result type: `Similar` (list of key + score pairs).

## Cross-Engine Patterns

### Vector + Graph: similarity-aware traversal

Find neighbors ranked by vector similarity:

```sql
NEIGHBORS 'node-1' BOTH BY SIMILARITY [0.1, 0.2, 0.3] LIMIT 5
```

This returns graph neighbors of `node-1`, re-ranked by cosine similarity to the
given vector.

### Vector + Graph: connected similarity

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 CONNECTED TO 'hub-node'
```

Restricts ANN results to embeddings whose keys correspond to nodes connected
to `hub-node` in the graph.

### Vector + Cache: semantic caching

```sql
CACHE SEMANTIC PUT 'query-text' 'response-text'
CACHE SEMANTIC GET 'similar-query-text'
```

Uses embedding similarity for cache lookups -- returns cached responses for
semantically similar (not just exact-match) queries.

### Unified entities with embeddings

```sql
ENTITY CREATE 'product-1' { name: 'Widget', price: 9.99 } EMBEDDING [0.1, 0.2, 0.3]
ENTITY UPDATE 'product-1' { price: 12.99 } EMBEDDING [0.15, 0.25, 0.35]
```

Stores properties, graph node, and embedding in one atomic operation.

## Performance Tips

1. **Build index after bulk inserts.** Call `EMBED BUILD INDEX` once after loading
   all embeddings, not after each insert. Index building is expensive and
   redundant if more inserts follow immediately.

2. **Use EMBED BATCH for bulk loading.** A single `EMBED BATCH` with 1000 pairs
   is faster than 1000 individual `EMBED STORE` calls due to reduced parsing
   and locking overhead.

3. **Choose the right metric for your data:**
   - `COSINE` -- best for normalized embeddings (most LLM outputs). Measures
     angle between vectors, invariant to magnitude.
   - `EUCLIDEAN` -- best for spatial data where absolute distance matters.
   - `DOT_PRODUCT` -- best when magnitude carries meaning (e.g., popularity-weighted
     embeddings). Fastest to compute but sensitive to vector norms.

4. **Use collections for multi-tenant isolation.** Rather than filtering after
   search, partition embeddings into collections. Each collection maintains
   its own index, so queries only search relevant vectors.

5. **Keep vector dimensions consistent.** All vectors in a collection should
   have the same dimensionality. Mixing dimensions causes runtime errors.

6. **LIMIT is not optional for large datasets.** Without LIMIT, SIMILAR returns
   all embeddings sorted by distance. Always specify LIMIT for production queries.

## Common Mistakes

**Forgetting to build the index:**

```sql
EMBED STORE 'k1' [0.1, 0.2]
SIMILAR [0.1, 0.2] LIMIT 5       -- slow: falls back to brute-force scan
EMBED BUILD INDEX                  -- should come before SIMILAR
SIMILAR [0.1, 0.2] LIMIT 5       -- fast: uses HNSW index
```

**Using SIMILAR TO (wrong keyword):**

```sql
SIMILAR [0.1, 0.2] LIMIT 5       -- correct
SIMILAR TO [0.1, 0.2] LIMIT 5    -- WRONG: no TO keyword in SIMILAR
```

**Using BY instead of METRIC:**

```sql
SIMILAR [0.1, 0.2] LIMIT 5 METRIC COSINE    -- correct
SIMILAR [0.1, 0.2] LIMIT 5 BY COSINE        -- WRONG: use METRIC
```

**Unquoted keys with special characters:**

```sql
EMBED STORE 'doc:1' [0.1, 0.2]    -- correct
EMBED STORE doc:1 [0.1, 0.2]      -- WRONG: colon splits into 3 tokens
EMBED STORE 'my-key' [0.1, 0.2]   -- correct
EMBED STORE my-key [0.1, 0.2]     -- WRONG: hyphen causes parse error
```

**Searching without LIMIT on large datasets:**

```sql
SIMILAR [0.1, 0.2] LIMIT 10       -- correct: bounded result set
SIMILAR [0.1, 0.2]                 -- risky: returns all embeddings
```
