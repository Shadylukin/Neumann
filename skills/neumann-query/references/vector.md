# Vector Engine -- Quick Reference

## Embedding Storage

```sql
-- Store a single embedding
EMBED STORE 'key' [0.1, 0.2, 0.3, 0.4]

-- Store in a named collection
EMBED STORE 'key' [0.1, 0.2, 0.3] IN my_collection

-- Retrieve an embedding
EMBED GET 'key'

-- Delete an embedding
EMBED DELETE 'key'

-- Build HNSW index (required for efficient similarity search)
EMBED BUILD INDEX

-- Batch store multiple embeddings
EMBED BATCH [('key1', [0.1, 0.2]), ('key2', [0.3, 0.4]), ('key3', [0.5, 0.6])]
```

Key must be a quoted string. Vector is enclosed in square brackets with comma-separated floats.

## Similarity Search

```sql
-- Search by vector
SIMILAR [0.1, 0.2, 0.3] LIMIT 10

-- Search by vector with explicit metric
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC COSINE
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC EUCLIDEAN
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC DOT_PRODUCT

-- Search by existing key (finds neighbors of that key's vector)
SIMILAR 'existing-key' LIMIT 5

-- Search in a specific collection
SIMILAR [0.1, 0.2, 0.3] LIMIT 5 IN my_collection

-- Filtered similarity search
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 WHERE category = 'science'

-- Cross-engine: similar + graph connectivity
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 CONNECTED TO 'node-id'
```

**Available metrics:**

| Metric | Keyword | Description |
|--------|---------|-------------|
| Cosine similarity | `COSINE` | Default. Normalized dot product. |
| Euclidean distance | `EUCLIDEAN` | L2 distance (lower = more similar). |
| Dot product | `DOT_PRODUCT` | Raw dot product (higher = more similar). |

**Result format:** Returns `Similar` result -- a list of `(key, score)` pairs sorted by relevance.

## Inspection Commands

```sql
-- List stored embeddings
SHOW EMBEDDINGS
SHOW EMBEDDINGS LIMIT 100

-- Show HNSW index status
SHOW VECTOR INDEX

-- Count total embeddings
COUNT EMBEDDINGS
```

## Common Gotchas

1. Always quote embedding keys: `EMBED STORE 'my-key'` not `EMBED STORE my-key`
2. Keys with colons must be quoted: `EMBED STORE 'doc:123'` not `EMBED STORE doc:123`
3. No `TO` keyword in SIMILAR: `SIMILAR [vec]` not `SIMILAR TO [vec]`
4. No `BY` keyword for metric: `METRIC COSINE` not `BY COSINE`
5. `LIMIT` comes before `METRIC`: `SIMILAR [vec] LIMIT 5 METRIC COSINE`
6. Build index before searching large collections: `EMBED BUILD INDEX`
