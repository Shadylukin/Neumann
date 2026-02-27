# Caching LLM Responses

Set up semantic caching for LLM API responses. By the end you will have
a cache that matches semantically similar queries to avoid redundant API
calls.

## Prerequisites

- Neumann installed ([Installation](../how-to/installation.md))
- A running Neumann shell

## Step 1: Start the Shell

```bash
neumann --wal-dir ./cache-data
```

## Step 2: Store an Exact Cache Entry

Cache an LLM response with its exact prompt:

```sql
CACHE PUT 'What is machine learning?' 'Machine learning is a subset of artificial intelligence that enables systems to learn from data...'
```

Retrieve it:

```sql
CACHE GET 'What is machine learning?'
```

You should see the cached response.

## Step 3: Store a Semantic Cache Entry

Store a response with an embedding vector for semantic matching:

```sql
CACHE SEMANTIC PUT 'What is machine learning?' 'Machine learning is a subset of artificial intelligence...' EMBEDDING [0.9, 0.1, 0.2, 0.05, 0.8, 0.15, 0.3, 0.02]
```

## Step 4: Query by Semantic Similarity

Try a semantically similar but differently worded query:

```sql
CACHE SEMANTIC GET 'Explain ML to me' THRESHOLD 0.8
```

If the embedding of "Explain ML to me" is similar enough to the stored
embedding (above the 0.8 threshold), you get a cache hit and the stored
response is returned.

## Step 5: Store Multiple Cache Entries

Build up a cache with several topics:

```sql
CACHE SEMANTIC PUT 'How do neural networks work?' 'Neural networks consist of layers of interconnected nodes...' EMBEDDING [0.85, 0.2, 0.3, 0.1, 0.7, 0.25, 0.15, 0.05]

CACHE SEMANTIC PUT 'What is Docker?' 'Docker is a platform for containerizing applications...' EMBEDDING [0.1, 0.8, 0.05, 0.7, 0.2, 0.6, 0.1, 0.3]

CACHE SEMANTIC PUT 'Explain REST APIs' 'REST APIs use HTTP methods to perform CRUD operations...' EMBEDDING [0.2, 0.6, 0.1, 0.8, 0.15, 0.5, 0.3, 0.2]
```

## Step 6: Check Cache Stats

```sql
CACHE LIST
```

You should see all cached entries.

## Step 7: Evict Stale Entries

Remove a specific entry:

```sql
CACHE DELETE 'What is Docker?'
```

Verify it was removed:

```sql
CACHE GET 'What is Docker?'
```

## Step 8: Combine with Other Engines

Store document metadata alongside the cache for richer context:

```sql
CREATE TABLE llm_usage (
    id INT PRIMARY KEY,
    prompt TEXT,
    model TEXT,
    tokens_used INT,
    cached INT
);
INSERT INTO llm_usage VALUES (1, 'What is machine learning?', 'gpt-4', 150, 0);
INSERT INTO llm_usage VALUES (2, 'Explain ML to me', 'gpt-4', 0, 1);
```

Track which queries hit the cache:

```sql
SELECT * FROM llm_usage WHERE cached = 1;
```

## Verification

You should have:

- Exact cache entries stored and retrieved (CACHE PUT/GET)
- Semantic cache entries with embeddings
- Semantic similarity matching returning cached responses
- Cache deletion working
- Usage tracking in a relational table

## Next Steps

- [Configure Semantic Cache](../how-to/configure-semantic-cache.md) --
  tune cache parameters
- [Vector Search with Filtering](vector-search-filtering.md) -- more
  embedding search patterns
- [Use Cases](use-cases.md) -- more application patterns
