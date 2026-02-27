# Vector Search with Metadata Filtering

Build an end-to-end filtered vector search system. You will store document
embeddings with metadata, then run similarity searches filtered by
attributes.

## Prerequisites

- Neumann installed ([Installation](../how-to/installation.md))
- A running Neumann shell

## Step 1: Start the Shell

```bash
neumann --wal-dir ./vector-search-data
```

## Step 2: Create a Metadata Table

Store document metadata in a relational table:

```sql
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    author TEXT,
    year INT
);
```

Insert sample documents:

```sql
INSERT INTO documents VALUES (1, 'Intro to Neural Networks', 'ml', 'Alice', 2024);
INSERT INTO documents VALUES (2, 'CSS Grid Layout', 'web', 'Bob', 2023);
INSERT INTO documents VALUES (3, 'Transformer Architecture', 'ml', 'Alice', 2025);
INSERT INTO documents VALUES (4, 'React Hooks Guide', 'web', 'Carol', 2024);
INSERT INTO documents VALUES (5, 'Reinforcement Learning', 'ml', 'Dave', 2025);
```

## Step 3: Store Embeddings

Store an embedding vector for each document. In a real system these would
come from an embedding model; here we use small example vectors:

```sql
EMBED STORE 'doc:1' [0.9, 0.1, 0.2, 0.0]
EMBED STORE 'doc:2' [0.0, 0.8, 0.1, 0.7]
EMBED STORE 'doc:3' [0.85, 0.15, 0.3, 0.05]
EMBED STORE 'doc:4' [0.05, 0.75, 0.15, 0.8]
EMBED STORE 'doc:5' [0.8, 0.2, 0.4, 0.1]
```

Verify storage:

```sql
EMBED GET 'doc:1'
```

You should see the stored vector.

## Step 4: Run Basic Similarity Search

Find documents similar to a query vector (representing "deep learning"):

```sql
SIMILAR [0.88, 0.12, 0.25, 0.02] LIMIT 3 METRIC COSINE
```

You should see doc:1, doc:3, and doc:5 as the top matches (ML documents).

## Step 5: Try Different Distance Metrics

```sql
SIMILAR [0.88, 0.12, 0.25, 0.02] LIMIT 3 METRIC EUCLIDEAN
SIMILAR [0.88, 0.12, 0.25, 0.02] LIMIT 3 METRIC DOT
```

Compare how results change with different metrics.

## Step 6: Filtered Search

Combine vector similarity with metadata filters. Find ML documents only:

```sql
SELECT * FROM documents WHERE category = 'ml';
```

Then use the IDs to verify your similarity results match the ML category.

## Step 7: Graph-Aware Search

Connect documents with a graph to find related content:

```sql
NODE CREATE collection { name: 'ml-papers' }
EDGE CREATE 'doc:1' -> 'node:1' : belongs_to
EDGE CREATE 'doc:3' -> 'node:1' : belongs_to
EDGE CREATE 'doc:5' -> 'node:1' : belongs_to
```

## Step 8: Cache Search Results

Cache frequently-used search results to avoid recomputation:

```sql
CACHE PUT 'deep-learning-query' '[0.88, 0.12, 0.25, 0.02]'
CACHE GET 'deep-learning-query'
```

## Verification

You should have:

- 5 documents in the relational table
- 5 embedding vectors stored
- Similarity search returning ML docs for an ML query vector
- Graph connections linking ML docs to a collection
- A cached query result

## Next Steps

- [Embeddings Search](../how-to/embeddings-search.md) -- more search
  options
- [Building a Knowledge Graph](knowledge-graph.md) -- combine with graph
  traversals
- [Query Language Reference](../reference/query-language.md) -- full
  command list
