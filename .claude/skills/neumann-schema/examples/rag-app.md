# RAG Application Schema

A retrieval-augmented generation (RAG) application that stores documents,
splits them into chunks, embeds chunks for semantic retrieval, and caches
LLM responses.

## Engines Used

- **Relational**: Document metadata and chunk tracking
- **Graph**: Document-collection relationships and cross-references
- **Vector**: Chunk embeddings for semantic retrieval
- **Cache**: LLM response caching by semantic similarity

## Schema

### Relational Tables

```sql
CREATE TABLE collections (name STRING, description STRING, created_at INT)
CREATE TABLE documents (title STRING, source STRING, collection_id INT, chunk_count INT, created_at INT)
CREATE TABLE chunks (doc_id INT, chunk_index INT, content STRING, token_count INT)
CREATE INDEX idx_chunks_doc ON chunks (doc_id)
```

### Graph Structure

```sql
-- Collection nodes
NODE CREATE collection { name: 'engineering-docs', owner: 'team-a' }

-- Document nodes linked to collections
NODE CREATE document { title: 'API Guide', doc_id: '1' }
EDGE CREATE 2 -> 1 : belongs_to

-- Cross-reference edges between documents
EDGE CREATE 2 -> 3 : references
```

### Vector Embeddings

Key format: `doc-{doc_id}-chunk-{chunk_index}`

```sql
-- Store chunk embeddings (384-dim for all-MiniLM-L6-v2)
EMBED STORE 'doc-1-chunk-0' [0.12, -0.34, 0.56, ...]
EMBED STORE 'doc-1-chunk-1' [0.08, -0.21, 0.44, ...]
EMBED STORE 'doc-2-chunk-0' [0.15, -0.28, 0.61, ...]
```

## Retrieval Pattern

### Step 1: Semantic Search

```sql
SIMILAR TO [0.12, -0.34, ...] LIMIT 5
-- Returns: [("doc-1-chunk-2", 0.94), ("doc-3-chunk-0", 0.88), ...]
```

### Step 2: Expand Context

Parse the chunk keys to find adjacent chunks and related documents.

```sql
-- Get surrounding chunks for context
SELECT * FROM chunks WHERE doc_id = 1 AND chunk_index BETWEEN 1 AND 3

-- Find related documents via graph
NEIGHBORS 2 OUTGOING WHERE label = 'references' DEPTH 1
```

### Step 3: Cache the Response

```sql
-- Check cache first (semantic match)
CACHE SEMANTIC GET 'What are the API rate limits?' THRESHOLD 0.9

-- If miss, call LLM and cache the response
CACHE PUT 'What are the API rate limits?' 'The rate limit is 1000 req/min...'
```

## Ingestion Pipeline

```python
# Python SDK example
from neumann import NeumannClient

with NeumannClient.connect("localhost:9200") as client:
    # 1. Create document record
    client.query("INSERT INTO documents (title, source) VALUES ('Guide', 'upload')")

    # 2. Split into chunks and store
    for i, chunk in enumerate(chunks):
        client.query(f"INSERT INTO chunks (doc_id, chunk_index, content) VALUES (1, {i}, '{chunk}')")

    # 3. Embed each chunk
    for i, embedding in enumerate(embeddings):
        vec_str = ", ".join(str(v) for v in embedding)
        client.query(f"EMBED STORE 'doc-1-chunk-{i}' [{vec_str}]")

    # 4. Create graph node
    client.query("NODE CREATE document { title: 'Guide', doc_id: '1' }")
```
