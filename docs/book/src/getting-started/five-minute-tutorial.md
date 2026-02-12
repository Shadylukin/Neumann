# Five-Minute Tutorial: Build a Mini RAG System

This tutorial builds a retrieval-augmented generation (RAG) knowledge base using
all three engines. By the end, you will have documents stored relationally, linked
in a graph, searchable by embeddings, and protected with vault secrets.

## Prerequisites

Start the shell with persistence:

```bash
neumann --wal-dir ./rag-data
```

## Step 1: Create the Document Store

Use a relational table for structured document metadata:

```sql
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    author TEXT,
    created TEXT
);

INSERT INTO documents VALUES (1, 'Intro to Neural Networks', 'ml', 'Alice', '2024-01-15');
INSERT INTO documents VALUES (2, 'Transformer Architecture', 'ml', 'Bob', '2024-02-20');
INSERT INTO documents VALUES (3, 'Database Indexing', 'systems', 'Carol', '2024-03-10');
INSERT INTO documents VALUES (4, 'Vector Search at Scale', 'systems', 'Alice', '2024-04-05');
INSERT INTO documents VALUES (5, 'Fine-Tuning LLMs', 'ml', 'Dave', '2024-05-12');
INSERT INTO documents VALUES (6, 'Consensus Protocols', 'distributed', 'Eve', '2024-06-01');
```

Verify:

```sql
SELECT * FROM documents ORDER BY id;
SELECT category, COUNT(*) FROM documents GROUP BY category;
```

## Step 2: Add Graph Relationships

Create nodes for documents and topics, then link them:

```sql
NODE CREATE document { title: 'Intro to Neural Networks', doc_id: 1 }
NODE CREATE document { title: 'Transformer Architecture', doc_id: 2 }
NODE CREATE document { title: 'Database Indexing', doc_id: 3 }
NODE CREATE document { title: 'Vector Search at Scale', doc_id: 4 }
NODE CREATE document { title: 'Fine-Tuning LLMs', doc_id: 5 }
NODE CREATE document { title: 'Consensus Protocols', doc_id: 6 }

NODE CREATE topic { name: 'machine-learning' }
NODE CREATE topic { name: 'systems' }
NODE CREATE topic { name: 'distributed-systems' }
```

List nodes to get IDs:

```sql
NODE LIST document
NODE LIST topic
```

Create edges linking documents to topics (use actual IDs from NODE LIST):

```sql
-- Documents 1, 2, 5 cover machine-learning
-- Documents 3, 4 cover systems
-- Documents 4, 6 cover distributed-systems
-- Document 2 references document 1
-- Document 5 references documents 1 and 2
```

Create citation relationships between documents:

```sql
EDGE CREATE 'doc2-id' -> 'doc1-id' : cites
EDGE CREATE 'doc5-id' -> 'doc1-id' : cites
EDGE CREATE 'doc5-id' -> 'doc2-id' : cites
```

Check the graph:

```sql
NEIGHBORS 'doc1-id' INCOMING : cites
PATH SHORTEST 'doc5-id' TO 'doc1-id'
```

## Step 3: Store Embeddings

Each document gets a vector representing its content. In a real system, you would
generate these with an embedding model (e.g., OpenAI, Cohere, or a local model).
Here we use hand-crafted 6-dimensional vectors:

```sql
-- [neural-nets, transformers, databases, vectors, llms, distributed]
EMBED STORE 'doc-1' [0.9, 0.3, 0.1, 0.2, 0.2, 0.1]
EMBED STORE 'doc-2' [0.7, 0.95, 0.1, 0.1, 0.4, 0.1]
EMBED STORE 'doc-3' [0.1, 0.1, 0.95, 0.3, 0.0, 0.2]
EMBED STORE 'doc-4' [0.2, 0.1, 0.5, 0.9, 0.1, 0.4]
EMBED STORE 'doc-5' [0.6, 0.5, 0.1, 0.1, 0.95, 0.1]
EMBED STORE 'doc-6' [0.1, 0.1, 0.3, 0.2, 0.1, 0.9]
```

Search by similarity:

```sql
-- Find documents similar to "Intro to Neural Networks"
SIMILAR 'doc-1' LIMIT 3

-- Search with a custom query vector (someone asking about transformers + LLMs)
SIMILAR [0.5, 0.8, 0.1, 0.1, 0.7, 0.1] LIMIT 3 METRIC COSINE
```

## Step 4: Graph-Aware Semantic Search

Combine vector similarity with graph connectivity. Find documents similar to a
query vector that are also connected to a specific topic node:

```sql
SIMILAR [0.8, 0.6, 0.1, 0.1, 0.5, 0.1] LIMIT 3 CONNECTED TO 'ml-topic-id'
```

This is the core RAG pattern: retrieve relevant documents using embedding
similarity, then filter by graph relationships for context-aware results.

## Step 5: Protect API Keys with Vault

Store the embedding API key securely:

```sql
VAULT SET 'openai_api_key' 'sk-proj-abc123...'
VAULT SET 'cohere_api_key' 'co-xyz789...'
VAULT GRANT 'alice' ON 'openai_api_key'
```

Retrieve when needed:

```sql
VAULT GET 'openai_api_key'
VAULT LIST
```

## Step 6: Cache LLM Responses

Initialize the cache and store responses to avoid repeated API calls:

```sql
CACHE INIT

CACHE PUT 'what are transformers?' 'Transformers are a neural network architecture based on self-attention mechanisms...'

CACHE SEMANTIC PUT 'explain attention mechanism' 'The attention mechanism allows models to focus on relevant parts of the input...' EMBEDDING [0.6, 0.9, 0.1, 0.1, 0.3, 0.1]
```

On subsequent queries, check the cache first:

```sql
CACHE GET 'what are transformers?'
CACHE SEMANTIC GET 'how does attention work?' THRESHOLD 0.8
```

## Step 7: Checkpoint

Save your work:

```sql
CHECKPOINT 'rag-setup-complete'
CHECKPOINTS
```

## What You Built

You now have a working RAG knowledge base with:

1. **Structured metadata** in a relational table (searchable with SQL)
2. **Semantic relationships** in a graph (topics, citations, authorship)
3. **Vector embeddings** for similarity search
4. **Graph-aware retrieval** combining similarity and structure
5. **Encrypted secrets** for API key management
6. **Response caching** to reduce LLM API costs
7. **Checkpoint** for safe rollback

## Next Steps

- [Use Cases](use-cases.md) -- More application patterns
- [Query Language Reference](../reference/query-language.md) -- All available commands
- [Python SDK](../sdks/python-quickstart.md) -- Build this in Python
- [Architecture Overview](../architecture/overview.md) -- How the engines work together
