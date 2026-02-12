# Use Cases

Neumann is designed for applications that need two or more of: structured data,
relationships, and semantic search. Here are concrete examples with query patterns.

---

## RAG Application

**Problem**: Build a retrieval-augmented generation system that retrieves relevant
context for an LLM.

**Why Neumann**: Traditional RAG uses a vector store for retrieval. Neumann adds
graph relationships (document structure, citations, topics) and relational metadata
(authors, dates, permissions) to improve retrieval quality.

### Schema

```sql
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    created TEXT,
    chunk_count INT
);
```

```sql
NODE CREATE collection { name: 'engineering-docs' }
-- For each document:
NODE CREATE document { title: 'Design Doc: Auth System', doc_id: 1 }
-- Link to collection:
EDGE CREATE 'doc-node-id' -> 'collection-id' : belongs_to
-- Link related documents:
EDGE CREATE 'doc-a-id' -> 'doc-b-id' : references
```

### Indexing

```sql
-- Store chunk embeddings (one per document chunk)
EMBED STORE 'doc-1-chunk-0' [0.1, 0.2, ...]
EMBED STORE 'doc-1-chunk-1' [0.3, 0.1, ...]
```

### Retrieval

```sql
-- Basic: find similar chunks
SIMILAR [0.2, 0.3, ...] LIMIT 10 METRIC COSINE

-- Graph-aware: find chunks connected to a specific collection
SIMILAR [0.2, 0.3, ...] LIMIT 10 CONNECTED TO 'collection-id'

-- Check cache before calling LLM
CACHE SEMANTIC GET 'how does auth work?' THRESHOLD 0.85
```

### Store LLM response

```sql
CACHE SEMANTIC PUT 'how does auth work?' 'The auth system uses JWT tokens...' EMBEDDING [0.2, 0.3, ...]
```

---

## Agent Memory

**Problem**: Give an AI agent persistent memory that supports both exact recall
and semantic search, with conversation structure.

**Why Neumann**: Agents need to recall specific facts (relational), navigate
conversation history (graph), and find semantically related memories (vector).

### Schema

```sql
CREATE TABLE memories (
    id INT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT,
    importance FLOAT,
    created TEXT
);
```

```sql
-- Create session nodes
NODE CREATE session { name: 'session-2024-01-15', user: 'alice' }

-- Create memory nodes
NODE CREATE memory { content: 'User prefers dark mode', type: 'preference' }

-- Link memories to sessions
EDGE CREATE 'memory-id' -> 'session-id' : observed_in

-- Link related memories
EDGE CREATE 'memory-a' -> 'memory-b' : related_to
```

### Store a Memory

```sql
INSERT INTO memories VALUES (1, 'User prefers dark mode', 'preference', 0.8, '2024-01-15');
NODE CREATE memory { content: 'User prefers dark mode', memory_id: 1 }
EMBED STORE 'memory-1' [0.1, 0.3, ...]
```

### Recall

```sql
-- Semantic recall: find memories similar to current context
SIMILAR [0.2, 0.3, ...] LIMIT 5

-- Structured recall: get high-importance memories
SELECT * FROM memories WHERE importance > 0.7 ORDER BY importance DESC LIMIT 10

-- Graph recall: get memories from a specific session
NEIGHBORS 'session-id' INCOMING : observed_in
```

---

## Knowledge Graph

**Problem**: Build a knowledge graph of entities with properties, relationships,
and similarity search.

**Why Neumann**: Knowledge graphs need rich entity properties (relational), typed
relationships (graph), and entity similarity (vector) in a single queryable system.

### Build the Graph

```sql
-- Create entity types
NODE CREATE company { name: 'Acme Corp', industry: 'Technology', founded: 2010 }
NODE CREATE person { name: 'Jane Smith', title: 'CEO' }
NODE CREATE product { name: 'Acme Cloud', category: 'IaaS' }

-- Create relationships
EDGE CREATE 'jane-id' -> 'acme-id' : works_at { role: 'CEO', since: '2018' }
EDGE CREATE 'acme-id' -> 'product-id' : produces
```

### Add Embeddings

```sql
-- Embed entity descriptions for similarity search
EMBED STORE 'acme-corp' [0.8, 0.3, 0.1, ...]
EMBED STORE 'jane-smith' [0.2, 0.7, 0.5, ...]
EMBED STORE 'acme-cloud' [0.9, 0.4, 0.2, ...]
```

### Query

```sql
-- Find entities similar to a description
SIMILAR [0.7, 0.3, 0.2, ...] LIMIT 5

-- Discover connections
NEIGHBORS 'acme-id' BOTH
PATH SHORTEST 'jane-id' TO 'product-id'

-- Graph analytics
PAGERANK
LOUVAIN
BETWEENNESS
```

---

## Access-Controlled Search

**Problem**: Build a search system where different users see different results
based on permissions.

**Why Neumann**: Vault stores access tokens, graph models the permission hierarchy,
and vector search handles the retrieval. No external auth system needed.

### Setup Permissions

```sql
-- Store API keys securely
VAULT SET 'admin_key' 'ak-admin-secret'
VAULT SET 'user_key' 'ak-user-secret'

-- Grant access based on roles
VAULT GRANT 'alice' ON 'admin_key'
VAULT GRANT 'bob' ON 'user_key'

-- Build permission graph
NODE CREATE role { name: 'admin' }
NODE CREATE role { name: 'viewer' }
NODE CREATE resource { name: 'confidential-docs' }

EDGE CREATE 'admin-role-id' -> 'resource-id' : can_access
```

### Query with Access Check

```sql
-- Find documents similar to query
SIMILAR [0.3, 0.5, ...] LIMIT 10

-- Verify access through graph
NEIGHBORS 'admin-role-id' OUTGOING : can_access

-- Rotate keys periodically
VAULT ROTATE 'admin_key' 'ak-new-admin-secret'
```

---

## Common Patterns

### Checkpoint Before Migrations

```sql
CHECKPOINT 'before-schema-v2'
-- Run migration...
-- If something goes wrong:
ROLLBACK TO 'before-schema-v2'
```

### Blob Attachments

```sql
-- Upload a file and link it to an entity
BLOB INIT
BLOB PUT 'report.pdf' FROM '/tmp/report.pdf' TAG 'quarterly' LINK 'entity-id'

-- Find all blobs for an entity
BLOBS FOR 'entity-id'

-- Find blobs by tag
BLOBS BY TAG 'quarterly'
```

### Chain for Audit Trail

```sql
-- Start a chain transaction for auditable operations
BEGIN CHAIN TRANSACTION
INSERT INTO audit_log VALUES (1, 'user_created', 'alice', '2024-01-15');
COMMIT CHAIN

-- Verify integrity
CHAIN VERIFY
CHAIN HISTORY 'audit_log/1'
```
