# AI Agent Memory Schema

An AI agent that maintains long-term memory with structured storage,
associative graph links, and semantic recall.

## Engines Used

- **Relational**: Structured memory records with type, importance, timestamps
- **Graph**: Associative links between memories and sessions
- **Vector**: Semantic embeddings for similarity-based recall
- **Vault**: Encrypted storage for sensitive user information

## Schema

### Relational Tables

```sql
CREATE TABLE memories (content STRING, memory_type STRING, importance FLOAT, source STRING, created_at INT)
CREATE TABLE sessions (started_at INT, ended_at INT, topic STRING, message_count INT)
CREATE TABLE facts (subject STRING, predicate STRING, object STRING, confidence FLOAT, memory_id INT)
CREATE INDEX idx_memories_type ON memories (memory_type)
CREATE INDEX idx_facts_subject ON facts (subject)
```

Memory types: `observation`, `reflection`, `fact`, `preference`, `instruction`.

### Graph Structure

```sql
-- Session nodes
NODE CREATE session { session_id: 's1', topic: 'project planning' }

-- Memory nodes
NODE CREATE memory { memory_id: '1', importance: '0.8', memory_type: 'observation' }
NODE CREATE memory { memory_id: '2', importance: '0.9', memory_type: 'fact' }

-- Edges: memory observed in session
EDGE CREATE 2 -> 1 : observed_in
EDGE CREATE 3 -> 1 : observed_in

-- Edges: associative links between memories
EDGE CREATE 2 -> 3 : related_to { strength: '0.7' }
```

### Vector Embeddings

Key format: `memory-{memory_id}`

```sql
EMBED STORE 'memory-1' [0.23, -0.45, 0.12, ...]
EMBED STORE 'memory-2' [0.18, -0.31, 0.44, ...]
```

### Vault (Sensitive Data)

```sql
VAULT STORE 'user:alice:api-key' 'sk-abc123...' AS agent
VAULT STORE 'user:alice:preferences' '{"timezone": "UTC", ...}' AS agent
```

## Recall Patterns

### Semantic Recall -- "What do I know about X?"

```sql
-- Find semantically similar memories
SIMILAR TO [0.23, -0.45, ...] LIMIT 10
-- Returns memories ranked by relevance to the query embedding
```

### Structured Recall -- "What facts about user preferences?"

```sql
SELECT * FROM memories WHERE memory_type = 'preference' ORDER BY importance DESC
SELECT * FROM facts WHERE subject = 'user' AND predicate = 'prefers'
```

### Associative Recall -- "What is connected to this memory?"

```sql
-- Find related memories via graph edges
NEIGHBORS 42 OUTGOING WHERE label = 'related_to' DEPTH 2

-- Find all memories from a session
NEIGHBORS 10 INCOMING WHERE label = 'observed_in'
```

### Combined Recall

Typical agent recall combines all three strategies:

```python
from neumann import NeumannClient

async def recall(client, query_embedding, context):
    # 1. Semantic: find relevant memories
    semantic = await client.query(f"SIMILAR TO [{vec}] LIMIT 5")

    # 2. Structured: get recent high-importance memories
    recent = await client.query(
        "SELECT * FROM memories WHERE importance > 0.7 ORDER BY created_at DESC LIMIT 5"
    )

    # 3. Associative: expand from semantic hits via graph
    for item in semantic.similar_items:
        memory_id = item.key.split("-")[1]
        related = await client.query(f"NEIGHBORS {memory_id} OUTGOING DEPTH 1")

    # Merge, deduplicate, rank by combined relevance + importance + recency
    return merged_memories
```

## Memory Consolidation

Periodically merge related memories and update importance scores.

```sql
-- Find clusters of related memories
LOUVAIN

-- Update importance based on access frequency
UPDATE memories SET importance = 0.9 WHERE id = 42

-- Archive old low-importance memories
DELETE FROM memories WHERE importance < 0.2 AND created_at < 1700000000
```
