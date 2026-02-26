# Knowledge Graph Schema

A knowledge graph with typed entities, relationships, and entity embeddings
for combined structural and semantic queries.

## Engines Used

- **Graph**: Entities as nodes, relationships as typed edges
- **Vector**: Entity embeddings for similarity and resolution
- **Relational**: Metadata tables for statistics and configuration

## Schema

### Graph Structure

Entities are nodes with a label indicating type. Properties are schemaless.

```sql
-- People
NODE CREATE person { name: 'Alice Chen', title: 'CTO', company: 'Acme' }
NODE CREATE person { name: 'Bob Park', title: 'VP Engineering', company: 'Acme' }

-- Companies
NODE CREATE company { name: 'Acme Corp', industry: 'tech', founded: '2015' }
NODE CREATE company { name: 'Beta Inc', industry: 'finance', founded: '2018' }

-- Products
NODE CREATE product { name: 'WidgetPro', category: 'SaaS', version: '3.0' }
```

### Typed Relationships

Edges carry a label (relationship type) and optional properties.

```sql
-- Employment
EDGE CREATE 1 -> 3 : works_at { since: '2020', role: 'CTO' }
EDGE CREATE 2 -> 3 : works_at { since: '2021', role: 'VP' }

-- Reporting
EDGE CREATE 2 -> 1 : reports_to

-- Business relationships
EDGE CREATE 3 -> 4 : partner_of { since: '2022' }
EDGE CREATE 3 -> 5 : produces
EDGE CREATE 4 -> 5 : uses { license: 'enterprise' }
```

### Entity Embeddings

Embed entities for similarity search and entity resolution.

```sql
-- Embed using entity description or concatenated properties
EMBED STORE 'person:1' [0.12, -0.34, ...]  -- Alice Chen, CTO at Acme
EMBED STORE 'person:2' [0.15, -0.29, ...]  -- Bob Park, VP at Acme
EMBED STORE 'company:3' [0.41, -0.18, ...]  -- Acme Corp, tech
EMBED STORE 'product:5' [0.22, -0.55, ...]  -- WidgetPro, SaaS
```

### Metadata Tables

```sql
CREATE TABLE entity_stats (entity_type STRING, count INT, last_updated INT)
CREATE TABLE relationship_types (name STRING, source_type STRING, target_type STRING, count INT)
```

## Query Patterns

### Direct Neighbors

```sql
-- Who does Alice work with?
NEIGHBORS 1 OUTGOING WHERE label = 'works_at'

-- Who reports to Alice?
NEIGHBORS 1 INCOMING WHERE label = 'reports_to'

-- All connections within 2 hops
NEIGHBORS 1 BOTH DEPTH 2
```

### Shortest Path

```sql
-- How are Alice and Beta Inc connected?
PATH SHORTEST FROM 1 TO 4
-- Returns: [1] -works_at-> [3] -partner_of-> [4]
```

### Graph Algorithms

```sql
-- Influence ranking
PAGERANK

-- Find communities
LOUVAIN

-- Key connectors
CENTRALITY BETWEENNESS
```

### Similarity Search

```sql
-- Find entities similar to "Alice Chen, CTO"
SIMILAR TO [0.12, -0.34, ...] LIMIT 5

-- Entity resolution: find potential duplicates
SIMILAR TO [0.12, -0.34, ...] LIMIT 3 THRESHOLD 0.95
```

### Combined Queries

```python
from neumann import NeumannClient

async def find_related_experts(client, topic_embedding):
    # 1. Find entities similar to the topic
    similar = await client.query(f"SIMILAR TO [{vec}] LIMIT 10")

    # 2. For each person found, get their company and connections
    for item in similar.similar_items:
        if item.key.startswith("person:"):
            node_id = item.key.split(":")[1]
            # Get company
            company = await client.query(
                f"NEIGHBORS {node_id} OUTGOING WHERE label = 'works_at'"
            )
            # Get collaborators
            peers = await client.query(
                f"NEIGHBORS {node_id} BOTH WHERE label = 'collaborates_with'"
            )
    return experts
```

## Entity Resolution Pattern

Use vector similarity to detect duplicate entities, then merge via graph.

```sql
-- Find near-duplicates
SIMILAR TO [embedding-of-new-entity] LIMIT 5 THRESHOLD 0.92

-- If match found, link as same_as rather than creating duplicate
EDGE CREATE new_node -> existing_node : same_as { confidence: '0.95' }
```
