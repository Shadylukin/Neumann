---
name: neumann-graph
description: Model and query graph data in Neumann. Use when creating nodes and edges, traversing relationships, finding paths, running graph algorithms, or using Cypher-style queries.
---

# Neumann Graph Engine Guide

## Graph Model

Neumann implements a directed property graph stored in `tensor_store`'s graph
tensor slab. Key concepts:

- **Nodes** have a label (type) and a property map (`{ key: value }`).
- **Edges** are directed (from -> to), have a type label, and a property map.
- **Node/edge IDs** are auto-generated strings returned on creation.
- **Direction keywords** are `OUTGOING`, `INCOMING`, `BOTH`. Never use `OUT`, `IN`, or `UNDIRECTED`.

The graph slab uses sharded locking -- concurrent reads and writes to different
nodes/edges proceed in parallel.

## CRUD Operations

### Nodes

**Create a node with label and properties:**

```sql
NODE CREATE person { name: 'Alice', age: 30, role: 'engineer' }
```

Returns: the auto-generated node ID (e.g., `node-a1b2c3`).

**Get a node by ID:**

```sql
NODE GET 'node-a1b2c3'
```

**Delete a node:**

```sql
NODE DELETE 'node-a1b2c3'
```

**List nodes (optional label filter, limit, offset):**

```sql
NODE LIST
NODE LIST person
NODE LIST person LIMIT 10
NODE LIST person LIMIT 10 OFFSET 20
```

### Edges

**Create an edge between two nodes:**

```sql
EDGE CREATE 'node-1' -> 'node-2' : knows { since: 2024, weight: 0.9 }
```

The `->` arrow and `:` type delimiter are required syntax. Properties are optional.

**Get an edge by ID:**

```sql
EDGE GET 'edge-xyz'
```

**Delete an edge:**

```sql
EDGE DELETE 'edge-xyz'
```

**List edges (optional type filter):**

```sql
EDGE LIST
EDGE LIST knows
EDGE LIST knows LIMIT 50 OFFSET 10
```

### Batch operations

**Batch create nodes:**

```sql
GRAPH BATCH CREATE NODES [(:person {name: 'Alice'}), (:person {name: 'Bob'}), (:company {name: 'Acme'})]
```

**Batch create edges:**

```sql
GRAPH BATCH CREATE EDGES [('node-1' -> 'node-2' : knows {since: 2024}), ('node-2' -> 'node-3' : works_at)]
```

**Batch delete:**

```sql
GRAPH BATCH DELETE NODES ['node-1', 'node-2']
GRAPH BATCH DELETE EDGES ['edge-1', 'edge-2']
```

**Batch update nodes:**

```sql
GRAPH BATCH UPDATE NODES [('node-1' {age: 31}), ('node-2' {age: 26})]
```

## Traversal and Paths

### NEIGHBORS

Find nodes adjacent to a given node.

```sql
NEIGHBORS 'node-1' OUTGOING
NEIGHBORS 'node-1' INCOMING
NEIGHBORS 'node-1' BOTH
```

**With edge type filter:**

```sql
NEIGHBORS 'node-1' OUTGOING knows
```

**Cross-engine: neighbors ranked by vector similarity:**

```sql
NEIGHBORS 'node-1' BOTH BY SIMILARITY [0.1, 0.2, 0.3] LIMIT 5
```

### PATH: shortest path (BFS)

```sql
PATH SHORTEST 'node-a' 'node-z'
```

Returns a list of node IDs forming the shortest unweighted path.

### PATH: all paths

```sql
PATH ALL 'node-a' 'node-z' MAX_DEPTH 5
```

### PATH: weighted shortest (Dijkstra)

```sql
PATH WEIGHTED 'node-a' 'node-z' WEIGHT distance
```

Uses the `distance` edge property as the weight.

### PATH: all weighted paths

```sql
PATH ALL_WEIGHTED 'node-a' 'node-z' WEIGHT cost MAX_DEPTH 4
```

### PATH: variable-length

```sql
PATH VARIABLE 'node-a' 'node-z' MIN_DEPTH 2 MAX_DEPTH 5
```

Finds paths with length between MIN_DEPTH and MAX_DEPTH hops.

## Graph Algorithms

All algorithms accept optional `DIRECTION` (OUTGOING/INCOMING/BOTH) and
edge type filters.

### PageRank

```sql
GRAPH PAGERANK
GRAPH PAGERANK DAMPING 0.85 TOLERANCE 0.001 ITERATIONS 100
GRAPH PAGERANK OUTGOING knows
```

Result type: `PageRank` (node ID + score pairs).

### Betweenness centrality

```sql
GRAPH BETWEENNESS CENTRALITY
GRAPH BETWEENNESS CENTRALITY SAMPLING 0.5 OUTGOING
```

`SAMPLING` controls the fraction of node pairs sampled (0.0-1.0). Lower values
are faster but less accurate.

### Closeness centrality

```sql
GRAPH CLOSENESS CENTRALITY
GRAPH CLOSENESS CENTRALITY INCOMING knows
```

### Eigenvector centrality

```sql
GRAPH EIGENVECTOR CENTRALITY
GRAPH EIGENVECTOR CENTRALITY ITERATIONS 100 TOLERANCE 0.001
```

### Community detection: Louvain

```sql
GRAPH LOUVAIN COMMUNITIES
GRAPH LOUVAIN COMMUNITIES RESOLUTION 1.0 PASSES 10 BOTH
```

Higher `RESOLUTION` produces smaller communities.

### Community detection: Label Propagation

```sql
GRAPH LABEL PROPAGATION
GRAPH LABEL PROPAGATION ITERATIONS 100 OUTGOING
```

## Advanced Features

### Constraints

```sql
GRAPH CONSTRAINT CREATE unique_name ON NODE person PROPERTY email UNIQUE
GRAPH CONSTRAINT CREATE exists_name ON NODE person PROPERTY name EXISTS
GRAPH CONSTRAINT CREATE type_age ON NODE person PROPERTY age TYPE 'integer'
GRAPH CONSTRAINT DROP unique_name
GRAPH CONSTRAINT LIST
GRAPH CONSTRAINT GET unique_name
```

### Graph indexes

```sql
GRAPH INDEX CREATE NODE PROPERTY name
GRAPH INDEX CREATE EDGE PROPERTY weight
GRAPH INDEX CREATE LABEL
GRAPH INDEX CREATE EDGE TYPE
GRAPH INDEX DROP NODE name
GRAPH INDEX DROP EDGE weight
GRAPH INDEX SHOW NODE
GRAPH INDEX SHOW EDGE
```

### Aggregation

```sql
GRAPH AGGREGATE COUNT NODES
GRAPH AGGREGATE COUNT NODES person
GRAPH AGGREGATE COUNT EDGES
GRAPH AGGREGATE COUNT EDGES knows
GRAPH AGGREGATE SUM NODE PROPERTY age person
GRAPH AGGREGATE AVG NODE PROPERTY age WHERE age > 18
GRAPH AGGREGATE MIN EDGE PROPERTY weight
GRAPH AGGREGATE MAX EDGE PROPERTY cost knows
```

Supported functions: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.

### Pattern matching

```sql
GRAPH PATTERN MATCH (a:person)-[:knows]->(b:person) LIMIT 10
GRAPH PATTERN COUNT (a)-[:works_at]->(b:company)
GRAPH PATTERN EXISTS (a:person)-[:knows]->(b)
```

Pattern syntax uses `(variable:Label)` for nodes and `[:TYPE]` for edges.

## Cypher (Experimental)

Neumann supports a subset of Cypher for graph queries.

### MATCH

```sql
MATCH (p:person)-[r:knows]->(q:person) WHERE p.age > 25 RETURN p.name, q.name LIMIT 10
OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m
```

### CREATE

```sql
CREATE (n:person {name: 'Alice', age: 30})-[:knows]->(m:person {name: 'Bob'})
```

### MERGE

```sql
MERGE (n:person {email: 'alice@example.com'}) ON CREATE SET n.name = 'Alice' ON MATCH SET n.last_seen = '2024-01-01'
```

### DELETE

```sql
DELETE n
DETACH DELETE n
```

`DETACH DELETE` removes the node and all its edges.

### Variable-length relationships

```sql
MATCH (a)-[*1..3]->(b) RETURN a, b
MATCH (a:person)-[:knows*1..5]->(b:person) RETURN a.name, b.name
```

## Direction Gotcha

This is the most common mistake when writing graph queries. The parser uses
`OUTGOING`, `INCOMING`, and `BOTH` as direction keywords.

```sql
NEIGHBORS 'n1' OUTGOING       -- correct
NEIGHBORS 'n1' OUT            -- WRONG: parse error
NEIGHBORS 'n1' IN             -- WRONG: IN is a different keyword (SQL IN)
NEIGHBORS 'n1' UNDIRECTED     -- WRONG: use BOTH
```

The same applies to algorithm direction filters:

```sql
GRAPH PAGERANK OUTGOING        -- correct
GRAPH PAGERANK OUT             -- WRONG
```
