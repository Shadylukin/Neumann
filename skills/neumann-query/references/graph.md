# Graph Engine -- Quick Reference

## Node CRUD

```sql
NODE CREATE label { key: value, key2: value2 }
NODE GET id
NODE DELETE id
NODE LIST [label] [LIMIT n] [OFFSET m]
```

- `id` is a numeric node ID (returned on creation).
- `label` is an unquoted identifier for the node type.
- Properties use `{ key: value }` syntax with colons, not equals.

## Edge CRUD

```sql
EDGE CREATE from_id -> to_id : edge_type { key: value }
EDGE GET id
EDGE DELETE id
EDGE LIST [type] [LIMIT n] [OFFSET m]
```

- Arrow `->` separates source and target node IDs.
- Colon `:` separates target from edge type label.
- Properties are optional after the label.

## Neighbors

```sql
NEIGHBORS 'node-id' OUTGOING [edge_type]
NEIGHBORS 'node-id' INCOMING [edge_type]
NEIGHBORS 'node-id' BOTH [edge_type]

-- Cross-engine: rank neighbors by vector similarity
NEIGHBORS 'node-id' BOTH BY SIMILARITY [0.1, 0.2, 0.3] LIMIT 5
```

Direction keywords: `OUTGOING`, `INCOMING`, `BOTH` (never `OUT`/`IN`).

## Path Finding

```sql
PATH SHORTEST 'from' 'to'
PATH ALL 'from' 'to' [MAX_DEPTH n]
PATH WEIGHTED 'from' 'to' WEIGHT property_name
PATH ALL_WEIGHTED 'from' 'to' WEIGHT property_name [MAX_DEPTH n]
PATH VARIABLE 'from' 'to' [MIN_DEPTH n] [MAX_DEPTH m]
```

## Graph Algorithms

All algorithms accept optional direction and edge type filters at the end.

```sql
-- PageRank centrality
GRAPH PAGERANK [DAMPING 0.85] [TOLERANCE 0.001] [ITERATIONS 100] [OUTGOING|INCOMING|BOTH] [edge_type]

-- Betweenness centrality
GRAPH BETWEENNESS CENTRALITY [SAMPLING 0.5] [OUTGOING|INCOMING|BOTH] [edge_type]

-- Closeness centrality
GRAPH CLOSENESS CENTRALITY [OUTGOING|INCOMING|BOTH] [edge_type]

-- Eigenvector centrality
GRAPH EIGENVECTOR CENTRALITY [ITERATIONS 100] [TOLERANCE 0.001] [OUTGOING|INCOMING|BOTH] [edge_type]

-- Louvain community detection
GRAPH LOUVAIN COMMUNITIES [RESOLUTION 1.0] [PASSES 10] [OUTGOING|INCOMING|BOTH] [edge_type]

-- Label propagation community detection
GRAPH LABEL PROPAGATION [ITERATIONS 100] [OUTGOING|INCOMING|BOTH] [edge_type]
```

## Graph Constraints

```sql
GRAPH CONSTRAINT CREATE name ON NODE [label] PROPERTY prop UNIQUE
GRAPH CONSTRAINT CREATE name ON NODE [label] PROPERTY prop EXISTS
GRAPH CONSTRAINT CREATE name ON NODE [label] PROPERTY prop TYPE 'string'
GRAPH CONSTRAINT CREATE name ON EDGE [type] PROPERTY prop UNIQUE
GRAPH CONSTRAINT DROP name
GRAPH CONSTRAINT LIST
GRAPH CONSTRAINT GET name
```

## Graph Indexes

```sql
GRAPH INDEX CREATE NODE PROPERTY prop
GRAPH INDEX CREATE EDGE PROPERTY prop
GRAPH INDEX CREATE LABEL
GRAPH INDEX CREATE EDGE TYPE
GRAPH INDEX DROP NODE prop
GRAPH INDEX DROP EDGE prop
GRAPH INDEX SHOW NODE
GRAPH INDEX SHOW EDGE
```

## Graph Aggregates

```sql
GRAPH AGGREGATE COUNT NODES [label]
GRAPH AGGREGATE COUNT EDGES [type]
GRAPH AGGREGATE SUM|AVG|MIN|MAX|COUNT NODE PROPERTY prop [label] [WHERE cond]
GRAPH AGGREGATE SUM|AVG|MIN|MAX|COUNT EDGE PROPERTY prop [type] [WHERE cond]
```

## Graph Pattern Matching

```sql
GRAPH PATTERN MATCH (a:Label)-[:TYPE]->(b:Label) [LIMIT n]
GRAPH PATTERN COUNT (a)-[:TYPE]->(b)
GRAPH PATTERN EXISTS (a:Label)-[:TYPE]->(b)
```

Patterns support node aliases, labels, edge types, and property filters.

## Graph Batch Operations

```sql
GRAPH BATCH CREATE NODES [(:Label {k: v}), (:Label2 {k: v})]
GRAPH BATCH CREATE EDGES [(from_id -> to_id : type {k: v})]
GRAPH BATCH DELETE NODES [id1, id2, id3]
GRAPH BATCH DELETE EDGES [id1, id2, id3]
GRAPH BATCH UPDATE NODES [(id {k: v}), (id2 {k: v})]
```

## Describe

```sql
DESCRIBE NODE label
DESCRIBE EDGE type
```

## Cypher (Experimental)

Cypher-style graph queries are also supported with standard Neo4j-like syntax:

```sql
-- Pattern matching
MATCH (n:Person)-[r:KNOWS]->(m:Person)
WHERE n.age > 25
RETURN n.name, m.name, r
ORDER BY n.name
SKIP 10
LIMIT 5

-- Optional match
OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m

-- Variable-length paths
MATCH (a)-[*1..5]->(b) RETURN a, b
MATCH path = (a)-[*]->(b) RETURN path

-- Create
CREATE (n:Person {name: 'Bob', age: 30})-[:KNOWS]->(m:Person {name: 'Carol'})

-- Merge (upsert)
MERGE (n:Person {name: 'Alice'})
ON CREATE SET n.created = 2024
ON MATCH SET n.updated = 2024

-- Delete
DELETE n
DETACH DELETE n
```
