# Building a Knowledge Graph

Build a knowledge graph from scratch using Neumann's graph engine. By the
end, you will have nodes representing people and topics, edges showing
relationships, and queries that traverse the graph.

## Prerequisites

- Neumann installed ([Installation](../how-to/installation.md))
- A running Neumann shell

## Step 1: Start the Shell

```bash
neumann --wal-dir ./knowledge-graph-data
```

You should see the `neumann>` prompt.

## Step 2: Create Nodes

Create nodes for people and topics:

```sql
NODE CREATE person { name: 'Alice', role: 'engineer' }
NODE CREATE person { name: 'Bob', role: 'designer' }
NODE CREATE person { name: 'Carol', role: 'manager' }
NODE CREATE topic { name: 'machine-learning' }
NODE CREATE topic { name: 'user-experience' }
NODE CREATE topic { name: 'project-planning' }
```

Verify the nodes exist:

```sql
NODE LIST
```

You should see 6 nodes with their auto-assigned IDs.

## Step 3: Create Edges

Connect people to their expertise topics:

```sql
EDGE CREATE 'node:1' -> 'node:4' : expert_in
EDGE CREATE 'node:1' -> 'node:5' : interested_in
EDGE CREATE 'node:2' -> 'node:5' : expert_in
EDGE CREATE 'node:3' -> 'node:6' : expert_in
EDGE CREATE 'node:3' -> 'node:4' : interested_in
```

Connect people to each other:

```sql
EDGE CREATE 'node:1' -> 'node:2' : collaborates_with
EDGE CREATE 'node:2' -> 'node:3' : reports_to
EDGE CREATE 'node:1' -> 'node:3' : reports_to
```

## Step 4: Query the Graph

Find all of Alice's connections:

```sql
TRAVERSE FROM 'node:1' DEPTH 1 DIRECTION OUTGOING
```

Find the shortest path between Alice and a topic:

```sql
PATH FROM 'node:1' TO 'node:6'
```

Find who is connected to machine-learning:

```sql
TRAVERSE FROM 'node:4' DEPTH 1 DIRECTION INCOMING
```

## Step 5: Add Properties and Query by Property

Update a node with additional properties:

```sql
NODE UPDATE 'node:1' SET experience_years = 5
```

## Step 6: Combine with Relational Data

Create a table to store project metadata alongside the graph:

```sql
CREATE TABLE projects (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT,
    lead_node TEXT
);
INSERT INTO projects VALUES (1, 'ML Pipeline', 'active', 'node:1');
INSERT INTO projects VALUES (2, 'Design System', 'active', 'node:2');
```

Now you can query both relational and graph data:

```sql
SELECT * FROM projects WHERE status = 'active';
TRAVERSE FROM 'node:1' DEPTH 2 DIRECTION OUTGOING
```

## Verification

You should see:

- 6 nodes (3 people, 3 topics)
- 8 edges (5 expertise, 3 collaboration)
- 2 projects in the relational table
- Traversal results showing connected nodes and edges

## Next Steps

- [Graph Traversals](../how-to/graph-traversals.md) -- advanced traversal
  patterns
- [Vector Search with Filtering](vector-search-filtering.md) -- add
  embeddings to your knowledge graph
- [Use Cases](use-cases.md) -- more application patterns
