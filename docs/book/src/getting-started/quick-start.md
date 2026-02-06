# Quick Start

Get up and running with Neumann in under 5 minutes. This guide walks you through
relational queries, graph operations, vector search, and the cross-engine "wow"
moment.

## Start the Shell

```bash
# In-memory (data lost on exit)
neumann

# With persistence (recommended)
neumann --wal-dir ./data
```

You will see:

```text
Neumann v0.1.0
Type 'help' for available commands.
neumann>
```

## 1. Relational Queries

Create a table and insert some data:

```sql
CREATE TABLE people (id INT PRIMARY KEY, name TEXT, role TEXT, team TEXT);

INSERT INTO people VALUES (1, 'Alice', 'Staff Engineer', 'Platform');
INSERT INTO people VALUES (2, 'Bob', 'Engineering Manager', 'Platform');
INSERT INTO people VALUES (3, 'Carol', 'Senior Engineer', 'ML');
INSERT INTO people VALUES (4, 'Dave', 'Junior Engineer', 'Platform');
```

Query it:

```sql
SELECT * FROM people WHERE team = 'Platform';
SELECT name, role FROM people ORDER BY name;
SELECT team, COUNT(*) AS headcount FROM people GROUP BY team;
```

## 2. Graph Operations

Create nodes with labels and properties:

```sql
NODE CREATE person { name: 'Alice', role: 'Staff Engineer' }
NODE CREATE person { name: 'Bob', role: 'Engineering Manager' }
NODE CREATE person { name: 'Carol', role: 'Senior Engineer' }
NODE CREATE person { name: 'Dave', role: 'Junior Engineer' }
```

List the nodes to see their auto-generated IDs:

```sql
NODE LIST person
```

Create edges (replace the IDs with the actual values from NODE LIST):

```sql
EDGE CREATE 'alice-node-id' -> 'bob-node-id' : reports_to
EDGE CREATE 'dave-node-id' -> 'bob-node-id' : reports_to
EDGE CREATE 'alice-node-id' -> 'dave-node-id' : mentors
```

Traverse the graph:

```sql
NEIGHBORS 'bob-node-id' INCOMING : reports_to
PATH SHORTEST 'dave-node-id' TO 'bob-node-id'
```

Run graph algorithms:

```sql
PAGERANK
```

## 3. Vector Search

Store embeddings with string keys:

```sql
EMBED STORE 'alice' [0.9, 0.4, 0.1, 0.7, 0.6, 0.3]
EMBED STORE 'bob' [0.6, 0.2, 0.1, 0.5, 0.3, 0.2]
EMBED STORE 'carol' [0.3, 0.9, 0.1, 0.4, 0.8, 0.1]
EMBED STORE 'dave' [0.4, 0.1, 0.2, 0.5, 0.2, 0.1]
```

Find similar items by key or by vector:

```sql
SIMILAR 'alice' LIMIT 3
SIMILAR [0.8, 0.5, 0.1, 0.6, 0.5, 0.2] LIMIT 3 METRIC COSINE
```

Check what is stored:

```sql
SHOW EMBEDDINGS
COUNT EMBEDDINGS
```

## 4. The Cross-Engine Moment

This is where Neumann shines. Combine graph traversal with vector similarity
in a single query:

```sql
SIMILAR 'alice' LIMIT 3 CONNECTED TO 'bob-node-id'
```

This finds embeddings similar to Alice's that are also connected to Bob in the
graph. No joins across separate databases needed.

Search across all engines with FIND:

```sql
FIND NODE person WHERE name = 'Alice'
```

Create unified entities that span relational, graph, and vector storage:

```sql
ENTITY CREATE 'project-x' { name: 'Project X', status: 'active' } EMBEDDING [0.5, 0.3, 0.7, 0.2, 0.4, 0.1]
ENTITY GET 'project-x'
```

## 5. Persistence

Save a checkpoint:

```sql
CHECKPOINT 'my-first-checkpoint'
CHECKPOINTS
```

If you started with `--wal-dir`, your data persists across restarts. You can also
save and load binary snapshots:

```sql
SAVE 'backup.bin'
LOAD 'backup.bin'
```

## Next Steps

- [Five-Minute Tutorial](five-minute-tutorial.md) -- Build a mini RAG system
- [Use Cases](use-cases.md) -- Real-world application patterns
- [Query Language Reference](../reference/query-language.md) -- Full command list
- [Architecture Overview](../architecture/overview.md) -- How it works under the hood
- [Python SDK](../sdks/python-quickstart.md) -- Use Neumann from Python
- [TypeScript SDK](../sdks/typescript-quickstart.md) -- Use Neumann from TypeScript

## Sample Dataset

A ready-made dataset is available in `samples/knowledge-base.nql`. Load it with:

```bash
neumann --wal-dir ./data
```

```text
neumann> \i samples/knowledge-base.nql
```
