# Quick Start

## Starting the Shell

```bash
neumann
```

You'll see the interactive prompt:

```
Neumann v0.1.0
Type 'help' for available commands.
neumann>
```

## Basic Operations

### Relational Data

```sql
-- Create a table
CREATE TABLE users (id INT, name STRING, age INT);

-- Insert data
INSERT INTO users VALUES (1, 'Alice', 30);
INSERT INTO users VALUES (2, 'Bob', 25);

-- Query data
SELECT * FROM users WHERE age > 20;
```

### Graph Data

```sql
-- Create nodes
CREATE NODE Person { name: 'Alice', role: 'Engineer' };
CREATE NODE Person { name: 'Bob', role: 'Manager' };

-- Create edges
CREATE EDGE REPORTS_TO FROM 'alice' TO 'bob';

-- Traverse
MATCH (p:Person)-[:REPORTS_TO]->(m:Person) RETURN p.name, m.name;
```

### Vector Data

```sql
-- Store embeddings
INSERT INTO embeddings (id, vec) VALUES (1, [0.1, 0.2, 0.3, 0.4]);
INSERT INTO embeddings (id, vec) VALUES (2, [0.15, 0.25, 0.35, 0.45]);

-- Similarity search
SELECT id FROM embeddings
WHERE SIMILAR(vec, [0.1, 0.2, 0.3, 0.4], 0.9)
LIMIT 10;
```

## Unified Queries

Neumann allows mixing data models in a single query:

```sql
-- Find similar users and their graph connections
SELECT u.name, e.similarity
FROM users u
JOIN embeddings e ON u.id = e.id
WHERE SIMILAR(e.vec, [0.1, 0.2, 0.3, 0.4], 0.8)
AND EXISTS (
  MATCH (u)-[:FRIEND]->()
);
```

## Persistence

Data is stored in memory by default. Enable WAL for persistence:

```bash
neumann --wal-dir ./data
```

## Next Steps

- [Building from Source](building-from-source.md) - Development setup
- [Architecture Overview](../architecture/overview.md) - System design
