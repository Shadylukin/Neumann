---
name: neumann-migrate
description: Migrate data to Neumann from other databases. Use when moving data from PostgreSQL, MySQL, MongoDB, Neo4j, Pinecone, Redis, or other systems to Neumann.
---

# Neumann Migration Guide

## Migration Strategy

Before migrating, map your source data model to Neumann's engines:

| Source | Neumann Engine | Command Prefix |
|--------|---------------|----------------|
| SQL tables | Relational engine | `CREATE TABLE`, `INSERT INTO` |
| Documents (MongoDB, Firestore) | Graph engine or Unified | `NODE CREATE` or `ENTITY CREATE` |
| Graph data (Neo4j, Neptune) | Graph engine | `NODE CREATE`, `EDGE CREATE` |
| Vectors (Pinecone, Weaviate, Qdrant) | Vector engine | `EMBED STORE`, `EMBED BATCH` |
| Key-value (Redis, DynamoDB) | Relational or Unified | `INSERT INTO` or `ENTITY CREATE` |
| Secrets (Vault, AWS Secrets Manager) | Vault engine | `VAULT SET` |

**General approach:**

1. Create a checkpoint before starting (`CHECKPOINT 'pre-migration'`).
2. Test with a small subset first.
3. Bulk load using batch commands.
4. Verify row/node/embedding counts match the source.
5. Build indexes after loading is complete.

## From SQL Databases (PostgreSQL, MySQL, SQLite)

### Step 1: Recreate the schema

Map source types to Neumann types:

| Source Type | Neumann Type |
|------------|-------------|
| `INTEGER`, `SERIAL`, `BIGSERIAL` | `INT` or `BIGINT` |
| `VARCHAR(n)`, `TEXT`, `CHAR(n)` | `VARCHAR(n)` or `TEXT` |
| `FLOAT`, `DOUBLE PRECISION`, `REAL` | `FLOAT` or `DOUBLE` |
| `DECIMAL(p,s)`, `NUMERIC(p,s)` | `DECIMAL(p,s)` |
| `BOOLEAN` | `BOOLEAN` |
| `DATE`, `TIME`, `TIMESTAMP` | `DATE`, `TIME`, `TIMESTAMP` |
| `BYTEA`, `BLOB` | `BLOB` |

```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INT,
    created_at TIMESTAMP
)
```

### Step 2: Bulk insert data

Use multi-row INSERT for efficiency:

```sql
INSERT INTO users (id, name, email, age) VALUES
    (1, 'Alice', 'alice@example.com', 30),
    (2, 'Bob', 'bob@example.com', 25),
    (3, 'Carol', 'carol@example.com', 35)
```

For large tables, batch in groups of 500-1000 rows per INSERT statement.

### Step 3: Recreate indexes

```sql
CREATE INDEX idx_users_email ON users (email)
CREATE INDEX idx_users_name ON users (name)
```

### Step 4: Verify

```sql
SELECT COUNT(*) FROM users
DESCRIBE users
```

## From Document Databases (MongoDB, Firestore, CouchDB)

Map documents to nodes or unified entities. Choose based on whether you
need graph relationships.

### Simple documents to entities

```sql
ENTITY CREATE 'user-1' { name: 'Alice', email: 'alice@example.com', tags: 'admin,editor' }
ENTITY CREATE 'user-2' { name: 'Bob', email: 'bob@example.com', tags: 'viewer' }
```

### Nested documents to nodes with edges

For a MongoDB document like `{ user: "Alice", address: { city: "NYC" } }`:

```sql
NODE CREATE person { name: 'Alice' }
NODE CREATE address { city: 'NYC', zip: '10001' }
EDGE CREATE 'person-node-id' -> 'address-node-id' : has_address
```

### Batch creation for bulk import

```sql
ENTITY BATCH CREATE [
    { key: 'doc-1', name: 'First', category: 'A' },
    { key: 'doc-2', name: 'Second', category: 'B' },
    { key: 'doc-3', name: 'Third', category: 'A' }
]
```

### Arrays as multiple edges

For a document with `{ user: "Alice", friends: ["Bob", "Carol"] }`:

```sql
NODE CREATE person { name: 'Alice' }
NODE CREATE person { name: 'Bob' }
NODE CREATE person { name: 'Carol' }
EDGE CREATE 'alice-id' -> 'bob-id' : friends_with
EDGE CREATE 'alice-id' -> 'carol-id' : friends_with
```

## From Graph Databases (Neo4j, Amazon Neptune, ArangoDB)

### Node mapping

Neo4j Cypher:

```cypher
CREATE (n:Person {name: "Alice", age: 30})
```

Neumann equivalent:

```sql
NODE CREATE person { name: 'Alice', age: 30 }
```

### Edge mapping

Neo4j Cypher:

```cypher
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS {since: 2020}]->(b)
```

Neumann equivalent:

```sql
EDGE CREATE 'alice-node-id' -> 'bob-node-id' : knows { since: 2020 }
```

### Batch import for large graphs

```sql
GRAPH BATCH CREATE NODES [
    (:person {name: 'Alice', age: 30}),
    (:person {name: 'Bob', age: 25}),
    (:company {name: 'Acme', industry: 'tech'})
]

GRAPH BATCH CREATE EDGES [
    ('alice-id' -> 'bob-id' : knows {since: 2020}),
    ('alice-id' -> 'acme-id' : works_at {role: 'engineer'})
]
```

### Recreate graph indexes and constraints

```sql
GRAPH INDEX CREATE NODE PROPERTY name
GRAPH INDEX CREATE LABEL
GRAPH CONSTRAINT CREATE unique_email ON NODE person PROPERTY email UNIQUE
```

## From Vector Databases (Pinecone, Weaviate, Qdrant, Milvus)

### Bulk vector import

Use `EMBED BATCH` for the fastest bulk loading:

```sql
EMBED BATCH [
    ('vec-1', [0.12, -0.34, 0.56, 0.78]),
    ('vec-2', [0.23, 0.45, -0.67, 0.89]),
    ('vec-3', [-0.11, 0.33, 0.55, -0.77])
]
```

### Collection mapping

If your source uses namespaces or collections:

```sql
EMBED STORE 'doc-1' [0.1, 0.2, 0.3] IN products
EMBED STORE 'doc-2' [0.4, 0.5, 0.6] IN products
EMBED STORE 'doc-3' [0.7, 0.8, 0.9] IN articles
```

### Metadata as entity properties

If vectors have associated metadata, use unified entities:

```sql
ENTITY CREATE 'doc-1' { title: 'Widget Manual', category: 'product' } EMBEDDING [0.1, 0.2, 0.3]
```

### Build index after bulk load

```sql
EMBED BUILD INDEX
```

Always build the index once after all embeddings are loaded, not after each insert.

### Verify

```sql
COUNT EMBEDDINGS
SHOW VECTOR INDEX
SIMILAR [0.1, 0.2, 0.3] LIMIT 3 METRIC COSINE
```

## Safety Checklist

1. **Checkpoint before migration:**

   ```sql
   CHECKPOINT 'pre-migration'
   ```

2. **Test with a small subset first.** Load 100 rows/nodes/vectors and verify
   correctness before running the full migration.

3. **Verify counts match:**

   ```sql
   SELECT COUNT(*) FROM users
   GRAPH AGGREGATE COUNT NODES person
   COUNT EMBEDDINGS
   ```

4. **Use transactions for atomicity** (when available):

   ```sql
   BEGIN CHAIN TRANSACTION
   -- migration commands here
   COMMIT CHAIN
   ```

5. **Rollback if something goes wrong:**

   ```sql
   CHECKPOINTS
   ROLLBACK TO 'pre-migration'
   ```

6. **Build indexes after loading, not during.** This applies to both relational
   indexes (`CREATE INDEX`) and vector indexes (`EMBED BUILD INDEX`).

## Batch Import Tips

- **Relational:** Multi-row `INSERT INTO ... VALUES (row1), (row2), ...` -- batch 500-1000 rows per statement.
- **Graph nodes:** `GRAPH BATCH CREATE NODES [...]` -- batch up to 1000 nodes per call.
- **Graph edges:** `GRAPH BATCH CREATE EDGES [...]` -- batch up to 1000 edges per call.
- **Vectors:** `EMBED BATCH [...]` -- batch up to 1000 vectors per call.
- **Entities:** `ENTITY BATCH CREATE [...]` -- batch up to 500 entities per call (each entity may create a node + embedding).
- **Quote all keys** that contain hyphens, colons, or other special characters.
- **Order matters:** Create nodes before edges that reference them. Create tables before inserting rows.
