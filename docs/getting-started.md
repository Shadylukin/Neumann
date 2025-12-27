# Getting Started

This guide walks you through your first 10 minutes with Neumann.

## Start the Shell

```bash
cd Neumann
cargo run --release --package neumann_shell
```

You'll see:
```
Neumann Shell v0.1.0
Type 'help' for commands, 'exit' to quit.
>
```

## Tutorial: Three Query Patterns

Neumann unifies relational, graph, and vector queries. Let's try all three.

### 1. Relational: Tables and SQL

Create a table and add some data:

```sql
> CREATE TABLE users (id INT, name TEXT, role TEXT)
OK

> INSERT INTO users VALUES (1, 'Alice', 'engineer')
1 row affected

> INSERT INTO users VALUES (2, 'Bob', 'designer')
1 row affected

> INSERT INTO users VALUES (3, 'Carol', 'manager')
1 row affected

> SELECT * FROM users
id | name  | role
---+-------+---------
1  | Alice | engineer
2  | Bob   | designer
3  | Carol | manager
(3 rows)

> SELECT name FROM users WHERE role = 'engineer'
name
-----
Alice
(1 row)
```

### 2. Graph: Nodes and Relationships

Create nodes and connect them:

```sql
> NODE CREATE person {name: 'Alice', team: 'platform'}
ID: node:1

> NODE CREATE person {name: 'Bob', team: 'design'}
ID: node:2

> NODE CREATE project {name: 'Neumann', status: 'active'}
ID: node:3

> EDGE CREATE node:1 -> node:3 : works_on {}
ID: edge:1

> EDGE CREATE node:2 -> node:3 : works_on {}
ID: edge:2

> NEIGHBORS node:3 INCOMING
node:1 (person) {name: "Alice", team: "platform"}
node:2 (person) {name: "Bob", team: "design"}
```

### 3. Vector: Embeddings and Similarity

Store embeddings and find similar items:

```sql
> EMBED STORE 'doc:1' [0.1, 0.2, 0.3, 0.4]
OK

> EMBED STORE 'doc:2' [0.15, 0.25, 0.35, 0.45]
OK

> EMBED STORE 'doc:3' [0.9, 0.8, 0.7, 0.6]
OK

> SIMILAR 'doc:1' LIMIT 2
1. doc:2 (0.9998)
2. doc:3 (0.7364)
```

## Saving Your Work

Neumann stores everything in memory. Save to disk:

```sql
> SAVE 'mydata.bin'
Saved to mydata.bin

> exit
```

Later, load it back:

```sql
> LOAD 'mydata.bin'
Loaded from mydata.bin
WAL replay: 0 operations

> SELECT * FROM users
-- Your data is back!
```

## What's Next?

### Extended Features

**Encrypted Secrets (Vault)**
```sql
> VAULT SET 'api_key' 'sk-secret123'
> VAULT GET 'api_key'
```

**LLM Response Caching**
```sql
> CACHE INIT
> CACHE PUT 'prompt:hello' 'Hello! How can I help?'
> CACHE GET 'prompt:hello'
```

**File Storage (Blob)**
```sql
> BLOB PUT 'report.txt' 'Quarterly results...'
> BLOBS
> BLOB TAG 'artifact:xxx' 'important'
```

### Learn More

- [Installation](installation.md) - Full setup instructions
- [Architecture](architecture.md) - How Neumann works
- [Query Reference](neumann-parser.md) - Complete syntax
- [Tensor Store API](tensor-store.md) - Rust library usage
- [Benchmarks](benchmarks.md) - Performance data

### Get Help

- [Discord](https://discord.gg/uN3KbAyKvw) - Community chat
- [GitHub Issues](https://github.com/Shadylukin/Neumann/issues) - Bug reports
- [GitHub Discussions](https://github.com/Shadylukin/Neumann/discussions) - Questions

## Quick Reference

| Pattern | Create | Query |
|---------|--------|-------|
| **Table** | `CREATE TABLE t (col TYPE)` | `SELECT * FROM t WHERE ...` |
| **Node** | `NODE CREATE label {props}` | `FIND NODE WHERE ...` |
| **Edge** | `EDGE CREATE n1 -> n2 : type` | `NEIGHBORS n OUTGOING` |
| **Vector** | `EMBED STORE 'key' [...]` | `SIMILAR 'key' LIMIT k` |
| **Secret** | `VAULT SET 'k' 'v'` | `VAULT GET 'k'` |
| **Cache** | `CACHE PUT 'k' 'v'` | `CACHE GET 'k'` |
| **File** | `BLOB PUT 'name' 'data'` | `BLOBS FOR entity` |
