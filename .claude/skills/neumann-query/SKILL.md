---
name: neumann-query
description: Write correct Neumann database queries. Use when writing NODE, EDGE, EMBED, SIMILAR, ENTITY, VAULT, CACHE, BLOB, CHECKPOINT, CHAIN, or CLUSTER commands, or any Neumann SQL query.
---

# Neumann Query Language Guide

## Critical Syntax Gotchas

These are the most common mistakes. Read this section first.

### 1. Direction keywords

Use `OUTGOING`, `INCOMING`, or `BOTH`. Never use `OUT`, `IN`, or `UNDIRECTED`.

```sql
NEIGHBORS 'node-1' OUTGOING    -- correct
NEIGHBORS 'node-1' OUT         -- WRONG: OUT is not a direction keyword
```

### 2. Colon-key tokenization

The parser splits `doc:1` into three tokens (`doc`, `:`, `1`). Always quote keys containing colons.

```sql
EMBED STORE 'doc:1' [0.1, 0.2]    -- correct
EMBED STORE doc:1 [0.1, 0.2]      -- WRONG: parsed as 3 separate tokens
```

### 3. Edge creation syntax

Use `EDGE CREATE from -> to : label`. The arrow `->` and colon `:` are required delimiters.

```sql
EDGE CREATE 'alice' -> 'bob' : knows { since: 2024 }    -- correct
EDGE CREATE (alice)-[:knows]->(bob)                      -- WRONG: that is Cypher syntax
```

### 4. Property blocks

Use curly braces with colons: `{ key: value }`. Not equals signs.

```sql
NODE CREATE person { name: 'Alice', age: 30 }    -- correct
NODE CREATE person { name = 'Alice' }            -- WRONG: use colon, not equals
```

### 5. Keywords as identifiers

The parser has 150+ reserved keywords. If a column or label name collides
(e.g., `node`, `edge`, `path`, `type`, `status`), rename it.

```sql
NODE CREATE person { name: 'Alice' }      -- correct
NODE CREATE node { name: 'Alice' }        -- WRONG: NODE is a reserved keyword
```

### 6. SIMILAR syntax

The format is `SIMILAR [vector] LIMIT n METRIC COSINE`. No `TO`, no `BY`.

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 5 METRIC COSINE       -- correct
SIMILAR TO [0.1, 0.2, 0.3]                           -- WRONG: no TO keyword
SIMILAR [0.1, 0.2] BY COSINE                         -- WRONG: use METRIC, not BY
```

### 7. EMBED STORE requires a quoted key

```sql
EMBED STORE 'my-key' [0.1, 0.2, 0.3]    -- correct
EMBED STORE my-key [0.1, 0.2, 0.3]      -- WRONG: unquoted key with hyphen fails
```

### 8. Prefix-based engine commands

VAULT, CACHE, and BLOB commands use prefix syntax, not SQL-style.

```sql
VAULT SET 'api-key' 'secret-value'    -- correct (prefix: VAULT SET)
SET VAULT 'api-key' 'secret-value'    -- WRONG: not SQL SET
```

### 9. Case insensitivity

All keywords are case-insensitive. String values preserve case.

```sql
select * from users          -- valid (same as SELECT * FROM users)
VAULT SET 'Key' 'Value'     -- 'Key' and 'Value' are case-sensitive strings
```

### 10. No semicolons needed

One command per line in batch mode. Semicolons are accepted but not required.

## Quick Reference

### Relational Engine

| Command | Syntax |
|---------|--------|
| SELECT | `SELECT col1, col2 FROM table WHERE cond ORDER BY col LIMIT n OFFSET m` |
| INSERT | `INSERT INTO table (col1, col2) VALUES (val1, val2)` |
| INSERT multi | `INSERT INTO table VALUES (v1, v2), (v3, v4)` |
| INSERT...SELECT | `INSERT INTO t1 (cols) SELECT cols FROM t2` |
| UPDATE | `UPDATE table SET col1 = val1 WHERE cond` |
| DELETE | `DELETE FROM table WHERE cond` |
| CREATE TABLE | `CREATE TABLE name (col1 INT PRIMARY KEY, col2 VARCHAR(255) NOT NULL)` |
| CREATE TABLE IF | `CREATE TABLE IF NOT EXISTS name (col1 INT)` |
| DROP TABLE | `DROP TABLE name` |
| ALTER TABLE | `ALTER TABLE name ADD COLUMN col INT` / `ALTER TABLE name DROP COLUMN col` |
| CREATE INDEX | `CREATE INDEX idx_name ON table (col)` |
| DROP INDEX | `DROP INDEX idx_name` |
| SHOW TABLES | `SHOW TABLES` |
| DESCRIBE | `DESCRIBE table` |
| DISTINCT | `SELECT DISTINCT col FROM table` |
| JOIN | `SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id` |
| GROUP BY | `SELECT col, COUNT(*) FROM table GROUP BY col HAVING COUNT(*) > 1` |
| UNION | `SELECT col FROM t1 UNION SELECT col FROM t2` |
| CASE | `SELECT CASE WHEN cond THEN val ELSE other END FROM table` |
| CAST | `SELECT CAST(col AS INT) FROM table` |
| EXISTS | `SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id)` |
| BETWEEN | `SELECT * FROM t WHERE col BETWEEN 1 AND 10` |
| IN | `SELECT * FROM t WHERE col IN (1, 2, 3)` |
| LIKE | `SELECT * FROM t WHERE col LIKE 'prefix%'` |
| IS NULL | `SELECT * FROM t WHERE col IS NULL` / `IS NOT NULL` |
| Subquery | `SELECT * FROM (SELECT col FROM t) AS sub` |

### Graph Engine

| Command | Syntax |
|---------|--------|
| NODE CREATE | `NODE CREATE label { key: value, key2: value2 }` |
| NODE GET | `NODE GET id` |
| NODE DELETE | `NODE DELETE id` |
| NODE LIST | `NODE LIST [label] [LIMIT n] [OFFSET m]` |
| EDGE CREATE | `EDGE CREATE from_id -> to_id : label { key: value }` |
| EDGE GET | `EDGE GET id` |
| EDGE DELETE | `EDGE DELETE id` |
| EDGE LIST | `EDGE LIST [type] [LIMIT n] [OFFSET m]` |
| NEIGHBORS | `NEIGHBORS 'id' OUTGOING` / `INCOMING` / `BOTH` |
| NEIGHBORS filtered | `NEIGHBORS 'id' OUTGOING label` |
| NEIGHBORS+vector | `NEIGHBORS 'id' BOTH BY SIMILARITY [vec] LIMIT 5` |
| PATH SHORTEST | `PATH SHORTEST 'from' 'to'` |
| PATH ALL | `PATH ALL 'from' 'to' MAX_DEPTH 5` |
| PATH WEIGHTED | `PATH WEIGHTED 'from' 'to' WEIGHT prop` |
| PATH ALL_WEIGHTED | `PATH ALL_WEIGHTED 'from' 'to' WEIGHT prop MAX_DEPTH 5` |
| PATH VARIABLE | `PATH VARIABLE 'from' 'to' MIN_DEPTH 2 MAX_DEPTH 5` |
| DESCRIBE NODE | `DESCRIBE NODE label` |
| DESCRIBE EDGE | `DESCRIBE EDGE type` |
| GRAPH PAGERANK | `GRAPH PAGERANK [DAMPING 0.85] [TOLERANCE 0.001] [ITERATIONS 100] [OUTGOING] [label]` |
| GRAPH BETWEENNESS | `GRAPH BETWEENNESS CENTRALITY [SAMPLING 0.5] [OUTGOING] [label]` |
| GRAPH CLOSENESS | `GRAPH CLOSENESS CENTRALITY [OUTGOING] [label]` |
| GRAPH EIGENVECTOR | `GRAPH EIGENVECTOR CENTRALITY [ITERATIONS 100] [TOLERANCE 0.001] [OUTGOING] [label]` |
| GRAPH LOUVAIN | `GRAPH LOUVAIN COMMUNITIES [RESOLUTION 1.0] [PASSES 10] [OUTGOING] [label]` |
| GRAPH LABEL_PROPAGATION | `GRAPH LABEL PROPAGATION [ITERATIONS 100] [OUTGOING] [label]` |
| GRAPH CONSTRAINT CREATE | `GRAPH CONSTRAINT CREATE name ON NODE [label] PROPERTY prop UNIQUE` / `EXISTS` / `TYPE 'string'` |
| GRAPH CONSTRAINT DROP | `GRAPH CONSTRAINT DROP name` |
| GRAPH CONSTRAINT LIST | `GRAPH CONSTRAINT LIST` |
| GRAPH CONSTRAINT GET | `GRAPH CONSTRAINT GET name` |
| GRAPH INDEX CREATE | `GRAPH INDEX CREATE NODE PROPERTY prop` / `EDGE PROPERTY prop` / `LABEL` / `EDGE TYPE` |
| GRAPH INDEX DROP | `GRAPH INDEX DROP NODE prop` / `EDGE prop` |
| GRAPH INDEX SHOW | `GRAPH INDEX SHOW NODE` / `GRAPH INDEX SHOW EDGE` |
| GRAPH AGGREGATE | `GRAPH AGGREGATE COUNT NODES [label]` / `COUNT EDGES [type]` |
| GRAPH AGGREGATE prop | `GRAPH AGGREGATE SUM NODE PROPERTY prop [label] [WHERE cond]` |
| GRAPH PATTERN MATCH | `GRAPH PATTERN MATCH (a:Label)-[:TYPE]->(b) [LIMIT n]` |
| GRAPH PATTERN COUNT | `GRAPH PATTERN COUNT (a)-[:TYPE]->(b)` |
| GRAPH PATTERN EXISTS | `GRAPH PATTERN EXISTS (a:Label)-[:TYPE]->(b)` |
| GRAPH BATCH CREATE NODES | `GRAPH BATCH CREATE NODES [(:Label {k:v}), (:Label2 {k:v})]` |
| GRAPH BATCH CREATE EDGES | `GRAPH BATCH CREATE EDGES [(from -> to : type {k:v})]` |
| GRAPH BATCH DELETE NODES | `GRAPH BATCH DELETE NODES [id1, id2]` |
| GRAPH BATCH DELETE EDGES | `GRAPH BATCH DELETE EDGES [id1, id2]` |
| GRAPH BATCH UPDATE NODES | `GRAPH BATCH UPDATE NODES [(id {k:v}), (id2 {k:v})]` |

### Vector Engine

| Command | Syntax |
|---------|--------|
| EMBED STORE | `EMBED STORE 'key' [0.1, 0.2, 0.3]` |
| EMBED STORE (collection) | `EMBED STORE 'key' [0.1, 0.2] IN my_collection` |
| EMBED GET | `EMBED GET 'key'` |
| EMBED DELETE | `EMBED DELETE 'key'` |
| EMBED BUILD INDEX | `EMBED BUILD INDEX` |
| EMBED BATCH | `EMBED BATCH [('k1', [0.1, 0.2]), ('k2', [0.3, 0.4])]` |
| SIMILAR (vector) | `SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC COSINE` |
| SIMILAR (key) | `SIMILAR 'key' LIMIT 5` |
| SIMILAR (collection) | `SIMILAR [0.1, 0.2] LIMIT 5 IN my_collection` |
| SIMILAR (filtered) | `SIMILAR [0.1, 0.2] LIMIT 5 WHERE category = 'tech'` |
| SIMILAR+graph | `SIMILAR [0.1, 0.2] LIMIT 10 CONNECTED TO 'node-id'` |
| SHOW EMBEDDINGS | `SHOW EMBEDDINGS [LIMIT n]` |
| SHOW VECTOR INDEX | `SHOW VECTOR INDEX` |
| COUNT EMBEDDINGS | `COUNT EMBEDDINGS` |

### Unified Engine (Entity + Find)

| Command | Syntax |
|---------|--------|
| ENTITY CREATE | `ENTITY CREATE 'key' { prop: val } [EMBEDDING [vec]]` |
| ENTITY GET | `ENTITY GET 'key'` |
| ENTITY UPDATE | `ENTITY UPDATE 'key' { prop: val } [EMBEDDING [vec]]` |
| ENTITY DELETE | `ENTITY DELETE 'key'` |
| ENTITY CONNECT | `ENTITY CONNECT 'from' -> 'to' : type` |
| ENTITY BATCH CREATE | `ENTITY BATCH CREATE [{ key: 'k1', prop: val }, { key: 'k2', prop: val }]` |
| FIND NODE | `FIND NODE [label] [WHERE cond] RETURN items [LIMIT n]` |
| FIND EDGE | `FIND EDGE [type] [WHERE cond] RETURN items [LIMIT n]` |
| FIND ROWS | `FIND ROWS FROM table [WHERE cond] RETURN items [LIMIT n]` |
| FIND PATH | `FIND PATH [from]-[:type]->[to] [WHERE cond] RETURN items` |

### Vault, Cache, Blob, Checkpoint, Chain, Cluster

See `references/auxiliary.md` for full syntax of all commands in these engines.

| Engine | Commands |
|--------|----------|
| Vault | `VAULT SET/GET/DELETE/LIST/ROTATE/GRANT/REVOKE` |
| Cache | `CACHE INIT/STATS/CLEAR/EVICT/GET/PUT/SEMANTIC GET/SEMANTIC PUT` |
| Blob | `BLOB INIT/PUT/GET/DELETE/INFO/LINK/UNLINK/LINKS/TAG/UNTAG/VERIFY/GC/REPAIR/STATS/META SET/META GET` |
| Blob queries | `BLOBS` / `BLOBS FOR entity` / `BLOBS BY TAG 'tag'` / `BLOBS WHERE TYPE = 'type'` / `BLOBS SIMILAR TO 'id' LIMIT n` |
| Checkpoint | `CHECKPOINT ['name']` / `CHECKPOINTS [LIMIT n]` / `ROLLBACK TO 'id'` |
| Chain | `BEGIN CHAIN TRANSACTION` / `COMMIT CHAIN` / `ROLLBACK CHAIN TO height` |
| Chain queries | `CHAIN HEIGHT/TIP/BLOCK n/VERIFY/HISTORY 'key'/SIMILAR [vec] LIMIT n/DRIFT FROM h1 TO h2` |
| Chain codebook | `SHOW CODEBOOK GLOBAL` / `SHOW CODEBOOK LOCAL 'domain'` / `ANALYZE CODEBOOK TRANSITIONS` |
| Cluster | `CLUSTER CONNECT 'addr'/DISCONNECT/STATUS/NODES/LEADER` |

### Cypher (Experimental)

| Command | Syntax |
|---------|--------|
| MATCH | `MATCH (n:Label)-[r:TYPE]->(m) WHERE n.prop = val RETURN n, m LIMIT 10` |
| OPTIONAL MATCH | `OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m` |
| CREATE | `CREATE (n:Label {prop: val})-[:TYPE]->(m:Label)` |
| MERGE | `MERGE (n:Label {key: val}) ON CREATE SET n.prop = val ON MATCH SET n.prop = val` |
| DELETE | `DELETE n` / `DETACH DELETE n` |
| Variable-length | `MATCH (a)-[*1..5]->(b) RETURN a, b` |

## Result Type Mapping

| Query | Result Variant |
|-------|---------------|
| SELECT | `Rows` |
| INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE | `Empty` or `Count` |
| NODE CREATE/GET | `Nodes` |
| NODE LIST | `Nodes` |
| EDGE CREATE/GET | `Edges` |
| EDGE LIST | `Edges` |
| NEIGHBORS | `Nodes` |
| PATH | `Path` (list of node IDs) |
| SIMILAR | `Similar` (key + score pairs) |
| EMBED STORE/GET/DELETE | `Value` or `Empty` |
| SHOW TABLES | `TableList` |
| SHOW EMBEDDINGS | `Similar` |
| COUNT EMBEDDINGS | `Count` |
| ENTITY operations | `Unified` |
| FIND | depends on pattern (Nodes, Edges, Rows, Path) |
| VAULT GET | `Value` |
| VAULT LIST | `Value` |
| CACHE GET/SEMANTIC GET | `Value` |
| BLOB GET | `Blob` (bytes) |
| BLOB INFO | `ArtifactInfo` |
| BLOBS / BLOBS FOR / BLOBS BY TAG | `ArtifactList` |
| BLOB STATS | `BlobStats` |
| CHECKPOINT/CHECKPOINTS | `CheckpointList` or `Value` |
| CHAIN operations | `Chain` |
| GRAPH PAGERANK | `PageRank` |
| GRAPH BETWEENNESS/CLOSENESS/EIGENVECTOR | `Centrality` |
| GRAPH LOUVAIN/LABEL PROPAGATION | `Communities` |
| GRAPH AGGREGATE | `Aggregate` or `Count` |
| GRAPH PATTERN | `PatternMatch` |
| GRAPH BATCH | `BatchResult` |
| GRAPH CONSTRAINT LIST | `Constraints` |
| GRAPH INDEX SHOW | `GraphIndexes` |
| SPATIAL WITHIN | `Spatial` |

## Cross-Engine Patterns

Neumann supports cross-engine queries that combine vector similarity with graph traversal:

**Vector + Graph** -- Find similar embeddings that are connected to a specific node:

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 CONNECTED TO 'node-id'
```

**Graph + Vector** -- Find neighbors of a node ranked by vector similarity:

```sql
NEIGHBORS 'node-id' BOTH BY SIMILARITY [0.1, 0.2, 0.3] LIMIT 5
```

**Unified Entity** -- Spans relational, graph, and vector in one command:

```sql
ENTITY CREATE 'user-1' { name: 'Alice', role: 'admin' } EMBEDDING [0.1, 0.2, 0.3]
ENTITY CONNECT 'user-1' -> 'team-1' : member_of
```

## Reference Pointers

For full syntax details beyond this quick reference:

- Relational SQL: `references/relational.md`
- Graph commands: `references/graph.md`
- Vector commands: `references/vector.md`
- Unified/Entity commands: `references/unified.md`
- Vault, Cache, Blob, Checkpoint, Chain, Cluster: `references/auxiliary.md`
- Complete reserved keyword list: `references/reserved-keywords.md`
- Parser source: `neumann_parser/src/parser.rs`
- AST definitions: `neumann_parser/src/ast.rs`
- Token definitions: `neumann_parser/src/token.rs`
