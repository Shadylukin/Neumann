# Query Language Reference

Neumann uses a SQL-inspired query language extended with graph, vector, blob, vault,
cache, and chain commands. All commands are case-insensitive.

---

## Relational Commands

### SELECT

```sql
SELECT [DISTINCT] columns
FROM table [alias]
[JOIN table ON condition | USING (columns)]
[WHERE condition]
[GROUP BY columns]
[HAVING condition]
[ORDER BY columns [ASC|DESC] [NULLS FIRST|LAST]]
[LIMIT n]
[OFFSET n]
```

Columns can be `*`, expressions, or `expr AS alias`. Supports subqueries in FROM
and WHERE clauses.

**Join types**: `INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`, `NATURAL`.

```sql
SELECT u.name, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.total > 100
ORDER BY o.total DESC
LIMIT 10
```

### INSERT

```sql
INSERT INTO table [(columns)] VALUES (values), ...
INSERT INTO table [(columns)] SELECT ...
```

```sql
INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)
INSERT INTO users VALUES (2, 'Bob', 25), (3, 'Carol', 28)
```

### UPDATE

```sql
UPDATE table SET column = value, ... [WHERE condition]
```

```sql
UPDATE users SET age = 31 WHERE name = 'Alice'
```

### DELETE

```sql
DELETE FROM table [WHERE condition]
```

```sql
DELETE FROM users WHERE age < 18
```

### CREATE TABLE

```sql
CREATE TABLE [IF NOT EXISTS] name (
    column type [constraints],
    ...
    [table_constraints]
)
```

**Column types**: `INT`, `INTEGER`, `BIGINT`, `SMALLINT`, `FLOAT`, `DOUBLE`, `REAL`,
`DECIMAL(p,s)`, `NUMERIC(p,s)`, `VARCHAR(n)`, `CHAR(n)`, `TEXT`, `BOOLEAN`, `DATE`,
`TIME`, `TIMESTAMP`, `BLOB`.

**Column constraints**: `NOT NULL`, `NULL`, `UNIQUE`, `PRIMARY KEY`,
`DEFAULT expr`, `CHECK(expr)`, `REFERENCES table(column) [ON DELETE|UPDATE action]`.

**Table constraints**: `PRIMARY KEY (columns)`, `UNIQUE (columns)`,
`FOREIGN KEY (columns) REFERENCES table(column)`, `CHECK(expr)`.

**Referential actions**: `CASCADE`, `RESTRICT`, `SET NULL`, `SET DEFAULT`, `NO ACTION`.

```sql
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total FLOAT DEFAULT 0.0,
    created TIMESTAMP,
    UNIQUE (user_id, created)
)
```

### DROP TABLE

```sql
DROP TABLE [IF EXISTS] name [CASCADE]
```

```sql
DROP TABLE IF EXISTS orders CASCADE
```

### CREATE INDEX

```sql
CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (columns)
```

```sql
CREATE INDEX idx_users_name ON users (name)
CREATE UNIQUE INDEX idx_email ON users (email)
```

### DROP INDEX

```sql
DROP INDEX [IF EXISTS] name
DROP INDEX ON table(column)
```

```sql
DROP INDEX idx_users_name
DROP INDEX ON users(email)
```

### SHOW TABLES

```sql
SHOW TABLES
```

Lists all relational tables.

### DESCRIBE

```sql
DESCRIBE TABLE name
DESCRIBE NODE label
DESCRIBE EDGE type
```

Shows the schema of a table, node label, or edge type.

```sql
DESCRIBE TABLE users
DESCRIBE NODE person
DESCRIBE EDGE reports_to
```

---

## Graph Commands

### NODE CREATE

```sql
NODE CREATE label { key: value, ... }
```

Creates a node with the given label and properties.

```sql
NODE CREATE person { name: 'Alice', role: 'Engineer', team: 'Platform' }
```

### NODE GET

```sql
NODE GET id
```

Retrieves a node by its ID.

```sql
NODE GET 'abc-123'
```

### NODE DELETE

```sql
NODE DELETE id
```

Deletes a node by its ID.

```sql
NODE DELETE 'abc-123'
```

### NODE LIST

```sql
NODE LIST [label] [LIMIT n] [OFFSET m]
```

Lists nodes, optionally filtered by label.

```sql
NODE LIST person LIMIT 10
NODE LIST
```

### EDGE CREATE

```sql
EDGE CREATE from_id -> to_id : edge_type [{ key: value, ... }]
```

Creates a directed edge between two nodes.

```sql
EDGE CREATE 'alice-id' -> 'bob-id' : reports_to { since: '2024-01' }
```

### EDGE GET

```sql
EDGE GET id
```

Retrieves an edge by its ID.

### EDGE DELETE

```sql
EDGE DELETE id
```

Deletes an edge by its ID.

### EDGE LIST

```sql
EDGE LIST [type] [LIMIT n] [OFFSET m]
```

Lists edges, optionally filtered by type.

```sql
EDGE LIST reports_to LIMIT 20
```

### NEIGHBORS

```sql
NEIGHBORS id [OUTGOING|INCOMING|BOTH] [: edge_type]
    [BY SIMILARITY [vector] LIMIT n]
```

Finds neighbors of a node. The optional `BY SIMILARITY` clause enables cross-engine
queries that combine graph traversal with vector similarity.

```sql
NEIGHBORS 'alice-id' OUTGOING : reports_to
NEIGHBORS 'node-1' BOTH BY SIMILARITY [0.1, 0.2, 0.3] LIMIT 5
```

### PATH

```sql
PATH [SHORTEST|ALL|WEIGHTED|ALL_WEIGHTED|VARIABLE] from_id TO to_id
    [MAX_DEPTH n] [MIN_DEPTH n] [WEIGHT property]
```

Finds paths between two nodes.

```sql
PATH SHORTEST 'alice-id' TO 'ceo-id'
PATH WEIGHTED 'a' TO 'b' WEIGHT cost MAX_DEPTH 5
PATH ALL 'start' TO 'end' MIN_DEPTH 2 MAX_DEPTH 4
```

---

## Graph Algorithms

### PAGERANK

```sql
PAGERANK [DAMPING d] [TOLERANCE t] [MAX_ITERATIONS n]
    [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Computes PageRank scores for all nodes.

```sql
PAGERANK DAMPING 0.85 MAX_ITERATIONS 100
PAGERANK EDGE_TYPE collaborates
```

### BETWEENNESS

```sql
BETWEENNESS [SAMPLING_RATIO r]
    [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Computes betweenness centrality for all nodes.

```sql
BETWEENNESS SAMPLING_RATIO 0.5
```

### CLOSENESS

```sql
CLOSENESS [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Computes closeness centrality for all nodes.

### EIGENVECTOR

```sql
EIGENVECTOR [MAX_ITERATIONS n] [TOLERANCE t]
    [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Computes eigenvector centrality for all nodes.

### LOUVAIN

```sql
LOUVAIN [RESOLUTION r] [MAX_PASSES n]
    [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Detects communities using the Louvain algorithm.

```sql
LOUVAIN RESOLUTION 1.0 MAX_PASSES 10
```

### LABEL_PROPAGATION

```sql
LABEL_PROPAGATION [MAX_ITERATIONS n]
    [DIRECTION OUTGOING|INCOMING|BOTH] [EDGE_TYPE type]
```

Detects communities using label propagation.

---

## Graph Constraints

### GRAPH CONSTRAINT CREATE

```sql
GRAPH CONSTRAINT CREATE name ON NODE|EDGE [(label)] property UNIQUE|EXISTS|TYPE 'type'
```

Creates a property constraint on nodes or edges.

```sql
GRAPH CONSTRAINT CREATE unique_email ON NODE (person) email UNIQUE
GRAPH CONSTRAINT CREATE requires_name ON NODE name EXISTS
```

### GRAPH CONSTRAINT DROP

```sql
GRAPH CONSTRAINT DROP name
```

### GRAPH CONSTRAINT LIST

```sql
GRAPH CONSTRAINT LIST
```

Lists all graph constraints.

### GRAPH CONSTRAINT GET

```sql
GRAPH CONSTRAINT GET name
```

---

## Graph Indexes

### GRAPH INDEX CREATE

```sql
GRAPH INDEX CREATE NODE PROPERTY property
GRAPH INDEX CREATE EDGE PROPERTY property
GRAPH INDEX CREATE LABEL
GRAPH INDEX CREATE EDGE_TYPE
```

Creates a graph property or label index.

### GRAPH INDEX DROP

```sql
GRAPH INDEX DROP NODE property
GRAPH INDEX DROP EDGE property
```

### GRAPH INDEX SHOW

```sql
GRAPH INDEX SHOW NODE
GRAPH INDEX SHOW EDGE
```

---

## Graph Aggregation

### COUNT NODES / COUNT EDGES

```sql
GRAPH AGGREGATE COUNT NODES [label]
GRAPH AGGREGATE COUNT EDGES [type]
```

```sql
GRAPH AGGREGATE COUNT NODES person
GRAPH AGGREGATE COUNT EDGES reports_to
```

### AGGREGATE property

```sql
GRAPH AGGREGATE SUM|AVG|MIN|MAX|COUNT NODE property [label] [WHERE condition]
GRAPH AGGREGATE SUM|AVG|MIN|MAX|COUNT EDGE property [type] [WHERE condition]
```

```sql
GRAPH AGGREGATE AVG NODE age person
GRAPH AGGREGATE SUM EDGE weight collaborates WHERE weight > 0.5
```

---

## Graph Pattern Matching

### PATTERN MATCH

```sql
GRAPH PATTERN MATCH (pattern) [LIMIT n]
GRAPH PATTERN COUNT (pattern)
GRAPH PATTERN EXISTS (pattern)
```

Matches structural patterns in the graph.

```sql
GRAPH PATTERN MATCH (a:person)-[:reports_to]->(b:person) LIMIT 10
GRAPH PATTERN EXISTS (a:person)-[:mentors]->(b:person)
```

---

## Graph Batch Operations

### GRAPH BATCH CREATE NODES

```sql
GRAPH BATCH CREATE NODES [(label { props }), ...]
```

### GRAPH BATCH CREATE EDGES

```sql
GRAPH BATCH CREATE EDGES [(from -> to : type { props }), ...]
```

### GRAPH BATCH DELETE NODES

```sql
GRAPH BATCH DELETE NODES [id1, id2, ...]
```

### GRAPH BATCH DELETE EDGES

```sql
GRAPH BATCH DELETE EDGES [id1, id2, ...]
```

### GRAPH BATCH UPDATE NODES

```sql
GRAPH BATCH UPDATE NODES [(id { props }), ...]
```

---

## Vector Commands

### EMBED STORE

```sql
EMBED STORE key [vector] [IN collection]
```

Stores a vector embedding with an associated key.

```sql
EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]
EMBED STORE 'doc2' [0.5, 0.6, 0.7, 0.8] IN my_collection
```

### EMBED GET

```sql
EMBED GET key [IN collection]
```

Retrieves a stored embedding.

```sql
EMBED GET 'doc1'
```

### EMBED DELETE

```sql
EMBED DELETE key [IN collection]
```

Deletes a stored embedding.

### EMBED BUILD INDEX

```sql
EMBED BUILD INDEX [IN collection]
```

Builds or rebuilds the HNSW index for similarity search.

### EMBED BATCH

```sql
EMBED BATCH [('key1', [v1, v2, ...]), ('key2', [v1, v2, ...])] [IN collection]
```

Stores multiple embeddings in a single operation.

```sql
EMBED BATCH [('doc1', [0.1, 0.2]), ('doc2', [0.3, 0.4])]
```

### SIMILAR

```sql
SIMILAR key|[vector] [LIMIT n] [METRIC COSINE|EUCLIDEAN|DOT_PRODUCT]
    [CONNECTED TO node_id] [IN collection] [WHERE condition]
```

Finds similar embeddings by key or vector. The optional `CONNECTED TO` clause
combines vector similarity with graph connectivity for cross-engine queries.

```sql
SIMILAR 'doc1' LIMIT 5
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC COSINE
SIMILAR [0.1, 0.2, 0.3] LIMIT 5 CONNECTED TO 'alice-id'
SIMILAR 'doc1' LIMIT 10 IN my_collection WHERE score > 0.8
```

### SHOW EMBEDDINGS

```sql
SHOW EMBEDDINGS [LIMIT n]
```

Lists stored embeddings.

### SHOW VECTOR INDEX

```sql
SHOW VECTOR INDEX
```

Shows information about the HNSW index.

### COUNT EMBEDDINGS

```sql
COUNT EMBEDDINGS
```

Returns the number of stored embeddings.

---

## Unified Entity Commands

### ENTITY CREATE

```sql
ENTITY CREATE key { properties } [EMBEDDING [vector]]
```

Creates a unified entity with optional embedding. A unified entity spans all engines:
it is stored as relational data, as a graph node, and optionally as a vector embedding.

```sql
ENTITY CREATE 'alice' { name: 'Alice', role: 'Engineer' } EMBEDDING [0.1, 0.2, 0.3]
```

### ENTITY GET

```sql
ENTITY GET key
```

Retrieves a unified entity with all its data across engines.

### ENTITY UPDATE

```sql
ENTITY UPDATE key { properties } [EMBEDDING [vector]]
```

Updates an existing unified entity.

```sql
ENTITY UPDATE 'alice' { role: 'Senior Engineer' } EMBEDDING [0.15, 0.25, 0.35]
```

### ENTITY DELETE

```sql
ENTITY DELETE key
```

Deletes a unified entity from all engines.

### ENTITY CONNECT

```sql
ENTITY CONNECT from_key -> to_key : edge_type
```

Creates a relationship between two unified entities.

```sql
ENTITY CONNECT 'alice' -> 'bob' : reports_to
```

### ENTITY BATCH

```sql
ENTITY BATCH CREATE [{ key: 'k1', props... }, { key: 'k2', props... }]
```

Creates multiple unified entities in a single operation.

### FIND

```sql
FIND NODE [label] [WHERE condition] [LIMIT n]
FIND EDGE [type] [WHERE condition] [LIMIT n]
FIND ROWS FROM table [WHERE condition] [LIMIT n]
FIND PATH from_label -[edge_type]-> to_label [WHERE condition] [LIMIT n]
```

Cross-engine search that queries across relational, graph, and vector engines.

```sql
FIND NODE person WHERE name = 'Alice'
FIND EDGE reports_to LIMIT 10
FIND ROWS FROM users WHERE age > 25
FIND PATH person -[reports_to]-> person LIMIT 5
```

---

## Vault Commands

### VAULT SET

```sql
VAULT SET key value
```

Stores an encrypted secret.

```sql
VAULT SET 'api_key' 'sk-abc123'
```

### VAULT GET

```sql
VAULT GET key
```

Retrieves a decrypted secret (requires appropriate access).

```sql
VAULT GET 'api_key'
```

### VAULT DELETE

```sql
VAULT DELETE key
```

Deletes a secret.

### VAULT LIST

```sql
VAULT LIST [pattern]
```

Lists secrets, optionally filtered by pattern.

```sql
VAULT LIST
VAULT LIST 'api_*'
```

### VAULT ROTATE

```sql
VAULT ROTATE key new_value
```

Rotates a secret to a new value while maintaining the same key.

```sql
VAULT ROTATE 'api_key' 'sk-new456'
```

### VAULT GRANT

```sql
VAULT GRANT entity ON key
```

Grants an entity access to a secret.

```sql
VAULT GRANT 'alice' ON 'api_key'
```

### VAULT REVOKE

```sql
VAULT REVOKE entity ON key
```

Revokes an entity's access to a secret.

```sql
VAULT REVOKE 'bob' ON 'api_key'
```

---

## Cache Commands

### CACHE INIT

```sql
CACHE INIT
```

Initializes the LLM response cache.

### CACHE STATS

```sql
CACHE STATS
```

Shows cache hit/miss statistics.

### CACHE CLEAR

```sql
CACHE CLEAR
```

Clears all cache entries.

### CACHE EVICT

```sql
CACHE EVICT [n]
```

Evicts the least recently used entries. If `n` is provided, evicts that many.

### CACHE GET

```sql
CACHE GET key
```

Retrieves a cached response by exact key.

```sql
CACHE GET 'what is machine learning?'
```

### CACHE PUT

```sql
CACHE PUT key value
```

Stores a response in the cache.

```sql
CACHE PUT 'what is ML?' 'Machine learning is...'
```

### CACHE SEMANTIC GET

```sql
CACHE SEMANTIC GET query [THRESHOLD n]
```

Performs a semantic similarity lookup in the cache. Returns the closest matching
cached response if it exceeds the similarity threshold.

```sql
CACHE SEMANTIC GET 'explain machine learning' THRESHOLD 0.85
```

### CACHE SEMANTIC PUT

```sql
CACHE SEMANTIC PUT query response EMBEDDING [vector]
```

Stores a response with its embedding for semantic matching.

```sql
CACHE SEMANTIC PUT 'what is ML?' 'Machine learning is...' EMBEDDING [0.1, 0.2, 0.3]
```

---

## Blob Storage Commands

### BLOB INIT

```sql
BLOB INIT
```

Initializes the blob storage engine.

### BLOB PUT

```sql
BLOB PUT filename [DATA value | FROM path]
    [TYPE content_type] [BY creator] [LINK entity, ...] [TAG tag, ...]
```

Uploads a blob with optional metadata.

```sql
BLOB PUT 'report.pdf' FROM '/tmp/report.pdf' TYPE 'application/pdf' TAG 'quarterly'
BLOB PUT 'config.json' DATA '{"key": "value"}' BY 'admin'
```

### BLOB GET

```sql
BLOB GET artifact_id [TO path]
```

Downloads a blob. If `TO` is specified, writes to the given file path.

```sql
BLOB GET 'art-123'
BLOB GET 'art-123' TO '/tmp/download.pdf'
```

### BLOB DELETE

```sql
BLOB DELETE artifact_id
```

Deletes a blob.

### BLOB INFO

```sql
BLOB INFO artifact_id
```

Shows metadata for a blob (size, checksum, creation date, tags, links).

### BLOB LINK

```sql
BLOB LINK artifact_id TO entity
```

Links a blob to an entity.

```sql
BLOB LINK 'art-123' TO 'alice'
```

### BLOB UNLINK

```sql
BLOB UNLINK artifact_id FROM entity
```

Removes a link between a blob and an entity.

### BLOB LINKS

```sql
BLOB LINKS artifact_id
```

Lists all entities linked to a blob.

### BLOB TAG

```sql
BLOB TAG artifact_id tag
```

Adds a tag to a blob.

```sql
BLOB TAG 'art-123' 'important'
```

### BLOB UNTAG

```sql
BLOB UNTAG artifact_id tag
```

Removes a tag from a blob.

### BLOB VERIFY

```sql
BLOB VERIFY artifact_id
```

Verifies the integrity of a blob by checking its checksum.

### BLOB GC

```sql
BLOB GC [FULL]
```

Runs garbage collection on blob storage. `FULL` performs a thorough sweep.

### BLOB REPAIR

```sql
BLOB REPAIR
```

Repairs blob storage by fixing inconsistencies.

### BLOB STATS

```sql
BLOB STATS
```

Shows blob storage statistics (total count, size, etc.).

### BLOB META SET

```sql
BLOB META SET artifact_id key value
```

Sets a custom metadata key-value pair on a blob.

```sql
BLOB META SET 'art-123' 'department' 'engineering'
```

### BLOB META GET

```sql
BLOB META GET artifact_id key
```

Gets a custom metadata value from a blob.

### BLOBS

```sql
BLOBS [pattern]
```

Lists all blobs, optionally filtered by filename pattern.

### BLOBS FOR

```sql
BLOBS FOR entity
```

Lists blobs linked to a specific entity.

```sql
BLOBS FOR 'alice'
```

### BLOBS BY TAG

```sql
BLOBS BY TAG tag
```

Lists blobs with a specific tag.

```sql
BLOBS BY TAG 'quarterly'
```

### BLOBS WHERE TYPE

```sql
BLOBS WHERE TYPE = content_type
```

Lists blobs with a specific content type.

```sql
BLOBS WHERE TYPE = 'application/pdf'
```

### BLOBS SIMILAR TO

```sql
BLOBS SIMILAR TO artifact_id [LIMIT n]
```

Finds blobs similar to a given blob.

---

## Checkpoint Commands

### CHECKPOINT

```sql
CHECKPOINT [name]
```

Creates a named checkpoint (snapshot) of the current state.

```sql
CHECKPOINT 'before-migration'
CHECKPOINT
```

### CHECKPOINTS

```sql
CHECKPOINTS [LIMIT n]
```

Lists all available checkpoints.

### ROLLBACK TO

```sql
ROLLBACK TO checkpoint_id
```

Restores the database to a previous checkpoint.

```sql
ROLLBACK TO 'before-migration'
```

---

## Chain Commands

The chain subsystem provides a tensor-native blockchain with Raft consensus.

### BEGIN CHAIN TRANSACTION

```sql
BEGIN CHAIN TRANSACTION
```

Starts a new chain transaction. All subsequent mutations are buffered until commit.

### COMMIT CHAIN

```sql
COMMIT CHAIN
```

Commits the current chain transaction, appending a new block.

### ROLLBACK CHAIN TO

```sql
ROLLBACK CHAIN TO height
```

Rolls back the chain to a specific block height.

### CHAIN HEIGHT

```sql
CHAIN HEIGHT
```

Returns the current chain height (number of blocks).

### CHAIN TIP

```sql
CHAIN TIP
```

Returns the most recent block.

### CHAIN BLOCK

```sql
CHAIN BLOCK height
```

Retrieves a block at the given height.

```sql
CHAIN BLOCK 42
```

### CHAIN VERIFY

```sql
CHAIN VERIFY
```

Verifies the integrity of the entire chain.

### CHAIN HISTORY

```sql
CHAIN HISTORY key
```

Gets the history of changes for a specific key across all blocks.

```sql
CHAIN HISTORY 'users/alice'
```

### CHAIN SIMILAR

```sql
CHAIN SIMILAR [embedding] [LIMIT n]
```

Searches the chain by embedding similarity.

```sql
CHAIN SIMILAR [0.1, 0.2, 0.3] LIMIT 5
```

### CHAIN DRIFT

```sql
CHAIN DRIFT FROM height TO height
```

Computes drift metrics between two chain heights.

```sql
CHAIN DRIFT FROM 10 TO 50
```

### SHOW CODEBOOK GLOBAL

```sql
SHOW CODEBOOK GLOBAL
```

Shows the global codebook used for tensor compression.

### SHOW CODEBOOK LOCAL

```sql
SHOW CODEBOOK LOCAL domain
```

Shows the local codebook for a specific domain.

```sql
SHOW CODEBOOK LOCAL 'embeddings'
```

### ANALYZE CODEBOOK TRANSITIONS

```sql
ANALYZE CODEBOOK TRANSITIONS
```

Analyzes transitions between codebook states.

---

## Cluster Commands

### CLUSTER CONNECT

```sql
CLUSTER CONNECT address
```

Connects to a cluster node.

```sql
CLUSTER CONNECT 'node2@192.168.1.10:7000'
```

### CLUSTER DISCONNECT

```sql
CLUSTER DISCONNECT
```

Disconnects from the cluster.

### CLUSTER STATUS

```sql
CLUSTER STATUS
```

Shows the current cluster status (membership, leader, term).

### CLUSTER NODES

```sql
CLUSTER NODES
```

Lists all cluster nodes and their states.

### CLUSTER LEADER

```sql
CLUSTER LEADER
```

Shows the current cluster leader.

---

## Cypher Commands (Experimental)

Neumann includes experimental support for Cypher-style graph queries.

### MATCH

```sql
[OPTIONAL] MATCH pattern [WHERE condition]
RETURN items [ORDER BY items] [SKIP n] [LIMIT n]
```

Pattern matching query with Cypher syntax.

```sql
MATCH (p:Person)-[:REPORTS_TO]->(m:Person)
RETURN p.name, m.name

MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
WHERE a.name = 'Alice'
RETURN b.name, COUNT(*) AS depth
ORDER BY depth
LIMIT 10
```

**Relationship patterns**: `-[r:TYPE]->` (outgoing), `<-[r:TYPE]-` (incoming),
`-[r:TYPE]-` (undirected). Variable-length: `-[*1..5]->`.

### CYPHER CREATE

```sql
CREATE (pattern)
```

Creates nodes and relationships.

```sql
CREATE (p:Person { name: 'Dave', role: 'Designer' })
CREATE (a)-[:KNOWS]->(b)
```

### CYPHER DELETE

```sql
[DETACH] DELETE variables
```

Deletes nodes or relationships. `DETACH DELETE` also removes all relationships.

### MERGE

```sql
MERGE (pattern) [ON CREATE SET ...] [ON MATCH SET ...]
```

Upsert: matches an existing pattern or creates it.

```sql
MERGE (p:Person { name: 'Alice' })
ON CREATE SET p.created = '2024-01-01'
ON MATCH SET p.updated = '2024-06-01'
```

---

## Shell Commands

These commands are available in the interactive shell but are not part of the
query language.

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `exit` / `quit` | Exit the shell |
| `clear` | Clear the screen |
| `tables` | Alias for SHOW TABLES |
| `save 'path'` | Save data to binary file |
| `load 'path'` | Load data from binary file |

### Persistence

Start the shell with WAL (write-ahead log) for durability:

```bash
neumann --wal-dir ./data
```
