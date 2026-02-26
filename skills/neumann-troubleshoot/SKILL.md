---
name: neumann-troubleshoot
description: Debug and troubleshoot Neumann issues. Use when encountering errors, performance problems, cluster issues, or unexpected query results.
---

# Neumann Troubleshooting Guide

## Parse Errors

Parse errors are the most common issue. The parser is strict and error messages
point to the exact token that failed.

### Reserved keyword collision

Neumann has 150+ reserved keywords. If a column name, node label, or edge type
collides with a keyword, you get a parse error.

**Symptoms:** `expected identifier, found keyword 'NODE'` or similar.

**Fix:** Rename the identifier. Common collisions: `node`, `edge`, `path`, `type`,
`status`, `key`, `value`, `name`, `index`, `limit`, `order`, `group`, `select`,
`table`, `set`, `delete`, `create`, `drop`, `insert`, `update`.

```sql
NODE CREATE node { ... }          -- WRONG: node is a keyword
NODE CREATE nd { ... }            -- correct: renamed to nd

CREATE TABLE status (...)         -- WRONG: status is a keyword
CREATE TABLE user_status (...)    -- correct: prefixed
```

### Unquoted keys with special characters

The parser splits tokens on `:`, `-`, and other punctuation.

```sql
EMBED STORE 'doc:1' [0.1, 0.2]   -- correct
EMBED STORE doc:1 [0.1, 0.2]     -- WRONG: 3 tokens (doc, :, 1)

NODE GET 'node-abc'               -- correct
NODE GET node-abc                 -- WRONG: parsed as node minus abc
```

### Wrong direction keywords

The parser only accepts `OUTGOING`, `INCOMING`, `BOTH`.

```sql
NEIGHBORS 'n1' OUTGOING           -- correct
NEIGHBORS 'n1' OUT                -- WRONG: OUT is not a direction keyword
NEIGHBORS 'n1' IN                 -- WRONG: IN means SQL IN operator
```

### Wrong SIMILAR syntax

```sql
SIMILAR [0.1, 0.2] LIMIT 5       -- correct
SIMILAR TO [0.1, 0.2] LIMIT 5    -- WRONG: no TO keyword
SIMILAR [0.1, 0.2] BY COSINE     -- WRONG: use METRIC, not BY
```

### Missing property block braces

```sql
NODE CREATE person { name: 'Alice' }    -- correct
NODE CREATE person name: 'Alice'        -- WRONG: missing braces
```

### Wrong property separator

```sql
NODE CREATE person { name: 'Alice', age: 30 }    -- correct (colon)
NODE CREATE person { name = 'Alice' }             -- WRONG: use colon, not equals
```

### Edge creation syntax

```sql
EDGE CREATE 'a' -> 'b' : knows         -- correct
EDGE CREATE 'a' 'b' knows              -- WRONG: missing -> and :
EDGE CREATE (a)-[:knows]->(b)          -- WRONG: that is Cypher, use EDGE syntax
```

## Connection Issues

### Port assignments

| Service | Default Port | Protocol |
|---------|-------------|----------|
| gRPC API | 9200 | HTTP/2 (gRPC) |
| Raft consensus | 9300 | TCP (length-prefixed frames) |

### Port conflict

**Symptom:** `Address already in use` on startup.

**Fix:** Check what is using the port:

```bash
lsof -i :9200
lsof -i :9300
```

Kill the conflicting process or change the Neumann bind address via
`NEUMANN_BIND_ADDR` environment variable.

### Health check

```bash
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

Expected response: `{ "status": "SERVING" }`.

### TLS issues

If TLS is enabled, use the correct certificate:

```bash
grpcurl -cacert ca.pem -cert client.pem -key client-key.pem localhost:9200 grpc.health.v1.Health/Check
```

### Authentication

If API key authentication is enabled, include the key in metadata:

```bash
grpcurl -plaintext -H 'authorization: Bearer <api-key>' localhost:9200 neumann.QueryService/Query
```

## Performance Issues

### Slow queries

**Check for missing indexes:**

```sql
DESCRIBE users                     -- shows column types but not indexes
SHOW TABLES                        -- verify table exists
SHOW VECTOR INDEX                  -- check if HNSW index is built
GRAPH INDEX SHOW NODE              -- check graph property indexes
GRAPH INDEX SHOW EDGE              -- check edge property indexes
```

**Common fixes:**

- Relational: `CREATE INDEX idx_name ON table (column)`
- Vector: `EMBED BUILD INDEX` (must be run after inserts)
- Graph: `GRAPH INDEX CREATE NODE PROPERTY prop_name`

### Slow SIMILAR queries

If `SIMILAR` is slow, the HNSW index likely was not built or is stale:

```sql
SHOW VECTOR INDEX
EMBED BUILD INDEX
```

### High memory usage

Check embedding count -- each embedding consumes memory proportional to its
dimensionality:

```sql
COUNT EMBEDDINGS
```

Check node/edge counts:

```sql
GRAPH AGGREGATE COUNT NODES
GRAPH AGGREGATE COUNT EDGES
```

### Large result sets

Always use `LIMIT` to bound result sizes:

```sql
SIMILAR [0.1, 0.2] LIMIT 10       -- bounded
NODE LIST person LIMIT 100         -- bounded
SELECT * FROM users LIMIT 1000     -- bounded
```

## Cluster Issues

### Split brain

**Symptom:** Two nodes both claim to be leader, or queries return different results
depending on which node is queried.

**Diagnose:**

```sql
CLUSTER STATUS
CLUSTER LEADER
CLUSTER NODES
```

Compare commit indices across nodes. The node with the higher commit index has
more up-to-date data. The other node likely lost connectivity and elected itself.

**Fix:** Restart the minority-side node. It will rejoin the cluster and
replicate from the true leader.

### Leader election timeout

**Symptom:** `CLUSTER LEADER` returns no leader or times out.

**Causes:**

- Network partition between nodes.
- All nodes started simultaneously (election collision). Wait for randomized
  timeout to resolve.
- Fewer than quorum nodes are reachable.

**Quorum requirements:**

| Cluster Size | Quorum | Can Tolerate Failures |
|-------------|--------|----------------------|
| 3 | 2 | 1 |
| 5 | 3 | 2 |
| 7 | 4 | 3 |

### Replication lag

**Symptom:** Writes on the leader are not immediately visible on followers.

This is normal in an eventually-consistent read path. For strong consistency,
read from the leader. For followers, wait briefly or verify via:

```sql
CLUSTER STATUS
```

Check that `commit_index` on followers is close to the leader's.

### Node won't join cluster

Verify:

1. `NEUMANN_CLUSTER_PEERS` includes all peer addresses in `node_id=IP:port` format.
2. `NEUMANN_CLUSTER_NODE_ID` is unique across all nodes.
3. Port 9300 is reachable between all nodes (Raft transport).
4. All nodes are running the same Neumann version.

## Data Issues

### Missing data

**Check you are querying the right engine:**

- Relational data: `SELECT * FROM table WHERE ...`
- Graph data: `NODE GET 'id'` or `NODE LIST label`
- Vector data: `EMBED GET 'key'`
- Unified data: `ENTITY GET 'key'`

A common mistake is inserting into one engine and querying another.

### Data integrity verification

```sql
CHAIN VERIFY
BLOB VERIFY 'blob-id'
```

### Recovery from checkpoints

```sql
CHECKPOINTS                        -- list available checkpoints
CHECKPOINTS LIMIT 5                -- show last 5
ROLLBACK TO 'checkpoint-id'        -- restore to a checkpoint
```

### WAL corruption

If the write-ahead log is corrupted, start Neumann with a fresh WAL directory.
Data persisted in checkpoints is still recoverable via `ROLLBACK TO`.

## Debug Logging

Set the `RUST_LOG` environment variable to control log verbosity.

**Levels:** `error`, `warn`, `info`, `debug`, `trace`.

```bash
# All modules at info level
RUST_LOG=info neumann

# Specific module at debug level
RUST_LOG=query_router=debug neumann

# Multiple modules at different levels
RUST_LOG=tensor_chain=debug,query_router=trace,tensor_store=info neumann

# Everything at trace (very verbose)
RUST_LOG=trace neumann
```

**Useful module targets for debugging:**

- `query_router` -- query parsing and routing
- `tensor_chain` -- Raft consensus and chain operations
- `tensor_store` -- storage operations
- `vector_engine` -- HNSW index and similarity search
- `graph_engine` -- graph traversal and algorithms
- `neumann_server` -- gRPC request handling

## Getting Help

When filing an issue, include:

1. Neumann version (from startup banner or `--version`).
2. Operating system and architecture.
3. The exact query that failed and the full error message.
4. Relevant log output (set `RUST_LOG=debug` to capture details).
5. Cluster configuration if applicable (node count, network topology).
6. Steps to reproduce the issue from a fresh state.
