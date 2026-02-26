# Auxiliary Engines -- Quick Reference

## Vault (Encrypted Secret Storage)

```sql
-- Store a secret
VAULT SET 'api-key' 'sk-abc123secret'

-- Retrieve a secret
VAULT GET 'api-key'

-- Delete a secret
VAULT DELETE 'api-key'

-- List secrets (optional pattern filter)
VAULT LIST
VAULT LIST 'api-*'

-- Rotate a secret (update with new value)
VAULT ROTATE 'api-key' 'sk-newvalue456'

-- Grant entity access to a secret
VAULT GRANT 'user-entity' ON 'api-key'

-- Revoke entity access
VAULT REVOKE 'user-entity' ON 'api-key'
```

## Cache (Multi-Layer LLM Response Cache)

```sql
-- Initialize cache subsystem
CACHE INIT

-- Show cache statistics
CACHE STATS

-- Clear all cache entries
CACHE CLEAR

-- Evict entries (optional count)
CACHE EVICT
CACHE EVICT 10

-- Exact key-value cache
CACHE GET 'query-key'
CACHE PUT 'query-key' 'cached-response'

-- Semantic cache (similarity-based lookup)
CACHE SEMANTIC GET 'what is machine learning'
CACHE SEMANTIC GET 'what is machine learning' THRESHOLD 0.85

-- Semantic cache store (requires embedding vector)
CACHE SEMANTIC PUT 'what is ML' 'Machine learning is...' EMBEDDING [0.1, 0.2, 0.3]
```

## Blob Storage (Content-Addressable)

### BLOB Commands

```sql
-- Initialize blob store
BLOB INIT

-- Store blob (inline data or from file path)
BLOB PUT 'report.pdf' 'base64-or-text-data'
BLOB PUT 'report.pdf' FROM '/path/to/file'

-- Store with options (TYPE, BY, LINK, TAG)
BLOB PUT 'image.png' FROM '/path/to/image.png' TYPE 'image/png' BY 'user-1' LINK 'entity-1' TAG 'screenshot'

-- Retrieve blob
BLOB GET 'artifact-id'
BLOB GET 'artifact-id' TO '/output/path'

-- Delete blob
BLOB DELETE 'artifact-id'

-- Show blob metadata
BLOB INFO 'artifact-id'

-- Link/unlink blob to entity
BLOB LINK 'artifact-id' TO 'entity-id'
BLOB UNLINK 'artifact-id' FROM 'entity-id'

-- Get all links for a blob
BLOB LINKS 'artifact-id'

-- Tag/untag
BLOB TAG 'artifact-id' 'important'
BLOB UNTAG 'artifact-id' 'important'

-- Integrity verification
BLOB VERIFY 'artifact-id'

-- Garbage collection
BLOB GC
BLOB GC FULL

-- Repair storage
BLOB REPAIR

-- Storage statistics
BLOB STATS

-- Metadata key-value pairs on blobs
BLOB META SET 'artifact-id' 'author' 'Alice'
BLOB META GET 'artifact-id' 'author'
```

### BLOBS Query Commands

```sql
-- List all blobs
BLOBS

-- List with filename pattern
BLOBS 'report*'

-- Find blobs linked to an entity
BLOBS FOR 'entity-id'

-- Find blobs by tag
BLOBS BY TAG 'important'

-- Find blobs by content type
BLOBS WHERE TYPE = 'image/png'

-- Find similar blobs
BLOBS SIMILAR TO 'artifact-id' LIMIT 10
```

## Checkpoint (Snapshot/Restore)

```sql
-- Create checkpoint (auto-named)
CHECKPOINT

-- Create named checkpoint
CHECKPOINT 'before-migration'

-- List checkpoints
CHECKPOINTS
CHECKPOINTS LIMIT 5

-- Rollback to a checkpoint
ROLLBACK TO 'checkpoint-id-or-name'
```

## Chain (Tensor-Native Blockchain)

### Transaction Commands

```sql
-- Begin a chain transaction
BEGIN CHAIN TRANSACTION

-- Commit the current chain transaction
COMMIT CHAIN

-- Rollback chain to a specific block height
ROLLBACK CHAIN TO 42
```

### Query Commands

```sql
-- Get current chain height
CHAIN HEIGHT

-- Get the tip (latest) block
CHAIN TIP

-- Get block at a specific height
CHAIN BLOCK 42

-- Verify chain integrity
CHAIN VERIFY

-- Get history for a specific key
CHAIN HISTORY 'my-key'

-- Similarity search on chain embeddings
CHAIN SIMILAR [0.1, 0.2, 0.3] LIMIT 5

-- Drift analysis between block ranges
CHAIN DRIFT FROM 10 TO 50
```

### Codebook Commands

```sql
-- Show global codebook
SHOW CODEBOOK GLOBAL

-- Show local codebook for a domain
SHOW CODEBOOK LOCAL 'domain-name'

-- Analyze codebook transitions
ANALYZE CODEBOOK TRANSITIONS
```

## Cluster (Distributed Operations)

```sql
-- Connect to cluster
CLUSTER CONNECT '192.168.1.1:9000'

-- Disconnect
CLUSTER DISCONNECT

-- Show cluster status
CLUSTER STATUS

-- List cluster nodes
CLUSTER NODES

-- Show current leader
CLUSTER LEADER
```

## Spatial (R-tree Index)

```sql
-- Insert spatial entry
SPATIAL INSERT 'key' BOUNDS x y width height

-- Range query within radius
SPATIAL WITHIN x y RADIUS r [LIMIT n]

-- Delete spatial entry
SPATIAL DELETE 'key' BOUNDS x y width height

-- Count spatial entries
SPATIAL COUNT
```
