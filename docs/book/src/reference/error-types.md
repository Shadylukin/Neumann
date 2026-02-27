# Error Types

Consolidated reference for error types across all Neumann crates.

## tensor_store

| Error | Cause |
| --- | --- |
| `KeyNotFound(String)` | Key does not exist |
| `SerializationError(String)` | Failed to serialize/deserialize |
| `IndexError(String)` | HNSW index operation failed |
| `DimensionMismatch { expected, got }` | Vector dimensions do not match |
| `WalError(String)` | WAL operation failed |
| `SnapshotError(String)` | Snapshot creation/restore failed |
| `TieredStorageError(String)` | Tiered storage migration failed |

## relational_engine

| Error | Cause |
| --- | --- |
| `TableNotFound(String)` | Table does not exist |
| `TableAlreadyExists(String)` | Table name already taken |
| `ColumnNotFound(String)` | Column not in table schema |
| `TypeMismatch { column, expected, got }` | Value type does not match column |
| `ConstraintViolation(String)` | UNIQUE, NOT NULL, or PRIMARY KEY violated |
| `TransactionError(String)` | Transaction abort or timeout |

## graph_engine

| Error | Cause |
| --- | --- |
| `NodeNotFound(u64)` | Node with given ID does not exist |
| `EdgeNotFound(u64)` | Edge with given ID does not exist |
| `InvalidDirection` | Invalid traversal direction |
| `CycleDetected` | Graph cycle found during acyclic operation |
| `StorageError(String)` | Underlying storage failure |

## vector_engine

| Error | Cause |
| --- | --- |
| `KeyNotFound(String)` | Embedding key does not exist |
| `DimensionMismatch { expected, got }` | Vector size mismatch |
| `InvalidMetric(String)` | Unknown distance metric |
| `IndexError(String)` | HNSW index failure |
| `CollectionNotFound(String)` | Collection does not exist |

## tensor_chain

| Error | Cause |
| --- | --- |
| `NotLeader { leader_hint }` | Node is not the Raft leader |
| `NoQuorum` | Cannot reach majority of nodes |
| `TransactionConflict { similarity }` | Delta embedding conflict detected |
| `TransactionTimeout` | Transaction exceeded timeout |
| `DeadlockDetected { victim_tx }` | Wait-for graph cycle found |
| `LockTimeout` | Lock acquisition timed out |
| `InvalidBlock(String)` | Block validation failed |

## tensor_vault

| Error | Cause |
| --- | --- |
| `SecretNotFound(String)` | Secret key does not exist |
| `AccessDenied(String)` | Insufficient permissions |
| `EncryptionError(String)` | AES-256-GCM operation failed |
| `KeyDerivationError(String)` | Key derivation failed |

## tensor_cache

| Error | Cause |
| --- | --- |
| `CacheMiss(String)` | Key not found in any layer |
| `InvalidThreshold` | Similarity threshold out of range |
| `EmbeddingError(String)` | Embedding operation failed |

## tensor_blob

| Error | Cause |
| --- | --- |
| `BlobNotFound(String)` | Blob hash not found |
| `IntegrityError(String)` | Content hash mismatch |
| `IoError(String)` | Filesystem or network I/O failure |

## tensor_checkpoint

| Error | Cause |
| --- | --- |
| `CheckpointNotFound(String)` | Checkpoint ID does not exist |
| `RetentionViolation` | Cannot delete within retention window |
| `CorruptedCheckpoint(String)` | Checkpoint data integrity failure |

## neumann_parser

| Error | Cause |
| --- | --- |
| `UnexpectedToken { expected, got }` | Parser expected different token |
| `UnexpectedEof` | Input ended before statement complete |
| `InvalidSyntax(String)` | General parse failure |
| `UnsupportedQuery(String)` | Query type not recognized |

## query_router

| Error | Cause |
| --- | --- |
| `ParseError(String)` | Query failed to parse |
| `EngineError(String)` | Underlying engine returned error |
| `UnsupportedOperation(String)` | Operation not supported by engine |

## See Also

- Per-crate API reference pages in [Crate APIs](api/tensor-store.md)
- [Troubleshooting](../how-to/troubleshooting.md)
