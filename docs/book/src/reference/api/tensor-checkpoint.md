# Tensor Checkpoint API Reference

Point-in-time snapshots of the database state for recovery operations.

> **See also:** [Checkpoint Design](../../explanation/checkpoint-design.md) |
> [Checkpoint and Restore How-To](../../how-to/checkpoint-restore.md)

---

## Core Types

| Type | Description |
| --- | --- |
| `CheckpointManager` | Main API for checkpoint operations |
| `CheckpointConfig` | Configuration (retention, auto-checkpoint, interactive mode) |
| `CheckpointState` | Full checkpoint data with snapshot and metadata |
| `CheckpointInfo` | Lightweight checkpoint listing info |
| `CheckpointTrigger` | Context for auto-checkpoints (command, operation, preview) |

### State Types

| Type | Description |
| --- | --- |
| `DestructiveOp` | Enum of destructive operations that trigger auto-checkpoints |
| `OperationPreview` | Summary and sample data for confirmation prompts |
| `CheckpointMetadata` | Statistics for validation (tables, nodes, embeddings) |
| `RelationalMeta` | Table and row counts |
| `GraphMeta` | Node and edge counts |
| `VectorMeta` | Embedding count |

## Error Types

| Variant | Description | Common Cause |
| --- | --- | --- |
| `NotFound` | Checkpoint not found by ID or name | Typo in checkpoint name or ID was pruned by retention |
| `Storage` | Blob storage error | Disk full, permissions issue |
| `Serialization` | Bincode serialization error | Corrupt in-memory state |
| `Deserialization` | Bincode deserialization error | Corrupt checkpoint file |
| `Blob` | Underlying blob store error | BlobStore not initialized |
| `Snapshot` | TensorStore snapshot error | Store locked or corrupted |
| `Cancelled` | Operation cancelled by user | User rejected confirmation prompt |
| `InvalidId` | Invalid checkpoint identifier | Empty or malformed ID string |
| `Retention` | Retention enforcement error | Failed to delete old checkpoints |

## CheckpointManager

```rust
impl CheckpointManager {
    /// Create manager with blob storage and configuration
    pub async fn new(
        blob: Arc<Mutex<BlobStore>>,
        config: CheckpointConfig
    ) -> Self;

    /// Create a manual checkpoint
    pub async fn create(
        &self,
        name: Option<&str>,
        store: &TensorStore
    ) -> Result<String>;

    /// Create an auto-checkpoint before destructive operation
    pub async fn create_auto(
        &self,
        command: &str,
        op: DestructiveOp,
        preview: OperationPreview,
        store: &TensorStore
    ) -> Result<String>;

    /// Rollback to a checkpoint by ID or name
    pub async fn rollback(
        &self,
        id_or_name: &str,
        store: &TensorStore
    ) -> Result<()>;

    /// List checkpoints, most recent first
    pub async fn list(
        &self,
        limit: Option<usize>
    ) -> Result<Vec<CheckpointInfo>>;

    /// Delete a checkpoint by ID or name
    pub async fn delete(&self, id_or_name: &str) -> Result<()>;

    /// Generate preview for a destructive operation
    pub fn generate_preview(
        &self,
        op: &DestructiveOp,
        sample_data: Vec<String>
    ) -> OperationPreview;

    /// Request user confirmation for an operation
    pub fn request_confirmation(
        &self,
        op: &DestructiveOp,
        preview: &OperationPreview
    ) -> bool;

    /// Set custom confirmation handler
    pub fn set_confirmation_handler(
        &mut self,
        handler: Arc<dyn ConfirmationHandler>
    );

    /// Check if auto-checkpoint is enabled
    pub fn auto_checkpoint_enabled(&self) -> bool;

    /// Check if interactive confirmation is enabled
    pub fn interactive_confirm_enabled(&self) -> bool;

    /// Access the current configuration
    pub fn config(&self) -> &CheckpointConfig;
}
```

## ConfirmationHandler

```rust
pub trait ConfirmationHandler: Send + Sync {
    fn confirm(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool;
}
```

Built-in implementations:

| Type | Behavior | Use Case |
| --- | --- | --- |
| `AutoConfirm` | Always returns true | Automated scripts, testing |
| `AutoReject` | Always returns false | Testing cancellation paths |

## CheckpointStorage

Internal storage layer for checkpoint persistence:

```rust
impl CheckpointStorage {
    /// Store a checkpoint state to blob storage
    pub async fn store(state: &CheckpointState, blob: &BlobStore) -> Result<String>;

    /// Load a checkpoint by ID or name
    pub async fn load(checkpoint_id: &str, blob: &BlobStore) -> Result<CheckpointState>;

    /// List all checkpoints (sorted by created_at descending)
    pub async fn list(blob: &BlobStore) -> Result<Vec<CheckpointInfo>>;

    /// Delete a checkpoint by artifact ID
    pub async fn delete(artifact_id: &str, blob: &BlobStore) -> Result<()>;
}
```

## PreviewGenerator

```rust
impl PreviewGenerator {
    pub fn new(sample_size: usize) -> Self;

    pub fn generate(&self, op: &DestructiveOp, sample_data: Vec<String>) -> OperationPreview;
}

// Utility functions
pub fn format_warning(op: &DestructiveOp) -> String;
pub fn format_confirmation_prompt(op: &DestructiveOp, preview: &OperationPreview) -> String;
```

## CheckpointConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_checkpoints` | `usize` | 10 | Maximum checkpoints before pruning |
| `auto_checkpoint` | `bool` | true | Enable auto-checkpoints before destructive ops |
| `interactive_confirm` | `bool` | true | Require confirmation for destructive ops |
| `preview_sample_size` | `usize` | 5 | Number of sample rows in previews |

### Builder Pattern

```rust
let config = CheckpointConfig::default()
    .with_max_checkpoints(20)
    .with_auto_checkpoint(true)
    .with_interactive_confirm(false)
    .with_preview_sample_size(10);
```

### Configuration Presets

| Preset | max_checkpoints | auto_checkpoint | interactive_confirm | Use Case |
| --- | --- | --- | --- | --- |
| Default | 10 | true | true | Interactive CLI usage |
| Automated | 20 | true | false | Batch processing scripts |
| Minimal | 3 | false | false | Memory-constrained environments |
| Safe | 50 | true | true | Production with high retention |

## DestructiveOp

Operations that trigger auto-checkpoints when `auto_checkpoint` is enabled:

| Operation | Variant | Fields | Affected Count |
| --- | --- | --- | --- |
| DELETE | `Delete` | `table`, `row_count` | `row_count` |
| DROP TABLE | `DropTable` | `table`, `row_count` | `row_count` |
| DROP INDEX | `DropIndex` | `table`, `column` | 1 |
| NODE DELETE | `NodeDelete` | `node_id`, `edge_count` | 1 + `edge_count` |
| EMBED DELETE | `EmbedDelete` | `key` | 1 |
| VAULT DELETE | `VaultDelete` | `key` | 1 |
| BLOB DELETE | `BlobDelete` | `artifact_id`, `size` | 1 |
| CACHE CLEAR | `CacheClear` | `entry_count` | `entry_count` |

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DestructiveOp {
    Delete { table: String, row_count: usize },
    DropTable { table: String, row_count: usize },
    DropIndex { table: String, column: String },
    NodeDelete { node_id: u64, edge_count: usize },
    EmbedDelete { key: String },
    VaultDelete { key: String },
    BlobDelete { artifact_id: String, size: usize },
    CacheClear { entry_count: usize },
}

impl DestructiveOp {
    pub fn operation_name(&self) -> &'static str;
    pub fn affected_count(&self) -> usize;
}
```

## SQL Commands

### CHECKPOINT

```sql
-- Named checkpoint
CHECKPOINT 'before-migration'

-- Auto-generated name (checkpoint-{timestamp})
CHECKPOINT
```

### CHECKPOINTS

```sql
-- List all checkpoints
CHECKPOINTS

-- List last N checkpoints
CHECKPOINTS LIMIT 10
```

Returns: `ID`, `Name`, `Created`, `Type` (manual/auto)

### ROLLBACK TO

```sql
-- By name
ROLLBACK TO 'checkpoint-name'

-- By ID
ROLLBACK TO 'uuid-string'
```

## Storage Format

Checkpoints are stored as blob artifacts using content-addressable storage:

| Property | Value |
| --- | --- |
| Tag | `_system:checkpoint` |
| Content-Type | `application/x-neumann-checkpoint` |
| Format | bincode-serialized `CheckpointState` |
| Filename | `checkpoint_{id}.ncp` |
| Creator | `system:checkpoint` |

### CheckpointState Structure

```rust
#[derive(Serialize, Deserialize)]
pub struct CheckpointState {
    pub id: String,           // UUID v4
    pub name: String,         // User-provided or auto-generated
    pub created_at: u64,      // Unix timestamp (seconds)
    pub trigger: Option<CheckpointTrigger>,  // For auto-checkpoints
    pub store_snapshot: Vec<u8>,  // Serialized SlabRouterSnapshot
    pub metadata: CheckpointMetadata,
}
```

### Artifact Metadata Keys

| Key | Type | Description |
| --- | --- | --- |
| `checkpoint_id` | String | UUID identifier |
| `checkpoint_name` | String | User-provided or auto-generated name |
| `created_at` | String | Unix timestamp (parsed to u64) |
| `trigger` | String | Operation name (for auto-checkpoints only) |

## Size Formatting

Blob sizes in previews are formatted for readability:

| Bytes | Display |
| --- | --- |
| < 1024 | "N bytes" |
| >= 1 KB | "N.NN KB" |
| >= 1 MB | "N.NN MB" |
| >= 1 GB | "N.NN GB" |

## Performance

| Store Size | Checkpoint Time | Rollback Time | Memory |
| --- | --- | --- | --- |
| 1K entries | ~5ms | ~3ms | ~100KB |
| 10K entries | ~50ms | ~30ms | ~1MB |
| 100K entries | ~500ms | ~300ms | ~10MB |
| 1M entries | ~5s | ~3s | ~100MB |

## Edge Cases and Gotchas

- **Name vs ID conflict**: If a checkpoint is named with a valid UUID format, it
  may conflict with ID lookup (exact match on ID is checked first, then name).
- **Auto-generated names**: When no name is provided, format is
  `checkpoint-{unix_seconds}`. Auto-checkpoints use `auto-before-{operation-name}`.
- **Timestamp edge cases**: System time before epoch produces timestamp 0; rapid
  creation may produce identical second-granularity timestamps.
- **Blob dependency**: Always call `init_blob()` before `init_checkpoint()` on
  the query router, or initialization will fail.

## Limitations

- Full snapshots only (no incremental checkpoints)
- Single-node operation (no distributed checkpoints)
- In-memory restore (entire snapshot loaded)
- No automatic scheduling (manual or trigger-based only)
- Not atomic (partial restore possible on failure)
- No encryption (checkpoints stored in plaintext)
- Bloom filter state not preserved (rebuilt on load if needed)

## Related Modules

| Module | Relationship |
| --- | --- |
| [tensor_blob](./tensor-blob.md) | Storage backend for checkpoint data |
| [tensor_store](./tensor-store.md) | Source of snapshots and restore target |
| [query_router](./query-router.md) | SQL command integration |
| [neumann_shell](./neumann-shell.md) | Interactive confirmation handling |
