# Tensor Checkpoint (Module 13)

Checkpoint and rollback system for Neumann database state recovery.

## Overview

Tensor Checkpoint provides point-in-time snapshots of the database state, enabling users to:
- Create manual checkpoints before important operations
- Automatically checkpoint before destructive operations
- Rollback to any previous checkpoint
- Manage checkpoint retention with automatic cleanup

## Quick Start

```sql
-- Create a named checkpoint
CHECKPOINT 'before-migration'

-- Create checkpoint with auto-generated name
CHECKPOINT

-- List all checkpoints
CHECKPOINTS

-- List last 5 checkpoints
CHECKPOINTS LIMIT 5

-- Rollback to a checkpoint by name
ROLLBACK TO 'before-migration'

-- Rollback to a checkpoint by ID
ROLLBACK TO 'a1b2c3d4-e5f6-...'
```

## Commands

### CHECKPOINT

Creates a new checkpoint of the current database state.

```sql
-- Named checkpoint
CHECKPOINT 'checkpoint-name'

-- Auto-generated name (checkpoint-YYYYMMDD-HHMMSS)
CHECKPOINT
```

**Returns**: Checkpoint ID string

### CHECKPOINTS

Lists existing checkpoints.

```sql
-- List all checkpoints
CHECKPOINTS

-- List last N checkpoints
CHECKPOINTS LIMIT 10
```

**Returns**: Table with columns:
- `ID` - Unique checkpoint identifier
- `Name` - User-provided or auto-generated name
- `Created` - Timestamp
- `Type` - "manual" or "auto"

### ROLLBACK TO

Restores database state to a previous checkpoint.

```sql
-- By name
ROLLBACK TO 'checkpoint-name'

-- By ID
ROLLBACK TO 'uuid-string'
```

**Warning**: This replaces all current data with the checkpoint state.

## Architecture

### Components

```
tensor_checkpoint/
  src/
    lib.rs          # CheckpointManager, CheckpointConfig
    state.rs        # CheckpointState, DestructiveOp, metadata types
    storage.rs      # Blob storage integration
    retention.rs    # Count-based purge logic
    preview.rs      # Destructive operation previews
    error.rs        # Error types
```

### Storage

Checkpoints are stored as blob artifacts in tensor_blob:
- **Tag**: `_system:checkpoint`
- **Content-Type**: `application/x-neumann-checkpoint`
- **Data**: bincode-serialized `CheckpointState`

### CheckpointState

```rust
pub struct CheckpointState {
    pub id: String,                      // UUID
    pub name: String,                    // User-provided or auto-generated
    pub created_at: u64,                 // Unix timestamp
    pub trigger: Option<CheckpointTrigger>, // For auto-checkpoints
    pub store_snapshot: Vec<u8>,         // Serialized TensorStore
    pub metadata: CheckpointMetadata,    // Stats for validation
}
```

## Configuration

### CheckpointConfig

```rust
pub struct CheckpointConfig {
    pub max_checkpoints: usize,       // Default: 10
    pub auto_checkpoint: bool,        // Default: true
    pub interactive_confirm: bool,    // Default: true
    pub preview_sample_size: usize,   // Default: 5
}
```

### Builder Pattern

```rust
let config = CheckpointConfig::default()
    .with_max_checkpoints(20)
    .with_auto_checkpoint(true)
    .with_interactive_confirm(false)
    .with_preview_sample_size(10);
```

## Auto-Checkpoints

When enabled, checkpoints are automatically created before destructive operations:

| Operation | Description |
|-----------|-------------|
| `DELETE` | DELETE FROM table |
| `DROP TABLE` | DROP TABLE name |
| `DROP INDEX` | DROP INDEX on column |
| `NODE DELETE` | Delete graph node |
| `EMBED DELETE` | Delete embedding |
| `VAULT DELETE` | Delete secret |
| `BLOB DELETE` | Delete blob artifact |
| `CACHE CLEAR` | Clear cache entries |

Auto-checkpoints are named `auto-before-{OPERATION}` and include:
- The triggering command
- Preview of affected data
- Count of affected rows/items

## Interactive Confirmation

When `interactive_confirm` is enabled, destructive operations show a preview:

```
WARNING: About to delete 5 rows from table 'users'

Affected data sample:
  1. id=1, name='Alice', age=25
  2. id=2, name='Bob', age=30
  ... and 3 more

Type 'yes' to proceed, anything else to cancel:
```

### ConfirmationHandler Trait

```rust
pub trait ConfirmationHandler: Send + Sync {
    fn confirm(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool;
}
```

Implementations:
- `AutoConfirm` - Always confirms (for testing/automation)
- Custom handlers for interactive shells

## Retention Management

Checkpoints are automatically pruned based on `max_checkpoints`:

1. After each checkpoint creation
2. List all checkpoints sorted by creation time
3. Delete oldest checkpoints exceeding the limit

## API Reference

### CheckpointManager

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

    /// List checkpoints with optional limit
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
}
```

### DestructiveOp

```rust
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
```

### OperationPreview

```rust
pub struct OperationPreview {
    pub summary: String,        // Human-readable summary
    pub sample_data: Vec<String>, // Sample of affected data
    pub affected_count: usize,  // Total items affected
}
```

## Error Handling

```rust
pub enum CheckpointError {
    NotFound(String),           // Checkpoint not found
    Storage(String),            // Blob storage error
    Serialization(String),      // Bincode error
    Snapshot(String),           // TensorStore snapshot error
}
```

## Examples

### Basic Usage

```rust
use tensor_checkpoint::{CheckpointManager, CheckpointConfig};
use tensor_blob::{BlobStore, BlobConfig};
use tensor_store::TensorStore;

// Initialize
let store = TensorStore::new();
let blob = BlobStore::new(store.clone(), BlobConfig::default()).await?;
let blob = Arc::new(Mutex::new(blob));

let config = CheckpointConfig::default();
let manager = CheckpointManager::new(blob, config).await;

// Create checkpoint
let id = manager.create(Some("before-migration"), &store).await?;

// ... make changes ...

// Rollback if needed
manager.rollback("before-migration", &store).await?;
```

### With Query Router

```rust
use query_router::QueryRouter;

let mut router = QueryRouter::new();
router.init_blob()?;
router.init_checkpoint()?;

// Execute checkpoint commands via SQL
router.execute_parsed("CHECKPOINT 'backup'")?;
router.execute_parsed("CHECKPOINTS")?;
router.execute_parsed("ROLLBACK TO 'backup'")?;
```

## Performance Considerations

- **Snapshot Size**: Full TensorStore serialization (scales with data size)
- **Storage**: Uses blob deduplication for efficient storage
- **Rollback Time**: Proportional to snapshot size (full restore)
- **Retention**: Automatic cleanup prevents unbounded growth

## Limitations

- Full snapshots only (no incremental checkpoints)
- Single-node operation (no distributed checkpoints)
- In-memory restore (entire snapshot loaded)
- No automatic scheduling (manual or trigger-based only)

## Future Enhancements

- Incremental checkpoints for faster creation
- Streaming restore for large datasets
- Scheduled automatic checkpoints
- Checkpoint comparison/diff
- Partial rollback (specific tables/entities)
