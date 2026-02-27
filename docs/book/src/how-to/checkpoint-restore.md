# Create and Restore Checkpoints

## Goal

Create point-in-time snapshots, restore from them, configure retention
policies, and manage the checkpoint lifecycle.

> **See also:** [Tensor Checkpoint API](../reference/api/tensor-checkpoint.md) |
> [Checkpoint Design](../explanation/checkpoint-design.md) |
> [Architecture](../reference/api/tensor-checkpoint.md)

## Create a Checkpoint

### Via SQL commands

```sql
-- Named checkpoint
CHECKPOINT 'before-migration'

-- Auto-generated name (checkpoint-{timestamp})
CHECKPOINT
```

### Via Rust API

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
```

### Via Query Router

```rust
use query_router::QueryRouter;

let mut router = QueryRouter::new();
router.init_blob()?;
router.init_checkpoint()?;

router.execute_parsed("CHECKPOINT 'backup'")?;
```

## List Checkpoints

### Via SQL

```sql
-- List all checkpoints
CHECKPOINTS

-- List last 5 checkpoints
CHECKPOINTS LIMIT 5
```

### Via Rust API

```rust
let checkpoints = manager.list(Some(10)).await?;
for cp in &checkpoints {
    println!("{} | {} | {} | {}", cp.id, cp.name, cp.created_at, cp.trigger_name());
}
```

## Restore from a Checkpoint

### Via SQL

```sql
-- By name
ROLLBACK TO 'before-migration'

-- By ID
ROLLBACK TO 'a1b2c3d4-...'
```

### Via Rust API

```rust
// Rollback by name
manager.rollback("before-migration", &store).await?;

// Rollback by ID
manager.rollback("a1b2c3d4-e5f6-...", &store).await?;
```

Rollback completely replaces the current store contents with the checkpoint
state. Consider creating a checkpoint before rollback in case you need to
undo the restore.

## Delete a Checkpoint

```rust
manager.delete("before-migration").await?;
```

Both checkpoint name and ID are accepted.

## Configure Retention

### Builder pattern

```rust
let config = CheckpointConfig::default()
    .with_max_checkpoints(20)     // Keep up to 20 checkpoints
    .with_auto_checkpoint(true)   // Auto-checkpoint before destructive ops
    .with_interactive_confirm(false)  // No confirmation prompts
    .with_preview_sample_size(10);   // Show 10 sample rows in previews
```

### Common presets

| Use Case | max_checkpoints | auto_checkpoint | interactive_confirm |
| --- | --- | --- | --- |
| Interactive CLI | 10 | true | true |
| Batch scripts | 20 | true | false |
| Memory-constrained | 3 | false | false |
| Production (high retention) | 50 | true | true |

Retention is enforced automatically after every checkpoint creation. When the
count exceeds `max_checkpoints`, the oldest checkpoints are pruned.

## Set Up Auto-Checkpoints

Auto-checkpoints are created automatically before destructive operations
(DELETE, DROP TABLE, NODE DELETE, etc.) when `auto_checkpoint` is enabled.

```rust
let config = CheckpointConfig::default()
    .with_auto_checkpoint(true)
    .with_interactive_confirm(true);

let mut router = QueryRouter::new();
router.init_blob()?;
router.init_checkpoint_with_config(config)?;

// This DELETE will:
// 1. Show a preview of affected rows
// 2. Ask for confirmation
// 3. Create an auto-checkpoint named "auto-before-DELETE"
// 4. Execute the delete
router.execute_parsed("DELETE FROM users WHERE age > 50")?;
```

## Implement a Custom Confirmation Handler

```rust
use tensor_checkpoint::{ConfirmationHandler, DestructiveOp, OperationPreview};
use std::io::{self, Write};

struct InteractiveHandler;

impl ConfirmationHandler for InteractiveHandler {
    fn confirm(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool {
        println!("{}", tensor_checkpoint::format_confirmation_prompt(op, preview));
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input.trim().to_lowercase() == "yes"
    }
}

// Usage
manager.set_confirmation_handler(Arc::new(InteractiveHandler));
```

### For testing: auto-confirm or auto-reject

```rust
use tensor_checkpoint::{AutoConfirm, AutoReject};

// Always proceed (for automated scripts)
manager.set_confirmation_handler(Arc::new(AutoConfirm));

// Always cancel (for testing cancellation paths)
manager.set_confirmation_handler(Arc::new(AutoReject));
```

## Manage Memory During Checkpointing

Full snapshots are held in memory during creation and rollback. For large
stores:

| Store Size | Checkpoint Time | Rollback Time | Memory |
| --- | --- | --- | --- |
| 1K entries | ~5ms | ~3ms | ~100KB |
| 10K entries | ~50ms | ~30ms | ~1MB |
| 100K entries | ~500ms | ~300ms | ~10MB |
| 1M entries | ~5s | ~3s | ~100MB |

Strategies for large datasets:

1. **Separate hot and cold data** into different stores and only checkpoint
   the critical store
2. **Reduce retention** with a lower `max_checkpoints` to limit total storage
3. **Schedule checkpoints** during low-activity periods to reduce contention

## Initialize Checkpoint Support

The blob store must be initialized before checkpoint support. On the query
router:

```rust
let mut router = QueryRouter::new();

// IMPORTANT: blob must be initialized first
router.init_blob()?;

// Then checkpoint support
router.init_checkpoint()?;

// Or with custom config
router.init_checkpoint_with_config(CheckpointConfig::default()
    .with_max_checkpoints(20))?;
```

Calling `init_checkpoint()` without `init_blob()` will return an error.
