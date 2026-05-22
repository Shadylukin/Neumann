// SPDX-License-Identifier: MIT OR Apache-2.0
//! `TensorCheckpoint` - Rollback/Checkpoint System for Neumann
//!
//! Provides checkpoint and rollback capabilities for the Neumann database:
//! - Auto-checkpoints before destructive operations
//! - Manual CHECKPOINT command for user-initiated snapshots
//! - Interactive confirmation with preview of affected data
//! - Count-based retention with automatic purge
//!
//! Checkpoints are stored on disk via `FileCheckpointStore`.

mod checkpoint_store;
mod error;
pub mod file_store;
mod preview;
mod retention;
mod state;

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

pub use checkpoint_store::CheckpointStore;
pub use error::{CheckpointError, Result};
pub use file_store::FileCheckpointStore;
pub use preview::{format_confirmation_prompt, format_warning, PreviewGenerator};
pub use retention::RetentionManager;
pub use state::{
    CheckpointInfo, CheckpointMetadata, CheckpointState, CheckpointTrigger, DestructiveOp,
    GraphMeta, OperationPreview, RelationalMeta, VectorMeta,
};
use tensor_store::TensorStore;

/// Configuration for the checkpoint manager.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum number of checkpoints to retain (oldest are purged).
    pub max_checkpoints: usize,
    /// Whether to auto-checkpoint before destructive operations.
    pub auto_checkpoint: bool,
    /// Whether to prompt the user for confirmation before destructive operations.
    pub interactive_confirm: bool,
    /// Maximum number of sample data items shown in operation previews.
    pub preview_sample_size: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            auto_checkpoint: true,
            interactive_confirm: true,
            preview_sample_size: 5,
        }
    }
}

impl CheckpointConfig {
    /// Create a default checkpoint configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of checkpoints to retain.
    #[must_use]
    pub const fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Enable or disable auto-checkpoints before destructive operations.
    #[must_use]
    pub const fn with_auto_checkpoint(mut self, enabled: bool) -> Self {
        self.auto_checkpoint = enabled;
        self
    }

    /// Enable or disable interactive confirmation prompts.
    #[must_use]
    pub const fn with_interactive_confirm(mut self, enabled: bool) -> Self {
        self.interactive_confirm = enabled;
        self
    }

    /// Set the number of sample data items shown in operation previews.
    #[must_use]
    pub const fn with_preview_sample_size(mut self, size: usize) -> Self {
        self.preview_sample_size = size;
        self
    }
}

/// Trait for handling confirmation prompts before destructive operations.
pub trait ConfirmationHandler: Send + Sync {
    /// Return `true` to proceed with the operation, `false` to cancel.
    fn confirm(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool;
}

/// No-op confirmation handler that always confirms.
pub struct AutoConfirm;

impl ConfirmationHandler for AutoConfirm {
    fn confirm(&self, _op: &DestructiveOp, _preview: &OperationPreview) -> bool {
        true
    }
}

/// Confirmation handler that always rejects (for testing).
pub struct AutoReject;

impl ConfirmationHandler for AutoReject {
    fn confirm(&self, _op: &DestructiveOp, _preview: &OperationPreview) -> bool {
        false
    }
}

/// Central coordinator for creating, listing, restoring, and deleting checkpoints.
///
/// All state-changing operations are serialized through an internal mutex.
/// `list()` is lock-free (reads only, tolerant of concurrent atomic file creation).
pub struct CheckpointManager {
    store: Arc<dyn CheckpointStore>,
    config: CheckpointConfig,
    retention: RetentionManager,
    preview_gen: PreviewGenerator,
    confirm_handler: RwLock<Option<Arc<dyn ConfirmationHandler>>>,
    /// Serializes create, rollback, delete operations.
    op_lock: Mutex<()>,
}

impl CheckpointManager {
    /// Create a checkpoint manager backed by the given store and configuration.
    #[must_use]
    pub fn new(store: Arc<dyn CheckpointStore>, config: CheckpointConfig) -> Self {
        let retention = RetentionManager::new(config.max_checkpoints);
        let preview_gen = PreviewGenerator::new(config.preview_sample_size);

        Self {
            store,
            config,
            retention,
            preview_gen,
            confirm_handler: RwLock::new(None),
            op_lock: Mutex::new(()),
        }
    }

    /// Register a handler to be called for destructive operation confirmation.
    pub fn set_confirmation_handler(&self, handler: Arc<dyn ConfirmationHandler>) {
        *self.confirm_handler.write() = Some(handler);
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub const fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Create a manual checkpoint with optional name.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot cannot be created or the checkpoint cannot be stored.
    pub fn create(&self, name: Option<&str>, tensor_store: &TensorStore) -> Result<String> {
        let _guard = self.op_lock.lock();

        let id = uuid::Uuid::new_v4().to_string();
        let name = name.map_or_else(
            || {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                format!("checkpoint-{now}")
            },
            String::from,
        );

        let metadata = Self::collect_metadata(tensor_store);
        let snapshot_bytes = tensor_store
            .snapshot_bytes()
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        let state = CheckpointState::new(id.clone(), name, snapshot_bytes, metadata);

        self.store.store(&state)?;
        self.retention.enforce(self.store.as_ref())?;

        Ok(id)
    }

    /// Create an auto-checkpoint before a destructive operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot cannot be created or the checkpoint cannot be stored.
    pub fn create_auto(
        &self,
        command: &str,
        op: DestructiveOp,
        preview: OperationPreview,
        tensor_store: &TensorStore,
    ) -> Result<String> {
        let _guard = self.op_lock.lock();

        let id = uuid::Uuid::new_v4().to_string();
        let name = format!(
            "auto-before-{}",
            op.operation_name().to_lowercase().replace(' ', "-")
        );

        let trigger = CheckpointTrigger::new(command.to_string(), op, preview);
        let metadata = Self::collect_metadata(tensor_store);
        let snapshot_bytes = tensor_store
            .snapshot_bytes()
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        let state =
            CheckpointState::new(id.clone(), name, snapshot_bytes, metadata).with_trigger(trigger);

        self.store.store(&state)?;
        self.retention.enforce(self.store.as_ref())?;

        Ok(id)
    }

    /// Request confirmation for a destructive operation.
    #[must_use]
    pub fn request_confirmation(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool {
        if !self.config.interactive_confirm {
            return true;
        }

        self.confirm_handler
            .read()
            .as_ref()
            .map_or(true, |handler| handler.confirm(op, preview))
    }

    /// Generate a preview for a destructive operation.
    #[must_use]
    pub fn generate_preview(
        &self,
        op: &DestructiveOp,
        sample_data: Vec<String>,
    ) -> OperationPreview {
        self.preview_gen.generate(op, sample_data)
    }

    /// List checkpoints, most recent first.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store cannot be enumerated.
    pub fn list(&self, limit: Option<usize>) -> Result<Vec<CheckpointInfo>> {
        self.store.list(limit)
    }

    /// Rollback to a checkpoint by ID or name.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is not found or the snapshot cannot be restored.
    pub fn rollback(&self, id_or_name: &str, tensor_store: &TensorStore) -> Result<()> {
        let _guard = self.op_lock.lock();

        let state = self.store.load(id_or_name)?;

        tensor_store
            .restore_from_bytes(&state.store_snapshot)
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        Ok(())
    }

    /// Delete a checkpoint by ID or name.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is not found or cannot be removed.
    pub fn delete(&self, id_or_name: &str) -> Result<()> {
        let _guard = self.op_lock.lock();
        self.store.delete(id_or_name)
    }

    /// Returns whether auto-checkpoints are enabled for destructive operations.
    #[must_use]
    pub const fn auto_checkpoint_enabled(&self) -> bool {
        self.config.auto_checkpoint
    }

    /// Returns whether interactive confirmation prompts are enabled.
    #[must_use]
    pub const fn interactive_confirm_enabled(&self) -> bool {
        self.config.interactive_confirm
    }

    fn collect_metadata(store: &TensorStore) -> CheckpointMetadata {
        let store_key_count = store.len();

        // Count relational tables
        let table_keys: Vec<_> = store.scan("_schema:");
        let table_count = table_keys.len();
        let mut total_rows = 0;

        for key in &table_keys {
            if let Some(table_name) = key.strip_prefix("_schema:") {
                total_rows += store.scan_count(&format!("{table_name}:"));
            }
        }

        // Count graph entities
        let node_count = store.scan_count("node:");
        let edge_count = store.scan_count("edge:");

        // Count embeddings
        let embedding_count = store.scan_count("_embed:");

        CheckpointMetadata::new(
            RelationalMeta::new(table_count, total_rows),
            GraphMeta::new(node_count, edge_count),
            VectorMeta::new(embedding_count),
            store_key_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    use super::*;

    fn make_tensor(key: &str, value: &str) -> TensorData {
        let mut t = TensorData::new();
        t.set(
            key,
            TensorValue::Scalar(ScalarValue::String(value.to_string())),
        );
        t
    }

    fn setup_with_dir() -> (CheckpointManager, TensorStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = TensorStore::new();
        let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
        let config = CheckpointConfig::default();
        let manager = CheckpointManager::new(file_store, config);
        (manager, store, dir)
    }

    #[test]
    fn test_create_manual_checkpoint() {
        let (manager, store, _dir) = setup_with_dir();

        store.put("user:1", make_tensor("name", "Alice")).unwrap();

        let id = manager.create(Some("my-checkpoint"), &store).unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "my-checkpoint");
    }

    #[test]
    fn test_create_auto_checkpoint() {
        let (manager, store, _dir) = setup_with_dir();

        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 5,
        };
        let preview = OperationPreview::new("Deleting 5 rows".to_string(), vec![], 5);

        let id = manager
            .create_auto("DELETE FROM users", op, preview, &store)
            .unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].name.starts_with("auto-before-"));
    }

    #[test]
    fn test_rollback() {
        let (manager, store, _dir) = setup_with_dir();

        store.put("user:1", make_tensor("name", "Alice")).unwrap();

        let id = manager.create(Some("before-delete"), &store).unwrap();

        store.delete("user:1").unwrap();
        assert!(!store.exists("user:1"));

        manager.rollback(&id, &store).unwrap();

        assert!(store.exists("user:1"));
        let data = store.get("user:1").unwrap();
        assert_eq!(
            data.get("name"),
            Some(&TensorValue::Scalar(ScalarValue::String(
                "Alice".to_string()
            )))
        );
    }

    #[test]
    fn test_rollback_by_name() {
        let (manager, store, _dir) = setup_with_dir();

        store.put("key", make_tensor("val", "original")).unwrap();

        manager.create(Some("named-checkpoint"), &store).unwrap();

        store.delete("key").unwrap();

        manager.rollback("named-checkpoint", &store).unwrap();

        assert!(store.exists("key"));
    }

    #[test]
    fn test_retention() {
        let dir = tempfile::tempdir().unwrap();
        let store = TensorStore::new();
        let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
        let config = CheckpointConfig::default().with_max_checkpoints(2);
        let manager = CheckpointManager::new(file_store, config);

        for i in 0..5 {
            manager.create(Some(&format!("cp-{i}")), &store).unwrap();
        }

        let list = manager.list(None).unwrap();
        assert_eq!(list.len(), 2);

        for cp in &list {
            assert!(cp.name.starts_with("cp-"));
        }
    }

    #[test]
    fn test_confirmation_handler() {
        let (manager, _store, _dir) = setup_with_dir();

        manager.set_confirmation_handler(Arc::new(AutoReject));

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(!manager.request_confirmation(&op, &preview));
    }

    #[test]
    fn test_metadata_collection() {
        let (manager, store, _dir) = setup_with_dir();

        store
            .put("_schema:users", make_tensor("name", "users"))
            .unwrap();
        store.put("users:1", make_tensor("name", "Alice")).unwrap();
        store.put("users:2", make_tensor("name", "Bob")).unwrap();
        store.put("node:1", make_tensor("label", "Person")).unwrap();
        store.put("edge:1", make_tensor("type", "KNOWS")).unwrap();

        let mut embed_data = TensorData::new();
        embed_data.set("vec", TensorValue::Vector(vec![1.0, 2.0]));
        store.put("_embed:doc1", embed_data).unwrap();

        let id = manager.create(None, &store).unwrap();
        let state = manager.list(None).unwrap();
        assert_eq!(state.len(), 1);

        // Load the full state to check metadata
        let dir = tempfile::tempdir().unwrap();
        let file_store = FileCheckpointStore::new(dir.path()).unwrap();
        // Re-create to check: metadata is in the checkpoint
        let loaded = manager.list(None).unwrap();
        assert!(!loaded.is_empty());
        // Verify id is valid
        assert!(!id.is_empty());

        // Just verify the file_store module works for metadata storage
        let metadata = CheckpointManager::collect_metadata(&store);
        assert_eq!(metadata.relational.table_count, 1);
        assert_eq!(metadata.relational.total_rows, 2);
        assert_eq!(metadata.graph.node_count, 1);
        assert_eq!(metadata.graph.edge_count, 1);
        assert_eq!(metadata.vector.embedding_count, 1);

        drop(file_store);
    }

    #[test]
    fn test_delete_checkpoint() {
        let (manager, store, _dir) = setup_with_dir();

        let id = manager.create(Some("to-delete"), &store).unwrap();
        assert_eq!(manager.list(None).unwrap().len(), 1);

        manager.delete(&id).unwrap();
        assert_eq!(manager.list(None).unwrap().len(), 0);
    }

    #[test]
    fn test_delete_by_name() {
        let (manager, store, _dir) = setup_with_dir();

        manager.create(Some("named-cp"), &store).unwrap();
        assert_eq!(manager.list(None).unwrap().len(), 1);

        manager.delete("named-cp").unwrap();
        assert_eq!(manager.list(None).unwrap().len(), 0);
    }

    #[test]
    fn test_delete_not_found() {
        let (manager, _store, _dir) = setup_with_dir();

        let result = manager.delete("non-existent");
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_rollback_not_found() {
        let (manager, store, _dir) = setup_with_dir();

        let result = manager.rollback("non-existent", &store);
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_config_methods() {
        let config = CheckpointConfig::new()
            .with_max_checkpoints(5)
            .with_auto_checkpoint(false)
            .with_interactive_confirm(false)
            .with_preview_sample_size(10);

        assert_eq!(config.max_checkpoints, 5);
        assert!(!config.auto_checkpoint);
        assert!(!config.interactive_confirm);
        assert_eq!(config.preview_sample_size, 10);
    }

    #[test]
    fn test_auto_checkpoint_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
        let config = CheckpointConfig::default().with_auto_checkpoint(false);
        let manager = CheckpointManager::new(file_store, config);

        assert!(!manager.auto_checkpoint_enabled());
    }

    #[test]
    fn test_interactive_confirm_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
        let config = CheckpointConfig::default().with_interactive_confirm(false);
        let manager = CheckpointManager::new(file_store, config);

        assert!(!manager.interactive_confirm_enabled());
    }

    #[test]
    fn test_request_confirmation_without_handler() {
        let (manager, _store, _dir) = setup_with_dir();

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(manager.request_confirmation(&op, &preview));
    }

    #[test]
    fn test_request_confirmation_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
        let config = CheckpointConfig::default().with_interactive_confirm(false);
        let manager = CheckpointManager::new(file_store, config);

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(manager.request_confirmation(&op, &preview));
    }

    #[test]
    fn test_auto_confirm_handler() {
        let (manager, _store, _dir) = setup_with_dir();

        manager.set_confirmation_handler(Arc::new(AutoConfirm));

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(manager.request_confirmation(&op, &preview));
    }

    #[test]
    fn test_generate_preview() {
        let (manager, _store, _dir) = setup_with_dir();

        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 10,
        };
        let sample = vec!["row1".to_string(), "row2".to_string()];

        let preview = manager.generate_preview(&op, sample);
        assert_eq!(preview.affected_count, 10);
        assert_eq!(preview.sample_data.len(), 2);
    }

    #[test]
    fn test_list_with_limit() {
        let (manager, store, _dir) = setup_with_dir();

        for i in 0..5 {
            manager.create(Some(&format!("cp-{i}")), &store).unwrap();
        }

        let list = manager.list(Some(3)).unwrap();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_config_accessor() {
        let (manager, _store, _dir) = setup_with_dir();

        let config = manager.config();
        assert_eq!(config.max_checkpoints, 10);
    }

    #[test]
    fn test_create_unnamed_checkpoint() {
        let (manager, store, _dir) = setup_with_dir();

        let id = manager.create(None, &store).unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].name.starts_with("checkpoint-"));
    }
}
