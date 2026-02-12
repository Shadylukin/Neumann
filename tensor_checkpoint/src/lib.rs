// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! `TensorCheckpoint` - Rollback/Checkpoint System for Neumann
//!
//! Provides checkpoint and rollback capabilities for the Neumann database:
//! - Auto-checkpoints before destructive operations
//! - Manual CHECKPOINT command for user-initiated snapshots
//! - Interactive confirmation with preview of affected data
//! - Count-based retention with automatic purge
//!
//! Checkpoints are stored in `tensor_blob` for S3-style content-addressable storage.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::use_self)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::format_push_string)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::match_same_arms)]

mod error;
mod preview;
mod retention;
mod state;
mod storage;

use std::sync::Arc;

pub use error::{CheckpointError, Result};
pub use preview::{format_confirmation_prompt, format_warning, PreviewGenerator};
pub use retention::RetentionManager;
pub use state::{
    CheckpointInfo, CheckpointMetadata, CheckpointState, CheckpointTrigger, DestructiveOp,
    GraphMeta, OperationPreview, RelationalMeta, VectorMeta,
};
pub use storage::CheckpointStorage;
use tensor_blob::BlobStore;
use tensor_store::TensorStore;
use tokio::sync::Mutex;

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
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of checkpoints to retain.
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Enable or disable auto-checkpoints before destructive operations.
    pub fn with_auto_checkpoint(mut self, enabled: bool) -> Self {
        self.auto_checkpoint = enabled;
        self
    }

    /// Enable or disable interactive confirmation prompts.
    pub fn with_interactive_confirm(mut self, enabled: bool) -> Self {
        self.interactive_confirm = enabled;
        self
    }

    /// Set the number of sample data items shown in operation previews.
    pub fn with_preview_sample_size(mut self, size: usize) -> Self {
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
pub struct CheckpointManager {
    blob: Arc<Mutex<BlobStore>>,
    config: CheckpointConfig,
    retention: RetentionManager,
    preview_gen: PreviewGenerator,
    confirm_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl CheckpointManager {
    /// Create a checkpoint manager backed by the given blob store and configuration.
    pub fn new(blob: Arc<Mutex<BlobStore>>, config: CheckpointConfig) -> Self {
        let retention = RetentionManager::new(config.max_checkpoints);
        let preview_gen = PreviewGenerator::new(config.preview_sample_size);

        Self {
            blob,
            config,
            retention,
            preview_gen,
            confirm_handler: None,
        }
    }

    /// Register a handler to be called for destructive operation confirmation.
    pub fn set_confirmation_handler(&mut self, handler: Arc<dyn ConfirmationHandler>) {
        self.confirm_handler = Some(handler);
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Create a manual checkpoint with optional name.
    pub async fn create(&self, name: Option<&str>, store: &TensorStore) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let name = name.map_or_else(
            || {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("checkpoint-{now}")
            },
            String::from,
        );

        let metadata = self.collect_metadata(store);
        let snapshot_bytes = store
            .snapshot_bytes()
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        let state = CheckpointState::new(id.clone(), name, snapshot_bytes, metadata);

        let blob = self.blob.lock().await;
        CheckpointStorage::store(&state, &blob).await?;
        self.retention.enforce(&blob).await?;

        Ok(id)
    }

    /// Create an auto-checkpoint before a destructive operation.
    pub async fn create_auto(
        &self,
        command: &str,
        op: DestructiveOp,
        preview: OperationPreview,
        store: &TensorStore,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let name = format!(
            "auto-before-{}",
            op.operation_name().to_lowercase().replace(' ', "-")
        );

        let trigger = CheckpointTrigger::new(command.to_string(), op, preview);
        let metadata = self.collect_metadata(store);
        let snapshot_bytes = store
            .snapshot_bytes()
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        let state =
            CheckpointState::new(id.clone(), name, snapshot_bytes, metadata).with_trigger(trigger);

        let blob = self.blob.lock().await;
        CheckpointStorage::store(&state, &blob).await?;
        self.retention.enforce(&blob).await?;

        Ok(id)
    }

    /// Request confirmation for a destructive operation.
    pub fn request_confirmation(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool {
        if !self.config.interactive_confirm {
            return true;
        }

        self.confirm_handler
            .as_ref()
            .map_or(true, |handler| handler.confirm(op, preview))
    }

    /// Generate a preview for a destructive operation.
    pub fn generate_preview(
        &self,
        op: &DestructiveOp,
        sample_data: Vec<String>,
    ) -> OperationPreview {
        self.preview_gen.generate(op, sample_data)
    }

    /// List checkpoints, most recent first.
    pub async fn list(&self, limit: Option<usize>) -> Result<Vec<CheckpointInfo>> {
        let mut checkpoints = {
            let blob = self.blob.lock().await;
            CheckpointStorage::list(&blob).await?
        };

        if let Some(limit) = limit {
            checkpoints.truncate(limit);
        }

        Ok(checkpoints)
    }

    /// Rollback to a checkpoint by ID or name.
    pub async fn rollback(&self, id_or_name: &str, store: &TensorStore) -> Result<()> {
        let state = {
            let blob = self.blob.lock().await;
            CheckpointStorage::load(id_or_name, &blob).await?
        };

        store
            .restore_from_bytes(&state.store_snapshot)
            .map_err(|e| CheckpointError::Snapshot(e.to_string()))?;

        Ok(())
    }

    /// Delete a checkpoint by ID or name.
    pub async fn delete(&self, id_or_name: &str) -> Result<()> {
        let checkpoints = {
            let blob = self.blob.lock().await;
            CheckpointStorage::list(&blob).await?
        };

        let artifact_id = checkpoints
            .into_iter()
            .find(|cp| cp.id == id_or_name || cp.name == id_or_name)
            .map(|cp| cp.artifact_id);

        match artifact_id {
            Some(id) => {
                let blob = self.blob.lock().await;
                CheckpointStorage::delete(&id, &blob).await?;
                Ok(())
            },
            None => Err(CheckpointError::NotFound(id_or_name.to_string())),
        }
    }

    /// Returns whether auto-checkpoints are enabled for destructive operations.
    pub fn auto_checkpoint_enabled(&self) -> bool {
        self.config.auto_checkpoint
    }

    /// Returns whether interactive confirmation prompts are enabled.
    pub fn interactive_confirm_enabled(&self) -> bool {
        self.config.interactive_confirm
    }

    fn collect_metadata(&self, store: &TensorStore) -> CheckpointMetadata {
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
    use tensor_blob::BlobConfig;
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

    async fn setup() -> (CheckpointManager, TensorStore) {
        let store = TensorStore::new();
        let blob = BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap();
        let blob = Arc::new(Mutex::new(blob));
        let config = CheckpointConfig::default();
        let manager = CheckpointManager::new(blob, config);
        (manager, store)
    }

    #[tokio::test]
    async fn test_create_manual_checkpoint() {
        let (manager, store) = setup().await;

        store.put("user:1", make_tensor("name", "Alice")).unwrap();

        let id = manager.create(Some("my-checkpoint"), &store).await.unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "my-checkpoint");
    }

    #[tokio::test]
    async fn test_create_auto_checkpoint() {
        let (manager, store) = setup().await;

        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 5,
        };
        let preview = OperationPreview::new("Deleting 5 rows".to_string(), vec![], 5);

        let id = manager
            .create_auto("DELETE FROM users", op, preview, &store)
            .await
            .unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).await.unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].name.starts_with("auto-before-"));
    }

    #[tokio::test]
    async fn test_rollback() {
        let (manager, store) = setup().await;

        store.put("user:1", make_tensor("name", "Alice")).unwrap();

        let id = manager.create(Some("before-delete"), &store).await.unwrap();

        store.delete("user:1").unwrap();
        assert!(!store.exists("user:1"));

        manager.rollback(&id, &store).await.unwrap();

        assert!(store.exists("user:1"));
        let data = store.get("user:1").unwrap();
        assert_eq!(
            data.get("name"),
            Some(&TensorValue::Scalar(ScalarValue::String(
                "Alice".to_string()
            )))
        );
    }

    #[tokio::test]
    async fn test_rollback_by_name() {
        let (manager, store) = setup().await;

        store.put("key", make_tensor("val", "original")).unwrap();

        manager
            .create(Some("named-checkpoint"), &store)
            .await
            .unwrap();

        store.delete("key").unwrap();

        manager.rollback("named-checkpoint", &store).await.unwrap();

        assert!(store.exists("key"));
    }

    #[tokio::test]
    async fn test_retention() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap();
        let blob = Arc::new(Mutex::new(blob));

        let config = CheckpointConfig::default().with_max_checkpoints(2);
        let manager = CheckpointManager::new(blob, config);

        for i in 0..5 {
            manager
                .create(Some(&format!("cp-{i}")), &store)
                .await
                .unwrap();
        }

        // Should only have 2 checkpoints remaining (most recent)
        let list = manager.list(None).await.unwrap();
        assert_eq!(list.len(), 2);

        // Verify both remaining checkpoints have valid names
        for cp in &list {
            assert!(cp.name.starts_with("cp-"));
        }
    }

    #[tokio::test]
    async fn test_confirmation_handler() {
        let (mut manager, _store) = setup().await;

        manager.set_confirmation_handler(Arc::new(AutoReject));

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(!manager.request_confirmation(&op, &preview));
    }

    #[tokio::test]
    async fn test_metadata_collection() {
        let (manager, store) = setup().await;

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

        let id = manager.create(None, &store).await.unwrap();
        let blob = manager.blob.lock().await;
        let state = CheckpointStorage::load(&id, &blob).await.unwrap();

        assert_eq!(state.metadata.relational.table_count, 1);
        assert_eq!(state.metadata.relational.total_rows, 2);
        assert_eq!(state.metadata.graph.node_count, 1);
        assert_eq!(state.metadata.graph.edge_count, 1);
        assert_eq!(state.metadata.vector.embedding_count, 1);
    }

    #[tokio::test]
    async fn test_delete_checkpoint() {
        let (manager, store) = setup().await;

        let id = manager.create(Some("to-delete"), &store).await.unwrap();
        assert_eq!(manager.list(None).await.unwrap().len(), 1);

        manager.delete(&id).await.unwrap();
        assert_eq!(manager.list(None).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_delete_by_name() {
        let (manager, store) = setup().await;

        manager.create(Some("named-cp"), &store).await.unwrap();
        assert_eq!(manager.list(None).await.unwrap().len(), 1);

        manager.delete("named-cp").await.unwrap();
        assert_eq!(manager.list(None).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_delete_not_found() {
        let (manager, _store) = setup().await;

        let result = manager.delete("non-existent").await;
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_rollback_not_found() {
        let (manager, store) = setup().await;

        let result = manager.rollback("non-existent", &store).await;
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_config_methods() {
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

    #[tokio::test]
    async fn test_auto_checkpoint_enabled() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap();
        let blob = Arc::new(Mutex::new(blob));

        let config = CheckpointConfig::default().with_auto_checkpoint(false);
        let manager = CheckpointManager::new(blob, config);

        assert!(!manager.auto_checkpoint_enabled());
    }

    #[tokio::test]
    async fn test_interactive_confirm_enabled() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap();
        let blob = Arc::new(Mutex::new(blob));

        let config = CheckpointConfig::default().with_interactive_confirm(false);
        let manager = CheckpointManager::new(blob, config);

        assert!(!manager.interactive_confirm_enabled());
    }

    #[tokio::test]
    async fn test_request_confirmation_without_handler() {
        let (manager, _store) = setup().await;

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        // No handler set, should auto-confirm
        assert!(manager.request_confirmation(&op, &preview));
    }

    #[tokio::test]
    async fn test_request_confirmation_disabled() {
        let store = TensorStore::new();
        let blob = BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap();
        let blob = Arc::new(Mutex::new(blob));

        let config = CheckpointConfig::default().with_interactive_confirm(false);
        let manager = CheckpointManager::new(blob, config);

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        // Interactive confirm disabled, should always return true
        assert!(manager.request_confirmation(&op, &preview));
    }

    #[tokio::test]
    async fn test_auto_confirm_handler() {
        let (mut manager, _store) = setup().await;

        manager.set_confirmation_handler(Arc::new(AutoConfirm));

        let op = DestructiveOp::Delete {
            table: "test".to_string(),
            row_count: 1,
        };
        let preview = OperationPreview::empty("test");

        assert!(manager.request_confirmation(&op, &preview));
    }

    #[tokio::test]
    async fn test_generate_preview() {
        let (manager, _store) = setup().await;

        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 10,
        };
        let sample = vec!["row1".to_string(), "row2".to_string()];

        let preview = manager.generate_preview(&op, sample);
        assert_eq!(preview.affected_count, 10);
        assert_eq!(preview.sample_data.len(), 2);
    }

    #[tokio::test]
    async fn test_list_with_limit() {
        let (manager, store) = setup().await;

        for i in 0..5 {
            manager
                .create(Some(&format!("cp-{i}")), &store)
                .await
                .unwrap();
        }

        let list = manager.list(Some(3)).await.unwrap();
        assert_eq!(list.len(), 3);
    }

    #[tokio::test]
    async fn test_config_accessor() {
        let (manager, _store) = setup().await;

        let config = manager.config();
        assert_eq!(config.max_checkpoints, 10);
    }

    #[tokio::test]
    async fn test_create_unnamed_checkpoint() {
        let (manager, store) = setup().await;

        let id = manager.create(None, &store).await.unwrap();
        assert!(!id.is_empty());

        let list = manager.list(None).await.unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].name.starts_with("checkpoint-"));
    }
}
