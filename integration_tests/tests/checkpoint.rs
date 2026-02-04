// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for the checkpoint/rollback system.

use std::sync::Arc;

use tensor_blob::{BlobConfig, BlobStore};
use tensor_checkpoint::{CheckpointConfig, CheckpointManager, DestructiveOp, OperationPreview};
use tensor_store::TensorStore;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_checkpoint_manager_create_and_list() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    // Create checkpoint
    let id = manager.create(Some("test-cp"), &store).await.unwrap();
    assert!(!id.is_empty());

    // List checkpoints
    let list = manager.list(None).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "test-cp");
}

#[tokio::test]
async fn test_checkpoint_manager_rollback() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    // Put some data
    use tensor_store::{ScalarValue, TensorData, TensorValue};
    let mut data = TensorData::new();
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String("Alice".to_string())),
    );
    store.put("user:1", data).unwrap();

    // Create checkpoint
    let id = manager.create(Some("before-delete"), &store).await.unwrap();

    // Delete data
    store.delete("user:1").unwrap();
    assert!(!store.exists("user:1"));

    // Rollback
    manager.rollback(&id, &store).await.unwrap();

    // Verify restoration
    assert!(store.exists("user:1"));
    let restored = store.get("user:1").unwrap();
    assert_eq!(
        restored.get("name"),
        Some(&TensorValue::Scalar(ScalarValue::String(
            "Alice".to_string()
        )))
    );
}

#[tokio::test]
async fn test_checkpoint_manager_rollback_by_name() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    use tensor_store::{ScalarValue, TensorData, TensorValue};
    let mut data = TensorData::new();
    data.set(
        "val",
        TensorValue::Scalar(ScalarValue::String("original".to_string())),
    );
    store.put("key", data).unwrap();

    manager.create(Some("named-cp"), &store).await.unwrap();

    store.delete("key").unwrap();

    // Rollback by name
    manager.rollback("named-cp", &store).await.unwrap();

    assert!(store.exists("key"));
}

#[tokio::test]
async fn test_checkpoint_auto_checkpoint() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    let op = DestructiveOp::Delete {
        table: "users".to_string(),
        row_count: 10,
    };
    let preview = OperationPreview::new("Deleting 10 rows".to_string(), vec![], 10);

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
async fn test_checkpoint_retention() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default().with_max_checkpoints(3);
    let manager = CheckpointManager::new(blob, config).await;

    // Create 5 checkpoints
    for i in 0..5 {
        manager
            .create(Some(&format!("cp-{i}")), &store)
            .await
            .unwrap();
    }

    // Should only have 3 (the most recent)
    let list = manager.list(None).await.unwrap();
    assert_eq!(list.len(), 3);
}

#[tokio::test]
async fn test_checkpoint_delete() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    let id = manager.create(Some("to-delete"), &store).await.unwrap();
    assert_eq!(manager.list(None).await.unwrap().len(), 1);

    manager.delete(&id).await.unwrap();
    assert_eq!(manager.list(None).await.unwrap().len(), 0);
}

#[tokio::test]
async fn test_checkpoint_delete_by_name() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    manager.create(Some("named-cp"), &store).await.unwrap();
    assert_eq!(manager.list(None).await.unwrap().len(), 1);

    manager.delete("named-cp").await.unwrap();
    assert_eq!(manager.list(None).await.unwrap().len(), 0);
}

#[tokio::test]
async fn test_checkpoint_rollback_not_found() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    let result = manager.rollback("non-existent", &store).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_checkpoint_delete_not_found() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    let result = manager.delete("non-existent").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_checkpoint_with_multiple_keys() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    use tensor_store::{ScalarValue, TensorData, TensorValue};

    // Add multiple keys
    for i in 0..10 {
        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
        store.put(format!("item:{i}"), data).unwrap();
    }

    // Create checkpoint
    manager.create(Some("multi-key"), &store).await.unwrap();

    // Delete some keys
    for i in 0..5 {
        store.delete(&format!("item:{i}")).unwrap();
    }

    // Verify deletion
    assert!(!store.exists("item:0"));
    assert!(store.exists("item:5"));

    // Rollback
    manager.rollback("multi-key", &store).await.unwrap();

    // Verify all keys restored
    for i in 0..10 {
        assert!(store.exists(&format!("item:{i}")));
    }
}

#[tokio::test]
async fn test_checkpoint_list_with_limit() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

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
async fn test_checkpoint_auto_naming() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

    // Create checkpoint without name
    let id = manager.create(None, &store).await.unwrap();
    assert!(!id.is_empty());

    let list = manager.list(None).await.unwrap();
    assert_eq!(list.len(), 1);
    assert!(list[0].name.starts_with("checkpoint-"));
}

#[tokio::test]
async fn test_checkpoint_confirmation_handler() {
    use tensor_checkpoint::AutoConfirm;

    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let mut manager = CheckpointManager::new(blob, config).await;

    manager.set_confirmation_handler(Arc::new(AutoConfirm));

    let op = DestructiveOp::Delete {
        table: "test".to_string(),
        row_count: 1,
    };
    let preview = OperationPreview::empty("test");

    assert!(manager.request_confirmation(&op, &preview));
}

#[tokio::test]
async fn test_checkpoint_preview_generation() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default();
    let manager = CheckpointManager::new(blob, config).await;

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
async fn test_checkpoint_config_accessors() {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));

    let config = CheckpointConfig::default()
        .with_auto_checkpoint(false)
        .with_interactive_confirm(false);
    let manager = CheckpointManager::new(blob, config).await;

    assert!(!manager.auto_checkpoint_enabled());
    assert!(!manager.interactive_confirm_enabled());
}
