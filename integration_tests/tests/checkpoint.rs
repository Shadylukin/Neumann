// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for the checkpoint/rollback system.

use std::sync::Arc;

use tensor_checkpoint::{
    CheckpointConfig, CheckpointManager, DestructiveOp, FileCheckpointStore, OperationPreview,
};
use tensor_store::TensorStore;

fn setup(max_checkpoints: usize) -> (CheckpointManager, TensorStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
    let config = CheckpointConfig::default().with_max_checkpoints(max_checkpoints);
    let manager = CheckpointManager::new(file_store, config);
    let store = TensorStore::new();
    (manager, store, dir)
}

#[test]
fn test_checkpoint_manager_create_and_list() {
    let (manager, store, _dir) = setup(10);

    let id = manager.create(Some("test-cp"), &store).unwrap();
    assert!(!id.is_empty());

    let list = manager.list(None).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "test-cp");
}

#[test]
fn test_checkpoint_manager_rollback() {
    let (manager, store, _dir) = setup(10);

    use tensor_store::{ScalarValue, TensorData, TensorValue};
    let mut data = TensorData::new();
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String("Alice".to_string())),
    );
    store.put("user:1", data).unwrap();

    let id = manager.create(Some("before-delete"), &store).unwrap();

    store.delete("user:1").unwrap();
    assert!(!store.exists("user:1"));

    manager.rollback(&id, &store).unwrap();

    assert!(store.exists("user:1"));
    let restored = store.get("user:1").unwrap();
    assert_eq!(
        restored.get("name"),
        Some(&TensorValue::Scalar(ScalarValue::String(
            "Alice".to_string()
        )))
    );
}

#[test]
fn test_checkpoint_manager_rollback_by_name() {
    let (manager, store, _dir) = setup(10);

    use tensor_store::{ScalarValue, TensorData, TensorValue};
    let mut data = TensorData::new();
    data.set(
        "val",
        TensorValue::Scalar(ScalarValue::String("original".to_string())),
    );
    store.put("key", data).unwrap();

    manager.create(Some("named-cp"), &store).unwrap();

    store.delete("key").unwrap();

    manager.rollback("named-cp", &store).unwrap();

    assert!(store.exists("key"));
}

#[test]
fn test_checkpoint_auto_checkpoint() {
    let (manager, store, _dir) = setup(10);

    let op = DestructiveOp::Delete {
        table: "users".to_string(),
        row_count: 10,
    };
    let preview = OperationPreview::new("Deleting 10 rows".to_string(), vec![], 10);

    let id = manager
        .create_auto("DELETE FROM users", op, preview, &store)
        .unwrap();
    assert!(!id.is_empty());

    let list = manager.list(None).unwrap();
    assert_eq!(list.len(), 1);
    assert!(list[0].name.starts_with("auto-before-"));
}

#[test]
fn test_checkpoint_retention() {
    let (manager, store, _dir) = setup(3);

    for i in 0..5 {
        manager.create(Some(&format!("cp-{i}")), &store).unwrap();
    }

    let list = manager.list(None).unwrap();
    assert_eq!(list.len(), 3);
}

#[test]
fn test_checkpoint_delete() {
    let (manager, store, _dir) = setup(10);

    let id = manager.create(Some("to-delete"), &store).unwrap();
    assert_eq!(manager.list(None).unwrap().len(), 1);

    manager.delete(&id).unwrap();
    assert_eq!(manager.list(None).unwrap().len(), 0);
}

#[test]
fn test_checkpoint_delete_by_name() {
    let (manager, store, _dir) = setup(10);

    manager.create(Some("named-cp"), &store).unwrap();
    assert_eq!(manager.list(None).unwrap().len(), 1);

    manager.delete("named-cp").unwrap();
    assert_eq!(manager.list(None).unwrap().len(), 0);
}

#[test]
fn test_checkpoint_rollback_not_found() {
    let (manager, store, _dir) = setup(10);

    let result = manager.rollback("non-existent", &store);
    assert!(result.is_err());
}

#[test]
fn test_checkpoint_delete_not_found() {
    let (manager, _store, _dir) = setup(10);

    let result = manager.delete("non-existent");
    assert!(result.is_err());
}

#[test]
fn test_checkpoint_with_multiple_keys() {
    let (manager, store, _dir) = setup(10);

    use tensor_store::{ScalarValue, TensorData, TensorValue};

    for i in 0..10 {
        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
        store.put(format!("item:{i}"), data).unwrap();
    }

    manager.create(Some("multi-key"), &store).unwrap();

    for i in 0..5 {
        store.delete(&format!("item:{i}")).unwrap();
    }

    assert!(!store.exists("item:0"));
    assert!(store.exists("item:5"));

    manager.rollback("multi-key", &store).unwrap();

    for i in 0..10 {
        assert!(store.exists(&format!("item:{i}")));
    }
}

#[test]
fn test_checkpoint_list_with_limit() {
    let (manager, store, _dir) = setup(10);

    for i in 0..5 {
        manager.create(Some(&format!("cp-{i}")), &store).unwrap();
    }

    let list = manager.list(Some(3)).unwrap();
    assert_eq!(list.len(), 3);
}

#[test]
fn test_checkpoint_auto_naming() {
    let (manager, store, _dir) = setup(10);

    let id = manager.create(None, &store).unwrap();
    assert!(!id.is_empty());

    let list = manager.list(None).unwrap();
    assert_eq!(list.len(), 1);
    assert!(list[0].name.starts_with("checkpoint-"));
}

#[test]
fn test_checkpoint_confirmation_handler() {
    use tensor_checkpoint::AutoConfirm;

    let (manager, _store, _dir) = setup(10);

    manager.set_confirmation_handler(Arc::new(AutoConfirm));

    let op = DestructiveOp::Delete {
        table: "test".to_string(),
        row_count: 1,
    };
    let preview = OperationPreview::empty("test");

    assert!(manager.request_confirmation(&op, &preview));
}

#[test]
fn test_checkpoint_preview_generation() {
    let (manager, _store, _dir) = setup(10);

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
fn test_checkpoint_config_accessors() {
    let dir = tempfile::tempdir().unwrap();
    let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
    let config = CheckpointConfig::default()
        .with_auto_checkpoint(false)
        .with_interactive_confirm(false);
    let manager = CheckpointManager::new(file_store, config);

    assert!(!manager.auto_checkpoint_enabled());
    assert!(!manager.interactive_confirm_enabled());
}
