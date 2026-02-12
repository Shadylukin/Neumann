// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use tensor_blob::{BlobStore, PutOptions};

use crate::{
    error::{CheckpointError, Result},
    state::{CheckpointInfo, CheckpointState},
};

const CHECKPOINT_TAG: &str = "_system:checkpoint";
const CHECKPOINT_CONTENT_TYPE: &str = "application/x-neumann-checkpoint";

pub struct CheckpointStorage;

impl CheckpointStorage {
    pub async fn store(state: &CheckpointState, blob: &BlobStore) -> Result<String> {
        let data =
            bitcode::serialize(state).map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        let filename = format!("checkpoint_{}.ncp", state.id);

        let trigger_desc = state
            .trigger
            .as_ref()
            .map(|t| t.operation.operation_name().to_string());

        let mut options = PutOptions::new()
            .with_content_type(CHECKPOINT_CONTENT_TYPE)
            .with_tag(CHECKPOINT_TAG)
            .with_meta("checkpoint_id", &state.id)
            .with_meta("checkpoint_name", &state.name)
            .with_meta("created_at", state.created_at.to_string())
            .with_created_by("system:checkpoint");

        if let Some(trigger) = &trigger_desc {
            options = options.with_meta("trigger", trigger);
        }

        let artifact_id = blob
            .put(&filename, &data, options)
            .await
            .map_err(CheckpointError::Blob)?;

        Ok(artifact_id)
    }

    pub async fn load(checkpoint_id: &str, blob: &BlobStore) -> Result<CheckpointState> {
        let artifact_id = Self::find_by_id_or_name(checkpoint_id, blob).await?;

        let data = blob
            .get(&artifact_id)
            .await
            .map_err(CheckpointError::Blob)?;

        let state: CheckpointState = bitcode::deserialize(&data)
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;

        Ok(state)
    }

    pub async fn list(blob: &BlobStore) -> Result<Vec<CheckpointInfo>> {
        let artifact_ids = blob
            .by_tag(CHECKPOINT_TAG)
            .await
            .map_err(CheckpointError::Blob)?;

        let mut checkpoints = Vec::new();
        for artifact_id in artifact_ids {
            if let Ok(meta) = blob.metadata(&artifact_id).await {
                let info = CheckpointInfo {
                    id: meta
                        .custom
                        .get("checkpoint_id")
                        .cloned()
                        .unwrap_or_default(),
                    name: meta
                        .custom
                        .get("checkpoint_name")
                        .cloned()
                        .unwrap_or_default(),
                    created_at: meta
                        .custom
                        .get("created_at")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(meta.created),
                    artifact_id: artifact_id.clone(),
                    size: meta.size,
                    trigger: meta.custom.get("trigger").cloned(),
                };
                checkpoints.push(info);
            }
        }

        checkpoints.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(checkpoints)
    }

    pub async fn delete(artifact_id: &str, blob: &BlobStore) -> Result<()> {
        blob.delete(artifact_id)
            .await
            .map_err(CheckpointError::Blob)
    }

    async fn find_by_id_or_name(id_or_name: &str, blob: &BlobStore) -> Result<String> {
        let checkpoints = Self::list(blob).await?;

        for cp in checkpoints {
            if cp.id == id_or_name || cp.name == id_or_name {
                return Ok(cp.artifact_id);
            }
        }

        Err(CheckpointError::NotFound(id_or_name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use tensor_blob::BlobConfig;
    use tensor_store::TensorStore;

    use super::*;
    use crate::state::{CheckpointMetadata, CheckpointTrigger, DestructiveOp, OperationPreview};

    async fn setup() -> BlobStore {
        let store = TensorStore::new();
        BlobStore::new(store, BlobConfig::default()).await.unwrap()
    }

    #[tokio::test]
    async fn test_store_and_load_checkpoint() {
        let blob = setup().await;

        let state = CheckpointState::new(
            "test-id-123".to_string(),
            "test-checkpoint".to_string(),
            vec![1, 2, 3, 4],
            CheckpointMetadata::default(),
        );

        let artifact_id = CheckpointStorage::store(&state, &blob).await.unwrap();
        assert!(!artifact_id.is_empty());

        let loaded = CheckpointStorage::load("test-id-123", &blob).await.unwrap();
        assert_eq!(loaded.id, "test-id-123");
        assert_eq!(loaded.name, "test-checkpoint");
        assert_eq!(loaded.store_snapshot, vec![1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn test_list_checkpoints() {
        let blob = setup().await;

        let state1 = CheckpointState::new(
            "id-1".to_string(),
            "first".to_string(),
            vec![1],
            CheckpointMetadata::default(),
        );

        let state2 = CheckpointState::new(
            "id-2".to_string(),
            "second".to_string(),
            vec![2],
            CheckpointMetadata::default(),
        );

        CheckpointStorage::store(&state1, &blob).await.unwrap();
        CheckpointStorage::store(&state2, &blob).await.unwrap();

        let list = CheckpointStorage::list(&blob).await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_store_with_trigger() {
        let blob = setup().await;

        let trigger = CheckpointTrigger::new(
            "DELETE FROM users".to_string(),
            DestructiveOp::Delete {
                table: "users".to_string(),
                row_count: 10,
            },
            OperationPreview::new("Deleting 10 rows".to_string(), vec![], 10),
        );

        let state = CheckpointState::new(
            "trigger-test".to_string(),
            "auto-before-delete".to_string(),
            vec![],
            CheckpointMetadata::default(),
        )
        .with_trigger(trigger);

        CheckpointStorage::store(&state, &blob).await.unwrap();

        let list = CheckpointStorage::list(&blob).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].trigger, Some("DELETE".to_string()));
    }

    #[tokio::test]
    async fn test_find_by_name() {
        let blob = setup().await;

        let state = CheckpointState::new(
            "uuid-123".to_string(),
            "my-named-checkpoint".to_string(),
            vec![5, 6, 7],
            CheckpointMetadata::default(),
        );

        CheckpointStorage::store(&state, &blob).await.unwrap();

        let loaded = CheckpointStorage::load("my-named-checkpoint", &blob)
            .await
            .unwrap();
        assert_eq!(loaded.id, "uuid-123");
    }

    #[tokio::test]
    async fn test_not_found() {
        let blob = setup().await;

        let result = CheckpointStorage::load("nonexistent", &blob).await;
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }
}
