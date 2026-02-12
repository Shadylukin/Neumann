// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use tensor_blob::BlobStore;

use crate::{error::Result, storage::CheckpointStorage};

pub struct RetentionManager {
    max_checkpoints: usize,
}

impl RetentionManager {
    pub fn new(max_checkpoints: usize) -> Self {
        Self { max_checkpoints }
    }

    /// Enforce retention policy by deleting oldest checkpoints beyond the limit.
    /// Returns the number of checkpoints deleted.
    pub async fn enforce(&self, blob: &BlobStore) -> Result<usize> {
        let checkpoints = CheckpointStorage::list(blob).await?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(0);
        }

        let to_remove = checkpoints.len() - self.max_checkpoints;
        let mut removed = 0;

        // Checkpoints are sorted by created_at descending, so oldest are at the end
        for checkpoint in checkpoints.iter().rev().take(to_remove) {
            if CheckpointStorage::delete(&checkpoint.artifact_id, blob)
                .await
                .is_ok()
            {
                removed += 1;
            }
        }

        Ok(removed)
    }

    pub fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }
}

#[cfg(test)]
mod tests {
    use tensor_blob::BlobConfig;
    use tensor_store::TensorStore;

    use super::*;
    use crate::state::{CheckpointMetadata, CheckpointState};

    async fn setup() -> BlobStore {
        let store = TensorStore::new();
        BlobStore::new(store, BlobConfig::default()).await.unwrap()
    }

    #[tokio::test]
    async fn test_no_deletion_under_limit() {
        let blob = setup().await;
        let retention = RetentionManager::new(5);

        for i in 0..3 {
            let state = CheckpointState::new(
                format!("id-{i}"),
                format!("checkpoint-{i}"),
                vec![i as u8],
                CheckpointMetadata::default(),
            );
            CheckpointStorage::store(&state, &blob).await.unwrap();
        }

        let deleted = retention.enforce(&blob).await.unwrap();
        assert_eq!(deleted, 0);

        let list = CheckpointStorage::list(&blob).await.unwrap();
        assert_eq!(list.len(), 3);
    }

    #[tokio::test]
    async fn test_deletion_at_limit() {
        let blob = setup().await;
        let retention = RetentionManager::new(2);

        for i in 0..5 {
            let state = CheckpointState::new(
                format!("id-{i}"),
                format!("checkpoint-{i}"),
                vec![i as u8],
                CheckpointMetadata::default(),
            );
            CheckpointStorage::store(&state, &blob).await.unwrap();
        }

        let deleted = retention.enforce(&blob).await.unwrap();
        assert_eq!(deleted, 3);

        let list = CheckpointStorage::list(&blob).await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_keeps_one() {
        let blob = setup().await;
        let retention = RetentionManager::new(1);

        for i in 0..3 {
            let state = CheckpointState::new(
                format!("id-{i}"),
                format!("checkpoint-{i}"),
                vec![i as u8],
                CheckpointMetadata::default(),
            );
            CheckpointStorage::store(&state, &blob).await.unwrap();
        }

        retention.enforce(&blob).await.unwrap();

        let list = CheckpointStorage::list(&blob).await.unwrap();
        assert_eq!(list.len(), 1);
        // One checkpoint remains (order not guaranteed with same timestamps)
        assert!(list[0].id.starts_with("id-"));
    }
}
