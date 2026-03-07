// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use crate::{checkpoint_store::CheckpointStore, error::Result};

/// Enforces a count-based retention policy, deleting the oldest checkpoints beyond the limit.
pub struct RetentionManager {
    max_checkpoints: usize,
}

impl RetentionManager {
    /// Create a retention manager that keeps at most `max_checkpoints` checkpoints.
    #[must_use]
    pub const fn new(max_checkpoints: usize) -> Self {
        Self { max_checkpoints }
    }

    /// Enforce retention policy by deleting oldest checkpoints beyond the limit.
    /// Returns the number of checkpoints deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint list cannot be read or a checkpoint cannot be deleted.
    pub fn enforce(&self, store: &dyn CheckpointStore) -> Result<usize> {
        let checkpoints = store.list(None)?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(0);
        }

        let to_remove = checkpoints.len() - self.max_checkpoints;
        let mut removed = 0;

        // Checkpoints are sorted by created_at descending, so oldest are at the end
        for checkpoint in checkpoints.iter().rev().take(to_remove) {
            if store.delete(&checkpoint.id).is_ok() {
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Returns the configured maximum number of checkpoints to retain.
    #[must_use]
    pub const fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        file_store::FileCheckpointStore,
        state::{CheckpointMetadata, CheckpointState},
    };

    fn setup() -> (FileCheckpointStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();
        (store, dir)
    }

    fn make_state(id: &str, name: &str, ts: u64) -> CheckpointState {
        let mut s = CheckpointState::new(
            id.to_string(),
            name.to_string(),
            vec![id.len() as u8],
            CheckpointMetadata::default(),
        );
        s.created_at = ts;
        s
    }

    #[test]
    fn test_no_deletion_under_limit() {
        let (store, _dir) = setup();
        let retention = RetentionManager::new(5);

        for i in 0..3u64 {
            store
                .store(&make_state(
                    &format!("id-{i}"),
                    &format!("cp-{i}"),
                    i * 1000,
                ))
                .unwrap();
        }

        let deleted = retention.enforce(&store).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.list(None).unwrap().len(), 3);
    }

    #[test]
    fn test_deletion_at_limit() {
        let (store, _dir) = setup();
        let retention = RetentionManager::new(2);

        for i in 0..5u64 {
            store
                .store(&make_state(
                    &format!("id-{i}"),
                    &format!("cp-{i}"),
                    i * 1000,
                ))
                .unwrap();
        }

        let deleted = retention.enforce(&store).unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(store.list(None).unwrap().len(), 2);
    }

    #[test]
    fn test_keeps_one() {
        let (store, _dir) = setup();
        let retention = RetentionManager::new(1);

        for i in 0..3u64 {
            store
                .store(&make_state(
                    &format!("id-{i}"),
                    &format!("cp-{i}"),
                    i * 1000,
                ))
                .unwrap();
        }

        retention.enforce(&store).unwrap();
        let list = store.list(None).unwrap();
        assert_eq!(list.len(), 1);
        // Should keep the newest
        assert_eq!(list[0].id, "id-2");
    }
}
