// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! File-based checkpoint storage backend.
//!
//! Each checkpoint is stored as a single `.ncp` file with a two-part layout:
//! ```text
//! [header_len: u32 LE]
//! [header: bitcode(CheckpointFileHeader)]
//! [body: bitcode(CheckpointBody)]
//! ```
//!
//! `list()` reads only the header portion for efficiency. `load()` reads
//! the entire file to reconstruct a `CheckpointState`.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    checkpoint_store::CheckpointStore,
    error::{CheckpointError, Result},
    state::{CheckpointInfo, CheckpointMetadata, CheckpointState, CheckpointTrigger},
};

/// Header written at the start of each `.ncp` file.
///
/// Read by `list()` without touching the body.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointFileHeader {
    id: String,
    name: String,
    created_at: u64,
    trigger: Option<CheckpointTrigger>,
    metadata: CheckpointMetadata,
    body_size: u64,
}

/// Body of a `.ncp` file containing the serialized store snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointBody {
    store_snapshot: Vec<u8>,
}

/// File-based checkpoint storage.
///
/// Stores each checkpoint as a `{id}.ncp` file under a directory.
/// Writes are atomic via temp file + rename.
pub struct FileCheckpointStore {
    dir: PathBuf,
}

impl FileCheckpointStore {
    /// Create a new file store, creating the directory if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns `CheckpointError::Storage` if the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .map_err(|e| CheckpointError::Storage(format!("create dir {}: {e}", dir.display())))?;
        Ok(Self { dir })
    }

    /// Read just the header from an `.ncp` file.
    fn read_header(path: &Path) -> Result<CheckpointFileHeader> {
        let data = fs::read(path)
            .map_err(|e| CheckpointError::Storage(format!("read {}: {e}", path.display())))?;

        if data.len() < 4 {
            return Err(CheckpointError::Deserialization(
                "file too small for header length".to_string(),
            ));
        }

        #[allow(clippy::cast_possible_truncation)]
        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + header_len {
            return Err(CheckpointError::Deserialization(
                "file truncated before header end".to_string(),
            ));
        }

        let header: CheckpointFileHeader = bitcode::deserialize(&data[4..4 + header_len])
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;

        Ok(header)
    }

    /// Read the full checkpoint state from an `.ncp` file.
    fn read_full(path: &Path) -> Result<CheckpointState> {
        let data = fs::read(path)
            .map_err(|e| CheckpointError::Storage(format!("read {}: {e}", path.display())))?;

        if data.len() < 4 {
            return Err(CheckpointError::Deserialization(
                "file too small".to_string(),
            ));
        }

        #[allow(clippy::cast_possible_truncation)]
        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let header_end = 4 + header_len;

        if data.len() < header_end {
            return Err(CheckpointError::Deserialization(
                "file truncated before header end".to_string(),
            ));
        }

        let header: CheckpointFileHeader = bitcode::deserialize(&data[4..header_end])
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;

        let body: CheckpointBody = bitcode::deserialize(&data[header_end..])
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;

        Ok(CheckpointState {
            id: header.id,
            name: header.name,
            created_at: header.created_at,
            trigger: header.trigger,
            store_snapshot: body.store_snapshot,
            metadata: header.metadata,
        })
    }

    /// Find the `.ncp` file matching an ID or name.
    ///
    /// Exact ID match takes priority. For name matches, the newest
    /// (by `created_at` DESC, then `id` ASC) is returned.
    fn find_file(&self, id_or_name: &str) -> Result<PathBuf> {
        let direct = self.dir.join(format!("{id_or_name}.ncp"));
        if direct.exists() {
            return Ok(direct);
        }

        // Scan headers for name match
        let mut best: Option<(PathBuf, u64, String)> = None;

        for entry in self.iter_ncp_files()? {
            if let Ok(header) = Self::read_header(&entry) {
                if header.id == id_or_name || header.name == id_or_name {
                    let dominated = best.as_ref().map_or(true, |(_, ts, id)| {
                        header.created_at > *ts || (header.created_at == *ts && header.id < *id)
                    });
                    if dominated {
                        best = Some((entry, header.created_at, header.id));
                    }
                }
            }
        }

        best.map(|(path, _, _)| path)
            .ok_or_else(|| CheckpointError::NotFound(id_or_name.to_string()))
    }

    /// Iterate over all `.ncp` files in the directory.
    fn iter_ncp_files(&self) -> Result<Vec<PathBuf>> {
        let entries = fs::read_dir(&self.dir).map_err(|e| {
            CheckpointError::Storage(format!("read dir {}: {e}", self.dir.display()))
        })?;

        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| CheckpointError::Storage(e.to_string()))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ncp") {
                files.push(path);
            }
        }
        Ok(files)
    }
}

impl CheckpointStore for FileCheckpointStore {
    fn store(&self, state: &CheckpointState) -> Result<String> {
        let header = CheckpointFileHeader {
            id: state.id.clone(),
            name: state.name.clone(),
            created_at: state.created_at,
            trigger: state.trigger.clone(),
            metadata: state.metadata.clone(),
            body_size: state.store_snapshot.len() as u64,
        };

        let body = CheckpointBody {
            store_snapshot: state.store_snapshot.clone(),
        };

        let header_bytes = bitcode::serialize(&header)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        let body_bytes =
            bitcode::serialize(&body).map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let header_len = header_bytes.len() as u32;

        let file_path = self.dir.join(format!("{}.ncp", state.id));
        let temp_path = self.dir.join(format!("{}.ncp.tmp", state.id));

        // Atomic write: temp file + rename
        {
            let mut file = fs::File::create(&temp_path).map_err(|e| {
                CheckpointError::Storage(format!("create temp {}: {e}", temp_path.display()))
            })?;
            file.write_all(&header_len.to_le_bytes())
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            file.write_all(&header_bytes)
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            file.write_all(&body_bytes)
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            file.flush()
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        }

        fs::rename(&temp_path, &file_path).map_err(|e| {
            CheckpointError::Storage(format!(
                "rename {} -> {}: {e}",
                temp_path.display(),
                file_path.display()
            ))
        })?;

        Ok(state.id.clone())
    }

    fn load(&self, id_or_name: &str) -> Result<CheckpointState> {
        let path = self.find_file(id_or_name)?;
        Self::read_full(&path)
    }

    fn list(&self, limit: Option<usize>) -> Result<Vec<CheckpointInfo>> {
        let files = self.iter_ncp_files()?;
        let mut infos = Vec::with_capacity(files.len());

        for path in &files {
            if let Ok(header) = Self::read_header(path) {
                let file_size = fs::metadata(path)
                    .map(|m| {
                        #[allow(clippy::cast_possible_truncation)]
                        let size = m.len() as usize;
                        size
                    })
                    .unwrap_or(0);

                infos.push(CheckpointInfo {
                    id: header.id.clone(),
                    name: header.name,
                    created_at: header.created_at,
                    size: file_size,
                    trigger: header
                        .trigger
                        .as_ref()
                        .map(|t| t.operation.operation_name().to_string()),
                });
            }
        }

        // Sort by created_at descending, then id ascending for determinism
        infos.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });

        if let Some(limit) = limit {
            infos.truncate(limit);
        }

        Ok(infos)
    }

    fn delete(&self, id_or_name: &str) -> Result<()> {
        let path = self.find_file(id_or_name)?;
        fs::remove_file(&path)
            .map_err(|e| CheckpointError::Storage(format!("delete {}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        CheckpointMetadata, DestructiveOp, GraphMeta, OperationPreview, RelationalMeta, VectorMeta,
    };

    fn make_state(id: &str, name: &str, data: &[u8]) -> CheckpointState {
        CheckpointState::new(
            id.to_string(),
            name.to_string(),
            data.to_vec(),
            CheckpointMetadata::default(),
        )
    }

    #[test]
    fn test_store_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        let state = make_state("id-1", "first", &[1, 2, 3, 4]);
        let returned_id = store.store(&state).unwrap();
        assert_eq!(returned_id, "id-1");

        let loaded = store.load("id-1").unwrap();
        assert_eq!(loaded.id, "id-1");
        assert_eq!(loaded.name, "first");
        assert_eq!(loaded.store_snapshot, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_load_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        let state = make_state("uuid-abc", "my-checkpoint", &[10, 20]);
        store.store(&state).unwrap();

        let loaded = store.load("my-checkpoint").unwrap();
        assert_eq!(loaded.id, "uuid-abc");
    }

    #[test]
    fn test_list_sorted_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        // Create states with different timestamps
        let mut s1 = make_state("id-1", "first", &[1]);
        s1.created_at = 1000;
        let mut s2 = make_state("id-2", "second", &[2]);
        s2.created_at = 2000;
        let mut s3 = make_state("id-3", "third", &[3]);
        s3.created_at = 1500;

        store.store(&s1).unwrap();
        store.store(&s2).unwrap();
        store.store(&s3).unwrap();

        let list = store.list(None).unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].id, "id-2"); // newest
        assert_eq!(list[1].id, "id-3");
        assert_eq!(list[2].id, "id-1"); // oldest
    }

    #[test]
    fn test_list_with_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        for i in 0u32..5 {
            let mut s = make_state(&format!("id-{i}"), &format!("cp-{i}"), &[i as u8]);
            s.created_at = u64::from(i) * 1000;
            store.store(&s).unwrap();
        }

        let list = store.list(Some(2)).unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_delete_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        store.store(&make_state("id-1", "first", &[1])).unwrap();
        store.store(&make_state("id-2", "second", &[2])).unwrap();

        store.delete("id-1").unwrap();

        let list = store.list(None).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "id-2");
    }

    #[test]
    fn test_delete_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        store.store(&make_state("id-1", "first", &[1])).unwrap();
        store.delete("first").unwrap();

        let list = store.list(None).unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        let result = store.load("nonexistent");
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_store_with_trigger_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        let trigger = CheckpointTrigger::new(
            "DELETE FROM users".to_string(),
            DestructiveOp::Delete {
                table: "users".to_string(),
                row_count: 42,
            },
            OperationPreview::new("Deleting 42 rows".to_string(), vec![], 42),
        );

        let metadata = CheckpointMetadata::new(
            RelationalMeta::new(3, 100),
            GraphMeta::new(10, 25),
            VectorMeta::new(50),
            500,
        );

        let state = CheckpointState::new(
            "trigger-test".to_string(),
            "auto-before-delete".to_string(),
            vec![1, 2, 3],
            metadata,
        )
        .with_trigger(trigger);

        store.store(&state).unwrap();

        // Load preserves trigger and metadata
        let loaded = store.load("trigger-test").unwrap();
        assert!(loaded.trigger.is_some());
        let t = loaded.trigger.unwrap();
        assert_eq!(t.command, "DELETE FROM users");
        assert_eq!(loaded.metadata.relational.table_count, 3);
        assert_eq!(loaded.metadata.graph.edge_count, 25);
        assert_eq!(loaded.metadata.vector.embedding_count, 50);

        // List shows trigger label
        let list = store.list(None).unwrap();
        assert_eq!(list[0].trigger, Some("DELETE".to_string()));
    }

    #[test]
    fn test_atomic_write_no_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        // Store should be atomic — file exists only after successful write
        let state = make_state("atomic-test", "atomic", &vec![0u8; 10_000]);
        store.store(&state).unwrap();

        // No temp file should remain
        let temps: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("tmp"))
            .collect();
        assert!(temps.is_empty(), "temp file should not remain after store");

        // The .ncp file should be valid
        let loaded = store.load("atomic-test").unwrap();
        assert_eq!(loaded.store_snapshot.len(), 10_000);
    }

    #[test]
    fn test_snapshot_sizes_stable() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path()).unwrap();

        // Create 3 checkpoints with same-sized data
        let data = vec![0u8; 1000];
        for i in 0u32..3 {
            let mut s = make_state(&format!("id-{i}"), &format!("cp-{i}"), &data);
            s.created_at = u64::from(i) * 1000;
            store.store(&s).unwrap();
        }

        let list = store.list(None).unwrap();
        assert_eq!(list.len(), 3);

        // All sizes should be similar (no unbounded growth)
        let sizes: Vec<usize> = list.iter().map(|info| info.size).collect();
        let max = *sizes.iter().max().unwrap();
        let min = *sizes.iter().min().unwrap();
        assert!(max - min < 100, "checkpoint sizes diverge: {sizes:?}");
    }

    #[test]
    fn test_creates_directory_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("checkpoints");
        let store = FileCheckpointStore::new(&nested).unwrap();

        store.store(&make_state("id-1", "test", &[1])).unwrap();
        assert!(nested.join("id-1.ncp").exists());
    }
}
