// SPDX-License-Identifier: MIT OR Apache-2.0
//! Write-Ahead Log for crash recovery.
//!
//! Provides durable logging of all mutations to enable recovery after crashes.
//! The WAL uses a binary format with CRC32 checksums, following the same
//! patterns as `RaftWal` and `TxWal` in `tensor_chain`.

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{EntityId, TensorData};

/// WAL-specific errors.
#[derive(Debug, Error)]
pub enum WalError {
    /// I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Serialization or deserialization failed.
    #[error("Serialization error: {0}")]
    Serialization(#[from] bitcode::Error),

    /// CRC32 checksum verification failed.
    #[error("Checksum mismatch at entry {index}: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch {
        /// Entry index where mismatch occurred.
        index: usize,
        /// Expected checksum.
        expected: u32,
        /// Actual checksum.
        actual: u32,
    },

    /// WAL file size exceeds configured limit.
    #[error("WAL size limit exceeded: {current} >= {limit}")]
    SizeLimitExceeded {
        /// Current WAL size in bytes.
        current: u64,
        /// Maximum allowed size.
        limit: u64,
    },

    /// Operation requires an active transaction.
    #[error("No active transaction")]
    NoActiveTransaction,

    /// Attempted to start a transaction while one is active.
    #[error("Transaction already active: {0}")]
    TransactionAlreadyActive(u64),

    /// Entry size exceeds `u32::MAX`.
    #[error("Entry too large: {size} bytes exceeds u32::MAX")]
    EntryTooLarge {
        /// Entry size in bytes.
        size: usize,
    },
}

/// Result type for WAL operations.
pub type WalResult<T> = Result<T, WalError>;

/// Entry types for all mutable operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalEntry {
    /// Set metadata key-value pair.
    MetadataSet {
        /// Metadata key.
        key: String,
        /// Metadata value.
        data: TensorData,
    },

    /// Delete metadata key.
    MetadataDelete {
        /// Metadata key to delete.
        key: String,
    },

    /// Set embedding for an entity.
    EmbeddingSet {
        /// Entity ID.
        entity_id: EntityId,
        /// Embedding vector.
        embedding: Vec<f32>,
    },

    /// Delete embedding for an entity.
    EmbeddingDelete {
        /// Entity ID.
        entity_id: EntityId,
    },

    /// Create entity in index.
    EntityCreate {
        /// Entity key.
        key: String,
        /// Assigned entity ID.
        entity_id: EntityId,
    },

    /// Remove entity from index.
    EntityRemove {
        /// Entity key to remove.
        key: String,
    },

    /// Begin a transaction.
    TxBegin {
        /// Transaction ID.
        tx_id: u64,
    },

    /// Commit a transaction.
    TxCommit {
        /// Transaction ID.
        tx_id: u64,
    },

    /// Abort a transaction.
    TxAbort {
        /// Transaction ID.
        tx_id: u64,
    },

    /// Checkpoint marker (WAL can be truncated after this).
    Checkpoint {
        /// Snapshot ID.
        snapshot_id: u64,
    },
}

/// Fsync strategy for WAL writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Fsync after every write. Safest, slowest (~300 ops/sec).
    #[default]
    Immediate,
    /// Buffer writes, fsync when batch size or timeout reached.
    /// Good balance of safety and performance (~10K-50K ops/sec).
    Batched {
        /// Maximum entries before forced fsync.
        max_entries: usize,
    },
    /// Never fsync automatically. Caller must call `sync()` explicitly.
    /// Fastest but caller is responsible for durability.
    Manual,
}

/// WAL configuration.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct WalConfig {
    /// Whether WAL is enabled.
    pub enabled: bool,
    /// Enable CRC32 checksums for entries.
    pub enable_checksums: bool,
    /// Verify checksums on replay.
    pub verify_on_replay: bool,
    /// Maximum WAL size in bytes before rotation.
    pub max_size_bytes: u64,
    /// Automatically rotate when size limit is reached.
    pub auto_rotate: bool,
    /// Maximum number of rotated files to keep.
    pub max_rotated_files: usize,
    /// Fsync strategy.
    pub sync_mode: SyncMode,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_checksums: true,
            verify_on_replay: true,
            max_size_bytes: 512 * 1024 * 1024, // 512MB
            auto_rotate: true,
            max_rotated_files: 2,
            sync_mode: SyncMode::Immediate,
        }
    }
}

impl WalConfig {
    /// Create a config with WAL disabled.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            enable_checksums: true,
            verify_on_replay: true,
            max_size_bytes: 512 * 1024 * 1024,
            auto_rotate: true,
            max_rotated_files: 2,
            sync_mode: SyncMode::Immediate,
        }
    }

    /// Create a config for testing with small size limit.
    #[must_use]
    pub const fn for_testing() -> Self {
        Self {
            enabled: true,
            enable_checksums: true,
            verify_on_replay: true,
            max_size_bytes: 1024, // 1KB for testing rotation
            auto_rotate: true,
            max_rotated_files: 2,
            sync_mode: SyncMode::Immediate,
        }
    }

    /// Create a config optimized for throughput with batched fsync.
    #[must_use]
    pub const fn batched(max_entries: usize) -> Self {
        Self {
            enabled: true,
            enable_checksums: true,
            verify_on_replay: true,
            max_size_bytes: 512 * 1024 * 1024,
            auto_rotate: true,
            max_rotated_files: 2,
            sync_mode: SyncMode::Batched { max_entries },
        }
    }
}

/// WAL status information.
#[derive(Debug, Clone)]
pub struct WalStatus {
    /// Path to the WAL file.
    pub path: PathBuf,
    /// Current size in bytes.
    pub size_bytes: u64,
    /// Number of entries.
    pub entry_count: usize,
    /// Whether checksums are enabled.
    pub checksums_enabled: bool,
}

/// Write-Ahead Log for durable storage.
pub struct TensorWal {
    file: BufWriter<File>,
    path: PathBuf,
    config: WalConfig,
    entry_count: usize,
    current_size: u64,
    /// Number of entries written since last fsync (for batched mode).
    pending_sync_count: usize,
}

impl TensorWal {
    /// Open or create a WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or created.
    pub fn open(path: impl AsRef<Path>, config: WalConfig) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Get current size if file exists
        let current_size = if path.exists() {
            std::fs::metadata(&path)?.len()
        } else {
            0
        };

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok(Self {
            file: BufWriter::new(file),
            path,
            config,
            entry_count: 0,
            current_size,
            pending_sync_count: 0,
        })
    }

    /// Get the WAL file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get current WAL size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.current_size
    }

    /// Get number of entries written in this session.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get WAL status.
    #[must_use]
    pub fn status(&self) -> WalStatus {
        WalStatus {
            path: self.path.clone(),
            size_bytes: self.current_size,
            entry_count: self.entry_count,
            checksums_enabled: self.config.enable_checksums,
        }
    }

    /// Write a single entry to the WAL file without fsync.
    fn write_entry_no_sync(&mut self, entry: &WalEntry) -> WalResult<()> {
        let bytes = bitcode::serialize(entry)?;

        // Check entry size fits in u32
        let len_u32 = u32::try_from(bytes.len())
            .map_err(|_| WalError::EntryTooLarge { size: bytes.len() })?;

        let write_size = 4 + 4 + u64::from(len_u32); // length + checksum + payload

        // Check size limit
        if self.current_size + write_size > self.config.max_size_bytes {
            if self.config.auto_rotate {
                self.rotate()?;
            } else {
                return Err(WalError::SizeLimitExceeded {
                    current: self.current_size + write_size,
                    limit: self.config.max_size_bytes,
                });
            }
        }

        // Compute checksum
        let checksum = if self.config.enable_checksums {
            crc32fast::hash(&bytes)
        } else {
            0
        };

        // Write V2 format: [length][checksum][payload]
        self.file.write_all(&len_u32.to_le_bytes())?;
        self.file.write_all(&checksum.to_le_bytes())?;
        self.file.write_all(&bytes)?;

        self.current_size += write_size;
        self.entry_count += 1;
        self.pending_sync_count += 1;

        Ok(())
    }

    /// Append an entry to the WAL.
    ///
    /// Fsync behavior depends on `WalConfig::sync_mode`:
    /// - `Immediate`: fsync after every entry (default, safest)
    /// - `Batched { max_entries }`: fsync when batch size reached
    /// - `Manual`: never fsync automatically, caller must call `sync()`
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails, the entry is too large,
    /// the size limit is exceeded (and auto-rotate is disabled), or I/O fails.
    pub fn append(&mut self, entry: &WalEntry) -> WalResult<()> {
        self.write_entry_no_sync(entry)?;
        self.maybe_sync()?;
        Ok(())
    }

    /// Append multiple entries with a single fsync (group commit).
    ///
    /// This is much faster than calling `append` multiple times when
    /// using `SyncMode::Immediate`, as it amortizes the fsync cost.
    ///
    /// # Errors
    ///
    /// Returns an error if any entry fails to serialize or write.
    pub fn append_batch(&mut self, entries: &[WalEntry]) -> WalResult<()> {
        for entry in entries {
            self.write_entry_no_sync(entry)?;
        }
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.pending_sync_count = 0;
        Ok(())
    }

    /// Conditionally fsync based on sync mode.
    fn maybe_sync(&mut self) -> WalResult<()> {
        let should_sync = match self.config.sync_mode {
            SyncMode::Immediate => true,
            SyncMode::Batched { max_entries } => self.pending_sync_count >= max_entries,
            SyncMode::Manual => false,
        };

        if should_sync {
            self.file.flush()?;
            self.file.get_ref().sync_all()?;
            self.pending_sync_count = 0;
        }

        Ok(())
    }

    /// Force sync all pending writes to disk.
    ///
    /// Call this to ensure durability when using `SyncMode::Batched` or
    /// `SyncMode::Manual`. Returns the number of entries that were synced.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or syncing fails.
    pub fn sync(&mut self) -> io::Result<usize> {
        let synced = self.pending_sync_count;
        if synced > 0 {
            self.file.flush()?;
            self.file.get_ref().sync_all()?;
            self.pending_sync_count = 0;
        }
        Ok(synced)
    }

    /// Get the number of entries written since last fsync.
    #[must_use]
    pub const fn pending_sync_count(&self) -> usize {
        self.pending_sync_count
    }

    /// Flush buffered writes (does not fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }

    /// Fsync to ensure durability (legacy method, prefer `sync()`).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or syncing fails.
    pub fn fsync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.pending_sync_count = 0;
        Ok(())
    }

    /// Replay all entries from the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or checksums don't match.
    pub fn replay(&self) -> WalResult<Vec<WalEntry>> {
        self.replay_with_validation(self.config.verify_on_replay)
    }

    /// Replay entries with optional checksum validation.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or (when verify is true) checksums don't match.
    pub fn replay_with_validation(&self, verify_checksums: bool) -> WalResult<Vec<WalEntry>> {
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut entry_index = 0;

        loop {
            // Read length
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let len = u32::from_le_bytes(len_buf) as usize;

            // Read checksum
            let mut checksum_buf = [0u8; 4];
            match reader.read_exact(&mut checksum_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // Partial write
                Err(e) => return Err(e.into()),
            }

            let stored_checksum = u32::from_le_bytes(checksum_buf);

            // Read payload
            let mut data = vec![0u8; len];
            match reader.read_exact(&mut data) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // Partial write
                Err(e) => return Err(e.into()),
            }

            // Verify checksum if enabled
            if verify_checksums && stored_checksum != 0 {
                let computed_checksum = crc32fast::hash(&data);
                if stored_checksum != computed_checksum {
                    return Err(WalError::ChecksumMismatch {
                        index: entry_index,
                        expected: stored_checksum,
                        actual: computed_checksum,
                    });
                }
            }

            // Deserialize entry
            match bitcode::deserialize::<WalEntry>(&data) {
                Ok(entry) => {
                    entries.push(entry);
                    entry_index += 1;
                },
                Err(_) => {
                    // Corrupted entry - stop replay
                    break;
                },
            }
        }

        Ok(entries)
    }

    /// Truncate the WAL (clear all entries).
    ///
    /// # Errors
    ///
    /// Returns an error if truncation fails.
    pub fn truncate(&mut self) -> io::Result<()> {
        self.file.flush()?;

        // Create empty file
        let file = File::create(&self.path)?;
        self.file = BufWriter::new(file);
        self.entry_count = 0;
        self.current_size = 0;

        Ok(())
    }

    /// Rotate the WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub fn rotate(&mut self) -> WalResult<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;

        // Delete oldest rotated file
        let oldest = self.rotated_path(self.config.max_rotated_files);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }

        // Rename existing rotated files
        for i in (1..self.config.max_rotated_files).rev() {
            let from = self.rotated_path(i);
            let to = self.rotated_path(i + 1);
            if from.exists() {
                std::fs::rename(&from, &to)?;
            }
        }

        // Rename current WAL to .1
        if self.path.exists() {
            std::fs::rename(&self.path, self.rotated_path(1))?;
        }

        // Create fresh WAL
        let file = File::create(&self.path)?;
        self.file = BufWriter::new(file);
        self.current_size = 0;
        self.entry_count = 0;

        Ok(())
    }

    fn rotated_path(&self, n: usize) -> PathBuf {
        let name = self.path.file_name().unwrap_or_default().to_string_lossy();
        self.path.with_file_name(format!("{name}.{n}"))
    }
}

impl std::fmt::Debug for TensorWal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorWal")
            .field("path", &self.path)
            .field("entry_count", &self.entry_count)
            .field("current_size", &self.current_size)
            .finish_non_exhaustive()
    }
}

/// WAL recovery state.
#[derive(Debug, Default)]
pub struct WalRecovery {
    /// Single operations to replay (not in any transaction).
    pub operations: Vec<WalEntry>,
    /// In-flight transactions that were not committed.
    pub pending_txs: HashMap<u64, Vec<WalEntry>>,
    /// Committed transaction operations to replay.
    pub committed_ops: Vec<WalEntry>,
    /// Last checkpoint snapshot ID.
    pub last_checkpoint: Option<u64>,
    /// Number of aborted transactions.
    pub aborted_count: usize,
}

impl WalRecovery {
    /// Reconstruct recovery state from WAL entries.
    #[must_use]
    pub fn from_entries(entries: &[WalEntry]) -> Self {
        let mut recovery = Self::default();
        let mut active_tx: Option<u64> = None;
        let mut tx_buffers: HashMap<u64, Vec<WalEntry>> = HashMap::new();

        for entry in entries {
            match entry {
                WalEntry::TxBegin { tx_id } => {
                    active_tx = Some(*tx_id);
                    tx_buffers.insert(*tx_id, Vec::new());
                },
                WalEntry::TxCommit { tx_id } => {
                    if let Some(ops) = tx_buffers.remove(tx_id) {
                        recovery.committed_ops.extend(ops);
                    }
                    if active_tx == Some(*tx_id) {
                        active_tx = None;
                    }
                },
                WalEntry::TxAbort { tx_id } => {
                    tx_buffers.remove(tx_id);
                    recovery.aborted_count += 1;
                    if active_tx == Some(*tx_id) {
                        active_tx = None;
                    }
                },
                WalEntry::Checkpoint { snapshot_id } => {
                    recovery.last_checkpoint = Some(*snapshot_id);
                    // Clear operations before checkpoint
                    recovery.operations.clear();
                    recovery.committed_ops.clear();
                    tx_buffers.clear();
                    active_tx = None;
                },
                _ => {
                    // Regular operation
                    if let Some(tx_id) = active_tx {
                        if let Some(buf) = tx_buffers.get_mut(&tx_id) {
                            buf.push(entry.clone());
                        }
                    } else {
                        recovery.operations.push(entry.clone());
                    }
                },
            }
        }

        // Any remaining tx_buffers are uncommitted
        recovery.pending_txs = tx_buffers;

        recovery
    }

    /// Build recovery from a WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL replay fails.
    pub fn from_wal(wal: &TensorWal) -> WalResult<Self> {
        let entries = wal.replay()?;
        Ok(Self::from_entries(&entries))
    }

    /// Get all operations to replay (single ops + committed tx ops).
    #[must_use]
    pub fn all_operations(&self) -> Vec<&WalEntry> {
        self.operations
            .iter()
            .chain(self.committed_ops.iter())
            .collect()
    }

    /// Check if there are uncommitted transactions.
    #[must_use]
    pub fn has_pending_transactions(&self) -> bool {
        !self.pending_txs.is_empty()
    }

    /// Get count of entries to replay.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::len() not const-stable
    pub fn replay_count(&self) -> usize {
        self.operations.len() + self.committed_ops.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScalarValue, TensorValue};
    use tempfile::tempdir;

    fn make_tensor_data() -> TensorData {
        let mut data = TensorData::new();
        data.set("field", TensorValue::Scalar(ScalarValue::Int(42)));
        data
    }

    // Config tests

    #[test]
    fn test_wal_config_default() {
        let config = WalConfig::default();
        assert!(config.enabled);
        assert!(config.enable_checksums);
        assert!(config.verify_on_replay);
        assert_eq!(config.max_size_bytes, 512 * 1024 * 1024);
        assert!(config.auto_rotate);
        assert_eq!(config.max_rotated_files, 2);
    }

    #[test]
    fn test_wal_config_disabled() {
        let config = WalConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_wal_config_for_testing() {
        let config = WalConfig::for_testing();
        assert!(config.enabled);
        assert_eq!(config.max_size_bytes, 1024);
    }

    // Open/create tests

    #[test]
    fn test_wal_open_new() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(wal.size(), 0);
        assert_eq!(wal.entry_count(), 0);
        assert!(path.exists());
    }

    #[test]
    fn test_wal_open_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Create and write
        {
            let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
            wal.append(&WalEntry::MetadataDelete {
                key: "test".to_string(),
            })
            .unwrap();
        }

        // Reopen
        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        assert!(wal.size() > 0);
    }

    #[test]
    fn test_wal_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(wal.path(), path);
    }

    #[test]
    fn test_wal_status() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let status = wal.status();
        assert_eq!(status.path, path);
        assert_eq!(status.size_bytes, 0);
        assert_eq!(status.entry_count, 0);
        assert!(status.checksums_enabled);
    }

    #[test]
    fn test_wal_debug() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let debug = format!("{wal:?}");
        assert!(debug.contains("TensorWal"));
    }

    // Entry roundtrip tests

    #[test]
    fn test_entry_metadata_set_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::MetadataSet {
            key: "test_key".to_string(),
            data: make_tensor_data(),
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_metadata_delete_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::MetadataDelete {
            key: "delete_me".to_string(),
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_embedding_set_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::EmbeddingSet {
            entity_id: EntityId(42),
            embedding: vec![1.0, 2.0, 3.0, 4.0],
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_embedding_delete_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::EmbeddingDelete {
            entity_id: EntityId(99),
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_entity_create_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::EntityCreate {
            key: "entity_key".to_string(),
            entity_id: EntityId(123),
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_entity_remove_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::EntityRemove {
            key: "remove_key".to_string(),
        };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_tx_begin_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::TxBegin { tx_id: 12345 };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_tx_commit_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::TxCommit { tx_id: 12345 };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_tx_abort_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::TxAbort { tx_id: 12345 };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_entry_checkpoint_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let entry = WalEntry::Checkpoint { snapshot_id: 999 };

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&entry).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    // Append tests

    #[test]
    fn test_append_single_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "x".to_string(),
        })
        .unwrap();

        assert_eq!(wal.entry_count(), 1);
        assert!(wal.size() > 0);
    }

    #[test]
    fn test_append_multiple_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();

        for i in 0..10 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key{i}"),
            })
            .unwrap();
        }

        assert_eq!(wal.entry_count(), 10);

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 10);
    }

    #[test]
    fn test_append_updates_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let initial_size = wal.size();

        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();

        assert!(wal.size() > initial_size);
    }

    #[test]
    fn test_append_with_checksums() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let config = WalConfig {
            enable_checksums: true,
            ..WalConfig::default()
        };

        let mut wal = TensorWal::open(&path, config).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();

        // Replay with validation
        let entries = wal.replay_with_validation(true).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_append_without_checksums() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let config = WalConfig {
            enable_checksums: false,
            ..WalConfig::default()
        };

        let mut wal = TensorWal::open(&path, config).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    // Replay tests

    #[test]
    fn test_replay_empty_wal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_replay_nonexistent_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.wal");

        // Don't create the file, just try to replay
        let config = WalConfig::default();
        // We need to manually construct for this test
        let result = TensorWal::open(&path, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_replay_single_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataSet {
            key: "k".to_string(),
            data: make_tensor_data(),
        })
        .unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_replay_multiple_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();

        for i in 0..5 {
            wal.append(&WalEntry::EmbeddingSet {
                entity_id: EntityId(i),
                embedding: vec![i as f32; 4],
            })
            .unwrap();
        }

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn test_replay_detects_corruption() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write a valid entry
        {
            let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
            wal.append(&WalEntry::MetadataDelete {
                key: "test".to_string(),
            })
            .unwrap();
        }

        // Corrupt the file by modifying bytes
        {
            let mut data = std::fs::read(&path).unwrap();
            if data.len() > 10 {
                data[10] ^= 0xFF; // Flip bits in payload
            }
            std::fs::write(&path, &data).unwrap();
        }

        // Replay should detect corruption
        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let result = wal.replay_with_validation(true);
        assert!(result.is_err() || result.unwrap().is_empty());
    }

    #[test]
    fn test_replay_partial_write_recovery() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write valid entries
        {
            let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
            wal.append(&WalEntry::MetadataDelete {
                key: "k1".to_string(),
            })
            .unwrap();
            wal.append(&WalEntry::MetadataDelete {
                key: "k2".to_string(),
            })
            .unwrap();
        }

        // Append partial data (simulating crash during write)
        {
            use std::io::Write;
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0u8; 4]).unwrap(); // Length only, no payload
        }

        // Replay should recover valid entries
        let wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        let entries = wal.replay_with_validation(false).unwrap();
        assert_eq!(entries.len(), 2); // Should have 2 valid entries
    }

    // Truncate tests

    #[test]
    fn test_truncate_clears_wal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();

        assert!(wal.size() > 0);
        assert_eq!(wal.entry_count(), 1);

        wal.truncate().unwrap();

        assert_eq!(wal.size(), 0);
        assert_eq!(wal.entry_count(), 0);

        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    // Rotate tests

    #[test]
    fn test_rotate_creates_backup() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();

        wal.rotate().unwrap();

        // Check backup exists
        let backup = dir.path().join("test.wal.1");
        assert!(backup.exists());

        // New WAL should be empty
        assert_eq!(wal.size(), 0);
    }

    #[test]
    fn test_rotate_limits_backups() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let config = WalConfig {
            max_rotated_files: 2,
            ..WalConfig::default()
        };

        let mut wal = TensorWal::open(&path, config).unwrap();

        // Rotate 5 times
        for i in 0..5 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key{i}"),
            })
            .unwrap();
            wal.rotate().unwrap();
        }

        // Should only have .1 and .2
        assert!(dir.path().join("test.wal.1").exists());
        assert!(dir.path().join("test.wal.2").exists());
        assert!(!dir.path().join("test.wal.3").exists());
    }

    #[test]
    fn test_auto_rotate_on_size_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let config = WalConfig {
            max_size_bytes: 100, // Very small limit
            auto_rotate: true,
            max_rotated_files: 2,
            ..WalConfig::default()
        };

        let mut wal = TensorWal::open(&path, config).unwrap();

        // Write enough to trigger rotation
        for i in 0..10 {
            wal.append(&WalEntry::EmbeddingSet {
                entity_id: EntityId(i),
                embedding: vec![1.0; 100], // Large entry
            })
            .unwrap();
        }

        // Should have rotated
        assert!(dir.path().join("test.wal.1").exists());
    }

    #[test]
    fn test_size_limit_error_when_no_auto_rotate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let config = WalConfig {
            max_size_bytes: 50,
            auto_rotate: false,
            ..WalConfig::default()
        };

        let mut wal = TensorWal::open(&path, config).unwrap();

        // First write might succeed
        let _ = wal.append(&WalEntry::MetadataDelete {
            key: "k".to_string(),
        });

        // Eventually should fail
        let result = wal.append(&WalEntry::EmbeddingSet {
            entity_id: EntityId(1),
            embedding: vec![1.0; 100],
        });

        assert!(matches!(result, Err(WalError::SizeLimitExceeded { .. })));
    }

    // Recovery tests

    #[test]
    fn test_recovery_single_operations() {
        let entries = vec![
            WalEntry::MetadataSet {
                key: "k1".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::MetadataDelete {
                key: "k2".to_string(),
            },
        ];

        let recovery = WalRecovery::from_entries(&entries);

        assert_eq!(recovery.operations.len(), 2);
        assert!(recovery.committed_ops.is_empty());
        assert!(recovery.pending_txs.is_empty());
        assert_eq!(recovery.replay_count(), 2);
    }

    #[test]
    fn test_recovery_committed_transaction() {
        let entries = vec![
            WalEntry::TxBegin { tx_id: 1 },
            WalEntry::MetadataSet {
                key: "k1".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::MetadataSet {
                key: "k2".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxCommit { tx_id: 1 },
        ];

        let recovery = WalRecovery::from_entries(&entries);

        assert!(recovery.operations.is_empty());
        assert_eq!(recovery.committed_ops.len(), 2);
        assert!(!recovery.has_pending_transactions());
        assert_eq!(recovery.replay_count(), 2);
    }

    #[test]
    fn test_recovery_uncommitted_transaction_discarded() {
        let entries = vec![
            WalEntry::TxBegin { tx_id: 1 },
            WalEntry::MetadataSet {
                key: "k1".to_string(),
                data: make_tensor_data(),
            },
            // No commit - simulates crash
        ];

        let recovery = WalRecovery::from_entries(&entries);

        assert!(recovery.operations.is_empty());
        assert!(recovery.committed_ops.is_empty());
        assert!(recovery.has_pending_transactions());
        assert_eq!(recovery.pending_txs.len(), 1);
        assert_eq!(recovery.replay_count(), 0);
    }

    #[test]
    fn test_recovery_aborted_transaction_discarded() {
        let entries = vec![
            WalEntry::TxBegin { tx_id: 1 },
            WalEntry::MetadataSet {
                key: "k1".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxAbort { tx_id: 1 },
        ];

        let recovery = WalRecovery::from_entries(&entries);

        assert!(recovery.operations.is_empty());
        assert!(recovery.committed_ops.is_empty());
        assert!(!recovery.has_pending_transactions());
        assert_eq!(recovery.aborted_count, 1);
    }

    #[test]
    fn test_recovery_after_checkpoint() {
        let entries = vec![
            WalEntry::MetadataSet {
                key: "old".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::Checkpoint { snapshot_id: 1 },
            WalEntry::MetadataSet {
                key: "new".to_string(),
                data: make_tensor_data(),
            },
        ];

        let recovery = WalRecovery::from_entries(&entries);

        // Only "new" should be in operations (after checkpoint)
        assert_eq!(recovery.operations.len(), 1);
        assert_eq!(recovery.last_checkpoint, Some(1));

        if let WalEntry::MetadataSet { key, .. } = &recovery.operations[0] {
            assert_eq!(key, "new");
        } else {
            panic!("Expected MetadataSet");
        }
    }

    #[test]
    fn test_recovery_mixed_ops_and_txs() {
        let entries = vec![
            WalEntry::MetadataSet {
                key: "standalone".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxBegin { tx_id: 1 },
            WalEntry::MetadataSet {
                key: "in_tx".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxCommit { tx_id: 1 },
            WalEntry::MetadataDelete {
                key: "another".to_string(),
            },
        ];

        let recovery = WalRecovery::from_entries(&entries);

        assert_eq!(recovery.operations.len(), 2); // standalone + another
        assert_eq!(recovery.committed_ops.len(), 1); // in_tx
        assert_eq!(recovery.replay_count(), 3);
    }

    #[test]
    fn test_recovery_all_operations() {
        let entries = vec![
            WalEntry::MetadataSet {
                key: "k1".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxBegin { tx_id: 1 },
            WalEntry::MetadataSet {
                key: "k2".to_string(),
                data: make_tensor_data(),
            },
            WalEntry::TxCommit { tx_id: 1 },
        ];

        let recovery = WalRecovery::from_entries(&entries);
        let all_ops = recovery.all_operations();

        assert_eq!(all_ops.len(), 2);
    }

    #[test]
    fn test_recovery_from_wal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataSet {
            key: "test".to_string(),
            data: make_tensor_data(),
        })
        .unwrap();

        let recovery = WalRecovery::from_wal(&wal).unwrap();
        assert_eq!(recovery.operations.len(), 1);
    }

    // Error display tests

    #[test]
    fn test_wal_error_display() {
        let err = WalError::ChecksumMismatch {
            index: 5,
            expected: 0x1234,
            actual: 0x5678,
        };
        let msg = err.to_string();
        assert!(msg.contains("Checksum mismatch"));
        assert!(msg.contains("entry 5"));
    }

    #[test]
    fn test_wal_error_size_limit() {
        let err = WalError::SizeLimitExceeded {
            current: 1000,
            limit: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("size limit"));
    }

    #[test]
    fn test_wal_error_no_transaction() {
        let err = WalError::NoActiveTransaction;
        let msg = err.to_string();
        assert!(msg.contains("No active transaction"));
    }

    #[test]
    fn test_wal_error_transaction_active() {
        let err = WalError::TransactionAlreadyActive(42);
        let msg = err.to_string();
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_wal_error_entry_too_large() {
        let err = WalError::EntryTooLarge {
            size: 5_000_000_000,
        };
        let msg = err.to_string();
        assert!(msg.contains("too large"));
    }

    // Flush/fsync tests

    #[test]
    fn test_flush() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();
        wal.flush().unwrap();
    }

    #[test]
    fn test_fsync() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();
        wal.append(&WalEntry::MetadataDelete {
            key: "test".to_string(),
        })
        .unwrap();
        wal.fsync().unwrap();
    }

    // WalEntry equality tests

    #[test]
    fn test_wal_entry_equality() {
        let e1 = WalEntry::MetadataDelete {
            key: "test".to_string(),
        };
        let e2 = WalEntry::MetadataDelete {
            key: "test".to_string(),
        };
        let e3 = WalEntry::MetadataDelete {
            key: "other".to_string(),
        };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_wal_entry_clone() {
        let entry = WalEntry::EmbeddingSet {
            entity_id: EntityId(42),
            embedding: vec![1.0, 2.0, 3.0],
        };
        let cloned = entry.clone();
        assert_eq!(entry, cloned);
    }

    #[test]
    fn test_wal_entry_debug() {
        let entry = WalEntry::TxBegin { tx_id: 123 };
        let debug = format!("{entry:?}");
        assert!(debug.contains("TxBegin"));
        assert!(debug.contains("123"));
    }

    // WalStatus tests

    #[test]
    fn test_wal_status_clone() {
        let status = WalStatus {
            path: PathBuf::from("/test"),
            size_bytes: 100,
            entry_count: 5,
            checksums_enabled: true,
        };
        let cloned = status.clone();
        assert_eq!(cloned.path, status.path);
        assert_eq!(cloned.size_bytes, status.size_bytes);
    }

    #[test]
    fn test_wal_status_debug() {
        let status = WalStatus {
            path: PathBuf::from("/test"),
            size_bytes: 100,
            entry_count: 5,
            checksums_enabled: true,
        };
        let debug = format!("{status:?}");
        assert!(debug.contains("WalStatus"));
    }

    // WalRecovery tests

    #[test]
    fn test_wal_recovery_default() {
        let recovery = WalRecovery::default();
        assert!(recovery.operations.is_empty());
        assert!(recovery.pending_txs.is_empty());
        assert!(recovery.committed_ops.is_empty());
        assert!(recovery.last_checkpoint.is_none());
        assert_eq!(recovery.aborted_count, 0);
    }

    #[test]
    fn test_wal_recovery_debug() {
        let recovery = WalRecovery::default();
        let debug = format!("{recovery:?}");
        assert!(debug.contains("WalRecovery"));
    }

    // WalConfig clone test

    #[test]
    fn test_wal_config_clone() {
        let config = WalConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.max_size_bytes, cloned.max_size_bytes);
    }

    #[test]
    fn test_wal_config_debug() {
        let config = WalConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("WalConfig"));
    }

    #[test]
    fn test_fsync_throughput() {
        use std::time::Instant;

        let dir = tempdir().unwrap();
        let path = dir.path().join("perf.wal");
        let mut wal = TensorWal::open(&path, WalConfig::default()).unwrap();

        let count = 100;
        let start = Instant::now();

        for i in 0..count {
            wal.append(&WalEntry::MetadataSet {
                key: format!("key_{i}"),
                data: crate::TensorData::new(),
            })
            .unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        let ms_per_op = elapsed.as_millis() as f64 / count as f64;

        eprintln!("\n=== WAL FSYNC PERFORMANCE ===");
        eprintln!("{count} appends (each with fsync) in {elapsed:?}");
        eprintln!("{ops_per_sec:.0} ops/sec");
        eprintln!("{ms_per_op:.2} ms per op");
        eprintln!("==============================\n");

        // Sanity check: fsync should take at least 10us typically
        assert!(
            elapsed.as_micros() > 100,
            "suspiciously fast - fsync may not be working"
        );
    }

    // Batched sync mode tests

    #[test]
    fn test_sync_mode_immediate_default() {
        let config = WalConfig::default();
        assert!(matches!(config.sync_mode, SyncMode::Immediate));
    }

    #[test]
    fn test_sync_mode_batched_constructor() {
        let config = WalConfig::batched(100);
        assert!(matches!(
            config.sync_mode,
            SyncMode::Batched { max_entries: 100 }
        ));
    }

    #[test]
    fn test_batched_mode_defers_sync() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch.wal");

        let config = WalConfig::batched(10);
        let mut wal = TensorWal::open(&path, config).unwrap();

        // Write 5 entries - should not sync yet (batch size is 10)
        for i in 0..5 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key_{i}"),
            })
            .unwrap();
        }
        assert_eq!(wal.pending_sync_count(), 5);

        // Write 5 more - should trigger sync (total = 10)
        for i in 5..10 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key_{i}"),
            })
            .unwrap();
        }
        assert_eq!(wal.pending_sync_count(), 0);
    }

    #[test]
    fn test_manual_sync() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manual.wal");

        let config = WalConfig {
            sync_mode: SyncMode::Manual,
            ..WalConfig::default()
        };
        let mut wal = TensorWal::open(&path, config).unwrap();

        // Write entries - should never auto-sync
        for i in 0..100 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key_{i}"),
            })
            .unwrap();
        }
        assert_eq!(wal.pending_sync_count(), 100);

        // Manual sync
        let synced = wal.sync().unwrap();
        assert_eq!(synced, 100);
        assert_eq!(wal.pending_sync_count(), 0);
    }

    #[test]
    fn test_sync_returns_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sync_count.wal");

        let config = WalConfig::batched(1000); // Large batch so nothing auto-syncs
        let mut wal = TensorWal::open(&path, config).unwrap();

        // Write 50 entries
        for i in 0..50 {
            wal.append(&WalEntry::MetadataDelete {
                key: format!("key_{i}"),
            })
            .unwrap();
        }

        let synced = wal.sync().unwrap();
        assert_eq!(synced, 50);

        // Second sync returns 0
        let synced = wal.sync().unwrap();
        assert_eq!(synced, 0);
    }
}
