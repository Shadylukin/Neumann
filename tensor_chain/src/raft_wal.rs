//! Write-Ahead Log (WAL) for Raft consensus.
//!
//! Ensures durability of Raft state transitions by persisting changes to disk
//! before applying them in memory. This prevents split-brain scenarios and
//! double-voting after crashes.
//!
//! ## Critical Invariants
//!
//! 1. Term and voted_for MUST be persisted before any state change
//! 2. All writes MUST be fsynced before returning
//! 3. Recovery MUST restore the exact state from the WAL
//!
//! ## Format Versions
//!
//! V1 (legacy): `[4-byte length][bincode payload]`
//! V2 (current): `[4-byte length][4-byte CRC32][bincode payload]`
//!
//! V2 format is automatically detected during replay by checking if the
//! checksum bytes could be a valid bincode discriminant.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// WAL-specific error types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Disk space low: {available} bytes available, {required} required")]
    DiskSpaceLow { available: u64, required: u64 },

    #[error("WAL size limit exceeded: {current} bytes, limit {max}")]
    SizeLimitExceeded { current: u64, max: u64 },

    #[error("Checksum mismatch at entry {index}: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch {
        index: u64,
        expected: u32,
        actual: u32,
    },

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<WalError> for io::Error {
    fn from(e: WalError) -> Self {
        match e {
            WalError::Io(io_err) => io_err,
            other => io::Error::other(other.to_string()),
        }
    }
}

/// Configuration for WAL behavior.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Enable CRC32 checksums for new entries (default: true).
    pub enable_checksums: bool,
    /// Verify checksums on replay (default: true).
    pub verify_on_replay: bool,
    /// Maximum WAL size in bytes before rotation (default: 1GB).
    pub max_size_bytes: u64,
    /// Minimum free disk space required (default: 100MB).
    pub min_free_space_bytes: u64,
    /// Maximum number of rotated files to keep (default: 3).
    pub max_rotated_files: usize,
    /// Automatically rotate when size limit is reached (default: true).
    pub auto_rotate: bool,
    /// Check disk space before writes (default: true).
    pub pre_check_space: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enable_checksums: true,
            verify_on_replay: true,
            max_size_bytes: 1024 * 1024 * 1024,      // 1GB
            min_free_space_bytes: 100 * 1024 * 1024, // 100MB
            max_rotated_files: 3,
            auto_rotate: true,
            pre_check_space: true,
        }
    }
}

/// A WAL entry for Raft state changes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum RaftWalEntry {
    /// Term changed (election started or higher term seen).
    TermChange { new_term: u64 },

    /// Vote cast in a term.
    VoteCast { term: u64, candidate_id: String },

    /// Combined term and vote update (most common during elections).
    TermAndVote {
        term: u64,
        voted_for: Option<String>,
    },

    /// Log entry appended (for recovery of log state).
    LogAppend {
        index: u64,
        term: u64,
        command_hash: [u8; 32],
    },

    /// Log truncated from a specific index.
    LogTruncate { from_index: u64 },

    /// Snapshot taken (WAL can be truncated after this point).
    SnapshotTaken {
        last_included_index: u64,
        last_included_term: u64,
    },
}

/// Write-Ahead Log for Raft state.
pub struct RaftWal {
    file: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
    current_size: u64,
    config: WalConfig,
}

impl RaftWal {
    /// Open or create a WAL file with default configuration.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_config(path, WalConfig::default())
    }

    /// Open or create a WAL file with custom configuration.
    pub fn open_with_config(path: impl AsRef<Path>, config: WalConfig) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Get current file size
        let current_size = file.metadata().map(|m| m.len()).unwrap_or(0);

        // Count existing entries
        let entry_count = Self::count_entries(&path)?;

        Ok(Self {
            file: BufWriter::new(file),
            path,
            entry_count,
            current_size,
            config,
        })
    }

    /// Count entries in an existing WAL file.
    fn count_entries(path: &Path) -> io::Result<u64> {
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut count = 0;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u32::from_le_bytes(len_buf) as usize;

            // Read potential checksum (4 bytes)
            let mut checksum_buf = [0u8; 4];
            match reader.read_exact(&mut checksum_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            // Detect format: V1 has payload starting immediately, V2 has checksum
            // V1 bincode starts with a discriminant (0-5 for our enum variants)
            // V2 checksum is unlikely to be a small number like 0-5
            let is_v2 = !Self::looks_like_bincode_start(&checksum_buf);

            if is_v2 {
                // V2: skip the remaining payload bytes
                let mut data = vec![0u8; len];
                match reader.read_exact(&mut data) {
                    Ok(()) => count += 1,
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
            } else {
                // V1: we already read 4 bytes of payload, read the rest
                if len > 4 {
                    let mut remaining = vec![0u8; len - 4];
                    match reader.read_exact(&mut remaining) {
                        Ok(()) => count += 1,
                        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(e),
                    }
                } else {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Check if bytes look like a bincode discriminant start.
    /// Bincode encodes enum variants as u32 little-endian.
    fn looks_like_bincode_start(bytes: &[u8; 4]) -> bool {
        let value = u32::from_le_bytes(*bytes);
        // Our enum has variants 0-5, so valid bincode discriminants are small
        value <= 10
    }

    /// Get available disk space for the WAL directory.
    fn available_disk_space(&self) -> io::Result<u64> {
        #[cfg(unix)]
        {
            let parent = self.path.parent().unwrap_or(Path::new("."));
            let stat = nix_statvfs(parent)?;
            Ok(stat.available_bytes)
        }
        #[cfg(not(unix))]
        {
            // On non-Unix systems, return a large value to skip the check
            Ok(u64::MAX)
        }
    }

    /// Check if there's enough disk space for a write.
    fn check_space(&self, write_size: u64) -> Result<(), WalError> {
        if !self.config.pre_check_space {
            return Ok(());
        }
        let available = self.available_disk_space()?;
        let required = write_size + self.config.min_free_space_bytes;
        if available < required {
            return Err(WalError::DiskSpaceLow {
                available,
                required,
            });
        }
        Ok(())
    }

    /// Check if WAL size exceeds the limit.
    fn check_size_limit(&self, additional_size: u64) -> Result<(), WalError> {
        let new_size = self.current_size + additional_size;
        if new_size > self.config.max_size_bytes {
            return Err(WalError::SizeLimitExceeded {
                current: new_size,
                max: self.config.max_size_bytes,
            });
        }
        Ok(())
    }

    /// Get the path for a rotated WAL file.
    fn rotated_path(&self, index: usize) -> PathBuf {
        let file_name = self.path.file_name().unwrap_or_default().to_string_lossy();
        self.path.with_file_name(format!("{}.{}", file_name, index))
    }

    /// Rotate the WAL file.
    pub fn rotate(&mut self) -> Result<(), WalError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;

        // Delete oldest FIRST to make room for the shift
        let oldest = self.rotated_path(self.config.max_rotated_files);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }

        // Rename: .N-1 -> .N, .N-2 -> .N-1, ..., .1 -> .2
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

    /// Append an entry to the WAL with fsync.
    pub fn append(&mut self, entry: &RaftWalEntry) -> io::Result<()> {
        let bytes = bincode::serialize(entry).map_err(io::Error::other)?;
        let write_size = 4 + 4 + bytes.len() as u64; // length + checksum + payload

        // Check disk space
        self.check_space(write_size)?;

        // Check size limit and auto-rotate if needed
        if let Err(WalError::SizeLimitExceeded { .. }) = self.check_size_limit(write_size) {
            if self.config.auto_rotate {
                self.rotate()?;
            } else {
                return Err(WalError::SizeLimitExceeded {
                    current: self.current_size + write_size,
                    max: self.config.max_size_bytes,
                }
                .into());
            }
        }

        // Compute checksum
        let checksum = if self.config.enable_checksums {
            crc32fast::hash(&bytes)
        } else {
            0
        };

        // Write length-prefixed entry with checksum (V2 format)
        self.file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.file.write_all(&checksum.to_le_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;

        // CRITICAL: fsync to ensure durability
        self.file.get_ref().sync_all()?;

        self.current_size += write_size;
        self.entry_count += 1;
        Ok(())
    }

    /// Truncate the WAL (after snapshot).
    pub fn truncate(&mut self) -> io::Result<()> {
        // Close and recreate
        let file = File::create(&self.path)?;
        self.file = BufWriter::new(file);
        self.entry_count = 0;
        self.current_size = 0;
        Ok(())
    }

    /// Replay all entries from the WAL.
    pub fn replay(&self) -> io::Result<Vec<RaftWalEntry>> {
        self.replay_with_validation(self.config.verify_on_replay)
    }

    /// Replay all entries with optional checksum verification.
    pub fn replay_with_validation(&self, verify_checksums: bool) -> io::Result<Vec<RaftWalEntry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut entry_index = 0u64;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u32::from_le_bytes(len_buf) as usize;

            // Read potential checksum (4 bytes)
            let mut checksum_buf = [0u8; 4];
            match reader.read_exact(&mut checksum_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            // Detect format: V1 vs V2
            let is_v2 = !Self::looks_like_bincode_start(&checksum_buf);

            let data = if is_v2 {
                // V2: checksum_buf contains CRC32, read payload separately
                let mut data = vec![0u8; len];
                match reader.read_exact(&mut data) {
                    Ok(()) => {},
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }

                // Verify checksum if enabled
                if verify_checksums {
                    let stored_checksum = u32::from_le_bytes(checksum_buf);
                    let computed_checksum = crc32fast::hash(&data);
                    if stored_checksum != 0 && stored_checksum != computed_checksum {
                        return Err(WalError::ChecksumMismatch {
                            index: entry_index,
                            expected: stored_checksum,
                            actual: computed_checksum,
                        }
                        .into());
                    }
                }

                data
            } else {
                // V1: checksum_buf is actually the start of payload
                let mut data = Vec::with_capacity(len);
                data.extend_from_slice(&checksum_buf);

                if len > 4 {
                    let mut remaining = vec![0u8; len - 4];
                    match reader.read_exact(&mut remaining) {
                        Ok(()) => {},
                        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(e),
                    }
                    data.extend_from_slice(&remaining);
                }

                data
            };

            match bincode::deserialize::<RaftWalEntry>(&data) {
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

    /// Get the number of entries in the WAL.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Get the path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current size of the WAL in bytes.
    pub fn current_size(&self) -> u64 {
        self.current_size
    }

    /// Get the current configuration.
    pub fn config(&self) -> &WalConfig {
        &self.config
    }
}

/// Helper for getting filesystem stats on Unix.
#[cfg(unix)]
struct StatVfsResult {
    available_bytes: u64,
}

#[cfg(unix)]
fn nix_statvfs(path: &Path) -> io::Result<StatVfsResult> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid path"))?;

    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };

    if result != 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(StatVfsResult {
        available_bytes: stat.f_bavail as u64 * stat.f_frsize as u64,
    })
}

/// Recovered state from the WAL.
#[derive(Debug, Clone, Default)]
pub struct RaftRecoveryState {
    /// Current term (highest seen).
    pub current_term: u64,
    /// Node voted for in current term.
    pub voted_for: Option<String>,
    /// Last snapshot index (if any).
    pub last_snapshot_index: Option<u64>,
    /// Last snapshot term (if any).
    pub last_snapshot_term: Option<u64>,
}

impl RaftRecoveryState {
    /// Reconstruct state from WAL entries.
    pub fn from_entries(entries: &[RaftWalEntry]) -> Self {
        let mut state = Self::default();

        for entry in entries {
            match entry {
                RaftWalEntry::TermChange { new_term } => {
                    if *new_term > state.current_term {
                        state.current_term = *new_term;
                        state.voted_for = None; // New term, vote reset
                    }
                },
                RaftWalEntry::VoteCast { term, candidate_id } => {
                    if *term >= state.current_term {
                        state.current_term = *term;
                        state.voted_for = Some(candidate_id.clone());
                    }
                },
                RaftWalEntry::TermAndVote { term, voted_for } => {
                    if *term >= state.current_term {
                        state.current_term = *term;
                        state.voted_for = voted_for.clone();
                    }
                },
                RaftWalEntry::SnapshotTaken {
                    last_included_index,
                    last_included_term,
                } => {
                    state.last_snapshot_index = Some(*last_included_index);
                    state.last_snapshot_term = Some(*last_included_term);
                },
                RaftWalEntry::LogAppend { .. } | RaftWalEntry::LogTruncate { .. } => {
                    // Log entries handled separately
                },
            }
        }

        state
    }

    /// Reconstruct state directly from a WAL.
    pub fn from_wal(wal: &RaftWal) -> io::Result<Self> {
        let entries = wal.replay()?;
        Ok(Self::from_entries(&entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Append entries
        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
            wal.append(&RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node1".to_string(),
            })
            .unwrap();
            wal.append(&RaftWalEntry::TermAndVote {
                term: 2,
                voted_for: Some("node2".to_string()),
            })
            .unwrap();
            assert_eq!(wal.entry_count(), 3);
        }

        // Replay
        {
            let wal = RaftWal::open(&wal_path).unwrap();
            let entries = wal.replay().unwrap();

            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 1 });
            assert_eq!(
                entries[1],
                RaftWalEntry::VoteCast {
                    term: 1,
                    candidate_id: "node1".to_string()
                }
            );
            assert_eq!(
                entries[2],
                RaftWalEntry::TermAndVote {
                    term: 2,
                    voted_for: Some("node2".to_string())
                }
            );
        }
    }

    #[test]
    fn test_wal_truncate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 2 })
            .unwrap();
        assert_eq!(wal.entry_count(), 2);

        wal.truncate().unwrap();
        assert_eq!(wal.entry_count(), 0);

        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_recovery_state_from_entries() {
        let entries = vec![
            RaftWalEntry::TermChange { new_term: 1 },
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node1".to_string(),
            },
            RaftWalEntry::TermChange { new_term: 2 },
            RaftWalEntry::TermAndVote {
                term: 3,
                voted_for: Some("node2".to_string()),
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 3);
        assert_eq!(state.voted_for, Some("node2".to_string()));
    }

    #[test]
    fn test_recovery_state_term_change_resets_vote() {
        let entries = vec![
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node1".to_string(),
            },
            RaftWalEntry::TermChange { new_term: 2 },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 2);
        assert_eq!(state.voted_for, None); // Vote reset on term change
    }

    #[test]
    fn test_recovery_state_snapshot() {
        let entries = vec![
            RaftWalEntry::TermAndVote {
                term: 5,
                voted_for: None,
            },
            RaftWalEntry::SnapshotTaken {
                last_included_index: 100,
                last_included_term: 4,
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 5);
        assert_eq!(state.last_snapshot_index, Some(100));
        assert_eq!(state.last_snapshot_term, Some(4));
    }

    #[test]
    fn test_wal_handles_partial_write() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write valid entries
        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
        }

        // Append garbage to simulate partial write
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0x05, 0x00, 0x00, 0x00]).unwrap(); // length 5
            file.write_all(&[0x01, 0x02]).unwrap(); // only 2 bytes (partial)
        }

        // Should recover the valid entry
        {
            let wal = RaftWal::open(&wal_path).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 1 });
        }
    }

    #[test]
    fn test_recovery_state_from_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermAndVote {
                term: 5,
                voted_for: Some("leader".to_string()),
            })
            .unwrap();
        }

        let wal = RaftWal::open(&wal_path).unwrap();
        let state = RaftRecoveryState::from_wal(&wal).unwrap();
        assert_eq!(state.current_term, 5);
        assert_eq!(state.voted_for, Some("leader".to_string()));
    }

    #[test]
    fn test_wal_entry_serialization() {
        let entries = vec![
            RaftWalEntry::TermChange { new_term: 42 },
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "test".to_string(),
            },
            RaftWalEntry::TermAndVote {
                term: 2,
                voted_for: None,
            },
            RaftWalEntry::LogAppend {
                index: 10,
                term: 3,
                command_hash: [0u8; 32],
            },
            RaftWalEntry::LogTruncate { from_index: 5 },
            RaftWalEntry::SnapshotTaken {
                last_included_index: 100,
                last_included_term: 5,
            },
        ];

        for entry in entries {
            let bytes = bincode::serialize(&entry).unwrap();
            let restored: RaftWalEntry = bincode::deserialize(&bytes).unwrap();
            assert_eq!(entry, restored);
        }
    }

    #[test]
    fn test_wal_fsync_actually_persists() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 999 })
                .unwrap();
            // Drop without explicit close
        }

        // File should exist and contain the entry
        assert!(wal_path.exists());
        let metadata = fs::metadata(&wal_path).unwrap();
        assert!(metadata.len() > 0);

        // Verify we can read it back
        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_recovery_state_multiple_votes_same_term() {
        let entries = vec![
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node1".to_string(),
            },
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node2".to_string(),
            },
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node3".to_string(),
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 1);
        assert_eq!(state.voted_for, Some("node3".to_string()));
    }

    #[test]
    fn test_recovery_state_vote_in_lower_term_ignored() {
        let entries = vec![
            RaftWalEntry::TermAndVote {
                term: 5,
                voted_for: Some("node5".to_string()),
            },
            RaftWalEntry::VoteCast {
                term: 3,
                candidate_id: "node3".to_string(),
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 5);
        assert_eq!(state.voted_for, Some("node5".to_string()));
    }

    #[test]
    fn test_recovery_state_term_and_vote_combined() {
        let entries = vec![
            RaftWalEntry::TermAndVote {
                term: 2,
                voted_for: None,
            },
            RaftWalEntry::VoteCast {
                term: 2,
                candidate_id: "node_a".to_string(),
            },
            RaftWalEntry::TermAndVote {
                term: 3,
                voted_for: Some("node_b".to_string()),
            },
            RaftWalEntry::VoteCast {
                term: 3,
                candidate_id: "node_c".to_string(),
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, 3);
        assert_eq!(state.voted_for, Some("node_c".to_string()));
    }

    #[test]
    fn test_raft_wal_append_returns_io_error_on_failure() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("readonly.wal");

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
        }

        let mut perms = fs::metadata(&wal_path).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&wal_path, perms).unwrap();

        let result = RaftWal::open(&wal_path);
        if result.is_ok() {
            let mut wal = result.unwrap();
            let append_result = wal.append(&RaftWalEntry::TermChange { new_term: 2 });
            assert!(append_result.is_err());
        }

        let mut perms = fs::metadata(&wal_path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        fs::set_permissions(&wal_path, perms).unwrap();
    }

    #[test]
    fn test_raft_wal_open_creates_file_if_missing() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("nonexistent.wal");

        assert!(!wal_path.exists());

        let wal = RaftWal::open(&wal_path).unwrap();
        assert!(wal_path.exists());
        assert_eq!(wal.entry_count(), 0);
    }

    #[test]
    fn test_raft_wal_count_entries_handles_empty_file() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("empty.wal");

        File::create(&wal_path).unwrap();
        assert!(wal_path.exists());

        let wal = RaftWal::open(&wal_path).unwrap();
        assert_eq!(wal.entry_count(), 0);

        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_raft_wal_very_large_term_number() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("large_term.wal");

        let large_term = u64::MAX - 1;

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange {
                new_term: large_term,
            })
            .unwrap();
            wal.append(&RaftWalEntry::VoteCast {
                term: large_term,
                candidate_id: "node_max".to_string(),
            })
            .unwrap();
        }

        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            RaftWalEntry::TermChange {
                new_term: large_term
            }
        );
        assert_eq!(
            entries[1],
            RaftWalEntry::VoteCast {
                term: large_term,
                candidate_id: "node_max".to_string()
            }
        );

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.current_term, large_term);
        assert_eq!(state.voted_for, Some("node_max".to_string()));
    }

    #[test]
    fn test_raft_wal_concurrent_append_thread_safety() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("concurrent.wal");

        let wal = Arc::new(Mutex::new(RaftWal::open(&wal_path).unwrap()));
        let mut handles = vec![];

        for i in 0..10 {
            let wal_clone = Arc::clone(&wal);
            let handle = thread::spawn(move || {
                let mut wal = wal_clone.lock().unwrap();
                wal.append(&RaftWalEntry::TermChange { new_term: i as u64 })
                    .unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let wal = wal.lock().unwrap();
        assert_eq!(wal.entry_count(), 10);

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 10);
    }

    #[test]
    fn test_raft_wal_reopen_append_continue() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("reopen.wal");

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 2 })
                .unwrap();
            assert_eq!(wal.entry_count(), 2);
        }

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            assert_eq!(wal.entry_count(), 2);
            wal.append(&RaftWalEntry::TermChange { new_term: 3 })
                .unwrap();
            assert_eq!(wal.entry_count(), 3);
        }

        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 1 });
        assert_eq!(entries[1], RaftWalEntry::TermChange { new_term: 2 });
        assert_eq!(entries[2], RaftWalEntry::TermChange { new_term: 3 });
    }

    #[test]
    fn test_raft_wal_multiple_snapshots_last_wins() {
        let entries = vec![
            RaftWalEntry::SnapshotTaken {
                last_included_index: 10,
                last_included_term: 1,
            },
            RaftWalEntry::SnapshotTaken {
                last_included_index: 50,
                last_included_term: 3,
            },
            RaftWalEntry::SnapshotTaken {
                last_included_index: 100,
                last_included_term: 5,
            },
        ];

        let state = RaftRecoveryState::from_entries(&entries);
        assert_eq!(state.last_snapshot_index, Some(100));
        assert_eq!(state.last_snapshot_term, Some(5));
    }

    #[test]
    fn test_raft_wal_log_append_entries_recorded() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("log_append.wal");

        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::LogAppend {
                index: 1,
                term: 1,
                command_hash: hash1,
            })
            .unwrap();
            wal.append(&RaftWalEntry::LogAppend {
                index: 2,
                term: 1,
                command_hash: hash2,
            })
            .unwrap();
        }

        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            RaftWalEntry::LogAppend {
                index: 1,
                term: 1,
                command_hash: hash1
            }
        );
        assert_eq!(
            entries[1],
            RaftWalEntry::LogAppend {
                index: 2,
                term: 1,
                command_hash: hash2
            }
        );
    }

    #[test]
    fn test_raft_wal_log_truncate_entry_recorded() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("log_truncate.wal");

        {
            let mut wal = RaftWal::open(&wal_path).unwrap();
            wal.append(&RaftWalEntry::LogAppend {
                index: 1,
                term: 1,
                command_hash: [0u8; 32],
            })
            .unwrap();
            wal.append(&RaftWalEntry::LogAppend {
                index: 2,
                term: 1,
                command_hash: [0u8; 32],
            })
            .unwrap();
            wal.append(&RaftWalEntry::LogTruncate { from_index: 2 })
                .unwrap();
        }

        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2], RaftWalEntry::LogTruncate { from_index: 2 });
    }

    // New tests for V2 features

    #[test]
    fn test_wal_v2_checksum_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("v2_checksum.wal");

        let config = WalConfig {
            enable_checksums: true,
            verify_on_replay: true,
            ..Default::default()
        };

        {
            let mut wal = RaftWal::open_with_config(&wal_path, config.clone()).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 42 })
                .unwrap();
            wal.append(&RaftWalEntry::VoteCast {
                term: 42,
                candidate_id: "node1".to_string(),
            })
            .unwrap();
        }

        let wal = RaftWal::open_with_config(&wal_path, config).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 42 });
    }

    #[test]
    fn test_wal_corrupted_checksum_detected() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("corrupted.wal");

        let config = WalConfig {
            enable_checksums: true,
            verify_on_replay: true,
            ..Default::default()
        };

        {
            let mut wal = RaftWal::open_with_config(&wal_path, config.clone()).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
        }

        // Corrupt the checksum (byte 4-7 after length)
        {
            let mut data = fs::read(&wal_path).unwrap();
            if data.len() > 7 {
                data[4] ^= 0xFF; // Flip bits in checksum
                data[5] ^= 0xFF;
                fs::write(&wal_path, data).unwrap();
            }
        }

        let wal = RaftWal::open_with_config(&wal_path, config).unwrap();
        let result = wal.replay();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Checksum mismatch"));
    }

    #[test]
    fn test_wal_checksum_verification_can_be_disabled() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("no_verify.wal");

        let write_config = WalConfig {
            enable_checksums: true,
            verify_on_replay: true,
            ..Default::default()
        };

        {
            let mut wal = RaftWal::open_with_config(&wal_path, write_config).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
        }

        // Corrupt the checksum
        {
            let mut data = fs::read(&wal_path).unwrap();
            if data.len() > 7 {
                data[4] ^= 0xFF;
                fs::write(&wal_path, data).unwrap();
            }
        }

        // With verification disabled, should succeed
        let read_config = WalConfig {
            enable_checksums: true,
            verify_on_replay: false,
            ..Default::default()
        };
        let wal = RaftWal::open_with_config(&wal_path, read_config).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_wal_rotation_at_size_limit() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rotate.wal");

        let config = WalConfig {
            max_size_bytes: 200, // Very small limit
            auto_rotate: true,
            max_rotated_files: 2,
            ..Default::default()
        };

        let mut wal = RaftWal::open_with_config(&wal_path, config).unwrap();

        // Write entries until rotation
        for i in 0..20 {
            wal.append(&RaftWalEntry::TermChange { new_term: i })
                .unwrap();
        }

        // Check that rotated files exist
        let rotated1 = dir.path().join("rotate.wal.1");
        assert!(rotated1.exists());
    }

    #[test]
    fn test_wal_manual_rotation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("manual_rotate.wal");

        let config = WalConfig {
            auto_rotate: false,
            max_rotated_files: 3,
            ..Default::default()
        };

        let mut wal = RaftWal::open_with_config(&wal_path, config).unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 2 })
            .unwrap();

        wal.rotate().unwrap();

        // New WAL should be empty
        assert_eq!(wal.entry_count(), 0);
        assert_eq!(wal.current_size(), 0);

        // Old entries should be in rotated file
        let rotated1 = dir.path().join("manual_rotate.wal.1");
        assert!(rotated1.exists());
    }

    #[test]
    fn test_wal_rotated_files_cleanup() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("cleanup.wal");

        let config = WalConfig {
            max_size_bytes: 100,
            auto_rotate: true,
            max_rotated_files: 2,
            ..Default::default()
        };

        let mut wal = RaftWal::open_with_config(&wal_path, config).unwrap();

        // Trigger multiple rotations
        for i in 0..50 {
            wal.append(&RaftWalEntry::TermChange { new_term: i })
                .unwrap();
        }

        // Only max_rotated_files should exist
        let rotated1 = dir.path().join("cleanup.wal.1");
        let rotated2 = dir.path().join("cleanup.wal.2");
        let rotated3 = dir.path().join("cleanup.wal.3");

        assert!(rotated1.exists());
        assert!(rotated2.exists());
        assert!(!rotated3.exists()); // Should have been deleted
    }

    #[test]
    fn test_wal_config_default() {
        let config = WalConfig::default();
        assert!(config.enable_checksums);
        assert!(config.verify_on_replay);
        assert_eq!(config.max_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.min_free_space_bytes, 100 * 1024 * 1024);
        assert_eq!(config.max_rotated_files, 3);
        assert!(config.auto_rotate);
        assert!(config.pre_check_space);
    }

    #[test]
    fn test_wal_current_size_tracking() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("size_track.wal");

        let mut wal = RaftWal::open(&wal_path).unwrap();
        assert_eq!(wal.current_size(), 0);

        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();
        let size_after_one = wal.current_size();
        assert!(size_after_one > 0);

        wal.append(&RaftWalEntry::TermChange { new_term: 2 })
            .unwrap();
        let size_after_two = wal.current_size();
        assert!(size_after_two > size_after_one);
    }

    #[test]
    fn test_wal_error_display() {
        let err = WalError::DiskSpaceLow {
            available: 1000,
            required: 5000,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("5000"));

        let err = WalError::SizeLimitExceeded {
            current: 2000,
            max: 1000,
        };
        assert!(err.to_string().contains("2000"));
        assert!(err.to_string().contains("1000"));

        let err = WalError::ChecksumMismatch {
            index: 5,
            expected: 0x12345678,
            actual: 0x87654321,
        };
        assert!(err.to_string().contains("entry 5"));
    }

    #[test]
    fn test_wal_v1_backward_compatible() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("v1_compat.wal");

        // Write V1 format manually (no checksum)
        {
            let entry = RaftWalEntry::TermChange { new_term: 42 };
            let bytes = bincode::serialize(&entry).unwrap();

            let mut file = File::create(&wal_path).unwrap();
            file.write_all(&(bytes.len() as u32).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
        }

        // Should be able to read V1 format
        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 42 });
    }

    #[test]
    fn test_wal_size_limit_without_auto_rotate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("no_auto_rotate.wal");

        let config = WalConfig {
            max_size_bytes: 25, // Very small - one entry is ~20 bytes
            auto_rotate: false,
            ..Default::default()
        };

        let mut wal = RaftWal::open_with_config(&wal_path, config).unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();

        // Next append should fail due to size limit (first entry ~20 bytes, limit 25)
        let result = wal.append(&RaftWalEntry::TermChange { new_term: 2 });
        assert!(result.is_err());
    }

    #[test]
    fn test_wal_checksums_disabled() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("no_checksum.wal");

        let config = WalConfig {
            enable_checksums: false,
            ..Default::default()
        };

        {
            let mut wal = RaftWal::open_with_config(&wal_path, config.clone()).unwrap();
            wal.append(&RaftWalEntry::TermChange { new_term: 1 })
                .unwrap();
        }

        let wal = RaftWal::open_with_config(&wal_path, config).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }
}
