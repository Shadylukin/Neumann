// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Write-Ahead Log (WAL) for Two-Phase Commit (2PC) transactions.
//!
//! Ensures durability of transaction phase transitions by persisting changes
//! to disk before applying them in memory. This enables recovery of in-flight
//! transactions after a crash.
//!
//! ## Critical Invariants
//!
//! 1. Phase transitions MUST be persisted before being applied
//! 2. All writes MUST be fsynced before returning
//! 3. Recovery MUST complete any prepared transactions
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

use crate::distributed_tx::TxPhase;
use crate::raft_wal::{WalConfig, WalError};

/// A WAL entry for 2PC transaction state changes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum TxWalEntry {
    /// Transaction begin with list of participants.
    TxBegin {
        tx_id: u64,
        participants: Vec<usize>,
    },

    /// Prepare vote from a participant.
    PrepareVote {
        tx_id: u64,
        shard: usize,
        vote: PrepareVoteKind,
    },

    /// Phase transition.
    PhaseChange {
        tx_id: u64,
        from: TxPhase,
        to: TxPhase,
    },

    /// Transaction completed (committed or aborted).
    TxComplete { tx_id: u64, outcome: TxOutcome },

    /// Individual lock released during commit/abort.
    LockRelease { tx_id: u64, lock_handle: u64 },

    /// All locks for a transaction have been released.
    AllLocksReleased { tx_id: u64 },

    /// Abort intent recorded before sending abort messages to participants.
    ///
    /// On recovery, any `AbortIntent` without a matching `TxComplete` (Aborted)
    /// indicates abort messages may not have been delivered and must be resent.
    AbortIntent {
        tx_id: u64,
        reason: String,
        shards: Vec<usize>,
    },
}

/// Vote type during prepare phase.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrepareVoteKind {
    /// Participant votes YES with the lock handle for recovery.
    Yes { lock_handle: u64 },
    /// Participant votes NO.
    No,
}

/// Outcome of a completed transaction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum TxOutcome {
    /// Transaction was committed.
    Committed,
    /// Transaction was aborted.
    Aborted,
}

/// Write-Ahead Log for 2PC transactions.
pub struct TxWal {
    file: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
    current_size: u64,
    config: WalConfig,
}

impl TxWal {
    /// Open or create a WAL file with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or created.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_config(path, WalConfig::default())
    }

    /// Open or create a WAL file with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or created.
    pub fn open_with_config(path: impl AsRef<Path>, config: WalConfig) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Get current file size
        let current_size = file.metadata().map(|m| m.len()).unwrap_or(0);

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
        let mut detected_format: Option<bool> = None; // None = unknown, Some(true) = V2, Some(false) = V1

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

            // Detect format on first entry, then use consistently
            let is_v2 = *detected_format
                .get_or_insert_with(|| !Self::looks_like_bincode_start(checksum_buf));

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

    /// Check if bytes look like a bitcode discriminant start (legacy V1 format detection).
    /// After migration to bitcode, always returns false since V2 format is assumed.
    const fn looks_like_bincode_start(_bytes: [u8; 4]) -> bool {
        // After migrating to bitcode, we always assume V2 format.
        // Old bincode V1 files are not readable with bitcode anyway.
        false
    }

    /// Get available disk space for the WAL directory.
    fn available_disk_space(&self) -> io::Result<u64> {
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::os::unix::ffi::OsStrExt;

            let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
            let c_path = CString::new(parent.as_os_str().as_bytes())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid path"))?;

            let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
            let result = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };

            if result != 0 {
                return Err(io::Error::last_os_error());
            }

            // Cast needed for cross-platform compatibility (types differ between Linux/macOS)
            #[allow(clippy::unnecessary_cast, clippy::cast_lossless)]
            Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
        }
        #[cfg(not(unix))]
        {
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
    const fn check_size_limit(&self, additional_size: u64) -> Result<(), WalError> {
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
        self.path.with_file_name(format!("{file_name}.{index}"))
    }

    /// Rotate the WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if file I/O or renaming fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if serialization, disk space check, or I/O fails.
    pub fn append(&mut self, entry: &TxWalEntry) -> io::Result<()> {
        let bytes = bitcode::serialize(entry).map_err(io::Error::other)?;
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
        #[allow(clippy::cast_possible_truncation)]
        let len_bytes = (bytes.len() as u32).to_le_bytes();
        self.file.write_all(&len_bytes)?;
        self.file.write_all(&checksum.to_le_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;

        // CRITICAL: fsync to ensure durability
        self.file.get_ref().sync_all()?;

        self.current_size += write_size;
        self.entry_count += 1;
        Ok(())
    }

    /// Truncate the WAL (after checkpoint).
    ///
    /// Uses atomic file operations to ensure crash safety. The old file
    /// is replaced atomically with an empty file.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing, truncation, or reopening fails.
    pub fn truncate(&mut self) -> io::Result<()> {
        // Flush any pending writes before truncation
        self.file.flush()?;

        // Atomically create an empty file at the WAL path
        crate::atomic_io::atomic_truncate(&self.path)?;

        // Reopen the file for append
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.file = BufWriter::new(file);
        self.entry_count = 0;
        self.current_size = 0;
        Ok(())
    }

    /// Replay all entries from the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or entries are corrupted.
    pub fn replay(&self) -> io::Result<Vec<TxWalEntry>> {
        self.replay_with_validation(self.config.verify_on_replay)
    }

    /// Replay all entries with optional checksum verification.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or a checksum mismatch is detected.
    pub fn replay_with_validation(&self, verify_checksums: bool) -> io::Result<Vec<TxWalEntry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut entry_index = 0u64;
        let mut detected_format: Option<bool> = None; // None = unknown, Some(true) = V2, Some(false) = V1

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

            // Detect format on first entry, then use consistently
            let is_v2 = *detected_format
                .get_or_insert_with(|| !Self::looks_like_bincode_start(checksum_buf));

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

            match bitcode::deserialize::<TxWalEntry>(&data) {
                Ok(entry) => {
                    entries.push(entry);
                    entry_index += 1;
                },
                Err(_) => break, // Corrupted entry - stop replay
            }
        }

        Ok(entries)
    }

    /// Get the number of entries in the WAL.
    #[must_use]
    pub const fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Get the path to the WAL file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current size of the WAL in bytes.
    #[must_use]
    pub const fn current_size(&self) -> u64 {
        self.current_size
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &WalConfig {
        &self.config
    }
}

/// Recovered transaction state from the WAL.
#[derive(Debug, Clone, Default)]
pub struct TxRecoveryState {
    /// Transactions that were in Prepared phase (need decision).
    pub prepared_txs: Vec<RecoveredPreparedTx>,
    /// Transactions that were in Committing phase (need completion).
    pub committing_txs: Vec<RecoveredPreparedTx>,
    /// Transactions that were in Aborting phase (need completion).
    pub aborting_txs: Vec<RecoveredPreparedTx>,
    /// Lock handles from completed transactions that were not fully released
    /// (`TxComplete` logged but `AllLocksReleased` not seen). These are orphaned
    /// locks that must be force-released during recovery.
    pub orphaned_locks: Vec<OrphanedLock>,
    /// Abort intents that were logged but never completed (no matching `TxComplete`).
    /// These aborts must be resent to participants on recovery.
    pub pending_abort_intents: Vec<RecoveredAbortIntent>,
}

/// An abort that was initiated but may not have been fully delivered.
#[derive(Debug, Clone)]
pub struct RecoveredAbortIntent {
    /// Transaction being aborted.
    pub tx_id: u64,
    /// Reason for the abort.
    pub reason: String,
    /// Shards that need to receive the abort.
    pub shards: Vec<usize>,
}

/// A lock handle that belongs to a completed transaction but was never released.
#[derive(Debug, Clone)]
pub struct OrphanedLock {
    /// Transaction that held this lock.
    pub tx_id: u64,
    /// The lock handle to release.
    pub lock_handle: u64,
}

/// A transaction in prepared state awaiting decision.
#[derive(Debug, Clone)]
pub struct RecoveredPreparedTx {
    /// Transaction ID.
    pub tx_id: u64,
    /// Participants in the transaction.
    pub participants: Vec<usize>,
    /// Votes received from participants.
    pub votes: Vec<(usize, PrepareVoteKind)>,
}

/// In-progress transaction state during recovery: (participants, votes, phase).
type InProgressTxState = (Vec<usize>, Vec<(usize, PrepareVoteKind)>, TxPhase);

impl TxRecoveryState {
    /// Reconstruct state from WAL entries.
    ///
    /// Tracks orphaned locks: if a `TxComplete` is logged but `AllLocksReleased` is not,
    /// the lock handles from `PrepareVote::Yes` entries are considered orphaned and
    /// must be force-released during recovery.
    #[must_use]
    pub fn from_entries(entries: &[TxWalEntry]) -> Self {
        let (
            in_progress,
            completed_lock_handles,
            released_locks,
            fully_released,
            abort_intents,
            completed_txs,
        ) = Self::scan_entries(entries);

        let mut state = Self::default();
        Self::classify_in_progress(&mut state, in_progress);
        Self::detect_orphaned_locks(
            &mut state,
            &completed_lock_handles,
            &released_locks,
            &fully_released,
        );
        Self::detect_pending_aborts(&mut state, abort_intents, &completed_txs);
        state
    }

    #[allow(clippy::type_complexity)]
    fn scan_entries(
        entries: &[TxWalEntry],
    ) -> (
        std::collections::HashMap<u64, InProgressTxState>,
        std::collections::HashMap<u64, Vec<u64>>,
        std::collections::HashMap<u64, std::collections::HashSet<u64>>,
        std::collections::HashSet<u64>,
        std::collections::HashMap<u64, (String, Vec<usize>)>,
        std::collections::HashSet<u64>,
    ) {
        use std::collections::{HashMap, HashSet};

        let mut in_progress: HashMap<u64, InProgressTxState> = HashMap::new();
        let mut completed_lock_handles: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut released_locks: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut fully_released: HashSet<u64> = HashSet::new();
        let mut abort_intents: HashMap<u64, (String, Vec<usize>)> = HashMap::new();
        let mut completed_txs: HashSet<u64> = HashSet::new();

        for entry in entries {
            match entry {
                TxWalEntry::TxBegin {
                    tx_id,
                    participants,
                } => {
                    in_progress.insert(
                        *tx_id,
                        (participants.clone(), Vec::new(), TxPhase::Preparing),
                    );
                },
                TxWalEntry::PrepareVote { tx_id, shard, vote } => {
                    if let Some((_, votes, _)) = in_progress.get_mut(tx_id) {
                        votes.push((*shard, *vote));
                    }
                },
                TxWalEntry::PhaseChange { tx_id, to, .. } => {
                    if let Some((_, _, phase)) = in_progress.get_mut(tx_id) {
                        *phase = *to;
                    }
                },
                TxWalEntry::TxComplete { tx_id, .. } => {
                    if let Some((_, votes, _)) = in_progress.get(tx_id) {
                        let handles: Vec<u64> = votes
                            .iter()
                            .filter_map(|(_, vote)| match vote {
                                PrepareVoteKind::Yes { lock_handle } => Some(*lock_handle),
                                PrepareVoteKind::No => None,
                            })
                            .collect();
                        if !handles.is_empty() {
                            completed_lock_handles.insert(*tx_id, handles);
                        }
                    }
                    in_progress.remove(tx_id);
                    completed_txs.insert(*tx_id);
                },
                TxWalEntry::LockRelease { tx_id, lock_handle } => {
                    released_locks
                        .entry(*tx_id)
                        .or_default()
                        .insert(*lock_handle);
                },
                TxWalEntry::AllLocksReleased { tx_id } => {
                    fully_released.insert(*tx_id);
                },
                TxWalEntry::AbortIntent {
                    tx_id,
                    reason,
                    shards,
                } => {
                    abort_intents.insert(*tx_id, (reason.clone(), shards.clone()));
                },
            }
        }

        (
            in_progress,
            completed_lock_handles,
            released_locks,
            fully_released,
            abort_intents,
            completed_txs,
        )
    }

    fn classify_in_progress(
        state: &mut Self,
        in_progress: std::collections::HashMap<u64, InProgressTxState>,
    ) {
        for (tx_id, (participants, votes, phase)) in in_progress {
            let recovered_tx = RecoveredPreparedTx {
                tx_id,
                participants,
                votes,
            };
            match phase {
                TxPhase::Prepared => state.prepared_txs.push(recovered_tx),
                TxPhase::Committing => state.committing_txs.push(recovered_tx),
                TxPhase::Aborting => state.aborting_txs.push(recovered_tx),
                TxPhase::Preparing | TxPhase::Committed | TxPhase::Aborted => {},
            }
        }
    }

    fn detect_orphaned_locks(
        state: &mut Self,
        completed_lock_handles: &std::collections::HashMap<u64, Vec<u64>>,
        released_locks: &std::collections::HashMap<u64, std::collections::HashSet<u64>>,
        fully_released: &std::collections::HashSet<u64>,
    ) {
        for (tx_id, handles) in completed_lock_handles {
            if fully_released.contains(tx_id) {
                continue;
            }
            let released = released_locks.get(tx_id);
            for handle in handles {
                let already_released = released.is_some_and(|set| set.contains(handle));
                if !already_released {
                    state.orphaned_locks.push(OrphanedLock {
                        tx_id: *tx_id,
                        lock_handle: *handle,
                    });
                }
            }
        }
    }

    fn detect_pending_aborts(
        state: &mut Self,
        abort_intents: std::collections::HashMap<u64, (String, Vec<usize>)>,
        completed_txs: &std::collections::HashSet<u64>,
    ) {
        for (tx_id, (reason, shards)) in abort_intents {
            if !completed_txs.contains(&tx_id) {
                state.pending_abort_intents.push(RecoveredAbortIntent {
                    tx_id,
                    reason,
                    shards,
                });
            }
        }
    }

    /// Reconstruct state directly from a WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL cannot be replayed.
    pub fn from_wal(wal: &TxWal) -> io::Result<Self> {
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
    fn test_tx_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1, 2],
            })
            .unwrap();
            wal.append(&TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            })
            .unwrap();
            wal.append(&TxWalEntry::PhaseChange {
                tx_id: 1,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            })
            .unwrap();
            assert_eq!(wal.entry_count(), 3);
        }

        {
            let wal = TxWal::open(&wal_path).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 3);
        }
    }

    #[test]
    fn test_tx_recovery_state_prepared() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 1,
                vote: PrepareVoteKind::Yes { lock_handle: 101 },
            },
            TxWalEntry::PhaseChange {
                tx_id: 1,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.prepared_txs.len(), 1);
        assert_eq!(state.prepared_txs[0].tx_id, 1);
        assert_eq!(state.prepared_txs[0].votes.len(), 2);
    }

    #[test]
    fn test_tx_recovery_state_committing() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 42,
                participants: vec![0],
            },
            TxWalEntry::PhaseChange {
                tx_id: 42,
                from: TxPhase::Prepared,
                to: TxPhase::Committing,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.committing_txs.len(), 1);
        assert_eq!(state.committing_txs[0].tx_id, 42);
    }

    #[test]
    fn test_tx_recovery_state_completed_ignored() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            },
            TxWalEntry::PhaseChange {
                tx_id: 1,
                from: TxPhase::Preparing,
                to: TxPhase::Committed,
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.prepared_txs.is_empty());
        assert!(state.committing_txs.is_empty());
        assert!(state.aborting_txs.is_empty());
    }

    #[test]
    fn test_tx_wal_truncate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();
        assert_eq!(wal.entry_count(), 1);

        wal.truncate().unwrap();
        assert_eq!(wal.entry_count(), 0);

        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_tx_wal_entry_serialization() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1, 2],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 1,
                vote: PrepareVoteKind::No,
            },
            TxWalEntry::PhaseChange {
                tx_id: 1,
                from: TxPhase::Preparing,
                to: TxPhase::Aborting,
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Aborted,
            },
        ];

        for entry in entries {
            let bytes = bitcode::serialize(&entry).unwrap();
            let restored: TxWalEntry = bitcode::deserialize(&bytes).unwrap();
            assert_eq!(entry, restored);
        }
    }

    #[test]
    fn test_tx_wal_open_permission_denied() {
        let dir = tempdir().unwrap();
        let readonly_dir = dir.path().join("readonly");
        std::fs::create_dir(&readonly_dir).unwrap();

        let mut perms = std::fs::metadata(&readonly_dir).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&readonly_dir, perms).unwrap();

        let result = TxWal::open(readonly_dir.join("tx.wal"));
        assert!(result.is_err());

        let mut perms = std::fs::metadata(&readonly_dir).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        std::fs::set_permissions(&readonly_dir, perms).unwrap();
    }

    #[test]
    fn test_tx_wal_append_disk_full_simulation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();

        drop(wal);

        let mut perms = std::fs::metadata(&wal_path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&wal_path, perms).unwrap();

        let result = TxWal::open(&wal_path);
        if let Ok(mut wal) = result {
            let append_result = wal.append(&TxWalEntry::TxBegin {
                tx_id: 2,
                participants: vec![1],
            });
            assert!(append_result.is_err());
        }

        let mut perms = std::fs::metadata(&wal_path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        std::fs::set_permissions(&wal_path, perms).unwrap();
    }

    #[test]
    fn test_tx_wal_replay_io_error() {
        let dir = tempdir().unwrap();
        let nonexistent_path = dir.path().join("nonexistent").join("tx.wal");

        let wal = TxWal {
            file: BufWriter::new(File::create(dir.path().join("temp.wal")).unwrap()),
            path: nonexistent_path,
            entry_count: 0,
            current_size: 0,
            config: WalConfig::default(),
        };

        let result = wal.replay();
        assert!(result.is_err());
    }

    #[test]
    fn test_tx_wal_truncate_error_handling() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();

        let readonly_dir = dir.path().join("readonly");
        std::fs::create_dir(&readonly_dir).unwrap();
        let readonly_wal_path = readonly_dir.join("tx.wal");
        std::fs::copy(&wal_path, &readonly_wal_path).unwrap();

        let mut perms = std::fs::metadata(&readonly_dir).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&readonly_dir, perms).unwrap();

        let mut readonly_wal = TxWal {
            file: BufWriter::new(File::open(&wal_path).unwrap()),
            path: readonly_dir.join("new_file.wal"),
            entry_count: 1,
            current_size: 0,
            config: WalConfig::default(),
        };

        let result = readonly_wal.truncate();
        assert!(result.is_err());

        let mut perms = std::fs::metadata(&readonly_dir).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        std::fs::set_permissions(&readonly_dir, perms).unwrap();
    }

    #[test]
    fn test_tx_wal_handles_partial_length_write() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            })
            .unwrap();
        }

        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0x01, 0x02]).unwrap();
        }

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_tx_wal_handles_partial_data_write() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            })
            .unwrap();
        }

        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&100u32.to_le_bytes()).unwrap();
            file.write_all(&[0x01, 0x02, 0x03]).unwrap();
        }

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_tx_wal_corrupted_bincode_stops_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            })
            .unwrap();
        }

        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            let garbage = [0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA];
            file.write_all(&(garbage.len() as u32).to_le_bytes())
                .unwrap();
            file.write_all(&garbage).unwrap();
        }

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_tx_wal_empty_replay_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
        assert_eq!(wal.entry_count(), 0);
    }

    #[test]
    fn test_tx_wal_very_large_entry_serializes_correctly() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let large_participants: Vec<usize> = (0..10000).collect();

        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 999,
            participants: large_participants.clone(),
        })
        .unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            TxWalEntry::TxBegin {
                tx_id,
                participants,
            } => {
                assert_eq!(*tx_id, 999);
                assert_eq!(participants.len(), 10000);
                assert_eq!(*participants, large_participants);
            },
            _ => panic!("Expected TxBegin entry"),
        }
    }

    #[test]
    fn test_tx_wal_maximum_entry_count() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let mut wal = TxWal::open(&wal_path).unwrap();
        for i in 0..1000 {
            wal.append(&TxWalEntry::TxBegin {
                tx_id: i,
                participants: vec![0],
            })
            .unwrap();
        }
        assert_eq!(wal.entry_count(), 1000);

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1000);
    }

    #[test]
    fn test_tx_wal_path_accessor() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        let wal = TxWal::open(&wal_path).unwrap();
        assert_eq!(wal.path(), wal_path.as_path());
    }

    #[test]
    fn test_tx_wal_entry_count_after_reopen() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            for i in 0..5 {
                wal.append(&TxWalEntry::TxBegin {
                    tx_id: i,
                    participants: vec![0],
                })
                .unwrap();
            }
            assert_eq!(wal.entry_count(), 5);
        }

        {
            let wal = TxWal::open(&wal_path).unwrap();
            assert_eq!(wal.entry_count(), 5);
        }
    }

    #[test]
    fn test_tx_recovery_state_vote_without_begin_ignored() {
        let entries = vec![
            TxWalEntry::PrepareVote {
                tx_id: 999,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PhaseChange {
                tx_id: 999,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.prepared_txs.is_empty());
        assert!(state.committing_txs.is_empty());
        assert!(state.aborting_txs.is_empty());
    }

    #[test]
    fn test_tx_recovery_state_multiple_transactions() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            },
            TxWalEntry::TxBegin {
                tx_id: 2,
                participants: vec![2, 3],
            },
            TxWalEntry::TxBegin {
                tx_id: 3,
                participants: vec![4],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PrepareVote {
                tx_id: 2,
                shard: 2,
                vote: PrepareVoteKind::Yes { lock_handle: 101 },
            },
            TxWalEntry::PhaseChange {
                tx_id: 1,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            },
            TxWalEntry::PhaseChange {
                tx_id: 2,
                from: TxPhase::Preparing,
                to: TxPhase::Committing,
            },
            TxWalEntry::PhaseChange {
                tx_id: 3,
                from: TxPhase::Preparing,
                to: TxPhase::Aborting,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.prepared_txs.len(), 1);
        assert_eq!(state.prepared_txs[0].tx_id, 1);
        assert_eq!(state.committing_txs.len(), 1);
        assert_eq!(state.committing_txs[0].tx_id, 2);
        assert_eq!(state.aborting_txs.len(), 1);
        assert_eq!(state.aborting_txs[0].tx_id, 3);
    }

    // New tests for V2 features

    #[test]
    fn test_tx_wal_v2_checksum_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("v2_checksum.wal");

        let config = WalConfig {
            enable_checksums: true,
            verify_on_replay: true,
            ..Default::default()
        };

        {
            let mut wal = TxWal::open_with_config(&wal_path, config.clone()).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 42,
                participants: vec![0, 1],
            })
            .unwrap();
            wal.append(&TxWalEntry::PrepareVote {
                tx_id: 42,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            })
            .unwrap();
        }

        let wal = TxWal::open_with_config(&wal_path, config).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_tx_wal_corrupted_checksum_detected() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("corrupted.wal");

        let config = WalConfig {
            enable_checksums: true,
            verify_on_replay: true,
            ..Default::default()
        };

        {
            let mut wal = TxWal::open_with_config(&wal_path, config.clone()).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            })
            .unwrap();
        }

        // Corrupt the checksum
        {
            let mut data = fs::read(&wal_path).unwrap();
            if data.len() > 7 {
                data[4] ^= 0xFF;
                data[5] ^= 0xFF;
                fs::write(&wal_path, data).unwrap();
            }
        }

        let wal = TxWal::open_with_config(&wal_path, config).unwrap();
        let result = wal.replay();
        assert!(result.is_err());
    }

    #[test]
    fn test_tx_wal_rotation_at_size_limit() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rotate.wal");

        let config = WalConfig {
            max_size_bytes: 200,
            auto_rotate: true,
            max_rotated_files: 2,
            ..Default::default()
        };

        let mut wal = TxWal::open_with_config(&wal_path, config).unwrap();

        for i in 0..20 {
            wal.append(&TxWalEntry::TxBegin {
                tx_id: i,
                participants: vec![0, 1, 2],
            })
            .unwrap();
        }

        let rotated1 = dir.path().join("rotate.wal.1");
        assert!(rotated1.exists());
    }

    #[test]
    fn test_tx_wal_manual_rotation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("manual_rotate.wal");

        let config = WalConfig {
            auto_rotate: false,
            max_rotated_files: 3,
            ..Default::default()
        };

        let mut wal = TxWal::open_with_config(&wal_path, config).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();

        wal.rotate().unwrap();

        assert_eq!(wal.entry_count(), 0);
        assert_eq!(wal.current_size(), 0);

        let rotated1 = dir.path().join("manual_rotate.wal.1");
        assert!(rotated1.exists());
    }

    #[test]
    fn test_tx_wal_v2_format_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("v2_format.wal");

        // Write V2 format manually (with checksum)
        {
            let entry = TxWalEntry::TxBegin {
                tx_id: 42,
                participants: vec![0, 1],
            };
            let bytes = bitcode::serialize(&entry).unwrap();
            let checksum = crc32fast::hash(&bytes);

            let mut file = File::create(&wal_path).unwrap();
            file.write_all(&(bytes.len() as u32).to_le_bytes()).unwrap();
            file.write_all(&checksum.to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
        }

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            TxWalEntry::TxBegin {
                tx_id,
                participants,
            } => {
                assert_eq!(*tx_id, 42);
                assert_eq!(*participants, vec![0, 1]);
            },
            _ => panic!("Expected TxBegin"),
        }
    }

    #[test]
    fn test_tx_wal_current_size_tracking() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("size_track.wal");

        let mut wal = TxWal::open(&wal_path).unwrap();
        assert_eq!(wal.current_size(), 0);

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();
        let size_after_one = wal.current_size();
        assert!(size_after_one > 0);

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 2,
            participants: vec![0],
        })
        .unwrap();
        let size_after_two = wal.current_size();
        assert!(size_after_two > size_after_one);
    }

    #[test]
    fn test_lock_release_entries_serialize_roundtrip() {
        let entries = vec![
            TxWalEntry::LockRelease {
                tx_id: 1,
                lock_handle: 42,
            },
            TxWalEntry::AllLocksReleased { tx_id: 1 },
        ];

        for entry in entries {
            let bytes = bitcode::serialize(&entry).unwrap();
            let restored: TxWalEntry = bitcode::deserialize(&bytes).unwrap();
            assert_eq!(entry, restored);
        }
    }

    #[test]
    fn test_lock_release_wal_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("lock_release.wal");

        {
            let mut wal = TxWal::open(&wal_path).unwrap();
            wal.append(&TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            })
            .unwrap();
            wal.append(&TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            })
            .unwrap();
            wal.append(&TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            })
            .unwrap();
            wal.append(&TxWalEntry::LockRelease {
                tx_id: 1,
                lock_handle: 100,
            })
            .unwrap();
            wal.append(&TxWalEntry::AllLocksReleased { tx_id: 1 })
                .unwrap();
        }

        let wal = TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn test_recovery_detects_orphaned_locks() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 1,
                vote: PrepareVoteKind::Yes { lock_handle: 101 },
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            },
            // Crash -- no LockRelease or AllLocksReleased entries
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.prepared_txs.is_empty());
        assert_eq!(state.orphaned_locks.len(), 2);
        let mut handles: Vec<u64> = state.orphaned_locks.iter().map(|o| o.lock_handle).collect();
        handles.sort_unstable();
        assert_eq!(handles, vec![100, 101]);
    }

    #[test]
    fn test_recovery_no_orphans_when_all_released() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            },
            TxWalEntry::LockRelease {
                tx_id: 1,
                lock_handle: 100,
            },
            TxWalEntry::AllLocksReleased { tx_id: 1 },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.orphaned_locks.is_empty());
    }

    #[test]
    fn test_recovery_partial_lock_release_detected() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 1,
                vote: PrepareVoteKind::Yes { lock_handle: 101 },
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            },
            // Only one lock released before crash
            TxWalEntry::LockRelease {
                tx_id: 1,
                lock_handle: 100,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.orphaned_locks.len(), 1);
        assert_eq!(state.orphaned_locks[0].lock_handle, 101);
    }

    #[test]
    fn test_recovery_no_orphans_for_no_votes() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::No,
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Aborted,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.orphaned_locks.is_empty());
    }

    #[test]
    fn test_tx_wal_size_limit_without_auto_rotate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("no_auto_rotate.wal");

        // Bitcode produces compact output: ~15-20 bytes per entry with checksum
        let config = WalConfig {
            max_size_bytes: 20, // Very small - one entry fits, second won't
            auto_rotate: false,
            ..Default::default()
        };

        let mut wal = TxWal::open_with_config(&wal_path, config).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();

        let result = wal.append(&TxWalEntry::TxBegin {
            tx_id: 2,
            participants: vec![0],
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_abort_intent_wal_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("abort_intent.wal");
        let mut wal = TxWal::open(&wal_path).unwrap();

        wal.append(&TxWalEntry::AbortIntent {
            tx_id: 42,
            reason: "timeout".to_string(),
            shards: vec![0, 1, 2],
        })
        .unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            TxWalEntry::AbortIntent {
                tx_id,
                reason,
                shards,
            } => {
                assert_eq!(*tx_id, 42);
                assert_eq!(reason, "timeout");
                assert_eq!(shards, &[0, 1, 2]);
            },
            other => panic!("Expected AbortIntent, got {other:?}"),
        }
    }

    #[test]
    fn test_recovery_detects_pending_abort_intents() {
        let entries = vec![
            TxWalEntry::AbortIntent {
                tx_id: 10,
                reason: "conflict".to_string(),
                shards: vec![0, 1],
            },
            // No TxComplete for tx 10 -- abort delivery was interrupted
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.pending_abort_intents.len(), 1);
        assert_eq!(state.pending_abort_intents[0].tx_id, 10);
        assert_eq!(state.pending_abort_intents[0].reason, "conflict");
        assert_eq!(state.pending_abort_intents[0].shards, vec![0, 1]);
    }

    #[test]
    fn test_recovery_no_pending_abort_when_completed() {
        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 20,
                participants: vec![0],
            },
            TxWalEntry::AbortIntent {
                tx_id: 20,
                reason: "timeout".to_string(),
                shards: vec![0],
            },
            TxWalEntry::TxComplete {
                tx_id: 20,
                outcome: TxOutcome::Aborted,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert!(state.pending_abort_intents.is_empty());
    }

    #[test]
    fn test_recovery_mixed_abort_intents() {
        let entries = vec![
            // TX 30: abort intent without completion (pending)
            TxWalEntry::AbortIntent {
                tx_id: 30,
                reason: "partition".to_string(),
                shards: vec![0, 1],
            },
            // TX 31: abort intent with completion (resolved)
            TxWalEntry::AbortIntent {
                tx_id: 31,
                reason: "conflict".to_string(),
                shards: vec![2],
            },
            TxWalEntry::TxComplete {
                tx_id: 31,
                outcome: TxOutcome::Aborted,
            },
        ];

        let state = TxRecoveryState::from_entries(&entries);
        assert_eq!(state.pending_abort_intents.len(), 1);
        assert_eq!(state.pending_abort_intents[0].tx_id, 30);
    }
}
