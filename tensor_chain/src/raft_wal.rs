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

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A WAL entry for Raft state changes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

impl RaftWal {
    /// Open or create a WAL file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Count existing entries
        let entry_count = Self::count_entries(&path)?;

        Ok(Self {
            file: BufWriter::new(file),
            path,
            entry_count,
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
            let mut data = vec![0u8; len];
            match reader.read_exact(&mut data) {
                Ok(()) => count += 1,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // Partial entry
                Err(e) => return Err(e),
            }
        }

        Ok(count)
    }

    /// Append an entry to the WAL with fsync.
    pub fn append(&mut self, entry: &RaftWalEntry) -> io::Result<()> {
        let bytes =
            bincode::serialize(entry).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Write length-prefixed entry
        self.file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;

        // CRITICAL: fsync to ensure durability
        self.file.get_ref().sync_all()?;

        self.entry_count += 1;
        Ok(())
    }

    /// Truncate the WAL (after snapshot).
    pub fn truncate(&mut self) -> io::Result<()> {
        // Close and recreate
        let file = File::create(&self.path)?;
        self.file = BufWriter::new(file);
        self.entry_count = 0;
        Ok(())
    }

    /// Replay all entries from the WAL.
    pub fn replay(&self) -> io::Result<Vec<RaftWalEntry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];

            match reader.read_exact(&mut data) {
                Ok(()) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    // Partial entry - WAL was corrupted, stop here
                    break;
                },
                Err(e) => return Err(e),
            }

            match bincode::deserialize::<RaftWalEntry>(&data) {
                Ok(entry) => entries.push(entry),
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
}
