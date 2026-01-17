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

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::distributed_tx::TxPhase;

/// A WAL entry for 2PC transaction state changes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

/// Vote type during prepare phase.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrepareVoteKind {
    /// Participant votes YES.
    Yes,
    /// Participant votes NO.
    No,
}

/// Outcome of a completed transaction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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
}

impl TxWal {
    /// Open or create a WAL file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

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
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        Ok(count)
    }

    /// Append an entry to the WAL with fsync.
    pub fn append(&mut self, entry: &TxWalEntry) -> io::Result<()> {
        let bytes = bincode::serialize(entry).map_err(io::Error::other)?;

        self.file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;

        // CRITICAL: fsync to ensure durability
        self.file.get_ref().sync_all()?;

        self.entry_count += 1;
        Ok(())
    }

    /// Truncate the WAL (after checkpoint).
    pub fn truncate(&mut self) -> io::Result<()> {
        let file = File::create(&self.path)?;
        self.file = BufWriter::new(file);
        self.entry_count = 0;
        Ok(())
    }

    /// Replay all entries from the WAL.
    pub fn replay(&self) -> io::Result<Vec<TxWalEntry>> {
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
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            match bincode::deserialize::<TxWalEntry>(&data) {
                Ok(entry) => entries.push(entry),
                Err(_) => break, // Corrupted entry - stop replay
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

/// Recovered transaction state from the WAL.
#[derive(Debug, Clone, Default)]
pub struct TxRecoveryState {
    /// Transactions that were in Prepared phase (need decision).
    pub prepared_txs: Vec<RecoveredPreparedTx>,
    /// Transactions that were in Committing phase (need completion).
    pub committing_txs: Vec<u64>,
    /// Transactions that were in Aborting phase (need completion).
    pub aborting_txs: Vec<u64>,
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
    pub fn from_entries(entries: &[TxWalEntry]) -> Self {
        use std::collections::HashMap;

        let mut in_progress: HashMap<u64, InProgressTxState> = HashMap::new();

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
                    in_progress.remove(tx_id);
                },
            }
        }

        let mut state = TxRecoveryState::default();

        for (tx_id, (participants, votes, phase)) in in_progress {
            match phase {
                TxPhase::Prepared => {
                    state.prepared_txs.push(RecoveredPreparedTx {
                        tx_id,
                        participants,
                        votes,
                    });
                },
                TxPhase::Committing => {
                    state.committing_txs.push(tx_id);
                },
                TxPhase::Aborting => {
                    state.aborting_txs.push(tx_id);
                },
                TxPhase::Preparing | TxPhase::Committed | TxPhase::Aborted => {
                    // Preparing: Transaction didn't reach decision, can be aborted
                    // Committed/Aborted: Should have TxComplete entry, but may be safe to ignore
                },
            }
        }

        state
    }

    /// Reconstruct state directly from a WAL.
    pub fn from_wal(wal: &TxWal) -> io::Result<Self> {
        let entries = wal.replay()?;
        Ok(Self::from_entries(&entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                vote: PrepareVoteKind::Yes,
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
                vote: PrepareVoteKind::Yes,
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 1,
                vote: PrepareVoteKind::Yes,
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
        assert_eq!(state.committing_txs, vec![42]);
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
                vote: PrepareVoteKind::Yes,
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
            let bytes = bincode::serialize(&entry).unwrap();
            let restored: TxWalEntry = bincode::deserialize(&bytes).unwrap();
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
                vote: PrepareVoteKind::Yes,
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
                vote: PrepareVoteKind::Yes,
            },
            TxWalEntry::PrepareVote {
                tx_id: 2,
                shard: 2,
                vote: PrepareVoteKind::Yes,
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
        assert_eq!(state.committing_txs[0], 2);
        assert_eq!(state.aborting_txs.len(), 1);
        assert_eq!(state.aborting_txs[0], 3);
    }
}
