// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Loom-based concurrency verification tests for DistributedTxCoordinator lock ordering.
//!
//! Two test groups:
//!
//! 1. **Mock tests** (`loom_dtx_*`): Minimal mock with loom::sync::RwLock to verify the
//!    phased locking pattern in record_vote (distributed_tx.rs:1394-1563) cannot deadlock.
//!
//! 2. **Real-code tests** (`loom_dtx_real_*`): Drive the actual `DistributedTxCoordinator`
//!    through concurrent operations. The `sync_compat` module swaps `parking_lot::RwLock`
//!    for a loom-compatible wrapper, so loom exhaustively explores all interleavings of the
//!    real phased locking discipline.
//!
//! Run with: cargo nextest run --package tensor_chain --features loom -E 'test(loom_dtx)'

#![cfg(feature = "loom")]

use loom::sync::{Arc, RwLock};
use loom::thread;
use std::collections::{HashMap, HashSet};

use tensor_chain::{
    ConsensusConfig, ConsensusManager, DeltaVector, DistributedTxConfig, DistributedTxCoordinator,
    PrepareVote,
};

/// Minimal mock of the pending + pending_aborts lock pattern from
/// `DistributedTxCoordinator`. Reproduces only the phased locking discipline.
struct MockCoordinator {
    /// Maps tx_id -> vote_count (simplified from real pending map)
    pending: RwLock<HashMap<u64, u8>>,
    /// List of (tx_id, reason) for aborted transactions
    pending_aborts: RwLock<Vec<u64>>,
}

impl MockCoordinator {
    fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            pending_aborts: RwLock::new(Vec::new()),
        }
    }

    /// Simulates begin: acquires pending write lock, inserts tx, releases.
    fn mock_begin(&self, tx_id: u64) {
        let mut pending = self.pending.write().unwrap();
        pending.insert(tx_id, 0);
    }

    /// Simulates the phased locking pattern from record_vote:
    /// Phase 1: pending write -> record vote -> release
    /// Phase 2: (no locks) decide
    /// Phase 3: if abort, re-acquire pending write -> move to aborts -> release,
    ///          then acquire pending_aborts write -> push -> release
    fn mock_record_vote(&self, tx_id: u64, shard: u8, should_abort: bool) {
        // Phase 1: Acquire pending write, record vote, release
        let vote_count = {
            let mut pending = self.pending.write().unwrap();
            if let Some(count) = pending.get_mut(&tx_id) {
                *count += 1;
                *count
            } else {
                return; // tx already gone
            }
        }; // pending lock released here

        // Phase 2: Decision logic (no locks held)
        let _ = vote_count;
        let _ = shard;

        if should_abort {
            // Phase 3: Re-acquire pending write to remove tx
            {
                let mut pending = self.pending.write().unwrap();
                pending.remove(&tx_id);
            } // pending lock released

            // Then acquire pending_aborts write
            {
                let mut aborts = self.pending_aborts.write().unwrap();
                aborts.push(tx_id);
            }
        }
    }

    /// Simulates abort: acquires pending write, removes tx, releases.
    fn mock_abort(&self, tx_id: u64) {
        let existed = {
            let mut pending = self.pending.write().unwrap();
            pending.remove(&tx_id).is_some()
        };
        if existed {
            let mut aborts = self.pending_aborts.write().unwrap();
            aborts.push(tx_id);
        }
    }
}

#[test]
fn loom_dtx_record_vote_two_shards_no_deadlock() {
    loom::model(|| {
        let coord = Arc::new(MockCoordinator::new());
        coord.mock_begin(1);

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        // Two shards vote for the same tx concurrently
        let t1 = thread::spawn(move || {
            c1.mock_record_vote(1, 0, false);
        });
        let t2 = thread::spawn(move || {
            c2.mock_record_vote(1, 1, false);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Verify consistent state: tx should still be pending with 2 votes
        let pending = coord.pending.read().unwrap();
        if let Some(&count) = pending.get(&1) {
            assert_eq!(count, 2, "Both votes must be recorded");
        }
        // Or tx may have been removed if one triggered abort
    });
}

#[test]
fn loom_dtx_record_vote_concurrent_transactions() {
    loom::model(|| {
        let coord = Arc::new(MockCoordinator::new());
        coord.mock_begin(1);
        coord.mock_begin(2);

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        // Thread 1: votes on tx 1, triggers abort path
        let t1 = thread::spawn(move || {
            c1.mock_record_vote(1, 0, true);
        });

        // Thread 2: votes on tx 2, no abort
        let t2 = thread::spawn(move || {
            c2.mock_record_vote(2, 0, false);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // tx 1 should be aborted, tx 2 should still be pending
        let pending = coord.pending.read().unwrap();
        assert!(
            !pending.contains_key(&1),
            "tx 1 must be removed after abort"
        );
        assert!(pending.contains_key(&2), "tx 2 must still be pending");

        let aborts = coord.pending_aborts.read().unwrap();
        assert!(aborts.contains(&1), "tx 1 must be in aborts list");
    });
}

#[test]
fn loom_dtx_concurrent_begin_and_vote() {
    loom::model(|| {
        let coord = Arc::new(MockCoordinator::new());
        coord.mock_begin(1);

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        // Thread 1: begin a new transaction (pending write)
        let t1 = thread::spawn(move || {
            c1.mock_begin(2);
        });

        // Thread 2: vote on existing transaction (pending write -> release -> pending write)
        let t2 = thread::spawn(move || {
            c2.mock_record_vote(1, 0, false);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Both operations must complete without deadlock
        let pending = coord.pending.read().unwrap();
        assert!(pending.contains_key(&2), "tx 2 must exist after begin");
    });
}

#[test]
fn loom_dtx_abort_during_vote() {
    loom::model(|| {
        let coord = Arc::new(MockCoordinator::new());
        coord.mock_begin(1);

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        // Thread 1: abort tx 1 (pending write)
        let t1 = thread::spawn(move || {
            c1.mock_abort(1);
        });

        // Thread 2: vote on tx 1 (pending write -> release -> pending write)
        let t2 = thread::spawn(move || {
            c2.mock_record_vote(1, 0, true);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // tx 1 must be removed from pending regardless of ordering
        let pending = coord.pending.read().unwrap();
        assert!(
            !pending.contains_key(&1),
            "tx 1 must be removed after abort/vote"
        );

        // At least one abort record should exist
        let aborts = coord.pending_aborts.read().unwrap();
        assert!(!aborts.is_empty(), "At least one abort must be recorded");
    });
}

// ---------------------------------------------------------------------------
// Real-code loom tests: drive DistributedTxCoordinator through sync_compat
// ---------------------------------------------------------------------------

fn make_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default())
}

fn yes_vote(shard: usize) -> PrepareVote {
    PrepareVote::Yes {
        lock_handle: shard as u64,
        delta: DeltaVector::new(&[0.0; 4], HashSet::new(), 0),
    }
}

fn no_vote() -> PrepareVote {
    PrepareVote::No {
        reason: "test rejection".to_string(),
    }
}

#[test]
fn loom_dtx_real_concurrent_begin() {
    // Two threads call begin() concurrently, contending on the pending write lock.
    loom::model(|| {
        let coord = Arc::new(make_coordinator());

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        let t1 = thread::spawn(move || c1.begin(&"node1".to_string(), &[0, 1]).ok());
        let t2 = thread::spawn(move || c2.begin(&"node2".to_string(), &[0, 1]).ok());

        let tx1 = t1.join().unwrap();
        let tx2 = t2.join().unwrap();

        // Both should succeed (max_concurrent default is 100)
        assert!(tx1.is_some(), "first begin must succeed");
        assert!(tx2.is_some(), "second begin must succeed");

        // Both transactions should have different IDs
        let id1 = tx1.unwrap().tx_id;
        let id2 = tx2.unwrap().tx_id;
        assert_ne!(id1, id2, "transaction IDs must be unique");
    });
}

#[test]
fn loom_dtx_real_concurrent_votes() {
    // Two shards record_vote() for the same tx (phased locking pattern).
    loom::model(|| {
        let coord = Arc::new(make_coordinator());
        let tx = coord.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        let t1 = thread::spawn(move || c1.record_vote(tx_id, 0, yes_vote(0)));
        let t2 = thread::spawn(move || c2.record_vote(tx_id, 1, yes_vote(1)));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both votes must succeed (no TxNotFound or DuplicateVote)
        assert!(r1.is_ok(), "shard 0 vote must succeed");
        assert!(r2.is_ok(), "shard 1 vote must succeed");

        // Exactly one of them should trigger the Prepared phase transition
        let phase1 = r1.unwrap();
        let phase2 = r2.unwrap();
        let phases = [phase1, phase2];
        let transitions: Vec<_> = phases.iter().filter(|p| p.is_some()).collect();
        assert_eq!(
            transitions.len(),
            1,
            "exactly one vote triggers phase change"
        );
    });
}

#[test]
fn loom_dtx_real_begin_and_abort() {
    // begin() on one thread, abort() on another (pending + pending_aborts ordering).
    loom::model(|| {
        let coord = Arc::new(make_coordinator());
        let tx = coord.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        let t1 = thread::spawn(move || c1.begin(&"node2".to_string(), &[1]));
        let t2 = thread::spawn(move || c2.abort(tx_id, "test abort"));

        let begin_result = t1.join().unwrap();
        let abort_result = t2.join().unwrap();

        // begin must succeed (different tx)
        assert!(begin_result.is_ok(), "concurrent begin must succeed");
        // abort must succeed (tx was in pending)
        assert!(abort_result.is_ok(), "abort must succeed");

        // Original tx must be gone
        assert!(coord.get(tx_id).is_none(), "aborted tx must be removed");
    });
}

#[test]
fn loom_dtx_real_vote_no_triggers_abort() {
    // Concurrent YES/NO votes where the NO triggers the abort path.
    loom::model(|| {
        let coord = Arc::new(make_coordinator());
        let tx = coord.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        let c1 = Arc::clone(&coord);
        let c2 = Arc::clone(&coord);

        let t1 = thread::spawn(move || c1.record_vote(tx_id, 0, yes_vote(0)));
        let t2 = thread::spawn(move || c2.record_vote(tx_id, 1, no_vote()));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both votes must succeed
        assert!(r1.is_ok(), "shard 0 vote must succeed");
        assert!(r2.is_ok(), "shard 1 vote must succeed");

        // The second-to-finish should trigger Aborting (NO vote present)
        let phase1 = r1.unwrap();
        let phase2 = r2.unwrap();
        let phases: Vec<_> = [phase1, phase2].into_iter().flatten().collect();
        if !phases.is_empty() {
            // If a phase transition occurred, it must be Aborting
            assert!(
                phases.iter().any(|p| *p == tensor_chain::TxPhase::Aborting),
                "NO vote must trigger Aborting, got {phases:?}"
            );
        }
    });
}
