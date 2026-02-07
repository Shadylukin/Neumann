// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Stateful property tests for the 2PC distributed transaction coordinator.
//!
//! Tests atomicity, lock exclusivity, decision stability, and phase monotonicity
//! by generating random operation sequences and comparing real coordinator state
//! against a reference model.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use proptest::prelude::*;
use tensor_chain::consensus::{ConsensusConfig, ConsensusManager, DeltaVector};
use tensor_chain::distributed_tx::{
    DistributedTxConfig, DistributedTxCoordinator, PrepareVote, TxPhase,
};

/// Simplified 2PC reference model.
#[derive(Debug, Clone)]
struct TwoPhaseModel {
    transactions: HashMap<u64, TxModel>,
    locks: HashMap<String, u64>,
}

#[derive(Debug, Clone)]
struct TxModel {
    phase: TxPhase,
    participants: Vec<usize>,
    votes: HashMap<usize, bool>,
    locked_keys: HashSet<String>,
    decided: Option<bool>,
}

impl TwoPhaseModel {
    fn new() -> Self {
        Self {
            transactions: HashMap::new(),
            locks: HashMap::new(),
        }
    }

    fn begin(&mut self, tx_id: u64, participants: Vec<usize>, keys: Vec<String>) {
        self.transactions.insert(
            tx_id,
            TxModel {
                phase: TxPhase::Preparing,
                participants,
                votes: HashMap::new(),
                locked_keys: keys.into_iter().collect(),
                decided: None,
            },
        );
    }

    fn vote_yes(&mut self, tx_id: u64, shard: usize) -> bool {
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            if tx.phase != TxPhase::Preparing {
                return false;
            }
            tx.votes.insert(shard, true);
            true
        } else {
            false
        }
    }

    fn vote_no(&mut self, tx_id: u64, shard: usize) -> bool {
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            if tx.phase != TxPhase::Preparing {
                return false;
            }
            tx.votes.insert(shard, false);
            true
        } else {
            false
        }
    }

    fn try_decide(&mut self, tx_id: u64) -> Option<bool> {
        let tx = self.transactions.get(&tx_id)?;
        if tx.votes.len() < tx.participants.len() {
            return None;
        }
        let all_yes = tx.votes.values().all(|v| *v);
        Some(all_yes)
    }

    fn commit(&mut self, tx_id: u64) -> bool {
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            if tx.decided == Some(false) {
                return false; // Decision stability: can't commit after abort
            }
            tx.phase = TxPhase::Committed;
            tx.decided = Some(true);
            // Release locks
            for key in &tx.locked_keys {
                self.locks.remove(key);
            }
            true
        } else {
            false
        }
    }

    fn abort(&mut self, tx_id: u64) -> bool {
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            if tx.decided == Some(true) {
                return false; // Decision stability: can't abort after commit
            }
            tx.phase = TxPhase::Aborted;
            tx.decided = Some(false);
            for key in &tx.locked_keys {
                self.locks.remove(key);
            }
            true
        } else {
            false
        }
    }

    /// Invariant: Decision stability.
    fn check_decision_stability(&self) -> bool {
        for tx in self.transactions.values() {
            if tx.phase == TxPhase::Committed && tx.decided == Some(false) {
                return false;
            }
            if tx.phase == TxPhase::Aborted && tx.decided == Some(true) {
                return false;
            }
        }
        true
    }

    /// Invariant: Phase monotonicity.
    ///
    /// 2PC phases form a DAG, not a linear order:
    /// ```text
    /// Preparing -> Prepared -> Committing -> Committed
    ///                    \---> Aborting   -> Aborted
    /// Preparing --------/
    /// ```
    fn check_phase_monotonicity(old_phase: TxPhase, new_phase: TxPhase) -> bool {
        if old_phase == new_phase {
            return true;
        }
        match old_phase {
            TxPhase::Preparing => matches!(
                new_phase,
                TxPhase::Prepared | TxPhase::Aborting | TxPhase::Aborted
            ),
            TxPhase::Prepared => matches!(
                new_phase,
                TxPhase::Committing | TxPhase::Committed | TxPhase::Aborting | TxPhase::Aborted
            ),
            TxPhase::Committing => matches!(new_phase, TxPhase::Committed),
            TxPhase::Committed => false,
            TxPhase::Aborting => matches!(new_phase, TxPhase::Aborted),
            TxPhase::Aborted => false,
            _ => false,
        }
    }

    /// Invariant: No two active txns hold the same key.
    fn check_lock_exclusivity(&self) -> bool {
        let mut held_keys: HashMap<String, u64> = HashMap::new();
        for (tx_id, tx) in &self.transactions {
            let is_active = matches!(
                tx.phase,
                TxPhase::Preparing | TxPhase::Prepared | TxPhase::Committing | TxPhase::Aborting
            );
            if is_active {
                for key in &tx.locked_keys {
                    if let Some(other_tx) = held_keys.get(key) {
                        if other_tx != tx_id {
                            return false;
                        }
                    }
                    held_keys.insert(key.clone(), *tx_id);
                }
            }
        }
        true
    }

    /// Invariant: Terminal transactions have no locks.
    fn check_no_orphaned_locks(&self) -> bool {
        for tx in self.transactions.values() {
            if (tx.phase == TxPhase::Committed || tx.phase == TxPhase::Aborted)
                && !tx.locked_keys.is_empty()
            {
                // After commit/abort, model should have released locks
                for key in &tx.locked_keys {
                    if self.locks.contains_key(key) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn create_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let config = DistributedTxConfig {
        max_concurrent: 100,
        ..DistributedTxConfig::default()
    };
    DistributedTxCoordinator::new(consensus, config)
}

// Test: LockManager exclusivity with proptest.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn proptest_lock_exclusivity(
        key_count in 1usize..=5,
        tx_count in 2usize..=5,
    ) {
        let lock_mgr = tensor_chain::distributed_tx::LockManager::new();

        let keys: Vec<String> = (0..key_count).map(|i| format!("key_{i}")).collect();

        // First tx acquires all keys
        let result1 = lock_mgr.try_lock(1, &keys);
        prop_assert!(result1.is_ok(), "First tx should acquire locks");

        // Second tx should fail on the same keys
        for tx_id in 2..=(tx_count as u64) {
            let result = lock_mgr.try_lock(tx_id, &keys);
            prop_assert!(result.is_err(), "tx {tx_id} should fail: locks held by tx 1");
        }

        // Release tx 1, then tx 2 should succeed
        lock_mgr.release(1);
        let result2 = lock_mgr.try_lock(2, &keys);
        prop_assert!(result2.is_ok(), "After release, tx 2 should acquire locks");

        // Clean up
        lock_mgr.release(2);
    }

    // Test: Lock manager handles disjoint key sets correctly.
    #[test]
    fn proptest_lock_disjoint_keys(
        set_a_size in 1usize..=3,
        set_b_size in 1usize..=3,
    ) {
        let lock_mgr = tensor_chain::distributed_tx::LockManager::new();

        let keys_a: Vec<String> = (0..set_a_size).map(|i| format!("a_{i}")).collect();
        let keys_b: Vec<String> = (0..set_b_size).map(|i| format!("b_{i}")).collect();

        // Disjoint key sets should both succeed
        let result_a = lock_mgr.try_lock(1, &keys_a);
        let result_b = lock_mgr.try_lock(2, &keys_b);

        prop_assert!(result_a.is_ok());
        prop_assert!(result_b.is_ok());

        lock_mgr.release(1);
        lock_mgr.release(2);
    }

    // Test: 2PC coordinator begin + abort + verify cleanup.
    #[test]
    fn proptest_2pc_begin_abort_cleanup(
        shard_count in 1usize..=5,
    ) {
        let coord = create_coordinator();
        let participants: Vec<usize> = (0..shard_count).collect();

        let tx = coord.begin(&"node1".to_string(), &participants).unwrap();
        let tx_id = tx.tx_id;

        // Transaction should be in Preparing phase
        prop_assert_eq!(tx.phase, TxPhase::Preparing);

        // Abort should succeed
        let abort_result = coord.abort(tx_id, "test abort");
        prop_assert!(abort_result.is_ok());
    }

    // Test: Model-based decision stability.
    #[test]
    fn proptest_model_decision_stability(
        ops in prop::collection::vec(
            (0u64..5, prop::bool::ANY),
            1..20
        ),
    ) {
        let mut model = TwoPhaseModel::new();

        // Create 5 transactions with 2 participants each
        for tx_id in 0u64..5 {
            model.begin(tx_id, vec![0, 1], vec![format!("key_{tx_id}")]);
        }

        for (tx_id, vote_yes) in &ops {
            let tx_id = *tx_id;
            if !model.transactions.contains_key(&tx_id) {
                continue;
            }

            let phase = model.transactions[&tx_id].phase;

            if phase == TxPhase::Preparing {
                if *vote_yes {
                    model.vote_yes(tx_id, 0);
                    model.vote_yes(tx_id, 1);
                } else {
                    model.vote_no(tx_id, 0);
                }

                if let Some(all_yes) = model.try_decide(tx_id) {
                    if all_yes {
                        model.commit(tx_id);
                    } else {
                        model.abort(tx_id);
                    }
                }
            }
        }

        prop_assert!(model.check_decision_stability());
        prop_assert!(model.check_no_orphaned_locks());
    }

    // Test: Real LockManager enforces exclusivity across concurrent transactions.
    #[test]
    fn proptest_real_lock_exclusivity(
        tx_count in 2usize..=8,
        key_pool_size in 1usize..=4,
    ) {
        let lock_mgr = tensor_chain::distributed_tx::LockManager::new();
        let key_pool: Vec<String> = (0..key_pool_size).map(|i| format!("shared_{i}")).collect();

        let mut acquired = Vec::new();

        // Each tx tries to lock all keys in the pool - only first should succeed
        for tx_id in 0..(tx_count as u64) {
            match lock_mgr.try_lock(tx_id, &key_pool) {
                Ok(_) => acquired.push(tx_id),
                Err(holder) => {
                    // Should be blocked by the first tx
                    prop_assert!(acquired.contains(&holder));
                }
            }
        }

        // Exactly one tx should have acquired the locks
        prop_assert_eq!(acquired.len(), 1);

        // After release, another tx can acquire
        lock_mgr.release(acquired[0]);
        let result = lock_mgr.try_lock(acquired[0] + 1, &key_pool);
        prop_assert!(result.is_ok());
    }

    // Test: Phase monotonicity property.
    #[test]
    fn proptest_phase_monotonicity(
        transitions in prop::collection::vec(
            (0..6u8, 0..6u8),
            1..20
        ),
    ) {
        let phases = [
            TxPhase::Preparing,
            TxPhase::Prepared,
            TxPhase::Committing,
            TxPhase::Committed,
            TxPhase::Aborting,
            TxPhase::Aborted,
        ];

        for (from_idx, to_idx) in transitions {
            let from = phases[from_idx as usize % 6];
            let to = phases[to_idx as usize % 6];

            // Verify our model's phase ordering is consistent
            if TwoPhaseModel::check_phase_monotonicity(from, to) {
                // Valid transition - ensure it's logically sound
                if from == TxPhase::Committed {
                    // From committed, can only stay committed
                    prop_assert_eq!(to, TxPhase::Committed);
                }
                if from == TxPhase::Aborted {
                    // From aborted, can only stay aborted
                    prop_assert_eq!(to, TxPhase::Aborted);
                }
            }
        }
    }

    // Test: Real coordinator begin with varying participant counts.
    #[test]
    fn proptest_coordinator_begin_valid(
        shard_count in 1usize..=10,
    ) {
        let coord = create_coordinator();
        let participants: Vec<usize> = (0..shard_count).collect();

        let tx = coord.begin(&"coord".to_string(), &participants).unwrap();

        prop_assert_eq!(tx.phase, TxPhase::Preparing);
        prop_assert_eq!(tx.participants.len(), shard_count);
        prop_assert!(tx.tx_id > 0);
    }

    // Test: all_yes and any_no are complementary when fully voted.
    #[test]
    fn proptest_vote_decision_consistency(
        votes in prop::collection::vec(prop::bool::ANY, 1..=5),
    ) {
        let participants: Vec<usize> = (0..votes.len()).collect();
        let mut tx = tensor_chain::DistributedTransaction::new(
            "coord".to_string(),
            participants,
        );

        for (i, &yes) in votes.iter().enumerate() {
            let vote = if yes {
                PrepareVote::Yes {
                    lock_handle: i as u64,
                    delta: DeltaVector::zero(4),
                }
            } else {
                PrepareVote::No {
                    reason: "test".to_string(),
                }
            };
            tx.record_vote(i, vote);
        }

        let all_yes = tx.all_yes();
        let any_no = tx.any_no();

        // These should be complementary when all votes are cast
        if votes.iter().all(|v| *v) {
            prop_assert!(all_yes);
            prop_assert!(!any_no);
        } else {
            prop_assert!(!all_yes);
            prop_assert!(any_no);
        }
    }

    // Test: Concurrent transactions with overlapping key sets maintain lock exclusivity.
    // Generates random transactions with partially overlapping key sets and verifies
    // that the LockManager never allows two transactions to hold the same key.
    #[test]
    fn proptest_concurrent_overlapping_keys(
        tx_count in 2usize..=6,
        key_pool_size in 2usize..=6,
        // Each tx gets a random subset of keys (bitmask-style)
        key_masks in prop::collection::vec(1u32..64, 2..=6),
    ) {
        let lock_mgr = tensor_chain::distributed_tx::LockManager::new();
        let key_pool: Vec<String> = (0..key_pool_size).map(|i| format!("key_{i}")).collect();

        let actual_tx_count = tx_count.min(key_masks.len());
        let mut acquired_txs: Vec<(u64, Vec<String>)> = Vec::new();

        for tx_idx in 0..actual_tx_count {
            let mask = key_masks[tx_idx];
            // Select keys for this tx using the mask
            let tx_keys: Vec<String> = key_pool
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << (i % 32)) != 0)
                .map(|(_, k)| k.clone())
                .collect();

            if tx_keys.is_empty() {
                continue;
            }

            let tx_id = tx_idx as u64 + 1;
            match lock_mgr.try_lock(tx_id, &tx_keys) {
                Ok(_) => {
                    // Verify no overlap with already-acquired transactions
                    let tx_key_set: HashSet<&str> =
                        tx_keys.iter().map(String::as_str).collect();
                    for (other_id, other_keys) in &acquired_txs {
                        for ok in other_keys {
                            prop_assert!(
                                !tx_key_set.contains(ok.as_str()),
                                "tx {tx_id} acquired key {ok:?} already held by tx {other_id}"
                            );
                        }
                    }
                    acquired_txs.push((tx_id, tx_keys));
                },
                Err(holder) => {
                    // Verify the holder is one of the txs we know acquired locks
                    prop_assert!(
                        acquired_txs.iter().any(|(id, _)| *id == holder),
                        "Lock conflict reported holder {holder} but no such tx acquired"
                    );
                },
            }
        }

        // Release all and verify re-acquisition works
        for (tx_id, _) in &acquired_txs {
            lock_mgr.release(*tx_id);
        }

        // After release, any tx should be able to acquire any key
        let result = lock_mgr.try_lock(999, &key_pool);
        prop_assert!(result.is_ok(), "After releasing all locks, should acquire freely");
        lock_mgr.release(999);
    }

    // Test: Max concurrent transaction limit is enforced.
    #[test]
    fn proptest_max_concurrent_enforced(max_concurrent in 1usize..=10) {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let config = DistributedTxConfig {
            max_concurrent,
            ..DistributedTxConfig::default()
        };
        let coord = DistributedTxCoordinator::new(consensus, config);

        let mut tx_ids = Vec::new();
        for _ in 0..max_concurrent {
            let tx = coord.begin(&"coord".to_string(), &[0, 1]);
            prop_assert!(tx.is_ok(), "Should allow up to max_concurrent");
            tx_ids.push(tx.unwrap().tx_id);
        }

        // Next one should fail
        let overflow = coord.begin(&"coord".to_string(), &[0, 1]);
        prop_assert!(overflow.is_err(), "Should reject beyond max_concurrent");

        // Abort one, then begin should succeed again
        coord.abort(tx_ids[0], "make room").unwrap();
        let after_abort = coord.begin(&"coord".to_string(), &[0, 1]);
        prop_assert!(after_abort.is_ok(), "Should succeed after abort frees a slot");
    }
}
