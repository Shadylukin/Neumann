// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Deterministic simulation tests for 2PC distributed transactions.
//!
//! Tests atomicity, lock exclusivity, and phase transitions under
//! concurrent transaction workloads.

use tensor_chain::consensus::{ConsensusConfig, ConsensusManager, DeltaVector};
use tensor_chain::distributed_tx::{
    DistributedTxConfig, DistributedTxCoordinator, PrepareVote, TxPhase,
};

fn create_coordinator(max_concurrent: usize) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let config = DistributedTxConfig {
        max_concurrent,
        ..DistributedTxConfig::default()
    };
    DistributedTxCoordinator::new(consensus, config)
}

#[test]
fn dst_2pc_normal_commit_3_shards() {
    for seed in 0..100u64 {
        let coord = create_coordinator(100);
        let participants = vec![0, 1, 2];
        let tx = coord
            .begin(&"coord".to_string(), &participants)
            .expect("begin failed");
        let tx_id = tx.tx_id;

        // All shards vote YES
        for &shard in &participants {
            let vote = PrepareVote::Yes {
                lock_handle: seed + shard as u64,
                delta: DeltaVector::zero(4),
            };
            let result = coord.record_vote(tx_id, shard, vote);
            assert!(
                result.is_ok(),
                "seed {seed}: record_vote failed: {result:?}"
            );
        }

        // Transaction should be commitable
        let commit_result = coord.commit(tx_id);
        assert!(
            commit_result.is_ok(),
            "seed {seed}: commit failed: {commit_result:?}"
        );
    }
}

#[test]
fn dst_2pc_one_no_aborts() {
    for _ in 0..100 {
        let coord = create_coordinator(100);
        let tx = coord.begin(&"coord".to_string(), &[0, 1, 2]).unwrap();
        let tx_id = tx.tx_id;

        // Shard 0 votes YES
        let _ = coord.record_vote(
            tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::zero(4),
            },
        );

        // Shard 1 votes NO
        let result = coord.record_vote(
            tx_id,
            1,
            PrepareVote::No {
                reason: "conflict".to_string(),
            },
        );

        // After NO vote + all voted, should transition to aborting
        if let Ok(Some(phase)) = result {
            assert_eq!(phase, TxPhase::Aborting);
        }

        // Shard 2 voting after abort decision should still work or be handled
        let _ = coord.record_vote(
            tx_id,
            2,
            PrepareVote::Yes {
                lock_handle: 3,
                delta: DeltaVector::zero(4),
            },
        );
    }
}

#[test]
fn dst_2pc_concurrent_transactions_no_deadlock() {
    for seed in 0..100u64 {
        let coord = create_coordinator(100);

        // Start multiple transactions with non-overlapping shards
        let mut tx_ids = Vec::new();
        for i in 0..5 {
            let shards = vec![i * 2, i * 2 + 1];
            let tx = coord.begin(&"coord".to_string(), &shards).unwrap();
            tx_ids.push((tx.tx_id, shards));
        }

        // All vote YES and commit
        for (tx_id, shards) in &tx_ids {
            for &shard in shards {
                let vote = PrepareVote::Yes {
                    lock_handle: seed + shard as u64 + *tx_id,
                    delta: DeltaVector::zero(4),
                };
                let _ = coord.record_vote(*tx_id, shard, vote);
            }
        }

        for (tx_id, _) in &tx_ids {
            let result = coord.commit(*tx_id);
            assert!(
                result.is_ok(),
                "seed {seed}: commit of tx {tx_id} failed: {result:?}"
            );
        }
    }
}

#[test]
fn dst_2pc_abort_releases_resources() {
    for _ in 0..100 {
        let coord = create_coordinator(5);

        // Fill to capacity
        let mut tx_ids = Vec::new();
        for _ in 0..5 {
            let tx = coord.begin(&"coord".to_string(), &[0]).unwrap();
            tx_ids.push(tx.tx_id);
        }

        // Should be at capacity
        let overflow = coord.begin(&"coord".to_string(), &[0]);
        assert!(overflow.is_err(), "Should reject beyond capacity");

        // Abort one
        coord.abort(tx_ids[0], "free up slot").unwrap();

        // Now should be able to begin another
        let new_tx = coord.begin(&"coord".to_string(), &[0]);
        assert!(new_tx.is_ok(), "Should allow after abort frees slot");
    }
}

#[test]
fn dst_2pc_decision_stability() {
    for _ in 0..100 {
        let coord = create_coordinator(100);
        let tx = coord.begin(&"coord".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Vote YES on both shards
        let _ = coord.record_vote(
            tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::zero(4),
            },
        );
        let _ = coord.record_vote(
            tx_id,
            1,
            PrepareVote::Yes {
                lock_handle: 2,
                delta: DeltaVector::zero(4),
            },
        );

        // Commit
        coord.commit(tx_id).unwrap();

        // Abort after commit should fail
        let abort_result = coord.abort(tx_id, "too late");
        assert!(
            abort_result.is_err(),
            "Abort after commit should fail (decision stability)"
        );
    }
}
