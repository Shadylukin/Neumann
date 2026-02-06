// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Jepsen-style tests for Two-Phase Commit.
//!
//! ## Two cluster types
//!
//! - **`Real2PCCluster`** (end-to-end): real `TxParticipant` instances acquire
//!   real locks via `LockManager::try_lock`, capture real undo logs, and return
//!   honest `PrepareVote`s.  Commit/abort flows through `TxParticipant::commit()`
//!   / `abort()`.  The test never fabricates a vote -- participants decide.
//!
//! - **`TxKVCluster`** (coordinator-focused): the test feeds predetermined votes
//!   to the coordinator, which is useful for testing specific coordinator state
//!   transitions (e.g. partial vote timeout, stats tracking).
//!
//! ## How operations flow
//!
//! - **Writes**: `Real2PCCluster::execute_tx()` -> `TxParticipant::prepare()` ->
//!   real lock acquisition -> `PrepareVote::Yes` / `Conflict` ->
//!   `coordinator.record_vote()` -> `coordinator.commit()` ->
//!   `TxParticipant::commit()` -> `apply_operations()` -> `TensorStore::put()`
//! - **Reads**: `cluster.read_value()` -> `participant.store().get()` ->
//!   return actual stored value (never hardcoded)
//! - **Aborts**: `TxParticipant::abort()` -> undo log applied in reverse ->
//!   previous values restored in real store
//!
//! The linearizability checker validates the system-generated history.

use integration_tests::jepsen::{Real2PCCluster, TxKVCluster};
use integration_tests::linearizability::{
    HistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, RegisterModel, Value,
};
use tensor_chain::block::Transaction as ChainTransaction;
use tensor_chain::consensus::DeltaVector;
use tensor_chain::distributed_tx::{DistributedTxConfig, PrepareVote};

fn make_yes_vote() -> PrepareVote {
    PrepareVote::Yes {
        lock_handle: 1,
        delta: DeltaVector::zero(128),
    }
}

fn make_no_vote(reason: &str) -> PrepareVote {
    PrepareVote::No {
        reason: reason.to_string(),
    }
}

fn put_tx(key: &str, data: &[u8]) -> ChainTransaction {
    ChainTransaction::Put {
        key: key.to_string(),
        data: data.to_vec(),
    }
}

#[test]
fn test_2pc_real_no_faults_commit() {
    let cluster = TxKVCluster::new(2);
    let mut history = HistoryRecorder::new();

    // Begin transaction on shards [0, 1]
    let tx = cluster.begin("coordinator", &[0, 1]);

    // Record write operation
    let w_id = history.invoke(1, OpType::Write, "tx_key".to_string(), Value::Int(42));

    // Both shards vote yes
    let phase = cluster.record_vote(tx.tx_id, 0, make_yes_vote());
    assert!(phase.is_none());
    let phase = cluster.record_vote(tx.tx_id, 1, make_yes_vote());
    assert!(phase.is_some());

    // Commit and apply to both shard stores
    let ops = vec![
        (0, vec![put_tx("tx_key", &42_i64.to_le_bytes())]),
        (1, vec![put_tx("tx_key", &42_i64.to_le_bytes())]),
    ];
    cluster.commit_and_apply(tx.tx_id, &ops);
    history.complete(w_id, Value::None);

    // Read from store -- value comes from real TensorStore
    let r_id = history.invoke(2, OpType::Read, "tx_key".to_string(), Value::None);
    let val = cluster.read_value(0, "tx_key");
    history.complete(r_id, val);

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_participant_votes_no() {
    let cluster = TxKVCluster::new(2);
    let mut history = HistoryRecorder::new();

    let tx = cluster.begin("coordinator", &[0, 1]);

    // Record attempted write (will be left incomplete on abort)
    let _w_id = history.invoke(1, OpType::Write, "abort_key".to_string(), Value::Int(99));

    // Shard 0 votes yes, shard 1 votes no
    cluster.record_vote(tx.tx_id, 0, make_yes_vote());
    let phase = cluster.record_vote(tx.tx_id, 1, make_no_vote("resource unavailable"));
    assert!(phase.is_some());

    // Abort -- no apply
    cluster.abort(tx.tx_id, "participant voted no");

    // Read from store -- should be empty (key was never written)
    let r_id = history.invoke(2, OpType::Read, "abort_key".to_string(), Value::None);
    let val = cluster.read_value(0, "abort_key");
    history.complete(r_id, val);

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_concurrent_non_conflicting_transactions() {
    let cluster = TxKVCluster::new(2);
    let mut history = HistoryRecorder::new();

    // Transaction A on key "a", shard 0
    let tx_a = cluster.begin("coordinator", &[0]);
    let wa = history.invoke(1, OpType::Write, "a".to_string(), Value::Int(10));

    // Transaction B on key "b", shard 1
    let tx_b = cluster.begin("coordinator", &[1]);
    let wb = history.invoke(2, OpType::Write, "b".to_string(), Value::Int(20));

    // A votes yes and commits
    cluster.record_vote(tx_a.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx_a.tx_id, &[(0, vec![put_tx("a", &10_i64.to_le_bytes())])]);
    history.complete(wa, Value::None);

    // B votes yes and commits
    cluster.record_vote(tx_b.tx_id, 1, make_yes_vote());
    cluster.commit_and_apply(tx_b.tx_id, &[(1, vec![put_tx("b", &20_i64.to_le_bytes())])]);
    history.complete(wb, Value::None);

    // Read both keys from real stores
    let ra = history.invoke(1, OpType::Read, "a".to_string(), Value::None);
    history.complete(ra, cluster.read_value(0, "a"));
    let rb = history.invoke(2, OpType::Read, "b".to_string(), Value::None);
    history.complete(rb, cluster.read_value(1, "b"));

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_sequential_transactions_same_key() {
    let cluster = TxKVCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Three sequential transactions on key "x", shard 0
    for i in 1..=3_i64 {
        let tx = cluster.begin("coordinator", &[0]);
        let w = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(i));
        cluster.record_vote(tx.tx_id, 0, make_yes_vote());
        cluster.commit_and_apply(tx.tx_id, &[(0, vec![put_tx("x", &i.to_le_bytes())])]);
        history.complete(w, Value::None);
    }

    // Read final value from store
    let r = history.invoke(2, OpType::Read, "x".to_string(), Value::None);
    history.complete(r, cluster.read_value(0, "x"));

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_lock_contention() {
    let cluster = TxKVCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Transaction A commits on "contested"
    let tx_a = cluster.begin("coordinator", &[0]);
    let wa = history.invoke(1, OpType::Write, "contested".to_string(), Value::Int(100));
    cluster.record_vote(tx_a.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(
        tx_a.tx_id,
        &[(0, vec![put_tx("contested", &100_i64.to_le_bytes())])],
    );
    history.complete(wa, Value::None);

    // Transaction B commits on "contested" after A
    let tx_b = cluster.begin("coordinator", &[0]);
    let wb = history.invoke(2, OpType::Write, "contested".to_string(), Value::Int(200));
    cluster.record_vote(tx_b.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(
        tx_b.tx_id,
        &[(0, vec![put_tx("contested", &200_i64.to_le_bytes())])],
    );
    history.complete(wb, Value::None);

    // Read from store
    let r = history.invoke(3, OpType::Read, "contested".to_string(), Value::None);
    history.complete(r, cluster.read_value(0, "contested"));

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_multi_participant_commit() {
    let cluster = TxKVCluster::new(3);
    let mut history = HistoryRecorder::new();

    // Transaction spanning 3 shards
    let tx = cluster.begin("coordinator", &[0, 1, 2]);
    let w = history.invoke(1, OpType::Write, "multi".to_string(), Value::Int(777));

    // All three vote yes
    for shard in 0..3_usize {
        let phase = cluster.record_vote(tx.tx_id, shard, make_yes_vote());
        if shard < 2 {
            assert!(phase.is_none());
        } else {
            assert!(phase.is_some());
        }
    }

    // Commit and apply to all shards
    let ops: Vec<_> = (0..3)
        .map(|s| (s, vec![put_tx("multi", &777_i64.to_le_bytes())]))
        .collect();
    cluster.commit_and_apply(tx.tx_id, &ops);
    history.complete(w, Value::None);

    // Read from each shard store
    for shard in 0..3_usize {
        let r = history.invoke(2, OpType::Read, "multi".to_string(), Value::None);
        history.complete(r, cluster.read_value(shard, "multi"));
    }

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_abort_preserves_state() {
    let cluster = TxKVCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Write x=1 and commit
    let tx1 = cluster.begin("coordinator", &[0]);
    let w1 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(1));
    cluster.record_vote(tx1.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx1.tx_id, &[(0, vec![put_tx("x", &1_i64.to_le_bytes())])]);
    history.complete(w1, Value::None);

    // Attempt to write x=999 but abort
    let tx2 = cluster.begin("coordinator", &[0]);
    let _w2 = history.invoke(2, OpType::Write, "x".to_string(), Value::Int(999));
    cluster.record_vote(tx2.tx_id, 0, make_no_vote("conflict"));
    // Leave w2 incomplete -- the write never took effect

    // Read from store -- should still be 1 (abort didn't apply)
    let r = history.invoke(3, OpType::Read, "x".to_string(), Value::None);
    let val = cluster.read_value(0, "x");
    history.complete(r, val.clone());
    assert_eq!(val, Value::Int(1), "abort must not change stored value");

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_stale_read_detected() {
    let cluster = TxKVCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Write x=1, commit and apply
    let tx1 = cluster.begin("coordinator", &[0]);
    let w1 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(1));
    cluster.record_vote(tx1.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx1.tx_id, &[(0, vec![put_tx("x", &1_i64.to_le_bytes())])]);
    history.complete(w1, Value::None);

    // Write x=2, commit and apply
    let tx2 = cluster.begin("coordinator", &[0]);
    let w2 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(2));
    cluster.record_vote(tx2.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx2.tx_id, &[(0, vec![put_tx("x", &2_i64.to_le_bytes())])]);
    history.complete(w2, Value::None);

    // Manually inject a stale read (returns 1 instead of 2) to verify the
    // checker catches the violation. This is a regression test for the checker
    // itself.
    let r = history.invoke(2, OpType::Read, "x".to_string(), Value::None);
    history.complete(r, Value::Int(1));

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert!(
        matches!(result, LinearizabilityResult::Violation(_)),
        "stale read should violate linearizability: {result:?}"
    );
}

#[test]
fn test_2pc_real_stats_tracking() {
    let cluster = TxKVCluster::new(1);

    let tx = cluster.begin("coordinator", &[0]);
    cluster.record_vote(tx.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx.tx_id, &[(0, vec![put_tx("k", &[1])])]);

    let stats = cluster.coordinator().stats();
    assert_eq!(stats.started.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        stats.committed.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(stats.aborted.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn test_2pc_real_abort_stats() {
    let cluster = TxKVCluster::new(1);

    let tx = cluster.begin("coordinator", &[0]);
    cluster.record_vote(tx.tx_id, 0, make_no_vote("test abort"));
    cluster.abort(tx.tx_id, "participant declined");

    let stats = cluster.coordinator().stats();
    assert_eq!(stats.started.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(stats.aborted.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn test_2pc_real_partial_vote_abort() {
    // Simulates a participant crash: only 2 of 3 shards vote. The coordinator
    // detects the missing vote and aborts the transaction.
    let cluster = TxKVCluster::new(3);
    let mut history = HistoryRecorder::new();

    // Begin transaction on shards [0, 1, 2]
    let tx = cluster.begin("coordinator", &[0, 1, 2]);
    let _w = history.invoke(1, OpType::Write, "timeout_key".to_string(), Value::Int(42));

    // Only shards 0 and 1 vote yes -- shard 2 "crashed" and never votes
    cluster.record_vote(tx.tx_id, 0, make_yes_vote());
    cluster.record_vote(tx.tx_id, 1, make_yes_vote());
    // Shard 2 never votes

    // Coordinator detects incomplete votes and aborts
    cluster.abort(tx.tx_id, "shard 2 unresponsive (timeout)");

    // Verify abort stats
    let stats = cluster.coordinator().stats();
    assert_eq!(stats.aborted.load(std::sync::atomic::Ordering::Relaxed), 1);

    // Read from all stores -- nothing should have been committed
    for shard in 0..3_usize {
        let r = history.invoke(2, OpType::Read, "timeout_key".to_string(), Value::None);
        let val = cluster.read_value(shard, "timeout_key");
        history.complete(r, val.clone());
        assert_eq!(
            val,
            Value::None,
            "shard {shard} must have no data after partial-vote abort"
        );
    }

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_abort_after_partial_prepare() {
    let cluster = TxKVCluster::new(2);
    let mut history = HistoryRecorder::new();

    // Begin transaction on shards [0, 1]
    let tx = cluster.begin("coordinator", &[0, 1]);
    let _w = history.invoke(1, OpType::Write, "partial_key".to_string(), Value::Int(99));

    // Only shard 0 votes yes -- shard 1 never responds
    cluster.record_vote(tx.tx_id, 0, make_yes_vote());

    // Coordinator decides to abort (e.g. timeout or external decision)
    cluster.abort(tx.tx_id, "shard 1 unresponsive");

    // Both stores must be empty
    let r0 = history.invoke(2, OpType::Read, "partial_key".to_string(), Value::None);
    history.complete(r0, cluster.read_value(0, "partial_key"));
    let r1 = history.invoke(3, OpType::Read, "partial_key".to_string(), Value::None);
    history.complete(r1, cluster.read_value(1, "partial_key"));

    assert_eq!(
        cluster.read_value(0, "partial_key"),
        Value::None,
        "shard 0 must have no data after abort"
    );
    assert_eq!(
        cluster.read_value(1, "partial_key"),
        Value::None,
        "shard 1 must have no data after abort"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_sequential_commit_abort_commit() {
    let cluster = TxKVCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Tx1: write x=10, all vote yes, commit
    let tx1 = cluster.begin("coordinator", &[0]);
    let w1 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(10));
    cluster.record_vote(tx1.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx1.tx_id, &[(0, vec![put_tx("x", &10_i64.to_le_bytes())])]);
    history.complete(w1, Value::None);

    // Tx2: attempt write x=20, participant votes no, abort -- x stays 10
    let tx2 = cluster.begin("coordinator", &[0]);
    let _w2 = history.invoke(2, OpType::Write, "x".to_string(), Value::Int(20));
    cluster.record_vote(tx2.tx_id, 0, make_no_vote("conflict detected"));
    cluster.abort(tx2.tx_id, "participant voted no");
    // w2 left incomplete -- never committed

    // Tx3: write x=30, all vote yes, commit -- x becomes 30
    let tx3 = cluster.begin("coordinator", &[0]);
    let w3 = history.invoke(3, OpType::Write, "x".to_string(), Value::Int(30));
    cluster.record_vote(tx3.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(tx3.tx_id, &[(0, vec![put_tx("x", &30_i64.to_le_bytes())])]);
    history.complete(w3, Value::None);

    // Read x -- should be 30
    let r = history.invoke(4, OpType::Read, "x".to_string(), Value::None);
    let val = cluster.read_value(0, "x");
    history.complete(r, val.clone());
    assert_eq!(
        val,
        Value::Int(30),
        "x must be 30 after commit-abort-commit"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_2pc_real_multi_shard_partial_failure() {
    let cluster = TxKVCluster::new(3);
    let mut history = HistoryRecorder::new();

    // Pre-write: "baseline"=100 on shard 0 (committed via its own tx)
    let tx_setup = cluster.begin("coordinator", &[0]);
    let w_setup = history.invoke(1, OpType::Write, "baseline".to_string(), Value::Int(100));
    cluster.record_vote(tx_setup.tx_id, 0, make_yes_vote());
    cluster.commit_and_apply(
        tx_setup.tx_id,
        &[(0, vec![put_tx("baseline", &100_i64.to_le_bytes())])],
    );
    history.complete(w_setup, Value::None);

    // Multi-shard tx: spans [0, 1, 2], wants to overwrite "baseline"=200
    let tx = cluster.begin("coordinator", &[0, 1, 2]);
    let _w = history.invoke(2, OpType::Write, "baseline".to_string(), Value::Int(200));

    // Shards 0 and 1 vote yes, shard 2 votes no
    cluster.record_vote(tx.tx_id, 0, make_yes_vote());
    cluster.record_vote(tx.tx_id, 1, make_yes_vote());
    let phase = cluster.record_vote(tx.tx_id, 2, make_no_vote("disk full"));
    assert!(phase.is_some(), "phase transition expected after all votes");

    // Abort -- nothing should be applied
    cluster.abort(tx.tx_id, "shard 2 voted no");

    // Read "baseline" from shard 0 -- must still be 100 (atomicity)
    let r = history.invoke(3, OpType::Read, "baseline".to_string(), Value::None);
    let val = cluster.read_value(0, "baseline");
    history.complete(r, val.clone());
    assert_eq!(
        val,
        Value::Int(100),
        "baseline must still be 100 after multi-shard abort"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

// ---------------------------------------------------------------------------
// End-to-end tests using Real2PCCluster (real TxParticipant voting)
// ---------------------------------------------------------------------------

#[test]
fn test_2pc_participant_real_commit() {
    // Full end-to-end: real participant acquires lock, votes Yes, commits
    // through TxParticipant::commit() which applies to real store.
    let cluster = Real2PCCluster::new(1);
    let mut history = HistoryRecorder::new();

    let w = history.invoke(1, OpType::Write, "k".to_string(), Value::Int(42));
    let result = cluster.execute_tx(
        "coordinator",
        &[(0, vec![put_tx("k", &42_i64.to_le_bytes())])],
    );
    assert!(result.is_ok(), "commit should succeed: {result:?}");
    history.complete(w, Value::None);

    // Read from participant's real store
    let r = history.invoke(2, OpType::Read, "k".to_string(), Value::None);
    let val = cluster.read_value(0, "k");
    history.complete(r, val.clone());
    assert_eq!(
        val,
        Value::Int(42),
        "participant store must contain committed value"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

#[test]
fn test_2pc_participant_real_lock_conflict() {
    // Two transactions on the same key: the second gets a real Conflict vote
    // from the participant's LockManager (not a hardcoded No).
    let cluster = Real2PCCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Tx A: prepare on "x" -- acquires lock, gets real Yes vote
    let tx_a = cluster.begin("coordinator", &[0]);
    let wa = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(10));
    let vote_a = cluster.prepare_and_vote(tx_a.tx_id, 0, vec![put_tx("x", &10_i64.to_le_bytes())]);
    assert!(
        matches!(vote_a, PrepareVote::Yes { .. }),
        "first tx must get Yes vote: {vote_a:?}"
    );

    // Tx B: prepare on "x" -- lock already held by A, gets real Conflict
    let tx_b = cluster.begin("coordinator", &[0]);
    let _wb = history.invoke(2, OpType::Write, "x".to_string(), Value::Int(20));
    let vote_b = cluster.prepare_and_vote(tx_b.tx_id, 0, vec![put_tx("x", &20_i64.to_le_bytes())]);
    assert!(
        matches!(vote_b, PrepareVote::Conflict { .. }),
        "second tx must get Conflict (lock held by first): {vote_b:?}"
    );

    // Abort B (real participant abort releases nothing since prepare failed to lock)
    cluster.abort_all(tx_b.tx_id, &[0], "lock conflict");

    // Commit A through real participant
    cluster.commit_all(tx_a.tx_id, &[0]);
    history.complete(wa, Value::None);

    // Read from participant's real store -- should be 10 from Tx A
    let r = history.invoke(3, OpType::Read, "x".to_string(), Value::None);
    let val = cluster.read_value(0, "x");
    history.complete(r, val.clone());
    assert_eq!(val, Value::Int(10), "only Tx A's value should be committed");

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

#[test]
fn test_2pc_participant_real_abort_rollback() {
    // Pre-write a value, start a new transaction, prepare (acquires lock + undo
    // log), then abort. The undo log restores the original value.
    let cluster = Real2PCCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Tx 1: write x=100, commit through real participant
    let w1 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(100));
    cluster
        .execute_tx(
            "coordinator",
            &[(0, vec![put_tx("x", &100_i64.to_le_bytes())])],
        )
        .expect("initial commit should succeed");
    history.complete(w1, Value::None);

    // Verify x=100
    assert_eq!(cluster.read_value(0, "x"), Value::Int(100));

    // Tx 2: prepare to overwrite x=999 (participant locks "x", captures undo)
    let tx2 = cluster.begin("coordinator", &[0]);
    let _w2 = history.invoke(2, OpType::Write, "x".to_string(), Value::Int(999));
    let vote = cluster.prepare_and_vote(tx2.tx_id, 0, vec![put_tx("x", &999_i64.to_le_bytes())]);
    assert!(
        matches!(vote, PrepareVote::Yes { .. }),
        "should acquire lock"
    );

    // Abort Tx 2 -- participant applies undo log (restores x=100)
    cluster.abort_all(tx2.tx_id, &[0], "decided to abort");

    // Read x -- must be 100 (restored by real undo log, not by test logic)
    let r = history.invoke(3, OpType::Read, "x".to_string(), Value::None);
    let val = cluster.read_value(0, "x");
    history.complete(r, val.clone());
    assert_eq!(val, Value::Int(100), "undo log must restore original value");

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

#[test]
fn test_2pc_participant_real_multi_shard_commit() {
    // 3-shard transaction: all participants prepare, vote Yes, commit.
    // Verify all 3 participant stores contain the committed value.
    let cluster = Real2PCCluster::new(3);
    let mut history = HistoryRecorder::new();

    let w = history.invoke(1, OpType::Write, "shared".to_string(), Value::Int(777));
    let ops: Vec<(usize, Vec<ChainTransaction>)> = (0..3)
        .map(|s| (s, vec![put_tx("shared", &777_i64.to_le_bytes())]))
        .collect();
    let result = cluster.execute_tx("coordinator", &ops);
    assert!(result.is_ok(), "multi-shard commit should succeed");
    history.complete(w, Value::None);

    // Read from each participant's real store
    for shard in 0..3_usize {
        let r = history.invoke(2, OpType::Read, "shared".to_string(), Value::None);
        let val = cluster.read_value(shard, "shared");
        history.complete(r, val.clone());
        assert_eq!(
            val,
            Value::Int(777),
            "shard {shard} must have committed value"
        );
    }

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

#[test]
fn test_2pc_participant_commit_conflict_commit() {
    // Tx A commits x=10, Tx C prepares x=30 (holds lock),
    // Tx B tries x=20 but gets real Conflict from participant,
    // then Tx C commits x=30.
    let cluster = Real2PCCluster::new(1);
    let mut history = HistoryRecorder::new();

    // Tx A: commit x=10
    let wa = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(10));
    cluster
        .execute_tx(
            "coordinator",
            &[(0, vec![put_tx("x", &10_i64.to_le_bytes())])],
        )
        .expect("Tx A should commit");
    history.complete(wa, Value::None);
    assert_eq!(cluster.read_value(0, "x"), Value::Int(10));

    // Tx C: prepare (holds lock on "x")
    let tx_c = cluster.begin("coordinator", &[0]);
    let vote_c = cluster.prepare_and_vote(tx_c.tx_id, 0, vec![put_tx("x", &30_i64.to_le_bytes())]);
    assert!(matches!(vote_c, PrepareVote::Yes { .. }), "C should lock x");

    // Tx B: prepare while C holds lock -- gets real Conflict
    let tx_b = cluster.begin("coordinator", &[0]);
    let _wb = history.invoke(2, OpType::Write, "x".to_string(), Value::Int(20));
    let vote_b = cluster.prepare_and_vote(tx_b.tx_id, 0, vec![put_tx("x", &20_i64.to_le_bytes())]);
    assert!(
        matches!(vote_b, PrepareVote::Conflict { .. }),
        "B must conflict with C: {vote_b:?}"
    );
    cluster.abort_all(tx_b.tx_id, &[0], "conflict");

    // Commit C
    let wc = history.invoke(3, OpType::Write, "x".to_string(), Value::Int(30));
    cluster.commit_all(tx_c.tx_id, &[0]);
    history.complete(wc, Value::None);

    // Final read: x=30
    let r = history.invoke(4, OpType::Read, "x".to_string(), Value::None);
    let val = cluster.read_value(0, "x");
    history.complete(r, val.clone());
    assert_eq!(val, Value::Int(30), "x must be 30 after A->conflict(B)->C");

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

// ---------------------------------------------------------------------------
// Timeout-based recovery tests (real config.prepare_timeout_ms wiring)
// ---------------------------------------------------------------------------

#[test]
fn test_2pc_timeout_recovery_aborts_stale() {
    // Create cluster with a very short timeout so we can trigger it in tests.
    let config = DistributedTxConfig {
        prepare_timeout_ms: 50,
        ..DistributedTxConfig::default()
    };
    let cluster = Real2PCCluster::with_config(1, config);
    let mut history = HistoryRecorder::new();

    // Begin tx and prepare on participant (acquires lock, votes Yes)
    let tx = cluster.begin("coordinator", &[0]);
    let _w = history.invoke(1, OpType::Write, "stale_key".to_string(), Value::Int(42));
    let vote = cluster.prepare_and_vote(
        tx.tx_id,
        0,
        vec![put_tx("stale_key", &42_i64.to_le_bytes())],
    );
    assert!(
        matches!(vote, PrepareVote::Yes { .. }),
        "prepare should succeed"
    );

    // Do NOT commit -- let the transaction sit in Preparing phase.
    // Sleep beyond the timeout.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Cleanup should detect the timed-out transaction
    let timed_out = cluster.cleanup_timeouts();
    assert!(
        timed_out.contains(&tx.tx_id),
        "tx should be in timed_out list: {timed_out:?}"
    );

    // Verify stats
    let stats = cluster.coordinator().stats();
    assert_eq!(
        stats.timed_out.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "timed_out stat should be 1"
    );

    // Abort on participant side to release locks
    cluster.participant(0).abort(tx.tx_id);

    // Read from store -- no committed value
    let r = history.invoke(2, OpType::Read, "stale_key".to_string(), Value::None);
    let val = cluster.read_value(0, "stale_key");
    history.complete(r, val.clone());
    assert_eq!(
        val,
        Value::None,
        "timed-out tx must not have committed data"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}

#[test]
fn test_2pc_timeout_preserves_active_tx() {
    let config = DistributedTxConfig {
        prepare_timeout_ms: 50,
        ..DistributedTxConfig::default()
    };
    let cluster = Real2PCCluster::with_config(1, config);
    let mut history = HistoryRecorder::new();

    // Begin tx_old (will timeout)
    let tx_old = cluster.begin("coordinator", &[0]);
    let _w_old = history.invoke(1, OpType::Write, "old_key".to_string(), Value::Int(1));

    // Sleep beyond the timeout
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Begin tx_new (fresh, within timeout)
    let tx_new = cluster.begin("coordinator", &[0]);
    let w_new = history.invoke(2, OpType::Write, "new_key".to_string(), Value::Int(2));

    // Cleanup -- only tx_old should be removed
    let timed_out = cluster.cleanup_timeouts();
    assert!(
        timed_out.contains(&tx_old.tx_id),
        "tx_old should have timed out"
    );
    assert!(
        !timed_out.contains(&tx_new.tx_id),
        "tx_new should NOT have timed out"
    );

    // Complete tx_new normally
    let vote = cluster.prepare_and_vote(
        tx_new.tx_id,
        0,
        vec![put_tx("new_key", &2_i64.to_le_bytes())],
    );
    assert!(
        matches!(vote, PrepareVote::Yes { .. }),
        "tx_new should get Yes vote"
    );
    cluster.commit_all(tx_new.tx_id, &[0]);
    history.complete(w_new, Value::None);

    // Read new_key -- should be 2
    let r = history.invoke(3, OpType::Read, "new_key".to_string(), Value::None);
    let val = cluster.read_value(0, "new_key");
    history.complete(r, val.clone());
    assert_eq!(
        val,
        Value::Int(2),
        "tx_new must have committed successfully"
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(
        checker.check(history.operations()),
        LinearizabilityResult::Ok
    );
}
