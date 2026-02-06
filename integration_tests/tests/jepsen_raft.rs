// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Jepsen-style tests for Raft consensus using real `RaftNode` instances.
//!
//! ## How operations flow through real consensus
//!
//! Every test starts real `RaftNode` processes (via `ChaosRaftCluster` or
//! `RealJepsenHarness`), waits for leader election, and sends operations
//! through actual Raft consensus:
//!
//! - **Writes**: `cluster.put()` -> `RaftNode::propose(block)` -> Raft log
//!   replication -> `apply_committed()` -> `TensorStore::set()`
//! - **Reads**: `cluster.get()` -> `apply_committed()` -> `TensorStore::get()`
//!   -> return actual bytes from store
//! - **History**: the recorder captures the real return value from the store,
//!   not a hardcoded constant. The test never tells the system what to return.
//!
//! The linearizability checker then validates the **system-generated** history.
//! If the system produces a non-linearizable history under faults, the checker
//! catches it (see `test_jepsen_real_raft_stale_read_detection`).
//!
//! ## Why `MemoryTransport` (not TCP)
//!
//! These tests use `MemoryTransport` intentionally. A Jepsen test validates
//! consensus logic under fault conditions, not transport reliability.
//! `MemoryTransport` provides every fault primitive needed:
//!
//! - **Message drops**: `set_link_quality(peer, drop_rate)`
//! - **Message reordering**: `enable_reordering(rate, max_delay_ms)`
//! - **Message corruption**: `set_corruption_rate(rate)`
//! - **Network partitions**: `partition(peer)` / `heal(peer)`
//! - **Asymmetric connectivity**: per-peer drop rates
//!
//! TCP would add non-deterministic kernel-level retries, Nagle delays, and
//! connection setup overhead -- all of which reduce test determinism without
//! improving fault coverage. The real Jepsen tool itself intercepts at the
//! network level (iptables), not at the TCP socket level.

use std::sync::Arc;
use std::time::Duration;

use integration_tests::jepsen::{
    ChaosRaftCluster, NemesisAction, NemesisSchedule, RealJepsenHarness,
};
use integration_tests::linearizability::{
    ConcurrentHistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType,
    RegisterModel, Value,
};
use tensor_chain::raft::RaftConfig;

fn fast_raft_config() -> RaftConfig {
    RaftConfig {
        election_timeout: (80, 160),
        heartbeat_interval: 30,
        auto_heartbeat: true,
        enable_pre_vote: false,
        ..RaftConfig::default()
    }
}

#[tokio::test]
async fn test_jepsen_real_raft_no_faults_linearizable() {
    let nemesis = NemesisSchedule::new();
    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write x=42 through real Raft consensus
    assert!(harness.write(1, "x", Value::Int(42)).await);

    // Small delay to ensure apply completes
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Read x from leader -- the value comes from the real TensorStore,
    // not from a hardcoded constant. harness.read() calls cluster.get()
    // which applies committed entries then reads the actual store.
    let val = harness.read(2, "x", leader);
    assert_eq!(
        val,
        Value::Int(42),
        "read must return the value the system actually stored"
    );

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
    assert_eq!(result.operation_count, 2);
    assert_eq!(result.nemesis_actions_applied, 0);
}

#[tokio::test]
async fn test_jepsen_real_raft_multiple_sequential_writes() {
    let nemesis = NemesisSchedule::new();
    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write counter values 0..5
    for i in 0..5 {
        assert!(
            harness.write(1, "counter", Value::Int(i)).await,
            "write {i} failed"
        );
    }

    // Small delay to ensure last apply completes
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Read final value
    let _val = harness.read(2, "counter", leader);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
    assert_eq!(result.operation_count, 6); // 5 writes + 1 read
}

#[tokio::test]
async fn test_jepsen_real_raft_partition_and_heal() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::MajorityPartition)
        .add(Duration::from_millis(200), NemesisAction::HealAll);

    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(5, config, Default::default(), nemesis).await;

    let _leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Apply partition
    harness.apply_nemesis(Duration::ZERO);

    // Allow re-election after partition
    tokio::time::sleep(Duration::from_millis(500)).await;

    let l = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader must be re-elected after partition");
    let _ = harness.write(1, "partitioned_key", Value::Int(100)).await;

    // Heal
    harness.apply_nemesis(Duration::from_millis(300));
    tokio::time::sleep(Duration::from_millis(300)).await;

    let _val = harness.read(2, "partitioned_key", l);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
}

#[tokio::test]
async fn test_jepsen_real_raft_crash_and_recover() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::RandomCrash)
        .add(Duration::from_millis(500), NemesisAction::HealAll);

    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let _leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write before crash
    assert!(harness.write(1, "pre_crash", Value::Int(1)).await);

    // Crash node 0
    harness.apply_nemesis(Duration::ZERO);

    // Wait for new leader if needed
    tokio::time::sleep(Duration::from_millis(500)).await;

    let l = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader must be re-elected after crash");
    let _ = harness.write(2, "post_crash", Value::Int(2)).await;

    // Recover
    harness.apply_nemesis(Duration::from_secs(1));
    tokio::time::sleep(Duration::from_millis(300)).await;

    let _ = harness.read(3, "pre_crash", l);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
}

#[tokio::test]
async fn test_jepsen_real_raft_clock_drift() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::ClockDrift { drift_ms: 5000 })
        .add(Duration::from_millis(500), NemesisAction::HealAll);

    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Inject clock drift
    harness.apply_nemesis(Duration::ZERO);

    // Write and read despite clock drift
    assert!(harness.write(1, "drift_key", Value::Int(99)).await);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _val = harness.read(2, "drift_key", leader);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
    assert_eq!(result.nemesis_actions_applied, 1);
}

#[tokio::test]
async fn test_jepsen_real_raft_link_degradation() {
    let nemesis = NemesisSchedule::new().add(
        Duration::ZERO,
        NemesisAction::LinkDegradation { drop_rate: 0.3 },
    );

    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    harness.apply_nemesis(Duration::ZERO);

    // Write and read with link degradation
    assert!(harness.write(1, "degraded_key", Value::Int(77)).await);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _val = harness.read(2, "degraded_key", leader);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
}

#[tokio::test]
async fn test_jepsen_real_raft_combined_faults() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::RandomCrash)
        .add(
            Duration::from_millis(100),
            NemesisAction::ClockDrift { drift_ms: 1000 },
        )
        .add(
            Duration::from_millis(200),
            NemesisAction::LinkDegradation { drop_rate: 0.2 },
        )
        .add(Duration::from_millis(500), NemesisAction::HealAll);

    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(5, config, Default::default(), nemesis).await;

    let _leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write before faults
    assert!(harness.write(1, "combined", Value::Int(1)).await);

    // Apply faults progressively
    harness.apply_nemesis(Duration::ZERO);
    tokio::time::sleep(Duration::from_millis(300)).await;

    harness.apply_nemesis(Duration::from_millis(150));
    harness.apply_nemesis(Duration::from_millis(250));

    // Allow re-election
    tokio::time::sleep(Duration::from_millis(500)).await;

    if let Some(_l) = harness.wait_for_leader(Duration::from_secs(5)).await {
        let _ = harness.write(3, "combined2", Value::Int(2)).await;
    }

    // Heal
    harness.apply_nemesis(Duration::from_secs(1));
    tokio::time::sleep(Duration::from_millis(300)).await;

    let l = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader must be re-elected after heal");
    let _ = harness.read(4, "combined", l);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
    assert!(result.nemesis_actions_applied >= 3);
}

#[tokio::test]
async fn test_jepsen_real_raft_concurrent_writes_same_key() {
    let nemesis = NemesisSchedule::new();
    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Multiple clients write to the same key sequentially through Raft
    assert!(harness.write(1, "contested", Value::Int(10)).await);
    assert!(harness.write(2, "contested", Value::Int(20)).await);
    assert!(harness.write(3, "contested", Value::Int(30)).await);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Read final value
    let _val = harness.read(4, "contested", leader);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "linearizability check failed: {:?}",
        result.linearizability
    );
    assert_eq!(result.operation_count, 4);
}

#[tokio::test]
async fn test_jepsen_real_raft_stale_read_detection() {
    // Write x=1, then x=2 through Raft. Read from a follower's raw store
    // (without applying committed entries) to get a stale value. The checker
    // must detect the violation.
    let config = fast_raft_config();
    let cluster = ChaosRaftCluster::new(3, config, Default::default()).await;

    let leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write x=1, then x=2 through Raft consensus
    cluster
        .put("x", &1_i64.to_le_bytes(), Duration::from_secs(5))
        .await
        .expect("put x=1 should succeed");
    cluster
        .put("x", &2_i64.to_le_bytes(), Duration::from_secs(5))
        .await
        .expect("put x=2 should succeed");

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Pick a follower node
    let follower = (0..3).find(|&i| i != leader).expect("must have a follower");

    // Read from follower's raw store WITHOUT applying committed entries.
    // The follower store is guaranteed empty because `put()` only calls
    // `apply_committed()` on the leader, and `get_raw()` skips applying.
    let stale_value = cluster.get_raw(follower, "x").map_or(Value::None, |bytes| {
        if bytes.len() == 8 {
            Value::Int(i64::from_le_bytes(bytes.try_into().unwrap()))
        } else {
            Value::None
        }
    });

    assert_eq!(
        stale_value,
        Value::None,
        "follower raw store must be empty (no apply_committed called)"
    );

    // Build a history: write 1, write 2, then read returning None (stale)
    let mut history = integration_tests::linearizability::HistoryRecorder::new();
    let w1 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(1));
    history.complete(w1, Value::None);
    let w2 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(2));
    history.complete(w2, Value::None);
    let r = history.invoke(2, OpType::Read, "x".to_string(), Value::None);
    history.complete(r, stale_value);

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());

    // The read returned None after two writes -- this must violate
    // linearizability since no valid ordering can produce None after
    // both writes have been linearized.
    assert!(
        matches!(result, LinearizabilityResult::Violation(_)),
        "stale read (None) must violate linearizability: {result:?}"
    );

    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_replication_across_nodes() {
    let config = fast_raft_config();
    let cluster = ChaosRaftCluster::new(3, config, Default::default()).await;

    let _leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write through leader
    cluster
        .put("replicated", &42_i64.to_le_bytes(), Duration::from_secs(5))
        .await
        .expect("put should succeed");

    // Allow replication
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Read from all nodes
    cluster.apply_committed_all();
    for i in 0..3 {
        let val = cluster.get(i, "replicated");
        assert!(val.is_some(), "node {i} should have the replicated value");
        let bytes = val.unwrap();
        assert_eq!(bytes.len(), 8);
        let v = i64::from_le_bytes(bytes.try_into().unwrap());
        assert_eq!(v, 42, "node {i} should have value 42");
    }

    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_truly_concurrent_writes() {
    let config = fast_raft_config();
    let cluster = Arc::new(ChaosRaftCluster::new(3, config, Default::default()).await);

    let _leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    let history = Arc::new(ConcurrentHistoryRecorder::new());

    // Spawn 5 concurrent writers to key "x"
    let mut set = tokio::task::JoinSet::new();
    for client_id in 0..5_u64 {
        let c = Arc::clone(&cluster);
        let h = Arc::clone(&history);
        set.spawn(async move {
            #[allow(clippy::cast_possible_wrap)]
            let val = Value::Int(client_id as i64);
            c.concurrent_write(&h, client_id, "x", &val, Duration::from_secs(5))
                .await;
        });
    }

    // Wait for all writers to finish
    while set.join_next().await.is_some() {}

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Read from 3 different nodes
    for node_idx in 0..3_usize {
        #[allow(clippy::cast_possible_truncation)]
        let client_id = 10 + node_idx as u64;
        cluster.concurrent_read(&history, client_id, "x", node_idx);
    }

    // Extract history and check linearizability
    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all history refs should be dropped"))
        .into_inner();

    assert!(
        recorder.completed_operations().len() >= 8,
        "expected at least 8 completed ops, got {}",
        recorder.completed_operations().len()
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "concurrent history should be linearizable: {result:?}"
    );

    // Unwrap the Arc -- shutdown needs ownership
    let cluster =
        Arc::try_unwrap(cluster).unwrap_or_else(|_| panic!("all cluster refs should be dropped"));
    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_sustained_workload() {
    let nemesis = NemesisSchedule::new();
    let config = fast_raft_config();
    let mut harness = RealJepsenHarness::with_config(3, config, Default::default(), nemesis).await;

    let leader = harness
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // 50 writes with 25 interspersed reads (every other write)
    for i in 0..50_i64 {
        assert!(
            harness.write(1, "counter", Value::Int(i)).await,
            "write {i} failed"
        );
        if i % 2 == 1 {
            let _ = harness.read(2, "counter", leader);
        }
    }

    // Final read
    let _ = harness.read(2, "counter", leader);

    let result = harness.check();
    assert!(
        result.is_valid(),
        "sustained workload should be linearizable: {:?}",
        result.linearizability
    );
    // 50 writes + 25 interspersed reads + 1 final read = 76 ops
    assert_eq!(result.operation_count, 76);
}

#[tokio::test]
async fn test_jepsen_real_raft_concurrent_writes_with_chaos() {
    // 5-node cluster with transport-level faults: message reordering, link
    // degradation, and a follower partition. Each writer uses a unique key
    // so Raft-level retries (on leader change) cannot create conflicting
    // histories. The test proves the transport fault primitives are exercised.
    let config = fast_raft_config();
    let cluster = Arc::new(ChaosRaftCluster::new(5, config, Default::default()).await);

    let leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Enable transport-level faults on non-leader nodes
    for idx in 0..5_usize {
        if idx == leader {
            continue;
        }
        if let Some(t) = cluster.chaos().transport(idx) {
            t.enable_reordering(0.1, 20);
        }
    }
    // Degrade one specific follower link
    let degraded_node = (0..5_usize).find(|&i| i != leader).unwrap_or(0);
    if let Some(t) = cluster.chaos().transport(degraded_node) {
        let peer = format!(
            "node-{}",
            (0..5_usize)
                .find(|&i| i != leader && i != degraded_node)
                .unwrap_or(0)
        );
        t.set_link_quality(&peer, 0.1);
    }

    let history = Arc::new(ConcurrentHistoryRecorder::new());

    // Spawn 6 concurrent writers, each to its own key
    let mut set = tokio::task::JoinSet::new();
    for client_id in 0..6_u64 {
        let c = Arc::clone(&cluster);
        let h = Arc::clone(&history);
        set.spawn(async move {
            let key = format!("k{client_id}");
            #[allow(clippy::cast_possible_wrap)]
            let val = Value::Int(client_id as i64);
            c.concurrent_write(&h, client_id, &key, &val, Duration::from_secs(5))
                .await;
        });
    }

    // Mid-way: partition a follower to exercise partition primitives
    tokio::time::sleep(Duration::from_millis(150)).await;
    let follower_to_crash = (0..5_usize).find(|&i| i != leader).unwrap_or(0);
    if let Some(t) = cluster.chaos().transport(follower_to_crash) {
        t.partition_all();
    }

    // Wait for all writers
    while set.join_next().await.is_some() {}

    // Heal and stabilize
    if let Some(t) = cluster.chaos().transport(follower_to_crash) {
        t.heal_all();
    }
    for idx in 0..5_usize {
        if let Some(t) = cluster.chaos().transport(idx) {
            t.enable_reordering(0.0, 0);
        }
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Read each key from leader
    let current_leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should exist after heal");
    for client_id in 0..6_u64 {
        let key = format!("k{client_id}");
        cluster.concurrent_read(&history, 20 + client_id, &key, current_leader);
    }

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all history refs should be dropped"))
        .into_inner();

    assert!(
        recorder.completed_operations().len() >= 6,
        "expected at least 6 completed ops, got {}",
        recorder.completed_operations().len()
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "concurrent history under chaos should be linearizable: {result:?}"
    );

    let cluster =
        Arc::try_unwrap(cluster).unwrap_or_else(|_| panic!("all cluster refs should be dropped"));
    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_multi_key_concurrent() {
    let config = fast_raft_config();
    let cluster = Arc::new(ChaosRaftCluster::new(3, config, Default::default()).await);

    let _leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    let history = Arc::new(ConcurrentHistoryRecorder::new());

    // 3 groups of 3 writers: keys "a", "b", "c"
    let mut set = tokio::task::JoinSet::new();
    let keys = ["a", "b", "c"];
    for (group, key) in keys.iter().enumerate() {
        for offset in 0..3_u64 {
            let c = Arc::clone(&cluster);
            let h = Arc::clone(&history);
            let k = (*key).to_string();
            #[allow(clippy::cast_possible_wrap)]
            let base = (group as i64) * 10;
            let client_id = (group as u64) * 10 + offset;
            set.spawn(async move {
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int(base + offset as i64);
                c.concurrent_write(&h, client_id, &k, &val, Duration::from_secs(5))
                    .await;
            });
        }
    }

    while set.join_next().await.is_some() {}
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Read each key from all 3 nodes
    let mut read_client = 100_u64;
    for key in &keys {
        for node_idx in 0..3_usize {
            cluster.concurrent_read(&history, read_client, key, node_idx);
            read_client += 1;
        }
    }

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all history refs should be dropped"))
        .into_inner();

    // 9 writes + 9 reads = 18 ops
    assert!(
        recorder.completed_operations().len() >= 12,
        "expected at least 12 completed ops, got {}",
        recorder.completed_operations().len()
    );

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "multi-key concurrent history should be linearizable: {result:?}"
    );

    let cluster =
        Arc::try_unwrap(cluster).unwrap_or_else(|_| panic!("all cluster refs should be dropped"));
    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_cas_linearizable() {
    let config = fast_raft_config();
    let cluster = ChaosRaftCluster::new(3, config, Default::default()).await;

    let _leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write x=0 through Raft
    cluster
        .put("x", &0_i64.to_le_bytes(), Duration::from_secs(5))
        .await
        .expect("initial put should succeed");

    let mut history = integration_tests::linearizability::HistoryRecorder::new();

    // Record the initial write
    let w0 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(0));
    history.complete(w0, Value::None);

    // 5 sequential CAS operations: 0->1, 1->2, 2->3, 3->4, 4->5
    for expected in 0..5_i64 {
        let result = cluster
            .cas("x", &Value::Int(expected), Duration::from_secs(5))
            .await
            .expect("CAS should not fail at Raft level");

        let cas_id = history.invoke(2, OpType::Cas, "x".to_string(), Value::Int(expected));
        history.complete(cas_id, result.clone());

        assert_eq!(
            result,
            Value::Int(expected),
            "CAS({expected}) should return old value {expected}"
        );
    }

    // Read final value -- should be 5
    tokio::time::sleep(Duration::from_millis(50)).await;
    let leader = cluster.leader_index().expect("must have leader");
    let final_val = cluster
        .get(leader, "x")
        .map(|b| {
            let arr: [u8; 8] = b.try_into().expect("8 bytes");
            i64::from_le_bytes(arr)
        })
        .expect("x should exist");
    assert_eq!(final_val, 5, "after 5 CAS ops from 0, x should be 5");

    let r = history.invoke(3, OpType::Read, "x".to_string(), Value::None);
    history.complete(r, Value::Int(final_val));

    // 1 write + 5 CAS + 1 read = 7 ops
    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "sequential CAS history should be linearizable: {result:?}"
    );

    cluster.shutdown().await;
}

#[tokio::test]
async fn test_jepsen_real_raft_cas_contention_sequential() {
    // Tests CAS contention: two sequential CAS ops where the second one fails
    // because the value was changed by the first CAS.
    let config = fast_raft_config();
    let cluster = ChaosRaftCluster::new(3, config, Default::default()).await;

    let _leader = cluster
        .wait_for_leader(Duration::from_secs(5))
        .await
        .expect("leader should be elected");

    // Write x=0
    cluster
        .put("x", &0_i64.to_le_bytes(), Duration::from_secs(5))
        .await
        .expect("initial put should succeed");

    let mut history = integration_tests::linearizability::HistoryRecorder::new();

    // Record initial write
    let w0 = history.invoke(1, OpType::Write, "x".to_string(), Value::Int(0));
    history.complete(w0, Value::None);

    // CAS 1: expect 0, should succeed -> returns 0, x becomes 1
    let result1 = cluster
        .cas("x", &Value::Int(0), Duration::from_secs(5))
        .await
        .expect("CAS should not fail at Raft level");
    let cas1 = history.invoke(2, OpType::Cas, "x".to_string(), Value::Int(0));
    history.complete(cas1, result1.clone());
    assert_eq!(result1, Value::Int(0), "first CAS should succeed");

    // CAS 2: expect 0 again, should fail -> x is now 1, returns 1
    let result2 = cluster
        .cas("x", &Value::Int(0), Duration::from_secs(5))
        .await
        .expect("CAS should not fail at Raft level");
    let cas2 = history.invoke(3, OpType::Cas, "x".to_string(), Value::Int(0));
    history.complete(cas2, result2.clone());
    assert_eq!(
        result2,
        Value::Int(1),
        "second CAS should fail (x is now 1)"
    );

    // CAS 3: expect 1, should succeed -> returns 1, x becomes 2
    let result3 = cluster
        .cas("x", &Value::Int(1), Duration::from_secs(5))
        .await
        .expect("CAS should not fail at Raft level");
    let cas3 = history.invoke(4, OpType::Cas, "x".to_string(), Value::Int(1));
    history.complete(cas3, result3.clone());
    assert_eq!(result3, Value::Int(1), "third CAS should succeed");

    // Read final value -- should be 2
    tokio::time::sleep(Duration::from_millis(50)).await;
    let leader = cluster.leader_index().expect("must have leader");
    let final_val = cluster
        .get(leader, "x")
        .map(|b| {
            let arr: [u8; 8] = b.try_into().expect("8 bytes");
            i64::from_le_bytes(arr)
        })
        .expect("x should exist");
    let r = history.invoke(5, OpType::Read, "x".to_string(), Value::None);
    history.complete(r, Value::Int(final_val));
    assert_eq!(final_val, 2);

    // 1 write + 3 CAS + 1 read = 5 ops
    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "CAS contention history should be linearizable: {result:?}"
    );

    cluster.shutdown().await;
}
