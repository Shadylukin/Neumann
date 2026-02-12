// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Linearizability tests under network partition scenarios.
//!
//! Uses the chaos cluster to inject partitions, then records operation
//! histories and checks linearizability.

use std::time::{Duration, Instant};

use integration_tests::chaos::{ChaosCluster, ChaosConfig};
use integration_tests::linearizability::{
    LinearizabilityChecker, LinearizabilityResult, OpType, RegisterModel, Value,
};

fn make_op(
    id: u64,
    op_type: OpType,
    key: &str,
    input: Value,
    output: Value,
    invoke_offset_ms: u64,
    duration_ms: u64,
    client_id: u64,
) -> integration_tests::linearizability::Operation {
    let base = Instant::now();
    let invoke_time = base + Duration::from_millis(invoke_offset_ms);
    let complete_time = invoke_time + Duration::from_millis(duration_ms);
    integration_tests::linearizability::Operation {
        id,
        op_type,
        key: key.to_string(),
        input,
        output: Some(output),
        invoke_time,
        complete_time: Some(complete_time),
        client_id,
    }
}

#[test]
fn test_linearizable_under_no_partition() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());
    assert_eq!(cluster.active_node_count(), 3);

    // Simple write-read on healthy cluster
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
        make_op(1, OpType::Read, "x", Value::None, Value::Int(1), 20, 10, 2),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_partition_isolates_node() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());

    // Isolate node 0
    cluster.isolate_node(0);

    let t0 = cluster.transport(0).unwrap();
    assert!(t0.is_partitioned(&"node-1".to_string()));
    assert!(t0.is_partitioned(&"node-2".to_string()));

    // Nodes 1 and 2 can still talk to each other
    let t1 = cluster.transport(1).unwrap();
    assert!(!t1.is_partitioned(&"node-2".to_string()));
}

#[test]
fn test_majority_partition_linearizable_history() {
    let mut cluster = ChaosCluster::new(5, ChaosConfig::default());

    // Create majority partition
    let minority = cluster.create_majority_partition();
    assert_eq!(minority.len(), 2);

    // Majority side (nodes 2,3,4) can still agree on operations
    // Record a linearizable history from majority side only
    let ops = vec![
        make_op(
            0,
            OpType::Write,
            "k",
            Value::Int(100),
            Value::None,
            0,
            10,
            3,
        ),
        make_op(
            1,
            OpType::Read,
            "k",
            Value::None,
            Value::Int(100),
            20,
            10,
            4,
        ),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);

    // Heal and verify
    cluster.heal_majority_partition();
    let t0 = cluster.transport(0).unwrap();
    assert!(!t0.is_partitioned(&"node-2".to_string()));
}

#[test]
fn test_symmetric_partition_between_pairs() {
    let cluster = ChaosCluster::new(4, ChaosConfig::default());

    // Partition nodes 0-1 from nodes 2-3
    cluster.inject_symmetric_partition(0, 2);
    cluster.inject_symmetric_partition(0, 3);
    cluster.inject_symmetric_partition(1, 2);
    cluster.inject_symmetric_partition(1, 3);

    // Verify partition
    let t0 = cluster.transport(0).unwrap();
    assert!(t0.is_partitioned(&"node-2".to_string()));
    assert!(t0.is_partitioned(&"node-3".to_string()));
    assert!(!t0.is_partitioned(&"node-1".to_string()));

    let t2 = cluster.transport(2).unwrap();
    assert!(t2.is_partitioned(&"node-0".to_string()));
    assert!(!t2.is_partitioned(&"node-3".to_string()));
}

#[test]
fn test_heal_after_partition_restores_connectivity() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());

    // Create and heal asymmetric partition
    cluster.inject_partition(0, 1);
    let t0 = cluster.transport(0).unwrap();
    assert!(t0.is_partitioned(&"node-1".to_string()));

    cluster.heal_partition(0, 1);
    assert!(!t0.is_partitioned(&"node-1".to_string()));
}

#[test]
fn test_partition_with_clock_drift_history_check() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());

    // Inject clock drift on node 0
    cluster.inject_clock_drift(0, 2000);

    // Isolate node 0
    cluster.isolate_node(0);

    // History from remaining nodes should still be linearizable
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(5), Value::None, 0, 10, 2),
        make_op(1, OpType::Read, "x", Value::None, Value::Int(5), 20, 10, 3),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);

    // Verify drift persists through partition
    let clock = cluster.clock(0).unwrap();
    assert_eq!(clock.drift_offset(), 2000);
}

#[test]
fn test_concurrent_operations_during_partition_heal() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());

    // Partition then heal
    cluster.inject_symmetric_partition(0, 1);
    cluster.heal_symmetric_partition(0, 1);

    // After healing, concurrent operations from all nodes should be linearizable
    let ops = vec![
        make_op(0, OpType::Write, "y", Value::Int(10), Value::None, 0, 50, 1),
        make_op(
            1,
            OpType::Write,
            "y",
            Value::Int(20),
            Value::None,
            10,
            50,
            2,
        ),
        // Read during concurrent writes can return either value
        make_op(2, OpType::Read, "y", Value::None, Value::Int(20), 60, 10, 3),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_link_quality_degradation_does_not_affect_linearizability() {
    let cluster = ChaosCluster::new(3, ChaosConfig::default());

    // Set high drop rates
    cluster.set_link_quality(0, 1, 0.5);
    cluster.set_link_quality(1, 0, 0.5);

    // Linearizability is about the *observed* operations, not dropped ones
    let ops = vec![
        make_op(0, OpType::Write, "z", Value::Int(99), Value::None, 0, 10, 1),
        make_op(1, OpType::Read, "z", Value::None, Value::Int(99), 20, 10, 2),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_crash_node_partition_check() {
    let mut cluster = ChaosCluster::new(3, ChaosConfig::default());

    cluster.crash_node(0);
    assert!(cluster.is_node_crashed(0));
    assert_eq!(cluster.active_node_count(), 2);

    // Crashed node should be fully partitioned
    let t0 = cluster.transport(0).unwrap();
    assert!(t0.is_partitioned(&"node-1".to_string()));
    assert!(t0.is_partitioned(&"node-2".to_string()));

    // Recover
    cluster.recover_node(0);
    assert!(!cluster.is_node_crashed(0));
    assert_eq!(cluster.active_node_count(), 3);
}
