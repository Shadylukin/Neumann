// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for combined chaos failure scenarios.
//!
//! Tests multiple simultaneous fault injection: partitions + drift + corruption.

use std::sync::Arc;

use integration_tests::chaos::{ChaosCluster, ChaosConfig};
use tensor_chain::{MemoryTransport, Message, Transport};

#[tokio::test]
async fn test_corruption_modifies_term() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable 100% corruption
    t1.set_corruption_rate(1.0);

    t1.send(&"node2".to_string(), Message::Ping { term: 10 })
        .await
        .unwrap();

    let (_, msg) = t2.recv().await.unwrap();

    // Term should be corrupted (incremented by 1)
    assert!(matches!(msg, Message::Ping { term: 11 }));
    assert_eq!(t1.corrupted_message_count(), 1);
}

#[tokio::test]
async fn test_corruption_zero_rate_no_modification() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    t1.set_corruption_rate(0.0);

    t1.send(&"node2".to_string(), Message::Ping { term: 42 })
        .await
        .unwrap();

    let (_, msg) = t2.recv().await.unwrap();
    assert!(matches!(msg, Message::Ping { term: 42 }));
    assert_eq!(t1.corrupted_message_count(), 0);
}

#[tokio::test]
async fn test_combined_reordering_and_corruption() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable both reordering and corruption at 100%
    t1.enable_reordering(1.0, 5);
    t1.set_corruption_rate(1.0);

    t1.send(&"node2".to_string(), Message::Ping { term: 10 })
        .await
        .unwrap();

    let (_, msg) = t2.recv().await.unwrap();

    // Both effects should apply: reordered AND corrupted
    assert!(matches!(msg, Message::Ping { term: 11 })); // corrupted term
    assert!(t1.reordered_message_count() > 0);
    assert!(t1.corrupted_message_count() > 0);
}

#[test]
fn test_chaos_cluster_with_config_applies_settings() {
    let config = ChaosConfig {
        partition_probability: 0.0,
        message_reorder_rate: 0.5,
        max_reorder_delay_ms: 50,
        clock_drift_ms: 1000,
        crash_probability: 0.0,
        corruption_rate: 0.1,
        link_drop_rate: 0.0,
    };

    let cluster = ChaosCluster::new(3, config);

    // Clock drift should be applied
    let clock = cluster.clock(0).unwrap();
    assert_eq!(clock.drift_offset(), 1000);

    let clock2 = cluster.clock(1).unwrap();
    assert_eq!(clock2.drift_offset(), 1000);
}

#[test]
fn test_chaos_cluster_crash_and_recover() {
    let config = ChaosConfig::default();
    let mut cluster = ChaosCluster::new(5, config);

    assert_eq!(cluster.active_node_count(), 5);

    // Crash two nodes
    cluster.crash_node(0);
    cluster.crash_node(1);
    assert_eq!(cluster.active_node_count(), 3);

    // Recover one
    cluster.recover_node(0);
    assert_eq!(cluster.active_node_count(), 4);

    // Recover the other
    cluster.recover_node(1);
    assert_eq!(cluster.active_node_count(), 5);
}

#[test]
fn test_chaos_cluster_majority_partition_and_heal() {
    let config = ChaosConfig::default();
    let mut cluster = ChaosCluster::new(5, config);

    let minority = cluster.create_majority_partition();
    assert_eq!(minority.len(), 2);

    // Majority nodes (2,3,4) should still be connected to each other
    let t2 = cluster.transport(2).unwrap();
    assert!(!t2.is_partitioned(&"node-3".to_string()));
    assert!(!t2.is_partitioned(&"node-4".to_string()));

    // Heal
    cluster.heal_majority_partition();

    // All connections restored
    let t0 = cluster.transport(0).unwrap();
    assert!(!t0.is_partitioned(&"node-2".to_string()));
    assert!(!t0.is_partitioned(&"node-3".to_string()));
}

#[test]
fn test_chaos_cluster_clock_drift_per_node() {
    let config = ChaosConfig::default();
    let cluster = ChaosCluster::new(3, config);

    // Apply different drift to each node
    cluster.inject_clock_drift(0, 1000);
    cluster.inject_clock_drift(1, -500);
    cluster.inject_clock_drift(2, 2000);

    assert_eq!(cluster.clock(0).unwrap().drift_offset(), 1000);
    assert_eq!(cluster.clock(1).unwrap().drift_offset(), -500);
    assert_eq!(cluster.clock(2).unwrap().drift_offset(), 2000);
}

#[test]
fn test_chaos_cluster_stats_aggregate() {
    let config = ChaosConfig::default();
    let cluster = ChaosCluster::new(3, config);

    // Initially all zeros
    assert_eq!(cluster.total_dropped_messages(), 0);
    assert_eq!(cluster.total_reordered_messages(), 0);
    assert_eq!(cluster.total_corrupted_messages(), 0);

    let all_stats = cluster.all_chaos_stats();
    assert_eq!(all_stats.len(), 3);
}

#[test]
fn test_chaos_cluster_reset_counters() {
    let config = ChaosConfig::default();
    let cluster = ChaosCluster::new(3, config);

    // Isolate a node to generate drops
    cluster.isolate_node(0);

    cluster.reset_all_counters();

    assert_eq!(cluster.total_dropped_messages(), 0);
    assert_eq!(cluster.total_reordered_messages(), 0);
    assert_eq!(cluster.total_corrupted_messages(), 0);
}

#[test]
fn test_chaos_config_presets_have_increasing_severity() {
    let mild = ChaosConfig::mild();
    let moderate = ChaosConfig::moderate();
    let aggressive = ChaosConfig::aggressive();

    assert!(moderate.partition_probability > mild.partition_probability);
    assert!(aggressive.partition_probability > moderate.partition_probability);

    assert!(moderate.message_reorder_rate > mild.message_reorder_rate);
    assert!(aggressive.message_reorder_rate > moderate.message_reorder_rate);

    assert!(moderate.clock_drift_ms.unsigned_abs() > mild.clock_drift_ms.unsigned_abs());
    assert!(aggressive.clock_drift_ms.unsigned_abs() > moderate.clock_drift_ms.unsigned_abs());
}
