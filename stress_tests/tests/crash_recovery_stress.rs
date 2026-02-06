// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stress tests for repeated node crash and recovery cycles.
//!
//! Tests ChaosCluster behavior under crash/recovery scenarios:
//! - Repeated single-node crash/recover cycles
//! - Majority crash and sequential recovery
//! - Combined crash and partition scenarios

use integration_tests::chaos::{ChaosCluster, ChaosConfig};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
#[ignore]
fn test_repeated_crash_recover_cycles() {
    let config = ChaosConfig::default();
    let mut cluster = ChaosCluster::new(5, config);

    let mut rng = ChaCha8Rng::seed_from_u64(77);

    for cycle in 0..200 {
        // Pick a random node to crash
        let victim = rng.random_range(0..5_usize);

        let active_before = cluster.active_node_count();

        // Skip if already crashed (can happen with random selection)
        if cluster.is_node_crashed(victim) {
            cluster.recover_node(victim);
            continue;
        }

        // Crash the node
        cluster.crash_node(victim);

        let active_after_crash = cluster.active_node_count();
        assert_eq!(
            active_after_crash,
            active_before - 1,
            "Active count should decrease by 1 after crashing node {} in cycle {}",
            victim,
            cycle
        );
        assert!(
            cluster.is_node_crashed(victim),
            "Node {} should be marked as crashed in cycle {}",
            victim,
            cycle
        );

        // Recover the node
        cluster.recover_node(victim);

        let active_after_recover = cluster.active_node_count();
        assert_eq!(
            active_after_recover, active_before,
            "Active count should be restored after recovering node {} in cycle {}",
            victim, cycle
        );
        assert!(
            !cluster.is_node_crashed(victim),
            "Node {} should not be crashed after recovery in cycle {}",
            victim,
            cycle
        );
    }

    // Final state: all nodes should be active
    assert_eq!(
        cluster.active_node_count(),
        5,
        "All 5 nodes should be active after 200 crash/recover cycles"
    );

    println!("Completed 200 crash/recover cycles successfully");
}

#[test]
#[ignore]
fn test_majority_crash_recovery() {
    let config = ChaosConfig::default();
    let mut cluster = ChaosCluster::new(5, config);

    assert_eq!(cluster.active_node_count(), 5);

    // Crash 3 nodes (majority for a 5-node cluster)
    cluster.crash_node(0);
    assert_eq!(cluster.active_node_count(), 4);
    assert!(cluster.is_node_crashed(0));

    cluster.crash_node(2);
    assert_eq!(cluster.active_node_count(), 3);
    assert!(cluster.is_node_crashed(2));

    cluster.crash_node(4);
    assert_eq!(cluster.active_node_count(), 2);
    assert!(cluster.is_node_crashed(4));

    // Verify only nodes 1 and 3 are still active
    assert!(!cluster.is_node_crashed(1));
    assert!(!cluster.is_node_crashed(3));

    // Recover nodes one by one
    cluster.recover_node(0);
    assert_eq!(cluster.active_node_count(), 3);
    assert!(!cluster.is_node_crashed(0));

    cluster.recover_node(2);
    assert_eq!(cluster.active_node_count(), 4);
    assert!(!cluster.is_node_crashed(2));

    cluster.recover_node(4);
    assert_eq!(cluster.active_node_count(), 5);
    assert!(!cluster.is_node_crashed(4));

    // All nodes should be active
    for i in 0..5 {
        assert!(
            !cluster.is_node_crashed(i),
            "Node {} should be active after full recovery",
            i
        );
    }

    println!("Majority crash/recovery test passed: 3 of 5 nodes crashed and recovered");
}

#[test]
#[ignore]
fn test_crash_with_partition() {
    let config = ChaosConfig::default();
    let mut cluster = ChaosCluster::new(5, config);

    assert_eq!(cluster.active_node_count(), 5);

    // Crash node 0
    cluster.crash_node(0);
    assert_eq!(cluster.active_node_count(), 4);
    assert!(cluster.is_node_crashed(0));

    // Create partition between nodes {1,2} and {3,4}
    cluster.inject_symmetric_partition(1, 3);
    cluster.inject_symmetric_partition(1, 4);
    cluster.inject_symmetric_partition(2, 3);
    cluster.inject_symmetric_partition(2, 4);

    // Verify partition state
    let t1 = cluster.transport(1).expect("transport 1 should exist");
    let t3 = cluster.transport(3).expect("transport 3 should exist");
    assert!(t1.is_partitioned(&"node-3".to_string()));
    assert!(t1.is_partitioned(&"node-4".to_string()));
    assert!(t3.is_partitioned(&"node-1".to_string()));
    assert!(t3.is_partitioned(&"node-2".to_string()));

    // Verify node 0 is still crashed
    assert!(cluster.is_node_crashed(0));
    assert_eq!(cluster.active_node_count(), 4);

    // Verify chaos stats are accessible
    let stats = cluster.all_chaos_stats();
    assert_eq!(stats.len(), 5);

    // Recover node 0
    cluster.recover_node(0);
    assert_eq!(cluster.active_node_count(), 5);
    assert!(!cluster.is_node_crashed(0));

    // Heal partition between {1,2} and {3,4}
    cluster.heal_symmetric_partition(1, 3);
    cluster.heal_symmetric_partition(1, 4);
    cluster.heal_symmetric_partition(2, 3);
    cluster.heal_symmetric_partition(2, 4);

    // Verify partition is healed
    let t1 = cluster.transport(1).expect("transport 1 should exist");
    let t3 = cluster.transport(3).expect("transport 3 should exist");
    assert!(!t1.is_partitioned(&"node-3".to_string()));
    assert!(!t1.is_partitioned(&"node-4".to_string()));
    assert!(!t3.is_partitioned(&"node-1".to_string()));
    assert!(!t3.is_partitioned(&"node-2".to_string()));

    // All nodes should be active and fully connected
    assert_eq!(cluster.active_node_count(), 5);
    for i in 0..5 {
        assert!(
            !cluster.is_node_crashed(i),
            "Node {} should be active after full recovery and partition heal",
            i
        );
    }

    println!("Crash-with-partition test passed: crash + partition + recovery + heal");
}
