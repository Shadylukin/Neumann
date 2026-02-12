// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for asymmetric and partial network partitions.
//!
//! Tests link quality degradation, one-way partitions, and bridge failures.

use std::sync::Arc;

use tensor_chain::{MemoryTransport, Message, Transport};

#[tokio::test]
async fn test_asymmetric_partition_one_way() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());
    t2.connect_to("node1".to_string(), t1.sender());

    // Only partition one direction: node1 -> node2
    t1.partition(&"node2".to_string());

    // node1 -> node2 should fail
    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());

    // node2 -> node1 should still work
    let result = t2
        .send(&"node1".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok());

    let (from, msg) = t1.recv().await.unwrap();
    assert_eq!(from, "node2");
    assert!(matches!(msg, Message::Ping { term: 2 }));
}

#[tokio::test]
async fn test_link_quality_degradation_full_drop() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // 100% drop rate via link quality
    t1.set_link_quality(&"node2".to_string(), 1.0);

    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());
    assert!(t1.dropped_message_count() > 0);
}

#[tokio::test]
async fn test_link_quality_zero_drop_rate() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // 0% drop rate - all messages should get through
    t1.set_link_quality(&"node2".to_string(), 0.0);

    for _ in 0..10 {
        let result = t1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.is_ok());
    }

    assert_eq!(t1.dropped_message_count(), 0);
}

#[tokio::test]
async fn test_link_quality_clear_restores_reliability() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Set 100% drop
    t1.set_link_quality(&"node2".to_string(), 1.0);

    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());

    // Clear link quality - should restore reliable delivery
    t1.clear_link_quality(&"node2".to_string());

    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_asymmetric_link_quality() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());
    t2.connect_to("node1".to_string(), t1.sender());

    // Degrade link only from node1 to node2
    t1.set_link_quality(&"node2".to_string(), 1.0);

    // node1 -> node2 should drop
    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());

    // node2 -> node1 should be fine (no link quality set on t2)
    let result = t2
        .send(&"node1".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_bridge_node_partition() {
    // 3-node topology: node1 <-> node2 <-> node3
    // node2 is the bridge between node1 and node3
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let t3 = Arc::new(MemoryTransport::new("node3".to_string()));

    // Fully connected mesh
    t1.connect_to("node2".to_string(), t2.sender());
    t1.connect_to("node3".to_string(), t3.sender());
    t2.connect_to("node1".to_string(), t1.sender());
    t2.connect_to("node3".to_string(), t3.sender());
    t3.connect_to("node1".to_string(), t1.sender());
    t3.connect_to("node2".to_string(), t2.sender());

    // Partition node1 from node3 (but both can still reach node2)
    t1.partition(&"node3".to_string());
    t3.partition(&"node1".to_string());

    // node1 -> node3 should fail
    let result = t1
        .send(&"node3".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());

    // node1 -> node2 should work
    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok());

    // node3 -> node2 should work
    let result = t3
        .send(&"node2".to_string(), Message::Ping { term: 3 })
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_partition_with_link_quality_partition_takes_priority() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Set link quality to 0% drop (reliable)
    t1.set_link_quality(&"node2".to_string(), 0.0);

    // But also partition - partition takes priority
    t1.partition(&"node2".to_string());

    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_link_quality_getter() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));

    assert!(t1.link_drop_rate(&"node2".to_string()).abs() < f32::EPSILON);

    t1.set_link_quality(&"node2".to_string(), 0.5);
    assert!((t1.link_drop_rate(&"node2".to_string()) - 0.5).abs() < f32::EPSILON);

    t1.clear_link_quality(&"node2".to_string());
    assert!(t1.link_drop_rate(&"node2".to_string()).abs() < f32::EPSILON);
}

#[tokio::test]
async fn test_chaos_cluster_link_quality() {
    use integration_tests::chaos::{ChaosCluster, ChaosConfig};

    let config = ChaosConfig::default();
    let cluster = ChaosCluster::new(3, config);

    // Set asymmetric link quality
    cluster.set_link_quality(0, 1, 1.0); // node0 -> node1 drops all

    let t0 = cluster.transport(0).unwrap();
    assert!((t0.link_drop_rate(&"node-1".to_string()) - 1.0).abs() < f32::EPSILON);

    // node1 -> node0 should be unaffected
    let t1 = cluster.transport(1).unwrap();
    assert!(t1.link_drop_rate(&"node-0".to_string()).abs() < f32::EPSILON);
}
