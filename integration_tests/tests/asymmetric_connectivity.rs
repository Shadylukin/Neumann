// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for asymmetric network partition detection.
//!
//! Validates that the system correctly handles scenarios where A can reach B
//! but B cannot reach A, including partition/heal lifecycle and indirect paths.

use std::sync::Arc;

use tensor_chain::{MemoryTransport, Message, Transport};

/// Create a 3-node mesh where all nodes can reach each other.
fn create_mesh_3() -> (
    Arc<MemoryTransport>,
    Arc<MemoryTransport>,
    Arc<MemoryTransport>,
) {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let t3 = Arc::new(MemoryTransport::new("node3".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());
    t1.connect_to("node3".to_string(), t3.sender());
    t2.connect_to("node1".to_string(), t1.sender());
    t2.connect_to("node3".to_string(), t3.sender());
    t3.connect_to("node1".to_string(), t1.sender());
    t3.connect_to("node2".to_string(), t2.sender());

    (t1, t2, t3)
}

#[tokio::test]
async fn test_asymmetric_partition_message_delivery() {
    let (t1, t2, _t3) = create_mesh_3();

    // Create asymmetric partition: node1 cannot send to node2
    t1.partition(&"node2".to_string());

    // node1 -> node2 should fail
    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_err(), "Partitioned direction should fail");

    // node2 -> node1 should succeed (asymmetric)
    let result = t2
        .send(&"node1".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok(), "Reverse direction should still work");

    // Verify message actually arrived
    let (from, msg) = t1.recv().await.unwrap();
    assert_eq!(from, "node2");
    assert!(matches!(msg, Message::Ping { term: 2 }));
}

#[tokio::test]
async fn test_asymmetric_partition_with_indirect_path() {
    let (t1, t2, t3) = create_mesh_3();

    // Asymmetric: node1 cannot reach node2 directly
    t1.partition(&"node2".to_string());

    // But node1 can still reach node3, and node3 can reach node2
    let result = t1
        .send(&"node3".to_string(), Message::Ping { term: 1 })
        .await;
    assert!(result.is_ok(), "node1 -> node3 should work");

    let result = t3
        .send(&"node2".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok(), "node3 -> node2 should work");

    // Drain messages
    let _ = t3.recv().await;
    let _ = t2.recv().await;
}

#[tokio::test]
async fn test_asymmetric_partition_both_directions_different_peers() {
    let (t1, t2, t3) = create_mesh_3();

    // node1 cannot reach node2, node2 cannot reach node3
    t1.partition(&"node2".to_string());
    t2.partition(&"node3".to_string());

    // node1 -> node2 fails
    assert!(t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await
        .is_err());

    // node2 -> node3 fails
    assert!(t2
        .send(&"node3".to_string(), Message::Ping { term: 2 })
        .await
        .is_err());

    // But node1 -> node3 still works (direct link intact)
    assert!(t1
        .send(&"node3".to_string(), Message::Ping { term: 3 })
        .await
        .is_ok());

    // And node3 -> node1 works
    assert!(t3
        .send(&"node1".to_string(), Message::Ping { term: 4 })
        .await
        .is_ok());

    // Drain
    let _ = t3.recv().await;
    let _ = t1.recv().await;
}

#[tokio::test]
async fn test_heal_after_asymmetric_partition() {
    let (t1, t2, _t3) = create_mesh_3();

    // Partition node1 -> node2
    t1.partition(&"node2".to_string());

    // Confirm partition
    assert!(t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await
        .is_err());

    // Heal the partition
    t1.heal(&"node2".to_string());

    // Now node1 -> node2 should work again
    let result = t1
        .send(&"node2".to_string(), Message::Ping { term: 2 })
        .await;
    assert!(result.is_ok(), "Healed partition should allow messages");

    // Drain
    let _ = t2.recv().await;
}

#[tokio::test]
async fn test_dropped_message_count_tracks_partitioned_sends() {
    let (t1, _t2, _t3) = create_mesh_3();

    assert_eq!(t1.dropped_message_count(), 0);

    t1.partition(&"node2".to_string());

    // Each failed send should increment dropped count
    for _ in 0..5 {
        let _ = t1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await;
    }

    assert!(
        t1.dropped_message_count() > 0,
        "Should track dropped messages from partitioned sends"
    );
}

#[tokio::test]
async fn test_partition_all_and_heal_all() {
    let (t1, t2, t3) = create_mesh_3();

    // Partition t1 from everyone
    t1.partition_all();

    assert!(t1
        .send(&"node2".to_string(), Message::Ping { term: 1 })
        .await
        .is_err());
    assert!(t1
        .send(&"node3".to_string(), Message::Ping { term: 1 })
        .await
        .is_err());

    // Others can still communicate with each other
    assert!(t2
        .send(&"node3".to_string(), Message::Ping { term: 2 })
        .await
        .is_ok());

    // Heal all
    t1.heal_all();

    assert!(t1
        .send(&"node2".to_string(), Message::Ping { term: 3 })
        .await
        .is_ok());
    assert!(t1
        .send(&"node3".to_string(), Message::Ping { term: 4 })
        .await
        .is_ok());

    // Drain
    let _ = t3.recv().await; // from t2
    let _ = t2.recv().await; // from t1
    let _ = t3.recv().await; // from t1
}

#[tokio::test]
async fn test_partitioned_peers_list() {
    let (t1, _t2, _t3) = create_mesh_3();

    assert!(t1.partitioned_peers().is_empty());

    t1.partition(&"node2".to_string());
    let peers = t1.partitioned_peers();
    assert_eq!(peers.len(), 1);
    assert!(peers.contains(&"node2".to_string()));

    t1.partition(&"node3".to_string());
    let peers = t1.partitioned_peers();
    assert_eq!(peers.len(), 2);

    t1.heal(&"node2".to_string());
    let peers = t1.partitioned_peers();
    assert_eq!(peers.len(), 1);
    assert!(peers.contains(&"node3".to_string()));
}
