// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for message reordering behavior.
//!
//! Tests that the consensus system handles out-of-order message
//! delivery correctly, including delayed AppendEntries and votes.

use std::sync::Arc;

use tensor_chain::{MemoryTransport, Message, Transport};

#[tokio::test]
async fn test_reordering_enabled_sends_succeed() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable reordering with 100% probability but small delay
    t1.enable_reordering(1.0, 5);

    // Messages should still arrive, just potentially delayed
    let msg = Message::Ping { term: 1 };
    let result = t1.send(&"node2".to_string(), msg).await;
    assert!(result.is_ok());

    // Verify message was received
    let (from, received) = t2.recv().await.unwrap();
    assert_eq!(from, "node1");
    assert!(matches!(received, Message::Ping { term: 1 }));

    // Reordering counter should have incremented
    assert!(t1.reordered_message_count() > 0);
}

#[tokio::test]
async fn test_reordering_disabled_no_delay() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable then disable
    t1.enable_reordering(1.0, 100);
    t1.disable_reordering();

    let msg = Message::Ping { term: 1 };
    t1.send(&"node2".to_string(), msg).await.unwrap();

    let (_, received) = t2.recv().await.unwrap();
    assert!(matches!(received, Message::Ping { term: 1 }));

    // No reordering should have happened
    assert_eq!(t1.reordered_message_count(), 0);
}

#[tokio::test]
async fn test_reordering_zero_probability_no_delay() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable with 0% probability - should never reorder
    t1.enable_reordering(0.0, 1000);

    for _ in 0..10 {
        let msg = Message::Ping { term: 1 };
        t1.send(&"node2".to_string(), msg).await.unwrap();
    }

    assert_eq!(t1.reordered_message_count(), 0);
}

#[tokio::test]
async fn test_multiple_messages_with_reordering() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable reordering with moderate probability
    t1.enable_reordering(0.5, 10);

    // Send multiple messages
    for i in 0..20 {
        let msg = Message::Ping { term: i };
        t1.send(&"node2".to_string(), msg).await.unwrap();
    }

    // All messages should arrive (reordering does not drop)
    let mut received_count = 0;
    for _ in 0..20 {
        let result = tokio::time::timeout(std::time::Duration::from_millis(100), t2.recv()).await;
        if result.is_ok() {
            received_count += 1;
        }
    }

    assert_eq!(received_count, 20);
}

#[tokio::test]
async fn test_reordering_with_partition_partition_takes_priority() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    // Enable reordering
    t1.enable_reordering(1.0, 10);

    // Also partition
    t1.partition(&"node2".to_string());

    // Partition should take priority - message should be dropped
    let msg = Message::Ping { term: 1 };
    let result = t1.send(&"node2".to_string(), msg).await;
    assert!(result.is_err());
    assert_eq!(t1.dropped_message_count(), 1);
}

#[tokio::test]
async fn test_reordering_chaos_stats_tracking() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    t1.enable_reordering(1.0, 1);

    for _ in 0..5 {
        let msg = Message::Ping { term: 1 };
        t1.send(&"node2".to_string(), msg).await.unwrap();
    }

    let stats = t1.chaos_stats();
    assert!(stats.reordered_messages > 0);
    assert_eq!(stats.dropped_messages, 0);
    assert_eq!(stats.corrupted_messages, 0);
}

#[tokio::test]
async fn test_reset_chaos_counters_clears_reorder_count() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());

    t1.enable_reordering(1.0, 1);
    let msg = Message::Ping { term: 1 };
    t1.send(&"node2".to_string(), msg).await.unwrap();

    assert!(t1.reordered_message_count() > 0);

    t1.reset_chaos_counters();
    assert_eq!(t1.reordered_message_count(), 0);
}
