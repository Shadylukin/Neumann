// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stress tests for rapid network partition creation and healing.
//!
//! Tests distributed transport behavior under high-frequency partition
//! churn scenarios:
//! - Rapid partition/heal cycles with connectivity verification
//! - Concurrent partition mutations and message sends
//! - Full node isolation and sequential rejoin

use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor_chain::{MemoryTransport, Message, Transport};

/// Create a fully connected N-node transport mesh.
fn create_transport_mesh(n: usize) -> (Vec<Arc<MemoryTransport>>, Vec<String>) {
    let node_ids: Vec<String> = (0..n).map(|i| format!("node-{i}")).collect();

    let transports: Vec<Arc<MemoryTransport>> = node_ids
        .iter()
        .map(|id| Arc::new(MemoryTransport::new(id.clone())))
        .collect();

    for (i, t1) in transports.iter().enumerate() {
        for (j, t2) in transports.iter().enumerate() {
            if i != j {
                t1.connect_to(node_ids[j].clone(), t2.sender());
            }
        }
    }

    (transports, node_ids)
}

#[tokio::test]
#[ignore]
async fn test_rapid_partition_heal_cycles() {
    let (transports, node_ids) = create_transport_mesh(5);
    let mut rng = ChaCha8Rng::seed_from_u64(99);

    let mut total_dropped = 0_u64;

    for cycle in 0..1000 {
        // Randomly pick two distinct nodes to partition
        let a = rng.random_range(0..5_usize);
        let mut b = rng.random_range(0..5_usize);
        while b == a {
            b = rng.random_range(0..5_usize);
        }

        // Create bidirectional partition
        transports[a].partition(&node_ids[b]);
        transports[b].partition(&node_ids[a]);

        // Verify partitioned path fails
        let result = transports[a]
            .send(&node_ids[b], Message::Ping { term: 1 })
            .await;
        assert!(
            result.is_err(),
            "Send should fail on partitioned path {a}->{b} in cycle {cycle}",
        );

        // Verify non-partitioned path works (pick a third node)
        let mut c = rng.random_range(0..5_usize);
        while c == a || c == b {
            c = rng.random_range(0..5_usize);
        }

        // Only check if a->c is not partitioned
        if !transports[a].is_partitioned(&node_ids[c]) {
            let result = transports[a]
                .send(&node_ids[c], Message::Ping { term: 1 })
                .await;
            assert!(
                result.is_ok(),
                "Send should succeed on non-partitioned path {a}->{c} in cycle {cycle}",
            );
        }

        // Heal the partition
        transports[a].heal(&node_ids[b]);
        transports[b].heal(&node_ids[a]);
    }

    // Collect final dropped message counts
    for t in &transports {
        total_dropped += t.dropped_message_count();
    }

    println!("Completed 1000 rapid partition/heal cycles");
    println!("Total messages dropped: {total_dropped}");

    // Should have dropped at least some messages
    assert!(
        total_dropped > 0,
        "Expected some dropped messages from partitioned sends"
    );
}

#[tokio::test]
#[ignore]
async fn test_concurrent_partition_and_send() {
    let (transports, node_ids) = create_transport_mesh(3);

    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Drain tasks: consume messages from each node to prevent channel backpressure
    let mut drain_handles = Vec::new();
    for transport in &transports {
        let t = transport.clone();
        let done_drain = done.clone();
        drain_handles.push(tokio::spawn(async move {
            let mut drained = 0_u64;
            while !done_drain.load(std::sync::atomic::Ordering::Relaxed) {
                // Use try_recv via recv with a timeout to drain messages
                match tokio::time::timeout(Duration::from_millis(1), t.recv()).await {
                    Ok(Ok(_)) => drained += 1,
                    _ => tokio::time::sleep(Duration::from_micros(100)).await,
                }
            }
            drained
        }));
    }

    // Task 1: Repeatedly partition and heal pairs
    let partition_transports = transports.to_vec();
    let partition_ids = node_ids.clone();
    let done_partition = done.clone();
    let partition_handle = tokio::spawn(async move {
        let mut rng = ChaCha8Rng::seed_from_u64(200);
        let mut cycles = 0_u64;

        while !done_partition.load(std::sync::atomic::Ordering::Relaxed) {
            let a = rng.random_range(0..3_usize);
            let mut b = rng.random_range(0..3_usize);
            while b == a {
                b = rng.random_range(0..3_usize);
            }

            partition_transports[a].partition(&partition_ids[b]);
            partition_transports[b].partition(&partition_ids[a]);

            tokio::time::sleep(Duration::from_micros(100)).await;

            partition_transports[a].heal(&partition_ids[b]);
            partition_transports[b].heal(&partition_ids[a]);

            cycles += 1;
        }

        cycles
    });

    // Task 2: Continuously send Ping messages between all pairs
    let send_transports = transports.to_vec();
    let send_ids = node_ids.clone();
    let done_send = done.clone();
    let send_handle = tokio::spawn(async move {
        let mut sent = 0_u64;
        let mut failed = 0_u64;

        while !done_send.load(std::sync::atomic::Ordering::Relaxed) {
            for (a, transport) in send_transports.iter().enumerate() {
                for (b, target_id) in send_ids.iter().enumerate() {
                    if a != b {
                        let result = transport.send(target_id, Message::Ping { term: 1 }).await;
                        sent += 1;
                        if result.is_err() {
                            failed += 1;
                        }
                    }
                }
            }

            tokio::time::sleep(Duration::from_micros(50)).await;
        }

        (sent, failed)
    });

    // Run for 2 seconds
    tokio::time::sleep(Duration::from_secs(2)).await;
    done.store(true, std::sync::atomic::Ordering::Relaxed);

    let partition_cycles = partition_handle.await.expect("partition task panicked");
    let (sent, failed) = send_handle.await.expect("send task panicked");

    let chaos_stats: Vec<_> = transports.iter().map(|t| t.chaos_stats()).collect();

    println!("Partition/heal cycles: {partition_cycles}");
    println!("Messages sent: {sent}, failed: {failed}");
    println!("Chaos stats: {chaos_stats:?}");

    for handle in drain_handles {
        let _ = handle.await;
    }

    // Both tasks should have completed without panicking
    assert!(
        partition_cycles > 0,
        "Should have completed some partition cycles"
    );
    assert!(sent > 0, "Should have sent some messages");
}

#[tokio::test]
#[ignore]
async fn test_full_isolation_and_rejoin() {
    let (transports, node_ids) = create_transport_mesh(5);

    // Isolate nodes one by one and verify they cannot send
    for (isolate_idx, transport) in transports.iter().enumerate() {
        transport.partition_all();

        // Verify isolated node cannot send to any peer
        for (target_idx, target_id) in node_ids.iter().enumerate() {
            if target_idx == isolate_idx {
                continue;
            }
            let result = transport.send(target_id, Message::Ping { term: 1 }).await;
            assert!(
                result.is_err(),
                "Isolated node {isolate_idx} should not be able to send to node {target_idx}",
            );
        }
    }

    // All nodes are now isolated. Verify none can communicate.
    for (sender_idx, sender_transport) in transports.iter().enumerate() {
        for (receiver_idx, receiver_id) in node_ids.iter().enumerate() {
            if sender_idx == receiver_idx {
                continue;
            }
            let result = sender_transport
                .send(receiver_id, Message::Ping { term: 1 })
                .await;
            assert!(
                result.is_err(),
                "All-isolated: node {sender_idx} should not reach node {receiver_idx}",
            );
        }
    }

    // Rejoin nodes one by one and verify progressive connectivity
    for (rejoin_idx, transport) in transports.iter().enumerate() {
        transport.heal_all();

        // Verify this node can send to other already-rejoined nodes
        for (other_idx, other_id) in node_ids.iter().enumerate().take(rejoin_idx) {
            let result = transport.send(other_id, Message::Ping { term: 1 }).await;
            assert!(
                result.is_ok(),
                "Rejoined node {rejoin_idx} should be able to send to rejoined node {other_idx}",
            );
        }
    }

    // All nodes rejoined. Verify full connectivity.
    for (sender_idx, sender_transport) in transports.iter().enumerate() {
        for (receiver_idx, receiver_id) in node_ids.iter().enumerate() {
            if sender_idx == receiver_idx {
                continue;
            }
            let result = sender_transport
                .send(receiver_id, Message::Ping { term: 1 })
                .await;
            assert!(
                result.is_ok(),
                "Fully rejoined: node {sender_idx} should reach node {receiver_idx}",
            );
        }
    }

    let total_dropped: u64 = transports.iter().map(|t| t.dropped_message_count()).sum();
    println!("Full isolation/rejoin test passed");
    println!("Total messages dropped during isolation: {total_dropped}");
}
