//! Stress tests for network partition resilience.
//!
//! Tests distributed system behavior under sustained partition scenarios:
//! - Repeated partition/heal cycles
//! - Concurrent operations during partitions
//! - Recovery time measurement
//! - Message drop statistics

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use tensor_chain::{MemoryTransport, Message, RaftConfig, RaftNode, RaftState, Transport};
use tokio::time::sleep;

/// Create a connected 5-node cluster for stress testing.
fn create_5_node_cluster() -> (Vec<Arc<RaftNode>>, Vec<Arc<MemoryTransport>>) {
    let node_ids: Vec<String> = (1..=5).map(|i| format!("node{}", i)).collect();

    // Create transports
    let transports: Vec<Arc<MemoryTransport>> = node_ids
        .iter()
        .map(|id| Arc::new(MemoryTransport::new(id.clone())))
        .collect();

    // Fully connect all transports
    for (i, t1) in transports.iter().enumerate() {
        for (j, t2) in transports.iter().enumerate() {
            if i != j {
                t1.connect_to(node_ids[j].clone(), t2.sender());
            }
        }
    }

    let config = RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    };

    // Create nodes
    let nodes: Vec<Arc<RaftNode>> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let peers: Vec<String> = node_ids.iter().filter(|p| *p != id).cloned().collect();
            Arc::new(RaftNode::new(
                id.clone(),
                peers,
                transports[i].clone(),
                config.clone(),
            ))
        })
        .collect();

    (nodes, transports)
}

/// Tick all nodes and process any pending messages.
async fn tick_all(nodes: &[Arc<RaftNode>]) {
    // First, tick all nodes (may send messages)
    for node in nodes {
        let _ = node.tick_async().await;
    }

    // Then process any pending messages (non-blocking)
    for node in nodes {
        // Try to receive and process up to 10 messages per tick
        for _ in 0..10 {
            match tokio::time::timeout(Duration::from_micros(100), node.transport().recv()).await {
                Ok(Ok((from, msg))) => {
                    let _ = node.handle_message_async(&from, msg).await;
                },
                _ => break, // No more messages or timeout
            }
        }
    }
}

/// Count leaders in the cluster.
fn count_leaders(nodes: &[Arc<RaftNode>]) -> usize {
    nodes
        .iter()
        .filter(|n| n.state() == RaftState::Leader)
        .count()
}

/// Get current leader if exactly one exists.
fn get_leader<'a>(nodes: &'a [Arc<RaftNode>]) -> Option<&'a Arc<RaftNode>> {
    let leaders: Vec<_> = nodes
        .iter()
        .filter(|n| n.state() == RaftState::Leader)
        .collect();
    if leaders.len() == 1 {
        Some(leaders[0])
    } else {
        None
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --test network_partition_stress -- --ignored
async fn stress_partition_heal_cycles() {
    let (nodes, transports) = create_5_node_cluster();

    // Initial stabilization
    nodes[0].start_election_async().await.unwrap();
    for _ in 0..100 {
        tick_all(&nodes).await;
        sleep(Duration::from_millis(10)).await;
    }

    let mut partition_count = 0;
    let mut heal_count = 0;
    let mut leader_changes = 0;
    let mut last_leader: Option<String> = None;

    // Run 20 partition/heal cycles
    for cycle in 0..20 {
        // Partition a random node
        let victim = cycle % 5;
        let victim_id = format!("node{}", victim + 1);

        // Create bidirectional partition
        transports[victim].partition_all();
        for (i, t) in transports.iter().enumerate() {
            if i != victim {
                t.partition(&victim_id);
            }
        }
        partition_count += 1;

        // Tick during partition
        for _ in 0..30 {
            tick_all(&nodes).await;
            sleep(Duration::from_millis(10)).await;
        }

        // Check for leader
        if let Some(leader) = get_leader(&nodes) {
            let leader_id = leader.node_id().to_string();
            if last_leader.as_ref() != Some(&leader_id) {
                leader_changes += 1;
                last_leader = Some(leader_id);
            }
        }

        // Heal partition
        transports[victim].heal_all();
        for t in transports.iter() {
            t.heal(&victim_id);
        }
        heal_count += 1;

        // Tick to stabilize
        for _ in 0..20 {
            tick_all(&nodes).await;
            sleep(Duration::from_millis(10)).await;
        }
    }

    // Final assertions
    assert_eq!(partition_count, 20);
    assert_eq!(heal_count, 20);

    // Should never have more than one leader
    let final_leaders = count_leaders(&nodes);
    assert!(
        final_leaders <= 1,
        "Split-brain detected: {} leaders",
        final_leaders
    );

    // Check message drop stats
    let total_dropped: u64 = transports.iter().map(|t| t.dropped_message_count()).sum();
    println!("Partition/heal cycles: {}/{}", partition_count, heal_count);
    println!("Leader changes: {}", leader_changes);
    println!("Total messages dropped: {}", total_dropped);
}

#[tokio::test]
#[ignore]
async fn stress_continuous_partitions() {
    let (nodes, transports) = create_5_node_cluster();

    // Initial election
    nodes[0].start_election_async().await.unwrap();
    for _ in 0..50 {
        tick_all(&nodes).await;
        sleep(Duration::from_millis(10)).await;
    }

    let start = Instant::now();
    let ops = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    // Run for 5 seconds with continuous partitions
    while start.elapsed() < Duration::from_secs(5) {
        // Random partition
        let victim = (start.elapsed().as_millis() as usize / 100) % 5;

        transports[victim].partition_all();

        // Tick a few times
        for _ in 0..5 {
            tick_all(&nodes).await;
            ops.fetch_add(1, Ordering::Relaxed);
        }

        // Heal
        transports[victim].heal_all();

        // Tick more
        for _ in 0..5 {
            tick_all(&nodes).await;
            ops.fetch_add(1, Ordering::Relaxed);
        }

        sleep(Duration::from_millis(20)).await;
    }

    let total_ops = ops.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!("Total tick operations: {}", total_ops);
    println!("Ops/sec: {:.0}", total_ops as f64 / elapsed.as_secs_f64());
    println!("Errors: {}", total_errors);

    // At end, should converge to single leader or no leader (not multiple)
    let leaders = count_leaders(&nodes);
    assert!(
        leaders <= 1,
        "Split-brain: {} leaders after stress",
        leaders
    );
}

#[tokio::test]
#[ignore]
async fn stress_majority_partition_quorum() {
    let (nodes, transports) = create_5_node_cluster();

    // Create partition FIRST, before any election
    // Partition minority (2 nodes) from majority (3 nodes)
    // Nodes 1,2 are minority, nodes 3,4,5 are majority
    let minority = vec!["node1", "node2"];
    let majority = vec!["node3", "node4", "node5"];

    // Create bidirectional partition
    for m in &minority {
        for maj in &majority {
            transports[m.chars().last().unwrap().to_digit(10).unwrap() as usize - 1]
                .partition(&maj.to_string());
        }
    }
    for maj in &majority {
        for m in &minority {
            transports[maj.chars().last().unwrap().to_digit(10).unwrap() as usize - 1]
                .partition(&m.to_string());
        }
    }

    // Also partition within minority so they can't vote for each other
    transports[0].partition(&"node2".to_string());
    transports[1].partition(&"node1".to_string());

    // Now trigger elections in both partitions
    nodes[0].start_election_async().await.unwrap(); // Minority node
    nodes[2].start_election_async().await.unwrap(); // Majority node

    // Run for a while partitioned
    for _ in 0..100 {
        tick_all(&nodes).await;
        sleep(Duration::from_millis(10)).await;
    }

    // Majority partition should be able to elect a leader
    let majority_leaders: Vec<_> = nodes[2..5]
        .iter()
        .filter(|n| n.state() == RaftState::Leader)
        .collect();

    // Minority should NOT have a leader (can't reach quorum)
    let minority_leaders: Vec<_> = nodes[0..2]
        .iter()
        .filter(|n| n.state() == RaftState::Leader)
        .collect();

    println!("Majority leaders: {}", majority_leaders.len());
    println!("Minority leaders: {}", minority_leaders.len());

    // Minority should not be able to elect leader
    assert_eq!(
        minority_leaders.len(),
        0,
        "Minority partition should not have leader"
    );

    // Majority should have at most one leader
    assert!(
        majority_leaders.len() <= 1,
        "Multiple leaders in majority partition"
    );
}

#[tokio::test]
#[ignore]
async fn stress_message_throughput_under_partition() {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());
    t2.connect_to("node1".to_string(), t1.sender());

    let messages_sent = Arc::new(AtomicU64::new(0));
    let messages_dropped = Arc::new(AtomicU64::new(0));
    let messages_received = Arc::new(AtomicU64::new(0));
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Spawn receiver task to drain messages from t2
    let t2_clone = t2.clone();
    let received_clone = messages_received.clone();
    let done_clone = done.clone();
    let receiver_handle = tokio::spawn(async move {
        while !done_clone.load(Ordering::Relaxed) {
            match tokio::time::timeout(Duration::from_millis(10), t2_clone.recv()).await {
                Ok(Ok(_)) => {
                    received_clone.fetch_add(1, Ordering::Relaxed);
                },
                _ => {},
            }
        }
    });

    let start = Instant::now();
    let mut partitioned = false;

    // Send messages for 3 seconds, toggling partition every 500ms
    while start.elapsed() < Duration::from_secs(3) {
        // Toggle partition every 500ms
        if start.elapsed().as_millis() / 500 % 2 == 0 {
            if partitioned {
                t1.heal(&"node2".to_string());
                partitioned = false;
            }
        } else if !partitioned {
            t1.partition(&"node2".to_string());
            partitioned = true;
        }

        // Try to send
        let result = t1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await;
        messages_sent.fetch_add(1, Ordering::Relaxed);
        if result.is_err() {
            messages_dropped.fetch_add(1, Ordering::Relaxed);
        }

        // Small delay
        sleep(Duration::from_micros(100)).await;
    }

    // Signal receiver to stop and wait for it
    done.store(true, Ordering::Relaxed);
    let _ = receiver_handle.await;

    let sent = messages_sent.load(Ordering::Relaxed);
    let dropped = messages_dropped.load(Ordering::Relaxed);
    let received = messages_received.load(Ordering::Relaxed);
    let transport_dropped = t1.dropped_message_count();

    println!("Messages sent: {}", sent);
    println!("Messages received: {}", received);
    println!("Messages dropped (tracked): {}", dropped);
    println!("Messages dropped (transport): {}", transport_dropped);
    println!("Drop rate: {:.1}%", (dropped as f64 / sent as f64) * 100.0);

    // Should have roughly 50% drop rate due to 50% partition time
    // Allow some variance
    let drop_rate = dropped as f64 / sent as f64;
    assert!(
        drop_rate > 0.3,
        "Expected some drops, got {:.1}%",
        drop_rate * 100.0
    );
    assert!(drop_rate < 0.7, "Too many drops: {:.1}%", drop_rate * 100.0);
}
