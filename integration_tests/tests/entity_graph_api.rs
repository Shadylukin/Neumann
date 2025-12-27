//! GraphEngine Entity API integration tests.
//!
//! Tests string-keyed entity operations for graph connectivity.
//! Note: Entities are auto-created when adding edges via add_entity_edge().

use graph_engine::GraphEngine;
use std::sync::Arc;
use tensor_store::TensorStore;

fn create_graph() -> Arc<GraphEngine> {
    let store = TensorStore::new();
    Arc::new(GraphEngine::with_store(store))
}

#[test]
fn test_add_entity_edge_basic() {
    let graph = create_graph();

    // Add edge between entities (auto-creates entities)
    let edge_key = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    assert!(!edge_key.is_empty());
    assert!(edge_key.contains("follows"));
}

#[test]
fn test_add_entity_edge_undirected() {
    let graph = create_graph();

    // Add undirected edge (auto-creates entities)
    let edge_key = graph
        .add_entity_edge_undirected("user:alice", "user:bob", "friends")
        .unwrap();

    assert!(!edge_key.is_empty());

    // Both directions should show neighbors
    let alice_neighbors = graph.get_entity_neighbors("user:alice").unwrap();
    let bob_neighbors = graph.get_entity_neighbors("user:bob").unwrap();

    assert!(alice_neighbors.contains(&"user:bob".to_string()));
    assert!(bob_neighbors.contains(&"user:alice".to_string()));
}

#[test]
fn test_get_entity_neighbors_out() {
    let graph = create_graph();

    // Alice follows Bob and Carol
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    graph
        .add_entity_edge("user:alice", "user:carol", "follows")
        .unwrap();

    // Check outgoing neighbors
    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));

    // Bob has no outgoing edges
    let bob_out = graph.get_entity_neighbors_out("user:bob").unwrap();
    assert!(bob_out.is_empty());
}

#[test]
fn test_get_entity_neighbors_in() {
    let graph = create_graph();

    // Bob and Carol follow Alice
    graph
        .add_entity_edge("user:bob", "user:alice", "follows")
        .unwrap();
    graph
        .add_entity_edge("user:carol", "user:alice", "follows")
        .unwrap();

    // Check incoming neighbors
    let neighbors = graph.get_entity_neighbors_in("user:alice").unwrap();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));

    // Bob has no incoming edges
    let bob_in = graph.get_entity_neighbors_in("user:bob").unwrap();
    assert!(bob_in.is_empty());
}

#[test]
fn test_get_entity_neighbors_both_directions() {
    let graph = create_graph();

    // Alice -> Bob (Alice follows Bob)
    // Carol -> Alice (Carol follows Alice)
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    graph
        .add_entity_edge("user:carol", "user:alice", "follows")
        .unwrap();

    // Alice has neighbors in both directions
    let all_neighbors = graph.get_entity_neighbors("user:alice").unwrap();
    assert_eq!(all_neighbors.len(), 2);
    assert!(all_neighbors.contains(&"user:bob".to_string()));
    assert!(all_neighbors.contains(&"user:carol".to_string()));
}

#[test]
fn test_delete_entity_edge() {
    let graph = create_graph();

    // Add edge
    let edge_key = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    // Verify edge exists
    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert_eq!(neighbors.len(), 1);

    // Delete edge
    graph.delete_entity_edge(&edge_key).unwrap();

    // Verify edge is gone
    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert!(neighbors.is_empty());
}

#[test]
fn test_delete_entity_edge_nonexistent() {
    let graph = create_graph();

    // Deleting non-existent edge should fail
    let result = graph.delete_entity_edge("nonexistent:edge:key");
    assert!(result.is_err());
}

#[test]
fn test_multiple_edge_types_same_entities() {
    let graph = create_graph();

    // Add multiple edge types
    let edge1 = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    let edge2 = graph
        .add_entity_edge("user:alice", "user:bob", "mentions")
        .unwrap();
    let edge3 = graph
        .add_entity_edge("user:alice", "user:bob", "likes")
        .unwrap();

    // All edge keys should be different
    assert_ne!(edge1, edge2);
    assert_ne!(edge2, edge3);
    assert_ne!(edge1, edge3);

    // Bob appears in neighbors
    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert!(neighbors.contains(&"user:bob".to_string()));
}

#[test]
fn test_entity_graph_traversal() {
    let graph = create_graph();

    // Create a chain: A -> B -> C -> D
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    graph
        .add_entity_edge("user:bob", "user:carol", "follows")
        .unwrap();
    graph
        .add_entity_edge("user:carol", "user:dave", "follows")
        .unwrap();

    // Traverse from alice
    let mut current = "user:alice".to_string();
    let mut path = vec![current.clone()];

    for _ in 0..3 {
        let neighbors = graph.get_entity_neighbors_out(&current).unwrap();
        if neighbors.is_empty() {
            break;
        }
        current = neighbors[0].clone();
        path.push(current.clone());
    }

    assert_eq!(path.len(), 4);
    assert_eq!(path[0], "user:alice");
    assert_eq!(path[3], "user:dave");
}

#[test]
fn test_entity_self_loop() {
    let graph = create_graph();

    // Add self-loop
    let edge_key = graph
        .add_entity_edge("user:narcissist", "user:narcissist", "admires")
        .unwrap();
    assert!(!edge_key.is_empty());

    // Should appear in both in and out neighbors
    let neighbors_out = graph.get_entity_neighbors_out("user:narcissist").unwrap();
    let neighbors_in = graph.get_entity_neighbors_in("user:narcissist").unwrap();

    assert!(neighbors_out.contains(&"user:narcissist".to_string()));
    assert!(neighbors_in.contains(&"user:narcissist".to_string()));
}

#[test]
fn test_concurrent_entity_edge_operations() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let graph = create_graph();
    let success_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // 4 threads add edges to different targets (avoid race on single hub)
    for t in 0..4 {
        let graph = Arc::clone(&graph);
        let counter = Arc::clone(&success_count);

        handles.push(thread::spawn(move || {
            for i in 0..5 {
                let spoke_key = format!("spoke:t{}:{}", t, i);
                let hub_key = format!("hub:{}", t); // Each thread has its own hub
                if graph
                    .add_entity_edge(&spoke_key, &hub_key, "connects")
                    .is_ok()
                {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All 20 edges should be created
    assert_eq!(success_count.load(Ordering::SeqCst), 20);

    // Each hub should have 5 incoming connections
    for t in 0..4 {
        let hub_key = format!("hub:{}", t);
        let neighbors = graph.get_entity_neighbors_in(&hub_key).unwrap();
        assert_eq!(neighbors.len(), 5);
    }
}

#[test]
fn test_entity_edge_delete_and_recreate() {
    let graph = create_graph();

    // Add, delete, add again
    let edge1 = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    graph.delete_entity_edge(&edge1).unwrap();

    let edge2 = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    // Should work and be a different edge key
    assert_ne!(edge1, edge2);

    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert!(neighbors.contains(&"user:bob".to_string()));
}

#[test]
fn test_entity_has_edges() {
    let graph = create_graph();

    // No edges initially (entity doesn't exist)
    assert!(!graph.entity_has_edges("user:alice"));

    // Add an edge
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    // Now alice has edges
    assert!(graph.entity_has_edges("user:alice"));
}

#[test]
fn test_entity_bidirectional_edges() {
    let graph = create_graph();

    // Alice follows Bob
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    // Bob follows Alice back
    graph
        .add_entity_edge("user:bob", "user:alice", "follows")
        .unwrap();

    // Both should see each other as neighbors
    let alice_neighbors = graph.get_entity_neighbors("user:alice").unwrap();
    let bob_neighbors = graph.get_entity_neighbors("user:bob").unwrap();

    assert!(alice_neighbors.contains(&"user:bob".to_string()));
    assert!(bob_neighbors.contains(&"user:alice".to_string()));
}

#[test]
fn test_entity_neighbors_empty_entity() {
    let graph = create_graph();

    // Create an entity by having it as an edge target
    graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();

    // Alice has outgoing, Bob has incoming
    let alice_out = graph.get_entity_neighbors_out("user:alice").unwrap();
    let bob_out = graph.get_entity_neighbors_out("user:bob").unwrap();

    assert_eq!(alice_out.len(), 1);
    assert!(bob_out.is_empty());
}

#[test]
fn test_entity_star_topology() {
    let graph = create_graph();

    // Create star topology with hub in center
    for i in 0..10 {
        graph
            .add_entity_edge(&format!("spoke:{}", i), "hub:center", "connects")
            .unwrap();
    }

    // Hub should have 10 incoming connections
    let hub_in = graph.get_entity_neighbors_in("hub:center").unwrap();
    assert_eq!(hub_in.len(), 10);

    // Hub should have 0 outgoing connections
    let hub_out = graph.get_entity_neighbors_out("hub:center").unwrap();
    assert!(hub_out.is_empty());

    // Each spoke should have 1 outgoing connection
    for i in 0..10 {
        let spoke_out = graph
            .get_entity_neighbors_out(&format!("spoke:{}", i))
            .unwrap();
        assert_eq!(spoke_out.len(), 1);
    }
}

#[test]
fn test_entity_ring_topology() {
    let graph = create_graph();

    // Create ring: 0 -> 1 -> 2 -> 3 -> 4 -> 0
    for i in 0..5 {
        let from = format!("node:{}", i);
        let to = format!("node:{}", (i + 1) % 5);
        graph.add_entity_edge(&from, &to, "next").unwrap();
    }

    // Each node should have 1 in and 1 out
    for i in 0..5 {
        let key = format!("node:{}", i);
        let neighbors_out = graph.get_entity_neighbors_out(&key).unwrap();
        let neighbors_in = graph.get_entity_neighbors_in(&key).unwrap();

        assert_eq!(neighbors_out.len(), 1);
        assert_eq!(neighbors_in.len(), 1);
    }
}

#[test]
fn test_entity_delete_preserves_other_edges() {
    let graph = create_graph();

    // Create multiple edges from alice
    let edge1 = graph
        .add_entity_edge("user:alice", "user:bob", "follows")
        .unwrap();
    let _edge2 = graph
        .add_entity_edge("user:alice", "user:carol", "follows")
        .unwrap();
    let _edge3 = graph
        .add_entity_edge("user:alice", "user:dave", "follows")
        .unwrap();

    // Delete one edge
    graph.delete_entity_edge(&edge1).unwrap();

    // Should still have 2 neighbors
    let neighbors = graph.get_entity_neighbors_out("user:alice").unwrap();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:carol".to_string()));
    assert!(neighbors.contains(&"user:dave".to_string()));
    assert!(!neighbors.contains(&"user:bob".to_string()));
}
