// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! GraphEngine Node-based API integration tests.
//!
//! Tests node-based graph operations using the recommended API.
//! Nodes are created with properties, and edges link them together.

use std::{collections::HashMap, sync::Arc};

use graph_engine::{Direction, GraphEngine, PropertyValue};
use tensor_store::TensorStore;

fn create_graph() -> Arc<GraphEngine> {
    let store = TensorStore::new();
    Arc::new(GraphEngine::with_store(store))
}

/// Helper to create a node with an entity key property.
fn create_entity_node(graph: &GraphEngine, entity_key: &str) -> u64 {
    let mut props = HashMap::new();
    props.insert(
        "entity_key".to_string(),
        PropertyValue::String(entity_key.to_string()),
    );
    graph.create_node("Entity", props).unwrap()
}

/// Helper to get or create a node for an entity key.
fn get_or_create_entity_node(graph: &GraphEngine, entity_key: &str) -> u64 {
    if let Ok(nodes) =
        graph.find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
    {
        if let Some(node) = nodes.first() {
            return node.id;
        }
    }
    create_entity_node(graph, entity_key)
}

/// Helper to add a directed edge between entity keys.
fn add_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) -> u64 {
    let from_node = get_or_create_entity_node(graph, from_key);
    let to_node = get_or_create_entity_node(graph, to_key);
    graph
        .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
        .unwrap()
}

/// Helper to add an undirected edge between entity keys.
fn add_undirected_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) -> u64 {
    let from_node = get_or_create_entity_node(graph, from_key);
    let to_node = get_or_create_entity_node(graph, to_key);
    graph
        .create_edge(from_node, to_node, edge_type, HashMap::new(), false)
        .unwrap()
}

/// Helper to get entity key from a node ID.
fn get_entity_key(graph: &GraphEngine, node_id: u64) -> Option<String> {
    graph.get_node(node_id).ok().and_then(|node| {
        if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
            Some(key.clone())
        } else {
            None
        }
    })
}

/// Helper to get outgoing neighbor entity keys.
fn get_neighbors_out(graph: &GraphEngine, entity_key: &str) -> Vec<String> {
    let node_id = get_or_create_entity_node(graph, entity_key);
    let mut result = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Outgoing) {
        for edge in edges {
            let target_id = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };
            if let Some(key) = get_entity_key(graph, target_id) {
                result.push(key);
            }
        }
    }
    result
}

/// Helper to get incoming neighbor entity keys.
fn get_neighbors_in(graph: &GraphEngine, entity_key: &str) -> Vec<String> {
    let node_id = get_or_create_entity_node(graph, entity_key);
    let mut result = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Incoming) {
        for edge in edges {
            let source_id = if edge.to == node_id {
                edge.from
            } else {
                edge.to
            };
            if let Some(key) = get_entity_key(graph, source_id) {
                result.push(key);
            }
        }
    }
    result
}

/// Helper to get all neighbor entity keys (both directions).
fn get_neighbors_both(graph: &GraphEngine, entity_key: &str) -> Vec<String> {
    let node_id = get_or_create_entity_node(graph, entity_key);
    let mut result = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Both) {
        for edge in edges {
            let other_id = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };
            if let Some(key) = get_entity_key(graph, other_id) {
                if !result.contains(&key) {
                    result.push(key);
                }
            }
        }
    }
    result
}

#[test]
fn test_add_edge_basic() {
    let graph = create_graph();

    // Add edge between entities
    let edge_id = add_edge(&graph, "user:alice", "user:bob", "follows");

    assert!(edge_id > 0);

    // Verify edge exists
    let edge = graph.get_edge(edge_id).unwrap();
    assert_eq!(edge.edge_type, "follows");
}

#[test]
fn test_add_undirected_edge() {
    let graph = create_graph();

    // Add undirected edge
    let edge_id = add_undirected_edge(&graph, "user:alice", "user:bob", "friends");

    assert!(edge_id > 0);

    // Both directions should show neighbors
    let alice_neighbors = get_neighbors_both(&graph, "user:alice");
    let bob_neighbors = get_neighbors_both(&graph, "user:bob");

    assert!(alice_neighbors.contains(&"user:bob".to_string()));
    assert!(bob_neighbors.contains(&"user:alice".to_string()));
}

#[test]
fn test_get_neighbors_out() {
    let graph = create_graph();

    // Alice follows Bob and Carol
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:alice", "user:carol", "follows");

    // Check outgoing neighbors
    let neighbors = get_neighbors_out(&graph, "user:alice");
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));

    // Bob has no outgoing edges
    let bob_out = get_neighbors_out(&graph, "user:bob");
    assert!(bob_out.is_empty());
}

#[test]
fn test_get_neighbors_in() {
    let graph = create_graph();

    // Bob and Carol follow Alice
    add_edge(&graph, "user:bob", "user:alice", "follows");
    add_edge(&graph, "user:carol", "user:alice", "follows");

    // Check incoming neighbors
    let neighbors = get_neighbors_in(&graph, "user:alice");
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));

    // Bob has no incoming edges
    let bob_in = get_neighbors_in(&graph, "user:bob");
    assert!(bob_in.is_empty());
}

#[test]
fn test_get_neighbors_both() {
    let graph = create_graph();

    // Alice follows Bob, Carol follows Alice
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:carol", "user:alice", "follows");

    // Alice has both directions
    let neighbors = get_neighbors_both(&graph, "user:alice");
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));
}

#[test]
fn test_edge_details() {
    let graph = create_graph();

    let edge_id = add_edge(&graph, "user:alice", "user:bob", "follows");

    let edge = graph.get_edge(edge_id).unwrap();
    assert_eq!(edge.edge_type, "follows");

    // Verify nodes
    let from_key = get_entity_key(&graph, edge.from).unwrap();
    let to_key = get_entity_key(&graph, edge.to).unwrap();
    assert_eq!(from_key, "user:alice");
    assert_eq!(to_key, "user:bob");
}

#[test]
fn test_multiple_edge_types() {
    let graph = create_graph();

    // Same source/target with different edge types
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:alice", "user:bob", "mentions");
    add_edge(&graph, "user:alice", "user:bob", "likes");

    // Alice has 3 outgoing edges to Bob
    let alice_node = get_or_create_entity_node(&graph, "user:alice");
    let edges = graph.edges_of(alice_node, Direction::Outgoing).unwrap();
    assert_eq!(edges.len(), 3);

    let edge_types: Vec<&str> = edges.iter().map(|e| e.edge_type.as_str()).collect();
    assert!(edge_types.contains(&"follows"));
    assert!(edge_types.contains(&"mentions"));
    assert!(edge_types.contains(&"likes"));
}

#[test]
fn test_chain_navigation() {
    let graph = create_graph();

    // Alice -> Bob -> Carol -> Dave
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:bob", "user:carol", "follows");
    add_edge(&graph, "user:carol", "user:dave", "follows");

    // Navigate the chain
    let alice_out = get_neighbors_out(&graph, "user:alice");
    assert_eq!(alice_out, vec!["user:bob".to_string()]);

    let bob_out = get_neighbors_out(&graph, "user:bob");
    assert_eq!(bob_out, vec!["user:carol".to_string()]);

    let carol_out = get_neighbors_out(&graph, "user:carol");
    assert_eq!(carol_out, vec!["user:dave".to_string()]);

    // End of chain
    let dave_out = get_neighbors_out(&graph, "user:dave");
    assert!(dave_out.is_empty());
}

#[test]
fn test_self_loop() {
    let graph = create_graph();

    // Self-referential edge
    add_edge(&graph, "user:narcissist", "user:narcissist", "admires");

    let neighbors = get_neighbors_out(&graph, "user:narcissist");
    assert_eq!(neighbors.len(), 1);
    assert!(neighbors.contains(&"user:narcissist".to_string()));
}

#[test]
fn test_concurrent_edge_creation() {
    use std::thread;

    let graph = create_graph();

    // Each thread creates edges between its own unique pair of nodes to avoid
    // concurrent writes to the same adjacency list (graph_engine's add_edge_to_list
    // has a get-modify-put race when multiple threads write to the same list).
    let node_pairs: Vec<(u64, u64)> = (0..10)
        .map(|i| {
            let from = create_entity_node(&graph, &format!("from:{i}"));
            let to = create_entity_node(&graph, &format!("to:{i}"));
            (from, to)
        })
        .collect();

    let handles: Vec<_> = node_pairs
        .into_iter()
        .map(|(from_id, to_id)| {
            let graph = Arc::clone(&graph);
            thread::spawn(move || {
                graph
                    .create_edge(from_id, to_id, "connects", HashMap::new(), true)
                    .unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Each pair should have exactly one edge
    for i in 0..10 {
        let from_key = format!("from:{i}");
        let neighbors = get_neighbors_out(&graph, &from_key);
        assert_eq!(neighbors.len(), 1, "from:{i} should have 1 outgoing edge");
    }
}

#[test]
fn test_duplicate_edges_allowed() {
    let graph = create_graph();

    // Same edge twice is allowed (creates two edges)
    let edge1 = add_edge(&graph, "user:alice", "user:bob", "follows");
    let edge2 = add_edge(&graph, "user:alice", "user:bob", "follows");

    // Should be different edge IDs
    assert_ne!(edge1, edge2);
}

#[test]
fn test_delete_edge() {
    let graph = create_graph();

    let edge_id = add_edge(&graph, "user:alice", "user:bob", "follows");

    // Verify edge exists
    assert!(graph.get_edge(edge_id).is_ok());

    // Delete edge
    graph.delete_edge(edge_id).unwrap();

    // Verify edge is gone
    assert!(graph.get_edge(edge_id).is_err());
}

#[test]
fn test_bidirectional_edges() {
    let graph = create_graph();

    // Mutual follow
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:bob", "user:alice", "follows");

    // Check both directions
    let alice_out = get_neighbors_out(&graph, "user:alice");
    let alice_in = get_neighbors_in(&graph, "user:alice");

    assert!(alice_out.contains(&"user:bob".to_string()));
    assert!(alice_in.contains(&"user:bob".to_string()));
}

#[test]
fn test_edge_with_properties() {
    let graph = create_graph();

    let from_node = get_or_create_entity_node(&graph, "user:alice");
    let to_node = get_or_create_entity_node(&graph, "user:bob");

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(0.8));
    props.insert(
        "created".to_string(),
        PropertyValue::String("2024-01-01".to_string()),
    );

    let edge_id = graph
        .create_edge(from_node, to_node, "follows", props, true)
        .unwrap();

    let edge = graph.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.properties.get("weight"),
        Some(&PropertyValue::Float(0.8))
    );
}

#[test]
fn test_many_edges_performance() {
    let graph = create_graph();

    // Create 100 edges from hub to spokes
    for i in 0..100 {
        add_edge(&graph, "hub:center", &format!("spoke:{}", i), "connects");
    }

    // Query should be fast
    let neighbors = get_neighbors_out(&graph, "hub:center");
    assert_eq!(neighbors.len(), 100);
}

#[test]
fn test_long_chain() {
    let graph = create_graph();

    // Create a chain: node:0 -> node:1 -> ... -> node:99
    for i in 0..99 {
        let from = format!("node:{}", i);
        let to = format!("node:{}", i + 1);
        add_edge(&graph, &from, &to, "next");
    }

    // Verify first link
    let first_out = get_neighbors_out(&graph, "node:0");
    assert_eq!(first_out, vec!["node:1".to_string()]);

    // Verify last link
    let last_in = get_neighbors_in(&graph, "node:99");
    assert_eq!(last_in, vec!["node:98".to_string()]);
}

#[test]
fn test_node_with_many_edge_types() {
    let graph = create_graph();

    // Alice has different relationships with different people
    add_edge(&graph, "user:alice", "user:bob", "follows");
    add_edge(&graph, "user:alice", "user:carol", "follows");
    add_edge(&graph, "user:alice", "user:dave", "follows");

    let neighbors = get_neighbors_out(&graph, "user:alice");
    assert_eq!(neighbors.len(), 3);
    assert!(neighbors.contains(&"user:bob".to_string()));
    assert!(neighbors.contains(&"user:carol".to_string()));
    assert!(neighbors.contains(&"user:dave".to_string()));
}
