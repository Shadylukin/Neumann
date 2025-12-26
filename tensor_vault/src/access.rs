//! Graph-based access control using topological path verification.

use graph_engine::GraphEngine;
use std::collections::{HashSet, VecDeque};

/// Access controller using graph topology for authorization.
pub struct AccessController;

impl AccessController {
    /// Check if a path exists from source to target in the graph.
    ///
    /// Uses BFS following only outgoing edges from each node.
    /// Returns true if path exists, false otherwise.
    pub fn check_path(graph: &GraphEngine, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }

        // BFS traversal - only follow outgoing edges
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(source.to_string());
        visited.insert(source.to_string());

        while let Some(current) = queue.pop_front() {
            // Only follow outgoing edges (directional access)
            let neighbors = graph.get_entity_neighbors_out(&current).unwrap_or_default();

            for neighbor in neighbors {
                if neighbor == target {
                    return true;
                }

                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor);
                }
            }
        }

        false
    }

    /// Get all entities that have direct access to a target via VAULT_ACCESS edges.
    pub fn get_direct_accessors(graph: &GraphEngine, target: &str) -> Vec<String> {
        let mut accessors = Vec::new();

        if let Ok(incoming) = graph.get_entity_incoming(target) {
            for edge_key in incoming {
                if let Ok((from, _, edge_type, _)) = graph.get_entity_edge(&edge_key) {
                    if edge_type == "VAULT_ACCESS" {
                        accessors.push(from);
                    }
                }
            }
        }

        accessors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_same_node() {
        let graph = GraphEngine::new();

        assert!(AccessController::check_path(
            &graph,
            "user:alice",
            "user:alice"
        ));
    }

    #[test]
    fn test_direct_path() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:api_key", "VAULT_ACCESS")
            .unwrap();

        assert!(AccessController::check_path(
            &graph,
            "user:alice",
            "secret:api_key"
        ));
        assert!(!AccessController::check_path(
            &graph,
            "user:bob",
            "secret:api_key"
        ));
    }

    #[test]
    fn test_transitive_path() {
        let graph = GraphEngine::new();

        // alice -> team -> secret
        graph
            .add_entity_edge("user:alice", "team:devs", "MEMBER")
            .unwrap();
        graph
            .add_entity_edge("team:devs", "secret:api_key", "VAULT_ACCESS")
            .unwrap();

        assert!(AccessController::check_path(
            &graph,
            "user:alice",
            "secret:api_key"
        ));
    }

    #[test]
    fn test_no_path() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:one", "VAULT_ACCESS")
            .unwrap();
        graph
            .add_entity_edge("user:bob", "secret:two", "VAULT_ACCESS")
            .unwrap();

        assert!(!AccessController::check_path(
            &graph,
            "user:alice",
            "secret:two"
        ));
    }

    #[test]
    fn test_cycle_handling() {
        let graph = GraphEngine::new();

        // Create a cycle: a -> b -> c -> a
        graph.add_entity_edge("node:a", "node:b", "LINK").unwrap();
        graph.add_entity_edge("node:b", "node:c", "LINK").unwrap();
        graph.add_entity_edge("node:c", "node:a", "LINK").unwrap();

        // Should not hang, should find path
        assert!(AccessController::check_path(&graph, "node:a", "node:c"));

        // d is not connected
        assert!(!AccessController::check_path(&graph, "node:a", "node:d"));
    }

    #[test]
    fn test_long_path() {
        let graph = GraphEngine::new();

        // Create a long chain: node:0 -> node:1 -> ... -> node:10
        for i in 0..10 {
            graph
                .add_entity_edge(&format!("node:{i}"), &format!("node:{}", i + 1), "NEXT")
                .unwrap();
        }

        // Forward direction works
        assert!(AccessController::check_path(&graph, "node:0", "node:10"));
        // Reverse direction should NOT work (directional access)
        assert!(!AccessController::check_path(&graph, "node:10", "node:0"));
    }

    #[test]
    fn test_directional_path() {
        let graph = GraphEngine::new();

        graph.add_entity_edge("node:a", "node:b", "LINK").unwrap();

        // Only forward direction works (a -> b)
        assert!(AccessController::check_path(&graph, "node:a", "node:b"));
        // Reverse does NOT work
        assert!(!AccessController::check_path(&graph, "node:b", "node:a"));
    }

    #[test]
    fn test_get_direct_accessors() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS")
            .unwrap();
        graph
            .add_entity_edge("user:bob", "secret:key", "VAULT_ACCESS")
            .unwrap();
        graph
            .add_entity_edge("user:carol", "secret:key", "OTHER_EDGE")
            .unwrap();

        let accessors = AccessController::get_direct_accessors(&graph, "secret:key");

        assert_eq!(accessors.len(), 2);
        assert!(accessors.contains(&"user:alice".to_string()));
        assert!(accessors.contains(&"user:bob".to_string()));
        assert!(!accessors.contains(&"user:carol".to_string()));
    }

    #[test]
    fn test_empty_graph() {
        let graph = GraphEngine::new();

        assert!(!AccessController::check_path(
            &graph,
            "user:alice",
            "secret:key"
        ));
    }

    #[test]
    fn test_concurrent_access_check() {
        use std::thread;

        let graph = Arc::new(GraphEngine::new());

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS")
            .unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let graph = Arc::clone(&graph);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let result =
                            AccessController::check_path(&graph, "user:alice", "secret:key");
                        assert!(result);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
