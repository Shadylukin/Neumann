//! Graph-based access control using topological path verification.

use crate::Permission;
use graph_engine::GraphEngine;
use std::collections::{HashSet, VecDeque};

/// Access controller using graph topology for authorization.
pub struct AccessController;

/// Edge type prefix for vault access grants.
const VAULT_ACCESS_PREFIX: &str = "VAULT_ACCESS";

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
                    if edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                        accessors.push(from);
                    }
                }
            }
        }

        accessors
    }

    /// Get the highest permission level for a requester on a target.
    ///
    /// Returns None if no access path exists.
    pub fn get_permission_level(
        graph: &GraphEngine,
        source: &str,
        target: &str,
    ) -> Option<Permission> {
        if source == target {
            return Some(Permission::Admin);
        }

        // BFS traversal tracking the best permission found along each path
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut best_permission: Option<Permission> = None;

        queue.push_back((source.to_string(), Permission::Admin));
        visited.insert(source.to_string());

        while let Some((current, current_perm)) = queue.pop_front() {
            // Get outgoing edges to check their types
            if let Ok(outgoing) = graph.get_entity_outgoing(&current) {
                for edge_key in outgoing {
                    if let Ok((_, to, edge_type, _)) = graph.get_entity_edge(&edge_key) {
                        // Determine permission from edge type
                        let edge_perm = if edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                            Permission::from_edge_type(&edge_type)
                        } else {
                            // Non-VAULT_ACCESS edges inherit current permission
                            Some(current_perm)
                        };

                        if let Some(perm) = edge_perm {
                            // Take the minimum permission along the path
                            let path_perm = Self::min_permission(current_perm, perm);

                            if to == target {
                                // Found target - update best permission if better
                                best_permission = Some(match best_permission {
                                    None => path_perm,
                                    Some(existing) => Self::max_permission(existing, path_perm),
                                });
                                continue;
                            }

                            if !visited.contains(&to) {
                                visited.insert(to.clone());
                                queue.push_back((to, path_perm));
                            }
                        }
                    }
                }
            }
        }

        best_permission
    }

    /// Check if a path exists with at least the required permission level.
    pub fn check_path_with_permission(
        graph: &GraphEngine,
        source: &str,
        target: &str,
        required: Permission,
    ) -> bool {
        match Self::get_permission_level(graph, source, target) {
            Some(perm) => perm.allows(required),
            None => false,
        }
    }

    fn min_permission(a: Permission, b: Permission) -> Permission {
        match (a, b) {
            (Permission::Read, _) | (_, Permission::Read) => Permission::Read,
            (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
            (Permission::Admin, Permission::Admin) => Permission::Admin,
        }
    }

    fn max_permission(a: Permission, b: Permission) -> Permission {
        match (a, b) {
            (Permission::Admin, _) | (_, Permission::Admin) => Permission::Admin,
            (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
            (Permission::Read, Permission::Read) => Permission::Read,
        }
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

    // === Permission Level Tests ===

    #[test]
    fn test_permission_level_direct_read() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Read));
    }

    #[test]
    fn test_permission_level_direct_write() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_WRITE")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Write));
    }

    #[test]
    fn test_permission_level_direct_admin() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_ADMIN")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_backward_compat() {
        let graph = GraphEngine::new();

        // Old-style VAULT_ACCESS edge should be treated as Admin
        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_transitive_minimum() {
        let graph = GraphEngine::new();

        // alice -> team (MEMBER) -> secret (VAULT_ACCESS_READ)
        // The path gives Alice only Read permission via the chain
        graph
            .add_entity_edge("user:alice", "team:devs", "MEMBER")
            .unwrap();
        graph
            .add_entity_edge("team:devs", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Read));
    }

    #[test]
    fn test_permission_level_best_of_multiple_paths() {
        let graph = GraphEngine::new();

        // Alice has two paths:
        // Path 1: alice -> secret (VAULT_ACCESS_READ)
        // Path 2: alice -> team -> secret (VAULT_ACCESS_ADMIN)
        // Should get Admin (best of both)
        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();
        graph
            .add_entity_edge("user:alice", "team:devs", "MEMBER")
            .unwrap();
        graph
            .add_entity_edge("team:devs", "secret:key", "VAULT_ACCESS_ADMIN")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_no_path() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:bob", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, None);
    }

    #[test]
    fn test_check_path_with_permission_read_ok() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();

        assert!(AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Read
        ));
    }

    #[test]
    fn test_check_path_with_permission_read_denied_write() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();

        // Read permission doesn't allow Write
        assert!(!AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Write
        ));
    }

    #[test]
    fn test_check_path_with_permission_write_allows_read() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_WRITE")
            .unwrap();

        // Write permission allows Read
        assert!(AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Read
        ));
    }

    #[test]
    fn test_check_path_with_permission_admin_allows_all() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_ADMIN")
            .unwrap();

        assert!(AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Read
        ));
        assert!(AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Write
        ));
        assert!(AccessController::check_path_with_permission(
            &graph,
            "user:alice",
            "secret:key",
            Permission::Admin
        ));
    }

    #[test]
    fn test_get_direct_accessors_with_permission_levels() {
        let graph = GraphEngine::new();

        graph
            .add_entity_edge("user:alice", "secret:key", "VAULT_ACCESS_READ")
            .unwrap();
        graph
            .add_entity_edge("user:bob", "secret:key", "VAULT_ACCESS_WRITE")
            .unwrap();
        graph
            .add_entity_edge("user:carol", "secret:key", "VAULT_ACCESS_ADMIN")
            .unwrap();

        let accessors = AccessController::get_direct_accessors(&graph, "secret:key");

        assert_eq!(accessors.len(), 3);
        assert!(accessors.contains(&"user:alice".to_string()));
        assert!(accessors.contains(&"user:bob".to_string()));
        assert!(accessors.contains(&"user:carol".to_string()));
    }
}
