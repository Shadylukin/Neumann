// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph-based access control using topological path verification.

#[cfg(test)]
use std::collections::HashMap;
use std::collections::{HashSet, VecDeque};

use graph_engine::{Direction, GraphEngine, PropertyValue};

use crate::attenuation::AttenuationPolicy;
use crate::signing::EdgeSigner;
use crate::Permission;

/// Access controller using graph topology for authorization.
pub struct AccessController;

// ========== Node-based Graph API Helpers ==========
// These functions provide a string-key interface on top of the node-based graph API.

/// Ensure the entity_key index exists.
fn ensure_entity_key_index(graph: &GraphEngine) {
    // Create index if it doesn't exist (idempotent operation)
    let _ = graph.create_node_property_index("entity_key");
}

/// Get or create a graph node for an entity key (test helper).
#[cfg(test)]
fn get_or_create_entity_node(graph: &GraphEngine, entity_key: &str) -> u64 {
    ensure_entity_key_index(graph);

    if let Ok(nodes) =
        graph.find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
    {
        if let Some(node) = nodes.first() {
            return node.id;
        }
    }

    let mut props = HashMap::new();
    props.insert(
        "entity_key".to_string(),
        PropertyValue::String(entity_key.to_string()),
    );
    graph.create_node("AccessEntity", props).unwrap_or(0)
}

/// Find node by entity key without creating.
fn find_entity_node(graph: &GraphEngine, entity_key: &str) -> Option<u64> {
    ensure_entity_key_index(graph);
    graph
        .find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
        .ok()
        .and_then(|nodes| nodes.first().map(|n| n.id))
}

/// Get the entity key for a node ID.
fn get_entity_key(graph: &GraphEngine, node_id: u64) -> Option<String> {
    graph.get_node(node_id).ok().and_then(|node| {
        if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
            Some(key.clone())
        } else {
            None
        }
    })
}

/// Edge info returned from graph traversal.
struct EdgeInfo {
    target_key: String,
    source_key: String,
    edge_type: String,
    signature: Option<Vec<u8>>,
    sig_timestamp: Option<i64>,
    /// Bottleneck capacity (1=Read, 2=Write, 3=Admin), if set.
    capacity: Option<i64>,
}

/// Get outgoing edges for an entity, returning (target_key, edge_type).
fn get_outgoing_edges(graph: &GraphEngine, entity_key: &str) -> Vec<(String, String)> {
    get_outgoing_edges_full(graph, entity_key)
        .into_iter()
        .map(|e| (e.target_key, e.edge_type))
        .collect()
}

/// Get outgoing edges with full info including signature properties.
fn get_outgoing_edges_full(graph: &GraphEngine, entity_key: &str) -> Vec<EdgeInfo> {
    let Some(node_id) = find_entity_node(graph, entity_key) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Outgoing) {
        for edge in edges {
            let target_id = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };
            if let Some(target_key) = get_entity_key(graph, target_id) {
                let signature = match edge.properties.get("vault_sig") {
                    Some(PropertyValue::Bytes(b)) => Some(b.clone()),
                    _ => None,
                };
                let sig_timestamp = match edge.properties.get("vault_sig_ts") {
                    Some(PropertyValue::Int(ts)) => Some(*ts),
                    _ => None,
                };
                let capacity = match edge.properties.get("vault_capacity") {
                    Some(PropertyValue::Int(c)) => Some(*c),
                    _ => None,
                };
                result.push(EdgeInfo {
                    target_key,
                    source_key: entity_key.to_string(),
                    edge_type: edge.edge_type.clone(),
                    signature,
                    sig_timestamp,
                    capacity,
                });
            }
        }
    }
    result
}

/// Get incoming edges for an entity, returning (source_key, edge_type).
fn get_incoming_edges(graph: &GraphEngine, entity_key: &str) -> Vec<(String, String)> {
    let Some(node_id) = find_entity_node(graph, entity_key) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Incoming) {
        for edge in edges {
            let source_id = if edge.to == node_id {
                edge.from
            } else {
                edge.to
            };
            if let Some(source_key) = get_entity_key(graph, source_id) {
                result.push((source_key, edge.edge_type.clone()));
            }
        }
    }
    result
}

/// Add an edge between two entity keys (test helper).
#[cfg(test)]
fn add_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) -> u64 {
    let from_node = get_or_create_entity_node(graph, from_key);
    let to_node = get_or_create_entity_node(graph, to_key);
    graph
        .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
        .unwrap_or(0)
}

/// Edge type prefix for vault access grants.
const VAULT_ACCESS_PREFIX: &str = "VAULT_ACCESS";

/// Allowlisted edge types that can be traversed during access control checks.
/// Only these edge types can grant transitive access permissions.
/// - VAULT_ACCESS_* edges grant explicit vault permissions
/// - MEMBER edges allow group membership traversal
const ALLOWED_TRAVERSAL_EDGES: &[&str] = &[
    "VAULT_ACCESS",
    "VAULT_ACCESS_READ",
    "VAULT_ACCESS_WRITE",
    "VAULT_ACCESS_ADMIN",
    "MEMBER",
];

/// Hard limit on BFS traversal depth to prevent DoS via long MEMBER chains.
const MAX_BFS_DEPTH: usize = 32;

/// Check if an edge type is allowed for access control traversal.
fn is_allowed_edge_type(edge_type: &str) -> bool {
    ALLOWED_TRAVERSAL_EDGES
        .iter()
        .any(|&allowed| edge_type.starts_with(allowed))
}

impl AccessController {
    /// Check if a path exists from source to target in the graph.
    ///
    /// Uses BFS following only outgoing edges from each node.
    /// Only traverses edges in the allowlist (VAULT_ACCESS_*, MEMBER).
    /// Returns true if path exists, false otherwise.
    pub fn check_path(graph: &GraphEngine, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }

        // BFS traversal - only follow outgoing edges with allowed types
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((source.to_string(), 0));
        visited.insert(source.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= MAX_BFS_DEPTH {
                continue;
            }

            for (to, edge_type) in get_outgoing_edges(graph, &current) {
                // Only traverse allowed edge types
                if !is_allowed_edge_type(&edge_type) {
                    continue;
                }

                if to == target {
                    return true;
                }

                if !visited.contains(&to) {
                    visited.insert(to.clone());
                    queue.push_back((to, depth + 1));
                }
            }
        }

        false
    }

    /// Get all entities that have direct access to a target via VAULT_ACCESS edges.
    pub fn get_direct_accessors(graph: &GraphEngine, target: &str) -> Vec<String> {
        let mut accessors = Vec::new();

        for (from, edge_type) in get_incoming_edges(graph, target) {
            if edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                accessors.push(from);
            }
        }

        accessors
    }

    /// Get the highest permission level for a requester on a target.
    ///
    /// Returns None if no access path exists.
    ///
    /// SECURITY: MEMBER edges allow graph traversal but do NOT grant permission
    /// to the target. Only VAULT_ACCESS_* edges grant actual permissions.
    /// This prevents privilege escalation via group membership.
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
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut best_permission: Option<Permission> = None;

        queue.push_back((source.to_string(), 0));
        visited.insert(source.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= MAX_BFS_DEPTH {
                continue;
            }

            for (to, edge_type) in get_outgoing_edges(graph, &current) {
                // Only traverse allowed edge types
                if !is_allowed_edge_type(&edge_type) {
                    continue;
                }

                // SECURITY FIX: Only VAULT_ACCESS_* edges can grant permission
                // MEMBER edges allow traversal but do NOT grant permission to target
                if edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                    if to == target {
                        // Found target via VAULT_ACCESS edge - extract permission
                        if let Some(perm) = Permission::from_edge_type(&edge_type) {
                            best_permission = Some(match best_permission {
                                None => perm,
                                Some(existing) => Self::max_permission(existing, perm),
                            });
                        }
                    }
                    // Don't continue traversal past VAULT_ACCESS edges
                    // (they point to secrets, not to groups)
                } else {
                    // MEMBER edge - allow traversal but no permission granted
                    if !visited.contains(&to) {
                        visited.insert(to.clone());
                        queue.push_back((to, depth + 1));
                    }
                }
            }
        }

        best_permission
    }

    /// Get the highest permission level with edge signature verification
    /// and distance-based attenuation.
    ///
    /// Like `get_permission_level`, but:
    /// - Verifies HMAC signatures on VAULT_ACCESS edges (tampered = skipped)
    /// - Unsigned (legacy) edges are accepted
    /// - Applies distance-based attenuation via `AttenuationPolicy`
    /// - BFS depth is bounded by the policy's `horizon`
    pub fn get_permission_level_verified(
        graph: &GraphEngine,
        source: &str,
        target: &str,
        signer: &EdgeSigner,
        policy: &AttenuationPolicy,
    ) -> Option<Permission> {
        if source == target {
            return Some(Permission::Admin);
        }

        let mut visited = HashSet::new();
        // (entity_key, depth) -- depth counts MEMBER hops traversed
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut best_permission: Option<Permission> = None;

        queue.push_back((source.to_string(), 0));
        visited.insert(source.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            // BFS depth bounded by attenuation horizon
            if depth >= policy.horizon {
                continue;
            }

            for edge in get_outgoing_edges_full(graph, &current) {
                if !is_allowed_edge_type(&edge.edge_type) {
                    continue;
                }

                if edge.edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                    if edge.target_key == target {
                        // Verify signature if present; skip tampered edges
                        if let (Some(sig), Some(ts)) = (&edge.signature, edge.sig_timestamp) {
                            if !signer.verify_edge(
                                &edge.source_key,
                                &edge.target_key,
                                &edge.edge_type,
                                ts,
                                sig,
                            ) {
                                continue;
                            }
                        }
                        // The VAULT_ACCESS hop itself counts as +1
                        let total_hops = depth + 1;
                        if let Some(perm) = Permission::from_edge_type(&edge.edge_type) {
                            if let Some(attenuated) = policy.attenuate(perm, total_hops) {
                                // Apply bottleneck: capacity limits the effective permission
                                let effective = match edge.capacity.and_then(Permission::from_level)
                                {
                                    Some(cap) => Self::min_permission(attenuated, cap),
                                    None => attenuated,
                                };
                                best_permission = Some(match best_permission {
                                    None => effective,
                                    Some(existing) => Self::max_permission(existing, effective),
                                });
                            }
                        }
                    }
                } else if !visited.contains(&edge.target_key) {
                    visited.insert(edge.target_key.clone());
                    queue.push_back((edge.target_key, depth + 1));
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

    /// Check permission with signature verification and attenuation.
    pub fn check_path_with_permission_verified(
        graph: &GraphEngine,
        source: &str,
        target: &str,
        required: Permission,
        signer: &EdgeSigner,
        policy: &AttenuationPolicy,
    ) -> bool {
        match Self::get_permission_level_verified(graph, source, target, signer, policy) {
            Some(perm) => perm.allows(required),
            None => false,
        }
    }

    fn max_permission(a: Permission, b: Permission) -> Permission {
        match (a, b) {
            (Permission::Admin, _) | (_, Permission::Admin) => Permission::Admin,
            (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
            (Permission::Read, Permission::Read) => Permission::Read,
        }
    }

    fn min_permission(a: Permission, b: Permission) -> Permission {
        match (a, b) {
            (Permission::Read, _) | (_, Permission::Read) => Permission::Read,
            (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
            (Permission::Admin, Permission::Admin) => Permission::Admin,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_node_creation_and_lookup() {
        let graph = GraphEngine::new();

        // Create a node
        let node_id = get_or_create_entity_node(&graph, "test:entity");
        assert!(node_id > 0, "Node should be created with positive ID");

        // Verify we can find it again
        let found = find_entity_node(&graph, "test:entity");
        assert_eq!(found, Some(node_id), "Should find the same node");

        // Verify the property is set correctly
        let node = graph.get_node(node_id).expect("Should get node");
        let key = node.properties.get("entity_key");
        assert_eq!(
            key,
            Some(&PropertyValue::String("test:entity".to_string())),
            "Node should have correct entity_key property"
        );
    }

    #[test]
    fn test_edge_creation_and_lookup() {
        let graph = GraphEngine::new();

        // Create edge
        let edge_id = add_edge(&graph, "from:a", "to:b", "TEST_EDGE");
        assert!(edge_id > 0, "Edge should be created");

        // Verify outgoing edges
        let outgoing = get_outgoing_edges(&graph, "from:a");
        assert_eq!(outgoing.len(), 1, "Should have 1 outgoing edge");
        assert_eq!(outgoing[0].0, "to:b", "Target should be to:b");
        assert_eq!(outgoing[0].1, "TEST_EDGE", "Edge type should be TEST_EDGE");
    }

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

        add_edge(&graph, "user:alice", "secret:api_key", "VAULT_ACCESS");

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
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:api_key", "VAULT_ACCESS");

        assert!(AccessController::check_path(
            &graph,
            "user:alice",
            "secret:api_key"
        ));
    }

    #[test]
    fn test_no_path() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:alice", "secret:one", "VAULT_ACCESS");
        add_edge(&graph, "user:bob", "secret:two", "VAULT_ACCESS");

        assert!(!AccessController::check_path(
            &graph,
            "user:alice",
            "secret:two"
        ));
    }

    #[test]
    fn test_cycle_handling() {
        let graph = GraphEngine::new();

        // Create a cycle: a -> b -> c -> a using allowed MEMBER edges
        add_edge(&graph, "node:a", "node:b", "MEMBER");
        add_edge(&graph, "node:b", "node:c", "MEMBER");
        add_edge(&graph, "node:c", "node:a", "MEMBER");

        // Should not hang, should find path
        assert!(AccessController::check_path(&graph, "node:a", "node:c"));

        // d is not connected
        assert!(!AccessController::check_path(&graph, "node:a", "node:d"));
    }

    #[test]
    fn test_long_path() {
        let graph = GraphEngine::new();

        // Create a long chain: node:0 -> node:1 -> ... -> node:10 using allowed MEMBER edges
        for i in 0..10 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }

        // Forward direction works
        assert!(AccessController::check_path(&graph, "node:0", "node:10"));
        // Reverse direction should NOT work (directional access)
        assert!(!AccessController::check_path(&graph, "node:10", "node:0"));
    }

    #[test]
    fn test_disallowed_edge_type_blocked() {
        let graph = GraphEngine::new();

        // Create path using disallowed edge type
        add_edge(&graph, "node:a", "node:b", "RANDOM_EDGE");

        // Path should NOT be found because RANDOM_EDGE is not allowlisted
        assert!(!AccessController::check_path(&graph, "node:a", "node:b"));
    }

    #[test]
    fn test_directional_path() {
        let graph = GraphEngine::new();

        add_edge(&graph, "node:a", "node:b", "MEMBER");

        // Only forward direction works (a -> b)
        assert!(AccessController::check_path(&graph, "node:a", "node:b"));
        // Reverse does NOT work
        assert!(!AccessController::check_path(&graph, "node:b", "node:a"));
    }

    #[test]
    fn test_get_direct_accessors() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS");
        add_edge(&graph, "user:bob", "secret:key", "VAULT_ACCESS");
        add_edge(&graph, "user:carol", "secret:key", "OTHER_EDGE");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_READ");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Read));
    }

    #[test]
    fn test_permission_level_direct_write() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_WRITE");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Write));
    }

    #[test]
    fn test_permission_level_direct_admin() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_ADMIN");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_backward_compat() {
        let graph = GraphEngine::new();

        // Old-style VAULT_ACCESS edge should be treated as Admin
        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_transitive_minimum() {
        let graph = GraphEngine::new();

        // alice -> team (MEMBER) -> secret (VAULT_ACCESS_READ)
        // The path gives Alice only Read permission via the chain
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:key", "VAULT_ACCESS_READ");

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
        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_READ");
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:key", "VAULT_ACCESS_ADMIN");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Admin));
    }

    #[test]
    fn test_permission_level_no_path() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:bob", "secret:key", "VAULT_ACCESS_READ");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, None);
    }

    #[test]
    fn test_check_path_with_permission_read_ok() {
        let graph = GraphEngine::new();

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_READ");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_READ");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_WRITE");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_ADMIN");

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

        add_edge(&graph, "user:alice", "secret:key", "VAULT_ACCESS_READ");
        add_edge(&graph, "user:bob", "secret:key", "VAULT_ACCESS_WRITE");
        add_edge(&graph, "user:carol", "secret:key", "VAULT_ACCESS_ADMIN");

        let accessors = AccessController::get_direct_accessors(&graph, "secret:key");

        assert_eq!(accessors.len(), 3);
        assert!(accessors.contains(&"user:alice".to_string()));
        assert!(accessors.contains(&"user:bob".to_string()));
        assert!(accessors.contains(&"user:carol".to_string()));
    }

    // === Security Tests for MEMBER Edge Permission ===

    #[test]
    fn test_member_edge_direct_to_secret_no_permission() {
        let graph = GraphEngine::new();

        // SECURITY: MEMBER edge directly to secret should NOT grant permission
        add_edge(&graph, "user:alice", "secret:key", "MEMBER");

        // Should return None - no VAULT_ACCESS_* edge to secret
        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, None);
    }

    #[test]
    fn test_member_chain_without_vault_access_no_permission() {
        let graph = GraphEngine::new();

        // alice -> team (MEMBER) -> secret (MEMBER)
        // SECURITY: This should NOT grant any permission
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:key", "MEMBER");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, None);
    }

    #[test]
    fn test_member_traversal_to_vault_access_grants_permission() {
        let graph = GraphEngine::new();

        // alice -> team (MEMBER) -> secret (VAULT_ACCESS_WRITE)
        // Should grant Write permission via the chain
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:key", "VAULT_ACCESS_WRITE");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Write));
    }

    #[test]
    fn test_member_with_mixed_access_paths() {
        let graph = GraphEngine::new();

        // alice -> team1 (MEMBER) -> secret (MEMBER) - no permission
        // alice -> team2 (MEMBER) -> secret (VAULT_ACCESS_READ) - Read permission
        // Should get Read from the valid path
        add_edge(&graph, "user:alice", "team:team1", "MEMBER");
        add_edge(&graph, "team:team1", "secret:key", "MEMBER");
        add_edge(&graph, "user:alice", "team:team2", "MEMBER");
        add_edge(&graph, "team:team2", "secret:key", "VAULT_ACCESS_READ");

        let perm = AccessController::get_permission_level(&graph, "user:alice", "secret:key");
        assert_eq!(perm, Some(Permission::Read));
    }

    #[test]
    fn test_check_path_still_works_with_member() {
        let graph = GraphEngine::new();

        // check_path (for path existence) should still work with MEMBER edges
        add_edge(&graph, "user:alice", "team:devs", "MEMBER");
        add_edge(&graph, "team:devs", "secret:key", "MEMBER");

        // Path exists (MEMBER edges connect the nodes)
        assert!(AccessController::check_path(
            &graph,
            "user:alice",
            "secret:key"
        ));
        // But no permission is granted
        assert_eq!(
            AccessController::get_permission_level(&graph, "user:alice", "secret:key"),
            None
        );
    }

    // === BFS Depth Limit Tests ===

    #[test]
    fn test_check_path_depth_limit_exceeded() {
        let graph = GraphEngine::new();

        // Chain of 35 MEMBER edges: node:0 -> node:1 -> ... -> node:35
        for i in 0..35 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }

        // 35 hops exceeds MAX_BFS_DEPTH (32), path should not be found
        assert!(!AccessController::check_path(&graph, "node:0", "node:35"));
    }

    #[test]
    fn test_check_path_within_depth_limit() {
        let graph = GraphEngine::new();

        // Chain of 30 MEMBER edges: node:0 -> node:1 -> ... -> node:30
        for i in 0..30 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }

        // 30 hops is within MAX_BFS_DEPTH (32), path should be found
        assert!(AccessController::check_path(&graph, "node:0", "node:30"));
    }

    #[test]
    fn test_get_permission_level_depth_limit_exceeded() {
        let graph = GraphEngine::new();

        // Chain of 35 MEMBER edges then a VAULT_ACCESS_WRITE edge
        for i in 0..35 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }
        add_edge(&graph, "node:35", "secret:key", "VAULT_ACCESS_WRITE");

        // 35 MEMBER hops exceeds MAX_BFS_DEPTH (32), should return None
        assert_eq!(
            AccessController::get_permission_level(&graph, "node:0", "secret:key"),
            None
        );
    }

    #[test]
    fn test_get_permission_level_within_depth_limit() {
        let graph = GraphEngine::new();

        // Chain of 30 MEMBER edges then a VAULT_ACCESS_WRITE edge
        for i in 0..30 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }
        add_edge(&graph, "node:30", "secret:key", "VAULT_ACCESS_WRITE");

        // 30 MEMBER hops is within MAX_BFS_DEPTH (32), should find permission
        assert_eq!(
            AccessController::get_permission_level(&graph, "node:0", "secret:key"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_check_path_at_exact_boundary() {
        let graph = GraphEngine::new();

        // Chain of exactly 31 MEMBER edges: node:0 -> ... -> node:31
        // At depth 31, node:31 is enqueued with depth=31 which is < 32, so it
        // will be processed and its edges explored. A target at node:32 requires
        // the edge from node:31 (depth=31) to be explored, which succeeds.
        for i in 0..32 {
            add_edge(
                &graph,
                &format!("node:{i}"),
                &format!("node:{}", i + 1),
                "MEMBER",
            );
        }

        // 31 hops (depth 31) should still find node:32 as a neighbor
        assert!(AccessController::check_path(&graph, "node:0", "node:32"));

        // But 33 hops should fail: node:32 is enqueued at depth=32 and skipped
        add_edge(&graph, "node:32", "node:33", "MEMBER");
        assert!(!AccessController::check_path(&graph, "node:0", "node:33"));
    }
}
