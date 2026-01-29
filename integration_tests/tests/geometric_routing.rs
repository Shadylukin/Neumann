// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for geometric routing in tensor_chain.
//!
//! Tests the geometric membership manager and geometric routing features.

use std::sync::Arc;

use tensor_chain::{
    geometric_membership::{GeometricMembershipConfig, GeometricMembershipManager},
    membership::{ClusterConfig, LocalNodeConfig, MembershipManager},
    network::MemoryTransport,
    ChainConfig, GeometricRoutingConfig, TensorChain,
};
use tensor_store::{SparseVector, TensorStore};

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_membership() -> Arc<MembershipManager> {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = ClusterConfig::new(
        "test-cluster",
        LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: "127.0.0.1:9100".parse().unwrap(),
        },
    )
    .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
    .with_peer("node3", "127.0.0.1:9102".parse().unwrap());

    Arc::new(MembershipManager::new(config, transport))
}

// ============================================================================
// Geometric Membership Manager Tests
// ============================================================================

#[test]
fn test_geometric_membership_basic() {
    let membership = create_test_membership();
    let config = GeometricMembershipConfig::default();
    let manager = GeometricMembershipManager::new(membership, config);

    // Initially no embeddings
    assert_eq!(manager.cached_embedding_count(), 0);
    assert!(manager.local_embedding().is_none());
}

#[test]
fn test_geometric_membership_with_embeddings() {
    let membership = create_test_membership();
    let manager = GeometricMembershipManager::with_defaults(membership);

    // Update local embedding
    let local_emb = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
    manager.update_local_embedding(local_emb);
    assert!(manager.local_embedding().is_some());

    // Record peer embeddings
    manager.record_peer_embedding(
        &"node2".to_string(),
        SparseVector::from_dense(&[0.9, 0.1, 0.0]),
    );
    manager.record_peer_embedding(
        &"node3".to_string(),
        SparseVector::from_dense(&[0.0, 1.0, 0.0]),
    );

    assert_eq!(manager.cached_embedding_count(), 2);
}

#[test]
fn test_geometric_membership_peer_ranking() {
    let membership = create_test_membership();
    let config = GeometricMembershipConfig {
        nearby_threshold: 0.5,
        geometric_weight: 0.5, // 50% geometric, 50% health
    };
    let manager = GeometricMembershipManager::new(membership, config);

    // Record peer embeddings
    manager.record_peer_embedding(
        &"node2".to_string(),
        SparseVector::from_dense(&[1.0, 0.0, 0.0]),
    );
    manager.record_peer_embedding(
        &"node3".to_string(),
        SparseVector::from_dense(&[0.0, 1.0, 0.0]),
    );

    // Query similar to node2
    let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
    let ranked = manager.ranked_peers(&query);

    assert_eq!(ranked.len(), 2);

    // node2 should have higher similarity
    let node2_ranked = ranked.iter().find(|p| p.node_id == "node2").unwrap();
    let node3_ranked = ranked.iter().find(|p| p.node_id == "node3").unwrap();
    assert!(node2_ranked.similarity > node3_ranked.similarity);
}

#[test]
fn test_geometric_membership_prune_stale() {
    let membership = create_test_membership();
    let manager = GeometricMembershipManager::with_defaults(membership);

    // Add embedding for non-existent node
    manager.record_peer_embedding(
        &"stale_node".to_string(),
        SparseVector::from_dense(&[0.5, 0.5, 0.0]),
    );
    manager.record_peer_embedding(
        &"node2".to_string(),
        SparseVector::from_dense(&[0.5, 0.5, 0.0]),
    );

    assert_eq!(manager.cached_embedding_count(), 2);

    // Prune stale embeddings
    manager.prune_stale_embeddings();

    // Only node2 should remain
    assert_eq!(manager.cached_embedding_count(), 1);
    assert!(manager.peer_embedding(&"node2".to_string()).is_some());
    assert!(manager.peer_embedding(&"stale_node".to_string()).is_none());
}

// ============================================================================
// TensorChain Geometric Routing Tests
// ============================================================================

#[test]
fn test_tensor_chain_geometric_routing_enabled() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");

    assert!(chain.is_geometric_routing_enabled());

    let config = chain.geometric_routing_config();
    assert!(config.enabled);
    assert!((config.min_similarity - 0.5).abs() < 0.001);
    assert!(config.fallback_to_hash);
}

#[test]
fn test_tensor_chain_geometric_routing_disabled() {
    let config =
        ChainConfig::new("node1").with_geometric_routing(GeometricRoutingConfig::disabled());
    let store = TensorStore::new();
    let chain = TensorChain::with_config(store, config);

    assert!(!chain.is_geometric_routing_enabled());
}

#[test]
fn test_tensor_chain_route_by_embedding_disabled() {
    let config =
        ChainConfig::new("node1").with_geometric_routing(GeometricRoutingConfig::disabled());
    let store = TensorStore::new();
    let chain = TensorChain::with_config(store, config);

    let embedding = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
    let routed = chain.route_by_embedding(&embedding);

    // When disabled, routes to local node
    assert_eq!(routed, "node1");
}

#[test]
fn test_tensor_chain_route_by_embedding_empty() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");

    let embedding = SparseVector::from_dense(&[]);
    let routed = chain.route_by_embedding(&embedding);

    // Empty embedding falls back to local node
    assert_eq!(routed, "node1");
}

#[test]
fn test_tensor_chain_route_by_embedding_no_peers() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");

    let embedding = SparseVector::from_dense(&[1.0, 0.5, 0.0]);
    let routed = chain.route_by_embedding(&embedding);

    // Without peers, routes to local node
    assert_eq!(routed, "node1");
}

// ============================================================================
// Geometric Routing Configuration Tests
// ============================================================================

#[test]
fn test_geometric_routing_config_default() {
    let config = GeometricRoutingConfig::default();
    assert!(config.enabled);
    assert!((config.min_similarity - 0.5).abs() < 0.001);
    assert!(config.fallback_to_hash);
}

#[test]
fn test_geometric_routing_config_disabled() {
    let config = GeometricRoutingConfig::disabled();
    assert!(!config.enabled);
    assert!((config.min_similarity - 0.5).abs() < 0.001);
    assert!(config.fallback_to_hash);
}

#[test]
fn test_geometric_routing_config_custom() {
    let config = GeometricRoutingConfig {
        enabled: true,
        min_similarity: 0.8,
        fallback_to_hash: false,
    };

    assert!(config.enabled);
    assert!((config.min_similarity - 0.8).abs() < 0.001);
    assert!(!config.fallback_to_hash);
}

// ============================================================================
// Geometric Membership Config Tests
// ============================================================================

#[test]
fn test_geometric_membership_config_default() {
    let config = GeometricMembershipConfig::default();
    assert!((config.nearby_threshold - 0.5).abs() < 0.001);
    assert!((config.geometric_weight - 0.3).abs() < 0.001);
}

#[test]
fn test_geometric_membership_config_custom() {
    let config = GeometricMembershipConfig {
        nearby_threshold: 0.7,
        geometric_weight: 0.6,
    };

    assert!((config.nearby_threshold - 0.7).abs() < 0.001);
    assert!((config.geometric_weight - 0.6).abs() < 0.001);
}

// ============================================================================
// End-to-End Geometric Routing Tests
// ============================================================================

#[test]
fn test_geometric_routing_with_chain_initialization() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");

    // Initialize the chain
    chain.initialize().unwrap();

    // Verify geometric routing is available
    assert!(chain.is_geometric_routing_enabled());

    // Route some embeddings
    let embeddings = vec![
        SparseVector::from_dense(&[1.0, 0.0, 0.0]),
        SparseVector::from_dense(&[0.0, 1.0, 0.0]),
        SparseVector::from_dense(&[0.0, 0.0, 1.0]),
    ];

    for embedding in &embeddings {
        let routed = chain.route_by_embedding(embedding);
        // All should route to local node (no peers)
        assert_eq!(routed, "node1");
    }
}

#[test]
fn test_geometric_membership_with_sparse_vectors() {
    let membership = create_test_membership();
    let manager = GeometricMembershipManager::with_defaults(membership);

    // Create sparse embeddings (typical in real usage)
    let mut sparse1 = SparseVector::new(128);
    sparse1.set(0, 1.0);
    sparse1.set(100, 0.5);

    let mut sparse2 = SparseVector::new(128);
    sparse2.set(0, 0.9);
    sparse2.set(50, 0.3);

    manager.record_peer_embedding(&"node2".to_string(), sparse1.clone());
    manager.record_peer_embedding(&"node3".to_string(), sparse2.clone());

    // Query with similar sparse embedding
    let mut query = SparseVector::new(128);
    query.set(0, 1.0);
    query.set(100, 0.4);

    let ranked = manager.ranked_peers(&query);
    assert_eq!(ranked.len(), 2);

    // node2 should be more similar (shares indices 0 and 100)
    let node2_ranked = ranked.iter().find(|p| p.node_id == "node2").unwrap();
    let node3_ranked = ranked.iter().find(|p| p.node_id == "node3").unwrap();
    assert!(node2_ranked.similarity > node3_ranked.similarity);
}
