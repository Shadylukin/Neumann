//! Geometric membership management with embedding-based peer scoring.
//!
//! Extends MembershipManager with geometric routing capabilities:
//! - Track peer embeddings from received messages
//! - Score peers by geometric proximity + health
//! - Find geometrically nearest healthy peers

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;
use tensor_store::SparseVector;

use crate::{
    block::NodeId,
    membership::{ClusterView, MembershipManager, NodeHealth},
};

/// Configuration for geometric membership scoring.
#[derive(Debug, Clone)]
pub struct GeometricMembershipConfig {
    /// Minimum similarity to consider a peer "nearby" (0.0-1.0).
    pub nearby_threshold: f32,
    /// Weight for geometric distance vs health in peer scoring (0.0-1.0).
    /// Higher values favor geometric proximity over health.
    pub geometric_weight: f32,
}

impl Default for GeometricMembershipConfig {
    fn default() -> Self {
        Self {
            nearby_threshold: 0.5,
            geometric_weight: 0.3, // 30% geometric, 70% health
        }
    }
}

/// Peer with its similarity score for ranking.
#[derive(Debug, Clone)]
pub struct RankedPeer {
    /// Node identifier.
    pub node_id: NodeId,
    /// Combined score (geometric + health).
    pub score: f32,
    /// Geometric similarity component.
    pub similarity: f32,
    /// Whether the peer is healthy.
    pub is_healthy: bool,
}

/// Geometric membership manager that extends MembershipManager with embedding awareness.
pub struct GeometricMembershipManager {
    /// Underlying membership manager.
    inner: Arc<MembershipManager>,
    /// Cached peer embeddings (updated from received messages).
    peer_embeddings: RwLock<HashMap<NodeId, SparseVector>>,
    /// Local node's state embedding.
    local_embedding: RwLock<Option<SparseVector>>,
    /// Configuration.
    config: GeometricMembershipConfig,
}

impl GeometricMembershipManager {
    /// Create a new geometric membership manager wrapping an existing MembershipManager.
    pub fn new(inner: Arc<MembershipManager>, config: GeometricMembershipConfig) -> Self {
        Self {
            inner,
            peer_embeddings: RwLock::new(HashMap::new()),
            local_embedding: RwLock::new(None),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(inner: Arc<MembershipManager>) -> Self {
        Self::new(inner, GeometricMembershipConfig::default())
    }

    /// Get the underlying membership manager.
    pub fn inner(&self) -> &Arc<MembershipManager> {
        &self.inner
    }

    /// Get the current cluster view.
    pub fn view(&self) -> ClusterView {
        self.inner.view()
    }

    /// Update the local node's state embedding.
    pub fn update_local_embedding(&self, embedding: SparseVector) {
        *self.local_embedding.write() = Some(embedding);
    }

    /// Get the local node's state embedding.
    pub fn local_embedding(&self) -> Option<SparseVector> {
        self.local_embedding.read().clone()
    }

    /// Record a peer's embedding from a received message.
    pub fn record_peer_embedding(&self, peer: &NodeId, embedding: SparseVector) {
        self.peer_embeddings.write().insert(peer.clone(), embedding);
    }

    /// Get a peer's cached embedding.
    pub fn peer_embedding(&self, peer: &NodeId) -> Option<SparseVector> {
        self.peer_embeddings.read().get(peer).cloned()
    }

    /// Get all peers ranked by combined geometric proximity and health.
    ///
    /// Score formula: `score = (1 - geo_weight) * health_score + geo_weight * similarity`
    /// - health_score: 1.0 for healthy, 0.5 for degraded, 0.0 for failed/unknown
    /// - similarity: cosine similarity between query and peer embeddings
    pub fn ranked_peers(&self, query: &SparseVector) -> Vec<RankedPeer> {
        let view = self.view();
        let embeddings = self.peer_embeddings.read();
        let local_id = self.inner.local_id();

        let mut peers: Vec<RankedPeer> = view
            .nodes
            .iter()
            .filter(|status| &status.node_id != local_id)
            .map(|status| {
                let health_score = match status.health {
                    NodeHealth::Healthy => 1.0,
                    NodeHealth::Degraded => 0.5,
                    NodeHealth::Failed | NodeHealth::Unknown => 0.0,
                };

                let similarity = embeddings
                    .get(&status.node_id)
                    .map(|emb| query.cosine_similarity(emb))
                    .unwrap_or(0.0);

                let score = (1.0 - self.config.geometric_weight) * health_score
                    + self.config.geometric_weight * similarity;

                RankedPeer {
                    node_id: status.node_id.clone(),
                    score,
                    similarity,
                    is_healthy: status.health == NodeHealth::Healthy,
                }
            })
            .collect();

        // Sort by score descending (highest first)
        peers.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        peers
    }

    /// Get the geometrically nearest healthy peer.
    pub fn nearest_healthy_peer(&self, query: &SparseVector) -> Option<NodeId> {
        self.ranked_peers(query)
            .into_iter()
            .find(|p| p.is_healthy)
            .map(|p| p.node_id)
    }

    /// Get all nearby healthy peers (similarity > threshold).
    pub fn nearby_healthy_peers(&self, query: &SparseVector) -> Vec<NodeId> {
        self.ranked_peers(query)
            .into_iter()
            .filter(|p| p.is_healthy && p.similarity >= self.config.nearby_threshold)
            .map(|p| p.node_id)
            .collect()
    }

    /// Get the number of cached peer embeddings.
    pub fn cached_embedding_count(&self) -> usize {
        self.peer_embeddings.read().len()
    }

    /// Clear stale embeddings for nodes that are no longer in the cluster.
    pub fn prune_stale_embeddings(&self) {
        let view = self.view();
        let known_nodes: std::collections::HashSet<_> =
            view.nodes.iter().map(|n| &n.node_id).collect();

        self.peer_embeddings
            .write()
            .retain(|node_id, _| known_nodes.contains(node_id));
    }
}

impl std::fmt::Debug for GeometricMembershipManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeometricMembershipManager")
            .field("config", &self.config)
            .field("cached_embeddings", &self.peer_embeddings.read().len())
            .field(
                "has_local_embedding",
                &self.local_embedding.read().is_some(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        membership::{ClusterConfig, LocalNodeConfig},
        network::MemoryTransport,
    };

    fn create_test_manager() -> GeometricMembershipManager {
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

        let membership = Arc::new(MembershipManager::new(config, transport));
        GeometricMembershipManager::with_defaults(membership)
    }

    #[test]
    fn test_geometric_membership_creation() {
        let manager = create_test_manager();
        assert_eq!(manager.cached_embedding_count(), 0);
        assert!(manager.local_embedding().is_none());
    }

    #[test]
    fn test_update_local_embedding() {
        let manager = create_test_manager();
        let embedding = SparseVector::from_dense(&[1.0, 0.0, 0.0]);

        manager.update_local_embedding(embedding.clone());

        let retrieved = manager.local_embedding().unwrap();
        assert_eq!(retrieved.to_dense(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_record_peer_embedding() {
        let manager = create_test_manager();
        let embedding = SparseVector::from_dense(&[0.0, 1.0, 0.0]);

        manager.record_peer_embedding(&"node2".to_string(), embedding.clone());

        assert_eq!(manager.cached_embedding_count(), 1);
        let retrieved = manager.peer_embedding(&"node2".to_string()).unwrap();
        assert_eq!(retrieved.to_dense(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_ranked_peers() {
        let manager = create_test_manager();

        // Record embeddings for peers
        manager.record_peer_embedding(
            &"node2".to_string(),
            SparseVector::from_dense(&[0.9, 0.1, 0.0]),
        );
        manager.record_peer_embedding(
            &"node3".to_string(),
            SparseVector::from_dense(&[0.1, 0.9, 0.0]),
        );

        // Query with embedding similar to node2
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let ranked = manager.ranked_peers(&query);

        // Should return peers (health unknown, so primarily ranked by similarity)
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_prune_stale_embeddings() {
        let manager = create_test_manager();

        // Record embedding for a non-existent node
        manager.record_peer_embedding(
            &"stale_node".to_string(),
            SparseVector::from_dense(&[0.5, 0.5, 0.0]),
        );
        manager.record_peer_embedding(
            &"node2".to_string(),
            SparseVector::from_dense(&[0.5, 0.5, 0.0]),
        );

        assert_eq!(manager.cached_embedding_count(), 2);

        manager.prune_stale_embeddings();

        // Only node2 should remain (it's in the cluster config)
        assert_eq!(manager.cached_embedding_count(), 1);
        assert!(manager.peer_embedding(&"node2".to_string()).is_some());
        assert!(manager.peer_embedding(&"stale_node".to_string()).is_none());
    }

    #[test]
    fn test_geometric_membership_config_default() {
        let config = GeometricMembershipConfig::default();
        assert!((config.nearby_threshold - 0.5).abs() < 0.01);
        assert!((config.geometric_weight - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_ranked_peer_debug() {
        let peer = RankedPeer {
            node_id: "node1".to_string(),
            score: 0.8,
            similarity: 0.9,
            is_healthy: true,
        };
        let debug = format!("{:?}", peer);
        assert!(debug.contains("RankedPeer"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_geometric_membership_debug() {
        let manager = create_test_manager();
        let debug = format!("{:?}", manager);
        assert!(debug.contains("GeometricMembershipManager"));
    }

    #[test]
    fn test_inner_membership() {
        let manager = create_test_manager();
        let inner = manager.inner();
        assert_eq!(inner.local_id(), "node1");
    }

    #[test]
    fn test_view() {
        let manager = create_test_manager();
        let view = manager.view();
        // View should have local node + peer nodes
        assert_eq!(view.total_count(), 3);
    }

    #[test]
    fn test_nearest_healthy_peer_no_healthy() {
        let manager = create_test_manager();
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);

        // No peers are healthy yet (unknown state), should return None
        let nearest = manager.nearest_healthy_peer(&query);
        assert!(nearest.is_none());
    }

    #[test]
    fn test_nearby_healthy_peers_empty() {
        let manager = create_test_manager();
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);

        // No healthy peers with high similarity
        let nearby = manager.nearby_healthy_peers(&query);
        assert!(nearby.is_empty());
    }

    #[test]
    fn test_config_custom() {
        let config = GeometricMembershipConfig {
            nearby_threshold: 0.8,
            geometric_weight: 0.5,
        };
        assert!((config.nearby_threshold - 0.8).abs() < 0.01);
        assert!((config.geometric_weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ranked_peers_with_multiple_embeddings() {
        let manager = create_test_manager();

        // Record embeddings for all peers
        manager.record_peer_embedding(
            &"node2".to_string(),
            SparseVector::from_dense(&[1.0, 0.0, 0.0]),
        );
        manager.record_peer_embedding(
            &"node3".to_string(),
            SparseVector::from_dense(&[0.0, 1.0, 0.0]),
        );

        // Query with embedding identical to node2
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let ranked = manager.ranked_peers(&query);

        assert_eq!(ranked.len(), 2);
        // node2 should have higher similarity
        let node2_rank = ranked.iter().find(|p| p.node_id == "node2");
        let node3_rank = ranked.iter().find(|p| p.node_id == "node3");
        assert!(node2_rank.is_some());
        assert!(node3_rank.is_some());
        assert!(node2_rank.unwrap().similarity > node3_rank.unwrap().similarity);
    }

    #[test]
    fn test_ranked_peer_clone() {
        let peer = RankedPeer {
            node_id: "node1".to_string(),
            score: 0.8,
            similarity: 0.9,
            is_healthy: true,
        };
        let cloned = peer.clone();
        assert_eq!(cloned.node_id, peer.node_id);
        assert!((cloned.score - peer.score).abs() < 0.001);
    }

    #[test]
    fn test_config_clone() {
        let config = GeometricMembershipConfig::default();
        let cloned = config.clone();
        assert!((cloned.nearby_threshold - config.nearby_threshold).abs() < 0.001);
    }
}
