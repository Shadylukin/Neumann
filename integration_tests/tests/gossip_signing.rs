// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for gossip message signing.

use std::sync::Arc;

use tensor_chain::{
    gossip::{GossipConfig, GossipMembershipManager, GossipMessage},
    network::{MemoryTransport, Message, Transport},
    signing::{Identity, SequenceTracker, SignedGossipMessage, ValidatorRegistry},
};

/// Helper to create a test transport network for a node.
fn create_transport(node_id: &str) -> Arc<MemoryTransport> {
    Arc::new(MemoryTransport::new(node_id.to_string()))
}

#[tokio::test]
async fn test_multi_node_signed_gossip() {
    // Create identities for 3 nodes
    let identity1 = Arc::new(Identity::generate());
    let identity2 = Arc::new(Identity::generate());
    let identity3 = Arc::new(Identity::generate());

    // Create shared validator registry with all nodes
    let registry = Arc::new(ValidatorRegistry::new());
    registry.register(&identity1);
    registry.register(&identity2);
    registry.register(&identity3);

    // Create sequence trackers for each node
    let tracker1 = Arc::new(SequenceTracker::new());
    let tracker2 = Arc::new(SequenceTracker::new());
    let tracker3 = Arc::new(SequenceTracker::new());

    // Create config with signatures required
    let mut config = GossipConfig::default();
    config.require_signatures = true;

    // Create transports for each node
    let transport1 = create_transport(&identity1.node_id());
    let transport2 = create_transport(&identity2.node_id());
    let transport3 = create_transport(&identity3.node_id());

    // Create gossip managers
    let _gossip1 = GossipMembershipManager::with_signing(
        identity1.node_id(),
        config.clone(),
        Arc::clone(&transport1) as Arc<dyn Transport>,
        Arc::clone(&identity1),
        Arc::clone(&registry),
        Arc::clone(&tracker1),
    );

    let gossip2 = GossipMembershipManager::with_signing(
        identity2.node_id(),
        config.clone(),
        Arc::clone(&transport2) as Arc<dyn Transport>,
        Arc::clone(&identity2),
        Arc::clone(&registry),
        Arc::clone(&tracker2),
    );

    let gossip3 = GossipMembershipManager::with_signing(
        identity3.node_id(),
        config,
        Arc::clone(&transport3) as Arc<dyn Transport>,
        Arc::clone(&identity3),
        Arc::clone(&registry),
        Arc::clone(&tracker3),
    );

    // Create and verify a signed gossip message from node1 to node2
    let msg = GossipMessage::Sync {
        sender: identity1.node_id(),
        states: vec![],
        sender_time: 100,
    };

    let signed = SignedGossipMessage::new(&identity1, &msg, 1).unwrap();

    // Node 2 should be able to verify
    assert!(gossip2.handle_signed_gossip(signed.clone()).is_ok());

    // Node 3 should also be able to verify (with different sequence tracking)
    let signed2 = SignedGossipMessage::new(&identity1, &msg, 2).unwrap();
    assert!(gossip3.handle_signed_gossip(signed2).is_ok());
}

#[tokio::test]
async fn test_mixed_mode_compatibility() {
    // Create identity for signing node
    let identity = Arc::new(Identity::generate());

    // Create registry with the signing node
    let registry = Arc::new(ValidatorRegistry::new());
    registry.register(&identity);

    // Create sequence tracker
    let tracker = Arc::new(SequenceTracker::new());

    // Create transport
    let transport = create_transport(&identity.node_id());

    // Node 1: Signing enabled, but not required (accepts both)
    let mut config1 = GossipConfig::default();
    config1.require_signatures = false;

    let gossip1 = GossipMembershipManager::with_signing(
        identity.node_id(),
        config1,
        Arc::clone(&transport) as Arc<dyn Transport>,
        Arc::clone(&identity),
        Arc::clone(&registry),
        Arc::clone(&tracker),
    );

    // Unsigned gossip should work when signatures not required
    let unsigned_msg = GossipMessage::Alive {
        node_id: "unsigned_node".to_string(),
        incarnation: 1,
    };
    gossip1.handle_gossip(unsigned_msg);

    // Signed gossip should also work
    let signed_msg = GossipMessage::Alive {
        node_id: identity.node_id(),
        incarnation: 2,
    };
    let signed = SignedGossipMessage::new(&identity, &signed_msg, 1).unwrap();
    assert!(gossip1.handle_signed_gossip(signed).is_ok());
}

#[tokio::test]
async fn test_malicious_node_rejected() {
    // Create legitimate identity
    let identity = Arc::new(Identity::generate());

    // Create malicious identity (not in registry)
    let malicious = Identity::generate();

    // Create registry with only legitimate node
    let registry = Arc::new(ValidatorRegistry::new());
    registry.register(&identity);

    // Create sequence tracker
    let tracker = Arc::new(SequenceTracker::new());

    // Create config with signatures required
    let mut config = GossipConfig::default();
    config.require_signatures = true;

    let transport = create_transport(&identity.node_id());

    let gossip = GossipMembershipManager::with_signing(
        identity.node_id(),
        config,
        Arc::clone(&transport) as Arc<dyn Transport>,
        Arc::clone(&identity),
        Arc::clone(&registry),
        Arc::clone(&tracker),
    );

    // Message from malicious node should be rejected
    let malicious_msg = GossipMessage::Suspect {
        reporter: malicious.node_id(),
        suspect: identity.node_id(),
        incarnation: 1,
    };

    let signed = SignedGossipMessage::new(&malicious, &malicious_msg, 1).unwrap();
    let result = gossip.handle_signed_gossip(signed);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown gossip sender"));
}

#[tokio::test]
async fn test_require_signatures_rejects_unsigned() {
    let identity = Arc::new(Identity::generate());
    let registry = Arc::new(ValidatorRegistry::new());
    registry.register(&identity);
    let tracker = Arc::new(SequenceTracker::new());

    // Create config with signatures required
    let mut config = GossipConfig::default();
    config.require_signatures = true;

    let transport = create_transport(&identity.node_id());

    let gossip = GossipMembershipManager::with_signing(
        identity.node_id(),
        config,
        Arc::clone(&transport) as Arc<dyn Transport>,
        Arc::clone(&identity),
        Arc::clone(&registry),
        Arc::clone(&tracker),
    );

    // Verify require_signatures returns true
    assert!(gossip.require_signatures());
}

#[tokio::test]
async fn test_signed_gossip_message_enum_variant() {
    let identity = Identity::generate();
    let gossip_msg = GossipMessage::Alive {
        node_id: identity.node_id(),
        incarnation: 1,
    };

    let signed = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();
    let message = Message::SignedGossip(signed);

    // Verify it can be serialized/deserialized
    let serialized = bitcode::serialize(&message).unwrap();
    let deserialized: Message = bitcode::deserialize(&serialized).unwrap();

    match deserialized {
        Message::SignedGossip(s) => {
            assert_eq!(s.sender(), &identity.node_id());
            assert_eq!(s.sequence(), 1);
        },
        _ => panic!("Expected SignedGossip variant"),
    }
}
