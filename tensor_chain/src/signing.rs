//! Cryptographic signing and identity binding for tensor-chain.
//!
//! Provides:
//! - Ed25519 key pair management
//! - NodeId derivation from public key (identity binding)
//! - Stable embedding derivation from public key (geometric approach)

// ZeroizeOnDrop derive macro generates code that triggers this warning
#![allow(unused_assignments)]

use blake2::{digest::consts::U16, digest::consts::U64, Blake2b, Digest};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use tensor_store::SparseVector;
use zeroize::ZeroizeOnDrop;

use crate::{ChainError, NodeId, Result};

/// Domain separator for NodeId derivation.
const NODE_ID_DOMAIN: &[u8] = b"neumann_node_id_v1";

/// Domain separator for embedding derivation.
const EMBEDDING_DOMAIN: &[u8] = b"neumann_node_embedding_v1";

/// Signed identity with private key (zeroized on drop).
#[derive(ZeroizeOnDrop)]
pub struct Identity {
    /// Ed25519 signing key (private).
    #[zeroize(skip)] // ed25519_dalek handles zeroization internally
    signing_key: SigningKey,
}

impl Identity {
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let signing_key = SigningKey::from_bytes(bytes);
        Ok(Self { signing_key })
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    pub fn verifying_key(&self) -> PublicIdentity {
        PublicIdentity {
            verifying_key: self.signing_key.verifying_key(),
        }
    }

    pub fn node_id(&self) -> NodeId {
        self.verifying_key().to_node_id()
    }

    pub fn to_embedding(&self) -> SparseVector {
        self.verifying_key().to_embedding()
    }

    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let sig = self.signing_key.sign(message);
        sig.to_bytes().to_vec()
    }

    /// Sign a message and return a SignedMessage envelope with replay protection.
    pub fn sign_message(&self, payload: &[u8], sequence: u64) -> SignedMessage {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Sign over: sender || sequence || timestamp || payload
        let mut to_sign = Vec::new();
        to_sign.extend(self.node_id().as_bytes());
        to_sign.extend(&sequence.to_le_bytes());
        to_sign.extend(&timestamp_ms.to_le_bytes());
        to_sign.extend(payload);

        let signature = self.sign(&to_sign);

        SignedMessage {
            sender: self.node_id(),
            public_key: self.public_key_bytes(),
            payload: payload.to_vec(),
            signature,
            sequence,
            timestamp_ms,
        }
    }
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't expose private key in debug output
        f.debug_struct("Identity")
            .field("node_id", &self.node_id())
            .finish()
    }
}

/// Public identity (verifying key only, no private key).
#[derive(Clone)]
pub struct PublicIdentity {
    verifying_key: VerifyingKey,
}

impl PublicIdentity {
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let verifying_key = VerifyingKey::from_bytes(bytes)
            .map_err(|e| ChainError::CryptoError(format!("Invalid public key: {e}")))?;
        Ok(Self { verifying_key })
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Uses BLAKE2b-128 for a compact but collision-resistant ID.
    pub fn to_node_id(&self) -> NodeId {
        let mut hasher = Blake2b::<U16>::new();
        hasher.update(NODE_ID_DOMAIN);
        hasher.update(self.verifying_key.as_bytes());
        let hash = hasher.finalize();
        hex::encode(hash)
    }

    /// Uses BLAKE2b-512 to generate 16 f32 coordinates in [-1, 1].
    pub fn to_embedding(&self) -> SparseVector {
        let mut hasher = Blake2b::<U64>::new();
        hasher.update(EMBEDDING_DOMAIN);
        hasher.update(self.verifying_key.as_bytes());
        let hash = hasher.finalize();

        // Convert 64 bytes to 16 f32 coordinates normalized to [-1, 1]
        let coords: Vec<f32> = hash
            .chunks(4)
            .map(|c| {
                let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                (bits as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect();

        SparseVector::from_dense(&coords)
    }

    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<()> {
        if signature.len() != 64 {
            return Err(ChainError::CryptoError(format!(
                "Invalid signature length: expected 64, got {}",
                signature.len()
            )));
        }

        let sig_bytes: &[u8; 64] = signature
            .try_into()
            .map_err(|_| ChainError::CryptoError("signature conversion failed".to_string()))?;
        let sig = Signature::from_bytes(sig_bytes);

        self.verifying_key
            .verify(message, &sig)
            .map_err(|e| ChainError::CryptoError(format!("Signature verification failed: {e}")))
    }
}

impl std::fmt::Debug for PublicIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicIdentity")
            .field("node_id", &self.to_node_id())
            .finish()
    }
}

/// Default maximum age for messages (5 minutes).
const DEFAULT_MAX_AGE_MS: u64 = 5 * 60 * 1000;

/// Tracks sequence numbers per sender for replay attack detection.
pub struct SequenceTracker {
    /// Last seen sequence number per sender.
    sequences: dashmap::DashMap<NodeId, u64>,
    /// Maximum message age in milliseconds.
    max_age_ms: u64,
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceTracker {
    /// Create a new sequence tracker with default max age (5 minutes).
    pub fn new() -> Self {
        Self {
            sequences: dashmap::DashMap::new(),
            max_age_ms: DEFAULT_MAX_AGE_MS,
        }
    }

    pub fn with_max_age_ms(max_age_ms: u64) -> Self {
        Self {
            sequences: dashmap::DashMap::new(),
            max_age_ms,
        }
    }

    /// Check if a message is valid (not a replay) and record its sequence.
    /// Returns Ok(()) if valid, Err if replay detected or timestamp too old.
    pub fn check_and_record(
        &self,
        sender: &NodeId,
        sequence: u64,
        timestamp_ms: u64,
    ) -> Result<()> {
        // Check timestamp freshness
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if timestamp_ms > now_ms + 60_000 {
            // Message is from the future (allow 1 minute clock skew)
            return Err(ChainError::CryptoError(
                "message timestamp is in the future".into(),
            ));
        }

        if now_ms > timestamp_ms + self.max_age_ms {
            return Err(ChainError::CryptoError(format!(
                "message too old: {} ms",
                now_ms - timestamp_ms
            )));
        }

        // Check and update sequence number
        let mut entry = self.sequences.entry(sender.clone()).or_insert(0);
        if sequence <= *entry {
            return Err(ChainError::CryptoError(format!(
                "replay detected: sequence {} <= last seen {}",
                sequence, *entry
            )));
        }

        *entry = sequence;
        Ok(())
    }

    pub fn last_sequence(&self, sender: &NodeId) -> Option<u64> {
        self.sequences.get(sender).map(|v| *v)
    }

    /// Clear all tracked sequences (for testing or resets).
    pub fn clear(&self) {
        self.sequences.clear();
    }
}

/// A signed message envelope with identity binding and replay protection.
#[derive(Debug, Clone)]
pub struct SignedMessage {
    /// Sender's NodeId (derived from public key).
    pub sender: NodeId,
    /// Sender's public key.
    pub public_key: [u8; 32],
    /// The message payload.
    pub payload: Vec<u8>,
    /// Ed25519 signature over the payload (including sequence and timestamp).
    pub signature: Vec<u8>,
    /// Monotonically increasing sequence number for replay protection.
    pub sequence: u64,
    /// Unix timestamp in milliseconds when the message was created.
    pub timestamp_ms: u64,
}

/// Registry of known validators with their public keys.
/// Used for verifying block signatures and validator messages.
pub struct ValidatorRegistry {
    validators: dashmap::DashMap<NodeId, PublicIdentity>,
}

impl Default for ValidatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidatorRegistry {
    pub fn new() -> Self {
        Self {
            validators: dashmap::DashMap::new(),
        }
    }

    pub fn register(&self, identity: &Identity) {
        let node_id = identity.node_id();
        let public = identity.verifying_key();
        self.validators.insert(node_id, public);
    }

    pub fn register_public_key(&self, public_key: &[u8; 32]) -> Result<NodeId> {
        let public = PublicIdentity::from_bytes(public_key)?;
        let node_id = public.to_node_id();
        self.validators.insert(node_id.clone(), public);
        Ok(node_id)
    }

    pub fn get(&self, node_id: &str) -> Option<PublicIdentity> {
        self.validators.get(node_id).map(|v| v.clone())
    }

    pub fn contains(&self, node_id: &str) -> bool {
        self.validators.contains_key(node_id)
    }

    pub fn remove(&self, node_id: &str) -> Option<PublicIdentity> {
        self.validators.remove(node_id).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.validators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }

    pub fn node_ids(&self) -> Vec<NodeId> {
        self.validators.iter().map(|v| v.key().clone()).collect()
    }
}

impl SignedMessage {
    /// Verify the signature and check identity binding.
    /// Returns the payload if valid.
    /// Note: Does not check replay protection - use verify_with_tracker for that.
    pub fn verify(&self) -> Result<&[u8]> {
        // 1. Reconstruct public identity
        let identity = PublicIdentity::from_bytes(&self.public_key)?;

        // 2. Verify NodeId matches public key (identity binding)
        let expected_node_id = identity.to_node_id();
        if self.sender != expected_node_id {
            return Err(ChainError::CryptoError(format!(
                "NodeId mismatch: claimed {} but key derives {}",
                self.sender, expected_node_id
            )));
        }

        // 3. Reconstruct signed data: sender || sequence || timestamp || payload
        let mut to_verify = Vec::new();
        to_verify.extend(self.sender.as_bytes());
        to_verify.extend(&self.sequence.to_le_bytes());
        to_verify.extend(&self.timestamp_ms.to_le_bytes());
        to_verify.extend(&self.payload);

        // 4. Verify signature
        identity.verify(&to_verify, &self.signature)?;

        Ok(&self.payload)
    }

    /// Verify the signature, check identity binding, and check replay protection.
    /// Returns the payload if valid and not a replay.
    pub fn verify_with_tracker(&self, tracker: &SequenceTracker) -> Result<&[u8]> {
        // First verify the signature
        self.verify()?;

        // Check replay protection
        tracker.check_and_record(&self.sender, self.sequence, self.timestamp_ms)?;

        Ok(&self.payload)
    }

    pub fn sender_embedding(&self) -> Result<SparseVector> {
        let identity = PublicIdentity::from_bytes(&self.public_key)?;

        // Verify identity binding
        let expected_node_id = identity.to_node_id();
        if self.sender != expected_node_id {
            return Err(ChainError::CryptoError(
                "NodeId does not match public key".into(),
            ));
        }

        Ok(identity.to_embedding())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_generation() {
        let identity = Identity::generate();
        let node_id = identity.node_id();

        // NodeId should be 32 hex characters (16 bytes)
        assert_eq!(node_id.len(), 32);
        assert!(node_id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_node_id_deterministic() {
        let identity = Identity::generate();
        let id1 = identity.node_id();
        let id2 = identity.node_id();

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_different_identities_different_node_ids() {
        let id1 = Identity::generate();
        let id2 = Identity::generate();

        assert_ne!(id1.node_id(), id2.node_id());
    }

    #[test]
    fn test_embedding_generation() {
        let identity = Identity::generate();
        let embedding = identity.to_embedding();

        // Should have 16 dimensions
        assert_eq!(embedding.dimension(), 16);

        // Values should be in [-1, 1] range
        for &val in embedding.values() {
            assert!(val >= -1.0 && val <= 1.0, "Value out of range: {}", val);
        }
    }

    #[test]
    fn test_embedding_deterministic() {
        let identity = Identity::generate();
        let emb1 = identity.to_embedding();
        let emb2 = identity.to_embedding();

        assert_eq!(emb1.to_dense(), emb2.to_dense());
    }

    #[test]
    fn test_sign_and_verify() {
        let identity = Identity::generate();
        let message = b"test message";

        let signature = identity.sign(message);
        let public = identity.verifying_key();

        assert!(public.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_wrong_message_fails_verification() {
        let identity = Identity::generate();
        let message = b"test message";
        let wrong_message = b"wrong message";

        let signature = identity.sign(message);
        let public = identity.verifying_key();

        assert!(public.verify(wrong_message, &signature).is_err());
    }

    #[test]
    fn test_signed_message_verification() {
        let identity = Identity::generate();
        let payload = b"important data";

        let signed = identity.sign_message(payload, 1);

        // Verification should succeed
        let verified_payload = signed.verify().unwrap();
        assert_eq!(verified_payload, payload);
    }

    #[test]
    fn test_signed_message_identity_binding() {
        let identity = Identity::generate();
        let payload = b"data";

        let mut signed = identity.sign_message(payload, 1);

        // Tamper with sender (claim different identity)
        signed.sender = "0000000000000000000000000000000".to_string();

        // Verification should fail due to identity mismatch
        assert!(signed.verify().is_err());
    }

    #[test]
    fn test_public_identity_from_bytes() {
        let identity = Identity::generate();
        let bytes = identity.public_key_bytes();

        let public = PublicIdentity::from_bytes(&bytes).unwrap();

        assert_eq!(public.to_node_id(), identity.node_id());
    }

    #[test]
    fn test_identity_from_bytes() {
        let original = Identity::generate();
        let signing_bytes: [u8; 32] = original.signing_key.to_bytes();

        let restored = Identity::from_bytes(&signing_bytes).unwrap();

        assert_eq!(original.node_id(), restored.node_id());
    }

    #[test]
    fn test_sender_embedding() {
        let identity = Identity::generate();
        let signed = identity.sign_message(b"data", 1);

        let embedding = signed.sender_embedding().unwrap();

        assert_eq!(embedding.to_dense(), identity.to_embedding().to_dense());
    }

    #[test]
    fn test_sequence_tracker_accepts_valid() {
        let tracker = SequenceTracker::new();
        let sender = "test_sender".to_string();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // First message should succeed
        tracker.check_and_record(&sender, 1, now_ms).unwrap();

        // Higher sequence should succeed
        tracker.check_and_record(&sender, 2, now_ms).unwrap();

        // Check last sequence
        assert_eq!(tracker.last_sequence(&sender), Some(2));
    }

    #[test]
    fn test_sequence_tracker_rejects_replay() {
        let tracker = SequenceTracker::new();
        let sender = "test_sender".to_string();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // First message succeeds
        tracker.check_and_record(&sender, 5, now_ms).unwrap();

        // Replay (same sequence) should fail
        let result = tracker.check_and_record(&sender, 5, now_ms);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("replay detected"));

        // Lower sequence should fail
        let result = tracker.check_and_record(&sender, 3, now_ms);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_tracker_rejects_old_messages() {
        let tracker = SequenceTracker::with_max_age_ms(1000); // 1 second max age
        let sender = "test_sender".to_string();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Message from 2 seconds ago should fail
        let old_timestamp = now_ms - 2000;
        let result = tracker.check_and_record(&sender, 1, old_timestamp);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too old"));
    }

    #[test]
    fn test_signed_message_with_replay_protection() {
        let identity = Identity::generate();
        let tracker = SequenceTracker::new();

        // First message should verify
        let msg1 = identity.sign_message(b"data1", 1);
        msg1.verify_with_tracker(&tracker).unwrap();

        // Second message with higher sequence should verify
        let msg2 = identity.sign_message(b"data2", 2);
        msg2.verify_with_tracker(&tracker).unwrap();

        // Replay of first message should fail
        let result = msg1.verify_with_tracker(&tracker);
        assert!(result.is_err());
    }

    // === ValidatorRegistry tests ===

    #[test]
    fn test_validator_registry_new() {
        let registry = ValidatorRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_validator_registry_register() {
        let registry = ValidatorRegistry::new();
        let identity = Identity::generate();
        let node_id = identity.node_id();

        registry.register(&identity);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(&node_id));
    }

    #[test]
    fn test_validator_registry_get() {
        let registry = ValidatorRegistry::new();
        let identity = Identity::generate();
        let node_id = identity.node_id();

        registry.register(&identity);

        let public = registry.get(&node_id).unwrap();
        assert_eq!(public.to_node_id(), node_id);
    }

    #[test]
    fn test_validator_registry_get_nonexistent() {
        let registry = ValidatorRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_validator_registry_register_public_key() {
        let registry = ValidatorRegistry::new();
        let identity = Identity::generate();
        let pub_key = identity.public_key_bytes();

        let node_id = registry.register_public_key(&pub_key).unwrap();
        assert_eq!(node_id, identity.node_id());
        assert!(registry.contains(&node_id));
    }

    #[test]
    fn test_validator_registry_remove() {
        let registry = ValidatorRegistry::new();
        let identity = Identity::generate();
        let node_id = identity.node_id();

        registry.register(&identity);
        assert!(registry.contains(&node_id));

        let removed = registry.remove(&node_id);
        assert!(removed.is_some());
        assert!(!registry.contains(&node_id));
    }

    #[test]
    fn test_validator_registry_node_ids() {
        let registry = ValidatorRegistry::new();
        let id1 = Identity::generate();
        let id2 = Identity::generate();

        registry.register(&id1);
        registry.register(&id2);

        let node_ids = registry.node_ids();
        assert_eq!(node_ids.len(), 2);
        assert!(node_ids.contains(&id1.node_id()));
        assert!(node_ids.contains(&id2.node_id()));
    }

    #[test]
    fn test_validator_registry_verify_signature() {
        let registry = ValidatorRegistry::new();
        let identity = Identity::generate();
        let node_id = identity.node_id();

        registry.register(&identity);

        let message = b"test message";
        let signature = identity.sign(message);

        let public = registry.get(&node_id).unwrap();
        assert!(public.verify(message, &signature).is_ok());
    }
}
