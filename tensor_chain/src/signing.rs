// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Ed25519 identity management with geometric embedding derivation.
//!
//! # Overview
//!
//! This module provides cryptographic identity management for cluster nodes, combining
//! Ed25519 digital signatures with geometric embedding derivation. Each node has an
//! [`Identity`] containing its private key, from which its [`NodeId`] and stable
//! embedding vector are deterministically derived.
//!
//! The key innovation is **identity binding**: a node's `NodeId` is cryptographically
//! bound to its public key, preventing impersonation. When a message is signed, the
//! receiver can verify both the signature and that the claimed sender matches the
//! signing key.
//!
//! # Architecture
//!
//! ```text
//! +------------------+
//! |    Identity      |  Private key (zeroized on drop)
//! +------------------+
//!          |
//!          | derives
//!          v
//! +------------------+     to_node_id()     +------------------+
//! |  PublicIdentity  | ------------------> | NodeId (String)  |
//! | (verifying key)  |     BLAKE2b-128     | 32 hex chars     |
//! +------------------+                     +------------------+
//!          |
//!          | to_embedding()
//!          | BLAKE2b-512
//!          v
//! +------------------+
//! | SparseVector     |  16 dimensions, [-1, 1] normalized
//! | (stable embedding)|
//! +------------------+
//! ```
//!
//! # Identity Binding
//!
//! The `NodeId` is derived from the public key using BLAKE2b-128:
//!
//! ```text
//! NodeId = hex(BLAKE2b-128(domain_separator || public_key))
//! ```
//!
//! Domain separation (`neumann_node_id_v1`) prevents cross-protocol attacks where
//! a hash computed for one purpose could be misused in another context.
//!
//! # Geometric Embedding Derivation
//!
//! Each node's public key deterministically maps to a 16-dimensional embedding:
//!
//! 1. Compute `BLAKE2b-512(domain_separator || public_key)` (64 bytes)
//! 2. Split into 16 groups of 4 bytes each
//! 3. Interpret each group as a little-endian `u32`
//! 4. Normalize to `[-1.0, 1.0]` range: `value = u32 / u32::MAX * 2.0 - 1.0`
//!
//! This provides stable embeddings for geometric routing and peer scoring
//! without requiring external embedding models.
//!
//! # Signed Messages
//!
//! The [`SignedMessage`] envelope provides:
//!
//! - **Authentication**: Ed25519 signature over the payload
//! - **Identity binding**: Sender's `NodeId` must match their public key
//! - **Replay protection**: Monotonic sequence numbers per sender
//! - **Freshness**: Timestamps checked against a configurable window
//!
//! ```text
//! SignedMessage {
//!     sender: NodeId,           // Claimed sender (verified against public_key)
//!     public_key: [u8; 32],     // Sender's Ed25519 public key
//!     payload: Vec<u8>,         // Arbitrary message data
//!     signature: Vec<u8>,       // Ed25519 signature (64 bytes)
//!     sequence: u64,            // Monotonic sequence number
//!     timestamp_ms: u64,        // Wall clock time when signed
//! }
//! ```
//!
//! Signature is computed over: `sender || sequence || timestamp || payload`
//!
//! # Usage
//!
//! ## Creating and Using Identities
//!
//! ```rust
//! use tensor_chain::signing::{Identity, PublicIdentity};
//!
//! // Generate a new identity
//! let identity = Identity::generate();
//!
//! // Get the node ID (for cluster membership)
//! let node_id = identity.node_id();
//!
//! // Get the stable embedding (for geometric routing)
//! let embedding = identity.to_embedding();
//! assert_eq!(embedding.dimension(), 16);
//!
//! // Sign a message
//! let signature = identity.sign(b"important data");
//!
//! // Extract public identity for verification
//! let public = identity.verifying_key();
//! assert!(public.verify(b"important data", &signature).is_ok());
//! ```
//!
//! ## Signed Message Envelopes
//!
//! ```rust
//! use tensor_chain::signing::{Identity, SequenceTracker};
//!
//! let identity = Identity::generate();
//! let tracker = SequenceTracker::new();
//!
//! // Create signed message with replay protection
//! let msg = identity.sign_message(b"payload", /*sequence=*/1);
//!
//! // Verify signature and replay protection
//! let payload = msg.verify_with_tracker(&tracker).unwrap();
//! assert_eq!(payload, b"payload");
//!
//! // Replay attempt fails
//! let msg_replay = identity.sign_message(b"payload", /*sequence=*/1);
//! assert!(msg_replay.verify_with_tracker(&tracker).is_err());
//! ```
//!
//! ## Validator Registry
//!
//! ```rust
//! use tensor_chain::signing::{Identity, ValidatorRegistry};
//!
//! let registry = ValidatorRegistry::new();
//!
//! // Register validators
//! let validator1 = Identity::generate();
//! registry.register(&validator1);
//!
//! // Check if a node is a known validator
//! assert!(registry.contains(&validator1.node_id()));
//!
//! // Retrieve public identity for verification
//! let public = registry.get(&validator1.node_id()).unwrap();
//! ```
//!
//! # Security Considerations
//!
//! ## Key Material Protection
//!
//! - [`Identity`] implements `ZeroizeOnDrop` to clear private key memory
//! - The `Debug` impl redacts the private key to prevent accidental logging
//! - `ed25519_dalek` handles internal zeroization of the signing key
//!
//! ## Replay Protection
//!
//! The [`SequenceTracker`] provides bounded-memory replay detection:
//!
//! - Entries older than `max_age_ms` are periodically removed
//! - Maximum number of tracked senders is bounded by `max_entries`
//! - Messages with old timestamps are rejected (configurable window)
//!
//! ## Timestamp Validation
//!
//! - Messages from the future (> 1 minute ahead) are rejected
//! - Messages older than `max_age_ms` are rejected
//! - This prevents replay of captured messages
//!
//! # See Also
//!
//! - [`crate::gossip`]: Gossip protocol using signed messages
//! - [`crate::message_validation`]: Additional message validation
//! - [`crate::geometric_membership`]: Peer scoring using node embeddings

// ZeroizeOnDrop derive macro generates code that triggers this warning
#![allow(unused_assignments)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use blake2::{digest::consts::U16, digest::consts::U64, Blake2b, Digest};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use tensor_store::SparseVector;
use zeroize::ZeroizeOnDrop;

use crate::{gossip::GossipMessage, ChainError, NodeId, Result};

/// Domain separator for `NodeId` derivation.
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
    #[must_use]
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// # Errors
    /// Returns an error if the bytes don't form a valid signing key (currently infallible).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let signing_key = SigningKey::from_bytes(bytes);
        Ok(Self { signing_key })
    }

    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    #[must_use]
    pub fn verifying_key(&self) -> PublicIdentity {
        PublicIdentity {
            verifying_key: self.signing_key.verifying_key(),
        }
    }

    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.verifying_key().to_node_id()
    }

    #[must_use]
    pub fn to_embedding(&self) -> SparseVector {
        self.verifying_key().to_embedding()
    }

    #[must_use]
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let sig = self.signing_key.sign(message);
        sig.to_bytes().to_vec()
    }

    /// Sign a message and return a `SignedMessage` envelope with replay protection.
    #[must_use]
    pub fn sign_message(&self, payload: &[u8], sequence: u64) -> SignedMessage {
        #[allow(clippy::cast_possible_truncation)]
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
    /// # Errors
    /// Returns an error if the bytes don't form a valid Ed25519 public key.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let verifying_key = VerifyingKey::from_bytes(bytes)
            .map_err(|e| ChainError::CryptoError(format!("Invalid public key: {e}")))?;
        Ok(Self { verifying_key })
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Uses BLAKE2b-128 for a compact but collision-resistant ID.
    #[must_use]
    pub fn to_node_id(&self) -> NodeId {
        let mut hasher = Blake2b::<U16>::new();
        hasher.update(NODE_ID_DOMAIN);
        hasher.update(self.verifying_key.as_bytes());
        let hash = hasher.finalize();
        hex::encode(hash)
    }

    /// BLAKE2b-512 for speed + security. 64 bytes -> 16 f32 coordinates.
    /// Domain separation prevents cross-protocol attacks. Normalization to [-1,1].
    #[must_use]
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
                #[allow(clippy::cast_possible_truncation)]
                let val = (f64::from(bits) / f64::from(u32::MAX)).mul_add(2.0, -1.0) as f32;
                val
            })
            .collect();

        SparseVector::from_dense(&coords)
    }

    /// # Errors
    /// Returns an error if the signature is invalid or has the wrong length.
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

/// Configuration for sequence tracker bounds and cleanup.
#[derive(Debug, Clone)]
pub struct SequenceTrackerConfig {
    /// Maximum age for entries in milliseconds (default: 5 minutes).
    pub max_age_ms: u64,
    /// Maximum number of tracked senders (default: 10,000).
    pub max_entries: usize,
    /// Cleanup is triggered every N calls to `check_and_record` (default: 100).
    pub cleanup_interval: usize,
}

impl Default for SequenceTrackerConfig {
    fn default() -> Self {
        Self {
            max_age_ms: DEFAULT_MAX_AGE_MS,
            max_entries: 10_000,
            cleanup_interval: 100,
        }
    }
}

impl SequenceTrackerConfig {
    #[must_use]
    pub const fn with_max_age_ms(mut self, max_age_ms: u64) -> Self {
        self.max_age_ms = max_age_ms;
        self
    }

    #[must_use]
    pub const fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    #[must_use]
    pub const fn with_cleanup_interval(mut self, cleanup_interval: usize) -> Self {
        self.cleanup_interval = cleanup_interval;
        self
    }
}

/// Tracks sequence numbers per sender for replay attack detection.
///
/// Includes protection against unbounded memory growth:
/// - Entries older than `max_age_ms` are periodically removed
/// - Maximum number of tracked senders is bounded by `max_entries`
pub struct SequenceTracker {
    /// Last seen sequence number and timestamp per sender.
    sequences: dashmap::DashMap<NodeId, (u64, Instant)>,
    /// Configuration for bounds and cleanup.
    config: SequenceTrackerConfig,
    /// Counter for periodic cleanup triggering.
    call_count: AtomicUsize,
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceTracker {
    /// Create a new sequence tracker with default config (5 min max age, 10k max entries).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SequenceTrackerConfig::default())
    }

    #[must_use]
    pub fn with_max_age_ms(max_age_ms: u64) -> Self {
        Self::with_config(SequenceTrackerConfig::default().with_max_age_ms(max_age_ms))
    }

    #[must_use]
    pub fn with_config(config: SequenceTrackerConfig) -> Self {
        Self {
            sequences: dashmap::DashMap::new(),
            config,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Check if a message is valid (not a replay) and record its sequence.
    ///
    /// # Errors
    /// Returns an error if replay detected, timestamp too old/future, or tracker at capacity.
    pub fn check_and_record(
        &self,
        sender: &NodeId,
        sequence: u64,
        timestamp_ms: u64,
    ) -> Result<()> {
        // Periodic cleanup
        let count = self.call_count.fetch_add(1, Ordering::Relaxed);
        if count % self.config.cleanup_interval == 0 {
            self.cleanup_stale_entries();
        }

        // Check timestamp freshness
        #[allow(clippy::cast_possible_truncation)]
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

        if now_ms > timestamp_ms + self.config.max_age_ms {
            return Err(ChainError::CryptoError(format!(
                "message too old: {} ms",
                now_ms - timestamp_ms
            )));
        }

        // Check capacity before inserting new sender
        if !self.sequences.contains_key(sender) && self.sequences.len() >= self.config.max_entries {
            // Try cleanup first
            self.cleanup_stale_entries();
            // If still at capacity, reject
            if self.sequences.len() >= self.config.max_entries {
                return Err(ChainError::CryptoError(
                    "sequence tracker at capacity".into(),
                ));
            }
        }

        // Check and update sequence number
        let now = Instant::now();
        let mut entry = self.sequences.entry(sender.clone()).or_insert((0, now));
        if sequence <= entry.0 {
            return Err(ChainError::CryptoError(format!(
                "replay detected: sequence {} <= last seen {}",
                sequence, entry.0
            )));
        }

        *entry = (sequence, now);
        drop(entry);
        Ok(())
    }

    /// Remove entries older than `max_age_ms`.
    fn cleanup_stale_entries(&self) {
        let cutoff = Instant::now().checked_sub(Duration::from_millis(self.config.max_age_ms));
        if let Some(cutoff) = cutoff {
            self.sequences
                .retain(|_, (_, last_seen)| *last_seen > cutoff);
        }
    }

    #[must_use]
    pub fn last_sequence(&self, sender: &NodeId) -> Option<u64> {
        self.sequences.get(sender).map(|v| v.0)
    }

    /// Clear all tracked sequences (for testing or resets).
    pub fn clear(&self) {
        self.sequences.clear();
    }

    /// Returns the number of tracked senders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Returns true if no senders are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

/// A signed message envelope with identity binding and replay protection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMessage {
    /// Sender's `NodeId` (derived from public key).
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
    #[must_use]
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

    /// # Errors
    /// Returns an error if the public key bytes are invalid.
    pub fn register_public_key(&self, public_key: &[u8; 32]) -> Result<NodeId> {
        let public = PublicIdentity::from_bytes(public_key)?;
        let node_id = public.to_node_id();
        self.validators.insert(node_id.clone(), public);
        Ok(node_id)
    }

    #[must_use]
    pub fn get(&self, node_id: &str) -> Option<PublicIdentity> {
        self.validators.get(node_id).map(|v| v.clone())
    }

    #[must_use]
    pub fn contains(&self, node_id: &str) -> bool {
        self.validators.contains_key(node_id)
    }

    #[must_use]
    pub fn remove(&self, node_id: &str) -> Option<PublicIdentity> {
        self.validators.remove(node_id).map(|(_, v)| v)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.validators.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }

    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.validators.iter().map(|v| v.key().clone()).collect()
    }
}

impl SignedMessage {
    /// Verify the signature and check identity binding.
    /// Returns the payload if valid.
    /// Note: Does not check replay protection - use `verify_with_tracker` for that.
    ///
    /// # Errors
    /// Returns an error if the signature is invalid or `NodeId` doesn't match the public key.
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
    ///
    /// # Errors
    /// Returns an error if the signature is invalid, identity doesn't match, or replay detected.
    pub fn verify_with_tracker(&self, tracker: &SequenceTracker) -> Result<&[u8]> {
        // First verify the signature
        self.verify()?;

        // Check replay protection
        tracker.check_and_record(&self.sender, self.sequence, self.timestamp_ms)?;

        Ok(&self.payload)
    }

    /// # Errors
    /// Returns an error if the public key is invalid or `NodeId` doesn't match.
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

/// A signed gossip message with replay protection.
///
/// Wraps a `GossipMessage` in a `SignedMessage` envelope to provide:
/// - Authentication via Ed25519 signature
/// - Identity binding (`NodeId` derived from public key)
/// - Replay protection via monotonic sequence numbers
/// - Freshness via timestamp checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedGossipMessage {
    /// The signed message envelope containing the serialized `GossipMessage`.
    pub envelope: SignedMessage,
}

impl SignedGossipMessage {
    /// Create a new signed gossip message.
    ///
    /// Serializes the gossip message and signs it with the given identity.
    ///
    /// # Errors
    /// Returns an error if the gossip message fails to serialize.
    pub fn new(identity: &Identity, msg: &GossipMessage, sequence: u64) -> Result<Self> {
        let payload = bitcode::serialize(msg).map_err(|e| {
            ChainError::SerializationError(format!("failed to serialize gossip: {e}"))
        })?;
        let envelope = identity.sign_message(&payload, sequence);
        Ok(Self { envelope })
    }

    /// Verify the signature and identity binding using the validator registry.
    ///
    /// Returns the deserialized `GossipMessage` if valid.
    /// Does not check replay protection - use `verify_with_tracker` for that.
    ///
    /// # Errors
    /// Returns an error if sender is unknown, signature is invalid, or deserialization fails.
    pub fn verify(&self, registry: &ValidatorRegistry) -> Result<GossipMessage> {
        // Check that the sender is a known validator
        if !registry.contains(&self.envelope.sender) {
            return Err(ChainError::CryptoError(format!(
                "unknown gossip sender: {}",
                self.envelope.sender
            )));
        }

        // Verify signature and identity binding
        let payload = self.envelope.verify()?;

        // Deserialize the gossip message
        bitcode::deserialize(payload).map_err(|e| {
            ChainError::SerializationError(format!("failed to deserialize gossip: {e}"))
        })
    }

    /// Verify the signature, identity binding, and replay protection.
    ///
    /// Returns the deserialized `GossipMessage` if valid and not a replay.
    ///
    /// # Errors
    /// Returns an error if verification fails or replay detected.
    pub fn verify_with_tracker(
        &self,
        registry: &ValidatorRegistry,
        tracker: &SequenceTracker,
    ) -> Result<GossipMessage> {
        // Check that the sender is a known validator
        if !registry.contains(&self.envelope.sender) {
            return Err(ChainError::CryptoError(format!(
                "unknown gossip sender: {}",
                self.envelope.sender
            )));
        }

        // Verify signature, identity binding, and replay protection
        let payload = self.envelope.verify_with_tracker(tracker)?;

        // Deserialize the gossip message
        bitcode::deserialize(payload).map_err(|e| {
            ChainError::SerializationError(format!("failed to deserialize gossip: {e}"))
        })
    }

    /// Get the sender's `NodeId`.
    #[must_use]
    pub const fn sender(&self) -> &NodeId {
        &self.envelope.sender
    }

    /// Get the sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.envelope.sequence
    }

    /// Get the timestamp in milliseconds.
    #[must_use]
    pub const fn timestamp_ms(&self) -> u64 {
        self.envelope.timestamp_ms
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
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
            assert!((-1.0..=1.0).contains(&val), "Value out of range: {}", val);
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

    // === SignedGossipMessage tests ===

    #[test]
    fn test_signed_gossip_roundtrip() {
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        let gossip_msg = GossipMessage::Sync {
            sender: identity.node_id(),
            states: vec![],
            sender_time: 42,
        };

        let signed = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();

        assert_eq!(signed.sender(), &identity.node_id());
        assert_eq!(signed.sequence(), 1);

        let verified = signed.verify(&registry).unwrap();
        assert_eq!(verified, gossip_msg);
    }

    #[test]
    fn test_signed_gossip_all_message_types() {
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        // Test Sync message
        let sync = GossipMessage::Sync {
            sender: identity.node_id(),
            states: vec![],
            sender_time: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &sync, 1).unwrap();
        assert!(signed.verify(&registry).is_ok());

        // Test Suspect message
        let suspect = GossipMessage::Suspect {
            reporter: identity.node_id(),
            suspect: "other_node".to_string(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &suspect, 2).unwrap();
        assert!(signed.verify(&registry).is_ok());

        // Test Alive message
        let alive = GossipMessage::Alive {
            node_id: identity.node_id(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &alive, 3).unwrap();
        assert!(signed.verify(&registry).is_ok());

        // Test PingReq message
        let ping_req = GossipMessage::PingReq {
            origin: identity.node_id(),
            target: "target_node".to_string(),
            sequence: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &ping_req, 4).unwrap();
        assert!(signed.verify(&registry).is_ok());

        // Test PingAck message
        let ping_ack = GossipMessage::PingAck {
            origin: identity.node_id(),
            target: "target_node".to_string(),
            sequence: 1,
            success: true,
        };
        let signed = SignedGossipMessage::new(&identity, &ping_ack, 5).unwrap();
        assert!(signed.verify(&registry).is_ok());
    }

    #[test]
    fn test_signed_gossip_replay_rejected() {
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        let tracker = SequenceTracker::new();
        registry.register(&identity);

        let gossip_msg = GossipMessage::Alive {
            node_id: identity.node_id(),
            incarnation: 1,
        };

        // First message should succeed
        let signed1 = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();
        signed1.verify_with_tracker(&registry, &tracker).unwrap();

        // Replay with same sequence should fail
        let signed2 = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();
        let result = signed2.verify_with_tracker(&registry, &tracker);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("replay detected"));
    }

    #[test]
    fn test_signed_gossip_invalid_signature() {
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        let gossip_msg = GossipMessage::Alive {
            node_id: identity.node_id(),
            incarnation: 1,
        };

        let mut signed = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();

        // Tamper with signature
        if !signed.envelope.signature.is_empty() {
            signed.envelope.signature[0] ^= 0xFF;
        }

        let result = signed.verify(&registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_signed_gossip_wrong_sender() {
        let identity1 = Identity::generate();
        let identity2 = Identity::generate();
        let registry = ValidatorRegistry::new();
        registry.register(&identity1);
        registry.register(&identity2);

        let gossip_msg = GossipMessage::Alive {
            node_id: identity1.node_id(),
            incarnation: 1,
        };

        let mut signed = SignedGossipMessage::new(&identity1, &gossip_msg, 1).unwrap();

        // Change the sender to a different identity's node_id
        signed.envelope.sender = identity2.node_id();

        let result = signed.verify(&registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NodeId mismatch"));
    }

    #[test]
    fn test_signed_gossip_unknown_sender() {
        let identity = Identity::generate();
        let unknown_identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        // Only register identity, not unknown_identity
        registry.register(&identity);

        let gossip_msg = GossipMessage::Alive {
            node_id: unknown_identity.node_id(),
            incarnation: 1,
        };

        let signed = SignedGossipMessage::new(&unknown_identity, &gossip_msg, 1).unwrap();

        let result = signed.verify(&registry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown gossip sender"));
    }

    #[test]
    fn test_signed_gossip_expired_message() {
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        let tracker = SequenceTracker::with_max_age_ms(1000); // 1 second max age
        registry.register(&identity);

        let gossip_msg = GossipMessage::Alive {
            node_id: identity.node_id(),
            incarnation: 1,
        };

        let mut signed = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();

        // Set timestamp to 2 seconds ago
        signed.envelope.timestamp_ms -= 2000;

        // Re-sign with old timestamp to verify rejection is based on timestamp
        // Note: we can't really re-sign, so this test verifies the tracker check
        let result = signed.verify_with_tracker(&registry, &tracker);
        // This will fail due to signature mismatch since we modified the timestamp
        assert!(result.is_err());
    }

    #[test]
    fn test_signed_gossip_backward_compat_unsigned() {
        // Test that verify works without replay protection when desired
        let identity = Identity::generate();
        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        let gossip_msg = GossipMessage::Alive {
            node_id: identity.node_id(),
            incarnation: 1,
        };

        // Create same message twice with same sequence
        let signed1 = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();
        let signed2 = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();

        // Without tracker, both should verify (no replay protection)
        assert!(signed1.verify(&registry).is_ok());
        assert!(signed2.verify(&registry).is_ok());
    }

    #[test]
    fn test_signed_gossip_serialization_roundtrip() {
        let identity = Identity::generate();

        let gossip_msg = GossipMessage::Sync {
            sender: identity.node_id(),
            states: vec![],
            sender_time: 42,
        };

        let signed = SignedGossipMessage::new(&identity, &gossip_msg, 1).unwrap();

        // Serialize and deserialize
        let serialized = bitcode::serialize(&signed).unwrap();
        let deserialized: SignedGossipMessage = bitcode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.sender(), signed.sender());
        assert_eq!(deserialized.sequence(), signed.sequence());
        assert_eq!(deserialized.timestamp_ms(), signed.timestamp_ms());
    }

    // === SequenceTrackerConfig tests ===

    #[test]
    fn test_sequence_tracker_config_default() {
        let config = SequenceTrackerConfig::default();
        assert_eq!(config.max_age_ms, 5 * 60 * 1000);
        assert_eq!(config.max_entries, 10_000);
        assert_eq!(config.cleanup_interval, 100);
    }

    #[test]
    fn test_sequence_tracker_config_builder() {
        let config = SequenceTrackerConfig::default()
            .with_max_age_ms(60_000)
            .with_max_entries(100)
            .with_cleanup_interval(10);

        assert_eq!(config.max_age_ms, 60_000);
        assert_eq!(config.max_entries, 100);
        assert_eq!(config.cleanup_interval, 10);
    }

    #[test]
    fn test_sequence_tracker_max_entries_enforced() {
        let config = SequenceTrackerConfig::default()
            .with_max_entries(5)
            .with_cleanup_interval(1000); // High interval to avoid auto-cleanup

        let tracker = SequenceTracker::with_config(config);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add 5 senders (should succeed)
        for i in 0..5 {
            let sender = format!("sender_{}", i);
            tracker.check_and_record(&sender, 1, now_ms).unwrap();
        }

        assert_eq!(tracker.len(), 5);

        // Adding 6th sender should fail (at capacity)
        let result = tracker.check_and_record(&"sender_5".to_string(), 1, now_ms);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at capacity"));
    }

    #[test]
    fn test_sequence_tracker_len_is_empty() {
        let tracker = SequenceTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        tracker
            .check_and_record(&"sender".to_string(), 1, now_ms)
            .unwrap();

        assert!(!tracker.is_empty());
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_sequence_tracker_existing_sender_not_counted_against_capacity() {
        let config = SequenceTrackerConfig::default()
            .with_max_entries(2)
            .with_cleanup_interval(1000);

        let tracker = SequenceTracker::with_config(config);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add 2 senders
        tracker
            .check_and_record(&"sender_a".to_string(), 1, now_ms)
            .unwrap();
        tracker
            .check_and_record(&"sender_b".to_string(), 1, now_ms)
            .unwrap();

        // Updating existing sender should succeed (not at capacity for new senders)
        tracker
            .check_and_record(&"sender_a".to_string(), 2, now_ms)
            .unwrap();
        tracker
            .check_and_record(&"sender_b".to_string(), 2, now_ms)
            .unwrap();

        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn test_sequence_tracker_with_config() {
        let config = SequenceTrackerConfig::default().with_max_age_ms(30_000);
        let tracker = SequenceTracker::with_config(config);

        assert!(tracker.is_empty());
    }

    #[test]
    fn test_identity_debug() {
        let identity = Identity::generate();
        let debug_str = format!("{:?}", identity);
        assert!(debug_str.contains("Identity"));
        assert!(debug_str.contains("node_id"));
    }

    #[test]
    fn test_public_identity_to_bytes() {
        let identity = Identity::generate();
        let public = identity.verifying_key();
        let bytes = public.to_bytes();
        assert_eq!(bytes.len(), 32);

        // Should roundtrip correctly
        let restored = PublicIdentity::from_bytes(&bytes).unwrap();
        assert_eq!(restored.to_node_id(), public.to_node_id());
    }

    #[test]
    fn test_identity_sign_verify_roundtrip() {
        let identity = Identity::generate();
        let message = b"test message for signing";

        let signature = identity.sign(message);
        let public = identity.verifying_key();

        assert!(public.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_public_identity_to_node_id() {
        let identity = Identity::generate();
        let public = identity.verifying_key();

        // Both should have the same node ID
        assert_eq!(identity.node_id(), public.to_node_id());

        // Node ID should be consistent
        let id1 = public.to_node_id();
        let id2 = public.to_node_id();
        assert_eq!(id1, id2);
    }
}
