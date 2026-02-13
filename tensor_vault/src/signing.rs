// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Edge HMAC signing and verification for graph-based access control.
//!
//! Signs VAULT_ACCESS edges with HMAC-BLAKE2b to detect tampering.
//! The signing key is derived from the master key via HKDF.

use blake2::{digest::consts::U32, digest::Mac, Blake2b, Digest};
use zeroize::Zeroizing;

use crate::key::{MasterKey, KEY_SIZE};

/// Domain separation tag for edge signing key derivation.
const EDGE_SIGNING_DOMAIN: &[u8] = b"neumann_vault_edge_signing_v1";

/// Signs and verifies graph edges using HMAC-BLAKE2b.
pub struct EdgeSigner {
    hmac_key: Zeroizing<[u8; KEY_SIZE]>,
}

impl EdgeSigner {
    /// Create a new edge signer from the master key.
    pub fn new(master_key: &MasterKey) -> Self {
        let hmac_key = Zeroizing::new(master_key.derive_subkey(EDGE_SIGNING_DOMAIN));
        Self { hmac_key }
    }

    /// Create a signer with zeroed key (for seal zeroization).
    pub fn from_zeroed() -> Self {
        Self {
            hmac_key: Zeroizing::new([0u8; KEY_SIZE]),
        }
    }

    /// Sign edge data: `HMAC-BLAKE2b(from || to || edge_type || timestamp)`.
    pub fn sign_edge(&self, from: &str, to: &str, edge_type: &str, timestamp: i64) -> Vec<u8> {
        self.compute_mac(from, to, edge_type, timestamp).to_vec()
    }

    /// Verify an edge signature (tries new MAC first, then legacy fallback).
    pub fn verify_edge(
        &self,
        from: &str,
        to: &str,
        edge_type: &str,
        timestamp: i64,
        signature: &[u8],
    ) -> bool {
        let expected = self.compute_mac(from, to, edge_type, timestamp);
        if constant_time_eq(&expected, signature) {
            return true;
        }
        // Legacy fallback: try old hand-rolled HMAC
        let legacy = self.compute_mac_legacy(from, to, edge_type, timestamp);
        constant_time_eq(&legacy, signature)
    }

    /// BLAKE2b-MAC512 keyed MAC for edge data (truncated to 32 bytes).
    fn compute_mac(&self, from: &str, to: &str, edge_type: &str, timestamp: i64) -> [u8; 32] {
        let mut mac =
            blake2::Blake2bMac512::new_from_slice(&*self.hmac_key).expect("valid key length");
        mac.update(from.as_bytes());
        mac.update(b"\x00");
        mac.update(to.as_bytes());
        mac.update(b"\x00");
        mac.update(edge_type.as_bytes());
        mac.update(b"\x00");
        mac.update(&timestamp.to_le_bytes());
        let result = mac.finalize().into_bytes();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result[..32]);
        out
    }

    /// Legacy hand-rolled HMAC-BLAKE2b construction (for backward compatibility).
    fn compute_mac_legacy(
        &self,
        from: &str,
        to: &str,
        edge_type: &str,
        timestamp: i64,
    ) -> [u8; 32] {
        let mut inner_key = *self.hmac_key;
        for byte in &mut inner_key {
            *byte ^= 0x36;
        }

        let mut inner = Blake2b::<U32>::new();
        inner.update(inner_key);
        inner.update(from.as_bytes());
        inner.update(b"\x00");
        inner.update(to.as_bytes());
        inner.update(b"\x00");
        inner.update(edge_type.as_bytes());
        inner.update(b"\x00");
        inner.update(timestamp.to_le_bytes());
        let inner_hash = inner.finalize();

        let mut outer_key = *self.hmac_key;
        for byte in &mut outer_key {
            *byte ^= 0x5c;
        }

        let mut outer = Blake2b::<U32>::new();
        outer.update(outer_key);
        outer.update(inner_hash);
        let result = outer.finalize();

        result.into()
    }
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signer() -> EdgeSigner {
        let key = MasterKey::from_bytes([42u8; KEY_SIZE]);
        EdgeSigner::new(&key)
    }

    #[test]
    fn test_sign_and_verify() {
        let signer = test_signer();

        let sig = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        assert!(signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000, &sig));
    }

    #[test]
    fn test_deterministic_signatures() {
        let signer = test_signer();

        let sig1 = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        let sig2 = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_different_inputs_different_signatures() {
        let signer = test_signer();

        let sig1 = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        let sig2 = signer.sign_edge("user:bob", "secret:key", "VAULT_ACCESS_READ", 1000);
        let sig3 = signer.sign_edge("user:alice", "secret:other", "VAULT_ACCESS_READ", 1000);
        let sig4 = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_WRITE", 1000);
        let sig5 = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 2000);

        assert_ne!(sig1, sig2);
        assert_ne!(sig1, sig3);
        assert_ne!(sig1, sig4);
        assert_ne!(sig1, sig5);
    }

    #[test]
    fn test_tampered_signature_rejected() {
        let signer = test_signer();

        let mut sig = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        // Tamper with the signature
        if let Some(byte) = sig.first_mut() {
            *byte ^= 0xff;
        }

        assert!(!signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000, &sig));
    }

    #[test]
    fn test_wrong_parameters_rejected() {
        let signer = test_signer();

        let sig = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);

        // Wrong from
        assert!(!signer.verify_edge("user:bob", "secret:key", "VAULT_ACCESS_READ", 1000, &sig));
        // Wrong to
        assert!(!signer.verify_edge(
            "user:alice",
            "secret:other",
            "VAULT_ACCESS_READ",
            1000,
            &sig
        ));
        // Wrong edge type
        assert!(!signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_WRITE", 1000, &sig));
        // Wrong timestamp
        assert!(!signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 9999, &sig));
    }

    #[test]
    fn test_different_keys_different_signatures() {
        let signer1 = EdgeSigner::new(&MasterKey::from_bytes([1u8; KEY_SIZE]));
        let signer2 = EdgeSigner::new(&MasterKey::from_bytes([2u8; KEY_SIZE]));

        let sig1 = signer1.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        let sig2 = signer2.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_wrong_key_verification_fails() {
        let signer1 = EdgeSigner::new(&MasterKey::from_bytes([1u8; KEY_SIZE]));
        let signer2 = EdgeSigner::new(&MasterKey::from_bytes([2u8; KEY_SIZE]));

        let sig = signer1.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);

        assert!(!signer2.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000, &sig));
    }

    #[test]
    fn test_empty_signature_rejected() {
        let signer = test_signer();
        assert!(!signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000, &[]));
    }

    #[test]
    fn test_short_signature_rejected() {
        let signer = test_signer();
        assert!(!signer.verify_edge(
            "user:alice",
            "secret:key",
            "VAULT_ACCESS_READ",
            1000,
            &[0u8; 16]
        ));
    }

    #[test]
    fn test_constant_time_eq_basic() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_signing_new_mac_roundtrip() {
        let signer = test_signer();
        let sig = signer.sign_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        assert!(signer.verify_edge("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000, &sig));
        assert_eq!(sig.len(), 32);
    }

    #[test]
    fn test_signing_legacy_fallback_verify() {
        let signer = test_signer();
        // Compute a legacy-style signature
        let legacy_sig =
            signer.compute_mac_legacy("user:alice", "secret:key", "VAULT_ACCESS_READ", 1000);
        // verify_edge should accept it via fallback
        assert!(signer.verify_edge(
            "user:alice",
            "secret:key",
            "VAULT_ACCESS_READ",
            1000,
            &legacy_sig
        ));
    }
}
