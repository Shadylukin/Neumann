//! Obfuscation layer for hiding metadata and storage patterns.
//!
//! Provides:
//! - Key obfuscation via HMAC
//! - Blind indexes for searchable encryption
//! - Padding for length hiding
//! - Pointer indirection for storage pattern hiding

use blake2::{digest::consts::U32, Blake2b, Digest};

use crate::{key::MasterKey, Result, VaultError};

/// Padding block sizes for length hiding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingSize {
    /// 256 bytes - for short secrets (API keys, tokens)
    Small = 256,
    /// 1 KB - for medium secrets (certificates, small configs)
    Medium = 1024,
    /// 4 KB - for large secrets (private keys, large configs)
    Large = 4096,
    /// 16 KB - for very large secrets
    ExtraLarge = 16384,
}

impl PaddingSize {
    /// Select appropriate padding size for given plaintext length.
    pub fn for_length(len: usize) -> Self {
        if len <= 240 {
            Self::Small
        } else if len <= 1000 {
            Self::Medium
        } else if len <= 4000 {
            Self::Large
        } else {
            Self::ExtraLarge
        }
    }
}

/// Obfuscation utilities using HMAC-like construction.
pub struct Obfuscator {
    /// Derived key for obfuscation (separate from encryption key)
    obfuscation_key: [u8; 32],
}

impl Obfuscator {
    /// Create obfuscator from master key.
    /// Derives a separate key for obfuscation to maintain key separation.
    pub fn new(master_key: &MasterKey) -> Self {
        // Derive obfuscation key using domain separation
        let mut obfuscation_key = [0u8; 32];
        let master_bytes = master_key.as_bytes();

        // Simple key derivation with domain separation
        // In production, use HKDF
        for (i, byte) in obfuscation_key.iter_mut().enumerate() {
            *byte = master_bytes[i] ^ 0x5c; // HMAC-style outer padding
        }

        // Mix in domain separator
        let domain = b"neumann_vault_obfuscation_v1";
        for (i, &b) in domain.iter().enumerate() {
            obfuscation_key[i % 32] ^= b;
        }

        Self { obfuscation_key }
    }

    /// Obfuscate a secret key for storage.
    /// Returns a hex-encoded HMAC-like hash that hides the original key.
    pub fn obfuscate_key(&self, key: &str) -> String {
        let hash = self.hmac_hash(key.as_bytes(), b"key");
        hex::encode(&hash[..16]) // Use first 16 bytes = 32 hex chars
    }

    /// Create a blind index for pattern matching.
    /// Allows searching for secrets by pattern without revealing the pattern.
    pub fn blind_index(&self, key: &str, pattern: &str) -> String {
        // Combine key and pattern for the index
        let mut input = key.as_bytes().to_vec();
        input.extend_from_slice(b"::");
        input.extend_from_slice(pattern.as_bytes());

        let hash = self.hmac_hash(&input, b"blind_index");
        hex::encode(&hash[..8]) // Shorter for index (16 hex chars)
    }

    /// Generate a random-looking storage key for pointer indirection.
    pub fn generate_storage_id(&self, key: &str, nonce: &[u8]) -> String {
        let mut input = key.as_bytes().to_vec();
        input.extend_from_slice(nonce);

        let hash = self.hmac_hash(&input, b"storage_id");
        format!("_vs:{}", hex::encode(&hash[..12])) // 24 hex chars
    }

    /// Obfuscate metadata (like timestamps, creator).
    pub fn obfuscate_metadata(&self, data: &[u8]) -> Vec<u8> {
        // XOR with fixed keystream derived from obfuscation key
        // Use empty data to derive constant keystream (not data-dependent)
        let keystream = self.hmac_hash(&[], b"metadata_stream");
        data.iter()
            .zip(keystream.iter().cycle())
            .map(|(d, k)| d ^ k)
            .collect()
    }

    /// Deobfuscate metadata.
    pub fn deobfuscate_metadata(&self, obfuscated: &[u8]) -> Vec<u8> {
        // XOR is its own inverse with same keystream
        self.obfuscate_metadata(obfuscated)
    }

    /// HMAC construction using BLAKE2b (cryptographically secure).
    fn hmac_hash(&self, data: &[u8], domain: &[u8]) -> [u8; 32] {
        // Inner hash: H((key XOR ipad) || domain || data)
        let mut inner_key = self.obfuscation_key;
        for byte in &mut inner_key {
            *byte ^= 0x36; // ipad
        }

        let mut inner_hasher = Blake2b::<U32>::new();
        inner_hasher.update(inner_key);
        inner_hasher.update(domain);
        inner_hasher.update(data);
        let inner_hash = inner_hasher.finalize();

        // Outer hash: H((key XOR opad) || inner_hash)
        let mut outer_key = self.obfuscation_key;
        for byte in &mut outer_key {
            *byte ^= 0x5c; // opad
        }

        let mut outer_hasher = Blake2b::<U32>::new();
        outer_hasher.update(outer_key);
        outer_hasher.update(inner_hash);
        let result = outer_hasher.finalize();

        result.into()
    }
}

/// Pad plaintext to hide its length.
pub fn pad_plaintext(plaintext: &[u8]) -> Vec<u8> {
    let target_size = PaddingSize::for_length(plaintext.len()) as usize;
    let padding_len = target_size - plaintext.len() - 2; // -2 for length prefix

    let mut padded = Vec::with_capacity(target_size);

    // Store original length as 2 bytes (max 65535)
    let len = plaintext.len() as u16;
    padded.push((len >> 8) as u8);
    padded.push((len & 0xff) as u8);

    // Original data
    padded.extend_from_slice(plaintext);

    // Random padding
    let mut rng_bytes = vec![0u8; padding_len];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut rng_bytes);
    padded.extend_from_slice(&rng_bytes);

    padded
}

/// Remove padding from plaintext.
pub fn unpad_plaintext(padded: &[u8]) -> Result<Vec<u8>> {
    if padded.len() < 2 {
        return Err(VaultError::CryptoError("Padded data too short".to_string()));
    }

    let len = ((padded[0] as usize) << 8) | (padded[1] as usize);

    if len + 2 > padded.len() {
        return Err(VaultError::CryptoError(
            "Invalid padding length".to_string(),
        ));
    }

    Ok(padded[2..2 + len].to_vec())
}

mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::KEY_SIZE;

    fn test_key() -> MasterKey {
        MasterKey::from_bytes([42u8; KEY_SIZE])
    }

    #[test]
    fn test_obfuscate_key_deterministic() {
        let obf = Obfuscator::new(&test_key());

        let key1 = obf.obfuscate_key("api_key");
        let key2 = obf.obfuscate_key("api_key");

        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_obfuscate_key_different_inputs() {
        let obf = Obfuscator::new(&test_key());

        let key1 = obf.obfuscate_key("api_key");
        let key2 = obf.obfuscate_key("db_password");

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_obfuscate_key_different_master_keys() {
        let obf1 = Obfuscator::new(&MasterKey::from_bytes([1u8; KEY_SIZE]));
        let obf2 = Obfuscator::new(&MasterKey::from_bytes([2u8; KEY_SIZE]));

        let key1 = obf1.obfuscate_key("api_key");
        let key2 = obf2.obfuscate_key("api_key");

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_blind_index() {
        let obf = Obfuscator::new(&test_key());

        let idx1 = obf.blind_index("api_key", "api:*");
        let idx2 = obf.blind_index("api_key", "api:*");
        let idx3 = obf.blind_index("api_key", "db:*");

        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
    }

    #[test]
    fn test_generate_storage_id() {
        let obf = Obfuscator::new(&test_key());

        let id1 = obf.generate_storage_id("key", &[1, 2, 3]);
        let id2 = obf.generate_storage_id("key", &[1, 2, 3]);
        let id3 = obf.generate_storage_id("key", &[4, 5, 6]);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(id1.starts_with("_vs:"));
    }

    #[test]
    fn test_metadata_obfuscation_roundtrip() {
        let obf = Obfuscator::new(&test_key());

        let original = b"user:alice";
        let obfuscated = obf.obfuscate_metadata(original);
        let recovered = obf.deobfuscate_metadata(&obfuscated);

        assert_ne!(obfuscated.as_slice(), original);
        assert_eq!(recovered.as_slice(), original);
    }

    #[test]
    fn test_padding_small() {
        let plaintext = b"short";
        let padded = pad_plaintext(plaintext);

        assert_eq!(padded.len(), PaddingSize::Small as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_medium() {
        let plaintext = vec![b'x'; 500];
        let padded = pad_plaintext(&plaintext);

        assert_eq!(padded.len(), PaddingSize::Medium as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_large() {
        let plaintext = vec![b'x'; 2000];
        let padded = pad_plaintext(&plaintext);

        assert_eq!(padded.len(), PaddingSize::Large as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_sizes() {
        assert_eq!(PaddingSize::for_length(10), PaddingSize::Small);
        assert_eq!(PaddingSize::for_length(240), PaddingSize::Small);
        assert_eq!(PaddingSize::for_length(241), PaddingSize::Medium);
        assert_eq!(PaddingSize::for_length(1000), PaddingSize::Medium);
        assert_eq!(PaddingSize::for_length(1001), PaddingSize::Large);
        assert_eq!(PaddingSize::for_length(4000), PaddingSize::Large);
        assert_eq!(PaddingSize::for_length(4001), PaddingSize::ExtraLarge);
    }

    #[test]
    fn test_hex_encode() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        let encoded = hex::encode(&data);
        assert_eq!(encoded, "deadbeef");

        // Empty data
        assert_eq!(hex::encode(&[]), "");

        // Single byte
        assert_eq!(hex::encode(&[0x0f]), "0f");
    }
}
