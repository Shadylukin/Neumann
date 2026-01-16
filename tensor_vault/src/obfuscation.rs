//! Obfuscation layer for hiding metadata and storage patterns.
//!
//! Provides:
//! - Key obfuscation via HMAC
//! - Blind indexes for searchable encryption
//! - Padding for length hiding
//! - Pointer indirection for storage pattern hiding
//! - AEAD metadata encryption with per-record nonces

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use blake2::{digest::consts::U32, Blake2b, Digest};
use rand::RngCore;

use crate::{key::MasterKey, Result, VaultError};

/// Nonce size for metadata AEAD (12 bytes = 96 bits, standard for AES-GCM).
pub const METADATA_NONCE_SIZE: usize = 12;

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
    /// Derived key for metadata AEAD encryption
    metadata_key: [u8; 32],
}

impl Obfuscator {
    /// Create obfuscator from master key.
    /// Derives separate keys for obfuscation and metadata encryption.
    pub fn new(master_key: &MasterKey) -> Self {
        let master_bytes = master_key.as_bytes();

        // Derive obfuscation key using domain separation
        let mut obfuscation_key = [0u8; 32];
        for (i, byte) in obfuscation_key.iter_mut().enumerate() {
            *byte = master_bytes[i] ^ 0x5c; // HMAC-style outer padding
        }
        let domain = b"neumann_vault_obfuscation_v1";
        for (i, &b) in domain.iter().enumerate() {
            obfuscation_key[i % 32] ^= b;
        }

        // Derive separate metadata key with different domain
        let mut metadata_key = [0u8; 32];
        for (i, byte) in metadata_key.iter_mut().enumerate() {
            *byte = master_bytes[i] ^ 0x36; // HMAC-style inner padding
        }
        let metadata_domain = b"neumann_vault_metadata_v1";
        for (i, &b) in metadata_domain.iter().enumerate() {
            metadata_key[i % 32] ^= b;
        }

        Self {
            obfuscation_key,
            metadata_key,
        }
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

    /// Encrypt metadata using AEAD (AES-256-GCM) with a random per-record nonce.
    /// Returns the ciphertext with the nonce prepended (nonce || ciphertext).
    pub fn encrypt_metadata(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.metadata_key)
            .map_err(|e| VaultError::CryptoError(format!("Invalid metadata key: {e}")))?;

        // Generate random nonce for each encryption
        let mut nonce_bytes = [0u8; METADATA_NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| VaultError::CryptoError(format!("Metadata encryption failed: {e}")))?;

        // Prepend nonce to ciphertext for storage
        let mut result = Vec::with_capacity(METADATA_NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt metadata using AEAD. Expects nonce || ciphertext format.
    pub fn decrypt_metadata(&self, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < METADATA_NONCE_SIZE {
            return Err(VaultError::CryptoError(
                "Metadata too short (missing nonce)".into(),
            ));
        }

        let (nonce_bytes, ciphertext) = encrypted.split_at(METADATA_NONCE_SIZE);

        let cipher = Aes256Gcm::new_from_slice(&self.metadata_key)
            .map_err(|e| VaultError::CryptoError(format!("Invalid metadata key: {e}")))?;

        let nonce = Nonce::from_slice(nonce_bytes);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| VaultError::CryptoError(format!("Metadata decryption failed: {e}")))
    }

    /// Obfuscate metadata using simple XOR (legacy, for backward compatibility).
    #[deprecated(note = "Use encrypt_metadata for new code")]
    pub fn obfuscate_metadata(&self, data: &[u8]) -> Vec<u8> {
        let keystream = self.hmac_hash(&[], b"metadata_stream");
        data.iter()
            .zip(keystream.iter().cycle())
            .map(|(d, k)| d ^ k)
            .collect()
    }

    /// Deobfuscate metadata (legacy, for backward compatibility).
    #[deprecated(note = "Use decrypt_metadata for new code")]
    pub fn deobfuscate_metadata(&self, obfuscated: &[u8]) -> Vec<u8> {
        #[allow(deprecated)]
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
    #[allow(deprecated)]
    fn test_metadata_obfuscation_roundtrip() {
        let obf = Obfuscator::new(&test_key());

        let original = b"user:alice";
        let obfuscated = obf.obfuscate_metadata(original);
        let recovered = obf.deobfuscate_metadata(&obfuscated);

        assert_ne!(obfuscated.as_slice(), original);
        assert_eq!(recovered.as_slice(), original);
    }

    #[test]
    fn test_metadata_aead_roundtrip() {
        let obf = Obfuscator::new(&test_key());

        let original = b"user:alice:timestamp:1234567890";
        let encrypted = obf.encrypt_metadata(original).unwrap();
        let decrypted = obf.decrypt_metadata(&encrypted).unwrap();

        // Encrypted should be different from original
        assert_ne!(encrypted.as_slice(), original.as_slice());
        // Should be longer due to nonce (12 bytes) + auth tag (16 bytes)
        assert!(encrypted.len() > original.len());
        // Decrypted should match original
        assert_eq!(decrypted.as_slice(), original);
    }

    #[test]
    fn test_metadata_aead_unique_nonces() {
        let obf = Obfuscator::new(&test_key());

        let data = b"same data";
        let encrypted1 = obf.encrypt_metadata(data).unwrap();
        let encrypted2 = obf.encrypt_metadata(data).unwrap();

        // Same data should produce different ciphertexts due to random nonces
        assert_ne!(encrypted1, encrypted2);

        // Both should decrypt to same original
        assert_eq!(
            obf.decrypt_metadata(&encrypted1).unwrap(),
            obf.decrypt_metadata(&encrypted2).unwrap()
        );
    }

    #[test]
    fn test_metadata_aead_tamper_detection() {
        let obf = Obfuscator::new(&test_key());

        let original = b"sensitive metadata";
        let mut encrypted = obf.encrypt_metadata(original).unwrap();

        // Tamper with the ciphertext
        if let Some(last) = encrypted.last_mut() {
            *last ^= 0xff;
        }

        // Decryption should fail due to authentication
        assert!(obf.decrypt_metadata(&encrypted).is_err());
    }

    #[test]
    fn test_metadata_aead_short_input() {
        let obf = Obfuscator::new(&test_key());

        // Too short - no nonce
        let result = obf.decrypt_metadata(&[1, 2, 3]);
        assert!(result.is_err());
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
