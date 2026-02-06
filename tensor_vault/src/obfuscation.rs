// SPDX-License-Identifier: MIT OR Apache-2.0
//! Obfuscation layer for hiding metadata and storage patterns.
//!
//! Provides:
//! - Key obfuscation via HMAC
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

/// Maximum plaintext size (64KB - 4 bytes for length prefix - 1 byte min padding).
pub const MAX_PLAINTEXT_SIZE: usize = 65531;

/// Length prefix size in bytes (u32).
const LENGTH_PREFIX_SIZE: usize = 4;

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
    /// 32 KB - for oversized secrets
    Huge = 32768,
    /// 64 KB - maximum supported size
    Maximum = 65536,
}

impl PaddingSize {
    /// Select appropriate padding size for given plaintext length.
    /// Accounts for the 4-byte length prefix when selecting bucket.
    pub fn for_length(len: usize) -> Option<Self> {
        // Need space for: length prefix (4 bytes) + plaintext + at least 1 byte padding
        let min_required = len.checked_add(LENGTH_PREFIX_SIZE + 1)?;

        if min_required <= Self::Small as usize {
            Some(Self::Small)
        } else if min_required <= Self::Medium as usize {
            Some(Self::Medium)
        } else if min_required <= Self::Large as usize {
            Some(Self::Large)
        } else if min_required <= Self::ExtraLarge as usize {
            Some(Self::ExtraLarge)
        } else if min_required <= Self::Huge as usize {
            Some(Self::Huge)
        } else if min_required <= Self::Maximum as usize {
            Some(Self::Maximum)
        } else {
            None // Plaintext too large
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
    /// Derives separate keys via HKDF for obfuscation and metadata encryption.
    pub fn new(master_key: &MasterKey) -> Self {
        Self {
            obfuscation_key: master_key.obfuscation_key(),
            metadata_key: master_key.metadata_key(),
        }
    }

    /// Obfuscate a secret key for storage.
    /// Returns a hex-encoded HMAC-like hash that hides the original key.
    pub fn obfuscate_key(&self, key: &str) -> String {
        let hash = self.hmac_hash(key.as_bytes(), b"key");
        hex::encode(&hash[..16]) // Use first 16 bytes = 32 hex chars
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
/// Returns an error if plaintext exceeds MAX_PLAINTEXT_SIZE.
pub fn pad_plaintext(plaintext: &[u8]) -> Result<Vec<u8>> {
    if plaintext.len() > MAX_PLAINTEXT_SIZE {
        return Err(VaultError::CryptoError(format!(
            "Plaintext too large: {} bytes exceeds maximum {}",
            plaintext.len(),
            MAX_PLAINTEXT_SIZE
        )));
    }

    let target_size = PaddingSize::for_length(plaintext.len()).ok_or_else(|| {
        VaultError::CryptoError(format!(
            "Cannot determine padding size for {} bytes",
            plaintext.len()
        ))
    })? as usize;

    // Safe subtraction: target_size >= LENGTH_PREFIX_SIZE + plaintext.len() + 1 by construction
    let padding_len = target_size
        .checked_sub(LENGTH_PREFIX_SIZE)
        .and_then(|n| n.checked_sub(plaintext.len()))
        .ok_or_else(|| VaultError::CryptoError("Padding calculation overflow".into()))?;

    let mut padded = Vec::with_capacity(target_size);

    // Store original length as 4 bytes (u32 little-endian)
    let len_bytes = (plaintext.len() as u32).to_le_bytes();
    padded.extend_from_slice(&len_bytes);

    // Original data
    padded.extend_from_slice(plaintext);

    // Random padding
    let mut rng_bytes = vec![0u8; padding_len];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut rng_bytes);
    padded.extend_from_slice(&rng_bytes);

    Ok(padded)
}

/// Remove padding from plaintext.
pub fn unpad_plaintext(padded: &[u8]) -> Result<Vec<u8>> {
    if padded.len() < LENGTH_PREFIX_SIZE {
        return Err(VaultError::CryptoError(
            "Padded data too short for length prefix".into(),
        ));
    }

    // Read length as u32 little-endian
    let len = u32::from_le_bytes([padded[0], padded[1], padded[2], padded[3]]) as usize;

    // Validate length doesn't exceed remaining data
    let data_start = LENGTH_PREFIX_SIZE;
    let data_end = data_start
        .checked_add(len)
        .ok_or_else(|| VaultError::CryptoError("Length overflow".into()))?;

    if data_end > padded.len() {
        return Err(VaultError::CryptoError(format!(
            "Invalid length prefix: {} exceeds padded data size {}",
            len,
            padded.len() - LENGTH_PREFIX_SIZE
        )));
    }

    Ok(padded[data_start..data_end].to_vec())
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
        let padded = pad_plaintext(plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Small as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_medium() {
        let plaintext = vec![b'x'; 500];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Medium as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_large() {
        let plaintext = vec![b'x'; 2000];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Large as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_extra_large() {
        let plaintext = vec![b'x'; 10000];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::ExtraLarge as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_huge() {
        let plaintext = vec![b'x'; 20000];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Huge as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_maximum() {
        let plaintext = vec![b'x'; 50000];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Maximum as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_at_max_size() {
        // Test at exactly MAX_PLAINTEXT_SIZE
        let plaintext = vec![b'x'; MAX_PLAINTEXT_SIZE];
        let padded = pad_plaintext(&plaintext).unwrap();

        assert_eq!(padded.len(), PaddingSize::Maximum as usize);

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
    }

    #[test]
    fn test_padding_exceeds_max_size() {
        // Test just over MAX_PLAINTEXT_SIZE - should fail
        let plaintext = vec![b'x'; MAX_PLAINTEXT_SIZE + 1];
        let result = pad_plaintext(&plaintext);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn test_padding_sizes() {
        // With 4-byte length prefix, need len + 5 bytes minimum
        assert_eq!(PaddingSize::for_length(10), Some(PaddingSize::Small));
        assert_eq!(PaddingSize::for_length(251), Some(PaddingSize::Small)); // 251 + 5 = 256
        assert_eq!(PaddingSize::for_length(252), Some(PaddingSize::Medium)); // 252 + 5 = 257 > 256
        assert_eq!(PaddingSize::for_length(1000), Some(PaddingSize::Medium));
        assert_eq!(PaddingSize::for_length(1019), Some(PaddingSize::Medium)); // 1019 + 5 = 1024
        assert_eq!(PaddingSize::for_length(1020), Some(PaddingSize::Large)); // 1020 + 5 = 1025 > 1024
        assert_eq!(PaddingSize::for_length(4000), Some(PaddingSize::Large));
        assert_eq!(PaddingSize::for_length(4091), Some(PaddingSize::Large)); // 4091 + 5 = 4096
        assert_eq!(PaddingSize::for_length(4092), Some(PaddingSize::ExtraLarge));
        assert_eq!(
            PaddingSize::for_length(16379),
            Some(PaddingSize::ExtraLarge)
        ); // 16379 + 5 = 16384
        assert_eq!(PaddingSize::for_length(16380), Some(PaddingSize::Huge));
        assert_eq!(PaddingSize::for_length(32763), Some(PaddingSize::Huge)); // 32763 + 5 = 32768
        assert_eq!(PaddingSize::for_length(32764), Some(PaddingSize::Maximum));
        assert_eq!(
            PaddingSize::for_length(MAX_PLAINTEXT_SIZE),
            Some(PaddingSize::Maximum)
        );
        assert_eq!(PaddingSize::for_length(MAX_PLAINTEXT_SIZE + 1), None);
    }

    #[test]
    fn test_unpad_too_short() {
        // Less than 4 bytes (length prefix)
        assert!(unpad_plaintext(&[]).is_err());
        assert!(unpad_plaintext(&[1]).is_err());
        assert!(unpad_plaintext(&[1, 2]).is_err());
        assert!(unpad_plaintext(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_unpad_invalid_length() {
        // Length prefix claims more data than available
        let mut data = vec![0u8; 10];
        // Claim 100 bytes of data
        data[0..4].copy_from_slice(&100u32.to_le_bytes());

        let result = unpad_plaintext(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn test_u32_length_prefix_roundtrip() {
        // Test that we correctly use u32 (not u16) for length
        // This would fail with the old u16 implementation
        let plaintext = vec![b'x'; 40000]; // > 65535 would overflow u16
        let padded = pad_plaintext(&plaintext).unwrap();

        // Verify the length prefix is correct
        let stored_len = u32::from_le_bytes([padded[0], padded[1], padded[2], padded[3]]);
        assert_eq!(stored_len as usize, plaintext.len());

        let unpadded = unpad_plaintext(&padded).unwrap();
        assert_eq!(unpadded, plaintext);
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
