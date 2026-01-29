// SPDX-License-Identifier: MIT OR Apache-2.0
//! Master key derivation using Argon2id with HKDF-based subkey separation.

use argon2::{Algorithm, Argon2, Params, Version};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{Result, VaultConfig, VaultError};

/// Salt size for Argon2id key derivation.
pub const SALT_SIZE: usize = 16;

/// AES-256 key size in bytes.
pub const KEY_SIZE: usize = 32;

/// Derived master key for encryption (zeroized on drop).
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct MasterKey {
    bytes: [u8; KEY_SIZE],
}

impl MasterKey {
    /// Derive a master key from raw input using Argon2id.
    ///
    /// If `config.salt` is `None`, generates a cryptographically random salt.
    /// Returns both the derived key and the salt used (for persistence).
    pub fn derive(input: &[u8], config: &VaultConfig) -> Result<(Self, [u8; SALT_SIZE])> {
        match config.salt {
            Some(salt) => {
                let key = Self::derive_with_salt(input, &salt, config)?;
                Ok((key, salt))
            },
            None => Self::derive_with_random_salt(input, config),
        }
    }

    /// Derive a master key with a specific salt.
    pub fn derive_with_salt(input: &[u8], salt: &[u8], config: &VaultConfig) -> Result<Self> {
        let params = Params::new(
            config.argon2_memory_cost,
            config.argon2_time_cost,
            config.argon2_parallelism,
            Some(KEY_SIZE),
        )
        .map_err(|e| VaultError::KeyDerivationError(format!("Invalid Argon2 params: {e}")))?;

        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        let mut key = [0u8; KEY_SIZE];
        argon2
            .hash_password_into(input, salt, &mut key)
            .map_err(|e| VaultError::KeyDerivationError(format!("Argon2 failed: {e}")))?;

        Ok(Self { bytes: key })
    }

    /// Derive a master key with a newly generated random salt.
    /// Returns both the key and the salt (which should be persisted).
    pub fn derive_with_random_salt(
        input: &[u8],
        config: &VaultConfig,
    ) -> Result<(Self, [u8; SALT_SIZE])> {
        let mut salt = [0u8; SALT_SIZE];
        rand::thread_rng().fill_bytes(&mut salt);
        let key = Self::derive_with_salt(input, &salt, config)?;
        Ok((key, salt))
    }

    /// Create from raw bytes (for testing and fuzzing).
    #[cfg(any(test, feature = "fuzzing"))]
    pub fn from_bytes(bytes: [u8; KEY_SIZE]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.bytes
    }

    /// Derive a subkey using HKDF with domain separation.
    /// Each domain produces a cryptographically independent key.
    #[allow(clippy::missing_panics_doc)] // HKDF expand never fails for 32-byte output
    pub fn derive_subkey(&self, domain: &[u8]) -> [u8; KEY_SIZE] {
        let hk = Hkdf::<Sha256>::new(None, &self.bytes);
        let mut output = [0u8; KEY_SIZE];
        hk.expand(domain, &mut output)
            .expect("HKDF expand should never fail with 32-byte output");
        output
    }

    /// Derive encryption key for AES-256-GCM.
    pub fn encryption_key(&self) -> [u8; KEY_SIZE] {
        self.derive_subkey(b"neumann_vault_encryption_v1")
    }

    /// Derive obfuscation key for HMAC-based operations.
    pub fn obfuscation_key(&self) -> [u8; KEY_SIZE] {
        self.derive_subkey(b"neumann_vault_obfuscation_v1")
    }

    /// Derive metadata encryption key.
    pub fn metadata_key(&self) -> [u8; KEY_SIZE] {
        self.derive_subkey(b"neumann_vault_metadata_v1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_with_explicit_salt_deterministic() {
        // Use explicit salt for deterministic derivation
        let config = VaultConfig::default().with_salt([42u8; SALT_SIZE]);
        let (key1, salt1) = MasterKey::derive(b"password123", &config).unwrap();
        let (key2, salt2) = MasterKey::derive(b"password123", &config).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
        assert_eq!(salt1, salt2);
    }

    #[test]
    fn test_derive_without_salt_generates_random() {
        // Without salt, each call generates a different random salt
        let config = VaultConfig::default();
        let (key1, salt1) = MasterKey::derive(b"password123", &config).unwrap();
        let (key2, salt2) = MasterKey::derive(b"password123", &config).unwrap();

        // Random salts should be different
        assert_ne!(salt1, salt2);
        // Same password with different salts produces different keys
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_different_passwords_different_keys() {
        // Use explicit salt for fair comparison
        let config = VaultConfig::default().with_salt([42u8; SALT_SIZE]);
        let (key1, _) = MasterKey::derive(b"password1", &config).unwrap();
        let (key2, _) = MasterKey::derive(b"password2", &config).unwrap();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_different_salts_different_keys() {
        let config1 = VaultConfig {
            salt: Some([1u8; 16]),
            ..VaultConfig::default()
        };
        let config2 = VaultConfig {
            salt: Some([2u8; 16]),
            ..VaultConfig::default()
        };

        let (key1, _) = MasterKey::derive(b"password", &config1).unwrap();
        let (key2, _) = MasterKey::derive(b"password", &config2).unwrap();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_empty_password() {
        let config = VaultConfig::default().with_salt([42u8; SALT_SIZE]);
        let (key, _) = MasterKey::derive(b"", &config).unwrap();

        assert_eq!(key.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_long_password() {
        let config = VaultConfig::default().with_salt([42u8; SALT_SIZE]);
        let long_password = vec![b'a'; 10000];
        let (key, _) = MasterKey::derive(&long_password, &config).unwrap();

        assert_eq!(key.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_key_is_32_bytes() {
        let config = VaultConfig::default().with_salt([42u8; SALT_SIZE]);
        let (key, _) = MasterKey::derive(b"test", &config).unwrap();

        assert_eq!(key.as_bytes().len(), 32);
    }

    #[test]
    fn test_from_bytes() {
        let bytes = [42u8; KEY_SIZE];
        let key = MasterKey::from_bytes(bytes);

        assert_eq!(key.as_bytes(), &bytes);
    }

    #[test]
    fn test_hkdf_subkey_derivation() {
        let key = MasterKey::from_bytes([1u8; KEY_SIZE]);

        let subkey1 = key.derive_subkey(b"domain1");
        let subkey2 = key.derive_subkey(b"domain2");

        // Different domains produce different keys
        assert_ne!(subkey1, subkey2);
        // Same domain produces same key
        assert_eq!(subkey1, key.derive_subkey(b"domain1"));
    }

    #[test]
    fn test_hkdf_subkeys_are_independent() {
        let key = MasterKey::from_bytes([42u8; KEY_SIZE]);

        let encryption = key.encryption_key();
        let obfuscation = key.obfuscation_key();
        let metadata = key.metadata_key();

        // All subkeys are different from each other
        assert_ne!(encryption, obfuscation);
        assert_ne!(encryption, metadata);
        assert_ne!(obfuscation, metadata);

        // And different from the master key
        assert_ne!(&encryption, key.as_bytes());
        assert_ne!(&obfuscation, key.as_bytes());
        assert_ne!(&metadata, key.as_bytes());
    }

    #[test]
    fn test_hkdf_subkeys_are_deterministic() {
        let key1 = MasterKey::from_bytes([99u8; KEY_SIZE]);
        let key2 = MasterKey::from_bytes([99u8; KEY_SIZE]);

        assert_eq!(key1.encryption_key(), key2.encryption_key());
        assert_eq!(key1.obfuscation_key(), key2.obfuscation_key());
        assert_eq!(key1.metadata_key(), key2.metadata_key());
    }

    #[test]
    fn test_derive_with_random_salt() {
        let config = VaultConfig::default();

        // Generate two keys with random salts
        let (key1, salt1) = MasterKey::derive_with_random_salt(b"password", &config).unwrap();
        let (key2, salt2) = MasterKey::derive_with_random_salt(b"password", &config).unwrap();

        // Random salts should be different
        assert_ne!(salt1, salt2);

        // Same password with different salts produces different keys
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_derive_with_salt_reproducible() {
        let config = VaultConfig::default();
        let salt = [7u8; SALT_SIZE];

        // Deriving with same salt should produce same key
        let key1 = MasterKey::derive_with_salt(b"password", &salt, &config).unwrap();
        let key2 = MasterKey::derive_with_salt(b"password", &salt, &config).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }
}
