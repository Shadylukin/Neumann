//! Master key derivation using Argon2id.

use crate::{Result, VaultConfig, VaultError};
use argon2::{Algorithm, Argon2, Params, Version};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// AES-256 key size in bytes.
pub const KEY_SIZE: usize = 32;

/// Default salt for key derivation.
const DEFAULT_SALT: &[u8; 16] = b"neumann_vault_16";

/// Derived master key for encryption (zeroized on drop).
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct MasterKey {
    bytes: [u8; KEY_SIZE],
}

impl MasterKey {
    /// Derive a master key from raw input using Argon2id.
    pub fn derive(input: &[u8], config: &VaultConfig) -> Result<Self> {
        let salt = config
            .salt
            .as_ref()
            .map_or(DEFAULT_SALT.as_slice(), |s| s.as_slice());

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

    /// Create from raw bytes (for testing and fuzzing).
    #[cfg(any(test, feature = "fuzzing"))]
    pub fn from_bytes(bytes: [u8; KEY_SIZE]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_deterministic() {
        let config = VaultConfig::default();
        let key1 = MasterKey::derive(b"password123", &config).unwrap();
        let key2 = MasterKey::derive(b"password123", &config).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_different_passwords_different_keys() {
        let config = VaultConfig::default();
        let key1 = MasterKey::derive(b"password1", &config).unwrap();
        let key2 = MasterKey::derive(b"password2", &config).unwrap();

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

        let key1 = MasterKey::derive(b"password", &config1).unwrap();
        let key2 = MasterKey::derive(b"password", &config2).unwrap();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_empty_password() {
        let config = VaultConfig::default();
        let key = MasterKey::derive(b"", &config).unwrap();

        assert_eq!(key.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_long_password() {
        let config = VaultConfig::default();
        let long_password = vec![b'a'; 10000];
        let key = MasterKey::derive(&long_password, &config).unwrap();

        assert_eq!(key.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_key_is_32_bytes() {
        let config = VaultConfig::default();
        let key = MasterKey::derive(b"test", &config).unwrap();

        assert_eq!(key.as_bytes().len(), 32);
    }

    #[test]
    fn test_from_bytes() {
        let bytes = [42u8; KEY_SIZE];
        let key = MasterKey::from_bytes(bytes);

        assert_eq!(key.as_bytes(), &bytes);
    }
}
