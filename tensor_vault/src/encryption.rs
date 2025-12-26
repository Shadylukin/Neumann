//! AES-256-GCM encryption for vault secrets.

use crate::key::MasterKey;
use crate::{Result, VaultError};
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;

/// 12-byte nonce for AES-GCM (96 bits is the standard).
pub const NONCE_SIZE: usize = 12;

/// Encryption cipher using AES-256-GCM.
pub struct Cipher {
    key: MasterKey,
}

impl Cipher {
    pub fn new(key: MasterKey) -> Self {
        Self { key }
    }

    /// Encrypt plaintext, returning (ciphertext, nonce).
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<(Vec<u8>, [u8; NONCE_SIZE])> {
        let cipher = Aes256Gcm::new_from_slice(self.key.as_bytes())
            .map_err(|e| VaultError::CryptoError(format!("Invalid key: {e}")))?;

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Encryption failed: {e}")))?;

        Ok((ciphertext, nonce_bytes))
    }

    /// Decrypt ciphertext using the provided nonce.
    pub fn decrypt(&self, ciphertext: &[u8], nonce_bytes: &[u8]) -> Result<Vec<u8>> {
        if nonce_bytes.len() != NONCE_SIZE {
            return Err(VaultError::CryptoError(format!(
                "Invalid nonce size: expected {NONCE_SIZE}, got {}",
                nonce_bytes.len()
            )));
        }

        let cipher = Aes256Gcm::new_from_slice(self.key.as_bytes())
            .map_err(|e| VaultError::CryptoError(format!("Invalid key: {e}")))?;

        let nonce = Nonce::from_slice(nonce_bytes);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| VaultError::CryptoError(format!("Decryption failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::KEY_SIZE;

    fn test_key() -> MasterKey {
        MasterKey::from_bytes([0u8; KEY_SIZE])
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let cipher = Cipher::new(test_key());

        let plaintext = b"hello, world!";
        let (ciphertext, nonce) = cipher.encrypt(plaintext).unwrap();

        assert_ne!(&ciphertext[..], plaintext);

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn test_decrypt_wrong_nonce_fails() {
        let cipher = Cipher::new(test_key());

        let plaintext = b"secret data";
        let (ciphertext, _nonce) = cipher.encrypt(plaintext).unwrap();

        let wrong_nonce = [0u8; NONCE_SIZE];
        let result = cipher.decrypt(&ciphertext, &wrong_nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_each_encryption_unique_nonce() {
        let cipher = Cipher::new(test_key());

        let plaintext = b"same text";
        let (_, nonce1) = cipher.encrypt(plaintext).unwrap();
        let (_, nonce2) = cipher.encrypt(plaintext).unwrap();

        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_empty_plaintext() {
        let cipher = Cipher::new(test_key());

        let plaintext = b"";
        let (ciphertext, nonce) = cipher.encrypt(plaintext).unwrap();

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn test_large_plaintext() {
        let cipher = Cipher::new(test_key());

        let plaintext = vec![0xABu8; 1024 * 1024]; // 1MB
        let (ciphertext, nonce) = cipher.encrypt(&plaintext).unwrap();

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_invalid_nonce_size() {
        let cipher = Cipher::new(test_key());

        let (ciphertext, _) = cipher.encrypt(b"data").unwrap();

        let short_nonce = [0u8; 8];
        let result = cipher.decrypt(&ciphertext, &short_nonce);
        assert!(matches!(result, Err(VaultError::CryptoError(_))));
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let cipher = Cipher::new(test_key());

        let (mut ciphertext, nonce) = cipher.encrypt(b"secret").unwrap();

        // Tamper with ciphertext
        if let Some(byte) = ciphertext.first_mut() {
            *byte ^= 0xFF;
        }

        let result = cipher.decrypt(&ciphertext, &nonce);
        assert!(result.is_err());
    }
}
