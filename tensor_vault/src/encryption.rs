// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! AES-256-GCM encryption for vault secrets.

use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use zeroize::Zeroizing;

use crate::{
    key::{MasterKey, KEY_SIZE},
    Result, VaultError,
};

/// 12-byte nonce for AES-GCM (96 bits is the standard).
pub const NONCE_SIZE: usize = 12;

/// Encryption cipher using AES-256-GCM with HKDF-derived key.
pub struct Cipher {
    encryption_key: Zeroizing<[u8; KEY_SIZE]>,
}

impl Cipher {
    /// Derive the encryption subkey from the master key via HKDF.
    pub fn new(master_key: &MasterKey) -> Self {
        Self {
            encryption_key: Zeroizing::new(master_key.encryption_key()),
        }
    }

    /// Create a cipher from a raw key (for transit encryption).
    pub fn from_raw_key(key: [u8; KEY_SIZE]) -> Self {
        Self {
            encryption_key: Zeroizing::new(key),
        }
    }

    /// Encrypt plaintext, returning (ciphertext, nonce).
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<(Vec<u8>, [u8; NONCE_SIZE])> {
        let cipher = Aes256Gcm::new_from_slice(&*self.encryption_key)
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

        let cipher = Aes256Gcm::new_from_slice(&*self.encryption_key)
            .map_err(|e| VaultError::CryptoError(format!("Invalid key: {e}")))?;

        let nonce = Nonce::from_slice(nonce_bytes);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| VaultError::CryptoError(format!("Decryption failed: {e}")))
    }

    /// Encrypt plaintext with additional authenticated data (AAD).
    ///
    /// Prepends a `0x02` version byte so decryption can distinguish AAD-bound
    /// ciphertexts from legacy bare encryptions.
    pub fn encrypt_with_aad(
        &self,
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<(Vec<u8>, [u8; NONCE_SIZE])> {
        let cipher = Aes256Gcm::new_from_slice(&*self.encryption_key)
            .map_err(|e| VaultError::CryptoError(format!("Invalid key: {e}")))?;

        let mut nonce_bytes = [0u8; NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let payload = Payload {
            msg: plaintext,
            aad,
        };
        let raw_ct = cipher
            .encrypt(nonce, payload)
            .map_err(|e| VaultError::CryptoError(format!("Encryption failed: {e}")))?;

        // Prepend version tag 0x02 to distinguish from legacy ciphertext
        let mut tagged = Vec::with_capacity(1 + raw_ct.len());
        tagged.push(0x02);
        tagged.extend_from_slice(&raw_ct);

        Ok((tagged, nonce_bytes))
    }

    /// Decrypt ciphertext with additional authenticated data (AAD).
    ///
    /// If the first byte is `0x02`, strips it and decrypts with AAD.
    /// Otherwise falls back to bare decryption for backward compatibility.
    pub fn decrypt_with_aad(
        &self,
        ciphertext: &[u8],
        nonce_bytes: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>> {
        if nonce_bytes.len() != NONCE_SIZE {
            return Err(VaultError::CryptoError(format!(
                "Invalid nonce size: expected {NONCE_SIZE}, got {}",
                nonce_bytes.len()
            )));
        }

        // Version-tagged: 0x02 means AAD-bound ciphertext
        if ciphertext.first() == Some(&0x02) {
            let cipher = Aes256Gcm::new_from_slice(&*self.encryption_key)
                .map_err(|e| VaultError::CryptoError(format!("Invalid key: {e}")))?;
            let nonce = Nonce::from_slice(nonce_bytes);
            let payload = Payload {
                msg: &ciphertext[1..],
                aad,
            };
            cipher
                .decrypt(nonce, payload)
                .map_err(|e| VaultError::CryptoError(format!("Decryption failed: {e}")))
        } else {
            // Legacy ciphertext without version tag -- bare decrypt
            self.decrypt(ciphertext, nonce_bytes)
        }
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
        let cipher = Cipher::new(&test_key());

        let plaintext = b"hello, world!";
        let (ciphertext, nonce) = cipher.encrypt(plaintext).unwrap();

        assert_ne!(&ciphertext[..], plaintext);

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn test_decrypt_wrong_nonce_fails() {
        let cipher = Cipher::new(&test_key());

        let plaintext = b"secret data";
        let (ciphertext, _nonce) = cipher.encrypt(plaintext).unwrap();

        let wrong_nonce = [0u8; NONCE_SIZE];
        let result = cipher.decrypt(&ciphertext, &wrong_nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_each_encryption_unique_nonce() {
        let cipher = Cipher::new(&test_key());

        let plaintext = b"same text";
        let (_, nonce1) = cipher.encrypt(plaintext).unwrap();
        let (_, nonce2) = cipher.encrypt(plaintext).unwrap();

        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_empty_plaintext() {
        let cipher = Cipher::new(&test_key());

        let plaintext = b"";
        let (ciphertext, nonce) = cipher.encrypt(plaintext).unwrap();

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn test_large_plaintext() {
        let cipher = Cipher::new(&test_key());

        let plaintext = vec![0xabu8; 1024 * 1024]; // 1MB
        let (ciphertext, nonce) = cipher.encrypt(&plaintext).unwrap();

        let decrypted = cipher.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_invalid_nonce_size() {
        let cipher = Cipher::new(&test_key());

        let (ciphertext, _) = cipher.encrypt(b"data").unwrap();

        let short_nonce = [0u8; 8];
        let result = cipher.decrypt(&ciphertext, &short_nonce);
        assert!(matches!(result, Err(VaultError::CryptoError(_))));
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let cipher = Cipher::new(&test_key());

        let (mut ciphertext, nonce) = cipher.encrypt(b"secret").unwrap();

        // Tamper with ciphertext
        if let Some(byte) = ciphertext.first_mut() {
            *byte ^= 0xff;
        }

        let result = cipher.decrypt(&ciphertext, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypt_decrypt_with_aad_roundtrip() {
        let cipher = Cipher::new(&test_key());
        let plaintext = b"hello with aad";
        let aad = b"context-binding-data";

        let (ciphertext, nonce) = cipher.encrypt_with_aad(plaintext, aad).unwrap();
        let decrypted = cipher.decrypt_with_aad(&ciphertext, &nonce, aad).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn test_wrong_aad_fails() {
        let cipher = Cipher::new(&test_key());
        let (ciphertext, nonce) = cipher.encrypt_with_aad(b"secret", b"correct-aad").unwrap();

        let result = cipher.decrypt_with_aad(&ciphertext, &nonce, b"wrong-aad");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_aad_works() {
        let cipher = Cipher::new(&test_key());
        let (ciphertext, nonce) = cipher.encrypt_with_aad(b"data", b"").unwrap();

        let decrypted = cipher.decrypt_with_aad(&ciphertext, &nonce, b"").unwrap();
        assert_eq!(&decrypted[..], b"data");
    }

    #[test]
    fn test_aad_legacy_fallback() {
        let cipher = Cipher::new(&test_key());
        // Encrypt without AAD (legacy)
        let (ciphertext, nonce) = cipher.encrypt(b"old data").unwrap();

        // decrypt_with_aad should fall back to bare decrypt for legacy data
        let decrypted = cipher
            .decrypt_with_aad(&ciphertext, &nonce, b"any-aad")
            .unwrap();
        assert_eq!(&decrypted[..], b"old data");
    }

    #[test]
    fn test_aad_version_tag_present() {
        let cipher = Cipher::new(&test_key());
        let (ciphertext, _) = cipher.encrypt_with_aad(b"tagged", b"aad").unwrap();

        assert_eq!(ciphertext[0], 0x02, "AAD ciphertext must start with 0x02");
    }

    #[test]
    fn test_aad_tampered_ciphertext_fails() {
        let cipher = Cipher::new(&test_key());
        let (mut ciphertext, nonce) = cipher.encrypt_with_aad(b"secret", b"aad").unwrap();

        // Tamper with a byte after the version tag
        if ciphertext.len() > 2 {
            ciphertext[2] ^= 0xff;
        }
        let result = cipher.decrypt_with_aad(&ciphertext, &nonce, b"aad");
        assert!(result.is_err());
    }

    #[test]
    fn test_uses_hkdf_derived_key_not_raw() {
        let master = test_key();
        // The HKDF-derived encryption key must differ from the raw master bytes
        assert_ne!(
            master.encryption_key(),
            *master.as_bytes(),
            "Cipher should use HKDF-derived key, not raw master key"
        );
    }
}
