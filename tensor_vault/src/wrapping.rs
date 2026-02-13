// SPDX-License-Identifier: MIT OR Apache-2.0
//! Response wrapping (cubbyhole) for single-use secret tokens.

use std::time::{SystemTime, UNIX_EPOCH};

use rand::RngCore;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{encryption::Cipher, obfuscation, Result, VaultError};

/// Storage prefix for wrapped secrets.
const WRAP_PREFIX: &str = "_vwrap:";

/// Metadata about a wrapping token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappingToken {
    /// The hex token string.
    pub token: String,
    /// When the token was created (unix millis).
    pub created_at_ms: i64,
    /// When the token expires (unix millis).
    pub expires_at_ms: i64,
    /// Whether the token has been consumed.
    pub consumed: bool,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(&bytes)
}

fn wrap_storage_key(token: &str) -> String {
    format!("{WRAP_PREFIX}{token}")
}

/// Wrap a secret value and store it behind a single-use token.
pub fn wrap_secret(
    store: &TensorStore,
    cipher: &Cipher,
    value: &str,
    ttl_ms: i64,
) -> Result<(String, WrappingToken)> {
    let token = generate_token();
    let now = now_ms();
    let expires = now + ttl_ms;

    // Encrypt the secret value with wrapping token as AAD
    let padded = obfuscation::pad_plaintext(value.as_bytes())?;
    let (ciphertext, nonce) = cipher.encrypt_with_aad(&padded, token.as_bytes())?;

    let mut tensor = TensorData::new();
    tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
    tensor.set(
        "_nonce",
        TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
    );
    tensor.set("_created_at", TensorValue::Scalar(ScalarValue::Int(now)));
    tensor.set(
        "_expires_at",
        TensorValue::Scalar(ScalarValue::Int(expires)),
    );
    tensor.set("_consumed", TensorValue::Scalar(ScalarValue::Bool(false)));

    let key = wrap_storage_key(&token);
    store
        .put(&key, tensor)
        .map_err(|e| VaultError::StorageError(e.to_string()))?;

    let info = WrappingToken {
        token: token.clone(),
        created_at_ms: now,
        expires_at_ms: expires,
        consumed: false,
    };

    Ok((token, info))
}

/// Unwrap a secret -- single use, deletes after read.
pub fn unwrap_secret(store: &TensorStore, cipher: &Cipher, token: &str) -> Result<String> {
    let key = wrap_storage_key(token);
    let tensor = store
        .get(&key)
        .map_err(|_| VaultError::NotFound(format!("wrapping token: {token}")))?;

    // Check consumed
    if matches!(
        tensor.get("_consumed"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    ) {
        return Err(VaultError::WrappingTokenConsumed(token.to_string()));
    }

    // Check expiry
    if let Some(TensorValue::Scalar(ScalarValue::Int(expires))) = tensor.get("_expires_at") {
        if now_ms() >= *expires {
            return Err(VaultError::WrappingTokenExpired(token.to_string()));
        }
    }

    let ciphertext = match tensor.get("_data") {
        Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
        _ => return Err(VaultError::CryptoError("wrapped data missing".to_string())),
    };

    let nonce = match tensor.get("_nonce") {
        Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
        _ => return Err(VaultError::CryptoError("wrapped nonce missing".to_string())),
    };

    let padded = cipher.decrypt_with_aad(&ciphertext, &nonce, token.as_bytes())?;
    let plaintext = obfuscation::unpad_plaintext(&padded)?;

    // Delete the wrapped entry (single-use)
    store.delete(&key).ok();

    String::from_utf8(plaintext).map_err(|e| VaultError::CryptoError(format!("invalid UTF-8: {e}")))
}

/// Get info about a wrapping token without consuming it.
pub fn wrapping_token_info(store: &TensorStore, token: &str) -> Option<WrappingToken> {
    let key = wrap_storage_key(token);
    let Ok(tensor) = store.get(&key) else {
        return None;
    };

    let created_at_ms = match tensor.get("_created_at") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };

    let expires_at_ms = match tensor.get("_expires_at") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };

    let consumed = matches!(
        tensor.get("_consumed"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    );

    Some(WrappingToken {
        token: token.to_string(),
        created_at_ms,
        expires_at_ms,
        consumed,
    })
}

/// Module-level hex encoding (avoid external dep since we only do it here + shamir).
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{MasterKey, KEY_SIZE};

    fn test_cipher() -> Cipher {
        Cipher::new(&MasterKey::from_bytes([42u8; KEY_SIZE]))
    }

    #[test]
    fn test_wrap_unwrap_roundtrip() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (token, info) = wrap_secret(&store, &cipher, "my_secret", 60_000).unwrap();
        assert!(!info.consumed);
        assert!(!token.is_empty());
        assert_eq!(token.len(), 64); // 32 bytes hex-encoded

        let value = unwrap_secret(&store, &cipher, &token).unwrap();
        assert_eq!(value, "my_secret");
    }

    #[test]
    fn test_unwrap_consumed_fails() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (token, _) = wrap_secret(&store, &cipher, "secret", 60_000).unwrap();
        unwrap_secret(&store, &cipher, &token).unwrap();

        // Second unwrap should fail (token deleted)
        let result = unwrap_secret(&store, &cipher, &token);
        assert!(result.is_err());
    }

    #[test]
    fn test_unwrap_expired_fails() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        // TTL of 0ms - immediately expired
        let (token, _) = wrap_secret(&store, &cipher, "secret", 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let result = unwrap_secret(&store, &cipher, &token);
        assert!(matches!(result, Err(VaultError::WrappingTokenExpired(_))));
    }

    #[test]
    fn test_unwrap_nonexistent_token() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let result = unwrap_secret(&store, &cipher, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrapping_token_info() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (token, _) = wrap_secret(&store, &cipher, "secret", 60_000).unwrap();

        let info = wrapping_token_info(&store, &token).unwrap();
        assert_eq!(info.token, token);
        assert!(!info.consumed);
        assert!(info.expires_at_ms > info.created_at_ms);
    }

    #[test]
    fn test_wrapping_token_info_nonexistent() {
        let store = TensorStore::new();
        let result = wrapping_token_info(&store, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_unique_tokens() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (t1, _) = wrap_secret(&store, &cipher, "secret1", 60_000).unwrap();
        let (t2, _) = wrap_secret(&store, &cipher, "secret2", 60_000).unwrap();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_wrap_empty_string() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (token, _) = wrap_secret(&store, &cipher, "", 60_000).unwrap();
        let value = unwrap_secret(&store, &cipher, &token).unwrap();
        assert_eq!(value, "");
    }

    #[test]
    fn test_wrap_large_value() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let large = "x".repeat(10_000);
        let (token, _) = wrap_secret(&store, &cipher, &large, 60_000).unwrap();
        let value = unwrap_secret(&store, &cipher, &token).unwrap();
        assert_eq!(value, large);
    }

    #[test]
    fn test_wrap_preserves_special_chars() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let special = "pass\nword\t with \"quotes\" & 'apos'";
        let (token, _) = wrap_secret(&store, &cipher, special, 60_000).unwrap();
        let value = unwrap_secret(&store, &cipher, &token).unwrap();
        assert_eq!(value, special);
    }

    #[test]
    fn test_token_info_after_unwrap_returns_none() {
        let store = TensorStore::new();
        let cipher = test_cipher();

        let (token, _) = wrap_secret(&store, &cipher, "secret", 60_000).unwrap();
        unwrap_secret(&store, &cipher, &token).unwrap();

        // After unwrap, token entry is deleted
        let info = wrapping_token_info(&store, &token);
        assert!(info.is_none());
    }
}
