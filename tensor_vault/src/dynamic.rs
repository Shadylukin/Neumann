// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dynamic (short-lived) secret generation.

use rand::RngCore;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{Result, VaultError};

/// Storage prefix for dynamic secret metadata.
const DYN_PREFIX: &str = "_vdyn:";

/// Template for generating dynamic secrets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretTemplate {
    /// Generate a random password with character-class constraints.
    Password(PasswordConfig),
    /// Generate a random token in hex or base64 encoding.
    Token(TokenConfig),
    /// Generate a prefixed API key with a random suffix.
    ApiKey(ApiKeyConfig),
}

/// Configuration for password generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordConfig {
    /// Desired password length (minimum 4 is enforced).
    pub length: usize,
    /// Character set to draw from.
    pub charset: PasswordCharset,
    /// Whether at least one uppercase letter is required.
    pub require_uppercase: bool,
    /// Whether at least one digit is required.
    pub require_digit: bool,
    /// Whether at least one special character is required.
    pub require_special: bool,
}

impl Default for PasswordConfig {
    fn default() -> Self {
        Self {
            length: 32,
            charset: PasswordCharset::Alphanumeric,
            require_uppercase: true,
            require_digit: true,
            require_special: false,
        }
    }
}

/// Character sets for password generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PasswordCharset {
    /// Letters (a-z, A-Z) and digits (0-9).
    Alphanumeric,
    /// Letters, digits, and special characters (!@#$%^&*()-_=+).
    AlphanumericSpecial,
    /// Hexadecimal characters (0-9, a-f).
    Hex,
}

/// Configuration for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    /// Number of random bytes to generate.
    pub length: usize,
    /// Encoding used for the output string.
    pub encoding: TokenEncoding,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            length: 32,
            encoding: TokenEncoding::Hex,
        }
    }
}

/// Encoding format for tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenEncoding {
    /// Lowercase hexadecimal encoding (2 characters per byte).
    Hex,
    /// URL-safe base64 encoding without padding.
    Base64,
}

/// Configuration for API key generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Static prefix prepended to the key (e.g. "nk").
    pub prefix: String,
    /// Number of random bytes for the hex-encoded suffix.
    pub random_length: usize,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            prefix: "nk".to_string(),
            random_length: 32,
        }
    }
}

/// Metadata for a generated dynamic secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicSecretMetadata {
    /// Unique identifier for this dynamic secret.
    pub secret_id: String,
    /// Template type used to generate the secret ("password", "token", or "api_key").
    pub template_type: String,
    /// Identity of the entity that requested this secret.
    pub requester: String,
    /// Expiration timestamp in milliseconds since epoch.
    pub expires_at_ms: i64,
    /// Whether the secret can only be read once before invalidation.
    pub one_time: bool,
    /// Whether the secret has already been consumed.
    pub consumed: bool,
}

fn generate_id() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("dyn_{}", hex_encode(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Generate a secret value from a template.
pub fn generate_from_template(template: &SecretTemplate) -> String {
    match template {
        SecretTemplate::Password(config) => generate_password(config),
        SecretTemplate::Token(config) => generate_token(config),
        SecretTemplate::ApiKey(config) => generate_api_key(config),
    }
}

fn generate_password(config: &PasswordConfig) -> String {
    let length = config.length.max(4);
    let charset: &[u8] = match config.charset {
        PasswordCharset::Alphanumeric => {
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        },
        PasswordCharset::AlphanumericSpecial => {
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
        },
        PasswordCharset::Hex => b"0123456789abcdef",
    };

    let mut rng = rand::thread_rng();
    let mut password: Vec<u8> = Vec::with_capacity(length);

    // Ensure required character classes
    let mut required = Vec::new();
    if config.require_uppercase {
        let upper = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut idx = [0u8; 1];
        rng.fill_bytes(&mut idx);
        required.push(upper[(idx[0] as usize) % upper.len()]);
    }
    if config.require_digit {
        let digits = b"0123456789";
        let mut idx = [0u8; 1];
        rng.fill_bytes(&mut idx);
        required.push(digits[(idx[0] as usize) % digits.len()]);
    }
    if config.require_special {
        let special = b"!@#$%^&*()-_=+";
        let mut idx = [0u8; 1];
        rng.fill_bytes(&mut idx);
        required.push(special[(idx[0] as usize) % special.len()]);
    }

    // Fill remainder with random chars
    let remaining = length.saturating_sub(required.len());
    for _ in 0..remaining {
        let mut idx = [0u8; 1];
        rng.fill_bytes(&mut idx);
        password.push(charset[(idx[0] as usize) % charset.len()]);
    }
    password.extend_from_slice(&required);

    // Shuffle using Fisher-Yates
    for i in (1..password.len()).rev() {
        let mut j_bytes = [0u8; 4];
        rng.fill_bytes(&mut j_bytes);
        let j = (u32::from_le_bytes(j_bytes) as usize) % (i + 1);
        password.swap(i, j);
    }

    String::from_utf8(password).unwrap_or_default()
}

fn generate_token(config: &TokenConfig) -> String {
    let byte_len = config.length;
    let mut bytes = vec![0u8; byte_len];
    rand::thread_rng().fill_bytes(&mut bytes);

    match config.encoding {
        TokenEncoding::Hex => hex_encode(&bytes),
        TokenEncoding::Base64 => {
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &bytes)
        },
    }
}

fn generate_api_key(config: &ApiKeyConfig) -> String {
    let mut bytes = vec![0u8; config.random_length];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("{}_{}", config.prefix, hex_encode(&bytes))
}

/// Store dynamic secret metadata.
pub fn store_metadata(
    store: &TensorStore,
    id: &str,
    template: &SecretTemplate,
    requester: &str,
    expires_at_ms: i64,
    one_time: bool,
) -> Result<()> {
    let key = format!("{DYN_PREFIX}{id}");
    let mut tensor = TensorData::new();

    let template_type = match template {
        SecretTemplate::Password(_) => "password",
        SecretTemplate::Token(_) => "token",
        SecretTemplate::ApiKey(_) => "api_key",
    };

    tensor.set(
        "_template_type",
        TensorValue::Scalar(ScalarValue::String(template_type.into())),
    );
    tensor.set(
        "_requester",
        TensorValue::Scalar(ScalarValue::String(requester.into())),
    );
    tensor.set(
        "_expires_at",
        TensorValue::Scalar(ScalarValue::Int(expires_at_ms)),
    );
    tensor.set(
        "_one_time",
        TensorValue::Scalar(ScalarValue::Bool(one_time)),
    );
    tensor.set("_consumed", TensorValue::Scalar(ScalarValue::Bool(false)));

    store
        .put(&key, tensor)
        .map_err(|e| VaultError::StorageError(e.to_string()))
}

/// Mark a dynamic secret as consumed.
pub fn mark_consumed(store: &TensorStore, id: &str) -> Result<()> {
    let key = format!("{DYN_PREFIX}{id}");
    if let Ok(mut tensor) = store.get(&key) {
        tensor.set("_consumed", TensorValue::Scalar(ScalarValue::Bool(true)));
        store
            .put(&key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;
    }
    Ok(())
}

/// List dynamic secret metadata.
pub fn list_metadata(store: &TensorStore) -> Vec<DynamicSecretMetadata> {
    let mut results = Vec::new();
    for key in store.scan(DYN_PREFIX) {
        if let Some(id) = key.strip_prefix(DYN_PREFIX) {
            if let Ok(tensor) = store.get(&key) {
                if let Some(meta) = parse_metadata(id, &tensor) {
                    results.push(meta);
                }
            }
        }
    }
    results
}

/// Get metadata for a specific dynamic secret.
pub fn get_metadata(store: &TensorStore, id: &str) -> Option<DynamicSecretMetadata> {
    let key = format!("{DYN_PREFIX}{id}");
    store.get(&key).ok().and_then(|t| parse_metadata(id, &t))
}

fn parse_metadata(id: &str, tensor: &TensorData) -> Option<DynamicSecretMetadata> {
    let template_type = match tensor.get("_template_type") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let requester = match tensor.get("_requester") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let expires_at_ms = match tensor.get("_expires_at") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    let one_time = matches!(
        tensor.get("_one_time"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    );
    let consumed = matches!(
        tensor.get("_consumed"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    );

    Some(DynamicSecretMetadata {
        secret_id: id.to_string(),
        template_type,
        requester,
        expires_at_ms,
        one_time,
        consumed,
    })
}

/// Revoke (delete) a dynamic secret's metadata.
pub fn revoke_metadata(store: &TensorStore, id: &str) {
    let key = format!("{DYN_PREFIX}{id}");
    store.delete(&key).ok();
}

/// Generate a new secret ID.
pub fn new_secret_id() -> String {
    generate_id()
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    #[test]
    fn test_generate_password_default() {
        let config = PasswordConfig::default();
        let pwd = generate_password(&config);
        assert_eq!(pwd.len(), 32);
        assert!(pwd.chars().any(|c| c.is_uppercase()));
        assert!(pwd.chars().any(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_generate_password_hex() {
        let config = PasswordConfig {
            length: 16,
            charset: PasswordCharset::Hex,
            require_uppercase: false,
            require_digit: false,
            require_special: false,
        };
        let pwd = generate_password(&config);
        assert_eq!(pwd.len(), 16);
        assert!(pwd.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_generate_password_with_special() {
        let config = PasswordConfig {
            length: 20,
            charset: PasswordCharset::AlphanumericSpecial,
            require_uppercase: true,
            require_digit: true,
            require_special: true,
        };
        let pwd = generate_password(&config);
        assert_eq!(pwd.len(), 20);
    }

    #[test]
    fn test_generate_token_hex() {
        let config = TokenConfig {
            length: 16,
            encoding: TokenEncoding::Hex,
        };
        let token = generate_token(&config);
        assert_eq!(token.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_generate_token_base64() {
        let config = TokenConfig {
            length: 24,
            encoding: TokenEncoding::Base64,
        };
        let token = generate_token(&config);
        assert!(!token.is_empty());
    }

    #[test]
    fn test_generate_api_key() {
        let config = ApiKeyConfig {
            prefix: "test".to_string(),
            random_length: 16,
        };
        let key = generate_api_key(&config);
        assert!(key.starts_with("test_"));
        assert_eq!(key.len(), 5 + 32); // "test_" + 32 hex chars
    }

    #[test]
    fn test_generate_from_template() {
        let templates = vec![
            SecretTemplate::Password(PasswordConfig::default()),
            SecretTemplate::Token(TokenConfig::default()),
            SecretTemplate::ApiKey(ApiKeyConfig::default()),
        ];

        for template in templates {
            let value = generate_from_template(&template);
            assert!(!value.is_empty());
        }
    }

    #[test]
    fn test_store_and_list_metadata() {
        let store = TensorStore::new();
        let id = "test_dyn_001";

        store_metadata(
            &store,
            id,
            &SecretTemplate::Password(PasswordConfig::default()),
            "user:alice",
            now_ms() + 60_000,
            false,
        )
        .unwrap();

        let metas = list_metadata(&store);
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].secret_id, id);
        assert_eq!(metas[0].requester, "user:alice");
        assert!(!metas[0].one_time);
        assert!(!metas[0].consumed);
    }

    #[test]
    fn test_mark_consumed() {
        let store = TensorStore::new();
        let id = "test_dyn_002";

        store_metadata(
            &store,
            id,
            &SecretTemplate::Token(TokenConfig::default()),
            "user:bob",
            now_ms() + 60_000,
            true,
        )
        .unwrap();

        mark_consumed(&store, id).unwrap();

        let meta = get_metadata(&store, id).unwrap();
        assert!(meta.consumed);
    }

    #[test]
    fn test_revoke_metadata() {
        let store = TensorStore::new();
        let id = "test_dyn_003";

        store_metadata(
            &store,
            id,
            &SecretTemplate::ApiKey(ApiKeyConfig::default()),
            "user:carol",
            now_ms() + 60_000,
            false,
        )
        .unwrap();

        revoke_metadata(&store, id);
        assert!(get_metadata(&store, id).is_none());
    }

    #[test]
    fn test_get_metadata_nonexistent() {
        let store = TensorStore::new();
        assert!(get_metadata(&store, "nonexistent").is_none());
    }

    #[test]
    fn test_unique_ids() {
        let id1 = new_secret_id();
        let id2 = new_secret_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("dyn_"));
    }

    #[test]
    fn test_password_minimum_length() {
        let config = PasswordConfig {
            length: 1, // Below minimum
            charset: PasswordCharset::Alphanumeric,
            require_uppercase: true,
            require_digit: true,
            require_special: false,
        };
        let pwd = generate_password(&config);
        assert!(pwd.len() >= 4); // Minimum enforced
    }

    #[test]
    fn test_template_serialization() {
        let template = SecretTemplate::Password(PasswordConfig::default());
        let json = serde_json::to_string(&template).unwrap();
        let deserialized: SecretTemplate = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, SecretTemplate::Password(_)));
    }
}
