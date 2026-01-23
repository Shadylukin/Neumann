//! Server configuration types.

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::error::{Result, ServerError};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind the server to.
    pub bind_addr: SocketAddr,
    /// TLS configuration (optional).
    pub tls: Option<TlsConfig>,
    /// Authentication configuration (optional).
    pub auth: Option<AuthConfig>,
    /// Maximum message size in bytes.
    pub max_message_size: usize,
    /// Maximum upload size for blob service in bytes.
    pub max_upload_size: usize,
    /// Enable gRPC-web support for browser clients.
    pub enable_grpc_web: bool,
    /// Enable reflection service for debugging.
    pub enable_reflection: bool,
    /// Blob streaming chunk size.
    pub blob_chunk_size: usize,
    /// Channel capacity for streaming responses (backpressure control).
    pub stream_channel_capacity: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:9200".parse().unwrap_or_else(|_| {
                // Safe fallback - this should never fail for a valid literal
                SocketAddr::from(([127, 0, 0, 1], 9200))
            }),
            tls: None,
            auth: None,
            max_message_size: 64 * 1024 * 1024, // 64MB
            max_upload_size: 512 * 1024 * 1024, // 512MB
            enable_grpc_web: true,
            enable_reflection: true,
            blob_chunk_size: 64 * 1024,  // 64KB chunks for streaming
            stream_channel_capacity: 32, // Bounded channel for backpressure
        }
    }
}

impl ServerConfig {
    /// Create a new server configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the bind address.
    #[must_use]
    pub fn with_bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Set TLS configuration.
    #[must_use]
    pub fn with_tls(mut self, tls: TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Set authentication configuration.
    #[must_use]
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Set maximum message size.
    #[must_use]
    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    /// Enable or disable gRPC-web support.
    #[must_use]
    pub fn with_grpc_web(mut self, enabled: bool) -> Self {
        self.enable_grpc_web = enabled;
        self
    }

    /// Enable or disable reflection service.
    #[must_use]
    pub fn with_reflection(mut self, enabled: bool) -> Self {
        self.enable_reflection = enabled;
        self
    }

    /// Set blob streaming chunk size.
    #[must_use]
    pub fn with_blob_chunk_size(mut self, size: usize) -> Self {
        self.blob_chunk_size = size;
        self
    }

    /// Set the maximum upload size for blob service.
    #[must_use]
    pub fn with_max_upload_size(mut self, size: usize) -> Self {
        self.max_upload_size = size;
        self
    }

    /// Set the channel capacity for streaming responses.
    ///
    /// Lower values provide better backpressure at the cost of throughput.
    /// Higher values allow more buffering but may use more memory.
    #[must_use]
    pub fn with_stream_channel_capacity(mut self, capacity: usize) -> Self {
        self.stream_channel_capacity = capacity;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_message_size == 0 {
            return Err(ServerError::Config(
                "max_message_size must be greater than 0".to_string(),
            ));
        }

        if self.blob_chunk_size == 0 {
            return Err(ServerError::Config(
                "blob_chunk_size must be greater than 0".to_string(),
            ));
        }

        if self.stream_channel_capacity == 0 {
            return Err(ServerError::Config(
                "stream_channel_capacity must be greater than 0".to_string(),
            ));
        }

        if let Some(ref tls) = self.tls {
            tls.validate()?;
        }

        if let Some(ref auth) = self.auth {
            auth.validate()?;
        }

        Ok(())
    }
}

/// TLS configuration for secure connections.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the certificate file (PEM format).
    pub cert_path: PathBuf,
    /// Path to the private key file (PEM format).
    pub key_path: PathBuf,
    /// Optional path to CA certificate for client authentication.
    pub ca_cert_path: Option<PathBuf>,
    /// Require client certificates.
    pub require_client_cert: bool,
}

impl TlsConfig {
    /// Create a new TLS configuration.
    #[must_use]
    pub fn new(cert_path: PathBuf, key_path: PathBuf) -> Self {
        Self {
            cert_path,
            key_path,
            ca_cert_path: None,
            require_client_cert: false,
        }
    }

    /// Set CA certificate for client authentication.
    #[must_use]
    pub fn with_ca_cert(mut self, path: PathBuf) -> Self {
        self.ca_cert_path = Some(path);
        self
    }

    /// Require client certificates.
    #[must_use]
    pub fn with_required_client_cert(mut self, required: bool) -> Self {
        self.require_client_cert = required;
        self
    }

    /// Validate the TLS configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.cert_path.exists() {
            return Err(ServerError::Config(format!(
                "certificate file not found: {}",
                self.cert_path.display()
            )));
        }

        if !self.key_path.exists() {
            return Err(ServerError::Config(format!(
                "key file not found: {}",
                self.key_path.display()
            )));
        }

        if let Some(ref ca_path) = self.ca_cert_path {
            if !ca_path.exists() {
                return Err(ServerError::Config(format!(
                    "CA certificate file not found: {}",
                    ca_path.display()
                )));
            }
        }

        Ok(())
    }
}

/// Authentication configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// API keys that are allowed to access the server.
    pub api_keys: Vec<ApiKey>,
    /// Header name for API key authentication.
    pub api_key_header: String,
    /// Allow unauthenticated access (for development).
    pub allow_anonymous: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            api_key_header: "x-api-key".to_string(),
            allow_anonymous: false,
        }
    }
}

impl AuthConfig {
    /// Create a new authentication configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an API key.
    #[must_use]
    pub fn with_api_key(mut self, key: ApiKey) -> Self {
        self.api_keys.push(key);
        self
    }

    /// Set the API key header name.
    #[must_use]
    pub fn with_header(mut self, header: String) -> Self {
        self.api_key_header = header;
        self
    }

    /// Allow anonymous access.
    #[must_use]
    pub fn with_anonymous(mut self, allowed: bool) -> Self {
        self.allow_anonymous = allowed;
        self
    }

    /// Validate the authentication configuration.
    pub fn validate(&self) -> Result<()> {
        if self.api_key_header.is_empty() {
            return Err(ServerError::Config(
                "api_key_header cannot be empty".to_string(),
            ));
        }

        if !self.allow_anonymous && self.api_keys.is_empty() {
            return Err(ServerError::Config(
                "at least one API key required when anonymous access is disabled".to_string(),
            ));
        }

        for key in &self.api_keys {
            key.validate()?;
        }

        Ok(())
    }

    /// Check if an API key is valid and return the associated identity.
    ///
    /// Uses constant-time comparison to prevent timing attacks, including
    /// key length. We iterate through all keys regardless of whether a
    /// match was found to prevent timing side channels.
    #[must_use]
    pub fn validate_key(&self, key: &str) -> Option<&str> {
        let key_bytes = key.as_bytes();

        // To prevent timing attacks, we must:
        // 1. Always check all keys (no early return on match)
        // 2. Use constant-time comparison that doesn't leak length
        let mut found_identity: Option<&str> = None;

        for api_key in &self.api_keys {
            let stored_bytes = api_key.key.as_bytes();

            // Constant-time length-safe comparison:
            // We compare byte-by-byte up to the longer key length,
            // padding the shorter one with zeros (which won't match printable keys).
            // The result is true only if lengths match AND all bytes match.
            let max_len = stored_bytes.len().max(key_bytes.len());

            let mut matches: u8 = 1;

            // Compare overlapping portion in constant time
            for i in 0..max_len {
                let stored_byte = if i < stored_bytes.len() {
                    stored_bytes[i]
                } else {
                    0
                };
                let key_byte = if i < key_bytes.len() { key_bytes[i] } else { 0 };
                matches &= u8::from(stored_byte == key_byte);
            }

            // Length must also match (constant-time via bitwise)
            let lengths_match = u8::from(stored_bytes.len() == key_bytes.len());
            matches &= lengths_match;

            // If match found, record identity (but don't return early)
            if matches == 1 {
                found_identity = Some(api_key.identity.as_str());
            }
        }

        found_identity
    }
}

/// An API key with associated identity.
#[derive(Debug, Clone)]
pub struct ApiKey {
    /// The API key value.
    pub key: String,
    /// The identity associated with this key.
    pub identity: String,
    /// Optional description for this key.
    pub description: Option<String>,
}

impl ApiKey {
    /// Create a new API key.
    #[must_use]
    pub fn new(key: String, identity: String) -> Self {
        Self {
            key,
            identity,
            description: None,
        }
    }

    /// Set description for this key.
    #[must_use]
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Validate the API key.
    pub fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(ServerError::Config("API key cannot be empty".to_string()));
        }

        if self.key.len() < 16 {
            return Err(ServerError::Config(
                "API key must be at least 16 characters".to_string(),
            ));
        }

        if self.identity.is_empty() {
            return Err(ServerError::Config(
                "API key identity cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_addr.port(), 9200);
        assert!(config.tls.is_none());
        assert!(config.auth.is_none());
        assert!(config.enable_grpc_web);
        assert!(config.enable_reflection);
    }

    #[test]
    fn test_config_builder() {
        let config = ServerConfig::new()
            .with_bind_addr("0.0.0.0:8080".parse().unwrap())
            .with_max_message_size(128 * 1024 * 1024)
            .with_grpc_web(false)
            .with_reflection(false)
            .with_blob_chunk_size(32 * 1024);

        assert_eq!(config.bind_addr.port(), 8080);
        assert_eq!(config.max_message_size, 128 * 1024 * 1024);
        assert!(!config.enable_grpc_web);
        assert!(!config.enable_reflection);
        assert_eq!(config.blob_chunk_size, 32 * 1024);
    }

    #[test]
    fn test_config_validation() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = ServerConfig::new().with_max_message_size(0);
        assert!(invalid_config.validate().is_err());

        let invalid_config = ServerConfig::new().with_blob_chunk_size(0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_auth_config_validation() {
        // Valid with anonymous access
        let auth = AuthConfig::new().with_anonymous(true);
        assert!(auth.validate().is_ok());

        // Invalid: no keys and no anonymous
        let auth = AuthConfig::new().with_anonymous(false);
        assert!(auth.validate().is_err());

        // Valid with API key
        let auth = AuthConfig::new()
            .with_anonymous(false)
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ));
        assert!(auth.validate().is_ok());
    }

    #[test]
    fn test_api_key_validation() {
        // Valid key
        let key = ApiKey::new("test-api-key-12345678".to_string(), "user:test".to_string());
        assert!(key.validate().is_ok());

        // Key too short
        let key = ApiKey::new("short".to_string(), "user:test".to_string());
        assert!(key.validate().is_err());

        // Empty identity
        let key = ApiKey::new("test-api-key-12345678".to_string(), String::new());
        assert!(key.validate().is_err());
    }

    #[test]
    fn test_validate_key() {
        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));

        assert_eq!(
            auth.validate_key("test-api-key-12345678"),
            Some("user:alice")
        );
        assert_eq!(auth.validate_key("wrong-key"), None);
    }

    #[test]
    fn test_validate_key_different_lengths() {
        // Test that keys of different lengths don't match (constant-time)
        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));

        // Shorter key
        assert_eq!(auth.validate_key("test-api-key"), None);
        // Longer key
        assert_eq!(auth.validate_key("test-api-key-12345678-extra"), None);
        // Same length, wrong content
        assert_eq!(auth.validate_key("test-api-key-XXXXXXXX"), None);
        // Exact match
        assert_eq!(
            auth.validate_key("test-api-key-12345678"),
            Some("user:alice")
        );
    }

    #[test]
    fn test_validate_key_multiple_keys() {
        // Verify all keys are checked (constant-time)
        let auth = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "first-key-12345678".to_string(),
                "user:first".to_string(),
            ))
            .with_api_key(ApiKey::new(
                "second-key-1234567".to_string(),
                "user:second".to_string(),
            ))
            .with_api_key(ApiKey::new(
                "third-key-12345678".to_string(),
                "user:third".to_string(),
            ));

        assert_eq!(auth.validate_key("first-key-12345678"), Some("user:first"));
        assert_eq!(auth.validate_key("second-key-1234567"), Some("user:second"));
        assert_eq!(auth.validate_key("third-key-12345678"), Some("user:third"));
        assert_eq!(auth.validate_key("unknown-key-12345"), None);
    }

    #[test]
    fn test_tls_config() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        );
        assert!(tls.validate().is_ok());

        // Non-existent cert file
        let tls = TlsConfig::new(
            PathBuf::from("/nonexistent/cert.pem"),
            key_file.path().to_path_buf(),
        );
        assert!(tls.validate().is_err());
    }

    #[test]
    fn test_api_key_with_description() {
        let key = ApiKey::new("test-api-key-12345678".to_string(), "user:test".to_string())
            .with_description("Test API key".to_string());

        assert_eq!(key.description, Some("Test API key".to_string()));
    }

    #[test]
    fn test_tls_config_with_ca() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        let ca_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_ca_cert(ca_file.path().to_path_buf())
        .with_required_client_cert(true);

        assert!(tls.validate().is_ok());
        assert!(tls.require_client_cert);
    }

    #[test]
    fn test_auth_config_with_header() {
        let auth = AuthConfig::new()
            .with_header("Authorization".to_string())
            .with_anonymous(true);

        assert_eq!(auth.api_key_header, "Authorization");
        assert!(auth.validate().is_ok());
    }

    #[test]
    fn test_empty_header_validation() {
        let auth = AuthConfig {
            api_keys: Vec::new(),
            api_key_header: String::new(),
            allow_anonymous: true,
        };
        assert!(auth.validate().is_err());
    }
}
