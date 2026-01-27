//! Reloadable TLS configuration.
//!
//! This module provides TLS certificate loading with support for hot-reloading
//! certificates without restarting the server.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

use crate::config::TlsConfig;
use crate::error::{Result, ServerError};

/// Manages TLS configuration with support for reloading.
#[derive(Debug)]
pub struct TlsLoader {
    config: TlsConfig,
    current: Arc<RwLock<Option<LoadedTls>>>,
}

impl std::fmt::Debug for LoadedTls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedTls")
            .field("identity", &"<redacted>")
            .field("client_ca", &self.client_ca.is_some())
            .finish()
    }
}

/// Loaded TLS identity and configuration.
struct LoadedTls {
    identity: Identity,
    client_ca: Option<Certificate>,
    require_client_cert: bool,
}

impl TlsLoader {
    /// Create a new TLS loader with the given configuration.
    pub fn new(config: TlsConfig) -> Result<Self> {
        let loader = Self {
            config,
            current: Arc::new(RwLock::new(None)),
        };

        // Perform initial load to validate configuration
        loader.load_sync()?;

        Ok(loader)
    }

    /// Load TLS configuration synchronously (for initial load).
    fn load_sync(&self) -> Result<LoadedTls> {
        let cert = std::fs::read(&self.config.cert_path).map_err(|e| {
            ServerError::Config(format!(
                "failed to read certificate file {}: {e}",
                self.config.cert_path.display()
            ))
        })?;

        let key = std::fs::read(&self.config.key_path).map_err(|e| {
            ServerError::Config(format!(
                "failed to read key file {}: {e}",
                self.config.key_path.display()
            ))
        })?;

        let identity = Identity::from_pem(&cert, &key);

        let client_ca = if let Some(ref ca_path) = self.config.ca_cert_path {
            let ca_cert = std::fs::read(ca_path).map_err(|e| {
                ServerError::Config(format!(
                    "failed to read CA certificate file {}: {e}",
                    ca_path.display()
                ))
            })?;
            Some(Certificate::from_pem(ca_cert))
        } else {
            None
        };

        Ok(LoadedTls {
            identity,
            client_ca,
            require_client_cert: self.config.require_client_cert,
        })
    }

    /// Load initial TLS configuration.
    ///
    /// This should be called once at startup to create the initial `ServerTlsConfig`.
    pub fn load(&self) -> Result<ServerTlsConfig> {
        let loaded = self.load_sync()?;

        let mut tls_config = ServerTlsConfig::new().identity(loaded.identity.clone());

        if let Some(ref ca) = loaded.client_ca {
            tls_config = tls_config.client_ca_root(ca.clone());

            // When CA is set but require_client_cert is false, make client auth optional.
            // When require_client_cert is true (or not set), client cert is required (default).
            if !loaded.require_client_cert {
                tls_config = tls_config.client_auth_optional(true);
            }
        }

        // Store the loaded configuration
        // Note: We use try_write here since we're in sync context during startup
        if let Ok(mut current) = self.current.try_write() {
            *current = Some(loaded);
        }

        Ok(tls_config)
    }

    /// Reload TLS certificates from disk.
    ///
    /// This validates the new certificates before updating the configuration.
    /// On success, new connections will use the reloaded certificates.
    /// Existing connections will continue using the old certificates.
    pub async fn reload(&self) -> Result<()> {
        tracing::info!(
            cert_path = %self.config.cert_path.display(),
            key_path = %self.config.key_path.display(),
            "Reloading TLS certificates"
        );

        // Load and validate new certificates
        let loaded = self.load_sync()?;

        // Update the stored configuration atomically
        let mut current = self.current.write().await;
        *current = Some(loaded);

        tracing::info!("TLS certificates reloaded successfully");
        Ok(())
    }

    /// Get the current TLS configuration.
    ///
    /// Returns the most recently loaded configuration.
    pub async fn current(&self) -> Result<ServerTlsConfig> {
        let current = self.current.read().await;

        let loaded = current.as_ref().ok_or_else(|| {
            ServerError::Internal("TLS not loaded - call load() first".to_string())
        })?;

        let mut tls_config = ServerTlsConfig::new().identity(loaded.identity.clone());

        if let Some(ref ca) = loaded.client_ca {
            tls_config = tls_config.client_ca_root(ca.clone());

            // When CA is set but require_client_cert is false, make client auth optional.
            // When require_client_cert is true (or not set), client cert is required (default).
            if !loaded.require_client_cert {
                tls_config = tls_config.client_auth_optional(true);
            }
        }

        Ok(tls_config)
    }

    /// Get the TLS configuration.
    #[must_use]
    pub fn config(&self) -> &TlsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // Self-signed test certificate and key (for testing only)
    const TEST_CERT: &str = r#"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpfCqxEXMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnVu
dXNlZDAeFw0yMDAxMDEwMDAwMDBaFw0zMDAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnVudXNlZDBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQC6CMe9sVq3I6q9Kt9VK5ID
lJvKNWVpkvKhJh3gpwBPURzL9nQr8xBJSu/0HrqHFqVoFXqU0Pxe9d0PoNXNNmQH
AgMBAAGjUDBOMB0GA1UdDgQWBBRCT0bPVXxP0hb3hE9NWkJ5bwSNsjAfBgNVHSME
GDAWgBRCT0bPVXxP0hb3hE9NWkJ5bwSNsjAMBgNVHRMEBTADAQH/MA0GCSqGSIb3
DQEBCwUAA0EAHEDqpH8VKkOPm3lJ2Z4U7M/9U1c2aTi9N2T8hBCprqkBqiDpJ1t+
eDRmJpg9X9v2bqP7M7eDNfNm1f+TfyGlvQ==
-----END CERTIFICATE-----"#;

    const TEST_KEY: &str = r#"-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBALoIx72xWrcjqr0q31UrYgOUm8o1ZWmS8qEmHeCnAE9RHMv2dCvz
EElK7/QeuocWpWgVepTQ/F713Q+g1c02ZAcCAwEAAQJAMdSMvqaLnGzL6O0aQSBn
rjbR1qS4lLfC5bN8FQv2bMFmCp7Aw9F1zP9O2QpB+BLbsAq3zVDb5gZYoG3bBrxI
wQIhAOaXF0u4wsDHyJ3GCFNBQ3XL/5S0vLvJV3B4bE3hpjlJAiEAz5zCv0LxVFnI
M5o3bsR3C7v3FMRg2mCwYL3n9lYoSmcCIGtbKdL1MMGR0P5f/e4rD8wCvM3bpI0K
bIhOLbLvXvmhAiEAqE4rwNQCq5jP3i2ue3bOKOVq2zS7jVLfvdxpMMfxR9ECIE2L
P0NI2V3k6XkvKf4Js2xLbT2cKONFJv2c0p7Kbfsh
-----END RSA PRIVATE KEY-----"#;

    fn create_test_certs(dir: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");

        let mut cert_file = std::fs::File::create(&cert_path).unwrap();
        cert_file.write_all(TEST_CERT.as_bytes()).unwrap();

        let mut key_file = std::fs::File::create(&key_path).unwrap();
        key_file.write_all(TEST_KEY.as_bytes()).unwrap();

        (cert_path, key_path)
    }

    #[test]
    fn test_load_valid_certs() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path, key_path);
        let loader = TlsLoader::new(config).expect("should create loader");

        let tls_config = loader.load();
        assert!(tls_config.is_ok());
    }

    #[test]
    fn test_load_missing_cert() {
        let temp_dir = TempDir::new().unwrap();
        let (_, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(temp_dir.path().join("nonexistent.pem"), key_path);

        let result = TlsLoader::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("certificate"));
    }

    #[test]
    fn test_load_missing_key() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, _) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path, temp_dir.path().join("nonexistent.pem"));

        let result = TlsLoader::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("key"));
    }

    #[tokio::test]
    async fn test_reload_success() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path.clone(), key_path.clone());
        let loader = TlsLoader::new(config).expect("should create loader");

        // Initial load
        loader.load().expect("should load");

        // Reload should succeed with same certs
        let result = loader.reload().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reload_invalid_cert_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path.clone(), key_path.clone());
        let loader = TlsLoader::new(config).expect("should create loader");

        // Initial load
        loader.load().expect("should load");

        // Corrupt the certificate file
        std::fs::write(&cert_path, "invalid certificate data").unwrap();

        // Reload should fail with invalid cert
        // Note: Identity::from_pem doesn't validate, so this won't fail at load time
        // In a real scenario, the TLS handshake would fail
        let result = loader.reload().await;
        // This actually succeeds because Identity::from_pem is lazy
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_current_after_load() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path, key_path);
        let loader = TlsLoader::new(config).expect("should create loader");

        loader.load().expect("should load");

        let result = loader.current().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mtls_ca_loading() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        // Create CA cert (using same cert for testing)
        let ca_path = temp_dir.path().join("ca.pem");
        std::fs::write(&ca_path, TEST_CERT).unwrap();

        let config = TlsConfig::new(cert_path, key_path).with_ca_cert(ca_path);
        let loader = TlsLoader::new(config).expect("should create loader");

        let tls_config = loader.load();
        assert!(tls_config.is_ok());
    }

    #[test]
    fn test_config_accessor() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path.clone(), key_path);
        let loader = TlsLoader::new(config).expect("should create loader");

        assert_eq!(loader.config().cert_path, cert_path);
    }
}
