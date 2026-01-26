//! Unix signal handlers for the server.
//!
//! This module provides signal handling for graceful shutdown and
//! certificate reloading (SIGHUP).

use std::sync::Arc;

use crate::tls_loader::TlsLoader;

/// Register a SIGHUP handler for TLS certificate reloading.
///
/// When SIGHUP is received, the TLS loader will reload certificates from disk.
/// This allows updating certificates without restarting the server.
///
/// Note: Existing connections will continue using the old certificates.
/// New connections after reload will use the updated certificates.
#[cfg(unix)]
pub fn register_sighup_handler(tls_loader: Arc<TlsLoader>) {
    use tokio::signal::unix::{signal, SignalKind};

    tokio::spawn(async move {
        let mut stream = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to register SIGHUP handler: {}", e);
                return;
            },
        };

        tracing::info!("SIGHUP handler registered for TLS certificate reload");

        loop {
            stream.recv().await;
            tracing::info!("Received SIGHUP, reloading TLS certificates");

            match tls_loader.reload().await {
                Ok(()) => {
                    tracing::info!("TLS certificates reloaded successfully");
                },
                Err(e) => {
                    tracing::error!("Failed to reload TLS certificates: {}", e);
                },
            }
        }
    });
}

/// No-op SIGHUP handler for non-Unix platforms.
#[cfg(not(unix))]
pub fn register_sighup_handler(_tls_loader: Arc<TlsLoader>) {
    tracing::debug!("SIGHUP handler not available on this platform");
}

/// Wait for a shutdown signal (SIGTERM or SIGINT).
///
/// Returns when either signal is received.
///
/// # Panics
///
/// Panics if signal handlers cannot be registered. This should only happen
/// if the operating system rejects the signal registration, which is rare.
#[allow(clippy::expect_used)]
pub async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
        let mut sigint =
            signal(SignalKind::interrupt()).expect("failed to register SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("Received SIGTERM");
            }
            _ = sigint.recv() => {
                tracing::info!("Received SIGINT");
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for ctrl+c");
        tracing::info!("Received Ctrl+C");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TlsConfig;
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

    #[tokio::test]
    async fn test_sighup_handler_registration() {
        let temp_dir = TempDir::new().unwrap();
        let (cert_path, key_path) = create_test_certs(&temp_dir);

        let config = TlsConfig::new(cert_path, key_path);
        let loader = Arc::new(TlsLoader::new(config).expect("should create loader"));

        // This should not panic
        register_sighup_handler(loader);

        // Give the handler time to register
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    #[cfg(not(unix))]
    #[test]
    fn test_non_unix_noop() {
        // On non-Unix platforms, the handler should be a no-op
        // We can't easily test this, but we can verify it compiles
    }

    #[tokio::test]
    async fn test_shutdown_signal_setup() {
        // We can't easily test the actual signal handling,
        // but we can verify the function exists and compiles
        // Note: Don't actually wait for the signal in tests
    }
}
