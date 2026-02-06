// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
/// Returns when either signal is received, or if signal registration fails.
///
/// # Errors
///
/// Returns an error if signal handlers cannot be registered. This should only
/// happen if the operating system rejects the signal registration, which is rare.
pub async fn wait_for_shutdown_signal() -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sigint = signal(SignalKind::interrupt())?;

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
        tokio::signal::ctrl_c().await?;
        tracing::info!("Received Ctrl+C");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TlsConfig;
    use std::path::PathBuf;

    // Use pre-generated valid certificates from fixtures directory
    fn valid_cert_paths() -> (PathBuf, PathBuf) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures");
        (dir.join("valid_cert.pem"), dir.join("valid_key.pem"))
    }

    #[tokio::test]
    async fn test_sighup_handler_registration() {
        let (cert_path, key_path) = valid_cert_paths();
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
