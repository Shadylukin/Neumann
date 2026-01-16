//! TLS support for TCP transport.
//!
//! Provides wrapper functions to upgrade TCP streams to TLS.

#[cfg(feature = "tls")]
use std::io::BufReader;
#[cfg(feature = "tls")]
use std::path::Path;
#[cfg(feature = "tls")]
use std::sync::Arc;

#[cfg(feature = "tls")]
use tokio::net::TcpStream;
#[cfg(feature = "tls")]
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
#[cfg(feature = "tls")]
use tokio_rustls::rustls::{ClientConfig, RootCertStore, ServerConfig};
#[cfg(feature = "tls")]
use tokio_rustls::{TlsAcceptor, TlsConnector};

#[cfg(feature = "tls")]
use super::config::TlsConfig;
#[cfg(feature = "tls")]
use super::error::{TcpError, TcpResult};

/// TLS stream type for server connections.
#[cfg(feature = "tls")]
pub type ServerTlsStream = tokio_rustls::server::TlsStream<TcpStream>;

/// TLS stream type for client connections.
#[cfg(feature = "tls")]
pub type ClientTlsStream = tokio_rustls::client::TlsStream<TcpStream>;

/// Load certificates from a PEM file.
#[cfg(feature = "tls")]
fn load_certs(path: &Path) -> TcpResult<Vec<CertificateDer<'static>>> {
    let file = std::fs::File::open(path).map_err(|e| {
        TcpError::TlsError(format!(
            "failed to open cert file {}: {}",
            path.display(),
            e
        ))
    })?;
    let mut reader = BufReader::new(file);

    let certs: Vec<_> = rustls_pemfile::certs(&mut reader)
        .filter_map(|r| r.ok())
        .collect();

    if certs.is_empty() {
        return Err(TcpError::TlsError(format!(
            "no certificates found in {}",
            path.display()
        )));
    }

    Ok(certs)
}

/// Load a private key from a PEM file.
#[cfg(feature = "tls")]
fn load_private_key(path: &Path) -> TcpResult<PrivateKeyDer<'static>> {
    let file = std::fs::File::open(path).map_err(|e| {
        TcpError::TlsError(format!("failed to open key file {}: {}", path.display(), e))
    })?;
    let mut reader = BufReader::new(file);

    // Try to read PKCS8 key first, then RSA, then EC
    let keys: Vec<_> = rustls_pemfile::pkcs8_private_keys(&mut reader)
        .filter_map(|r| r.ok())
        .collect();

    if !keys.is_empty() {
        return Ok(PrivateKeyDer::Pkcs8(keys.into_iter().next().unwrap()));
    }

    // Reopen and try RSA
    let file = std::fs::File::open(path).map_err(|e| {
        TcpError::TlsError(format!("failed to open key file {}: {}", path.display(), e))
    })?;
    let mut reader = BufReader::new(file);
    let keys: Vec<_> = rustls_pemfile::rsa_private_keys(&mut reader)
        .filter_map(|r| r.ok())
        .collect();

    if !keys.is_empty() {
        return Ok(PrivateKeyDer::Pkcs1(keys.into_iter().next().unwrap()));
    }

    Err(TcpError::TlsError(format!(
        "no private key found in {}",
        path.display()
    )))
}

/// Wrap a TCP stream with TLS for server-side connections.
#[cfg(feature = "tls")]
pub async fn wrap_server(stream: TcpStream, config: &TlsConfig) -> TcpResult<ServerTlsStream> {
    let certs = load_certs(&config.cert_path)?;
    let key = load_private_key(&config.key_path)?;

    let server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| TcpError::TlsError(format!("TLS config error: {}", e)))?;

    let acceptor = TlsAcceptor::from(Arc::new(server_config));

    acceptor
        .accept(stream)
        .await
        .map_err(|e| TcpError::TlsError(format!("TLS handshake failed: {}", e)))
}

/// Wrap a TCP stream with TLS for client-side connections.
#[cfg(feature = "tls")]
pub async fn wrap_client(
    stream: TcpStream,
    config: &TlsConfig,
    server_name: &str,
) -> TcpResult<ClientTlsStream> {
    let mut root_store = RootCertStore::empty();

    // Load CA certs if provided
    if let Some(ca_path) = &config.ca_cert_path {
        let ca_certs = load_certs(ca_path)?;
        for cert in ca_certs {
            root_store
                .add(cert)
                .map_err(|e| TcpError::TlsError(format!("failed to add CA cert: {}", e)))?;
        }
    }

    // SECURITY: Use should_verify() which always returns true in release builds
    let client_config = if config.should_verify() {
        ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth()
    } else {
        // For testing only - skip certificate verification
        // This branch is unreachable in release builds
        ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(InsecureVerifier))
            .with_no_client_auth()
    };

    let connector = TlsConnector::from(Arc::new(client_config));

    let domain = ServerName::try_from(server_name.to_string())
        .map_err(|_| TcpError::TlsError(format!("invalid server name: {}", server_name)))?;

    connector
        .connect(domain, stream)
        .await
        .map_err(|e| TcpError::TlsError(format!("TLS handshake failed: {}", e)))
}

/// Insecure certificate verifier for testing.
#[cfg(feature = "tls")]
#[derive(Debug)]
struct InsecureVerifier;

#[cfg(feature = "tls")]
impl tokio_rustls::rustls::client::danger::ServerCertVerifier for InsecureVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: tokio_rustls::rustls::pki_types::UnixTime,
    ) -> Result<tokio_rustls::rustls::client::danger::ServerCertVerified, tokio_rustls::rustls::Error>
    {
        Ok(tokio_rustls::rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &tokio_rustls::rustls::DigitallySignedStruct,
    ) -> Result<
        tokio_rustls::rustls::client::danger::HandshakeSignatureValid,
        tokio_rustls::rustls::Error,
    > {
        Ok(tokio_rustls::rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &tokio_rustls::rustls::DigitallySignedStruct,
    ) -> Result<
        tokio_rustls::rustls::client::danger::HandshakeSignatureValid,
        tokio_rustls::rustls::Error,
    > {
        Ok(tokio_rustls::rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<tokio_rustls::rustls::SignatureScheme> {
        vec![
            tokio_rustls::rustls::SignatureScheme::RSA_PKCS1_SHA256,
            tokio_rustls::rustls::SignatureScheme::RSA_PKCS1_SHA384,
            tokio_rustls::rustls::SignatureScheme::RSA_PKCS1_SHA512,
            tokio_rustls::rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            tokio_rustls::rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            tokio_rustls::rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            tokio_rustls::rustls::SignatureScheme::RSA_PSS_SHA256,
            tokio_rustls::rustls::SignatureScheme::RSA_PSS_SHA384,
            tokio_rustls::rustls::SignatureScheme::RSA_PSS_SHA512,
            tokio_rustls::rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(all(test, feature = "tls"))]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn test_cert_path() -> PathBuf {
        // Tests would use temp certs generated by rcgen
        PathBuf::from("/tmp/test_cert.pem")
    }

    fn test_key_path() -> PathBuf {
        PathBuf::from("/tmp/test_key.pem")
    }

    #[test]
    fn test_tls_config_creation() {
        let config = TlsConfig::new(test_cert_path(), test_key_path());
        assert!(!config.require_client_auth);
        assert!(!config.insecure_skip_verify);
    }

    #[test]
    fn test_tls_config_with_ca() {
        let config = TlsConfig::new(test_cert_path(), test_key_path()).with_ca_cert("/tmp/ca.pem");
        assert!(config.ca_cert_path.is_some());
    }

    #[test]
    fn test_tls_config_with_client_auth() {
        let config = TlsConfig::new(test_cert_path(), test_key_path()).with_client_auth();
        assert!(config.require_client_auth);
    }

    #[test]
    fn test_load_certs_missing_file() {
        let result = load_certs(Path::new("/nonexistent/cert.pem"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_private_key_missing_file() {
        let result = load_private_key(Path::new("/nonexistent/key.pem"));
        assert!(result.is_err());
    }
}
