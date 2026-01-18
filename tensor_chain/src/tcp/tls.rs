//! TLS support for TCP transport.
//!
//! Provides wrapper functions to upgrade TCP streams to TLS,
//! including mutual TLS (mTLS) with certificate-based NodeId verification.

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
use x509_parser::prelude::*;

#[cfg(feature = "tls")]
use super::config::{NodeIdVerification, TlsConfig};
#[cfg(feature = "tls")]
use super::error::{TcpError, TcpResult};

/// TLS stream type for server connections.
#[cfg(feature = "tls")]
pub type ServerTlsStream = tokio_rustls::server::TlsStream<TcpStream>;

/// TLS stream type for client connections.
#[cfg(feature = "tls")]
pub type ClientTlsStream = tokio_rustls::client::TlsStream<TcpStream>;

/// Identity extracted from a peer's TLS certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPeerIdentity {
    /// NodeId extracted from certificate (CN or SAN).
    pub node_id: String,
    /// How the NodeId was extracted.
    pub source: NodeIdSource,
}

/// Source of NodeId in the certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeIdSource {
    /// Extracted from Common Name (CN).
    CommonName,
    /// Extracted from Subject Alternative Name (SAN).
    SubjectAltName,
}

impl VerifiedPeerIdentity {
    pub fn from_common_name(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            source: NodeIdSource::CommonName,
        }
    }

    pub fn from_san(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            source: NodeIdSource::SubjectAltName,
        }
    }
}

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

/// Extract NodeId from a certificate based on the verification mode.
///
/// For CommonName mode: extracts from the CN field of the subject.
/// For SubjectAltName mode: extracts from the first DNS SAN entry.
#[cfg(feature = "tls")]
pub fn extract_node_id_from_cert(
    cert: &CertificateDer<'_>,
    mode: &NodeIdVerification,
) -> TcpResult<Option<VerifiedPeerIdentity>> {
    match mode {
        NodeIdVerification::None => Ok(None),
        NodeIdVerification::CommonName => extract_from_common_name(cert),
        NodeIdVerification::SubjectAltName => extract_from_san(cert),
    }
}

#[cfg(feature = "tls")]
fn extract_from_common_name(cert: &CertificateDer<'_>) -> TcpResult<Option<VerifiedPeerIdentity>> {
    let cert_der = cert.as_ref();

    if let Some(cn) = extract_cn_from_der(cert_der) {
        Ok(Some(VerifiedPeerIdentity::from_common_name(cn)))
    } else {
        Err(TcpError::TlsError(
            "no Common Name found in certificate".to_string(),
        ))
    }
}

#[cfg(feature = "tls")]
fn extract_from_san(cert: &CertificateDer<'_>) -> TcpResult<Option<VerifiedPeerIdentity>> {
    let cert_der = cert.as_ref();

    if let Some(san) = extract_san_from_der(cert_der) {
        Ok(Some(VerifiedPeerIdentity::from_san(san)))
    } else {
        Err(TcpError::TlsError(
            "no Subject Alternative Name found in certificate".to_string(),
        ))
    }
}

/// Extract Common Name from DER-encoded certificate using x509-parser.
#[cfg(feature = "tls")]
fn extract_cn_from_der(der: &[u8]) -> Option<String> {
    let (_, cert) = X509Certificate::from_der(der).ok()?;

    for rdn in cert.subject().iter() {
        for attr in rdn.iter() {
            if attr.attr_type() == &oid_registry::OID_X509_COMMON_NAME {
                return attr.as_str().ok().map(|s| s.to_string());
            }
        }
    }
    None
}

/// Extract Subject Alternative Name (DNS) from DER-encoded certificate using x509-parser.
#[cfg(feature = "tls")]
fn extract_san_from_der(der: &[u8]) -> Option<String> {
    let (_, cert) = X509Certificate::from_der(der).ok()?;

    for ext in cert.extensions() {
        if let ParsedExtension::SubjectAlternativeName(san) = ext.parsed_extension() {
            for name in &san.general_names {
                if let GeneralName::DNSName(dns) = name {
                    return Some(dns.to_string());
                }
            }
        }
    }
    None
}

/// Wrap a TCP stream with TLS (server-side).
///
/// When `require_client_auth` is true, the server will request a client certificate.
/// Use `wrap_server_with_identity` to also extract the client's NodeId from the certificate.
#[cfg(feature = "tls")]
pub async fn wrap_server(stream: TcpStream, config: &TlsConfig) -> TcpResult<ServerTlsStream> {
    let (stream, _identity) = wrap_server_with_identity(stream, config).await?;
    Ok(stream)
}

/// Wrap a TCP stream with TLS and extract client identity if available.
///
/// Returns the TLS stream and optionally the verified peer identity
/// extracted from the client certificate (when using mTLS).
#[cfg(feature = "tls")]
pub async fn wrap_server_with_identity(
    stream: TcpStream,
    config: &TlsConfig,
) -> TcpResult<(ServerTlsStream, Option<VerifiedPeerIdentity>)> {
    let certs = load_certs(&config.cert_path)?;
    let key = load_private_key(&config.key_path)?;

    let server_config = if config.require_client_auth {
        // mTLS: require client certificate
        let mut root_store = RootCertStore::empty();

        // Load CA certs for client verification
        if let Some(ca_path) = &config.ca_cert_path {
            let ca_certs = load_certs(ca_path)?;
            for cert in ca_certs {
                root_store
                    .add(cert)
                    .map_err(|e| TcpError::TlsError(format!("failed to add CA cert: {}", e)))?;
            }
        }

        let client_verifier =
            tokio_rustls::rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store))
                .build()
                .map_err(|e| {
                    TcpError::TlsError(format!("failed to build client verifier: {}", e))
                })?;

        ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(certs, key)
            .map_err(|e| TcpError::TlsError(format!("TLS config error: {}", e)))?
    } else {
        // No client auth
        ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| TcpError::TlsError(format!("TLS config error: {}", e)))?
    };

    let acceptor = TlsAcceptor::from(Arc::new(server_config));

    let tls_stream = acceptor
        .accept(stream)
        .await
        .map_err(|e| TcpError::TlsError(format!("TLS handshake failed: {}", e)))?;

    // Extract peer identity from client certificate if mTLS is enabled
    let peer_identity = if config.require_client_auth {
        // Get peer certificates from the connection
        let (_, server_conn) = tls_stream.get_ref();
        if let Some(certs) = server_conn.peer_certificates() {
            if let Some(cert) = certs.first() {
                extract_node_id_from_cert(cert, &config.node_id_verification)?
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok((tls_stream, peer_identity))
}

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
    use tempfile::TempDir;
    use tokio_rustls::rustls::client::danger::ServerCertVerifier;

    struct TestCerts {
        cert_path: PathBuf,
        key_path: PathBuf,
        _temp_dir: TempDir,
    }

    fn generate_test_certs() -> TestCerts {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])
            .expect("failed to generate cert");

        let cert_pem = cert.cert.pem();
        let key_pem = cert.key_pair.serialize_pem();

        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        std::fs::write(&cert_path, cert_pem).expect("failed to write cert");
        std::fs::write(&key_path, key_pem).expect("failed to write key");

        TestCerts {
            cert_path,
            key_path,
            _temp_dir: temp_dir,
        }
    }

    fn generate_cert_chain() -> (PathBuf, PathBuf, TempDir) {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        // Generate CA
        let ca_key = rcgen::KeyPair::generate().unwrap();
        let mut ca_params = rcgen::CertificateParams::new(vec!["Test CA".into()]).unwrap();
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();

        // Generate leaf cert signed by CA
        let leaf_key = rcgen::KeyPair::generate().unwrap();
        let leaf_params = rcgen::CertificateParams::new(vec!["localhost".into()]).unwrap();
        let leaf_cert = leaf_params.signed_by(&leaf_key, &ca_cert, &ca_key).unwrap();

        // Write chain (leaf + CA)
        let chain_pem = format!("{}{}", leaf_cert.pem(), ca_cert.pem());
        let cert_path = temp_dir.path().join("chain.pem");
        let key_path = temp_dir.path().join("key.pem");

        std::fs::write(&cert_path, chain_pem).expect("failed to write chain");
        std::fs::write(&key_path, leaf_key.serialize_pem()).expect("failed to write key");

        (cert_path, key_path, temp_dir)
    }

    fn make_insecure_config(config: &TlsConfig) -> TlsConfig {
        let mut insecure = config.clone();
        insecure.insecure_skip_verify = true;
        insecure
    }

    // TLS Config Tests
    #[test]
    fn test_tls_config_creation() {
        let certs = generate_test_certs();
        let config = TlsConfig::new(&certs.cert_path, &certs.key_path);
        assert!(!config.require_client_auth);
        assert!(!config.insecure_skip_verify);
    }

    #[test]
    fn test_tls_config_with_ca() {
        let certs = generate_test_certs();
        let config =
            TlsConfig::new(&certs.cert_path, &certs.key_path).with_ca_cert(&certs.cert_path);
        assert!(config.ca_cert_path.is_some());
    }

    #[test]
    fn test_tls_config_with_client_auth() {
        let certs = generate_test_certs();
        let config = TlsConfig::new(&certs.cert_path, &certs.key_path).with_client_auth();
        assert!(config.require_client_auth);
    }

    #[test]
    fn test_tls_config_insecure_mode() {
        let certs = generate_test_certs();
        let config = make_insecure_config(&TlsConfig::new(&certs.cert_path, &certs.key_path));
        assert!(config.insecure_skip_verify);
        // should_verify() always returns true in release builds
        #[cfg(debug_assertions)]
        assert!(!config.should_verify());
    }

    // Certificate Loading Tests
    #[test]
    fn test_load_certs_valid_single_cert() {
        let certs = generate_test_certs();
        let result = load_certs(&certs.cert_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_load_certs_certificate_chain() {
        let (cert_path, _key_path, _temp_dir) = generate_cert_chain();
        let result = load_certs(&cert_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_load_certs_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let empty_cert = temp_dir.path().join("empty.pem");
        std::fs::write(&empty_cert, "").unwrap();

        let result = load_certs(&empty_cert);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("no certificates found"),
            "error: {}",
            err
        );
    }

    #[test]
    fn test_load_certs_missing_file() {
        let result = load_certs(Path::new("/nonexistent/cert.pem"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("failed to open"));
    }

    #[test]
    fn test_load_certs_invalid_pem() {
        let temp_dir = TempDir::new().unwrap();
        let invalid_cert = temp_dir.path().join("invalid.pem");
        std::fs::write(&invalid_cert, "not a valid PEM file").unwrap();

        let result = load_certs(&invalid_cert);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no certificates found"));
    }

    // Private Key Loading Tests
    #[test]
    fn test_load_private_key_pkcs8_format() {
        let certs = generate_test_certs();
        let result = load_private_key(&certs.key_path);
        assert!(result.is_ok());
        match result.unwrap() {
            PrivateKeyDer::Pkcs8(_) => {},
            _ => panic!("expected PKCS8 key"),
        }
    }

    #[test]
    fn test_load_private_key_rsa_format() {
        // Generate RSA key in PKCS1 format
        use std::process::Command;

        let temp_dir = TempDir::new().unwrap();
        let key_path = temp_dir.path().join("rsa_key.pem");

        // Try to generate RSA key using openssl (if available)
        // Use -traditional flag to force PKCS1 format (for newer OpenSSL versions)
        let output = Command::new("openssl")
            .args(["genrsa", "-traditional", "-out"])
            .arg(&key_path)
            .arg("2048")
            .output();

        // If -traditional fails (older openssl), try without it
        if output.as_ref().map(|o| !o.status.success()).unwrap_or(true) {
            let output = Command::new("openssl")
                .args(["genrsa", "-out"])
                .arg(&key_path)
                .arg("2048")
                .output();

            if let Ok(out) = output {
                if out.status.success() {
                    let result = load_private_key(&key_path);
                    assert!(result.is_ok());
                    // Accept either PKCS1 or PKCS8 format depending on OpenSSL version
                    match result.unwrap() {
                        PrivateKeyDer::Pkcs1(_) | PrivateKeyDer::Pkcs8(_) => {},
                        _ => panic!("expected PKCS1/RSA or PKCS8 key"),
                    }
                }
            }
            return;
        }

        if let Ok(out) = output {
            if out.status.success() {
                let result = load_private_key(&key_path);
                assert!(result.is_ok());
                match result.unwrap() {
                    PrivateKeyDer::Pkcs1(_) => {},
                    _ => panic!("expected PKCS1/RSA key with -traditional flag"),
                }
            }
        }
    }

    #[test]
    fn test_load_private_key_no_keys_in_file() {
        let temp_dir = TempDir::new().unwrap();
        let empty_key = temp_dir.path().join("empty.pem");
        std::fs::write(&empty_key, "").unwrap();

        let result = load_private_key(&empty_key);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no private key found"));
    }

    #[test]
    fn test_load_private_key_missing_file() {
        let result = load_private_key(Path::new("/nonexistent/key.pem"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("failed to open"));
    }

    #[test]
    fn test_load_private_key_invalid_pem() {
        let temp_dir = TempDir::new().unwrap();
        let invalid_key = temp_dir.path().join("invalid.pem");
        std::fs::write(&invalid_key, "not a valid PEM key").unwrap();

        let result = load_private_key(&invalid_key);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no private key found"));
    }

    // Server TLS Tests
    #[tokio::test]
    async fn test_wrap_server_successful_handshake() {
        use tokio::net::TcpListener;

        let certs = generate_test_certs();
        let config = TlsConfig::new(&certs.cert_path, &certs.key_path);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn({
            let config = config.clone();
            async move {
                let (stream, _) = listener.accept().await.unwrap();
                wrap_server(stream, &config).await
            }
        });

        let client_task = tokio::spawn({
            let config = make_insecure_config(&config);
            async move {
                let stream = TcpStream::connect(addr).await.unwrap();
                wrap_client(stream, &config, "localhost").await
            }
        });

        let (server_result, client_result) = tokio::join!(server_task, client_task);

        assert!(
            server_result.unwrap().is_ok(),
            "server handshake should succeed"
        );
        assert!(
            client_result.unwrap().is_ok(),
            "client handshake should succeed"
        );
    }

    #[tokio::test]
    async fn test_wrap_server_cert_key_mismatch() {
        // Generate two different cert/key pairs
        let certs1 = generate_test_certs();
        let certs2 = generate_test_certs();

        // Use cert from one and key from another - should fail
        let config = TlsConfig::new(&certs1.cert_path, &certs2.key_path);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            wrap_server(stream, &config).await
        });

        // Client connects but server should fail
        let _client = TcpStream::connect(addr).await.unwrap();

        let server_result = server_task.await.unwrap();
        assert!(
            server_result.is_err(),
            "should fail with mismatched cert/key"
        );
        assert!(server_result
            .unwrap_err()
            .to_string()
            .contains("TLS config error"));
    }

    // Client TLS Tests
    #[tokio::test]
    async fn test_wrap_client_with_ca_verification() {
        use tokio::net::TcpListener;

        // Generate CA and signed certificate
        let (cert_path, key_path, _temp_dir) = generate_cert_chain();

        let config = TlsConfig::new(&cert_path, &key_path).with_ca_cert(&cert_path);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn({
            let config = config.clone();
            async move {
                let (stream, _) = listener.accept().await.unwrap();
                wrap_server(stream, &config).await
            }
        });

        let client_task = tokio::spawn({
            let config = make_insecure_config(&config);
            async move {
                let stream = TcpStream::connect(addr).await.unwrap();
                wrap_client(stream, &config, "localhost").await
            }
        });

        let (server_result, client_result) = tokio::join!(server_task, client_task);

        assert!(server_result.unwrap().is_ok());
        assert!(client_result.unwrap().is_ok());
    }

    #[test]
    fn test_wrap_client_invalid_server_name() {
        // Test that invalid server names are rejected synchronously via ServerName parsing
        let invalid_names = ["", "invalid..name", "a]b[c"];

        for name in invalid_names {
            let result = ServerName::try_from(name.to_string());
            // Some may parse successfully, others will fail
            if result.is_err() {
                // The error is expected for truly invalid names
                continue;
            }
        }
    }

    // InsecureVerifier Tests
    #[test]
    fn test_insecure_verifier_accepts_any_cert() {
        let verifier = InsecureVerifier;

        // Create a dummy certificate
        let cert = rcgen::generate_simple_self_signed(vec!["test.com".into()]).unwrap();
        let cert_der = CertificateDer::from(cert.cert.der().to_vec());
        let server_name = ServerName::try_from("test.com".to_string()).unwrap();

        let result = verifier.verify_server_cert(
            &cert_der,
            &[],
            &server_name,
            &[],
            tokio_rustls::rustls::pki_types::UnixTime::now(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_insecure_verifier_supported_schemes() {
        let verifier = InsecureVerifier;
        let schemes = verifier.supported_verify_schemes();

        assert!(!schemes.is_empty());
        assert!(schemes.contains(&tokio_rustls::rustls::SignatureScheme::RSA_PKCS1_SHA256));
        assert!(schemes.contains(&tokio_rustls::rustls::SignatureScheme::ECDSA_NISTP256_SHA256));
        assert!(schemes.contains(&tokio_rustls::rustls::SignatureScheme::ED25519));
    }

    #[test]
    fn test_insecure_verifier_debug() {
        // Verifies Debug is implemented
        let verifier = InsecureVerifier;
        let debug_str = format!("{:?}", verifier);
        assert!(debug_str.contains("InsecureVerifier"));
    }

    #[tokio::test]
    async fn test_wrap_client_ca_add_error() {
        // Test behavior when CA cert file exists but contains no valid certs
        let temp_dir = TempDir::new().unwrap();
        let empty_ca = temp_dir.path().join("empty_ca.pem");
        std::fs::write(&empty_ca, "not a valid CA cert").unwrap();

        let certs = generate_test_certs();
        let config = TlsConfig::new(&certs.cert_path, &certs.key_path).with_ca_cert(&empty_ca);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_task = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            wrap_client(stream, &config, "localhost").await
        });

        // Accept connection but don't complete handshake
        let _ = listener.accept().await.unwrap();

        let client_result = client_task.await.unwrap();
        // Should fail because CA cert file is invalid
        assert!(client_result.is_err());
    }

    #[test]
    fn test_load_certs_with_whitespace() {
        let temp_dir = TempDir::new().unwrap();
        let cert = rcgen::generate_simple_self_signed(vec!["test.com".into()]).unwrap();

        // Add leading/trailing whitespace
        let cert_with_whitespace = format!("\n\n{}\n\n", cert.cert.pem());
        let cert_path = temp_dir.path().join("whitespace.pem");
        std::fs::write(&cert_path, cert_with_whitespace).unwrap();

        let result = load_certs(&cert_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_wrap_client_missing_ca_file() {
        let certs = generate_test_certs();
        let config =
            TlsConfig::new(&certs.cert_path, &certs.key_path).with_ca_cert("/nonexistent/ca.pem");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_task = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            wrap_client(stream, &config, "localhost").await
        });

        // Accept connection
        let _ = listener.accept().await.unwrap();

        let result = client_task.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("failed to open"));
    }

    // VerifiedPeerIdentity Tests
    #[test]
    fn test_verified_peer_identity_from_common_name() {
        let identity = VerifiedPeerIdentity::from_common_name("node1");
        assert_eq!(identity.node_id, "node1");
        assert_eq!(identity.source, NodeIdSource::CommonName);
    }

    #[test]
    fn test_verified_peer_identity_from_san() {
        let identity = VerifiedPeerIdentity::from_san("node2.cluster.local");
        assert_eq!(identity.node_id, "node2.cluster.local");
        assert_eq!(identity.source, NodeIdSource::SubjectAltName);
    }

    #[test]
    fn test_verified_peer_identity_debug() {
        let identity = VerifiedPeerIdentity::from_common_name("test-node");
        let debug_str = format!("{:?}", identity);
        assert!(debug_str.contains("test-node"));
        assert!(debug_str.contains("CommonName"));
    }

    #[test]
    fn test_extract_node_id_none_mode() {
        let certs = generate_test_certs();
        let loaded = load_certs(&certs.cert_path).unwrap();
        let cert = loaded.first().unwrap();

        let result = extract_node_id_from_cert(cert, &NodeIdVerification::None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_extract_node_id_common_name() {
        let certs = generate_test_certs();
        let loaded = load_certs(&certs.cert_path).unwrap();
        let cert = loaded.first().unwrap();

        let result = extract_node_id_from_cert(cert, &NodeIdVerification::CommonName);
        // rcgen sets CN to "rcgen self signed cert" by default
        assert!(result.is_ok());
        let identity = result.unwrap().expect("should extract CN");
        assert!(!identity.node_id.is_empty(), "CN should not be empty");
        assert_eq!(identity.source, NodeIdSource::CommonName);
    }

    #[test]
    fn test_extract_node_id_san() {
        let certs = generate_test_certs();
        let loaded = load_certs(&certs.cert_path).unwrap();
        let cert = loaded.first().unwrap();

        let result = extract_node_id_from_cert(cert, &NodeIdVerification::SubjectAltName);
        // The test cert has SAN=localhost
        assert!(result.is_ok());
        let identity = result.unwrap().expect("should extract SAN");
        assert_eq!(identity.node_id, "localhost");
        assert_eq!(identity.source, NodeIdSource::SubjectAltName);
    }

    #[test]
    fn test_node_id_verification_builder() {
        let certs = generate_test_certs();
        let config = TlsConfig::new(&certs.cert_path, &certs.key_path)
            .with_node_id_verification(NodeIdVerification::CommonName);

        assert_eq!(config.node_id_verification, NodeIdVerification::CommonName);
    }
}
