//! Integration tests for TLS security hardening.

use tensor_chain::{
    NodeIdVerification, SecurityConfig, SecurityMode, TcpTransportConfig, TlsConfig,
};

#[test]
fn test_strict_mode_rejects_plaintext() {
    // Strict mode requires TLS - validation should fail without it
    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_security_mode(SecurityMode::Strict);

    let result = config.validate_security();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("TLS required"));
}

#[test]
fn test_strict_mode_requires_client_cert() {
    // TLS configured but mTLS not enabled
    let tls = TlsConfig::new("/cert.pem", "/key.pem").with_ca_cert("/ca.pem");

    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_tls(tls)
        .with_security_mode(SecurityMode::Strict);

    let result = config.validate_security();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("mutual TLS required"));
}

#[test]
fn test_node_id_mismatch_validation() {
    // TLS + mTLS configured but no NodeId verification
    let tls = TlsConfig::new("/cert.pem", "/key.pem")
        .with_ca_cert("/ca.pem")
        .with_client_auth();

    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_tls(tls)
        .with_security_mode(SecurityMode::Strict);

    let result = config.validate_security();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("NodeId verification required"));
}

#[test]
fn test_development_mode_allows_plaintext() {
    // Development mode should allow no TLS
    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_security(SecurityConfig::development().without_warnings());

    let result = config.validate_security();
    assert!(result.is_ok());
}

#[test]
fn test_permissive_mode_allows_no_mtls() {
    // Permissive mode requires TLS but not mTLS
    let tls = TlsConfig::new("/cert.pem", "/key.pem");

    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_tls(tls)
        .with_security_mode(SecurityMode::Permissive);

    let result = config.validate_security();
    assert!(result.is_ok());
}

#[test]
fn test_strict_mode_full_security() {
    // All security settings properly configured
    let tls = TlsConfig::new_secure("/cert.pem", "/key.pem", "/ca.pem");

    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_tls(tls)
        .with_security_mode(SecurityMode::Strict);

    let result = config.validate_security();
    assert!(result.is_ok());
    assert!(config.is_secure());
}

#[test]
fn test_legacy_mode_for_migration() {
    // Legacy mode maintains backward compatibility
    let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
        .with_security(SecurityConfig::legacy().without_warnings());

    let result = config.validate_security();
    assert!(result.is_ok());

    // But it's not considered secure for production
    assert!(!config.is_secure());
}

#[test]
fn test_tls_config_new_secure_enforces_all_settings() {
    let tls = TlsConfig::new_secure("/cert.pem", "/key.pem", "/ca.pem");

    // All security settings should be enabled
    assert!(tls.require_client_auth);
    assert!(tls.ca_cert_path.is_some());
    assert_eq!(tls.node_id_verification, NodeIdVerification::CommonName);
    assert!(!tls.insecure_skip_verify);
}

#[test]
fn test_security_mode_serde_roundtrip() {
    let modes = vec![
        SecurityMode::Strict,
        SecurityMode::Permissive,
        SecurityMode::Development,
        SecurityMode::Legacy,
    ];

    for mode in modes {
        let serialized = bincode::serialize(&mode).unwrap();
        let deserialized: SecurityMode = bincode::deserialize(&serialized).unwrap();
        assert_eq!(mode, deserialized);
    }
}

#[test]
fn test_security_config_builder() {
    let config = SecurityConfig::strict().without_warnings();

    assert_eq!(config.mode, SecurityMode::Strict);
    assert!(!config.warn_on_insecure);
}
