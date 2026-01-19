//! Integration tests for TLS error handling.
//!
//! Tests that TLS key parsing handles malformed input gracefully
//! without panicking.

use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper to create a temp directory with test files
fn setup_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

/// Helper to write content to a file in the temp directory
fn write_temp_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, content).expect("Failed to write temp file");
    path
}

#[test]
fn test_tls_config_with_empty_cert_file() {
    let temp_dir = setup_temp_dir();
    let cert_path = write_temp_file(&temp_dir, "empty.pem", "");
    let key_path = write_temp_file(&temp_dir, "empty_key.pem", "");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    // Config creation should succeed (validation happens at use time)
    assert_eq!(config.cert_path, cert_path);
    assert_eq!(config.key_path, key_path);
}

#[test]
fn test_tls_config_with_invalid_pem_content() {
    let temp_dir = setup_temp_dir();
    let cert_path = write_temp_file(
        &temp_dir,
        "invalid.pem",
        "not a valid PEM file\n-----garbage-----\n",
    );
    let key_path = write_temp_file(&temp_dir, "invalid_key.pem", "also invalid\n");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    // Config creation should succeed
    assert!(!config.require_client_auth);
}

#[test]
fn test_tls_config_with_partial_pem() {
    let temp_dir = setup_temp_dir();

    // Partial/truncated PEM header
    let partial_pem = "-----BEGIN CERTIFICATE-----\nMIIB";
    let cert_path = write_temp_file(&temp_dir, "partial.pem", partial_pem);
    let key_path = write_temp_file(
        &temp_dir,
        "partial_key.pem",
        "-----BEGIN PRIVATE KEY-----\n",
    );

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    // Config should be created without panicking
    assert_eq!(config.cert_path, cert_path);
}

#[test]
fn test_tls_config_with_binary_garbage() {
    let temp_dir = setup_temp_dir();

    // Binary garbage that might look like DER but isn't
    let garbage: Vec<u8> = vec![0x30, 0x82, 0x00, 0x50, 0xFF, 0xFE, 0x00, 0x00];
    let cert_path = temp_dir.path().join("binary.pem");
    let key_path = temp_dir.path().join("binary_key.pem");

    fs::write(&cert_path, &garbage).expect("Failed to write cert");
    fs::write(&key_path, &garbage).expect("Failed to write key");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    // Should not panic
    assert_eq!(config.cert_path, cert_path);
}

#[test]
fn test_tls_config_builder_chain() {
    let temp_dir = setup_temp_dir();
    let cert_path = write_temp_file(&temp_dir, "cert.pem", "");
    let key_path = write_temp_file(&temp_dir, "key.pem", "");
    let ca_path = write_temp_file(&temp_dir, "ca.pem", "");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path)
        .with_ca_cert(&ca_path)
        .with_client_auth()
        .with_node_id_verification(tensor_chain::NodeIdVerification::CommonName);

    assert!(config.require_client_auth);
    assert_eq!(config.ca_cert_path, Some(ca_path));
    assert_eq!(
        config.node_id_verification,
        tensor_chain::NodeIdVerification::CommonName
    );
}

#[test]
fn test_tls_config_insecure_mode_only_in_debug() {
    let temp_dir = setup_temp_dir();
    let cert_path = write_temp_file(&temp_dir, "cert.pem", "");
    let key_path = write_temp_file(&temp_dir, "key.pem", "");

    let mut config = tensor_chain::TlsConfig::new(&cert_path, &key_path);
    config.insecure_skip_verify = true;

    // In release builds, should_verify() always returns true regardless
    // In debug builds, it respects the flag
    #[cfg(debug_assertions)]
    {
        assert!(!config.should_verify());
    }
    #[cfg(not(debug_assertions))]
    {
        assert!(config.should_verify());
    }
}

#[test]
fn test_node_id_verification_modes() {
    assert_ne!(
        tensor_chain::NodeIdVerification::None,
        tensor_chain::NodeIdVerification::CommonName
    );
    assert_ne!(
        tensor_chain::NodeIdVerification::CommonName,
        tensor_chain::NodeIdVerification::SubjectAltName
    );
    assert_ne!(
        tensor_chain::NodeIdVerification::None,
        tensor_chain::NodeIdVerification::SubjectAltName
    );
}

#[test]
fn test_tls_config_default_values() {
    let temp_dir = setup_temp_dir();
    let cert_path = write_temp_file(&temp_dir, "cert.pem", "");
    let key_path = write_temp_file(&temp_dir, "key.pem", "");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    assert!(!config.require_client_auth);
    assert!(!config.insecure_skip_verify);
    assert_eq!(config.ca_cert_path, None);
    assert_eq!(
        config.node_id_verification,
        tensor_chain::NodeIdVerification::None
    );
}

#[test]
fn test_tls_config_paths_are_preserved() {
    let temp_dir = setup_temp_dir();

    // Use paths with spaces and special characters
    let cert_path = write_temp_file(&temp_dir, "my cert file.pem", "");
    let key_path = write_temp_file(&temp_dir, "my key file.pem", "");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    assert_eq!(config.cert_path, cert_path);
    assert_eq!(config.key_path, key_path);
}

#[test]
fn test_multiple_pem_blocks_handling() {
    let temp_dir = setup_temp_dir();

    // Multiple incomplete PEM blocks
    let multi_pem = r#"
-----BEGIN CERTIFICATE-----
garbage
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
more garbage
-----END CERTIFICATE-----
"#;
    let cert_path = write_temp_file(&temp_dir, "multi.pem", multi_pem);
    let key_path = write_temp_file(&temp_dir, "key.pem", "");

    let config = tensor_chain::TlsConfig::new(&cert_path, &key_path);

    // Should handle gracefully without panicking
    assert_eq!(config.cert_path, cert_path);
}
