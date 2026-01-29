// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz test for TLS key parsing.
//!
//! Ensures that malformed PEM/DER key data does not cause panics.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    // Write fuzz data to temporary files
    let mut cert_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut key_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };

    if cert_file.write_all(data).is_err() {
        return;
    }
    if key_file.write_all(data).is_err() {
        return;
    }

    // Create TLS config - should never panic
    let config = tensor_chain::TlsConfig::new(cert_file.path(), key_file.path());

    // Access config fields to ensure they're valid
    let _ = config.cert_path;
    let _ = config.key_path;
    let _ = config.require_client_auth;
    let _ = config.insecure_skip_verify;

    // Try builder methods
    let config = config.with_client_auth();
    let _ = config.require_client_auth;

    // Try with_node_id_verification (clone since methods consume self)
    let _ = config.clone().with_node_id_verification(tensor_chain::NodeIdVerification::CommonName);
    let _ = config.clone().with_node_id_verification(tensor_chain::NodeIdVerification::SubjectAltName);
    let _ = config.with_node_id_verification(tensor_chain::NodeIdVerification::None);
});
