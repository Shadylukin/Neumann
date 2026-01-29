// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{SecurityConfig, SecurityMode, TcpTransportConfig, TlsConfig, NodeIdVerification};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    security_mode: u8,
    has_tls: bool,
    require_client_auth: bool,
    node_id_verification: u8,
    has_ca_cert: bool,
    warn_on_insecure: bool,
}

impl FuzzInput {
    fn to_security_mode(&self) -> SecurityMode {
        match self.security_mode % 4 {
            0 => SecurityMode::Strict,
            1 => SecurityMode::Permissive,
            2 => SecurityMode::Development,
            _ => SecurityMode::Legacy,
        }
    }

    fn to_node_id_verification(&self) -> NodeIdVerification {
        match self.node_id_verification % 3 {
            0 => NodeIdVerification::None,
            1 => NodeIdVerification::CommonName,
            _ => NodeIdVerification::SubjectAltName,
        }
    }
}

fuzz_target!(|input: FuzzInput| {
    let mode = input.to_security_mode();

    // Create security config
    let security = SecurityConfig {
        mode,
        warn_on_insecure: input.warn_on_insecure,
    };

    // Create base config
    let mut config = TcpTransportConfig::new("fuzz_node", "127.0.0.1:9100".parse().unwrap())
        .with_security(security);

    // Optionally add TLS
    if input.has_tls {
        let mut tls = TlsConfig::new("/cert.pem", "/key.pem");

        if input.has_ca_cert {
            tls.ca_cert_path = Some("/ca.pem".into());
        }

        if input.require_client_auth {
            tls.require_client_auth = true;
        }

        tls.node_id_verification = input.to_node_id_verification();

        config = config.with_tls(tls);
    }

    // Validate the configuration
    let result = config.validate_security();

    // Verify consistency: if validation passes, check that the configuration
    // is actually valid according to the rules
    if result.is_ok() {
        // If mode requires TLS, TLS must be present
        if mode.requires_tls() {
            assert!(config.tls.is_some(), "TLS required but validation passed without TLS");
        }

        // If mode requires mTLS, client auth must be enabled
        if mode.requires_mtls() {
            if let Some(ref tls) = config.tls {
                assert!(tls.require_client_auth, "mTLS required but client auth disabled");
            }
        }

        // If mode requires NodeId verification, it must be configured
        if mode.requires_node_id_verification() {
            if let Some(ref tls) = config.tls {
                assert_ne!(
                    tls.node_id_verification,
                    NodeIdVerification::None,
                    "NodeId verification required but not configured"
                );
            }
        }
    }

    // Verify is_secure() consistency
    let is_secure = config.is_secure();
    if is_secure {
        assert!(result.is_ok(), "is_secure() true but validation failed");
        assert!(
            matches!(mode, SecurityMode::Strict | SecurityMode::Permissive),
            "is_secure() true but mode is {:?}",
            mode
        );
    }

    // Test serialization roundtrip
    let serialized = bitcode::serialize(&config.security).unwrap();
    let deserialized: SecurityConfig = bitcode::deserialize(&serialized).unwrap();
    assert_eq!(config.security.mode, deserialized.mode);
    assert_eq!(config.security.warn_on_insecure, deserialized.warn_on_insecure);
});
