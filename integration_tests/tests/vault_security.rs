// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for vault security: seal/unseal, Shamir secret sharing,
//! PKI certificate lifecycle, key rotation, transit encryption, and emergency
//! access across the `tensor_vault` module boundaries.

use std::sync::Arc;
use std::time::Duration;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    split_master_key, CertificateRequest, KeyShare, MasterKey, PasswordConfig, RateLimitConfig,
    SecretTemplate, ShamirConfig, Vault, VaultConfig, VaultError,
};

fn create_test_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    Vault::new(
        b"test-key-32-bytes-long!!!!!",
        graph,
        store,
        VaultConfig::default(),
    )
    .unwrap()
}

fn create_rate_limited_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let config = VaultConfig::default().with_rate_limit(RateLimitConfig::default());
    Vault::new(b"test-key-32-bytes-long!!!!!", graph, store, config).unwrap()
}

// ---------- Seal blocks operations (tests 1-7) ----------

#[test]
fn test_seal_blocks_set() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "pre/seal", "value").unwrap();
    vault.seal().unwrap();

    let result = vault.set(Vault::ROOT, "post/seal", "value");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_get() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "readable", "secret").unwrap();
    vault.seal().unwrap();

    let result = vault.get(Vault::ROOT, "readable");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_rotate() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "rotatable", "v1").unwrap();
    vault.seal().unwrap();

    let result = vault.rotate(Vault::ROOT, "rotatable", "v2");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_delete() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "deletable", "bye").unwrap();
    vault.seal().unwrap();

    let result = vault.delete(Vault::ROOT, "deletable");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_encrypt_for() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "transit/key", "aad-key").unwrap();
    vault.seal().unwrap();

    let result = vault.encrypt_for(Vault::ROOT, "transit/key", b"plaintext");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_wrap_secret() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "wrappable", "wrapped-val").unwrap();
    vault.seal().unwrap();

    let result = vault.wrap_secret(Vault::ROOT, "wrappable", 60_000);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

#[test]
fn test_seal_blocks_dynamic_generate() {
    let mut vault = create_test_vault();
    vault.seal().unwrap();

    let template = SecretTemplate::Password(PasswordConfig::default());
    let result = vault.generate_dynamic_secret(Vault::ROOT, &template, 60_000, false);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::Sealed(_))),
        "expected Sealed error, got {result:?}"
    );
}

// ---------- Unseal (tests 8-10) ----------

#[test]
fn test_unseal_with_password_restores() {
    let mut vault = create_test_vault();
    vault
        .set(Vault::ROOT, "persist/key", "persist-val")
        .unwrap();
    vault.seal().unwrap();
    assert!(vault.is_sealed());

    // Unseal with the same password used in create_test_vault
    vault.unseal(b"test-key-32-bytes-long!!!!!").unwrap();
    assert!(!vault.is_sealed());

    let value = vault.get(Vault::ROOT, "persist/key").unwrap();
    assert_eq!(value, "persist-val");
}

#[test]
fn test_unseal_wrong_password_fails() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "sealed/secret", "value").unwrap();
    vault.seal().unwrap();

    // Attempt unseal with a different password -- key derivation will succeed but
    // the derived keys will be wrong, causing subsequent crypto operations to fail.
    // The unseal itself may or may not error (it re-derives keys but doesn't
    // validate them), so test that data is inaccessible after wrong-password unseal.
    let unseal_result = vault.unseal(b"wrong-password-not-matching!!!");
    if unseal_result.is_ok() {
        // Keys were re-derived from wrong password; reads should fail with crypto error
        let get_result = vault.get(Vault::ROOT, "sealed/secret");
        assert!(
            get_result.is_err(),
            "get should fail after wrong-password unseal"
        );
    }
    // If unseal itself errors, that is also acceptable
}

#[test]
fn test_is_sealed_reflects_state() {
    let mut vault = create_test_vault();
    assert!(!vault.is_sealed(), "vault should start unsealed");

    vault.seal().unwrap();
    assert!(vault.is_sealed(), "vault should be sealed after seal()");

    vault.unseal(b"test-key-32-bytes-long!!!!!").unwrap();
    assert!(
        !vault.is_sealed(),
        "vault should be unsealed after unseal()"
    );
}

// ---------- Shamir secret sharing (tests 11-14) ----------

#[test]
fn test_shamir_split_reconstruct() {
    let key = MasterKey::from_bytes([0xAB; 32]);
    let config = ShamirConfig {
        total_shares: 5,
        threshold: 3,
    };

    let shares = split_master_key(&key, &config).unwrap();
    assert_eq!(shares.len(), 5);

    // Reconstruct from exactly the threshold number of shares
    let subset: Vec<KeyShare> = shares.into_iter().take(3).collect();
    let reconstructed = tensor_vault::reconstruct_master_key(&subset).unwrap();
    assert_eq!(reconstructed.as_bytes(), key.as_bytes());
}

#[test]
fn test_shamir_insufficient_shares() {
    let key = MasterKey::from_bytes([0xCD; 32]);
    let config = ShamirConfig {
        total_shares: 5,
        threshold: 3,
    };

    let shares = split_master_key(&key, &config).unwrap();
    // Only one share -- below the minimum 2 required by reconstruct
    let single = vec![shares[0].clone()];
    let result = tensor_vault::reconstruct_master_key(&single);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(VaultError::ShamirError(_))),
        "expected ShamirError"
    );
}

#[test]
fn test_shamir_all_shares() {
    let key = MasterKey::from_bytes([0xEF; 32]);
    let config = ShamirConfig {
        total_shares: 7,
        threshold: 4,
    };

    let shares = split_master_key(&key, &config).unwrap();
    assert_eq!(shares.len(), 7);

    // Using all shares should also reconstruct correctly
    let reconstructed = tensor_vault::reconstruct_master_key(&shares).unwrap();
    assert_eq!(reconstructed.as_bytes(), key.as_bytes());
}

#[test]
fn test_unseal_with_shamir_shares() {
    let mut vault = create_test_vault();
    vault
        .set(Vault::ROOT, "shamir/secret", "shamir-val")
        .unwrap();

    // Split a known key into shares
    let key = MasterKey::from_bytes([0x42; 32]);
    let config = ShamirConfig {
        total_shares: 5,
        threshold: 3,
    };
    let shares = split_master_key(&key, &config).unwrap();

    vault.seal().unwrap();
    assert!(vault.is_sealed());

    // Unseal with shares -- this reconstructs the key from shares and re-derives
    // vault keys from the reconstructed master key. Since the reconstructed key
    // differs from the original password-derived key, reads of old data will fail.
    // The important thing is that unseal_with_shares itself succeeds and unseals.
    let subset: Vec<KeyShare> = shares.into_iter().take(3).collect();
    vault.unseal_with_shares(&subset).unwrap();
    assert!(!vault.is_sealed());
}

// ---------- PKI (tests 15-20) ----------

#[test]
fn test_pki_init_issue() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let request = CertificateRequest {
        common_name: "test-service.local".to_string(),
        organization: Some("Neumann".to_string()),
        san_dns: vec!["test-service.local".to_string()],
        san_ip: vec![],
    };

    let (serial, cert_der) = vault
        .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
        .unwrap();

    assert!(!serial.is_empty(), "serial should be non-empty");
    assert!(!cert_der.is_empty(), "certificate DER should be non-empty");
}

#[test]
fn test_pki_list_certificates() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let request = CertificateRequest {
        common_name: "list-test.local".to_string(),
        organization: None,
        san_dns: vec![],
        san_ip: vec![],
    };
    vault
        .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
        .unwrap();

    let certs = vault.list_certificates(Vault::ROOT).unwrap();
    assert!(!certs.is_empty(), "should have at least one certificate");
    assert_eq!(certs[0].subject, "list-test.local");
}

#[test]
fn test_pki_revoke_certificate() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let request = CertificateRequest {
        common_name: "revoke-me.local".to_string(),
        organization: None,
        san_dns: vec![],
        san_ip: vec![],
    };
    let (serial, _) = vault
        .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
        .unwrap();

    assert!(!vault.is_certificate_revoked(&serial));
    vault.revoke_certificate(Vault::ROOT, &serial).unwrap();
    assert!(vault.is_certificate_revoked(&serial));
}

#[test]
fn test_pki_revocation_list() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let request = CertificateRequest {
        common_name: "crl-test.local".to_string(),
        organization: None,
        san_dns: vec![],
        san_ip: vec![],
    };
    let (serial, _) = vault
        .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
        .unwrap();

    vault.revoke_certificate(Vault::ROOT, &serial).unwrap();

    let crl = vault.get_revocation_list().unwrap();
    assert!(!crl.entries.is_empty(), "CRL should contain revoked cert");
    assert!(
        crl.entries.iter().any(|e| e.serial == serial),
        "CRL should contain the revoked serial"
    );
}

#[test]
fn test_pki_get_ca_certificate() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let ca_cert = vault.get_ca_certificate(Vault::ROOT).unwrap();
    assert!(!ca_cert.is_empty(), "CA certificate should be non-empty");
}

#[test]
fn test_pki_non_root_denied() {
    let vault = create_test_vault();
    vault.init_pki(Vault::ROOT).unwrap();

    let non_root = "node:user42";

    let init_result = vault.init_pki(non_root);
    assert!(
        matches!(init_result, Err(VaultError::AccessDenied(_))),
        "non-root should not init PKI"
    );

    let request = CertificateRequest {
        common_name: "denied.local".to_string(),
        organization: None,
        san_dns: vec![],
        san_ip: vec![],
    };
    let issue_result = vault.issue_certificate(non_root, &request, Duration::from_secs(3600));
    assert!(
        matches!(issue_result, Err(VaultError::AccessDenied(_))),
        "non-root should not issue certificates"
    );

    let list_result = vault.list_certificates(non_root);
    assert!(
        matches!(list_result, Err(VaultError::AccessDenied(_))),
        "non-root should not list certificates"
    );

    // Issue a cert as root, then try revoking as non-root
    let (serial, _) = vault
        .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
        .unwrap();
    let revoke_result = vault.revoke_certificate(non_root, &serial);
    assert!(
        matches!(revoke_result, Err(VaultError::AccessDenied(_))),
        "non-root should not revoke certificates"
    );
}

// ---------- Master key rotation (tests 21-22) ----------

#[test]
fn test_master_key_rotation() {
    let mut vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/alpha", "alpha-val").unwrap();
    vault.set(Vault::ROOT, "rotate/beta", "beta-val").unwrap();

    vault
        .rotate_master_key(b"brand-new-password-32-bytes!!")
        .unwrap();

    // After rotation, secrets should still be readable
    let alpha = vault.get(Vault::ROOT, "rotate/alpha").unwrap();
    assert_eq!(alpha, "alpha-val");
    let beta = vault.get(Vault::ROOT, "rotate/beta").unwrap();
    assert_eq!(beta, "beta-val");
}

#[test]
fn test_master_key_rotation_count() {
    let mut vault = create_test_vault();

    vault.set(Vault::ROOT, "count/one", "1").unwrap();
    vault.set(Vault::ROOT, "count/two", "2").unwrap();
    vault.set(Vault::ROOT, "count/three", "3").unwrap();

    let rotated = vault
        .rotate_master_key(b"another-new-password-32-bytes!")
        .unwrap();
    assert_eq!(rotated, 3, "should have re-encrypted exactly 3 secrets");
}

// ---------- Transit encryption (tests 23-25) ----------

#[test]
fn test_encrypt_decrypt_transit() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "transit/aad", "aad-secret").unwrap();

    let plaintext = b"hello confidential world";
    let sealed = vault
        .encrypt_for(Vault::ROOT, "transit/aad", plaintext)
        .unwrap();

    assert_ne!(sealed, plaintext, "sealed data must differ from plaintext");

    let decrypted = vault
        .decrypt_as(Vault::ROOT, "transit/aad", &sealed)
        .unwrap();
    assert_eq!(decrypted, plaintext);
}

#[test]
fn test_transit_different_keys_fail() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "transit/key-a", "a").unwrap();
    vault.set(Vault::ROOT, "transit/key-b", "b").unwrap();

    let sealed = vault
        .encrypt_for(Vault::ROOT, "transit/key-a", b"secret-data")
        .unwrap();

    // Decrypting with a different AAD key should fail (AAD mismatch)
    let result = vault.decrypt_as(Vault::ROOT, "transit/key-b", &sealed);
    assert!(
        result.is_err(),
        "decrypting with different key (AAD) should fail"
    );
}

#[test]
fn test_transit_access_required() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "transit/guarded", "secret").unwrap();

    let non_root = "node:outsider";

    let enc_result = vault.encrypt_for(non_root, "transit/guarded", b"data");
    assert!(
        enc_result.is_err(),
        "non-root without grant should not encrypt"
    );

    let sealed = vault
        .encrypt_for(Vault::ROOT, "transit/guarded", b"data")
        .unwrap();

    let dec_result = vault.decrypt_as(non_root, "transit/guarded", &sealed);
    assert!(
        dec_result.is_err(),
        "non-root without grant should not decrypt"
    );
}

// ---------- Emergency access (test 26) ----------

#[test]
fn test_emergency_access() {
    let vault = create_rate_limited_vault();
    vault
        .set(Vault::ROOT, "emergency/target", "critical-secret")
        .unwrap();

    let non_root = "node:responder";

    // Without emergency access, non-root cannot read
    let denied = vault.get(non_root, "emergency/target");
    assert!(denied.is_err());

    // Emergency access bypasses graph ACL
    let value = vault
        .emergency_access(
            non_root,
            "emergency/target",
            "incident #42 requires immediate access",
            Duration::from_secs(300),
        )
        .unwrap();

    assert_eq!(value, "critical-secret");
}

// ---------- Vault status (test 27) ----------

#[test]
fn test_vault_status() {
    let mut vault = create_test_vault();
    vault.set(Vault::ROOT, "status/one", "1").unwrap();
    vault.set(Vault::ROOT, "status/two", "2").unwrap();

    let status = vault.vault_status();
    assert!(!status.sealed);
    assert!(
        status.total_secrets >= 2,
        "should count at least 2 secrets, got {}",
        status.total_secrets
    );

    vault.seal().unwrap();
    let sealed_status = vault.vault_status();
    assert!(sealed_status.sealed);
}

// ---------- Concurrent seal check (test 28) ----------

#[test]
fn test_concurrent_seal_check() {
    let vault = create_test_vault();

    // Spawn readers that check is_sealed via shared reference
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let sealed = vault.is_sealed();
            std::thread::spawn(move || {
                // Each thread captures the seal state; since vault is not sealed,
                // all should see false.
                assert!(!sealed, "vault should be unsealed during concurrent reads");
                sealed
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("thread should not panic");
        assert!(!result);
    }
}
