// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for vault dynamic secrets, rotation policies, wrapping
//! tokens, secret engines, and similarity search.
//!
//! Covers cross-module interactions between `dynamic.rs`, `rotation.rs`,
//! `wrapping.rs`, `engine.rs`, and `similarity.rs`.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    ApiKeyConfig, PasswordConfig, PkiEngine, RotationGenerator, RotationPolicy, SecretTemplate,
    TokenConfig, Vault, VaultConfig, VaultError,
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

// ==========================================================================
// Dynamic secret tests
// ==========================================================================

#[test]
fn test_dynamic_password_generation_and_retrieval() {
    let vault = create_test_vault();
    let template = SecretTemplate::Password(PasswordConfig::default());

    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    assert!(id.starts_with("dyn_"));
    assert!(!value.is_empty());

    let retrieved = vault.get_dynamic_secret(Vault::ROOT, &id).unwrap();
    assert_eq!(retrieved, value);
}

#[test]
fn test_dynamic_token_generation() {
    let vault = create_test_vault();
    let template = SecretTemplate::Token(TokenConfig::default());

    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    assert!(id.starts_with("dyn_"));
    assert!(!value.is_empty());
}

#[test]
fn test_dynamic_api_key_generation() {
    let vault = create_test_vault();
    let template = SecretTemplate::ApiKey(ApiKeyConfig::default());

    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    assert!(id.starts_with("dyn_"));
    assert!(value.starts_with("nk_"));
}

#[test]
fn test_dynamic_secret_one_time_consumed() {
    let vault = create_test_vault();
    let template = SecretTemplate::Password(PasswordConfig::default());

    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, true)
        .unwrap();

    // First retrieval succeeds
    let first = vault.get_dynamic_secret(Vault::ROOT, &id).unwrap();
    assert_eq!(first, value);

    // Second retrieval fails because it was consumed
    let second = vault.get_dynamic_secret(Vault::ROOT, &id);
    assert!(second.is_err());
}

#[test]
fn test_dynamic_secret_non_one_time_multiple_gets() {
    let vault = create_test_vault();
    let template = SecretTemplate::Token(TokenConfig::default());

    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    // Multiple retrievals succeed
    for _ in 0..5 {
        let retrieved = vault.get_dynamic_secret(Vault::ROOT, &id).unwrap();
        assert_eq!(retrieved, value);
    }
}

#[test]
fn test_dynamic_secret_expired_after_ttl() {
    let vault = create_test_vault();
    let template = SecretTemplate::Password(PasswordConfig::default());

    // Use a very short TTL (1ms). The vault timestamps have 1-second
    // granularity (current_timestamp returns seconds), so we must sleep
    // longer than 1 second for the expiration check to see a new second.
    let (id, _value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 1, false)
        .unwrap();

    // Sleep past the TTL boundary (must cross a second boundary)
    thread::sleep(Duration::from_millis(1500));

    let result = vault.get_dynamic_secret(Vault::ROOT, &id);
    assert!(result.is_err());
    assert!(matches!(result, Err(VaultError::SecretExpired(_))));
}

#[test]
fn test_dynamic_secret_list_root_only() {
    let vault = create_test_vault();
    let template = SecretTemplate::Password(PasswordConfig::default());

    vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();
    vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, true)
        .unwrap();

    let list = vault.list_dynamic_secrets(Vault::ROOT).unwrap();
    assert!(list.len() >= 2);

    // Non-root cannot list
    let denied = vault.list_dynamic_secrets("node:user1");
    assert!(denied.is_err());
}

#[test]
fn test_dynamic_secret_revoke() {
    let vault = create_test_vault();
    let template = SecretTemplate::Token(TokenConfig::default());

    let (id, _value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    // Revoke it
    vault.revoke_dynamic_secret(Vault::ROOT, &id).unwrap();

    // Get should fail after revocation (metadata removed)
    let result = vault.get_dynamic_secret(Vault::ROOT, &id);
    assert!(result.is_err());
}

#[test]
fn test_dynamic_secret_unique_ids() {
    let vault = create_test_vault();
    let template = SecretTemplate::Password(PasswordConfig::default());

    let (id1, _) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();
    let (id2, _) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    assert_ne!(id1, id2);
}

// ==========================================================================
// Wrapping token tests
// ==========================================================================

#[test]
fn test_wrap_and_unwrap_secret() {
    let vault = create_test_vault();

    vault
        .set(Vault::ROOT, "wrap/target", "wrapped-value")
        .unwrap();

    let token = vault
        .wrap_secret(Vault::ROOT, "wrap/target", 60_000)
        .unwrap();
    assert!(!token.is_empty());

    let unwrapped = vault.unwrap_secret(&token).unwrap();
    assert_eq!(unwrapped, "wrapped-value");
}

#[test]
fn test_wrapping_token_single_use() {
    let vault = create_test_vault();

    vault
        .set(Vault::ROOT, "wrap/single", "single-use-value")
        .unwrap();

    let token = vault
        .wrap_secret(Vault::ROOT, "wrap/single", 60_000)
        .unwrap();

    // First unwrap succeeds
    let value = vault.unwrap_secret(&token).unwrap();
    assert_eq!(value, "single-use-value");

    // Second unwrap fails because the token was deleted after use
    let second = vault.unwrap_secret(&token);
    assert!(second.is_err());
}

#[test]
fn test_wrapping_token_expired() {
    let vault = create_test_vault();

    vault
        .set(Vault::ROOT, "wrap/expire", "expire-value")
        .unwrap();

    // Use a very short TTL
    let token = vault.wrap_secret(Vault::ROOT, "wrap/expire", 1).unwrap();

    thread::sleep(Duration::from_millis(50));

    let result = vault.unwrap_secret(&token);
    assert!(result.is_err());
    assert!(matches!(result, Err(VaultError::WrappingTokenExpired(_))));
}

#[test]
fn test_wrapping_token_info() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "wrap/info", "info-value").unwrap();

    let token = vault.wrap_secret(Vault::ROOT, "wrap/info", 60_000).unwrap();

    let info = vault.wrapping_token_info(&token);
    assert!(info.is_some());

    let info = info.unwrap();
    assert_eq!(info.token, token);
    assert!(!info.consumed);
    assert!(info.expires_at_ms > info.created_at_ms);
}

#[test]
fn test_wrapping_token_info_nonexistent() {
    let vault = create_test_vault();

    let info = vault.wrapping_token_info("nonexistent-token-hex");
    assert!(info.is_none());
}

#[test]
fn test_wrap_multiple_secrets_independently() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "wrap/a", "value-a").unwrap();
    vault.set(Vault::ROOT, "wrap/b", "value-b").unwrap();

    let token_a = vault.wrap_secret(Vault::ROOT, "wrap/a", 60_000).unwrap();
    let token_b = vault.wrap_secret(Vault::ROOT, "wrap/b", 60_000).unwrap();

    assert_ne!(token_a, token_b);

    let val_a = vault.unwrap_secret(&token_a).unwrap();
    let val_b = vault.unwrap_secret(&token_b).unwrap();

    assert_eq!(val_a, "value-a");
    assert_eq!(val_b, "value-b");
}

// ==========================================================================
// Rotation policy tests
// ==========================================================================

#[test]
fn test_set_and_get_rotation_policy() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/secret", "initial").unwrap();

    let policy = RotationPolicy {
        secret_key: "rotate/secret".to_string(),
        interval_ms: 3_600_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 300_000,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/secret", policy)
        .unwrap();

    let retrieved = vault.get_rotation_policy(Vault::ROOT, "rotate/secret");
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.secret_key, "rotate/secret");
    assert_eq!(retrieved.interval_ms, 3_600_000);
}

#[test]
fn test_remove_rotation_policy() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/remove", "initial").unwrap();

    let policy = RotationPolicy {
        secret_key: "rotate/remove".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Token(TokenConfig::default()),
        notify_before_ms: 100,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/remove", policy)
        .unwrap();

    assert!(vault
        .get_rotation_policy(Vault::ROOT, "rotate/remove")
        .is_some());

    vault
        .remove_rotation_policy(Vault::ROOT, "rotate/remove")
        .unwrap();

    assert!(vault
        .get_rotation_policy(Vault::ROOT, "rotate/remove")
        .is_none());
}

#[test]
fn test_check_pending_rotations_overdue() {
    let vault = create_test_vault();

    vault
        .set(Vault::ROOT, "rotate/overdue", "old-value")
        .unwrap();

    // Set policy with last_rotated_ms=0 and interval_ms=1, so it is overdue
    let policy = RotationPolicy {
        secret_key: "rotate/overdue".to_string(),
        interval_ms: 1,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/overdue", policy)
        .unwrap();

    thread::sleep(Duration::from_millis(10));

    let pending = vault.check_pending_rotations();
    assert!(!pending.is_empty());

    let found = pending
        .iter()
        .any(|p| p.secret_key == "rotate/overdue" && p.overdue_ms > 0);
    assert!(found, "Expected overdue rotation for rotate/overdue");
}

#[test]
fn test_execute_rotation_generates_new_value() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/exec", "original").unwrap();

    let policy = RotationPolicy {
        secret_key: "rotate/exec".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/exec", policy)
        .unwrap();

    let new_value = vault.execute_rotation(Vault::ROOT, "rotate/exec").unwrap();
    assert!(!new_value.is_empty());
    assert_ne!(new_value, "original");

    // The stored value should now be the new value
    let stored = vault.get(Vault::ROOT, "rotate/exec").unwrap();
    assert_eq!(stored, new_value);
}

#[test]
fn test_execute_rotation_with_no_generator_fails() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/nogen", "manual").unwrap();

    let policy = RotationPolicy {
        secret_key: "rotate/nogen".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::None,
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/nogen", policy)
        .unwrap();

    let result = vault.execute_rotation(Vault::ROOT, "rotate/nogen");
    assert!(result.is_err());
    assert!(matches!(result, Err(VaultError::InvalidKey(_))));
}

#[test]
fn test_execute_rotation_without_policy_fails() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/nopolicy", "value").unwrap();

    let result = vault.execute_rotation(Vault::ROOT, "rotate/nopolicy");
    assert!(result.is_err());
    assert!(matches!(result, Err(VaultError::NotFound(_))));
}

#[test]
fn test_list_rotation_policies() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/a", "value-a").unwrap();
    vault.set(Vault::ROOT, "rotate/b", "value-b").unwrap();

    let policy_a = RotationPolicy {
        secret_key: "rotate/a".to_string(),
        interval_ms: 3_600_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 0,
    };
    let policy_b = RotationPolicy {
        secret_key: "rotate/b".to_string(),
        interval_ms: 7_200_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Token(TokenConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/a", policy_a)
        .unwrap();
    vault
        .set_rotation_policy(Vault::ROOT, "rotate/b", policy_b)
        .unwrap();

    let policies = vault.list_rotation_policies();
    assert!(policies.len() >= 2);
}

#[test]
fn test_execute_rotation_with_token_generator() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rotate/token", "old-token").unwrap();

    let policy = RotationPolicy {
        secret_key: "rotate/token".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Token(TokenConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rotate/token", policy)
        .unwrap();

    let new_value = vault.execute_rotation(Vault::ROOT, "rotate/token").unwrap();
    assert!(!new_value.is_empty());
    assert_ne!(new_value, "old-token");
}

// ==========================================================================
// Engine registry tests
// ==========================================================================

#[test]
fn test_register_and_list_engines() {
    let vault = create_test_vault();

    let pki = PkiEngine::new();
    vault.register_engine(Box::new(pki)).unwrap();

    let engines = vault.list_engines();
    assert!(engines.contains(&"pki".to_string()));
}

#[test]
fn test_engine_generate_with_pki() {
    let vault = create_test_vault();

    let pki = PkiEngine::new();
    vault.register_engine(Box::new(pki)).unwrap();

    let params = serde_json::json!({ "common_name": "test.example.com" });
    let result = vault.engine_generate(Vault::ROOT, "pki", &params).unwrap();

    assert!(result.contains("test.example.com"));
}

#[test]
fn test_engine_generate_nonexistent_engine() {
    let vault = create_test_vault();

    let params = serde_json::json!({});
    let result = vault.engine_generate(Vault::ROOT, "nonexistent", &params);

    assert!(result.is_err());
    assert!(matches!(result, Err(VaultError::EngineNotFound(_))));
}

#[test]
fn test_engine_revoke() {
    let vault = create_test_vault();

    let pki = PkiEngine::new();
    vault.register_engine(Box::new(pki)).unwrap();

    let params = serde_json::json!({});
    let secret_id = vault.engine_generate(Vault::ROOT, "pki", &params).unwrap();

    let revoke_result = vault.engine_revoke(Vault::ROOT, "pki", &secret_id);
    assert!(revoke_result.is_ok());
}

#[test]
fn test_unregister_engine() {
    let vault = create_test_vault();

    let pki = PkiEngine::new();
    vault.register_engine(Box::new(pki)).unwrap();

    assert!(vault.list_engines().contains(&"pki".to_string()));

    vault.unregister_engine("pki").unwrap();

    assert!(!vault.list_engines().contains(&"pki".to_string()));

    // Generating from an unregistered engine should fail
    let params = serde_json::json!({});
    let result = vault.engine_generate(Vault::ROOT, "pki", &params);
    assert!(matches!(result, Err(VaultError::EngineNotFound(_))));
}

// ==========================================================================
// Similarity and duplication tests
// ==========================================================================

#[test]
fn test_find_similar_secrets() {
    let vault = create_test_vault();

    // Store several secrets so the similarity index has data to compare
    vault.set(Vault::ROOT, "db/password", "pass1").unwrap();
    vault.set(Vault::ROOT, "db/backup", "pass2").unwrap();
    vault.set(Vault::ROOT, "api/key", "key1").unwrap();

    let results = vault.find_similar(Vault::ROOT, "db/password", 5).unwrap();

    // Should return results (may vary depending on HNSW behavior)
    // At minimum we should get a result set without errors
    assert!(results.len() <= 5);
}

#[test]
fn test_check_duplication_among_secrets() {
    let vault = create_test_vault();

    // Store secrets with similar operational profiles
    vault.set(Vault::ROOT, "dup/a", "value-alpha").unwrap();
    vault.set(Vault::ROOT, "dup/b", "value-beta").unwrap();

    // Use a low threshold to increase likelihood of matches
    let duplicates = vault.check_duplication(Vault::ROOT, 0.0).unwrap();

    // With a threshold of 0.0, should find at least the pair
    assert!(!duplicates.is_empty());
}

// ==========================================================================
// Cross-module interaction tests
// ==========================================================================

#[test]
fn test_dynamic_secret_with_wrapping() {
    let vault = create_test_vault();

    // Generate a dynamic secret, then wrap it via a regular set
    let template = SecretTemplate::Password(PasswordConfig::default());
    let (id, value) = vault
        .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
        .unwrap();

    // Wrap the dynamic secret
    let token = vault.wrap_secret(Vault::ROOT, &id, 60_000).unwrap();

    // Unwrap it
    let unwrapped = vault.unwrap_secret(&token).unwrap();
    assert_eq!(unwrapped, value);
}

#[test]
fn test_rotation_then_diff_versions() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rot/diff", "version-one").unwrap();

    let policy = RotationPolicy {
        secret_key: "rot/diff".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rot/diff", policy)
        .unwrap();

    vault.execute_rotation(Vault::ROOT, "rot/diff").unwrap();

    // Now we should have version 1 (original) and version 2 (rotated)
    let diff = vault.diff_versions(Vault::ROOT, "rot/diff", 1, 2).unwrap();

    assert_eq!(diff.key, "rot/diff");
    assert_eq!(diff.version_a, 1);
    assert_eq!(diff.version_b, 2);
    assert_eq!(diff.value_a, "version-one");
    assert_ne!(diff.value_b, "version-one");
}

#[test]
fn test_changelog_after_rotation() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "rot/changelog", "initial").unwrap();

    let policy = RotationPolicy {
        secret_key: "rot/changelog".to_string(),
        interval_ms: 1_000,
        last_rotated_ms: 0,
        generator: RotationGenerator::Password(PasswordConfig::default()),
        notify_before_ms: 0,
    };

    vault
        .set_rotation_policy(Vault::ROOT, "rot/changelog", policy)
        .unwrap();

    vault
        .execute_rotation(Vault::ROOT, "rot/changelog")
        .unwrap();

    let entries = vault.changelog(Vault::ROOT, "rot/changelog").unwrap();
    assert!(!entries.is_empty());
}

#[test]
fn test_list_paginated_secrets() {
    let vault = create_test_vault();

    for i in 0..10 {
        vault
            .set(Vault::ROOT, &format!("page/item{i}"), &format!("val{i}"))
            .unwrap();
    }

    let page = vault.list_paginated(Vault::ROOT, "page/*", 0, 3).unwrap();

    assert_eq!(page.limit, 3);
    assert!(page.secrets.len() <= 3);
    assert!(page.total >= 10);
    assert!(page.has_more);

    // Second page
    let page2 = vault.list_paginated(Vault::ROOT, "page/*", 3, 3).unwrap();
    assert_eq!(page2.offset, 3);
}
