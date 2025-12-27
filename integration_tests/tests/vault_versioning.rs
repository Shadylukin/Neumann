//! Vault versioning integration tests.
//!
//! Tests secret versioning, rollback, and version management.

use graph_engine::GraphEngine;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};

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

fn create_vault_with_max_versions(max: usize) -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let config = VaultConfig::default().with_max_versions(max);
    Vault::new(b"test-key-32-bytes-long!!!!!", graph, store, config).unwrap()
}

#[test]
fn test_version_increments_on_set() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 1);

    vault.set(Vault::ROOT, "secret", "v2").unwrap();
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

    vault.set(Vault::ROOT, "secret", "v3").unwrap();
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
}

#[test]
fn test_get_version_specific() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "first").unwrap();
    vault.set(Vault::ROOT, "secret", "second").unwrap();
    vault.set(Vault::ROOT, "secret", "third").unwrap();

    // Get each version explicitly
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
        "first"
    );
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
        "second"
    );
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 3).unwrap(),
        "third"
    );

    // Current (latest) should be third
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "third");
}

#[test]
fn test_get_version_not_found() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "only_one").unwrap();

    // Version 2 doesn't exist
    let result = vault.get_version(Vault::ROOT, "secret", 2);
    assert!(result.is_err());

    // Very high version doesn't exist
    let result = vault.get_version(Vault::ROOT, "secret", 100);
    assert!(result.is_err());

    // Note: Version 0 uses saturating_sub, so it maps to index 0 (version 1)
    // This is implementation-specific behavior
    let result = vault.get_version(Vault::ROOT, "secret", 0);
    // Version 0 maps to version 1 due to saturating subtraction
    assert_eq!(result.unwrap(), "only_one");
}

#[test]
fn test_list_versions() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    thread::sleep(Duration::from_millis(10));
    vault.set(Vault::ROOT, "secret", "v2").unwrap();
    thread::sleep(Duration::from_millis(10));
    vault.set(Vault::ROOT, "secret", "v3").unwrap();

    let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();

    assert_eq!(versions.len(), 3);
    assert_eq!(versions[0].version, 1);
    assert_eq!(versions[1].version, 2);
    assert_eq!(versions[2].version, 3);

    // Timestamps should be in order
    assert!(versions[0].created_at <= versions[1].created_at);
    assert!(versions[1].created_at <= versions[2].created_at);
}

#[test]
fn test_rollback_to_previous_version() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "original").unwrap();
    vault.set(Vault::ROOT, "secret", "modified").unwrap();
    vault.set(Vault::ROOT, "secret", "latest").unwrap();

    // Current should be latest
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "latest");
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);

    // Rollback to version 1
    vault.rollback(Vault::ROOT, "secret", 1).unwrap();

    // Current should now be "original" (as a new version)
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "original");

    // Should now have 4 versions (rollback creates a new version)
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 4);

    // Version 4 should equal version 1
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 4).unwrap(),
        vault.get_version(Vault::ROOT, "secret", 1).unwrap()
    );
}

#[test]
fn test_max_versions_pruning() {
    let vault = create_vault_with_max_versions(3);

    // Set 5 versions
    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    vault.set(Vault::ROOT, "secret", "v2").unwrap();
    vault.set(Vault::ROOT, "secret", "v3").unwrap();
    vault.set(Vault::ROOT, "secret", "v4").unwrap();
    vault.set(Vault::ROOT, "secret", "v5").unwrap();

    // Should only have 3 versions (v3, v4, v5)
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);

    let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();
    assert_eq!(versions.len(), 3);

    // Version 1 should now be "v3" (oldest kept)
    assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v3");
    assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v4");
    assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v5");
}

#[test]
fn test_rotate_keeps_versions() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "initial").unwrap();
    vault.rotate(Vault::ROOT, "secret", "rotated").unwrap();

    // Should have 2 versions
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

    // Can get both versions
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
        "initial"
    );
    assert_eq!(
        vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
        "rotated"
    );
}

#[test]
fn test_rotate_respects_max_versions() {
    let vault = create_vault_with_max_versions(2);

    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    vault.rotate(Vault::ROOT, "secret", "v2").unwrap();
    vault.rotate(Vault::ROOT, "secret", "v3").unwrap();

    // Should only have 2 versions
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

    // Oldest should be v2
    assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v2");
    assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v3");
}

#[test]
fn test_delete_removes_all_versions() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    vault.set(Vault::ROOT, "secret", "v2").unwrap();
    vault.set(Vault::ROOT, "secret", "v3").unwrap();

    // Delete the secret
    vault.delete(Vault::ROOT, "secret").unwrap();

    // Secret should not exist
    let result = vault.get(Vault::ROOT, "secret");
    assert!(result.is_err());

    // Versions should not exist
    let result = vault.get_version(Vault::ROOT, "secret", 1);
    assert!(result.is_err());

    let result = vault.list_versions(Vault::ROOT, "secret");
    assert!(result.is_err());
}

#[test]
fn test_version_access_control() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vault = Vault::new(
        b"test-key-32-bytes-long!!!!!",
        graph.clone(),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    vault.set(Vault::ROOT, "secret", "value").unwrap();

    // Alice has no access
    let result = vault.get_version("user:alice", "secret", 1);
    assert!(result.is_err());

    // Grant access to alice
    vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

    // Now alice can access versions
    assert_eq!(
        vault.get_version("user:alice", "secret", 1).unwrap(),
        "value"
    );
}

#[test]
fn test_multiple_secrets_versioned_independently() {
    let vault = create_test_vault();

    // Create versions for secret_a
    vault.set(Vault::ROOT, "secret_a", "a1").unwrap();
    vault.set(Vault::ROOT, "secret_a", "a2").unwrap();
    vault.set(Vault::ROOT, "secret_a", "a3").unwrap();

    // Create versions for secret_b
    vault.set(Vault::ROOT, "secret_b", "b1").unwrap();
    vault.set(Vault::ROOT, "secret_b", "b2").unwrap();

    // Verify independent versioning
    assert_eq!(vault.current_version(Vault::ROOT, "secret_a").unwrap(), 3);
    assert_eq!(vault.current_version(Vault::ROOT, "secret_b").unwrap(), 2);

    assert_eq!(vault.get_version(Vault::ROOT, "secret_a", 1).unwrap(), "a1");
    assert_eq!(vault.get_version(Vault::ROOT, "secret_b", 1).unwrap(), "b1");
}

#[test]
fn test_version_timestamps_monotonic() {
    let vault = create_test_vault();

    for i in 0..5 {
        vault
            .set(Vault::ROOT, "secret", &format!("v{}", i))
            .unwrap();
        thread::sleep(Duration::from_millis(5));
    }

    let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();

    for i in 1..versions.len() {
        assert!(
            versions[i].created_at >= versions[i - 1].created_at,
            "Version {} timestamp ({}) should be >= version {} timestamp ({})",
            i + 1,
            versions[i].created_at,
            i,
            versions[i - 1].created_at
        );
    }
}

#[test]
fn test_rollback_multiple_times() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret", "v1").unwrap();
    vault.set(Vault::ROOT, "secret", "v2").unwrap();
    vault.set(Vault::ROOT, "secret", "v3").unwrap();

    // Rollback to v1
    vault.rollback(Vault::ROOT, "secret", 1).unwrap();
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v1");

    // Add new version
    vault.set(Vault::ROOT, "secret", "v5").unwrap();

    // Rollback to v2
    vault.rollback(Vault::ROOT, "secret", 2).unwrap();
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v2");
}

#[test]
fn test_concurrent_version_updates() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vault = Arc::new(
        Vault::new(
            b"test-key-32-bytes-long!!!!!",
            graph,
            store,
            VaultConfig::default(),
        )
        .unwrap(),
    );

    // Initialize secret
    vault.set(Vault::ROOT, "counter", "0").unwrap();

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for t in 0..4 {
        let vault = Arc::clone(&vault);
        let counter = Arc::clone(&success_count);

        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let value = format!("thread{}_{}", t, i);
                if vault.set(Vault::ROOT, "counter", &value).is_ok() {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All updates should succeed
    assert_eq!(success_count.load(Ordering::SeqCst), 40);

    // Version count depends on max_versions config
    let version = vault.current_version(Vault::ROOT, "counter").unwrap();
    assert!(version >= 1);
}

#[test]
fn test_version_with_special_characters() {
    let vault = create_test_vault();

    // Values with special characters
    let values = [
        "value with spaces",
        "value\nwith\nnewlines",
        "value\twith\ttabs",
        "unicode: \u{1F600}\u{1F389}",
        "quotes: \"hello\"",
    ];

    for (i, value) in values.iter().enumerate() {
        vault
            .set(Vault::ROOT, "special", value)
            .expect(&format!("Failed to set value {}", i));
    }

    // Verify all versions are retrievable
    for (i, expected) in values.iter().enumerate() {
        let actual = vault
            .get_version(Vault::ROOT, "special", (i + 1) as u32)
            .unwrap();
        assert_eq!(&actual, *expected, "Version {} mismatch", i + 1);
    }
}

#[test]
fn test_min_version_is_one() {
    let vault = create_vault_with_max_versions(1);

    // Even with max_versions=1, we should keep at least 1 version
    vault.set(Vault::ROOT, "secret", "only").unwrap();

    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 1);
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "only");

    // Update
    vault.set(Vault::ROOT, "secret", "new").unwrap();

    // Still only 1 version
    assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 1);
    assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "new");
}

#[test]
fn test_version_large_values() {
    let vault = create_test_vault();

    // Create large values
    let large_v1 = "x".repeat(10_000);
    let large_v2 = "y".repeat(10_000);
    let large_v3 = "z".repeat(10_000);

    vault.set(Vault::ROOT, "large", &large_v1).unwrap();
    vault.set(Vault::ROOT, "large", &large_v2).unwrap();
    vault.set(Vault::ROOT, "large", &large_v3).unwrap();

    // Verify all versions
    assert_eq!(
        vault.get_version(Vault::ROOT, "large", 1).unwrap(),
        large_v1
    );
    assert_eq!(
        vault.get_version(Vault::ROOT, "large", 2).unwrap(),
        large_v2
    );
    assert_eq!(
        vault.get_version(Vault::ROOT, "large", 3).unwrap(),
        large_v3
    );
}
