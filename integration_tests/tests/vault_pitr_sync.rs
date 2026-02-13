// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for vault PITR, sync targets, geo routing, namespacing,
//! scoped views, and rate limiting cross-module interactions.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    FileSyncTarget, GeoCoordinate, GeoRouter, Permission, RateLimitConfig, RoutingConfig,
    TargetGeometry, Vault, VaultConfig, VaultError,
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

fn create_test_vault_with_config(config: VaultConfig) -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    Vault::new(b"test-key-32-bytes-long!!!!!", graph, store, config).unwrap()
}

// ========== PITR Tests (1-7) ==========

#[test]
fn test_create_snapshot() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "pitr/secret1", "value1").unwrap();
    vault.set(Vault::ROOT, "pitr/secret2", "value2").unwrap();

    let snap = vault.create_snapshot(Vault::ROOT, "test backup").unwrap();

    assert!(snap.id.starts_with("snap_"));
    assert_eq!(snap.label, "test backup");
    assert!(snap.created_at_ms > 0);
}

#[test]
fn test_list_snapshots() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "ls/key", "val").unwrap();

    vault.create_snapshot(Vault::ROOT, "snap-a").unwrap();
    vault.create_snapshot(Vault::ROOT, "snap-b").unwrap();
    vault.create_snapshot(Vault::ROOT, "snap-c").unwrap();

    let snapshots = vault.list_snapshots();
    assert_eq!(snapshots.len(), 3);
}

#[test]
fn test_restore_snapshot() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "rs/alpha", "original_a").unwrap();
    vault.set(Vault::ROOT, "rs/beta", "original_b").unwrap();

    let snap = vault
        .create_snapshot(Vault::ROOT, "before modification")
        .unwrap();

    // Modify secrets after snapshot
    vault.set(Vault::ROOT, "rs/alpha", "modified_a").unwrap();
    vault.set(Vault::ROOT, "rs/beta", "modified_b").unwrap();

    // Verify modified values
    assert_eq!(vault.get(Vault::ROOT, "rs/alpha").unwrap(), "modified_a");
    assert_eq!(vault.get(Vault::ROOT, "rs/beta").unwrap(), "modified_b");

    // Restore from snapshot
    vault.restore_snapshot(Vault::ROOT, &snap.id).unwrap();

    // Originals should be back
    assert_eq!(vault.get(Vault::ROOT, "rs/alpha").unwrap(), "original_a");
    assert_eq!(vault.get(Vault::ROOT, "rs/beta").unwrap(), "original_b");
}

#[test]
fn test_restore_count() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "rc/one", "v1").unwrap();
    vault.set(Vault::ROOT, "rc/two", "v2").unwrap();
    vault.set(Vault::ROOT, "rc/three", "v3").unwrap();

    let snap = vault.create_snapshot(Vault::ROOT, "count check").unwrap();

    // Modify to force a different state
    vault.set(Vault::ROOT, "rc/one", "changed").unwrap();

    let count = vault.restore_snapshot(Vault::ROOT, &snap.id).unwrap();
    // At least 3 secret entries should be restored (may include blob data)
    assert!(count >= 3, "expected at least 3, got {count}");
}

#[test]
fn test_delete_snapshot() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "ds/key", "val").unwrap();

    let snap = vault.create_snapshot(Vault::ROOT, "to delete").unwrap();

    // Verify it exists
    let before = vault.list_snapshots();
    assert!(before.iter().any(|s| s.id == snap.id));

    // Delete it
    vault.delete_snapshot(Vault::ROOT, &snap.id).unwrap();

    // Verify it's gone
    let after = vault.list_snapshots();
    assert!(!after.iter().any(|s| s.id == snap.id));
}

#[test]
fn test_snapshot_non_root_denied() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "nd/key", "val").unwrap();

    let result = vault.create_snapshot("user:mallory", "sneaky");
    assert!(result.is_err());
    match result.unwrap_err() {
        VaultError::AccessDenied(msg) => {
            assert!(msg.contains("root"), "expected root in message, got: {msg}");
        },
        other => panic!("expected AccessDenied, got: {other}"),
    }
}

#[test]
fn test_snapshot_add_restore() {
    let vault = create_test_vault();

    // Create initial secrets and snapshot
    vault.set(Vault::ROOT, "sar/initial", "present").unwrap();
    let snap = vault.create_snapshot(Vault::ROOT, "initial state").unwrap();

    // Add more secrets after snapshot
    vault
        .set(Vault::ROOT, "sar/added", "after snapshot")
        .unwrap();

    // Restore to snapshot -- the "added" secret persists (restore overwrites, does not delete)
    // but the initial one should be back at original value
    vault.set(Vault::ROOT, "sar/initial", "changed").unwrap();
    vault.restore_snapshot(Vault::ROOT, &snap.id).unwrap();

    assert_eq!(vault.get(Vault::ROOT, "sar/initial").unwrap(), "present");
}

// ========== Sync Tests (8-12) ==========

#[test]
fn test_register_sync_target() {
    let vault = create_test_vault();
    let dir = tempfile::tempdir().unwrap();

    vault
        .register_sync_target(Box::new(FileSyncTarget::new(
            "file-target",
            dir.path().to_path_buf(),
        )))
        .unwrap();

    let targets = vault.list_sync_targets();
    assert!(targets.contains(&"file-target".to_string()));
}

#[test]
fn test_subscribe_unsubscribe() {
    let vault = create_test_vault();
    let dir = tempfile::tempdir().unwrap();

    vault
        .register_sync_target(Box::new(FileSyncTarget::new(
            "sub-target",
            dir.path().to_path_buf(),
        )))
        .unwrap();
    vault.set(Vault::ROOT, "sub/key", "sub-value").unwrap();

    // Subscribe
    vault
        .subscribe_sync(Vault::ROOT, "sub/key", "sub-target")
        .unwrap();

    // Trigger sync should succeed with count > 0
    let count = vault.trigger_sync(Vault::ROOT, "sub/key").unwrap();
    assert!(count > 0, "expected sync count > 0 after subscribe");

    // Unsubscribe
    vault
        .unsubscribe_sync(Vault::ROOT, "sub/key", "sub-target")
        .unwrap();

    // Trigger sync should now return 0
    let count_after = vault.trigger_sync(Vault::ROOT, "sub/key").unwrap();
    assert_eq!(count_after, 0, "expected 0 after unsubscribe");
}

#[test]
fn test_trigger_sync() {
    let vault = create_test_vault();
    let dir = tempfile::tempdir().unwrap();

    vault
        .register_sync_target(Box::new(FileSyncTarget::new(
            "trigger-target",
            dir.path().to_path_buf(),
        )))
        .unwrap();
    vault.set(Vault::ROOT, "trigger/db", "password123").unwrap();

    vault
        .subscribe_sync(Vault::ROOT, "trigger/db", "trigger-target")
        .unwrap();

    let count = vault.trigger_sync(Vault::ROOT, "trigger/db").unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_sync_health() {
    let vault = create_test_vault();
    let dir = tempfile::tempdir().unwrap();

    vault
        .register_sync_target(Box::new(FileSyncTarget::new(
            "health-target",
            dir.path().to_path_buf(),
        )))
        .unwrap();

    let health = vault.sync_health();
    assert_eq!(health.len(), 1);
    // FileSyncTarget reports healthy if the base_dir exists
    let (name, healthy) = &health[0];
    assert_eq!(name, "health-target");
    assert!(healthy, "expected healthy target");
}

#[test]
fn test_geo_router_route() {
    let config = RoutingConfig {
        sync_fanout: 2,
        ..RoutingConfig::default()
    };
    let router = GeoRouter::new(config);

    router.update_geometry(TargetGeometry {
        target_name: "us-east".to_string(),
        location: GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        },
        avg_latency_ms: 20.0,
        avg_throughput: 100.0,
        failure_rate: 0.01,
        last_health_check_ms: 0,
    });
    router.update_geometry(TargetGeometry {
        target_name: "eu-west".to_string(),
        location: GeoCoordinate {
            x: 50.0,
            y: 50.0,
            z: None,
        },
        avg_latency_ms: 80.0,
        avg_throughput: 90.0,
        failure_rate: 0.02,
        last_health_check_ms: 0,
    });

    let available = vec!["us-east".to_string(), "eu-west".to_string()];
    let decision = router.route("secret/key", None, &available);

    assert_eq!(decision.selected_targets.len(), 2);
    assert!(decision.excluded_targets.is_empty());
    // Both should be selected; the one with lower latency scores higher
    assert_eq!(decision.selected_targets[0].target_name, "us-east");
}

// ========== GeoRouter Tests (13-14) ==========

#[test]
fn test_geo_router_exclude() {
    let config = RoutingConfig {
        max_latency_ms: 100.0,
        sync_fanout: 3,
        ..RoutingConfig::default()
    };
    let router = GeoRouter::new(config);

    router.update_geometry(TargetGeometry {
        target_name: "fast".to_string(),
        location: GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        },
        avg_latency_ms: 30.0,
        avg_throughput: 100.0,
        failure_rate: 0.0,
        last_health_check_ms: 0,
    });
    router.update_geometry(TargetGeometry {
        target_name: "slow".to_string(),
        location: GeoCoordinate {
            x: 10.0,
            y: 10.0,
            z: None,
        },
        avg_latency_ms: 500.0,
        avg_throughput: 50.0,
        failure_rate: 0.0,
        last_health_check_ms: 0,
    });

    let available = vec!["fast".to_string(), "slow".to_string()];
    let decision = router.route("secret/key", None, &available);

    assert_eq!(decision.selected_targets.len(), 1);
    assert_eq!(decision.selected_targets[0].target_name, "fast");
    assert_eq!(decision.excluded_targets.len(), 1);
    assert_eq!(decision.excluded_targets[0].target_name, "slow");
}

#[test]
fn test_geo_router_ema_update() {
    let router = GeoRouter::new(RoutingConfig::default());
    router.update_geometry(TargetGeometry {
        target_name: "ema-target".to_string(),
        location: GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        },
        avg_latency_ms: 100.0,
        avg_throughput: 100.0,
        failure_rate: 0.0,
        last_health_check_ms: 0,
    });

    // Record several results with higher latency
    for _ in 0..5 {
        router.record_sync_result("ema-target", 200.0, true);
    }

    let geometries = router.geometries();
    let target = geometries
        .iter()
        .find(|g| g.target_name == "ema-target")
        .expect("target should exist");

    // After 5 EMA updates from 100 toward 200, the latency should have converged
    // upward significantly (each step: alpha=0.2 * 200 + 0.8 * prev)
    assert!(
        target.avg_latency_ms > 140.0,
        "expected latency to converge toward 200, got {}",
        target.avg_latency_ms
    );
    // Failure rate should remain near 0 since all results were successful
    assert!(
        target.failure_rate < 0.01,
        "expected low failure rate, got {}",
        target.failure_rate
    );
}

// ========== Namespace Tests (15-18) ==========

#[test]
fn test_namespace_isolation() {
    let vault = create_test_vault();

    let ns_prod = vault.namespace("prod", Vault::ROOT);
    let ns_staging = vault.namespace("staging", Vault::ROOT);

    ns_prod.set("db/password", "prod-secret").unwrap();
    ns_staging.set("db/password", "staging-secret").unwrap();

    assert_eq!(ns_prod.get("db/password").unwrap(), "prod-secret");
    assert_eq!(ns_staging.get("db/password").unwrap(), "staging-secret");
}

#[test]
fn test_namespace_list() {
    let vault = create_test_vault();
    let ns = vault.namespace("list-ns", Vault::ROOT);

    ns.set("config/a", "va").unwrap();
    ns.set("config/b", "vb").unwrap();
    ns.set("config/c", "vc").unwrap();

    let keys = ns.list("config/*").unwrap();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"config/a".to_string()));
    assert!(keys.contains(&"config/b".to_string()));
    assert!(keys.contains(&"config/c".to_string()));
}

#[test]
fn test_namespace_encrypt_decrypt() {
    let vault = create_test_vault();
    let ns = vault.namespace("enc-ns", Vault::ROOT);

    // set stores encrypted data; get decrypts it
    ns.set("secret/token", "my-super-secret-token").unwrap();
    let retrieved = ns.get("secret/token").unwrap();
    assert_eq!(retrieved, "my-super-secret-token");

    // Rotate and verify updated value
    ns.rotate("secret/token", "rotated-token").unwrap();
    let after_rotate = ns.get("secret/token").unwrap();
    assert_eq!(after_rotate, "rotated-token");
}

#[test]
fn test_namespace_batch() {
    let vault = create_test_vault();
    let ns = vault.namespace("batch-ns", Vault::ROOT);

    // batch_set
    let entries = vec![
        ("DB_HOST", "localhost"),
        ("DB_PORT", "5432"),
        ("DB_NAME", "mydb"),
    ];
    ns.batch_set(&entries).unwrap();

    // batch_get
    let results = ns.batch_get(&["DB_HOST", "DB_PORT", "DB_NAME"]).unwrap();
    assert_eq!(results.len(), 3);

    for (key, value_result) in &results {
        let value = value_result.as_ref().unwrap();
        match key.as_str() {
            "DB_HOST" => assert_eq!(value, "localhost"),
            "DB_PORT" => assert_eq!(value, "5432"),
            "DB_NAME" => assert_eq!(value, "mydb"),
            other => panic!("unexpected key: {other}"),
        }
    }
}

// ========== Scoped Tests (19-23) ==========

#[test]
fn test_scoped_lifecycle() {
    let vault = create_test_vault();
    let scoped = vault.scope(Vault::ROOT);

    // Set
    scoped.set("life/secret", "initial").unwrap();
    assert_eq!(scoped.get("life/secret").unwrap(), "initial");

    // Rotate
    scoped.rotate("life/secret", "rotated").unwrap();
    assert_eq!(scoped.get("life/secret").unwrap(), "rotated");

    // List
    let keys = scoped.list("life/*").unwrap();
    assert!(keys.contains(&"life/secret".to_string()));

    // Delete
    scoped.delete("life/secret").unwrap();
    assert!(scoped.get("life/secret").is_err());
}

#[test]
fn test_scoped_delegation() {
    let vault = create_test_vault();

    // Root sets secrets and grants to parent
    vault.set(Vault::ROOT, "del/api-key", "key-abc").unwrap();
    vault.set(Vault::ROOT, "del/db-pass", "pass-xyz").unwrap();
    vault
        .grant(Vault::ROOT, "user:parent", "del/api-key")
        .unwrap();
    vault
        .grant(Vault::ROOT, "user:parent", "del/db-pass")
        .unwrap();

    let scoped = vault.scope("user:parent");

    // Delegate read access on selected secrets to child
    let record = scoped
        .delegate(
            "user:child",
            &["del/api-key", "del/db-pass"],
            Permission::Read,
            Some(Duration::from_secs(3600)),
        )
        .unwrap();
    assert_eq!(record.child, "user:child");

    // Child should now be able to read
    assert_eq!(vault.get("user:child", "del/api-key").unwrap(), "key-abc");
    assert_eq!(vault.get("user:child", "del/db-pass").unwrap(), "pass-xyz");

    // Revoke delegation
    let revoked_keys = scoped.revoke_delegation("user:child").unwrap();
    assert!(!revoked_keys.is_empty());

    // Child should no longer have access
    assert!(vault.get("user:child", "del/api-key").is_err());
}

#[test]
fn test_scoped_wrapping() {
    let vault = create_test_vault();
    let scoped = vault.scope(Vault::ROOT);

    scoped.set("wrap/token", "wrapped-value").unwrap();

    // Wrap
    let token = scoped.wrap_secret("wrap/token", 60_000).unwrap();
    assert!(!token.is_empty());

    // Unwrap
    let value = scoped.unwrap_secret(&token).unwrap();
    assert_eq!(value, "wrapped-value");

    // Unwrap again should fail (single-use token)
    assert!(scoped.unwrap_secret(&token).is_err());
}

#[test]
fn test_scoped_dependency() {
    let vault = create_test_vault();
    let scoped = vault.scope(Vault::ROOT);

    scoped.set("dep/parent", "parent-val").unwrap();
    scoped.set("dep/child", "child-val").unwrap();
    scoped.set("dep/grandchild", "gc-val").unwrap();

    // Build dependency chain: parent -> child -> grandchild
    scoped.add_dependency("dep/parent", "dep/child").unwrap();
    scoped
        .add_dependency("dep/child", "dep/grandchild")
        .unwrap();

    // Impact analysis from parent should show affected secrets
    let report = scoped.impact_analysis("dep/parent").unwrap();
    // root_secret uses an obfuscated key internally, so just verify it's not empty
    assert!(!report.root_secret.is_empty());
    assert!(
        report.affected_secrets.len() >= 2,
        "expected at least 2 affected secrets, got {}",
        report.affected_secrets.len()
    );
}

#[test]
fn test_scoped_dynamic() {
    let vault = create_test_vault();
    let scoped = vault.scope(Vault::ROOT);

    let template = tensor_vault::SecretTemplate::Password(tensor_vault::PasswordConfig::default());

    // Generate
    let (secret_id, value) = scoped
        .generate_dynamic_secret(&template, 600_000, false)
        .unwrap();
    assert!(!secret_id.is_empty());
    assert!(!value.is_empty());

    // Get
    let retrieved = scoped.get_dynamic_secret(&secret_id).unwrap();
    assert!(!retrieved.is_empty());

    // List
    let list = scoped.list_dynamic_secrets().unwrap();
    assert!(list.iter().any(|m| m.secret_id == secret_id));

    // Revoke
    scoped.revoke_dynamic_secret(&secret_id).unwrap();
    let list_after = scoped.list_dynamic_secrets().unwrap();
    assert!(!list_after.iter().any(|m| m.secret_id == secret_id));
}

// ========== Rate Limit Tests (24-25) ==========

#[test]
fn test_rate_limit_enforced() {
    let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
        max_gets: 2,
        max_lists: 100,
        max_sets: 100,
        max_grants: 100,
        max_break_glass: 100,
        max_wraps: 100,
        max_generates: 100,
        window: Duration::from_secs(60),
    });

    let vault = create_test_vault_with_config(config);
    vault.set(Vault::ROOT, "rl/secret", "rate-value").unwrap();

    // Grant a non-root entity access (root is exempt from rate limits)
    vault
        .grant(Vault::ROOT, "user:limited", "rl/secret")
        .unwrap();

    // First two gets succeed for non-root entity
    assert!(vault.get("user:limited", "rl/secret").is_ok());
    assert!(vault.get("user:limited", "rl/secret").is_ok());

    // Third get should fail with RateLimited
    let result = vault.get("user:limited", "rl/secret");
    assert!(result.is_err());
    match result.unwrap_err() {
        VaultError::RateLimited(msg) => {
            assert!(
                msg.contains("Rate limit"),
                "expected rate limit message, got: {msg}"
            );
        },
        other => panic!("expected RateLimited, got: {other}"),
    }
}

#[test]
fn test_rate_limit_separate_entities() {
    let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
        max_gets: 2,
        max_lists: 100,
        max_sets: 100,
        max_grants: 100,
        max_break_glass: 100,
        max_wraps: 100,
        max_generates: 100,
        window: Duration::from_secs(60),
    });

    let vault = create_test_vault_with_config(config);
    vault.set(Vault::ROOT, "rls/shared", "shared-val").unwrap();
    vault
        .grant(Vault::ROOT, "user:alice", "rls/shared")
        .unwrap();
    vault.grant(Vault::ROOT, "user:bob", "rls/shared").unwrap();

    // Alice uses her 2 gets
    assert!(vault.get("user:alice", "rls/shared").is_ok());
    assert!(vault.get("user:alice", "rls/shared").is_ok());

    // Bob should still have his own quota
    assert!(vault.get("user:bob", "rls/shared").is_ok());
    assert!(vault.get("user:bob", "rls/shared").is_ok());

    // Both are now exhausted
    assert!(matches!(
        vault.get("user:alice", "rls/shared"),
        Err(VaultError::RateLimited(_))
    ));
    assert!(matches!(
        vault.get("user:bob", "rls/shared"),
        Err(VaultError::RateLimited(_))
    ));
}

// ========== Concurrent Test (26) ==========

#[test]
fn test_concurrent_snapshot_ops() {
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

    // Seed some secrets
    for i in 0..5 {
        vault
            .set(
                Vault::ROOT,
                &format!("conc/secret-{i}"),
                &format!("val-{i}"),
            )
            .unwrap();
    }

    let mut create_handles = Vec::new();
    let mut list_handles = Vec::new();

    // Spawn 4 threads that each create a snapshot
    for i in 0..4 {
        let v = Arc::clone(&vault);
        create_handles.push(thread::spawn(move || {
            v.create_snapshot(Vault::ROOT, &format!("concurrent-snap-{i}"))
                .unwrap()
        }));
    }

    // Spawn 2 threads that list snapshots while creation is happening
    for _ in 0..2 {
        let v = Arc::clone(&vault);
        list_handles.push(thread::spawn(move || {
            thread::sleep(Duration::from_millis(1));
            let snaps = v.list_snapshots();
            // At most 4 snapshots should be visible
            assert!(snaps.len() <= 4, "should not have more than 4 snapshots");
        }));
    }

    for handle in create_handles {
        handle.join().expect("thread should not panic");
    }
    for handle in list_handles {
        handle.join().expect("thread should not panic");
    }

    // After all threads complete, all 4 snapshots should be present
    let final_snapshots = vault.list_snapshots();
    assert_eq!(final_snapshots.len(), 4);
}
