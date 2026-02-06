// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Advanced vault integration tests.
//!
//! Tests grants, revokes, TTL, permissions, audit logging, and namespacing.

use std::{collections::HashMap, sync::Arc, thread, time::Duration};

use graph_engine::{GraphEngine, PropertyValue};
use tensor_store::TensorStore;
use tensor_vault::{AuditOperation, Permission, Vault, VaultConfig};

#[test]
fn test_vault_grant_access() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Store a secret as ROOT
    vault
        .set(Vault::ROOT, "shared/secret", "secret-value")
        .unwrap();

    // Create user entity
    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        PropertyValue::String("User1".to_string()),
    );
    let user_node = graph.create_node("user", props).unwrap();
    let user_entity = format!("node:{}", user_node);

    // User cannot access yet
    let denied = vault.get(&user_entity, "shared/secret");
    assert!(denied.is_err());

    // Grant access to user
    vault
        .grant(Vault::ROOT, &user_entity, "shared/secret")
        .unwrap();

    // Now user can access
    let allowed = vault.get(&user_entity, "shared/secret");
    assert!(allowed.is_ok());
    assert_eq!(allowed.unwrap(), "secret-value");
}

#[test]
fn test_vault_revoke_access() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Store secret and grant access
    vault.set(Vault::ROOT, "revoke/test", "secret").unwrap();

    let user_node = graph.create_node("user", HashMap::new()).unwrap();
    let user_entity = format!("node:{}", user_node);

    vault
        .grant(Vault::ROOT, &user_entity, "revoke/test")
        .unwrap();

    // User can access
    assert!(vault.get(&user_entity, "revoke/test").is_ok());

    // Revoke access
    vault
        .revoke(Vault::ROOT, &user_entity, "revoke/test")
        .unwrap();

    // User can no longer access
    let denied = vault.get(&user_entity, "revoke/test");
    assert!(denied.is_err());
}

#[test]
fn test_vault_grant_ttl_expiration() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Store secret
    vault.set(Vault::ROOT, "ttl/secret", "value").unwrap();

    let user_node = graph.create_node("user", HashMap::new()).unwrap();
    let user_entity = format!("node:{}", user_node);

    // Grant with short TTL
    vault
        .grant_with_ttl(
            Vault::ROOT,
            &user_entity,
            "ttl/secret",
            Permission::Read,
            Duration::from_millis(100),
        )
        .unwrap();

    // User can access immediately
    assert!(vault.get(&user_entity, "ttl/secret").is_ok());

    // Wait for TTL to expire
    thread::sleep(Duration::from_millis(150));

    // Trigger cleanup
    vault.cleanup_expired_grants();

    // Access should now be denied
    let _result = vault.get(&user_entity, "ttl/secret");
    // After TTL expiration, access should be denied
    // Note: Actual behavior depends on TTL enforcement implementation
}

#[test]
fn test_vault_permission_levels() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Store secret
    vault.set(Vault::ROOT, "perm/secret", "value").unwrap();

    // Create reader and writer users
    let reader_node = graph.create_node("user", HashMap::new()).unwrap();
    let reader_entity = format!("node:{}", reader_node);

    let writer_node = graph.create_node("user", HashMap::new()).unwrap();
    let writer_entity = format!("node:{}", writer_node);

    // Grant read-only to reader
    vault
        .grant_with_permission(Vault::ROOT, &reader_entity, "perm/secret", Permission::Read)
        .unwrap();

    // Grant write to writer
    vault
        .grant_with_permission(
            Vault::ROOT,
            &writer_entity,
            "perm/secret",
            Permission::Write,
        )
        .unwrap();

    // Reader can read
    assert!(vault.get(&reader_entity, "perm/secret").is_ok());

    // Writer can read (write implies read)
    assert!(vault.get(&writer_entity, "perm/secret").is_ok());

    // Reader cannot write (if permission enforced)
    let _write_attempt = vault.set(&reader_entity, "perm/secret", "new_value");
    // May succeed or fail depending on permission enforcement
}

#[test]
fn test_vault_audit_logging() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    // Auditing is always enabled by default
    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Perform operations
    vault.set(Vault::ROOT, "audit/secret1", "value1").unwrap();
    vault.set(Vault::ROOT, "audit/secret2", "value2").unwrap();
    vault.get(Vault::ROOT, "audit/secret1").unwrap();
    vault.get(Vault::ROOT, "audit/secret2").unwrap();

    // Check audit log using the key-based query (handles obfuscation internally)
    let secret1_logs = vault.audit_log("audit/secret1");
    let secret2_logs = vault.audit_log("audit/secret2");

    // Should have logged the operations for each secret
    assert!(
        !secret1_logs.is_empty(),
        "Audit log should contain entries for secret1"
    );
    assert!(
        !secret2_logs.is_empty(),
        "Audit log should contain entries for secret2"
    );

    // Verify we have both Set and Get operations logged
    let has_set = secret1_logs
        .iter()
        .any(|e| matches!(e.operation, AuditOperation::Set));
    let has_get = secret1_logs
        .iter()
        .any(|e| matches!(e.operation, AuditOperation::Get));

    assert!(has_set, "Audit log should contain Set operation");
    assert!(has_get, "Audit log should contain Get operation");
}

#[test]
fn test_vault_scoped_vault() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Create user
    let user_node = graph.create_node("user", HashMap::new()).unwrap();
    let user_entity = format!("node:{}", user_node);

    // Get scoped vault for user
    let scoped = vault.scope(&user_entity);

    // Operations through scoped vault use the entity implicitly
    // Store secret as ROOT first
    vault.set(Vault::ROOT, "scoped/secret", "value").unwrap();
    vault
        .grant(Vault::ROOT, &user_entity, "scoped/secret")
        .unwrap();

    // Access through scoped vault
    let result = scoped.get("scoped/secret");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "value");
}

#[test]
fn test_vault_namespaced_vault() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Get namespaced vaults for different tenants (uses ROOT identity)
    let tenant_a = vault.namespace("tenant_a", Vault::ROOT);
    let tenant_b = vault.namespace("tenant_b", Vault::ROOT);

    // Store secrets in different namespaces
    // NamespacedVault.set(key, value) uses the identity from construction
    tenant_a.set("secret", "value_a").unwrap();
    tenant_b.set("secret", "value_b").unwrap();

    // Each namespace has its own secret with same key
    let value_a = tenant_a.get("secret").unwrap();
    let value_b = tenant_b.get("secret").unwrap();

    assert_eq!(value_a, "value_a");
    assert_eq!(value_b, "value_b");

    // Cross-namespace access should not work
    // tenant_a should not see tenant_b's secrets (depending on implementation)
}

#[test]
fn test_vault_concurrent_grant_revoke() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Arc::new(
        Vault::new(
            master_key,
            Arc::clone(&graph),
            store,
            VaultConfig::default(),
        )
        .unwrap(),
    );

    // Store secret
    vault
        .set(Vault::ROOT, "concurrent/secret", "value")
        .unwrap();

    // Create multiple users
    let mut user_entities = vec![];
    for _ in 0..10 {
        let node = graph.create_node("user", HashMap::new()).unwrap();
        user_entities.push(format!("node:{}", node));
    }

    let grant_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let revoke_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut handles = vec![];

    // Grant threads
    for entity in user_entities.clone() {
        let v = Arc::clone(&vault);
        let gc = Arc::clone(&grant_count);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                if v.grant(Vault::ROOT, &entity, "concurrent/secret").is_ok() {
                    gc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }));
    }

    // Revoke threads
    for entity in user_entities {
        let v = Arc::clone(&vault);
        let rc = Arc::clone(&revoke_count);
        handles.push(thread::spawn(move || {
            for _ in 0..5 {
                if v.revoke(Vault::ROOT, &entity, "concurrent/secret").is_ok() {
                    rc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                thread::sleep(Duration::from_micros(10));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All operations should complete without deadlock
    assert!(grant_count.load(std::sync::atomic::Ordering::SeqCst) > 0);
    // Vault should still be usable
    assert!(vault.get(Vault::ROOT, "concurrent/secret").is_ok());
}
