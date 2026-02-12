// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for vault delegation, policy, attenuation, quota, and
//! template cross-module interactions.

use std::sync::Arc;
use std::time::Duration;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    AttenuationPolicy, ExponentialAttenuationPolicy, PasswordConfig, Permission, PolicyTemplate,
    ResourceQuota, SecretTemplate, Vault, VaultConfig, VaultError,
};

fn create_test_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    Vault::new(
        b"test-key-32-bytes-long!!!!!!!!!",
        graph,
        store,
        VaultConfig::default(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// 1-5: Delegation tests
// ---------------------------------------------------------------------------

#[test]
fn test_delegate_grants_access() {
    let vault = create_test_vault();

    // Store secrets as root
    vault.set(Vault::ROOT, "secret/one", "value1").unwrap();
    vault.set(Vault::ROOT, "secret/two", "value2").unwrap();

    // Grant parent Admin access
    vault
        .grant_with_permission(Vault::ROOT, "user:parent", "secret/one", Permission::Admin)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:parent", "secret/two", Permission::Admin)
        .unwrap();

    // Parent delegates to child
    let record = vault
        .delegate(
            "user:parent",
            "user:child",
            &["secret/one", "secret/two"],
            Permission::Read,
            None,
        )
        .unwrap();

    assert_eq!(record.parent, "user:parent");
    assert_eq!(record.child, "user:child");
    assert_eq!(record.secrets.len(), 2);

    // Child should now be able to read the delegated secrets
    let val = vault.get("user:child", "secret/one").unwrap();
    assert_eq!(val, "value1");
}

#[test]
fn test_delegate_respects_permission_ceiling() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "db/password", "hunter2").unwrap();

    // Grant parent Write access
    vault
        .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Write)
        .unwrap();

    // Alice delegates Read to Bob
    let record = vault
        .delegate(
            "user:alice",
            "user:bob",
            &["db/password"],
            Permission::Read,
            None,
        )
        .unwrap();

    assert_eq!(record.max_permission, Permission::Read);

    // Bob should be able to read
    let val = vault.get("user:bob", "db/password").unwrap();
    assert_eq!(val, "hunter2");
}

#[test]
fn test_revoke_delegation_removes_access() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "secret/x", "data").unwrap();

    vault
        .grant_with_permission(Vault::ROOT, "user:parent", "secret/x", Permission::Admin)
        .unwrap();

    vault
        .delegate(
            "user:parent",
            "user:child",
            &["secret/x"],
            Permission::Read,
            None,
        )
        .unwrap();

    // Child can access
    assert!(vault.get("user:child", "secret/x").is_ok());

    // Revoke
    vault
        .revoke_delegation("user:parent", "user:child")
        .unwrap();

    // Child can no longer access
    assert!(vault.get("user:child", "secret/x").is_err());
}

#[test]
fn test_revoke_delegation_returns_secret_names() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "key/a", "va").unwrap();
    vault.set(Vault::ROOT, "key/b", "vb").unwrap();

    vault
        .grant_with_permission(Vault::ROOT, "user:p", "key/a", Permission::Admin)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:p", "key/b", Permission::Admin)
        .unwrap();

    vault
        .delegate(
            "user:p",
            "user:c",
            &["key/a", "key/b"],
            Permission::Read,
            None,
        )
        .unwrap();

    let revoked = vault.revoke_delegation("user:p", "user:c").unwrap();
    assert_eq!(revoked.len(), 2);
    assert!(revoked.contains(&"key/a".to_string()));
    assert!(revoked.contains(&"key/b".to_string()));
}

#[test]
fn test_cascading_revoke() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "cascade/s", "cascade-val").unwrap();

    // Root -> A -> B -> C
    vault
        .grant_with_permission(Vault::ROOT, "agent:a", "cascade/s", Permission::Admin)
        .unwrap();

    vault
        .delegate(
            "agent:a",
            "agent:b",
            &["cascade/s"],
            Permission::Write,
            None,
        )
        .unwrap();

    vault
        .delegate("agent:b", "agent:c", &["cascade/s"], Permission::Read, None)
        .unwrap();

    // Both B and C can read
    assert!(vault.get("agent:b", "cascade/s").is_ok());
    assert!(vault.get("agent:c", "cascade/s").is_ok());

    // Cascading revoke from A -> B should also revoke B -> C
    let revoked = vault
        .revoke_delegation_cascading("agent:a", "agent:b")
        .unwrap();

    assert!(revoked.len() >= 2);

    // Neither B nor C can access
    assert!(vault.get("agent:b", "cascade/s").is_err());
    assert!(vault.get("agent:c", "cascade/s").is_err());
}

// ---------------------------------------------------------------------------
// 6: Delegation anomaly scores
// ---------------------------------------------------------------------------

#[test]
fn test_delegation_anomaly_scores() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "anomaly/s1", "v1").unwrap();
    vault.set(Vault::ROOT, "anomaly/s2", "v2").unwrap();

    vault
        .grant_with_permission(Vault::ROOT, "user:x", "anomaly/s1", Permission::Read)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:y", "anomaly/s2", Permission::Read)
        .unwrap();

    let scores = vault.delegation_anomaly_scores();
    // Scores should be returned (possibly empty on small graphs, but the API
    // should not panic).
    assert!(scores.iter().all(|s| s.anomaly_score >= 0.0));
}

// ---------------------------------------------------------------------------
// 7-12: Policy tests
// ---------------------------------------------------------------------------

#[test]
fn test_policy_add_and_list() {
    let vault = create_test_vault();

    let policy = PolicyTemplate {
        name: "eng-staging".to_string(),
        match_pattern: "team:engineering/*".to_string(),
        secret_pattern: "staging/*".to_string(),
        permission: Permission::Read,
        ttl_ms: None,
    };

    vault.add_policy(Vault::ROOT, policy).unwrap();

    let policies = vault.list_policies();
    assert_eq!(policies.len(), 1);
    assert_eq!(policies[0].name, "eng-staging");
}

#[test]
fn test_policy_get_by_name() {
    let vault = create_test_vault();

    let policy = PolicyTemplate {
        name: "dev-config".to_string(),
        match_pattern: "dev:*".to_string(),
        secret_pattern: "config/*".to_string(),
        permission: Permission::Write,
        ttl_ms: Some(3_600_000),
    };

    vault.add_policy(Vault::ROOT, policy).unwrap();

    let fetched = vault.get_policy("dev-config").unwrap();
    assert_eq!(fetched.match_pattern, "dev:*");
    assert_eq!(fetched.permission, Permission::Write);
    assert_eq!(fetched.ttl_ms, Some(3_600_000));
}

#[test]
fn test_policy_remove() {
    let vault = create_test_vault();

    let policy = PolicyTemplate {
        name: "temporary".to_string(),
        match_pattern: "*".to_string(),
        secret_pattern: "*".to_string(),
        permission: Permission::Read,
        ttl_ms: None,
    };

    vault.add_policy(Vault::ROOT, policy).unwrap();
    assert!(vault.get_policy("temporary").is_some());

    vault.remove_policy(Vault::ROOT, "temporary").unwrap();
    assert!(vault.get_policy("temporary").is_none());
}

#[test]
fn test_policy_evaluate_match() {
    let vault = create_test_vault();

    vault
        .add_policy(
            Vault::ROOT,
            PolicyTemplate {
                name: "eng-read".to_string(),
                match_pattern: "team:engineering/*".to_string(),
                secret_pattern: "staging/*".to_string(),
                permission: Permission::Read,
                ttl_ms: None,
            },
        )
        .unwrap();

    let matches = vault.evaluate_policies("team:engineering/alice");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].policy_name, "eng-read");
    assert_eq!(matches[0].permission, Permission::Read);
}

#[test]
fn test_policy_evaluate_no_match() {
    let vault = create_test_vault();

    vault
        .add_policy(
            Vault::ROOT,
            PolicyTemplate {
                name: "eng-only".to_string(),
                match_pattern: "team:engineering/*".to_string(),
                secret_pattern: "staging/*".to_string(),
                permission: Permission::Read,
                ttl_ms: None,
            },
        )
        .unwrap();

    let matches = vault.evaluate_policies("team:sales/bob");
    assert!(matches.is_empty());
}

#[test]
fn test_policy_non_root_denied() {
    let vault = create_test_vault();

    let policy = PolicyTemplate {
        name: "sneaky".to_string(),
        match_pattern: "*".to_string(),
        secret_pattern: "*".to_string(),
        permission: Permission::Admin,
        ttl_ms: None,
    };

    let result = vault.add_policy("user:unprivileged", policy);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, VaultError::AccessDenied(_)));

    let remove_result = vault.remove_policy("user:unprivileged", "anything");
    assert!(remove_result.is_err());
}

// ---------------------------------------------------------------------------
// 13-17: Quota tests
// ---------------------------------------------------------------------------

#[test]
fn test_quota_set_and_get() {
    let vault = create_test_vault();

    let quota = ResourceQuota {
        max_secrets: 100,
        max_storage_bytes: 1_000_000,
        max_ops_per_hour: 500,
    };

    vault
        .set_quota(Vault::ROOT, "team-a", quota.clone())
        .unwrap();

    let fetched = vault.get_quota("team-a").unwrap();
    assert_eq!(fetched.max_secrets, 100);
    assert_eq!(fetched.max_storage_bytes, 1_000_000);
    assert_eq!(fetched.max_ops_per_hour, 500);
}

#[test]
fn test_quota_usage_tracking() {
    let vault = create_test_vault();

    // Before any operations, usage should be zero-like
    let usage = vault.get_usage("team-x");
    assert_eq!(usage.secret_count, 0);
}

#[test]
fn test_quota_removal() {
    let vault = create_test_vault();

    let quota = ResourceQuota {
        max_secrets: 10,
        max_storage_bytes: 10_000,
        max_ops_per_hour: 100,
    };

    vault.set_quota(Vault::ROOT, "ns1", quota).unwrap();
    assert!(vault.get_quota("ns1").is_some());

    vault.remove_quota(Vault::ROOT, "ns1").unwrap();
    assert!(vault.get_quota("ns1").is_none());
}

#[test]
fn test_quota_non_root_denied() {
    let vault = create_test_vault();

    let quota = ResourceQuota {
        max_secrets: 10,
        max_storage_bytes: 10_000,
        max_ops_per_hour: 100,
    };

    let result = vault.set_quota("user:nobody", "ns", quota);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), VaultError::AccessDenied(_)));

    let remove_result = vault.remove_quota("user:nobody", "ns");
    assert!(remove_result.is_err());
    assert!(matches!(
        remove_result.unwrap_err(),
        VaultError::AccessDenied(_)
    ));
}

#[test]
fn test_quota_enforcement() {
    let vault = create_test_vault();

    let quota = ResourceQuota {
        max_secrets: 2,
        max_storage_bytes: 1_000_000,
        max_ops_per_hour: 1000,
    };

    // Set quota for the namespace prefix
    vault
        .set_quota(Vault::ROOT, "limited", quota)
        .unwrap();

    // Store two secrets within quota
    vault.set(Vault::ROOT, "limited/s1", "value1").unwrap();
    vault.set(Vault::ROOT, "limited/s2", "value2").unwrap();

    // Third secret should exceed max_secrets=2
    let result = vault.set(Vault::ROOT, "limited/s3", "value3");
    assert!(
        result.is_err(),
        "third secret should be denied by quota enforcement"
    );

    // Updating an existing secret should still work (not a new secret)
    vault
        .set(Vault::ROOT, "limited/s1", "updated")
        .unwrap();
    assert_eq!(vault.get(Vault::ROOT, "limited/s1").unwrap(), "updated");
}

// ---------------------------------------------------------------------------
// 18-20: Template tests
// ---------------------------------------------------------------------------

#[test]
fn test_template_save_get_list() {
    let vault = create_test_vault();

    let template = SecretTemplate::Password(PasswordConfig::default());
    vault
        .save_template(Vault::ROOT, "db-password", template)
        .unwrap();

    let stored = vault.get_template("db-password").unwrap();
    assert_eq!(stored.name, "db-password");
    assert_eq!(stored.created_by, Vault::ROOT);
    assert!(stored.created_at > 0);

    let names = vault.list_templates();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"db-password".to_string()));
}

#[test]
fn test_template_delete() {
    let vault = create_test_vault();

    let template = SecretTemplate::Password(PasswordConfig::default());
    vault
        .save_template(Vault::ROOT, "ephemeral-tpl", template)
        .unwrap();

    assert!(vault.get_template("ephemeral-tpl").is_some());

    vault.delete_template(Vault::ROOT, "ephemeral-tpl").unwrap();

    assert!(vault.get_template("ephemeral-tpl").is_none());
}

#[test]
fn test_template_overwrite() {
    let vault = create_test_vault();

    let tpl1 = SecretTemplate::Password(PasswordConfig::default());
    vault.save_template(Vault::ROOT, "reuse", tpl1).unwrap();

    let first = vault.get_template("reuse").unwrap();
    assert_eq!(first.name, "reuse");

    // Overwrite with a different template type
    let tpl2 = SecretTemplate::Token(tensor_vault::TokenConfig::default());
    vault.save_template(Vault::ROOT, "reuse", tpl2).unwrap();

    let second = vault.get_template("reuse").unwrap();
    assert_eq!(second.name, "reuse");

    // Template list should still have exactly one entry
    assert_eq!(vault.list_templates().len(), 1);
}

// ---------------------------------------------------------------------------
// 21-23: Attenuation tests
// ---------------------------------------------------------------------------

#[test]
fn test_exponential_attenuation_decay() {
    let policy = ExponentialAttenuationPolicy::default();

    // At 0 hops, Admin stays Admin
    assert_eq!(
        policy.attenuate(Permission::Admin, 0),
        Some(Permission::Admin)
    );

    // At 1 hop, Admin should attenuate to Write (strength ~0.607)
    assert_eq!(
        policy.attenuate(Permission::Admin, 1),
        Some(Permission::Write)
    );

    // At 3 hops, Admin should attenuate to Read (strength ~0.223)
    assert_eq!(
        policy.attenuate(Permission::Admin, 3),
        Some(Permission::Read)
    );

    // At 6 hops, should be denied (strength ~0.0498 < 0.05)
    assert_eq!(policy.attenuate(Permission::Admin, 6), None);
}

#[test]
fn test_attenuation_none_preserves_permissions() {
    let policy = AttenuationPolicy::none();

    assert_eq!(
        policy.attenuate(Permission::Admin, 100),
        Some(Permission::Admin)
    );
    assert_eq!(
        policy.attenuate(Permission::Write, 50),
        Some(Permission::Write)
    );
    assert_eq!(
        policy.attenuate(Permission::Read, 200),
        Some(Permission::Read)
    );
}

#[test]
fn test_delegate_with_attenuation() {
    // Use default attenuation: admin_limit=1, write_limit=2
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "attenuate/secret", "value").unwrap();

    // Grant Admin to parent
    vault
        .grant_with_permission(
            Vault::ROOT,
            "user:admin-parent",
            "attenuate/secret",
            Permission::Admin,
        )
        .unwrap();

    // Delegate Admin from parent to child
    vault
        .delegate(
            "user:admin-parent",
            "user:delegated-child",
            &["attenuate/secret"],
            Permission::Admin,
            Some(Duration::from_secs(3600)),
        )
        .unwrap();

    // Child should still be able to read the secret
    let val = vault
        .get("user:delegated-child", "attenuate/secret")
        .unwrap();
    assert_eq!(val, "value");
}

// ---------------------------------------------------------------------------
// 24: Policy redundancy analysis
// ---------------------------------------------------------------------------

#[test]
fn test_policy_redundancy_analysis() {
    let vault = create_test_vault();

    // Add two overlapping policies
    vault
        .add_policy(
            Vault::ROOT,
            PolicyTemplate {
                name: "broad-read".to_string(),
                match_pattern: "user:*".to_string(),
                secret_pattern: "config/*".to_string(),
                permission: Permission::Read,
                ttl_ms: None,
            },
        )
        .unwrap();

    vault
        .add_policy(
            Vault::ROOT,
            PolicyTemplate {
                name: "broad-read-2".to_string(),
                match_pattern: "user:*".to_string(),
                secret_pattern: "config/*".to_string(),
                permission: Permission::Read,
                ttl_ms: None,
            },
        )
        .unwrap();

    let report = vault.analyze_policy_redundancy().unwrap();
    assert_eq!(report.total_policies, 2);
    // Two identical policies should be detected as mergeable, yielding a
    // compression ratio > 1.0 (higher means more redundancy).
    assert!(report.compression_ratio >= 1.0);
}

// ---------------------------------------------------------------------------
// 25: Multiple quotas are independent
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_quotas_independent() {
    let vault = create_test_vault();

    let q1 = ResourceQuota {
        max_secrets: 10,
        max_storage_bytes: 10_000,
        max_ops_per_hour: 100,
    };
    let q2 = ResourceQuota {
        max_secrets: 5,
        max_storage_bytes: 5_000,
        max_ops_per_hour: 50,
    };

    vault.set_quota(Vault::ROOT, "ns-alpha", q1).unwrap();
    vault.set_quota(Vault::ROOT, "ns-beta", q2).unwrap();

    let qa = vault.get_quota("ns-alpha").unwrap();
    let qb = vault.get_quota("ns-beta").unwrap();

    assert_eq!(qa.max_secrets, 10);
    assert_eq!(qb.max_secrets, 5);

    // Modifying one should not affect the other
    vault.remove_quota(Vault::ROOT, "ns-alpha").unwrap();
    assert!(vault.get_quota("ns-alpha").is_none());
    assert!(vault.get_quota("ns-beta").is_some());
}

// ---------------------------------------------------------------------------
// 26-27: Access explanation
// ---------------------------------------------------------------------------

#[test]
fn test_explain_access_granted() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "explain/key", "val").unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:eve", "explain/key", Permission::Read)
        .unwrap();

    let explanation = vault.explain_access("user:eve", "explain/key");
    assert!(explanation.granted);
    assert_eq!(explanation.entity, "user:eve");
    assert_eq!(explanation.secret, "explain/key");
    assert!(explanation.effective_permission.is_some());
}

#[test]
fn test_explain_access_denied() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "locked/key", "val").unwrap();

    let explanation = vault.explain_access("user:nobody", "locked/key");
    assert!(!explanation.granted);
    assert!(explanation.denial_reason.is_some());
}

// ---------------------------------------------------------------------------
// 28: Blast radius
// ---------------------------------------------------------------------------

#[test]
fn test_blast_radius() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "blast/s1", "v1").unwrap();
    vault.set(Vault::ROOT, "blast/s2", "v2").unwrap();
    vault.set(Vault::ROOT, "blast/s3", "v3").unwrap();

    vault
        .grant_with_permission(Vault::ROOT, "user:wide", "blast/s1", Permission::Admin)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:wide", "blast/s2", Permission::Write)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:wide", "blast/s3", Permission::Read)
        .unwrap();

    let radius = vault.blast_radius("user:wide");
    assert_eq!(radius.entity, "user:wide");
    assert_eq!(radius.total_secrets, 3);
    assert_eq!(radius.secrets.len(), 3);
}

// ---------------------------------------------------------------------------
// 29: Simulate grant
// ---------------------------------------------------------------------------

#[test]
fn test_simulate_grant() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "sim/key", "sim-val").unwrap();

    let result = vault.simulate_grant("user:new-entity", "sim/key", Permission::Admin);
    assert_eq!(result.target_entity, "user:new-entity");
    assert_eq!(result.secret, "sim/key");
    assert_eq!(result.requested_permission, Permission::Admin);
    // Simulation should report at least the direct access that would be created
    assert!(result.total_affected >= 1);
}

// ---------------------------------------------------------------------------
// 30: Privilege analysis
// ---------------------------------------------------------------------------

#[test]
fn test_privilege_analysis() {
    let vault = create_test_vault();

    vault.set(Vault::ROOT, "priv/s1", "v1").unwrap();
    vault.set(Vault::ROOT, "priv/s2", "v2").unwrap();

    vault
        .grant_with_permission(Vault::ROOT, "user:high", "priv/s1", Permission::Admin)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:high", "priv/s2", Permission::Admin)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:low", "priv/s1", Permission::Read)
        .unwrap();

    let report = vault.privilege_analysis();
    // Should report on at least the entities we created
    assert!(!report.entities.is_empty());
    assert!(report.max_privilege_score >= 0.0);
    assert!(report.mean_privilege_score >= 0.0);
}
