// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for vault temporal analysis, access tensors, anomaly
//! detection, dependency tracking, and audit log cross-module interactions.

use std::sync::Arc;
use std::thread;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    AccessTensorConfig, AnomalyThresholds, DependencyWeight, TemporalAnalysisConfig, Vault,
    VaultConfig, VaultError,
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

fn create_test_vault_with_thresholds(thresholds: AnomalyThresholds) -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let config = VaultConfig::default().with_anomaly_thresholds(thresholds);
    Vault::new(b"test-key-32-bytes-long!!!!!!!!!", graph, store, config).unwrap()
}

/// Helper to get a recent start_time_ms that will capture operations done "now".
fn recent_start_ms() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    // Start 1 hour before now so current operations land in bucket 0
    now - 3_600_000
}

// =========================================================================
// Access Tensor Tests
// =========================================================================

#[test]
fn test_access_tensor_empty_vault_all_zeros() {
    let vault = create_test_vault();
    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 24,
        start_time_ms: None,
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let (entities, secrets, buckets) = tensor.dimensions();
    // Empty vault has no audit entries, so dimensions should be zero
    assert_eq!(entities, 0);
    assert_eq!(secrets, 0);
    assert_eq!(buckets, 24);
    // All data should be zeros
    assert!(tensor.raw_data().iter().all(|&v| v == 0.0));
}

#[test]
fn test_access_tensor_after_operations_non_zero() {
    let vault = create_test_vault();
    vault
        .set(Vault::ROOT, "tensor/secret-a", "value-a")
        .unwrap();
    vault
        .set(Vault::ROOT, "tensor/secret-b", "value-b")
        .unwrap();
    vault.get(Vault::ROOT, "tensor/secret-a").unwrap();
    vault.get(Vault::ROOT, "tensor/secret-a").unwrap();
    vault.get(Vault::ROOT, "tensor/secret-b").unwrap();

    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 24,
        start_time_ms: Some(recent_start_ms()),
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let (entities, secrets, _buckets) = tensor.dimensions();
    // Should have at least one entity and one secret from the operations
    assert!(entities > 0, "expected at least one entity");
    assert!(secrets > 0, "expected at least one secret");
    // There should be non-zero data somewhere
    assert!(
        tensor.raw_data().iter().any(|&v| v > 0.0),
        "expected non-zero entries after vault operations"
    );
}

#[test]
fn test_access_tensor_dimensions_match_entities_and_secrets() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "dim/s1", "v1").unwrap();
    vault.set(Vault::ROOT, "dim/s2", "v2").unwrap();
    vault.set(Vault::ROOT, "dim/s3", "v3").unwrap();

    let num_buckets = 48;
    let config = AccessTensorConfig {
        bucket_size_ms: 1_800_000,
        num_buckets,
        start_time_ms: Some(recent_start_ms()),
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let (n_entities, n_secrets, n_buckets) = tensor.dimensions();
    assert_eq!(n_buckets, num_buckets);
    // Data length should equal product of dimensions
    assert_eq!(tensor.raw_data().len(), n_entities * n_secrets * n_buckets);
}

#[test]
fn test_access_tensor_entity_profiles() {
    let vault = create_test_vault();
    let start = recent_start_ms();
    vault.set(Vault::ROOT, "prof/key1", "val1").unwrap();
    vault.get(Vault::ROOT, "prof/key1").unwrap();

    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 24,
        start_time_ms: Some(start),
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let profiles = tensor.entity_profiles();
    // ROOT performed operations, so there should be at least one profile
    assert!(!profiles.is_empty(), "expected at least one entity profile");
    // At least one profile should have recorded accesses
    assert!(
        profiles.iter().any(|p| p.total_accesses > 0),
        "at least one profile should have accesses"
    );
}

#[test]
fn test_access_tensor_secret_profiles() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "sprof/alpha", "a").unwrap();
    vault.get(Vault::ROOT, "sprof/alpha").unwrap();

    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 24,
        start_time_ms: None,
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let secret_profiles = tensor.secret_profiles();
    assert!(
        !secret_profiles.is_empty(),
        "expected at least one secret profile"
    );
}

#[test]
fn test_access_tensor_operation_filter() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "filter/s", "val").unwrap();
    vault.get(Vault::ROOT, "filter/s").unwrap();

    // Filter to only Get operations; use a start time that captures current ops
    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 168,
        start_time_ms: Some(recent_start_ms()),
        operations: Some(vec!["Get".to_string()]),
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    // With only Get filtered, Set operations should be excluded
    let total: f32 = tensor.raw_data().iter().sum();
    // At least the one Get should appear
    assert!(total > 0.0, "expected Get operations in tensor");
}

// =========================================================================
// Temporal Analysis Tests
// =========================================================================

#[test]
fn test_temporal_analysis_empty_vault_no_drift() {
    let vault = create_test_vault();
    let tensor_config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    let analysis_config = TemporalAnalysisConfig {
        tt_config: None,
        drift_window: 24,
        drift_threshold: 0.3,
        min_accesses: 5,
    };
    let report = vault
        .analyze_temporal_patterns(tensor_config, analysis_config)
        .unwrap();
    assert_eq!(report.total_entities_analyzed, 0);
    assert!(report.drift_detections.is_empty());
    assert!(report.seasonal_patterns.is_empty());
}

#[test]
fn test_temporal_analysis_with_operations() {
    let vault = create_test_vault();
    // Generate enough operations to meet the min_accesses threshold
    for i in 0..10 {
        vault
            .set(Vault::ROOT, &format!("temporal/{i}"), &format!("v{i}"))
            .unwrap();
    }

    let tensor_config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    let analysis_config = TemporalAnalysisConfig {
        tt_config: None,
        drift_window: 12,
        drift_threshold: 0.3,
        min_accesses: 1,
    };
    let report = vault
        .analyze_temporal_patterns(tensor_config, analysis_config)
        .unwrap();
    // With min_accesses = 1, ROOT entity should be analyzed
    assert!(
        report.total_entities_analyzed > 0,
        "expected at least one entity analyzed"
    );
}

#[test]
fn test_temporal_analysis_compression_ratio_bounded() {
    let vault = create_test_vault();
    for i in 0..20 {
        vault
            .set(Vault::ROOT, &format!("comp/{i}"), &format!("v{i}"))
            .unwrap();
    }

    let tensor_config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    let analysis_config = TemporalAnalysisConfig {
        tt_config: None,
        drift_window: 12,
        drift_threshold: 0.3,
        min_accesses: 1,
    };
    let report = vault
        .analyze_temporal_patterns(tensor_config, analysis_config)
        .unwrap();
    // Compression ratio should be non-negative
    assert!(
        report.mean_compression_ratio >= 0.0,
        "compression ratio should be non-negative"
    );
}

#[test]
fn test_temporal_analysis_drift_detection_high_threshold() {
    let vault = create_test_vault();
    for i in 0..5 {
        vault
            .set(Vault::ROOT, &format!("drift/{i}"), &format!("v{i}"))
            .unwrap();
    }

    let tensor_config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    // Very high threshold: nothing should be flagged as drifting
    let analysis_config = TemporalAnalysisConfig {
        tt_config: None,
        drift_window: 12,
        drift_threshold: 10.0,
        min_accesses: 1,
    };
    let report = vault
        .analyze_temporal_patterns(tensor_config, analysis_config)
        .unwrap();
    for d in &report.drift_detections {
        assert!(
            !d.is_drifting,
            "no drift should be flagged with threshold 10.0"
        );
    }
}

// =========================================================================
// Anomaly Detection Tests
// =========================================================================

#[test]
fn test_anomaly_monitor_accessible() {
    let vault = create_test_vault();
    let monitor = vault.anomaly_monitor();
    // Monitor should exist and be accessible
    let _profiles = monitor;
}

#[test]
fn test_anomaly_after_rapid_operations() {
    let thresholds = AnomalyThresholds {
        frequency_spike_limit: 3,
        frequency_window_ms: 60_000,
        bulk_operation_threshold: 2,
        inactive_threshold_ms: 1_000,
    };
    let vault = create_test_vault_with_thresholds(thresholds);

    // Perform rapid operations
    for i in 0..10 {
        vault
            .set(Vault::ROOT, &format!("anom/{i}"), &format!("v{i}"))
            .unwrap();
    }

    // The anomaly monitor should have tracked the ROOT agent
    let monitor = vault.anomaly_monitor();
    // Check that profiles exist (at least ROOT should have one)
    let has_profile = monitor.get_profile(Vault::ROOT).is_some();
    assert!(
        has_profile,
        "ROOT should have an anomaly profile after operations"
    );
}

#[test]
fn test_anomaly_persist_and_reload() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let salt = [1_u8; 16];
    let thresholds = AnomalyThresholds::default();
    // Must use with_anomaly_thresholds so the vault calls AnomalyMonitor::load
    let config = VaultConfig::default()
        .with_salt(salt)
        .with_anomaly_thresholds(thresholds.clone());
    let vault = Vault::new(
        b"test-key-32-bytes-long!!!!!!!!!",
        Arc::clone(&graph),
        store.clone(),
        config,
    )
    .unwrap();

    // Generate some operations to create anomaly profiles
    vault.set(Vault::ROOT, "persist/s1", "val").unwrap();
    vault.get(Vault::ROOT, "persist/s1").unwrap();

    // Persist profiles
    vault.persist_anomaly_profiles();

    // Create new vault from same store with same salt and thresholds to reload profiles
    let config2 = VaultConfig::default()
        .with_salt(salt)
        .with_anomaly_thresholds(thresholds);
    let vault2 = Vault::new(b"test-key-32-bytes-long!!!!!!!!!", graph, store, config2).unwrap();

    // The new vault should have loaded profiles from store
    let monitor = vault2.anomaly_monitor();
    let profile = monitor.get_profile(Vault::ROOT);
    assert!(
        profile.is_some(),
        "anomaly profile should persist across vault instances"
    );
}

#[test]
fn test_anomaly_custom_thresholds() {
    let thresholds = AnomalyThresholds {
        frequency_spike_limit: 100,
        frequency_window_ms: 30_000,
        bulk_operation_threshold: 50,
        inactive_threshold_ms: 3_600_000,
    };
    let vault = create_test_vault_with_thresholds(thresholds);
    // Vault should be usable with custom thresholds
    vault.set(Vault::ROOT, "custom/th", "val").unwrap();
    let val = vault.get(Vault::ROOT, "custom/th").unwrap();
    assert_eq!(val, "val");
}

// =========================================================================
// Dependency Tests
// =========================================================================

#[test]
fn test_dependency_add_and_get() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "dep/a", "va").unwrap();
    vault.set(Vault::ROOT, "dep/b", "vb").unwrap();

    // A -> B (B depends on A)
    vault.add_dependency(Vault::ROOT, "dep/a", "dep/b").unwrap();

    // get_dependencies returns children (secrets that depend on A)
    let children = vault.get_dependencies(Vault::ROOT, "dep/a").unwrap();
    assert!(
        !children.is_empty(),
        "dep/a should have at least one child dependency"
    );
}

#[test]
fn test_dependency_get_dependents() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "depd/a", "va").unwrap();
    vault.set(Vault::ROOT, "depd/b", "vb").unwrap();

    vault
        .add_dependency(Vault::ROOT, "depd/a", "depd/b")
        .unwrap();

    // get_dependents returns parents (secrets B depends on)
    let parents = vault.get_dependents(Vault::ROOT, "depd/b").unwrap();
    assert!(
        !parents.is_empty(),
        "depd/b should have at least one parent dependency"
    );
}

#[test]
fn test_dependency_remove() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "rem/a", "va").unwrap();
    vault.set(Vault::ROOT, "rem/b", "vb").unwrap();

    vault.add_dependency(Vault::ROOT, "rem/a", "rem/b").unwrap();

    let children_before = vault.get_dependencies(Vault::ROOT, "rem/a").unwrap();
    assert!(!children_before.is_empty());

    vault
        .remove_dependency(Vault::ROOT, "rem/a", "rem/b")
        .unwrap();

    let children_after = vault.get_dependencies(Vault::ROOT, "rem/a").unwrap();
    assert!(children_after.is_empty(), "dependency should be removed");
}

#[test]
fn test_dependency_chain_impact_analysis() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "chain/a", "va").unwrap();
    vault.set(Vault::ROOT, "chain/b", "vb").unwrap();
    vault.set(Vault::ROOT, "chain/c", "vc").unwrap();

    // A -> B -> C (C depends on B, B depends on A)
    vault
        .add_dependency(Vault::ROOT, "chain/a", "chain/b")
        .unwrap();
    vault
        .add_dependency(Vault::ROOT, "chain/b", "chain/c")
        .unwrap();

    let report = vault.impact_analysis(Vault::ROOT, "chain/a").unwrap();
    // Both B and C are affected when A changes
    assert!(
        report.affected_secrets.len() >= 2,
        "impact analysis should find both B and C affected, found {}",
        report.affected_secrets.len()
    );
    assert!(report.depth >= 2, "depth should be at least 2 for A->B->C");
}

#[test]
fn test_dependency_cyclic_detection() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "cyc/a", "va").unwrap();
    vault.set(Vault::ROOT, "cyc/b", "vb").unwrap();

    // A -> B
    vault.add_dependency(Vault::ROOT, "cyc/a", "cyc/b").unwrap();

    // B -> A would create a cycle
    let result = vault.add_dependency(Vault::ROOT, "cyc/b", "cyc/a");
    assert!(result.is_err(), "cyclic dependency should be rejected");
    match result.unwrap_err() {
        VaultError::CyclicDependency(_) => {},
        other => panic!("expected CyclicDependency, got: {other:?}"),
    }
}

#[test]
fn test_dependency_self_cycle_detection() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "self/a", "va").unwrap();

    // A -> A should be detected as a cycle
    let result = vault.add_dependency(Vault::ROOT, "self/a", "self/a");
    assert!(result.is_err(), "self-dependency should be rejected");
    match result.unwrap_err() {
        VaultError::CyclicDependency(_) => {},
        other => panic!("expected CyclicDependency, got: {other:?}"),
    }
}

#[test]
fn test_dependency_transitive_cycle_detection() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "tcyc/a", "va").unwrap();
    vault.set(Vault::ROOT, "tcyc/b", "vb").unwrap();
    vault.set(Vault::ROOT, "tcyc/c", "vc").unwrap();

    // A -> B -> C
    vault
        .add_dependency(Vault::ROOT, "tcyc/a", "tcyc/b")
        .unwrap();
    vault
        .add_dependency(Vault::ROOT, "tcyc/b", "tcyc/c")
        .unwrap();

    // C -> A would create a transitive cycle
    let result = vault.add_dependency(Vault::ROOT, "tcyc/c", "tcyc/a");
    assert!(result.is_err(), "transitive cycle should be rejected");
    match result.unwrap_err() {
        VaultError::CyclicDependency(_) => {},
        other => panic!("expected CyclicDependency, got: {other:?}"),
    }
}

#[test]
fn test_dependency_weighted_add() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "wdep/a", "va").unwrap();
    vault.set(Vault::ROOT, "wdep/b", "vb").unwrap();

    vault
        .add_weighted_dependency(
            Vault::ROOT,
            "wdep/a",
            "wdep/b",
            DependencyWeight::Critical,
            Some("critical link"),
        )
        .unwrap();

    let children = vault.get_dependencies(Vault::ROOT, "wdep/a").unwrap();
    assert!(
        !children.is_empty(),
        "weighted dependency should create a child link"
    );
}

#[test]
fn test_dependency_weighted_impact_analysis() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "wimp/a", "va").unwrap();
    vault.set(Vault::ROOT, "wimp/b", "vb").unwrap();
    vault.set(Vault::ROOT, "wimp/c", "vc").unwrap();

    vault
        .add_weighted_dependency(
            Vault::ROOT,
            "wimp/a",
            "wimp/b",
            DependencyWeight::Critical,
            None,
        )
        .unwrap();
    vault
        .add_weighted_dependency(Vault::ROOT, "wimp/b", "wimp/c", DependencyWeight::Low, None)
        .unwrap();

    let report = vault
        .weighted_impact_analysis(Vault::ROOT, "wimp/a")
        .unwrap();
    assert!(
        !report.affected_secrets.is_empty(),
        "weighted impact should find affected secrets"
    );
    assert!(
        report.total_impact_score > 0.0,
        "total impact score should be positive"
    );
    // Critical weight should produce higher score than Low
    if report.affected_secrets.len() >= 2 {
        let critical_item = report
            .affected_secrets
            .iter()
            .find(|s| s.edge_weight == DependencyWeight::Critical);
        let low_item = report
            .affected_secrets
            .iter()
            .find(|s| s.edge_weight == DependencyWeight::Low);
        if let (Some(c), Some(l)) = (critical_item, low_item) {
            assert!(
                c.impact_score > l.impact_score,
                "critical weight ({}) should score higher than low ({})",
                c.impact_score,
                l.impact_score
            );
        }
    }
}

#[test]
fn test_dependency_rotation_plan() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "rot/a", "va").unwrap();
    vault.set(Vault::ROOT, "rot/b", "vb").unwrap();
    vault.set(Vault::ROOT, "rot/c", "vc").unwrap();

    vault.add_dependency(Vault::ROOT, "rot/a", "rot/b").unwrap();
    vault.add_dependency(Vault::ROOT, "rot/b", "rot/c").unwrap();

    let plan = vault.rotation_plan(Vault::ROOT, "rot/a").unwrap();
    // The plan should include at least the dependent secrets
    assert!(
        !plan.rotation_order.is_empty(),
        "rotation plan should include steps"
    );
    assert!(
        plan.total_secrets > 0,
        "should have at least one secret in plan"
    );
}

// =========================================================================
// Audit Tests
// =========================================================================

#[test]
fn test_audit_log_operations_generate_entries() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "audit/s1", "val").unwrap();
    vault.get(Vault::ROOT, "audit/s1").unwrap();

    let entries = vault.audit_log("audit/s1").unwrap();
    assert!(
        entries.len() >= 2,
        "set + get should produce at least 2 audit entries, got {}",
        entries.len()
    );
}

#[test]
fn test_audit_by_entity_filters() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "aent/s1", "val").unwrap();
    vault.get(Vault::ROOT, "aent/s1").unwrap();

    let entries = vault.audit_by_entity(Vault::ROOT).unwrap();
    assert!(!entries.is_empty(), "ROOT should have audit entries");
    for entry in &entries {
        assert_eq!(
            entry.entity,
            Vault::ROOT,
            "all entries should be from ROOT entity"
        );
    }
}

#[test]
fn test_audit_since_timestamp() {
    let vault = create_test_vault();
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    vault.set(Vault::ROOT, "asince/s1", "val").unwrap();

    let entries = vault.audit_since(before).unwrap();
    assert!(
        !entries.is_empty(),
        "should find entries since the timestamp"
    );
    for entry in &entries {
        assert!(
            entry.timestamp >= before,
            "entry timestamp {} should be >= since {}",
            entry.timestamp,
            before
        );
    }
}

#[test]
fn test_audit_since_future_returns_empty() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "afut/s1", "val").unwrap();

    let far_future = i64::MAX / 2;
    let entries = vault.audit_since(far_future).unwrap();
    assert!(
        entries.is_empty(),
        "future timestamp should return no entries"
    );
}

#[test]
fn test_audit_recent_limits() {
    let vault = create_test_vault();
    for i in 0..10 {
        vault
            .set(Vault::ROOT, &format!("arec/{i}"), &format!("v{i}"))
            .unwrap();
    }

    let entries = vault.audit_recent(5).unwrap();
    assert!(
        entries.len() <= 5,
        "audit_recent(5) should return at most 5 entries, got {}",
        entries.len()
    );
}

#[test]
fn test_audit_recent_zero_returns_empty() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "azero/s1", "val").unwrap();

    let entries = vault.audit_recent(0).unwrap();
    assert!(entries.is_empty(), "audit_recent(0) should return empty");
}

// =========================================================================
// Cross-Module Interaction Tests
// =========================================================================

#[test]
fn test_dependency_operations_produce_audit_entries() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "daud/a", "va").unwrap();
    vault.set(Vault::ROOT, "daud/b", "vb").unwrap();

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    vault
        .add_dependency(Vault::ROOT, "daud/a", "daud/b")
        .unwrap();

    let entries = vault.audit_since(before).unwrap();
    // add_dependency should generate an audit entry
    assert!(
        !entries.is_empty(),
        "add_dependency should produce audit entries"
    );
}

#[test]
fn test_impact_analysis_produces_audit_entry() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "iaud/a", "va").unwrap();
    vault.set(Vault::ROOT, "iaud/b", "vb").unwrap();
    vault
        .add_dependency(Vault::ROOT, "iaud/a", "iaud/b")
        .unwrap();

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    vault.impact_analysis(Vault::ROOT, "iaud/a").unwrap();

    let entries = vault.audit_since(before).unwrap();
    assert!(
        !entries.is_empty(),
        "impact_analysis should produce an audit entry"
    );
}

#[test]
fn test_concurrent_vault_operations_and_tensor_build() {
    let vault = Arc::new(create_test_vault());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let v = Arc::clone(&vault);
            thread::spawn(move || {
                for i in 0..5 {
                    let key = format!("conc/{t}/{i}");
                    v.set(Vault::ROOT, &key, &format!("val-{t}-{i}")).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 24,
        start_time_ms: None,
        operations: None,
    };
    let tensor = vault.build_access_tensor(config).unwrap();
    let total: f32 = tensor.raw_data().iter().sum();
    assert!(
        total > 0.0,
        "concurrent operations should produce non-zero tensor data"
    );
}

#[test]
fn test_build_access_tensor_then_analyze() {
    let vault = create_test_vault();
    for i in 0..8 {
        vault
            .set(Vault::ROOT, &format!("ba/{i}"), &format!("v{i}"))
            .unwrap();
    }

    // First build the tensor
    let tensor_config = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    let tensor = vault.build_access_tensor(tensor_config).unwrap();
    assert!(tensor.raw_data().iter().any(|&v| v > 0.0));

    // Then analyze via the combined method
    let tensor_config2 = AccessTensorConfig {
        bucket_size_ms: 3_600_000,
        num_buckets: 48,
        start_time_ms: None,
        operations: None,
    };
    let analysis_config = TemporalAnalysisConfig {
        tt_config: None,
        drift_window: 12,
        drift_threshold: 0.3,
        min_accesses: 1,
    };
    let report = vault
        .analyze_temporal_patterns(tensor_config2, analysis_config)
        .unwrap();
    // Both paths should be consistent
    assert!(
        report.total_entities_analyzed > 0 || tensor.raw_data().iter().all(|&v| v == 0.0),
        "analysis should find entities if tensor has data"
    );
}

#[test]
fn test_anomaly_profile_tracks_operations_count() {
    let vault = create_test_vault();
    for i in 0..5 {
        vault
            .set(Vault::ROOT, &format!("track/{i}"), &format!("v{i}"))
            .unwrap();
    }

    let monitor = vault.anomaly_monitor();
    let profile = monitor.get_profile(Vault::ROOT);
    assert!(profile.is_some(), "ROOT should have a profile");
    let p = profile.unwrap();
    assert!(
        p.total_ops >= 5,
        "profile should track at least 5 operations, got {}",
        p.total_ops
    );
}

#[test]
fn test_weighted_impact_analysis_produces_audit_entry() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "waud/a", "va").unwrap();
    vault.set(Vault::ROOT, "waud/b", "vb").unwrap();
    vault
        .add_weighted_dependency(
            Vault::ROOT,
            "waud/a",
            "waud/b",
            DependencyWeight::High,
            None,
        )
        .unwrap();

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    vault
        .weighted_impact_analysis(Vault::ROOT, "waud/a")
        .unwrap();

    let entries = vault.audit_since(before).unwrap();
    assert!(
        !entries.is_empty(),
        "weighted_impact_analysis should produce an audit entry"
    );
}

#[test]
fn test_rotation_plan_produces_audit_entry() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "raud/a", "va").unwrap();
    vault.set(Vault::ROOT, "raud/b", "vb").unwrap();
    vault
        .add_dependency(Vault::ROOT, "raud/a", "raud/b")
        .unwrap();

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    vault.rotation_plan(Vault::ROOT, "raud/a").unwrap();

    let entries = vault.audit_since(before).unwrap();
    assert!(
        !entries.is_empty(),
        "rotation_plan should produce an audit entry"
    );
}
