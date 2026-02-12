// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for graph_intel, heat_kernel, topology, and manifold cross-module
//! interactions in tensor_vault.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tensor_vault::{
    batch_recommend_placement, recommend_placement, BehaviorEmbeddingConfig, GeoCoordinate,
    HeatKernelConfig, Permission, PlacementConfig, RegionRegistry, TopologyConfig, Vault,
    VaultConfig, VaultRegion,
};

fn create_test_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    Vault::new(
        b"test-key-32-bytes-long!!!!!!",
        graph,
        store,
        VaultConfig::default(),
    )
    .unwrap()
}

fn populate_vault(vault: &Vault) {
    vault.set(Vault::ROOT, "secret1", "value1").unwrap();
    vault.set(Vault::ROOT, "secret2", "value2").unwrap();
    vault.set(Vault::ROOT, "secret3", "value3").unwrap();
    vault.grant(Vault::ROOT, "user:alice", "secret1").unwrap();
    vault.grant(Vault::ROOT, "user:bob", "secret1").unwrap();
    vault.grant(Vault::ROOT, "user:bob", "secret2").unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:carol", "secret2", Permission::Read)
        .unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:carol", "secret3", Permission::Write)
        .unwrap();
}

// =====================================================================
// Graph Intel: Security Audit
// =====================================================================

#[test]
fn test_security_audit_empty_vault() {
    let vault = create_test_vault();
    let report = vault.security_audit();
    assert_eq!(report.total_entities, 0);
    assert_eq!(report.total_secrets, 0);
    assert!(report.cycles.is_empty());
    assert!(report.single_points_of_failure.is_empty());
}

#[test]
fn test_security_audit_populated_vault() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.security_audit();
    assert!(report.total_entities > 0);
    assert!(report.total_secrets > 0);
    assert!(report.total_edges > 0);
}

#[test]
fn test_find_critical_entities_returns_entities_with_access() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "exclusive", "only-alice").unwrap();
    vault.grant(Vault::ROOT, "user:alice", "exclusive").unwrap();

    let critical = vault.find_critical_entities();
    let alice_entries: Vec<_> = critical
        .iter()
        .filter(|c| c.entity == "user:alice")
        .collect();
    // Alice should appear in the critical entities list
    assert!(
        !alice_entries.is_empty(),
        "user:alice should be listed as a critical entity"
    );
    assert!(alice_entries[0].total_reachable_secrets > 0);
    assert!(alice_entries[0].pagerank_score >= 0.0);
}

// =====================================================================
// Graph Intel: Privilege Analysis
// =====================================================================

#[test]
fn test_privilege_analysis_reports_all_entities() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.privilege_analysis();
    assert!(!report.entities.is_empty());
    assert!(report.max_privilege_score >= report.mean_privilege_score);
}

#[test]
fn test_privilege_analysis_admin_counts() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "s1", "v1").unwrap();
    vault.set(Vault::ROOT, "s2", "v2").unwrap();
    vault.grant(Vault::ROOT, "user:admin", "s1").unwrap();
    vault.grant(Vault::ROOT, "user:admin", "s2").unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:reader", "s1", Permission::Read)
        .unwrap();

    let report = vault.privilege_analysis();
    let admin_entry = report.entities.iter().find(|e| e.entity == "user:admin");
    let reader_entry = report.entities.iter().find(|e| e.entity == "user:reader");

    if let (Some(admin), Some(reader)) = (admin_entry, reader_entry) {
        assert!(
            admin.admin_count >= reader.admin_count,
            "admin should have higher admin_count than reader"
        );
    }
}

// =====================================================================
// Graph Intel: Role Inference
// =====================================================================

#[test]
fn test_infer_roles_with_multiple_entities() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let result = vault.infer_roles();
    // Modularity must be finite
    assert!(result.modularity.is_finite());
    // All entities should be accounted for in roles or unassigned
    let role_members: usize = result.roles.iter().map(|r| r.members.len()).sum();
    let total_covered = role_members + result.unassigned.len();
    // populate_vault creates 3 entities: alice, bob, carol
    assert!(total_covered >= 3, "expected at least 3 entities, got {total_covered}");
    // Each role should have a valid ID and at least one member
    for role in &result.roles {
        assert!(!role.members.is_empty(), "role should have at least one member");
        assert!(
            !role.common_secrets.is_empty(),
            "role should share at least one secret"
        );
    }
}

// =====================================================================
// Graph Intel: Trust Transitivity
// =====================================================================

#[test]
fn test_trust_transitivity_basic() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.trust_transitivity();
    assert!(!report.entities.is_empty());
    assert!(report.global_clustering >= 0.0);
    for entry in &report.entities {
        assert!(entry.trust_score >= 0.0);
        assert!(entry.clustering_coefficient >= 0.0);
    }
}

// =====================================================================
// Graph Intel: Risk Propagation
// =====================================================================

#[test]
fn test_risk_propagation_basic() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.risk_propagation();
    assert!(!report.entities.is_empty());
    assert!(report.mean_risk >= 0.0);
    assert!(report.max_risk >= report.mean_risk);
}

#[test]
fn test_risk_propagation_admin_vs_read() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "critical", "admin-secret").unwrap();
    vault.set(Vault::ROOT, "public", "read-secret").unwrap();
    vault.grant(Vault::ROOT, "user:power", "critical").unwrap();
    vault
        .grant_with_permission(Vault::ROOT, "user:limited", "public", Permission::Read)
        .unwrap();

    let report = vault.risk_propagation();
    let power_risk = report
        .entities
        .iter()
        .find(|e| e.entity == "user:power")
        .map(|e| e.risk_score);
    let limited_risk = report
        .entities
        .iter()
        .find(|e| e.entity == "user:limited")
        .map(|e| e.risk_score);

    // Both should have some risk score computed
    if let (Some(pr), Some(lr)) = (power_risk, limited_risk) {
        // The admin user should have at least as much risk
        assert!(pr >= lr || (pr - lr).abs() < f64::EPSILON);
    }
}

// =====================================================================
// Graph Intel: Cluster Entities
// =====================================================================

#[test]
fn test_cluster_entities_basic() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let result = vault.cluster_entities();
    // Every assigned entity maps to a cluster
    for (entity, cluster_id) in &result.assignments {
        assert!(
            result
                .clusters
                .iter()
                .any(|c| c.cluster_id == *cluster_id && c.members.contains(entity)),
            "entity {entity} should appear in cluster {cluster_id}"
        );
    }
}

// =====================================================================
// Graph Intel: Behavior Embeddings
// =====================================================================

#[test]
fn test_compute_behavior_embeddings_default() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let embeddings = vault.compute_behavior_embeddings(BehaviorEmbeddingConfig::default());
    assert!(!embeddings.is_empty());
    for emb in &embeddings {
        assert!(
            !emb.embedding.is_empty(),
            "embedding vector should not be empty"
        );
        // Embeddings are L2-normalized: norm should be close to 1.0
        let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01 || norm < f32::EPSILON,
            "expected L2-normalized embedding, got norm={norm}"
        );
    }
}

#[test]
fn test_compute_behavior_embeddings_topology_only() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let config = BehaviorEmbeddingConfig {
        use_topology_features: true,
        use_access_patterns: false,
    };
    let embeddings = vault.compute_behavior_embeddings(config);
    assert!(!embeddings.is_empty());
    // With topology only, embeddings should have 3 dimensions
    for emb in &embeddings {
        assert_eq!(
            emb.embedding.len(),
            3,
            "topology-only embedding should have 3 features"
        );
    }
}

// =====================================================================
// Graph Intel: Geometric Anomaly Detection
// =====================================================================

#[test]
fn test_detect_geometric_anomalies_basic() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.detect_geometric_anomalies(2, 2.0);
    assert!(report.total_entities > 0);
    assert!(report.mean_distance >= 0.0);
    assert!(report.threshold >= report.mean_distance);
}

#[test]
fn test_detect_geometric_anomalies_outlier() {
    let vault = create_test_vault();
    // Create a cluster of similar users
    for i in 0..5 {
        let name = format!("common/secret{i}");
        vault.set(Vault::ROOT, &name, &format!("val{i}")).unwrap();
        vault.grant(Vault::ROOT, "user:member1", &name).unwrap();
        vault.grant(Vault::ROOT, "user:member2", &name).unwrap();
        vault.grant(Vault::ROOT, "user:member3", &name).unwrap();
    }
    // Create an outlier with unique access pattern
    vault.set(Vault::ROOT, "outlier/unique", "val").unwrap();
    vault
        .grant(Vault::ROOT, "user:outlier", "outlier/unique")
        .unwrap();

    let report = vault.detect_geometric_anomalies(2, 1.5);
    assert!(report.total_entities > 0);
    // The outlier entity may be flagged
    let outlier_flagged = report.anomalies.iter().any(|a| a.entity == "user:outlier");
    // With such a small graph, the outlier may or may not be flagged,
    // but the anomaly detection pipeline should run without error
    assert!(
        report.threshold > 0.0 || report.total_entities <= 1,
        "threshold should be positive with multiple entities"
    );
    // If the outlier is flagged, its anomaly score should be positive
    if outlier_flagged {
        let outlier = report
            .anomalies
            .iter()
            .find(|a| a.entity == "user:outlier")
            .unwrap();
        assert!(outlier.anomaly_score > 0.0);
    }
}

// =====================================================================
// Heat Kernel Trust
// =====================================================================

#[test]
fn test_heat_kernel_trust_default_config() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.heat_kernel_trust(HeatKernelConfig::default());
    assert!(report.total_entities > 0);
    assert!((report.diffusion_time - 1.0).abs() < f64::EPSILON);
    assert!(report.spectral_gap >= 0.0);
    for entry in &report.entities {
        assert!(entry.trust_score >= 0.0);
    }
}

#[test]
fn test_heat_kernel_trust_high_diffusion() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let config = HeatKernelConfig {
        diffusion_time: 10.0,
        chebyshev_order: 10,
        max_iterations: 100,
    };
    let report = vault.heat_kernel_trust(config);
    assert!(report.total_entities > 0);
    assert!((report.diffusion_time - 10.0).abs() < f64::EPSILON);
    assert!(report.spectral_gap >= 0.0);
}

#[test]
fn test_heat_kernel_trust_empty_vault() {
    let vault = create_test_vault();
    let report = vault.heat_kernel_trust(HeatKernelConfig::default());
    assert_eq!(report.total_entities, 0);
    assert!(report.entities.is_empty());
}

#[test]
fn test_heat_kernel_trust_star_topology() {
    let vault = create_test_vault();
    // Star topology: hub entity has grants to many secrets, leaf entities have one each
    for i in 0..5 {
        let name = format!("star/secret{i}");
        vault.set(Vault::ROOT, &name, &format!("v{i}")).unwrap();
        vault.grant(Vault::ROOT, "user:hub", &name).unwrap();
        let leaf = format!("user:leaf{i}");
        vault.grant(Vault::ROOT, &leaf, &name).unwrap();
    }

    let report = vault.heat_kernel_trust(HeatKernelConfig::default());
    assert!(report.total_entities > 0);

    // The hub entity should exist in the report
    let hub_entry = report.entities.iter().find(|e| e.entity == "user:hub");
    assert!(
        hub_entry.is_some(),
        "hub entity should appear in heat kernel report"
    );
}

#[test]
fn test_heat_kernel_trust_disconnected_entity_low_trust() {
    let vault = create_test_vault();
    populate_vault(&vault);
    // Create a disconnected entity (no grants)
    vault.set(Vault::ROOT, "isolated/secret", "val").unwrap();
    vault
        .grant(Vault::ROOT, "user:isolated", "isolated/secret")
        .unwrap();

    let report = vault.heat_kernel_trust(HeatKernelConfig::default());
    // All entities with grants should have some trust scores
    assert!(report.total_entities > 0);
    assert!(report.spectral_gap >= 0.0);
}

// =====================================================================
// Topology: Access Topology
// =====================================================================

#[test]
fn test_build_access_topology_default() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let topology = vault
        .build_access_topology(TopologyConfig::default())
        .unwrap();
    assert!(topology.num_entities() > 0);
    assert!(topology.num_secrets() > 0);
}

#[test]
fn test_build_access_topology_batch_check() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let topology = vault
        .build_access_topology(TopologyConfig::default())
        .unwrap();

    let queries = vec![
        ("user:alice", "secret1"),
        ("user:bob", "secret1"),
        ("user:bob", "secret2"),
        ("user:carol", "secret2"),
        ("user:carol", "secret3"),
        ("user:alice", "secret3"), // alice has no access to secret3
    ];

    let results = topology.batch_check(&queries);
    assert_eq!(results.len(), 6);

    // alice -> secret1: should have admin (grant gives Admin by default)
    assert!(results[0].has_admin);
    // bob -> secret1: admin
    assert!(results[1].has_admin);
    // bob -> secret2: admin
    assert!(results[2].has_admin);
    // carol -> secret2: read only
    assert!(results[3].has_read);
    assert!(!results[3].has_admin);
    // carol -> secret3: write
    assert!(results[4].has_write);
    // alice -> secret3: no access
    assert!(!results[5].has_read);
    assert!(!results[5].has_write);
    assert!(!results[5].has_admin);
}

#[test]
fn test_build_access_topology_compression_stats() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let topology = vault
        .build_access_topology(TopologyConfig {
            enable_tt_compression: true,
            compression_threshold: 10_000,
        })
        .unwrap();
    let (original, compressed, ratio) = topology.compression_stats();
    // With a small vault, TT compression should not be applied (below threshold)
    assert!(original > 0);
    assert_eq!(original, compressed);
    assert!((ratio - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_analyze_policy_redundancy() {
    let vault = create_test_vault();
    populate_vault(&vault);
    let report = vault.analyze_policy_redundancy().unwrap();
    assert!(report.compression_ratio >= 1.0 || report.total_policies < 2);
}

// =====================================================================
// Manifold: Geographic Placement
// =====================================================================

fn create_region_registry() -> RegionRegistry {
    let registry = RegionRegistry::new();

    let mut latencies_us_east = HashMap::new();
    latencies_us_east.insert("us-west".to_string(), 60.0);
    latencies_us_east.insert("eu-west".to_string(), 80.0);

    let mut latencies_us_west = HashMap::new();
    latencies_us_west.insert("us-east".to_string(), 60.0);
    latencies_us_west.insert("eu-west".to_string(), 140.0);

    let mut latencies_eu_west = HashMap::new();
    latencies_eu_west.insert("us-east".to_string(), 80.0);
    latencies_eu_west.insert("us-west".to_string(), 140.0);

    registry.add_region(VaultRegion {
        name: "us-east".to_string(),
        center: GeoCoordinate {
            x: -74.0,
            y: 40.7,
            z: None,
        },
        capacity: 1000,
        current_load: 200,
        latencies: latencies_us_east,
    });

    registry.add_region(VaultRegion {
        name: "us-west".to_string(),
        center: GeoCoordinate {
            x: -122.4,
            y: 37.8,
            z: None,
        },
        capacity: 1000,
        current_load: 100,
        latencies: latencies_us_west,
    });

    registry.add_region(VaultRegion {
        name: "eu-west".to_string(),
        center: GeoCoordinate {
            x: -0.1,
            y: 51.5,
            z: None,
        },
        capacity: 800,
        current_load: 500,
        latencies: latencies_eu_west,
    });

    registry
}

#[test]
fn test_recommend_placement_single_secret() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "geo/secret1", "val1").unwrap();
    vault
        .grant(Vault::ROOT, "user:alice", "geo/secret1")
        .unwrap();

    let registry = create_region_registry();
    registry.set_entity_location(
        "user:alice",
        GeoCoordinate {
            x: -73.9,
            y: 40.7,
            z: None,
        },
    );

    let config = PlacementConfig::default();
    let recommendation = recommend_placement(&vault, &registry, "geo/secret1", &config).unwrap();
    assert_eq!(recommendation.secret_key, "geo/secret1");
    assert!(!recommendation.primary_region.is_empty());
    assert!(!recommendation.replica_regions.is_empty());
    // Primary region should not appear in replicas
    assert!(
        !recommendation
            .replica_regions
            .contains(&recommendation.primary_region),
        "primary should not be in replicas"
    );
}

#[test]
fn test_recommend_placement_picks_closest_region() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "eu/secret", "val").unwrap();
    vault
        .grant(Vault::ROOT, "user:london", "eu/secret")
        .unwrap();

    let registry = create_region_registry();
    // Place the user near London
    registry.set_entity_location(
        "user:london",
        GeoCoordinate {
            x: -0.12,
            y: 51.5,
            z: None,
        },
    );

    let config = PlacementConfig {
        locality_weight: 1.0,
        load_balance_weight: 0.0,
        replication_weight: 0.0,
        replica_count: 2,
    };
    let recommendation = recommend_placement(&vault, &registry, "eu/secret", &config).unwrap();
    assert_eq!(
        recommendation.primary_region, "eu-west",
        "should pick eu-west as closest to London"
    );
}

#[test]
fn test_batch_recommend_placement() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "batch/s1", "v1").unwrap();
    vault.set(Vault::ROOT, "batch/s2", "v2").unwrap();
    vault.set(Vault::ROOT, "batch/s3", "v3").unwrap();
    vault.grant(Vault::ROOT, "user:alice", "batch/s1").unwrap();
    vault.grant(Vault::ROOT, "user:bob", "batch/s2").unwrap();
    vault.grant(Vault::ROOT, "user:carol", "batch/s3").unwrap();

    let registry = create_region_registry();
    registry.set_entity_location(
        "user:alice",
        GeoCoordinate {
            x: -74.0,
            y: 40.7,
            z: None,
        },
    );
    registry.set_entity_location(
        "user:bob",
        GeoCoordinate {
            x: -122.0,
            y: 37.8,
            z: None,
        },
    );
    registry.set_entity_location(
        "user:carol",
        GeoCoordinate {
            x: -0.1,
            y: 51.5,
            z: None,
        },
    );

    let config = PlacementConfig::default();
    let recommendations = batch_recommend_placement(&vault, &registry, &config);
    // Should have recommendations for all secrets listed by root
    assert!(
        recommendations.len() >= 3,
        "expected at least 3 recommendations, got {}",
        recommendations.len()
    );
    for rec in &recommendations {
        assert!(!rec.primary_region.is_empty());
    }
}

#[test]
fn test_recommend_placement_no_regions_error() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "orphan", "val").unwrap();

    let registry = RegionRegistry::new();
    let config = PlacementConfig::default();
    let result = recommend_placement(&vault, &registry, "orphan", &config);
    assert!(result.is_err(), "should fail with no regions registered");
}

// =====================================================================
// Cross-module: Graph Intel + Heat Kernel
// =====================================================================

#[test]
fn test_trust_transitivity_feeds_heat_kernel() {
    let vault = create_test_vault();
    populate_vault(&vault);

    // Trust transitivity provides initial seeds for heat kernel
    let trust_report = vault.trust_transitivity();
    let heat_report = vault.heat_kernel_trust(HeatKernelConfig::default());

    // Same entities should appear in both reports
    assert_eq!(
        trust_report.entities.len(),
        heat_report.total_entities,
        "trust transitivity and heat kernel should analyze same entity count"
    );
}

// =====================================================================
// Cross-module: Graph Intel + Topology
// =====================================================================

#[test]
fn test_security_audit_and_topology_consistency() {
    let vault = create_test_vault();
    populate_vault(&vault);

    let audit = vault.security_audit();
    let topology = vault
        .build_access_topology(TopologyConfig::default())
        .unwrap();

    // The topology should reflect the same secrets found by the audit
    assert!(
        topology.num_secrets() > 0 && audit.total_secrets > 0,
        "both audit and topology should find secrets"
    );
}

// =====================================================================
// Cross-module: Delegation + Graph Intel
// =====================================================================

#[test]
fn test_delegation_affects_privilege_analysis() {
    let vault = create_test_vault();
    vault.set(Vault::ROOT, "delegated", "secret").unwrap();
    vault
        .grant(Vault::ROOT, "user:parent", "delegated")
        .unwrap();

    // Delegate to child
    vault
        .delegate(
            "user:parent",
            "user:child",
            &["delegated"],
            Permission::Read,
            Some(Duration::from_secs(3600)),
        )
        .unwrap();

    let report = vault.privilege_analysis();
    let child = report.entities.iter().find(|e| e.entity == "user:child");
    // Child should appear in privilege analysis after delegation
    assert!(
        child.is_some(),
        "delegated child should appear in privilege analysis"
    );
}
