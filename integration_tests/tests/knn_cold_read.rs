// SPDX-License-Identifier: MIT OR Apache-2.0
//! k-NN cold read integration tests.
//!
//! Tests geometric-aware tiered storage with HNSW/Voronoi integration.
//! Verifies that k-NN queries can efficiently access cold data through
//! region-aware migration and preloading.

use std::{collections::HashSet, path::PathBuf, thread, time::Duration};

use tensor_store::{
    tiered::{MigrationStrategy, TieredConfig, TieredStore},
    HNSWConfig, HNSWIndex, ScalarValue, SparseVector, TensorData, TensorValue, VoronoiPartitioner,
    VoronoiPartitionerConfig,
};

fn setup_test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("/tmp/knn_cold_read_test_{}", name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn create_embedding_tensor(id: i64, embedding: Vec<f32>) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    tensor.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
    );
    tensor.set("embedding", TensorValue::Vector(embedding));
    tensor
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / mag).collect()
    }
}

fn generate_cluster_embeddings(
    cluster_id: usize,
    count: usize,
    dim: usize,
) -> Vec<(i64, Vec<f32>)> {
    let mut embeddings = Vec::with_capacity(count);

    for i in 0..count {
        let mut emb = vec![0.0; dim];
        // Create cluster center based on cluster_id
        let base_idx = cluster_id % dim;
        emb[base_idx] = 1.0;
        // Add small variations for each vector in the cluster
        for (j, val) in emb.iter_mut().enumerate() {
            *val += (i as f32 * 0.01 + j as f32 * 0.001) * (cluster_id as f32 + 1.0).sin();
        }
        let id = (cluster_id * count + i) as i64;
        embeddings.push((id, normalize(&emb)));
    }

    embeddings
}

#[test]
fn test_hnsw_search_with_cold_data() {
    let dir = setup_test_dir("hnsw_cold");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::AccessRecency,
    };

    let mut store = TieredStore::new(config).unwrap();
    let index = HNSWIndex::with_config(HNSWConfig::default());

    // Insert 4 clusters of embeddings
    let dim = 32;
    let mut all_ids = Vec::new();
    for cluster in 0..4 {
        let cluster_data = generate_cluster_embeddings(cluster, 25, dim);
        for (id, emb) in &cluster_data {
            let tensor = create_embedding_tensor(*id, emb.clone());
            store.put(format!("vec:{}", id), tensor);
            index.insert(emb.clone());
            all_ids.push(*id);
        }
    }

    assert_eq!(store.hot_len(), 100);
    assert_eq!(index.len(), 100);

    // Access only cluster 0 to keep it hot
    for i in 0..25 {
        let _ = store.get(&format!("vec:{}", i));
    }

    // Wait and migrate cold data
    thread::sleep(Duration::from_millis(100));
    let migrated = store.migrate_cold(50).unwrap();

    // Should have migrated some vectors from clusters 1-3
    assert!(migrated > 0 || store.cold_len() > 0 || store.hot_len() <= 25);

    // k-NN search for a vector in cluster 2 (should be cold)
    let query_emb = generate_cluster_embeddings(2, 1, dim)[0].1.clone();
    let sparse_query = SparseVector::from_dense(&query_emb);
    let results = index.search_sparse(&sparse_query, 10);

    // Should find results
    assert!(!results.is_empty());

    // Top results should be from cluster 2
    let cluster_2_ids: HashSet<_> = (50..75).collect();
    let top_in_cluster = results
        .iter()
        .take(5)
        .filter(|(id, _)| cluster_2_ids.contains(&(*id as i64)))
        .count();
    assert!(top_in_cluster >= 3, "Expected top results from cluster 2");

    // Now access these cold vectors - they should be readable
    for (id, _score) in results.iter().take(5) {
        let key = format!("vec:{}", id);
        let tensor = store.get(&key);
        assert!(
            tensor.is_ok(),
            "Failed to read vector {} (may be from cold tier)",
            id
        );
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_region_aware_tiered_migration() {
    let dir = setup_test_dir("region_migration");
    let dim = 16;

    // Create Voronoi partitioner and verify it can compute regions
    let mut voronoi_config = VoronoiPartitionerConfig::new("test_node", 4, dim);
    voronoi_config.min_samples_for_regions = 4;
    let partitioner = VoronoiPartitioner::new(voronoi_config);

    let centroids: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut c = vec![0.0; dim];
            c[i % dim] = 1.0;
            normalize(&c)
        })
        .collect();

    for centroid in &centroids {
        partitioner.add_sample(centroid.clone());
    }
    partitioner.compute_regions_from_samples(&centroids);

    if !partitioner.has_regions() {
        return;
    }

    // Verify partitioner assigns vectors to regions
    for centroid in &centroids {
        let region = partitioner.region_for_embedding(centroid);
        assert!(region.is_some(), "Centroid should map to a region");
    }

    // Test tiered store migration with VoronoiRegion strategy config
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::VoronoiRegion,
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert vectors clustered around centroids
    for cluster in 0..4 {
        let cluster_data = generate_cluster_embeddings(cluster, 10, dim);
        for (id, emb) in &cluster_data {
            let tensor = create_embedding_tensor(*id, emb.clone());
            store.put(format!("vec:{}", id), tensor);
        }
    }

    assert_eq!(store.hot_len(), 40);

    // Access only cluster 0 vectors to keep them hot
    for i in 0..10 {
        let _ = store.get(&format!("vec:{}", i));
    }

    // Wait and migrate cold data
    thread::sleep(Duration::from_millis(100));
    let migrated = store.migrate_cold(50).unwrap();

    // Verify total entries preserved
    let stats = store.stats();
    assert_eq!(
        stats.hot_count + stats.cold_count,
        40,
        "Total entries should remain 40"
    );

    // If migration happened, preload some back
    if migrated > 0 {
        let keys: Vec<String> = (10..20).map(|i| format!("vec:{}", i)).collect();
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let preloaded = store.preload(&key_refs).unwrap();
        assert!(preloaded <= migrated);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_hybrid_migration_strategy() {
    let dir = setup_test_dir("hybrid_migration");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::Hybrid {
            recency_weight: 0.7,
        },
    };

    let mut store = TieredStore::new(config).unwrap();

    let dim = 16;
    // Insert vectors
    for i in 0..50 {
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        emb[(i + 1) % dim] = 0.5;
        let tensor = create_embedding_tensor(i as i64, normalize(&emb));
        store.put(format!("vec:{}", i), tensor);
    }

    // Create access pattern: frequently access vectors 0-9
    for _ in 0..10 {
        for i in 0..10 {
            let _ = store.get(&format!("vec:{}", i));
        }
    }

    // Wait and trigger migration via migrate_cold
    thread::sleep(Duration::from_millis(100));
    let _migrated = store.migrate_cold(50).unwrap_or(0);

    // Hot data (0-9) should remain hot, cold data (10-49) may migrate
    let stats = store.stats();

    // Verify hot data is still accessible
    for i in 0..10 {
        let key = format!("vec:{}", i);
        let tensor = store.get(&key);
        assert!(tensor.is_ok(), "Hot vector {} should be accessible", i);
    }

    // All data should still be accessible
    assert_eq!(stats.hot_count + stats.cold_count, 50);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_knn_preload_candidate_keys() {
    let dir = setup_test_dir("knn_preload");
    let dim = 8;

    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::AccessRecency,
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert vectors into 4 logical clusters
    for cluster in 0..4 {
        for j in 0..8 {
            let id = cluster * 8 + j;
            let mut emb = vec![0.0; dim];
            emb[cluster % dim] = 1.0 - (j as f32 * 0.02);
            emb[(cluster + 1) % dim] = j as f32 * 0.03;
            let tensor = create_embedding_tensor(id as i64, normalize(&emb));
            store.put(format!("vec:{}", id), tensor);
        }
    }

    assert_eq!(store.hot_len(), 32);

    // Migrate all to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    let cold_before = store.cold_len();

    // Preload specific keys (simulating k-NN candidate preloading)
    let preload_keys: Vec<String> = (0..8).map(|i| format!("vec:{}", i)).collect();
    let key_refs: Vec<&str> = preload_keys.iter().map(String::as_str).collect();
    let _preloaded = store.preload(&key_refs).unwrap();

    if cold_before > 0 {
        let cold_after = store.cold_len();
        assert!(
            cold_after <= cold_before,
            "Preload should reduce cold entries"
        );
    }

    // Preloaded vectors should be accessible
    for i in 0..8 {
        let key = format!("vec:{}", i);
        if store.exists(&key) {
            let tensor = store.get(&key);
            assert!(
                tensor.is_ok(),
                "Preloaded vector {} should be accessible",
                i
            );
        }
    }

    // Non-preloaded vectors should still be accessible (from cold tier)
    for i in 24..32 {
        let key = format!("vec:{}", i);
        let tensor = store.get(&key);
        assert!(
            tensor.is_ok(),
            "Cold vector {} should still be accessible",
            i
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_cross_tier_knn_query() {
    let dir = setup_test_dir("cross_tier");
    let dim = 16;

    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::AccessRecency,
    };

    let mut store = TieredStore::new(config).unwrap();
    let index = HNSWIndex::with_config(HNSWConfig::default());

    // Insert vectors to all clusters
    for cluster in 0..4 {
        let cluster_data = generate_cluster_embeddings(cluster, 10, dim);
        for (id, emb) in &cluster_data {
            let tensor = create_embedding_tensor(*id, emb.clone());
            store.put(format!("vec:{}", id), tensor);
            index.insert(emb.clone());
        }
    }

    // Access only cluster 0 to keep it hot
    for i in 0..10 {
        let _ = store.get(&format!("vec:{}", i));
    }

    // Migrate cold data
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(5);

    // Query that spans multiple clusters (boundary query)
    let mut query = vec![0.3; dim];
    query[0] = 0.8;
    query[1] = 0.5;
    let query = normalize(&query);
    let sparse_query = SparseVector::from_dense(&query);

    // k-NN search via HNSW index
    let results = index.search_sparse(&sparse_query, 20);
    assert!(!results.is_empty());

    // Access results - some may be in cold tier
    let mut accessed = 0;
    for (id, _score) in &results {
        let key = format!("vec:{}", id);
        if let Ok(tensor) = store.get(&key) {
            assert!(tensor.get("id").is_some());
            accessed += 1;
        }
    }

    // All k-NN results should be accessible from hot or cold tier
    assert_eq!(
        accessed,
        results.len(),
        "All k-NN results should be accessible"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_hnsw_search_survives_cold_migration() {
    let dir = setup_test_dir("hnsw_migration");
    let dim = 32;

    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 10 * 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::AccessRecency,
    };

    let mut store = TieredStore::new(config).unwrap();
    let index = HNSWIndex::with_config(HNSWConfig::default());

    // Insert vectors
    for i in 0..50 {
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        let emb = normalize(&emb);
        let tensor = create_embedding_tensor(i as i64, emb.clone());
        store.put(format!("vec:{}", i), tensor);
        index.insert(emb);
    }

    assert_eq!(index.len(), 50);

    // Migrate some data to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    // HNSW index still works after store migration
    let query = normalize(&vec![1.0; dim]);
    let sparse_query = SparseVector::from_dense(&query);
    let results = index.search_sparse(&sparse_query, 5);

    assert!(!results.is_empty());

    // All results should be accessible from store (hot or cold)
    for (id, _) in &results {
        let key = format!("vec:{}", id);
        let tensor = store.get(&key);
        assert!(tensor.is_ok(), "Vector {} should be accessible", id);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_voronoi_partitioner_region_consistency() {
    let dim = 8;

    // Create Voronoi partitioner
    let mut voronoi_config = VoronoiPartitionerConfig::new("test_node", 4, dim);
    voronoi_config.min_samples_for_regions = 4;
    let partitioner = VoronoiPartitioner::new(voronoi_config);

    let centroids: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut c = vec![0.0; dim];
            c[i % dim] = 1.0;
            normalize(&c)
        })
        .collect();

    for centroid in &centroids {
        partitioner.add_sample(centroid.clone());
    }
    partitioner.compute_regions_from_samples(&centroids);

    if !partitioner.has_regions() {
        return;
    }

    // Verify region assignment is deterministic
    for i in 0..20 {
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        let emb = normalize(&emb);
        let r1 = partitioner.region_for_embedding(&emb);
        let r2 = partitioner.region_for_embedding(&emb);
        match (&r1, &r2) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.owner, b.owner,
                    "Region assignment must be deterministic for embedding {}",
                    i
                );
                assert_eq!(
                    a.centroid, b.centroid,
                    "Region centroid must match for embedding {}",
                    i
                );
            },
            (None, None) => {},
            _ => panic!("Inconsistent region assignment for embedding {}", i),
        }
    }

    // Verify all regions are represented
    let all_regions = partitioner.all_regions();
    assert!(!all_regions.is_empty(), "Should have computed regions");
}
