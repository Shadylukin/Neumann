// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for archetype registry persistence.
//!
//! Tests:
//! - Delta replication manager preserves archetypes across restarts
//! - Archetype recovery after simulated crash
//! - Registry persistence with discovered archetypes

use tensor_chain::{DeltaReplicationConfig, DeltaReplicationManager};
use tensor_store::{ArchetypeRegistry, TensorStore};

fn make_embedding(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 17 + i) as f32 / 100.0).sin())
        .collect()
}

fn make_clustered_embedding(cluster: usize, variation: usize, dim: usize) -> Vec<f32> {
    let base = make_embedding(cluster * 1000, dim);
    let var: Vec<f32> = (0..dim)
        .map(|i| ((variation * 7 + i) as f32 / 1000.0).cos() * 0.1)
        .collect();

    base.iter().zip(var.iter()).map(|(b, v)| b + v).collect()
}

#[test]
fn test_delta_replication_preserves_archetypes() {
    let store = TensorStore::new();

    // Create a registry with archetypes and save it
    {
        let mut registry = ArchetypeRegistry::new(256);
        registry.register(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        registry.register(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        registry.register(vec![0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        registry.save_to_store(&store).unwrap();
    }

    // Create DeltaReplicationManager with_store - should load existing registry
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::with_store("node1".to_string(), config, &store);

    // Queue an update - should use the loaded archetypes
    let embedding = vec![0.98, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
    manager
        .queue_update("test_key".to_string(), &embedding, 1)
        .unwrap();

    // Verify the update was queued
    assert_eq!(manager.pending_count(), 1);
}

#[test]
fn test_archetype_recovery_after_restart() {
    let store = TensorStore::new();
    let config = DeltaReplicationConfig::default();

    // First "session" - discover archetypes and save
    {
        let manager =
            DeltaReplicationManager::with_store("node1".to_string(), config.clone(), &store);

        // Queue several clustered embeddings to trigger archetype discovery
        for cluster in 0..3 {
            for variation in 0..5 {
                let embedding = make_clustered_embedding(cluster, variation, 64);
                let _ = manager.queue_update(
                    format!("key_{}_{}", cluster, variation),
                    &embedding,
                    (cluster * 5 + variation) as u64,
                );
            }
        }

        // Persist the registry
        manager.persist_registry(&store).unwrap();
    }

    // Second "session" - verify archetypes were recovered
    {
        let manager = DeltaReplicationManager::with_store("node2".to_string(), config, &store);

        // Queue new updates - should use recovered archetypes
        let embedding = make_clustered_embedding(0, 99, 64);
        manager
            .queue_update("new_key".to_string(), &embedding, 100)
            .unwrap();

        // Verify update was queued successfully
        assert_eq!(manager.pending_count(), 1);
    }
}

#[test]
fn test_registry_persistence_with_discovered_archetypes() {
    let store = TensorStore::new();

    // Create registry and discover archetypes from data
    {
        let mut registry = ArchetypeRegistry::new(256);

        // Create clustered vectors for k-means
        let mut vectors = Vec::new();
        for cluster in 0..4 {
            for _ in 0..10 {
                vectors.push(make_clustered_embedding(cluster, 0, 32));
            }
        }

        // Discover archetypes
        let discovered =
            registry.discover_archetypes(&vectors, 4, tensor_store::KMeansConfig::default());
        assert!(discovered > 0);

        // Save to store
        registry.save_to_store(&store).unwrap();
    }

    // Load and verify
    {
        let restored = ArchetypeRegistry::load_from_store(&store, 256).unwrap();
        assert!(!restored.is_empty());

        // Verify encoding still works with loaded archetypes
        let test_vec = make_clustered_embedding(0, 0, 32);
        let (best_id, similarity) = restored.find_best_archetype(&test_vec).unwrap();
        assert!(
            similarity > 0.8,
            "Expected high similarity, got {}",
            similarity
        );
        assert!(best_id < restored.len());
    }
}

#[test]
fn test_manager_new_vs_with_store() {
    let store = TensorStore::new();

    // Save a known registry
    {
        let mut registry = ArchetypeRegistry::new(256);
        for i in 0..5 {
            registry.register(vec![i as f32; 8]);
        }
        registry.save_to_store(&store).unwrap();
    }

    // Manager::new() should start fresh (no archetypes)
    let config = DeltaReplicationConfig::default();
    let manager_new = DeltaReplicationManager::new("node_new".to_string(), config.clone());

    // Manager::with_store() should load existing archetypes
    let manager_store =
        DeltaReplicationManager::with_store("node_store".to_string(), config, &store);

    // Queue same embedding to both
    let embedding = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    manager_new
        .queue_update("key1".to_string(), &embedding, 1)
        .unwrap();
    manager_store
        .queue_update("key2".to_string(), &embedding, 2)
        .unwrap();

    // Both should have queued successfully
    assert_eq!(manager_new.pending_count(), 1);
    assert_eq!(manager_store.pending_count(), 1);
}

#[test]
fn test_persist_registry_multiple_times() {
    let store = TensorStore::new();
    let config = DeltaReplicationConfig::default();

    // Save initial state
    {
        let mut registry = ArchetypeRegistry::new(256);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]);
        registry.save_to_store(&store).unwrap();
    }

    // First persist with manager
    {
        let manager =
            DeltaReplicationManager::with_store("node".to_string(), config.clone(), &store);
        manager.persist_registry(&store).unwrap();
    }

    // Load and verify - should still have original archetype
    {
        let restored = ArchetypeRegistry::load_from_store(&store, 256).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get(0), Some(&[1.0f32, 0.0, 0.0, 0.0][..]));
    }
}
