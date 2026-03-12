// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for delta replication manager.
//!
//! Tests basic delta encoding and replication functionality.

use tensor_chain::{DeltaBatch, DeltaReplicationConfig, DeltaReplicationManager, DeltaUpdate};

// ============================================================================
// Helper Functions
// ============================================================================

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

// ============================================================================
// Basic Tests
// ============================================================================

#[test]
fn test_delta_replication_manager_creation() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    let stats = manager.stats();
    assert_eq!(stats.updates_sent, 0);
}

#[test]
fn test_queue_update() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue an update
    let embedding = make_embedding(42, 64);
    manager
        .queue_update("test_key".to_string(), &embedding, 1)
        .unwrap();

    assert_eq!(manager.pending_count(), 1);
}

#[test]
fn test_queue_multiple_updates() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue multiple updates
    for i in 0..10 {
        let embedding = make_embedding(i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    assert_eq!(manager.pending_count(), 10);
}

#[test]
fn test_create_batch() {
    let config = DeltaReplicationConfig {
        max_batch_size: 5,
        ..Default::default()
    };
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue updates
    for i in 0..10 {
        let embedding = make_embedding(i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    // Create a batch
    let batch = manager.create_batch(false);
    assert!(batch.is_some());

    let batch = batch.unwrap();
    assert!(!batch.is_empty());
}

#[test]
fn test_flush_all_batches() {
    let config = DeltaReplicationConfig {
        max_batch_size: 3,
        ..Default::default()
    };
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue updates
    for i in 0..10 {
        let embedding = make_embedding(i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    // Flush all
    let batches = manager.flush();
    assert!(!batches.is_empty());

    // After flush, pending should be 0
    assert_eq!(manager.pending_count(), 0);
}

#[test]
fn test_initialize_archetypes() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Create sample embeddings
    let samples: Vec<Vec<f32>> = (0..20).map(|i| make_embedding(i, 64)).collect();

    // Initialize archetypes
    let count = manager.initialize_archetypes(&samples, 4);
    assert!(count > 0);
}

#[test]
fn test_archetype_sync() {
    let config1 = DeltaReplicationConfig::default();
    let config2 = DeltaReplicationConfig::default();
    let manager1 = DeltaReplicationManager::new("node1".to_string(), config1);
    let manager2 = DeltaReplicationManager::new("node2".to_string(), config2);

    // Initialize archetypes on manager1
    let samples: Vec<Vec<f32>> = (0..20).map(|i| make_embedding(i, 64)).collect();
    manager1.initialize_archetypes(&samples, 4);

    // Get archetypes from manager1
    let archetypes = manager1.get_archetype_sync();

    // Apply to manager2
    let count = manager2.apply_archetype_sync(archetypes);
    assert!(count > 0);
}

#[test]
fn test_delta_batch_operations() {
    let mut batch = DeltaBatch::new("source_node".to_string(), 1);

    // Add updates
    let embedding = make_embedding(42, 64);
    let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);
    batch.add(update);

    assert_eq!(batch.len(), 1);
    assert!(!batch.is_empty());
    assert!(batch.memory_bytes() > 0);
}

#[test]
fn test_clustered_data_encoding() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Create clustered samples for archetypes
    let samples: Vec<Vec<f32>> = (0..4)
        .flat_map(|cluster| (0..5).map(move |var| make_clustered_embedding(cluster, var, 64)))
        .collect();

    // Initialize archetypes
    manager.initialize_archetypes(&samples, 4);

    // Queue updates from same clusters
    for i in 0..20 {
        let cluster = i % 4;
        let embedding = make_clustered_embedding(cluster, 10 + i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    // Create batch
    let batch = manager.create_batch(true);
    assert!(batch.is_some());
}

#[test]
fn test_empty_flush() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Flush with no pending updates
    let batches = manager.flush();
    assert!(batches.is_empty());
}

#[test]
fn test_apply_batch_callback() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue updates
    for i in 0..5 {
        let embedding = make_embedding(i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    // Create batch
    let batch = manager.create_batch(true).unwrap();

    // Apply with callback (takes key and embedding)
    let mut applied_count = 0;
    let result = manager.apply_batch(&batch, |_key, _embedding| {
        applied_count += 1;
        Ok(())
    });

    assert!(result.is_ok());
    assert!(applied_count > 0);
}

#[test]
fn test_stats_tracking() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Initialize archetypes
    let samples: Vec<Vec<f32>> = (0..20).map(|i| make_embedding(i, 64)).collect();
    manager.initialize_archetypes(&samples, 4);

    // Queue updates
    for i in 0..10 {
        let embedding = make_embedding(i, 64);
        manager
            .queue_update(format!("key_{}", i), &embedding, i as u64)
            .unwrap();
    }

    // Create batch to trigger encoding
    let _ = manager.create_batch(true);

    let stats = manager.stats();
    // Just verify stats exist and are accessible
    let _ = stats.bytes_sent;
    let _ = stats.updates_sent;
}
