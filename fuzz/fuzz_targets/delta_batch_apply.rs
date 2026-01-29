// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz test for delta replication batch creation and application.
//!
//! Tests that:
//! - Delta batch serialization is stable
//! - Queue operations handle backpressure correctly
//! - Batch creation handles edge cases

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{DeltaReplicationConfig, DeltaReplicationManager};

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    /// Key bytes (converted to string)
    key: Vec<u8>,
    /// Embedding values
    embedding: Vec<u8>,
    /// Version number
    version: u64,
    /// Max pending queue size
    max_pending: u8,
    /// Batch size
    batch_size: u8,
    /// Whether to force full update
    force_full: bool,
    /// Number of updates to queue
    update_count: u8,
}

fn make_key(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "default_key".to_string()
    } else {
        let s: String = bytes
            .iter()
            .take(64)
            .map(|&b| {
                let c = (b % 26) + b'a';
                c as char
            })
            .collect();
        if s.is_empty() {
            "default_key".to_string()
        } else {
            s
        }
    }
}

fn make_embedding(bytes: &[u8]) -> Vec<f32> {
    if bytes.is_empty() {
        vec![0.0; 4]
    } else {
        bytes.iter().take(128).map(|&b| b as f32 / 255.0).collect()
    }
}

fuzz_target!(|input: FuzzInput| {
    // Create manager with fuzzed config
    let max_pending = (input.max_pending as usize).max(1).min(1000);
    let max_batch_size = (input.batch_size as usize).max(1).min(100);

    let config = DeltaReplicationConfig {
        max_pending,
        max_batch_size,
        min_archetype_similarity: 0.8,
        ..Default::default()
    };

    let manager = DeltaReplicationManager::new("fuzz_node".to_string(), config);

    let key = make_key(&input.key);
    let embedding = make_embedding(&input.embedding);

    // Test 1: Queue updates (with backpressure handling)
    let update_count = (input.update_count as usize).min(max_pending * 2);
    let mut queued = 0;
    let mut backpressure_hit = false;

    for i in 0..update_count {
        let result = manager.queue_update(
            format!("{}_{}", key, i),
            &embedding,
            input.version.wrapping_add(i as u64),
        );

        match result {
            Ok(()) => queued += 1,
            Err(_) => {
                backpressure_hit = true;
                break;
            },
        }
    }

    // Verify backpressure works correctly
    if queued >= max_pending {
        // Should have hit backpressure at some point
        assert!(backpressure_hit || queued == max_pending);
    }

    // Test 2: Batch creation
    let batch = manager.create_batch(input.force_full);
    if let Some(batch) = batch {
        // Verify batch properties
        assert!(!batch.updates.is_empty());
        assert_eq!(batch.source, "fuzz_node");

        // Verify serialization roundtrip
        let serialized = match bitcode::serialize(&batch) {
            Ok(bytes) => bytes,
            Err(_) => return,
        };

        let deserialized: tensor_chain::DeltaBatch = match bitcode::deserialize(&serialized) {
            Ok(b) => b,
            Err(_) => panic!("Failed to deserialize batch we just serialized"),
        };

        assert_eq!(batch.source, deserialized.source);
        assert_eq!(batch.sequence, deserialized.sequence);
        assert_eq!(batch.updates.len(), deserialized.updates.len());
    }

    // Test 3: Stats tracking
    let stats = manager.stats();
    if queued > 0 {
        assert!(stats.queue_depth <= max_pending);
    }

    // Test 4: Flush
    let batches = manager.flush();
    // After flush, pending count should be 0
    assert_eq!(manager.pending_count(), 0);

    // All flushed batches should be valid
    for batch in batches {
        assert!(!batch.updates.is_empty());
    }
});
