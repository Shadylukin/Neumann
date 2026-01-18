//! Integration tests for security validation features.
//!
//! Tests message validation, delta checksums, and TLS authentication.

use tensor_chain::{
    AppendEntries, CompositeValidator, DeltaBatch, DeltaReplicationConfig, DeltaReplicationManager,
    DeltaUpdate, Message, MessageValidationConfig, MessageValidator, QueryRequest, RequestVote,
    RequestVoteResponse, TxPrepareMsg,
};
use tensor_store::SparseVector;

// ============================================================================
// Helper Functions
// ============================================================================

fn make_embedding(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 17 + i) as f32 / 100.0).sin())
        .collect()
}

fn make_sparse_vector(dim: usize, nnz: usize) -> SparseVector {
    let positions: Vec<u32> = (0..nnz as u32).collect();
    let values: Vec<f32> = positions.iter().map(|i| (*i as f32 + 1.0) * 0.1).collect();
    SparseVector::from_parts(dim, positions, values)
}

fn default_validation_config() -> MessageValidationConfig {
    MessageValidationConfig {
        enabled: true,
        max_term: 1_000_000,
        max_shard_id: 65536,
        max_tx_timeout_ms: 300_000,
        max_node_id_len: 256,
        max_key_len: 4096,
        max_embedding_dimension: 65536,
        max_embedding_magnitude: 1e6,
        max_query_len: 1_048_576,
        max_message_age_ms: 300_000,
    }
}

// ============================================================================
// Message Validation Integration Tests
// ============================================================================

#[test]
fn test_message_validation_request_vote_valid() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::RequestVote(RequestVote {
        term: 1,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: make_sparse_vector(64, 10),
    });

    let result = validator.validate(&msg, &"node2".to_string());
    assert!(
        result.is_ok(),
        "Valid RequestVote should pass: {:?}",
        result
    );
}

#[test]
fn test_message_validation_request_vote_zero_term() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::RequestVote(RequestVote {
        term: 0,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: make_sparse_vector(64, 10),
    });

    let result = validator.validate(&msg, &"node2".to_string());
    assert!(result.is_err(), "Zero term should fail validation");
}

#[test]
fn test_message_validation_request_vote_excessive_term() {
    let config = MessageValidationConfig {
        max_term: 100,
        ..default_validation_config()
    };
    let validator = CompositeValidator::new(config);

    let msg = Message::RequestVote(RequestVote {
        term: 101,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: make_sparse_vector(64, 10),
    });

    let result = validator.validate(&msg, &"node2".to_string());
    assert!(result.is_err(), "Excessive term should fail validation");
}

#[test]
fn test_message_validation_request_vote_response_valid() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::RequestVoteResponse(RequestVoteResponse {
        term: 1,
        voter_id: "node1".to_string(),
        vote_granted: true,
    });

    let result = validator.validate(&msg, &"node1".to_string());
    assert!(
        result.is_ok(),
        "Valid RequestVoteResponse should pass: {:?}",
        result
    );
}

#[test]
fn test_message_validation_append_entries_valid() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::AppendEntries(AppendEntries {
        term: 1,
        leader_id: "leader".to_string(),
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
        block_embedding: Some(make_sparse_vector(64, 5)),
    });

    let result = validator.validate(&msg, &"leader".to_string());
    assert!(
        result.is_ok(),
        "Valid AppendEntries should pass: {:?}",
        result
    );
}

#[test]
fn test_message_validation_tx_prepare_valid() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::TxPrepare(TxPrepareMsg {
        tx_id: 1,
        coordinator: "coord".to_string(),
        shard_id: 0,
        operations: vec![],
        delta_embedding: make_sparse_vector(64, 5),
        timeout_ms: 5000,
    });

    let result = validator.validate(&msg, &"coord".to_string());
    assert!(result.is_ok(), "Valid TxPrepare should pass: {:?}", result);
}

#[test]
fn test_message_validation_tx_prepare_excessive_timeout() {
    let config = MessageValidationConfig {
        max_tx_timeout_ms: 10_000,
        ..default_validation_config()
    };
    let validator = CompositeValidator::new(config);

    let msg = Message::TxPrepare(TxPrepareMsg {
        tx_id: 1,
        coordinator: "coord".to_string(),
        shard_id: 0,
        operations: vec![],
        delta_embedding: make_sparse_vector(64, 5),
        timeout_ms: 20_000,
    });

    let result = validator.validate(&msg, &"coord".to_string());
    assert!(result.is_err(), "Excessive timeout should fail validation");
}

#[test]
fn test_message_validation_query_request_valid() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let msg = Message::QueryRequest(QueryRequest {
        query_id: 1,
        query: "SELECT * FROM users".to_string(),
        shard_id: 0,
        embedding: Some(make_sparse_vector(64, 5)),
        timeout_ms: 5000,
    });

    let result = validator.validate(&msg, &"client".to_string());
    assert!(
        result.is_ok(),
        "Valid QueryRequest should pass: {:?}",
        result
    );
}

#[test]
fn test_message_validation_query_request_too_long() {
    let config = MessageValidationConfig {
        max_query_len: 100,
        ..default_validation_config()
    };
    let validator = CompositeValidator::new(config);

    let msg = Message::QueryRequest(QueryRequest {
        query_id: 1,
        query: "x".repeat(200),
        shard_id: 0,
        embedding: None,
        timeout_ms: 5000,
    });

    let result = validator.validate(&msg, &"client".to_string());
    assert!(result.is_err(), "Excessively long query should fail");
}

#[test]
fn test_message_validation_disabled() {
    let config = MessageValidationConfig {
        enabled: false,
        ..default_validation_config()
    };
    let validator = CompositeValidator::new(config);

    // This would normally fail (zero term)
    let msg = Message::RequestVote(RequestVote {
        term: 0,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: make_sparse_vector(64, 10),
    });

    let result = validator.validate(&msg, &"node2".to_string());
    assert!(
        result.is_ok(),
        "Disabled validation should pass all messages"
    );
}

#[test]
fn test_message_validation_node_id_too_long() {
    let config = MessageValidationConfig {
        max_node_id_len: 10,
        ..default_validation_config()
    };
    let validator = CompositeValidator::new(config);

    let msg = Message::RequestVote(RequestVote {
        term: 1,
        candidate_id: "a".repeat(20),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: make_sparse_vector(64, 10),
    });

    let result = validator.validate(&msg, &"node2".to_string());
    assert!(result.is_err(), "Excessively long node_id should fail");
}

#[test]
fn test_message_validation_ping_pong() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    let ping = Message::Ping { term: 1 };
    let pong = Message::Pong { term: 1 };

    assert!(validator.validate(&ping, &"node1".to_string()).is_ok());
    assert!(validator.validate(&pong, &"node1".to_string()).is_ok());
}

// ============================================================================
// Delta Checksum Integration Tests
// ============================================================================

#[test]
fn test_delta_update_checksum_roundtrip() {
    let embedding = make_embedding(42, 64);
    let update = DeltaUpdate::full("test_key".to_string(), &embedding, 1);

    // Compute checksum
    let checksum = update.compute_checksum();

    // Checksum should be deterministic
    assert_eq!(checksum, update.compute_checksum());

    // Create update with checksum
    let update_with_checksum = update.with_checksum();
    assert!(update_with_checksum.checksum.is_some());
    assert!(update_with_checksum.verify_checksum());
}

#[test]
fn test_delta_update_legacy_verification() {
    let embedding = make_embedding(42, 64);
    let update = DeltaUpdate::full("test_key".to_string(), &embedding, 1);

    // Legacy update without checksum should pass verification
    assert!(
        update.verify_checksum(),
        "Legacy update should pass verification"
    );
}

#[test]
fn test_delta_update_corrupted_checksum() {
    let embedding = make_embedding(42, 64);
    let mut update = DeltaUpdate::full("test_key".to_string(), &embedding, 1).with_checksum();

    // Corrupt the checksum
    if let Some(ref mut checksum) = update.checksum {
        checksum[0] = checksum[0].wrapping_add(1);
    }

    // Verification should fail
    assert!(
        !update.verify_checksum(),
        "Corrupted update should fail verification"
    );
}

#[test]
fn test_delta_batch_checksum_roundtrip() {
    let mut batch = DeltaBatch::new("source_node".to_string(), 1);

    // Add multiple updates
    for i in 0..5 {
        let embedding = make_embedding(i, 64);
        let update = DeltaUpdate::full(format!("key_{}", i), &embedding, i as u64);
        batch.add(update);
    }

    // Compute batch checksum
    let checksum = batch.compute_checksum();

    // Checksum should be deterministic
    assert_eq!(checksum, batch.compute_checksum());

    // Batch without checksum should pass verification (legacy)
    assert!(batch.verify().is_ok());

    // Batch with checksum should pass verification
    let batch_with_checksum = batch.with_checksum();
    assert!(batch_with_checksum.verify().is_ok());

    // All updates should have checksums
    for update in &batch_with_checksum.updates {
        assert!(update.checksum.is_some());
    }
}

#[test]
fn test_delta_batch_serialization_with_checksums() {
    let mut batch = DeltaBatch::new("source_node".to_string(), 1);

    for i in 0..3 {
        let embedding = make_embedding(i, 64);
        let update = DeltaUpdate::full(format!("key_{}", i), &embedding, i as u64);
        batch.add(update);
    }

    let batch_with_checksum = batch.with_checksum();

    // Serialize and deserialize
    let bytes = bincode::serialize(&batch_with_checksum).expect("serialize");
    let decoded: DeltaBatch = bincode::deserialize(&bytes).expect("deserialize");

    // Checksums should survive serialization
    assert_eq!(batch_with_checksum.checksum, decoded.checksum);

    // Decoded batch should pass verification
    assert!(decoded.verify().is_ok());
}

#[test]
fn test_delta_replication_with_checksums() {
    let config = DeltaReplicationConfig::default();
    let manager = DeltaReplicationManager::new("node1".to_string(), config);

    // Queue updates
    for i in 0..5 {
        let embedding = make_embedding(i, 64);
        manager.queue_update(format!("key_{}", i), &embedding, i as u64);
    }

    // Create batch
    let batch = manager.create_batch(true).unwrap();

    // Add checksums
    let batch_with_checksum = batch.with_checksum();

    // Apply with verification
    let mut applied = 0;
    let result = manager.apply_batch(&batch_with_checksum, |_key, _emb| {
        applied += 1;
        Ok(())
    });

    assert!(result.is_ok());
    assert!(applied > 0);
}

#[test]
fn test_delta_update_different_data_different_checksums() {
    let embedding1 = make_embedding(1, 64);
    let embedding2 = make_embedding(2, 64);

    let update1 = DeltaUpdate::full("key1".to_string(), &embedding1, 1);
    let update2 = DeltaUpdate::full("key2".to_string(), &embedding2, 2);

    let checksum1 = update1.compute_checksum();
    let checksum2 = update2.compute_checksum();

    // Different data should produce different checksums
    assert_ne!(checksum1, checksum2);
}

#[test]
fn test_delta_batch_mixed_checksummed_and_legacy() {
    let mut batch = DeltaBatch::new("source".to_string(), 1);

    // Add checksummed update
    let emb1 = make_embedding(1, 64);
    batch.add(DeltaUpdate::full("key1".to_string(), &emb1, 1).with_checksum());

    // Add legacy update (no checksum)
    let emb2 = make_embedding(2, 64);
    batch.add(DeltaUpdate::full("key2".to_string(), &emb2, 2));

    // Mixed batch should pass verification
    assert!(batch.verify().is_ok());

    // After with_checksum, all should have checksums
    let batch_all_checksums = batch.with_checksum();
    for update in &batch_all_checksums.updates {
        assert!(update.checksum.is_some());
    }
}

#[test]
fn test_delta_batch_empty() {
    let batch = DeltaBatch::new("source".to_string(), 1);

    // Empty batch should have valid checksum
    let checksum = batch.compute_checksum();
    assert_ne!(checksum, [0u8; 32]);

    // Empty batch should pass verification
    assert!(batch.verify().is_ok());
}

#[test]
fn test_delta_update_checksum_includes_all_fields() {
    let embedding = make_embedding(1, 64);

    let update1 = DeltaUpdate::full("key".to_string(), &embedding, 1);
    let update2 = DeltaUpdate::full("key".to_string(), &embedding, 2); // Different version

    // Different versions should produce different checksums
    assert_ne!(update1.compute_checksum(), update2.compute_checksum());

    let update3 = DeltaUpdate::full("different_key".to_string(), &embedding, 1);

    // Different keys should produce different checksums
    assert_ne!(update1.compute_checksum(), update3.compute_checksum());
}

// ============================================================================
// Integration: Validation + Checksums Together
// ============================================================================

#[test]
fn test_security_pipeline_complete() {
    // Test that message validation and checksums work together in a realistic scenario
    let validation_config = default_validation_config();
    let validator = CompositeValidator::new(validation_config);

    // Create a valid message
    let msg = Message::RequestVote(RequestVote {
        term: 1,
        candidate_id: "node1".to_string(),
        last_log_index: 10,
        last_log_term: 1,
        state_embedding: make_sparse_vector(64, 10),
    });

    // Validate message
    assert!(validator.validate(&msg, &"node2".to_string()).is_ok());

    // Create delta batch with checksums
    let mut batch = DeltaBatch::new("node1".to_string(), 1);
    for i in 0..3 {
        let emb = make_embedding(i, 64);
        batch.add(DeltaUpdate::full(format!("entity_{}", i), &emb, i as u64));
    }
    let batch = batch.with_checksum();

    // Verify batch integrity
    assert!(batch.verify().is_ok());

    // All updates should be verifiable
    for update in &batch.updates {
        assert!(update.verify_checksum());
    }
}

#[test]
fn test_high_throughput_validation() {
    let config = default_validation_config();
    let validator = CompositeValidator::new(config);

    // Validate many messages quickly
    for i in 0..1000 {
        let msg = Message::RequestVote(RequestVote {
            term: (i % 100 + 1) as u64,
            candidate_id: format!("node{}", i % 10),
            last_log_index: i as u64,
            last_log_term: (i % 50) as u64,
            state_embedding: make_sparse_vector(64, 5),
        });

        let result = validator.validate(&msg, &format!("sender{}", i % 10));
        assert!(result.is_ok(), "Message {} should be valid", i);
    }
}

#[test]
fn test_high_throughput_checksums() {
    let mut batch = DeltaBatch::new("source".to_string(), 1);

    // Add many updates
    for i in 0..100 {
        let emb = make_embedding(i, 64);
        batch.add(DeltaUpdate::full(format!("key_{}", i), &emb, i as u64));
    }

    let batch = batch.with_checksum();

    // All should verify quickly
    for update in &batch.updates {
        assert!(update.verify_checksum());
    }

    assert!(batch.verify().is_ok());
}
