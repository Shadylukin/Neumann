// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{
    CompositeValidator, Message, MessageValidationConfig, MessageValidator, QueryRequest,
    RequestVote, RequestVoteResponse, TxPrepareMsg,
};

#[derive(Arbitrary, Debug)]
struct ValidationInput {
    config: FuzzValidationConfig,
    message: FuzzMessage,
    from_node_id: String,
}

#[derive(Arbitrary, Debug)]
struct FuzzValidationConfig {
    enabled: bool,
    max_term: u64,
    max_shard_id: u16,
    max_tx_timeout_ms: u32,
    max_node_id_len: u8,
    max_embedding_dimension: u16,
    max_query_len: u16,
}

#[derive(Arbitrary, Debug)]
enum FuzzMessage {
    RequestVote {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
        embedding_dim: u16,
        embedding_nnz: u8,
        embedding_values: Vec<f32>,
    },
    RequestVoteResponse {
        term: u64,
        voter_id: String,
        vote_granted: bool,
    },
    TxPrepare {
        tx_id: u64,
        coordinator: String,
        shard_id: u16,
        timeout_ms: u64,
        embedding_dim: u16,
        embedding_nnz: u8,
        embedding_values: Vec<f32>,
    },
    QueryRequest {
        query_id: u64,
        query: String,
        shard_id: u16,
        timeout_ms: u64,
        has_embedding: bool,
        embedding_dim: u16,
        embedding_nnz: u8,
        embedding_values: Vec<f32>,
    },
    Ping {
        term: u64,
    },
    Pong {
        term: u64,
    },
}

fn make_sparse_vector(dim: u16, nnz: u8, values: &[f32]) -> tensor_store::SparseVector {
    use tensor_store::SparseVector;

    let dimension = (dim as usize).max(1);
    let nnz = (nnz as usize).min(dimension).min(values.len());

    if nnz == 0 {
        return SparseVector::new(dimension);
    }

    let positions: Vec<u32> = (0..nnz as u32).collect();
    let vals: Vec<f32> = values.iter().take(nnz).cloned().collect();

    SparseVector::from_parts(dimension, positions, vals)
}

fuzz_target!(|input: ValidationInput| {
    // Build validation config with clamped values
    let config = MessageValidationConfig {
        enabled: input.config.enabled,
        max_term: input.config.max_term.max(1),
        max_shard_id: (input.config.max_shard_id as usize).max(1),
        max_tx_timeout_ms: (input.config.max_tx_timeout_ms as u64).max(1),
        max_node_id_len: (input.config.max_node_id_len as usize).max(1),
        max_key_len: 4096,
        max_embedding_dimension: (input.config.max_embedding_dimension as usize).max(1),
        max_embedding_magnitude: 1e6,
        max_query_len: (input.config.max_query_len as usize).max(1),
        max_message_age_ms: 5 * 60 * 1000,
        max_blocks_per_request: 1000,
        max_snapshot_chunk_size: 10 * 1024 * 1024,
    };

    let validator = CompositeValidator::new(config);

    // Limit from_node_id length
    let from: String = input.from_node_id.chars().take(256).collect();

    // Build message based on fuzz input
    let msg = match input.message {
        FuzzMessage::RequestVote {
            term,
            candidate_id,
            last_log_index,
            last_log_term,
            embedding_dim,
            embedding_nnz,
            embedding_values,
        } => {
            let candidate_id: String = candidate_id.chars().take(256).collect();
            let embedding = make_sparse_vector(embedding_dim, embedding_nnz, &embedding_values);

            Message::RequestVote(RequestVote {
                term,
                candidate_id,
                last_log_index,
                last_log_term,
                state_embedding: embedding,
            })
        },

        FuzzMessage::RequestVoteResponse {
            term,
            voter_id,
            vote_granted,
        } => {
            let voter_id: String = voter_id.chars().take(256).collect();

            Message::RequestVoteResponse(RequestVoteResponse {
                term,
                voter_id,
                vote_granted,
            })
        },

        FuzzMessage::TxPrepare {
            tx_id,
            coordinator,
            shard_id,
            timeout_ms,
            embedding_dim,
            embedding_nnz,
            embedding_values,
        } => {
            let coordinator: String = coordinator.chars().take(256).collect();
            let embedding = make_sparse_vector(embedding_dim, embedding_nnz, &embedding_values);

            Message::TxPrepare(TxPrepareMsg {
                tx_id,
                coordinator,
                shard_id: shard_id as usize,
                operations: vec![],
                delta_embedding: embedding,
                timeout_ms,
            })
        },

        FuzzMessage::QueryRequest {
            query_id,
            query,
            shard_id,
            timeout_ms,
            has_embedding,
            embedding_dim,
            embedding_nnz,
            embedding_values,
        } => {
            let query: String = query.chars().take(1024 * 1024).collect();
            let embedding = if has_embedding {
                Some(make_sparse_vector(
                    embedding_dim,
                    embedding_nnz,
                    &embedding_values,
                ))
            } else {
                None
            };

            Message::QueryRequest(QueryRequest {
                query_id,
                query,
                shard_id: shard_id as usize,
                embedding,
                timeout_ms,
            })
        },

        FuzzMessage::Ping { term } => Message::Ping { term },

        FuzzMessage::Pong { term } => Message::Pong { term },
    };

    // Validate the message - should not panic regardless of input
    let result = validator.validate(&msg, &from);

    // Verify consistency: if validation is disabled, should always pass
    if !input.config.enabled {
        assert!(result.is_ok(), "disabled validation should always pass");
    }

    // If result is Ok, verify the message would be processable
    // If result is Err, the error should be descriptive
    match result {
        Ok(()) => {
            // Valid message - no additional checks needed
        },
        Err(e) => {
            // Error should have meaningful message
            let error_str = e.to_string();
            assert!(!error_str.is_empty(), "error message should not be empty");
        },
    }
});
