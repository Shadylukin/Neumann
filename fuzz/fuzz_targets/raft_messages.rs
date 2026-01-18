//! Fuzz test for Raft message serialization and handling.
//!
//! Tests that:
//! - Message serialization/deserialization is stable
//! - RaftNode handles arbitrary messages without panicking
//! - Term and index boundaries are handled correctly

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use tensor_chain::{
    network::{AppendEntries, AppendEntriesResponse, RequestVote, RequestVoteResponse},
    MemoryTransport, Message, RaftConfig, RaftNode,
};
use tensor_store::SparseVector;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Message type selector
    message_type: u8,
    /// Term value
    term: u64,
    /// Index value
    index: u64,
    /// Node ID bytes (will be converted to string)
    node_id: Vec<u8>,
    /// Whether vote was granted
    vote_granted: bool,
    /// Whether append was successful
    success: bool,
    /// State embedding values
    embedding: Vec<u8>,
    /// Number of log entries
    entry_count: u8,
}

fn make_node_id(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "node1".to_string()
    } else {
        let s: String = bytes
            .iter()
            .take(32)
            .map(|&b| {
                let c = (b % 26) + b'a';
                c as char
            })
            .collect();
        if s.is_empty() {
            "node1".to_string()
        } else {
            s
        }
    }
}

fn make_embedding(bytes: &[u8]) -> SparseVector {
    let dense: Vec<f32> = bytes.iter().take(128).map(|&b| b as f32 / 255.0).collect();
    SparseVector::from_dense(&dense)
}

fuzz_target!(|input: FuzzInput| {
    let node_id = make_node_id(&input.node_id);
    let from_id = format!("from_{}", &node_id);

    // Create various message types based on selector
    let message = match input.message_type % 6 {
        0 => {
            // RequestVote
            Message::RequestVote(RequestVote {
                term: input.term,
                candidate_id: node_id.clone(),
                last_log_index: input.index,
                last_log_term: input.term.saturating_sub(1),
                state_embedding: make_embedding(&input.embedding),
            })
        },
        1 => {
            // RequestVoteResponse
            Message::RequestVoteResponse(RequestVoteResponse {
                term: input.term,
                vote_granted: input.vote_granted,
                voter_id: node_id.clone(),
            })
        },
        2 => {
            // AppendEntries (empty)
            Message::AppendEntries(AppendEntries {
                term: input.term,
                leader_id: node_id.clone(),
                prev_log_index: input.index,
                prev_log_term: input.term.saturating_sub(1),
                entries: vec![],
                leader_commit: input.index.saturating_sub(1),
                block_embedding: if input.embedding.is_empty() {
                    None
                } else {
                    Some(make_embedding(&input.embedding))
                },
            })
        },
        3 => {
            // AppendEntriesResponse
            Message::AppendEntriesResponse(AppendEntriesResponse {
                term: input.term,
                success: input.success,
                match_index: input.index,
                follower_id: node_id.clone(),
                used_fast_path: input.success, // Use success flag as proxy
            })
        },
        4 => {
            // Ping
            Message::Ping { term: input.term }
        },
        _ => {
            // Pong
            Message::Pong { term: input.term }
        },
    };

    // Test 1: Serialization roundtrip
    let serialized = match bincode::serialize(&message) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };

    let deserialized: Message = match bincode::deserialize(&serialized) {
        Ok(m) => m,
        Err(_) => panic!("Failed to deserialize message we just serialized"),
    };

    // Verify message type preserved
    match (&message, &deserialized) {
        (Message::RequestVote(a), Message::RequestVote(b)) => {
            assert_eq!(a.term, b.term);
            assert_eq!(a.candidate_id, b.candidate_id);
        },
        (Message::RequestVoteResponse(a), Message::RequestVoteResponse(b)) => {
            assert_eq!(a.term, b.term);
            assert_eq!(a.vote_granted, b.vote_granted);
        },
        (Message::AppendEntries(a), Message::AppendEntries(b)) => {
            assert_eq!(a.term, b.term);
            assert_eq!(a.leader_id, b.leader_id);
        },
        (Message::AppendEntriesResponse(a), Message::AppendEntriesResponse(b)) => {
            assert_eq!(a.term, b.term);
            assert_eq!(a.success, b.success);
        },
        (Message::Ping { term: a }, Message::Ping { term: b }) => {
            assert_eq!(a, b);
        },
        (Message::Pong { term: a }, Message::Pong { term: b }) => {
            assert_eq!(a, b);
        },
        _ => panic!("Message type changed during serialization"),
    }

    // Test 2: RaftNode should handle messages without panicking
    let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
    let config = RaftConfig {
        election_timeout: (150, 300),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    };
    let node = RaftNode::new(
        "test_node".to_string(),
        vec![from_id.clone()],
        transport,
        config,
    );

    // Handle the message - should not panic
    let _response = node.handle_message(&from_id, &message);

    // Test 3: JSON serialization (for debugging)
    if let Ok(json) = serde_json::to_string(&message) {
        assert!(!json.is_empty());
    }
});
