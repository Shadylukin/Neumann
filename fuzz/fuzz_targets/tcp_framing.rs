// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::tcp::{Handshake, LengthDelimitedCodec};
use tensor_chain::{AppendEntries, Message, RequestVote, RequestVoteResponse};
use tensor_store::SparseVector;

#[derive(Arbitrary, Debug)]
struct FramingInput {
    // Raw bytes to try decoding
    raw_bytes: Vec<u8>,
    // Max frame length (constrained to reasonable range)
    max_frame_length: u16,
    // Test case selection
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    DecodeRaw,
    EncodeRequestVote {
        term: u64,
        node_id: String,
        log_index: u64,
        log_term: u64,
        embedding: Vec<f32>,
    },
    EncodeVoteResponse {
        term: u64,
        voter_id: String,
        granted: bool,
    },
    EncodeAppendEntriesHeartbeat {
        term: u64,
        leader_id: String,
        prev_index: u64,
        prev_term: u64,
        commit_index: u64,
        embedding: Vec<f32>,
    },
    EncodePing {
        term: u64,
    },
    Handshake {
        node_id: String,
        capabilities: Vec<String>,
    },
}

fuzz_target!(|input: FramingInput| {
    // Ensure reasonable max frame length
    let max_frame_length = (input.max_frame_length as usize).clamp(16, 1024 * 1024);
    let codec = LengthDelimitedCodec::new(max_frame_length);

    match input.test_case {
        TestCase::DecodeRaw => {
            // Try to decode raw bytes as payload (without length prefix)
            let _ = codec.decode_payload(&input.raw_bytes);
        },

        TestCase::EncodeRequestVote {
            term,
            node_id,
            log_index,
            log_term,
            embedding,
        } => {
            // Limit sizes
            let embedding: Vec<f32> = embedding.into_iter().take(256).collect();
            let node_id: String = node_id.chars().take(64).collect();

            let msg = Message::RequestVote(RequestVote {
                term,
                candidate_id: node_id,
                last_log_index: log_index,
                last_log_term: log_term,
                state_embedding: SparseVector::from_dense(&embedding),
            });

            test_roundtrip(&codec, &msg);
        },

        TestCase::EncodeVoteResponse {
            term,
            voter_id,
            granted,
        } => {
            let voter_id: String = voter_id.chars().take(64).collect();

            let msg = Message::RequestVoteResponse(RequestVoteResponse {
                term,
                voter_id,
                vote_granted: granted,
            });

            test_roundtrip(&codec, &msg);
        },

        TestCase::EncodeAppendEntriesHeartbeat {
            term,
            leader_id,
            prev_index,
            prev_term,
            commit_index,
            embedding,
        } => {
            let embedding: Vec<f32> = embedding.into_iter().take(256).collect();
            let leader_id: String = leader_id.chars().take(64).collect();

            // Create a heartbeat (empty entries)
            let msg = Message::AppendEntries(AppendEntries {
                term,
                leader_id,
                prev_log_index: prev_index,
                prev_log_term: prev_term,
                entries: vec![], // Heartbeat has no entries
                leader_commit: commit_index,
                block_embedding: if embedding.is_empty() {
                    None
                } else {
                    Some(SparseVector::from_dense(&embedding))
                },
            });

            test_roundtrip(&codec, &msg);
        },

        TestCase::EncodePing { term } => {
            let msg = Message::Ping { term };
            test_roundtrip(&codec, &msg);
        },

        TestCase::Handshake {
            node_id,
            capabilities,
        } => {
            let node_id: String = node_id.chars().take(64).collect();
            let capabilities: Vec<String> = capabilities
                .into_iter()
                .take(10)
                .map(|s| s.chars().take(32).collect())
                .collect();

            let mut handshake = Handshake::new(node_id);
            for cap in capabilities {
                handshake = handshake.with_capability(cap);
            }

            // Encode handshake
            if let Ok(encoded) = handshake.encode() {
                // Verify length prefix
                assert!(encoded.len() >= 4);
                let length =
                    u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
                assert_eq!(length, encoded.len() - 4);

                // Decode should succeed
                let decoded: Result<Handshake, _> = bitcode::deserialize(&encoded[4..]);
                assert!(decoded.is_ok(), "Failed to decode valid handshake");
            }
        },
    }
});

fn test_roundtrip(codec: &LengthDelimitedCodec, msg: &Message) {
    // Encode message
    if let Ok(encoded) = codec.encode(msg) {
        // Should be able to decode what we encoded
        if encoded.len() > 4 {
            let payload = &encoded[4..];
            let decoded = codec.decode_payload(payload);
            assert!(decoded.is_ok(), "Failed to decode valid message");
        }
    }
}
