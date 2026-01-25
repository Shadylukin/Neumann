#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{Message, PreVote, PreVoteResponse, TimeoutNow};
use tensor_store::SparseVector;

#[derive(Arbitrary, Debug)]
enum TestCase {
    PreVoteSerialization {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
        embedding: Vec<f32>,
    },
    PreVoteResponseSerialization {
        term: u64,
        vote_granted: bool,
        voter_id: String,
    },
    TimeoutNowSerialization {
        term: u64,
        leader_id: String,
    },
    DeserializeArbitrary {
        bytes: Vec<u8>,
    },
    MessageVariantRoundtrip {
        variant: MessageVariant,
    },
}

#[derive(Arbitrary, Debug)]
enum MessageVariant {
    PreVote {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
        embedding_dim: u16,
    },
    PreVoteResponse {
        term: u64,
        vote_granted: bool,
        voter_id: String,
    },
    TimeoutNow {
        term: u64,
        leader_id: String,
    },
}

fuzz_target!(|test_case: TestCase| {
    match test_case {
        TestCase::PreVoteSerialization {
            term,
            candidate_id,
            last_log_index,
            last_log_term,
            embedding,
        } => {
            // Limit sizes
            let embedding: Vec<f32> = embedding.into_iter().take(256).collect();
            let candidate_id: String = candidate_id.chars().take(64).collect();

            let pv = PreVote {
                term,
                candidate_id: candidate_id.clone(),
                last_log_index,
                last_log_term,
                state_embedding: SparseVector::from_dense(&embedding),
            };

            // Test bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&pv) {
                if let Ok(restored) = bitcode::deserialize::<PreVote>(&bytes) {
                    assert_eq!(restored.term, term);
                    assert_eq!(restored.candidate_id, candidate_id);
                    assert_eq!(restored.last_log_index, last_log_index);
                    assert_eq!(restored.last_log_term, last_log_term);
                }
            }

            // Test as Message variant
            let msg = Message::PreVote(pv);
            if let Ok(bytes) = bitcode::serialize(&msg) {
                let _ = bitcode::deserialize::<Message>(&bytes);
            }
        },

        TestCase::PreVoteResponseSerialization {
            term,
            vote_granted,
            voter_id,
        } => {
            let voter_id: String = voter_id.chars().take(64).collect();

            let pvr = PreVoteResponse {
                term,
                vote_granted,
                voter_id: voter_id.clone(),
            };

            // Test bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&pvr) {
                if let Ok(restored) = bitcode::deserialize::<PreVoteResponse>(&bytes) {
                    assert_eq!(restored.term, term);
                    assert_eq!(restored.vote_granted, vote_granted);
                    assert_eq!(restored.voter_id, voter_id);
                }
            }

            // Test as Message variant
            let msg = Message::PreVoteResponse(pvr);
            if let Ok(bytes) = bitcode::serialize(&msg) {
                let _ = bitcode::deserialize::<Message>(&bytes);
            }
        },

        TestCase::TimeoutNowSerialization { term, leader_id } => {
            let leader_id: String = leader_id.chars().take(64).collect();

            let tn = TimeoutNow {
                term,
                leader_id: leader_id.clone(),
            };

            // Test bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&tn) {
                if let Ok(restored) = bitcode::deserialize::<TimeoutNow>(&bytes) {
                    assert_eq!(restored.term, term);
                    assert_eq!(restored.leader_id, leader_id);
                }
            }

            // Test as Message variant
            let msg = Message::TimeoutNow(tn);
            if let Ok(bytes) = bitcode::serialize(&msg) {
                let _ = bitcode::deserialize::<Message>(&bytes);
            }
        },

        TestCase::DeserializeArbitrary { bytes } => {
            // Try to deserialize arbitrary bytes as various types
            // Should not panic even on invalid input
            let _ = bitcode::deserialize::<PreVote>(&bytes);
            let _ = bitcode::deserialize::<PreVoteResponse>(&bytes);
            let _ = bitcode::deserialize::<TimeoutNow>(&bytes);
            let _ = bitcode::deserialize::<Message>(&bytes);
        },

        TestCase::MessageVariantRoundtrip { variant } => {
            let msg = match variant {
                MessageVariant::PreVote {
                    term,
                    candidate_id,
                    last_log_index,
                    last_log_term,
                    embedding_dim,
                } => {
                    let candidate_id: String = candidate_id.chars().take(64).collect();
                    let dim = (embedding_dim as usize).min(256);

                    Message::PreVote(PreVote {
                        term,
                        candidate_id,
                        last_log_index,
                        last_log_term,
                        state_embedding: SparseVector::new(dim),
                    })
                },
                MessageVariant::PreVoteResponse {
                    term,
                    vote_granted,
                    voter_id,
                } => {
                    let voter_id: String = voter_id.chars().take(64).collect();

                    Message::PreVoteResponse(PreVoteResponse {
                        term,
                        vote_granted,
                        voter_id,
                    })
                },
                MessageVariant::TimeoutNow { term, leader_id } => {
                    let leader_id: String = leader_id.chars().take(64).collect();

                    Message::TimeoutNow(TimeoutNow { term, leader_id })
                },
            };

            // Test roundtrip
            if let Ok(bytes) = bitcode::serialize(&msg) {
                if let Ok(restored) = bitcode::deserialize::<Message>(&bytes) {
                    // Verify the variant matches
                    match (&msg, &restored) {
                        (Message::PreVote(_), Message::PreVote(_)) => {},
                        (Message::PreVoteResponse(_), Message::PreVoteResponse(_)) => {},
                        (Message::TimeoutNow(_), Message::TimeoutNow(_)) => {},
                        _ => panic!("Message variant mismatch after roundtrip"),
                    }
                }
            }
        },
    }
});
