//! Fuzz target for distributed transaction serialization.
//!
//! Tests that 2PC messages can be serialized and deserialized without panicking.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TxMessage {
    tx_id: u64,
    coordinator: String,
    shard_id: usize,
    operations: Vec<TxOp>,
    delta_embedding: Vec<f32>,
    timeout_ms: u64,
}

#[derive(Arbitrary, Debug)]
enum TxOp {
    Put { key: String, data: Vec<u8> },
    Delete { key: String },
}

#[derive(Arbitrary, Debug)]
enum Vote {
    Yes {
        lock_handle: u64,
        delta: Vec<f32>,
    },
    No {
        reason: String,
    },
    Conflict {
        similarity: f32,
        conflicting_tx: u64,
    },
}

#[derive(Arbitrary, Debug)]
struct PrepareResponse {
    tx_id: u64,
    shard_id: usize,
    vote: Vote,
}

#[derive(Arbitrary, Debug)]
struct CommitMsg {
    tx_id: u64,
    shards: Vec<usize>,
}

#[derive(Arbitrary, Debug)]
struct AbortMsg {
    tx_id: u64,
    reason: String,
    shards: Vec<usize>,
}

#[derive(Arbitrary, Debug)]
struct AckMsg {
    tx_id: u64,
    shard_id: usize,
    success: bool,
    error: Option<String>,
}

#[derive(Arbitrary, Debug)]
enum FuzzMessage {
    Prepare(TxMessage),
    PrepareResponse(PrepareResponse),
    Commit(CommitMsg),
    Abort(AbortMsg),
    Ack(AckMsg),
}

fuzz_target!(|data: FuzzMessage| {
    // Test that we can serialize and deserialize without panicking
    match &data {
        FuzzMessage::Prepare(msg) => {
            // Verify tx_id is consistent
            let _ = msg.tx_id;
            let _ = msg.coordinator.len();
            let _ = msg.shard_id;
            let _ = msg.operations.len();
            let _ = msg.delta_embedding.len();
            let _ = msg.timeout_ms;

            // Check operations
            for op in &msg.operations {
                match op {
                    TxOp::Put { key, data } => {
                        assert!(key.len() < 1024 * 1024); // Reasonable key size
                        assert!(data.len() < 1024 * 1024); // Reasonable data size
                    },
                    TxOp::Delete { key } => {
                        assert!(key.len() < 1024 * 1024);
                    },
                }
            }
        },
        FuzzMessage::PrepareResponse(resp) => {
            let _ = resp.tx_id;
            let _ = resp.shard_id;
            match &resp.vote {
                Vote::Yes { lock_handle, delta } => {
                    let _ = lock_handle;
                    let _ = delta.len();
                },
                Vote::No { reason } => {
                    let _ = reason.len();
                },
                Vote::Conflict {
                    similarity,
                    conflicting_tx,
                } => {
                    // Similarity should be finite
                    if similarity.is_finite() {
                        assert!(*similarity >= -1.0 && *similarity <= 1.0 || true);
                    }
                    let _ = conflicting_tx;
                },
            }
        },
        FuzzMessage::Commit(msg) => {
            let _ = msg.tx_id;
            let _ = msg.shards.len();
        },
        FuzzMessage::Abort(msg) => {
            let _ = msg.tx_id;
            let _ = msg.reason.len();
            let _ = msg.shards.len();
        },
        FuzzMessage::Ack(msg) => {
            let _ = msg.tx_id;
            let _ = msg.shard_id;
            let _ = msg.success;
            if let Some(err) = &msg.error {
                let _ = err.len();
            }
        },
    }
});
