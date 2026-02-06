// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz test for TxAbortMsg serialization roundtrip.
//!
//! Tests that TxAbortMsg serialization is robust:
//! - Valid messages serialize and deserialize correctly
//! - Invalid bytes don't cause panics

#![no_main]
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
struct TxAbortMsg {
    tx_id: u64,
    reason: String,
    shards: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
struct TxAckMsg {
    tx_id: u64,
    shard_id: usize,
    success: bool,
    error: Option<String>,
}

fuzz_target!(|data: &[u8]| {
    // Try to deserialize as TxAbortMsg
    if let Ok(msg) = bitcode::deserialize::<TxAbortMsg>(data) {
        // If valid, re-serialize and verify roundtrip
        let serialized = bitcode::serialize(&msg).expect("Serialization should succeed");
        let deserialized: TxAbortMsg =
            bitcode::deserialize(&serialized).expect("Deserialization should succeed");
        assert_eq!(msg, deserialized, "Roundtrip should preserve data");
    }

    // Try to deserialize as TxAckMsg
    if let Ok(msg) = bitcode::deserialize::<TxAckMsg>(data) {
        // If valid, re-serialize and verify roundtrip
        let serialized = bitcode::serialize(&msg).expect("Serialization should succeed");
        let deserialized: TxAckMsg =
            bitcode::deserialize(&serialized).expect("Deserialization should succeed");
        assert_eq!(msg, deserialized, "Roundtrip should preserve data");
    }

    // Test creating valid messages from fuzz data
    if data.len() >= 16 {
        let tx_id = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let shard_count = (data[8] % 10) as usize;

        let abort_msg = TxAbortMsg {
            tx_id,
            reason: String::from_utf8_lossy(&data[9..data.len().min(50)]).to_string(),
            shards: (0..shard_count).collect(),
        };

        // Serialize and deserialize
        let serialized = bitcode::serialize(&abort_msg).expect("Serialization should succeed");
        let deserialized: TxAbortMsg =
            bitcode::deserialize(&serialized).expect("Deserialization should succeed");
        assert_eq!(abort_msg, deserialized);
    }
});
