// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::{CompositeValidator, Message, MessageValidationConfig, MessageValidator, SnapshotRequest};

fuzz_target!(|data: (String, u64, u64, String)| {
    let (requester_id, offset, chunk_size, sender) = data;

    // Create validation config with reasonable limits
    let config = MessageValidationConfig {
        max_snapshot_chunk_size: 10 * 1024 * 1024, // 10 MB
        max_node_id_len: 256,
        ..MessageValidationConfig::default()
    };
    let validator = CompositeValidator::new(config);

    // Create SnapshotRequest with fuzzed data
    let msg = Message::SnapshotRequest(SnapshotRequest {
        requester_id,
        offset,
        chunk_size,
    });

    // Validation should never panic
    let result = validator.validate(&msg, &sender);

    // Invariant: zero chunk size should always fail
    if chunk_size == 0 {
        assert!(result.is_err(), "Zero chunk size should be rejected");
    }

    // Invariant: chunk size > limit should always fail
    if chunk_size > 10 * 1024 * 1024 {
        assert!(result.is_err(), "Oversized chunk should be rejected");
    }
});
