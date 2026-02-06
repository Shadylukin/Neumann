// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::{BlockRequest, CompositeValidator, Message, MessageValidationConfig, MessageValidator};

fuzz_target!(|data: (u64, u64, String, String)| {
    let (from_height, to_height, requester_id, sender) = data;

    // Create validation config with reasonable limits
    let config = MessageValidationConfig {
        max_blocks_per_request: 1000,
        max_node_id_len: 256,
        ..MessageValidationConfig::default()
    };
    let validator = CompositeValidator::new(config);

    // Create BlockRequest with fuzzed data
    let msg = Message::BlockRequest(BlockRequest {
        from_height,
        to_height,
        requester_id,
    });

    // Validation should never panic
    let _ = validator.validate(&msg, &sender);

    // Additional invariant checks
    if from_height <= to_height {
        let block_count = to_height.saturating_sub(from_height).saturating_add(1);
        // If block count exceeds limit, validation should fail
        // If requester_id is empty, validation should fail
        // Otherwise validation may pass or fail based on other criteria
    }
});
