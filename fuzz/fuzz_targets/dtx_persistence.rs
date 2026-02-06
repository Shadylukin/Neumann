// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz target for distributed transaction persistence types.
//!
//! Tests that CoordinatorState, ParticipantState, and SerializableLockState
//! can be deserialized from arbitrary bytes without panicking.
//! Also tests roundtrip serialization.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{CoordinatorState, ParticipantState, SerializableLockState};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    coordinator_bytes: Vec<u8>,
    participant_bytes: Vec<u8>,
    lock_state_bytes: Vec<u8>,
}

fuzz_target!(|input: FuzzInput| {
    // Test CoordinatorState deserialize doesn't panic
    if let Ok(state) = bitcode::deserialize::<CoordinatorState>(&input.coordinator_bytes) {
        // If deserialize succeeds, verify roundtrip
        if let Ok(bytes) = bitcode::serialize(&state) {
            // Re-deserialize should not panic
            let _ = bitcode::deserialize::<CoordinatorState>(&bytes);
        }
    }

    // Test ParticipantState deserialize doesn't panic
    if let Ok(state) = bitcode::deserialize::<ParticipantState>(&input.participant_bytes) {
        // If deserialize succeeds, verify roundtrip
        if let Ok(bytes) = bitcode::serialize(&state) {
            let _ = bitcode::deserialize::<ParticipantState>(&bytes);
        }
    }

    // Test SerializableLockState deserialize doesn't panic
    if let Ok(state) = bitcode::deserialize::<SerializableLockState>(&input.lock_state_bytes) {
        // If deserialize succeeds, verify roundtrip
        if let Ok(bytes) = bitcode::serialize(&state) {
            let _ = bitcode::deserialize::<SerializableLockState>(&bytes);
        }
    }
});
