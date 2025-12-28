#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_checkpoint::CheckpointState;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes
    if let Ok(state) = bincode::deserialize::<CheckpointState>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bincode::serialize(&state) {
            if let Ok(state2) = bincode::deserialize::<CheckpointState>(&reserialized) {
                // Verify key fields match
                assert_eq!(state.id, state2.id, "State ID mismatch");
                assert_eq!(state.name, state2.name, "State name mismatch");
            }
        }
    }
});
