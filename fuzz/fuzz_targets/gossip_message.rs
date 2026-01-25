#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::GossipMessage;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes
    if let Ok(msg) = bitcode::deserialize::<GossipMessage>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bitcode::serialize(&msg) {
            if let Ok(msg2) = bitcode::deserialize::<GossipMessage>(&reserialized) {
                // Verify roundtrip preserves data
                assert_eq!(msg, msg2, "Gossip message roundtrip mismatch");
            }
        }
    }
});
