#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::RaftWalEntry;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes
    if let Ok(entry) = bincode::deserialize::<RaftWalEntry>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bincode::serialize(&entry) {
            if let Ok(entry2) = bincode::deserialize::<RaftWalEntry>(&reserialized) {
                // Verify roundtrip preserves data
                assert_eq!(entry, entry2, "WAL entry roundtrip mismatch");
            }
        }
    }
});
