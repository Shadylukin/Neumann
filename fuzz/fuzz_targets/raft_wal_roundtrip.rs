// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::RaftWalEntry;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes
    if let Ok(entry) = bitcode::deserialize::<RaftWalEntry>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bitcode::serialize(&entry) {
            if let Ok(entry2) = bitcode::deserialize::<RaftWalEntry>(&reserialized) {
                // Verify roundtrip preserves data
                assert_eq!(entry, entry2, "WAL entry roundtrip mismatch");
            }
        }
    }
});
