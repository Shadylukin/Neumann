#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_store::WalEntry;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes for WalEntry
    if let Ok(entry) = bitcode::deserialize::<WalEntry>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bitcode::serialize(&entry) {
            if let Ok(entry2) = bitcode::deserialize::<WalEntry>(&reserialized) {
                // Verify roundtrip equality
                assert_eq!(entry, entry2, "WalEntry roundtrip mismatch");
            }
        }
    }
});
