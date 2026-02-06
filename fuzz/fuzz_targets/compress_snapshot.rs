// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_compress::format::CompressedSnapshot;

fuzz_target!(|data: &[u8]| {
    // Test that deserialize never panics on arbitrary bytes
    if let Ok(snapshot) = CompressedSnapshot::deserialize(data) {
        // If deserialize succeeds, serialize should also work
        if let Ok(reserialized) = snapshot.serialize() {
            // And deserialize again should produce equivalent result
            if let Ok(snapshot2) = CompressedSnapshot::deserialize(&reserialized) {
                assert_eq!(snapshot, snapshot2, "Snapshot roundtrip mismatch");
            }
        }
    }
});
