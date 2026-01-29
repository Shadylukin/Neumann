// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_compress::incremental::DeltaSnapshot;

fuzz_target!(|data: &[u8]| {
    // Test that DeltaSnapshot::deserialize never panics on arbitrary bytes
    if let Ok(delta) = DeltaSnapshot::deserialize(data) {
        // If deserialize succeeds, serialize should also work
        if let Ok(reserialized) = delta.serialize() {
            // Roundtrip should produce equivalent result
            if let Ok(delta2) = DeltaSnapshot::deserialize(&reserialized) {
                assert_eq!(
                    delta.header.base_id, delta2.header.base_id,
                    "Delta roundtrip base_id mismatch"
                );
                assert_eq!(
                    delta.header.sequence_range, delta2.header.sequence_range,
                    "Delta roundtrip sequence_range mismatch"
                );
                assert_eq!(
                    delta.entries.len(),
                    delta2.entries.len(),
                    "Delta roundtrip entries length mismatch"
                );
            }
        }
    }
});
