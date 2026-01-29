// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_store::ArchetypeRegistry;

fuzz_target!(|data: &[u8]| {
    // Test bincode deserialize on arbitrary bytes
    if let Ok(registry) = bitcode::deserialize::<ArchetypeRegistry>(data) {
        // If deserialize succeeds, serialize should work
        if let Ok(reserialized) = bitcode::serialize(&registry) {
            if let Ok(registry2) = bitcode::deserialize::<ArchetypeRegistry>(&reserialized) {
                // Verify key fields match
                assert_eq!(registry.len(), registry2.len(), "Registry length mismatch");

                // Verify all archetypes match
                for i in 0..registry.len() {
                    assert_eq!(
                        registry.get(i),
                        registry2.get(i),
                        "Archetype {} mismatch",
                        i
                    );
                    assert_eq!(
                        registry.magnitude_sq(i),
                        registry2.magnitude_sq(i),
                        "Magnitude squared {} mismatch",
                        i
                    );
                }
            }
        }
    }
});
