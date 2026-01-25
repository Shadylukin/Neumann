#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{DeltaBatch, DeltaUpdate};

#[derive(Arbitrary, Debug)]
struct ChecksumInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    /// Test single update checksum computation and verification
    SingleUpdate {
        key: String,
        archetype_id: u32,
        delta_indices: Vec<u32>,
        delta_values: Vec<f32>,
        version: u64,
        dimension: u16,
    },
    /// Test batch checksum with multiple updates
    BatchChecksum {
        source: String,
        sequence: u64,
        updates: Vec<FuzzUpdate>,
        is_final: bool,
    },
    /// Test checksum verification with corruption
    CorruptedUpdate {
        key: String,
        archetype_id: u32,
        delta_indices: Vec<u32>,
        delta_values: Vec<f32>,
        version: u64,
        dimension: u16,
        corrupt_byte: u8,
    },
    /// Test checksum serialization roundtrip
    SerializationRoundtrip {
        key: String,
        archetype_id: u32,
        delta_indices: Vec<u32>,
        delta_values: Vec<f32>,
        version: u64,
        dimension: u16,
    },
    /// Test batch with mixed checksummed and legacy updates
    MixedBatch {
        source: String,
        sequence: u64,
        checksummed_updates: Vec<FuzzUpdate>,
        legacy_updates: Vec<FuzzUpdate>,
    },
}

#[derive(Arbitrary, Debug, Clone)]
struct FuzzUpdate {
    key: String,
    archetype_id: u32,
    delta_indices: Vec<u32>,
    delta_values: Vec<f32>,
    version: u64,
    dimension: u16,
}

impl FuzzUpdate {
    fn to_delta_update(&self) -> DeltaUpdate {
        let key: String = self.key.chars().take(256).collect();
        let dimension = (self.dimension as usize).max(1);

        // Ensure indices are valid and sorted
        let mut indices: Vec<u32> = self
            .delta_indices
            .iter()
            .take(dimension.min(100))
            .map(|&i| i % dimension as u32)
            .collect();
        indices.sort();
        indices.dedup();

        // Match values to indices
        let values: Vec<f32> = self
            .delta_values
            .iter()
            .take(indices.len())
            .cloned()
            .collect();

        DeltaUpdate {
            key,
            archetype_id: self.archetype_id,
            delta_indices: indices,
            delta_values: values,
            version: self.version,
            dimension,
            checksum: None,
        }
    }
}

fuzz_target!(|input: ChecksumInput| {
    match input.test_case {
        TestCase::SingleUpdate {
            key,
            archetype_id,
            delta_indices,
            delta_values,
            version,
            dimension,
        } => {
            let fuzz_update = FuzzUpdate {
                key,
                archetype_id,
                delta_indices,
                delta_values,
                version,
                dimension,
            };

            let update = fuzz_update.to_delta_update();

            // Compute checksum
            let checksum = update.compute_checksum();

            // Checksum should be deterministic
            assert_eq!(
                checksum,
                update.compute_checksum(),
                "checksum should be deterministic"
            );

            // Update without checksum should pass verification (legacy)
            assert!(
                update.verify_checksum(),
                "legacy update should pass verification"
            );

            // Update with checksum should pass verification
            let update_with_checksum = update.with_checksum();
            assert!(
                update_with_checksum.verify_checksum(),
                "checksummed update should pass verification"
            );

            // Checksum should match
            assert_eq!(
                update_with_checksum.checksum,
                Some(checksum),
                "checksum should match"
            );
        },

        TestCase::BatchChecksum {
            source,
            sequence,
            updates,
            is_final,
        } => {
            let source: String = source.chars().take(256).collect();
            let updates: Vec<DeltaUpdate> = updates
                .iter()
                .take(10)
                .map(|u| u.to_delta_update())
                .collect();

            let mut batch = DeltaBatch::new(source.clone(), sequence);
            for update in updates {
                batch.add(update);
            }
            if is_final {
                batch = batch.finalize();
            }

            // Compute batch checksum
            let checksum = batch.compute_checksum();

            // Checksum should be deterministic
            assert_eq!(
                checksum,
                batch.compute_checksum(),
                "batch checksum should be deterministic"
            );

            // Batch without checksum should pass verification (legacy)
            assert!(batch.verify().is_ok(), "legacy batch should pass verification");

            // Batch with checksum should pass verification
            let batch_with_checksum = batch.with_checksum();
            assert!(
                batch_with_checksum.verify().is_ok(),
                "checksummed batch should pass verification"
            );

            // All updates should now have checksums
            for update in &batch_with_checksum.updates {
                assert!(
                    update.checksum.is_some(),
                    "all updates should have checksums"
                );
            }
        },

        TestCase::CorruptedUpdate {
            key,
            archetype_id,
            delta_indices,
            delta_values,
            version,
            dimension,
            corrupt_byte,
        } => {
            let fuzz_update = FuzzUpdate {
                key,
                archetype_id,
                delta_indices,
                delta_values,
                version,
                dimension,
            };

            let mut update = fuzz_update.to_delta_update().with_checksum();

            // Corrupt the checksum
            if let Some(ref mut checksum) = update.checksum {
                let idx = (corrupt_byte as usize) % 32;
                checksum[idx] = checksum[idx].wrapping_add(1);
            }

            // Verification should fail
            assert!(
                !update.verify_checksum(),
                "corrupted update should fail verification"
            );
        },

        TestCase::SerializationRoundtrip {
            key,
            archetype_id,
            delta_indices,
            delta_values,
            version,
            dimension,
        } => {
            let fuzz_update = FuzzUpdate {
                key,
                archetype_id,
                delta_indices,
                delta_values,
                version,
                dimension,
            };

            let update = fuzz_update.to_delta_update().with_checksum();

            // Serialize and deserialize
            let bytes = bitcode::serialize(&update).expect("serialization should succeed");
            let decoded: DeltaUpdate =
                bitcode::deserialize(&bytes).expect("deserialization should succeed");

            // Checksum should survive serialization
            assert_eq!(
                update.checksum, decoded.checksum,
                "checksum should survive serialization"
            );

            // Decoded update should pass verification
            assert!(
                decoded.verify_checksum(),
                "decoded update should pass verification"
            );

            // Key and values should match
            assert_eq!(update.key, decoded.key, "key should match");
            assert_eq!(update.version, decoded.version, "version should match");
        },

        TestCase::MixedBatch {
            source,
            sequence,
            checksummed_updates,
            legacy_updates,
        } => {
            let source: String = source.chars().take(256).collect();

            let mut batch = DeltaBatch::new(source.clone(), sequence);

            // Add checksummed updates
            for u in checksummed_updates.iter().take(5) {
                batch.add(u.to_delta_update().with_checksum());
            }

            // Add legacy updates (no checksum)
            for u in legacy_updates.iter().take(5) {
                batch.add(u.to_delta_update());
            }

            // Mixed batch should pass verification (legacy updates are accepted)
            assert!(
                batch.verify().is_ok(),
                "mixed batch should pass verification"
            );

            // After calling with_checksum, all updates should have checksums
            let batch_with_checksum = batch.with_checksum();
            for update in &batch_with_checksum.updates {
                assert!(
                    update.checksum.is_some(),
                    "all updates should have checksums after with_checksum"
                );
            }
        },
    }
});
