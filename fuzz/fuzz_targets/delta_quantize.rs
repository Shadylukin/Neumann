// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz test for DeltaUpdate serialization, checksums, and invariants.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::DeltaUpdate;

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    key: String,
    /// Values in range [0, 255] to create valid f32 values
    values: Vec<u8>,
    version: u64,
    /// Whether to test as full update or delta update
    is_full: bool,
    archetype_id: u32,
}

fuzz_target!(|input: FuzzInput| {
    // Skip empty inputs
    if input.values.is_empty() {
        return;
    }

    // Limit key length and values count
    let key = if input.key.len() > 256 {
        let end = input
            .key
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= 256)
            .last()
            .unwrap_or(0);
        input.key[..end].to_string()
    } else if input.key.is_empty() {
        "key".to_string()
    } else {
        input.key
    };

    let values: Vec<f32> = input
        .values
        .iter()
        .take(1024)
        .map(|&v| f32::from(v) / 255.0)
        .collect();

    if values.is_empty() {
        return;
    }

    // Create DeltaUpdate
    let update = if input.is_full {
        DeltaUpdate::full(key.clone(), &values, input.version)
    } else {
        let indices: Vec<u32> = (0..values.len() as u32).collect();
        DeltaUpdate {
            key: key.clone(),
            archetype_id: input.archetype_id,
            delta_indices: indices,
            delta_values: values.clone(),
            version: input.version,
            dimension: values.len(),
            checksum: None,
        }
    };

    // Test 1: Checksum computation is deterministic
    let checksum1 = update.compute_checksum();
    let checksum2 = update.compute_checksum();
    assert_eq!(checksum1, checksum2, "Checksum should be deterministic");

    // Test 2: with_checksum then verify_checksum succeeds
    let update_with_checksum = update.clone().with_checksum();
    assert!(
        update_with_checksum.verify_checksum(),
        "Checksum verification should pass after with_checksum"
    );

    // Test 3: Serialization roundtrip with bitcode
    let serialized = match bitcode::serialize(&update) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };

    let deserialized: DeltaUpdate = match bitcode::deserialize(&serialized) {
        Ok(u) => u,
        Err(_) => panic!("Failed to deserialize what we just serialized"),
    };

    assert_eq!(update.key, deserialized.key);
    assert_eq!(update.version, deserialized.version);
    assert_eq!(update.dimension, deserialized.dimension);
    assert_eq!(update.archetype_id, deserialized.archetype_id);
    assert_eq!(update.delta_values.len(), deserialized.delta_values.len());
    assert_eq!(
        update.delta_indices.len(),
        deserialized.delta_indices.len()
    );

    // Test 4: nnz is consistent
    assert!(update.nnz() <= update.delta_values.len());

    // Test 5: Memory bytes is reasonable
    assert!(update.memory_bytes() > 0);

    // Test 6: Compression ratio is positive
    let ratio = update.compression_ratio();
    assert!(ratio > 0.0, "Compression ratio should be positive");

    // Test 7: Serialization roundtrip preserves checksum
    let with_cs = update.clone().with_checksum();
    let serialized_cs = bitcode::serialize(&with_cs).unwrap();
    let deserialized_cs: DeltaUpdate = bitcode::deserialize(&serialized_cs).unwrap();
    assert!(
        deserialized_cs.verify_checksum(),
        "Checksum should survive serialization roundtrip"
    );
});
