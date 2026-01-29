// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz test for quantized delta update serialization and operations.
//!
//! Tests that:
//! - Quantization/dequantization roundtrip preserves values within error bounds
//! - Serialization/deserialization is stable
//! - No panics on arbitrary input

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{DeltaUpdate, QuantizedDeltaUpdate};

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
        input.key[..256].to_string()
    } else if input.key.is_empty() {
        "key".to_string()
    } else {
        input.key
    };

    let values: Vec<f32> = input
        .values
        .iter()
        .take(1024) // Limit to reasonable size
        .map(|&v| v as f32 / 255.0) // Normalize to [0, 1]
        .collect();

    if values.is_empty() {
        return;
    }

    // Create DeltaUpdate
    let update = if input.is_full {
        DeltaUpdate::full(key, &values, input.version)
    } else {
        // Create a delta-style update with indices
        let indices: Vec<u32> = (0..values.len() as u32).collect();
        DeltaUpdate {
            key,
            archetype_id: input.archetype_id,
            delta_indices: indices,
            delta_values: values.clone(),
            version: input.version,
            dimension: values.len(),
            checksum: None,
        }
    };

    // Test 1: Quantization should succeed
    let quantized = match update.quantize() {
        Some(q) => q,
        None => return, // Quantization can fail for edge cases
    };

    // Test 2: Serialization roundtrip
    let serialized = match bitcode::serialize(&quantized) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };

    let deserialized: QuantizedDeltaUpdate = match bitcode::deserialize(&serialized) {
        Ok(q) => q,
        Err(_) => panic!("Failed to deserialize what we just serialized"),
    };

    // Verify key data preserved
    assert_eq!(quantized.key, deserialized.key);
    assert_eq!(quantized.version, deserialized.version);
    assert_eq!(quantized.dimension, deserialized.dimension);
    assert_eq!(quantized.archetype_id, deserialized.archetype_id);
    assert_eq!(
        quantized.quantized_values.data.len(),
        deserialized.quantized_values.data.len()
    );

    // Test 3: Dequantization should succeed
    let dequantized = quantized.dequantize();

    // Test 4: Values should be within error bounds (~1% for int8 quantization)
    assert_eq!(dequantized.delta_values.len(), update.delta_values.len());
    for (orig, deq) in update
        .delta_values
        .iter()
        .zip(dequantized.delta_values.iter())
    {
        let error = (orig - deq).abs();
        // Allow up to 2% error for int8 quantization
        assert!(
            error < 0.02 || (orig.abs() < 0.001 && deq.abs() < 0.02),
            "Quantization error too large: orig={}, deq={}, error={}",
            orig,
            deq,
            error
        );
    }

    // Test 5: Compression should be achieved for reasonable sizes
    if values.len() >= 64 {
        let orig_bytes = update.memory_bytes();
        let quant_bytes = quantized.memory_bytes();
        // Quantized should be smaller for larger vectors
        assert!(
            quant_bytes <= orig_bytes,
            "Quantized ({}) should not be larger than original ({}) for {} values",
            quant_bytes,
            orig_bytes,
            values.len()
        );
    }

    // Test 6: Memory bytes should be reasonable
    assert!(quantized.memory_bytes() > 0);
    assert!(quantized.compression_ratio() > 0.0);
});
