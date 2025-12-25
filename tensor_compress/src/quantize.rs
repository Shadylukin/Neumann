//! Vector quantization for f32 embeddings.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantizationError {
    #[error("empty vector")]
    EmptyVector,
    #[error("invalid binary data length: expected {expected}, got {actual}")]
    InvalidBinaryLength { expected: usize, actual: usize },
}

/// Quantized int8 vector with dequantization parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedInt8 {
    pub data: Vec<i8>,
    pub min: f32,
    pub scale: f32,
}

/// Binary quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantizedBinary {
    pub data: Vec<u8>,
    pub len: usize,
}

/// Quantize f32 vector to int8 using min-max scaling.
///
/// # Errors
/// Returns `EmptyVector` if the input is empty.
#[allow(clippy::cast_possible_truncation)]
pub fn quantize_int8(vector: &[f32]) -> Result<QuantizedInt8, QuantizationError> {
    if vector.is_empty() {
        return Err(QuantizationError::EmptyVector);
    }

    let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    let scale = if range == 0.0 { 1.0 } else { range / 255.0 };

    let data: Vec<i8> = vector
        .iter()
        .map(|&v| {
            let normalized = (v - min) / scale;
            (normalized - 128.0).round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    Ok(QuantizedInt8 { data, min, scale })
}

#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn dequantize_int8(quantized: &QuantizedInt8) -> Vec<f32> {
    quantized
        .data
        .iter()
        .map(|&q| {
            (f64::from(q) + 128.0).mul_add(f64::from(quantized.scale), f64::from(quantized.min))
                as f32
        })
        .collect()
}

/// Binary quantization: each f32 becomes 1 bit (positive = 1, non-positive = 0).
#[must_use]
pub fn quantize_binary(vector: &[f32]) -> QuantizedBinary {
    let len = vector.len();
    let byte_len = len.div_ceil(8);
    let mut data = vec![0u8; byte_len];

    for (i, &v) in vector.iter().enumerate() {
        if v > 0.0 {
            data[i / 8] |= 1 << (i % 8);
        }
    }

    QuantizedBinary { data, len }
}

/// Dequantize binary back to f32 (1.0 for set bits, -1.0 for unset).
///
/// # Errors
/// Returns `InvalidBinaryLength` if data length doesn't match expected.
pub fn dequantize_binary(quantized: &QuantizedBinary) -> Result<Vec<f32>, QuantizationError> {
    let expected_bytes = quantized.len.div_ceil(8);
    if quantized.data.len() != expected_bytes {
        return Err(QuantizationError::InvalidBinaryLength {
            expected: expected_bytes,
            actual: quantized.data.len(),
        });
    }

    let mut result = Vec::with_capacity(quantized.len);
    for i in 0..quantized.len {
        let bit = (quantized.data[i / 8] >> (i % 8)) & 1;
        result.push(if bit == 1 { 1.0 } else { -1.0 });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_int8_roundtrip() {
        let original = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantized = quantize_int8(&original).unwrap();
        let restored = dequantize_int8(&quantized);

        for (orig, rest) in original.iter().zip(&restored) {
            assert!(
                (orig - rest).abs() < 0.01,
                "Error too large: {orig} vs {rest}"
            );
        }
    }

    #[test]
    fn test_quantize_int8_negative() {
        let original = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let quantized = quantize_int8(&original).unwrap();
        let restored = dequantize_int8(&quantized);

        for (orig, rest) in original.iter().zip(&restored) {
            assert!(
                (orig - rest).abs() < 0.01,
                "Error too large: {orig} vs {rest}"
            );
        }
    }

    #[test]
    fn test_quantize_int8_uniform() {
        let original = vec![0.5, 0.5, 0.5, 0.5];
        let quantized = quantize_int8(&original).unwrap();
        let restored = dequantize_int8(&quantized);

        for (orig, rest) in original.iter().zip(&restored) {
            assert!(
                (orig - rest).abs() < 0.01,
                "Error too large: {orig} vs {rest}"
            );
        }
    }

    #[test]
    fn test_quantize_int8_empty() {
        let result = quantize_int8(&[]);
        assert!(matches!(result, Err(QuantizationError::EmptyVector)));
    }

    #[test]
    fn test_quantize_int8_single() {
        let original = vec![42.0];
        let quantized = quantize_int8(&original).unwrap();
        let restored = dequantize_int8(&quantized);
        assert!((original[0] - restored[0]).abs() < 0.01);
    }

    #[test]
    fn test_quantize_int8_compression_ratio() {
        let original: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let quantized = quantize_int8(&original).unwrap();

        let original_bytes = original.len() * 4;
        let quantized_bytes = quantized.data.len() + 8;
        let ratio = original_bytes as f64 / quantized_bytes as f64;

        assert!(ratio > 3.9, "Expected ~4x compression, got {ratio:.2}x");
    }

    #[test]
    fn test_quantize_binary_roundtrip() {
        let original = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 1.0];
        let quantized = quantize_binary(&original);
        let restored = dequantize_binary(&quantized).unwrap();

        assert_eq!(restored.len(), original.len());
        for (orig, rest) in original.iter().zip(&restored) {
            let expected = if *orig > 0.0 { 1.0 } else { -1.0 };
            assert_eq!(*rest, expected);
        }
    }

    #[test]
    fn test_quantize_binary_compression_ratio() {
        let original: Vec<f32> = (0..1024)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let quantized = quantize_binary(&original);

        let original_bytes = original.len() * 4;
        let quantized_bytes = quantized.data.len() + 8;
        let ratio = original_bytes as f64 / quantized_bytes as f64;

        assert!(ratio > 30.0, "Expected ~32x compression, got {ratio:.2}x");
    }

    #[test]
    fn test_quantize_binary_odd_length() {
        let original = vec![1.0, -1.0, 0.5, -0.5, 0.1];
        let quantized = quantize_binary(&original);
        let restored = dequantize_binary(&quantized).unwrap();
        assert_eq!(restored.len(), 5);
    }

    #[test]
    fn test_quantize_binary_empty() {
        let quantized = quantize_binary(&[]);
        let restored = dequantize_binary(&quantized).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_quantized_int8_serialize() {
        let original = vec![0.1, 0.2, 0.3];
        let quantized = quantize_int8(&original).unwrap();
        let bytes = bincode::serialize(&quantized).unwrap();
        let decoded: QuantizedInt8 = bincode::deserialize(&bytes).unwrap();
        assert_eq!(quantized, decoded);
    }

    #[test]
    fn test_quantized_binary_serialize() {
        let original = vec![1.0, -1.0, 0.5];
        let quantized = quantize_binary(&original);
        let bytes = bincode::serialize(&quantized).unwrap();
        let decoded: QuantizedBinary = bincode::deserialize(&bytes).unwrap();
        assert_eq!(quantized, decoded);
    }

    #[test]
    fn test_dequantize_binary_invalid_length() {
        let invalid = QuantizedBinary {
            data: vec![0xFF],
            len: 100,
        };
        let result = dequantize_binary(&invalid);
        assert!(matches!(
            result,
            Err(QuantizationError::InvalidBinaryLength { .. })
        ));
    }
}
