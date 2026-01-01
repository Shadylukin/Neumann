//! Tensor-native compression library.
//!
//! Exploits the mathematical structure of high-dimensional embeddings using
//! Tensor Train decomposition, achieving 10-20x compression for 4096+ dimensions.
//!
//! # Compression Methods
//!
//! - **Tensor Train (TT)**: Decomposes vectors into products of smaller tensors (recommended)
//! - **Delta + varint**: Lossless compression for sorted ID sequences
//! - **Run-length encoding**: Lossless compression for repeated values
//! - **Legacy quantization**: int8/binary for backward compatibility

mod delta;
pub mod decompose;
pub mod incremental;
mod rle;
pub mod streaming;
pub mod tensor_train;

pub mod format;

// Legacy module (deprecated, kept for backward compatibility)
mod quantize;

pub use delta::{
    compress_ids, decompress_ids, delta_decode, delta_encode, varint_decode, varint_encode,
};
pub use decompose::{DecomposeError, Matrix, SvdResult, TensorView};
pub use rle::{rle_decode, rle_encode, RleEncoded};
pub use tensor_train::{
    tt_cosine_similarity, tt_decompose, tt_decompose_batch, tt_dot_product,
    tt_euclidean_distance, tt_norm, tt_reconstruct, tt_scale, TTConfig, TTCore, TTError,
    TTVector,
};

// Legacy exports (deprecated)
#[deprecated(since = "0.2.0", note = "Use TensorMode::TensorTrain instead")]
pub use quantize::{
    dequantize_binary, dequantize_int8, quantize_binary, quantize_int8, QuantizationError,
    QuantizedBinary, QuantizedInt8,
};

use serde::{Deserialize, Serialize};

/// Tensor compression mode for vectors and embeddings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorMode {
    /// Tensor Train decomposition (recommended for 1024+ dimensions).
    /// Achieves 10-20x compression with <1% error.
    TensorTrain(TTConfig),
    /// Legacy int8 quantization (4x compression, ~1% error).
    #[deprecated(since = "0.2.0", note = "Use TensorTrain for better compression")]
    LegacyInt8,
    /// Legacy binary quantization (32x compression, lossy).
    #[deprecated(since = "0.2.0", note = "Use TensorTrain for better compression")]
    LegacyBinary,
}

impl TensorMode {
    /// Create TT mode optimized for the given dimension.
    #[must_use]
    pub fn tensor_train(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::for_dim(dim))
    }

    /// High compression TT preset.
    #[must_use]
    pub fn high_compression(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::high_compression(dim))
    }

    /// High accuracy TT preset.
    #[must_use]
    pub fn high_accuracy(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::high_accuracy(dim))
    }
}

/// Compression configuration for snapshots.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompressionConfig {
    /// Tensor compression mode for vectors/embeddings.
    pub tensor_mode: Option<TensorMode>,
    /// Delta encoding for sorted ID lists.
    pub delta_encoding: bool,
    /// RLE for repeated scalar values.
    pub rle_encoding: bool,
}

impl CompressionConfig {
    /// High compression preset for 4096+ dimension embeddings.
    #[must_use]
    pub fn high_compression() -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::high_compression(4096))),
            delta_encoding: true,
            rle_encoding: true,
        }
    }

    /// Balanced compression preset.
    #[must_use]
    pub fn balanced(dim: usize) -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::for_dim(dim))),
            delta_encoding: true,
            rle_encoding: true,
        }
    }

    /// High accuracy preset (lower compression).
    #[must_use]
    pub fn high_accuracy(dim: usize) -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::high_accuracy(dim))),
            delta_encoding: true,
            rle_encoding: true,
        }
    }
}

/// Vector quantization mode (legacy, use `TensorMode` instead).
#[deprecated(since = "0.2.0", note = "Use TensorMode instead")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantMode {
    /// Scalar quantization to int8. 4x size reduction, ~1% max error.
    Int8,
    /// Binary quantization (sign bit only). 32x reduction, lossy.
    Binary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(config.tensor_mode.is_none());
        assert!(!config.delta_encoding);
        assert!(!config.rle_encoding);
    }

    #[test]
    fn test_compression_config_serialize() {
        let config = CompressionConfig {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::for_dim(64))),
            delta_encoding: true,
            rle_encoding: true,
        };
        let bytes = bincode::serialize(&config).unwrap();
        let decoded: CompressionConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn test_tensor_mode_presets() {
        let tt = TensorMode::tensor_train(4096);
        assert!(matches!(tt, TensorMode::TensorTrain(_)));

        let high = TensorMode::high_compression(4096);
        assert!(matches!(high, TensorMode::TensorTrain(_)));

        let accurate = TensorMode::high_accuracy(4096);
        assert!(matches!(accurate, TensorMode::TensorTrain(_)));
    }

    #[test]
    fn test_compression_config_presets() {
        let high = CompressionConfig::high_compression();
        assert!(high.tensor_mode.is_some());
        assert!(high.delta_encoding);
        assert!(high.rle_encoding);

        let balanced = CompressionConfig::balanced(768);
        assert!(balanced.tensor_mode.is_some());

        let accurate = CompressionConfig::high_accuracy(768);
        assert!(accurate.tensor_mode.is_some());
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_quant_mode() {
        // Verify legacy QuantMode still works for backward compatibility
        assert_eq!(QuantMode::Int8, QuantMode::Int8);
        assert_ne!(QuantMode::Int8, QuantMode::Binary);
    }
}
