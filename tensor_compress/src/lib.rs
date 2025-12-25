//! Bespoke compression for tensor data.
//!
//! Provides compression algorithms optimized for tensor semantics:
//! - Vector quantization (int8, binary) for embeddings
//! - Delta encoding with varint for sorted ID sequences
//! - Run-length encoding for repeated values

mod delta;
mod quantize;
mod rle;

pub mod format;

pub use delta::{
    compress_ids, decompress_ids, delta_decode, delta_encode, varint_decode, varint_encode,
};
pub use quantize::{
    dequantize_binary, dequantize_int8, quantize_binary, quantize_int8, QuantizationError,
};
pub use rle::{rle_decode, rle_encode, RleEncoded};

use serde::{Deserialize, Serialize};

/// Compression configuration for snapshots.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressionConfig {
    pub vector_quantization: Option<QuantMode>,
    pub delta_encoding: bool,
    pub rle_encoding: bool,
}

/// Vector quantization mode.
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
        assert!(config.vector_quantization.is_none());
        assert!(!config.delta_encoding);
        assert!(!config.rle_encoding);
    }

    #[test]
    fn test_compression_config_serialize() {
        let config = CompressionConfig {
            vector_quantization: Some(QuantMode::Int8),
            delta_encoding: true,
            rle_encoding: true,
        };
        let bytes = bincode::serialize(&config).unwrap();
        let decoded: CompressionConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn test_quant_mode_variants() {
        assert_eq!(QuantMode::Int8, QuantMode::Int8);
        assert_ne!(QuantMode::Int8, QuantMode::Binary);
    }
}
