// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tensor-native compression library.
//!
//! Exploits the mathematical structure of high-dimensional embeddings using
//! Tensor Train decomposition, achieving 10-20x compression for 4096+ dimensions.
//!
//! # Compression Methods
//!
//! - **Tensor Train (TT)**: Decomposes vectors into products of smaller tensors (recommended)
//! - **Sparse**: Native format for vectors with >50% zeros (stores only non-zeros)
//! - **Delta + varint**: Lossless compression for sorted ID sequences
//! - **Run-length encoding**: Lossless compression for repeated values

pub mod decompose;
mod delta;
pub mod incremental;
mod rle;
pub mod streaming;
pub mod streaming_tt;
pub mod tensor_train;

pub mod format;

pub use decompose::{svd_truncated, DecomposeError, Matrix, SvdResult, TensorView};
pub use delta::{
    compress_ids, decompress_ids, delta_decode, delta_encode, varint_decode, varint_encode,
};
pub use format::{
    compress_dense_as_sparse, compress_sparse, should_use_sparse, should_use_sparse_threshold,
    sparse_storage_size,
};
pub use rle::{rle_decode, rle_encode, RleEncoded};
use serde::{Deserialize, Serialize};
pub use streaming_tt::{
    convert_vectors_to_streaming_tt, read_streaming_tt_all, streaming_tt_similarity_search,
    StreamingTTHeader, StreamingTTReader, StreamingTTWriter, STREAMING_TT_MAGIC,
    STREAMING_TT_VERSION,
};
pub use tensor_train::{
    tt_cosine_similarity, tt_cosine_similarity_batch, tt_decompose, tt_decompose_batch,
    tt_dot_product, tt_dot_product_batch, tt_euclidean_distance, tt_euclidean_distance_batch,
    tt_norm, tt_reconstruct, tt_scale, TTConfig, TTCore, TTError, TTVector,
};

/// Tensor compression mode for vectors and embeddings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorMode {
    /// Tensor Train decomposition (recommended for 1024+ dimensions).
    /// Achieves 10-20x compression with <1% error.
    TensorTrain(TTConfig),
}

impl TensorMode {
    /// Create TT mode optimized for the given dimension.
    ///
    /// # Panics
    /// Panics if dimension is 0.
    #[must_use]
    pub fn tensor_train(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::for_dim(dim).expect("invalid dimension"))
    }

    /// Create TT mode, returning error for invalid dimension.
    ///
    /// # Errors
    /// Returns error if dimension is 0 or has no valid factorization.
    pub fn try_tensor_train(dim: usize) -> Result<Self, TTError> {
        Ok(Self::TensorTrain(TTConfig::for_dim(dim)?))
    }

    /// High compression TT preset.
    ///
    /// # Panics
    /// Panics if dimension is 0.
    #[must_use]
    pub fn high_compression(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::high_compression(dim).expect("invalid dimension"))
    }

    /// High accuracy TT preset.
    ///
    /// # Panics
    /// Panics if dimension is 0.
    #[must_use]
    pub fn high_accuracy(dim: usize) -> Self {
        Self::TensorTrain(TTConfig::high_accuracy(dim).expect("invalid dimension"))
    }
}

/// Common embedding dimension constants.
pub struct CompressionDefaults;

impl CompressionDefaults {
    /// `MiniLM` and small models.
    pub const SMALL: usize = 64;
    /// `all-MiniLM-L6-v2`.
    pub const MEDIUM: usize = 384;
    /// BERT, sentence-transformers.
    pub const STANDARD: usize = 768;
    /// `OpenAI` `text-embedding-ada-002`.
    pub const LARGE: usize = 1536;
    /// `LLaMA` and large models.
    pub const XLARGE: usize = 4096;

    /// Auto-detect optimal compression config from vector length.
    #[must_use]
    pub fn config_for(vector: &[f32]) -> CompressionConfig {
        CompressionConfig::balanced(vector.len())
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
    ///
    /// # Panics
    /// Should not panic as 4096 is a valid dimension.
    #[must_use]
    pub fn high_compression() -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(
                TTConfig::high_compression(CompressionDefaults::XLARGE).expect("valid dimension"),
            )),
            delta_encoding: true,
            rle_encoding: true,
        }
    }

    /// Balanced compression preset.
    ///
    /// # Panics
    /// Panics if dimension is 0.
    #[must_use]
    pub fn balanced(dim: usize) -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(
                TTConfig::for_dim(dim).expect("invalid dimension"),
            )),
            delta_encoding: true,
            rle_encoding: true,
        }
    }

    /// High accuracy preset (lower compression).
    ///
    /// # Panics
    /// Panics if dimension is 0.
    #[must_use]
    pub fn high_accuracy(dim: usize) -> Self {
        Self {
            tensor_mode: Some(TensorMode::TensorTrain(
                TTConfig::high_accuracy(dim).expect("invalid dimension"),
            )),
            delta_encoding: true,
            rle_encoding: true,
        }
    }
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
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::for_dim(64).unwrap())),
            delta_encoding: true,
            rle_encoding: true,
        };
        let bytes = bitcode::serialize(&config).unwrap();
        let decoded: CompressionConfig = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn test_ttconfig_validation() {
        // Valid config
        assert!(TTConfig::for_dim(64).is_ok());
        assert!(TTConfig::for_dim(768).is_ok());
        assert!(TTConfig::for_dim(4096).is_ok());

        // Invalid: zero dimension
        assert!(TTConfig::for_dim(0).is_err());

        // Invalid: manual construction with bad tolerance
        let bad_config = TTConfig {
            shape: vec![4, 4, 4],
            max_rank: 8,
            tolerance: 0.0,
        };
        assert!(bad_config.validate().is_err());

        let bad_config2 = TTConfig {
            shape: vec![4, 4, 4],
            max_rank: 8,
            tolerance: 1.5,
        };
        assert!(bad_config2.validate().is_err());
    }

    #[test]
    fn test_compression_defaults() {
        assert_eq!(CompressionDefaults::SMALL, 64);
        assert_eq!(CompressionDefaults::STANDARD, 768);
        assert_eq!(CompressionDefaults::XLARGE, 4096);

        let vector = vec![0.0f32; 768];
        let config = CompressionDefaults::config_for(&vector);
        assert!(config.tensor_mode.is_some());
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
    fn test_try_tensor_train() {
        // Valid dimensions should succeed
        let mode = TensorMode::try_tensor_train(64);
        assert!(mode.is_ok());
        assert!(matches!(mode.unwrap(), TensorMode::TensorTrain(_)));

        // Zero dimension should fail
        let mode = TensorMode::try_tensor_train(0);
        assert!(mode.is_err());
    }
}
