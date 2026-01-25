//! Compressed snapshot format for tensor data.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    compress_ids, decompress_ids, rle_decode, rle_encode, tt_decompose, tt_reconstruct,
    CompressionConfig, RleEncoded, TTCore, TensorMode,
};

/// Magic bytes identifying a Neumann snapshot file.
pub const MAGIC: [u8; 4] = *b"NEUM";

/// Current format version (v3 adds TT compression).
pub const VERSION: u16 = 3;

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u16),
    #[error("serialization error: {0}")]
    Serialization(#[from] bitcode::Error),
    #[error("tensor train error: {0}")]
    TensorTrain(#[from] crate::TTError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Snapshot file header.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u16,
    pub config: CompressionConfig,
    pub entry_count: u64,
}

impl Header {
    #[must_use]
    pub fn new(config: CompressionConfig, entry_count: u64) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            config,
            entry_count,
        }
    }

    /// # Errors
    /// Returns error if magic bytes or version are invalid.
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.magic != MAGIC {
            return Err(FormatError::InvalidMagic);
        }
        if self.version > VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Encoding used for a field value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldEncoding {
    Raw,
    DeltaVarint,
    Rle,
}

/// Compressed representation of a scalar value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressedScalar {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
}

/// Compressed representation of a tensor value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressedValue {
    Scalar(CompressedScalar),
    /// Raw f32 vector (no compression).
    VectorRaw(Vec<f32>),
    /// Tensor Train compressed vector (recommended for 1024+ dimensions).
    VectorTT {
        cores: Vec<TTCore>,
        original_dim: usize,
        shape: Vec<usize>,
        ranks: Vec<usize>,
    },
    /// Sparse vector (stores only non-zero positions and values).
    /// Efficient when >50% of values are zero.
    VectorSparse {
        /// Total dimension of the vector.
        dimension: usize,
        /// Positions of non-zero values (delta + varint encoded).
        positions: Vec<u8>,
        /// Non-zero values.
        values: Vec<f32>,
    },
    /// Delta + varint encoded ID list (for sorted u64 sequences).
    IdList(Vec<u8>),
    /// RLE encoded i64 values.
    RleInt(RleEncoded<i64>),
    /// Raw pointer.
    Pointer(String),
    /// Raw pointers.
    Pointers(Vec<String>),
}

/// Compressed tensor data entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedEntry {
    pub key: String,
    pub fields: HashMap<String, CompressedValue>,
}

/// Complete compressed snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedSnapshot {
    pub header: Header,
    pub entries: Vec<CompressedEntry>,
}

impl CompressedSnapshot {
    /// # Errors
    /// Returns error if serialization fails.
    pub fn serialize(&self) -> Result<Vec<u8>, FormatError> {
        Ok(bitcode::serialize(self)?)
    }

    /// # Errors
    /// Returns error if deserialization or validation fails.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, FormatError> {
        let snapshot: Self = bitcode::deserialize(bytes)?;
        snapshot.header.validate()?;
        Ok(snapshot)
    }
}

/// Compress a vector field based on configuration and key hints.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn compress_vector(
    vector: &[f32],
    key: &str,
    field_name: &str,
    config: &CompressionConfig,
) -> Result<CompressedValue, FormatError> {
    let is_embedding =
        key.starts_with("emb:") || field_name == "_embedding" || field_name == "vector";

    if is_embedding {
        if let Some(TensorMode::TensorTrain(tt_config)) = &config.tensor_mode {
            let tt = tt_decompose(vector, tt_config)?;
            return Ok(CompressedValue::VectorTT {
                cores: tt.cores,
                original_dim: tt.original_dim,
                shape: tt.shape,
                ranks: tt.ranks,
            });
        }
    }

    if config.delta_encoding && looks_like_id_list(vector, field_name) {
        let ids: Vec<u64> = vector.iter().map(|&f| f as u64).collect();
        let compressed = compress_ids(&ids);
        return Ok(CompressedValue::IdList(compressed));
    }

    Ok(CompressedValue::VectorRaw(vector.to_vec()))
}

/// Decompress a vector value.
///
/// # Errors
/// Returns error if decompression fails.
#[allow(clippy::cast_precision_loss)]
pub fn decompress_vector(value: &CompressedValue) -> Result<Vec<f32>, FormatError> {
    match value {
        CompressedValue::VectorRaw(v) => Ok(v.clone()),
        CompressedValue::VectorTT {
            cores,
            original_dim,
            shape,
            ranks,
        } => {
            let tt = crate::TTVector {
                cores: cores.clone(),
                original_dim: *original_dim,
                shape: shape.clone(),
                ranks: ranks.clone(),
            };
            Ok(tt_reconstruct(&tt))
        },
        CompressedValue::IdList(bytes) => {
            let ids = decompress_ids(bytes);
            Ok(ids.iter().map(|&id| id as f32).collect())
        },
        CompressedValue::VectorSparse {
            dimension,
            positions,
            values,
        } => {
            // Decompress positions from delta+varint encoding
            let pos_ids = decompress_ids(positions);
            // Reconstruct dense vector
            let mut dense = vec![0.0f32; *dimension];
            for (pos, &val) in pos_ids.iter().zip(values.iter()) {
                if let Ok(idx) = usize::try_from(*pos) {
                    if idx < *dimension {
                        dense[idx] = val;
                    }
                }
            }
            Ok(dense)
        },
        _ => Ok(Vec::new()),
    }
}

/// Compress i64 values with RLE if beneficial.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compress_ints(values: &[i64], config: &CompressionConfig) -> CompressedValue {
    if config.rle_encoding && values.len() > 1 {
        let encoded = rle_encode(values);
        if encoded.runs() < values.len() / 2 {
            return CompressedValue::RleInt(encoded);
        }
    }

    CompressedValue::VectorRaw(values.iter().map(|&v| v as f32).collect())
}

/// Decompress i64 values.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn decompress_ints(value: &CompressedValue) -> Vec<i64> {
    match value {
        CompressedValue::RleInt(encoded) => rle_decode(encoded),
        CompressedValue::VectorRaw(v) => v.iter().map(|&f| f as i64).collect(),
        _ => Vec::new(),
    }
}

/// Compress a sparse vector from position/value arrays.
///
/// Positions must be sorted and unique. Values are the non-zero elements
/// at those positions. The dimension is the total size of the vector.
///
/// Uses delta + varint encoding for positions, achieving excellent compression
/// for vectors with clustered non-zeros.
#[must_use]
pub fn compress_sparse(dimension: usize, positions: &[u32], values: &[f32]) -> CompressedValue {
    debug_assert_eq!(positions.len(), values.len());

    // Convert positions to u64 for delta encoding
    let pos_u64: Vec<u64> = positions.iter().map(|&p| u64::from(p)).collect();
    let compressed_positions = compress_ids(&pos_u64);

    CompressedValue::VectorSparse {
        dimension,
        positions: compressed_positions,
        values: values.to_vec(),
    }
}

/// Compress a dense vector as sparse if it has enough zeros.
///
/// Returns `Some(VectorSparse)` if the vector is >50% zeros (worth compressing),
/// otherwise returns `None` and the caller should use another format.
#[must_use]
pub fn compress_dense_as_sparse(vector: &[f32]) -> Option<CompressedValue> {
    // Count non-zeros, skipping indices that don't fit in u32
    let non_zeros: Vec<(u32, f32)> = vector
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .filter_map(|(i, &v)| u32::try_from(i).ok().map(|idx| (idx, v)))
        .collect();

    // Only use sparse format if >50% are zeros
    if non_zeros.len() * 2 > vector.len() {
        return None;
    }

    let positions: Vec<u32> = non_zeros.iter().map(|(p, _)| *p).collect();
    let values: Vec<f32> = non_zeros.iter().map(|(_, v)| *v).collect();

    Some(compress_sparse(vector.len(), &positions, &values))
}

/// Estimate the storage size of a sparse compressed vector.
#[must_use]
pub fn sparse_storage_size(_dimension: usize, nnz: usize) -> usize {
    // dimension (8) + positions_len (8) + positions_data (variable) + values (4*nnz)
    // positions use delta+varint, roughly 1-2 bytes per position for clustered data
    8 + 8 + nnz * 2 + nnz * 4
}

/// Check if sparse format would be more efficient than dense.
#[must_use]
pub fn should_use_sparse(dimension: usize, nnz: usize) -> bool {
    let dense_size = dimension * 4;
    let sparse_size = sparse_storage_size(dimension, nnz);
    sparse_size < dense_size
}

/// Check if a vector should use sparse format based on zero threshold.
///
/// A vector is considered sparse if more than `threshold` fraction of values
/// are effectively zero (absolute value less than 1e-6).
///
/// For threshold=0.5, this means at least 50% of values must be zero.
#[must_use]
pub fn should_use_sparse_threshold(vector: &[f32], threshold: f32) -> bool {
    const SCALE: usize = 1000;

    if vector.is_empty() {
        return false;
    }
    let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
    // Sparse if: (len - nnz) / len >= threshold
    // Rearranged: nnz <= len * (1 - threshold)
    // Multiply both sides by 1000 to use integer math with 3 decimal precision:
    // nnz * 1000 <= len * (1000 - threshold * 1000)
    // threshold is clamped to [0, 1], so threshold_int is in [0, 1000]
    let threshold_clamped = threshold.clamp(0.0, 1.0);
    let threshold_scaled = (f64::from(threshold_clamped) * 1000.0)
        .round()
        .clamp(0.0, 1000.0);
    // Safe conversion: value is guaranteed to be in [0, 1000]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let threshold_int = threshold_scaled as usize;
    let max_nnz = vector
        .len()
        .saturating_mul(SCALE.saturating_sub(threshold_int));
    let nnz_scaled = nnz.saturating_mul(SCALE);
    nnz_scaled <= max_nnz && should_use_sparse(vector.len(), nnz)
}

/// Heuristic: does this vector look like an ID list?
fn looks_like_id_list(vector: &[f32], field_name: &str) -> bool {
    if field_name == "ids" || field_name.ends_with("_ids") {
        return true;
    }

    if vector.len() < 2 {
        return false;
    }

    // Check first value
    if vector[0] < 0.0 || vector[0].fract() != 0.0 {
        return false;
    }

    let mut prev = vector[0];
    for &v in &vector[1..] {
        if v < prev || v < 0.0 || v.fract() != 0.0 {
            return false;
        }
        prev = v;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TTConfig;

    #[test]
    fn test_header_new() {
        let config = CompressionConfig::default();
        let header = Header::new(config, 100);
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.entry_count, 100);
    }

    #[test]
    fn test_header_validate_ok() {
        let header = Header::new(CompressionConfig::default(), 0);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_header_validate_bad_magic() {
        let mut header = Header::new(CompressionConfig::default(), 0);
        header.magic = *b"BAAD";
        assert!(matches!(header.validate(), Err(FormatError::InvalidMagic)));
    }

    #[test]
    fn test_header_validate_future_version() {
        let mut header = Header::new(CompressionConfig::default(), 0);
        header.version = 999;
        assert!(matches!(
            header.validate(),
            Err(FormatError::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn test_compress_vector_tt() {
        let config = CompressionConfig {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::for_dim(64).unwrap())),
            ..Default::default()
        };
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let compressed = compress_vector(&vector, "emb:doc1", "vector", &config).unwrap();
        assert!(matches!(compressed, CompressedValue::VectorTT { .. }));

        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 64);
    }

    #[test]
    fn test_compress_vector_id_list() {
        let config = CompressionConfig {
            delta_encoding: true,
            ..Default::default()
        };
        let vector: Vec<f32> = (100..200).map(|i| i as f32).collect();
        let compressed = compress_vector(&vector, "index", "ids", &config).unwrap();
        assert!(matches!(compressed, CompressedValue::IdList(_)));

        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 100);
        for (orig, dec) in vector.iter().zip(&decompressed) {
            assert_eq!(*orig as u64, *dec as u64);
        }
    }

    #[test]
    fn test_compress_vector_raw() {
        let config = CompressionConfig::default();
        let vector = vec![0.1, 0.2, 0.3];
        let compressed = compress_vector(&vector, "random", "data", &config).unwrap();
        assert!(matches!(compressed, CompressedValue::VectorRaw(_)));
    }

    #[test]
    fn test_compress_ints_rle() {
        let config = CompressionConfig {
            rle_encoding: true,
            ..Default::default()
        };
        let values = vec![1i64; 100];
        let compressed = compress_ints(&values, &config);
        assert!(matches!(compressed, CompressedValue::RleInt(_)));

        let decompressed = decompress_ints(&compressed);
        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_compress_ints_no_benefit() {
        let config = CompressionConfig {
            rle_encoding: true,
            ..Default::default()
        };
        let values: Vec<i64> = (0..100).collect();
        let compressed = compress_ints(&values, &config);
        assert!(matches!(compressed, CompressedValue::VectorRaw(_)));
    }

    #[test]
    fn test_looks_like_id_list() {
        assert!(looks_like_id_list(&[1.0, 2.0, 3.0, 4.0], "ids"));
        assert!(looks_like_id_list(&[100.0, 101.0, 102.0], "row_ids"));
        assert!(!looks_like_id_list(&[3.0, 2.0, 1.0], "data"));
        assert!(!looks_like_id_list(&[1.5, 2.5, 3.5], "data"));
        assert!(!looks_like_id_list(&[-1.0, 0.0, 1.0], "data"));
        assert!(!looks_like_id_list(&[1.0], "data"));
    }

    #[test]
    fn test_compressed_snapshot_roundtrip() {
        let header = Header::new(CompressionConfig::default(), 1);
        let entry = CompressedEntry {
            key: "test".to_string(),
            fields: HashMap::from([(
                "value".to_string(),
                CompressedValue::Scalar(CompressedScalar::Int(42)),
            )]),
        };
        let snapshot = CompressedSnapshot {
            header,
            entries: vec![entry],
        };

        let bytes = snapshot.serialize().unwrap();
        let restored = CompressedSnapshot::deserialize(&bytes).unwrap();

        assert_eq!(restored.entries.len(), 1);
        assert_eq!(restored.entries[0].key, "test");
    }

    #[test]
    fn test_tt_compressed_snapshot_roundtrip() {
        let config = CompressionConfig {
            tensor_mode: Some(TensorMode::TensorTrain(TTConfig::for_dim(64).unwrap())),
            delta_encoding: true,
            rle_encoding: true,
        };
        let header = Header::new(config.clone(), 1);

        // Create a TT-compressed vector entry
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let compressed_vec = compress_vector(&vector, "emb:test", "vector", &config).unwrap();

        let entry = CompressedEntry {
            key: "embedding".to_string(),
            fields: HashMap::from([("vector".to_string(), compressed_vec)]),
        };
        let snapshot = CompressedSnapshot {
            header,
            entries: vec![entry],
        };

        let bytes = snapshot.serialize().unwrap();
        let restored = CompressedSnapshot::deserialize(&bytes).unwrap();

        assert_eq!(restored.entries.len(), 1);
        let restored_vec = &restored.entries[0].fields["vector"];
        assert!(matches!(restored_vec, CompressedValue::VectorTT { .. }));

        // Verify decompression works
        let decompressed = decompress_vector(restored_vec).unwrap();
        assert_eq!(decompressed.len(), 64);
    }

    #[test]
    fn test_format_error_display() {
        let err = FormatError::InvalidMagic;
        assert_eq!(err.to_string(), "invalid magic bytes");

        let err = FormatError::UnsupportedVersion(5);
        assert_eq!(err.to_string(), "unsupported version: 5");
    }

    #[test]
    fn test_compress_sparse_basic() {
        let positions = vec![0, 5, 10];
        let values = vec![1.0, 2.0, 3.0];
        let compressed = compress_sparse(100, &positions, &values);

        match compressed {
            CompressedValue::VectorSparse {
                dimension,
                positions: _,
                values: v,
            } => {
                assert_eq!(dimension, 100);
                assert_eq!(v.len(), 3);
                assert_eq!(v, vec![1.0, 2.0, 3.0]);
            },
            _ => panic!("expected VectorSparse"),
        }
    }

    #[test]
    fn test_compress_sparse_roundtrip() {
        let positions = vec![0, 10, 50, 99];
        let values = vec![0.5, 1.5, 2.5, 3.5];
        let compressed = compress_sparse(100, &positions, &values);

        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 100);
        assert_eq!(decompressed[0], 0.5);
        assert_eq!(decompressed[10], 1.5);
        assert_eq!(decompressed[50], 2.5);
        assert_eq!(decompressed[99], 3.5);
        // All other values should be zero
        assert_eq!(decompressed[1], 0.0);
        assert_eq!(decompressed[49], 0.0);
    }

    #[test]
    fn test_compress_dense_as_sparse() {
        // Dense vector with many zeros
        let mut dense = vec![0.0f32; 100];
        dense[5] = 1.0;
        dense[25] = 2.0;
        dense[75] = 3.0;

        let compressed = compress_dense_as_sparse(&dense);
        assert!(compressed.is_some());

        let compressed = compressed.unwrap();
        match &compressed {
            CompressedValue::VectorSparse {
                dimension, values, ..
            } => {
                assert_eq!(*dimension, 100);
                assert_eq!(values.len(), 3);
            },
            _ => panic!("expected VectorSparse"),
        }

        // Verify roundtrip
        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed[5], 1.0);
        assert_eq!(decompressed[25], 2.0);
        assert_eq!(decompressed[75], 3.0);
    }

    #[test]
    fn test_compress_dense_as_sparse_too_dense() {
        // Dense vector with too many non-zeros (>50%)
        let dense: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let compressed = compress_dense_as_sparse(&dense);
        assert!(compressed.is_none());
    }

    #[test]
    fn test_should_use_sparse() {
        // 10 non-zeros in 1000-dim vector = 1% density, definitely sparse
        // dense: 4000, sparse: 16 + 10*2 + 10*4 = 76
        assert!(should_use_sparse(1000, 10));

        // 660+ non-zeros in 1000-dim vector is too dense
        // dense: 4000, sparse: 16 + 660*6 = 3976 (just under)
        // At 667+: sparse: 16 + 667*6 = 4018 > 4000
        assert!(!should_use_sparse(1000, 667));

        // 500 non-zeros is still sparse efficient
        // dense: 4000, sparse: 16 + 500*6 = 3016
        assert!(should_use_sparse(1000, 500));
    }

    #[test]
    fn test_sparse_storage_size() {
        // sparse_storage_size = 8 + 8 + nnz*2 + nnz*4 = 16 + nnz*6
        let size = sparse_storage_size(1000, 10);
        assert_eq!(size, 16 + 10 * 6); // 76 bytes

        // Empty sparse vector: 16 bytes overhead
        let size = sparse_storage_size(1000, 0);
        assert_eq!(size, 16);
    }

    #[test]
    fn test_sparse_empty_vector() {
        let compressed = compress_sparse(100, &[], &[]);
        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 100);
        assert!(decompressed.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sparse_snapshot_roundtrip() {
        let header = Header::new(CompressionConfig::default(), 1);

        // Create a sparse-compressed vector entry
        let positions = vec![0, 50, 99];
        let values = vec![1.0, 2.0, 3.0];
        let compressed_vec = compress_sparse(100, &positions, &values);

        let entry = CompressedEntry {
            key: "sparse_embedding".to_string(),
            fields: HashMap::from([("vector".to_string(), compressed_vec)]),
        };
        let snapshot = CompressedSnapshot {
            header,
            entries: vec![entry],
        };

        let bytes = snapshot.serialize().unwrap();
        let restored = CompressedSnapshot::deserialize(&bytes).unwrap();

        assert_eq!(restored.entries.len(), 1);
        let restored_vec = &restored.entries[0].fields["vector"];
        assert!(matches!(restored_vec, CompressedValue::VectorSparse { .. }));

        let decompressed = decompress_vector(restored_vec).unwrap();
        assert_eq!(decompressed.len(), 100);
        assert_eq!(decompressed[0], 1.0);
        assert_eq!(decompressed[50], 2.0);
        assert_eq!(decompressed[99], 3.0);
    }
}
