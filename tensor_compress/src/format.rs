//! Compressed snapshot format for tensor data.

use crate::{
    compress_ids, decompress_ids, dequantize_binary, dequantize_int8, quantize_binary,
    quantize_int8, rle_decode, rle_encode, CompressionConfig, QuantMode, RleEncoded,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Magic bytes identifying a Neumann snapshot file.
pub const MAGIC: [u8; 4] = *b"NEUM";

/// Current format version.
pub const VERSION: u16 = 2;

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u16),
    #[error("serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("quantization error: {0}")]
    Quantization(#[from] crate::QuantizationError),
}

/// Snapshot file header.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u16,
    pub config: CompressionConfig,
    pub entry_count: u64,
}

impl Header {
    #[must_use]
    pub const fn new(config: CompressionConfig, entry_count: u64) -> Self {
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
    QuantizedInt8,
    QuantizedBinary,
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
    /// Int8 quantized vector.
    VectorInt8 {
        data: Vec<i8>,
        min: f32,
        scale: f32,
    },
    /// Binary quantized vector.
    VectorBinary {
        data: Vec<u8>,
        len: usize,
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
        Ok(bincode::serialize(self)?)
    }

    /// # Errors
    /// Returns error if deserialization or validation fails.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, FormatError> {
        let snapshot: Self = bincode::deserialize(bytes)?;
        snapshot.header.validate()?;
        Ok(snapshot)
    }
}

/// Compress a vector field based on configuration and key hints.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn compress_vector(
    vector: &[f32],
    key: &str,
    field_name: &str,
    config: &CompressionConfig,
) -> CompressedValue {
    let is_embedding =
        key.starts_with("emb:") || field_name == "_embedding" || field_name == "vector";

    if is_embedding {
        if let Some(mode) = &config.vector_quantization {
            return match mode {
                QuantMode::Int8 => {
                    if let Ok(q) = quantize_int8(vector) {
                        CompressedValue::VectorInt8 {
                            data: q.data,
                            min: q.min,
                            scale: q.scale,
                        }
                    } else {
                        CompressedValue::VectorRaw(vector.to_vec())
                    }
                },
                QuantMode::Binary => {
                    let q = quantize_binary(vector);
                    CompressedValue::VectorBinary {
                        data: q.data,
                        len: q.len,
                    }
                },
            };
        }
    }

    if config.delta_encoding && looks_like_id_list(vector, field_name) {
        let ids: Vec<u64> = vector.iter().map(|&f| f as u64).collect();
        let compressed = compress_ids(&ids);
        return CompressedValue::IdList(compressed);
    }

    CompressedValue::VectorRaw(vector.to_vec())
}

/// Decompress a vector value.
///
/// # Errors
/// Returns error if decompression fails (e.g., invalid quantized data).
#[allow(clippy::cast_precision_loss)]
pub fn decompress_vector(value: &CompressedValue) -> Result<Vec<f32>, FormatError> {
    match value {
        CompressedValue::VectorRaw(v) => Ok(v.clone()),
        CompressedValue::VectorInt8 { data, min, scale } => {
            let quantized = crate::quantize::QuantizedInt8 {
                data: data.clone(),
                min: *min,
                scale: *scale,
            };
            Ok(dequantize_int8(&quantized))
        },
        CompressedValue::VectorBinary { data, len } => {
            let quantized = crate::quantize::QuantizedBinary {
                data: data.clone(),
                len: *len,
            };
            Ok(dequantize_binary(&quantized)?)
        },
        CompressedValue::IdList(bytes) => {
            let ids = decompress_ids(bytes);
            Ok(ids.iter().map(|&id| id as f32).collect())
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
    fn test_compress_vector_embedding_int8() {
        let config = CompressionConfig {
            vector_quantization: Some(QuantMode::Int8),
            ..Default::default()
        };
        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let compressed = compress_vector(&vector, "emb:doc1", "vector", &config);
        assert!(matches!(compressed, CompressedValue::VectorInt8 { .. }));

        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 4);
    }

    #[test]
    fn test_compress_vector_embedding_binary() {
        let config = CompressionConfig {
            vector_quantization: Some(QuantMode::Binary),
            ..Default::default()
        };
        let vector = vec![0.1, -0.2, 0.3, -0.4];
        let compressed = compress_vector(&vector, "emb:doc1", "vector", &config);
        assert!(matches!(compressed, CompressedValue::VectorBinary { .. }));

        let decompressed = decompress_vector(&compressed).unwrap();
        assert_eq!(decompressed.len(), 4);
    }

    #[test]
    fn test_compress_vector_id_list() {
        let config = CompressionConfig {
            delta_encoding: true,
            ..Default::default()
        };
        let vector: Vec<f32> = (100..200).map(|i| i as f32).collect();
        let compressed = compress_vector(&vector, "index", "ids", &config);
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
        let compressed = compress_vector(&vector, "random", "data", &config);
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
    fn test_format_error_display() {
        let err = FormatError::InvalidMagic;
        assert_eq!(err.to_string(), "invalid magic bytes");

        let err = FormatError::UnsupportedVersion(5);
        assert_eq!(err.to_string(), "unsupported version: 5");
    }
}
