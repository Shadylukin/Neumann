//! Network message compression for TCP transport.
//!
//! Provides frame-level LZ4 compression with capability negotiation.
//! Compression is transparent to the protocol layer.

use serde::{Deserialize, Serialize};

use super::error::{TcpError, TcpResult};

/// Compression method used for network messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression.
    #[default]
    None,
    /// LZ4 compression (fast, reasonable ratio).
    Lz4,
}

/// Configuration for network message compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Whether compression is enabled.
    pub enabled: bool,
    /// Compression method to use.
    pub method: CompressionMethod,
    /// Minimum payload size for compression (bytes).
    /// Messages smaller than this are sent uncompressed.
    pub min_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: CompressionMethod::Lz4,
            min_size: 256,
        }
    }
}

impl CompressionConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            method: CompressionMethod::None,
            min_size: 256,
        }
    }

    pub fn with_method(mut self, method: CompressionMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size;
        self
    }
}

/// Frame flags byte encoding.
///
/// ```text
/// bit 0: compressed (1 = LZ4 compressed)
/// bits 1-7: reserved (must be 0)
/// ```
pub mod flags {
    /// No compression.
    pub const NONE: u8 = 0x00;
    /// LZ4 compression.
    pub const LZ4: u8 = 0x01;
}

/// Maximum decompressed message size (16 MB).
/// SECURITY: Prevents memory DoS from malicious size prefixes.
pub const MAX_DECOMPRESSED_SIZE: usize = 16 * 1024 * 1024;

pub fn frame_flags(method: CompressionMethod) -> u8 {
    match method {
        CompressionMethod::None => flags::NONE,
        CompressionMethod::Lz4 => flags::LZ4,
    }
}

pub fn method_from_flags(flag_byte: u8) -> TcpResult<CompressionMethod> {
    match flag_byte & 0x01 {
        0 => Ok(CompressionMethod::None),
        1 => Ok(CompressionMethod::Lz4),
        _ => unreachable!(), // Masked to 1 bit
    }
}

pub fn compress(data: &[u8], method: CompressionMethod) -> Vec<u8> {
    match method {
        CompressionMethod::None => data.to_vec(),
        CompressionMethod::Lz4 => lz4_flex::compress_prepend_size(data),
    }
}

/// Decompress data using the specified method.
/// SECURITY: Validates the claimed size before decompression to prevent memory DoS.
pub fn decompress(data: &[u8], method: CompressionMethod) -> TcpResult<Vec<u8>> {
    match method {
        CompressionMethod::None => Ok(data.to_vec()),
        CompressionMethod::Lz4 => {
            // SECURITY: Validate claimed size BEFORE decompression
            if data.len() < 4 {
                return Err(TcpError::Compression {
                    operation: "decompress",
                    message: "LZ4 data too short for size prefix".to_string(),
                });
            }

            // LZ4 prepends the uncompressed size as 4 bytes little-endian
            let claimed_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

            if claimed_size > MAX_DECOMPRESSED_SIZE {
                return Err(TcpError::Compression {
                    operation: "decompress",
                    message: format!(
                        "Claimed decompressed size {} exceeds maximum {}",
                        claimed_size, MAX_DECOMPRESSED_SIZE
                    ),
                });
            }

            // Now safe to decompress
            lz4_flex::decompress_size_prepended(data).map_err(|e| TcpError::Compression {
                operation: "decompress",
                message: e.to_string(),
            })
        },
    }
}

pub fn is_beneficial(original_len: usize, compressed_len: usize) -> bool {
    compressed_len < original_len
}

/// The compression capability identifier for handshake.
pub const COMPRESSION_CAPABILITY: &str = "compression";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_lz4() {
        let data = b"Hello, this is a test string that should compress well. ".repeat(10);
        let compressed = compress(&data, CompressionMethod::Lz4);
        let decompressed = decompress(&compressed, CompressionMethod::Lz4).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_compress_decompress_none() {
        let data = b"uncompressed data";
        let compressed = compress(data, CompressionMethod::None);
        assert_eq!(data.as_slice(), compressed.as_slice());
        let decompressed = decompress(&compressed, CompressionMethod::None).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_lz4_compression_ratio() {
        // Test that repeated data compresses well
        let data = b"AAAA".repeat(1000);
        let compressed = compress(&data, CompressionMethod::Lz4);
        assert!(
            compressed.len() < data.len() / 2,
            "expected significant compression, got {} -> {}",
            data.len(),
            compressed.len()
        );
    }

    #[test]
    fn test_frame_flags() {
        assert_eq!(frame_flags(CompressionMethod::None), flags::NONE);
        assert_eq!(frame_flags(CompressionMethod::Lz4), flags::LZ4);
    }

    #[test]
    fn test_method_from_flags() {
        assert_eq!(
            method_from_flags(flags::NONE).unwrap(),
            CompressionMethod::None
        );
        assert_eq!(
            method_from_flags(flags::LZ4).unwrap(),
            CompressionMethod::Lz4
        );
    }

    #[test]
    fn test_method_from_flags_ignores_reserved_bits() {
        // Reserved bits should be ignored
        assert_eq!(method_from_flags(0x00).unwrap(), CompressionMethod::None);
        assert_eq!(method_from_flags(0x01).unwrap(), CompressionMethod::Lz4);
        // High bits set but bit 0 clear = no compression
        assert_eq!(method_from_flags(0xFE).unwrap(), CompressionMethod::None);
        // High bits set with bit 0 set = LZ4
        assert_eq!(method_from_flags(0xFF).unwrap(), CompressionMethod::Lz4);
    }

    #[test]
    fn test_config_default() {
        let config = CompressionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.method, CompressionMethod::Lz4);
        assert_eq!(config.min_size, 256);
    }

    #[test]
    fn test_config_disabled() {
        let config = CompressionConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.method, CompressionMethod::None);
    }

    // === Security Tests for LZ4 Size Validation ===

    #[test]
    fn test_lz4_decompress_too_short() {
        // Less than 4 bytes (no size prefix)
        let result = decompress(&[1, 2, 3], CompressionMethod::Lz4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn test_lz4_decompress_oversized_claim() {
        // Craft a malicious payload claiming enormous decompressed size
        let mut malicious = Vec::new();
        // Claim 1 GB decompressed size (way over our 16 MB limit)
        malicious.extend_from_slice(&(1024u32 * 1024 * 1024).to_le_bytes());
        // Add some garbage "compressed" data
        malicious.extend_from_slice(&[0x00; 100]);

        let result = decompress(&malicious, CompressionMethod::Lz4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_lz4_decompress_valid_size_accepted() {
        // Valid compressed data with reasonable size
        let data = b"Hello, compression test!".repeat(10);
        let compressed = compress(&data, CompressionMethod::Lz4);
        let decompressed = decompress(&compressed, CompressionMethod::Lz4).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_lz4_decompress_just_over_max_rejected() {
        // Test size just OVER the limit - should be rejected by size check
        let mut data = Vec::new();
        // Claim exactly MAX_DECOMPRESSED_SIZE + 1 bytes
        data.extend_from_slice(&((MAX_DECOMPRESSED_SIZE + 1) as u32).to_le_bytes());
        // Add minimal "compressed" data
        data.extend_from_slice(&[0x00; 10]);

        let result = decompress(&data, CompressionMethod::Lz4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Should contain "exceeds maximum" - the size check should catch this
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_config_builder() {
        let config = CompressionConfig::default()
            .with_method(CompressionMethod::None)
            .with_min_size(512);
        assert_eq!(config.method, CompressionMethod::None);
        assert_eq!(config.min_size, 512);
    }

    #[test]
    fn test_is_beneficial() {
        assert!(is_beneficial(1000, 500));
        assert!(!is_beneficial(100, 100));
        assert!(!is_beneficial(100, 150));
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let compressed = compress(data, CompressionMethod::Lz4);
        let decompressed = decompress(&compressed, CompressionMethod::Lz4).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_small_data() {
        let data = b"tiny";
        let compressed = compress(data, CompressionMethod::Lz4);
        let decompressed = decompress(&compressed, CompressionMethod::Lz4).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_invalid_lz4_data() {
        let invalid = b"not valid lz4 data";
        let result = decompress(invalid, CompressionMethod::Lz4);
        assert!(matches!(result, Err(TcpError::Compression { .. })));
    }

    #[test]
    fn test_compression_method_default() {
        let method = CompressionMethod::default();
        assert_eq!(method, CompressionMethod::None);
    }

    #[test]
    fn test_compression_method_debug() {
        let method = CompressionMethod::Lz4;
        let debug = format!("{:?}", method);
        assert!(debug.contains("Lz4"));
    }

    #[test]
    fn test_config_debug() {
        let config = CompressionConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CompressionConfig"));
        assert!(debug.contains("enabled"));
    }

    #[test]
    fn test_config_clone() {
        let config = CompressionConfig::default().with_min_size(1024);
        let cloned = config.clone();
        assert_eq!(cloned.min_size, 1024);
    }
}
