//! Delta encoding with variable-length integers for sorted ID sequences.

/// Delta-encode a sorted list of IDs.
/// Stores first value followed by differences between consecutive values.
#[must_use]
pub fn delta_encode(ids: &[u64]) -> Vec<u64> {
    if ids.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(ids.len());
    result.push(ids[0]);

    for window in ids.windows(2) {
        result.push(window[1].saturating_sub(window[0]));
    }

    result
}

/// Decode delta-encoded IDs back to original sorted list.
#[must_use]
pub fn delta_decode(deltas: &[u64]) -> Vec<u64> {
    if deltas.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(deltas.len());
    let mut current = deltas[0];
    result.push(current);

    for &delta in &deltas[1..] {
        current = current.saturating_add(delta);
        result.push(current);
    }

    result
}

/// Variable-length encode u64 values.
/// Uses 7 bits per byte with high bit as continuation flag.
#[must_use]
pub fn varint_encode(values: &[u64]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len() * 2);

    for &value in values {
        let mut v = value;
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                result.push(byte);
                break;
            }
            result.push(byte | 0x80);
        }
    }

    result
}

/// Decode variable-length encoded bytes back to u64 values.
#[must_use]
pub fn varint_decode(bytes: &[u8]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut current: u64 = 0;
    let mut shift = 0;

    for &byte in bytes {
        current |= u64::from(byte & 0x7F) << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            result.push(current);
            current = 0;
            shift = 0;
        }
    }

    result
}

/// Combined delta + varint encoding for maximum compression of sorted IDs.
#[must_use]
pub fn compress_ids(ids: &[u64]) -> Vec<u8> {
    let deltas = delta_encode(ids);
    varint_encode(&deltas)
}

/// Decompress delta + varint encoded IDs.
#[must_use]
pub fn decompress_ids(bytes: &[u8]) -> Vec<u64> {
    let deltas = varint_decode(bytes);
    delta_decode(&deltas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encode_empty() {
        assert!(delta_encode(&[]).is_empty());
    }

    #[test]
    fn test_delta_encode_single() {
        assert_eq!(delta_encode(&[42]), vec![42]);
    }

    #[test]
    fn test_delta_encode_sequential() {
        let ids = vec![100, 101, 102, 103, 104];
        let encoded = delta_encode(&ids);
        assert_eq!(encoded, vec![100, 1, 1, 1, 1]);
    }

    #[test]
    fn test_delta_encode_gaps() {
        let ids = vec![10, 20, 100, 101, 200];
        let encoded = delta_encode(&ids);
        assert_eq!(encoded, vec![10, 10, 80, 1, 99]);
    }

    #[test]
    fn test_delta_roundtrip() {
        let original = vec![1, 5, 10, 100, 1000, 10000];
        let encoded = delta_encode(&original);
        let decoded = delta_decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_delta_decode_empty() {
        assert!(delta_decode(&[]).is_empty());
    }

    #[test]
    fn test_varint_encode_small() {
        let values = vec![0, 1, 127];
        let encoded = varint_encode(&values);
        assert_eq!(encoded, vec![0, 1, 127]);
    }

    #[test]
    fn test_varint_encode_medium() {
        let values = vec![128];
        let encoded = varint_encode(&values);
        assert_eq!(encoded, vec![0x80, 0x01]);
    }

    #[test]
    fn test_varint_encode_large() {
        let values = vec![16384];
        let encoded = varint_encode(&values);
        assert_eq!(encoded, vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn test_varint_roundtrip() {
        let original = vec![0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];
        let encoded = varint_encode(&original);
        let decoded = varint_decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_varint_empty() {
        assert!(varint_encode(&[]).is_empty());
        assert!(varint_decode(&[]).is_empty());
    }

    #[test]
    fn test_compress_ids_sequential() {
        let ids: Vec<u64> = (1000..1100).collect();
        let compressed = compress_ids(&ids);
        let decompressed = decompress_ids(&compressed);
        assert_eq!(ids, decompressed);

        // Sequential IDs should compress very well
        let original_bytes = ids.len() * 8;
        let compressed_bytes = compressed.len();
        let ratio = original_bytes as f64 / compressed_bytes as f64;
        assert!(ratio > 4.0, "Expected >4x compression, got {:.2}x", ratio);
    }

    #[test]
    fn test_compress_ids_sparse() {
        let ids = vec![100, 1000, 10000, 100_000, 1_000_000];
        let compressed = compress_ids(&ids);
        let decompressed = decompress_ids(&compressed);
        assert_eq!(ids, decompressed);
    }

    #[test]
    fn test_compress_ids_empty() {
        assert!(compress_ids(&[]).is_empty());
        assert!(decompress_ids(&[]).is_empty());
    }

    #[test]
    fn test_compression_ratio_best_case() {
        // 10,000 sequential IDs starting at 0
        let ids: Vec<u64> = (0..10_000).collect();
        let compressed = compress_ids(&ids);

        let original_bytes = ids.len() * 8;
        let compressed_bytes = compressed.len();
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        // Sequential should give ~8x compression (1 byte per delta)
        assert!(ratio > 7.0, "Expected ~8x compression, got {:.2}x", ratio);
    }

    #[test]
    fn test_compression_ratio_worst_case() {
        // Random large gaps (worst case for delta)
        let ids = vec![0, u64::MAX / 4, u64::MAX / 2, u64::MAX];
        let compressed = compress_ids(&ids);
        let decompressed = decompress_ids(&compressed);
        assert_eq!(ids, decompressed);
    }
}
