// SPDX-License-Identifier: MIT OR Apache-2.0
//! Snapshot format v2/v3 with backward compatibility.
//!
//! Format evolution:
//! - v2: `HashMap<String, TensorData>` serialized with bincode
//! - v3: `SlabRouterSnapshot` with specialized slab storage and TT compression
//!
//! Embeddings are automatically compressed using Tensor Train decomposition
//! (10-20x compression for 768+ dimensional vectors).

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    slab_router::{SlabRouter, SlabRouterSnapshot},
    TensorData,
};

/// Magic bytes for v3 format identification.
const V3_MAGIC: [u8; 4] = *b"NEUM";

/// Current version number.
const CURRENT_VERSION: u32 = 3;

/// Flag indicating compressed payload.
const FLAG_COMPRESSED: u32 = 0x01;

/// Default Zstd compression level for snapshots.
const SNAPSHOT_COMPRESSION_LEVEL: i32 = 3;

/// Snapshot format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotVersion {
    /// Legacy HashMap-based format.
    V2,
    /// SlabRouter-based format with magic header.
    V3,
}

/// Size of the raw header in bytes (magic + version + flags + `entry_count`).
const HEADER_SIZE: usize = 4 + 4 + 4 + 8;

/// Header for v3 snapshot format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotHeader {
    /// Magic bytes for format identification.
    pub magic: [u8; 4],
    /// Format version number.
    pub version: u32,
    /// Reserved flags.
    pub flags: u32,
    /// Total entry count.
    pub entry_count: u64,
}

impl SnapshotHeader {
    /// Serialize header to raw bytes (fixed 20-byte format).
    fn to_raw_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..20].copy_from_slice(&self.entry_count.to_le_bytes());
        buf
    }

    /// Deserialize header from raw bytes.
    const fn from_raw_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        Self {
            magic: [buf[0], buf[1], buf[2], buf[3]],
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            flags: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            entry_count: u64::from_le_bytes([
                buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
            ]),
        }
    }
}

impl SnapshotHeader {
    /// Creates a new v3 snapshot header (uncompressed).
    #[must_use]
    pub const fn new(entry_count: u64) -> Self {
        Self {
            magic: V3_MAGIC,
            version: CURRENT_VERSION,
            flags: 0,
            entry_count,
        }
    }

    /// Creates a new v3 snapshot header with compression flag.
    #[must_use]
    pub const fn new_compressed(entry_count: u64) -> Self {
        Self {
            magic: V3_MAGIC,
            version: CURRENT_VERSION,
            flags: FLAG_COMPRESSED,
            entry_count,
        }
    }

    /// Returns true if the snapshot is compressed.
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        self.flags & FLAG_COMPRESSED != 0
    }

    /// # Errors
    ///
    /// Returns an error if the magic bytes are invalid or the version is unsupported.
    pub fn validate(&self) -> Result<(), SnapshotFormatError> {
        if self.magic != V3_MAGIC {
            return Err(SnapshotFormatError::InvalidMagic);
        }
        if self.version != CURRENT_VERSION {
            return Err(SnapshotFormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// V3 snapshot container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V3Snapshot {
    /// Snapshot header with version info.
    pub header: SnapshotHeader,
    /// Slab router state.
    pub router: SlabRouterSnapshot,
}

/// Errors from snapshot operations.
#[derive(Debug, Clone)]
pub enum SnapshotFormatError {
    /// Invalid magic bytes.
    InvalidMagic,
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// I/O error.
    IoError(String),
    /// Serialization error.
    SerializationError(String),
}

impl std::fmt::Display for SnapshotFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::IoError(msg) => write!(f, "io error: {msg}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for SnapshotFormatError {}

impl From<std::io::Error> for SnapshotFormatError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

impl From<bitcode::Error> for SnapshotFormatError {
    fn from(e: bitcode::Error) -> Self {
        Self::SerializationError(e.to_string())
    }
}

/// Detect the snapshot format version from a file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened.
#[instrument(skip(path), fields(path = %path.as_ref().display()))]
pub fn detect_version<P: AsRef<Path>>(path: P) -> Result<SnapshotVersion, SnapshotFormatError> {
    let mut file = File::open(path.as_ref())?;
    let mut magic = [0u8; 4];

    if file.read_exact(&mut magic).is_err() {
        return Ok(SnapshotVersion::V2);
    }

    if magic == V3_MAGIC {
        Ok(SnapshotVersion::V3)
    } else {
        Ok(SnapshotVersion::V2)
    }
}

/// Save a `SlabRouter` to v3 snapshot format (compressed by default).
///
/// # Errors
///
/// Returns an error if the file cannot be created or serialization fails.
#[instrument(skip(router), fields(path = %path.as_ref().display()))]
pub fn save_v3<P: AsRef<Path>>(router: &SlabRouter, path: P) -> Result<(), SnapshotFormatError> {
    save_v3_with_compression(router, path, true)
}

/// Save a `SlabRouter` to v3 snapshot format without compression.
///
/// # Errors
///
/// Returns an error if the file cannot be created or serialization fails.
pub fn save_v3_uncompressed<P: AsRef<Path>>(
    router: &SlabRouter,
    path: P,
) -> Result<(), SnapshotFormatError> {
    save_v3_with_compression(router, path, false)
}

/// Save a `SlabRouter` to v3 snapshot format with optional compression.
fn save_v3_with_compression<P: AsRef<Path>>(
    router: &SlabRouter,
    path: P,
    compress: bool,
) -> Result<(), SnapshotFormatError> {
    let path = path.as_ref();
    let temp_path = path.with_extension("tmp");

    let router_snapshot = router.snapshot();
    // Estimate total entry count from various slabs
    let entry_count = (router.len() + router.index.len()) as u64;

    let header = if compress {
        SnapshotHeader::new_compressed(entry_count)
    } else {
        SnapshotHeader::new(entry_count)
    };

    let snapshot = V3Snapshot {
        header,
        router: router_snapshot,
    };

    let mut file = File::create(&temp_path)?;

    // Write header as raw bytes (fixed size for easy parsing)
    file.write_all(&snapshot.header.to_raw_bytes())?;

    // Serialize router data
    let router_bytes = bitcode::serialize(&snapshot.router)?;

    if compress {
        let compressed = zstd::encode_all(&router_bytes[..], SNAPSHOT_COMPRESSION_LEVEL)
            .map_err(|e| SnapshotFormatError::IoError(e.to_string()))?;
        file.write_all(&compressed)?;
    } else {
        file.write_all(&router_bytes)?;
    }

    std::fs::rename(&temp_path, path)?;

    Ok(())
}

/// Load a `SlabRouter` from snapshot (auto-detecting version).
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the format is invalid, or deserialization fails.
#[instrument(fields(path = %path.as_ref().display()))]
pub fn load<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let path = path.as_ref();
    match detect_version(path)? {
        SnapshotVersion::V2 => load_v2(path),
        SnapshotVersion::V3 => load_v3(path),
    }
}

/// Load from v2 format (HashMap-based).
fn load_v2<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let data: HashMap<String, TensorData> = bitcode::deserialize(&bytes)?;

    let router = SlabRouter::new();
    for (key, value) in data {
        // Best-effort: put each key into the router
        if let Err(e) = router.put(&key, value) {
            tracing::warn!(
                key = %key,
                error = %e,
                "Failed to restore entry during v2 snapshot load"
            );
        }
    }

    Ok(router)
}

/// Load from v3 format (handles both compressed and uncompressed).
#[instrument(fields(path = %path.as_ref().display()))]
fn load_v3<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let mut file = File::open(path.as_ref())?;

    // Read the fixed-size header
    let mut header_buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_buf)?;
    let header = SnapshotHeader::from_raw_bytes(&header_buf);
    header.validate()?;

    if header.is_compressed() {
        // Read remaining compressed data
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        // Decompress and deserialize router
        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| SnapshotFormatError::IoError(e.to_string()))?;
        let router: SlabRouterSnapshot = bitcode::deserialize(&decompressed)?;

        Ok(SlabRouter::restore(router))
    } else {
        // Uncompressed: read remaining router data
        let mut router_bytes = Vec::new();
        file.read_to_end(&mut router_bytes)?;
        let router: SlabRouterSnapshot = bitcode::deserialize(&router_bytes)?;

        Ok(SlabRouter::restore(router))
    }
}

/// Convert v2 snapshot to v3 format.
///
/// # Errors
///
/// Returns an error if loading v2 or saving v3 fails.
pub fn migrate_v2_to_v3<P: AsRef<Path>, Q: AsRef<Path>>(
    v2_path: P,
    v3_path: Q,
) -> Result<(), SnapshotFormatError> {
    let router = load_v2(v2_path)?;
    save_v3(&router, v3_path)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::{ScalarValue, TensorValue};

    fn create_test_data() -> TensorData {
        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("test".to_string())),
        );
        data.set("count", TensorValue::Scalar(ScalarValue::Int(42)));
        data
    }

    #[test]
    fn test_header_new() {
        let header = SnapshotHeader::new(100);
        assert_eq!(header.magic, V3_MAGIC);
        assert_eq!(header.version, CURRENT_VERSION);
        assert_eq!(header.entry_count, 100);
    }

    #[test]
    fn test_header_validate() {
        let header = SnapshotHeader::new(100);
        assert!(header.validate().is_ok());

        let invalid = SnapshotHeader {
            magic: *b"XXXX",
            version: 3,
            flags: 0,
            entry_count: 0,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_header_validate_version() {
        let header = SnapshotHeader {
            magic: V3_MAGIC,
            version: 99,
            flags: 0,
            entry_count: 0,
        };
        let result = header.validate();
        assert!(matches!(
            result,
            Err(SnapshotFormatError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn test_save_load_v3() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.v3.bin");

        let router = SlabRouter::new();
        router.put("user:1", create_test_data()).unwrap();
        router.put("user:2", create_test_data()).unwrap();

        save_v3(&router, &path).unwrap();

        let loaded = load(&path).unwrap();
        assert!(loaded.exists("user:1"));
        assert!(loaded.exists("user:2"));
    }

    #[test]
    fn test_detect_version_v3() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.v3.bin");

        let router = SlabRouter::new();
        router.put("key", create_test_data()).unwrap();
        save_v3(&router, &path).unwrap();

        assert_eq!(detect_version(&path).unwrap(), SnapshotVersion::V3);
    }

    #[test]
    fn test_detect_version_v2() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.v2.bin");

        // Create v2 format (plain HashMap)
        let mut data = HashMap::new();
        data.insert("key".to_string(), create_test_data());

        let mut file = File::create(&path).unwrap();
        let bytes = bitcode::serialize(&data).unwrap();
        file.write_all(&bytes).unwrap();

        assert_eq!(detect_version(&path).unwrap(), SnapshotVersion::V2);
    }

    #[test]
    fn test_load_v2_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.v2.bin");

        // Create v2 format
        let mut data = HashMap::new();
        data.insert("user:1".to_string(), create_test_data());
        data.insert("user:2".to_string(), create_test_data());

        let mut file = File::create(&path).unwrap();
        let bytes = bitcode::serialize(&data).unwrap();
        file.write_all(&bytes).unwrap();

        // Load with auto-detection
        let loaded = load(&path).unwrap();
        assert!(loaded.exists("user:1"));
        assert!(loaded.exists("user:2"));
    }

    #[test]
    fn test_migrate_v2_to_v3() {
        let dir = tempdir().unwrap();
        let v2_path = dir.path().join("test.v2.bin");
        let v3_path = dir.path().join("test.v3.bin");

        // Create v2 format
        let mut data = HashMap::new();
        data.insert("key1".to_string(), create_test_data());
        data.insert("key2".to_string(), create_test_data());

        let mut file = File::create(&v2_path).unwrap();
        let bytes = bitcode::serialize(&data).unwrap();
        file.write_all(&bytes).unwrap();

        // Migrate
        migrate_v2_to_v3(&v2_path, &v3_path).unwrap();

        // Verify v3 format
        assert_eq!(detect_version(&v3_path).unwrap(), SnapshotVersion::V3);

        let loaded = load(&v3_path).unwrap();
        assert!(loaded.exists("key1"));
        assert!(loaded.exists("key2"));
    }

    #[test]
    fn test_error_display() {
        let err = SnapshotFormatError::InvalidMagic;
        assert!(err.to_string().contains("invalid magic"));

        let err = SnapshotFormatError::UnsupportedVersion(99);
        assert!(err.to_string().contains("99"));

        let err = SnapshotFormatError::IoError("test".to_string());
        assert!(err.to_string().contains("io error"));

        let err = SnapshotFormatError::SerializationError("test".to_string());
        assert!(err.to_string().contains("serialization"));
    }

    #[test]
    fn test_save_with_embeddings() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.emb.bin");

        let router = SlabRouter::new();

        // Add embedding
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding.clone()));
        router.put("emb:vec1", data).unwrap();

        save_v3(&router, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert!(loaded.exists("emb:vec1"));
    }

    #[test]
    fn test_save_with_cache() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cache.bin");

        let router = SlabRouter::new();
        router.put("_cache:item1", create_test_data()).unwrap();
        router.put("_cache:item2", create_test_data()).unwrap();

        save_v3(&router, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert!(loaded.exists("_cache:item1"));
        assert!(loaded.exists("_cache:item2"));
    }

    #[test]
    fn test_save_with_blobs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.blob.bin");

        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "_data",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3, 4, 5])),
        );
        router.put("_blob:chunk1", data).unwrap();

        save_v3(&router, &path).unwrap();
        let loaded = load(&path).unwrap();

        // Blob metadata should exist
        assert!(loaded.exists("_blob:chunk1"));
    }

    #[test]
    fn test_roundtrip_all_types() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.all.bin");

        let router = SlabRouter::new();

        // Metadata
        router.put("meta:1", create_test_data()).unwrap();

        // Cache
        router.put("_cache:item", create_test_data()).unwrap();

        // Graph
        router.put("node:1", create_test_data()).unwrap();
        router.put("edge:1", create_test_data()).unwrap();

        // Table
        router.put("table:users", create_test_data()).unwrap();

        save_v3(&router, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert!(loaded.exists("meta:1"));
        assert!(loaded.exists("_cache:item"));
        assert!(loaded.exists("node:1"));
        assert!(loaded.exists("edge:1"));
        assert!(loaded.exists("table:users"));
    }

    #[test]
    fn test_empty_router_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.empty.bin");

        let router = SlabRouter::new();

        save_v3(&router, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn test_detect_version_nonexistent_file() {
        let result = detect_version("/nonexistent/path/to/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load("/nonexistent/path/to/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_header_debug() {
        let header = SnapshotHeader::new(100);
        let debug = format!("{:?}", header);
        assert!(debug.contains("SnapshotHeader"));
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_v3_snapshot_debug() {
        let router = SlabRouter::new();
        let snapshot = V3Snapshot {
            header: SnapshotHeader::new(0),
            router: router.snapshot(),
        };
        let debug = format!("{:?}", snapshot);
        assert!(debug.contains("V3Snapshot"));
        assert!(debug.contains("header"));
    }

    #[test]
    fn test_error_clone() {
        let err = SnapshotFormatError::InvalidMagic;
        let cloned = err.clone();
        assert!(matches!(cloned, SnapshotFormatError::InvalidMagic));

        let err = SnapshotFormatError::UnsupportedVersion(42);
        let cloned = err.clone();
        assert!(matches!(
            cloned,
            SnapshotFormatError::UnsupportedVersion(42)
        ));

        let err = SnapshotFormatError::IoError("test".to_string());
        let cloned = err.clone();
        assert!(matches!(cloned, SnapshotFormatError::IoError(_)));

        let err = SnapshotFormatError::SerializationError("test".to_string());
        let cloned = err.clone();
        assert!(matches!(cloned, SnapshotFormatError::SerializationError(_)));
    }

    #[test]
    fn test_load_v2_empty_hashmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.v2.empty.bin");

        // Create empty v2 format
        let data: HashMap<String, TensorData> = HashMap::new();
        let mut file = File::create(&path).unwrap();
        let bytes = bitcode::serialize(&data).unwrap();
        file.write_all(&bytes).unwrap();

        let loaded = load(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: SnapshotFormatError = io_err.into();
        assert!(matches!(err, SnapshotFormatError::IoError(_)));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_snapshot_version_equality() {
        assert_eq!(SnapshotVersion::V2, SnapshotVersion::V2);
        assert_eq!(SnapshotVersion::V3, SnapshotVersion::V3);
        assert_ne!(SnapshotVersion::V2, SnapshotVersion::V3);
    }

    #[test]
    fn test_snapshot_version_debug() {
        let v2 = SnapshotVersion::V2;
        let v3 = SnapshotVersion::V3;
        assert!(format!("{:?}", v2).contains("V2"));
        assert!(format!("{:?}", v3).contains("V3"));
    }

    #[test]
    fn test_snapshot_version_clone() {
        let v2 = SnapshotVersion::V2;
        let cloned = v2.clone();
        assert_eq!(v2, cloned);
    }

    #[test]
    fn test_header_clone() {
        let header = SnapshotHeader::new(100);
        let cloned = header.clone();
        assert_eq!(header.magic, cloned.magic);
        assert_eq!(header.version, cloned.version);
        assert_eq!(header.entry_count, cloned.entry_count);
    }

    #[test]
    fn test_header_compressed() {
        let header = SnapshotHeader::new_compressed(100);
        assert!(header.is_compressed());
        assert_eq!(header.flags & FLAG_COMPRESSED, FLAG_COMPRESSED);

        let uncompressed = SnapshotHeader::new(100);
        assert!(!uncompressed.is_compressed());
        assert_eq!(uncompressed.flags, 0);
    }

    #[test]
    fn test_save_v3_uncompressed_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("uncompressed.snap");

        let router = SlabRouter::new();
        router.put("user:1", create_test_data()).unwrap();
        router.put("user:2", create_test_data()).unwrap();

        // Save uncompressed
        save_v3_uncompressed(&router, &path).unwrap();

        // Load and verify
        let loaded = load(&path).unwrap();
        assert!(loaded.exists("user:1"));
        assert!(loaded.exists("user:2"));
    }

    #[test]
    fn test_save_v3_compressed_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("compressed.snap");

        let router = SlabRouter::new();
        for i in 0..100 {
            let mut data = create_test_data();
            data.set("index", TensorValue::Scalar(ScalarValue::Int(i)));
            router.put(&format!("user:{}", i), data).unwrap();
        }

        // Save compressed (default)
        save_v3(&router, &path).unwrap();

        // Load and verify
        let loaded = load(&path).unwrap();
        for i in 0..100 {
            assert!(loaded.exists(&format!("user:{}", i)));
        }
    }

    #[test]
    fn test_compressed_vs_uncompressed_size() {
        let dir = tempdir().unwrap();
        let compressed_path = dir.path().join("compressed.snap");
        let uncompressed_path = dir.path().join("uncompressed.snap");

        let router = SlabRouter::new();
        // Add repetitive data (compresses well)
        for i in 0..500 {
            let mut data = TensorData::new();
            data.set(
                "name",
                TensorValue::Scalar(ScalarValue::String("test_user".to_string())),
            );
            data.set("embedding", TensorValue::Vector(vec![0.5; 128]));
            router.put(&format!("emb:{}", i), data).unwrap();
        }

        save_v3(&router, &compressed_path).unwrap();
        save_v3_uncompressed(&router, &uncompressed_path).unwrap();

        let compressed_size = std::fs::metadata(&compressed_path).unwrap().len();
        let uncompressed_size = std::fs::metadata(&uncompressed_path).unwrap().len();

        // Compressed should be smaller for repetitive data
        assert!(
            compressed_size < uncompressed_size,
            "Compressed {} should be less than uncompressed {}",
            compressed_size,
            uncompressed_size
        );

        // Both should load correctly
        let loaded_compressed = load(&compressed_path).unwrap();
        let loaded_uncompressed = load(&uncompressed_path).unwrap();

        assert_eq!(loaded_compressed.len(), 500);
        assert_eq!(loaded_uncompressed.len(), 500);
    }
}
