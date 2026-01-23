//! Snapshot format v2/v3 with backward compatibility.
//!
//! Format evolution:
//! - v2: `HashMap<String, TensorData>` serialized with bincode
//! - v3: `SlabRouterSnapshot` with specialized slab storage
//!
//! The loader automatically detects format version and loads appropriately.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    slab_router::{SlabRouter, SlabRouterSnapshot},
    TensorData,
};

/// Magic bytes for v3 format identification.
const V3_MAGIC: [u8; 4] = *b"NEUM";

/// Current version number.
const CURRENT_VERSION: u32 = 3;

/// Snapshot format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotVersion {
    V2,
    V3,
}

/// Header for v3 snapshot format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u32,
    pub entry_count: u64,
}

impl SnapshotHeader {
    #[must_use]
    pub const fn new(entry_count: u64) -> Self {
        Self {
            magic: V3_MAGIC,
            version: CURRENT_VERSION,
            flags: 0,
            entry_count,
        }
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
    pub header: SnapshotHeader,
    pub router: SlabRouterSnapshot,
}

/// Errors from snapshot operations.
#[derive(Debug, Clone)]
pub enum SnapshotFormatError {
    InvalidMagic,
    UnsupportedVersion(u32),
    IoError(String),
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

impl From<bincode::Error> for SnapshotFormatError {
    fn from(e: bincode::Error) -> Self {
        Self::SerializationError(e.to_string())
    }
}

/// Detect the snapshot format version from a file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened.
pub fn detect_version<P: AsRef<Path>>(path: P) -> Result<SnapshotVersion, SnapshotFormatError> {
    let mut file = File::open(path)?;
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

/// Save a `SlabRouter` to v3 snapshot format.
///
/// # Errors
///
/// Returns an error if the file cannot be created or serialization fails.
pub fn save_v3<P: AsRef<Path>>(router: &SlabRouter, path: P) -> Result<(), SnapshotFormatError> {
    let path = path.as_ref();
    let temp_path = path.with_extension("tmp");

    let router_snapshot = router.snapshot();
    // Estimate total entry count from various slabs
    let entry_count = (router.len() + router.index.len()) as u64;

    let snapshot = V3Snapshot {
        header: SnapshotHeader::new(entry_count),
        router: router_snapshot,
    };

    let file = File::create(&temp_path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &snapshot)?;

    std::fs::rename(&temp_path, path)?;

    Ok(())
}

/// Load a `SlabRouter` from snapshot (auto-detecting version).
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the format is invalid, or deserialization fails.
pub fn load<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let path = path.as_ref();
    match detect_version(path)? {
        SnapshotVersion::V2 => load_v2(path),
        SnapshotVersion::V3 => load_v3(path),
    }
}

/// Load from v2 format (HashMap-based).
fn load_v2<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data: HashMap<String, TensorData> = bincode::deserialize_from(reader)?;

    let router = SlabRouter::new();
    for (key, value) in data {
        // Best-effort: put each key into the router
        let _ = router.put(&key, value);
    }

    Ok(router)
}

/// Load from v3 format.
fn load_v3<P: AsRef<Path>>(path: P) -> Result<SlabRouter, SnapshotFormatError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let snapshot: V3Snapshot = bincode::deserialize_from(reader)?;

    snapshot.header.validate()?;

    Ok(SlabRouter::restore(snapshot.router))
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

        let file = File::create(&path).unwrap();
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).unwrap();

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

        let file = File::create(&path).unwrap();
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).unwrap();

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

        let file = File::create(&v2_path).unwrap();
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).unwrap();

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
        let file = File::create(&path).unwrap();
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).unwrap();

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
}
