use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::Path,
};

use memmap2::{Mmap, MmapMut};

use crate::TensorData;

const MAGIC: &[u8; 4] = b"MMAP";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 16;

#[derive(Debug)]
pub enum MmapError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    Serialization(String),
    NotFound(String),
    EmptyFile,
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::Io(e) => write!(f, "I/O error: {}", e),
            MmapError::InvalidMagic => write!(f, "Invalid file magic"),
            MmapError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            MmapError::Serialization(s) => write!(f, "Serialization error: {}", s),
            MmapError::NotFound(k) => write!(f, "Key not found: {}", k),
            MmapError::EmptyFile => write!(f, "Empty file"),
        }
    }
}

impl std::error::Error for MmapError {}

impl From<io::Error> for MmapError {
    fn from(e: io::Error) -> Self {
        MmapError::Io(e)
    }
}

impl From<bincode::Error> for MmapError {
    fn from(e: bincode::Error) -> Self {
        MmapError::Serialization(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, MmapError>;

/// Entry location in the memory-mapped file.
#[derive(Clone, Copy)]
struct EntryLocation {
    data_offset: usize,
    data_len: usize,
}

/// Memory-mapped read-only store for cold tensor data.
///
/// File format:
/// ```text
/// [Header: 16 bytes]
///   Magic: 4 bytes "MMAP"
///   Version: 4 bytes (little-endian u32)
///   Entry count: 8 bytes (little-endian u64)
///
/// [Entries: variable]
///   Key length: 4 bytes (little-endian u32)
///   Key: variable bytes (UTF-8)
///   Data length: 4 bytes (little-endian u32)
///   Data: variable bytes (bincode-serialized TensorData)
/// ```
pub struct MmapStore {
    mmap: Mmap,
    index: HashMap<String, EntryLocation>,
    entry_count: u64,
}

impl MmapStore {
    /// Open an existing mmap store file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Err(MmapError::EmptyFile);
        }

        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(MmapError::InvalidMagic);
        }

        // Validate header
        if &mmap[0..4] != MAGIC {
            return Err(MmapError::InvalidMagic);
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(MmapError::UnsupportedVersion(version));
        }

        let entry_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());

        // Build index by scanning entries
        let mut index = HashMap::new();
        let mut offset = HEADER_SIZE;

        for _ in 0..entry_count {
            if offset + 4 > mmap.len() {
                break;
            }

            let key_len = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + key_len > mmap.len() {
                break;
            }

            let key = String::from_utf8_lossy(&mmap[offset..offset + key_len]).to_string();
            offset += key_len;

            if offset + 4 > mmap.len() {
                break;
            }

            let data_len =
                u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            let data_offset = offset;
            offset += data_len;

            index.insert(
                key,
                EntryLocation {
                    data_offset,
                    data_len,
                },
            );
        }

        Ok(Self {
            mmap,
            index,
            entry_count,
        })
    }

    /// Get a tensor by key, deserializing on demand.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        let loc = self
            .index
            .get(key)
            .ok_or_else(|| MmapError::NotFound(key.to_string()))?;

        let data = &self.mmap[loc.data_offset..loc.data_offset + loc.data_len];
        let tensor: TensorData = bincode::deserialize(data)?;
        Ok(tensor)
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entry_count as usize
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    /// Get total memory-mapped size in bytes.
    pub fn mmap_size(&self) -> usize {
        self.mmap.len()
    }
}

/// Builder for creating mmap store files.
pub struct MmapStoreBuilder {
    file: File,
    entry_count: u64,
    current_offset: u64,
}

impl MmapStoreBuilder {
    /// Create a new mmap store file.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Write header with placeholder count
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?;

        Ok(Self {
            file,
            entry_count: 0,
            current_offset: HEADER_SIZE as u64,
        })
    }

    /// Add an entry to the store.
    pub fn add(&mut self, key: &str, tensor: &TensorData) -> Result<()> {
        let key_bytes = key.as_bytes();
        let data = bincode::serialize(tensor)?;

        // Write key length + key
        self.file
            .write_all(&(key_bytes.len() as u32).to_le_bytes())?;
        self.file.write_all(key_bytes)?;

        // Write data length + data
        self.file.write_all(&(data.len() as u32).to_le_bytes())?;
        self.file.write_all(&data)?;

        self.entry_count += 1;
        self.current_offset += 4 + key_bytes.len() as u64 + 4 + data.len() as u64;

        Ok(())
    }

    /// Finalize the store and return the file size.
    pub fn finish(mut self) -> Result<u64> {
        // Update entry count in header
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&self.entry_count.to_le_bytes())?;
        self.file.flush()?;

        Ok(self.current_offset)
    }
}

/// Mutable memory-mapped store for appending entries.
///
/// Uses a pre-allocated file that can be extended.
pub struct MmapStoreMut {
    file: File,
    mmap: MmapMut,
    index: HashMap<String, EntryLocation>,
    entry_count: u64,
    write_offset: usize,
    capacity: usize,
}

impl MmapStoreMut {
    /// Create a new mutable mmap store with initial capacity.
    pub fn create<P: AsRef<Path>>(path: P, initial_capacity: usize) -> Result<Self> {
        let capacity = initial_capacity.max(HEADER_SIZE + 1024);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(capacity as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        mmap[0..4].copy_from_slice(MAGIC);
        mmap[4..8].copy_from_slice(&VERSION.to_le_bytes());
        mmap[8..16].copy_from_slice(&0u64.to_le_bytes());

        Ok(Self {
            file,
            mmap,
            index: HashMap::new(),
            entry_count: 0,
            write_offset: HEADER_SIZE,
            capacity,
        })
    }

    /// Open an existing mutable mmap store.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Err(MmapError::EmptyFile);
        }

        let capacity = metadata.len() as usize;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(MmapError::InvalidMagic);
        }

        if &mmap[0..4] != MAGIC {
            return Err(MmapError::InvalidMagic);
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(MmapError::UnsupportedVersion(version));
        }

        let entry_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());

        // Build index
        let mut index = HashMap::new();
        let mut offset = HEADER_SIZE;

        for _ in 0..entry_count {
            if offset + 4 > mmap.len() {
                break;
            }

            let key_len = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + key_len > mmap.len() {
                break;
            }

            let key = String::from_utf8_lossy(&mmap[offset..offset + key_len]).to_string();
            offset += key_len;

            if offset + 4 > mmap.len() {
                break;
            }

            let data_len =
                u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            let data_offset = offset;
            offset += data_len;

            index.insert(
                key,
                EntryLocation {
                    data_offset,
                    data_len,
                },
            );
        }

        Ok(Self {
            file,
            mmap,
            index,
            entry_count,
            write_offset: offset,
            capacity,
        })
    }

    /// Insert or update an entry.
    ///
    /// Note: Updates append a new version; old data becomes garbage.
    /// Use compaction to reclaim space.
    pub fn insert(&mut self, key: &str, tensor: &TensorData) -> Result<()> {
        let key_bytes = key.as_bytes();
        let data = bincode::serialize(tensor)?;

        let entry_size = 4 + key_bytes.len() + 4 + data.len();

        // Grow file if needed
        if self.write_offset + entry_size > self.capacity {
            let new_capacity = (self.capacity * 2).max(self.write_offset + entry_size + 1024);
            self.file.set_len(new_capacity as u64)?;
            self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
            self.capacity = new_capacity;
        }

        // Write entry
        let offset = self.write_offset;
        self.mmap[offset..offset + 4].copy_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        self.mmap[offset + 4..offset + 4 + key_bytes.len()].copy_from_slice(key_bytes);
        let data_offset = offset + 4 + key_bytes.len() + 4;
        self.mmap[offset + 4 + key_bytes.len()..data_offset]
            .copy_from_slice(&(data.len() as u32).to_le_bytes());
        self.mmap[data_offset..data_offset + data.len()].copy_from_slice(&data);

        // Update index (new entry or replace old location)
        let is_new = !self.index.contains_key(key);
        self.index.insert(
            key.to_string(),
            EntryLocation {
                data_offset,
                data_len: data.len(),
            },
        );

        self.write_offset += entry_size;

        if is_new {
            self.entry_count += 1;
            // Update header count
            self.mmap[8..16].copy_from_slice(&self.entry_count.to_le_bytes());
        }

        Ok(())
    }

    /// Get a tensor by key.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        let loc = self
            .index
            .get(key)
            .ok_or_else(|| MmapError::NotFound(key.to_string()))?;

        let data = &self.mmap[loc.data_offset..loc.data_offset + loc.data_len];
        let tensor: TensorData = bincode::deserialize(data)?;
        Ok(tensor)
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Get current used size in bytes.
    pub fn used_size(&self) -> usize {
        self.write_offset
    }

    /// Get total allocated capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compact the store by removing garbage (old versions of updated keys).
    ///
    /// Returns the new file size after compaction.
    pub fn compact<P: AsRef<Path>>(&self, output_path: P) -> Result<u64> {
        let mut builder = MmapStoreBuilder::create(output_path)?;

        for key in self.index.keys() {
            let tensor = self.get(key)?;
            builder.add(key, &tensor)?;
        }

        builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{ScalarValue, TensorValue};

    fn create_test_tensor(id: i64) -> TensorData {
        let mut tensor = TensorData::new();
        tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
        );
        tensor.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4]));
        tensor
    }

    #[test]
    fn test_mmap_store_builder_and_read() {
        let path = "/tmp/test_mmap_store.bin";
        let _ = fs::remove_file(path);

        // Build store
        {
            let mut builder = MmapStoreBuilder::create(path).unwrap();
            for i in 0..100 {
                let tensor = create_test_tensor(i);
                builder.add(&format!("key_{}", i), &tensor).unwrap();
            }
            let size = builder.finish().unwrap();
            assert!(size > HEADER_SIZE as u64);
        }

        // Read store
        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 100);

            for i in 0..100 {
                let key = format!("key_{}", i);
                assert!(store.contains(&key));

                let tensor = store.get(&key).unwrap();
                match tensor.get("id") {
                    Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                        assert_eq!(*id, i);
                    },
                    _ => panic!("Expected int id"),
                }
            }

            assert!(!store.contains("nonexistent"));
            assert!(store.get("nonexistent").is_err());
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut() {
        let path = "/tmp/test_mmap_store_mut.bin";
        let _ = fs::remove_file(path);

        // Create and write
        {
            let mut store = MmapStoreMut::create(path, 4096).unwrap();
            for i in 0..50 {
                let tensor = create_test_tensor(i);
                store.insert(&format!("key_{}", i), &tensor).unwrap();
            }
            store.flush().unwrap();
            assert_eq!(store.len(), 50);
        }

        // Reopen and add more
        {
            let mut store = MmapStoreMut::open(path).unwrap();
            assert_eq!(store.len(), 50);

            for i in 50..100 {
                let tensor = create_test_tensor(i);
                store.insert(&format!("key_{}", i), &tensor).unwrap();
            }
            store.flush().unwrap();
            assert_eq!(store.len(), 100);
        }

        // Verify all data
        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 100);

            for i in 0..100 {
                let tensor = store.get(&format!("key_{}", i)).unwrap();
                match tensor.get("id") {
                    Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                        assert_eq!(*id, i);
                    },
                    _ => panic!("Expected int id"),
                }
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_update_and_compact() {
        let path = "/tmp/test_mmap_compact.bin";
        let compact_path = "/tmp/test_mmap_compact_out.bin";
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(compact_path);

        // Create with updates (creates garbage)
        {
            let mut store = MmapStoreMut::create(path, 4096).unwrap();

            // Initial insert
            for i in 0..10 {
                let tensor = create_test_tensor(i);
                store.insert(&format!("key_{}", i), &tensor).unwrap();
            }

            // Update some keys (creates garbage)
            for i in 0..5 {
                let tensor = create_test_tensor(i + 1000);
                store.insert(&format!("key_{}", i), &tensor).unwrap();
            }

            store.flush().unwrap();
            let used_before = store.used_size();

            // Compact
            store.compact(compact_path).unwrap();

            // Verify compacted store is smaller
            let compacted = MmapStore::open(compact_path).unwrap();
            assert_eq!(compacted.len(), 10);
            assert!(compacted.mmap_size() < used_before);

            // Verify updated values
            for i in 0..5 {
                let tensor = compacted.get(&format!("key_{}", i)).unwrap();
                match tensor.get("id") {
                    Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                        assert_eq!(*id, i + 1000);
                    },
                    _ => panic!("Expected int id"),
                }
            }
        }

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(compact_path);
    }

    #[test]
    fn test_mmap_store_empty_file() {
        let path = "/tmp/test_mmap_empty.bin";
        let _ = fs::remove_file(path);

        // Create empty file
        File::create(path).unwrap();

        // Should fail to open
        assert!(matches!(MmapStore::open(path), Err(MmapError::EmptyFile)));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_invalid_magic() {
        let path = "/tmp/test_mmap_invalid.bin";
        let _ = fs::remove_file(path);

        // Create file with invalid magic
        let mut file = File::create(path).unwrap();
        file.write_all(b"BAAD").unwrap();
        file.write_all(&[0u8; 12]).unwrap();
        drop(file);

        assert!(matches!(
            MmapStore::open(path),
            Err(MmapError::InvalidMagic)
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_large_entries() {
        let path = "/tmp/test_mmap_large.bin";
        let _ = fs::remove_file(path);

        // Create tensor with large embedding
        let mut tensor = TensorData::new();
        tensor.set("embedding", TensorValue::Vector(vec![0.5; 1024]));

        {
            let mut store = MmapStoreMut::create(path, 1024).unwrap();

            // This should trigger file growth
            for i in 0..100 {
                store.insert(&format!("large_{}", i), &tensor).unwrap();
            }

            store.flush().unwrap();
            assert!(store.capacity() > 1024);
        }

        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 100);

            let loaded = store.get("large_0").unwrap();
            match loaded.get("embedding") {
                Some(TensorValue::Vector(v)) => {
                    assert_eq!(v.len(), 1024);
                    assert!((v[0] - 0.5).abs() < 1e-6);
                },
                _ => panic!("Expected vector"),
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_error_display() {
        let io_err = MmapError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let magic_err = MmapError::InvalidMagic;
        assert!(magic_err.to_string().contains("Invalid file magic"));

        let ver_err = MmapError::UnsupportedVersion(99);
        assert!(ver_err.to_string().contains("Unsupported version: 99"));

        let ser_err = MmapError::Serialization("test".to_string());
        assert!(ser_err.to_string().contains("Serialization error: test"));

        let not_found = MmapError::NotFound("key1".to_string());
        assert!(not_found.to_string().contains("Key not found: key1"));

        let empty = MmapError::EmptyFile;
        assert!(empty.to_string().contains("Empty file"));
    }

    #[test]
    fn test_mmap_error_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(MmapError::NotFound("test".to_string()));
        assert!(err.to_string().contains("Key not found"));
    }

    #[test]
    fn test_mmap_store_keys_iteration() {
        let path = "/tmp/test_mmap_keys.bin";
        let _ = fs::remove_file(path);

        {
            let mut builder = MmapStoreBuilder::create(path).unwrap();
            for i in 0..5 {
                builder
                    .add(&format!("key_{}", i), &create_test_tensor(i))
                    .unwrap();
            }
            builder.finish().unwrap();
        }

        {
            let store = MmapStore::open(path).unwrap();
            let keys: Vec<_> = store.keys().collect();
            assert_eq!(keys.len(), 5);
            assert!(store.mmap_size() > HEADER_SIZE);
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_keys_and_len() {
        let path = "/tmp/test_mmap_mut_keys.bin";
        let _ = fs::remove_file(path);

        {
            let mut store = MmapStoreMut::create(path, 4096).unwrap();
            assert!(store.is_empty());

            for i in 0..5 {
                store
                    .insert(&format!("key_{}", i), &create_test_tensor(i))
                    .unwrap();
            }

            assert_eq!(store.len(), 5);
            assert!(!store.is_empty());

            let keys: Vec<_> = store.keys().collect();
            assert_eq!(keys.len(), 5);

            assert!(store.used_size() > HEADER_SIZE);
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_update() {
        let path = "/tmp/test_mmap_mut_update.bin";
        let _ = fs::remove_file(path);

        {
            let mut store = MmapStoreMut::create(path, 4096).unwrap();
            store.insert("key_1", &create_test_tensor(1)).unwrap();
            store.insert("key_1", &create_test_tensor(100)).unwrap();

            // Count should stay 1
            assert_eq!(store.len(), 1);

            // Value should be updated
            let tensor = store.get("key_1").unwrap();
            match tensor.get("id") {
                Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                    assert_eq!(*id, 100);
                },
                _ => panic!("Expected int id"),
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_reopen() {
        let path = "/tmp/test_mmap_mut_reopen.bin";
        let _ = fs::remove_file(path);

        {
            let mut store = MmapStoreMut::create(path, 4096).unwrap();
            for i in 0..5 {
                store
                    .insert(&format!("key_{}", i), &create_test_tensor(i))
                    .unwrap();
            }
            store.flush().unwrap();
        }

        {
            let mut store = MmapStoreMut::open(path).unwrap();
            assert_eq!(store.len(), 5);
            assert!(store.contains("key_0"));
            assert!(!store.contains("nonexistent"));

            // Add more
            for i in 5..10 {
                store
                    .insert(&format!("key_{}", i), &create_test_tensor(i))
                    .unwrap();
            }
            store.flush().unwrap();
        }

        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 10);
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_empty_open() {
        let path = "/tmp/test_mmap_mut_empty.bin";
        let _ = fs::remove_file(path);

        File::create(path).unwrap();
        assert!(matches!(
            MmapStoreMut::open(path),
            Err(MmapError::EmptyFile)
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_invalid_magic() {
        let path = "/tmp/test_mmap_mut_invalid.bin";
        let _ = fs::remove_file(path);

        let mut file = File::create(path).unwrap();
        file.write_all(b"BAAD").unwrap();
        file.write_all(&[0u8; 12]).unwrap();
        drop(file);

        assert!(matches!(
            MmapStoreMut::open(path),
            Err(MmapError::InvalidMagic)
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_unsupported_version() {
        let path = "/tmp/test_mmap_unsupported.bin";
        let _ = fs::remove_file(path);

        let mut file = File::create(path).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(&99u32.to_le_bytes()).unwrap(); // Invalid version
        file.write_all(&0u64.to_le_bytes()).unwrap();
        drop(file);

        assert!(matches!(
            MmapStore::open(path),
            Err(MmapError::UnsupportedVersion(99))
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_unicode_keys() {
        let path = "/tmp/test_mmap_unicode.bin";
        let _ = fs::remove_file(path);

        {
            let mut builder = MmapStoreBuilder::create(path).unwrap();
            builder.add("键_中文", &create_test_tensor(1)).unwrap();
            builder.add("キー_日本語", &create_test_tensor(2)).unwrap();
            builder.add("ключ_русский", &create_test_tensor(3)).unwrap();
            builder.finish().unwrap();
        }

        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 3);
            assert!(store.contains("键_中文"));
            assert!(store.contains("キー_日本語"));
            assert!(store.contains("ключ_русский"));
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_builder_empty() {
        let path = "/tmp/test_mmap_builder_empty.bin";
        let _ = fs::remove_file(path);

        {
            let builder = MmapStoreBuilder::create(path).unwrap();
            builder.finish().unwrap();
        }

        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 0);
            assert!(store.is_empty());
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_error_debug() {
        let err = MmapError::NotFound("test_key".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("NotFound"));
        assert!(debug.contains("test_key"));

        let err = MmapError::InvalidMagic;
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidMagic"));

        let err = MmapError::UnsupportedVersion(42);
        let debug = format!("{:?}", err);
        assert!(debug.contains("UnsupportedVersion"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_mmap_store_single_entry() {
        let path = "/tmp/test_mmap_single.bin";
        let _ = fs::remove_file(path);

        {
            let mut builder = MmapStoreBuilder::create(path).unwrap();
            builder.add("only_key", &create_test_tensor(42)).unwrap();
            builder.finish().unwrap();
        }

        {
            let store = MmapStore::open(path).unwrap();
            assert_eq!(store.len(), 1);
            assert!(!store.is_empty());
            let keys: Vec<_> = store.keys().collect();
            assert_eq!(keys.len(), 1);
            assert_eq!(keys[0], "only_key");
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_mut_get_nonexistent() {
        let path = "/tmp/test_mmap_mut_get_nonexistent.bin";
        let _ = fs::remove_file(path);

        {
            let mut store = MmapStoreMut::create(path, 1024).unwrap();
            store.insert("key1", &create_test_tensor(1)).unwrap();

            let result = store.get("nonexistent");
            assert!(result.is_err());
            assert!(matches!(result, Err(MmapError::NotFound(_))));
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_capacity_growth() {
        let path = "/tmp/test_mmap_growth.bin";
        let _ = fs::remove_file(path);

        {
            // Start with small capacity
            let mut store = MmapStoreMut::create(path, 64).unwrap();
            let initial_capacity = store.capacity();

            // Add data that exceeds initial capacity
            for i in 0..50 {
                store
                    .insert(&format!("key_{}", i), &create_test_tensor(i))
                    .unwrap();
            }

            // Capacity should have grown
            assert!(store.capacity() > initial_capacity);
            assert_eq!(store.len(), 50);
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_store_entry_location_copy() {
        let loc1 = EntryLocation {
            data_offset: 100,
            data_len: 50,
        };
        let loc2 = loc1; // Copy
        assert_eq!(loc1.data_offset, loc2.data_offset);
        assert_eq!(loc1.data_len, loc2.data_len);
    }

    #[test]
    fn test_mmap_error_source() {
        use std::error::Error;

        // MmapError does not implement source(), so all variants return None
        let io_err = MmapError::Io(io::Error::new(io::ErrorKind::NotFound, "not found"));
        assert!(io_err.source().is_none());

        let magic_err = MmapError::InvalidMagic;
        assert!(magic_err.source().is_none());

        let ser_err = MmapError::Serialization("test".to_string());
        assert!(ser_err.source().is_none());

        let not_found = MmapError::NotFound("key".to_string());
        assert!(not_found.source().is_none());
    }
}
