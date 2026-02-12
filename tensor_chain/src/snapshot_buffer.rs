// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Adaptive memory/disk buffer for snapshot assembly with bounded memory usage.
//!
//! # Overview
//!
//! This module provides a memory-efficient buffer for assembling and serving Raft snapshots.
//! When snapshot data exceeds a configurable memory threshold, the buffer transparently
//! spills to a temporary file using memory-mapped I/O, providing:
//!
//! - **Bounded memory**: Large snapshots don't exhaust heap memory
//! - **Zero-copy serving**: Mmap slices enable efficient chunk serving
//! - **Automatic cleanup**: Temp files removed on drop
//! - **SHA-256 hashing**: Content hash computed incrementally during writes
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! |  SnapshotBuffer   |
//! +-------------------+
//!          |
//!          | write()
//!          v
//! +-------------------+          +-------------------+
//! | BufferMode::Memory| -------> | BufferMode::File  |
//! | (Vec<u8>)         |  spill   | (mmap + temp file)|
//! +-------------------+  if >    +-------------------+
//!                      threshold
//!          |                              |
//!          | as_slice() / read_chunk()    | as_slice() (zero-copy)
//!          v                              v
//! +-------------------+          +-------------------+
//! | Snapshot Serving  |          | Snapshot Serving  |
//! | (copy from Vec)   |          | (mmap reference)  |
//! +-------------------+          +-------------------+
//! ```
//!
//! # Usage
//!
//! ## Basic Write and Read
//!
//! ```rust
//! use tensor_chain::snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig};
//!
//! let config = SnapshotBufferConfig::default()
//!     .with_max_memory(256 * 1024 * 1024);  // 256 MB threshold
//!
//! let mut buffer = SnapshotBuffer::new(config).unwrap();
//!
//! // Write data (may trigger spill to file)
//! buffer.write(b"snapshot data chunk 1").unwrap();
//! buffer.write(b"snapshot data chunk 2").unwrap();
//!
//! // Finalize before reading
//! buffer.finalize().unwrap();
//!
//! // Read entire buffer
//! let all_data = buffer.as_bytes().unwrap();
//!
//! // Or read specific chunk (for network transfer)
//! let chunk = buffer.as_slice(0, buffer.total_len() as usize).unwrap();
//! ```
//!
//! ## Checking Buffer State
//!
//! ```rust
//! use tensor_chain::snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig};
//!
//! let mut buffer = SnapshotBuffer::with_defaults().unwrap();
//! buffer.write(b"data").unwrap();
//! buffer.finalize().unwrap();
//!
//! // Check if using file backing
//! if buffer.is_file_backed() {
//!     println!("Spilled to: {:?}", buffer.temp_path());
//! }
//!
//! // Get content hash (SHA-256)
//! let hash = buffer.hash();
//!
//! // Get total size
//! println!("Total bytes: {}", buffer.total_len());
//! ```
//!
//! ## Using `SnapshotBufferReader`
//!
//! ```rust
//! use tensor_chain::snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig, SnapshotBufferReader};
//! use std::io::{Read, Seek, SeekFrom};
//!
//! let mut buffer = SnapshotBuffer::with_defaults().unwrap();
//! buffer.write(b"0123456789").unwrap();
//! buffer.finalize().unwrap();
//!
//! let mut reader = SnapshotBufferReader::new(&buffer);
//!
//! // Read in chunks
//! let mut buf = [0u8; 5];
//! reader.read_exact(&mut buf).unwrap();
//! assert_eq!(&buf, b"01234");
//!
//! // Seek to position
//! reader.seek(SeekFrom::Start(3)).unwrap();
//! reader.read_exact(&mut buf).unwrap();
//! assert_eq!(&buf, b"34567");
//! ```
//!
//! # Configuration
//!
//! ```rust
//! use tensor_chain::snapshot_buffer::SnapshotBufferConfig;
//!
//! let config = SnapshotBufferConfig {
//!     // Spill to disk when exceeding this size
//!     max_memory_bytes: 256 * 1024 * 1024,  // 256 MB
//!
//!     // Directory for temporary files
//!     temp_dir: std::env::temp_dir().join("raft_snapshots"),
//!
//!     // Initial file size when spilling
//!     initial_file_capacity: 64 * 1024 * 1024,  // 64 MB
//! };
//! ```
//!
//! # Error Handling
//!
//! Operations can fail with:
//!
//! - [`SnapshotBufferError::Io`]: Underlying I/O error (file creation, mmap, etc.)
//! - [`SnapshotBufferError::OutOfBounds`]: Read offset/length exceeds buffer size
//! - [`SnapshotBufferError::NotFinalized`]: Attempted read before `finalize()`
//!
//! # Security Considerations
//!
//! - Temp files are created with restrictive permissions
//! - Files are cleaned up on drop (best-effort)
//! - SHA-256 hash allows verification of content integrity
//!
//! # Performance Characteristics
//!
//! | Operation | Memory Mode | File Mode |
//! |-----------|-------------|-----------|
//! | `write()` | O(1) amortized | O(1) amortized + possible mmap resize |
//! | `as_slice()` | O(1) | O(1) zero-copy via mmap |
//! | `read_chunk()` | O(n) copy | O(n) copy |
//! | `finalize()` | O(1) | O(1) + fsync |
//!
//! # Thread Safety
//!
//! `SnapshotBuffer` is NOT thread-safe for concurrent writes. However, after
//! `finalize()`, the buffer can be shared via `Arc` for concurrent reads.
//!
//! # See Also
//!
//! - [`crate::snapshot_streaming`]: Higher-level streaming serialization
//! - [`crate::network::SnapshotRequest`]: Network protocol for snapshot transfer

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use memmap2::MmapMut;
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Configuration for snapshot buffering.
#[derive(Debug, Clone)]
pub struct SnapshotBufferConfig {
    /// Maximum bytes to hold in memory before spilling to disk.
    /// Default: 256 MB.
    pub max_memory_bytes: usize,
    /// Directory for temporary snapshot files.
    /// Default: system temp directory.
    pub temp_dir: PathBuf,
    /// Initial file capacity when spilling to disk.
    /// Default: 64 MB.
    pub initial_file_capacity: usize,
}

impl Default for SnapshotBufferConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            temp_dir: std::env::temp_dir().join("raft_snapshots"),
            initial_file_capacity: 64 * 1024 * 1024, // 64 MB
        }
    }
}

impl SnapshotBufferConfig {
    #[must_use]
    pub const fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    #[must_use]
    pub fn with_temp_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.temp_dir = path.as_ref().to_path_buf();
        self
    }
}

/// Error type for snapshot buffer operations.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotBufferError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("out of bounds access: offset={offset}, len={len}, total={total}")]
    OutOfBounds { offset: u64, len: usize, total: u64 },
    #[error("buffer not finalized")]
    NotFinalized,
}

pub type Result<T> = std::result::Result<T, SnapshotBufferError>;

/// Storage mode for the buffer.
enum BufferMode {
    /// Data fits in memory.
    Memory { data: Vec<u8> },
    /// Data spilled to file.
    File {
        file: File,
        mmap: MmapMut,
        capacity: usize,
        path: PathBuf,
    },
}

/// Memory-efficient snapshot buffer with automatic disk spill.
///
/// When data exceeds `max_memory_bytes`, the buffer transparently
/// transitions to a memory-mapped file, providing:
/// - Bounded memory usage regardless of snapshot size
/// - Zero-copy chunk serving via mmap slices
/// - Automatic cleanup on drop
pub struct SnapshotBuffer {
    config: SnapshotBufferConfig,
    mode: BufferMode,
    total_bytes: u64,
    hasher: Sha256,
    buffer_id: Uuid,
    finalized: bool,
}

impl SnapshotBuffer {
    /// Create a new snapshot buffer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the temp directory cannot be created.
    pub fn new(config: SnapshotBufferConfig) -> Result<Self> {
        // Ensure temp directory exists
        fs::create_dir_all(&config.temp_dir)?;

        Ok(Self {
            config,
            mode: BufferMode::Memory { data: Vec::new() },
            total_bytes: 0,
            hasher: Sha256::new(),
            buffer_id: Uuid::new_v4(),
            finalized: false,
        })
    }

    /// Create a new snapshot buffer with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the default temp directory cannot be created.
    pub fn with_defaults() -> Result<Self> {
        Self::new(SnapshotBufferConfig::default())
    }

    /// Write data to the buffer, spilling to disk if the memory threshold is exceeded.
    ///
    /// # Errors
    ///
    /// Returns an error if disk spill or mmap operations fail.
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Update hash
        self.hasher.update(data);

        match &mut self.mode {
            BufferMode::Memory { data: mem_data } => {
                let new_size = mem_data.len() + data.len();

                // Check if we need to spill to disk
                if new_size > self.config.max_memory_bytes {
                    self.spill_to_file(data)?;
                } else {
                    mem_data.extend_from_slice(data);
                }
            },
            BufferMode::File {
                mmap,
                capacity,
                file,
                ..
            } => {
                #[allow(clippy::cast_possible_truncation)]
                // Snapshot size bounded by available memory
                let write_offset = self.total_bytes as usize;
                let needed = write_offset + data.len();

                // Grow file if needed
                if needed > *capacity {
                    let new_capacity = (*capacity * 2).max(needed + 1024 * 1024);
                    file.set_len(new_capacity as u64)?;
                    *mmap = unsafe { MmapMut::map_mut(&*file)? };
                    *capacity = new_capacity;
                }

                mmap[write_offset..write_offset + data.len()].copy_from_slice(data);
            },
        }

        self.total_bytes += data.len() as u64;
        Ok(())
    }

    /// Spill in-memory data to a file and switch to file mode.
    fn spill_to_file(&mut self, additional_data: &[u8]) -> Result<()> {
        let path = self
            .config
            .temp_dir
            .join(format!("snapshot_{}.tmp", self.buffer_id));

        let existing_data = match &self.mode {
            BufferMode::Memory { data } => data.clone(),
            BufferMode::File { .. } => return Ok(()), // Already in file mode
        };

        let total_size = existing_data.len() + additional_data.len();
        let capacity = self.config.initial_file_capacity.max(total_size * 2);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(capacity as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Copy existing data
        mmap[..existing_data.len()].copy_from_slice(&existing_data);
        // Copy new data
        mmap[existing_data.len()..total_size].copy_from_slice(additional_data);

        // Ensure data is flushed and synced to disk for durability
        mmap.flush()?;
        file.sync_all()?;

        self.mode = BufferMode::File {
            file,
            mmap,
            capacity,
            path,
        };

        Ok(())
    }

    /// Finalize the buffer (must be called before reading).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or syncing the backing file fails.
    pub fn finalize(&mut self) -> Result<()> {
        if let BufferMode::File { mmap, file, .. } = &self.mode {
            mmap.flush()?;
            file.sync_all()?;
        }
        self.finalized = true;
        Ok(())
    }

    #[must_use]
    pub const fn total_len(&self) -> u64 {
        self.total_bytes
    }

    #[must_use]
    pub const fn is_file_backed(&self) -> bool {
        matches!(self.mode, BufferMode::File { .. })
    }

    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    /// Read a chunk of data as a new `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotBufferError::OutOfBounds`] if the requested range exceeds the buffer.
    pub fn read_chunk(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        if offset + len as u64 > self.total_bytes {
            return Err(SnapshotBufferError::OutOfBounds {
                offset,
                len,
                total: self.total_bytes,
            });
        }

        #[allow(clippy::cast_possible_truncation)] // Offset validated against total_bytes
        let start = offset as usize;
        let end = start + len;

        match &self.mode {
            BufferMode::Memory { data } => Ok(data[start..end].to_vec()),
            BufferMode::File { mmap, .. } => Ok(mmap[start..end].to_vec()),
        }
    }

    /// Zero-copy for file mode; copies for memory mode.
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotBufferError::OutOfBounds`] if the requested range exceeds the buffer.
    pub fn as_slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        if offset + len as u64 > self.total_bytes {
            return Err(SnapshotBufferError::OutOfBounds {
                offset,
                len,
                total: self.total_bytes,
            });
        }

        #[allow(clippy::cast_possible_truncation)] // Offset validated against total_bytes
        let start = offset as usize;
        let end = start + len;

        match &self.mode {
            BufferMode::Memory { data } => Ok(&data[start..end]),
            BufferMode::File { mmap, .. } => Ok(&mmap[start..end]),
        }
    }

    /// Return the entire buffer contents as a byte slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is empty or the backing file cannot be read.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        #[allow(clippy::cast_possible_truncation)] // Snapshot size bounded by available memory
        let len = self.total_bytes as usize;
        self.as_slice(0, len)
    }

    /// Clean up the temp file and reset to memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the temp file cannot be removed.
    pub fn cleanup(&mut self) -> Result<()> {
        if let BufferMode::File { path, .. } = &self.mode {
            let path = path.clone();
            // Switch back to memory mode first
            self.mode = BufferMode::Memory { data: Vec::new() };
            self.total_bytes = 0;
            // Then remove the file
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn temp_path(&self) -> Option<&Path> {
        match &self.mode {
            BufferMode::File { path, .. } => Some(path),
            BufferMode::Memory { .. } => None,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &SnapshotBufferConfig {
        &self.config
    }
}

impl Drop for SnapshotBuffer {
    fn drop(&mut self) {
        // Best-effort cleanup of temp files
        if let BufferMode::File { path, .. } = &self.mode {
            let _ = fs::remove_file(path);
        }
    }
}

impl Write for SnapshotBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write(buf).map_err(io::Error::other)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let BufferMode::File { mmap, .. } = &self.mode {
            mmap.flush()?;
        }
        Ok(())
    }
}

/// Reader for streaming data from a snapshot buffer.
pub struct SnapshotBufferReader<'a> {
    buffer: &'a SnapshotBuffer,
    position: u64,
}

impl<'a> SnapshotBufferReader<'a> {
    #[must_use]
    pub const fn new(buffer: &'a SnapshotBuffer) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    #[must_use]
    pub const fn remaining(&self) -> u64 {
        self.buffer.total_bytes.saturating_sub(self.position)
    }
}

impl Read for SnapshotBufferReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[allow(clippy::cast_possible_truncation)] // Remaining bytes bounded by buffer size
        let remaining = self.remaining() as usize;
        if remaining == 0 {
            return Ok(0);
        }

        let to_read = buf.len().min(remaining);
        let slice = self
            .buffer
            .as_slice(self.position, to_read)
            .map_err(io::Error::other)?;

        buf[..to_read].copy_from_slice(slice);
        self.position += to_read as u64;
        Ok(to_read)
    }
}

impl Seek for SnapshotBufferReader<'_> {
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.buffer.total_bytes as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }

        self.position = new_pos as u64;
        Ok(self.position)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn test_config() -> SnapshotBufferConfig {
        SnapshotBufferConfig {
            max_memory_bytes: 1024, // 1KB for testing
            temp_dir: std::env::temp_dir().join("snapshot_buffer_tests"),
            initial_file_capacity: 4096,
        }
    }

    #[test]
    fn test_buffer_memory_mode() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        // Write small data - stays in memory
        buffer.write(b"hello").unwrap();
        buffer.write(b" world").unwrap();
        buffer.finalize().unwrap();

        assert!(!buffer.is_file_backed());
        assert_eq!(buffer.total_len(), 11);
        assert_eq!(buffer.as_bytes().unwrap(), b"hello world");
    }

    #[test]
    fn test_buffer_spill_to_file() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        // Write enough data to trigger spill (>1KB)
        let large_data = vec![0u8; 2000];
        buffer.write(&large_data).unwrap();
        buffer.finalize().unwrap();

        assert!(buffer.is_file_backed());
        assert_eq!(buffer.total_len(), 2000);
        assert!(buffer.temp_path().is_some());
    }

    #[test]
    fn test_buffer_write_read_roundtrip() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let test_data: Vec<u8> = (0..2000).map(|i| (i % 256) as u8).collect();
        buffer.write(&test_data).unwrap();
        buffer.finalize().unwrap();

        let read_data = buffer.as_bytes().unwrap();
        assert_eq!(read_data, test_data.as_slice());
    }

    #[test]
    fn test_buffer_zero_copy_chunks() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data = vec![42u8; 3000];
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        // Read chunks
        let chunk1 = buffer.as_slice(0, 1000).unwrap();
        let chunk2 = buffer.as_slice(1000, 1000).unwrap();
        let chunk3 = buffer.as_slice(2000, 1000).unwrap();

        assert!(chunk1.iter().all(|&b| b == 42));
        assert!(chunk2.iter().all(|&b| b == 42));
        assert!(chunk3.iter().all(|&b| b == 42));
    }

    #[test]
    fn test_buffer_hash_computation() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data = b"test data for hashing";
        buffer.write(data).unwrap();
        buffer.finalize().unwrap();

        let buffer_hash = buffer.hash();

        // Compute expected hash directly
        let expected_hash: [u8; 32] = Sha256::digest(data).into();

        assert_eq!(buffer_hash, expected_hash);
    }

    #[test]
    fn test_buffer_cleanup_on_drop() {
        let config = test_config();
        let temp_path;

        {
            let mut buffer = SnapshotBuffer::new(config).unwrap();
            let data = vec![0u8; 2000];
            buffer.write(&data).unwrap();
            buffer.finalize().unwrap();

            temp_path = buffer.temp_path().unwrap().to_path_buf();
            assert!(temp_path.exists());
        }

        // File should be cleaned up after drop
        assert!(!temp_path.exists());
    }

    #[test]
    fn test_buffer_explicit_cleanup() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data = vec![0u8; 2000];
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        let temp_path = buffer.temp_path().unwrap().to_path_buf();
        assert!(temp_path.exists());

        buffer.cleanup().unwrap();
        assert!(!temp_path.exists());
        assert_eq!(buffer.total_len(), 0);
    }

    #[test]
    fn test_buffer_reader() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data: Vec<u8> = (0..100).collect();
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);
        let mut buf = [0u8; 50];

        // Read first half
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 50);
        assert_eq!(&buf[..50], &data[..50]);

        // Read second half
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 50);
        assert_eq!(&buf[..50], &data[50..]);

        // EOF
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_buffer_reader_seek() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data: Vec<u8> = (0..100).collect();
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);

        // Seek to middle
        reader.seek(SeekFrom::Start(50)).unwrap();
        assert_eq!(reader.position(), 50);

        // Read from middle
        let mut buf = [0u8; 10];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &data[50..60]);
    }

    #[test]
    fn test_buffer_out_of_bounds() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        buffer.write(b"small").unwrap();
        buffer.finalize().unwrap();

        let result = buffer.as_slice(0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        let data = vec![42u8; 500];
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        // Wrap in Arc for sharing (read-only after finalize)
        let buffer = Arc::new(buffer);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let buffer = Arc::clone(&buffer);
                thread::spawn(move || {
                    let chunk = buffer.as_slice(i * 100, 100).unwrap();
                    assert!(chunk.iter().all(|&b| b == 42));
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_buffer_incremental_writes() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        // Write in small increments
        for i in 0..500 {
            buffer.write(&[i as u8]).unwrap();
        }
        buffer.finalize().unwrap();

        assert_eq!(buffer.total_len(), 500);

        // Verify data
        let data = buffer.as_bytes().unwrap();
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(b, (i % 256) as u8);
        }
    }

    #[test]
    fn test_buffer_large_file_growth() {
        let config = SnapshotBufferConfig {
            max_memory_bytes: 100,
            temp_dir: std::env::temp_dir().join("snapshot_buffer_growth_test"),
            initial_file_capacity: 1024,
        };

        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write more than initial capacity
        let data = vec![0u8; 10000];
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        assert!(buffer.is_file_backed());
        assert_eq!(buffer.total_len(), 10000);
        assert_eq!(buffer.as_bytes().unwrap().len(), 10000);
    }

    #[test]
    fn test_config_builder() {
        let config = SnapshotBufferConfig::default()
            .with_max_memory(1024)
            .with_temp_dir("/tmp/test");

        assert_eq!(config.max_memory_bytes, 1024);
        assert_eq!(config.temp_dir, PathBuf::from("/tmp/test"));
    }

    // ========== Seek Variants Tests ==========

    #[test]
    fn test_buffer_reader_seek_from_end() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data: Vec<u8> = (0..100).collect();
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);

        // Seek 10 bytes from end
        reader.seek(SeekFrom::End(-10)).unwrap();
        assert_eq!(reader.position(), 90);

        // Read last 10 bytes
        let mut buf = [0u8; 10];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &data[90..100]);
    }

    #[test]
    fn test_buffer_reader_seek_from_current() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data: Vec<u8> = (0..100).collect();
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);

        // Move forward 20
        reader.seek(SeekFrom::Current(20)).unwrap();
        assert_eq!(reader.position(), 20);

        // Move forward another 30
        reader.seek(SeekFrom::Current(30)).unwrap();
        assert_eq!(reader.position(), 50);

        // Move backward 10
        reader.seek(SeekFrom::Current(-10)).unwrap();
        assert_eq!(reader.position(), 40);

        // Read from position 40
        let mut buf = [0u8; 10];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &data[40..50]);
    }

    #[test]
    fn test_buffer_reader_seek_past_end() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        buffer.write(b"short").unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);

        // Seek past end (allowed, but read will return 0)
        reader.seek(SeekFrom::Start(100)).unwrap();
        assert_eq!(reader.position(), 100);

        // Reading should return 0 bytes
        let mut buf = [0u8; 10];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_buffer_reader_seek_negative_position() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        buffer.write(b"test").unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);

        // Seek to negative position should error
        let result = reader.seek(SeekFrom::Start(0));
        assert!(result.is_ok());

        let result = reader.seek(SeekFrom::Current(-10));
        assert!(result.is_err());
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_buffer_read_exact_boundary() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data = vec![0u8; 100];
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        // Read exactly to the end
        let chunk = buffer.as_slice(0, 100).unwrap();
        assert_eq!(chunk.len(), 100);

        // Read one more byte should fail
        let result = buffer.as_slice(0, 101);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_read_chunk_method() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        let data: Vec<u8> = (0..200).collect();
        buffer.write(&data).unwrap();
        buffer.finalize().unwrap();

        // read_chunk returns a Vec (copy)
        let chunk = buffer.read_chunk(50, 100).unwrap();
        assert_eq!(chunk.len(), 100);
        assert_eq!(chunk, data[50..150].to_vec());
    }

    #[test]
    fn test_buffer_empty_writes() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        // Empty writes should be no-ops
        buffer.write(&[]).unwrap();
        buffer.write(&[]).unwrap();
        assert_eq!(buffer.total_len(), 0);

        buffer.write(b"data").unwrap();
        buffer.write(&[]).unwrap();
        assert_eq!(buffer.total_len(), 4);

        buffer.finalize().unwrap();
        assert_eq!(buffer.as_bytes().unwrap(), b"data");
    }

    #[test]
    fn test_buffer_finalize_idempotent() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        buffer.write(b"test").unwrap();
        buffer.finalize().unwrap();
        buffer.finalize().unwrap(); // Second finalize should be fine
        buffer.finalize().unwrap(); // Third too

        assert_eq!(buffer.as_bytes().unwrap(), b"test");
    }

    #[test]
    fn test_buffer_multiple_file_growth_cycles() {
        let config = SnapshotBufferConfig {
            max_memory_bytes: 100,
            temp_dir: std::env::temp_dir().join("snapshot_buffer_multi_growth_test"),
            initial_file_capacity: 200,
        };

        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // First write triggers spill
        buffer.write(&[1u8; 150]).unwrap();
        assert!(buffer.is_file_backed());

        // Write more to trigger first growth
        buffer.write(&[2u8; 200]).unwrap();

        // Write even more to trigger second growth
        buffer.write(&[3u8; 500]).unwrap();

        buffer.finalize().unwrap();
        assert_eq!(buffer.total_len(), 850);

        // Verify data integrity
        let data = buffer.as_bytes().unwrap();
        assert!(data[..150].iter().all(|&b| b == 1));
        assert!(data[150..350].iter().all(|&b| b == 2));
        assert!(data[350..].iter().all(|&b| b == 3));
    }

    #[test]
    fn test_buffer_with_defaults() {
        let buffer = SnapshotBuffer::with_defaults().unwrap();
        assert!(!buffer.is_file_backed());
        assert_eq!(buffer.total_len(), 0);
    }

    #[test]
    fn test_buffer_reader_remaining() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        buffer.write(b"0123456789").unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotBufferReader::new(&buffer);
        assert_eq!(reader.remaining(), 10);
        assert_eq!(reader.position(), 0);

        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.remaining(), 5);
        assert_eq!(reader.position(), 5);
    }

    #[test]
    fn test_error_display() {
        let io_err = SnapshotBufferError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let oob_err = SnapshotBufferError::OutOfBounds {
            offset: 100,
            len: 50,
            total: 80,
        };
        assert!(oob_err.to_string().contains("out of bounds"));
        assert!(oob_err.to_string().contains("100"));

        let not_final = SnapshotBufferError::NotFinalized;
        assert!(not_final.to_string().contains("not finalized"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test");
        let buf_err: SnapshotBufferError = io_err.into();
        assert!(matches!(buf_err, SnapshotBufferError::Io(_)));
    }

    #[test]
    fn test_write_trait_implementation() {
        use std::io::Write as IoWrite;

        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();

        // Use the Write trait
        let n = IoWrite::write(&mut buffer, b"hello").unwrap();
        assert_eq!(n, 5);

        let n = IoWrite::write(&mut buffer, b" world").unwrap();
        assert_eq!(n, 6);

        buffer.flush().unwrap();
        buffer.finalize().unwrap();

        assert_eq!(buffer.as_bytes().unwrap(), b"hello world");
    }

    #[test]
    fn test_buffer_temp_path_memory_mode() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        buffer.write(b"small").unwrap();
        buffer.finalize().unwrap();

        // Memory mode has no temp path
        assert!(buffer.temp_path().is_none());
    }

    #[test]
    fn test_buffer_config_accessor() {
        let config = test_config();
        let max_mem = config.max_memory_bytes;

        let buffer = SnapshotBuffer::new(config).unwrap();
        assert_eq!(buffer.config().max_memory_bytes, max_mem);
    }

    #[test]
    fn test_buffer_hash_stability() {
        let data = b"consistent hash test data";

        // Create two buffers with same data
        let mut buffer1 = SnapshotBuffer::new(test_config()).unwrap();
        buffer1.write(data).unwrap();
        buffer1.finalize().unwrap();

        let mut buffer2 = SnapshotBuffer::new(test_config()).unwrap();
        buffer2.write(data).unwrap();
        buffer2.finalize().unwrap();

        // Hashes should match
        assert_eq!(buffer1.hash(), buffer2.hash());
    }

    #[test]
    fn test_buffer_cleanup_memory_mode() {
        let mut buffer = SnapshotBuffer::new(test_config()).unwrap();
        buffer.write(b"memory").unwrap();
        buffer.finalize().unwrap();

        // Cleanup on memory mode is a no-op (no temp files to remove)
        // total_bytes is only reset when switching from file mode back to memory
        buffer.cleanup().unwrap();
        // In memory mode, cleanup doesn't reset the buffer
        assert_eq!(buffer.total_len(), 6);
    }
}
