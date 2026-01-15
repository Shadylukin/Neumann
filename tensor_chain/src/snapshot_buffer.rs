//! Memory-efficient snapshot buffering with file-backed overflow.
//!
//! Provides bounded memory usage for Raft snapshots by automatically
//! spilling to disk when data exceeds a configurable threshold.

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
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    pub fn with_temp_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.temp_dir = path.as_ref().to_path_buf();
        self
    }
}

/// Error type for snapshot buffer operations.
#[derive(Debug)]
pub enum SnapshotBufferError {
    Io(io::Error),
    OutOfBounds { offset: u64, len: usize, total: u64 },
    NotFinalized,
}

impl std::fmt::Display for SnapshotBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::OutOfBounds { offset, len, total } => {
                write!(
                    f,
                    "out of bounds access: offset={}, len={}, total={}",
                    offset, len, total
                )
            },
            Self::NotFinalized => write!(f, "buffer not finalized"),
        }
    }
}

impl std::error::Error for SnapshotBufferError {}

impl From<io::Error> for SnapshotBufferError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
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
    /// Create a new snapshot buffer.
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

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(SnapshotBufferConfig::default())
    }

    /// Write data to the buffer.
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

        self.mode = BufferMode::File {
            file,
            mmap,
            capacity,
            path,
        };

        Ok(())
    }

    /// Finalize the buffer (must be called before reading).
    pub fn finalize(&mut self) -> Result<()> {
        if let BufferMode::File { mmap, .. } = &self.mode {
            mmap.flush()?;
        }
        self.finalized = true;
        Ok(())
    }

    /// Get total bytes written.
    #[must_use]
    pub fn total_len(&self) -> u64 {
        self.total_bytes
    }

    /// Check if buffer is in file mode.
    #[must_use]
    pub fn is_file_backed(&self) -> bool {
        matches!(self.mode, BufferMode::File { .. })
    }

    /// Get the SHA-256 hash of all written data.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    /// Read a chunk of data at the given offset.
    pub fn read_chunk(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        if offset + len as u64 > self.total_bytes {
            return Err(SnapshotBufferError::OutOfBounds {
                offset,
                len,
                total: self.total_bytes,
            });
        }

        let start = offset as usize;
        let end = start + len;

        match &self.mode {
            BufferMode::Memory { data } => Ok(data[start..end].to_vec()),
            BufferMode::File { mmap, .. } => Ok(mmap[start..end].to_vec()),
        }
    }

    /// Get a slice of the buffer (zero-copy for file mode).
    pub fn as_slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        if offset + len as u64 > self.total_bytes {
            return Err(SnapshotBufferError::OutOfBounds {
                offset,
                len,
                total: self.total_bytes,
            });
        }

        let start = offset as usize;
        let end = start + len;

        match &self.mode {
            BufferMode::Memory { data } => Ok(&data[start..end]),
            BufferMode::File { mmap, .. } => Ok(&mmap[start..end]),
        }
    }

    /// Get the entire buffer as bytes.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        self.as_slice(0, self.total_bytes as usize)
    }

    /// Clean up any temporary files.
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

    /// Get the temp file path if in file mode.
    #[must_use]
    pub fn temp_path(&self) -> Option<&Path> {
        match &self.mode {
            BufferMode::File { path, .. } => Some(path),
            BufferMode::Memory { .. } => None,
        }
    }

    /// Get the buffer configuration.
    #[must_use]
    pub fn config(&self) -> &SnapshotBufferConfig {
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
        self.write(buf)
            .map_err(io::Error::other)?;
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
    /// Create a new reader starting at offset 0.
    pub fn new(buffer: &'a SnapshotBuffer) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    /// Get current position.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.buffer.total_bytes.saturating_sub(self.position)
    }
}

impl Read for SnapshotBufferReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
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
        reader.read(&mut buf).unwrap();
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
}
