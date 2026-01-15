//! Streaming serialization/deserialization for Raft snapshots.
//!
//! Provides incremental log entry writing and reading to avoid loading
//! entire snapshots into memory at once.

use std::io;

use crate::{
    network::LogEntry,
    snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig, SnapshotBufferError},
};

/// Error type for snapshot streaming operations.
#[derive(Debug)]
pub enum StreamingError {
    Io(io::Error),
    Buffer(SnapshotBufferError),
    Serialization(String),
    InvalidFormat(String),
    UnexpectedEof,
}

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Buffer(e) => write!(f, "buffer error: {}", e),
            Self::Serialization(s) => write!(f, "serialization error: {}", s),
            Self::InvalidFormat(s) => write!(f, "invalid format: {}", s),
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
        }
    }
}

impl std::error::Error for StreamingError {}

impl From<io::Error> for StreamingError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<SnapshotBufferError> for StreamingError {
    fn from(e: SnapshotBufferError) -> Self {
        Self::Buffer(e)
    }
}

impl From<bincode::Error> for StreamingError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, StreamingError>;

/// Magic bytes for streaming snapshot format.
const STREAMING_MAGIC: [u8; 4] = *b"SNAP";

/// Streaming snapshot format version.
const STREAMING_VERSION: u32 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 16;

/// Maximum entry size (100 MB).
const MAX_ENTRY_SIZE: usize = 100 * 1024 * 1024;

/// Streaming snapshot writer for incremental log entry serialization.
///
/// Writes log entries one at a time using a length-prefixed format,
/// avoiding the need to hold all entries in memory.
pub struct SnapshotWriter {
    buffer: SnapshotBuffer,
    entry_count: u64,
    last_index: u64,
    last_term: u64,
}

impl SnapshotWriter {
    /// Create a new snapshot writer with the given buffer configuration.
    pub fn new(config: SnapshotBufferConfig) -> Result<Self> {
        let mut buffer = SnapshotBuffer::new(config)?;

        // Write header placeholder
        let header = [0u8; HEADER_SIZE];
        buffer.write(&header)?;

        Ok(Self {
            buffer,
            entry_count: 0,
            last_index: 0,
            last_term: 0,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(SnapshotBufferConfig::default())
    }

    /// Write a single log entry.
    pub fn write_entry(&mut self, entry: &LogEntry) -> Result<()> {
        let bytes = bincode::serialize(entry)?;
        let len = bytes.len() as u32;

        if bytes.len() > MAX_ENTRY_SIZE {
            return Err(StreamingError::InvalidFormat(format!(
                "entry too large: {} bytes (max {})",
                bytes.len(),
                MAX_ENTRY_SIZE
            )));
        }

        // Write length prefix (4 bytes, little-endian)
        self.buffer.write(&len.to_le_bytes())?;
        // Write serialized entry
        self.buffer.write(&bytes)?;

        self.entry_count += 1;
        self.last_index = entry.index;
        self.last_term = entry.term;

        Ok(())
    }

    /// Get the number of entries written.
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Get the last written index.
    #[must_use]
    pub fn last_index(&self) -> u64 {
        self.last_index
    }

    /// Get the last written term.
    #[must_use]
    pub fn last_term(&self) -> u64 {
        self.last_term
    }

    /// Get total bytes written (including header).
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.buffer.total_len()
    }

    /// Finalize the snapshot and return the buffer.
    ///
    /// This writes the header with entry count and returns the
    /// completed buffer for chunk serving.
    pub fn finish(mut self) -> Result<SnapshotBuffer> {
        // Get current data (after header)
        let total_len = self.buffer.total_len();

        // Build header
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&self.entry_count.to_le_bytes());

        // We need to rewrite the header at the start
        // Since SnapshotBuffer doesn't support random writes, we need to create a new buffer
        let config = self.buffer.config().clone();
        let mut new_buffer = SnapshotBuffer::new(config)?;

        // Write real header
        new_buffer.write(&header)?;

        // Copy existing data (skip placeholder header)
        if total_len > HEADER_SIZE as u64 {
            let data = self.buffer.as_slice(
                HEADER_SIZE as u64,
                (total_len - HEADER_SIZE as u64) as usize,
            )?;
            new_buffer.write(data)?;
        }

        new_buffer.finalize()?;

        // Clean up old buffer
        self.buffer.cleanup()?;

        Ok(new_buffer)
    }
}

/// Streaming snapshot reader for incremental log entry deserialization.
///
/// Reads log entries on-demand using an iterator interface,
/// avoiding the need to deserialize all entries at once.
pub struct SnapshotReader<'a> {
    buffer: &'a SnapshotBuffer,
    read_offset: u64,
    entry_count: u64,
    entries_read: u64,
}

impl<'a> SnapshotReader<'a> {
    /// Create a new snapshot reader.
    pub fn new(buffer: &'a SnapshotBuffer) -> Result<Self> {
        if buffer.total_len() < HEADER_SIZE as u64 {
            return Err(StreamingError::InvalidFormat(
                "buffer too small".to_string(),
            ));
        }

        // Read and validate header
        let header = buffer.as_slice(0, HEADER_SIZE)?;

        if header[0..4] != STREAMING_MAGIC {
            return Err(StreamingError::InvalidFormat("invalid magic".to_string()));
        }

        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version > STREAMING_VERSION {
            return Err(StreamingError::InvalidFormat(format!(
                "unsupported version: {}",
                version
            )));
        }

        let entry_count = u64::from_le_bytes(header[8..16].try_into().unwrap());

        Ok(Self {
            buffer,
            read_offset: HEADER_SIZE as u64,
            entry_count,
            entries_read: 0,
        })
    }

    /// Get the total number of entries in the snapshot.
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Get the number of entries read so far.
    #[must_use]
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }

    /// Get remaining entries to read.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.entry_count.saturating_sub(self.entries_read)
    }

    /// Read the next log entry.
    pub fn read_entry(&mut self) -> Result<Option<LogEntry>> {
        if self.entries_read >= self.entry_count {
            return Ok(None);
        }

        // Read length prefix
        if self.read_offset + 4 > self.buffer.total_len() {
            return Err(StreamingError::UnexpectedEof);
        }

        let len_bytes = self.buffer.as_slice(self.read_offset, 4)?;
        let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        self.read_offset += 4;

        if len > MAX_ENTRY_SIZE {
            return Err(StreamingError::InvalidFormat(format!(
                "entry too large: {} bytes",
                len
            )));
        }

        // Read entry data
        if self.read_offset + len as u64 > self.buffer.total_len() {
            return Err(StreamingError::UnexpectedEof);
        }

        let entry_bytes = self.buffer.as_slice(self.read_offset, len)?;
        let entry: LogEntry = bincode::deserialize(entry_bytes)?;
        self.read_offset += len as u64;

        self.entries_read += 1;

        Ok(Some(entry))
    }
}

impl Iterator for SnapshotReader<'_> {
    type Item = Result<LogEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_entry() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Deserialize a snapshot from raw bytes (for backwards compatibility).
pub fn deserialize_entries(data: &[u8]) -> Result<Vec<LogEntry>> {
    // First try streaming format
    if data.len() >= HEADER_SIZE && data[0..4] == STREAMING_MAGIC {
        let config = SnapshotBufferConfig {
            max_memory_bytes: data.len() + 1024, // Ensure memory mode
            ..SnapshotBufferConfig::default()
        };

        let mut buffer = SnapshotBuffer::new(config)?;
        buffer.write(data)?;
        buffer.finalize()?;

        let reader = SnapshotReader::new(&buffer)?;
        reader.collect()
    } else {
        // Fall back to legacy format (direct bincode Vec<LogEntry>)
        let entries: Vec<LogEntry> = bincode::deserialize(data)?;
        Ok(entries)
    }
}

/// Serialize entries to streaming format.
pub fn serialize_entries(
    entries: &[LogEntry],
    config: SnapshotBufferConfig,
) -> Result<SnapshotBuffer> {
    let mut writer = SnapshotWriter::new(config)?;
    for entry in entries {
        writer.write_entry(entry)?;
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, BlockHeader};

    fn test_config() -> SnapshotBufferConfig {
        SnapshotBufferConfig {
            max_memory_bytes: 1024 * 1024, // 1MB for testing
            temp_dir: std::env::temp_dir().join("snapshot_streaming_tests"),
            initial_file_capacity: 4096,
        }
    }

    fn create_test_entry(index: u64, term: u64) -> LogEntry {
        let header = BlockHeader::new(index, [0u8; 32], [0u8; 32], [0u8; 32], "test".to_string());
        let block = Block::new(header, vec![]);
        LogEntry::new(term, index, block)
    }

    #[test]
    fn test_streaming_single_entry() {
        let mut writer = SnapshotWriter::new(test_config()).unwrap();

        let entry = create_test_entry(1, 1);
        writer.write_entry(&entry).unwrap();

        assert_eq!(writer.entry_count(), 1);
        assert_eq!(writer.last_index(), 1);
        assert_eq!(writer.last_term(), 1);

        let buffer = writer.finish().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        assert_eq!(reader.entry_count(), 1);

        let read_entry = reader.read_entry().unwrap().unwrap();
        assert_eq!(read_entry.index, 1);
        assert_eq!(read_entry.term, 1);

        assert!(reader.read_entry().unwrap().is_none());
    }

    #[test]
    fn test_streaming_large_snapshot() {
        let mut writer = SnapshotWriter::new(test_config()).unwrap();

        // Write 10k entries
        for i in 1..=10000 {
            let entry = create_test_entry(i, (i / 100) + 1);
            writer.write_entry(&entry).unwrap();
        }

        assert_eq!(writer.entry_count(), 10000);

        let buffer = writer.finish().unwrap();

        let reader = SnapshotReader::new(&buffer).unwrap();
        assert_eq!(reader.entry_count(), 10000);

        // Read all via iterator
        let entries: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 10000);

        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.index, (i + 1) as u64);
        }
    }

    #[test]
    fn test_streaming_iterator() {
        let mut writer = SnapshotWriter::new(test_config()).unwrap();

        for i in 1..=5 {
            writer.write_entry(&create_test_entry(i, 1)).unwrap();
        }

        let buffer = writer.finish().unwrap();
        let reader = SnapshotReader::new(&buffer).unwrap();

        let indices: Vec<u64> = reader.map(|r| r.unwrap().index).collect();

        assert_eq!(indices, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_streaming_partial_read() {
        let mut writer = SnapshotWriter::new(test_config()).unwrap();

        for i in 1..=10 {
            writer.write_entry(&create_test_entry(i, 1)).unwrap();
        }

        let buffer = writer.finish().unwrap();
        let mut reader = SnapshotReader::new(&buffer).unwrap();

        // Read only first 5
        for _ in 0..5 {
            let _ = reader.read_entry().unwrap();
        }

        assert_eq!(reader.entries_read(), 5);
        assert_eq!(reader.remaining(), 5);
    }

    #[test]
    fn test_streaming_corrupted_magic() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();
        buffer
            .write(b"BAAD\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            .unwrap();
        buffer.finalize().unwrap();

        let result = SnapshotReader::new(&buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let entries: Vec<LogEntry> = (1..=100).map(|i| create_test_entry(i, 1)).collect();

        let buffer = serialize_entries(&entries, test_config()).unwrap();
        let bytes = buffer.as_bytes().unwrap();

        let deserialized = deserialize_entries(bytes).unwrap();
        assert_eq!(deserialized.len(), 100);

        for (i, entry) in deserialized.iter().enumerate() {
            assert_eq!(entry.index, (i + 1) as u64);
        }
    }

    #[test]
    fn test_empty_snapshot() {
        let writer = SnapshotWriter::new(test_config()).unwrap();
        assert_eq!(writer.entry_count(), 0);

        let buffer = writer.finish().unwrap();

        let reader = SnapshotReader::new(&buffer).unwrap();
        assert_eq!(reader.entry_count(), 0);
        assert!(reader.collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn test_bytes_written_tracking() {
        let mut writer = SnapshotWriter::new(test_config()).unwrap();

        let initial_bytes = writer.bytes_written();
        assert_eq!(initial_bytes, HEADER_SIZE as u64);

        writer.write_entry(&create_test_entry(1, 1)).unwrap();

        assert!(writer.bytes_written() > initial_bytes);
    }
}
