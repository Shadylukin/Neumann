// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Memory-efficient streaming serialization for large Raft snapshots.
//!
//! # Overview
//!
//! This module provides streaming serialization and deserialization for Raft log snapshots,
//! enabling efficient handling of snapshots that may contain millions of log entries.
//! Instead of loading entire snapshots into memory, entries are processed incrementally
//! using a length-prefixed wire format.
//!
//! Key features:
//! - **Incremental writing**: Entries serialized one at a time via [`SnapshotWriter`]
//! - **Lazy reading**: Entries deserialized on-demand via [`SnapshotReader`] iterator
//! - **Memory bounded**: Uses [`SnapshotBuffer`] for automatic disk spill
//! - **Backwards compatible**: Falls back to legacy format for old snapshots
//!
//! # Wire Format
//!
//! ```text
//! +--------+--------+--------+--------+--------+...
//! | Header (16 bytes)        | Entry 1         | Entry 2 | ...
//! +--------+--------+--------+--------+--------+...
//!
//! Header:
//! +0       +4       +8       +16
//! | SNAP   | Version| Entry Count (u64)      |
//! | (4B)   | (u32)  | (8B)                   |
//!
//! Entry:
//! +0       +4       +N
//! | Length | Bincode-serialized LogEntry |
//! | (u32)  | (variable)                  |
//! ```
//!
//! # Architecture
//!
//! ```text
//! +------------------+     write_entry()    +------------------+
//! | Raft State       | ------------------> | SnapshotWriter   |
//! | Machine          |                     +------------------+
//! +------------------+                     | - buffer         |
//!                                          | - entry_count    |
//!                                          | - last_index     |
//!                                          +------------------+
//!                                                  |
//!                                                  | finish()
//!                                                  v
//!                                          +------------------+
//!                                          | SnapshotBuffer   |
//!                                          | (memory or file) |
//!                                          +------------------+
//!                                                  |
//!                                          SnapshotReader::new()
//!                                                  |
//!                                                  v
//! +------------------+     read_entry()    +------------------+
//! | Follower Node    | <------------------ | SnapshotReader   |
//! | (applying snap)  |                     | (implements      |
//! +------------------+                     |  Iterator)       |
//!                                          +------------------+
//! ```
//!
//! # Usage
//!
//! ## Writing Snapshots
//!
//! ```rust
//! use tensor_chain::snapshot_streaming::SnapshotWriter;
//! use tensor_chain::snapshot_buffer::SnapshotBufferConfig;
//! use tensor_chain::network::LogEntry;
//! use tensor_chain::{Block, BlockHeader};
//!
//! // Create writer with custom config
//! let config = SnapshotBufferConfig::default();
//! let mut writer = SnapshotWriter::new(config).unwrap();
//!
//! // Write entries incrementally
//! // for entry in log_entries {
//! //     writer.write_entry(&entry)?;
//! // }
//!
//! // Check progress
//! println!("Entries: {}, Bytes: {}", writer.entry_count(), writer.bytes_written());
//!
//! // Finalize and get buffer for chunk serving
//! let buffer = writer.finish().unwrap();
//! ```
//!
//! ## Reading Snapshots
//!
//! ```rust
//! use tensor_chain::snapshot_streaming::SnapshotReader;
//! use tensor_chain::snapshot_buffer::SnapshotBuffer;
//!
//! // Given a SnapshotBuffer received from leader
//! // let buffer: SnapshotBuffer = ...;
//!
//! // Create reader (validates header)
//! // let reader = SnapshotReader::new(&buffer)?;
//!
//! // Read entries lazily via iterator
//! // for result in reader {
//! //     let entry = result?;
//! //     state_machine.apply(entry);
//! // }
//!
//! // Or read specific count
//! // let mut reader = SnapshotReader::new(&buffer)?;
//! // for _ in 0..reader.entry_count() {
//! //     if let Some(entry) = reader.read_entry()? {
//! //         // process entry
//! //     }
//! // }
//! ```
//!
//! ## Legacy Format Compatibility
//!
//! ```rust
//! use tensor_chain::snapshot_streaming::deserialize_entries;
//!
//! // Automatically handles both streaming and legacy formats
//! // let data: &[u8] = ...;
//! // let entries = deserialize_entries(data)?;
//! ```
//!
//! # Error Handling
//!
//! Streaming operations can fail with:
//!
//! - [`StreamingError::Io`]: Underlying I/O error from buffer
//! - [`StreamingError::Buffer`]: Buffer-specific error (e.g., out of bounds)
//! - [`StreamingError::Serialization`]: Bincode serialization/deserialization error
//! - [`StreamingError::InvalidFormat`]: Magic mismatch, unsupported version, entry too large
//! - [`StreamingError::UnexpectedEof`]: Entry count mismatch or truncated data
//!
//! # Security Considerations
//!
//! - Entry size is limited to 100 MB to prevent memory exhaustion
//! - Magic bytes and version are validated to reject corrupted data
//! - Entry count in header is validated against actual entries
//!
//! # Performance
//!
//! - Zero-copy reads when buffer is in file mode (mmap)
//! - Memory usage bounded by `SnapshotBufferConfig::max_memory_bytes`
//! - Suitable for snapshots with millions of entries
//!
//! # See Also
//!
//! - [`crate::snapshot_buffer`]: Adaptive memory/disk buffer
//! - [`crate::network::LogEntry`]: Log entry type being serialized
//! - [`crate::raft`]: Raft consensus that uses snapshots

use std::io;

use crate::{
    network::LogEntry,
    snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig, SnapshotBufferError},
};

/// Error type for snapshot streaming operations.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("buffer error: {0}")]
    Buffer(#[from] SnapshotBufferError),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("unexpected end of data")]
    UnexpectedEof,
}

impl From<bitcode::Error> for StreamingError {
    fn from(e: bitcode::Error) -> Self {
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

/// Convert a slice to a fixed-size array, returning an error if lengths don't match.
fn slice_to_array<const N: usize>(slice: &[u8]) -> Result<[u8; N]> {
    slice.try_into().map_err(|_| {
        StreamingError::InvalidFormat(format!("expected {N} bytes, got {}", slice.len()))
    })
}

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
    /// # Errors
    /// Returns an error if the snapshot buffer cannot be created.
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

    /// # Errors
    /// Returns an error if the snapshot buffer cannot be created.
    pub fn with_defaults() -> Result<Self> {
        Self::new(SnapshotBufferConfig::default())
    }

    /// # Errors
    /// Returns an error if serialization or writing fails, or entry exceeds size limit.
    pub fn write_entry(&mut self, entry: &LogEntry) -> Result<()> {
        let bytes = bitcode::serialize(entry)?;
        #[allow(clippy::cast_possible_truncation)]
        let len = bytes.len() as u32;

        if bytes.len() > MAX_ENTRY_SIZE {
            return Err(StreamingError::InvalidFormat(format!(
                "entry too large: {} bytes (max {MAX_ENTRY_SIZE})",
                bytes.len(),
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

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    #[must_use]
    pub fn last_index(&self) -> u64 {
        self.last_index
    }

    #[must_use]
    pub fn last_term(&self) -> u64 {
        self.last_term
    }

    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.buffer.total_len()
    }

    /// Finalize the snapshot and return the buffer.
    ///
    /// This writes the header with entry count and returns the
    /// completed buffer for chunk serving.
    ///
    /// # Errors
    /// Returns an error if writing the header or copying data fails.
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
            #[allow(clippy::cast_possible_truncation)]
            let data_len = (total_len - HEADER_SIZE as u64) as usize;
            let data = self.buffer.as_slice(HEADER_SIZE as u64, data_len)?;
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
    /// # Errors
    /// Returns an error if the buffer is too small or has an invalid header.
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

        let version = u32::from_le_bytes(slice_to_array(&header[4..8])?);
        if version > STREAMING_VERSION {
            return Err(StreamingError::InvalidFormat(format!(
                "unsupported version: {version}"
            )));
        }

        let entry_count = u64::from_le_bytes(slice_to_array(&header[8..16])?);

        Ok(Self {
            buffer,
            read_offset: HEADER_SIZE as u64,
            entry_count,
            entries_read: 0,
        })
    }

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    #[must_use]
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }

    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.entry_count.saturating_sub(self.entries_read)
    }

    /// # Errors
    /// Returns an error if reading or deserializing an entry fails.
    pub fn read_entry(&mut self) -> Result<Option<LogEntry>> {
        if self.entries_read >= self.entry_count {
            return Ok(None);
        }

        // Read length prefix
        if self.read_offset + 4 > self.buffer.total_len() {
            return Err(StreamingError::UnexpectedEof);
        }

        let len_bytes = self.buffer.as_slice(self.read_offset, 4)?;
        let len = u32::from_le_bytes(slice_to_array(len_bytes)?) as usize;
        self.read_offset += 4;

        if len > MAX_ENTRY_SIZE {
            return Err(StreamingError::InvalidFormat(format!(
                "entry too large: {len} bytes"
            )));
        }

        // Read entry data
        if self.read_offset + len as u64 > self.buffer.total_len() {
            return Err(StreamingError::UnexpectedEof);
        }

        let entry_bytes = self.buffer.as_slice(self.read_offset, len)?;
        let entry: LogEntry = bitcode::deserialize(entry_bytes)?;
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
///
/// # Errors
/// Returns an error if the data is corrupted or cannot be deserialized.
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
        let entries: Vec<LogEntry> = bitcode::deserialize(data)?;
        Ok(entries)
    }
}

/// # Errors
/// Returns an error if serialization or buffer creation fails.
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
#[allow(clippy::field_reassign_with_default)]
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

    // ========== Error Handling Tests ==========

    #[test]
    fn test_streaming_reader_version_mismatch() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write header with version 99 (unsupported)
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&99u32.to_le_bytes()); // Invalid version
        header[8..16].copy_from_slice(&0u64.to_le_bytes());
        buffer.write(&header).unwrap();
        buffer.finalize().unwrap();

        let result = SnapshotReader::new(&buffer);
        assert!(result.is_err());
        match result {
            Err(StreamingError::InvalidFormat(msg)) => {
                assert!(msg.contains("unsupported version"));
            },
            _ => panic!("expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_streaming_reader_buffer_too_small() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write less than HEADER_SIZE bytes
        buffer.write(b"SNAP").unwrap(); // Only 4 bytes
        buffer.finalize().unwrap();

        let result = SnapshotReader::new(&buffer);
        assert!(result.is_err());
        match result {
            Err(StreamingError::InvalidFormat(msg)) => {
                assert!(msg.contains("too small"));
            },
            _ => panic!("expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_streaming_length_prefix_eof() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write valid header claiming 1 entry but no entry data
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes()); // Claims 1 entry
        buffer.write(&header).unwrap();
        // No entry data follows
        buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        let result = reader.read_entry();
        assert!(result.is_err());
        assert!(matches!(result, Err(StreamingError::UnexpectedEof)));
    }

    #[test]
    fn test_streaming_entry_data_eof() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write valid header claiming 1 entry
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        buffer.write(&header).unwrap();

        // Write length prefix claiming 100 bytes but only write 10
        buffer.write(&100u32.to_le_bytes()).unwrap();
        buffer.write(&[0u8; 10]).unwrap(); // Only 10 bytes, not 100
        buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        let result = reader.read_entry();
        assert!(result.is_err());
        assert!(matches!(result, Err(StreamingError::UnexpectedEof)));
    }

    #[test]
    fn test_streaming_corrupted_entry_count() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write header claiming 5 entries
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&5u64.to_le_bytes()); // Claims 5 entries
        buffer.write(&header).unwrap();

        // Only write 1 valid entry
        let entry = create_test_entry(1, 1);
        let bytes = bitcode::serialize(&entry).unwrap();
        buffer.write(&(bytes.len() as u32).to_le_bytes()).unwrap();
        buffer.write(&bytes).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        assert_eq!(reader.entry_count(), 5); // Claims 5

        // First entry should succeed
        assert!(reader.read_entry().unwrap().is_some());

        // Second entry should fail with EOF
        let result = reader.read_entry();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_invalid_magic_variations() {
        let test_cases = [
            b"XXXX",             // Wrong magic
            b"snap",             // Lowercase
            b"SNP\0",            // Truncated
            b"\x00\x00\x00\x00", // Null bytes
        ];

        for magic in test_cases {
            let config = test_config();
            let mut buffer = SnapshotBuffer::new(config).unwrap();

            let mut header = [0u8; HEADER_SIZE];
            header[0..4].copy_from_slice(magic);
            header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
            header[8..16].copy_from_slice(&0u64.to_le_bytes());
            buffer.write(&header).unwrap();
            buffer.finalize().unwrap();

            let result = SnapshotReader::new(&buffer);
            assert!(result.is_err(), "expected error for magic {:?}", magic);
        }
    }

    #[test]
    fn test_streaming_large_version_number() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        // Write header with max u32 version
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        header[8..16].copy_from_slice(&0u64.to_le_bytes());
        buffer.write(&header).unwrap();
        buffer.finalize().unwrap();

        let result = SnapshotReader::new(&buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_iterator_with_error() {
        // Use writer to create valid starting point, then manually construct corrupted snapshot
        let mut writer = SnapshotWriter::new(test_config()).unwrap();
        writer.write_entry(&create_test_entry(1, 1)).unwrap();
        let buffer = writer.finish().unwrap();

        // Get bytes, then create new buffer with corrupted entry count
        let bytes = buffer.as_bytes().unwrap();

        // Create a modified buffer claiming more entries than exist
        let config = test_config();
        let mut corrupt_buffer = SnapshotBuffer::new(config).unwrap();

        // Write modified header with entry_count = 3 (but only 1 entry exists)
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&3u64.to_le_bytes()); // Claim 3 entries
        corrupt_buffer.write(&header).unwrap();

        // Copy entry data (skip original header)
        if bytes.len() > HEADER_SIZE {
            corrupt_buffer.write(&bytes[HEADER_SIZE..]).unwrap();
        }
        corrupt_buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&corrupt_buffer).unwrap();
        assert_eq!(reader.entry_count(), 3);

        // First entry should succeed
        let first = reader.read_entry();
        assert!(first.is_ok());
        assert!(first.unwrap().is_some());

        // Second read should fail with EOF
        let second = reader.read_entry();
        assert!(second.is_err());
    }

    #[test]
    fn test_streaming_entry_too_large() {
        // Test that we reject entries larger than MAX_ENTRY_SIZE on read
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        buffer.write(&header).unwrap();

        // Write length prefix > MAX_ENTRY_SIZE (100MB + 1)
        let too_large = (MAX_ENTRY_SIZE + 1) as u32;
        buffer.write(&too_large.to_le_bytes()).unwrap();
        buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        let result = reader.read_entry();
        assert!(result.is_err());
        match result {
            Err(StreamingError::InvalidFormat(msg)) => {
                assert!(msg.contains("too large"));
            },
            _ => panic!("expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_streaming_deserialize_error() {
        let config = test_config();
        let mut buffer = SnapshotBuffer::new(config).unwrap();

        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&STREAMING_MAGIC);
        header[4..8].copy_from_slice(&STREAMING_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        buffer.write(&header).unwrap();

        // Write a small length prefix with garbage data
        buffer.write(&10u32.to_le_bytes()).unwrap();
        buffer.write(&[0xFF; 10]).unwrap(); // Invalid bincode data
        buffer.finalize().unwrap();

        let mut reader = SnapshotReader::new(&buffer).unwrap();
        let result = reader.read_entry();
        assert!(result.is_err());
        assert!(matches!(result, Err(StreamingError::Serialization(_))));
    }

    #[test]
    fn test_streaming_with_defaults() {
        let writer = SnapshotWriter::with_defaults().unwrap();
        assert_eq!(writer.entry_count(), 0);
        assert_eq!(writer.last_index(), 0);
        assert_eq!(writer.last_term(), 0);
    }

    #[test]
    fn test_deserialize_legacy_format() {
        // Test that deserialize_entries handles legacy bincode format
        let entries: Vec<LogEntry> = (1..=5).map(|i| create_test_entry(i, 1)).collect();

        // Serialize using legacy bincode format (not streaming)
        let legacy_bytes = bitcode::serialize(&entries).unwrap();

        // Should be able to deserialize
        let result = deserialize_entries(&legacy_bytes).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_error_display() {
        let io_err = StreamingError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let buf_err = StreamingError::Buffer(SnapshotBufferError::Io(io::Error::other("test")));
        assert!(buf_err.to_string().contains("buffer error"));

        let ser_err = StreamingError::Serialization("test".to_string());
        assert!(ser_err.to_string().contains("serialization error"));

        let fmt_err = StreamingError::InvalidFormat("test".to_string());
        assert!(fmt_err.to_string().contains("invalid format"));

        let eof_err = StreamingError::UnexpectedEof;
        assert!(eof_err.to_string().contains("unexpected end"));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From<io::Error>
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test");
        let streaming_err: StreamingError = io_err.into();
        assert!(matches!(streaming_err, StreamingError::Io(_)));

        // Test From<SnapshotBufferError>
        let buf_err = SnapshotBufferError::Io(io::Error::other("test"));
        let streaming_err: StreamingError = buf_err.into();
        assert!(matches!(streaming_err, StreamingError::Buffer(_)));
    }
}
