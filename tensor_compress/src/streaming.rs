//! Streaming compression for memory-bounded snapshot I/O.
//!
//! Enables processing large snapshots without loading the entire dataset into memory.
//! Uses a trailer-based header format where entry count is written at the end.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::io_other_error)]
#![allow(clippy::iter_with_drain)]

use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use crate::{
    format::{CompressedEntry, CompressedSnapshot, FormatError, Header},
    CompressionConfig,
};

/// Magic bytes for streaming format.
pub const STREAMING_MAGIC: [u8; 4] = *b"NEUS";

/// Streaming format version.
pub const STREAMING_VERSION: u16 = 1;

/// Streaming snapshot header (written as trailer at end of file).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamingHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub config: CompressionConfig,
    pub entry_count: u64,
    pub data_start: u64,
}

impl StreamingHeader {
    fn validate(&self) -> Result<(), FormatError> {
        if self.magic != STREAMING_MAGIC {
            return Err(FormatError::InvalidMagic);
        }
        if self.version > STREAMING_VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Streaming snapshot writer that writes entries incrementally.
pub struct StreamingWriter<W: Write> {
    writer: BufWriter<W>,
    config: CompressionConfig,
    entry_count: u64,
    bytes_written: u64,
    data_start: u64,
}

impl<W: Write> StreamingWriter<W> {
    /// Create a new streaming writer.
    pub fn new(writer: W, config: CompressionConfig) -> io::Result<Self> {
        let mut buf_writer = BufWriter::new(writer);

        // Write placeholder magic at start (will help detect incomplete files)
        buf_writer.write_all(&STREAMING_MAGIC)?;
        let data_start = STREAMING_MAGIC.len() as u64;

        Ok(Self {
            writer: buf_writer,
            config,
            entry_count: 0,
            bytes_written: data_start,
            data_start,
        })
    }

    /// Write a single compressed entry.
    pub fn write_entry(&mut self, entry: &CompressedEntry) -> Result<(), FormatError> {
        let bytes = bincode::serialize(entry)?;
        let len = bytes.len() as u32;

        // Write length prefix then entry data
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;

        self.entry_count += 1;
        self.bytes_written += 4 + bytes.len() as u64;
        Ok(())
    }

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

impl<W: Write + Seek> StreamingWriter<W> {
    /// Finish writing, write trailer with final entry count.
    /// Returns the inner writer.
    pub fn finish(mut self) -> Result<W, FormatError> {
        let header = StreamingHeader {
            magic: STREAMING_MAGIC,
            version: STREAMING_VERSION,
            config: self.config,
            entry_count: self.entry_count,
            data_start: self.data_start,
        };

        let trailer_bytes = bincode::serialize(&header)?;
        self.writer.write_all(&trailer_bytes)?;

        // Write trailer size at very end for easy seeking
        let trailer_len = trailer_bytes.len() as u64;
        self.writer.write_all(&trailer_len.to_le_bytes())?;

        self.writer.flush()?;
        self.writer
            .into_inner()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()).into())
    }
}

/// Streaming snapshot reader that yields entries one at a time.
pub struct StreamingReader<R: Read> {
    reader: BufReader<R>,
    header: StreamingHeader,
    entries_read: u64,
}

/// Maximum trailer size to prevent allocation attacks (1 MB).
const MAX_TRAILER_SIZE: u64 = 1024 * 1024;

/// Maximum entry size to prevent allocation attacks (100 MB).
const MAX_ENTRY_SIZE: usize = 100 * 1024 * 1024;

impl<R: Read + Seek> StreamingReader<R> {
    /// Open a streaming snapshot, reading header from trailer.
    pub fn open(mut reader: R) -> Result<Self, FormatError> {
        // Read trailer size from end
        reader.seek(SeekFrom::End(-8))?;
        let mut trailer_len_bytes = [0u8; 8];
        reader.read_exact(&mut trailer_len_bytes)?;
        let trailer_len = u64::from_le_bytes(trailer_len_bytes);

        // Sanity check trailer size to prevent allocation attacks
        if trailer_len > MAX_TRAILER_SIZE {
            return Err(FormatError::InvalidMagic);
        }

        // Read trailer
        reader.seek(SeekFrom::End(-(8 + trailer_len as i64)))?;
        let mut trailer_bytes = vec![0u8; trailer_len as usize];
        reader.read_exact(&mut trailer_bytes)?;

        let header: StreamingHeader = bincode::deserialize(&trailer_bytes)?;
        header.validate()?;

        // Seek to data start
        reader.seek(SeekFrom::Start(header.data_start))?;

        Ok(Self {
            reader: BufReader::new(reader),
            header,
            entries_read: 0,
        })
    }

    #[must_use]
    pub fn header(&self) -> &StreamingHeader {
        &self.header
    }

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.header.entry_count
    }

    #[must_use]
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }

    #[must_use]
    pub fn has_next(&self) -> bool {
        self.entries_read < self.header.entry_count
    }
}

impl<R: Read> Iterator for StreamingReader<R> {
    type Item = Result<CompressedEntry, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.entries_read >= self.header.entry_count {
            return None;
        }

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len_bytes) {
            return Some(Err(FormatError::from(bincode::Error::from(e))));
        }
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Sanity check entry size to prevent allocation attacks
        if len > MAX_ENTRY_SIZE {
            return Some(Err(FormatError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "entry size exceeds maximum",
            ))));
        }

        // Read entry data
        let mut entry_bytes = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut entry_bytes) {
            return Some(Err(FormatError::from(bincode::Error::from(e))));
        }

        match bincode::deserialize(&entry_bytes) {
            Ok(entry) => {
                self.entries_read += 1;
                Some(Ok(entry))
            },
            Err(e) => Some(Err(FormatError::from(e))),
        }
    }
}

/// Convert a non-streaming snapshot to streaming format.
pub fn convert_to_streaming<W: Write + Seek>(
    snapshot: &CompressedSnapshot,
    writer: W,
) -> Result<u64, FormatError> {
    let config = snapshot.header.config.clone();
    let mut stream_writer = StreamingWriter::new(writer, config)?;

    for entry in &snapshot.entries {
        stream_writer.write_entry(entry)?;
    }

    stream_writer.finish()?;
    Ok(snapshot.entries.len() as u64)
}

/// Read streaming format into a full snapshot (for compatibility).
pub fn read_streaming_to_snapshot<R: Read + Seek>(
    reader: R,
) -> Result<CompressedSnapshot, FormatError> {
    let stream_reader = StreamingReader::open(reader)?;
    let config = stream_reader.header.config.clone();
    let entry_count = stream_reader.entry_count();

    let entries: Result<Vec<_>, _> = stream_reader.collect();
    let entries = entries?;

    let header = Header::new(config, entry_count);
    Ok(CompressedSnapshot { header, entries })
}

/// Merge multiple streaming snapshots into one.
pub fn merge_streaming<W: Write + Seek, R: Read + Seek>(
    mut readers: Vec<R>,
    writer: W,
    config: CompressionConfig,
) -> Result<u64, FormatError> {
    let mut stream_writer = StreamingWriter::new(writer, config)?;

    for reader in readers.drain(..) {
        let stream_reader = StreamingReader::open(reader)?;
        for entry_result in stream_reader {
            let entry = entry_result?;
            stream_writer.write_entry(&entry)?;
        }
    }

    let count = stream_writer.entry_count();
    stream_writer.finish()?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Cursor};

    use super::*;
    use crate::format::{CompressedScalar, CompressedValue};

    fn make_test_entry(key: &str, value: i64) -> CompressedEntry {
        CompressedEntry {
            key: key.to_string(),
            fields: HashMap::from([(
                "value".to_string(),
                CompressedValue::Scalar(CompressedScalar::Int(value)),
            )]),
        }
    }

    #[test]
    fn test_streaming_writer_new() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let writer = StreamingWriter::new(cursor, config).unwrap();
        assert_eq!(writer.entry_count(), 0);
        assert_eq!(writer.bytes_written(), 4); // Magic bytes
    }

    #[test]
    fn test_streaming_write_single_entry() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config).unwrap();

        let entry = make_test_entry("key1", 42);
        writer.write_entry(&entry).unwrap();

        assert_eq!(writer.entry_count(), 1);
        assert!(writer.bytes_written() > 4);
    }

    #[test]
    fn test_streaming_roundtrip() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config.clone()).unwrap();

        // Write multiple entries
        for i in 0..10 {
            let entry = make_test_entry(&format!("key{}", i), i);
            writer.write_entry(&entry).unwrap();
        }

        let written = writer.finish().unwrap();

        // Read back
        let reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.entry_count(), 10);

        let entries: Vec<_> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 10);
        assert_eq!(entries[0].key, "key0");
        assert_eq!(entries[9].key, "key9");
    }

    #[test]
    fn test_streaming_iterator() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config).unwrap();

        for i in 0..5 {
            writer
                .write_entry(&make_test_entry(&format!("k{}", i), i))
                .unwrap();
        }

        let written = writer.finish().unwrap();
        let reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();

        let mut count = 0;
        for entry in reader {
            entry.unwrap();
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_streaming_has_next() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config).unwrap();
        writer.write_entry(&make_test_entry("k1", 1)).unwrap();
        let written = writer.finish().unwrap();

        let mut reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();
        assert!(reader.has_next());
        reader.next();
        assert!(!reader.has_next());
    }

    #[test]
    fn test_streaming_empty_snapshot() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let writer = StreamingWriter::new(cursor, config).unwrap();
        let written = writer.finish().unwrap();

        let reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.entry_count(), 0);
        assert!(!reader.has_next());

        let entries: Vec<_> = reader.collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_streaming_header_validation() {
        let mut bad_data = vec![0u8; 100];
        bad_data[0..4].copy_from_slice(b"BAAD");

        // Write fake trailer size
        let fake_trailer_len = 20u64;
        bad_data[92..100].copy_from_slice(&fake_trailer_len.to_le_bytes());

        let result = StreamingReader::open(Cursor::new(bad_data));
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_to_streaming() {
        let header = Header::new(CompressionConfig::default(), 2);
        let snapshot = CompressedSnapshot {
            header,
            entries: vec![make_test_entry("a", 1), make_test_entry("b", 2)],
        };

        let cursor = Cursor::new(Vec::new());
        let count = convert_to_streaming(&snapshot, cursor).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_read_streaming_to_snapshot() {
        // Create streaming data
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config).unwrap();
        writer.write_entry(&make_test_entry("x", 10)).unwrap();
        writer.write_entry(&make_test_entry("y", 20)).unwrap();
        let written = writer.finish().unwrap();

        // Convert back to snapshot
        let snapshot = read_streaming_to_snapshot(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.header.entry_count, 2);
    }

    #[test]
    fn test_merge_streaming() {
        // Create two streaming files
        let create_streaming = |entries: Vec<(&str, i64)>| {
            let cursor = Cursor::new(Vec::new());
            let config = CompressionConfig::default();
            let mut writer = StreamingWriter::new(cursor, config).unwrap();
            for (k, v) in entries {
                writer.write_entry(&make_test_entry(k, v)).unwrap();
            }
            Cursor::new(writer.finish().unwrap().into_inner())
        };

        let stream1 = create_streaming(vec![("a", 1), ("b", 2)]);
        let stream2 = create_streaming(vec![("c", 3), ("d", 4)]);

        let output = Cursor::new(Vec::new());
        let count =
            merge_streaming(vec![stream1, stream2], output, CompressionConfig::default()).unwrap();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_streaming_large_snapshot() {
        let cursor = Cursor::new(Vec::new());
        let config = CompressionConfig::default();
        let mut writer = StreamingWriter::new(cursor, config).unwrap();

        // Write 1000 entries
        for i in 0..1000 {
            writer
                .write_entry(&make_test_entry(&format!("key_{}", i), i))
                .unwrap();
        }

        let written = writer.finish().unwrap();
        let reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.entry_count(), 1000);

        // Verify we can iterate through all
        let entries: Vec<_> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 1000);
    }
}
