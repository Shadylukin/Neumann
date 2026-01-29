// SPDX-License-Identifier: MIT OR Apache-2.0
//! Streaming TT decomposition for memory-bounded I/O.
//!
//! Enables processing large vector collections without loading all into memory.
//! Uses a trailer-based format where entry count is written at the end.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::io_other_error)]

use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use crate::{
    format::FormatError,
    tensor_train::{tt_cosine_similarity, tt_decompose, TTConfig, TTError, TTVector},
};

/// Magic bytes for streaming TT format.
pub const STREAMING_TT_MAGIC: [u8; 4] = *b"NEUT";

/// Streaming TT format version.
pub const STREAMING_TT_VERSION: u16 = 1;

/// Streaming TT header (written as trailer at end of file).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamingTTHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub config: TTConfig,
    pub vector_count: u64,
    pub data_start: u64,
}

impl StreamingTTHeader {
    fn validate(&self) -> Result<(), FormatError> {
        if self.magic != STREAMING_TT_MAGIC {
            return Err(FormatError::InvalidMagic);
        }
        if self.version > STREAMING_TT_VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Maximum trailer size (1 MB).
const MAX_TRAILER_SIZE: u64 = 1024 * 1024;

/// Maximum entry size (100 MB).
const MAX_ENTRY_SIZE: usize = 100 * 1024 * 1024;

/// Streaming TT writer that writes TT vectors incrementally.
pub struct StreamingTTWriter<W: Write> {
    writer: BufWriter<W>,
    config: TTConfig,
    vector_count: u64,
    bytes_written: u64,
    data_start: u64,
}

impl<W: Write> StreamingTTWriter<W> {
    /// Create a new streaming TT writer.
    pub fn new(writer: W, config: TTConfig) -> io::Result<Self> {
        let mut buf_writer = BufWriter::new(writer);

        buf_writer.write_all(&STREAMING_TT_MAGIC)?;
        let data_start = STREAMING_TT_MAGIC.len() as u64;

        Ok(Self {
            writer: buf_writer,
            config,
            vector_count: 0,
            bytes_written: data_start,
            data_start,
        })
    }

    /// Decompose and write a single vector.
    pub fn write_vector(&mut self, vector: &[f32]) -> Result<(), TTError> {
        let tt = tt_decompose(vector, &self.config)?;
        self.write_tt(&tt).map_err(|e| match e {
            FormatError::Io(io_err) => TTError::InvalidShape(io_err.to_string()),
            other => TTError::InvalidShape(other.to_string()),
        })
    }

    /// Write a pre-decomposed TT vector.
    pub fn write_tt(&mut self, tt: &TTVector) -> Result<(), FormatError> {
        let bytes = bitcode::serialize(tt)?;
        let len = bytes.len() as u32;

        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;

        self.vector_count += 1;
        self.bytes_written += 4 + bytes.len() as u64;
        Ok(())
    }

    #[must_use]
    pub fn vector_count(&self) -> u64 {
        self.vector_count
    }

    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    #[must_use]
    pub fn config(&self) -> &TTConfig {
        &self.config
    }
}

impl<W: Write + Seek> StreamingTTWriter<W> {
    /// Finish writing, write trailer with final vector count.
    pub fn finish(mut self) -> Result<W, FormatError> {
        let header = StreamingTTHeader {
            magic: STREAMING_TT_MAGIC,
            version: STREAMING_TT_VERSION,
            config: self.config,
            vector_count: self.vector_count,
            data_start: self.data_start,
        };

        let trailer_bytes = bitcode::serialize(&header)?;
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

/// Streaming TT reader that yields TT vectors one at a time.
pub struct StreamingTTReader<R: Read> {
    reader: BufReader<R>,
    header: StreamingTTHeader,
    vectors_read: u64,
}

impl<R: Read + Seek> StreamingTTReader<R> {
    /// Open a streaming TT file, reading header from trailer.
    pub fn open(mut reader: R) -> Result<Self, FormatError> {
        reader.seek(SeekFrom::End(-8))?;
        let mut trailer_len_bytes = [0u8; 8];
        reader.read_exact(&mut trailer_len_bytes)?;
        let trailer_len = u64::from_le_bytes(trailer_len_bytes);

        if trailer_len > MAX_TRAILER_SIZE {
            return Err(FormatError::InvalidMagic);
        }

        reader.seek(SeekFrom::End(-(8 + trailer_len as i64)))?;
        let mut trailer_bytes = vec![0u8; trailer_len as usize];
        reader.read_exact(&mut trailer_bytes)?;

        let header: StreamingTTHeader = bitcode::deserialize(&trailer_bytes)?;
        header.validate()?;

        reader.seek(SeekFrom::Start(header.data_start))?;

        Ok(Self {
            reader: BufReader::new(reader),
            header,
            vectors_read: 0,
        })
    }

    #[must_use]
    pub fn header(&self) -> &StreamingTTHeader {
        &self.header
    }

    #[must_use]
    pub fn vector_count(&self) -> u64 {
        self.header.vector_count
    }

    #[must_use]
    pub fn vectors_read(&self) -> u64 {
        self.vectors_read
    }

    #[must_use]
    pub fn has_next(&self) -> bool {
        self.vectors_read < self.header.vector_count
    }

    #[must_use]
    pub fn config(&self) -> &TTConfig {
        &self.header.config
    }
}

impl<R: Read> Iterator for StreamingTTReader<R> {
    type Item = Result<TTVector, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.vectors_read >= self.header.vector_count {
            return None;
        }

        let mut len_bytes = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len_bytes) {
            return Some(Err(FormatError::Io(e)));
        }
        let len = u32::from_le_bytes(len_bytes) as usize;

        if len > MAX_ENTRY_SIZE {
            return Some(Err(FormatError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "entry size exceeds maximum",
            ))));
        }

        let mut entry_bytes = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut entry_bytes) {
            return Some(Err(FormatError::Io(e)));
        }

        match bitcode::deserialize(&entry_bytes) {
            Ok(tt) => {
                self.vectors_read += 1;
                Some(Ok(tt))
            },
            Err(e) => Some(Err(FormatError::from(e))),
        }
    }
}

/// Convert dense vectors to streaming TT format.
pub fn convert_vectors_to_streaming_tt<W, I>(
    vectors: I,
    writer: W,
    config: &TTConfig,
) -> Result<u64, TTError>
where
    W: Write + Seek,
    I: Iterator<Item = Vec<f32>>,
{
    let mut stream_writer = StreamingTTWriter::new(writer, config.clone())
        .map_err(|e| TTError::InvalidShape(e.to_string()))?;

    for vector in vectors {
        stream_writer.write_vector(&vector)?;
    }

    let count = stream_writer.vector_count();
    stream_writer
        .finish()
        .map_err(|e| TTError::InvalidShape(e.to_string()))?;
    Ok(count)
}

/// Perform similarity search on streaming TT data.
///
/// Returns top-k (index, similarity) pairs sorted by descending similarity.
pub fn streaming_tt_similarity_search<R: Read + Seek>(
    reader: R,
    query_tt: &TTVector,
    top_k: usize,
) -> Result<Vec<(u64, f32)>, FormatError> {
    let stream_reader = StreamingTTReader::open(reader)?;

    let mut results: Vec<(u64, f32)> = Vec::new();

    for (idx, tt_result) in stream_reader.enumerate() {
        let tt = tt_result?;
        if let Ok(sim) = tt_cosine_similarity(query_tt, &tt) {
            results.push((idx as u64, sim));
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    Ok(results)
}

/// Read all TT vectors from streaming format into memory.
pub fn read_streaming_tt_all<R: Read + Seek>(reader: R) -> Result<Vec<TTVector>, FormatError> {
    let stream_reader = StreamingTTReader::open(reader)?;
    stream_reader.collect()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn make_test_vector(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((i + seed) as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_streaming_tt_writer_new() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let writer = StreamingTTWriter::new(cursor, config).unwrap();
        assert_eq!(writer.vector_count(), 0);
        assert_eq!(writer.bytes_written(), 4); // Magic bytes
    }

    #[test]
    fn test_streaming_tt_write_single_vector() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();

        let vector = make_test_vector(0, 64);
        writer.write_vector(&vector).unwrap();

        assert_eq!(writer.vector_count(), 1);
        assert!(writer.bytes_written() > 4);
    }

    #[test]
    fn test_streaming_tt_roundtrip() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let mut writer = StreamingTTWriter::new(cursor, config.clone()).unwrap();

        // Write multiple vectors
        for i in 0..10 {
            let vector = make_test_vector(i, 64);
            writer.write_vector(&vector).unwrap();
        }

        let written = writer.finish().unwrap();

        // Read back
        let reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.vector_count(), 10);
        assert_eq!(reader.config(), &config);

        let tts: Vec<_> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(tts.len(), 10);
    }

    #[test]
    fn test_streaming_tt_iterator() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();

        for i in 0..5 {
            writer.write_vector(&make_test_vector(i, 64)).unwrap();
        }

        let written = writer.finish().unwrap();
        let reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();

        let mut count = 0;
        for tt in reader {
            tt.unwrap();
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_streaming_tt_has_next() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();
        writer.write_vector(&make_test_vector(0, 64)).unwrap();
        let written = writer.finish().unwrap();

        let mut reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
        assert!(reader.has_next());
        reader.next();
        assert!(!reader.has_next());
    }

    #[test]
    fn test_streaming_tt_empty() {
        let cursor = Cursor::new(Vec::new());
        let config = TTConfig::for_dim(64).unwrap();
        let writer = StreamingTTWriter::new(cursor, config).unwrap();
        let written = writer.finish().unwrap();

        let reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.vector_count(), 0);
        assert!(!reader.has_next());

        let tts: Vec<_> = reader.collect();
        assert!(tts.is_empty());
    }

    #[test]
    fn test_streaming_tt_header_validation() {
        let mut bad_data = vec![0u8; 100];
        bad_data[0..4].copy_from_slice(b"BAAD");

        // Write fake trailer size
        let fake_trailer_len = 20u64;
        bad_data[92..100].copy_from_slice(&fake_trailer_len.to_le_bytes());

        let result = StreamingTTReader::open(Cursor::new(bad_data));
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_vectors_to_streaming_tt() {
        let config = TTConfig::for_dim(64).unwrap();
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| make_test_vector(i, 64)).collect();

        let cursor = Cursor::new(Vec::new());
        let count = convert_vectors_to_streaming_tt(vectors.into_iter(), cursor, &config).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_streaming_tt_similarity_search() {
        let config = TTConfig::for_dim(64).unwrap();

        // Create streaming data
        let cursor = Cursor::new(Vec::new());
        let mut writer = StreamingTTWriter::new(cursor, config.clone()).unwrap();
        for i in 0..10 {
            writer.write_vector(&make_test_vector(i, 64)).unwrap();
        }
        let written = writer.finish().unwrap();

        // Create query
        let query_vec = make_test_vector(0, 64);
        let query_tt = tt_decompose(&query_vec, &config).unwrap();

        // Search
        let results =
            streaming_tt_similarity_search(Cursor::new(written.into_inner()), &query_tt, 3)
                .unwrap();
        assert_eq!(results.len(), 3);

        // First result should be index 0 (most similar to itself)
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.9); // Self-similarity should be high
    }

    #[test]
    fn test_read_streaming_tt_all() {
        let config = TTConfig::for_dim(64).unwrap();

        let cursor = Cursor::new(Vec::new());
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();
        for i in 0..3 {
            writer.write_vector(&make_test_vector(i, 64)).unwrap();
        }
        let written = writer.finish().unwrap();

        let tts = read_streaming_tt_all(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(tts.len(), 3);
    }

    #[test]
    fn test_streaming_tt_large_batch() {
        let config = TTConfig::for_dim(64).unwrap();

        let cursor = Cursor::new(Vec::new());
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();

        // Write 100 vectors
        for i in 0..100 {
            writer.write_vector(&make_test_vector(i, 64)).unwrap();
        }

        let written = writer.finish().unwrap();
        let reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
        assert_eq!(reader.vector_count(), 100);

        let tts: Vec<_> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(tts.len(), 100);
    }

    #[test]
    fn test_streaming_tt_write_pre_decomposed() {
        let config = TTConfig::for_dim(64).unwrap();

        // Pre-decompose a vector
        let vector = make_test_vector(42, 64);
        let tt = tt_decompose(&vector, &config).unwrap();

        // Write as pre-decomposed TT
        let cursor = Cursor::new(Vec::new());
        let mut writer = StreamingTTWriter::new(cursor, config).unwrap();
        writer.write_tt(&tt).unwrap();
        let written = writer.finish().unwrap();

        // Read back and verify
        let mut reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
        let read_tt = reader.next().unwrap().unwrap();
        assert_eq!(tt.shape, read_tt.shape);
        assert_eq!(tt.original_dim, read_tt.original_dim);
    }

    #[test]
    fn test_streaming_tt_invalid_magic() {
        use crate::format::FormatError;

        // Create data with wrong magic bytes (too short to even read trailer)
        let bad_data = vec![0u8; 100];
        let result = StreamingTTReader::open(Cursor::new(bad_data));
        assert!(result.is_err());

        // Also test via header validation directly
        let header = StreamingTTHeader {
            magic: [0, 0, 0, 0], // Wrong magic
            version: STREAMING_TT_VERSION,
            config: TTConfig::for_dim(64).unwrap(),
            vector_count: 0,
            data_start: 4,
        };
        assert!(matches!(header.validate(), Err(FormatError::InvalidMagic)));
    }

    #[test]
    fn test_streaming_tt_unsupported_version() {
        use crate::format::FormatError;

        let header = StreamingTTHeader {
            magic: STREAMING_TT_MAGIC,
            version: STREAMING_TT_VERSION + 100, // Future version
            config: TTConfig::for_dim(64).unwrap(),
            vector_count: 0,
            data_start: 4,
        };
        assert!(matches!(
            header.validate(),
            Err(FormatError::UnsupportedVersion(_))
        ));
    }
}
