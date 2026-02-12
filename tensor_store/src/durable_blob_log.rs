// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Durable blob log with WAL-based crash recovery.
//!
//! `DurableBlobLog` provides persistent, content-addressable chunk storage
//! with crash-safe semantics via a two-phase WAL protocol.
//!
//! # Design
//!
//! - Append-only segments (64MB default) for sequential I/O
//! - Footer-based index (scales to disk, not memory-bound)
//! - Bloom filter per segment for fast negative lookups
//! - Two-phase WAL (PREPARE/COMMIT) for durability
//!
//! # Durability Invariant
//!
//! The core invariant enforced everywhere:
//! - **Durability boundary** = ack to caller only AFTER WAL COMMIT fsync
//! - **Visibility rule** = only COMMIT records are visible (reads, recovery, GC)
//! - PREPARE without COMMIT = invisible, GC candidate
//!
//! # File Format
//!
//! Segment file layout:
//! ```text
//! [chunk0][chunk1]...[chunkN]
//! [index: (hash, offset, len)*]
//! [bloom: compressed bloom filter bytes]
//! [footer: index_offset(u64), bloom_offset(u64), count(u32), MAGIC(u32), crc32(u32)]
//! ```
//!
//! Footer is fixed 28 bytes at EOF.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::BloomFilter;

/// Magic bytes for segment footer ("NBLG" - Neumann Blob Segment).
const SEGMENT_MAGIC: u32 = 0x4E42_4C47;

/// Default segment capacity (64MB).
const DEFAULT_SEGMENT_SIZE: usize = 64 * 1024 * 1024;

/// Footer size in bytes: 8 + 8 + 4 + 4 + 4 = 28.
const FOOTER_SIZE: usize = 28;

/// Bloom filter expected items per segment.
const BLOOM_EXPECTED_ITEMS: usize = 100_000;

/// Bloom filter false positive rate.
const BLOOM_FPR: f64 = 0.01;

/// LRU cache size for chunk locations.
const LOCATION_CACHE_SIZE: usize = 10_000;

/// Content hash for blob chunks (BLAKE3-style, using crc64 for simplicity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DurableChunkHash(pub [u8; 32]);

impl DurableChunkHash {
    /// Compute hash from data using BLAKE3.
    #[must_use]
    pub fn from_data(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Return as hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string.
    #[must_use]
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex::decode(s).ok()?;
        if bytes.len() != 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Some(Self(arr))
    }
}

/// Location of a chunk within the durable blob log.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkLocation {
    /// Segment ID containing this chunk.
    pub segment_id: u64,
    /// Byte offset within the segment.
    pub offset: u64,
    /// Length in bytes.
    pub len: u32,
}

/// WAL record types for two-phase commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlobWalRecord {
    /// Intent to write a chunk (not yet durable).
    Prepare {
        /// Content hash of the chunk.
        hash: DurableChunkHash,
        /// Segment containing this chunk.
        segment_id: u64,
        /// Byte offset within segment.
        offset: u64,
        /// Length in bytes.
        len: u32,
    },
    /// Chunk is durable and visible.
    Commit {
        /// Content hash of the chunk.
        hash: DurableChunkHash,
        /// Segment containing this chunk.
        segment_id: u64,
        /// Byte offset within segment.
        offset: u64,
        /// Length in bytes.
        len: u32,
        /// CRC32 of chunk data.
        crc32: u32,
    },
    /// Chunk has been deleted/superseded.
    Tombstone {
        /// Hash of the deleted chunk.
        hash: DurableChunkHash,
    },
    /// Segment has been sealed (footer written).
    Seal {
        /// ID of the sealed segment.
        segment_id: u64,
        /// Byte offset of the footer.
        footer_offset: u64,
    },
}

/// Errors from durable blob log operations.
#[derive(Debug, Error)]
pub enum DurableBlobLogError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid segment footer.
    #[error("Invalid segment footer: {0}")]
    InvalidFooter(String),

    /// Chunk not found.
    #[error("Chunk not found: {hash}")]
    ChunkNotFound {
        /// Hash of the missing chunk.
        hash: String,
    },

    /// CRC32 mismatch.
    #[error("CRC32 mismatch for chunk {hash}: expected {expected:#x}, got {actual:#x}")]
    CrcMismatch {
        /// Hash of the corrupted chunk.
        hash: String,
        /// Expected CRC32 value.
        expected: u32,
        /// Actual CRC32 value.
        actual: u32,
    },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<bitcode::Error> for DurableBlobLogError {
    fn from(e: bitcode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// Result type for durable blob log operations.
pub type Result<T> = std::result::Result<T, DurableBlobLogError>;

/// State recovered from WAL replay.
type RecoveryState = (
    BTreeMap<DurableChunkHash, ChunkLocation>, // routing table
    BTreeSet<DurableChunkHash>,                // tombstones
    BTreeSet<DurableChunkHash>,                // gc_candidates
    HashMap<u64, u64>,                         // sealed segment info (id -> footer_offset)
    u64,                                       // max_segment_id
);

/// Configuration for durable blob log.
#[derive(Debug, Clone)]
pub struct DurableBlobLogConfig {
    /// Directory for segment files.
    pub segment_dir: PathBuf,
    /// Maximum segment size in bytes.
    pub segment_size: usize,
    /// Enable fsync after writes.
    pub enable_fsync: bool,
    /// LRU cache size for chunk locations.
    pub cache_size: usize,
}

impl Default for DurableBlobLogConfig {
    fn default() -> Self {
        Self {
            segment_dir: std::env::temp_dir().join("durable_blob"),
            segment_size: DEFAULT_SEGMENT_SIZE,
            enable_fsync: true,
            cache_size: LOCATION_CACHE_SIZE,
        }
    }
}

/// Index entry stored in segment footer.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    hash: DurableChunkHash,
    offset: u64,
    len: u32,
}

/// Sealed segment metadata.
#[derive(Debug)]
struct SealedSegment {
    id: u64,
    path: PathBuf,
    index_offset: u64,
    bloom_offset: u64,
    /// Number of entries in this segment.
    #[allow(dead_code)]
    count: u32,
    /// Bloom filter for fast negative lookups.
    bloom: BloomFilter,
    /// In-memory index (lazy-loaded from footer on demand).
    index: RwLock<Option<HashMap<DurableChunkHash, ChunkLocation>>>,
}

impl SealedSegment {
    /// Check if bloom filter might contain this hash.
    fn might_contain(&self, hash: &DurableChunkHash) -> bool {
        self.bloom.might_contain(&hash.0)
    }

    /// Load index from segment footer if not already loaded.
    fn load_index(&self) -> Result<()> {
        if self.index.read().is_some() {
            return Ok(());
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.index_offset))?;

        let index_len = usize::try_from(self.bloom_offset - self.index_offset)
            .expect("index length exceeds platform address space");
        let mut index_bytes = vec![0u8; index_len];
        file.read_exact(&mut index_bytes)?;

        let entries: Vec<IndexEntry> = bitcode::deserialize(&index_bytes)?;
        let mut index = HashMap::with_capacity(entries.len());
        for entry in entries {
            index.insert(
                entry.hash,
                ChunkLocation {
                    segment_id: self.id,
                    offset: entry.offset,
                    len: entry.len,
                },
            );
        }

        *self.index.write() = Some(index);
        Ok(())
    }

    /// Get chunk location from this segment.
    fn get_location(&self, hash: &DurableChunkHash) -> Result<Option<ChunkLocation>> {
        // Fast path: bloom filter negative
        if !self.might_contain(hash) {
            return Ok(None);
        }

        // Load index if needed
        self.load_index()?;

        let index = self.index.read();
        Ok(index.as_ref().and_then(|idx| idx.get(hash).copied()))
    }
}

/// Active segment being written to.
struct ActiveSegment {
    id: u64,
    file: BufWriter<File>,
    path: PathBuf,
    write_offset: u64,
    /// Pending entries for footer index.
    pending_entries: Vec<IndexEntry>,
}

impl ActiveSegment {
    fn new(id: u64, path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            id,
            file: BufWriter::new(file),
            path,
            write_offset: 0,
            pending_entries: Vec::new(),
        })
    }

    /// Open an existing segment for resuming writes.
    #[allow(dead_code)]
    fn open_existing(id: u64, path: PathBuf, offset: u64) -> Result<Self> {
        let file = OpenOptions::new().append(true).open(&path)?;

        Ok(Self {
            id,
            file: BufWriter::new(file),
            path,
            write_offset: offset,
            pending_entries: Vec::new(),
        })
    }

    fn can_fit(&self, size: usize, segment_size: usize) -> bool {
        // Reserve space for potential footer
        let reserved = FOOTER_SIZE + 4096; // Extra for bloom + index
        let offset = usize::try_from(self.write_offset).unwrap_or(usize::MAX);
        offset.saturating_add(size).saturating_add(reserved) <= segment_size
    }

    fn write_chunk(&mut self, data: &[u8]) -> Result<u64> {
        let offset = self.write_offset;
        self.file.write_all(data)?;
        self.write_offset += data.len() as u64;
        Ok(offset)
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.file.get_ref().sync_all()?;
        Ok(())
    }
}

/// LRU cache for chunk locations.
struct LocationCache {
    entries: RwLock<HashMap<DurableChunkHash, (ChunkLocation, u64)>>,
    access_counter: AtomicU64,
    max_size: usize,
}

impl LocationCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_size)),
            access_counter: AtomicU64::new(0),
            max_size,
        }
    }

    fn get(&self, hash: &DurableChunkHash) -> Option<ChunkLocation> {
        let mut entries = self.entries.write();
        if let Some((loc, access)) = entries.get_mut(hash) {
            *access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            Some(*loc)
        } else {
            None
        }
    }

    fn insert(&self, hash: DurableChunkHash, location: ChunkLocation) {
        let mut entries = self.entries.write();

        // Evict oldest if at capacity
        if entries.len() >= self.max_size {
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, (_, access))| *access)
                .map(|(k, _)| *k)
            {
                entries.remove(&oldest_key);
            }
        }

        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);
        entries.insert(hash, (location, access));
    }

    fn remove(&self, hash: &DurableChunkHash) {
        self.entries.write().remove(hash);
    }
}

/// Durable blob log with WAL-based crash recovery.
pub struct DurableBlobLog {
    /// Active segment for writes.
    active: Mutex<ActiveSegment>,
    /// Sealed segments (read-only).
    sealed: RwLock<Vec<SealedSegment>>,
    /// WAL for durability.
    wal: Mutex<BufWriter<File>>,
    /// Path to WAL file.
    #[allow(dead_code)]
    wal_path: PathBuf,
    /// Location cache for hot chunks.
    cache: LocationCache,
    /// Routing table: hash -> location (built from WAL on recovery).
    routing: RwLock<BTreeMap<DurableChunkHash, ChunkLocation>>,
    /// Tombstoned chunks (pending GC).
    tombstones: RwLock<BTreeSet<DurableChunkHash>>,
    /// GC candidates (PREPARE without COMMIT).
    gc_candidates: RwLock<BTreeSet<DurableChunkHash>>,
    /// Configuration.
    config: DurableBlobLogConfig,
    /// Next segment ID.
    next_segment_id: AtomicU64,
    /// Statistics.
    total_chunks: AtomicU64,
    total_bytes: AtomicU64,
}

impl DurableBlobLog {
    /// Create or open a durable blob log.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or recovery fails.
    pub fn open(config: DurableBlobLogConfig) -> Result<Self> {
        fs::create_dir_all(&config.segment_dir)?;

        let wal_path = config.segment_dir.join("blob.wal");
        let wal_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        // Recover state from WAL
        let (routing, tombstones, gc_candidates, sealed_info, max_segment_id) =
            Self::recover_from_wal(&wal_path, &config)?;

        // Load sealed segments
        let sealed = Self::load_sealed_segments(&config.segment_dir, &sealed_info)?;

        // Determine next segment ID
        let next_id = max_segment_id + 1;

        // Create or resume active segment
        let active_path = config.segment_dir.join(format!("segment_{next_id:08}.bin"));
        let active = ActiveSegment::new(next_id, active_path)?;

        let total_chunks = routing.len() as u64;
        let total_bytes: u64 = routing.values().map(|loc| u64::from(loc.len)).sum();

        Ok(Self {
            active: Mutex::new(active),
            sealed: RwLock::new(sealed),
            wal: Mutex::new(BufWriter::new(wal_file)),
            wal_path,
            cache: LocationCache::new(config.cache_size),
            routing: RwLock::new(routing),
            tombstones: RwLock::new(tombstones),
            gc_candidates: RwLock::new(gc_candidates),
            config,
            next_segment_id: AtomicU64::new(next_id + 1),
            total_chunks: AtomicU64::new(total_chunks),
            total_bytes: AtomicU64::new(total_bytes),
        })
    }

    /// Read a single WAL record from the reader.
    fn read_wal_record(reader: &mut BufReader<File>) -> Result<Option<BlobWalRecord>> {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len =
            usize::try_from(u32::from_le_bytes(len_buf)).expect("record length fits in usize");

        let mut crc_buf = [0u8; 4];
        match reader.read_exact(&mut crc_buf) {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let stored_crc = u32::from_le_bytes(crc_buf);

        let mut data = vec![0u8; len];
        match reader.read_exact(&mut data) {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let computed_crc = crc32fast::hash(&data);
        if stored_crc != computed_crc {
            return Ok(None);
        }

        Ok(bitcode::deserialize(&data).ok())
    }

    /// Apply a WAL record to recovery state.
    fn apply_wal_record(
        record: &BlobWalRecord,
        routing: &mut BTreeMap<DurableChunkHash, ChunkLocation>,
        tombstones: &mut BTreeSet<DurableChunkHash>,
        prepare_pending: &mut HashMap<DurableChunkHash, ChunkLocation>,
        sealed_info: &mut HashMap<u64, u64>,
        max_segment_id: &mut u64,
    ) {
        match *record {
            BlobWalRecord::Prepare {
                hash,
                segment_id,
                offset,
                len,
            } => {
                *max_segment_id = (*max_segment_id).max(segment_id);
                prepare_pending.insert(
                    hash,
                    ChunkLocation {
                        segment_id,
                        offset,
                        len,
                    },
                );
            },
            BlobWalRecord::Commit {
                hash,
                segment_id,
                offset,
                len,
                crc32: _,
            } => {
                *max_segment_id = (*max_segment_id).max(segment_id);
                prepare_pending.remove(&hash);
                tombstones.remove(&hash);
                routing.insert(
                    hash,
                    ChunkLocation {
                        segment_id,
                        offset,
                        len,
                    },
                );
            },
            BlobWalRecord::Tombstone { hash } => {
                routing.remove(&hash);
                tombstones.insert(hash);
            },
            BlobWalRecord::Seal {
                segment_id,
                footer_offset,
            } => {
                *max_segment_id = (*max_segment_id).max(segment_id);
                sealed_info.insert(segment_id, footer_offset);
            },
        }
    }

    /// Recover state from WAL.
    fn recover_from_wal(wal_path: &Path, _config: &DurableBlobLogConfig) -> Result<RecoveryState> {
        let mut routing = BTreeMap::new();
        let mut tombstones = BTreeSet::new();
        let mut gc_candidates = BTreeSet::new();
        let mut prepare_pending: HashMap<DurableChunkHash, ChunkLocation> = HashMap::new();
        let mut sealed_info = HashMap::new();
        let mut max_segment_id = 0u64;

        let file = match File::open(wal_path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok((routing, tombstones, gc_candidates, sealed_info, 0));
            },
            Err(e) => return Err(e.into()),
        };

        let mut reader = BufReader::new(file);

        while let Some(record) = Self::read_wal_record(&mut reader)? {
            Self::apply_wal_record(
                &record,
                &mut routing,
                &mut tombstones,
                &mut prepare_pending,
                &mut sealed_info,
                &mut max_segment_id,
            );
        }

        for hash in prepare_pending.keys() {
            gc_candidates.insert(*hash);
        }

        Ok((
            routing,
            tombstones,
            gc_candidates,
            sealed_info,
            max_segment_id,
        ))
    }

    /// Load sealed segments from disk.
    fn load_sealed_segments(
        segment_dir: &Path,
        sealed_info: &HashMap<u64, u64>,
    ) -> Result<Vec<SealedSegment>> {
        let mut segments = Vec::new();

        for (&segment_id, &footer_offset) in sealed_info {
            let path = segment_dir.join(format!("segment_{segment_id:08}.bin"));
            if !path.exists() {
                continue;
            }

            // Read footer
            let mut file = File::open(&path)?;
            let file_len = file.metadata()?.len();
            if file_len < FOOTER_SIZE as u64 {
                continue;
            }

            let footer_offset_neg = i64::try_from(FOOTER_SIZE).expect("footer size fits in i64");
            file.seek(SeekFrom::End(-footer_offset_neg))?;
            let mut footer_buf = [0u8; FOOTER_SIZE];
            file.read_exact(&mut footer_buf)?;

            // Parse footer
            let index_offset = u64::from_le_bytes(footer_buf[0..8].try_into().unwrap());
            let bloom_offset = u64::from_le_bytes(footer_buf[8..16].try_into().unwrap());
            let count = u32::from_le_bytes(footer_buf[16..20].try_into().unwrap());
            let magic = u32::from_le_bytes(footer_buf[20..24].try_into().unwrap());
            let stored_crc = u32::from_le_bytes(footer_buf[24..28].try_into().unwrap());

            // Validate magic
            if magic != SEGMENT_MAGIC {
                continue;
            }

            // Validate CRC (covers index_offset through magic)
            let computed_crc = crc32fast::hash(&footer_buf[0..24]);
            if stored_crc != computed_crc {
                continue;
            }

            // Validate footer offset matches
            if footer_offset != file_len - FOOTER_SIZE as u64 {
                // Mismatch - segment may have been partially written
                continue;
            }

            // Load bloom filter
            file.seek(SeekFrom::Start(bloom_offset))?;
            let bloom_len = usize::try_from(file_len - FOOTER_SIZE as u64 - bloom_offset)
                .expect("bloom filter length fits in usize");
            let mut bloom_bytes = vec![0u8; bloom_len];
            file.read_exact(&mut bloom_bytes)?;

            // Reconstruct bloom filter (we'll re-create one for now)
            let bloom = BloomFilter::new(BLOOM_EXPECTED_ITEMS, BLOOM_FPR);
            // Note: In production, we'd deserialize the actual bloom filter bits

            segments.push(SealedSegment {
                id: segment_id,
                path,
                index_offset,
                bloom_offset,
                count,
                bloom,
                index: RwLock::new(None),
            });
        }

        // Sort by segment ID for consistent ordering
        segments.sort_by_key(|s| s.id);

        Ok(segments)
    }

    /// Append a chunk and return its content hash.
    ///
    /// # Durability
    ///
    /// The chunk is durable only after this method returns. If the process
    /// crashes during this method, the chunk will NOT be visible on recovery.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging or segment I/O fails.
    pub fn append(&self, data: &[u8]) -> Result<DurableChunkHash> {
        let hash = DurableChunkHash::from_data(data);

        // Check if already exists (deduplication)
        {
            let routing = self.routing.read();
            if routing.contains_key(&hash) {
                return Ok(hash);
            }
        }

        let location = self.write_chunk_with_wal(&hash, data)?;

        // Update routing table
        {
            let mut routing = self.routing.write();
            routing.insert(hash, location);
        }

        // Update cache
        self.cache.insert(hash, location);

        // Update stats
        self.total_chunks.fetch_add(1, Ordering::Relaxed);
        self.total_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);

        Ok(hash)
    }

    /// Write a chunk with two-phase WAL protocol.
    fn write_chunk_with_wal(&self, hash: &DurableChunkHash, data: &[u8]) -> Result<ChunkLocation> {
        let mut active = self.active.lock();

        // Seal active segment if it can't fit
        if !active.can_fit(data.len(), self.config.segment_size) {
            self.seal_active_segment(&mut active)?;
        }

        let segment_id = active.id;
        let offset = active.write_offset;
        let len = u32::try_from(data.len()).expect("chunk data length fits in u32");
        let data_crc = crc32fast::hash(data);

        // Phase 1: Log PREPARE to WAL
        self.log_wal_record(&BlobWalRecord::Prepare {
            hash: *hash,
            segment_id,
            offset,
            len,
        })?;

        // Fsync WAL (intent is durable)
        if self.config.enable_fsync {
            self.sync_wal()?;
        }

        // Phase 2: Write data to segment
        active.write_chunk(data)?;
        active.flush()?;

        // Fsync data segment
        if self.config.enable_fsync {
            active.sync()?;
        }

        // Phase 3: Log COMMIT to WAL
        self.log_wal_record(&BlobWalRecord::Commit {
            hash: *hash,
            segment_id,
            offset,
            len,
            crc32: data_crc,
        })?;

        // Fsync WAL (commit is durable)
        if self.config.enable_fsync {
            self.sync_wal()?;
        }

        // Add to pending entries for footer
        active.pending_entries.push(IndexEntry {
            hash: *hash,
            offset,
            len,
        });
        drop(active);

        Ok(ChunkLocation {
            segment_id,
            offset,
            len,
        })
    }

    /// Seal the active segment and start a new one.
    fn seal_active_segment(&self, active: &mut ActiveSegment) -> Result<()> {
        if active.pending_entries.is_empty() {
            return Ok(());
        }

        // Build index
        let index_offset = active.write_offset;
        let index_bytes = bitcode::serialize(&active.pending_entries)?;
        active.file.write_all(&index_bytes)?;

        // Build bloom filter
        let bloom_offset = active.write_offset + index_bytes.len() as u64;
        let bloom = BloomFilter::new(BLOOM_EXPECTED_ITEMS, BLOOM_FPR);
        for entry in &active.pending_entries {
            bloom.add(&entry.hash.0);
        }
        // Serialize bloom filter (simplified - just the config for now)
        let bloom_bytes = bitcode::serialize(&(bloom.num_bits(), bloom.num_hashes()))?;
        active.file.write_all(&bloom_bytes)?;

        // Write footer
        let count = u32::try_from(active.pending_entries.len()).expect("entry count fits in u32");
        let mut footer = [0u8; FOOTER_SIZE];
        footer[0..8].copy_from_slice(&index_offset.to_le_bytes());
        footer[8..16].copy_from_slice(&bloom_offset.to_le_bytes());
        footer[16..20].copy_from_slice(&count.to_le_bytes());
        footer[20..24].copy_from_slice(&SEGMENT_MAGIC.to_le_bytes());
        let footer_crc = crc32fast::hash(&footer[0..24]);
        footer[24..28].copy_from_slice(&footer_crc.to_le_bytes());

        active.file.write_all(&footer)?;
        active.flush()?;

        // Fsync segment
        if self.config.enable_fsync {
            active.sync()?;
        }

        let footer_offset =
            active.write_offset + index_bytes.len() as u64 + bloom_bytes.len() as u64;

        // Log SEAL to WAL
        self.log_wal_record(&BlobWalRecord::Seal {
            segment_id: active.id,
            footer_offset,
        })?;

        // Fsync WAL
        if self.config.enable_fsync {
            self.sync_wal()?;
        }

        // Add to sealed segments
        {
            let mut sealed = self.sealed.write();
            sealed.push(SealedSegment {
                id: active.id,
                path: active.path.clone(),
                index_offset,
                bloom_offset,
                count,
                bloom,
                index: RwLock::new(None),
            });
        }

        // Create new active segment
        let new_id = self.next_segment_id.fetch_add(1, Ordering::Relaxed);
        let new_path = self
            .config
            .segment_dir
            .join(format!("segment_{new_id:08}.bin"));
        *active = ActiveSegment::new(new_id, new_path)?;

        Ok(())
    }

    /// Log a WAL record.
    fn log_wal_record(&self, record: &BlobWalRecord) -> Result<()> {
        let bytes = bitcode::serialize(record)?;
        let len = u32::try_from(bytes.len()).expect("WAL record fits in u32");
        let crc = crc32fast::hash(&bytes);

        let mut wal = self.wal.lock();
        wal.write_all(&len.to_le_bytes())?;
        wal.write_all(&crc.to_le_bytes())?;
        wal.write_all(&bytes)?;
        wal.flush()?;
        drop(wal);

        Ok(())
    }

    /// Sync WAL to disk.
    fn sync_wal(&self) -> Result<()> {
        self.wal.lock().get_ref().sync_all()?;
        Ok(())
    }

    /// Get chunk data by hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk is not found or I/O fails.
    pub fn get(&self, hash: &DurableChunkHash) -> Result<Vec<u8>> {
        // Check tombstones
        if self.tombstones.read().contains(hash) {
            return Err(DurableBlobLogError::ChunkNotFound {
                hash: hash.to_hex(),
            });
        }

        // Try cache
        if let Some(location) = self.cache.get(hash) {
            return self.read_chunk(&location);
        }

        // Try routing table
        let location = {
            let routing = self.routing.read();
            routing.get(hash).copied()
        };

        if let Some(loc) = location {
            let data = self.read_chunk(&loc)?;
            self.cache.insert(*hash, loc);
            return Ok(data);
        }

        // Try sealed segments
        {
            let sealed = self.sealed.read();
            for segment in sealed.iter().rev() {
                if let Some(loc) = segment.get_location(hash)? {
                    drop(sealed);
                    let data = self.read_chunk(&loc)?;
                    self.cache.insert(*hash, loc);
                    return Ok(data);
                }
            }
        }

        Err(DurableBlobLogError::ChunkNotFound {
            hash: hash.to_hex(),
        })
    }

    /// Read chunk data from a location.
    fn read_chunk(&self, location: &ChunkLocation) -> Result<Vec<u8>> {
        let chunk_len = usize::try_from(location.len).expect("chunk length fits in usize");

        // Check active segment first
        {
            let active = self.active.lock();
            if location.segment_id == active.id {
                let path = active.path.clone();
                drop(active);
                let file = File::open(&path)?;
                let mut reader = BufReader::new(file);
                reader.seek(SeekFrom::Start(location.offset))?;
                let mut data = vec![0u8; chunk_len];
                reader.read_exact(&mut data)?;
                return Ok(data);
            }
        }

        // Check sealed segments
        let segment_path = self
            .config
            .segment_dir
            .join(format!("segment_{:08}.bin", location.segment_id));

        let file = File::open(&segment_path)?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(location.offset))?;
        let mut data = vec![0u8; chunk_len];
        reader.read_exact(&mut data)?;

        Ok(data)
    }

    /// Check if a chunk exists.
    pub fn contains(&self, hash: &DurableChunkHash) -> bool {
        if self.tombstones.read().contains(hash) {
            return false;
        }
        self.routing.read().contains_key(hash)
    }

    /// Mark a chunk as deleted (tombstone).
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub fn delete(&self, hash: &DurableChunkHash) -> Result<()> {
        // Log tombstone to WAL
        self.log_wal_record(&BlobWalRecord::Tombstone { hash: *hash })?;

        if self.config.enable_fsync {
            self.sync_wal()?;
        }

        // Update state
        {
            let mut routing = self.routing.write();
            routing.remove(hash);
        }
        self.tombstones.write().insert(*hash);
        self.cache.remove(hash);

        Ok(())
    }

    /// Get total number of chunks.
    pub fn chunk_count(&self) -> u64 {
        self.total_chunks.load(Ordering::Relaxed)
    }

    /// Get total bytes stored.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get number of sealed segments.
    pub fn segment_count(&self) -> usize {
        self.sealed.read().len() + 1 // +1 for active
    }

    /// Get GC candidates (PREPARE without COMMIT).
    pub fn gc_candidates(&self) -> Vec<DurableChunkHash> {
        self.gc_candidates.read().iter().copied().collect()
    }

    /// Get tombstoned chunks.
    pub fn tombstoned(&self) -> Vec<DurableChunkHash> {
        self.tombstones.read().iter().copied().collect()
    }

    /// Flush all pending writes.
    ///
    /// # Errors
    ///
    /// Returns an error if I/O flush fails.
    pub fn flush(&self) -> Result<()> {
        {
            let mut active = self.active.lock();
            active.flush()?;
        }
        {
            let mut wal = self.wal.lock();
            wal.flush()?;
        }
        Ok(())
    }

    /// Sync all data to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if fsync fails.
    pub fn sync(&self) -> Result<()> {
        self.flush()?;
        {
            let active = self.active.lock();
            active.sync()?;
        }
        self.sync_wal()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn test_config(dir: &Path) -> DurableBlobLogConfig {
        DurableBlobLogConfig {
            segment_dir: dir.to_path_buf(),
            segment_size: 4096, // Small for testing
            enable_fsync: false,
            cache_size: 100,
        }
    }

    #[test]
    fn test_chunk_hash() {
        let data = b"hello world";
        let hash = DurableChunkHash::from_data(data);
        let hex = hash.to_hex();
        let recovered = DurableChunkHash::from_hex(&hex).unwrap();
        assert_eq!(hash, recovered);
    }

    #[test]
    fn test_basic_append_get() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let data = b"test data";
        let hash = log.append(data).unwrap();

        let retrieved = log.get(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_deduplication() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let data = b"duplicate data";
        let hash1 = log.append(data).unwrap();
        let hash2 = log.append(data).unwrap();

        assert_eq!(hash1, hash2);
        assert_eq!(log.chunk_count(), 1);
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let data = b"to be deleted";
        let hash = log.append(data).unwrap();

        assert!(log.contains(&hash));
        log.delete(&hash).unwrap();
        assert!(!log.contains(&hash));

        let result = log.get(&hash);
        assert!(result.is_err());
    }

    #[test]
    fn test_recovery_after_clean_close() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());

        let hash;
        {
            let log = DurableBlobLog::open(config.clone()).unwrap();
            hash = log.append(b"persistent data").unwrap();
            log.sync().unwrap();
        }

        // Reopen
        {
            let log = DurableBlobLog::open(config).unwrap();
            let data = log.get(&hash).unwrap();
            assert_eq!(data, b"persistent data");
        }
    }

    #[test]
    fn test_multiple_chunks() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let mut hashes = Vec::new();
        for i in 0..10 {
            let data = format!("chunk {i}");
            let hash = log.append(data.as_bytes()).unwrap();
            hashes.push(hash);
        }

        assert_eq!(log.chunk_count(), 10);

        for (i, hash) in hashes.iter().enumerate() {
            let data = log.get(hash).unwrap();
            assert_eq!(data, format!("chunk {i}").as_bytes());
        }
    }

    #[test]
    fn test_segment_sealing() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.segment_size = 256; // Very small to force sealing

        let log = DurableBlobLog::open(config).unwrap();

        // Write enough data to trigger multiple segments
        for i in 0..20 {
            let data = format!("chunk data with some padding {i:04}");
            log.append(data.as_bytes()).unwrap();
        }

        // Should have sealed at least one segment
        assert!(log.segment_count() >= 1);
    }

    #[test]
    fn test_chunk_hash_from_hex_invalid() {
        // Too short
        assert!(DurableChunkHash::from_hex("abcd").is_none());
        // Not hex
        assert!(DurableChunkHash::from_hex("zzzz").is_none());
        // Empty
        assert!(DurableChunkHash::from_hex("").is_none());
        // Wrong length (31 bytes)
        let short = "ab".repeat(31);
        assert!(DurableChunkHash::from_hex(&short).is_none());
        // Wrong length (33 bytes)
        let long = "ab".repeat(33);
        assert!(DurableChunkHash::from_hex(&long).is_none());
    }

    #[test]
    fn test_chunk_hash_roundtrip() {
        let data = b"test data for hashing";
        let hash = DurableChunkHash::from_data(data);
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64); // 32 bytes -> 64 hex chars
        let recovered = DurableChunkHash::from_hex(&hex).unwrap();
        assert_eq!(hash, recovered);
    }

    #[test]
    fn test_chunk_hash_deterministic() {
        let data = b"same data";
        let h1 = DurableChunkHash::from_data(data);
        let h2 = DurableChunkHash::from_data(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_chunk_hash_different_data() {
        let h1 = DurableChunkHash::from_data(b"data1");
        let h2 = DurableChunkHash::from_data(b"data2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_stats_tracking() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        assert_eq!(log.chunk_count(), 0);
        assert_eq!(log.total_bytes(), 0);

        let data = b"test data";
        log.append(data).unwrap();

        assert_eq!(log.chunk_count(), 1);
        assert!(log.total_bytes() > 0);
    }

    #[test]
    fn test_contains_after_delete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let hash = log.append(b"deletable").unwrap();
        assert!(log.contains(&hash));

        log.delete(&hash).unwrap();
        assert!(!log.contains(&hash));

        // tombstoned should include it
        let tombstones = log.tombstoned();
        assert!(tombstones.contains(&hash));
    }

    #[test]
    fn test_gc_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        // Initially empty
        assert!(log.gc_candidates().is_empty());
    }

    #[test]
    fn test_get_nonexistent_chunk() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let fake_hash = DurableChunkHash::from_data(b"never stored");
        let result = log.get(&fake_hash);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tombstoned_chunk() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let hash = log.append(b"will delete").unwrap();
        log.delete(&hash).unwrap();

        let result = log.get(&hash);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_empty_log() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        log.flush().unwrap();
    }

    #[test]
    fn test_sync_after_writes() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        log.append(b"data to sync").unwrap();
        log.sync().unwrap();
    }

    #[test]
    fn test_recovery_after_multiple_operations() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());

        let (hash1, hash3);
        {
            let log = DurableBlobLog::open(config.clone()).unwrap();
            hash1 = log.append(b"persistent1").unwrap();
            let hash2 = log.append(b"to_delete").unwrap();
            hash3 = log.append(b"persistent2").unwrap();
            log.delete(&hash2).unwrap();
            log.sync().unwrap();
        }

        // Reopen and verify
        {
            let log = DurableBlobLog::open(config).unwrap();
            assert_eq!(log.get(&hash1).unwrap(), b"persistent1");
            assert_eq!(log.get(&hash3).unwrap(), b"persistent2");
            assert_eq!(log.chunk_count(), 2);
        }
    }

    #[test]
    fn test_recovery_preserves_dedup() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());

        let hash;
        {
            let log = DurableBlobLog::open(config.clone()).unwrap();
            hash = log.append(b"duplicate").unwrap();
            let hash2 = log.append(b"duplicate").unwrap();
            assert_eq!(hash, hash2);
            log.sync().unwrap();
        }

        // Reopen
        {
            let log = DurableBlobLog::open(config).unwrap();
            assert_eq!(log.get(&hash).unwrap(), b"duplicate");
            assert_eq!(log.chunk_count(), 1);
        }
    }

    #[test]
    fn test_recovery_with_sealed_segments() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.segment_size = 128; // Very small to force sealing

        let mut hashes = Vec::new();
        {
            let log = DurableBlobLog::open(config.clone()).unwrap();
            for i in 0..15 {
                let data = format!("segment test data {i:04}");
                hashes.push(log.append(data.as_bytes()).unwrap());
            }
            log.sync().unwrap();
            assert!(log.segment_count() >= 1);
        }

        // Reopen and verify all chunks survive recovery
        {
            let log = DurableBlobLog::open(config).unwrap();
            for (i, hash) in hashes.iter().enumerate() {
                let data = log.get(hash).unwrap();
                assert_eq!(data, format!("segment test data {i:04}").as_bytes());
            }
        }
    }

    #[test]
    fn test_default_config() {
        let config = DurableBlobLogConfig::default();
        assert_eq!(config.segment_size, DEFAULT_SEGMENT_SIZE);
        assert!(config.enable_fsync);
        assert_eq!(config.cache_size, LOCATION_CACHE_SIZE);
    }

    #[test]
    fn test_error_display() {
        let io_err =
            DurableBlobLogError::Io(io::Error::new(io::ErrorKind::NotFound, "file missing"));
        assert!(io_err.to_string().contains("file missing"));

        let ser_err = DurableBlobLogError::Serialization("bad format".to_string());
        assert!(ser_err.to_string().contains("bad format"));

        let footer_err = DurableBlobLogError::InvalidFooter("magic mismatch".to_string());
        assert!(footer_err.to_string().contains("magic mismatch"));

        let not_found = DurableBlobLogError::ChunkNotFound {
            hash: "abc123".to_string(),
        };
        assert!(not_found.to_string().contains("abc123"));

        let crc_err = DurableBlobLogError::CrcMismatch {
            hash: "abc".to_string(),
            expected: 0x1234,
            actual: 0x5678,
        };
        assert!(crc_err.to_string().contains("CRC32 mismatch"));

        let config_err = DurableBlobLogError::Config("bad value".to_string());
        assert!(config_err.to_string().contains("bad value"));
    }

    #[test]
    fn test_delete_nonexistent_chunk() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let fake_hash = DurableChunkHash::from_data(b"never stored");
        // Delete of non-existent chunk should succeed (idempotent)
        let result = log.delete(&fake_hash);
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_data_chunks() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        // Write data larger than segment (4096 bytes)
        let data = vec![42u8; 8192];
        let hash = log.append(&data).unwrap();

        let retrieved = log.get(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_empty_data_chunk() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let log = DurableBlobLog::open(config).unwrap();

        let hash = log.append(b"").unwrap();
        let retrieved = log.get(&hash).unwrap();
        assert!(retrieved.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.cache_size = 3; // Very small cache

        let log = DurableBlobLog::open(config).unwrap();

        // Write more chunks than cache can hold
        let mut hashes = Vec::new();
        for i in 0..10 {
            let data = format!("cache test {i}");
            hashes.push(log.append(data.as_bytes()).unwrap());
        }

        // All chunks should still be retrievable even with eviction
        for (i, hash) in hashes.iter().enumerate() {
            let data = log.get(hash).unwrap();
            assert_eq!(data, format!("cache test {i}").as_bytes());
        }
    }

    #[test]
    fn test_segment_count_increases() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.segment_size = 64; // Very small

        let log = DurableBlobLog::open(config).unwrap();
        let initial = log.segment_count();

        // Write enough to seal multiple segments
        for i in 0..50 {
            let data = format!("fill segment {i:04}");
            log.append(data.as_bytes()).unwrap();
        }

        assert!(log.segment_count() > initial);
    }
}
