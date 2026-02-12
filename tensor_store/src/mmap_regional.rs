// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Region-aware memory-mapped storage for geometric locality.
//!
//! `RegionalMmapStore` extends the base mmap storage with Voronoi-region-aware
//! organization. Data is stored in sorted runs keyed by `LocalityKey`, which
//! groups geometrically similar vectors together on disk.
//!
//! # Design
//!
//! - **Sorted Runs**: Append-only runs sorted by `(region_id, sequence)`
//! - **Region-Aware Reads**: Load entire regions for efficient k-NN queries
//! - **Merge Without Rewrite**: Monotonic sequences enable append-only merging
//! - **LSM-Tree Style**: Multiple runs compacted into larger sorted runs
//!
//! # File Format
//!
//! Each sorted run is a separate file:
//! ```text
//! [Header: 32 bytes]
//!   Magic: 4 bytes "RMAP"
//!   Version: 4 bytes (little-endian u32)
//!   Entry count: 8 bytes (little-endian u64)
//!   Min region: 4 bytes (little-endian u32)
//!   Max region: 4 bytes (little-endian u32)
//!   Reserved: 8 bytes
//!
//! [Entries: sorted by LocalityKey]
//!   LocalityKey: 8 bytes (little-endian u64)
//!   Key length: 4 bytes (little-endian u32)
//!   Key: variable bytes (UTF-8)
//!   Data length: 4 bytes (little-endian u32)
//!   Data: variable bytes (Zstd-compressed bitcode)
//!
//! [Region Index: at end]
//!   Index entry count: 4 bytes (little-endian u32)
//!   [Index entries: (region_id: u32, start_offset: u64, entry_count: u32)*]
//!   Index offset: 8 bytes (little-endian u64, points to start of index)
//! ```

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use memmap2::Mmap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{
    voronoi::{LocalityKey, LocalityKeyGenerator, VoronoiPartitioner},
    TensorData,
};

const MAGIC: &[u8; 4] = b"RMAP";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;
const ZSTD_LEVEL: i32 = 3;

/// Errors from regional mmap operations.
#[derive(Debug)]
pub enum RegionalMmapError {
    /// I/O error.
    Io(io::Error),
    /// Invalid file magic.
    InvalidMagic,
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// Serialization error.
    Serialization(String),
    /// Key not found.
    NotFound(String),
    /// Region not found.
    RegionNotFound(u32),
    /// Empty file.
    EmptyFile,
    /// No regions computed.
    NoRegions,
}

impl std::fmt::Display for RegionalMmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic => write!(f, "Invalid file magic"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {v}"),
            Self::Serialization(s) => write!(f, "Serialization error: {s}"),
            Self::NotFound(k) => write!(f, "Key not found: {k}"),
            Self::RegionNotFound(r) => write!(f, "Region not found: {r}"),
            Self::EmptyFile => write!(f, "Empty file"),
            Self::NoRegions => write!(f, "No Voronoi regions computed"),
        }
    }
}

impl std::error::Error for RegionalMmapError {}

impl From<io::Error> for RegionalMmapError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bitcode::Error> for RegionalMmapError {
    fn from(e: bitcode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// Result type for regional mmap operations.
pub type Result<T> = std::result::Result<T, RegionalMmapError>;

/// Index entry for fast region lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegionIndexEntry {
    region_id: u32,
    start_offset: u64,
    entry_count: u32,
}

/// Entry location within a sorted run.
#[derive(Debug, Clone, Copy)]
struct EntryLocation {
    locality_key: LocalityKey,
    #[allow(dead_code)] // Reserved for direct key access
    key_offset: usize,
    #[allow(dead_code)] // Reserved for direct key access
    key_len: usize,
    data_offset: usize,
    data_len: usize,
}

/// Metadata for a sorted run file.
#[derive(Debug)]
pub struct SortedRun {
    /// Unique run ID.
    pub id: u64,
    /// Path to the run file.
    pub path: PathBuf,
    /// Number of entries in this run.
    pub entry_count: u64,
    /// Minimum region ID in this run.
    pub min_region: u32,
    /// Maximum region ID in this run.
    pub max_region: u32,
    /// Memory-mapped file (lazy-loaded).
    mmap: RwLock<Option<Mmap>>,
    /// In-memory key index (lazy-loaded).
    key_index: RwLock<Option<HashMap<String, EntryLocation>>>,
    /// Region index for fast lookups.
    region_index: RwLock<Option<Vec<RegionIndexEntry>>>,
}

impl SortedRun {
    /// Create metadata for an existing run file.
    fn from_path(id: u64, path: PathBuf) -> Result<Self> {
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        if metadata.len() < HEADER_SIZE as u64 {
            return Err(RegionalMmapError::EmptyFile);
        }

        // Read header
        // SAFETY: File opened read-only, mapping is safe
        let mmap = unsafe { Mmap::map(&file)? };

        if &mmap[0..4] != MAGIC {
            return Err(RegionalMmapError::InvalidMagic);
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(RegionalMmapError::UnsupportedVersion(version));
        }

        let entry_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        let min_region = u32::from_le_bytes(mmap[16..20].try_into().unwrap());
        let max_region = u32::from_le_bytes(mmap[20..24].try_into().unwrap());

        Ok(Self {
            id,
            path,
            entry_count,
            min_region,
            max_region,
            mmap: RwLock::new(Some(mmap)),
            key_index: RwLock::new(None),
            region_index: RwLock::new(None),
        })
    }

    /// Check if this run might contain the given region.
    const fn might_contain_region(&self, region_id: u32) -> bool {
        region_id >= self.min_region && region_id <= self.max_region
    }

    /// Load the key index if not already loaded.
    fn ensure_index_loaded(&self) -> Result<()> {
        if self.key_index.read().is_some() {
            return Ok(());
        }

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard.as_ref().ok_or(RegionalMmapError::EmptyFile)?;

        // Read region index from end of file
        let file_len = mmap.len();
        if file_len < 8 {
            return Err(RegionalMmapError::EmptyFile);
        }

        let index_offset_u64 = u64::from_le_bytes(mmap[file_len - 8..file_len].try_into().unwrap());
        let index_offset = usize::try_from(index_offset_u64)
            .map_err(|_| RegionalMmapError::Serialization("index offset too large".to_string()))?;

        if index_offset >= file_len - 8 {
            return Err(RegionalMmapError::Serialization(
                "Invalid index offset".to_string(),
            ));
        }

        let index_entry_count =
            u32::from_le_bytes(mmap[index_offset..index_offset + 4].try_into().unwrap()) as usize;

        let mut region_index = Vec::with_capacity(index_entry_count);
        let mut offset = index_offset + 4;
        for _ in 0..index_entry_count {
            let region_id = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
            let start_offset =
                u64::from_le_bytes(mmap[offset + 4..offset + 12].try_into().unwrap());
            let entry_count =
                u32::from_le_bytes(mmap[offset + 12..offset + 16].try_into().unwrap());
            region_index.push(RegionIndexEntry {
                region_id,
                start_offset,
                entry_count,
            });
            offset += 16;
        }

        // Build key index by scanning entries
        let mut key_index = HashMap::new();
        let mut scan_offset = HEADER_SIZE;

        while scan_offset < index_offset {
            if scan_offset + 16 > index_offset {
                break;
            }

            let locality_key = LocalityKey::from_u64(u64::from_le_bytes(
                mmap[scan_offset..scan_offset + 8].try_into().unwrap(),
            ));
            scan_offset += 8;

            let key_len =
                u32::from_le_bytes(mmap[scan_offset..scan_offset + 4].try_into().unwrap()) as usize;
            scan_offset += 4;

            let key_offset = scan_offset;
            let key =
                String::from_utf8_lossy(&mmap[scan_offset..scan_offset + key_len]).to_string();
            scan_offset += key_len;

            let data_len =
                u32::from_le_bytes(mmap[scan_offset..scan_offset + 4].try_into().unwrap()) as usize;
            scan_offset += 4;

            let data_offset = scan_offset;
            scan_offset += data_len;

            key_index.insert(
                key,
                EntryLocation {
                    locality_key,
                    key_offset,
                    key_len,
                    data_offset,
                    data_len,
                },
            );
        }

        drop(mmap_guard);
        *self.key_index.write() = Some(key_index);
        *self.region_index.write() = Some(region_index);

        Ok(())
    }

    /// Get a single entry by key.
    fn get(&self, key: &str) -> Result<TensorData> {
        self.ensure_index_loaded()?;

        let loc = {
            let key_index = self.key_index.read();
            key_index
                .as_ref()
                .and_then(|idx| idx.get(key).copied())
                .ok_or_else(|| RegionalMmapError::NotFound(key.to_string()))?
        };

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard.as_ref().ok_or(RegionalMmapError::EmptyFile)?;

        let compressed = &mmap[loc.data_offset..loc.data_offset + loc.data_len];
        let decompressed = zstd::decode_all(compressed)
            .map_err(|e| RegionalMmapError::Serialization(format!("Zstd: {e}")))?;
        drop(mmap_guard);

        Ok(bitcode::deserialize(&decompressed)?)
    }

    /// Get all entries for a specific region.
    fn get_region(&self, region_id: u32) -> Result<Vec<(String, TensorData)>> {
        if !self.might_contain_region(region_id) {
            return Ok(Vec::new());
        }

        self.ensure_index_loaded()?;

        // Find region entry - extract values before dropping lock
        let region_index_guard = self.region_index.read();
        let region_entry = region_index_guard
            .as_ref()
            .and_then(|idx| idx.iter().find(|e| e.region_id == region_id));
        let Some(entry) = region_entry else {
            drop(region_index_guard);
            return Ok(Vec::new());
        };
        let start_offset = entry.start_offset;
        let entry_count = entry.entry_count;
        drop(region_index_guard);

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard.as_ref().ok_or(RegionalMmapError::EmptyFile)?;

        let entry_count_usize = entry_count as usize;
        let mut results = Vec::with_capacity(entry_count_usize);
        let mut offset = usize::try_from(start_offset)
            .map_err(|_| RegionalMmapError::Serialization("offset too large".to_string()))?;

        for _ in 0..entry_count {
            // Read locality key
            let lk = LocalityKey::from_u64(u64::from_le_bytes(
                mmap[offset..offset + 8].try_into().unwrap(),
            ));

            // Verify region matches (entries are sorted by locality key)
            if lk.region_id() != region_id {
                break;
            }
            offset += 8;

            // Read key
            let key_len = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let key = String::from_utf8_lossy(&mmap[offset..offset + key_len]).to_string();
            offset += key_len;

            // Read data
            let data_len =
                u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let compressed = &mmap[offset..offset + data_len];
            offset += data_len;

            let decompressed = zstd::decode_all(compressed)
                .map_err(|e| RegionalMmapError::Serialization(format!("Zstd: {e}")))?;
            let tensor: TensorData = bitcode::deserialize(&decompressed)?;

            results.push((key, tensor));
        }

        drop(mmap_guard);
        Ok(results)
    }

    /// Check if a key exists.
    fn contains(&self, key: &str) -> bool {
        if self.ensure_index_loaded().is_err() {
            return false;
        }
        self.key_index
            .read()
            .as_ref()
            .is_some_and(|idx| idx.contains_key(key))
    }
}

/// Builder for creating sorted run files.
pub struct SortedRunBuilder {
    file: BufWriter<File>,
    path: PathBuf,
    entries: Vec<(LocalityKey, String, Vec<u8>)>,
    entry_count: u64,
}

impl SortedRunBuilder {
    /// Create a new sorted run file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            file: BufWriter::new(file),
            path,
            entries: Vec::new(),
            entry_count: 0,
        })
    }

    /// Add an entry to the run.
    ///
    /// Entries are buffered and sorted before writing.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn add(&mut self, locality_key: LocalityKey, key: &str, tensor: &TensorData) -> Result<()> {
        let serialized = bitcode::serialize(tensor)?;
        let compressed = zstd::encode_all(&serialized[..], ZSTD_LEVEL)
            .map_err(|e| RegionalMmapError::Serialization(format!("Zstd: {e}")))?;

        self.entries
            .push((locality_key, key.to_string(), compressed));
        self.entry_count += 1;

        Ok(())
    }

    /// Finalize the run file.
    ///
    /// Sorts entries by locality key and writes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    ///
    /// # Panics
    ///
    /// Panics if a key or data length exceeds `u32::MAX`.
    pub fn finish(mut self) -> Result<SortedRun> {
        // Sort by locality key
        self.entries.sort_by_key(|(lk, _, _)| *lk);

        // Compute min/max region
        let min_region = self.entries.first().map_or(0, |(lk, _, _)| lk.region_id());
        let max_region = self.entries.last().map_or(0, |(lk, _, _)| lk.region_id());

        // Write header
        self.file.write_all(MAGIC)?;
        self.file.write_all(&VERSION.to_le_bytes())?;
        self.file.write_all(&self.entry_count.to_le_bytes())?;
        self.file.write_all(&min_region.to_le_bytes())?;
        self.file.write_all(&max_region.to_le_bytes())?;
        self.file.write_all(&[0u8; 8])?; // Reserved

        // Build region index while writing entries
        let mut region_index: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        let mut current_offset = HEADER_SIZE as u64;

        for (locality_key, key, data) in &self.entries {
            let region_id = locality_key.region_id();

            // Track region start offset and count
            region_index
                .entry(region_id)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((current_offset, 1));

            // Write entry
            let key_len = u32::try_from(key.len()).expect("key length fits in u32");
            let data_len = u32::try_from(data.len()).expect("data length fits in u32");

            self.file.write_all(&locality_key.to_u64().to_le_bytes())?;
            self.file.write_all(&key_len.to_le_bytes())?;
            self.file.write_all(key.as_bytes())?;
            self.file.write_all(&data_len.to_le_bytes())?;
            self.file.write_all(data)?;

            current_offset += 8 + 4 + u64::from(key_len) + 4 + u64::from(data_len);
        }

        // Write region index
        let index_offset = current_offset;
        let region_count = u32::try_from(region_index.len()).expect("region count fits in u32");
        self.file.write_all(&region_count.to_le_bytes())?;

        for (region_id, (start_offset, entry_count)) in &region_index {
            self.file.write_all(&region_id.to_le_bytes())?;
            self.file.write_all(&start_offset.to_le_bytes())?;
            self.file.write_all(&entry_count.to_le_bytes())?;
        }

        // Write index offset at end
        self.file.write_all(&index_offset.to_le_bytes())?;
        self.file.flush()?;

        // Drop the file handle to ensure data is flushed
        drop(self.file);

        // Load the run from disk to get proper mmap
        SortedRun::from_path(0, self.path)
    }
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Number of runs compacted.
    pub runs_compacted: usize,
    /// Total entries before compaction.
    pub entries_before: u64,
    /// Entries after compaction (after dedup).
    pub entries_after: u64,
    /// Bytes saved.
    pub bytes_saved: u64,
}

/// Configuration for regional mmap store.
#[derive(Debug, Clone)]
pub struct RegionalMmapConfig {
    /// Directory for run files.
    pub run_dir: PathBuf,
    /// Maximum entries per run before flush.
    pub max_entries_per_run: usize,
    /// Number of runs that trigger compaction.
    pub compaction_threshold: usize,
}

impl Default for RegionalMmapConfig {
    fn default() -> Self {
        Self {
            run_dir: std::env::temp_dir().join("regional_mmap"),
            max_entries_per_run: 100_000,
            compaction_threshold: 4,
        }
    }
}

/// Region-aware memory-mapped store.
///
/// Organizes data by Voronoi regions for efficient k-NN queries.
/// Uses an LSM-tree style architecture with sorted runs.
#[derive(Debug)]
pub struct RegionalMmapStore {
    /// Voronoi partitioner for region assignment.
    partitioner: VoronoiPartitioner,
    /// Locality key generator (for manual key creation).
    #[allow(dead_code)] // Available via partitioner.locality_generator()
    key_generator: LocalityKeyGenerator,
    /// Sealed sorted runs (read-only).
    runs: RwLock<Vec<SortedRun>>,
    /// In-memory buffer for pending writes.
    buffer: RwLock<Vec<(LocalityKey, String, TensorData)>>,
    /// Configuration.
    config: RegionalMmapConfig,
    /// Next run ID.
    next_run_id: AtomicU64,
}

impl RegionalMmapStore {
    /// Create a new regional mmap store.
    ///
    /// # Errors
    ///
    /// Returns an error if the run directory cannot be created.
    pub fn new(partitioner: VoronoiPartitioner, config: RegionalMmapConfig) -> Result<Self> {
        fs::create_dir_all(&config.run_dir)?;

        let key_generator = LocalityKeyGenerator::new();

        Ok(Self {
            partitioner,
            key_generator,
            runs: RwLock::new(Vec::new()),
            buffer: RwLock::new(Vec::new()),
            config,
            next_run_id: AtomicU64::new(0),
        })
    }

    /// Open an existing regional mmap store.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory doesn't exist or run files are invalid.
    pub fn open(partitioner: VoronoiPartitioner, config: RegionalMmapConfig) -> Result<Self> {
        let mut runs = Vec::new();
        let mut max_run_id = 0u64;

        // Scan for existing run files
        if config.run_dir.exists() {
            for entry in fs::read_dir(&config.run_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "rmap") {
                    // Parse run ID from filename (run_NNNNNNNN.rmap)
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Some(id_str) = stem.strip_prefix("run_") {
                            if let Ok(id) = id_str.parse::<u64>() {
                                max_run_id = max_run_id.max(id);
                                if let Ok(run) = SortedRun::from_path(id, path) {
                                    runs.push(run);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            fs::create_dir_all(&config.run_dir)?;
        }

        // Sort runs by ID
        runs.sort_by_key(|r| r.id);

        let key_generator = LocalityKeyGenerator::new();

        Ok(Self {
            partitioner,
            key_generator,
            runs: RwLock::new(runs),
            buffer: RwLock::new(Vec::new()),
            config,
            next_run_id: AtomicU64::new(max_run_id + 1),
        })
    }

    /// Store a tensor with its embedding.
    ///
    /// The embedding is used to determine the Voronoi region for locality.
    ///
    /// # Errors
    ///
    /// Returns an error if regions haven't been computed or buffer flush fails.
    pub fn put(&self, key: &str, tensor: &TensorData, embedding: &[f32]) -> Result<LocalityKey> {
        // Get locality key from partitioner
        let locality_key = self
            .partitioner
            .locality_key_for_embedding(embedding)
            .ok_or(RegionalMmapError::NoRegions)?;

        // Add to buffer
        {
            let mut buffer = self.buffer.write();
            buffer.push((locality_key, key.to_string(), tensor.clone()));

            // Flush if buffer is full
            if buffer.len() >= self.config.max_entries_per_run {
                let entries = std::mem::take(&mut *buffer);
                drop(buffer);
                self.flush_buffer(entries)?;
            }
        }

        Ok(locality_key)
    }

    /// Store a tensor with a pre-computed locality key.
    ///
    /// Use this when you already have the locality key from a previous operation.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the buffer to disk fails.
    pub fn put_with_key(
        &self,
        key: &str,
        tensor: &TensorData,
        locality_key: LocalityKey,
    ) -> Result<()> {
        let mut buffer = self.buffer.write();
        buffer.push((locality_key, key.to_string(), tensor.clone()));

        if buffer.len() >= self.config.max_entries_per_run {
            let entries = std::mem::take(&mut *buffer);
            drop(buffer);
            self.flush_buffer(entries)?;
        }

        Ok(())
    }

    /// Flush the in-memory buffer to a new sorted run.
    fn flush_buffer(&self, entries: Vec<(LocalityKey, String, TensorData)>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_path = self.config.run_dir.join(format!("run_{run_id:08}.rmap"));

        let mut builder = SortedRunBuilder::create(&run_path)?;
        for (locality_key, key, tensor) in entries {
            builder.add(locality_key, &key, &tensor)?;
        }

        let mut run = builder.finish()?;
        run.id = run_id;

        self.runs.write().push(run);

        // Check if compaction is needed
        if self.runs.read().len() >= self.config.compaction_threshold {
            // Compaction would be triggered here in production
            // For now, we skip automatic compaction
        }

        Ok(())
    }

    /// Get a tensor by key.
    ///
    /// Searches runs in reverse order (newest first).
    ///
    /// # Errors
    ///
    /// Returns an error if the key is not found.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        // Check buffer first
        let buffer_guard = self.buffer.read();
        for (_, k, t) in buffer_guard.iter().rev() {
            if k == key {
                let result = t.clone();
                drop(buffer_guard);
                return Ok(result);
            }
        }
        drop(buffer_guard);

        // Search runs in reverse order (newest first)
        let runs_guard = self.runs.read();
        for run in runs_guard.iter().rev() {
            if run.contains(key) {
                let result = run.get(key);
                drop(runs_guard);
                return result;
            }
        }
        drop(runs_guard);

        Err(RegionalMmapError::NotFound(key.to_string()))
    }

    /// Get all entries for a specific region.
    ///
    /// Returns entries from all runs, with newest version of each key.
    ///
    /// # Errors
    ///
    /// Returns an error if I/O fails.
    pub fn get_region(&self, region_id: u32) -> Result<Vec<(String, TensorData)>> {
        // Use key as dedup key (newer entries overwrite older)
        let mut results: HashMap<String, TensorData> = HashMap::new();

        // Collect from runs (older runs first for proper overwrite semantics)
        let runs_guard = self.runs.read();
        for run in runs_guard.iter() {
            let region_entries = run.get_region(region_id)?;
            for (key, tensor) in region_entries {
                results.insert(key, tensor);
            }
        }
        drop(runs_guard);

        // Collect from buffer (newest, overwrites run data)
        let buffer_guard = self.buffer.read();
        for (lk, key, tensor) in buffer_guard.iter() {
            if lk.region_id() == region_id {
                results.insert(key.clone(), tensor.clone());
            }
        }
        drop(buffer_guard);

        Ok(results.into_iter().collect())
    }

    /// Flush any pending writes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        let entries = std::mem::take(&mut *self.buffer.write());
        if !entries.is_empty() {
            self.flush_buffer(entries)?;
        }
        Ok(())
    }

    /// Compact all runs into a single run.
    ///
    /// # Errors
    ///
    /// Returns an error if compaction fails.
    pub fn compact(&self) -> Result<CompactionStats> {
        // Flush buffer first
        self.flush()?;

        let runs = self.runs.read();
        if runs.len() <= 1 {
            return Ok(CompactionStats {
                runs_compacted: 0,
                entries_before: 0,
                entries_after: 0,
                bytes_saved: 0,
            });
        }

        // Collect all entries from all runs
        let mut all_entries: BTreeMap<String, (LocalityKey, TensorData)> = BTreeMap::new();
        let mut total_entries_before = 0u64;

        for run in runs.iter() {
            run.ensure_index_loaded()?;
            total_entries_before += run.entry_count;

            let key_index = run.key_index.read();
            if let Some(idx) = key_index.as_ref() {
                for (key, loc) in idx {
                    let tensor = run.get(key)?;
                    // Later entries (higher run ID) overwrite earlier ones
                    all_entries.insert(key.clone(), (loc.locality_key, tensor));
                }
            }
        }

        drop(runs);

        // Create new compacted run
        let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_path = self.config.run_dir.join(format!("run_{run_id:08}.rmap"));

        let mut builder = SortedRunBuilder::create(&run_path)?;
        for (key, (locality_key, tensor)) in &all_entries {
            builder.add(*locality_key, key, tensor)?;
        }

        let mut new_run = builder.finish()?;
        new_run.id = run_id;

        let entries_after = all_entries.len() as u64;

        // Calculate old size
        let mut old_size = 0u64;
        let old_runs = std::mem::take(&mut *self.runs.write());
        let runs_compacted = old_runs.len();
        for run in &old_runs {
            if let Ok(meta) = fs::metadata(&run.path) {
                old_size += meta.len();
            }
        }

        // Delete old runs
        for run in old_runs {
            let _ = fs::remove_file(&run.path);
        }

        // Add new run
        let new_size = fs::metadata(&new_run.path).map(|m| m.len()).unwrap_or(0);
        self.runs.write().push(new_run);

        Ok(CompactionStats {
            runs_compacted,
            entries_before: total_entries_before,
            entries_after,
            bytes_saved: old_size.saturating_sub(new_size),
        })
    }

    /// Get the number of sorted runs.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.read().len()
    }

    /// Get total entry count (approximate, may include duplicates).
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        let buffer_count = self.buffer.read().len() as u64;
        let run_count: u64 = self.runs.read().iter().map(|r| r.entry_count).sum();
        buffer_count + run_count
    }

    /// Get the Voronoi partitioner.
    #[must_use]
    pub const fn partitioner(&self) -> &VoronoiPartitioner {
        &self.partitioner
    }

    /// Generate a locality key for an embedding.
    ///
    /// Returns `None` if regions haven't been computed.
    #[must_use]
    pub fn locality_key_for(&self, embedding: &[f32]) -> Option<LocalityKey> {
        self.partitioner.locality_key_for_embedding(embedding)
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::{
        partitioner::Partitioner, voronoi::VoronoiPartitionerConfig, ScalarValue, TensorValue,
    };

    fn create_test_partitioner() -> VoronoiPartitioner {
        let mut config = VoronoiPartitionerConfig::new("node1", 3, 4);
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        // Compute regions
        let samples: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        partitioner.compute_regions_from_samples(&samples);

        partitioner
    }

    fn create_test_tensor(id: i64) -> TensorData {
        let mut tensor = TensorData::new();
        tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String(format!("entity_{id}"))),
        );
        tensor
    }

    #[test]
    fn test_sorted_run_builder() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rmap");

        let mut builder = SortedRunBuilder::create(&path).unwrap();

        // Add entries out of order
        builder
            .add(LocalityKey::new(1, 0), "key_b", &create_test_tensor(2))
            .unwrap();
        builder
            .add(LocalityKey::new(0, 0), "key_a", &create_test_tensor(1))
            .unwrap();
        builder
            .add(LocalityKey::new(1, 1), "key_c", &create_test_tensor(3))
            .unwrap();

        let run = builder.finish().unwrap();

        assert_eq!(run.entry_count, 3);
        assert_eq!(run.min_region, 0);
        assert_eq!(run.max_region, 1);
    }

    #[test]
    fn test_sorted_run_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rmap");

        // Create run
        {
            let mut builder = SortedRunBuilder::create(&path).unwrap();
            builder
                .add(LocalityKey::new(0, 0), "key_a", &create_test_tensor(1))
                .unwrap();
            builder
                .add(LocalityKey::new(0, 1), "key_b", &create_test_tensor(2))
                .unwrap();
            builder.finish().unwrap();
        }

        // Read run
        let run = SortedRun::from_path(0, path).unwrap();
        assert_eq!(run.entry_count, 2);

        let tensor = run.get("key_a").unwrap();
        assert_eq!(
            tensor.get("id"),
            Some(&TensorValue::Scalar(ScalarValue::Int(1)))
        );

        assert!(run.contains("key_a"));
        assert!(!run.contains("nonexistent"));
    }

    #[test]
    fn test_sorted_run_get_region() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rmap");

        // Create run with entries in multiple regions
        {
            let mut builder = SortedRunBuilder::create(&path).unwrap();
            builder
                .add(LocalityKey::new(0, 0), "r0_a", &create_test_tensor(1))
                .unwrap();
            builder
                .add(LocalityKey::new(0, 1), "r0_b", &create_test_tensor(2))
                .unwrap();
            builder
                .add(LocalityKey::new(1, 0), "r1_a", &create_test_tensor(3))
                .unwrap();
            builder
                .add(LocalityKey::new(2, 0), "r2_a", &create_test_tensor(4))
                .unwrap();
            builder.finish().unwrap();
        }

        let run = SortedRun::from_path(0, path).unwrap();

        // Get region 0
        let region0 = run.get_region(0).unwrap();
        assert_eq!(region0.len(), 2);

        // Get region 1
        let region1 = run.get_region(1).unwrap();
        assert_eq!(region1.len(), 1);
        assert_eq!(region1[0].0, "r1_a");

        // Get nonexistent region
        let region99 = run.get_region(99).unwrap();
        assert!(region99.is_empty());
    }

    #[test]
    fn test_regional_mmap_store_basic() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 10,
            compaction_threshold: 4,
        };

        let partitioner = create_test_partitioner();
        let store = RegionalMmapStore::new(partitioner, config).unwrap();

        // Put some entries
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        store
            .put("key1", &create_test_tensor(1), &embedding)
            .unwrap();
        store
            .put("key2", &create_test_tensor(2), &embedding)
            .unwrap();

        // Get entries (from buffer)
        let tensor = store.get("key1").unwrap();
        assert_eq!(
            tensor.get("id"),
            Some(&TensorValue::Scalar(ScalarValue::Int(1)))
        );
    }

    #[test]
    fn test_regional_mmap_store_flush() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 5,
            compaction_threshold: 10,
        };

        let partitioner = create_test_partitioner();
        let store = RegionalMmapStore::new(partitioner, config).unwrap();

        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        // Add entries to trigger flush
        for i in 0..6 {
            store
                .put(&format!("key_{i}"), &create_test_tensor(i), &embedding)
                .unwrap();
        }

        // Should have created a run
        assert!(store.run_count() >= 1);

        // Can still read
        let tensor = store.get("key_0").unwrap();
        assert_eq!(
            tensor.get("id"),
            Some(&TensorValue::Scalar(ScalarValue::Int(0)))
        );
    }

    #[test]
    fn test_regional_mmap_store_compact() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 3,
            compaction_threshold: 10,
        };

        let partitioner = create_test_partitioner();
        let store = RegionalMmapStore::new(partitioner, config).unwrap();

        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        // Create multiple runs
        for i in 0..12 {
            store
                .put(&format!("key_{i}"), &create_test_tensor(i), &embedding)
                .unwrap();
        }
        store.flush().unwrap();

        let runs_before = store.run_count();
        assert!(runs_before > 1);

        // Compact
        let stats = store.compact().unwrap();
        assert!(stats.runs_compacted > 0);

        // Should have single run
        assert_eq!(store.run_count(), 1);

        // Data still accessible
        for i in 0..12 {
            let tensor = store.get(&format!("key_{i}")).unwrap();
            assert_eq!(
                tensor.get("id"),
                Some(&TensorValue::Scalar(ScalarValue::Int(i)))
            );
        }
    }

    #[test]
    fn test_regional_mmap_store_get_region() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 100,
            compaction_threshold: 10,
        };

        let partitioner = create_test_partitioner();
        let store = RegionalMmapStore::new(partitioner, config).unwrap();

        // Add entries with different embeddings (different regions)
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];

        // Get locality keys for each embedding to verify consistency
        let lk1 = store.put("e1_a", &create_test_tensor(1), &emb1).unwrap();
        let lk2 = store.put("e1_b", &create_test_tensor(2), &emb1).unwrap();
        let lk3 = store.put("e2_a", &create_test_tensor(3), &emb2).unwrap();

        // Entries with same embedding should have same region_id
        assert_eq!(
            lk1.region_id(),
            lk2.region_id(),
            "Same embedding should map to same region"
        );

        // Flush to disk
        store.flush().unwrap();

        // Get region for emb1
        let region_id = store.locality_key_for(&emb1).unwrap().region_id();
        assert_eq!(
            region_id,
            lk1.region_id(),
            "Region lookup should be consistent"
        );

        let region_data = store.get_region(region_id).unwrap();

        // Should have at least the 2 entries with emb1 (e1_a, e1_b)
        // Note: If emb2 also maps to same region, we'd have 3
        assert!(
            region_data.len() >= 2,
            "Expected at least 2 entries in region {}, got {}. Keys: {:?}. lk3.region_id={}",
            region_id,
            region_data.len(),
            region_data.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            lk3.region_id()
        );

        // Make sure e2_a is in its own region OR same region
        let region_id_emb2 = lk3.region_id();
        if region_id != region_id_emb2 {
            let region_data_2 = store.get_region(region_id_emb2).unwrap();
            assert!(!region_data_2.is_empty(), "emb2 region should have entries");
        }
    }

    #[test]
    fn test_regional_mmap_store_reopen() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 5,
            compaction_threshold: 10,
        };

        // Write data
        {
            let partitioner = create_test_partitioner();
            let store = RegionalMmapStore::new(partitioner, config.clone()).unwrap();
            let embedding = vec![1.0, 0.0, 0.0, 0.0];

            for i in 0..10 {
                store
                    .put(&format!("key_{i}"), &create_test_tensor(i), &embedding)
                    .unwrap();
            }
            store.flush().unwrap();
        }

        // Reopen and verify
        {
            let partitioner = create_test_partitioner();
            let store = RegionalMmapStore::open(partitioner, config).unwrap();
            assert!(store.run_count() >= 1);

            for i in 0..10 {
                let tensor = store.get(&format!("key_{i}")).unwrap();
                assert_eq!(
                    tensor.get("id"),
                    Some(&TensorValue::Scalar(ScalarValue::Int(i)))
                );
            }
        }
    }

    #[test]
    fn test_regional_mmap_error_display() {
        let io_err = RegionalMmapError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let magic_err = RegionalMmapError::InvalidMagic;
        assert!(magic_err.to_string().contains("Invalid file magic"));

        let ver_err = RegionalMmapError::UnsupportedVersion(99);
        assert!(ver_err.to_string().contains("Unsupported version: 99"));

        let not_found = RegionalMmapError::NotFound("key1".to_string());
        assert!(not_found.to_string().contains("Key not found: key1"));

        let region_err = RegionalMmapError::RegionNotFound(42);
        assert!(region_err.to_string().contains("Region not found: 42"));

        let no_regions = RegionalMmapError::NoRegions;
        assert!(no_regions.to_string().contains("No Voronoi regions"));
    }

    #[test]
    fn test_compaction_stats() {
        let stats = CompactionStats {
            runs_compacted: 5,
            entries_before: 1000,
            entries_after: 800,
            bytes_saved: 50000,
        };

        assert_eq!(stats.runs_compacted, 5);
        assert_eq!(stats.entries_before, 1000);
        assert_eq!(stats.entries_after, 800);
        assert_eq!(stats.bytes_saved, 50000);
    }

    #[test]
    fn test_regional_mmap_config_default() {
        let config = RegionalMmapConfig::default();
        assert_eq!(config.max_entries_per_run, 100_000);
        assert_eq!(config.compaction_threshold, 4);
    }

    #[test]
    fn test_put_with_key() {
        let dir = tempdir().unwrap();
        let config = RegionalMmapConfig {
            run_dir: dir.path().to_path_buf(),
            max_entries_per_run: 100,
            compaction_threshold: 10,
        };

        let partitioner = create_test_partitioner();
        let store = RegionalMmapStore::new(partitioner, config).unwrap();

        let key = LocalityKey::new(0, 42);
        store
            .put_with_key("manual_key", &create_test_tensor(99), key)
            .unwrap();

        let tensor = store.get("manual_key").unwrap();
        assert_eq!(
            tensor.get("id"),
            Some(&TensorValue::Scalar(ScalarValue::Int(99)))
        );
    }
}
