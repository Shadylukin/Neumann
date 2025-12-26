use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub mod hnsw;
pub mod sparse_vector;

pub use hnsw::{HNSWConfig, HNSWIndex};
pub use sparse_vector::SparseVector;

/// Reserved field prefixes for unified entity storage.
///
/// These prefixes are used by the different engines to store their data
/// within a single TensorData entity, enabling cross-engine queries.
pub mod fields {
    /// Graph: outgoing edge pointers (`Vec<String>`)
    pub const OUT: &str = "_out";
    /// Graph: incoming edge pointers (`Vec<String>`)
    pub const IN: &str = "_in";
    /// Vector: embedding vector (`Vec<f32>`)
    pub const EMBEDDING: &str = "_embedding";
    /// Graph/Relational: entity type/label
    pub const LABEL: &str = "_label";
    /// System: entity type discriminator ("node", "edge", "row")
    pub const TYPE: &str = "_type";
    /// System: entity ID
    pub const ID: &str = "_id";
    /// Graph: edge source node
    pub const FROM: &str = "_from";
    /// Graph: edge target node
    pub const TO: &str = "_to";
    /// Graph: edge type
    pub const EDGE_TYPE: &str = "_edge_type";
    /// Graph: whether edge is directed
    pub const DIRECTED: &str = "_directed";
    /// Relational: table name for row entities
    pub const TABLE: &str = "_table";
}

/// Thread-safe Bloom filter for fast negative lookups.
///
/// A Bloom filter is a probabilistic data structure that can quickly tell you:
/// - Definitely NOT in set (no false negatives)
/// - POSSIBLY in set (may have false positives)
///
/// This is useful for avoiding expensive lookups when the key doesn't exist.
pub struct BloomFilter {
    bits: Box<[AtomicU64]>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with the given expected number of items and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to insert
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal size: m = -n*ln(p) / (ln(2)^2)
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let num_bits =
            (-(expected_items as f64) * false_positive_rate.ln() / ln2_squared).ceil() as usize;
        let num_bits = num_bits.max(64); // Minimum 64 bits

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_items as f64) * std::f64::consts::LN_2).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 16); // Between 1 and 16 hash functions

        // Allocate bit array (using u64 blocks)
        let num_blocks = num_bits.div_ceil(64);
        let bits: Vec<AtomicU64> = (0..num_blocks).map(|_| AtomicU64::new(0)).collect();

        Self {
            bits: bits.into_boxed_slice(),
            num_bits,
            num_hashes,
        }
    }

    /// Create a Bloom filter with default parameters for typical key-value usage.
    /// Expects ~10,000 items with 1% false positive rate.
    pub fn with_defaults() -> Self {
        Self::new(10_000, 0.01)
    }

    /// Add a key to the Bloom filter.
    pub fn add<K: Hash>(&self, key: &K) {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(key, i);
            let block_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bits[block_index].fetch_or(1 << bit_offset, Ordering::Relaxed);
        }
    }

    /// Check if a key might be in the set.
    /// Returns false if the key is definitely NOT in the set.
    /// Returns true if the key MIGHT be in the set (could be false positive).
    #[inline]
    pub fn might_contain<K: Hash>(&self, key: &K) -> bool {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(key, i);
            let block_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            if (self.bits[block_index].load(Ordering::Relaxed) & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear all bits in the filter.
    pub fn clear(&self) {
        for block in self.bits.iter() {
            block.store(0, Ordering::Relaxed);
        }
    }

    /// Compute hash index for a key with a given seed.
    #[inline]
    fn hash_index<K: Hash>(&self, key: &K, seed: usize) -> usize {
        let mut hasher = SipHasher::new_with_seed(seed as u64);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_bits
    }

    /// Get the number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get the number of hash functions used.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

// Simple SipHash-like hasher with configurable seed
struct SipHasher {
    state: u64,
    seed: u64,
}

impl SipHasher {
    fn new_with_seed(seed: u64) -> Self {
        Self {
            state: seed ^ 0x736f_6d65_7073_6575,
            seed,
        }
    }
}

impl Hasher for SipHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state = self.state.wrapping_mul(31).wrapping_add(*byte as u64);
            self.state ^= self.seed;
        }
    }
}

/// Represents different types of values a tensor can hold
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorValue {
    /// Scalar values (properties): integers, floats, strings, booleans
    Scalar(ScalarValue),
    /// Vector values (embeddings): f32 arrays for similarity search
    Vector(Vec<f32>),
    /// Sparse vector values: only non-zero positions stored
    ///
    /// Philosophy: Zero represents absence of information, not a stored value.
    /// The dimension defines the boundary/shell of meaningful space.
    Sparse(SparseVector),
    /// Pointer to another tensor (relationships)
    Pointer(String),
    /// List of pointers (multiple relationships)
    Pointers(Vec<String>),
}

/// Scalar value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

/// An entity that can hold scalar properties, vector embeddings, and pointers to other tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorData {
    fields: HashMap<String, TensorValue>,
}

impl TensorData {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: TensorValue) {
        self.fields.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&TensorValue> {
        self.fields.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<TensorValue> {
        self.fields.remove(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }

    pub fn has(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn entity_type(&self) -> Option<&str> {
        match self.get(fields::TYPE) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn entity_id(&self) -> Option<i64> {
        match self.get(fields::ID) {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => Some(*id),
            _ => None,
        }
    }

    pub fn label(&self) -> Option<&str> {
        match self.get(fields::LABEL) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn embedding(&self) -> Option<&Vec<f32>> {
        match self.get(fields::EMBEDDING) {
            Some(TensorValue::Vector(v)) => Some(v),
            _ => None,
        }
    }

    pub fn outgoing_edges(&self) -> Option<&Vec<String>> {
        match self.get(fields::OUT) {
            Some(TensorValue::Pointers(p)) => Some(p),
            _ => None,
        }
    }

    pub fn incoming_edges(&self) -> Option<&Vec<String>> {
        match self.get(fields::IN) {
            Some(TensorValue::Pointers(p)) => Some(p),
            _ => None,
        }
    }

    pub fn set_entity_type(&mut self, entity_type: &str) {
        self.set(
            fields::TYPE,
            TensorValue::Scalar(ScalarValue::String(entity_type.to_string())),
        );
    }

    pub fn set_entity_id(&mut self, id: i64) {
        self.set(fields::ID, TensorValue::Scalar(ScalarValue::Int(id)));
    }

    pub fn set_label(&mut self, label: &str) {
        self.set(
            fields::LABEL,
            TensorValue::Scalar(ScalarValue::String(label.to_string())),
        );
    }

    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.set(fields::EMBEDDING, TensorValue::Vector(embedding));
    }

    pub fn set_outgoing_edges(&mut self, edges: Vec<String>) {
        self.set(fields::OUT, TensorValue::Pointers(edges));
    }

    pub fn set_incoming_edges(&mut self, edges: Vec<String>) {
        self.set(fields::IN, TensorValue::Pointers(edges));
    }

    /// Adds edge if not already present.
    pub fn add_outgoing_edge(&mut self, edge_key: String) {
        let mut edges = match self.get(fields::OUT) {
            Some(TensorValue::Pointers(p)) => p.clone(),
            _ => Vec::new(),
        };
        if !edges.contains(&edge_key) {
            edges.push(edge_key);
        }
        self.set(fields::OUT, TensorValue::Pointers(edges));
    }

    /// Adds edge if not already present.
    pub fn add_incoming_edge(&mut self, edge_key: String) {
        let mut edges = match self.get(fields::IN) {
            Some(TensorValue::Pointers(p)) => p.clone(),
            _ => Vec::new(),
        };
        if !edges.contains(&edge_key) {
            edges.push(edge_key);
        }
        self.set(fields::IN, TensorValue::Pointers(edges));
    }

    pub fn has_embedding(&self) -> bool {
        self.has(fields::EMBEDDING)
    }

    pub fn has_edges(&self) -> bool {
        self.has(fields::OUT) || self.has(fields::IN)
    }

    /// Returns fields that don't start with underscore.
    pub fn user_fields(&self) -> impl Iterator<Item = (&String, &TensorValue)> {
        self.fields.iter().filter(|(k, _)| !k.starts_with('_'))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorValue)> {
        self.fields.iter()
    }
}

pub type Result<T> = std::result::Result<T, TensorStoreError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorStoreError {
    NotFound(String),
}

impl std::fmt::Display for TensorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorStoreError::NotFound(key) => write!(f, "Key not found: {}", key),
        }
    }
}

impl std::error::Error for TensorStoreError {}

/// Errors that can occur during snapshot operations.
#[derive(Debug)]
pub enum SnapshotError {
    /// Failed to create or open the file.
    IoError(std::io::Error),
    /// Failed to serialize or deserialize data.
    SerializationError(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::IoError(e) => write!(f, "I/O error: {}", e),
            SnapshotError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SnapshotError::IoError(e) => Some(e),
            SnapshotError::SerializationError(_) => None,
        }
    }
}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        SnapshotError::IoError(e)
    }
}

impl From<bincode::Error> for SnapshotError {
    fn from(e: bincode::Error) -> Self {
        SnapshotError::SerializationError(e.to_string())
    }
}

/// Thread-safe key-value store for tensor data using sharded concurrent HashMap.
///
/// Uses DashMap internally for lock-free concurrent reads and sharded writes.
/// This provides better performance under write contention compared to a single RwLock.
///
/// # Concurrency Model
///
/// - **Reads**: Lock-free, can proceed in parallel with other reads and writes to different shards
/// - **Writes**: Only block other writes to the same shard (~16 shards by default)
/// - **No poisoning**: Unlike RwLock, panics don't poison the entire store
///
/// # Bloom Filter
///
/// An optional Bloom filter can be enabled to accelerate negative lookups for sparse key spaces.
/// When enabled, `get()` and `exists()` will first check the Bloom filter and return immediately
/// if the key is definitely not present, avoiding the HashMap lookup.
/// Clone creates a shared reference to the same underlying storage.
#[derive(Clone)]
pub struct TensorStore {
    data: Arc<DashMap<String, TensorData>>,
    bloom_filter: Option<Arc<BloomFilter>>,
}

impl TensorStore {
    const PARALLEL_THRESHOLD: usize = 1000;

    pub fn new() -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            bloom_filter: None,
        }
    }

    /// Create a store with a specific capacity hint for better initial allocation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Arc::new(DashMap::with_capacity(capacity)),
            bloom_filter: None,
        }
    }

    /// Create a store with a Bloom filter for fast negative lookups.
    ///
    /// This is useful for sparse key spaces where most lookups are misses.
    /// The Bloom filter provides O(1) rejection of non-existent keys.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to store
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn with_bloom_filter(expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            bloom_filter: Some(Arc::new(BloomFilter::new(
                expected_items,
                false_positive_rate,
            ))),
        }
    }

    /// Create a store with default Bloom filter settings.
    ///
    /// Uses defaults: 10,000 expected items, 1% false positive rate.
    pub fn with_default_bloom_filter() -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            bloom_filter: Some(Arc::new(BloomFilter::with_defaults())),
        }
    }

    /// Check if the store has a Bloom filter enabled.
    pub fn has_bloom_filter(&self) -> bool {
        self.bloom_filter.is_some()
    }

    pub fn put(&self, key: impl Into<String>, tensor: TensorData) -> Result<()> {
        let key = key.into();
        if let Some(ref filter) = self.bloom_filter {
            filter.add(&key);
        }
        self.data.insert(key, tensor);
        Ok(())
    }

    /// Returns cloned data to ensure thread safety.
    ///
    /// If a Bloom filter is enabled, this will first check the filter and return
    /// `NotFound` immediately if the key is definitely not present.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        // Fast path: check Bloom filter first
        if let Some(ref filter) = self.bloom_filter {
            if !filter.might_contain(&key) {
                return Err(TensorStoreError::NotFound(key.to_string()));
            }
        }
        self.data
            .get(key)
            .map(|r| r.value().clone())
            .ok_or_else(|| TensorStoreError::NotFound(key.to_string()))
    }

    pub fn delete(&self, key: &str) -> Result<()> {
        self.data
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| TensorStoreError::NotFound(key.to_string()))
    }

    /// Check if a key exists in the store.
    ///
    /// If a Bloom filter is enabled, this will first check the filter and return
    /// `false` immediately if the key is definitely not present.
    pub fn exists(&self, key: &str) -> bool {
        // Fast path: check Bloom filter first
        if let Some(ref filter) = self.bloom_filter {
            if !filter.might_contain(&key) {
                return false;
            }
        }
        self.data.contains_key(key)
    }

    pub fn scan(&self, prefix: &str) -> Vec<String> {
        if self.data.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .filter(|r| r.key().starts_with(prefix))
                .map(|r| r.key().clone())
                .collect()
        } else {
            self.data
                .iter()
                .filter(|r| r.key().starts_with(prefix))
                .map(|r| r.key().clone())
                .collect()
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&self) {
        self.data.clear();
        if let Some(ref filter) = self.bloom_filter {
            filter.clear();
        }
    }

    pub fn scan_count(&self, prefix: &str) -> usize {
        if self.data.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .filter(|r| r.key().starts_with(prefix))
                .count()
        } else {
            self.data
                .iter()
                .filter(|r| r.key().starts_with(prefix))
                .count()
        }
    }

    /// Save a snapshot of the store to a file.
    ///
    /// The snapshot is written atomically by first writing to a temporary file
    /// and then renaming it to the target path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = TensorStore::new();
    /// store.put("key", tensor).unwrap();
    /// store.save_snapshot("data.bin")?;
    /// ```
    pub fn save_snapshot<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), SnapshotError> {
        let path = path.as_ref();

        // Create temp file in same directory for atomic rename
        let temp_path = path.with_extension("tmp");

        // Collect all data into a HashMap for serialization
        let snapshot: HashMap<String, TensorData> = self
            .data
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        // Write to temp file
        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &snapshot)?;

        // Atomic rename
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Load a store from a snapshot file.
    ///
    /// Returns a new TensorStore with the data from the snapshot.
    /// Note: Bloom filter state is not persisted and will be rebuilt if enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = TensorStore::load_snapshot("data.bin")?;
    /// let tensor = store.get("key")?;
    /// ```
    pub fn load_snapshot<P: AsRef<Path>>(path: P) -> std::result::Result<Self, SnapshotError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: HashMap<String, TensorData> = bincode::deserialize_from(reader)?;

        let store = TensorStore::new();
        for (key, value) in snapshot {
            store.data.insert(key, value);
        }

        Ok(store)
    }

    /// Load a store from a snapshot file with a Bloom filter.
    ///
    /// The Bloom filter is rebuilt from the loaded keys.
    pub fn load_snapshot_with_bloom_filter<P: AsRef<Path>>(
        path: P,
        expected_items: usize,
        false_positive_rate: f64,
    ) -> std::result::Result<Self, SnapshotError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: HashMap<String, TensorData> = bincode::deserialize_from(reader)?;

        let bloom = BloomFilter::new(expected_items, false_positive_rate);
        let data = DashMap::new();

        for (key, value) in snapshot {
            bloom.add(&key);
            data.insert(key, value);
        }

        Ok(Self {
            data: Arc::new(data),
            bloom_filter: Some(Arc::new(bloom)),
        })
    }

    /// Save a compressed snapshot using bespoke tensor compression.
    ///
    /// Compression includes:
    /// - Vector quantization (int8 or binary) for embeddings
    /// - Delta + varint encoding for sorted ID lists
    /// - Run-length encoding for repeated values
    ///
    /// # Errors
    /// Returns error if file creation or serialization fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn save_snapshot_compressed<P: AsRef<Path>>(
        &self,
        path: P,
        config: tensor_compress::CompressionConfig,
    ) -> std::result::Result<(), SnapshotError> {
        use tensor_compress::format::{
            compress_vector, CompressedEntry, CompressedScalar, CompressedSnapshot,
            CompressedValue, Header,
        };

        let path = path.as_ref();
        let temp_path = path.with_extension("tmp");

        let mut entries = Vec::with_capacity(self.data.len());

        for entry in self.data.iter() {
            let key = entry.key().clone();
            let tensor = entry.value();

            let mut fields = HashMap::new();
            for (field_name, value) in tensor.iter() {
                let compressed = match value {
                    TensorValue::Scalar(s) => CompressedValue::Scalar(match s {
                        ScalarValue::Null => CompressedScalar::Null,
                        ScalarValue::Bool(b) => CompressedScalar::Bool(*b),
                        ScalarValue::Int(i) => CompressedScalar::Int(*i),
                        ScalarValue::Float(f) => CompressedScalar::Float(*f),
                        ScalarValue::String(s) => CompressedScalar::String(s.clone()),
                        ScalarValue::Bytes(b) => {
                            CompressedScalar::String(format!("bytes:{}", b.len()))
                        },
                    }),
                    TensorValue::Vector(v) => compress_vector(v, &key, field_name, &config),
                    TensorValue::Sparse(sv) => {
                        // Convert sparse to dense for compression, then compress
                        // Future: add native sparse compression format
                        compress_vector(&sv.to_dense(), &key, field_name, &config)
                    },
                    TensorValue::Pointer(p) => CompressedValue::Pointer(p.clone()),
                    TensorValue::Pointers(ps) => CompressedValue::Pointers(ps.clone()),
                };
                fields.insert(field_name.to_string(), compressed);
            }

            entries.push(CompressedEntry { key, fields });
        }

        let header = Header::new(config, entries.len() as u64);
        let snapshot = CompressedSnapshot { header, entries };

        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &snapshot)?;

        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Load a compressed snapshot.
    ///
    /// # Errors
    /// Returns error if file read or deserialization fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn load_snapshot_compressed<P: AsRef<Path>>(
        path: P,
    ) -> std::result::Result<Self, SnapshotError> {
        use tensor_compress::format::{decompress_vector, CompressedSnapshot, CompressedValue};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: CompressedSnapshot = bincode::deserialize_from(reader)?;

        snapshot
            .header
            .validate()
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;

        let store = TensorStore::new();

        for entry in snapshot.entries {
            let mut tensor = TensorData::new();

            for (field_name, value) in entry.fields {
                let tensor_value = match value {
                    CompressedValue::Scalar(s) => {
                        use tensor_compress::format::CompressedScalar;
                        TensorValue::Scalar(match s {
                            CompressedScalar::Null => ScalarValue::Null,
                            CompressedScalar::Bool(b) => ScalarValue::Bool(b),
                            CompressedScalar::Int(i) => ScalarValue::Int(i),
                            CompressedScalar::Float(f) => ScalarValue::Float(f),
                            CompressedScalar::String(s) => ScalarValue::String(s),
                        })
                    },
                    CompressedValue::VectorRaw(v) => TensorValue::Vector(v),
                    CompressedValue::VectorInt8 { .. }
                    | CompressedValue::VectorBinary { .. }
                    | CompressedValue::IdList(_) => {
                        let v = decompress_vector(&value)
                            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;
                        TensorValue::Vector(v)
                    },
                    CompressedValue::RleInt(encoded) => {
                        let ints = tensor_compress::rle_decode(&encoded);
                        TensorValue::Vector(ints.iter().map(|&i| i as f32).collect())
                    },
                    CompressedValue::Pointer(p) => TensorValue::Pointer(p),
                    CompressedValue::Pointers(ps) => TensorValue::Pointers(ps),
                };

                tensor.set(&field_name, tensor_value);
            }

            store.data.insert(entry.key, tensor);
        }

        Ok(store)
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified entity store that provides a shared storage layer for all engines.
///
/// EntityStore wraps a TensorStore and provides entity-oriented access patterns
/// that enable cross-engine queries. All engines can share the same EntityStore
/// to enable unified entity access.
///
/// # Entity Key Format
///
/// Entities use the format `{type}:{id}`, for example:
/// - `user:1` - A user entity
/// - `post:42` - A post entity
/// - `edge:123` - An edge entity
///
/// # Unified Entity Model
///
/// A single entity can have:
/// - Relational fields (scalars like name, age, email)
/// - Graph connections (outgoing/incoming edge pointers)
/// - Vector embeddings (for similarity search)
///
/// ```text
/// user:1
/// ├── Relational: name="Alice", age=30, email="..."
/// ├── Graph: _out=["edge:1", "edge:2"], _in=["edge:3"]
/// └── Vector: _embedding=[0.1, 0.2, 0.3, ...]
/// ```
#[derive(Clone)]
pub struct EntityStore {
    store: Arc<TensorStore>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            store: Arc::new(TensorStore::new()),
        }
    }

    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store: Arc::new(store),
        }
    }

    pub fn with_arc(store: Arc<TensorStore>) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    pub fn store_arc(&self) -> Arc<TensorStore> {
        Arc::clone(&self.store)
    }

    pub fn entity_key(entity_type: &str, id: u64) -> String {
        format!("{}:{}", entity_type, id)
    }

    pub fn parse_key(key: &str) -> Option<(&str, u64)> {
        let parts: Vec<&str> = key.splitn(2, ':').collect();
        if parts.len() == 2 {
            parts[1].parse().ok().map(|id| (parts[0], id))
        } else {
            None
        }
    }

    pub fn get(&self, key: &str) -> Result<TensorData> {
        self.store.get(key)
    }

    pub fn put(&self, key: impl Into<String>, data: TensorData) -> Result<()> {
        self.store.put(key, data)
    }

    pub fn delete(&self, key: &str) -> Result<()> {
        self.store.delete(key)
    }

    pub fn exists(&self, key: &str) -> bool {
        self.store.exists(key)
    }

    /// Returns existing entity or creates empty TensorData if not found.
    pub fn get_or_create(&self, key: &str) -> TensorData {
        self.store.get(key).unwrap_or_else(|_| TensorData::new())
    }

    /// Atomically read-modify-write an entity.
    pub fn update<F>(&self, key: &str, updater: F) -> Result<()>
    where
        F: FnOnce(&mut TensorData),
    {
        let mut data = self.get_or_create(key);
        updater(&mut data);
        self.store.put(key, data)
    }

    pub fn scan_type(&self, entity_type: &str) -> Vec<String> {
        self.store.scan(&format!("{}:", entity_type))
    }

    pub fn scan_with_embeddings(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| {
                if let Ok(data) = self.store.get(key) {
                    data.has_embedding()
                } else {
                    false
                }
            })
            .collect()
    }

    pub fn scan_with_edges(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| {
                if let Ok(data) = self.store.get(key) {
                    data.has_edges()
                } else {
                    false
                }
            })
            .collect()
    }

    pub fn get_embedding(&self, key: &str) -> Option<Vec<f32>> {
        self.store
            .get(key)
            .ok()
            .and_then(|data| data.embedding().cloned())
    }

    /// Creates entity if it doesn't exist.
    pub fn set_embedding(&self, key: &str, embedding: Vec<f32>) -> Result<()> {
        self.update(key, |data| {
            data.set_embedding(embedding);
        })
    }

    /// Updates both from and to nodes with edge pointers.
    pub fn add_edge(&self, from_key: &str, to_key: &str, edge_key: &str) -> Result<()> {
        self.update(from_key, |data| {
            data.add_outgoing_edge(edge_key.to_string());
        })?;

        self.update(to_key, |data| {
            data.add_incoming_edge(edge_key.to_string());
        })
    }

    pub fn outgoing_neighbors(&self, key: &str) -> Result<Vec<String>> {
        let data = self.get(key)?;
        Ok(data.outgoing_edges().cloned().unwrap_or_default())
    }

    pub fn incoming_neighbors(&self, key: &str) -> Result<Vec<String>> {
        let data = self.get(key)?;
        Ok(data.incoming_edges().cloned().unwrap_or_default())
    }

    pub fn clear(&self) {
        self.store.clear();
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn count_type(&self, entity_type: &str) -> usize {
        self.store.scan_count(&format!("{}:", entity_type))
    }
}

impl Default for EntityStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // TensorData tests

    #[test]
    fn tensor_data_stores_scalars() {
        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        tensor.set("score", TensorValue::Scalar(ScalarValue::Float(95.5)));
        tensor.set("active", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("nullable", TensorValue::Scalar(ScalarValue::Null));

        assert_eq!(tensor.len(), 5);
        assert!(tensor.has("name"));
        assert!(!tensor.has("nonexistent"));

        match tensor.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn tensor_data_stores_vectors() {
        let mut tensor = TensorData::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        tensor.set("embedding", TensorValue::Vector(embedding.clone()));

        match tensor.get("embedding") {
            Some(TensorValue::Vector(v)) => assert_eq!(v, &embedding),
            _ => panic!("expected vector"),
        }
    }

    #[test]
    fn tensor_data_stores_pointers() {
        let mut tensor = TensorData::new();
        tensor.set("friend", TensorValue::Pointer("user:2".into()));
        tensor.set(
            "posts",
            TensorValue::Pointers(vec!["post:1".into(), "post:2".into()]),
        );

        match tensor.get("friend") {
            Some(TensorValue::Pointer(p)) => assert_eq!(p, "user:2"),
            _ => panic!("expected pointer"),
        }

        match tensor.get("posts") {
            Some(TensorValue::Pointers(ps)) => assert_eq!(ps.len(), 2),
            _ => panic!("expected pointers"),
        }
    }

    #[test]
    fn tensor_data_remove_field() {
        let mut tensor = TensorData::new();
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(1)));

        assert!(tensor.has("key"));
        let removed = tensor.remove("key");
        assert!(removed.is_some());
        assert!(!tensor.has("key"));
        assert!(tensor.remove("key").is_none());
    }

    #[test]
    fn tensor_data_overwrite_field() {
        let mut tensor = TensorData::new();
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(1)));
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(2)));

        match tensor.get("key") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 2),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn tensor_data_empty() {
        let tensor = TensorData::new();
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
        assert!(tensor.get("anything").is_none());
    }

    #[test]
    fn tensor_data_keys_iteration() {
        let mut tensor = TensorData::new();
        tensor.set("a", TensorValue::Scalar(ScalarValue::Int(1)));
        tensor.set("b", TensorValue::Scalar(ScalarValue::Int(2)));

        let keys: Vec<_> = tensor.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    // TensorStore tests

    #[test]
    fn store_put_get() {
        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));

        store.put("key1", tensor).unwrap();

        let retrieved = store.get("key1").unwrap();
        match retrieved.get("value") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 42),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_get_not_found() {
        let store = TensorStore::new();
        let result = store.get("nonexistent");
        assert!(matches!(result, Err(TensorStoreError::NotFound(_))));
    }

    #[test]
    fn store_delete() {
        let store = TensorStore::new();
        store.put("key1", TensorData::new()).unwrap();

        assert!(store.exists("key1"));
        store.delete("key1").unwrap();
        assert!(!store.exists("key1"));
    }

    #[test]
    fn store_delete_not_found() {
        let store = TensorStore::new();
        let result = store.delete("nonexistent");
        assert!(matches!(result, Err(TensorStoreError::NotFound(_))));
    }

    #[test]
    fn store_exists() {
        let store = TensorStore::new();
        assert!(!store.exists("key"));
        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
    }

    #[test]
    fn store_overwrite() {
        let store = TensorStore::new();
        let mut t1 = TensorData::new();
        t1.set("v", TensorValue::Scalar(ScalarValue::Int(1)));
        let mut t2 = TensorData::new();
        t2.set("v", TensorValue::Scalar(ScalarValue::Int(2)));

        store.put("key", t1).unwrap();
        store.put("key", t2).unwrap();

        let retrieved = store.get("key").unwrap();
        match retrieved.get("v") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 2),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_scan_basic() {
        let store = TensorStore::new();
        store.put("user:1", TensorData::new()).unwrap();
        store.put("user:2", TensorData::new()).unwrap();
        store.put("post:1", TensorData::new()).unwrap();

        let users = store.scan("user:");
        assert_eq!(users.len(), 2);
        assert!(users.contains(&"user:1".to_string()));
        assert!(users.contains(&"user:2".to_string()));
    }

    #[test]
    fn store_scan_empty_prefix() {
        let store = TensorStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        let all = store.scan("");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn store_scan_no_match() {
        let store = TensorStore::new();
        store.put("user:1", TensorData::new()).unwrap();

        let results = store.scan("post:");
        assert!(results.is_empty());
    }

    #[test]
    fn store_len_and_is_empty() {
        let store = TensorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.put("key", TensorData::new()).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn store_10k_entities() {
        let store = TensorStore::new();

        for i in 0..10_000 {
            let mut tensor = TensorData::new();
            tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
            tensor.set("embedding", TensorValue::Vector(vec![i as f32; 128]));
            store.put(format!("entity:{}", i), tensor).unwrap();
        }

        assert_eq!(store.len(), 10_000);

        let tensor = store.get("entity:5000").unwrap();
        match tensor.get("id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => assert_eq!(*id, 5000),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_concurrent_writes() {
        let store = Arc::new(TensorStore::new());
        let mut handles = vec![];

        for t in 0..4 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let mut tensor = TensorData::new();
                    tensor.set("value", TensorValue::Scalar(ScalarValue::Int(i)));
                    store.put(format!("thread{}:key{}", t, i), tensor).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 4000);
    }

    #[test]
    fn store_concurrent_read_write() {
        let store = Arc::new(TensorStore::new());

        for i in 0..100 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        let mut handles = vec![];

        for _ in 0..4 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = store.get(&format!("key:{}", i));
                }
            }));
        }

        for t in 0..2 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let start = 100 + t * 100;
                for i in start..(start + 100) {
                    store.put(format!("key:{}", i), TensorData::new()).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 300);
    }

    #[test]
    fn store_concurrent_writes_same_keys() {
        // This test verifies DashMap's sharded locking under contention
        // Multiple threads write to overlapping keys
        let store = Arc::new(TensorStore::new());
        let mut handles = vec![];

        for t in 0..8 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    let key = format!("key:{}", i % 100); // Only 100 unique keys
                    let mut tensor = TensorData::new();
                    tensor.set("thread", TensorValue::Scalar(ScalarValue::Int(t)));
                    tensor.set("iter", TensorValue::Scalar(ScalarValue::Int(i)));
                    store.put(key, tensor).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have exactly 100 keys (last write wins)
        assert_eq!(store.len(), 100);
    }

    #[test]
    fn store_scan_many_prefixes() {
        let store = TensorStore::new();

        for i in 0..100 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
            store
                .put(format!("comment:{}", i), TensorData::new())
                .unwrap();
        }

        assert_eq!(store.scan("user:").len(), 100);
        assert_eq!(store.scan("post:").len(), 100);
        assert_eq!(store.scan("comment:").len(), 100);
        assert_eq!(store.scan("").len(), 300);
    }

    #[test]
    fn store_empty_key() {
        let store = TensorStore::new();
        store.put("", TensorData::new()).unwrap();
        assert!(store.exists(""));
        store.delete("").unwrap();
        assert!(!store.exists(""));
    }

    #[test]
    fn store_unicode_keys() {
        let store = TensorStore::new();
        store.put("user:café", TensorData::new()).unwrap();
        store.put("user:東京", TensorData::new()).unwrap();

        assert!(store.exists("user:café"));
        assert!(store.exists("user:東京"));
    }

    #[test]
    fn store_clear() {
        let store = TensorStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn store_scan_count() {
        let store = TensorStore::new();
        for i in 0..50 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
        }
        for i in 0..30 {
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
        }

        assert_eq!(store.scan_count("user:"), 50);
        assert_eq!(store.scan_count("post:"), 30);
        assert_eq!(store.scan_count(""), 80);
        assert_eq!(store.scan_count("nonexistent:"), 0);
    }

    #[test]
    fn store_with_capacity() {
        let store = TensorStore::with_capacity(1000);
        assert!(store.is_empty());

        for i in 0..1000 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }
        assert_eq!(store.len(), 1000);
    }

    #[test]
    fn tensor_data_stores_bytes() {
        let mut tensor = TensorData::new();
        let data = vec![0x00, 0xFF, 0x42];
        tensor.set(
            "binary",
            TensorValue::Scalar(ScalarValue::Bytes(data.clone())),
        );

        match tensor.get("binary") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => assert_eq!(b, &data),
            _ => panic!("expected bytes"),
        }
    }

    #[test]
    fn error_display_not_found() {
        let err = TensorStoreError::NotFound("test_key".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test_key"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &TensorStoreError::NotFound("x".into());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn store_default_trait() {
        let store = TensorStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn tensor_data_default_trait() {
        let tensor = TensorData::default();
        assert!(tensor.is_empty());
    }

    #[test]
    fn tensor_data_clone() {
        let mut original = TensorData::new();
        original.set("key", TensorValue::Scalar(ScalarValue::Int(42)));

        let cloned = original.clone();
        assert_eq!(cloned.len(), 1);
        match cloned.get("key") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 42),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn tensor_value_clone_all_variants() {
        let scalar = TensorValue::Scalar(ScalarValue::Int(1));
        let vector = TensorValue::Vector(vec![1.0, 2.0]);
        let pointer = TensorValue::Pointer("ref".into());
        let pointers = TensorValue::Pointers(vec!["a".into(), "b".into()]);

        assert_eq!(scalar.clone(), scalar);
        assert_eq!(vector.clone(), vector);
        assert_eq!(pointer.clone(), pointer);
        assert_eq!(pointers.clone(), pointers);
    }

    #[test]
    fn scalar_value_clone_all_variants() {
        let null = ScalarValue::Null;
        let bool_val = ScalarValue::Bool(true);
        let int_val = ScalarValue::Int(42);
        let float_val = ScalarValue::Float(3.14);
        let string_val = ScalarValue::String("test".into());
        let bytes_val = ScalarValue::Bytes(vec![1, 2, 3]);

        assert_eq!(null.clone(), null);
        assert_eq!(bool_val.clone(), bool_val);
        assert_eq!(int_val.clone(), int_val);
        assert_eq!(float_val.clone(), float_val);
        assert_eq!(string_val.clone(), string_val);
        assert_eq!(bytes_val.clone(), bytes_val);
    }

    #[test]
    fn tensor_store_error_clone() {
        let err = TensorStoreError::NotFound("key".into());
        assert_eq!(err.clone(), err);
    }

    #[test]
    fn tensor_data_debug() {
        let tensor = TensorData::new();
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("TensorData"));
    }

    #[test]
    fn tensor_value_debug() {
        let val = TensorValue::Scalar(ScalarValue::Int(1));
        let debug_str = format!("{:?}", val);
        assert!(debug_str.contains("Scalar"));
    }

    #[test]
    fn tensor_store_error_debug() {
        let err = TensorStoreError::NotFound("key".into());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NotFound"));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // crossbeam-epoch has known Miri issues with stacked borrows
    fn store_parallel_scan_large_dataset() {
        let store = TensorStore::new();

        // Insert enough entries to trigger parallel scan (>1000)
        for i in 0..1500 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
        }
        for i in 0..500 {
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
        }

        assert_eq!(store.len(), 2000);

        // These should use parallel iteration
        let users = store.scan("user:");
        assert_eq!(users.len(), 1500);

        let posts = store.scan("post:");
        assert_eq!(posts.len(), 500);

        assert_eq!(store.scan_count("user:"), 1500);
        assert_eq!(store.scan_count("post:"), 500);
        assert_eq!(store.scan_count(""), 2000);
    }

    // Bloom filter tests

    #[test]
    fn bloom_filter_basic() {
        let filter = BloomFilter::new(100, 0.01);

        filter.add(&"key1");
        filter.add(&"key2");
        filter.add(&"key3");

        // Added keys should be found (no false negatives)
        assert!(filter.might_contain(&"key1"));
        assert!(filter.might_contain(&"key2"));
        assert!(filter.might_contain(&"key3"));

        // Non-added keys should likely not be found (may have false positives)
        // We check multiple to reduce chance of false positive affecting test
        let mut misses = 0;
        for i in 100..200 {
            if !filter.might_contain(&format!("nonexistent{}", i)) {
                misses += 1;
            }
        }
        // With 1% false positive rate, should have ~99% misses
        assert!(
            misses > 90,
            "Too many false positives: {} misses out of 100",
            misses
        );
    }

    #[test]
    fn bloom_filter_clear() {
        let filter = BloomFilter::new(100, 0.01);

        filter.add(&"key1");
        assert!(filter.might_contain(&"key1"));

        filter.clear();

        // After clear, key should not be found
        assert!(!filter.might_contain(&"key1"));
    }

    #[test]
    fn bloom_filter_defaults() {
        let filter = BloomFilter::with_defaults();

        // Should be able to add items
        filter.add(&"test_key");
        assert!(filter.might_contain(&"test_key"));

        // Check configuration (10k items, 1% FP rate)
        assert!(filter.num_bits() > 0);
        assert!(filter.num_hashes() > 0);
    }

    #[test]
    fn bloom_filter_many_items() {
        let filter = BloomFilter::new(1000, 0.01);

        // Add 1000 items
        for i in 0..1000 {
            filter.add(&format!("item{}", i));
        }

        // All added items should be found
        for i in 0..1000 {
            assert!(filter.might_contain(&format!("item{}", i)));
        }
    }

    #[test]
    fn store_with_bloom_filter() {
        let store = TensorStore::with_bloom_filter(100, 0.01);
        assert!(store.has_bloom_filter());

        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("key1", tensor).unwrap();

        // Key should be found
        assert!(store.exists("key1"));
        assert!(store.get("key1").is_ok());

        // Non-existent key should return not found
        assert!(!store.exists("nonexistent"));
        assert!(store.get("nonexistent").is_err());
    }

    #[test]
    fn store_with_default_bloom_filter() {
        let store = TensorStore::with_default_bloom_filter();
        assert!(store.has_bloom_filter());

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
    }

    #[test]
    fn store_bloom_filter_accelerates_negative_lookups() {
        let store = TensorStore::with_bloom_filter(100, 0.01);

        // Add some keys
        for i in 0..50 {
            store
                .put(format!("existing:{}", i), TensorData::new())
                .unwrap();
        }

        // Existing keys should be found
        for i in 0..50 {
            assert!(store.exists(&format!("existing:{}", i)));
        }

        // Non-existing keys should not be found
        // The Bloom filter will reject most of these without HashMap lookup
        for i in 1000..1050 {
            assert!(!store.exists(&format!("missing:{}", i)));
        }
    }

    #[test]
    fn store_bloom_filter_clear() {
        let store = TensorStore::with_bloom_filter(100, 0.01);

        store.put("key1", TensorData::new()).unwrap();
        assert!(store.exists("key1"));

        store.clear();

        // After clear, key should not be found
        assert!(!store.exists("key1"));
        assert!(store.is_empty());
    }

    #[test]
    fn store_without_bloom_filter() {
        let store = TensorStore::new();
        assert!(!store.has_bloom_filter());

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
        assert!(!store.exists("missing"));
    }

    #[test]
    fn bloom_filter_concurrent_access() {
        let filter = Arc::new(BloomFilter::new(10000, 0.01));
        let mut handles = vec![];

        // Multiple threads adding keys
        for t in 0..4 {
            let filter = Arc::clone(&filter);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    filter.add(&format!("thread{}:key{}", t, i));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All keys should be found
        for t in 0..4 {
            for i in 0..100 {
                assert!(filter.might_contain(&format!("thread{}:key{}", t, i)));
            }
        }
    }

    // TensorData entity helper tests

    #[test]
    fn tensor_data_entity_type_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.entity_type().is_none());

        tensor.set_entity_type("node");
        assert_eq!(tensor.entity_type(), Some("node"));

        tensor.set_entity_id(42);
        assert_eq!(tensor.entity_id(), Some(42));

        tensor.set_label("Person");
        assert_eq!(tensor.label(), Some("Person"));
    }

    #[test]
    fn tensor_data_embedding_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.embedding().is_none());
        assert!(!tensor.has_embedding());

        tensor.set_embedding(vec![0.1, 0.2, 0.3]);
        assert!(tensor.has_embedding());
        assert_eq!(tensor.embedding(), Some(&vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn tensor_data_edge_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.outgoing_edges().is_none());
        assert!(tensor.incoming_edges().is_none());
        assert!(!tensor.has_edges());

        tensor.set_outgoing_edges(vec!["edge:1".to_string()]);
        tensor.set_incoming_edges(vec!["edge:2".to_string()]);
        assert!(tensor.has_edges());

        assert_eq!(tensor.outgoing_edges(), Some(&vec!["edge:1".to_string()]));
        assert_eq!(tensor.incoming_edges(), Some(&vec!["edge:2".to_string()]));
    }

    #[test]
    fn tensor_data_add_edges_deduplicates() {
        let mut tensor = TensorData::new();

        tensor.add_outgoing_edge("edge:1".to_string());
        tensor.add_outgoing_edge("edge:1".to_string());
        tensor.add_outgoing_edge("edge:2".to_string());

        let edges = tensor.outgoing_edges().unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&"edge:1".to_string()));
        assert!(edges.contains(&"edge:2".to_string()));
    }

    #[test]
    fn tensor_data_user_fields() {
        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        tensor.set_entity_type("user");
        tensor.set_entity_id(1);

        let user_fields: Vec<_> = tensor.user_fields().collect();
        assert_eq!(user_fields.len(), 2);

        let all_fields: Vec<_> = tensor.iter().collect();
        assert_eq!(all_fields.len(), 4);
    }

    // EntityStore tests

    #[test]
    fn entity_store_basic_operations() {
        let store = EntityStore::new();
        assert!(store.is_empty());

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", data).unwrap();

        assert!(store.exists("user:1"));
        assert_eq!(store.len(), 1);

        let retrieved = store.get("user:1").unwrap();
        match retrieved.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }

        store.delete("user:1").unwrap();
        assert!(!store.exists("user:1"));
    }

    #[test]
    fn entity_store_entity_key() {
        assert_eq!(EntityStore::entity_key("user", 42), "user:42");
        assert_eq!(EntityStore::entity_key("post", 1), "post:1");
    }

    #[test]
    fn entity_store_parse_key() {
        assert_eq!(EntityStore::parse_key("user:42"), Some(("user", 42)));
        assert_eq!(EntityStore::parse_key("post:1"), Some(("post", 1)));
        assert_eq!(EntityStore::parse_key("invalid"), None);
        assert_eq!(EntityStore::parse_key("user:abc"), None);
    }

    #[test]
    fn entity_store_get_or_create() {
        let store = EntityStore::new();

        let data = store.get_or_create("user:1");
        assert!(data.is_empty());

        let mut existing = TensorData::new();
        existing.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Bob".into())),
        );
        store.put("user:2", existing).unwrap();

        let data2 = store.get_or_create("user:2");
        assert!(!data2.is_empty());
    }

    #[test]
    fn entity_store_update() {
        let store = EntityStore::new();

        store
            .update("user:1", |data| {
                data.set(
                    "name",
                    TensorValue::Scalar(ScalarValue::String("Alice".into())),
                );
            })
            .unwrap();

        store
            .update("user:1", |data| {
                data.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
            })
            .unwrap();

        let data = store.get("user:1").unwrap();
        assert!(data.has("name"));
        assert!(data.has("age"));
    }

    #[test]
    fn entity_store_embeddings() {
        let store = EntityStore::new();

        store.set_embedding("user:1", vec![0.1, 0.2, 0.3]).unwrap();
        store.set_embedding("user:2", vec![0.4, 0.5, 0.6]).unwrap();

        assert_eq!(store.get_embedding("user:1"), Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(store.get_embedding("user:3"), None);

        let with_embeddings = store.scan_with_embeddings();
        assert_eq!(with_embeddings.len(), 2);
    }

    #[test]
    fn entity_store_edges() {
        let store = EntityStore::new();

        let mut user1 = TensorData::new();
        user1.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", user1).unwrap();

        let mut user2 = TensorData::new();
        user2.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Bob".into())),
        );
        store.put("user:2", user2).unwrap();

        store.add_edge("user:1", "user:2", "edge:1").unwrap();

        let outgoing = store.outgoing_neighbors("user:1").unwrap();
        assert_eq!(outgoing, vec!["edge:1"]);

        let incoming = store.incoming_neighbors("user:2").unwrap();
        assert_eq!(incoming, vec!["edge:1"]);

        let with_edges = store.scan_with_edges();
        assert_eq!(with_edges.len(), 2);
    }

    #[test]
    fn entity_store_scan_type() {
        let store = EntityStore::new();

        store.put("user:1", TensorData::new()).unwrap();
        store.put("user:2", TensorData::new()).unwrap();
        store.put("post:1", TensorData::new()).unwrap();

        let users = store.scan_type("user");
        assert_eq!(users.len(), 2);

        let posts = store.scan_type("post");
        assert_eq!(posts.len(), 1);

        assert_eq!(store.count_type("user"), 2);
        assert_eq!(store.count_type("post"), 1);
    }

    #[test]
    fn entity_store_with_arc() {
        let tensor_store = Arc::new(TensorStore::new());
        let store1 = EntityStore::with_arc(Arc::clone(&tensor_store));
        let store2 = EntityStore::with_arc(Arc::clone(&tensor_store));

        store1.put("shared:1", TensorData::new()).unwrap();
        assert!(store2.exists("shared:1"));
    }

    #[test]
    fn entity_store_clone() {
        let store1 = EntityStore::new();
        store1.put("key:1", TensorData::new()).unwrap();

        let store2 = store1.clone();
        assert!(store2.exists("key:1"));

        store2.put("key:2", TensorData::new()).unwrap();
        assert!(store1.exists("key:2"));
    }

    #[test]
    fn entity_store_default() {
        let store = EntityStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn entity_store_clear() {
        let store = EntityStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn entity_store_unified_entity() {
        let store = EntityStore::new();

        store
            .update("user:1", |data| {
                data.set_entity_type("user");
                data.set_entity_id(1);
                data.set(
                    "name",
                    TensorValue::Scalar(ScalarValue::String("Alice".into())),
                );
                data.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
            })
            .unwrap();

        store.set_embedding("user:1", vec![0.1, 0.2, 0.3]).unwrap();

        store
            .add_edge("user:1", "user:2", "edge:follows:1")
            .unwrap();

        let data = store.get("user:1").unwrap();
        assert_eq!(data.entity_type(), Some("user"));
        assert_eq!(data.entity_id(), Some(1));
        assert!(data.has_embedding());
        assert!(data.has_edges());

        let user_fields: Vec<_> = data.user_fields().collect();
        assert_eq!(user_fields.len(), 2);
    }

    // Snapshot tests

    #[test]
    fn snapshot_save_and_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_basic.bin");

        // Create and populate store
        let store = TensorStore::new();
        let mut tensor1 = TensorData::new();
        tensor1.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor1.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        store.put("user:1", tensor1).unwrap();

        let mut tensor2 = TensorData::new();
        tensor2.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
        store.put("user:2", tensor2).unwrap();

        // Save snapshot
        store.save_snapshot(&path).unwrap();

        // Load into new store
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        // Verify data
        assert_eq!(loaded.len(), 2);
        assert!(loaded.exists("user:1"));
        assert!(loaded.exists("user:2"));

        let user1 = loaded.get("user:1").unwrap();
        match user1.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }

        let user2 = loaded.get("user:2").unwrap();
        match user2.get("embedding") {
            Some(TensorValue::Vector(v)) => assert_eq!(v, &vec![0.1, 0.2, 0.3]),
            _ => panic!("expected vector"),
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_empty_store() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_empty.bin");

        let store = TensorStore::new();
        store.save_snapshot(&path).unwrap();

        let loaded = TensorStore::load_snapshot(&path).unwrap();
        assert!(loaded.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_all_scalar_types() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_scalars.bin");

        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("null", TensorValue::Scalar(ScalarValue::Null));
        tensor.set("bool", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("int", TensorValue::Scalar(ScalarValue::Int(-42)));
        tensor.set("float", TensorValue::Scalar(ScalarValue::Float(3.14)));
        tensor.set(
            "string",
            TensorValue::Scalar(ScalarValue::String("hello".into())),
        );
        tensor.set(
            "bytes",
            TensorValue::Scalar(ScalarValue::Bytes(vec![0xFF, 0x00, 0xAB])),
        );
        store.put("test", tensor).unwrap();

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        let t = loaded.get("test").unwrap();
        assert_eq!(t.get("null"), Some(&TensorValue::Scalar(ScalarValue::Null)));
        assert_eq!(
            t.get("bool"),
            Some(&TensorValue::Scalar(ScalarValue::Bool(true)))
        );
        assert_eq!(
            t.get("int"),
            Some(&TensorValue::Scalar(ScalarValue::Int(-42)))
        );
        assert_eq!(
            t.get("float"),
            Some(&TensorValue::Scalar(ScalarValue::Float(3.14)))
        );
        assert_eq!(
            t.get("string"),
            Some(&TensorValue::Scalar(ScalarValue::String("hello".into())))
        );
        assert_eq!(
            t.get("bytes"),
            Some(&TensorValue::Scalar(ScalarValue::Bytes(vec![
                0xFF, 0x00, 0xAB
            ])))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_pointers() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_pointers.bin");

        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("single", TensorValue::Pointer("ref:1".into()));
        tensor.set(
            "multi",
            TensorValue::Pointers(vec!["ref:2".into(), "ref:3".into()]),
        );
        store.put("test", tensor).unwrap();

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        let t = loaded.get("test").unwrap();
        assert_eq!(t.get("single"), Some(&TensorValue::Pointer("ref:1".into())));
        assert_eq!(
            t.get("multi"),
            Some(&TensorValue::Pointers(vec!["ref:2".into(), "ref:3".into()]))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_large_dataset() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_large.bin");

        let store = TensorStore::new();
        for i in 0..1000 {
            let mut tensor = TensorData::new();
            tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
            tensor.set("embedding", TensorValue::Vector(vec![i as f32; 128]));
            store.put(format!("entity:{}", i), tensor).unwrap();
        }

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        assert_eq!(loaded.len(), 1000);

        // Verify a few entries
        for i in [0, 500, 999] {
            let t = loaded.get(&format!("entity:{}", i)).unwrap();
            match t.get("id") {
                Some(TensorValue::Scalar(ScalarValue::Int(id))) => assert_eq!(*id, i),
                _ => panic!("expected int"),
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_with_bloom_filter() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_bloom.bin");

        let store = TensorStore::new();
        for i in 0..100 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        store.save_snapshot(&path).unwrap();

        // Load with bloom filter
        let loaded = TensorStore::load_snapshot_with_bloom_filter(&path, 200, 0.01).unwrap();

        assert!(loaded.has_bloom_filter());
        assert_eq!(loaded.len(), 100);

        // Bloom filter should work
        assert!(loaded.exists("key:50"));
        assert!(!loaded.exists("nonexistent"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_load_nonexistent_file() {
        let result = TensorStore::load_snapshot("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_error_display() {
        let io_err = SnapshotError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(format!("{}", io_err).contains("I/O error"));

        let ser_err = SnapshotError::SerializationError("bad data".into());
        assert!(format!("{}", ser_err).contains("Serialization error"));
    }

    #[test]
    fn snapshot_error_source() {
        use std::error::Error;

        let io_err =
            SnapshotError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(io_err.source().is_some());

        let ser_err = SnapshotError::SerializationError("test".into());
        assert!(ser_err.source().is_none());
    }

    #[test]
    fn snapshot_compressed_roundtrip() {
        let store = TensorStore::new();

        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("test".into())),
        );
        tensor.set("count", TensorValue::Scalar(ScalarValue::Int(42)));
        tensor.set("vector", TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4]));
        store.put("emb:test1", tensor).unwrap();

        let mut tensor2 = TensorData::new();
        tensor2.set("value", TensorValue::Scalar(ScalarValue::Float(3.14)));
        store.put("other", tensor2).unwrap();

        let config = tensor_compress::CompressionConfig {
            vector_quantization: Some(tensor_compress::QuantMode::Int8),
            delta_encoding: true,
            rle_encoding: true,
        };

        let temp = std::env::temp_dir().join("test_compressed.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        assert_eq!(loaded.len(), 2);

        let t1 = loaded.get("emb:test1").unwrap();
        assert!(t1.has("name"));
        assert!(t1.has("vector"));

        let t2 = loaded.get("other").unwrap();
        assert!(t2.has("value"));

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn snapshot_compressed_with_quantization() {
        let store = TensorStore::new();

        let embedding: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) - 0.5).collect();
        let mut tensor = TensorData::new();
        tensor.set("_embedding", TensorValue::Vector(embedding.clone()));
        store.put("emb:doc1", tensor).unwrap();

        let config = tensor_compress::CompressionConfig {
            vector_quantization: Some(tensor_compress::QuantMode::Int8),
            ..Default::default()
        };

        let temp = std::env::temp_dir().join("test_quant.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let file_size = std::fs::metadata(&temp).unwrap().len();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        let restored = loaded.get("emb:doc1").unwrap();
        let restored_vec = restored.get("_embedding").unwrap();

        if let TensorValue::Vector(v) = restored_vec {
            assert_eq!(v.len(), 768);
            for (orig, rest) in embedding.iter().zip(v) {
                assert!((orig - rest).abs() < 0.02, "Quantization error too large");
            }
        } else {
            panic!("Expected vector");
        }

        let uncompressed_size = 768 * 4;
        assert!(
            file_size < uncompressed_size as u64,
            "Compressed file should be smaller"
        );

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn snapshot_compressed_empty_store() {
        let store = TensorStore::new();
        let config = tensor_compress::CompressionConfig::default();

        let temp = std::env::temp_dir().join("test_empty_compressed.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_file(&temp).ok();
    }
}
