use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;

/// Represents different types of values a tensor can hold
#[derive(Debug, Clone, PartialEq)]
pub enum TensorValue {
    /// Scalar values (properties): integers, floats, strings, booleans
    Scalar(ScalarValue),
    /// Vector values (embeddings): f32 arrays for similarity search
    Vector(Vec<f32>),
    /// Pointer to another tensor (relationships)
    Pointer(String),
    /// List of pointers (multiple relationships)
    Pointers(Vec<String>),
}

/// Scalar value types
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

/// An entity that can hold scalar properties, vector embeddings, and pointers to other tensors.
#[derive(Debug, Clone, Default)]
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
}

pub type Result<T> = std::result::Result<T, TensorStoreError>;

#[derive(Debug, Clone, PartialEq)]
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
pub struct TensorStore {
    data: DashMap<String, TensorData>,
}

impl TensorStore {
    const PARALLEL_THRESHOLD: usize = 1000;

    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
        }
    }

    /// Create a store with a specific capacity hint for better initial allocation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: DashMap::with_capacity(capacity),
        }
    }

    pub fn put(&self, key: impl Into<String>, tensor: TensorData) -> Result<()> {
        self.data.insert(key.into(), tensor);
        Ok(())
    }

    /// Returns cloned data to ensure thread safety.
    pub fn get(&self, key: &str) -> Result<TensorData> {
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

    pub fn exists(&self, key: &str) -> bool {
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
}

impl Default for TensorStore {
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
}
