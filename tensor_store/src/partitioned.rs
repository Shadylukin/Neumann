//! Partition-aware store wrapper for distributed operations.
//!
//! Provides a wrapper around TensorStore that integrates with the partitioner
//! to support distributed storage operations.

use std::sync::Arc;

use crate::{
    consistent_hash::{ConsistentHashConfig, ConsistentHashPartitioner},
    partitioner::{PartitionResult, Partitioner, PhysicalNodeId},
    slab_router::SlabRouterError,
    TensorData, TensorStore,
};

/// Error type for partitioned store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionedError {
    /// Key belongs to a remote partition.
    RemotePartition(PartitionResult),
    /// Underlying store error.
    StoreError(String),
    /// No partitioner configured.
    NoPartitioner,
}

impl std::fmt::Display for PartitionedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RemotePartition(result) => {
                write!(
                    f,
                    "key belongs to remote partition on node {}",
                    result.primary
                )
            },
            Self::StoreError(msg) => write!(f, "store error: {}", msg),
            Self::NoPartitioner => write!(f, "no partitioner configured"),
        }
    }
}

impl std::error::Error for PartitionedError {}

impl From<SlabRouterError> for PartitionedError {
    fn from(e: SlabRouterError) -> Self {
        Self::StoreError(e.to_string())
    }
}

/// Result type for partitioned operations.
pub type PartitionedResult<T> = std::result::Result<T, PartitionedError>;

/// Result of a partitioned get operation.
#[derive(Debug, Clone)]
pub struct PartitionedGet {
    /// The data if found.
    pub data: Option<TensorData>,
    /// Partition information.
    pub partition: PartitionResult,
}

/// Result of a partitioned put operation.
#[derive(Debug, Clone)]
pub struct PartitionedPut {
    /// Whether the put succeeded.
    pub success: bool,
    /// Partition information.
    pub partition: PartitionResult,
}

/// Partition-aware wrapper around TensorStore.
///
/// Provides methods that:
/// - Check partition ownership before local operations
/// - Return partition information with each operation
/// - Support both local-only and partition-aware modes
pub struct PartitionedStore {
    /// Underlying store.
    store: TensorStore,
    /// Partitioner (optional for single-node mode).
    partitioner: Option<Arc<dyn Partitioner>>,
}

impl std::fmt::Debug for PartitionedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartitionedStore")
            .field("has_partitioner", &self.partitioner.is_some())
            .field("store_len", &self.store.len())
            .finish()
    }
}

impl PartitionedStore {
    /// Create a new partitioned store without a partitioner (single-node mode).
    pub fn new(store: TensorStore) -> Self {
        Self {
            store,
            partitioner: None,
        }
    }

    /// Create a partitioned store with a consistent hash partitioner.
    pub fn with_consistent_hash(store: TensorStore, config: ConsistentHashConfig) -> Self {
        Self {
            store,
            partitioner: Some(Arc::new(ConsistentHashPartitioner::new(config))),
        }
    }

    /// Create a partitioned store with a custom partitioner.
    pub fn with_partitioner(store: TensorStore, partitioner: Arc<dyn Partitioner>) -> Self {
        Self {
            store,
            partitioner: Some(partitioner),
        }
    }

    pub fn partition_for(&self, key: &str) -> Option<PartitionResult> {
        self.partitioner.as_ref().map(|p| p.partition(key))
    }

    /// Check if a key belongs to the local node.
    pub fn is_local(&self, key: &str) -> bool {
        match &self.partitioner {
            Some(p) => p.is_local(key),
            None => true, // Single-node mode: all keys are local
        }
    }

    /// Get partition info and data for a key.
    ///
    /// Returns both the data (if found) and partition information.
    /// Does NOT check partition ownership - use `get_local` for that.
    pub fn get_partitioned(&self, key: &str) -> PartitionedGet {
        let partition = self
            .partitioner
            .as_ref()
            .map(|p| p.partition(key))
            .unwrap_or_else(|| PartitionResult::local("local", 0));

        let data = self.store.get(key).ok();

        PartitionedGet { data, partition }
    }

    /// Get data only if the key belongs to this node.
    ///
    /// Returns `Err(RemotePartition)` if the key belongs to another node.
    pub fn get_local(&self, key: &str) -> PartitionedResult<Option<TensorData>> {
        if let Some(ref p) = self.partitioner {
            let partition = p.partition(key);
            if !partition.is_local {
                return Err(PartitionedError::RemotePartition(partition));
            }
        }

        Ok(self.store.get(key).ok())
    }

    /// Put data with partition information.
    ///
    /// Returns partition information. Does NOT check partition ownership.
    pub fn put_partitioned(
        &self,
        key: impl Into<String>,
        data: TensorData,
    ) -> PartitionedResult<PartitionedPut> {
        let key = key.into();
        let partition = self
            .partitioner
            .as_ref()
            .map(|p| p.partition(&key))
            .unwrap_or_else(|| PartitionResult::local("local", 0));

        let success = self.store.put(&key, data).is_ok();

        Ok(PartitionedPut { success, partition })
    }

    /// Put data only if the key belongs to this node.
    ///
    /// Returns `Err(RemotePartition)` if the key belongs to another node.
    pub fn put_local(&self, key: impl Into<String>, data: TensorData) -> PartitionedResult<()> {
        let key = key.into();

        if let Some(ref p) = self.partitioner {
            let partition = p.partition(&key);
            if !partition.is_local {
                return Err(PartitionedError::RemotePartition(partition));
            }
        }

        self.store
            .put(&key, data)
            .map_err(|e| PartitionedError::StoreError(e.to_string()))
    }

    /// Delete data only if the key belongs to this node.
    ///
    /// Returns `Err(RemotePartition)` if the key belongs to another node.
    pub fn delete_local(&self, key: &str) -> PartitionedResult<()> {
        if let Some(ref p) = self.partitioner {
            let partition = p.partition(key);
            if !partition.is_local {
                return Err(PartitionedError::RemotePartition(partition));
            }
        }

        self.store
            .delete(key)
            .map_err(|e| PartitionedError::StoreError(e.to_string()))
    }

    /// Check if a key exists locally.
    ///
    /// Returns false for remote keys without checking the store.
    pub fn exists_local(&self, key: &str) -> bool {
        if let Some(ref p) = self.partitioner {
            if !p.is_local(key) {
                return false;
            }
        }

        self.store.exists(key)
    }

    /// Get all local keys with a given prefix.
    ///
    /// Note: This scans the local store and filters by partition ownership.
    pub fn scan_local(&self, prefix: &str) -> Vec<String> {
        let keys = self.store.scan(prefix);

        match &self.partitioner {
            Some(p) => keys.into_iter().filter(|k| p.is_local(k)).collect(),
            None => keys,
        }
    }

    pub fn local_count(&self) -> usize {
        match &self.partitioner {
            Some(p) => self
                .store
                .scan("")
                .into_iter()
                .filter(|k| p.is_local(k))
                .count(),
            None => self.store.len(),
        }
    }

    /// Get total keys in the local store (including remote keys during migration).
    pub fn total_count(&self) -> usize {
        self.store.len()
    }

    /// Access the underlying store directly.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Access the partitioner.
    pub fn partitioner(&self) -> Option<&Arc<dyn Partitioner>> {
        self.partitioner.as_ref()
    }

    pub fn local_node(&self) -> Option<&PhysicalNodeId> {
        self.partitioner.as_ref().map(|p| p.local_node())
    }

    pub fn nodes(&self) -> Vec<PhysicalNodeId> {
        self.partitioner
            .as_ref()
            .map(|p| p.nodes())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consistent_hash::ConsistentHashConfig, ScalarValue, TensorValue};

    fn make_tensor(value: i64) -> TensorData {
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(value)));
        data
    }

    #[test]
    fn test_single_node_mode() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        // All keys should be local in single-node mode
        assert!(partitioned.is_local("any_key"));
        assert!(partitioned.partition_for("any_key").is_none());
    }

    #[test]
    fn test_put_get_local() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        let tensor = make_tensor(42);
        partitioned.put_local("key1", tensor.clone()).unwrap();

        let result = partitioned.get_local("key1").unwrap();
        assert!(result.is_some());

        let data = result.unwrap();
        assert_eq!(
            data.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(42)))
        );
    }

    #[test]
    fn test_partitioned_get() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        let tensor = make_tensor(100);
        partitioned.put_local("test_key", tensor).unwrap();

        let result = partitioned.get_partitioned("test_key");
        assert!(result.data.is_some());
        assert!(result.partition.is_local);
    }

    #[test]
    fn test_partitioned_put() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        let tensor = make_tensor(200);
        let result = partitioned.put_partitioned("key", tensor).unwrap();

        assert!(result.success);
        assert!(result.partition.is_local);
    }

    #[test]
    fn test_delete_local() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        let tensor = make_tensor(42);
        partitioned.put_local("to_delete", tensor).unwrap();
        assert!(partitioned.exists_local("to_delete"));

        partitioned.delete_local("to_delete").unwrap();
        assert!(!partitioned.exists_local("to_delete"));
    }

    #[test]
    fn test_scan_local() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        for i in 0..5 {
            let tensor = make_tensor(i);
            partitioned
                .put_local(format!("prefix:{}", i), tensor)
                .unwrap();
        }

        let keys = partitioned.scan_local("prefix:");
        assert_eq!(keys.len(), 5);
    }

    #[test]
    fn test_local_count() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        for i in 0..10 {
            let tensor = make_tensor(i);
            partitioned.put_local(format!("key{}", i), tensor).unwrap();
        }

        assert_eq!(partitioned.local_count(), 10);
        assert_eq!(partitioned.total_count(), 10);
    }

    #[test]
    fn test_with_consistent_hash() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        // Create partitioner manually so we can add the node
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // With only local node, all keys should be local
        assert!(partitioned.is_local("any_key"));
    }

    #[test]
    fn test_remote_key_detection() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Some keys should be remote
        let mut has_remote = false;
        for i in 0..100 {
            if !partitioned.is_local(&format!("key{}", i)) {
                has_remote = true;
                break;
            }
        }
        assert!(has_remote, "Should have some remote keys");
    }

    #[test]
    fn test_put_local_rejects_remote() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Find a remote key
        let mut remote_key = None;
        for i in 0..100 {
            let key = format!("key{}", i);
            if !partitioned.is_local(&key) {
                remote_key = Some(key);
                break;
            }
        }

        if let Some(key) = remote_key {
            let tensor = make_tensor(42);
            let result = partitioned.put_local(&key, tensor);
            assert!(matches!(result, Err(PartitionedError::RemotePartition(_))));
        }
    }

    #[test]
    fn test_get_local_rejects_remote() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Find a remote key
        let mut remote_key = None;
        for i in 0..100 {
            let key = format!("key{}", i);
            if !partitioned.is_local(&key) {
                remote_key = Some(key);
                break;
            }
        }

        if let Some(key) = remote_key {
            let result = partitioned.get_local(&key);
            assert!(matches!(result, Err(PartitionedError::RemotePartition(_))));
        }
    }

    #[test]
    fn test_delete_local_rejects_remote() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Find a remote key
        let mut remote_key = None;
        for i in 0..100 {
            let key = format!("key{}", i);
            if !partitioned.is_local(&key) {
                remote_key = Some(key);
                break;
            }
        }

        if let Some(key) = remote_key {
            let result = partitioned.delete_local(&key);
            assert!(matches!(result, Err(PartitionedError::RemotePartition(_))));
        }
    }

    #[test]
    fn test_exists_local_returns_false_for_remote() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Find a remote key
        for i in 0..100 {
            let key = format!("key{}", i);
            if !partitioned.is_local(&key) {
                // Should return false for remote keys (not in local store)
                assert!(!partitioned.exists_local(&key));
                return;
            }
        }
    }

    #[test]
    fn test_local_node() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("my_node");
        let partitioned = PartitionedStore::with_consistent_hash(store, config);

        assert_eq!(partitioned.local_node(), Some(&"my_node".to_string()));
    }

    #[test]
    fn test_nodes() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        let nodes = partitioned.nodes();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&"node1".to_string()));
        assert!(nodes.contains(&"node2".to_string()));
    }

    #[test]
    fn test_store_accessor() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        let tensor = make_tensor(42);
        partitioned.put_local("test", tensor).unwrap();

        // Access underlying store directly
        assert!(partitioned.store().exists("test"));
    }

    #[test]
    fn test_partitioner_accessor() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);
        assert!(partitioned.partitioner().is_none());

        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1");
        let partitioned = PartitionedStore::with_consistent_hash(store, config);
        assert!(partitioned.partitioner().is_some());
    }

    #[test]
    fn test_error_display() {
        let remote_err = PartitionedError::RemotePartition(PartitionResult::remote("node2", 42));
        assert!(remote_err.to_string().contains("remote partition"));
        assert!(remote_err.to_string().contains("node2"));

        let store_err = PartitionedError::StoreError("test error".to_string());
        assert!(store_err.to_string().contains("store error"));

        let no_part_err = PartitionedError::NoPartitioner;
        assert!(no_part_err.to_string().contains("no partitioner"));
    }

    #[test]
    fn test_error_debug() {
        let err = PartitionedError::StoreError("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("StoreError"));
    }

    #[test]
    fn test_partitioned_get_debug() {
        let result = PartitionedGet {
            data: None,
            partition: PartitionResult::local("node1", 42),
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("PartitionedGet"));
    }

    #[test]
    fn test_partitioned_put_debug() {
        let result = PartitionedPut {
            success: true,
            partition: PartitionResult::local("node1", 42),
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("PartitionedPut"));
    }

    #[test]
    fn test_partitioned_get_clone() {
        let result = PartitionedGet {
            data: Some(make_tensor(42)),
            partition: PartitionResult::local("node1", 42),
        };
        let cloned = result.clone();
        assert_eq!(result.partition, cloned.partition);
    }

    #[test]
    fn test_partitioned_put_clone() {
        let result = PartitionedPut {
            success: true,
            partition: PartitionResult::local("node1", 42),
        };
        let cloned = result.clone();
        assert_eq!(result.success, cloned.success);
        assert_eq!(result.partition, cloned.partition);
    }

    #[test]
    fn test_error_equality() {
        let a = PartitionedError::NoPartitioner;
        let b = PartitionedError::NoPartitioner;
        assert_eq!(a, b);

        let c = PartitionedError::StoreError("test".to_string());
        let d = PartitionedError::StoreError("test".to_string());
        assert_eq!(c, d);
    }
}
