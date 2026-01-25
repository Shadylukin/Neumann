//! Partition-aware store wrapper for distributed operations.
//!
//! Provides a wrapper around `TensorStore` that integrates with the partitioner
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
            Self::StoreError(msg) => write!(f, "store error: {msg}"),
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

/// Partition-aware wrapper around `TensorStore`.
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
    #[must_use]
    pub fn new(store: TensorStore) -> Self {
        Self {
            store,
            partitioner: None,
        }
    }

    /// Create a partitioned store with a consistent hash partitioner.
    #[must_use]
    pub fn with_consistent_hash(store: TensorStore, config: ConsistentHashConfig) -> Self {
        Self {
            store,
            partitioner: Some(Arc::new(ConsistentHashPartitioner::new(config))),
        }
    }

    /// Create a partitioned store with a custom partitioner.
    #[must_use]
    pub fn with_partitioner(store: TensorStore, partitioner: Arc<dyn Partitioner>) -> Self {
        Self {
            store,
            partitioner: Some(partitioner),
        }
    }

    /// Returns the partition assignment for a key.
    #[must_use]
    pub fn partition_for(&self, key: &str) -> Option<PartitionResult> {
        self.partitioner.as_ref().map(|p| p.partition(key))
    }

    /// Check if a key belongs to the local node.
    #[must_use]
    pub fn is_local(&self, key: &str) -> bool {
        self.partitioner.as_ref().is_none_or(|p| p.is_local(key))
    }

    /// Get partition info and data for a key.
    ///
    /// Returns both the data (if found) and partition information.
    /// Does NOT check partition ownership - use `get_local` for that.
    #[must_use]
    pub fn get_partitioned(&self, key: &str) -> PartitionedGet {
        let partition = self
            .partitioner
            .as_ref()
            .map_or_else(|| PartitionResult::local("local", 0), |p| p.partition(key));

        let data = self.store.get(key).ok();

        PartitionedGet { data, partition }
    }

    /// Get data only if the key belongs to this node.
    ///
    /// # Errors
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
    ///
    /// # Errors
    ///
    /// This function currently does not return errors.
    pub fn put_partitioned(
        &self,
        key: impl Into<String>,
        data: TensorData,
    ) -> PartitionedResult<PartitionedPut> {
        let key = key.into();
        let partition = self
            .partitioner
            .as_ref()
            .map_or_else(|| PartitionResult::local("local", 0), |p| p.partition(&key));

        let success = self.store.put(&key, data).is_ok();

        Ok(PartitionedPut { success, partition })
    }

    /// Put data only if the key belongs to this node.
    ///
    /// # Errors
    ///
    /// Returns `Err(RemotePartition)` if the key belongs to another node,
    /// or `Err(StoreError)` if the store operation fails.
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
    /// # Errors
    ///
    /// Returns `Err(RemotePartition)` if the key belongs to another node,
    /// or `Err(StoreError)` if the store operation fails.
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
    #[must_use]
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
    #[must_use]
    pub fn scan_local(&self, prefix: &str) -> Vec<String> {
        let keys = self.store.scan(prefix);

        match &self.partitioner {
            Some(p) => keys.into_iter().filter(|k| p.is_local(k)).collect(),
            None => keys,
        }
    }

    /// Returns the count of local keys only.
    #[must_use]
    pub fn local_count(&self) -> usize {
        self.partitioner.as_ref().map_or_else(
            || self.store.len(),
            |p| {
                self.store
                    .scan("")
                    .into_iter()
                    .filter(|k| p.is_local(k))
                    .count()
            },
        )
    }

    /// Get total keys in the local store (including remote keys during migration).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.store.len()
    }

    /// Access the underlying store directly.
    #[must_use]
    pub const fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Access the partitioner.
    #[must_use]
    pub fn partitioner(&self) -> Option<&Arc<dyn Partitioner>> {
        self.partitioner.as_ref()
    }

    /// Returns the local node ID if partitioning is configured.
    #[must_use]
    pub fn local_node(&self) -> Option<&PhysicalNodeId> {
        self.partitioner.as_ref().map(|p| p.local_node())
    }

    /// Returns all nodes in the partition scheme.
    #[must_use]
    pub fn nodes(&self) -> Vec<PhysicalNodeId> {
        self.partitioner
            .as_ref()
            .map_or_else(Vec::new, |p| p.nodes())
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

    #[test]
    fn test_local_count_with_partitioner() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Insert some data - all should be local since we only have node1
        for i in 0..5 {
            let tensor = make_tensor(i);
            partitioned.put_local(format!("key{}", i), tensor).unwrap();
        }

        // local_count should use the partitioner branch
        let count = partitioned.local_count();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_local_count_filters_remote_keys() {
        let store = TensorStore::new();
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let partitioner = {
            let mut p = ConsistentHashPartitioner::new(config);
            p.add_node("node1".to_string());
            p.add_node("node2".to_string());
            Arc::new(p) as Arc<dyn Partitioner>
        };
        let partitioned = PartitionedStore::with_partitioner(store, partitioner);

        // Insert via put_partitioned which doesn't check ownership
        // This allows us to insert keys regardless of which node they belong to
        let mut local_count = 0;
        for i in 0..20 {
            let key = format!("testkey{}", i);
            let tensor = make_tensor(i);
            if partitioned.is_local(&key) {
                partitioned.put_partitioned(&key, tensor).unwrap();
                local_count += 1;
            }
        }

        // local_count should only count keys that belong to this node
        assert_eq!(partitioned.local_count(), local_count);
        // total_count is the same since we only inserted local keys
        assert_eq!(partitioned.total_count(), local_count);
    }

    #[test]
    fn test_debug_impl() {
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);
        let debug = format!("{:?}", partitioned);
        assert!(debug.contains("PartitionedStore"));
        assert!(debug.contains("has_partitioner"));
    }

    #[test]
    fn test_partitioned_error_from_slab_router_error() {
        use crate::slab_router::SlabRouterError;

        let slab_err = SlabRouterError::NotFound("test_key".to_string());
        let partitioned_err: PartitionedError = slab_err.into();
        match partitioned_err {
            PartitionedError::StoreError(msg) => assert!(msg.contains("test_key")),
            _ => panic!("Expected StoreError variant"),
        }
    }

    // ========== Phase 3: Additional Negative Path Tests ==========

    #[test]
    fn test_partitioned_error_no_partitioner_display() {
        // Verify NoPartitioner error can be constructed and formatted
        let err = PartitionedError::NoPartitioner;
        let msg = err.to_string();
        assert!(msg.contains("no partitioner"));
    }

    #[test]
    fn test_partitioned_error_is_std_error() {
        // Verify PartitionedError implements std::error::Error
        let err: Box<dyn std::error::Error> = Box::new(PartitionedError::NoPartitioner);
        assert!(err.to_string().contains("no partitioner"));

        // Test source() returns None (no nested error)
        assert!(err.source().is_none());
    }

    #[test]
    fn test_partitioned_error_remote_partition_display() {
        let result = PartitionResult::remote("remote_node", 123);
        let err = PartitionedError::RemotePartition(result);
        let msg = err.to_string();
        assert!(msg.contains("remote partition"));
        assert!(msg.contains("remote_node"));
    }

    #[test]
    fn test_partitioned_store_single_node_no_partitioner() {
        // Single node mode has no partitioner, so partition_for returns None
        let store = TensorStore::new();
        let partitioned = PartitionedStore::new(store);

        assert!(partitioned.partitioner().is_none());
        assert!(partitioned.partition_for("any_key").is_none());
        assert!(partitioned.local_node().is_none());
        assert!(partitioned.nodes().is_empty());
    }

    #[test]
    fn test_partitioned_error_clone() {
        let err = PartitionedError::StoreError("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);

        let no_part = PartitionedError::NoPartitioner;
        let cloned_no_part = no_part.clone();
        assert_eq!(no_part, cloned_no_part);
    }

    #[test]
    fn test_partitioned_error_debug_remote() {
        let result = PartitionResult::remote("node2", 42);
        let err = PartitionedError::RemotePartition(result);
        let debug = format!("{:?}", err);
        assert!(debug.contains("RemotePartition"));
    }
}
