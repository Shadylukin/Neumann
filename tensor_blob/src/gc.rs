use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tensor_store::TensorStore;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::interval;

use crate::config::GcConfig;
use crate::error::Result;
use crate::metadata::GcStats;
use crate::streaming::{get_int, get_pointers};

/// Background garbage collector for orphaned chunks.
pub struct GarbageCollector {
    store: TensorStore,
    config: GcConfig,
    shutdown_tx: broadcast::Sender<()>,
}

impl GarbageCollector {
    pub fn new(store: TensorStore, config: GcConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            store,
            config,
            shutdown_tx,
        }
    }

    /// Start background GC task. Returns a handle to the task.
    pub fn start(self: Arc<Self>) -> JoinHandle<()> {
        let gc = Arc::clone(&self);
        tokio::spawn(async move {
            gc.run().await;
        })
    }

    /// Get a shutdown sender for graceful termination.
    pub fn shutdown_sender(&self) -> broadcast::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Send shutdown signal.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    async fn run(&self) {
        let mut interval = interval(self.config.check_interval);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let _ = self.gc_cycle().await;
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    }

    /// Run a single GC cycle, processing up to batch_size chunks.
    pub async fn gc_cycle(&self) -> GcStats {
        let mut deleted = 0;
        let mut freed_bytes = 0;

        let now = current_timestamp();
        let min_created = now.saturating_sub(self.config.min_age.as_secs());

        // Find chunks with zero refs
        let chunk_keys = self.store.scan("_blob:chunk:");

        for chunk_key in chunk_keys.into_iter().take(self.config.batch_size) {
            if let Ok(tensor) = self.store.get(&chunk_key) {
                let refs = get_int(&tensor, "_refs").unwrap_or(0);
                let created = get_int(&tensor, "_created").unwrap_or(0) as u64;

                // Zero refs and old enough
                if refs == 0 && created < min_created {
                    let size = get_int(&tensor, "_size").unwrap_or(0) as usize;

                    if self.store.delete(&chunk_key).is_ok() {
                        deleted += 1;
                        freed_bytes += size;
                    }
                }
            }
        }

        GcStats {
            deleted,
            freed_bytes,
        }
    }

    /// Full GC: recount all references from scratch.
    pub async fn full_gc(&self) -> Result<GcStats> {
        // 1. Build reference set from all artifacts
        let mut referenced: HashSet<String> = HashSet::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(chunks) = get_pointers(&tensor, "_chunks") {
                    referenced.extend(chunks);
                }
            }
        }

        // 2. Delete unreferenced chunks
        let mut deleted = 0;
        let mut freed_bytes = 0;

        for chunk_key in self.store.scan("_blob:chunk:") {
            if !referenced.contains(&chunk_key) {
                if let Ok(tensor) = self.store.get(&chunk_key) {
                    let size = get_int(&tensor, "_size").unwrap_or(0) as usize;

                    if self.store.delete(&chunk_key).is_ok() {
                        deleted += 1;
                        freed_bytes += size;
                    }
                }
            }
        }

        Ok(GcStats {
            deleted,
            freed_bytes,
        })
    }

    /// Count orphaned chunks (chunks with zero references).
    pub fn count_orphans(&self) -> usize {
        let mut count = 0;

        for chunk_key in self.store.scan("_blob:chunk:") {
            if let Ok(tensor) = self.store.get(&chunk_key) {
                let refs = get_int(&tensor, "_refs").unwrap_or(0);
                if refs == 0 {
                    count += 1;
                }
            }
        }

        count
    }
}

/// Decrement chunk reference count. Used when deleting artifacts.
pub fn decrement_chunk_refs(store: &TensorStore, chunk_key: &str) -> Result<()> {
    if let Ok(mut tensor) = store.get(chunk_key) {
        let refs = get_int(&tensor, "_refs").unwrap_or(1);
        let new_refs = (refs - 1).max(0);
        tensor.set(
            "_refs",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(new_refs)),
        );
        store.put(chunk_key, tensor)?;
    }
    Ok(())
}

/// Increment chunk reference count. Used for deduplication.
pub fn increment_chunk_refs(store: &TensorStore, chunk_key: &str) -> Result<()> {
    if let Ok(mut tensor) = store.get(chunk_key) {
        let refs = get_int(&tensor, "_refs").unwrap_or(0);
        tensor.set(
            "_refs",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(refs + 1)),
        );
        store.put(chunk_key, tensor)?;
    }
    Ok(())
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::Chunk;
    use std::time::Duration;
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    fn create_test_store() -> TensorStore {
        TensorStore::new()
    }

    fn store_chunk(store: &TensorStore, data: &[u8], refs: i64) -> String {
        let chunk = Chunk::new(data.to_vec());
        let chunk_key = chunk.key();

        let mut tensor = TensorData::new();
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("blob_chunk".to_string())),
        );
        tensor.set(
            "_data",
            TensorValue::Scalar(ScalarValue::Bytes(data.to_vec())),
        );
        tensor.set(
            "_size",
            TensorValue::Scalar(ScalarValue::Int(data.len() as i64)),
        );
        tensor.set("_refs", TensorValue::Scalar(ScalarValue::Int(refs)));
        tensor.set("_created", TensorValue::Scalar(ScalarValue::Int(0))); // Old timestamp

        store.put(&chunk_key, tensor).unwrap();
        chunk_key
    }

    fn store_artifact(store: &TensorStore, id: &str, chunks: Vec<String>) {
        let mut tensor = TensorData::new();
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("blob_artifact".to_string())),
        );
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::String(id.to_string())),
        );
        tensor.set("_chunks", TensorValue::Pointers(chunks));

        let meta_key = format!("_blob:meta:{id}");
        store.put(&meta_key, tensor).unwrap();
    }

    #[tokio::test]
    async fn test_gc_cycle_deletes_orphans() {
        let store = create_test_store();

        // Create a chunk with 0 refs (orphan)
        let orphan_key = store_chunk(&store, b"orphan data", 0);

        // Create a chunk with refs (should be kept)
        let kept_key = store_chunk(&store, b"kept data", 1);

        // Run GC
        let config = GcConfig {
            check_interval: Duration::from_secs(1),
            batch_size: 100,
            min_age: Duration::from_secs(0), // No age requirement for test
        };
        let gc = GarbageCollector::new(store.clone(), config);
        let stats = gc.gc_cycle().await;

        assert_eq!(stats.deleted, 1);
        assert!(!store.exists(&orphan_key));
        assert!(store.exists(&kept_key));
    }

    #[tokio::test]
    async fn test_gc_respects_batch_size() {
        let store = create_test_store();

        // Create 5 orphan chunks
        for i in 0..5 {
            store_chunk(&store, &[i as u8; 10], 0);
        }

        // Run GC with batch size of 2
        let config = GcConfig {
            check_interval: Duration::from_secs(1),
            batch_size: 2,
            min_age: Duration::from_secs(0),
        };
        let gc = GarbageCollector::new(store.clone(), config);
        let stats = gc.gc_cycle().await;

        // Should only delete up to 2 chunks per cycle
        assert!(stats.deleted <= 2);
    }

    #[tokio::test]
    async fn test_full_gc() {
        let store = create_test_store();

        // Create chunks
        let chunk1 = store_chunk(&store, b"chunk 1", 1);
        let chunk2 = store_chunk(&store, b"chunk 2", 1);
        let _orphan = store_chunk(&store, b"orphan", 1);

        // Create artifact referencing only chunk1 and chunk2
        store_artifact(&store, "artifact1", vec![chunk1.clone(), chunk2.clone()]);

        // Run full GC
        let config = GcConfig::default();
        let gc = GarbageCollector::new(store.clone(), config);
        let stats = gc.full_gc().await.unwrap();

        // Orphan should be deleted
        assert_eq!(stats.deleted, 1);
        assert!(store.exists(&chunk1));
        assert!(store.exists(&chunk2));
    }

    #[tokio::test]
    async fn test_count_orphans() {
        let store = create_test_store();

        store_chunk(&store, b"orphan 1", 0);
        store_chunk(&store, b"orphan 2", 0);
        store_chunk(&store, b"referenced", 1);

        let config = GcConfig::default();
        let gc = GarbageCollector::new(store, config);

        assert_eq!(gc.count_orphans(), 2);
    }

    #[test]
    fn test_decrement_chunk_refs() {
        let store = create_test_store();
        let chunk_key = store_chunk(&store, b"data", 3);

        decrement_chunk_refs(&store, &chunk_key).unwrap();

        let tensor = store.get(&chunk_key).unwrap();
        assert_eq!(get_int(&tensor, "_refs"), Some(2));
    }

    #[test]
    fn test_decrement_chunk_refs_saturating() {
        let store = create_test_store();
        let chunk_key = store_chunk(&store, b"data", 0);

        decrement_chunk_refs(&store, &chunk_key).unwrap();

        let tensor = store.get(&chunk_key).unwrap();
        assert_eq!(get_int(&tensor, "_refs"), Some(0)); // Doesn't go negative
    }

    #[test]
    fn test_increment_chunk_refs() {
        let store = create_test_store();
        let chunk_key = store_chunk(&store, b"data", 1);

        increment_chunk_refs(&store, &chunk_key).unwrap();

        let tensor = store.get(&chunk_key).unwrap();
        assert_eq!(get_int(&tensor, "_refs"), Some(2));
    }

    #[tokio::test]
    async fn test_gc_shutdown() {
        let store = create_test_store();
        let config = GcConfig {
            check_interval: Duration::from_millis(10),
            batch_size: 100,
            min_age: Duration::from_secs(0),
        };
        let gc = Arc::new(GarbageCollector::new(store, config));

        let handle = gc.clone().start();

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Shutdown
        gc.shutdown();

        // Task should complete
        let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok());
    }
}
