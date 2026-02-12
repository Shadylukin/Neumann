// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use std::collections::HashSet;

use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{
    chunker::StreamingHasher,
    error::{BlobError, Result},
    gc::decrement_chunk_refs,
    metadata::RepairStats,
    streaming::{get_bytes, get_int, get_pointers, get_string},
};

/// Verify the integrity of an artifact by checking its checksum.
///
/// # Errors
///
/// Returns an error if the artifact or its chunks are not found.
pub fn verify_artifact(store: &TensorStore, artifact_id: &str) -> Result<bool> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    let expected_checksum = get_string(&tensor, "_checksum")
        .ok_or_else(|| BlobError::NotFound(format!("checksum for {artifact_id}")))?;

    let chunks = get_pointers(&tensor, "_chunks")
        .ok_or_else(|| BlobError::NotFound(format!("chunks for {artifact_id}")))?;

    let mut hasher = StreamingHasher::new();

    for chunk_key in &chunks {
        let chunk_tensor = store
            .get(chunk_key)
            .map_err(|_| BlobError::ChunkMissing(chunk_key.clone()))?;

        let chunk_data = get_bytes(&chunk_tensor, "_data")
            .ok_or_else(|| BlobError::ChunkMissing(chunk_key.clone()))?;

        hasher.update(&chunk_data);
    }

    let actual_checksum = hasher.finalize();
    Ok(actual_checksum == expected_checksum)
}

/// Verify a single chunk's integrity by checking its content hash matches its key.
///
/// # Errors
///
/// Returns an error if the chunk is not found or has an invalid key format.
pub fn verify_chunk(store: &TensorStore, chunk_key: &str) -> Result<bool> {
    // Extract expected hash from key: "_blob:chunk:sha256:..."
    let expected_hash = chunk_key
        .strip_prefix("_blob:chunk:")
        .ok_or_else(|| BlobError::InvalidArtifactId(chunk_key.to_string()))?;

    let tensor = store
        .get(chunk_key)
        .map_err(|_| BlobError::ChunkMissing(chunk_key.to_string()))?;

    let data = get_bytes(&tensor, "_data")
        .ok_or_else(|| BlobError::ChunkMissing(chunk_key.to_string()))?;

    let actual_hash = crate::chunker::compute_hash(&data);
    Ok(actual_hash == expected_hash)
}

/// Repair the blob store by fixing reference counts and removing orphans.
///
/// # Errors
///
/// Returns an error if store operations fail.
pub fn repair(store: &TensorStore) -> Result<RepairStats> {
    let mut stats = RepairStats::default();

    // 1. Build true reference counts from all artifacts
    let mut true_refs: std::collections::HashMap<String, i64> = std::collections::HashMap::new();

    for meta_key in store.scan("_blob:meta:") {
        stats.artifacts_checked += 1;

        if let Ok(tensor) = store.get(&meta_key) {
            if let Some(chunks) = get_pointers(&tensor, "_chunks") {
                for chunk_key in chunks {
                    *true_refs.entry(chunk_key).or_insert(0) += 1;
                }
            }
        }
    }

    // 2. Verify all chunks and fix reference counts
    let mut orphan_keys = Vec::new();

    for chunk_key in store.scan("_blob:chunk:") {
        stats.chunks_verified += 1;

        if let Ok(mut tensor) = store.get(&chunk_key) {
            let current_refs = get_int(&tensor, "_refs").unwrap_or(0);
            let expected_refs = true_refs.get(&chunk_key).copied().unwrap_or(0);

            if current_refs != expected_refs {
                tensor.set(
                    "_refs",
                    TensorValue::Scalar(ScalarValue::Int(expected_refs)),
                );
                store.put(&chunk_key, tensor)?;
                stats.refs_fixed += 1;
            }

            if expected_refs == 0 {
                orphan_keys.push(chunk_key);
            }
        }
    }

    // 3. Delete orphans
    for orphan_key in orphan_keys {
        if store.delete(&orphan_key).is_ok() {
            stats.orphans_deleted += 1;
        }
    }

    Ok(stats)
}

/// Check if all chunks for an artifact exist.
///
/// # Errors
///
/// Returns an error if the artifact is not found.
pub fn check_chunks_exist(store: &TensorStore, artifact_id: &str) -> Result<Vec<String>> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    let chunks = get_pointers(&tensor, "_chunks")
        .ok_or_else(|| BlobError::NotFound(format!("chunks for {artifact_id}")))?;

    let mut missing = Vec::new();
    for chunk_key in chunks {
        if !store.exists(&chunk_key) {
            missing.push(chunk_key);
        }
    }

    Ok(missing)
}

/// Find all chunks not referenced by any artifact.
#[must_use]
pub fn find_orphaned_chunks(store: &TensorStore) -> Vec<String> {
    // Build set of all referenced chunks
    let mut referenced: HashSet<String> = HashSet::new();

    for meta_key in store.scan("_blob:meta:") {
        if let Ok(tensor) = store.get(&meta_key) {
            if let Some(chunks) = get_pointers(&tensor, "_chunks") {
                referenced.extend(chunks);
            }
        }
    }

    // Find unreferenced chunks
    store
        .scan("_blob:chunk:")
        .into_iter()
        .filter(|key| !referenced.contains(key))
        .collect()
}

/// Delete a specific artifact and decrement chunk references.
///
/// # Errors
///
/// Returns an error if the artifact is not found or deletion fails.
pub fn delete_artifact(store: &TensorStore, artifact_id: &str) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    // Get chunks and decrement refs
    if let Some(chunks) = get_pointers(&tensor, "_chunks") {
        for chunk_key in chunks {
            decrement_chunk_refs(store, &chunk_key)?;
        }
    }

    // Clean up secondary index entries for links
    if let Some(linked_to) = get_pointers(&tensor, "_linked_to") {
        for entity in linked_to {
            let idx_key = format!("_blob:idx:link:{entity}:{artifact_id}");
            let _ = store.delete(&idx_key);
        }
    }

    // Clean up secondary index entries for tags
    if let Some(tags) = get_pointers(&tensor, "_tags") {
        for tag_ref in tags {
            if let Some(tag) = tag_ref.strip_prefix("tag:") {
                let idx_key = format!("_blob:idx:tag:{tag}:{artifact_id}");
                let _ = store.delete(&idx_key);
            }
        }
    }

    // Clean up secondary index entry for content type
    if let Some(ct) = get_string(&tensor, "_content_type") {
        let idx_key = format!("_blob:idx:ct:{ct}:{artifact_id}");
        let _ = store.delete(&idx_key);
    }

    // Delete metadata
    store.delete(&meta_key)?;

    Ok(())
}

/// Update artifact metadata field.
///
/// # Errors
///
/// Returns an error if the artifact is not found or update fails.
pub fn update_artifact_field(
    store: &TensorStore,
    artifact_id: &str,
    field: &str,
    value: TensorValue,
) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let mut tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    tensor.set(field, value);

    // Update modified timestamp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0);
    tensor.set("_modified", TensorValue::Scalar(ScalarValue::Int(now)));

    store.put(&meta_key, tensor)?;
    Ok(())
}

/// Add a link to an artifact.
///
/// # Errors
///
/// Returns an error if the artifact is not found or update fails.
pub fn add_artifact_link(store: &TensorStore, artifact_id: &str, entity: &str) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let mut tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    let mut linked_to = get_pointers(&tensor, "_linked_to").unwrap_or_default();
    if !linked_to.contains(&entity.to_string()) {
        linked_to.push(entity.to_string());
        tensor.set("_linked_to", TensorValue::Pointers(linked_to));
        store.put(&meta_key, tensor)?;

        // Write secondary index entry for link lookup
        let idx_key = format!("_blob:idx:link:{entity}:{artifact_id}");
        store.put(&idx_key, TensorData::new())?;
    }

    Ok(())
}

/// Remove a link from an artifact.
///
/// # Errors
///
/// Returns an error if the artifact is not found or update fails.
pub fn remove_artifact_link(store: &TensorStore, artifact_id: &str, entity: &str) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let mut tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    if let Some(mut linked_to) = get_pointers(&tensor, "_linked_to") {
        linked_to.retain(|e| e != entity);
        tensor.set("_linked_to", TensorValue::Pointers(linked_to));
        store.put(&meta_key, tensor)?;

        // Remove secondary index entry for link lookup
        let idx_key = format!("_blob:idx:link:{entity}:{artifact_id}");
        let _ = store.delete(&idx_key);
    }

    Ok(())
}

/// Add a tag to an artifact.
///
/// # Errors
///
/// Returns an error if the artifact is not found or update fails.
pub fn add_artifact_tag(store: &TensorStore, artifact_id: &str, tag: &str) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let mut tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    let tag_ref = format!("tag:{tag}");
    let mut tags = get_pointers(&tensor, "_tags").unwrap_or_default();
    if !tags.contains(&tag_ref) {
        tags.push(tag_ref);
        tensor.set("_tags", TensorValue::Pointers(tags));
        store.put(&meta_key, tensor)?;

        // Write secondary index entry for tag lookup
        let idx_key = format!("_blob:idx:tag:{tag}:{artifact_id}");
        store.put(&idx_key, TensorData::new())?;
    }

    Ok(())
}

/// Remove a tag from an artifact.
///
/// # Errors
///
/// Returns an error if the artifact is not found or update fails.
pub fn remove_artifact_tag(store: &TensorStore, artifact_id: &str, tag: &str) -> Result<()> {
    let meta_key = format!("_blob:meta:{artifact_id}");
    let mut tensor = store
        .get(&meta_key)
        .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

    let tag_ref = format!("tag:{tag}");
    if let Some(mut tags) = get_pointers(&tensor, "_tags") {
        tags.retain(|t| t != &tag_ref);
        tensor.set("_tags", TensorValue::Pointers(tags));
        store.put(&meta_key, tensor)?;

        // Remove secondary index entry for tag lookup
        let idx_key = format!("_blob:idx:tag:{tag}:{artifact_id}");
        let _ = store.delete(&idx_key);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use tensor_store::TensorData;

    use super::*;
    use crate::chunker::Chunk;

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

        store.put(&chunk_key, tensor).unwrap();
        chunk_key
    }

    fn store_artifact(store: &TensorStore, id: &str, chunks: Vec<String>, checksum: &str) {
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
        tensor.set(
            "_checksum",
            TensorValue::Scalar(ScalarValue::String(checksum.to_string())),
        );

        let meta_key = format!("_blob:meta:{id}");
        store.put(&meta_key, tensor).unwrap();
    }

    #[test]
    fn test_verify_artifact_valid() {
        let store = create_test_store();
        let data = b"test data";
        let chunk_key = store_chunk(&store, data, 1);
        let checksum = crate::chunker::compute_hash(data);

        store_artifact(&store, "test", vec![chunk_key], &checksum);

        let valid = verify_artifact(&store, "test").unwrap();
        assert!(valid);
    }

    #[test]
    fn test_verify_artifact_invalid_checksum() {
        let store = create_test_store();
        let data = b"test data";
        let chunk_key = store_chunk(&store, data, 1);

        store_artifact(&store, "test", vec![chunk_key], "sha256:invalid");

        let valid = verify_artifact(&store, "test").unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_verify_artifact_not_found() {
        let store = create_test_store();
        let result = verify_artifact(&store, "nonexistent");
        assert!(matches!(result, Err(BlobError::NotFound(_))));
    }

    #[test]
    fn test_verify_chunk_valid() {
        let store = create_test_store();
        let data = b"chunk data";
        let chunk_key = store_chunk(&store, data, 1);

        let valid = verify_chunk(&store, &chunk_key).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_check_chunks_exist_all_present() {
        let store = create_test_store();
        let chunk1 = store_chunk(&store, b"chunk 1", 1);
        let chunk2 = store_chunk(&store, b"chunk 2", 1);

        store_artifact(&store, "test", vec![chunk1, chunk2], "sha256:test");

        let missing = check_chunks_exist(&store, "test").unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn test_check_chunks_exist_some_missing() {
        let store = create_test_store();
        let chunk1 = store_chunk(&store, b"chunk 1", 1);
        let missing_key = "_blob:chunk:sha256:nonexistent".to_string();

        store_artifact(
            &store,
            "test",
            vec![chunk1, missing_key.clone()],
            "sha256:test",
        );

        let missing = check_chunks_exist(&store, "test").unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], missing_key);
    }

    #[test]
    fn test_find_orphaned_chunks() {
        let store = create_test_store();
        let referenced = store_chunk(&store, b"referenced", 1);
        let orphan = store_chunk(&store, b"orphan", 0);

        store_artifact(&store, "test", vec![referenced], "sha256:test");

        let orphans = find_orphaned_chunks(&store);
        assert_eq!(orphans.len(), 1);
        assert!(orphans.contains(&orphan));
    }

    #[test]
    fn test_repair() {
        let store = create_test_store();

        // Create chunks with wrong ref counts
        let chunk1 = store_chunk(&store, b"chunk 1", 5); // Wrong ref count
        let orphan = store_chunk(&store, b"orphan", 1); // Orphan with refs

        store_artifact(&store, "test", vec![chunk1.clone()], "sha256:test");

        let stats = repair(&store).unwrap();

        assert_eq!(stats.artifacts_checked, 1);
        assert!(stats.refs_fixed > 0);
        assert_eq!(stats.orphans_deleted, 1);

        // Verify chunk1 has correct ref count
        let tensor = store.get(&chunk1).unwrap();
        assert_eq!(get_int(&tensor, "_refs"), Some(1));

        // Verify orphan was deleted
        assert!(!store.exists(&orphan));
    }

    #[test]
    fn test_delete_artifact() {
        let store = create_test_store();
        let chunk = store_chunk(&store, b"data", 1);

        store_artifact(&store, "test", vec![chunk.clone()], "sha256:test");
        assert!(store.exists("_blob:meta:test"));

        delete_artifact(&store, "test").unwrap();

        assert!(!store.exists("_blob:meta:test"));

        // Chunk should have decremented refs
        let tensor = store.get(&chunk).unwrap();
        assert_eq!(get_int(&tensor, "_refs"), Some(0));
    }

    #[test]
    fn test_add_remove_link() {
        let store = create_test_store();
        store_artifact(&store, "test", vec![], "sha256:test");

        add_artifact_link(&store, "test", "user:alice").unwrap();

        let tensor = store.get("_blob:meta:test").unwrap();
        let links = get_pointers(&tensor, "_linked_to").unwrap();
        assert!(links.contains(&"user:alice".to_string()));

        remove_artifact_link(&store, "test", "user:alice").unwrap();

        let tensor = store.get("_blob:meta:test").unwrap();
        let links = get_pointers(&tensor, "_linked_to").unwrap_or_default();
        assert!(!links.contains(&"user:alice".to_string()));
    }

    #[test]
    fn test_add_remove_tag() {
        let store = create_test_store();
        store_artifact(&store, "test", vec![], "sha256:test");

        add_artifact_tag(&store, "test", "important").unwrap();

        let tensor = store.get("_blob:meta:test").unwrap();
        let tags = get_pointers(&tensor, "_tags").unwrap();
        assert!(tags.contains(&"tag:important".to_string()));

        remove_artifact_tag(&store, "test", "important").unwrap();

        let tensor = store.get("_blob:meta:test").unwrap();
        let tags = get_pointers(&tensor, "_tags").unwrap_or_default();
        assert!(!tags.contains(&"tag:important".to_string()));
    }

    #[test]
    fn test_update_artifact_field() {
        let store = create_test_store();
        store_artifact(&store, "test", vec![], "sha256:test");

        update_artifact_field(
            &store,
            "test",
            "_filename",
            TensorValue::Scalar(ScalarValue::String("new_name.txt".to_string())),
        )
        .unwrap();

        let tensor = store.get("_blob:meta:test").unwrap();
        assert_eq!(
            get_string(&tensor, "_filename"),
            Some("new_name.txt".to_string())
        );
        assert!(get_int(&tensor, "_modified").is_some());
    }
}
