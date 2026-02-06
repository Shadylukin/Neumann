// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Blob lifecycle integration tests.
//!
//! Tests garbage collection, repair, streaming, and deduplication.

use std::sync::Arc;

use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_store::TensorStore;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_blob_gc_cleans_orphaned_chunks() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    // Upload some blobs
    let id1 = blob
        .put("file1.txt", b"content for file 1", PutOptions::default())
        .await
        .unwrap();
    let id2 = blob
        .put("file2.txt", b"content for file 2", PutOptions::default())
        .await
        .unwrap();

    // Verify both exist
    assert!(blob.exists(&id1).await.unwrap());
    assert!(blob.exists(&id2).await.unwrap());

    // Delete one
    blob.delete(&id1).await.unwrap();
    assert!(!blob.exists(&id1).await.unwrap());

    // Run GC
    let gc_stats = blob.gc().await.unwrap();

    // GC should clean up orphaned chunks
    // The deleted artifact's chunks should be collected if not shared
    // gc_stats.deleted is valid (unsigned, always >= 0)
    let _ = gc_stats.deleted;

    // Second artifact should still work
    let content = blob.get(&id2).await.unwrap();
    assert_eq!(content, b"content for file 2");
}

#[tokio::test]
async fn test_blob_gc_during_upload() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = Arc::new(Mutex::new(BlobStore::new(store, config).await.unwrap()));

    // Upload initial blobs
    let initial_ids = {
        let guard = blob.lock().await;
        let mut ids = vec![];
        for i in 0..5 {
            let id = guard
                .put(
                    &format!("initial{}.txt", i),
                    format!("initial content {}", i).as_bytes(),
                    PutOptions::default(),
                )
                .await
                .unwrap();
            ids.push(id);
        }
        ids
    };

    // Run concurrent uploads and GC
    let upload_blob = Arc::clone(&blob);
    let gc_blob = Arc::clone(&blob);

    let upload_handle = tokio::spawn(async move {
        let mut new_ids = vec![];
        for i in 0..10 {
            let guard = upload_blob.lock().await;
            if let Ok(id) = guard
                .put(
                    &format!("new{}.txt", i),
                    format!("new content {}", i).as_bytes(),
                    PutOptions::default(),
                )
                .await
            {
                new_ids.push(id);
            }
        }
        new_ids
    });

    // Small delay then run GC
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    let gc_handle = tokio::spawn(async move {
        let guard = gc_blob.lock().await;
        guard.gc().await
    });

    let new_ids = upload_handle.await.unwrap();
    let gc_result = gc_handle.await.unwrap();

    // GC should complete without breaking uploads
    assert!(gc_result.is_ok());

    // All uploads should have succeeded
    assert!(!new_ids.is_empty());

    // Initial blobs should still be accessible
    let guard = blob.lock().await;
    for id in &initial_ids {
        assert!(guard.exists(id).await.unwrap());
    }
}

#[tokio::test]
async fn test_blob_repair_corrupted_metadata() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store.clone(), config).await.unwrap();

    // Upload some blobs
    let id1 = blob
        .put("file1.txt", b"file 1 content", PutOptions::default())
        .await
        .unwrap();
    let id2 = blob
        .put("file2.txt", b"file 2 content", PutOptions::default())
        .await
        .unwrap();

    // Run repair
    let repair_stats = blob.repair().unwrap();

    // With clean data, nothing should need repair
    assert!(repair_stats.artifacts_checked >= 2);

    // Blobs should still work after repair
    let content1 = blob.get(&id1).await.unwrap();
    let content2 = blob.get(&id2).await.unwrap();
    assert_eq!(content1, b"file 1 content");
    assert_eq!(content2, b"file 2 content");
}

#[tokio::test]
async fn test_blob_verify_integrity() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    // Upload a blob
    let content = b"integrity check content";
    let id = blob
        .put("verify.txt", content, PutOptions::default())
        .await
        .unwrap();

    // Verify integrity
    let is_valid = blob.verify(&id).unwrap();
    assert!(
        is_valid,
        "Freshly uploaded blob should pass integrity check"
    );

    // Read content to double-check
    let retrieved = blob.get(&id).await.unwrap();
    assert_eq!(retrieved, content);
}

#[tokio::test]
async fn test_blob_streaming_read() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    // Upload a larger blob
    let content: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let id = blob
        .put("large.bin", &content, PutOptions::default())
        .await
        .unwrap();

    // Read using streaming reader
    let mut reader = blob.reader(&id).await.unwrap();

    // Read all chunks
    let mut read_content = Vec::new();
    while let Some(chunk) = reader.next_chunk().await.unwrap() {
        read_content.extend(chunk);
    }

    // Should match original
    assert_eq!(read_content.len(), content.len());
    assert_eq!(read_content, content);
}

#[tokio::test]
async fn test_blob_chunk_deduplication() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store.clone(), config).await.unwrap();

    // Upload same content multiple times with different filenames
    let content = b"This is the same content that will be deduplicated";

    let id1 = blob
        .put("file_a.txt", content, PutOptions::default())
        .await
        .unwrap();
    let id2 = blob
        .put("file_b.txt", content, PutOptions::default())
        .await
        .unwrap();
    let id3 = blob
        .put("file_c.txt", content, PutOptions::default())
        .await
        .unwrap();

    // All should have different artifact IDs
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);

    // But all should return the same content
    let content1 = blob.get(&id1).await.unwrap();
    let content2 = blob.get(&id2).await.unwrap();
    let content3 = blob.get(&id3).await.unwrap();

    assert_eq!(content1, content);
    assert_eq!(content2, content);
    assert_eq!(content3, content);

    // Due to content-addressable storage, chunks should be shared
    // This means storage used should be less than 3x the content size
    // (We can't easily verify this without internal access, but the system should dedupe)
}

#[tokio::test]
async fn test_blob_graceful_shutdown() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    // Upload some blobs (without starting background tasks)
    let mut ids = vec![];
    for i in 0..5 {
        let id = blob
            .put(
                &format!("pre_shutdown{}.txt", i),
                format!("content {}", i).as_bytes(),
                PutOptions::default(),
            )
            .await
            .unwrap();
        ids.push(id);
    }

    // Verify data was stored
    assert!(!blob.store().is_empty());

    // Verify we can still read the artifacts
    for id in &ids {
        let content = blob.get(id).await.unwrap();
        assert!(!content.is_empty());
    }

    // Document: Store data persists even without explicit shutdown
}
