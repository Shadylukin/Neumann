// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for tensor_blob embedding search and linking features.

use integration_tests::create_router_with_blob;
use query_router::QueryResult;

/// Helper: create a blob-enabled router with identity set.
fn blob_router() -> query_router::QueryRouter {
    let mut router = create_router_with_blob();
    router.set_identity("user:integration");
    router
}

/// Helper: PUT a blob and return its artifact ID.
fn put_blob(router: &query_router::QueryRouter, name: &str, content: &str) -> String {
    let result = router
        .execute_parsed(&format!("BLOB PUT '{name}' '{content}'"))
        .unwrap();
    match result {
        QueryResult::Value(id) => id,
        other => panic!("Expected Value with artifact ID, got {other:?}"),
    }
}

#[test]
fn test_blob_put_get_roundtrip() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "hello.txt", "Hello, World!");

    let get_result = router
        .execute_parsed(&format!("BLOB GET '{artifact_id}'"))
        .unwrap();
    match get_result {
        QueryResult::Blob(data) => {
            assert_eq!(String::from_utf8_lossy(&data), "Hello, World!");
        },
        other => panic!("Expected Blob result, got {other:?}"),
    }
}

#[test]
fn test_blob_link_unlink_lifecycle() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "linked_doc.txt", "document data");

    // Link blob to an entity
    router
        .execute_parsed(&format!("BLOB LINK '{artifact_id}' TO 'project:42'"))
        .unwrap();

    // BLOBS FOR should find it
    let for_result = router.execute_parsed("BLOBS FOR 'project:42'").unwrap();
    match &for_result {
        QueryResult::ArtifactList(list) => {
            assert!(
                list.iter().any(|id| id == &artifact_id),
                "Linked artifact not found in BLOBS FOR result"
            );
        },
        other => panic!("Expected ArtifactList, got {other:?}"),
    }

    // Unlink
    router
        .execute_parsed(&format!("BLOB UNLINK '{artifact_id}' FROM 'project:42'"))
        .unwrap();

    // BLOBS FOR should now be empty
    let for_after = router.execute_parsed("BLOBS FOR 'project:42'").unwrap();
    match &for_after {
        QueryResult::ArtifactList(list) => {
            assert!(
                !list.iter().any(|id| id == &artifact_id),
                "Artifact should no longer be linked"
            );
        },
        other => panic!("Expected ArtifactList, got {other:?}"),
    }
}

#[test]
fn test_blob_tag_workflow() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "tagged_file.txt", "tag me");

    // Tag the blob
    router
        .execute_parsed(&format!("BLOB TAG '{artifact_id}' 'urgent'"))
        .unwrap();

    // BLOBS BY TAG should find it
    let bytag = router.execute_parsed("BLOBS BY TAG 'urgent'").unwrap();
    match &bytag {
        QueryResult::ArtifactList(list) => {
            assert!(
                list.iter().any(|id| id == &artifact_id),
                "Tagged artifact not found in BLOBS BY TAG"
            );
        },
        other => panic!("Expected ArtifactList, got {other:?}"),
    }

    // Untag
    router
        .execute_parsed(&format!("BLOB UNTAG '{artifact_id}' 'urgent'"))
        .unwrap();

    // BLOBS BY TAG should no longer find it
    let bytag_after = router.execute_parsed("BLOBS BY TAG 'urgent'").unwrap();
    match &bytag_after {
        QueryResult::ArtifactList(list) => {
            assert!(
                !list.iter().any(|id| id == &artifact_id),
                "Artifact should not appear after untag"
            );
        },
        other => panic!("Expected ArtifactList, got {other:?}"),
    }
}

#[test]
fn test_blob_metadata_set_get() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "meta_doc.txt", "metadata test");

    // Set custom metadata
    router
        .execute_parsed(&format!("BLOB META SET '{artifact_id}' 'author' 'alice'"))
        .unwrap();
    router
        .execute_parsed(&format!("BLOB META SET '{artifact_id}' 'version' '2.0'"))
        .unwrap();

    // Get metadata values
    let author = router
        .execute_parsed(&format!("BLOB META GET '{artifact_id}' 'author'"))
        .unwrap();
    match author {
        QueryResult::Value(v) => assert_eq!(v, "alice"),
        other => panic!("Expected Value, got {other:?}"),
    }

    let version = router
        .execute_parsed(&format!("BLOB META GET '{artifact_id}' 'version'"))
        .unwrap();
    match version {
        QueryResult::Value(v) => assert_eq!(v, "2.0"),
        other => panic!("Expected Value, got {other:?}"),
    }

    // Nonexistent key returns "(not found)"
    let missing = router
        .execute_parsed(&format!("BLOB META GET '{artifact_id}' 'missing_key'"))
        .unwrap();
    match missing {
        QueryResult::Value(v) => assert_eq!(v, "(not found)"),
        other => panic!("Expected Value, got {other:?}"),
    }
}

#[test]
fn test_blob_list_and_info() {
    let router = blob_router();
    let id1 = put_blob(&router, "file_a.txt", "content A");
    let id2 = put_blob(&router, "file_b.txt", "content B");
    let id3 = put_blob(&router, "file_c.txt", "content C");

    // BLOBS should list all three
    let list = router.execute_parsed("BLOBS").unwrap();
    match &list {
        QueryResult::ArtifactList(items) => {
            assert_eq!(items.len(), 3, "Expected 3 blobs in list");
            assert!(items.contains(&id1));
            assert!(items.contains(&id2));
            assert!(items.contains(&id3));
        },
        other => panic!("Expected ArtifactList, got {other:?}"),
    }

    // INFO for each blob
    let info1 = router
        .execute_parsed(&format!("BLOB INFO '{id1}'"))
        .unwrap();
    match info1 {
        QueryResult::ArtifactInfo(info) => {
            assert_eq!(info.filename, "file_a.txt");
            assert_eq!(info.size, 9); // "content A" is 9 bytes
        },
        other => panic!("Expected ArtifactInfo, got {other:?}"),
    }

    let info2 = router
        .execute_parsed(&format!("BLOB INFO '{id2}'"))
        .unwrap();
    match info2 {
        QueryResult::ArtifactInfo(info) => {
            assert_eq!(info.filename, "file_b.txt");
            assert_eq!(info.size, 9);
        },
        other => panic!("Expected ArtifactInfo, got {other:?}"),
    }
}

#[test]
fn test_blob_delete_removes() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "ephemeral.txt", "gone soon");

    // Verify it exists
    let get_result = router
        .execute_parsed(&format!("BLOB GET '{artifact_id}'"))
        .unwrap();
    assert!(matches!(get_result, QueryResult::Blob(_)));

    // Delete
    router
        .execute_parsed(&format!("BLOB DELETE '{artifact_id}'"))
        .unwrap();

    // GET should now fail
    let gone = router.execute_parsed(&format!("BLOB GET '{artifact_id}'"));
    assert!(gone.is_err(), "GET after DELETE should fail");
}

#[test]
fn test_blob_gc_runs() {
    let router = blob_router();
    let id = put_blob(&router, "gc_target.txt", "garbage collect me");

    // Delete the blob first
    router
        .execute_parsed(&format!("BLOB DELETE '{id}'"))
        .unwrap();

    // Run GC
    let gc_result = router.execute_parsed("BLOB GC").unwrap();
    match gc_result {
        QueryResult::Value(v) => {
            assert!(v.contains("Deleted"), "GC result should mention Deleted");
            assert!(v.contains("freed"), "GC result should mention freed");
        },
        other => panic!("Expected Value, got {other:?}"),
    }
}

#[test]
fn test_blob_stats() {
    let router = blob_router();

    // Empty stats
    let empty_stats = router.execute_parsed("BLOB STATS").unwrap();
    match &empty_stats {
        QueryResult::BlobStats(stats) => {
            assert_eq!(stats.artifact_count, 0);
        },
        other => panic!("Expected BlobStats, got {other:?}"),
    }

    // Add some blobs
    put_blob(&router, "stat1.txt", "data one");
    put_blob(&router, "stat2.txt", "data two");
    put_blob(&router, "stat3.txt", "data three");

    let stats = router.execute_parsed("BLOB STATS").unwrap();
    match stats {
        QueryResult::BlobStats(s) => {
            assert_eq!(s.artifact_count, 3, "Expected 3 artifacts");
            assert!(s.total_bytes > 0, "Expected non-zero total bytes");
        },
        other => panic!("Expected BlobStats, got {other:?}"),
    }
}

#[test]
fn test_blob_multiple_tags() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "multi_tag.txt", "many tags");

    // Add multiple tags
    router
        .execute_parsed(&format!("BLOB TAG '{artifact_id}' 'priority'"))
        .unwrap();
    router
        .execute_parsed(&format!("BLOB TAG '{artifact_id}' 'review'"))
        .unwrap();
    router
        .execute_parsed(&format!("BLOB TAG '{artifact_id}' 'draft'"))
        .unwrap();

    // Each tag should find the blob
    for tag in &["priority", "review", "draft"] {
        let result = router
            .execute_parsed(&format!("BLOBS BY TAG '{tag}'"))
            .unwrap();
        match &result {
            QueryResult::ArtifactList(list) => {
                assert!(
                    list.contains(&artifact_id),
                    "Blob should be found by tag '{tag}'"
                );
            },
            other => panic!("Expected ArtifactList for tag '{tag}', got {other:?}"),
        }
    }

    // Verify INFO shows all tags
    let info = router
        .execute_parsed(&format!("BLOB INFO '{artifact_id}'"))
        .unwrap();
    match info {
        QueryResult::ArtifactInfo(info) => {
            assert!(info.tags.contains(&"priority".to_string()));
            assert!(info.tags.contains(&"review".to_string()));
            assert!(info.tags.contains(&"draft".to_string()));
        },
        other => panic!("Expected ArtifactInfo, got {other:?}"),
    }
}

#[test]
fn test_blob_link_multiple_entities() {
    let router = blob_router();
    let artifact_id = put_blob(&router, "shared_doc.txt", "shared across entities");

    // Link to multiple entities
    router
        .execute_parsed(&format!("BLOB LINK '{artifact_id}' TO 'task:100'"))
        .unwrap();
    router
        .execute_parsed(&format!("BLOB LINK '{artifact_id}' TO 'task:200'"))
        .unwrap();
    router
        .execute_parsed(&format!("BLOB LINK '{artifact_id}' TO 'project:alpha'"))
        .unwrap();

    // Each entity should see the blob
    for entity in &["task:100", "task:200", "project:alpha"] {
        let result = router
            .execute_parsed(&format!("BLOBS FOR '{entity}'"))
            .unwrap();
        match &result {
            QueryResult::ArtifactList(list) => {
                assert!(
                    list.contains(&artifact_id),
                    "Blob should be linked to '{entity}'"
                );
            },
            other => panic!("Expected ArtifactList for entity '{entity}', got {other:?}"),
        }
    }

    // BLOB INFO should show all links
    let info = router
        .execute_parsed(&format!("BLOB INFO '{artifact_id}'"))
        .unwrap();
    match info {
        QueryResult::ArtifactInfo(info) => {
            assert_eq!(info.linked_to.len(), 3, "Expected 3 links");
            assert!(info.linked_to.contains(&"task:100".to_string()));
            assert!(info.linked_to.contains(&"task:200".to_string()));
            assert!(info.linked_to.contains(&"project:alpha".to_string()));
        },
        other => panic!("Expected ArtifactInfo, got {other:?}"),
    }
}
