// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! gRPC integration tests for `PointsService`.
//!
//! These tests verify actual gRPC behavior for vector point operations by starting
//! a server and connecting with gRPC clients.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use neumann_server::proto::vector::collections_service_client::CollectionsServiceClient;
use neumann_server::proto::vector::points_service_client::PointsServiceClient;
use neumann_server::proto::vector::{
    CreateCollectionRequest, DeletePointsRequest, GetPointsRequest, Point, QueryPointsRequest,
    ScrollPointsRequest, UpsertPointsRequest,
};
use neumann_server::{AuthConfig, NeumannServer, RateLimitConfig, ServerConfig};
use parking_lot::RwLock;
use query_router::QueryRouter;
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use vector_engine::VectorEngine;

/// Helper to start a vector test server and return its address, shutdown channel, and clients.
async fn start_vector_test_server(
    config_override: Option<ServerConfig>,
) -> (
    SocketAddr,
    oneshot::Sender<()>,
    PointsServiceClient<Channel>,
    CollectionsServiceClient<Channel>,
) {
    let router = Arc::new(RwLock::new(QueryRouter::new()));
    let vector_engine = Arc::new(VectorEngine::new());

    // Find an available port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let config = config_override.unwrap_or_else(|| ServerConfig {
        bind_addr: addr,
        enable_grpc_web: false,
        ..Default::default()
    });
    let config = config.with_bind_addr(addr);

    let server = NeumannServer::new(router, config).with_vector_engine(vector_engine);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let points_client = PointsServiceClient::new(channel.clone());
    let collections_client = CollectionsServiceClient::new(channel);

    (addr, shutdown_tx, points_client, collections_client)
}

/// Helper to setup a test collection with a specific dimension.
async fn setup_test_collection(
    collections: &mut CollectionsServiceClient<Channel>,
    name: &str,
    dimension: u32,
) -> Result<(), tonic::Status> {
    collections
        .create(CreateCollectionRequest {
            name: name.to_string(),
            dimension,
            distance: "cosine".to_string(),
        })
        .await?;
    Ok(())
}

/// Helper to upsert test points into a collection.
async fn upsert_test_points(
    points: &mut PointsServiceClient<Channel>,
    collection: &str,
    count: usize,
    dimension: usize,
) -> Result<(), tonic::Status> {
    let test_points: Vec<Point> = (0..count)
        .map(|i| Point {
            id: format!("point_{i}"),
            vector: (0..dimension).map(|j| (i * 10 + j) as f32 / 10.0).collect(),
            payload: std::collections::HashMap::new(),
        })
        .collect();

    points
        .upsert(UpsertPointsRequest {
            collection: collection.to_string(),
            points: test_points,
        })
        .await?;
    Ok(())
}

// Basic Operations Tests

#[tokio::test]
async fn test_points_upsert_single() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    // Setup collection
    setup_test_collection(&mut collections, "test_single", 3)
        .await
        .unwrap();

    // Upsert single point
    let response = points
        .upsert(UpsertPointsRequest {
            collection: "test_single".to_string(),
            points: vec![Point {
                id: "point_1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: std::collections::HashMap::new(),
            }],
        })
        .await
        .unwrap();

    assert_eq!(response.into_inner().upserted, 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_upsert_batch() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_batch", 3)
        .await
        .unwrap();

    // Upsert 10 points
    upsert_test_points(&mut points, "test_batch", 10, 3)
        .await
        .unwrap();

    drop(shutdown);
}

#[tokio::test]
async fn test_points_upsert_with_payload() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_payload", 3)
        .await
        .unwrap();

    // Upsert point with JSON payload
    let mut payload = std::collections::HashMap::new();
    payload.insert("name".to_string(), b"\"test_document\"".to_vec());
    payload.insert("count".to_string(), b"42".to_vec());

    let response = points
        .upsert(UpsertPointsRequest {
            collection: "test_payload".to_string(),
            points: vec![Point {
                id: "point_meta".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload,
            }],
        })
        .await
        .unwrap();

    assert_eq!(response.into_inner().upserted, 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_get_existing() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_get", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_get", 5, 3)
        .await
        .unwrap();

    // Retrieve existing points with vectors
    let response = points
        .get(GetPointsRequest {
            collection: "test_get".to_string(),
            ids: vec!["point_0".to_string(), "point_2".to_string()],
            with_payload: false,
            with_vector: true,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.points.len(), 2);
    assert!(!result.points[0].vector.is_empty());

    drop(shutdown);
}

#[tokio::test]
async fn test_points_get_without_vector() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_get_no_vec", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_get_no_vec", 3, 3)
        .await
        .unwrap();

    // Retrieve without vectors (metadata only)
    let response = points
        .get(GetPointsRequest {
            collection: "test_get_no_vec".to_string(),
            ids: vec!["point_1".to_string()],
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.points.len(), 1);
    assert!(result.points[0].vector.is_empty());

    drop(shutdown);
}

#[tokio::test]
async fn test_points_get_nonexistent() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_get_missing", 3)
        .await
        .unwrap();

    // Get nonexistent points returns empty
    let response = points
        .get(GetPointsRequest {
            collection: "test_get_missing".to_string(),
            ids: vec!["nonexistent_1".to_string(), "nonexistent_2".to_string()],
            with_payload: false,
            with_vector: true,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.points.len(), 0);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_delete_existing() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_delete", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_delete", 5, 3)
        .await
        .unwrap();

    // Delete existing points
    let response = points
        .delete(DeletePointsRequest {
            collection: "test_delete".to_string(),
            ids: vec!["point_1".to_string(), "point_3".to_string()],
        })
        .await
        .unwrap();

    assert_eq!(response.into_inner().deleted, 2);

    // Verify they're gone
    let get_response = points
        .get(GetPointsRequest {
            collection: "test_delete".to_string(),
            ids: vec!["point_1".to_string()],
            with_payload: false,
            with_vector: true,
        })
        .await
        .unwrap();

    assert_eq!(get_response.into_inner().points.len(), 0);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_delete_nonexistent() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_delete_missing", 3)
        .await
        .unwrap();

    // Delete nonexistent points returns count=0
    let response = points
        .delete(DeletePointsRequest {
            collection: "test_delete_missing".to_string(),
            ids: vec!["missing_1".to_string(), "missing_2".to_string()],
        })
        .await
        .unwrap();

    assert_eq!(response.into_inner().deleted, 0);

    drop(shutdown);
}

// Query Operations Tests

#[tokio::test]
async fn test_points_query_basic() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_query", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_query", 10, 3)
        .await
        .unwrap();

    // Query for k=5 nearest neighbors
    let response = points
        .query(QueryPointsRequest {
            collection: "test_query".to_string(),
            vector: vec![0.5, 1.0, 1.5],
            limit: 5,
            offset: 0,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert!(result.results.len() <= 5);
    // Verify results are sorted by score (descending)
    for i in 1..result.results.len() {
        assert!(result.results[i - 1].score >= result.results[i].score);
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_points_query_with_offset() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_query_offset", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_query_offset", 10, 3)
        .await
        .unwrap();

    // Query with offset pagination
    let response = points
        .query(QueryPointsRequest {
            collection: "test_query_offset".to_string(),
            vector: vec![0.0, 0.0, 0.0],
            limit: 3,
            offset: 2,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert!(result.results.len() <= 3);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_query_with_score_threshold() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_query_threshold", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_query_threshold", 10, 3)
        .await
        .unwrap();

    // Query with score threshold filter
    let response = points
        .query(QueryPointsRequest {
            collection: "test_query_threshold".to_string(),
            vector: vec![0.0, 0.0, 0.0],
            limit: 10,
            offset: 0,
            score_threshold: Some(0.8),
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    // All results should meet threshold
    for scored_point in &result.results {
        assert!(scored_point.score >= 0.8);
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_points_query_empty_collection() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_query_empty", 3)
        .await
        .unwrap();

    // Query empty collection returns empty results
    let response = points
        .query(QueryPointsRequest {
            collection: "test_query_empty".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            limit: 5,
            offset: 0,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.results.len(), 0);

    drop(shutdown);
}

// Scroll/Pagination Tests

#[tokio::test]
async fn test_points_scroll_first_page() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_scroll", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_scroll", 20, 3)
        .await
        .unwrap();

    // Scroll first page with limit=10
    let response = points
        .scroll(ScrollPointsRequest {
            collection: "test_scroll".to_string(),
            offset_id: None,
            limit: 10,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.points.len(), 10);
    assert!(result.next_offset.is_some());

    drop(shutdown);
}

#[tokio::test]
async fn test_points_scroll_pagination() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_scroll_pages", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_scroll_pages", 15, 3)
        .await
        .unwrap();

    // First page
    let page1 = points
        .scroll(ScrollPointsRequest {
            collection: "test_scroll_pages".to_string(),
            offset_id: None,
            limit: 5,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(page1.points.len(), 5);
    let offset = page1.next_offset.unwrap();

    // Second page using offset
    let page2 = points
        .scroll(ScrollPointsRequest {
            collection: "test_scroll_pages".to_string(),
            offset_id: Some(offset),
            limit: 5,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(page2.points.len(), 5);
    // Verify no duplicate IDs
    let page1_ids: std::collections::HashSet<_> = page1.points.iter().map(|p| &p.id).collect();
    let page2_ids: std::collections::HashSet<_> = page2.points.iter().map(|p| &p.id).collect();
    assert!(page1_ids.is_disjoint(&page2_ids));

    drop(shutdown);
}

#[tokio::test]
async fn test_points_scroll_no_more_results() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_scroll_end", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_scroll_end", 5, 3)
        .await
        .unwrap();

    // Scroll with limit larger than total
    let response = points
        .scroll(ScrollPointsRequest {
            collection: "test_scroll_end".to_string(),
            offset_id: None,
            limit: 10,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    let result = response.into_inner();
    assert_eq!(result.points.len(), 5);
    assert!(result.next_offset.is_none());

    drop(shutdown);
}

// Error Handling Tests

#[tokio::test]
async fn test_points_upsert_invalid_dimension() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_invalid_dim", 3)
        .await
        .unwrap();

    // Upsert with wrong dimension should fail
    let result = points
        .upsert(UpsertPointsRequest {
            collection: "test_invalid_dim".to_string(),
            points: vec![Point {
                id: "bad_point".to_string(),
                vector: vec![1.0, 2.0],
                payload: std::collections::HashMap::new(),
            }],
        })
        .await;

    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Internal);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_query_missing_collection() {
    let (_addr, shutdown, mut points, _collections) = start_vector_test_server(None).await;

    // Query non-existent collection - VectorEngine returns empty results for missing collections
    let result = points
        .query(QueryPointsRequest {
            collection: "nonexistent_collection".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            limit: 5,
            offset: 0,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await;

    // VectorEngine returns Ok with empty results for non-existent collections
    assert!(result.is_ok());
    let response = result.unwrap().into_inner();
    assert_eq!(response.results.len(), 0);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_operation_after_collection_delete() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_deleted", 3)
        .await
        .unwrap();
    upsert_test_points(&mut points, "test_deleted", 5, 3)
        .await
        .unwrap();

    // Verify points exist
    let before_delete = points
        .query(QueryPointsRequest {
            collection: "test_deleted".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            limit: 5,
            offset: 0,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!before_delete.results.is_empty());

    // Delete collection
    use neumann_server::proto::vector::DeleteCollectionRequest;
    let delete_response = collections
        .delete(DeleteCollectionRequest {
            name: "test_deleted".to_string(),
        })
        .await
        .unwrap();
    assert!(delete_response.into_inner().deleted);

    // Operations after delete return empty results
    let after_delete = points
        .query(QueryPointsRequest {
            collection: "test_deleted".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            limit: 5,
            offset: 0,
            score_threshold: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(after_delete.results.len(), 0);

    drop(shutdown);
}

// Auth & Rate Limiting Tests

#[tokio::test]
async fn test_points_auth_required() {
    let config = ServerConfig::default().with_auth(
        AuthConfig::new()
            .with_api_key(neumann_server::config::ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ))
            .with_anonymous(false),
    );

    let (_addr, shutdown, mut points, _collections) = start_vector_test_server(Some(config)).await;

    // Request without auth should fail
    let result = points
        .upsert(UpsertPointsRequest {
            collection: "test_auth".to_string(),
            points: vec![],
        })
        .await;

    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_auth_success() {
    let config = ServerConfig::default().with_auth(
        AuthConfig::new()
            .with_api_key(neumann_server::config::ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ))
            .with_anonymous(false),
    );

    let (_addr, shutdown, mut points, mut collections) =
        start_vector_test_server(Some(config)).await;

    // Create collection with auth
    let mut request = tonic::Request::new(CreateCollectionRequest {
        name: "test_auth_ok".to_string(),
        dimension: 3,
        distance: "cosine".to_string(),
    });
    request.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    collections.create(request).await.unwrap();

    // Upsert with valid auth should succeed
    let mut request = tonic::Request::new(UpsertPointsRequest {
        collection: "test_auth_ok".to_string(),
        points: vec![Point {
            id: "point_1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            payload: std::collections::HashMap::new(),
        }],
    });
    request.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );

    let result = points.upsert(request).await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_points_rate_limit_exceeded() {
    // Set max_vector_ops to 3: 1 for collection create, 2 for upserts
    let config = ServerConfig::default()
        .with_auth(
            AuthConfig::new()
                .with_api_key(neumann_server::config::ApiKey::new(
                    "test-api-key-12345678".to_string(),
                    "user:test".to_string(),
                ))
                .with_anonymous(false),
        )
        .with_rate_limit(RateLimitConfig::new().with_max_vector_ops(3));

    let (_addr, shutdown, mut points, mut collections) =
        start_vector_test_server(Some(config)).await;

    // Setup collection (counts as 1 vector op)
    let mut request = tonic::Request::new(CreateCollectionRequest {
        name: "test_rate_limit".to_string(),
        dimension: 3,
        distance: "cosine".to_string(),
    });
    request.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    collections.create(request).await.unwrap();

    // First upsert should succeed (2nd vector op)
    let mut req1 = tonic::Request::new(UpsertPointsRequest {
        collection: "test_rate_limit".to_string(),
        points: vec![Point {
            id: "point_1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            payload: std::collections::HashMap::new(),
        }],
    });
    req1.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    assert!(
        points.upsert(req1).await.is_ok(),
        "First upsert should succeed"
    );

    // Second upsert should succeed (3rd vector op, at limit)
    let mut req2 = tonic::Request::new(UpsertPointsRequest {
        collection: "test_rate_limit".to_string(),
        points: vec![Point {
            id: "point_2".to_string(),
            vector: vec![2.0, 3.0, 4.0],
            payload: std::collections::HashMap::new(),
        }],
    });
    req2.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    assert!(
        points.upsert(req2).await.is_ok(),
        "Second upsert should succeed"
    );

    // Third upsert should exceed rate limit (4th vector op)
    let mut request3 = tonic::Request::new(UpsertPointsRequest {
        collection: "test_rate_limit".to_string(),
        points: vec![Point {
            id: "point_3".to_string(),
            vector: vec![3.0, 4.0, 5.0],
            payload: std::collections::HashMap::new(),
        }],
    });
    request3.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    let result = points.upsert(request3).await;
    assert!(result.is_err(), "Third upsert should be rate limited");
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);

    drop(shutdown);
}

// Concurrency Tests

#[tokio::test]
async fn test_points_concurrent_upsert() {
    let (_addr, shutdown, points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_concurrent", 3)
        .await
        .unwrap();

    // Spawn 10 concurrent upsert operations
    let mut handles = vec![];
    for i in 0..10 {
        let mut client = points.clone();
        handles.push(tokio::spawn(async move {
            client
                .upsert(UpsertPointsRequest {
                    collection: "test_concurrent".to_string(),
                    points: vec![Point {
                        id: format!("concurrent_{i}"),
                        vector: vec![i as f32, (i + 1) as f32, (i + 2) as f32],
                        payload: std::collections::HashMap::new(),
                    }],
                })
                .await
        }));
    }

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles).await;

    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    // Verify all points were inserted
    let mut client = points.clone();
    let scroll_response = client
        .scroll(ScrollPointsRequest {
            collection: "test_concurrent".to_string(),
            offset_id: None,
            limit: 20,
            with_payload: false,
            with_vector: false,
        })
        .await
        .unwrap();

    assert_eq!(scroll_response.into_inner().points.len(), 10);

    drop(shutdown);
}

#[tokio::test]
async fn test_points_concurrent_query() {
    let (_addr, shutdown, points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_concurrent_query", 3)
        .await
        .unwrap();

    let mut points_mut = points.clone();
    upsert_test_points(&mut points_mut, "test_concurrent_query", 20, 3)
        .await
        .unwrap();

    // Spawn 5 concurrent query operations
    let mut handles = vec![];
    for i in 0..5 {
        let mut client = points.clone();
        handles.push(tokio::spawn(async move {
            client
                .query(QueryPointsRequest {
                    collection: "test_concurrent_query".to_string(),
                    vector: vec![i as f32, (i + 1) as f32, (i + 2) as f32],
                    limit: 5,
                    offset: 0,
                    score_threshold: None,
                    with_payload: false,
                    with_vector: false,
                })
                .await
        }));
    }

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles).await;

    for result in results {
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.is_ok());
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_points_health_after_failures() {
    let (_addr, shutdown, mut points, mut collections) = start_vector_test_server(None).await;

    setup_test_collection(&mut collections, "test_health", 3)
        .await
        .unwrap();

    // Trigger 5 consecutive failures by querying with wrong dimension
    for _ in 0..5 {
        let _ = points
            .upsert(UpsertPointsRequest {
                collection: "test_health".to_string(),
                points: vec![Point {
                    id: "bad".to_string(),
                    vector: vec![1.0],
                    payload: std::collections::HashMap::new(),
                }],
            })
            .await;
    }

    // Service should continue to accept requests even after failures
    let result = points
        .upsert(UpsertPointsRequest {
            collection: "test_health".to_string(),
            points: vec![Point {
                id: "good".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: std::collections::HashMap::new(),
            }],
        })
        .await;

    assert!(result.is_ok());

    drop(shutdown);
}
