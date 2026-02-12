// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! gRPC integration tests for `CollectionsService`.
//!
//! These tests verify actual gRPC behavior for vector collection management by starting
//! a server and connecting with gRPC clients.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use neumann_server::proto::vector::collections_service_client::CollectionsServiceClient;
use neumann_server::proto::vector::points_service_client::PointsServiceClient;
use neumann_server::proto::vector::{
    CreateCollectionRequest, DeleteCollectionRequest, GetCollectionRequest, ListCollectionsRequest,
    Point, UpsertPointsRequest,
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
    CollectionsServiceClient<Channel>,
    PointsServiceClient<Channel>,
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

    let collections_client = CollectionsServiceClient::new(channel.clone());
    let points_client = PointsServiceClient::new(channel);

    (addr, shutdown_tx, collections_client, points_client)
}

// Basic CRUD Tests

#[tokio::test]
async fn test_collections_create_default() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection with cosine metric (default)
    let response = collections
        .create(CreateCollectionRequest {
            name: "test_cosine".to_string(),
            dimension: 128,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    assert!(response.into_inner().created);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_create_euclidean() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection with euclidean (L2) metric
    let response = collections
        .create(CreateCollectionRequest {
            name: "test_euclidean".to_string(),
            dimension: 256,
            distance: "euclidean".to_string(),
        })
        .await
        .unwrap();

    assert!(response.into_inner().created);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_create_dot() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection with dot product metric
    let response = collections
        .create(CreateCollectionRequest {
            name: "test_dot".to_string(),
            dimension: 512,
            distance: "dot".to_string(),
        })
        .await
        .unwrap();

    assert!(response.into_inner().created);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_create_invalid_metric() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection with invalid distance metric
    let result = collections
        .create(CreateCollectionRequest {
            name: "test_invalid".to_string(),
            dimension: 64,
            distance: "manhattan".to_string(),
        })
        .await;

    // Invalid metric should fail
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_create_duplicate() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create first collection
    let _ = collections
        .create(CreateCollectionRequest {
            name: "test_duplicate".to_string(),
            dimension: 128,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Try to create duplicate
    let result = collections
        .create(CreateCollectionRequest {
            name: "test_duplicate".to_string(),
            dimension: 128,
            distance: "cosine".to_string(),
        })
        .await;

    // Should fail with AlreadyExists
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::AlreadyExists);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_get_existing() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection
    collections
        .create(CreateCollectionRequest {
            name: "test_get".to_string(),
            dimension: 384,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Get collection info
    let response = collections
        .get(GetCollectionRequest {
            name: "test_get".to_string(),
        })
        .await
        .unwrap();

    let info = response.into_inner();
    assert_eq!(info.name, "test_get");
    assert_eq!(info.dimension, 384);
    assert_eq!(info.distance, "cosine");
    assert_eq!(info.points_count, 0);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_get_nonexistent() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Get non-existent collection
    let result = collections
        .get(GetCollectionRequest {
            name: "nonexistent".to_string(),
        })
        .await;

    // Should fail with NotFound
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_delete_existing() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create collection
    collections
        .create(CreateCollectionRequest {
            name: "test_delete".to_string(),
            dimension: 128,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Delete collection
    let response = collections
        .delete(DeleteCollectionRequest {
            name: "test_delete".to_string(),
        })
        .await
        .unwrap();

    assert!(response.into_inner().deleted);

    // Verify it's gone
    let result = collections
        .get(GetCollectionRequest {
            name: "test_delete".to_string(),
        })
        .await;

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_delete_nonexistent() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Delete non-existent collection
    let result = collections
        .delete(DeleteCollectionRequest {
            name: "nonexistent".to_string(),
        })
        .await;

    // Should fail with NotFound
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_list_empty() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // List collections before creating any
    let response = collections.list(ListCollectionsRequest {}).await.unwrap();

    let list = response.into_inner();
    assert_eq!(list.collections.len(), 0);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_list_multiple() {
    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(None).await;

    // Create multiple collections
    for i in 1..=3 {
        collections
            .create(CreateCollectionRequest {
                name: format!("collection_{i}"),
                dimension: 128 * i,
                distance: "cosine".to_string(),
            })
            .await
            .unwrap();
    }

    // List all collections
    let response = collections.list(ListCollectionsRequest {}).await.unwrap();

    let list = response.into_inner();
    assert_eq!(list.collections.len(), 3);

    // Verify names
    let names: std::collections::HashSet<_> = list.collections.into_iter().collect();
    assert!(names.contains("collection_1"));
    assert!(names.contains("collection_2"));
    assert!(names.contains("collection_3"));

    drop(shutdown);
}

// Integration Tests

#[tokio::test]
async fn test_collections_get_with_points() {
    let (_addr, shutdown, mut collections, mut points) = start_vector_test_server(None).await;

    // Create collection
    collections
        .create(CreateCollectionRequest {
            name: "test_points_count".to_string(),
            dimension: 3,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Add some points
    points
        .upsert(UpsertPointsRequest {
            collection: "test_points_count".to_string(),
            points: vec![
                Point {
                    id: "point_1".to_string(),
                    vector: vec![1.0, 2.0, 3.0],
                    payload: std::collections::HashMap::new(),
                },
                Point {
                    id: "point_2".to_string(),
                    vector: vec![4.0, 5.0, 6.0],
                    payload: std::collections::HashMap::new(),
                },
            ],
        })
        .await
        .unwrap();

    // Get collection info
    let response = collections
        .get(GetCollectionRequest {
            name: "test_points_count".to_string(),
        })
        .await
        .unwrap();

    let info = response.into_inner();
    assert_eq!(info.points_count, 2);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_delete_cascade() {
    let (_addr, shutdown, mut collections, mut points) = start_vector_test_server(None).await;

    // Create collection
    collections
        .create(CreateCollectionRequest {
            name: "test_cascade".to_string(),
            dimension: 3,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Add points
    points
        .upsert(UpsertPointsRequest {
            collection: "test_cascade".to_string(),
            points: vec![Point {
                id: "point_1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: std::collections::HashMap::new(),
            }],
        })
        .await
        .unwrap();

    // Delete collection
    collections
        .delete(DeleteCollectionRequest {
            name: "test_cascade".to_string(),
        })
        .await
        .unwrap();

    // Recreate with same name
    collections
        .create(CreateCollectionRequest {
            name: "test_cascade".to_string(),
            dimension: 3,
            distance: "cosine".to_string(),
        })
        .await
        .unwrap();

    // Get info - should have 0 points
    let response = collections
        .get(GetCollectionRequest {
            name: "test_cascade".to_string(),
        })
        .await
        .unwrap();

    let info = response.into_inner();
    assert_eq!(info.points_count, 0);

    drop(shutdown);
}

// Auth & Rate Limiting Tests

#[tokio::test]
async fn test_collections_auth_required() {
    let config = ServerConfig::default().with_auth(
        AuthConfig::new()
            .with_api_key(neumann_server::config::ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ))
            .with_anonymous(false),
    );

    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(Some(config)).await;

    // Request without auth should fail
    let result = collections
        .create(CreateCollectionRequest {
            name: "test_auth".to_string(),
            dimension: 128,
            distance: "cosine".to_string(),
        })
        .await;

    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_auth_success() {
    let config = ServerConfig::default().with_auth(
        AuthConfig::new()
            .with_api_key(neumann_server::config::ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ))
            .with_anonymous(false),
    );

    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(Some(config)).await;

    // Request with valid auth should succeed
    let mut request = tonic::Request::new(CreateCollectionRequest {
        name: "test_auth_ok".to_string(),
        dimension: 128,
        distance: "cosine".to_string(),
    });
    request.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );

    let result = collections.create(request).await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_rate_limit() {
    let config = ServerConfig::default()
        .with_auth(
            AuthConfig::new()
                .with_api_key(neumann_server::config::ApiKey::new(
                    "test-api-key-12345678".to_string(),
                    "user:test".to_string(),
                ))
                .with_anonymous(false),
        )
        .with_rate_limit(RateLimitConfig::new().with_max_vector_ops(2));

    let (_addr, shutdown, mut collections, _points) = start_vector_test_server(Some(config)).await;

    // Create 2 collections (at limit)
    for i in 0..2 {
        let mut req = tonic::Request::new(CreateCollectionRequest {
            name: format!("collection_{i}"),
            dimension: 128,
            distance: "cosine".to_string(),
        });
        req.metadata_mut().insert(
            "x-api-key",
            MetadataValue::from_static("test-api-key-12345678"),
        );
        let result = collections.create(req).await;
        assert!(result.is_ok(), "Request {i} should succeed");
    }

    // Third request should exceed rate limit
    let mut request3 = tonic::Request::new(CreateCollectionRequest {
        name: "collection_3".to_string(),
        dimension: 128,
        distance: "cosine".to_string(),
    });
    request3.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test-api-key-12345678"),
    );
    let result = collections.create(request3).await;
    assert!(result.is_err(), "Third request should be rate limited");
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);

    drop(shutdown);
}

#[tokio::test]
async fn test_collections_concurrent_create() {
    let (_addr, shutdown, collections, _points) = start_vector_test_server(None).await;

    // Spawn 5 concurrent collection creates
    let mut handles = vec![];
    for i in 0..5 {
        let mut client = collections.clone();
        handles.push(tokio::spawn(async move {
            client
                .create(CreateCollectionRequest {
                    name: format!("concurrent_{i}"),
                    dimension: 128,
                    distance: "cosine".to_string(),
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

    // Verify all collections were created
    let mut client = collections.clone();
    let list_response = client.list(ListCollectionsRequest {}).await.unwrap();

    assert_eq!(list_response.into_inner().collections.len(), 5);

    drop(shutdown);
}
