// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for the web admin UI handlers.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use graph_engine::GraphEngine;
use http_body_util::BodyExt;
use neumann_server::web::{self, AdminContext};
use relational_engine::{ColumnType, RelationalEngine, Schema};
use tower::ServiceExt;
use vector_engine::{VectorCollectionConfig, VectorEngine};

/// Create a test admin context with engines.
fn create_test_context() -> Arc<AdminContext> {
    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());
    Arc::new(AdminContext::new(relational, vector, graph))
}

/// Create a test context with sample data.
fn create_populated_context() -> Arc<AdminContext> {
    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());

    // Add a table
    let columns = vec![
        relational_engine::Column::new("id", ColumnType::Int),
        relational_engine::Column::new("name", ColumnType::String),
    ];
    let schema = Schema::new(columns);
    relational.create_table("users", schema).ok();

    // Add some vectors to default collection
    vector.store_embedding("vec1", vec![1.0, 0.0, 0.0]).ok();
    vector.store_embedding("vec2", vec![0.0, 1.0, 0.0]).ok();

    // Create a named collection
    let config = VectorCollectionConfig::default().with_dimension(3);
    vector.create_collection("test_coll", config).ok();
    vector
        .store_in_collection("test_coll", "p1", vec![1.0, 0.0, 0.0])
        .ok();

    // Add graph nodes and edges
    if let Ok(n1) = graph.create_node("Person", Default::default()) {
        if let Ok(n2) = graph.create_node("Person", Default::default()) {
            graph
                .create_edge(n1, n2, "KNOWS", Default::default(), true)
                .ok();
        }
    }

    Arc::new(AdminContext::new(relational, vector, graph))
}

/// Helper to get response body as string.
async fn body_string(body: Body) -> String {
    let bytes = body.collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ========== Dashboard Tests ==========

#[tokio::test]
async fn test_dashboard_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let html = body_string(response.into_body()).await;
    assert!(html.contains("NEUMANN"));
}

#[tokio::test]
async fn test_dashboard_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Relational Engine Tests ==========

#[tokio::test]
async fn test_relational_tables_list_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let html = body_string(response.into_body()).await;
    assert!(html.contains("TABLE") || html.contains("RELATIONAL"));
}

#[tokio::test]
async fn test_relational_tables_list_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let html = body_string(response.into_body()).await;
    assert!(html.contains("users") || html.contains("TABLE"));
}

#[tokio::test]
async fn test_relational_table_detail() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational/users")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_relational_table_detail_not_found() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_relational_table_rows() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational/users/rows")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_relational_table_rows_pagination() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/relational/users/rows?page=0&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Vector Engine Tests ==========

#[tokio::test]
async fn test_vector_collections_list_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collections_list_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_default_collection() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/_default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_default_points_list() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/_default/points")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collection_detail() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/test_coll")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collection_points() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/test_coll/points")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_point_detail() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/test_coll/points/p1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_default_search_form() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/_default/search")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collection_search_form() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/test_coll/search")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Graph Engine Tests ==========

#[tokio::test]
async fn test_graph_overview_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_overview_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_nodes_list_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/nodes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_nodes_list_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/nodes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_nodes_list_with_label_filter() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/nodes?label=Person")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_edges_list_empty() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/edges")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_edges_list_with_data() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/edges")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_edges_list_with_type_filter() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/edges?edge_type=KNOWS")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_path_finder_form() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/path")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_form() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_dashboard() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_dashboard_with_category() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard?category=centrality")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=pagerank")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form_unknown() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_api_subgraph() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/graph/subgraph?limit=50")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_string(response.into_body()).await;
    assert!(json.contains("nodes") || json.contains("links"));
}

#[tokio::test]
async fn test_graph_api_subgraph_with_center() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/graph/subgraph?center=1&depth=2&limit=50")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Metrics Tests ==========

#[tokio::test]
async fn test_metrics_dashboard() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_api() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Achievements Tests ==========

#[tokio::test]
async fn test_achievements_dashboard() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/achievements")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Test AdminContext Builders ==========

#[tokio::test]
async fn test_admin_context_with_metrics() {
    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());

    let ctx = AdminContext::new(relational, vector, graph).with_metrics(None);
    assert!(ctx.metrics.is_none());
}

#[tokio::test]
async fn test_admin_context_with_auth() {
    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());

    let ctx = AdminContext::new(relational, vector, graph).with_auth(None);
    assert!(ctx.auth_config.is_none());
}

// ========== Algorithm Execution Tests ==========

#[tokio::test]
async fn test_graph_algorithms_execute_pagerank() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(
                    "algorithm=pagerank&damping=0.85&tolerance=0.000001&max_iterations=100",
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_betweenness() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=betweenness&direction=both"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_closeness() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=closeness&direction=outgoing"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_eigenvector() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=eigenvector&max_iterations=50"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_degree() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=degree&direction=incoming"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_louvain() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=louvain&resolution=1.0"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_label_propagation() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=label_propagation&max_iterations=10"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_weakly_connected() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=weakly_connected"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_strongly_connected() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=strongly_connected"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_shortest_path() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=shortest_path&source=1&target=2"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_dijkstra() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=dijkstra&source=1&target=2"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_astar() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=astar&source=1&target=2"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_all_paths() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(
                    "algorithm=all_paths&source=1&target=2&max_depth=3",
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_kcore() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=kcore&k=2"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_mst() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=mst"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_triangles() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=triangles"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_biconnected() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=biconnected"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_similarity() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(
                    "algorithm=similarity&similarity_metric=jaccard&top_k=10",
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_unknown() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/algorithms/execute")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("algorithm=unknown_algorithm"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Algorithm Form Tests ==========

#[tokio::test]
async fn test_graph_algorithms_execute_form_betweenness() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=betweenness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form_louvain() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=louvain")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form_shortest_path() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=shortest_path")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form_triangles() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=triangles")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_execute_form_similarity() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/execute?algorithm=similarity")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Dashboard Category Tests ==========

#[tokio::test]
async fn test_graph_algorithms_dashboard_community() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard?category=community")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_dashboard_pathfinding() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard?category=pathfinding")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_dashboard_structure() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard?category=structure")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_algorithms_dashboard_similarity() {
    let ctx = create_test_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/algorithms/dashboard?category=similarity")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Vector Search POST Tests ==========

#[tokio::test]
async fn test_vector_default_search_submit() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/vector/_default/search")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("query_vector=1.0,0.0,0.0&k=5"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collection_search_submit() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/vector/test_coll/search")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("query_vector=1.0,0.0,0.0&k=5"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Graph Path Finder POST Tests ==========

#[tokio::test]
async fn test_graph_path_finder_submit() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graph/path")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("source=1&target=2"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Vector Points Pagination Tests ==========

#[tokio::test]
async fn test_vector_default_points_pagination() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/_default/points?page=0&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_vector_collection_points_pagination() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/vector/test_coll/points?page=0&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ========== Graph Pagination Tests ==========

#[tokio::test]
async fn test_graph_nodes_pagination() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/nodes?page=0&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graph_edges_pagination() {
    let ctx = create_populated_context();
    let app = web::router(ctx);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/graph/edges?page=0&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
