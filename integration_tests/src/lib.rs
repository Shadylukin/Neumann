//! Integration test helpers for Neumann.
//!
//! Provides utilities for setting up multi-engine test scenarios.

use std::sync::Arc;

use graph_engine::GraphEngine;
use query_router::{QueryResult, QueryRouter};
use relational_engine::RelationalEngine;
use tensor_cache::CacheConfig;
use tensor_store::TensorStore;
use vector_engine::VectorEngine;

/// Create a QueryRouter with all engines sharing a single TensorStore.
pub fn create_shared_router() -> QueryRouter {
    let store = TensorStore::new();
    QueryRouter::with_shared_store(store)
}

/// Create a QueryRouter with vault initialized.
pub fn create_router_with_vault(master_key: &[u8]) -> QueryRouter {
    let mut router = create_shared_router();
    router.init_vault(master_key).expect("vault init failed");
    router
}

/// Create a QueryRouter with cache initialized.
pub fn create_router_with_cache() -> QueryRouter {
    let mut router = create_shared_router();
    router.init_cache();
    router
}

/// Create a QueryRouter with cache initialized with custom embedding dimension.
pub fn create_router_with_cache_dim(dim: usize) -> QueryRouter {
    let mut router = create_shared_router();
    let config = CacheConfig {
        embedding_dim: dim,
        ..CacheConfig::default()
    };
    router.init_cache_with_config(config);
    router
}

/// Create a QueryRouter with blob store initialized.
pub fn create_router_with_blob() -> QueryRouter {
    let mut router = create_shared_router();
    router.init_blob().expect("blob init failed");
    router
}

/// Create a QueryRouter with all optional features enabled.
pub fn create_router_with_all_features(master_key: &[u8]) -> QueryRouter {
    let mut router = create_shared_router();
    router.init_vault(master_key).expect("vault init failed");
    router.init_cache_with_config(CacheConfig::default());
    router.init_blob().expect("blob init failed");
    router
}

/// Generate sample embeddings for testing.
pub fn sample_embeddings(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect())
        .collect()
}

/// Generate normalized sample embeddings (unit length).
pub fn sample_embeddings_normalized(count: usize, dim: usize) -> Vec<Vec<f32>> {
    sample_embeddings(count, dim)
        .into_iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect()
}

/// Get the shared TensorStore from a router's vector engine.
pub fn get_store_from_router(router: &QueryRouter) -> TensorStore {
    router.vector().store().clone()
}

/// Create engines with a shared store for direct engine testing.
pub fn create_shared_engines() -> (TensorStore, RelationalEngine, GraphEngine, VectorEngine) {
    let store = TensorStore::new();
    let relational = RelationalEngine::with_store(store.clone());
    let graph = GraphEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store.clone());
    (store, relational, graph, vector)
}

/// Create engines wrapped in Arc for concurrent testing.
pub fn create_shared_engines_arc() -> (
    TensorStore,
    Arc<RelationalEngine>,
    Arc<GraphEngine>,
    Arc<VectorEngine>,
) {
    let (store, relational, graph, vector) = create_shared_engines();
    (
        store,
        Arc::new(relational),
        Arc::new(graph),
        Arc::new(vector),
    )
}

// ========== Phase 6: New Test Helpers ==========

/// Create a router with test graph data (users and posts connected by edges).
pub fn create_test_graph_router() -> QueryRouter {
    let router = create_shared_router();

    // Create user nodes
    router.execute("NODE CREATE user name='Alice'").unwrap();
    router.execute("NODE CREATE user name='Bob'").unwrap();
    router.execute("NODE CREATE user name='Carol'").unwrap();

    // Create post nodes
    router.execute("NODE CREATE post title='Post1'").unwrap();
    router.execute("NODE CREATE post title='Post2'").unwrap();

    // Create edges (user wrote post)
    router.execute("EDGE CREATE 1 -> 4 wrote").unwrap();
    router.execute("EDGE CREATE 2 -> 5 wrote").unwrap();
    router.execute("EDGE CREATE 1 -> 2 follows").unwrap();
    router.execute("EDGE CREATE 2 -> 3 follows").unwrap();

    router
}

/// Create a router with test embeddings for similarity search.
pub fn create_test_vector_router(count: usize, dim: usize) -> QueryRouter {
    let router = create_shared_router();
    let embeddings = sample_embeddings(count, dim);

    for (i, emb) in embeddings.iter().enumerate() {
        let emb_str = emb
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        router
            .execute(&format!("EMBED doc:{} {}", i, emb_str))
            .unwrap();
    }

    router
}

/// Create a router with unified entities (graph + embeddings).
pub fn create_test_unified_router(count: usize) -> QueryRouter {
    let router = create_shared_router();
    let embeddings = sample_embeddings(count, 4);

    for (i, emb) in embeddings.iter().enumerate() {
        let emb_str = emb
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        router
            .execute_parsed(&format!(
                "ENTITY CREATE 'entity:{}' {{ idx: {} }} EMBEDDING [{}]",
                i, i, emb_str
            ))
            .unwrap();
    }

    router
}

/// Assert that a FIND result contains expected number of items.
pub fn assert_find_count(result: &QueryResult, expected: usize) {
    match result {
        QueryResult::Unified(unified) => {
            assert_eq!(unified.items.len(), expected);
        },
        QueryResult::Nodes(nodes) => {
            assert_eq!(nodes.len(), expected);
        },
        QueryResult::Edges(edges) => {
            assert_eq!(edges.len(), expected);
        },
        QueryResult::Rows(rows) => {
            assert_eq!(rows.len(), expected);
        },
        _ => {},
    }
}

/// Assert that a query result is not empty.
pub fn assert_result_not_empty(result: &QueryResult) -> bool {
    match result {
        QueryResult::Empty => false,
        QueryResult::Unified(u) => !u.items.is_empty(),
        QueryResult::Nodes(n) => !n.is_empty(),
        QueryResult::Edges(e) => !e.is_empty(),
        QueryResult::Rows(r) => !r.is_empty(),
        QueryResult::Similar(s) => !s.is_empty(),
        QueryResult::Ids(i) => !i.is_empty(),
        QueryResult::Count(c) => *c > 0,
        QueryResult::Value(v) => !v.is_empty(),
        QueryResult::Path(p) => !p.is_empty(),
        QueryResult::TableList(t) => !t.is_empty(),
        QueryResult::Blob(b) => !b.is_empty(),
        _ => true, // Other variants (ArtifactInfo, etc.) are considered not empty
    }
}

/// Format embedding as comma-separated string for query.
pub fn format_embedding(emb: &[f32]) -> String {
    emb.iter()
        .map(|v| format!("{:.4}", v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build EMBED BATCH query from key-embedding pairs.
pub fn build_embed_batch_query(items: &[(&str, &[f32])]) -> String {
    let formatted: Vec<String> = items
        .iter()
        .map(|(key, emb)| format!("('{}', [{}])", key, format_embedding(emb)))
        .collect();
    format!("EMBED BATCH [{}]", formatted.join(", "))
}
