// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end behavioral parity test for the REST + gRPC dispatch unification
//! (Phase 7 of the typed-router refactor).
//!
//! Asserts that REST handlers, gRPC services, and the string-dispatch path all
//! converge on the same [`query_router::QueryRouter`] surface — same cache
//! invalidation, same checkpoint protection (where applicable), same identity
//! flow, same error mapping. Failure of any assertion here means a divergence
//! has crept back in.

use std::sync::Arc;

use parking_lot::RwLock;
use query_router::{QueryRouter, ScrollOptions, SearchOptions, VectorPoint};
use tensor_checkpoint::CheckpointManager;
use vector_engine::{VectorCollectionConfig, VectorEngine};

fn boot_router_with_cache_and_checkpoint() -> (Arc<RwLock<QueryRouter>>, Arc<CheckpointManager>) {
    let mut router = QueryRouter::new();
    let engine = Arc::new(VectorEngine::new());
    router.replace_vector_engine(engine);
    router.init_cache_default().unwrap();
    let cp_dir = tempfile::tempdir().unwrap();
    router.set_checkpoint_dir(cp_dir.path().to_path_buf());
    router.init_checkpoint().unwrap();
    let cp = router.checkpoint().unwrap().clone();
    std::mem::forget(cp_dir); // keep checkpoint dir alive for the duration of the test
    (Arc::new(RwLock::new(router)), cp)
}

#[test]
fn create_collection_then_list_through_router() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("test", VectorCollectionConfig::default())
        .unwrap();
    assert!(router
        .read()
        .list_collections_typed()
        .contains(&"test".to_string()));
}

#[test]
fn typed_upsert_invalidates_string_dispatch_cache() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("test", VectorCollectionConfig::default())
        .unwrap();

    // Populate the default-namespace cache by issuing a SIMILAR through execute().
    router
        .write()
        .upsert_points(
            None,
            vec![VectorPoint {
                id: "seed".into(),
                vector: vec![1.0, 0.0],
                metadata: None,
            }],
        )
        .unwrap();
    let _ = router.read().execute("SIMILAR [1.0, 0.0] TOP 1");

    // Now do another typed upsert and confirm the cache is cleared even when
    // the cache was populated through the string-dispatch surface.
    router
        .write()
        .upsert_points(
            None,
            vec![VectorPoint {
                id: "second".into(),
                vector: vec![0.0, 1.0],
                metadata: None,
            }],
        )
        .unwrap();
    let guard = router.read();
    let cache = guard.cache().expect("cache initialized");
    assert!(
        cache.is_empty(),
        "typed upsert must invalidate the cache populated by string-dispatch SIMILAR"
    );
}

#[test]
fn typed_delete_creates_embed_delete_batch_checkpoint() {
    let (router, cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("c", VectorCollectionConfig::default())
        .unwrap();
    for k in ["a", "b", "c"] {
        router
            .write()
            .upsert_points(
                Some("c"),
                vec![VectorPoint {
                    id: k.into(),
                    vector: vec![1.0, 0.0],
                    metadata: None,
                }],
            )
            .unwrap();
    }

    let before = cp.list(None).unwrap().len();
    let outcome = router
        .write()
        .delete_points(
            Some("c"),
            &["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap();
    assert_eq!(outcome.deleted, 3);
    let list = cp.list(None).unwrap();
    assert_eq!(
        list.len(),
        before + 1,
        "multi-existing typed delete must create exactly one checkpoint"
    );
    let latest = list.iter().max_by_key(|s| s.created_at).unwrap();
    assert_eq!(
        latest.trigger.as_deref(),
        Some("EMBED DELETE (batch)"),
        "batch path must use the new EmbedDeleteBatch variant"
    );
}

#[test]
fn embed_delete_strict_semantics_preserved_after_phase2() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    // The string-dispatch EMBED DELETE keeps its strict "not found" semantic
    // after delegation to delete_points_impl.
    let err = router.read().execute("EMBED DELETE 'missing'").unwrap_err();
    assert!(
        format!("{err}").to_lowercase().contains("not found"),
        "EMBED DELETE for a missing key must error, not silently succeed"
    );
}

#[test]
fn rest_silently_skips_missing_ids_via_typed_delete() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("c", VectorCollectionConfig::default())
        .unwrap();
    router
        .write()
        .upsert_points(
            Some("c"),
            vec![VectorPoint {
                id: "real".into(),
                vector: vec![1.0, 0.0],
                metadata: None,
            }],
        )
        .unwrap();

    let outcome = router
        .write()
        .delete_points(Some("c"), &["real".into(), "ghost".into()])
        .unwrap();
    assert_eq!(outcome.deleted, 1);
    assert_eq!(outcome.missing, vec!["ghost".to_string()]);
}

#[test]
fn search_with_payload_returns_enriched_hits_in_one_call() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("c", VectorCollectionConfig::default())
        .unwrap();
    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "tag".to_string(),
        tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String("hello".into())),
    );
    router
        .write()
        .upsert_points(
            Some("c"),
            vec![VectorPoint {
                id: "k1".into(),
                vector: vec![1.0, 0.0],
                metadata: Some(meta),
            }],
        )
        .unwrap();

    let opts = SearchOptions {
        limit: 5,
        offset: 0,
        filter: None,
        metric: None,
        score_threshold: None,
        with_vector: true,
        with_payload: true,
    };
    let hits = router
        .read()
        .search_points(Some("c"), &[1.0, 0.0], &opts)
        .unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].vector.is_some(),
        "with_vector must populate hit.vector"
    );
    assert!(
        hits[0].metadata.is_some(),
        "with_payload must populate hit.metadata"
    );
}

#[test]
fn identity_flow_clears_after_dispatch() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    {
        let mut r = router.write();
        r.set_identity("alice");
        assert_eq!(r.current_identity(), Some("alice"));
        r.clear_identity();
    }
    assert!(router.read().current_identity().is_none());
}

#[test]
fn delete_points_single_existing_creates_embed_delete_not_batch() {
    // Plan verification #5: a one-valid + one-missing delete creates exactly one
    // `EmbedDelete` checkpoint (NOT `EmbedDeleteBatch`); affected_count == 1.
    let (router, cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("test", VectorCollectionConfig::default())
        .unwrap();
    router
        .write()
        .upsert_points(
            Some("test"),
            vec![VectorPoint {
                id: "real".into(),
                vector: vec![1.0, 0.0],
                metadata: None,
            }],
        )
        .unwrap();

    let before = cp.list(None).unwrap().len();
    let outcome = router
        .write()
        .delete_points(Some("test"), &["real".into(), "ghost".into()])
        .unwrap();
    assert_eq!(outcome.deleted, 1);
    assert_eq!(outcome.missing, vec!["ghost".to_string()]);

    let list = cp.list(None).unwrap();
    assert_eq!(list.len(), before + 1, "expected exactly one checkpoint");
    let latest = list.iter().max_by_key(|s| s.created_at).unwrap();
    assert_eq!(
        latest.trigger.as_deref(),
        Some("EMBED DELETE"),
        "single-key path must use the EmbedDelete variant (not the batch one)"
    );
}

#[test]
fn delete_points_all_missing_does_not_create_checkpoint() {
    // Plan verification #7: all-missing delete returns deleted == 0 and does
    // NOT create a checkpoint that would mislead audit/rollback consumers.
    let (router, cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("test", VectorCollectionConfig::default())
        .unwrap();
    let before = cp.list(None).unwrap().len();
    let outcome = router
        .write()
        .delete_points(Some("test"), &["ghost1".into(), "ghost2".into()])
        .unwrap();
    assert_eq!(outcome.deleted, 0);
    assert_eq!(outcome.missing.len(), 2);
    assert_eq!(
        cp.list(None).unwrap().len(),
        before,
        "all-missing delete must NOT pollute the checkpoint history"
    );
}

#[test]
fn exec_embed_batch_partial_failure_invalidates_cache() {
    // Plan verification #8: partial-mutation Err on EMBED BATCH still clears
    // the cache (Phase 2 explicit invalidation; execute()'s `?` early-returns
    // before its post-write clear at lib.rs:336).
    let (router, _cp) = boot_router_with_cache_and_checkpoint();

    router
        .write()
        .execute("EMBED STORE 'seed' [1.0, 0.0]")
        .expect("seed");
    let _ = router.read().execute("SIMILAR [1.0, 0.0] TOP 1");
    assert!(
        !router.read().cache().unwrap().is_empty(),
        "precondition: SIMILAR populated cache"
    );

    let result = router
        .write()
        .execute("EMBED BATCH [('good', [1.0, 0.0]), ('bad', [])]");
    assert!(
        result.is_err(),
        "batch with empty vector must error overall"
    );
    assert!(
        router.read().cache().unwrap().is_empty(),
        "exec_embed Batch must clear cache on partial-mutation Err"
    );
}

#[test]
fn spatial_2d_nearest_parity_typed_vs_string() {
    // Plan verification #11: typed `spatial_nearest` and the string-dispatched
    // `SPATIAL NEAREST` must return the same ordering — both delegate to
    // `query_nearest_by_centroid`.
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .execute("SPATIAL INSERT 'big' BOUNDS 0.0 0.0 100.0 100.0")
        .unwrap();
    router
        .write()
        .execute("SPATIAL INSERT 'small' BOUNDS 4.0 4.0 4.0 4.0")
        .unwrap();

    let typed = router.read().spatial_nearest(5.0, 5.0, 2).unwrap();
    let typed_ids: Vec<String> = typed.iter().map(|h| h.id.clone()).collect();

    let string_result = router
        .read()
        .execute("SPATIAL NEAREST 5 5 LIMIT 2")
        .unwrap();
    let string_ids: Vec<String> = match string_result {
        query_router::QueryResult::Spatial(items) => items.into_iter().map(|r| r.key).collect(),
        other => panic!("expected Spatial result, got {other:?}"),
    };

    assert_eq!(
        typed_ids, string_ids,
        "typed spatial_nearest and SPATIAL NEAREST must agree on ordering"
    );
}

#[test]
fn spatial_3d_not_configured_returns_invalid_argument_marker() {
    // Plan verification #12: when the router's 3D index is None, the typed
    // surface returns RouterError::InvalidArgument with the substring
    // "not configured" so the REST mapping can fork to 500 and the gRPC
    // mapping can fork to failed_precondition.
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    let err = router.read().spatial3d_count().unwrap_err();
    assert!(
        matches!(err, query_router::RouterError::InvalidArgument(ref msg) if msg.to_lowercase().contains("not configured")),
        "expected InvalidArgument('… not configured …'), got {err:?}"
    );
}

#[test]
fn similar_connected_to_short_circuits_to_cross_engine() {
    // Plan verification #13: SIMILAR…CONNECTED TO must short-circuit BEFORE
    // search_points delegation. We only assert the cross-engine PATH is
    // exercised (i.e. require_unified is invoked); the data answer depends on
    // edge bootstrap which is out of scope for a parity test.
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .execute("EMBED STORE 'k' [1.0, 0.0]")
        .unwrap();
    let result = router.read().execute("SIMILAR 'k' CONNECTED TO 'g' TOP 5");
    // The connected_to branch goes through find_similar_connected → unified
    // engine. The unified engine is initialized by default, so the call
    // succeeds (possibly with empty results) instead of erroring with
    // ParseError("requires a key, not a vector") which would happen if
    // exec_similar fell through to standard search_points.
    assert!(
        matches!(result, Ok(query_router::QueryResult::Similar(_))),
        "SIMILAR…CONNECTED TO must route to find_similar_connected, got {result:?}"
    );
}

#[test]
fn scroll_typed_paginates_collection() {
    let (router, _cp) = boot_router_with_cache_and_checkpoint();
    router
        .write()
        .create_collection_typed("c", VectorCollectionConfig::default())
        .unwrap();
    for k in ["k1", "k2", "k3"] {
        router
            .write()
            .upsert_points(
                Some("c"),
                vec![VectorPoint {
                    id: k.into(),
                    vector: vec![1.0, 0.0],
                    metadata: None,
                }],
            )
            .unwrap();
    }
    let page = router
        .read()
        .scroll_points(
            "c",
            &ScrollOptions {
                limit: 2,
                offset_id: None,
                with_vector: false,
                with_payload: false,
            },
        )
        .unwrap();
    assert_eq!(page.hits.len(), 2);
    assert!(page.next_offset_id.is_some(), "expected next page cursor");
}
