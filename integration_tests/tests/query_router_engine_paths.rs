// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for query_router engine dispatch paths.
//! Tests exercise CHAIN, CHECKPOINT, VAULT, CACHE, GRAPH ALGORITHM,
//! GRAPH CONSTRAINT, GRAPH INDEX, CLUSTER, CYPHER, and cursor paths.

use integration_tests::{
    create_router_with_blob, create_router_with_cache, create_router_with_vault,
    create_shared_router, create_test_graph_router,
};
use query_router::{PaginationOptions, QueryResult};
use tensor_vault::Vault;

// ========== CHAIN tests ==========

#[test]
fn test_chain_height() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN HEIGHT").unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_tip() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN TIP").unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_verify() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN VERIFY").unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_history() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN HISTORY 'some_key'").unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_drift() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN DRIFT FROM 0 TO 100").unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_similar() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router
        .execute_parsed("CHAIN SIMILAR [1.0, 2.0, 3.0] LIMIT 5")
        .unwrap();
    match result {
        QueryResult::Chain(_) => {},
        other => panic!("Expected Chain result, got {other:?}"),
    }
}

#[test]
fn test_chain_block_genesis() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    // Block 0 is the genesis block
    let result = router.execute_parsed("CHAIN BLOCK 0");
    // May succeed or fail depending on chain state, but should not panic
    let _ = result;
}

#[test]
fn test_chain_block_nonexistent() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    router.set_identity("user:integration");

    let result = router.execute_parsed("CHAIN BLOCK 99999");
    assert!(result.is_err());
}

#[test]
fn test_chain_not_initialized_errors() {
    let router = create_shared_router();
    let result = router.execute_parsed("CHAIN HEIGHT");
    assert!(result.is_err());
}

#[test]
fn test_chain_no_identity_errors() {
    let mut router = create_shared_router();
    router.init_chain("integration_node").unwrap();
    // No identity set
    let result = router.execute_parsed("CHAIN HEIGHT");
    assert!(result.is_err());
}

// ========== CHECKPOINT tests ==========

#[test]
fn test_checkpoint_create_and_rollback() {
    let mut router = create_router_with_blob();
    router.init_checkpoint().unwrap();

    // Insert some data via embeddings (avoids CREATE TABLE syntax issues)
    router.execute("EMBED cp_key 1.0, 2.0, 3.0").unwrap();

    // Create checkpoint
    let result = router.execute_parsed("CHECKPOINT 'before-change'").unwrap();
    let checkpoint_id = match &result {
        QueryResult::Value(v) => {
            assert!(v.contains("Checkpoint created"));
            v.replace("Checkpoint created: ", "")
        },
        other => panic!("Expected Value result, got {other:?}"),
    };

    // Modify data
    router.execute("EMBED cp_key 99.0, 99.0, 99.0").unwrap();

    // Rollback
    let result = router
        .execute_parsed(&format!("ROLLBACK TO '{checkpoint_id}'"))
        .unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("Rolled back")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_checkpoint_list() {
    let mut router = create_router_with_blob();
    router.init_checkpoint().unwrap();

    // Create a checkpoint
    router.execute_parsed("CHECKPOINT 'list-test'").unwrap();

    let result = router.execute_parsed("CHECKPOINTS").unwrap();
    match result {
        QueryResult::CheckpointList(list) => {
            assert!(!list.is_empty());
        },
        other => panic!("Expected CheckpointList result, got {other:?}"),
    }
}

#[test]
fn test_checkpoint_auto_name() {
    let mut router = create_router_with_blob();
    router.init_checkpoint().unwrap();

    // Checkpoint without explicit name
    let result = router.execute_parsed("CHECKPOINT").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("Checkpoint created")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_checkpoint_not_initialized_errors() {
    let router = create_shared_router();
    let result = router.execute_parsed("CHECKPOINT");
    assert!(result.is_err());
}

#[test]
fn test_rollback_not_initialized_errors() {
    let router = create_shared_router();
    let result = router.execute_parsed("ROLLBACK TO 'some-id'");
    assert!(result.is_err());
}

// ========== VAULT tests ==========

#[test]
fn test_vault_put_and_get() {
    let mut router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    router.set_identity(Vault::ROOT);

    router
        .execute_parsed("VAULT SET 'my_secret' 'hunter2'")
        .unwrap();
    let result = router.execute_parsed("VAULT GET 'my_secret'").unwrap();
    match result {
        QueryResult::Value(v) => assert_eq!(v, "hunter2"),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_vault_delete() {
    let mut router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    router.set_identity(Vault::ROOT);

    router
        .execute_parsed("VAULT SET 'del_key' 'del_val'")
        .unwrap();
    router.execute_parsed("VAULT DELETE 'del_key'").unwrap();
    let result = router.execute_parsed("VAULT GET 'del_key'");
    assert!(result.is_err());
}

#[test]
fn test_vault_list() {
    let mut router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    router.set_identity(Vault::ROOT);

    router.execute_parsed("VAULT SET 'list_a' 'val_a'").unwrap();
    router.execute_parsed("VAULT SET 'list_b' 'val_b'").unwrap();

    let result = router.execute_parsed("VAULT LIST").unwrap();
    match result {
        QueryResult::Value(v) => {
            assert!(v.contains("list_a"));
            assert!(v.contains("list_b"));
        },
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_vault_rotate() {
    let mut router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    router.set_identity(Vault::ROOT);

    router
        .execute_parsed("VAULT SET 'rotate_k' 'old_value'")
        .unwrap();
    router
        .execute_parsed("VAULT ROTATE 'rotate_k' 'new_value'")
        .unwrap();

    let result = router.execute_parsed("VAULT GET 'rotate_k'").unwrap();
    match result {
        QueryResult::Value(v) => assert_eq!(v, "new_value"),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_vault_no_identity_errors() {
    let router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    // No identity set
    let result = router.execute_parsed("VAULT SET 'k' 'v'");
    assert!(result.is_err());
}

#[test]
fn test_vault_get_nonexistent_errors() {
    let mut router = create_router_with_vault(b"test-master-key-32-bytes-long!!");
    router.set_identity(Vault::ROOT);

    let result = router.execute_parsed("VAULT GET 'does_not_exist'");
    assert!(result.is_err());
}

// ========== CACHE tests ==========

#[test]
fn test_cache_put_and_get() {
    let mut router = create_router_with_cache();
    router.set_identity("user:cache_test");

    let result = router
        .execute_parsed("CACHE PUT 'cache_key' 'cache_value'")
        .unwrap();
    assert!(matches!(result, QueryResult::Value(ref s) if s == "OK"));

    let result = router.execute_parsed("CACHE GET 'cache_key'").unwrap();
    match result {
        QueryResult::Value(v) => assert_eq!(v, "cache_value"),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cache_get_miss() {
    let mut router = create_router_with_cache();
    router.set_identity("user:cache_test");

    let result = router
        .execute_parsed("CACHE GET 'nonexistent_key'")
        .unwrap();
    match result {
        QueryResult::Value(v) => assert_eq!(v, "(not found)"),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cache_stats() {
    let mut router = create_router_with_cache();
    router.set_identity("user:cache_test");

    let result = router.execute_parsed("CACHE STATS").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("hits") || v.contains("Cache")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cache_clear() {
    let mut router = create_router_with_cache();
    router.set_identity("user:cache_test");

    router
        .execute_parsed("CACHE PUT 'to_clear' 'value'")
        .unwrap();
    let result = router.execute_parsed("CACHE CLEAR").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("cleared") || v.contains("Cache")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cache_init_command() {
    let mut router = create_router_with_cache();
    router.set_identity("user:cache_test");

    let result = router.execute_parsed("CACHE INIT").unwrap();
    match result {
        QueryResult::Value(v) => assert!(!v.is_empty()),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

// ========== GRAPH ALGORITHM tests ==========

#[test]
fn test_graph_pagerank() {
    let router = create_test_graph_router();

    let result = router.execute_parsed("GRAPH PAGERANK").unwrap();
    match result {
        QueryResult::PageRank(pr) => {
            assert!(!pr.items.is_empty());
        },
        other => panic!("Expected PageRank result, got {other:?}"),
    }
}

#[test]
fn test_graph_betweenness_centrality() {
    let router = create_test_graph_router();

    let result = router
        .execute_parsed("GRAPH BETWEENNESS CENTRALITY")
        .unwrap();
    match result {
        QueryResult::Centrality(c) => {
            assert!(!c.items.is_empty());
        },
        other => panic!("Expected Centrality result, got {other:?}"),
    }
}

#[test]
fn test_graph_closeness_centrality() {
    let router = create_test_graph_router();

    let result = router.execute_parsed("GRAPH CLOSENESS CENTRALITY").unwrap();
    match result {
        QueryResult::Centrality(c) => {
            assert!(!c.items.is_empty());
        },
        other => panic!("Expected Centrality result, got {other:?}"),
    }
}

#[test]
fn test_graph_pagerank_with_edge_type() {
    let router = create_test_graph_router();

    let result = router
        .execute_parsed("GRAPH PAGERANK EDGE TYPE follows")
        .unwrap();
    match result {
        QueryResult::PageRank(pr) => {
            let _ = pr;
        },
        other => panic!("Expected PageRank result, got {other:?}"),
    }
}

// ========== GRAPH CONSTRAINT tests ==========

#[test]
fn test_graph_constraint_create_unique() {
    let router = create_shared_router();

    router.execute("NODE CREATE User name='Alice'").unwrap();

    let result = router
        .execute_parsed("CONSTRAINT CREATE email_uniq ON NODE User PROPERTY email UNIQUE")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty | QueryResult::Value(_)));
}

#[test]
fn test_graph_constraint_create_exists() {
    let router = create_shared_router();

    router.execute("NODE CREATE Person name='Bob'").unwrap();

    let result = router
        .execute_parsed("CONSTRAINT CREATE name_exists ON NODE Person PROPERTY name EXISTS")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty | QueryResult::Value(_)));
}

#[test]
fn test_graph_constraint_list() {
    let router = create_shared_router();

    let result = router.execute_parsed("CONSTRAINT LIST");
    // Should succeed even with no constraints
    let _ = result;
}

// ========== GRAPH INDEX tests ==========

#[test]
fn test_graph_index_create_node_property() {
    let router = create_shared_router();

    router.execute("NODE CREATE Person name='Test'").unwrap();

    let result = router
        .execute_parsed("GRAPH INDEX CREATE ON NODE PROPERTY name")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty));
}

#[test]
fn test_graph_index_create_edge_property() {
    let router = create_test_graph_router();

    let result = router
        .execute_parsed("GRAPH INDEX CREATE ON EDGE PROPERTY weight")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty));
}

#[test]
fn test_graph_index_create_label() {
    let router = create_shared_router();

    let result = router
        .execute_parsed("GRAPH INDEX CREATE ON LABEL")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty));
}

#[test]
fn test_graph_index_show_node() {
    let router = create_shared_router();

    let result = router.execute_parsed("GRAPH INDEX SHOW ON NODE").unwrap();
    match result {
        QueryResult::GraphIndexes(_) => {},
        other => panic!("Expected GraphIndexes result, got {other:?}"),
    }
}

#[test]
fn test_graph_index_create_edge_type() {
    let router = create_shared_router();

    let result = router
        .execute_parsed("GRAPH INDEX CREATE ON EDGE TYPE")
        .unwrap();
    assert!(matches!(result, QueryResult::Empty));
}

// ========== CLUSTER tests ==========

#[test]
fn test_cluster_status_single_node() {
    let router = create_shared_router();

    let result = router.execute("CLUSTER STATUS").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("single-node")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cluster_nodes_single_node() {
    let router = create_shared_router();

    let result = router.execute("CLUSTER NODES").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("single-node") || v.contains("No cluster")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cluster_leader_single_node() {
    let router = create_shared_router();

    let result = router.execute("CLUSTER LEADER").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("single-node") || v.contains("No leader")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_cluster_disconnect_no_cluster() {
    let router = create_shared_router();

    let result = router.execute_parsed("CLUSTER DISCONNECT");
    assert!(result.is_err());
}

// ========== CYPHER tests ==========

#[test]
fn test_cypher_create_node() {
    let router = create_shared_router();

    let result = router.execute_parsed("CREATE (n:Animal {species: 'Dog'})");
    // Should succeed or at least not panic
    let _ = result;
}

#[test]
fn test_cypher_match_nodes() {
    let router = create_test_graph_router();

    let result = router.execute_parsed("MATCH (n:user) RETURN n");
    // Cypher MATCH may return nodes or empty depending on implementation
    let _ = result;
}

#[test]
fn test_cypher_match_with_where() {
    let router = create_test_graph_router();

    let result = router.execute_parsed("MATCH (n:user) WHERE n.name = 'Alice' RETURN n");
    let _ = result;
}

#[test]
fn test_cypher_delete() {
    let router = create_shared_router();
    router
        .execute("NODE CREATE test_label name='ToDelete'")
        .unwrap();

    let result = router.execute_parsed("MATCH (n:test_label) DELETE n");
    let _ = result;
}

// ========== PAGINATION tests ==========

#[test]
fn test_pagination_first_page() {
    let router = create_shared_router();

    // Create table with correct syntax (name:type)
    router
        .execute("CREATE TABLE paged (id:int, val:string)")
        .unwrap();
    for i in 0..25 {
        router
            .execute_parsed(&format!("INSERT INTO paged VALUES ({i}, 'row_{i}')"))
            .unwrap();
    }

    let options = PaginationOptions {
        page_size: Some(10),
        count_total: true,
        ..Default::default()
    };
    let paged = router
        .execute_paginated("SELECT * FROM paged", options)
        .unwrap();

    match &paged.result {
        QueryResult::Rows(rows) => assert_eq!(rows.len(), 10),
        other => panic!("Expected Rows result, got {other:?}"),
    }
    assert!(paged.has_more);
    assert_eq!(paged.total_count, Some(25));
    assert!(paged.next_cursor.is_some());
}

#[test]
fn test_pagination_with_cursor() {
    let router = create_shared_router();

    router
        .execute("CREATE TABLE paged2 (id:int, val:string)")
        .unwrap();
    for i in 0..15 {
        router
            .execute_parsed(&format!("INSERT INTO paged2 VALUES ({i}, 'item_{i}')"))
            .unwrap();
    }

    // First page
    let options = PaginationOptions {
        page_size: Some(10),
        count_total: true,
        ..Default::default()
    };
    let first_page = router
        .execute_paginated("SELECT * FROM paged2", options)
        .unwrap();
    assert!(first_page.has_more);
    let next_cursor = first_page.next_cursor.clone().unwrap();

    // Second page using cursor
    let options2 = PaginationOptions {
        cursor: Some(next_cursor),
        page_size: Some(10),
        count_total: true,
        ..Default::default()
    };
    let second_page = router
        .execute_paginated("SELECT * FROM paged2", options2)
        .unwrap();

    match &second_page.result {
        QueryResult::Rows(rows) => assert_eq!(rows.len(), 5),
        other => panic!("Expected Rows result, got {other:?}"),
    }
    assert!(!second_page.has_more);
}

#[test]
fn test_pagination_empty_result() {
    let router = create_shared_router();

    router.execute("CREATE TABLE paged_empty (id:int)").unwrap();

    let options = PaginationOptions {
        page_size: Some(10),
        ..Default::default()
    };
    let paged = router
        .execute_paginated("SELECT * FROM paged_empty", options)
        .unwrap();

    match &paged.result {
        QueryResult::Rows(rows) => assert!(rows.is_empty()),
        other => panic!("Expected Rows result, got {other:?}"),
    }
    assert!(!paged.has_more);
}

// ========== SHOW commands ==========

#[test]
fn test_show_tables() {
    let router = create_shared_router();

    router.execute("CREATE TABLE show_test (id:int)").unwrap();

    let result = router.execute_parsed("SHOW TABLES").unwrap();
    match result {
        QueryResult::TableList(tables) => {
            assert!(tables.contains(&"show_test".to_string()));
        },
        other => panic!("Expected TableList result, got {other:?}"),
    }
}

#[test]
fn test_show_embeddings() {
    let router = create_shared_router();
    router.execute("EMBED vec_a 1.0, 2.0, 3.0").unwrap();

    let result = router.execute_parsed("SHOW EMBEDDINGS").unwrap();
    match result {
        QueryResult::Value(v) => assert!(v.contains("vec_a")),
        other => panic!("Expected Value result, got {other:?}"),
    }
}

#[test]
fn test_count_embeddings() {
    let router = create_shared_router();
    router.execute("EMBED count_a 1.0, 2.0").unwrap();
    router.execute("EMBED count_b 3.0, 4.0").unwrap();

    let result = router.execute_parsed("COUNT EMBEDDINGS").unwrap();
    match result {
        QueryResult::Count(c) => assert!(c >= 2),
        other => panic!("Expected Count result, got {other:?}"),
    }
}

// ========== Combined multi-feature tests ==========

#[test]
fn test_checkpoint_roundtrip_with_data() {
    let mut router = create_router_with_blob();
    router.init_checkpoint().unwrap();

    // Store embedding data
    router.execute("EMBED cp_emb 1.0, 2.0, 3.0").unwrap();

    // Create checkpoint
    let cp_result = router.execute_parsed("CHECKPOINT 'embed-snap'").unwrap();
    let checkpoint_id = match cp_result {
        QueryResult::Value(v) => v.replace("Checkpoint created: ", ""),
        other => panic!("Expected Value, got {other:?}"),
    };

    // Overwrite the embedding
    router.execute("EMBED cp_emb 99.0, 99.0, 99.0").unwrap();

    // Rollback
    let rb = router
        .execute_parsed(&format!("ROLLBACK TO '{checkpoint_id}'"))
        .unwrap();
    match rb {
        QueryResult::Value(v) => assert!(v.contains("Rolled back")),
        other => panic!("Expected Value, got {other:?}"),
    }
}

#[test]
fn test_vault_with_cache_integration() {
    let mut router = create_shared_router();
    router
        .init_vault(b"test-master-key-32-bytes-long!!")
        .unwrap();
    router.init_cache();
    router.set_identity(Vault::ROOT);

    // Store secret via vault
    router
        .execute_parsed("VAULT SET 'combo_key' 'combo_val'")
        .unwrap();

    // Also use cache
    router
        .execute_parsed("CACHE PUT 'cached_key' 'cached_val'")
        .unwrap();

    // Both should be independently retrievable
    let vault_result = router.execute_parsed("VAULT GET 'combo_key'").unwrap();
    match vault_result {
        QueryResult::Value(v) => assert_eq!(v, "combo_val"),
        other => panic!("Expected Value, got {other:?}"),
    }

    let cache_result = router.execute_parsed("CACHE GET 'cached_key'").unwrap();
    match cache_result {
        QueryResult::Value(v) => assert_eq!(v, "cached_val"),
        other => panic!("Expected Value, got {other:?}"),
    }
}
