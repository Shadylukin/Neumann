//! Integration tests for tensor_chain module.
//!
//! Tests the tensor-native blockchain functionality including:
//! - Chain creation and block operations
//! - Transaction workflows
//! - Query integration via QueryRouter

use query_router::{ChainResult, QueryResult, QueryRouter};
use neumann_parser::parse;

#[test]
fn test_chain_initialization() {
    let mut router = QueryRouter::new();

    // Chain should not be initialized initially
    let stmt = parse("CHAIN HEIGHT").unwrap();
    let result = router.execute_statement(&stmt);
    assert!(result.is_err());

    // Initialize chain
    router.init_chain("test_node").expect("Failed to init chain");

    // Now it should work
    let result = router.execute_statement(&stmt).unwrap();
    if let QueryResult::Chain(ChainResult::Height(h)) = result {
        assert_eq!(h, 0, "Fresh chain should have height 0");
    } else {
        panic!("Expected ChainResult::Height");
    }
}

#[test]
fn test_chain_genesis_block() {
    let mut router = QueryRouter::new();
    router.init_chain("genesis_test").unwrap();

    // Get tip - should be genesis
    let stmt = parse("CHAIN TIP").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Tip { height, hash }) = result {
        assert_eq!(height, 0);
        assert!(!hash.is_empty());
    } else {
        panic!("Expected ChainResult::Tip");
    }
}

#[test]
fn test_chain_verification() {
    let mut router = QueryRouter::new();
    router.init_chain("verify_test").unwrap();

    let stmt = parse("CHAIN VERIFY").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Verified { ok, errors }) = result {
        assert!(ok, "Fresh chain should verify");
        assert!(errors.is_empty(), "No errors expected");
    } else {
        panic!("Expected ChainResult::Verified");
    }
}

#[test]
fn test_chain_begin_transaction() {
    let mut router = QueryRouter::new();
    router.init_chain("tx_test").unwrap();

    // Begin a transaction
    let stmt = parse("BEGIN CHAIN TRANSACTION").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::TransactionBegun { tx_id }) = result {
        assert!(!tx_id.is_empty(), "Transaction ID should not be empty");
    } else {
        panic!("Expected ChainResult::TransactionBegun");
    }
}

#[test]
fn test_chain_history_empty() {
    let mut router = QueryRouter::new();
    router.init_chain("history_test").unwrap();

    let stmt = parse("CHAIN HISTORY 'nonexistent_key'").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::History(entries)) = result {
        assert!(entries.is_empty(), "No history for nonexistent key");
    } else {
        panic!("Expected ChainResult::History");
    }
}

#[test]
fn test_chain_drift_metrics() {
    let mut router = QueryRouter::new();
    router.init_chain("drift_test").unwrap();

    let stmt = parse("CHAIN DRIFT FROM 0 TO 100").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Drift(drift)) = result {
        assert_eq!(drift.from_height, 0);
        assert_eq!(drift.to_height, 100);
    } else {
        panic!("Expected ChainResult::Drift");
    }
}

#[test]
fn test_chain_block_not_found() {
    let mut router = QueryRouter::new();
    router.init_chain("block_test").unwrap();

    // Try to get a block that doesn't exist
    let stmt = parse("CHAIN BLOCK 999").unwrap();
    let result = router.execute_statement(&stmt);
    assert!(result.is_err(), "Block 999 should not exist");
}

#[test]
fn test_show_codebook_global() {
    let mut router = QueryRouter::new();
    router.init_chain("codebook_test").unwrap();

    let stmt = parse("SHOW CODEBOOK GLOBAL").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Codebook(info)) = result {
        assert_eq!(info.scope, "global");
    } else {
        panic!("Expected ChainResult::Codebook");
    }
}

#[test]
fn test_show_codebook_local() {
    let mut router = QueryRouter::new();
    router.init_chain("codebook_local_test").unwrap();

    let stmt = parse("SHOW CODEBOOK LOCAL 'users'").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Codebook(info)) = result {
        assert_eq!(info.scope, "local");
        assert_eq!(info.domain, Some("users".to_string()));
    } else {
        panic!("Expected ChainResult::Codebook");
    }
}

#[test]
fn test_analyze_transitions() {
    let mut router = QueryRouter::new();
    router.init_chain("analyze_test").unwrap();

    let stmt = parse("ANALYZE CODEBOOK TRANSITIONS").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::TransitionAnalysis(analysis)) = result {
        assert_eq!(analysis.total_transitions, 0);
        assert_eq!(analysis.valid_transitions, 0);
        assert_eq!(analysis.invalid_transitions, 0);
    } else {
        panic!("Expected ChainResult::TransitionAnalysis");
    }
}

#[test]
fn test_chain_similar_empty() {
    let mut router = QueryRouter::new();
    router.init_chain("similar_test").unwrap();

    let stmt = parse("CHAIN SIMILAR [1.0, 0.0, 0.0] LIMIT 10").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Similar(results)) = result {
        // No blocks yet, so no similar blocks
        assert!(results.is_empty());
    } else {
        panic!("Expected ChainResult::Similar");
    }
}

#[test]
fn test_rollback_chain() {
    let mut router = QueryRouter::new();
    router.init_chain("rollback_test").unwrap();

    let stmt = parse("ROLLBACK CHAIN TO 0").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::RolledBack { to_height }) = result {
        assert_eq!(to_height, 0);
    } else {
        panic!("Expected ChainResult::RolledBack");
    }
}

#[test]
fn test_commit_chain() {
    let mut router = QueryRouter::new();
    router.init_chain("commit_test").unwrap();

    let stmt = parse("COMMIT CHAIN").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Committed { height, .. }) = result {
        assert_eq!(height, 0); // Fresh chain at height 0
    } else {
        panic!("Expected ChainResult::Committed");
    }
}
