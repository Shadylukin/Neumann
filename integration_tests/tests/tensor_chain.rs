// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for tensor_chain module.
//!
//! Tests the tensor-native blockchain functionality including:
//! - Chain creation and block operations
//! - Transaction workflows
//! - Query integration via QueryRouter

use neumann_parser::parse;
use query_router::{ChainResult, QueryResult, QueryRouter};

#[test]
fn test_chain_initialization() {
    let mut router = QueryRouter::new();
    router.set_identity("test-user");

    // Chain should not be initialized initially
    let stmt = parse("CHAIN HEIGHT").unwrap();
    let result = router.execute_statement(&stmt);
    assert!(result.is_err());

    // Initialize chain
    router
        .init_chain("test_node")
        .expect("Failed to init chain");

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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
    router.init_chain("block_test").unwrap();

    // Try to get a block that doesn't exist
    let stmt = parse("CHAIN BLOCK 999").unwrap();
    let result = router.execute_statement(&stmt);
    assert!(result.is_err(), "Block 999 should not exist");
}

#[test]
fn test_show_codebook_global() {
    let mut router = QueryRouter::new();
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
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
    router.set_identity("test-user");
    router.init_chain("commit_test").unwrap();

    let stmt = parse("COMMIT CHAIN").unwrap();
    let result = router.execute_statement(&stmt).unwrap();

    if let QueryResult::Chain(ChainResult::Committed { height, .. }) = result {
        assert_eq!(height, 0); // Fresh chain at height 0
    } else {
        panic!("Expected ChainResult::Committed");
    }
}

#[test]
fn test_codebook_persistence_roundtrip() {
    use tensor_chain::{
        ChainConfig, CodebookConfig, GlobalCodebook, TensorChain, ValidationConfig,
    };
    use tensor_store::TensorStore;

    let store = TensorStore::new();

    // Create chain with custom codebook
    let centroids = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let global = GlobalCodebook::from_centroids(centroids);
    let config = ChainConfig::new("persist_test");

    let chain = TensorChain::with_codebook(
        store.clone(),
        config.clone(),
        global,
        CodebookConfig::default(),
        ValidationConfig::default(),
    );
    chain.initialize().unwrap();

    // Save codebook
    let saved_count = chain.save_global_codebook().unwrap();
    assert_eq!(saved_count, 4);

    // Create a new chain using load_or_create - should load existing codebook
    let chain2 = TensorChain::load_or_create(store, config);

    assert_eq!(chain2.codebook_manager().global().len(), 4);
    assert_eq!(chain2.codebook_manager().global().dimension(), 4);
}

#[test]
fn test_transaction_with_codebook_quantization() {
    use tensor_chain::{
        ChainConfig, CodebookConfig, GlobalCodebook, TensorChain, Transaction, ValidationConfig,
    };
    use tensor_store::TensorStore;

    let store = TensorStore::new();

    // Create codebook with orthogonal centroids
    let centroids = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let global = GlobalCodebook::from_centroids(centroids);
    let config = ChainConfig::new("quant_test").with_auto_merge(false);

    let chain = TensorChain::with_codebook(
        store,
        config,
        global,
        CodebookConfig::default(),
        ValidationConfig::default(),
    );
    chain.initialize().unwrap();

    // Create transaction with delta close to first centroid
    let tx = chain.begin().unwrap();
    tx.add_operation(Transaction::Put {
        key: "test_key".to_string(),
        data: vec![1, 2, 3],
    })
    .unwrap();
    tx.set_before_embedding(&[0.0, 0.0, 0.0, 0.0]);
    tx.compute_delta(&[0.95, 0.05, 0.0, 0.0]); // Close to [1,0,0,0]

    chain.commit(&tx).unwrap();

    // Verify block has quantized codes
    let block = chain.get_tip().unwrap().unwrap();
    assert!(!block.header.quantized_codes.is_empty());
    assert_eq!(block.header.quantized_codes[0], 0); // Nearest centroid is 0
}

#[test]
fn test_load_or_create_fresh_store() {
    use tensor_chain::{ChainConfig, TensorChain};
    use tensor_store::TensorStore;

    let store = TensorStore::new();
    let config = ChainConfig::new("fresh_test");

    // Fresh store should create empty codebook
    let chain = TensorChain::load_or_create(store, config);

    assert_eq!(chain.codebook_manager().global().len(), 0);
    // Default dimension is 128 (not 4)
    assert_eq!(chain.codebook_manager().global().dimension(), 128);
}

#[test]
fn test_codebook_manager_accessor() {
    use tensor_chain::TensorChain;
    use tensor_store::TensorStore;

    let store = TensorStore::new();
    let chain = TensorChain::new(store, "accessor_test");

    let manager = chain.codebook_manager();
    // Default dimension is 128 (not 4)
    assert_eq!(manager.global().dimension(), 128);
    assert!(manager.global().is_empty());
}

#[test]
fn test_transition_validator_with_codebook() {
    use std::sync::Arc;

    use tensor_chain::{GlobalCodebook, TransitionValidator, ValidationConfig};

    // Create codebook with single centroid
    let centroids = vec![vec![1.0, 0.0, 0.0]];
    let global = Arc::new(GlobalCodebook::from_centroids(centroids));
    let config = ValidationConfig {
        state_threshold: 0.9,
        max_transition_magnitude: 0.5,
        strict_transition: true,
        codebook_config: Default::default(),
    };
    let validator = TransitionValidator::new(global, config);

    // Valid transition (close to centroid)
    let validation = validator.validate_transition("test", &[1.0, 0.0, 0.0], &[0.95, 0.05, 0.0]);
    assert!(validation.is_valid);
    assert!(validation.magnitude < 0.5);

    // Invalid transition (too far from centroid)
    let validation = validator.validate_transition("test", &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!(!validation.is_valid);
}

#[test]
fn test_state_machine_transaction_persistence() {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_chain::{
        compute_state_root, network::MemoryTransport, Block, BlockHeader, Chain, RaftConfig,
        RaftNode, TensorStateMachine, Transaction,
    };
    use tensor_store::{SparseVector, TensorStore, TensorValue};

    // Set up components
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
    chain.initialize().unwrap();

    // Get genesis block hash for valid prev_hash
    let genesis = chain.get_tip().unwrap().unwrap();
    let prev_hash = genesis.hash();

    let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
    let raft = Arc::new(RaftNode::new(
        "test_node".to_string(),
        vec![],
        transport,
        RaftConfig::default(),
    ));

    let state_machine = TensorStateMachine::new(chain.clone(), raft, store.clone());

    // Manually apply the transaction to compute the correct state_root
    let mut data = tensor_store::TensorData::new();
    data.set(
        "data",
        TensorValue::Scalar(tensor_store::ScalarValue::Bytes(vec![42, 43, 44])),
    );
    store.put("test_key", data).unwrap();
    let state_root = compute_state_root(&store).unwrap();

    // Create a block with Put transaction using valid prev_hash and computed state_root
    let block = Block {
        header: BlockHeader {
            height: 1,
            prev_hash,
            tx_root: [0u8; 32],
            state_root,
            timestamp: 0,
            proposer: "test".to_string(),
            signature: vec![],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
            quantized_codes: vec![],
        },
        transactions: vec![Transaction::Put {
            key: "test_key".to_string(),
            data: vec![42, 43, 44],
        }],
        signatures: vec![],
    };

    // Apply block (this will validate state_root and append to chain)
    state_machine.apply_block(&block).unwrap();

    // Verify data was persisted to store
    let retrieved = store.get("test_key").expect("Key should exist");
    let data_val = retrieved.get("data").expect("Should have data field");
    if let TensorValue::Scalar(tensor_store::ScalarValue::Bytes(bytes)) = data_val {
        assert_eq!(bytes, &vec![42, 43, 44], "Data should match");
    } else {
        panic!("Expected Bytes scalar value");
    }
}

#[test]
fn test_state_machine_embed_transaction_persistence() {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_chain::{
        compute_state_root, network::MemoryTransport, Block, BlockHeader, Chain, RaftConfig,
        RaftNode, TensorStateMachine, Transaction,
    };
    use tensor_store::{SparseVector, TensorStore, TensorValue};

    // Set up components
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
    chain.initialize().unwrap();

    // Get genesis block hash for valid prev_hash
    let genesis = chain.get_tip().unwrap().unwrap();
    let prev_hash = genesis.hash();

    let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
    let raft = Arc::new(RaftNode::new(
        "test_node".to_string(),
        vec![],
        transport,
        RaftConfig::default(),
    ));

    let state_machine = TensorStateMachine::new(chain.clone(), raft, store.clone());

    // Manually apply the transaction to compute the correct state_root
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let mut data = tensor_store::TensorData::new();
    data.set("vector", TensorValue::Vector(embedding.clone()));
    store.put("emb:doc_1", data).unwrap();
    let state_root = compute_state_root(&store).unwrap();

    // Create a block with Embed transaction
    let block = Block {
        header: BlockHeader {
            height: 1,
            prev_hash,
            tx_root: [0u8; 32],
            state_root,
            timestamp: 0,
            proposer: "test".to_string(),
            signature: vec![],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
            quantized_codes: vec![],
        },
        transactions: vec![Transaction::Embed {
            key: "doc_1".to_string(),
            vector: embedding.clone(),
        }],
        signatures: vec![],
    };

    // Apply block (this will validate state_root and append to chain)
    state_machine.apply_block(&block).unwrap();

    // Verify embedding was persisted with correct key prefix
    let retrieved = store.get("emb:doc_1").expect("Embedding key should exist");
    let vec_val = retrieved.get("vector").expect("Should have vector field");
    if let TensorValue::Vector(vec) = vec_val {
        assert_eq!(vec, &embedding, "Embedding should match");
    } else {
        panic!("Expected Vector value");
    }
}

#[test]
fn test_state_machine_node_create_persistence() {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_chain::{
        compute_state_root, network::MemoryTransport, Block, BlockHeader, Chain, RaftConfig,
        RaftNode, TensorStateMachine, Transaction,
    };
    use tensor_store::{ScalarValue, SparseVector, TensorStore, TensorValue};

    // Set up components
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
    chain.initialize().unwrap();

    // Get genesis block hash for valid prev_hash
    let genesis = chain.get_tip().unwrap().unwrap();
    let prev_hash = genesis.hash();

    let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
    let raft = Arc::new(RaftNode::new(
        "test_node".to_string(),
        vec![],
        transport,
        RaftConfig::default(),
    ));

    let state_machine = TensorStateMachine::new(chain.clone(), raft, store.clone());

    // Manually apply the transaction to compute the correct state_root
    let mut data = tensor_store::TensorData::new();
    data.set(
        "_id",
        TensorValue::Scalar(ScalarValue::String("user_123".to_string())),
    );
    data.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("node".to_string())),
    );
    data.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("User".to_string())),
    );
    store.put("node:user_123", data).unwrap();
    let state_root = compute_state_root(&store).unwrap();

    // Create a block with NodeCreate transaction
    let block = Block {
        header: BlockHeader {
            height: 1,
            prev_hash,
            tx_root: [0u8; 32],
            state_root,
            timestamp: 0,
            proposer: "test".to_string(),
            signature: vec![],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
            quantized_codes: vec![],
        },
        transactions: vec![Transaction::NodeCreate {
            key: "user_123".to_string(),
            label: "User".to_string(),
        }],
        signatures: vec![],
    };

    // Apply block (this will validate state_root and append to chain)
    state_machine.apply_block(&block).unwrap();

    // Verify node was persisted with correct key prefix
    let retrieved = store.get("node:user_123").expect("Node key should exist");

    let label_val = retrieved.get("_label").expect("Should have _label field");
    if let TensorValue::Scalar(ScalarValue::String(label)) = label_val {
        assert_eq!(label, "User", "Label should match");
    } else {
        panic!("Expected String scalar value for label");
    }

    let type_val = retrieved.get("_type").expect("Should have _type field");
    if let TensorValue::Scalar(ScalarValue::String(typ)) = type_val {
        assert_eq!(typ, "node", "Type should be 'node'");
    } else {
        panic!("Expected String scalar value for type");
    }
}
