# Tensor Chain

Module 12 of Neumann. Tensor-native blockchain with semantic conflict detection, hierarchical codebook-based validation, and Tensor-Raft distributed consensus.

## Design Principles

1. **Semantic Transactions**: Changes encoded as delta embeddings, enabling similarity-based conflict detection
2. **Hierarchical Codebooks**: Global (static for consensus) + Local (EMA-adaptive per domain) vector quantization
3. **Auto-Merge**: Orthogonal transactions merge via vector addition, reducing contention
4. **Tensor-Raft**: Modified Raft with similarity fast-path for block validation
5. **Two-Phase Finality**: Committed (Raft quorum) -> Finalized (checkpointed)
6. **Queryable History**: Search chain by semantic similarity, not just key lookup

## Quick Start

```rust
use tensor_chain::{TensorChain, ChainConfig, Transaction};
use tensor_store::TensorStore;

// Create chain
let store = TensorStore::new();
let chain = TensorChain::new(store, "node1");
chain.initialize()?;

// Begin a transaction
let tx = chain.begin()?;
tx.add_operation(Transaction::Put {
    key: "users:123".to_string(),
    data: vec![1, 2, 3, 4],
})?;

// Commit (creates new block)
let block_hash = chain.commit(tx)?;

// Query history
let history = chain.history("users:123")?;
```

## Architecture

```
tensor_chain/
  lib.rs              # TensorChain API, ChainConfig
  block.rs            # Block, BlockHeader, Transaction types
  chain.rs            # Chain linked via graph edges
  transaction.rs      # Workspace isolation, delta tracking
  codebook.rs         # GlobalCodebook, LocalCodebook, CodebookManager
  validation.rs       # TransitionValidator
  consensus.rs        # Semantic conflict detection, auto-merge
  raft.rs             # Tensor-Raft consensus state machine
  network.rs          # Transport trait, MemoryTransport
  error.rs            # ChainError types
  membership.rs       # Cluster membership and health checking
  delta_replication.rs # Delta-compressed state replication
  tcp/
    mod.rs            # Module exports
    config.rs         # TCP transport configuration
    transport.rs      # TcpTransport implementation
    connection.rs     # Connection and ConnectionPool
    framing.rs        # Length-delimited wire protocol
    listener.rs       # Accept loop for incoming connections
    reconnect.rs      # Exponential backoff reconnection
    error.rs          # TCP-specific error types
```

## Transaction Workflow

### 1. Begin Transaction

Creates an isolated workspace for tracking operations:

```rust
let tx = chain.begin()?;  // Returns Arc<TransactionWorkspace>
```

The workspace tracks:
- Pending operations (Put, Delete, Update)
- Delta embedding (semantic change vector)
- Affected keys for conflict detection

### 2. Add Operations

```rust
// Put data
tx.add_operation(Transaction::Put {
    key: "users:123".to_string(),
    data: serialized_data,
})?;

// Delete data
tx.add_operation(Transaction::Delete {
    key: "users:456".to_string(),
})?;
```

### 3. Commit

```rust
let block_hash = chain.commit(tx)?;
```

Commit performs:
1. Mark workspace as committing
2. Compute delta embedding from operations
3. Build block with transactions
4. Validate via codebook (if enabled)
5. Append block to chain
6. Clean up workspace

### 4. Rollback

```rust
chain.rollback(tx)?;  // Discard workspace, no block created
```

## Hierarchical Codebook System

Two-tier vector quantization for state validation:

### Global Codebook (Consensus Layer)

Static codebook shared across all nodes for deterministic validation:

```rust
use tensor_chain::{GlobalCodebook, CodebookEntry};

// Create from training data via k-means++
let codebook = GlobalCodebook::from_kmeans(&training_vectors, 256, 100);

// Or from pre-computed centroids
let codebook = GlobalCodebook::from_centroids(centroids);

// Quantize a state vector
let (entry_id, similarity) = codebook.quantize(&state_vector)?;

// Compute residual for local refinement
let (entry_id, residual) = codebook.compute_residual(&state_vector)?;

// Validate state proximity
if codebook.is_valid_state(&state_vector, 0.9) {
    // State is within threshold of a known valid state
}
```

### Local Codebook (Domain Layer)

Adaptive per-domain codebook capturing residuals via EMA:

```rust
use tensor_chain::{LocalCodebook, PruningStrategy};

let mut local = LocalCodebook::new("users", 128, 256, 0.1);  // dim=128, max=256, alpha=0.1

// Set pruning strategy for bounded memory
local.set_pruning_strategy(PruningStrategy::Hybrid {
    recency_weight: 0.5,
    frequency_weight: 0.5,
});

// Quantize and adapt
let (id, sim) = local.quantize_and_update(&residual, 0.9);

// Get statistics
let stats = local.stats();
println!("Entries: {}, Updates: {}", stats.entry_count, stats.total_updates);
```

### Hierarchical Quantization

```rust
use tensor_chain::{CodebookManager, CodebookConfig};

let config = CodebookConfig {
    local_capacity: 256,
    ema_alpha: 0.1,
    similarity_threshold: 0.9,
    residual_threshold: 0.05,
    validity_threshold: 0.8,
};

let manager = CodebookManager::new(global_codebook, config);

// Full hierarchical quantization
let result = manager.quantize("users", &state_vector)?;
// result.global_entry_id, result.local_entry_id, result.codes

// Validate state
if manager.is_valid_state("users", &state_vector) {
    // State is valid in either global or local codebook
}

// Validate transition
if manager.is_valid_transition("users", &from_state, &to_state, max_distance) {
    // Transition is allowed
}
```

### Codebook Persistence

Codebooks are persisted to TensorStore for recovery across restarts:

```rust
use tensor_chain::{TensorChain, ChainConfig, GlobalCodebook};
use tensor_store::TensorStore;

// Option 1: Use load_or_create (recommended for production)
// Automatically loads existing codebook or creates empty one
let store = TensorStore::new();
let chain = TensorChain::load_or_create(store, ChainConfig::new("node1"));

// Option 2: Manual save/load
let chain = TensorChain::with_codebook(store, config, codebook, cb_config, val_config);

// Save codebook to store (key pattern: _codebook:global:{id})
let count = chain.save_global_codebook()?;
println!("Saved {} codebook entries", count);

// Load codebook from store
if let Some(loaded) = chain.load_global_codebook()? {
    println!("Loaded codebook with {} entries", loaded.len());
}

// Access codebook manager
let manager = chain.codebook_manager();
println!("Global codebook dimension: {}", manager.global().dimension());

// Access transition validator
let validator = chain.transition_validator();
let validation = validator.validate_transition("domain", &from, &to);
```

Storage format:
- Centroids: `_codebook:global:{entry_id}` with `_embedding` vector
- Metadata: `_codebook:global:_meta` with `entry_count` and `dimension`

## Semantic Conflict Detection

Conflicts classified by cosine similarity of delta embeddings:

| cos(d1, d2) | Key Overlap | Class | Action |
|-------------|-------------|-------|--------|
| < 0.1 | Any | Orthogonal | Auto-merge (vector add) |
| 0.1-0.7 | None | LowConflict | Weighted merge |
| 0.1-0.7 | Some | Ambiguous | Reject |
| >= 0.7 | Any | Conflicting | Reject |
| ~1.0 | All | Identical | Deduplicate |
| <= -0.95 | All | Opposite | Cancel (no-op) |

### Using the Consensus Manager

```rust
use tensor_chain::{ConsensusManager, ConsensusConfig, DeltaVector};
use std::collections::HashSet;

let manager = ConsensusManager::new(ConsensusConfig::default());

// Create delta vectors from transaction deltas
let keys1: HashSet<String> = ["users:1"].iter().map(|s| s.to_string()).collect();
let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys1, 1);

let keys2: HashSet<String> = ["users:2"].iter().map(|s| s.to_string()).collect();
let d2 = DeltaVector::new(vec![0.0, 1.0, 0.0], keys2, 2);

// Detect conflict
let result = manager.detect_conflict(&d1, &d2);
println!("Class: {:?}, Can merge: {}", result.class, result.can_merge);

// Attempt merge
let merge_result = manager.merge(&d1, &d2);
if merge_result.success {
    let merged = merge_result.merged_delta.unwrap();
    println!("Merged vector: {:?}", merged.vector);
}

// Merge multiple deltas
let all_result = manager.merge_all(&[d1, d2, d3]);
```

### Delta Vector Operations

```rust
// Vector addition (orthogonal merge)
let sum = d1.add(&d2);

// Weighted average (low-conflict merge)
let avg = d1.weighted_average(&d2, 0.5, 0.5);

// Project out conflicting component
let safe = d1.project_non_conflicting(&conflict_direction);

// Scale delta
let scaled = d1.scale(0.5);
```

## Tensor-Raft Consensus

Modified Raft protocol with tensor-native optimizations:

### Node States

- **Follower**: Receives log entries from leader
- **Candidate**: Requesting votes for leadership
- **Leader**: Handles client requests, replicates log

### Configuration

```rust
use tensor_chain::{RaftNode, RaftConfig};

let config = RaftConfig {
    election_timeout: (150, 300),     // ms
    heartbeat_interval: 50,            // ms
    similarity_threshold: 0.95,        // for fast-path
    enable_fast_path: true,
    quorum_size: None,                 // auto: majority
};
```

### Creating a Node

```rust
use tensor_chain::{RaftNode, MemoryTransport};
use std::sync::Arc;

let transport = Arc::new(MemoryTransport::new("node1".to_string()));
let peers = vec!["node2".to_string(), "node3".to_string()];

let node = RaftNode::new("node1".to_string(), peers, transport, config);
```

### Leader Operations

```rust
if node.is_leader() {
    // Propose a block
    let index = node.propose(block)?;

    // Check commit status
    let committed = node.commit_index();

    // Finalize committed entries
    node.finalize_to(height)?;
}
```

### Similarity Fast-Path

When enabled, followers can skip full validation if the block embedding is similar to recent blocks from the same leader:

```rust
// Leader includes embedding in AppendEntries
let ae = AppendEntries {
    block_embedding: Some(block.header.delta_embedding.clone()),
    // ...
};

// Follower checks similarity
// If similarity >= threshold, uses fast-path validation
```

### Network Transport

```rust
use tensor_chain::{Transport, MemoryTransport, Message};

// Memory transport for testing
let transport = MemoryTransport::new("node1".to_string());

// Connect to peer
transport.connect(&peer_config).await?;

// Send message
transport.send(&peer_id, message).await?;

// Broadcast
transport.broadcast(message).await?;

// Receive messages
let (from, msg) = transport.recv().await?;
```

## TCP Transport

Production-ready TCP transport with persistent connections, automatic reconnection, and length-delimited framing.

### Configuration

```rust
use tensor_chain::tcp::{TcpTransport, TcpTransportConfig};

let config = TcpTransportConfig {
    bind_addr: "0.0.0.0:9100".parse()?,
    connection_timeout: Duration::from_secs(5),
    max_message_size: 16 * 1024 * 1024,  // 16MB
    max_connections_per_peer: 2,
    keepalive_interval: Duration::from_secs(30),
    reconnect_delay: Duration::from_millis(100),
    max_reconnect_delay: Duration::from_secs(30),
};
```

### Creating a TCP Transport

```rust
let transport = TcpTransport::new("node1".to_string(), config).await?;

// Connect to peers
transport.connect_peer("node2", "192.168.1.11:9100").await?;
transport.connect_peer("node3", "192.168.1.12:9100").await?;

// Use with Raft
let node = RaftNode::new("node1".to_string(), peers, Arc::new(transport), raft_config);
```

### Wire Protocol

Messages use length-delimited framing:
```
+----------------+------------------+
| Length (4B BE) | bincode payload  |
+----------------+------------------+
```

### Features

- **Persistent Connections**: Pool of 2 connections per peer for low latency
- **Automatic Reconnection**: Exponential backoff with jitter
- **Backpressure**: Bounded per-peer queue (1000 messages)
- **Health Monitoring**: Integrated with membership layer

## Cluster Membership

Static cluster configuration with health checking and failure detection. The membership system integrates with Raft to provide health-aware leader election.

### Membership-Aware Raft Voting

When a MembershipManager is attached to a RaftNode, votes are only granted to healthy candidates:

```rust
use tensor_chain::{RaftNode, RaftConfig, MembershipManager, ClusterConfig};
use std::sync::Arc;

// Create membership manager
let membership = Arc::new(MembershipManager::new(cluster_config, transport.clone()));

// Create Raft node with membership
let node = RaftNode::with_membership(
    "node1".to_string(),
    vec!["node2".to_string(), "node3".to_string()],
    transport,
    RaftConfig::default(),
    membership.clone(),
);

// Node will:
// 1. Reject votes from unhealthy candidates
// 2. Skip unhealthy peers when sending heartbeats
```

### Configuration

```rust
use tensor_chain::{MembershipManager, ClusterConfig, NodeConfig, HealthConfig};

let config = ClusterConfig {
    cluster_id: "neumann-prod-1".to_string(),
    local: NodeConfig {
        node_id: "node1".to_string(),
        bind_address: "0.0.0.0".to_string(),
        bind_port: 9100,
    },
    peers: vec![
        PeerConfig { node_id: "node2".to_string(), address: "192.168.1.11".to_string(), port: 9100 },
        PeerConfig { node_id: "node3".to_string(), address: "192.168.1.12".to_string(), port: 9100 },
    ],
    health: HealthConfig {
        ping_interval_ms: 1000,
        failure_threshold: 3,
        ping_timeout_ms: 500,
        startup_grace_ms: 5000,
    },
};
```

### Health Checking

```rust
let manager = MembershipManager::new(config, transport).await?;

// Start health monitoring
manager.start_health_checks().await;

// Get cluster view
let view = manager.view();
println!("Healthy nodes: {:?}", view.healthy_nodes);
println!("Failed nodes: {:?}", view.failed_nodes);

// Check specific node
let status = manager.node_status("node2")?;
println!("Node2 health: {:?}, RTT: {:?}ms", status.health, status.rtt_ms);

// Register for membership changes
manager.on_view_change(|old_view, new_view| {
    println!("Cluster view changed: {} -> {} healthy nodes",
             old_view.healthy_nodes.len(), new_view.healthy_nodes.len());
});
```

### Node Health States

| State | Description |
|-------|-------------|
| `Healthy` | Responding to pings within timeout |
| `Degraded` | Responding but with elevated latency |
| `Failed` | Exceeded failure threshold |
| `Unknown` | Not yet checked (startup grace period) |

## Delta Replication

Bandwidth-efficient state replication using delta-encoded vectors.

### Core Concept

Instead of replicating full embeddings, delta replication sends only the difference from a shared archetype:

```
Full embedding:    [0.1, 0.2, 0.3, ..., 0.9]  (128 floats = 512 bytes)
Delta encoding:    archetype_id=42, delta=[0.01, -0.02, ...]  (4 bytes + sparse)
Bandwidth saving:  4-10x compression for clustered data
```

### Using Delta Replication

```rust
use tensor_chain::{DeltaReplicationManager, DeltaUpdate, ReplicationBatch};
use tensor_store::ArchetypeRegistry;

let registry = Arc::new(ArchetypeRegistry::new(128, 256));
let manager = DeltaReplicationManager::new(registry.clone());

// Encode for replication
let update = manager.encode("users:123", &embedding)?;
println!("Archetype: {}, Delta nnz: {}", update.archetype_id, update.sparse_delta.nnz());

// Batch multiple updates
let batch = manager.create_batch(updates)?;
let compressed = batch.to_bytes()?;
println!("Batch size: {} bytes, compression: {:.1}x",
         compressed.len(), batch.compression_ratio());

// Apply on receiver
let received_batch = ReplicationBatch::from_bytes(&compressed)?;
manager.apply_batch(&received_batch, |key, embedding| {
    store.put_embedding(key, embedding)
})?;
```

### Batch Operations

```rust
// Create streaming batch for large transfers
let mut writer = manager.batch_writer(1000);  // max 1000 updates
for (key, embedding) in data {
    writer.add(key, embedding)?;
}
let batch = writer.finish()?;

// Iterate received batch
for update in batch.iter() {
    println!("Key: {}, Full: {}", update.key, update.is_full_update());
}
```

### Int8 Quantization for Delta Updates

Delta updates can be further compressed using int8 quantization, reducing bandwidth by ~4x:

```rust
use tensor_chain::{DeltaUpdate, QuantizedDeltaUpdate, DeltaReplicationConfig};

// Enable quantization in config
let config = DeltaReplicationConfig::default().with_quantization();

// Create a delta update
let update = DeltaUpdate::full("users:123", &embedding, version);

// Quantize for transmission
if let Some(quantized) = update.quantize() {
    // Serialize quantized update (4x smaller than f32)
    let bytes = bincode::serialize(&quantized)?;

    // Compression ratio
    println!("Compression: {:.1}x", quantized.compression_ratio());
    println!("Memory: {} bytes", quantized.memory_bytes());
}

// Dequantize on receiver
let restored = quantized.dequantize();
// Error < 2% for normalized values

// Or decode directly with registry
let full_embedding = quantized.decode(&registry)?;
```

Quantization error bounds:
- Normalized values [0, 1]: < 1% error
- General f32 values: < 2% error
- Preserves semantic similarity for downstream operations

### Compression Statistics

| Data Pattern | Compression Ratio | Notes |
|--------------|-------------------|-------|
| Random embeddings | 1.0-1.5x | No shared structure |
| Clustered (k=10) | 3-5x | Moderate clustering |
| Highly clustered (k=3) | 6-10x | Strong archetypes |
| Incremental updates | 10-20x | Sparse deltas |
| Int8 quantization | 4x | On top of delta encoding |

## Chain Operations

### Query Chain State

```rust
// Get current height
let height = chain.height();

// Get tip block
let tip = chain.get_tip()?;

// Get genesis block
let genesis = chain.get_genesis()?;

// Get block at height
let block = chain.get_block(42)?;

// Get blocks in range
let blocks = chain.get_blocks(10, 20)?;

// Iterate all blocks
for block in chain.iter() {
    println!("Block {}: {}", block.header.height, block.hash());
}
```

### Query History

```rust
// Get change history for a key
let history = chain.history("users:123")?;
for (height, transaction) in history {
    println!("Height {}: {:?}", height, transaction);
}
```

### Verify Chain Integrity

```rust
chain.verify()?;  // Verifies all block links and hashes
```

## SQL Extensions

### Transaction Control

```sql
-- Begin a chain transaction
BEGIN CHAIN TRANSACTION;

-- Commit (creates new block)
COMMIT CHAIN;

-- Rollback to a specific height
ROLLBACK CHAIN TO 10;
```

### Chain Queries

```sql
-- Get current chain height
CHAIN HEIGHT;

-- Get tip block info
CHAIN TIP;

-- Get specific block
CHAIN BLOCK 42;

-- Verify chain integrity
CHAIN VERIFY;

-- Get key history
CHAIN HISTORY 'users:123';

-- Find similar blocks by embedding
CHAIN SIMILAR [0.1, 0.2, 0.3, ...] LIMIT 10;

-- Analyze state drift between heights
CHAIN DRIFT FROM 0 TO 100;
```

### Codebook Management

```sql
-- Show global codebook info
SHOW CODEBOOK GLOBAL;

-- Show local codebook for domain
SHOW CODEBOOK LOCAL 'users';

-- Analyze transition validity
ANALYZE CODEBOOK TRANSITIONS;
```

## API Reference

### TensorChain

```rust
pub struct TensorChain {
    // Unified API for chain operations
}

impl TensorChain {
    // Construction
    pub fn new(store: TensorStore, node_id: impl Into<NodeId>) -> Self;
    pub fn with_config(store: TensorStore, config: ChainConfig) -> Self;
    pub fn with_codebook(
        store: TensorStore,
        config: ChainConfig,
        global_codebook: GlobalCodebook,
        codebook_config: CodebookConfig,
        validation_config: ValidationConfig,
    ) -> Self;
    pub fn load_or_create(store: TensorStore, config: ChainConfig) -> Self;
    pub fn initialize(&self) -> Result<()>;

    // Transaction Management
    pub fn begin(&self) -> Result<Arc<TransactionWorkspace>>;
    pub fn commit(&self, workspace: Arc<TransactionWorkspace>) -> Result<BlockHash>;
    pub fn rollback(&self, workspace: Arc<TransactionWorkspace>) -> Result<()>;

    // Chain Queries
    pub fn height(&self) -> u64;
    pub fn tip_hash(&self) -> BlockHash;
    pub fn get_block(&self, height: u64) -> Result<Option<Block>>;
    pub fn get_tip(&self) -> Result<Option<Block>>;
    pub fn get_genesis(&self) -> Result<Option<Block>>;
    pub fn get_blocks(&self, start: u64, end: u64) -> Result<Vec<Block>>;
    pub fn iter(&self) -> ChainIterator<'_>;

    // History and Verification
    pub fn history(&self, key: &str) -> Result<Vec<(u64, Transaction)>>;
    pub fn verify(&self) -> Result<()>;

    // Codebook Access
    pub fn codebook_manager(&self) -> &CodebookManager;
    pub fn transition_validator(&self) -> &TransitionValidator;

    // Codebook Persistence
    pub fn save_global_codebook(&self) -> Result<usize>;
    pub fn load_global_codebook(&self) -> Result<Option<GlobalCodebook>>;

    // Advanced
    pub fn active_transactions(&self) -> usize;
    pub fn store(&self) -> &TensorStore;
    pub fn graph(&self) -> &GraphEngine;
    pub fn append_block(&self, block: Block) -> Result<BlockHash>;
    pub fn new_block(&self) -> BlockBuilder;
}
```

### ChainConfig

```rust
pub struct ChainConfig {
    pub node_id: NodeId,
    pub max_txs_per_block: usize,     // Default: 1000
    pub conflict_threshold: f32,       // Default: 0.7
    pub auto_merge: bool,              // Default: true
}
```

### Block

```rust
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
    pub signatures: Vec<ValidatorSignature>,
}

pub struct BlockHeader {
    pub height: u64,
    pub prev_hash: [u8; 32],
    pub tx_root: [u8; 32],
    pub state_root: [u8; 32],
    pub delta_embedding: Vec<f32>,
    pub quantized_codes: Vec<u16>,
    pub timestamp: u64,
    pub proposer: NodeId,
    pub signature: [u8; 32],
}
```

### Transaction

```rust
pub enum Transaction {
    Put { key: String, data: Vec<u8> },
    Delete { key: String },
    Update { key: String, data: Vec<u8> },
}
```

### ChainError

```rust
pub enum ChainError {
    BlockNotFound(String),
    InvalidBlock(String),
    TransactionError(String),
    ConsensusError(String),
    CodebookError(String),
    ValidationError(String),
    StorageError(String),
    AlreadyInitialized,
    NotInitialized,
}
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| begin() | ~1us | Creates workspace |
| commit() (1 tx) | ~50us | Includes block build |
| commit() (100 tx) | ~200us | Scales with tx count |
| get_block() | ~5us | O(1) via graph lookup |
| history() | ~100us | Scans block transactions |
| verify() (1000 blocks) | ~10ms | Full chain verification |
| quantize() (global) | ~2us | O(n) over codebook |
| quantize() (hierarchical) | ~5us | Global + local lookup |
| detect_conflict() | ~1us | Cosine similarity |
| merge() (orthogonal) | ~2us | Vector addition |

## Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| validation.rs | 100% | State validation |
| block.rs | 99.44% | Block types |
| codebook.rs | 99.36% | Quantization |
| transaction.rs | 99.35% | Transaction workspace |
| lib.rs | 99.20% | Core API |
| consensus.rs | 99.01% | Conflict detection |
| distributed_tx.rs | 98.78% | 2PC coordinator |
| raft.rs | 98.47% | Raft state machine |
| chain.rs | 97.73% | Chain operations |
| delta_replication.rs | 95.28% | Delta compression |
| membership.rs | 95.0% | Cluster membership |
| network.rs | 94.51% | Transport trait |
| tcp/*.rs | 94.89% | TCP transport |
| **Total** | **>95%** | |

All modules meet the 95% minimum coverage threshold required for critical infrastructure. Comprehensive tests cover semantic conflict detection, distributed transactions, Raft consensus, and delta replication.

## Dependencies

- `tensor_store`: Core storage layer (includes ArchetypeRegistry for delta encoding)
- `tensor_compress`: Int8 quantization for delta embeddings
- `graph_engine`: Block linking via edges
- `parking_lot`: Lock primitives
- `sha2`: Block hashing (SHA-256)
- `bincode`: Serialization
- `uuid`: Transaction/node IDs
- `dashmap`: Concurrent state maps
- `tokio`: Async runtime for TCP transport
- `bytes`: Buffer management for framing
- `socket2`: Low-level socket configuration
- `serde`: Serialization for cluster config

## Unique Value Propositions

1. **Semantic Conflict Detection**: Cosine similarity catches logical conflicts even when bytes differ
2. **100x Compression Potential**: Int8 quantization (4x) + codebook discretization (8-32x)
3. **Queryable History**: Vector search over chain ("find transactions like X")
4. **Tensor-Native Smart Contracts**: Constraints as geometric bounds, not bytecode
5. **Proof by Reconstruction**: Validity = reconstruction error < threshold
6. **Chain Drift Metric**: Detect corruption by tracking error vs hop count
7. **Delta Replication**: 4-10x bandwidth reduction via archetype-based encoding
8. **Semantic Sharding**: Route data by embedding similarity, not just hash
9. **Orthogonal Transaction Merge**: Auto-merge non-conflicting concurrent updates

## Cross-Shard Distributed Transactions

Two-phase commit (2PC) protocol with delta-based conflict detection for transactions spanning multiple shards.

### Core Concept

Traditional 2PC blocks on locks; tensor-native 2PC uses embedding similarity to detect and resolve conflicts:

```
Phase 1: PREPARE
    Coordinator -> PrepareRequest to each shard
    Each shard: acquire locks, compute delta, check conflicts
    Shard -> PrepareVote (Yes/No/Conflict)

Phase 2: COMMIT or ABORT
    If all Yes && cross-shard deltas orthogonal: COMMIT
    Otherwise: ABORT
```

### Using the Distributed Transaction Coordinator

```rust
use tensor_chain::{
    DistributedTxCoordinator, DistributedTxConfig, PrepareRequest, PrepareVote,
    ConsensusManager, ConsensusConfig, DeltaVector,
};

// Create coordinator
let consensus = ConsensusManager::new(ConsensusConfig::default());
let config = DistributedTxConfig {
    max_concurrent: 100,
    prepare_timeout_ms: 5000,
    commit_timeout_ms: 10000,
    orthogonal_threshold: 0.1,  // cos < 0.1 = orthogonal
};
let coordinator = DistributedTxCoordinator::new(consensus, config);

// Begin distributed transaction across shards 0, 1, 2
let tx = coordinator.begin("coordinator_node".to_string(), vec![0, 1, 2])?;
let tx_id = tx.tx_id;

// Simulate votes from shards (in real system, sent over network)
let delta0 = DeltaVector::new(vec![1.0, 0.0, 0.0], vec!["key_a".to_string()], tx_id);
let delta1 = DeltaVector::new(vec![0.0, 1.0, 0.0], vec!["key_b".to_string()], tx_id);
let delta2 = DeltaVector::new(vec![0.0, 0.0, 1.0], vec!["key_c".to_string()], tx_id);

// Record votes - coordinator tracks state
coordinator.record_vote(tx_id, 0, PrepareVote::Yes { lock_handle: 1, delta: delta0 });
coordinator.record_vote(tx_id, 1, PrepareVote::Yes { lock_handle: 2, delta: delta1 });
let phase = coordinator.record_vote(tx_id, 2, PrepareVote::Yes { lock_handle: 3, delta: delta2 });

// All orthogonal -> auto-merge and commit
assert_eq!(phase, Some(TxPhase::Prepared));
coordinator.commit(tx_id)?;

// Check statistics
let stats = coordinator.stats();
println!("Committed: {}, Orthogonal merges: {}",
         stats.committed.load(Ordering::Relaxed),
         stats.orthogonal_merges.load(Ordering::Relaxed));
```

### Lock Manager

Key-level locking for transaction isolation:

```rust
use tensor_chain::LockManager;

let lock_manager = LockManager::new();

// Acquire locks for a transaction
let keys = vec!["account:1".to_string(), "account:2".to_string()];
let handle = lock_manager.try_lock(tx_id, &keys)?;

// Check lock status
assert!(lock_manager.is_locked("account:1"));
assert_eq!(lock_manager.lock_holder("account:1"), Some(tx_id));

// Conflict detection: another tx trying same key fails
let conflict_result = lock_manager.try_lock(other_tx_id, &keys);
assert!(conflict_result.is_err());

// Release on commit/abort
lock_manager.release_by_handle(handle);
// Or release all locks for a transaction
lock_manager.release(tx_id);
```

### Transaction Participant

Each shard runs a participant that handles prepare/commit/abort:

```rust
use tensor_chain::{TxParticipant, PrepareRequest, Transaction};

let participant = TxParticipant::new();

// Handle prepare request from coordinator
let request = PrepareRequest {
    tx_id: 42,
    coordinator: "coord".to_string(),
    operations: vec![Transaction::Put {
        key: "local_key".to_string(),
        data: vec![1, 2, 3],
    }],
    delta_embedding: vec![0.5, 0.5, 0.0],
    timeout_ms: 5000,
};

let vote = participant.prepare(request);
match vote {
    PrepareVote::Yes { lock_handle, delta } => {
        // Ready to commit, locks held
    }
    PrepareVote::No { reason } => {
        // Cannot prepare (e.g., validation failed)
    }
    PrepareVote::Conflict { similarity, conflicting_tx } => {
        // Detected conflict with another transaction
    }
}

// On commit decision
participant.commit(42);

// Or on abort decision
participant.abort(42);

// Cleanup stale prepared transactions
let stale = participant.cleanup_stale(Duration::from_secs(30));
```

### 2PC Network Messages

Messages for distributed transaction coordination:

```rust
use tensor_chain::{TxPrepareMsg, TxPrepareResponseMsg, TxCommitMsg, TxAbortMsg, TxAckMsg, TxVote};

// Prepare request
let prepare = Message::TxPrepare(TxPrepareMsg {
    tx_id: 1,
    coordinator: "node1".to_string(),
    shard_id: 0,
    operations: vec![/* transactions */],
    delta_embedding: vec![0.1, 0.2, 0.3],
    timeout_ms: 5000,
});

// Vote response
let response = Message::TxPrepareResponse(TxPrepareResponseMsg {
    tx_id: 1,
    shard_id: 0,
    vote: TxVote::Yes {
        lock_handle: 123,
        delta: vec![0.1, 0.2, 0.3],
        affected_keys: vec!["key1".to_string()],
    },
});

// Commit/abort
let commit = Message::TxCommit(TxCommitMsg { tx_id: 1, shards: vec![0, 1, 2] });
let abort = Message::TxAbort(TxAbortMsg { tx_id: 1, reason: "conflict".to_string(), shards: vec![0, 1] });
```

### Conflict Resolution

| Vote Pattern | Cross-Shard Delta | Result |
|--------------|-------------------|--------|
| All Yes | Orthogonal (cos < 0.1) | COMMIT with merge |
| All Yes | Conflicting (cos >= 0.1) | ABORT |
| Any No | - | ABORT |
| Any Conflict | - | ABORT |
| Timeout | - | ABORT |

### Failure Handling

| Failure | Recovery |
|---------|----------|
| Coordinator crash before prepare | Participants timeout, release locks |
| Participant crash during prepare | Coordinator aborts after timeout |
| Participant crash after YES vote | Coordinator retries commit (participant recovers from WAL) |
| Network partition | Both sides abort (conservative) |

## Orthogonal Transaction Auto-Merge

Concurrent transactions with cosine similarity < 0.1 automatically merge via vector addition, reducing contention.

### Configuration

```rust
use tensor_chain::{AutoMergeConfig, ChainConfig};

let config = ChainConfig {
    auto_merge: AutoMergeConfig {
        enabled: true,
        orthogonal_threshold: 0.1,  // cos < 0.1 = orthogonal
        max_merge_batch: 10,        // Max transactions to merge
        merge_window_ms: 100,       // Time window for batching
    },
    ..Default::default()
};

let chain = TensorChain::with_config(store, config);
```

### Workspace Embedding Tracking

```rust
use tensor_chain::TransactionWorkspace;

let tx = chain.begin()?;

// Workspace tracks embedding changes
tx.set_before_embedding(current_state_embedding);

// After operations...
tx.add_operation(Transaction::Put { key: "k1".to_string(), data: vec![1, 2, 3] })?;

// Compute delta embedding
let delta = tx.compute_delta();
if tx.has_delta() {
    let delta_vector = tx.to_delta_vector();
    println!("Delta magnitude: {}", delta_vector.magnitude());
}
```

### Merge Candidate Detection

```rust
use tensor_chain::TransactionManager;

let tx_manager = chain.transaction_manager();

// Find transactions that can merge with current
let candidates = tx_manager.find_merge_candidates(&workspace, 0.1);
for candidate in candidates {
    println!("Can merge tx {}: similarity = {:.3}",
             candidate.workspace.id(), candidate.similarity);
}
```

### Batch Conflict Detection

```rust
use tensor_chain::{ConsensusManager, DeltaVector};

let manager = ConsensusManager::new(config);

// Check all pairs for conflicts
let conflicts = manager.batch_detect_conflicts(&deltas);
for conflict in conflicts {
    println!("Conflict between {} and {}: {:?}",
             conflict.index_a, conflict.index_b, conflict.result.class);
}

// Find maximal orthogonal subset
let orthogonal_indices = manager.find_orthogonal_set(&deltas);
println!("Can merge {} of {} transactions", orthogonal_indices.len(), deltas.len());
```

## Similarity Fast-Path Validation

Skip full validation when block embedding similarity > 0.95 to recent blocks from same leader.

### Configuration

```rust
use tensor_chain::{FastPathValidator, ValidationMode};

let validator = FastPathValidator::new(
    0.95,   // similarity_threshold
    3,      // min_leader_history (blocks before fast-path eligible)
    10,     // full_validation_interval (periodic full check)
);
```

### Fast-Path State Tracking

```rust
use tensor_chain::{FastPathState, RaftNode};

// RaftNode tracks leader embeddings
let node = RaftNode::new(id, peers, transport, config);

// Fast-path state per leader
let fast_path = node.fast_path_state();

// Check if fast-path can be used
let result = fast_path.can_use_fast_path(
    &leader_id,
    &incoming_embedding,
    0.95,  // threshold
    3,     // min_history
);

match result.mode {
    ValidationMode::FastPath => {
        // Skip full validation, apply block directly
        fast_path.stats().record_accepted();
    }
    ValidationMode::Full => {
        // Run full validation
        fast_path.stats().record_rejected();
    }
    ValidationMode::Trusted => {
        // Trusted source, minimal validation
    }
}
```

### Fast-Path Statistics

```rust
let stats = fast_path.stats();
let acceptance_rate = stats.acceptance_rate();
println!("Fast-path acceptance: {:.1}%", acceptance_rate * 100.0);
println!("Fast-path: {} accepted, {} rejected, {} full validations",
         stats.fast_path_accepted.load(Ordering::Relaxed),
         stats.fast_path_rejected.load(Ordering::Relaxed),
         stats.full_validation_required.load(Ordering::Relaxed));
```

### Security Constraints

Fast-path is only used when:
1. Leader has established history (min 3 blocks)
2. Similarity exceeds threshold (default 0.95)
3. Not a periodic full-validation block (every 10th)
4. No anomalies detected in recent history

## Security Considerations

1. **Block Signatures**: Blake2b HMAC on block headers
2. **Transaction Root**: Merkle root of all transactions in block
3. **State Root**: Merkle root of chain state for proofs
4. **Quorum Validation**: Requires majority consensus for commits
5. **No Bypass**: All mutations go through transaction workspace
6. **2PC Atomicity**: Distributed transactions either fully commit or fully abort
7. **Lock Isolation**: Key-level locks prevent concurrent modification
8. **Fast-Path Guards**: Periodic full validation prevents drift accumulation
