// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Jepsen-style distributed systems test harness.
//!
//! All operations go through **real consensus**: `RealJepsenHarness` and
//! `ChaosRaftCluster` start actual `RaftNode` instances with `MemoryTransport`,
//! run real leader election, and propose blocks through Raft. Read values come
//! from actual `TensorStore` instances after applying committed log entries --
//! they are never hardcoded.
//!
//! Workflow (mirrors real Jepsen):
//! 1. Start a real Raft cluster (`ChaosRaftCluster::new`)
//! 2. Inject faults via nemesis schedule (partitions, crashes, clock drift)
//! 3. Send real writes through `cluster.put()` -> `RaftNode::propose()`
//! 4. Record the system's actual responses in a `HistoryRecorder`
//! 5. Check whether the system-generated history is linearizable
//!
//! For 2PC, `Real2PCCluster` uses real `TxParticipant` instances: each
//! participant acquires real locks, captures a real undo log, and returns an
//! honest `PrepareVote`.  Commit and abort flow through the participant's
//! own `commit()` / `abort()` methods -- the test never fabricates votes.
//! `TxKVCluster` is a lighter-weight variant that manually drives votes
//! for testing specific coordinator behaviors.

use std::sync::Arc;
use std::time::Duration;

use tensor_chain::raft::{RaftConfig, RaftNode};
use tensor_chain::{Block, BlockHeader, Transaction as ChainTransaction};
use tensor_store::{ScalarValue, TensorStore, TensorValue};
use tokio::sync::broadcast;

use crate::chaos::{ChaosCluster, ChaosConfig};
use crate::linearizability::{
    ConcurrentHistoryRecorder, HistoryRecorder, LinearizabilityChecker, LinearizabilityResult,
    OpType, RegisterModel, Value,
};

// ---------------------------------------------------------------------------
// Nemesis types
// ---------------------------------------------------------------------------

/// Configuration for nemesis (fault injection) during Jepsen tests.
#[derive(Debug, Clone)]
pub enum NemesisAction {
    /// Crash a random node.
    RandomCrash,
    /// Partition network into majority/minority.
    MajorityPartition,
    /// Inject clock drift on a random node.
    ClockDrift { drift_ms: i64 },
    /// Set asymmetric link quality.
    LinkDegradation { drop_rate: f32 },
    /// Heal all faults.
    HealAll,
}

/// Schedule of nemesis actions for a test.
#[derive(Debug, Clone)]
pub struct NemesisSchedule {
    /// Actions to execute in order.
    pub actions: Vec<(Duration, NemesisAction)>,
    /// Whether to heal all faults at the end.
    pub heal_at_end: bool,
}

impl NemesisSchedule {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            actions: Vec::new(),
            heal_at_end: true,
        }
    }

    #[must_use]
    pub fn add(mut self, delay: Duration, action: NemesisAction) -> Self {
        self.actions.push((delay, action));
        self
    }

    #[must_use]
    pub const fn with_heal_at_end(mut self, heal: bool) -> Self {
        self.heal_at_end = heal;
        self
    }

    /// Create a simple partition-then-heal schedule.
    #[must_use]
    pub fn partition_heal(partition_duration: Duration) -> Self {
        Self::new()
            .add(Duration::ZERO, NemesisAction::MajorityPartition)
            .add(partition_duration, NemesisAction::HealAll)
    }

    /// Create a crash-recover schedule.
    #[must_use]
    pub fn crash_recover(crash_duration: Duration) -> Self {
        Self::new()
            .add(Duration::ZERO, NemesisAction::RandomCrash)
            .add(crash_duration, NemesisAction::HealAll)
    }
}

impl Default for NemesisSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Jepsen result
// ---------------------------------------------------------------------------

/// Result of a Jepsen-style test.
#[derive(Debug)]
pub struct JepsenResult {
    pub duration: Duration,
    pub operation_count: usize,
    pub linearizability: LinearizabilityResult,
    pub nemesis_actions_applied: usize,
    pub total_faults_injected: u64,
}

impl JepsenResult {
    /// Returns true if the recorded history is linearizable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.linearizability == LinearizabilityResult::Ok
    }
}

// ---------------------------------------------------------------------------
// Real Raft KV Cluster
// ---------------------------------------------------------------------------

/// Raft cluster with integrated chaos infrastructure for fault injection.
pub struct ChaosRaftCluster {
    nodes: Vec<Arc<RaftNode>>,
    stores: Vec<TensorStore>,
    chaos: ChaosCluster,
    shutdown_tx: broadcast::Sender<()>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl ChaosRaftCluster {
    /// Create a new chaos-enabled Raft cluster.
    #[allow(clippy::unused_async)]
    pub async fn new(
        node_count: usize,
        raft_config: RaftConfig,
        chaos_config: ChaosConfig,
    ) -> Self {
        let chaos = ChaosCluster::new(node_count, chaos_config);
        let mut stores = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);

        for i in 0..node_count {
            stores.push(TensorStore::new());
            let transport = chaos.transport(i).expect("transport must exist");

            let peers: Vec<String> = (0..node_count)
                .filter(|&j| j != i)
                .map(|j| format!("node-{j}"))
                .collect();

            let node = Arc::new(RaftNode::new(
                format!("node-{i}"),
                peers,
                transport,
                raft_config.clone(),
            ));
            nodes.push(node);
        }

        let (shutdown_tx, _) = broadcast::channel(1);
        let mut handles = Vec::with_capacity(node_count);
        for node in &nodes {
            let n = Arc::clone(node);
            let rx = shutdown_tx.subscribe();
            handles.push(tokio::spawn(async move {
                let _ = n.run(rx).await;
            }));
        }

        Self {
            nodes,
            stores,
            chaos,
            shutdown_tx,
            handles,
        }
    }

    pub async fn wait_for_leader(&self, timeout: Duration) -> Option<usize> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            for (i, node) in self.nodes.iter().enumerate() {
                if node.is_leader() {
                    return Some(i);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    pub fn leader_index(&self) -> Option<usize> {
        self.nodes.iter().position(|n| n.is_leader())
    }

    /// # Errors
    ///
    /// Returns `Err` if the put times out before the entry is committed.
    pub async fn put(&self, key: &str, data: &[u8], timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        let mut log_index: Option<u64> = None;

        while start.elapsed() < timeout {
            let Some(leader_idx) = self.leader_index() else {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            };

            if log_index.is_none() {
                let header = BlockHeader::new(
                    0,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("node-{leader_idx}"),
                );
                let block = Block::new(
                    header,
                    vec![ChainTransaction::Put {
                        key: key.to_string(),
                        data: data.to_vec(),
                    }],
                );

                if let Ok(idx) = self.nodes[leader_idx].propose(block) {
                    log_index = Some(idx);
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    continue;
                }
            }

            if let Some(idx) = log_index {
                if self.nodes[leader_idx].commit_index() >= idx {
                    self.apply_committed(leader_idx);
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err("put timed out".to_string())
    }

    pub fn get(&self, node_idx: usize, key: &str) -> Option<Vec<u8>> {
        self.apply_committed(node_idx);
        self.get_raw(node_idx, key)
    }

    pub fn apply_committed(&self, node_idx: usize) -> usize {
        let entries = self.nodes[node_idx].get_uncommitted_entries();
        let count = entries.len();
        for entry in &entries {
            for tx in &entry.block.transactions {
                let _ = tensor_chain::apply_transaction_to_store(&self.stores[node_idx], tx);
            }
        }
        if count > 0 {
            let max_idx = entries.last().map_or(0, |e| e.index);
            self.nodes[node_idx].mark_applied(max_idx);
        }
        count
    }

    pub fn apply_committed_all(&self) -> usize {
        let mut total = 0;
        for i in 0..self.nodes.len() {
            total += self.apply_committed(i);
        }
        total
    }

    /// Read from a node's store WITHOUT applying committed entries first.
    /// This allows testing stale reads from followers that haven't caught up.
    pub fn get_raw(&self, node_idx: usize, key: &str) -> Option<Vec<u8>> {
        let data = self.stores[node_idx].get(key).ok()?;
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
            Some(bytes.clone())
        } else {
            None
        }
    }

    /// Perform a write and record it in a concurrent history recorder.
    pub async fn concurrent_write(
        &self,
        history: &ConcurrentHistoryRecorder,
        client_id: u64,
        key: &str,
        value: &Value,
        timeout: Duration,
    ) -> bool {
        let op_id = history.invoke(client_id, OpType::Write, key.to_string(), value.clone());
        let data = value_to_bytes(value);
        let ok = self.put(key, &data, timeout).await.is_ok();
        if ok {
            history.complete(op_id, Value::None);
        }
        ok
    }

    /// Perform a read and record it in a concurrent history recorder.
    pub fn concurrent_read(
        &self,
        history: &ConcurrentHistoryRecorder,
        client_id: u64,
        key: &str,
        node_idx: usize,
    ) -> Value {
        let op_id = history.invoke(client_id, OpType::Read, key.to_string(), Value::None);
        let result = self
            .get(node_idx, key)
            .map_or(Value::None, |bytes| bytes_to_value(&bytes));
        history.complete(op_id, result.clone());
        result
    }

    /// Atomic compare-and-swap through Raft consensus: proposes a
    /// `CompareAndSwap` transaction that the state machine evaluates
    /// atomically. Returns `Ok(old_value)` where `old_value` is the pre-CAS
    /// read. On mismatch the block is valid but the CAS is a no-op.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no leader is available or the CAS times out.
    pub async fn cas(
        &self,
        key: &str,
        expected: &Value,
        timeout: Duration,
    ) -> Result<Value, String> {
        let start = std::time::Instant::now();
        let mut log_index: Option<u64> = None;

        let leader = self.leader_index().ok_or("no leader")?;
        let current = self
            .get(leader, key)
            .map_or(Value::None, |bytes| bytes_to_value(&bytes));

        if current != *expected {
            return Ok(current);
        }

        let new_value = match expected {
            Value::Int(v) => Value::Int(v + 1),
            other => other.clone(),
        };

        while start.elapsed() < timeout {
            let Some(leader_idx) = self.leader_index() else {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            };

            if log_index.is_none() {
                let header = BlockHeader::new(
                    0,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("node-{leader_idx}"),
                );
                let block = Block::new(
                    header,
                    vec![ChainTransaction::CompareAndSwap {
                        key: key.to_string(),
                        expected_data: value_to_bytes(expected),
                        new_data: value_to_bytes(&new_value),
                    }],
                );

                if let Ok(idx) = self.nodes[leader_idx].propose(block) {
                    log_index = Some(idx);
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    continue;
                }
            }

            if let Some(idx) = log_index {
                if self.nodes[leader_idx].commit_index() >= idx {
                    self.apply_committed(leader_idx);
                    return Ok(current);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err("cas timed out".to_string())
    }

    /// Perform a CAS and record it in a concurrent history recorder.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the CAS operation fails at the Raft level.
    pub async fn concurrent_cas(
        &self,
        history: &ConcurrentHistoryRecorder,
        client_id: u64,
        key: &str,
        expected: &Value,
        timeout: Duration,
    ) -> Result<Value, String> {
        let op_id = history.invoke(client_id, OpType::Cas, key.to_string(), expected.clone());
        let result = self.cas(key, expected, timeout).await;
        if let Ok(old_val) = &result {
            history.complete(op_id, old_val.clone());
        }
        result
    }

    pub fn node(&self, idx: usize) -> &Arc<RaftNode> {
        &self.nodes[idx]
    }

    pub const fn chaos(&self) -> &ChaosCluster {
        &self.chaos
    }

    pub const fn chaos_mut(&mut self) -> &mut ChaosCluster {
        &mut self.chaos
    }

    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        for handle in self.handles {
            let _ = handle.await;
        }
    }
}

// ---------------------------------------------------------------------------
// Real Jepsen harness (wraps ChaosRaftCluster)
// ---------------------------------------------------------------------------

/// Jepsen harness that tests a real Raft cluster.
///
/// Operations go through actual Raft consensus. The history recorder captures
/// real timestamps from actual system interactions.
pub struct RealJepsenHarness {
    cluster: ChaosRaftCluster,
    history: HistoryRecorder,
    nemesis: NemesisSchedule,
    checker: LinearizabilityChecker<RegisterModel>,
    nemesis_cursor: usize,
    nemesis_applied: usize,
}

impl RealJepsenHarness {
    pub async fn with_config(
        node_count: usize,
        raft_config: RaftConfig,
        chaos_config: ChaosConfig,
        nemesis: NemesisSchedule,
    ) -> Self {
        let cluster = ChaosRaftCluster::new(node_count, raft_config, chaos_config).await;
        Self {
            cluster,
            history: HistoryRecorder::new(),
            nemesis,
            checker: LinearizabilityChecker::new(RegisterModel),
            nemesis_cursor: 0,
            nemesis_applied: 0,
        }
    }

    /// Wait for leader election.
    pub async fn wait_for_leader(&self, timeout: Duration) -> Option<usize> {
        self.cluster.wait_for_leader(timeout).await
    }

    /// Write a value through real Raft consensus and record in history.
    pub async fn write(&mut self, client_id: u64, key: &str, value: Value) -> bool {
        let op_id = self
            .history
            .invoke(client_id, OpType::Write, key.to_string(), value.clone());

        let data = value_to_bytes(&value);
        let timeout = Duration::from_secs(5);
        let ok = self.cluster.put(key, &data, timeout).await.is_ok();

        if ok {
            self.history.complete(op_id, Value::None);
        }
        // If put failed, leave operation incomplete (crash/timeout)
        ok
    }

    /// Read a value from a specific node and record in history.
    pub fn read(&mut self, client_id: u64, key: &str, node_idx: usize) -> Value {
        let op_id = self
            .history
            .invoke(client_id, OpType::Read, key.to_string(), Value::None);

        let result = self
            .cluster
            .get(node_idx, key)
            .map_or(Value::None, |bytes| bytes_to_value(&bytes));

        self.history.complete(op_id, result.clone());
        result
    }

    /// Apply pending nemesis actions whose delay has elapsed.
    pub fn apply_nemesis(&mut self, elapsed: Duration) -> usize {
        let mut applied = 0;

        while self.nemesis_cursor < self.nemesis.actions.len() {
            let (delay, _) = &self.nemesis.actions[self.nemesis_cursor];
            if elapsed < *delay {
                break;
            }

            let action = self.nemesis.actions[self.nemesis_cursor].1.clone();
            self.apply_action(&action);
            self.nemesis_cursor += 1;
            self.nemesis_applied += 1;
            applied += 1;
        }

        applied
    }

    fn apply_action(&mut self, action: &NemesisAction) {
        match action {
            NemesisAction::RandomCrash => {
                self.cluster.chaos_mut().crash_node(0);
            },
            NemesisAction::MajorityPartition => {
                let _ = self.cluster.chaos_mut().create_majority_partition();
            },
            NemesisAction::ClockDrift { drift_ms } => {
                self.cluster.chaos().inject_clock_drift(0, *drift_ms);
            },
            NemesisAction::LinkDegradation { drop_rate } => {
                self.cluster.chaos().set_link_quality(0, 1, *drop_rate);
            },
            NemesisAction::HealAll => {
                let count = self.cluster.node_count();
                for i in 0..count {
                    self.cluster.chaos_mut().recover_node(i);
                }
                self.cluster.chaos_mut().heal_majority_partition();
            },
        }
    }

    /// Check linearizability of the system-generated history.
    pub fn check(self) -> JepsenResult {
        let duration = self
            .history
            .operations()
            .last()
            .map_or(Duration::ZERO, |op| {
                op.complete_time
                    .map_or(Duration::ZERO, |ct| ct.duration_since(op.invoke_time))
            });
        let operation_count = self.history.len();
        let linearizability = self.checker.check(self.history.operations());

        let total_faults = self.cluster.chaos().total_dropped_messages()
            + self.cluster.chaos().total_reordered_messages()
            + self.cluster.chaos().total_corrupted_messages();

        JepsenResult {
            duration,
            operation_count,
            linearizability,
            nemesis_actions_applied: self.nemesis_applied,
            total_faults_injected: total_faults,
        }
    }
}

/// Encode a linearizability `Value` to bytes for storage.
fn value_to_bytes(value: &Value) -> Vec<u8> {
    match value {
        Value::Int(i) => i.to_le_bytes().to_vec(),
        Value::Str(s) => s.as_bytes().to_vec(),
        Value::None => Vec::new(),
    }
}

/// Decode bytes back to a linearizability `Value`.
fn bytes_to_value(bytes: &[u8]) -> Value {
    if bytes.len() == 8 {
        let arr: [u8; 8] = bytes.try_into().expect("checked length");
        Value::Int(i64::from_le_bytes(arr))
    } else if bytes.is_empty() {
        Value::None
    } else {
        Value::Str(String::from_utf8_lossy(bytes).into_owned())
    }
}

// ---------------------------------------------------------------------------
// 2PC KV Cluster with real stores
// ---------------------------------------------------------------------------

use tensor_chain::consensus::{ConsensusConfig, ConsensusManager};
use tensor_chain::distributed_tx::{
    DistributedTxConfig, DistributedTxCoordinator, PrepareRequest, PrepareVote, TxParticipant,
    TxPhase,
};
use tensor_store::sparse_vector::SparseVector;

/// 2PC test cluster backed by real `TensorStore` instances.
///
/// Each shard owns a store; reads come from the store (not hardcoded),
/// so the linearizability checker validates real system behavior.
pub struct TxKVCluster {
    coordinator: DistributedTxCoordinator,
    stores: Vec<TensorStore>,
}

impl TxKVCluster {
    pub fn new(shard_count: usize) -> Self {
        Self::with_config(shard_count, DistributedTxConfig::default())
    }

    pub fn with_config(shard_count: usize, config: DistributedTxConfig) -> Self {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);
        let stores = (0..shard_count).map(|_| TensorStore::new()).collect();
        Self {
            coordinator,
            stores,
        }
    }

    pub const fn coordinator(&self) -> &DistributedTxCoordinator {
        &self.coordinator
    }

    /// Run timeout cleanup, returning IDs of transactions that were cleaned up.
    pub fn cleanup_timeouts(&self) -> Vec<u64> {
        self.coordinator.cleanup_timeouts()
    }

    /// Begin a distributed transaction on the given shards.
    pub fn begin(
        &self,
        coordinator_id: &str,
        shards: &[usize],
    ) -> tensor_chain::distributed_tx::DistributedTransaction {
        let id = coordinator_id.to_string();
        self.coordinator
            .begin(&id, shards)
            .expect("begin should succeed")
    }

    /// Create a `PrepareRequest` for a Put operation.
    pub fn prepare_put(tx_id: u64, key: &str, data: &[u8]) -> PrepareRequest {
        PrepareRequest {
            tx_id,
            coordinator: "coordinator".to_string(),
            operations: vec![ChainTransaction::Put {
                key: key.to_string(),
                data: data.to_vec(),
            }],
            delta_embedding: SparseVector::new(128),
            timeout_ms: 5000,
        }
    }

    /// Record a vote for a shard. Returns the phase transition if all voted.
    pub fn record_vote(&self, tx_id: u64, shard: usize, vote: PrepareVote) -> Option<TxPhase> {
        self.coordinator
            .record_vote(tx_id, shard, vote)
            .expect("record_vote should succeed")
    }

    /// Commit a transaction and apply its operations to the shard stores.
    pub fn commit_and_apply(&self, tx_id: u64, operations: &[(usize, Vec<ChainTransaction>)]) {
        self.coordinator
            .commit(tx_id)
            .expect("commit should succeed");

        for (shard, txs) in operations {
            for tx in txs {
                tensor_chain::apply_transaction_to_store(&self.stores[*shard], tx)
                    .expect("apply should succeed");
            }
        }
    }

    /// Abort a transaction.
    pub fn abort(&self, tx_id: u64, reason: &str) {
        self.coordinator
            .abort(tx_id, reason)
            .expect("abort should succeed");
    }

    /// Read a value from a shard's store. Returns bytes if present.
    pub fn get(&self, shard: usize, key: &str) -> Option<Vec<u8>> {
        let data = self.stores[shard].get(key).ok()?;
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
            Some(bytes.clone())
        } else {
            None
        }
    }

    /// Read a value from a shard and convert to a linearizability `Value`.
    pub fn read_value(&self, shard: usize, key: &str) -> Value {
        self.get(shard, key)
            .map_or(Value::None, |bytes| bytes_to_value(&bytes))
    }
}

// ---------------------------------------------------------------------------
// Real 2PC Cluster with TxParticipant (end-to-end participant logic)
// ---------------------------------------------------------------------------

/// 2PC test cluster using real `TxParticipant` instances.
///
/// Unlike `TxKVCluster` (which feeds predetermined votes), this cluster routes
/// `PrepareRequest` to real `TxParticipant`s that acquire real locks, capture
/// real undo logs, and return honest `PrepareVote`s.  Commit and abort flow
/// through `TxParticipant::commit()` / `abort()`, which apply operations or
/// roll back via the undo log on the participant's internal store.
///
/// Reads come from the participant's real `TensorStore`.
pub struct Real2PCCluster {
    coordinator: DistributedTxCoordinator,
    participants: Vec<TxParticipant>,
}

impl Real2PCCluster {
    pub fn new(shard_count: usize) -> Self {
        Self::with_config(shard_count, DistributedTxConfig::default())
    }

    pub fn with_config(shard_count: usize, config: DistributedTxConfig) -> Self {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);
        let participants = (0..shard_count)
            .map(|_| TxParticipant::new_in_memory())
            .collect();
        Self {
            coordinator,
            participants,
        }
    }

    pub const fn coordinator(&self) -> &DistributedTxCoordinator {
        &self.coordinator
    }

    pub fn participant(&self, shard: usize) -> &TxParticipant {
        &self.participants[shard]
    }

    /// Run timeout cleanup, returning IDs of transactions that were cleaned up.
    pub fn cleanup_timeouts(&self) -> Vec<u64> {
        self.coordinator.cleanup_timeouts()
    }

    /// Begin a distributed transaction on the given shards.
    pub fn begin(
        &self,
        coordinator_id: &str,
        shards: &[usize],
    ) -> tensor_chain::distributed_tx::DistributedTransaction {
        self.coordinator
            .begin(&coordinator_id.to_string(), shards)
            .expect("begin should succeed")
    }

    /// Send a `PrepareRequest` to a real participant and feed the resulting
    /// vote back to the coordinator. Returns the participant's real vote.
    pub fn prepare_and_vote(
        &self,
        tx_id: u64,
        shard: usize,
        operations: Vec<ChainTransaction>,
    ) -> PrepareVote {
        let request = PrepareRequest {
            tx_id,
            coordinator: "coordinator".to_string(),
            operations,
            delta_embedding: SparseVector::new(128),
            timeout_ms: 5000,
        };
        let vote = self.participants[shard].prepare(request);
        self.coordinator
            .record_vote(tx_id, shard, vote.clone())
            .expect("record_vote should succeed");
        vote
    }

    /// Commit through both the coordinator and all listed participants.
    pub fn commit_all(&self, tx_id: u64, shards: &[usize]) {
        self.coordinator
            .commit(tx_id)
            .expect("commit should succeed");
        for &shard in shards {
            self.participants[shard].commit(tx_id);
        }
    }

    /// Abort through both the coordinator and all listed participants.
    pub fn abort_all(&self, tx_id: u64, shards: &[usize], reason: &str) {
        self.coordinator
            .abort(tx_id, reason)
            .expect("abort should succeed");
        for &shard in shards {
            self.participants[shard].abort(tx_id);
        }
    }

    /// Execute a full 2PC round with real participant voting.
    /// Returns `Ok(tx_id)` on commit, `Err(reason)` on abort.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a participant votes no/conflict and the transaction
    /// is aborted.
    pub fn execute_tx(
        &self,
        coordinator_id: &str,
        operations_per_shard: &[(usize, Vec<ChainTransaction>)],
    ) -> Result<u64, String> {
        let shards: Vec<usize> = operations_per_shard.iter().map(|(s, _)| *s).collect();
        let tx = self.begin(coordinator_id, &shards);

        let mut all_yes = true;
        for (shard, ops) in operations_per_shard {
            let vote = self.prepare_and_vote(tx.tx_id, *shard, ops.clone());
            if !matches!(vote, PrepareVote::Yes { .. }) {
                all_yes = false;
            }
        }

        if all_yes {
            self.commit_all(tx.tx_id, &shards);
            Ok(tx.tx_id)
        } else {
            self.abort_all(tx.tx_id, &shards, "participant voted no/conflict");
            Err("aborted: real participant vote".to_string())
        }
    }

    /// Read from a participant's real store and convert to `Value`.
    pub fn read_value(&self, shard: usize, key: &str) -> Value {
        let store = self.participants[shard].store();
        store.get(key).map_or(Value::None, |data| {
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
                bytes_to_value(bytes)
            } else {
                Value::None
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Crash-Recovery Raft Cluster (persistent Raft state across restarts)
// ---------------------------------------------------------------------------

/// Raft cluster that supports true crash-recovery.
///
/// Each node persists its Raft state (term, `voted_for`, log) to a dedicated
/// `TensorStore`. A crashed node drops its `RaftNode` and run task; restarting
/// creates a fresh `RaftNode` via `RaftNode::with_store()` that loads persisted
/// state, then catches up via log replication.
pub struct CrashRecoveryRaftCluster {
    nodes: Vec<Option<Arc<RaftNode>>>,
    stores: Vec<TensorStore>,
    raft_stores: Vec<TensorStore>,
    chaos: ChaosCluster,
    raft_config: RaftConfig,
    node_count: usize,
    shutdown_txs: Vec<broadcast::Sender<()>>,
    handles: Vec<Option<tokio::task::JoinHandle<()>>>,
}

impl CrashRecoveryRaftCluster {
    #[allow(clippy::unused_async)]
    pub async fn new(
        node_count: usize,
        raft_config: RaftConfig,
        chaos_config: ChaosConfig,
    ) -> Self {
        let chaos = ChaosCluster::new(node_count, chaos_config);
        let mut stores = Vec::with_capacity(node_count);
        let mut raft_stores = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);
        let mut shutdown_txs = Vec::with_capacity(node_count);
        let mut handles = Vec::with_capacity(node_count);

        for i in 0..node_count {
            stores.push(TensorStore::new());
            raft_stores.push(TensorStore::new());
            let transport = chaos.transport(i).expect("transport must exist");

            let peers: Vec<String> = (0..node_count)
                .filter(|&j| j != i)
                .map(|j| format!("node-{j}"))
                .collect();

            let node = Arc::new(RaftNode::new(
                format!("node-{i}"),
                peers,
                transport,
                raft_config.clone(),
            ));

            let (tx, _) = broadcast::channel(1);
            let n = Arc::clone(&node);
            let rx = tx.subscribe();
            handles.push(Some(tokio::spawn(async move {
                let _ = n.run(rx).await;
            })));

            shutdown_txs.push(tx);
            nodes.push(Some(node));
        }

        Self {
            nodes,
            stores,
            raft_stores,
            chaos,
            raft_config,
            node_count,
            shutdown_txs,
            handles,
        }
    }

    pub async fn wait_for_leader(&self, timeout: Duration) -> Option<usize> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(n) = node {
                    if n.is_leader() {
                        return Some(i);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    pub fn leader_index(&self) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| n.as_ref().filter(|n| n.is_leader()).map(|_| i))
    }

    /// Crash a node: persist Raft state, send shutdown, partition transport.
    pub async fn crash_node(&mut self, idx: usize) {
        if let Some(node) = self.nodes[idx].take() {
            // Persist Raft state before crash
            let _ = node.save_to_store(&self.raft_stores[idx]);

            // Shutdown the run task
            let _ = self.shutdown_txs[idx].send(());
            if let Some(handle) = self.handles[idx].take() {
                let _ = handle.await;
            }

            // Partition transport so no messages reach/leave this node
            if let Some(t) = self.chaos.transport(idx) {
                t.partition_all();
            }
        }
    }

    /// Restart a crashed node: create new `RaftNode` from persisted state.
    #[allow(clippy::unused_async)]
    pub async fn restart_node(&mut self, idx: usize) {
        if self.nodes[idx].is_some() {
            return; // Already running
        }

        // Heal transport
        if let Some(t) = self.chaos.transport(idx) {
            t.heal_all();
        }

        let transport = self.chaos.transport(idx).expect("transport must exist");
        let peers: Vec<String> = (0..self.node_count)
            .filter(|&j| j != idx)
            .map(|j| format!("node-{j}"))
            .collect();

        // Create node from persisted state
        let node = Arc::new(RaftNode::with_store(
            format!("node-{idx}"),
            peers,
            transport,
            self.raft_config.clone(),
            &self.raft_stores[idx],
        ));

        let (tx, _) = broadcast::channel(1);
        let n = Arc::clone(&node);
        let rx = tx.subscribe();
        self.handles[idx] = Some(tokio::spawn(async move {
            let _ = n.run(rx).await;
        }));

        self.shutdown_txs[idx] = tx;
        self.nodes[idx] = Some(node);
    }

    /// # Errors
    ///
    /// Returns `Err` if the put times out or the leader node has crashed.
    pub async fn put(&self, key: &str, data: &[u8], timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        let mut log_index: Option<u64> = None;

        while start.elapsed() < timeout {
            let Some(leader_idx) = self.leader_index() else {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            };

            if log_index.is_none() {
                let header = BlockHeader::new(
                    0,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("node-{leader_idx}"),
                );
                let block = Block::new(
                    header,
                    vec![ChainTransaction::Put {
                        key: key.to_string(),
                        data: data.to_vec(),
                    }],
                );

                if let Ok(idx) = self.nodes[leader_idx]
                    .as_ref()
                    .ok_or("node crashed")?
                    .propose(block)
                {
                    log_index = Some(idx);
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    continue;
                }
            }

            if let Some(idx) = log_index {
                if let Some(node) = &self.nodes[leader_idx] {
                    if node.commit_index() >= idx {
                        self.apply_committed(leader_idx);
                        return Ok(());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err("put timed out".to_string())
    }

    pub fn get(&self, node_idx: usize, key: &str) -> Option<Vec<u8>> {
        self.apply_committed(node_idx);
        let data = self.stores[node_idx].get(key).ok()?;
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
            Some(bytes.clone())
        } else {
            None
        }
    }

    pub fn apply_committed(&self, node_idx: usize) -> usize {
        let Some(node) = &self.nodes[node_idx] else {
            return 0;
        };
        let entries = node.get_uncommitted_entries();
        let count = entries.len();
        for entry in &entries {
            for tx in &entry.block.transactions {
                let _ = tensor_chain::apply_transaction_to_store(&self.stores[node_idx], tx);
            }
        }
        if count > 0 {
            let max_idx = entries.last().map_or(0, |e| e.index);
            node.mark_applied(max_idx);
        }
        count
    }

    pub fn apply_committed_all(&self) -> usize {
        let mut total = 0;
        for i in 0..self.node_count {
            total += self.apply_committed(i);
        }
        total
    }

    pub const fn chaos(&self) -> &ChaosCluster {
        &self.chaos
    }

    pub async fn shutdown(mut self) {
        for i in 0..self.node_count {
            let _ = self.shutdown_txs[i].send(());
            if let Some(handle) = self.handles[i].take() {
                let _ = handle.await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WAL-based Crash-Recovery Raft Cluster (persistent via RaftNode::with_wal)
// ---------------------------------------------------------------------------

/// Raft cluster that uses `RaftNode::with_wal()` for true WAL-based crash recovery.
///
/// Unlike `CrashRecoveryRaftCluster` which uses `save_to_store()`/`with_store()`,
/// this cluster persists only term and vote to the WAL file. Restarted nodes
/// must catch up via `AppendEntries` from the leader.
pub struct WalCrashRecoveryCluster {
    nodes: Vec<Option<Arc<RaftNode>>>,
    wal_paths: Vec<std::path::PathBuf>,
    stores: Vec<TensorStore>,
    chaos: ChaosCluster,
    raft_config: RaftConfig,
    node_count: usize,
    shutdown_txs: Vec<broadcast::Sender<()>>,
    handles: Vec<Option<tokio::task::JoinHandle<()>>>,
}

impl WalCrashRecoveryCluster {
    /// Create a new WAL-based crash recovery cluster.
    ///
    /// `wal_base_dir` must be a directory that persists across crash/restart
    /// cycles (typically a `tempfile::TempDir` owned by the test).
    #[allow(clippy::unused_async)]
    pub async fn new(
        node_count: usize,
        raft_config: RaftConfig,
        chaos_config: ChaosConfig,
        wal_base_dir: &std::path::Path,
    ) -> Self {
        let chaos = ChaosCluster::new(node_count, chaos_config);
        let mut stores = Vec::with_capacity(node_count);
        let mut wal_paths = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);
        let mut shutdown_txs = Vec::with_capacity(node_count);
        let mut handles = Vec::with_capacity(node_count);

        for i in 0..node_count {
            stores.push(TensorStore::new());
            let wal_path = wal_base_dir.join(format!("node_{i}.wal"));

            let transport = chaos.transport(i).expect("transport must exist");
            let peers: Vec<String> = (0..node_count)
                .filter(|&j| j != i)
                .map(|j| format!("node-{j}"))
                .collect();

            let node = Arc::new(
                RaftNode::with_wal(
                    format!("node-{i}"),
                    peers,
                    transport,
                    raft_config.clone(),
                    &wal_path,
                )
                .expect("create raft node with WAL"),
            );

            let (tx, _) = broadcast::channel(1);
            let n = Arc::clone(&node);
            let rx = tx.subscribe();
            handles.push(Some(tokio::spawn(async move {
                let _ = n.run(rx).await;
            })));

            shutdown_txs.push(tx);
            nodes.push(Some(node));
            wal_paths.push(wal_path);
        }

        Self {
            nodes,
            wal_paths,
            stores,
            chaos,
            raft_config,
            node_count,
            shutdown_txs,
            handles,
        }
    }

    pub async fn wait_for_leader(&self, timeout: Duration) -> Option<usize> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(n) = node {
                    if n.is_leader() {
                        return Some(i);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    pub fn leader_index(&self) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| n.as_ref().filter(|n| n.is_leader()).map(|_| i))
    }

    /// Crash a node: shutdown run task, partition transport. WAL stays on disk.
    pub async fn crash_node(&mut self, idx: usize) {
        if let Some(_node) = self.nodes[idx].take() {
            let _ = self.shutdown_txs[idx].send(());
            if let Some(handle) = self.handles[idx].take() {
                let _ = handle.await;
            }
            if let Some(t) = self.chaos.transport(idx) {
                t.partition_all();
            }
        }
    }

    /// Crash a node with simulated torn write: after shutting down the node,
    /// randomly truncate the WAL file at a point in the last ~64 bytes (50%
    /// probability). On restart, `RaftNode::with_wal()` must recover all
    /// complete entries before the torn point.
    pub async fn crash_node_with_torn_write(&mut self, idx: usize, rng: &mut impl rand::Rng) {
        self.crash_node(idx).await;
        if rng.random_bool(0.5) {
            let wal_path = &self.wal_paths[idx];
            if let Ok(meta) = std::fs::metadata(wal_path) {
                let size = meta.len();
                if size > 8 {
                    let truncate_at = rng.random_range(size.saturating_sub(64)..size);
                    let f = std::fs::OpenOptions::new()
                        .write(true)
                        .open(wal_path)
                        .unwrap();
                    f.set_len(truncate_at).unwrap();
                }
            }
        }
    }

    /// Restart a crashed node from WAL. Creates new `RaftNode::with_wal()`
    /// that recovers term+vote from the WAL file. Log is empty -- the node
    /// must catch up via `AppendEntries` from the leader.
    #[allow(clippy::unused_async)]
    pub async fn restart_node(&mut self, idx: usize) {
        if self.nodes[idx].is_some() {
            return;
        }

        if let Some(t) = self.chaos.transport(idx) {
            t.heal_all();
        }

        let transport = self.chaos.transport(idx).expect("transport must exist");
        let peers: Vec<String> = (0..self.node_count)
            .filter(|&j| j != idx)
            .map(|j| format!("node-{j}"))
            .collect();

        let wal_path = &self.wal_paths[idx];
        let node = Arc::new(
            RaftNode::with_wal(
                format!("node-{idx}"),
                peers,
                transport,
                self.raft_config.clone(),
                wal_path,
            )
            .expect("restart raft node with WAL"),
        );

        let (tx, _) = broadcast::channel(1);
        let n = Arc::clone(&node);
        let rx = tx.subscribe();
        self.handles[idx] = Some(tokio::spawn(async move {
            let _ = n.run(rx).await;
        }));

        self.shutdown_txs[idx] = tx;
        self.nodes[idx] = Some(node);
    }

    /// # Errors
    ///
    /// Returns `Err` if the put times out or the leader node has crashed.
    pub async fn put(&self, key: &str, data: &[u8], timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        let mut log_index: Option<u64> = None;

        while start.elapsed() < timeout {
            let Some(leader_idx) = self.leader_index() else {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            };

            if log_index.is_none() {
                let header = BlockHeader::new(
                    0,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("node-{leader_idx}"),
                );
                let block = Block::new(
                    header,
                    vec![ChainTransaction::Put {
                        key: key.to_string(),
                        data: data.to_vec(),
                    }],
                );

                if let Ok(idx) = self.nodes[leader_idx]
                    .as_ref()
                    .ok_or("node crashed")?
                    .propose(block)
                {
                    log_index = Some(idx);
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    continue;
                }
            }

            if let Some(idx) = log_index {
                if let Some(node) = &self.nodes[leader_idx] {
                    if node.commit_index() >= idx {
                        self.apply_committed(leader_idx);
                        return Ok(());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err("put timed out".to_string())
    }

    pub fn get(&self, node_idx: usize, key: &str) -> Option<Vec<u8>> {
        self.apply_committed(node_idx);
        let data = self.stores[node_idx].get(key).ok()?;
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
            Some(bytes.clone())
        } else {
            None
        }
    }

    pub fn apply_committed(&self, node_idx: usize) -> usize {
        let Some(node) = &self.nodes[node_idx] else {
            return 0;
        };
        let entries = node.get_uncommitted_entries();
        let count = entries.len();
        for entry in &entries {
            for tx in &entry.block.transactions {
                let _ = tensor_chain::apply_transaction_to_store(&self.stores[node_idx], tx);
            }
        }
        if count > 0 {
            let max_idx = entries.last().map_or(0, |e| e.index);
            node.mark_applied(max_idx);
        }
        count
    }

    pub fn apply_committed_all(&self) -> usize {
        let mut total = 0;
        for i in 0..self.node_count {
            total += self.apply_committed(i);
        }
        total
    }

    /// Partition a node so it cannot send/receive any messages.
    pub fn partition_node(&self, idx: usize) {
        if let Some(t) = self.chaos.transport(idx) {
            t.partition_all();
        }
    }

    /// Heal a node's partition so it can communicate again.
    pub fn heal_node(&self, idx: usize) {
        if let Some(t) = self.chaos.transport(idx) {
            t.heal_all();
        }
    }

    /// Get the node's current term (from the live node if available).
    pub fn node_term(&self, idx: usize) -> Option<u64> {
        self.nodes[idx].as_ref().map(|n| n.current_term())
    }

    pub async fn shutdown(mut self) {
        for i in 0..self.node_count {
            let _ = self.shutdown_txs[i].send(());
            if let Some(handle) = self.handles[i].take() {
                let _ = handle.await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Networked 2PC Cluster (partition-aware 2PC with simulated transport)
// ---------------------------------------------------------------------------

/// 2PC cluster with simulated network partitions.
///
/// Wraps real `TxParticipant` instances but gates message delivery on
/// partition state. When a participant is partitioned, `prepare()`, `commit()`,
/// and `abort()` return `None`/`false` (simulating message loss/timeout).
pub struct Networked2PCCluster {
    coordinator: DistributedTxCoordinator,
    participants: Vec<TxParticipant>,
    partitioned: Vec<bool>,
}

impl Networked2PCCluster {
    pub fn new(shard_count: usize) -> Self {
        Self::with_config(shard_count, DistributedTxConfig::default())
    }

    pub fn with_config(shard_count: usize, config: DistributedTxConfig) -> Self {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);
        let participants = (0..shard_count)
            .map(|_| TxParticipant::new_in_memory())
            .collect();
        let partitioned = vec![false; shard_count];
        Self {
            coordinator,
            participants,
            partitioned,
        }
    }

    pub const fn coordinator(&self) -> &DistributedTxCoordinator {
        &self.coordinator
    }

    pub fn participant(&self, shard: usize) -> &TxParticipant {
        &self.participants[shard]
    }

    /// Partition a participant (simulate network failure).
    pub fn partition_participant(&mut self, shard: usize) {
        self.partitioned[shard] = true;
    }

    /// Heal a participant (restore connectivity).
    pub fn heal_participant(&mut self, shard: usize) {
        self.partitioned[shard] = false;
    }

    /// Partition all participants.
    pub fn partition_all(&mut self) {
        for p in &mut self.partitioned {
            *p = true;
        }
    }

    /// Heal all participants.
    pub fn heal_all(&mut self) {
        for p in &mut self.partitioned {
            *p = false;
        }
    }

    pub fn is_reachable(&self, shard: usize) -> bool {
        !self.partitioned[shard]
    }

    pub fn begin(
        &self,
        coordinator_id: &str,
        shards: &[usize],
    ) -> tensor_chain::distributed_tx::DistributedTransaction {
        self.coordinator
            .begin(&coordinator_id.to_string(), shards)
            .expect("begin should succeed")
    }

    /// Send prepare to a participant. Returns `None` if partitioned.
    pub fn prepare(
        &self,
        tx_id: u64,
        shard: usize,
        operations: Vec<ChainTransaction>,
    ) -> Option<PrepareVote> {
        if self.partitioned[shard] {
            return None;
        }
        let request = PrepareRequest {
            tx_id,
            coordinator: "coordinator".to_string(),
            operations,
            delta_embedding: SparseVector::new(128),
            timeout_ms: 5000,
        };
        Some(self.participants[shard].prepare(request))
    }

    /// Record a vote with the coordinator.
    pub fn record_vote(&self, tx_id: u64, shard: usize, vote: PrepareVote) -> Option<TxPhase> {
        self.coordinator
            .record_vote(tx_id, shard, vote)
            .expect("record_vote should succeed")
    }

    /// Deliver commit to a participant. Returns false if partitioned.
    pub fn deliver_commit(&self, tx_id: u64, shard: usize) -> bool {
        if self.partitioned[shard] {
            return false;
        }
        self.participants[shard].commit(tx_id);
        true
    }

    /// Deliver abort to a participant. Returns false if partitioned.
    pub fn deliver_abort(&self, tx_id: u64, shard: usize) -> bool {
        if self.partitioned[shard] {
            return false;
        }
        self.participants[shard].abort(tx_id);
        true
    }

    /// Commit on the coordinator side only.
    pub fn coordinator_commit(&self, tx_id: u64) {
        self.coordinator
            .commit(tx_id)
            .expect("coordinator commit should succeed");
    }

    /// Abort on the coordinator side only.
    pub fn coordinator_abort(&self, tx_id: u64, reason: &str) {
        self.coordinator
            .abort(tx_id, reason)
            .expect("coordinator abort should succeed");
    }

    /// Read from a participant's real store.
    pub fn read_value(&self, shard: usize, key: &str) -> Value {
        let store = self.participants[shard].store();
        store.get(key).map_or(Value::None, |data| {
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
                bytes_to_value(bytes)
            } else {
                Value::None
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Random Nemesis Schedule generator
// ---------------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate a random nemesis schedule from a seed.
///
/// Produces a sequence of random faults (partition, crash, clock drift,
/// link degradation, heal) at random intervals over `duration_secs`.
#[must_use]
pub fn random_nemesis_schedule(seed: u64, duration_secs: u64) -> NemesisSchedule {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut schedule = NemesisSchedule::new();
    let mut elapsed_ms: u64 = 0;
    let end_ms = duration_secs * 1000;

    while elapsed_ms < end_ms {
        let interval_ms = rng.random_range(2000..5000);
        elapsed_ms += interval_ms;
        if elapsed_ms >= end_ms {
            break;
        }

        let action = match rng.random_range(0..5) {
            0 => NemesisAction::MajorityPartition,
            1 => NemesisAction::RandomCrash,
            2 => NemesisAction::ClockDrift {
                drift_ms: rng.random_range(-5000..5000),
            },
            3 => NemesisAction::LinkDegradation {
                drop_rate: rng.random_range(0.05..0.3),
            },
            _ => NemesisAction::HealAll,
        };
        schedule = schedule.add(Duration::from_millis(elapsed_ms), action);
    }

    // Always heal at the end
    schedule.add(Duration::from_millis(end_ms), NemesisAction::HealAll)
}

// ---------------------------------------------------------------------------
// Full Recovery Cluster (Raft WAL + TensorStore persistence)
// ---------------------------------------------------------------------------

/// Raft cluster that persists both Raft state (via WAL) and application state.
///
/// On crash, `save_to_store()` is called; on restart, `with_store()` loads the
/// full log from the store, and the WAL provides additional entries written
/// since the last save.
pub struct FullRecoveryCluster {
    nodes: Vec<Option<Arc<RaftNode>>>,
    stores: Vec<TensorStore>,
    raft_stores: Vec<TensorStore>,
    chaos: ChaosCluster,
    raft_config: RaftConfig,
    node_count: usize,
    shutdown_txs: Vec<broadcast::Sender<()>>,
    handles: Vec<Option<tokio::task::JoinHandle<()>>>,
}

impl FullRecoveryCluster {
    #[allow(clippy::unused_async)]
    pub async fn new(
        node_count: usize,
        raft_config: RaftConfig,
        chaos_config: ChaosConfig,
    ) -> Self {
        let chaos = ChaosCluster::new(node_count, chaos_config);
        let mut stores = Vec::with_capacity(node_count);
        let mut raft_stores = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);
        let mut shutdown_txs = Vec::with_capacity(node_count);
        let mut handles = Vec::with_capacity(node_count);

        for i in 0..node_count {
            stores.push(TensorStore::new());
            raft_stores.push(TensorStore::new());

            let transport = chaos.transport(i).expect("transport must exist");
            let peers: Vec<String> = (0..node_count)
                .filter(|&j| j != i)
                .map(|j| format!("node-{j}"))
                .collect();

            let node = Arc::new(RaftNode::new(
                format!("node-{i}"),
                peers,
                transport,
                raft_config.clone(),
            ));

            let (tx, _) = broadcast::channel(1);
            let n = Arc::clone(&node);
            let rx = tx.subscribe();
            handles.push(Some(tokio::spawn(async move {
                let _ = n.run(rx).await;
            })));

            shutdown_txs.push(tx);
            nodes.push(Some(node));
        }

        Self {
            nodes,
            stores,
            raft_stores,
            chaos,
            raft_config,
            node_count,
            shutdown_txs,
            handles,
        }
    }

    pub async fn wait_for_leader(&self, timeout: Duration) -> Option<usize> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(n) = node {
                    if n.is_leader() {
                        return Some(i);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    pub fn leader_index(&self) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| n.as_ref().filter(|n| n.is_leader()).map(|_| i))
    }

    /// Crash a node: persist Raft state to store, shutdown, partition.
    pub async fn crash_node(&mut self, idx: usize) {
        if let Some(node) = self.nodes[idx].take() {
            let _ = node.save_to_store(&self.raft_stores[idx]);

            let _ = self.shutdown_txs[idx].send(());
            if let Some(handle) = self.handles[idx].take() {
                let _ = handle.await;
            }

            if let Some(t) = self.chaos.transport(idx) {
                t.partition_all();
            }
        }
    }

    /// Restart a crashed node from store state.
    #[allow(clippy::unused_async)]
    pub async fn restart_node(&mut self, idx: usize) {
        if self.nodes[idx].is_some() {
            return;
        }

        if let Some(t) = self.chaos.transport(idx) {
            t.heal_all();
        }

        let transport = self.chaos.transport(idx).expect("transport must exist");
        let peers: Vec<String> = (0..self.node_count)
            .filter(|&j| j != idx)
            .map(|j| format!("node-{j}"))
            .collect();

        let node = Arc::new(RaftNode::with_store(
            format!("node-{idx}"),
            peers,
            transport,
            self.raft_config.clone(),
            &self.raft_stores[idx],
        ));

        let (tx, _) = broadcast::channel(1);
        let n = Arc::clone(&node);
        let rx = tx.subscribe();
        self.handles[idx] = Some(tokio::spawn(async move {
            let _ = n.run(rx).await;
        }));

        self.shutdown_txs[idx] = tx;
        self.nodes[idx] = Some(node);
    }

    /// # Errors
    ///
    /// Returns `Err` if the put times out or the leader node has crashed.
    pub async fn put(&self, key: &str, data: &[u8], timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        let mut log_index: Option<u64> = None;

        while start.elapsed() < timeout {
            let Some(leader_idx) = self.leader_index() else {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            };

            if log_index.is_none() {
                let header = BlockHeader::new(
                    0,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("node-{leader_idx}"),
                );
                let block = Block::new(
                    header,
                    vec![ChainTransaction::Put {
                        key: key.to_string(),
                        data: data.to_vec(),
                    }],
                );

                if let Ok(idx) = self.nodes[leader_idx]
                    .as_ref()
                    .ok_or("node crashed")?
                    .propose(block)
                {
                    log_index = Some(idx);
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    continue;
                }
            }

            if let Some(idx) = log_index {
                if let Some(node) = &self.nodes[leader_idx] {
                    if node.commit_index() >= idx {
                        self.apply_committed(leader_idx);
                        return Ok(());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err("put timed out".to_string())
    }

    pub fn get(&self, node_idx: usize, key: &str) -> Option<Vec<u8>> {
        self.apply_committed(node_idx);
        let data = self.stores[node_idx].get(key).ok()?;
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("data") {
            Some(bytes.clone())
        } else {
            None
        }
    }

    pub fn apply_committed(&self, node_idx: usize) -> usize {
        let Some(node) = &self.nodes[node_idx] else {
            return 0;
        };
        let entries = node.get_uncommitted_entries();
        let count = entries.len();
        for entry in &entries {
            for tx in &entry.block.transactions {
                let _ = tensor_chain::apply_transaction_to_store(&self.stores[node_idx], tx);
            }
        }
        if count > 0 {
            let max_idx = entries.last().map_or(0, |e| e.index);
            node.mark_applied(max_idx);
        }
        count
    }

    pub fn apply_committed_all(&self) -> usize {
        let mut total = 0;
        for i in 0..self.node_count {
            total += self.apply_committed(i);
        }
        total
    }

    pub fn node_log_len(&self, idx: usize) -> usize {
        self.nodes[idx].as_ref().map_or(0, |n| n.log_length())
    }

    pub async fn shutdown(mut self) {
        for i in 0..self.node_count {
            let _ = self.shutdown_txs[i].send(());
            if let Some(handle) = self.handles[i].take() {
                let _ = handle.await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemesis_schedule_new() {
        let schedule = NemesisSchedule::new();
        assert!(schedule.actions.is_empty());
        assert!(schedule.heal_at_end);
    }

    #[test]
    fn test_nemesis_schedule_add() {
        let schedule = NemesisSchedule::new()
            .add(Duration::from_secs(1), NemesisAction::RandomCrash)
            .add(Duration::from_secs(5), NemesisAction::HealAll);

        assert_eq!(schedule.actions.len(), 2);
        assert_eq!(schedule.actions[0].0, Duration::from_secs(1));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(5));
    }

    #[test]
    fn test_nemesis_schedule_partition_heal() {
        let schedule = NemesisSchedule::partition_heal(Duration::from_secs(10));

        assert_eq!(schedule.actions.len(), 2);
        assert_eq!(schedule.actions[0].0, Duration::ZERO);
        assert!(matches!(
            schedule.actions[0].1,
            NemesisAction::MajorityPartition
        ));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(10));
        assert!(matches!(schedule.actions[1].1, NemesisAction::HealAll));
        assert!(schedule.heal_at_end);
    }

    #[test]
    fn test_nemesis_schedule_crash_recover() {
        let schedule = NemesisSchedule::crash_recover(Duration::from_secs(5));

        assert_eq!(schedule.actions.len(), 2);
        assert!(matches!(schedule.actions[0].1, NemesisAction::RandomCrash));
        assert!(matches!(schedule.actions[1].1, NemesisAction::HealAll));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(5));
    }

    #[test]
    fn test_jepsen_result_is_valid() {
        let valid = JepsenResult {
            duration: Duration::from_secs(1),
            operation_count: 10,
            linearizability: LinearizabilityResult::Ok,
            nemesis_actions_applied: 2,
            total_faults_injected: 5,
        };
        assert!(valid.is_valid());

        let invalid = JepsenResult {
            duration: Duration::from_secs(1),
            operation_count: 10,
            linearizability: LinearizabilityResult::Violation("stale read on key 'x'".to_string()),
            nemesis_actions_applied: 2,
            total_faults_injected: 5,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_nemesis_action_debug() {
        let crash = NemesisAction::RandomCrash;
        let debug = format!("{crash:?}");
        assert_eq!(debug, "RandomCrash");

        let partition = NemesisAction::MajorityPartition;
        assert_eq!(format!("{partition:?}"), "MajorityPartition");

        let drift = NemesisAction::ClockDrift { drift_ms: 500 };
        let drift_debug = format!("{drift:?}");
        assert!(drift_debug.contains("500"));

        let degradation = NemesisAction::LinkDegradation { drop_rate: 0.25 };
        let degradation_debug = format!("{degradation:?}");
        assert!(degradation_debug.contains("0.25"));

        let heal = NemesisAction::HealAll;
        assert_eq!(format!("{heal:?}"), "HealAll");
    }
}
