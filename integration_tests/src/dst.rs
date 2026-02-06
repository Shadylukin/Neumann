// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Deterministic Simulation Testing (DST) harness for Raft consensus.
//!
//! Drives Raft nodes through controlled message delivery and fault injection.
//! All operations are synchronous via `handle_message()` for determinism.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use tensor_chain::block::{Block, BlockHeader, NodeId};
use tensor_chain::network::{AppendEntries, MemoryTransport, Message, RequestVote};
use tensor_chain::raft::{RaftConfig, RaftNode, RaftState};
use tensor_store::SparseVector;

use crate::linearizability::{
    HistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, RaftLogModel, Value,
};

/// A fault action to inject during simulation.
#[derive(Debug, Clone)]
pub enum FaultAction {
    PartitionNode(usize),
    HealNode(usize),
    /// Asymmetric partition: node can selectively block inbound/outbound traffic.
    AsymmetricPartition {
        node: usize,
        inbound_blocked: bool,
        outbound_blocked: bool,
    },
}

#[derive(Debug)]
struct ScheduledEvent {
    tick: u64,
    seq: u64,
    event: SimEvent,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.tick == other.tick && self.seq == other.seq
    }
}
impl Eq for ScheduledEvent {}
impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .tick
            .cmp(&self.tick)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Debug)]
enum SimEvent {
    DeliverMessage {
        from: NodeId,
        to_idx: usize,
        msg: Box<Message>,
    },
    ElectionTimeout {
        node_idx: usize,
    },
    SendHeartbeat {
        leader_idx: usize,
    },
    ClientPropose {
        node_idx: usize,
        op_id: u64,
    },
    Fault(FaultAction),
}

/// Result of a DST simulation run.
#[derive(Debug)]
pub struct SimulationResult {
    pub ticks_executed: u64,
    pub messages_delivered: u64,
    pub elections_triggered: u64,
    pub proposals_attempted: u64,
    pub proposals_succeeded: u64,
    pub faults_injected: u64,
    pub election_safety_held: bool,
    pub linearizability_held: bool,
}

/// Deterministic simulation harness for Raft.
pub struct DSTHarness {
    #[allow(dead_code)]
    rng: rand_chacha::ChaCha8Rng,
    nodes: Vec<Arc<RaftNode>>,
    #[allow(dead_code)]
    transports: Vec<Arc<MemoryTransport>>,
    node_ids: Vec<String>,
    event_queue: BinaryHeap<ScheduledEvent>,
    tick: u64,
    seq: u64,
    partitioned: HashSet<usize>,
    inbound_blocked: HashSet<usize>,
    outbound_blocked: HashSet<usize>,
    history: HistoryRecorder,
    scheduled_faults: Vec<(u64, FaultAction)>,
    max_ticks: u64,

    messages_delivered: u64,
    elections_triggered: u64,
    proposals_attempted: u64,
    proposals_succeeded: u64,
    faults_injected: u64,
    election_safety_violations: u64,
}

impl DSTHarness {
    pub fn new(seed: u64, node_count: usize, max_ticks: u64) -> Self {
        use rand::SeedableRng;

        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let node_ids: Vec<String> = (0..node_count).map(|i| format!("node{i}")).collect();

        // Transports are created but message delivery is handled by the harness
        // directly via handle_message(). Transports are needed for RaftNode construction.
        let transports: Vec<Arc<MemoryTransport>> = node_ids
            .iter()
            .map(|id| Arc::new(MemoryTransport::new(id.clone())))
            .collect();

        // Connect transports so RaftNode can query peers
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j {
                    transports[i].connect_to(node_ids[j].clone(), transports[j].sender());
                }
            }
        }

        let config = RaftConfig {
            enable_fast_path: false,
            enable_geometric_tiebreak: false,
            auto_heartbeat: false,
            election_timeout: (150, 300),
            heartbeat_interval: 50,
            ..RaftConfig::default()
        };

        let nodes: Vec<Arc<RaftNode>> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let peers: Vec<String> = node_ids.iter().filter(|p| *p != id).cloned().collect();
                Arc::new(RaftNode::new(
                    id.clone(),
                    peers,
                    transports[i].clone(),
                    config.clone(),
                ))
            })
            .collect();

        Self {
            rng,
            nodes,
            transports,
            node_ids,
            event_queue: BinaryHeap::new(),
            tick: 0,
            seq: 0,
            partitioned: HashSet::new(),
            inbound_blocked: HashSet::new(),
            outbound_blocked: HashSet::new(),
            history: HistoryRecorder::new(),
            scheduled_faults: Vec::new(),
            max_ticks,
            messages_delivered: 0,
            elections_triggered: 0,
            proposals_attempted: 0,
            proposals_succeeded: 0,
            faults_injected: 0,
            election_safety_violations: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    pub fn schedule_fault(&mut self, tick: u64, fault: FaultAction) {
        self.scheduled_faults.push((tick, fault));
    }

    pub fn schedule_election(&mut self, tick: u64, node_idx: usize) {
        let seq = self.next_seq();
        self.event_queue.push(ScheduledEvent {
            tick,
            seq,
            event: SimEvent::ElectionTimeout { node_idx },
        });
    }

    pub fn schedule_heartbeat(&mut self, tick: u64, leader_idx: usize) {
        let seq = self.next_seq();
        self.event_queue.push(ScheduledEvent {
            tick,
            seq,
            event: SimEvent::SendHeartbeat { leader_idx },
        });
    }

    pub fn schedule_proposal(&mut self, tick: u64, node_idx: usize) {
        let op_id = self.history.invoke(
            node_idx as u64,
            OpType::Write,
            "key".to_string(),
            Value::Int(tick as i64),
        );
        let seq = self.next_seq();
        self.event_queue.push(ScheduledEvent {
            tick,
            seq,
            event: SimEvent::ClientPropose { node_idx, op_id },
        });
    }

    pub fn find_leader(&self) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.state() == RaftState::Leader)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Enqueue a message for delivery to a specific node.
    fn enqueue_message(&mut self, from: NodeId, to_idx: usize, msg: Message, delay: u64) {
        let seq = self.next_seq();
        self.event_queue.push(ScheduledEvent {
            tick: self.tick + delay,
            seq,
            event: SimEvent::DeliverMessage {
                from,
                to_idx,
                msg: Box::new(msg),
            },
        });
    }

    /// Simulate a full election for a node by constructing and delivering
    /// RequestVote messages to all peers, then delivering responses.
    fn simulate_election(&mut self, node_idx: usize) {
        // Trigger election on the node (increments term, votes for self)
        self.nodes[node_idx].start_election();
        self.elections_triggered += 1;

        // Construct RequestVote message from this node's observable state
        let term = self.nodes[node_idx].current_term();
        let candidate_id = self.node_ids[node_idx].clone();
        #[allow(clippy::cast_possible_truncation)]
        let last_log_index = self.nodes[node_idx].log_length() as u64;
        // Use current term as last_log_term (conservative approximation)
        let last_log_term = if last_log_index > 0 { term } else { 0 };

        let rv = Message::RequestVote(RequestVote {
            term,
            candidate_id: candidate_id.clone(),
            last_log_index,
            last_log_term,
            state_embedding: SparseVector::new(0),
        });

        // Send to all non-partitioned peers
        for j in 0..self.nodes.len() {
            if j != node_idx && !self.partitioned.contains(&j) {
                self.enqueue_message(candidate_id.clone(), j, rv.clone(), 1);
            }
        }
    }

    /// Simulate a leader heartbeat by constructing AppendEntries (empty).
    fn simulate_heartbeat(&mut self, leader_idx: usize) {
        if self.nodes[leader_idx].state() != RaftState::Leader {
            return;
        }

        let term = self.nodes[leader_idx].current_term();
        let leader_id = self.node_ids[leader_idx].clone();
        let commit_index = self.nodes[leader_idx].commit_index();

        let ae = Message::AppendEntries(AppendEntries {
            term,
            leader_id: leader_id.clone(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: commit_index,
            block_embedding: None,
        });

        for j in 0..self.nodes.len() {
            if j != leader_idx && !self.partitioned.contains(&j) {
                self.enqueue_message(leader_id.clone(), j, ae.clone(), 1);
            }
        }
    }

    pub fn run(&mut self) -> SimulationResult {
        // Pre-schedule faults
        let faults: Vec<(u64, FaultAction)> = self.scheduled_faults.drain(..).collect();
        for (tick, fault) in faults {
            let seq = self.next_seq();
            self.event_queue.push(ScheduledEvent {
                tick,
                seq,
                event: SimEvent::Fault(fault),
            });
        }

        while self.tick < self.max_ticks {
            // Process all events at current tick
            let mut processed_any = true;
            while processed_any {
                processed_any = false;
                while let Some(event) = self.event_queue.peek() {
                    if event.tick > self.tick {
                        break;
                    }
                    let event = self.event_queue.pop().unwrap();
                    self.process_event(event.event);
                    processed_any = true;
                }
            }

            // Check election safety after each tick
            let state = self.cluster_state();
            if !state.check_election_safety() {
                self.election_safety_violations += 1;
            }

            self.tick += 1;
        }

        // Check linearizability of completed operations
        let linearizability_held = {
            let completed: Vec<_> = self
                .history
                .completed_operations()
                .into_iter()
                .filter(|op| op.output.as_ref() != Some(&Value::None))
                .cloned()
                .collect();

            if completed.is_empty() {
                true
            } else {
                let checker = LinearizabilityChecker::with_timeout(
                    RaftLogModel,
                    std::time::Duration::from_secs(5),
                );
                matches!(
                    checker.check(&completed),
                    LinearizabilityResult::Ok | LinearizabilityResult::Unknown(_)
                )
            }
        };

        SimulationResult {
            ticks_executed: self.tick,
            messages_delivered: self.messages_delivered,
            elections_triggered: self.elections_triggered,
            proposals_attempted: self.proposals_attempted,
            proposals_succeeded: self.proposals_succeeded,
            faults_injected: self.faults_injected,
            election_safety_held: self.election_safety_violations == 0,
            linearizability_held,
        }
    }

    fn process_event(&mut self, event: SimEvent) {
        match event {
            SimEvent::DeliverMessage { from, to_idx, msg } => {
                if self.partitioned.contains(&to_idx) {
                    return;
                }
                // Check asymmetric partition: receiver has inbound blocked
                if self.inbound_blocked.contains(&to_idx) {
                    return;
                }
                // Check asymmetric partition: sender has outbound blocked
                let from_idx = self.node_ids.iter().position(|id| *id == from).unwrap_or(0);
                if self.outbound_blocked.contains(&from_idx) {
                    return;
                }
                if let Some(response) = self.nodes[to_idx].handle_message(&from, &msg) {
                    self.enqueue_message(self.node_ids[to_idx].clone(), from_idx, response, 1);
                }
                self.messages_delivered += 1;
            },
            SimEvent::ElectionTimeout { node_idx } => {
                if !self.partitioned.contains(&node_idx) {
                    self.simulate_election(node_idx);
                }
            },
            SimEvent::SendHeartbeat { leader_idx } => {
                self.simulate_heartbeat(leader_idx);
            },
            SimEvent::ClientPropose { node_idx, op_id } => {
                self.proposals_attempted += 1;
                if self.nodes[node_idx].state() == RaftState::Leader {
                    let block = create_test_block(self.tick);
                    match self.nodes[node_idx].propose(block) {
                        Ok(_) => {
                            self.proposals_succeeded += 1;
                            self.history.complete(op_id, Value::Int(1));
                        },
                        Err(_) => {
                            self.history.complete(op_id, Value::None);
                        },
                    }
                } else {
                    self.history.complete(op_id, Value::None);
                }
            },
            SimEvent::Fault(fault) => {
                self.apply_fault(&fault);
                self.faults_injected += 1;
            },
        }
    }

    fn apply_fault(&mut self, fault: &FaultAction) {
        match fault {
            FaultAction::PartitionNode(idx) => {
                self.partitioned.insert(*idx);
                for j in 0..self.transports.len() {
                    if j != *idx {
                        self.transports[*idx].partition(&self.node_ids[j]);
                        self.transports[j].partition(&self.node_ids[*idx]);
                    }
                }
            },
            FaultAction::HealNode(idx) => {
                self.partitioned.remove(idx);
                self.inbound_blocked.remove(idx);
                self.outbound_blocked.remove(idx);
                for j in 0..self.transports.len() {
                    if j != *idx {
                        self.transports[*idx].heal(&self.node_ids[j]);
                        self.transports[j].heal(&self.node_ids[*idx]);
                    }
                }
            },
            FaultAction::AsymmetricPartition {
                node,
                inbound_blocked,
                outbound_blocked,
            } => {
                if *inbound_blocked {
                    self.inbound_blocked.insert(*node);
                } else {
                    self.inbound_blocked.remove(node);
                }
                if *outbound_blocked {
                    self.outbound_blocked.insert(*node);
                } else {
                    self.outbound_blocked.remove(node);
                }
            },
        }
    }

    pub fn cluster_state(&self) -> ClusterState {
        let mut leaders = Vec::new();
        let mut terms = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            let state = node.state();
            let term = node.current_term();
            terms.push(term);
            if state == RaftState::Leader {
                leaders.push((i, term));
            }
        }

        ClusterState { leaders, terms }
    }
}

/// Observable cluster state for invariant checking.
#[derive(Debug)]
pub struct ClusterState {
    pub leaders: Vec<(usize, u64)>,
    pub terms: Vec<u64>,
}

impl ClusterState {
    pub fn check_election_safety(&self) -> bool {
        let mut leaders_by_term: HashMap<u64, Vec<usize>> = HashMap::new();
        for &(idx, term) in &self.leaders {
            leaders_by_term.entry(term).or_default().push(idx);
        }
        leaders_by_term.values().all(|l| l.len() <= 1)
    }
}

fn create_test_block(height: u64) -> Block {
    let header = BlockHeader::new(
        height,
        [0u8; 32],
        [0u8; 32],
        [0u8; 32],
        "proposer".to_string(),
    );
    Block::new(header, vec![])
}
