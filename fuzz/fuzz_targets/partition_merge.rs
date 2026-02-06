// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz target for partition merge reconciliation.
//!
//! Tests the merge protocol state machine and reconciler logic with
//! arbitrary inputs to find edge cases.

#![no_main]

use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};
use tensor_chain::{
    DataReconciler, MembershipReconciler, MembershipViewSummary, PartitionStateSummary,
    PendingTxState, TransactionReconciler,
};
use tensor_chain::distributed_tx::TxPhase;
use tensor_chain::gossip::GossipNodeState;
use tensor_chain::membership::NodeHealth;
use tensor_store::SparseVector;

/// Fuzzable membership view summary.
#[derive(Arbitrary, Debug)]
struct FuzzMembershipView {
    node_id: String,
    lamport_time: u64,
    generation: u64,
    states: Vec<FuzzNodeState>,
}

/// Fuzzable node state.
#[derive(Arbitrary, Debug)]
struct FuzzNodeState {
    node_id: String,
    health: u8,
    timestamp: u64,
    incarnation: u64,
}

/// Fuzzable partition state summary.
#[derive(Arbitrary, Debug)]
struct FuzzStateSummary {
    node_id: String,
    last_committed_index: u64,
    last_committed_term: u64,
    state_hash: [u8; 32],
    embedding_indices: Vec<(u32, f32)>,
}

/// Fuzzable pending transaction.
#[derive(Arbitrary, Debug)]
struct FuzzPendingTx {
    tx_id: u64,
    coordinator: String,
    phase: u8,
    votes: Vec<(u32, bool)>,
    started_at: u64,
}

/// Input for merge protocol fuzzing.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    local_view: FuzzMembershipView,
    remote_view: FuzzMembershipView,
    local_state: FuzzStateSummary,
    remote_state: FuzzStateSummary,
    local_txs: Vec<FuzzPendingTx>,
    remote_txs: Vec<FuzzPendingTx>,
    data_thresholds: (f32, f32),
    tx_timeout_ms: u64,
}

fn health_from_u8(v: u8) -> NodeHealth {
    match v % 4 {
        0 => NodeHealth::Healthy,
        1 => NodeHealth::Degraded,
        2 => NodeHealth::Failed,
        _ => NodeHealth::Unknown,
    }
}

fn phase_from_u8(v: u8) -> TxPhase {
    match v % 6 {
        0 => TxPhase::Preparing,
        1 => TxPhase::Prepared,
        2 => TxPhase::Committing,
        3 => TxPhase::Committed,
        4 => TxPhase::Aborting,
        _ => TxPhase::Aborted,
    }
}

fn to_membership_view(fuzz: &FuzzMembershipView) -> MembershipViewSummary {
    let states: Vec<GossipNodeState> = fuzz
        .states
        .iter()
        .map(|s| GossipNodeState {
            node_id: s.node_id.clone(),
            health: health_from_u8(s.health),
            timestamp: s.timestamp,
            updated_at: s.timestamp,
            incarnation: s.incarnation,
        })
        .collect();

    MembershipViewSummary::new(fuzz.node_id.clone(), fuzz.lamport_time, fuzz.generation)
        .with_states(states)
}

fn to_state_summary(fuzz: &FuzzStateSummary) -> PartitionStateSummary {
    let mut summary = PartitionStateSummary::new(fuzz.node_id.clone())
        .with_log_position(fuzz.last_committed_index, fuzz.last_committed_term)
        .with_hash(fuzz.state_hash);

    // Create sparse vector from fuzz indices
    if !fuzz.embedding_indices.is_empty() {
        let max_dim = fuzz.embedding_indices.iter().map(|(i, _)| *i).max().unwrap_or(0) as usize + 1;
        let dim = max_dim.max(16).min(1024);
        let mut emb = SparseVector::new(dim);
        for (idx, val) in &fuzz.embedding_indices {
            if (*idx as usize) < dim {
                emb.set(*idx as usize, *val);
            }
        }
        summary = summary.with_embedding(emb);
    }

    summary
}

fn to_pending_tx(fuzz: &FuzzPendingTx) -> PendingTxState {
    let mut tx = PendingTxState::new(
        fuzz.tx_id,
        fuzz.coordinator.clone(),
        phase_from_u8(fuzz.phase),
    );
    tx.started_at = fuzz.started_at;
    for (shard, vote) in &fuzz.votes {
        tx.votes.insert(*shard as usize, *vote);
    }
    tx
}

fuzz_target!(|input: FuzzInput| {
    // Test membership reconciliation
    let local_view = to_membership_view(&input.local_view);
    let remote_view = to_membership_view(&input.remote_view);
    let (merged_view, _conflicts) = match MembershipReconciler::merge(&local_view, &remote_view) {
        Ok(result) => result,
        Err(_) => return, // Invalid state detected, skip this input
    };

    // Verify merged view properties
    // 1. All nodes from both views should be in merged
    for state in &local_view.node_states {
        assert!(
            merged_view.node_states.iter().any(|s| s.node_id == state.node_id),
            "local node missing from merged view"
        );
    }
    for state in &remote_view.node_states {
        assert!(
            merged_view.node_states.iter().any(|s| s.node_id == state.node_id),
            "remote node missing from merged view"
        );
    }

    // 2. Lamport time should advance
    assert!(
        merged_view.lamport_time >= local_view.lamport_time,
        "lamport time should not decrease"
    );
    assert!(
        merged_view.lamport_time >= remote_view.lamport_time,
        "lamport time should not decrease"
    );

    // Test data reconciliation
    let local_state = to_state_summary(&input.local_state);
    let remote_state = to_state_summary(&input.remote_state);

    let orthogonal = input.data_thresholds.0.abs().clamp(0.01, 0.5);
    let identical = input.data_thresholds.1.abs().clamp(0.5, 0.999);
    let reconciler = DataReconciler::new(orthogonal, identical);
    let _result = reconciler.reconcile(&local_state, &remote_state);

    // Test transaction reconciliation
    let local_txs: Vec<_> = input.local_txs.iter().map(to_pending_tx).collect();
    let remote_txs: Vec<_> = input.remote_txs.iter().map(to_pending_tx).collect();

    let timeout = input.tx_timeout_ms.max(1); // Avoid zero timeout
    let tx_reconciler = TransactionReconciler { tx_timeout_ms: timeout };
    let tx_result = match tx_reconciler.reconcile(&local_txs, &remote_txs) {
        Ok(result) => result,
        Err(_) => return, // Invalid state detected, skip this input
    };

    // Verify transaction reconciliation properties
    // 1. Each tx should be in exactly one of: to_commit, to_abort, or neither (if complete)
    for tx in local_txs.iter().chain(remote_txs.iter()) {
        let in_commit = tx_result.to_commit.contains(&tx.tx_id);
        let in_abort = tx_result.to_abort.contains(&tx.tx_id);
        assert!(
            !(in_commit && in_abort),
            "transaction {} in both commit and abort",
            tx.tx_id
        );
    }
});
