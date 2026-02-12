// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Loom-based concurrency verification tests for Raft lock ordering.
//!
//! These tests use a mock that mirrors the phased locking discipline from
//! `raft.rs`. The real `RaftNode` has 15+ RwLocks and uses `std::time::Instant`
//! (not loom-compatible), so we replicate only the lock ordering patterns that
//! matter for deadlock freedom:
//!
//! - Lock hierarchy: persistent -> leadership -> volatile
//! - Explicit drops before calling methods with different lock ordering
//! - Scoped blocks for brief leadership writes inside persistent write scope
//!
//! Run with: cargo nextest run --package tensor_chain --features loom -E 'test(loom_raft)'

#![cfg(feature = "loom")]

use loom::sync::{Arc, Mutex, RwLock};
use loom::thread;

// Mirror of the three core locks from RaftNode with their state.
struct MockRaftNode {
    // Lock ordering: persistent -> leadership -> volatile
    persistent: RwLock<PersistentState>,
    leadership: RwLock<LeadershipState>,
    volatile: RwLock<VolatileState>,
    votes_received: RwLock<Vec<String>>,
    wal: Mutex<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct PersistentState {
    current_term: u64,
    voted_for: Option<String>,
    log: Vec<(u64, Vec<u8>)>,
}

#[derive(Debug, Clone, PartialEq)]
enum Role {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
struct LeadershipState {
    role: Role,
    current_leader: Option<String>,
    next_index: Vec<u64>,
    match_index: Vec<u64>,
}

#[derive(Debug, Clone)]
struct VolatileState {
    commit_index: u64,
}

impl MockRaftNode {
    fn new(node_id: &str, peer_count: usize) -> Self {
        Self {
            persistent: RwLock::new(PersistentState {
                current_term: 0,
                voted_for: Some(node_id.to_string()),
                log: Vec::new(),
            }),
            leadership: RwLock::new(LeadershipState {
                role: Role::Follower,
                current_leader: None,
                next_index: vec![0; peer_count],
                match_index: vec![0; peer_count],
            }),
            volatile: RwLock::new(VolatileState { commit_index: 0 }),
            votes_received: RwLock::new(Vec::new()),
            wal: Mutex::new(Vec::new()),
        }
    }

    // Mirrors raft.rs start_election (line 2467):
    // persistent.write -> leadership.write -> votes_received.write
    fn start_election(&self, node_id: &str) {
        // Phase 1: persistent.write - increment term, vote for self
        let term = {
            let mut ps = self.persistent.write().unwrap();
            ps.current_term += 1;
            ps.voted_for = Some(node_id.to_string());
            ps.current_term
        };

        // Phase 2: leadership.write - become candidate
        {
            let mut ls = self.leadership.write().unwrap();
            ls.role = Role::Candidate;
            ls.current_leader = None;
        }

        // Phase 3: votes_received.write - reset and add self-vote
        {
            let mut votes = self.votes_received.write().unwrap();
            votes.clear();
            votes.push(node_id.to_string());
        }

        // WAL: persist vote
        {
            let mut wal = self.wal.lock().unwrap();
            wal.push(term as u8);
        }
    }

    // Mirrors raft.rs handle_append_entries (line 2171):
    // persistent.write -> { leadership.write (scoped) } -> volatile.write
    fn handle_append_entries(
        &self,
        leader_term: u64,
        leader_id: &str,
        entries: &[(u64, Vec<u8>)],
        leader_commit: u64,
    ) {
        // Phase 1: persistent.write - check/update term, append entries
        let should_update_commit = {
            let mut ps = self.persistent.write().unwrap();
            if leader_term < ps.current_term {
                return;
            }
            if leader_term > ps.current_term {
                ps.current_term = leader_term;
                ps.voted_for = None;
            }

            // Scoped leadership.write - become follower, update leader
            // (mirrors the scoped block at raft.rs:2198-2209)
            {
                let mut ls = self.leadership.write().unwrap();
                ls.role = Role::Follower;
                ls.current_leader = Some(leader_id.to_string());
            } // leadership released before continuing with persistent

            // Append entries while still holding persistent
            for entry in entries {
                ps.log.push(entry.clone());
            }

            leader_commit > 0
        }; // persistent released

        // Phase 2: volatile.write - advance commit index
        if should_update_commit {
            let mut vs = self.volatile.write().unwrap();
            if leader_commit > vs.commit_index {
                vs.commit_index = leader_commit;
            }
        }

        // WAL: persist entries
        {
            let mut wal = self.wal.lock().unwrap();
            wal.push(entries.len() as u8);
        }
    }

    // Mirrors raft.rs handle_request_vote (line 1787):
    // persistent.write -> leadership.write (nested when stepping down)
    fn handle_request_vote(
        &self,
        candidate_term: u64,
        candidate_id: &str,
        last_log_index: u64,
    ) -> bool {
        let mut ps = self.persistent.write().unwrap();

        // Step down if candidate has higher term
        if candidate_term > ps.current_term {
            ps.current_term = candidate_term;
            ps.voted_for = None;

            // Nested leadership.write (mirrors raft.rs:1811)
            let mut ls = self.leadership.write().unwrap();
            ls.role = Role::Follower;
            ls.current_leader = None;
        }

        if candidate_term < ps.current_term {
            return false;
        }

        // Check if we can grant vote
        let can_vote = ps.voted_for.is_none() || ps.voted_for.as_deref() == Some(candidate_id);
        let log_ok = last_log_index >= ps.log.len() as u64;

        if can_vote && log_ok {
            ps.voted_for = Some(candidate_id.to_string());
            true
        } else {
            false
        }
    }

    // Mirrors raft.rs handle_request_vote_response (line 1898):
    // leadership.read (guard) -> persistent.write -> votes_received.write
    // -> drop both -> become_leader (leadership.write)
    fn handle_vote_response(&self, node_id: &str, voter: &str, granted: bool, quorum: usize) {
        // Guard check: must be candidate
        {
            let ls = self.leadership.read().unwrap();
            if ls.role != Role::Candidate {
                return;
            }
        } // leadership.read released

        // persistent.write for term check
        let ps = self.persistent.write().unwrap();

        // votes_received.write to record vote
        let won = if granted {
            let mut votes = self.votes_received.write().unwrap();
            votes.push(voter.to_string());
            votes.len() >= quorum
        } else {
            false
        };

        // CRITICAL: drop persistent before become_leader (mirrors raft.rs:1942)
        let term = ps.current_term;
        drop(ps);

        if won {
            self.become_leader(node_id, term);
        }
    }

    // Mirrors raft.rs become_leader (line 2518):
    // persistent.read (gather data) -> drop -> leadership.write
    fn become_leader(&self, node_id: &str, _term: u64) {
        let log_len = {
            let ps = self.persistent.read().unwrap();
            ps.log.len() as u64
        }; // persistent.read released

        let mut ls = self.leadership.write().unwrap();
        ls.role = Role::Leader;
        ls.current_leader = Some(node_id.to_string());
        for ni in &mut ls.next_index {
            *ni = log_len + 1;
        }
        for mi in &mut ls.match_index {
            *mi = 0;
        }
    }

    // Mirrors raft.rs propose (line 2600):
    // leadership.read (guard) -> persistent.write -> drop -> WAL
    fn propose(&self, data: Vec<u8>) -> bool {
        // Guard: must be leader
        {
            let ls = self.leadership.read().unwrap();
            if ls.role != Role::Leader {
                return false;
            }
        }

        let term = {
            let mut ps = self.persistent.write().unwrap();
            let term = ps.current_term;
            ps.log.push((term, data));
            term
        }; // persistent released before WAL

        // WAL write
        {
            let mut wal = self.wal.lock().unwrap();
            wal.push(term as u8);
        }

        true
    }

    // Mirrors raft.rs try_advance_commit_index (line 2564):
    // leadership.read -> persistent.read -> volatile.write
    fn try_advance_commit_index(&self, new_commit: u64) {
        let is_leader = {
            let ls = self.leadership.read().unwrap();
            ls.role == Role::Leader
        };

        if !is_leader {
            return;
        }

        let log_len = {
            let ps = self.persistent.read().unwrap();
            ps.log.len() as u64
        };

        let effective = std::cmp::min(new_commit, log_len);
        let mut vs = self.volatile.write().unwrap();
        if effective > vs.commit_index {
            vs.commit_index = effective;
        }
    }

    // Mirrors raft.rs handle_append_entries_response (line 2291):
    // leadership.read (guard) -> persistent.write -> { leadership.write (scoped) }
    // -> drop persistent -> try_advance_commit_index
    fn handle_append_entries_response(&self, peer_idx: usize, success: bool, match_index: u64) {
        // Guard
        {
            let ls = self.leadership.read().unwrap();
            if ls.role != Role::Leader {
                return;
            }
        }

        let ps = self.persistent.write().unwrap();

        // Scoped leadership.write to update match/next index
        {
            let mut ls = self.leadership.write().unwrap();
            if success {
                if peer_idx < ls.match_index.len() {
                    ls.match_index[peer_idx] = match_index;
                    ls.next_index[peer_idx] = match_index + 1;
                }
            } else if peer_idx < ls.next_index.len() && ls.next_index[peer_idx] > 1 {
                ls.next_index[peer_idx] -= 1;
            }
        } // leadership.write released

        // CRITICAL: drop persistent before try_advance_commit_index
        // (mirrors the explicit drop at raft.rs:2368)
        drop(ps);

        if success {
            self.try_advance_commit_index(match_index);
        }
    }
}

#[test]
fn loom_raft_concurrent_append_entries() {
    // Two threads both process AppendEntries, contending on
    // persistent.write -> leadership.write -> volatile.write
    loom::model(|| {
        let node = Arc::new(MockRaftNode::new("node0", 2));

        let n1 = Arc::clone(&node);
        let n2 = Arc::clone(&node);

        let t1 = thread::spawn(move || {
            n1.handle_append_entries(1, "leader1", &[(1, vec![1])], 1);
        });

        let t2 = thread::spawn(move || {
            n2.handle_append_entries(1, "leader1", &[(1, vec![2])], 1);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Both entries should be appended (exact count depends on ordering)
        let ps = node.persistent.read().unwrap();
        assert!(!ps.log.is_empty(), "At least one entry must be appended");

        let vs = node.volatile.read().unwrap();
        assert!(vs.commit_index >= 1, "Commit index must advance");
    });
}

#[test]
fn loom_raft_election_during_heartbeat() {
    // start_election (persistent.write -> leadership.write) vs
    // handle_append_entries (persistent.write -> {leadership.write} -> volatile.write)
    // contend on persistent
    loom::model(|| {
        let node = Arc::new(MockRaftNode::new("node0", 2));

        let n1 = Arc::clone(&node);
        let n2 = Arc::clone(&node);

        let t1 = thread::spawn(move || {
            n1.start_election("node0");
        });

        let t2 = thread::spawn(move || {
            n2.handle_append_entries(1, "leader1", &[(1, vec![1])], 1);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Node must be in a valid state regardless of interleaving
        let ls = node.leadership.read().unwrap();
        assert!(
            ls.role == Role::Candidate || ls.role == Role::Follower,
            "Must be Candidate or Follower after election + AE"
        );
    });
}

#[test]
fn loom_raft_concurrent_vote_and_response() {
    // handle_request_vote (persistent.write -> leadership.write nested) vs
    // handle_vote_response (leadership.read -> persistent.write -> votes.write -> leadership.write)
    // contend on persistent and leadership
    loom::model(|| {
        let node = Arc::new(MockRaftNode::new("node0", 2));

        // First become candidate
        node.start_election("node0");

        let n1 = Arc::clone(&node);
        let n2 = Arc::clone(&node);

        let t1 = thread::spawn(move || {
            // Incoming vote request from higher-term candidate
            n1.handle_request_vote(5, "node2", 10);
        });

        let t2 = thread::spawn(move || {
            // Vote response granting our election
            n2.handle_vote_response("node0", "node1", true, 2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // State must be consistent: either we stepped down (higher term) or became leader
        let ls = node.leadership.read().unwrap();
        let ps = node.persistent.read().unwrap();
        match ls.role {
            Role::Leader => {
                // Won election before seeing higher term
                assert!(ps.current_term >= 1);
            },
            Role::Follower => {
                // Stepped down due to higher term
                assert!(ps.current_term >= 5);
            },
            Role::Candidate => {
                // Vote response processed but didn't reach quorum,
                // and vote request didn't have higher term
                assert!(ps.current_term >= 1);
            },
        }
    });
}

#[test]
fn loom_raft_propose_during_commit_advance() {
    // propose (leadership.read -> persistent.write) vs
    // handle_append_entries_response (leadership.read -> persistent.write ->
    //   {leadership.write} -> drop persistent -> try_advance_commit_index)
    loom::model(|| {
        let node = Arc::new(MockRaftNode::new("node0", 2));

        // Become leader first
        {
            let mut ps = node.persistent.write().unwrap();
            ps.current_term = 1;
            ps.log.push((1, vec![0]));
        }
        {
            let mut ls = node.leadership.write().unwrap();
            ls.role = Role::Leader;
            ls.current_leader = Some("node0".to_string());
            ls.next_index = vec![2, 2];
            ls.match_index = vec![0, 0];
        }

        let n1 = Arc::clone(&node);
        let n2 = Arc::clone(&node);

        let t1 = thread::spawn(move || {
            n1.propose(vec![42]);
        });

        let t2 = thread::spawn(move || {
            // Peer 0 acknowledges the first entry
            n2.handle_append_entries_response(0, true, 1);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Log must have at least the initial entry
        let ps = node.persistent.read().unwrap();
        assert!(!ps.log.is_empty(), "Log must not be empty");

        // Commit index may or may not have advanced depending on ordering
        let vs = node.volatile.read().unwrap();
        assert!(
            vs.commit_index <= ps.log.len() as u64,
            "Commit index must not exceed log length"
        );
    });
}
