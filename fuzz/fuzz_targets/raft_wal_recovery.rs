//! Fuzz test for Raft WAL recovery.
//!
//! Tests that:
//! - Recovery handles arbitrary sequences of WAL entries
//! - No duplicate votes are accepted for the same term
//! - Snapshot term updates are properly reflected
//! - Recovery state is consistent and valid

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{RaftRecoveryState, RaftWalEntry};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Sequence of entries to replay
    entries: Vec<WalEntryInput>,
}

#[derive(Debug, Arbitrary)]
struct WalEntryInput {
    /// Entry type selector
    entry_type: u8,
    /// Term value
    term: u64,
    /// Index value (for snapshot)
    index: u64,
    /// Candidate ID bytes
    candidate_id: Vec<u8>,
    /// Whether voted_for is Some
    has_vote: bool,
}

fn make_candidate_id(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "node1".to_string()
    } else {
        let s: String = bytes
            .iter()
            .take(16)
            .map(|&b| {
                let c = (b % 26) + b'a';
                c as char
            })
            .collect();
        if s.is_empty() {
            "node1".to_string()
        } else {
            s
        }
    }
}

fn make_entry(input: &WalEntryInput) -> RaftWalEntry {
    match input.entry_type % 6 {
        0 => RaftWalEntry::TermChange {
            new_term: input.term,
        },
        1 => RaftWalEntry::VoteCast {
            term: input.term,
            candidate_id: make_candidate_id(&input.candidate_id),
        },
        2 => RaftWalEntry::TermAndVote {
            term: input.term,
            voted_for: if input.has_vote {
                Some(make_candidate_id(&input.candidate_id))
            } else {
                None
            },
        },
        3 => RaftWalEntry::SnapshotTaken {
            last_included_index: input.index,
            last_included_term: input.term,
        },
        4 => RaftWalEntry::LogAppend {
            index: input.index,
            term: input.term,
            command_hash: [0u8; 32],
        },
        _ => RaftWalEntry::LogTruncate {
            from_index: input.index,
        },
    }
}

fuzz_target!(|input: FuzzInput| {
    // Limit entries to prevent OOM
    if input.entries.len() > 1000 {
        return;
    }

    let entries: Vec<RaftWalEntry> = input.entries.iter().map(make_entry).collect();

    // Recover state from entries
    let state = RaftRecoveryState::from_entries(&entries);

    // Invariant 1: current_term must be non-negative (always true for u64)
    // Invariant 2: If we have a snapshot, term should be at least snapshot term
    if let (Some(snapshot_term), true) = (state.last_snapshot_term, state.current_term < 1) {
        // This should never happen - snapshot term should update current_term
        assert!(
            state.current_term >= snapshot_term,
            "current_term should be >= snapshot term"
        );
    }

    // Invariant 3: Only one vote per term
    // To verify this, we check that recovery produces a valid state
    // by simulating the same recovery and tracking votes per term
    let mut tracked_term: u64 = 0;
    let mut tracked_vote: Option<String> = None;

    for entry in &entries {
        match entry {
            RaftWalEntry::TermChange { new_term } => {
                if *new_term > tracked_term {
                    tracked_term = *new_term;
                    tracked_vote = None;
                }
            }
            RaftWalEntry::VoteCast { term, candidate_id } => {
                if *term > tracked_term {
                    tracked_term = *term;
                    tracked_vote = Some(candidate_id.clone());
                } else if *term == tracked_term && tracked_vote.is_none() {
                    tracked_vote = Some(candidate_id.clone());
                }
                // Else: duplicate vote ignored
            }
            RaftWalEntry::TermAndVote { term, voted_for } => {
                if *term > tracked_term {
                    tracked_term = *term;
                    tracked_vote = voted_for.clone();
                } else if *term == tracked_term && tracked_vote.is_none() {
                    tracked_vote = voted_for.clone();
                }
                // Else: duplicate vote ignored
            }
            RaftWalEntry::SnapshotTaken {
                last_included_term,
                ..
            } => {
                if *last_included_term > tracked_term {
                    tracked_term = *last_included_term;
                    tracked_vote = None;
                }
            }
            _ => {}
        }
    }

    // Verify our tracking matches recovery state
    assert_eq!(
        state.current_term, tracked_term,
        "Recovery term mismatch: state={}, tracked={}",
        state.current_term, tracked_term
    );
    assert_eq!(
        state.voted_for, tracked_vote,
        "Recovery vote mismatch: state={:?}, tracked={:?}",
        state.voted_for, tracked_vote
    );

    // Invariant 4: Serialization roundtrip for entries should work
    for entry in &entries {
        let serialized = bincode::serialize(entry).expect("Entry should serialize");
        let deserialized: RaftWalEntry =
            bincode::deserialize(&serialized).expect("Entry should deserialize");
        assert_eq!(entry, &deserialized, "Entry roundtrip mismatch");
    }
});
