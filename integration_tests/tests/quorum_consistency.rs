//! Integration tests for quorum calculation consistency.

use tensor_chain::quorum_size;

#[test]
fn test_quorum_size_matches_raft_node() {
    // Verify the centralized function matches RaftNode behavior
    for peer_count in 0..=10 {
        let total_nodes = peer_count + 1;
        let expected = (total_nodes / 2) + 1;
        assert_eq!(
            quorum_size(total_nodes),
            expected,
            "Mismatch for {} total nodes",
            total_nodes
        );
    }
}

#[test]
fn test_quorum_equivalence_with_old_formula() {
    // Verify new formula matches the div_ceil formula that was used
    for peer_count in 0usize..=100 {
        let old_formula = (peer_count + 2).div_ceil(2);
        let new_formula = quorum_size(peer_count + 1);
        assert_eq!(
            old_formula, new_formula,
            "Formula mismatch for peer_count={}",
            peer_count
        );
    }
}

#[test]
fn test_quorum_safety_property() {
    // Fundamental Raft safety: quorum overlap guarantees consistency
    for n in 1..=50 {
        let q = quorum_size(n);
        // Any two quorums must overlap by at least 1 node
        // This means 2*q > n, i.e., q > n/2
        assert!(
            2 * q > n,
            "Quorum overlap not guaranteed for n={}, q={}",
            n,
            q
        );
    }
}
