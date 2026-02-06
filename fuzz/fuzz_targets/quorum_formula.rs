// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|cluster_size: u16| {
    // Use u16 to get reasonable cluster sizes (0-65535)
    let n = cluster_size as usize;
    if n == 0 {
        return; // Skip zero-node clusters
    }

    let quorum = tensor_chain::quorum_size(n);

    // Property 1: Quorum is always at least 1
    assert!(quorum >= 1, "quorum must be >= 1");

    // Property 2: Quorum never exceeds cluster size
    assert!(quorum <= n, "quorum {} exceeds cluster size {}", quorum, n);

    // Property 3: Quorum is a strict majority (> N/2)
    assert!(quorum > n / 2, "quorum {} not majority of {}", quorum, n);

    // Property 4: No split brain - two disjoint quorums impossible
    assert!(2 * quorum > n, "split brain possible: n={}, q={}", n, quorum);

    // Property 5: Formula consistency
    let expected = (n / 2) + 1;
    assert_eq!(quorum, expected, "formula mismatch for n={}", n);

    // Property 6: Equivalence with old div_ceil formula (for peer_count = n-1)
    if n >= 1 {
        let peer_count = n - 1;
        let old_formula = (peer_count + 2).div_ceil(2);
        assert_eq!(
            quorum, old_formula,
            "old formula mismatch: n={}, peer_count={}",
            n, peer_count
        );
    }
});
