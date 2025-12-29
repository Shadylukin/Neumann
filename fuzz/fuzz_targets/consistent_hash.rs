#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{ConsistentHashConfig, ConsistentHashPartitioner, Partitioner};

#[derive(Arbitrary, Debug)]
struct ConsistentHashInput {
    // Limit virtual nodes to avoid OOM
    virtual_nodes: u8,
    // Node names (limit to 10 nodes)
    nodes: Vec<String>,
    // Keys to lookup
    keys: Vec<String>,
}

fuzz_target!(|input: ConsistentHashInput| {
    // Need at least 1 virtual node and avoid excessive counts
    let virtual_nodes = (input.virtual_nodes as usize).clamp(1, 256);

    // Need at least 1 node
    if input.nodes.is_empty() {
        return;
    }

    // Limit node count, filter empty, and deduplicate
    let mut seen = std::collections::HashSet::new();
    let nodes: Vec<String> = input
        .nodes
        .into_iter()
        .filter(|s| !s.is_empty() && seen.insert(s.clone()))
        .take(10)
        .collect();
    if nodes.is_empty() {
        return;
    }

    // Create partitioner with first node as local
    let local_node = nodes[0].clone();
    let config = ConsistentHashConfig::new(local_node.clone()).with_virtual_nodes(virtual_nodes);
    let mut partitioner = ConsistentHashPartitioner::new(config);

    // Add all nodes
    for node in &nodes {
        partitioner.add_node(node.clone());
    }

    // Verify all nodes were added
    assert_eq!(partitioner.nodes().len(), nodes.len());

    // Verify total partitions
    assert_eq!(partitioner.total_partitions(), nodes.len() * virtual_nodes);

    // Lookup each key and verify result is valid
    for key in &input.keys {
        let result = partitioner.partition(key);

        // Result should point to a known node
        assert!(
            nodes.contains(&result.primary),
            "Partition result should be a known node"
        );

        // is_local should be consistent with local_node
        if result.primary == local_node {
            assert!(result.is_local, "Should be local for local node");
        } else {
            assert!(!result.is_local, "Should not be local for remote node");
        }
    }

    // Same key should always return same result (deterministic)
    for key in input.keys.iter().take(10) {
        let result1 = partitioner.partition(key);
        let result2 = partitioner.partition(key);
        assert_eq!(result1.primary, result2.primary, "Partitioning should be deterministic");
        assert_eq!(result1.partition, result2.partition);
    }

    // Remove a node and verify
    if nodes.len() > 1 {
        let to_remove = nodes[1].clone();
        let removed = partitioner.remove_node(&to_remove);
        assert_eq!(removed.len(), virtual_nodes);
        assert_eq!(partitioner.nodes().len(), nodes.len() - 1);
        assert_eq!(partitioner.total_partitions(), (nodes.len() - 1) * virtual_nodes);
    }
});
