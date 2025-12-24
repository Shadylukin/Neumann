use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graph_engine::{Direction, GraphEngine, PropertyValue};
use std::collections::HashMap;

fn create_props(id: i64) -> HashMap<String, PropertyValue> {
    let mut props = HashMap::new();
    props.insert("id".to_string(), PropertyValue::Int(id));
    props.insert(
        "name".to_string(),
        PropertyValue::String(format!("node_{}", id)),
    );
    props
}

fn bench_create_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_nodes");

    for size in [100, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let engine = GraphEngine::new();
                for i in 0..size {
                    engine
                        .create_node("Person", create_props(i as i64))
                        .unwrap();
                }
                black_box(&engine);
            });
        });
    }

    group.finish();
}

fn bench_create_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_edges");

    group.bench_function("1000_edges_directed", |b| {
        b.iter(|| {
            let engine = GraphEngine::new();
            let mut ids = vec![];
            for _ in 0..1000 {
                ids.push(engine.create_node("Person", HashMap::new()).unwrap());
            }
            // Create chain: 0->1->2->...->999
            for i in 0..999 {
                engine
                    .create_edge(ids[i], ids[i + 1], "KNOWS", HashMap::new(), true)
                    .unwrap();
            }
            black_box(&engine);
        });
    });

    group.bench_function("1000_edges_undirected", |b| {
        b.iter(|| {
            let engine = GraphEngine::new();
            let mut ids = vec![];
            for _ in 0..1000 {
                ids.push(engine.create_node("Person", HashMap::new()).unwrap());
            }
            for i in 0..999 {
                engine
                    .create_edge(ids[i], ids[i + 1], "FRIENDS", HashMap::new(), false)
                    .unwrap();
            }
            black_box(&engine);
        });
    });

    group.finish();
}

fn bench_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors");

    // Create a star graph: center node connected to N nodes
    for fan_out in [10, 50, 100].iter() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();

        for i in 0..*fan_out {
            let leaf = engine.create_node("Leaf", create_props(i as i64)).unwrap();
            engine
                .create_edge(center, leaf, "CONNECTED", HashMap::new(), true)
                .unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("fan_out", fan_out),
            fan_out,
            |b, _fan_out| {
                b.iter(|| {
                    black_box(engine.neighbors(center, None, Direction::Outgoing).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_traverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("traverse");

    // Create a binary tree of depth D
    fn create_binary_tree(engine: &GraphEngine, depth: usize) -> u64 {
        fn create_subtree(engine: &GraphEngine, current_depth: usize, max_depth: usize) -> u64 {
            let node = engine.create_node("Node", HashMap::new()).unwrap();
            if current_depth < max_depth {
                let left = create_subtree(engine, current_depth + 1, max_depth);
                let right = create_subtree(engine, current_depth + 1, max_depth);
                engine
                    .create_edge(node, left, "CHILD", HashMap::new(), true)
                    .unwrap();
                engine
                    .create_edge(node, right, "CHILD", HashMap::new(), true)
                    .unwrap();
            }
            node
        }
        create_subtree(engine, 0, depth)
    }

    for depth in [5, 7, 9].iter() {
        // depth 5 = 31 nodes, depth 7 = 127 nodes, depth 9 = 511 nodes
        let engine = GraphEngine::new();
        let root = create_binary_tree(&engine, *depth);
        let expected_nodes = (1 << (depth + 1)) - 1; // 2^(d+1) - 1

        group.bench_with_input(
            BenchmarkId::new("binary_tree_depth", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    let result = engine
                        .traverse(root, Direction::Outgoing, depth + 1, None)
                        .unwrap();
                    assert_eq!(result.len(), expected_nodes);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_find_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_path");

    // Create a chain: 0 -> 1 -> 2 -> ... -> N
    for chain_length in [10, 50, 100].iter() {
        let engine = GraphEngine::new();
        let mut node_ids = vec![];

        for i in 0..*chain_length {
            node_ids.push(engine.create_node("Node", create_props(i as i64)).unwrap());
        }

        for i in 0..(chain_length - 1) {
            engine
                .create_edge(
                    node_ids[i as usize],
                    node_ids[(i + 1) as usize],
                    "NEXT",
                    HashMap::new(),
                    true,
                )
                .unwrap();
        }

        let first = node_ids[0];
        let last = *node_ids.last().unwrap();

        group.bench_with_input(
            BenchmarkId::new("chain_length", chain_length),
            chain_length,
            |b, _| {
                b.iter(|| {
                    black_box(engine.find_path(first, last).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_find_path_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_path_branching");

    // Create a graph with multiple paths to test BFS efficiency
    // Grid-like structure: NxN nodes
    for grid_size in [5, 10].iter() {
        let engine = GraphEngine::new();
        let mut nodes = vec![];

        // Create NxN grid of nodes
        for i in 0..(grid_size * grid_size) {
            nodes.push(engine.create_node("Node", create_props(i as i64)).unwrap());
        }

        // Connect horizontally and vertically
        for row in 0..*grid_size {
            for col in 0..*grid_size {
                let idx = (row * grid_size + col) as usize;
                // Connect to right neighbor
                if col < grid_size - 1 {
                    let right_idx = idx + 1;
                    engine
                        .create_edge(
                            nodes[idx],
                            nodes[right_idx],
                            "ADJACENT",
                            HashMap::new(),
                            false,
                        )
                        .unwrap();
                }
                // Connect to bottom neighbor
                if row < grid_size - 1 {
                    let bottom_idx = idx + *grid_size as usize;
                    engine
                        .create_edge(
                            nodes[idx],
                            nodes[bottom_idx],
                            "ADJACENT",
                            HashMap::new(),
                            false,
                        )
                        .unwrap();
                }
            }
        }

        let top_left = nodes[0];
        let bottom_right = *nodes.last().unwrap();

        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", grid_size, grid_size)),
            grid_size,
            |b, _| {
                b.iter(|| {
                    black_box(engine.find_path(top_left, bottom_right).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_create_nodes,
    bench_create_edges,
    bench_neighbors,
    bench_traverse,
    bench_find_path,
    bench_find_path_branching,
);

criterion_main!(benches);
