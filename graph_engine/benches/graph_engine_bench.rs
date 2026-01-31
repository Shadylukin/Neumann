// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graph_engine::{Direction, GraphEngine, PropertyValue};

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
                    black_box(
                        engine
                            .neighbors(center, None, Direction::Outgoing, None)
                            .unwrap(),
                    );
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
                        .traverse(root, Direction::Outgoing, depth + 1, None, None)
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
                    black_box(engine.find_path(first, last, None).unwrap());
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
                    black_box(engine.find_path(top_left, bottom_right, None).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ========== Concurrent Benchmarks ==========

fn bench_concurrent_node_creation(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_node_creation");

    for thread_count in [2, 4, 8, 16, 32] {
        let nodes_per_thread = 1000;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());
                        let barrier = Arc::new(Barrier::new(threads));

                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..nodes_per_thread {
                                        eng.create_node(
                                            "Bench",
                                            create_props((t * nodes_per_thread + i) as i64),
                                        )
                                        .unwrap();
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_update_node(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_update_node");

    for thread_count in [2, 4, 8, 16, 32] {
        let updates_per_thread = 500;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());

                        // Pre-create nodes
                        let mut node_ids = Vec::new();
                        for t in 0..threads {
                            node_ids.push(
                                engine
                                    .create_node("Target", create_props(t as i64))
                                    .unwrap(),
                            );
                        }
                        let node_ids = Arc::new(node_ids);

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                let ids = Arc::clone(&node_ids);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..updates_per_thread {
                                        let mut props = HashMap::new();
                                        props.insert(
                                            "counter".to_string(),
                                            PropertyValue::Int(i as i64),
                                        );
                                        eng.update_node(ids[t], None, props).unwrap();
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_batch_operations(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_batch_operations");

    for thread_count in [2, 4, 8, 16] {
        let batch_size = 100;
        let batches_per_thread = 10;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());

                        // Pre-create base nodes for edges
                        let mut base_ids = Vec::new();
                        for i in 0..100 {
                            base_ids
                                .push(engine.create_node("Base", create_props(i as i64)).unwrap());
                        }
                        let base_ids = Arc::new(base_ids);

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                let bases = Arc::clone(&base_ids);
                                thread::spawn(move || {
                                    bar.wait();
                                    for batch_idx in 0..batches_per_thread {
                                        if (t + batch_idx) % 2 == 0 {
                                            // Batch nodes
                                            let nodes: Vec<graph_engine::NodeInput> = (0
                                                ..batch_size)
                                                .map(|i| graph_engine::NodeInput {
                                                    labels: vec!["Batch".to_string()],
                                                    properties: {
                                                        let mut p = HashMap::new();
                                                        p.insert(
                                                            "idx".to_string(),
                                                            PropertyValue::Int(i as i64),
                                                        );
                                                        p
                                                    },
                                                })
                                                .collect();
                                            let _ = eng.batch_create_nodes(nodes);
                                        } else {
                                            // Batch edges
                                            let edges: Vec<graph_engine::EdgeInput> = (0
                                                ..batch_size.min(99))
                                                .map(|i| graph_engine::EdgeInput {
                                                    from: bases[i % 100],
                                                    to: bases[(i + 1) % 100],
                                                    edge_type: "LINK".to_string(),
                                                    properties: HashMap::new(),
                                                    directed: true,
                                                })
                                                .collect();
                                            let _ = eng.batch_create_edges(edges);
                                        }
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_striped_lock_contention(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("striped_lock_contention");

    for thread_count in [2, 4, 8, 16, 32, 64] {
        let ops_per_thread = 500;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());
                        engine.create_node_property_index("key").unwrap();

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..ops_per_thread {
                                        let mut props = HashMap::new();
                                        // Distribute across stripes
                                        let shard = (t * ops_per_thread + i) % 64;
                                        let key = format!("{shard:02x}_t{t}_i{i}");
                                        props.insert("key".to_string(), PropertyValue::String(key));
                                        eng.create_node("Contended", props).unwrap();
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_neighbor_lookup(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_neighbor_lookup");

    for thread_count in [2, 4, 8, 16, 32] {
        let lookups_per_thread = 500;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        // Create star graph: 1 hub + 1000 spokes
                        let engine = Arc::new(GraphEngine::new());
                        let hub = engine.create_node("Hub", HashMap::new()).unwrap();

                        for i in 0..1000 {
                            let spoke =
                                engine.create_node("Spoke", create_props(i as i64)).unwrap();
                            engine
                                .create_edge(hub, spoke, "CONNECTED", HashMap::new(), true)
                                .unwrap();
                        }

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|_| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    bar.wait();
                                    for _ in 0..lookups_per_thread {
                                        black_box(
                                            eng.neighbors(hub, None, Direction::Outgoing, None)
                                                .unwrap(),
                                        );
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_traversal(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_traversal");

    // Create binary tree helper
    fn create_binary_tree(engine: &GraphEngine, depth: usize) -> u64 {
        fn create_subtree(engine: &GraphEngine, current_depth: usize, max_depth: usize) -> u64 {
            let node = engine.create_node("TreeNode", HashMap::new()).unwrap();
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

    for thread_count in [2, 4, 8, 16] {
        let traversals_per_thread = 50;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        // Create binary tree depth 10 (1023 nodes)
                        let engine = Arc::new(GraphEngine::new());
                        let root = create_binary_tree(&engine, 10);

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|_| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    bar.wait();
                                    for _ in 0..traversals_per_thread {
                                        black_box(
                                            eng.traverse(root, Direction::Outgoing, 11, None, None)
                                                .unwrap(),
                                        );
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_mixed_read_write(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_mixed_read_write");

    for thread_count in [4, 8, 16, 32] {
        let ops_per_thread = 200;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());

                        // Pre-create 1000 nodes
                        let mut node_ids = Vec::new();
                        for i in 0..1000 {
                            node_ids
                                .push(engine.create_node("Mixed", create_props(i as i64)).unwrap());
                        }
                        let node_ids = Arc::new(node_ids);

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                let ids = Arc::clone(&node_ids);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..ops_per_thread {
                                        // 50% readers, 50% writers
                                        if t % 2 == 0 {
                                            // Reader: get node
                                            let idx = (t * ops_per_thread + i) % 1000;
                                            black_box(eng.get_node(ids[idx]).unwrap());
                                        } else {
                                            // Writer: update node
                                            let idx = (t * ops_per_thread + i) % 1000;
                                            let mut props = HashMap::new();
                                            props.insert(
                                                "updated".to_string(),
                                                PropertyValue::Int(i as i64),
                                            );
                                            eng.update_node(ids[idx], None, props).unwrap();
                                        }
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_property_index_query(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_property_index_query");

    for thread_count in [2, 4, 8, 16, 32] {
        let queries_per_thread = 200;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());
                        engine.create_node_property_index("category").unwrap();

                        // Create 10000 nodes with 10 category values
                        for i in 0..10000 {
                            let mut props = HashMap::new();
                            let category = format!("cat_{}", i % 10);
                            props.insert("category".to_string(), PropertyValue::String(category));
                            props.insert("idx".to_string(), PropertyValue::Int(i as i64));
                            engine.create_node("Categorized", props).unwrap();
                        }

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..queries_per_thread {
                                        let category = format!("cat_{}", (t + i) % 10);
                                        black_box(
                                            eng.find_nodes_by_property(
                                                "category",
                                                &PropertyValue::String(category),
                                            )
                                            .unwrap(),
                                        );
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_large_graph(c: &mut Criterion) {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_large_graph");
    group.sample_size(10); // Fewer samples due to setup cost

    for thread_count in [4, 8, 16] {
        let ops_per_thread = 100;

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let engine = Arc::new(GraphEngine::new());

                        // Create 10000 nodes
                        let mut node_ids = Vec::new();
                        for i in 0..10000 {
                            node_ids
                                .push(engine.create_node("Large", create_props(i as i64)).unwrap());
                        }

                        // Create 50000 edges (5 per node on average)
                        for i in 0..50000 {
                            let from = node_ids[i % 10000];
                            let to = node_ids[(i * 7 + 13) % 10000];
                            if from != to {
                                let _ = engine.create_edge(from, to, "LINK", HashMap::new(), true);
                            }
                        }

                        let node_ids = Arc::new(node_ids);
                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let eng = Arc::clone(&engine);
                                let bar = Arc::clone(&barrier);
                                let ids = Arc::clone(&node_ids);
                                thread::spawn(move || {
                                    bar.wait();
                                    for i in 0..ops_per_thread {
                                        // Mix of operations
                                        match i % 4 {
                                            0 => {
                                                // Get node
                                                let idx = (t * 1000 + i * 10) % 10000;
                                                black_box(eng.get_node(ids[idx]).unwrap());
                                            },
                                            1 => {
                                                // Get neighbors
                                                let idx = (t * 1000 + i * 10) % 10000;
                                                black_box(
                                                    eng.neighbors(
                                                        ids[idx],
                                                        None,
                                                        Direction::Outgoing,
                                                        None,
                                                    )
                                                    .unwrap(),
                                                );
                                            },
                                            2 => {
                                                // Update node
                                                let idx = (t * 1000 + i * 10) % 10000;
                                                let mut props = HashMap::new();
                                                props.insert(
                                                    "modified".to_string(),
                                                    PropertyValue::Int(i as i64),
                                                );
                                                eng.update_node(ids[idx], None, props).unwrap();
                                            },
                                            _ => {
                                                // Create edge
                                                let from = ids[(t * 1000 + i * 10) % 10000];
                                                let to = ids[(t * 1000 + i * 10 + 100) % 10000];
                                                let _ = eng.create_edge(
                                                    from,
                                                    to,
                                                    "NEW",
                                                    HashMap::new(),
                                                    true,
                                                );
                                            },
                                        }
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
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

criterion_group!(
    concurrent_benches,
    bench_concurrent_node_creation,
    bench_concurrent_update_node,
    bench_concurrent_batch_operations,
    bench_striped_lock_contention,
    bench_concurrent_neighbor_lookup,
    bench_concurrent_traversal,
    bench_concurrent_mixed_read_write,
    bench_concurrent_property_index_query,
    bench_concurrent_large_graph,
);

criterion_main!(benches, concurrent_benches);
