//! Integration tests for distributed query execution.
//!
//! Tests query planning and result merging.

use std::sync::Arc;

use query_router::distributed::{
    AggregateFunction, MergeStrategy, QueryPlan, QueryPlanner, ResultMerger, ShardResult,
};
use query_router::{QueryResult, SimilarResult};
use relational_engine::{Row, Value};
use tensor_store::{ConsistentHashConfig, ConsistentHashPartitioner, Partitioner};

fn make_similar_result(key: &str, score: f32) -> SimilarResult {
    SimilarResult {
        key: key.to_string(),
        score,
    }
}

fn make_row(id: u64, values: Vec<(&str, Value)>) -> Row {
    Row {
        id,
        values: values
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    }
}

fn setup_partitioner(num_shards: usize) -> Arc<dyn Partitioner + Send + Sync> {
    let nodes: Vec<String> = (0..num_shards).map(|i| format!("shard{}", i)).collect();
    let config = ConsistentHashConfig::new("shard0").with_virtual_nodes(100);
    let mut partitioner = ConsistentHashPartitioner::new(config);
    for node in nodes {
        partitioner.add_node(node);
    }
    Arc::new(partitioner)
}

// ============================================================================
// Query Planning Tests
// ============================================================================

#[test]
fn test_query_planner_point_lookup() {
    let partitioner = setup_partitioner(3);
    let planner = QueryPlanner::new(partitioner.clone(), 0);

    let plan = planner.plan("GET user:123");

    match plan {
        QueryPlan::Local { .. } | QueryPlan::Remote { .. } => {
            // Point lookups should be single-shard
        },
        QueryPlan::ScatterGather { .. } => {
            panic!("Point lookup should not scatter-gather");
        },
    }
}

#[test]
fn test_query_planner_table_scan() {
    let partitioner = setup_partitioner(3);
    let planner = QueryPlanner::new(partitioner, 0);

    let plan = planner.plan("SELECT * FROM users");

    match plan {
        QueryPlan::ScatterGather { shards, merge, .. } => {
            assert_eq!(shards.len(), 3);
            assert!(matches!(merge, MergeStrategy::Union));
        },
        _ => panic!("Table scan should scatter-gather"),
    }
}

#[test]
fn test_query_planner_similarity_search() {
    let partitioner = setup_partitioner(3);
    let planner = QueryPlanner::new(partitioner, 0);

    let plan = planner.plan("SIMILAR TO [0.1, 0.2, 0.3] LIMIT 10");

    match plan {
        QueryPlan::ScatterGather { merge, .. } => {
            assert!(matches!(merge, MergeStrategy::TopK(10)));
        },
        _ => panic!("Similarity search should scatter-gather with TopK"),
    }
}

#[test]
fn test_query_planner_aggregate_count() {
    let partitioner = setup_partitioner(3);
    let planner = QueryPlanner::new(partitioner, 0);

    let plan = planner.plan("SELECT COUNT(*) FROM users");

    match plan {
        QueryPlan::ScatterGather { merge, .. } => {
            assert!(matches!(
                merge,
                MergeStrategy::Aggregate(AggregateFunction::Count)
            ));
        },
        _ => panic!("COUNT should scatter-gather with Aggregate"),
    }
}

// ============================================================================
// Result Merger Tests - Union
// ============================================================================

#[test]
fn test_merge_union_rows() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Rows(vec![
                make_row(1, vec![("name", Value::String("Alice".to_string()))]),
                make_row(2, vec![("name", Value::String("Bob".to_string()))]),
            ]),
            10_000,
        ),
        ShardResult::success(
            1,
            QueryResult::Rows(vec![make_row(
                3,
                vec![("name", Value::String("Charlie".to_string()))],
            )]),
            15_000,
        ),
    ];

    let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();

    match merged {
        QueryResult::Rows(rows) => {
            assert_eq!(rows.len(), 3);
        },
        _ => panic!("Expected Rows result"),
    }
}

#[test]
fn test_merge_union_similar_results() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Similar(vec![
                make_similar_result("key1", 0.9),
                make_similar_result("key2", 0.8),
            ]),
            10_000,
        ),
        ShardResult::success(
            1,
            QueryResult::Similar(vec![make_similar_result("key3", 0.85)]),
            15_000,
        ),
    ];

    let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();

    match merged {
        QueryResult::Similar(similar) => {
            assert_eq!(similar.len(), 3);
        },
        _ => panic!("Expected Similar result"),
    }
}

#[test]
fn test_merge_union_empty_shards() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Rows(vec![make_row(1, vec![("x", Value::Int(1))])]),
            10_000,
        ),
        ShardResult::success(1, QueryResult::Rows(vec![]), 5_000),
        ShardResult::success(
            2,
            QueryResult::Rows(vec![make_row(2, vec![("x", Value::Int(2))])]),
            8_000,
        ),
    ];

    let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();

    match merged {
        QueryResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
        },
        _ => panic!("Expected Rows result"),
    }
}

// ============================================================================
// Result Merger Tests - TopK
// ============================================================================

#[test]
fn test_merge_top_k() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Similar(vec![
                make_similar_result("k0_1", 0.95),
                make_similar_result("k0_2", 0.85),
                make_similar_result("k0_3", 0.75),
            ]),
            10_000,
        ),
        ShardResult::success(
            1,
            QueryResult::Similar(vec![
                make_similar_result("k1_1", 0.90),
                make_similar_result("k1_2", 0.80),
            ]),
            15_000,
        ),
    ];

    let merged = ResultMerger::merge(results, &MergeStrategy::TopK(3)).unwrap();

    match merged {
        QueryResult::Similar(similar) => {
            assert_eq!(similar.len(), 3);
            // Should be sorted by score descending
            assert_eq!(similar[0].key, "k0_1"); // 0.95
            assert_eq!(similar[1].key, "k1_1"); // 0.90
            assert_eq!(similar[2].key, "k0_2"); // 0.85
        },
        _ => panic!("Expected Similar result"),
    }
}

#[test]
fn test_merge_top_k_fewer_than_k() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Similar(vec![make_similar_result("key1", 0.9)]),
            10_000,
        ),
        ShardResult::success(
            1,
            QueryResult::Similar(vec![make_similar_result("key2", 0.8)]),
            15_000,
        ),
    ];

    // Request k=10, but only 2 results available
    let merged = ResultMerger::merge(results, &MergeStrategy::TopK(10)).unwrap();

    match merged {
        QueryResult::Similar(similar) => {
            assert_eq!(similar.len(), 2);
        },
        _ => panic!("Expected Similar result"),
    }
}

#[test]
fn test_merge_top_k_preserves_order() {
    let results = vec![
        ShardResult::success(
            0,
            QueryResult::Similar(vec![
                make_similar_result("a", 0.1),
                make_similar_result("b", 0.5),
            ]),
            10_000,
        ),
        ShardResult::success(
            1,
            QueryResult::Similar(vec![
                make_similar_result("c", 0.3),
                make_similar_result("d", 0.9),
            ]),
            15_000,
        ),
    ];

    let merged = ResultMerger::merge(results, &MergeStrategy::TopK(4)).unwrap();

    match merged {
        QueryResult::Similar(similar) => {
            let scores: Vec<f32> = similar.iter().map(|s| s.score).collect();
            for i in 1..scores.len() {
                assert!(
                    scores[i - 1] >= scores[i],
                    "Results should be sorted descending"
                );
            }
        },
        _ => panic!("Expected Similar result"),
    }
}

// ============================================================================
// Result Merger Tests - Aggregates
// ============================================================================

#[test]
fn test_merge_aggregate_count() {
    let results = vec![
        ShardResult::success(0, QueryResult::Count(100), 10_000),
        ShardResult::success(1, QueryResult::Count(150), 15_000),
        ShardResult::success(2, QueryResult::Count(50), 12_000),
    ];

    let merged =
        ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Count)).unwrap();

    match merged {
        QueryResult::Count(n) => {
            assert_eq!(n, 300);
        },
        _ => panic!("Expected Count result"),
    }
}

#[test]
fn test_merge_aggregate_sum() {
    let results = vec![
        ShardResult::success(0, QueryResult::Count(100), 10_000),
        ShardResult::success(1, QueryResult::Count(200), 15_000),
    ];

    let merged =
        ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum)).unwrap();

    match merged {
        QueryResult::Count(n) => {
            assert_eq!(n, 300);
        },
        _ => panic!("Expected Count result"),
    }
}

// ============================================================================
// Empty Results Tests
// ============================================================================

#[test]
fn test_merge_empty_results() {
    let results: Vec<ShardResult> = vec![];
    let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
    assert!(matches!(merged, QueryResult::Empty));
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_merge_performance_large_result_set() {
    let shard_count = 10;
    let rows_per_shard = 1000;

    let results: Vec<ShardResult> = (0..shard_count)
        .map(|shard| {
            let rows: Vec<Row> = (0..rows_per_shard)
                .map(|i| {
                    make_row(
                        (shard * rows_per_shard + i) as u64,
                        vec![("value", Value::Int((shard * rows_per_shard + i) as i64))],
                    )
                })
                .collect();

            ShardResult::success(shard, QueryResult::Rows(rows), 10_000)
        })
        .collect();

    let start = std::time::Instant::now();
    let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
    let elapsed = start.elapsed();

    match merged {
        QueryResult::Rows(rows) => {
            assert_eq!(rows.len(), shard_count * rows_per_shard);
        },
        _ => panic!("Expected Rows"),
    }

    // Should merge 10k rows in under 100ms
    assert!(
        elapsed.as_millis() < 100,
        "Merge took {}ms, expected <100ms",
        elapsed.as_millis()
    );
}

#[test]
fn test_top_k_merge_performance() {
    let shard_count = 10;
    let results_per_shard = 1000;
    let k = 100;

    let results: Vec<ShardResult> = (0..shard_count)
        .map(|shard| {
            let similar: Vec<SimilarResult> = (0..results_per_shard)
                .map(|i| SimilarResult {
                    key: format!("shard{}_{}", shard, i),
                    score: ((shard * 1000 + i) as f32 / 10000.0),
                })
                .collect();

            ShardResult::success(shard, QueryResult::Similar(similar), 10_000)
        })
        .collect();

    let start = std::time::Instant::now();
    let merged = ResultMerger::merge(results, &MergeStrategy::TopK(k)).unwrap();
    let elapsed = start.elapsed();

    match merged {
        QueryResult::Similar(similar) => {
            assert_eq!(similar.len(), k);
            // Verify sorted
            for i in 1..similar.len() {
                assert!(similar[i - 1].score >= similar[i].score);
            }
        },
        _ => panic!("Expected Similar"),
    }

    // Should merge and select top-k from 10k results in under 50ms
    assert!(
        elapsed.as_millis() < 50,
        "TopK merge took {}ms, expected <50ms",
        elapsed.as_millis()
    );
}
