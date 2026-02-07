// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Distributed query execution with semantic routing and scatter-gather.
//!
//! Routes queries to appropriate shards based on key or embedding similarity.
//! Supports multi-shard scatter-gather for table scans and similarity searches.

use std::sync::Arc;

use relational_engine::{Row, Value};
use serde::{Deserialize, Serialize};
use tensor_store::{PartitionResult, Partitioner, SemanticPartitioner};

use crate::{QueryResult, Result, SimilarResult};

/// Shard identifier.
pub type ShardId = usize;

/// Query execution plan.
#[derive(Debug, Clone)]
pub enum QueryPlan {
    /// Execute locally on this node.
    Local { query: String },
    /// Forward to a remote shard.
    Remote { shard: ShardId, query: String },
    /// Scatter to multiple shards and gather results.
    ScatterGather {
        shards: Vec<ShardId>,
        query: String,
        merge: MergeStrategy,
    },
}

/// Strategy for merging results from multiple shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Union all results (for SELECT, NODE queries).
    Union,
    /// Keep top K by similarity score (for SIMILAR queries).
    TopK(usize),
    /// Aggregate results (SUM, COUNT, AVG).
    Aggregate(AggregateFunction),
    /// First non-empty result (for point lookups).
    FirstNonEmpty,
    /// Concatenate all results in order.
    Concat,
}

/// Aggregate function for distributed aggregation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Sum all values.
    Sum,
    /// Count all results.
    Count,
    /// Average (sum/count).
    Avg,
    /// Maximum value.
    Max,
    /// Minimum value.
    Min,
}

/// Result from a single shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardResult {
    /// Shard that produced this result.
    pub shard: ShardId,
    /// Query result from the shard.
    pub result: QueryResult,
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Whether this shard had any errors.
    pub error: Option<String>,
}

impl ShardResult {
    /// Create a successful shard result.
    #[must_use]
    pub const fn success(shard: ShardId, result: QueryResult, execution_time_us: u64) -> Self {
        Self {
            shard,
            result,
            execution_time_us,
            error: None,
        }
    }

    /// Create an error shard result.
    #[must_use]
    pub const fn error(shard: ShardId, error: String) -> Self {
        Self {
            shard,
            result: QueryResult::Empty,
            execution_time_us: 0,
            error: Some(error),
        }
    }
}

/// Configuration for distributed query execution.
#[derive(Debug, Clone)]
pub struct DistributedQueryConfig {
    /// Maximum concurrent shard queries.
    pub max_concurrent: usize,
    /// Query timeout per shard in milliseconds.
    pub shard_timeout_ms: u64,
    /// Retry count for failed shards.
    pub retry_count: usize,
    /// Whether to fail fast on first shard error.
    pub fail_fast: bool,
}

impl Default for DistributedQueryConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            shard_timeout_ms: 5000,
            retry_count: 2,
            fail_fast: false,
        }
    }
}

/// Query planner for distributed execution.
#[derive(Debug)]
pub struct QueryPlanner {
    /// Partitioner for routing decisions.
    partitioner: Arc<dyn Partitioner + Send + Sync>,
    /// Semantic partitioner for embedding-based shard routing.
    semantic_partitioner: Option<Arc<SemanticPartitioner>>,
    /// Local shard ID (reserved for future optimizations like local-first execution).
    #[allow(dead_code)]
    local_shard: ShardId,
}

impl QueryPlanner {
    /// Create a new query planner.
    pub fn new(partitioner: Arc<dyn Partitioner + Send + Sync>, local_shard: ShardId) -> Self {
        Self {
            partitioner,
            semantic_partitioner: None,
            local_shard,
        }
    }

    /// Set the semantic partitioner for embedding-based routing.
    #[must_use]
    pub fn with_semantic_partitioner(mut self, partitioner: Arc<SemanticPartitioner>) -> Self {
        self.semantic_partitioner = Some(partitioner);
        self
    }

    /// Plan query execution.
    #[must_use]
    pub fn plan(&self, query: &str) -> QueryPlan {
        // Parse query to determine routing
        let query_type = Self::classify_query(query);

        match query_type {
            QueryType::PointLookup { key } => {
                let result = self.partitioner.partition(&key);
                if result.is_local {
                    QueryPlan::Local {
                        query: query.to_string(),
                    }
                } else {
                    QueryPlan::Remote {
                        shard: self.shard_from_result(&result),
                        query: query.to_string(),
                    }
                }
            },
            QueryType::SimilaritySearch { k } => {
                // Scatter to all shards, merge top K
                QueryPlan::ScatterGather {
                    shards: self.all_shards(),
                    query: query.to_string(),
                    merge: MergeStrategy::TopK(k),
                }
            },
            QueryType::TableScan => {
                // Scatter to all shards, union results
                QueryPlan::ScatterGather {
                    shards: self.all_shards(),
                    query: query.to_string(),
                    merge: MergeStrategy::Union,
                }
            },
            QueryType::Aggregate { func } => {
                // Scatter to all shards, aggregate
                QueryPlan::ScatterGather {
                    shards: self.all_shards(),
                    query: query.to_string(),
                    merge: MergeStrategy::Aggregate(func),
                }
            },
            QueryType::Unknown => {
                // Default to local execution
                QueryPlan::Local {
                    query: query.to_string(),
                }
            },
        }
    }

    /// Plan query with explicit embedding for semantic routing.
    #[must_use]
    pub fn plan_with_embedding(&self, query: &str, embedding: &[f32]) -> QueryPlan {
        // Get semantically relevant shards
        let relevant_shards = self.shards_for_embedding(embedding);

        if relevant_shards.is_empty() {
            // Fall back to all shards
            return self.plan(query);
        }

        let query_type = Self::classify_query(query);

        match query_type {
            QueryType::SimilaritySearch { k } => QueryPlan::ScatterGather {
                shards: relevant_shards,
                query: query.to_string(),
                merge: MergeStrategy::TopK(k),
            },
            _ => self.plan(query),
        }
    }

    /// Get all shards in the cluster.
    fn all_shards(&self) -> Vec<ShardId> {
        let nodes = self.partitioner.nodes();
        (0..nodes.len()).collect()
    }

    /// Convert partition result to shard ID.
    fn shard_from_result(&self, result: &PartitionResult) -> ShardId {
        let nodes = self.partitioner.nodes();
        nodes.iter().position(|n| *n == result.primary).unwrap_or(0)
    }

    /// Get shards relevant to an embedding using semantic routing.
    fn shards_for_embedding(&self, embedding: &[f32]) -> Vec<ShardId> {
        if let Some(sp) = &self.semantic_partitioner {
            let results = sp.shards_for_embedding(embedding);
            if !results.is_empty() {
                return results.into_iter().map(|(shard, _score)| shard).collect();
            }
        }
        // Fall back to all shards if no semantic partitioner or no results
        self.all_shards()
    }

    /// Classify query type for routing.
    fn classify_query(query: &str) -> QueryType {
        let query_upper = query.to_uppercase();
        let query_trimmed = query_upper.trim();

        // Point lookups
        if query_trimmed.starts_with("GET ")
            || query_trimmed.starts_with("NODE GET ")
            || query_trimmed.starts_with("ENTITY GET ")
        {
            // Extract key from query
            if let Some(key) = Self::extract_key(query) {
                return QueryType::PointLookup { key };
            }
        }

        // Similarity search
        if query_trimmed.starts_with("SIMILAR ") {
            let k = Self::extract_top_k(query).unwrap_or(10);
            return QueryType::SimilaritySearch { k };
        }

        // Table scans
        if query_trimmed.starts_with("SELECT ") || query_trimmed.starts_with("NODE LIST") {
            // Check for aggregates
            if query_trimmed.contains("COUNT(") {
                return QueryType::Aggregate {
                    func: AggregateFunction::Count,
                };
            }
            if query_trimmed.contains("SUM(") {
                return QueryType::Aggregate {
                    func: AggregateFunction::Sum,
                };
            }
            if query_trimmed.contains("AVG(") {
                return QueryType::Aggregate {
                    func: AggregateFunction::Avg,
                };
            }
            return QueryType::TableScan;
        }

        QueryType::Unknown
    }

    /// Extract key from a point lookup query.
    fn extract_key(query: &str) -> Option<String> {
        let parts: Vec<&str> = query.split_whitespace().collect();
        if parts.len() >= 2 {
            // Handle "GET key", "NODE GET key", etc.
            for (i, part) in parts.iter().enumerate() {
                if part.eq_ignore_ascii_case("GET") && i + 1 < parts.len() {
                    return Some(parts[i + 1].to_string());
                }
            }
        }
        None
    }

    /// Extract TOP K value from query.
    fn extract_top_k(query: &str) -> Option<usize> {
        let query_upper = query.to_uppercase();
        if let Some(pos) = query_upper.find("TOP ") {
            let rest = &query_upper[pos + 4..];
            let num_str: String = rest.chars().take_while(char::is_ascii_digit).collect();
            return num_str.parse().ok();
        }
        None
    }
}

/// Query type classification.
#[derive(Debug)]
enum QueryType {
    /// Single key lookup.
    PointLookup { key: String },
    /// Similarity search with top K.
    SimilaritySearch { k: usize },
    /// Full table/entity scan.
    TableScan,
    /// Aggregate query.
    Aggregate { func: AggregateFunction },
    /// Unknown query type.
    Unknown,
}

/// Merger for combining results from multiple shards.
#[derive(Debug)]
pub struct ResultMerger;

impl ResultMerger {
    /// Merge shard results using the specified strategy.
    ///
    /// # Errors
    ///
    /// This function currently never returns an error, but returns `Result` for
    /// forward compatibility with future merge strategies that may fail.
    pub fn merge(results: Vec<ShardResult>, strategy: &MergeStrategy) -> Result<QueryResult> {
        // Filter out errors if not fail-fast
        let successful: Vec<_> = results.into_iter().filter(|r| r.error.is_none()).collect();

        if successful.is_empty() {
            return Ok(QueryResult::Empty);
        }

        Ok(match strategy {
            MergeStrategy::Union => Self::merge_union(successful),
            MergeStrategy::TopK(k) => Self::merge_top_k(successful, *k),
            MergeStrategy::Aggregate(func) => Self::merge_aggregate(successful, *func),
            MergeStrategy::FirstNonEmpty => Self::merge_first_non_empty(successful),
            MergeStrategy::Concat => Self::merge_concat(successful),
        })
    }

    /// Merge results using union (combine all).
    fn merge_union(results: Vec<ShardResult>) -> QueryResult {
        let mut all_rows = Vec::new();
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        let mut all_similar = Vec::new();

        for shard_result in results {
            match shard_result.result {
                QueryResult::Rows(rows) => all_rows.extend(rows),
                QueryResult::Nodes(nodes) => all_nodes.extend(nodes),
                QueryResult::Edges(edges) => all_edges.extend(edges),
                QueryResult::Similar(similar) => all_similar.extend(similar),
                QueryResult::Count(n) => {
                    // Safety: usize to i64 wraps on 64-bit if n > i64::MAX, but count
                    // values are expected to be within reasonable bounds
                    #[allow(clippy::cast_possible_wrap)]
                    let count_val = n as i64;
                    all_rows.push(Row {
                        id: 0,
                        values: vec![("count".to_string(), Value::Int(count_val))],
                    });
                },
                _ => {},
            }
        }

        // Return appropriate type based on what we collected
        if !all_similar.is_empty() {
            return QueryResult::Similar(all_similar);
        }
        if !all_nodes.is_empty() {
            return QueryResult::Nodes(all_nodes);
        }
        if !all_edges.is_empty() {
            return QueryResult::Edges(all_edges);
        }
        if !all_rows.is_empty() {
            return QueryResult::Rows(all_rows);
        }

        QueryResult::Empty
    }

    /// Merge similarity results keeping top K.
    fn merge_top_k(results: Vec<ShardResult>, k: usize) -> QueryResult {
        let mut all_similar: Vec<SimilarResult> = Vec::new();

        for shard_result in results {
            if let QueryResult::Similar(similar) = shard_result.result {
                all_similar.extend(similar);
            }
        }

        // Sort by score descending
        all_similar.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top K
        all_similar.truncate(k);

        QueryResult::Similar(all_similar)
    }

    /// Merge using aggregate function.
    fn merge_aggregate(results: Vec<ShardResult>, func: AggregateFunction) -> QueryResult {
        let mut values: Vec<i64> = Vec::new();

        for shard_result in results {
            match shard_result.result {
                QueryResult::Count(n) => {
                    // Safety: usize to i64 wraps on 64-bit if n > i64::MAX, but count
                    // values are expected to be within reasonable bounds
                    #[allow(clippy::cast_possible_wrap)]
                    let count_val = n as i64;
                    values.push(count_val);
                },
                QueryResult::Value(s) => {
                    if let Ok(n) = s.parse::<i64>() {
                        values.push(n);
                    }
                },
                _ => {},
            }
        }

        if values.is_empty() {
            return QueryResult::Count(0);
        }

        // Safety: i64 to usize casts below may truncate on 32-bit systems or lose sign,
        // but aggregate results are expected to be non-negative and within usize range.
        // The len() to i64 cast may wrap if len > i64::MAX, but this is unrealistic.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_possible_wrap
        )]
        let result = match func {
            AggregateFunction::Sum | AggregateFunction::Count => {
                values.iter().sum::<i64>() as usize
            },
            AggregateFunction::Max => *values.iter().max().unwrap_or(&0) as usize,
            AggregateFunction::Min => *values.iter().min().unwrap_or(&0) as usize,
            AggregateFunction::Avg => (values.iter().sum::<i64>() / (values.len() as i64)) as usize,
        };

        QueryResult::Count(result)
    }

    /// Return first non-empty result.
    fn merge_first_non_empty(results: Vec<ShardResult>) -> QueryResult {
        for shard_result in results {
            if !matches!(&shard_result.result, QueryResult::Empty) {
                return shard_result.result;
            }
        }
        QueryResult::Empty
    }

    /// Concatenate all results in order.
    fn merge_concat(results: Vec<ShardResult>) -> QueryResult {
        // Same as union for most types
        Self::merge_union(results)
    }
}

/// Statistics for distributed query execution.
#[derive(Debug, Clone, Default)]
pub struct DistributedQueryStats {
    /// Total queries executed.
    pub queries_executed: u64,
    /// Local queries (no distribution needed).
    pub local_queries: u64,
    /// Remote single-shard queries.
    pub remote_queries: u64,
    /// Scatter-gather queries.
    pub scatter_gather_queries: u64,
    /// Total shards contacted.
    pub shards_contacted: u64,
    /// Average latency in microseconds.
    pub avg_latency_us: u64,
    /// Shard errors encountered.
    pub shard_errors: u64,
}

impl DistributedQueryStats {
    /// Record a query execution.
    pub const fn record_query(&mut self, plan: &QueryPlan, latency_us: u64, errors: usize) {
        self.queries_executed += 1;

        match plan {
            QueryPlan::Local { .. } => {
                self.local_queries += 1;
                self.shards_contacted += 1;
            },
            QueryPlan::Remote { .. } => {
                self.remote_queries += 1;
                self.shards_contacted += 1;
            },
            QueryPlan::ScatterGather { shards, .. } => {
                self.scatter_gather_queries += 1;
                self.shards_contacted += shards.len() as u64;
            },
        }

        self.shard_errors += errors as u64;

        // Update average latency
        if self.queries_executed == 1 {
            self.avg_latency_us = latency_us;
        } else {
            self.avg_latency_us = (self.avg_latency_us * (self.queries_executed - 1) + latency_us)
                / self.queries_executed;
        }
    }
}

#[cfg(test)]
mod tests {
    use tensor_store::{ConsistentHashConfig, ConsistentHashPartitioner};

    use super::*;

    fn create_test_partitioner() -> Arc<dyn Partitioner + Send + Sync> {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);
        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());
        Arc::new(partitioner)
    }

    #[test]
    fn test_query_plan_local() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("GET some_key");
        assert!(
            matches!(plan, QueryPlan::Local { .. } | QueryPlan::Remote { .. }),
            "Expected Local or Remote plan"
        );
    }

    #[test]
    fn test_query_plan_scatter_gather() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("SELECT users");
        assert!(
            matches!(
                plan,
                QueryPlan::ScatterGather {
                    merge: MergeStrategy::Union,
                    ..
                }
            ),
            "Expected ScatterGather with Union merge"
        );
    }

    #[test]
    fn test_query_plan_similar() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("SIMILAR key TOP 5");
        assert!(
            matches!(
                plan,
                QueryPlan::ScatterGather {
                    merge: MergeStrategy::TopK(5),
                    ..
                }
            ),
            "Expected ScatterGather with TopK(5) merge"
        );
    }

    #[test]
    fn test_query_plan_aggregate() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("SELECT COUNT(*) FROM users");
        assert!(
            matches!(
                plan,
                QueryPlan::ScatterGather {
                    merge: MergeStrategy::Aggregate(AggregateFunction::Count),
                    ..
                }
            ),
            "Expected ScatterGather with Count aggregate"
        );
    }

    #[test]
    fn test_merge_union() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(20), 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        let QueryResult::Rows(rows) = merged else {
            panic!("Expected Rows result");
        };
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_merge_top_k() {
        let results = vec![
            ShardResult::success(
                0,
                QueryResult::Similar(vec![
                    SimilarResult {
                        key: "a".to_string(),
                        score: 0.9,
                    },
                    SimilarResult {
                        key: "b".to_string(),
                        score: 0.8,
                    },
                ]),
                100,
            ),
            ShardResult::success(
                1,
                QueryResult::Similar(vec![SimilarResult {
                    key: "c".to_string(),
                    score: 0.95,
                }]),
                150,
            ),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::TopK(2)).unwrap();
        match merged {
            QueryResult::Similar(similar) => {
                assert_eq!(similar.len(), 2);
                assert_eq!(similar[0].key, "c"); // Highest score
                assert_eq!(similar[1].key, "a");
            },
            _ => panic!("Expected Similar result"),
        }
    }

    #[test]
    fn test_merge_aggregate_sum() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(20), 150),
            ShardResult::success(2, QueryResult::Count(30), 200),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 60),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_merge_aggregate_avg() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(20), 150),
            ShardResult::success(2, QueryResult::Count(30), 200),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Avg))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 20),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_merge_first_non_empty() {
        let results = vec![
            ShardResult::success(0, QueryResult::Empty, 100),
            ShardResult::success(1, QueryResult::Value("found".to_string()), 150),
            ShardResult::success(2, QueryResult::Value("also_found".to_string()), 200),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::FirstNonEmpty).unwrap();
        match merged {
            QueryResult::Value(s) => assert_eq!(s, "found"),
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_shard_result_success() {
        let result = ShardResult::success(0, QueryResult::Count(10), 100);
        assert_eq!(result.shard, 0);
        assert!(result.error.is_none());
        assert_eq!(result.execution_time_us, 100);
    }

    #[test]
    fn test_shard_result_error() {
        let result = ShardResult::error(1, "timeout".to_string());
        assert_eq!(result.shard, 1);
        assert!(result.error.is_some());
        assert_eq!(result.error.unwrap(), "timeout");
    }

    #[test]
    fn test_config_default() {
        let config = DistributedQueryConfig::default();
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.shard_timeout_ms, 5000);
        assert_eq!(config.retry_count, 2);
        assert!(!config.fail_fast);
    }

    #[test]
    fn test_stats_record_local() {
        let mut stats = DistributedQueryStats::default();
        let plan = QueryPlan::Local {
            query: "GET key".to_string(),
        };

        stats.record_query(&plan, 100, 0);

        assert_eq!(stats.queries_executed, 1);
        assert_eq!(stats.local_queries, 1);
        assert_eq!(stats.shards_contacted, 1);
        assert_eq!(stats.avg_latency_us, 100);
    }

    #[test]
    fn test_stats_record_scatter_gather() {
        let mut stats = DistributedQueryStats::default();
        let plan = QueryPlan::ScatterGather {
            shards: vec![0, 1, 2],
            query: "SELECT users".to_string(),
            merge: MergeStrategy::Union,
        };

        stats.record_query(&plan, 500, 1);

        assert_eq!(stats.queries_executed, 1);
        assert_eq!(stats.scatter_gather_queries, 1);
        assert_eq!(stats.shards_contacted, 3);
        assert_eq!(stats.shard_errors, 1);
    }

    #[test]
    fn test_merge_empty_results() {
        let results: Vec<ShardResult> = vec![];
        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        assert!(matches!(merged, QueryResult::Empty));
    }

    #[test]
    fn test_merge_filters_errors() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::error(1, "timeout".to_string()),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 10), // Only successful shard
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_planner_extract_key() {
        // Test various GET formats
        assert_eq!(
            QueryPlanner::extract_key("GET mykey"),
            Some("mykey".to_string())
        );
        assert_eq!(
            QueryPlanner::extract_key("NODE GET user:123"),
            Some("user:123".to_string())
        );
    }

    #[test]
    fn test_planner_extract_top_k() {
        assert_eq!(QueryPlanner::extract_top_k("SIMILAR key TOP 5"), Some(5));
        assert_eq!(
            QueryPlanner::extract_top_k("SIMILAR key TOP 100"),
            Some(100)
        );
        assert_eq!(QueryPlanner::extract_top_k("SIMILAR key"), None);
    }

    #[test]
    fn test_aggregate_function_equality() {
        assert_eq!(AggregateFunction::Sum, AggregateFunction::Sum);
        assert_ne!(AggregateFunction::Sum, AggregateFunction::Count);
    }

    #[test]
    fn test_all_shards() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let shards = planner.all_shards();
        assert_eq!(shards.len(), 3);
        assert_eq!(shards, vec![0, 1, 2]);
    }

    #[test]
    fn test_plan_with_embedding() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let plan = planner.plan_with_embedding("SIMILAR key TOP 10", &embedding);

        match plan {
            QueryPlan::ScatterGather { .. } => {},
            _ => panic!("Expected ScatterGather plan"),
        }
    }

    #[test]
    fn test_merge_max() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(50), 150),
            ShardResult::success(2, QueryResult::Count(30), 200),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Max))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 50),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_merge_min() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(50), 150),
            ShardResult::success(2, QueryResult::Count(30), 200),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Min))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 10),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_stats_avg_latency_updates() {
        let mut stats = DistributedQueryStats::default();
        let plan = QueryPlan::Local {
            query: "GET key".to_string(),
        };

        stats.record_query(&plan, 100, 0);
        assert_eq!(stats.avg_latency_us, 100);

        stats.record_query(&plan, 200, 0);
        assert_eq!(stats.avg_latency_us, 150);
    }

    #[test]
    fn test_merge_concat() {
        let results = vec![
            ShardResult::success(0, QueryResult::Count(10), 100),
            ShardResult::success(1, QueryResult::Count(20), 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Concat).unwrap();
        match merged {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 2),
            _ => panic!("Expected Rows result"),
        }
    }

    #[test]
    fn test_merge_union_nodes() {
        use crate::NodeResult;

        let results = vec![
            ShardResult::success(
                0,
                QueryResult::Nodes(vec![
                    NodeResult {
                        id: 1,
                        label: "Person".to_string(),
                        properties: std::collections::HashMap::new(),
                    },
                    NodeResult {
                        id: 2,
                        label: "Person".to_string(),
                        properties: std::collections::HashMap::new(),
                    },
                ]),
                100,
            ),
            ShardResult::success(
                1,
                QueryResult::Nodes(vec![NodeResult {
                    id: 3,
                    label: "Person".to_string(),
                    properties: std::collections::HashMap::new(),
                }]),
                150,
            ),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        match merged {
            QueryResult::Nodes(nodes) => assert_eq!(nodes.len(), 3),
            _ => panic!("Expected Nodes result"),
        }
    }

    #[test]
    fn test_merge_union_edges() {
        use crate::EdgeResult;

        let results = vec![
            ShardResult::success(
                0,
                QueryResult::Edges(vec![EdgeResult {
                    id: 1,
                    from: 1,
                    to: 2,
                    label: "KNOWS".to_string(),
                }]),
                100,
            ),
            ShardResult::success(
                1,
                QueryResult::Edges(vec![EdgeResult {
                    id: 2,
                    from: 2,
                    to: 3,
                    label: "KNOWS".to_string(),
                }]),
                150,
            ),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        match merged {
            QueryResult::Edges(edges) => assert_eq!(edges.len(), 2),
            _ => panic!("Expected Edges result"),
        }
    }

    #[test]
    fn test_merge_union_empty_all() {
        let results = vec![
            ShardResult::success(0, QueryResult::Empty, 100),
            ShardResult::success(1, QueryResult::Empty, 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        assert!(matches!(merged, QueryResult::Empty));
    }

    #[test]
    fn test_merge_first_non_empty_all_empty() {
        let results = vec![
            ShardResult::success(0, QueryResult::Empty, 100),
            ShardResult::success(1, QueryResult::Empty, 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::FirstNonEmpty).unwrap();
        assert!(matches!(merged, QueryResult::Empty));
    }

    #[test]
    fn test_merge_aggregate_value_strings() {
        let results = vec![
            ShardResult::success(0, QueryResult::Value("100".to_string()), 100),
            ShardResult::success(1, QueryResult::Value("200".to_string()), 150),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 300),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_merge_aggregate_empty_values() {
        let results = vec![
            ShardResult::success(0, QueryResult::Empty, 100),
            ShardResult::success(1, QueryResult::Empty, 150),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 0),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_query_plan_node_list() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("NODE LIST users");
        match plan {
            QueryPlan::ScatterGather {
                merge: MergeStrategy::Union,
                ..
            } => {},
            _ => panic!("Expected ScatterGather with Union merge"),
        }
    }

    #[test]
    fn test_query_plan_select_sum() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("SELECT SUM(amount) FROM orders");
        match plan {
            QueryPlan::ScatterGather {
                merge: MergeStrategy::Aggregate(AggregateFunction::Sum),
                ..
            } => {},
            _ => panic!("Expected ScatterGather with Sum aggregate"),
        }
    }

    #[test]
    fn test_query_plan_select_avg() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("SELECT AVG(price) FROM products");
        match plan {
            QueryPlan::ScatterGather {
                merge: MergeStrategy::Aggregate(AggregateFunction::Avg),
                ..
            } => {},
            _ => panic!("Expected ScatterGather with Avg aggregate"),
        }
    }

    #[test]
    fn test_query_plan_unknown() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        // Unknown query type should default to local
        let plan = planner.plan("FOOBAR something");
        match plan {
            QueryPlan::Local { .. } => {},
            _ => panic!("Expected Local plan for unknown query"),
        }
    }

    #[test]
    fn test_plan_with_embedding_non_similar() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        // Non-similarity query with embedding should fall back to plan()
        let plan = planner.plan_with_embedding("SELECT * FROM users", &embedding);

        match plan {
            QueryPlan::ScatterGather { .. } => {},
            _ => panic!("Expected ScatterGather plan"),
        }
    }

    #[test]
    fn test_extract_key_no_get() {
        // Query without GET keyword
        assert!(QueryPlanner::extract_key("something else").is_none());
    }

    #[test]
    fn test_extract_key_empty() {
        // Empty query
        assert!(QueryPlanner::extract_key("").is_none());
    }

    #[test]
    fn test_query_plan_node_get() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("NODE GET user:123");
        match plan {
            QueryPlan::Local { .. } | QueryPlan::Remote { .. } => {},
            _ => panic!("Expected Local or Remote plan"),
        }
    }

    #[test]
    fn test_query_plan_entity_get() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        let plan = planner.plan("ENTITY GET entity:456");
        match plan {
            QueryPlan::Local { .. } | QueryPlan::Remote { .. } => {},
            _ => panic!("Expected Local or Remote plan"),
        }
    }

    #[test]
    fn test_merge_top_k_non_similar_results() {
        // TopK merge with non-similar results should handle gracefully
        let results = vec![
            ShardResult::success(0, QueryResult::Empty, 100),
            ShardResult::success(1, QueryResult::Count(10), 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::TopK(5)).unwrap();
        match merged {
            QueryResult::Similar(similar) => assert!(similar.is_empty()),
            _ => panic!("Expected Similar result"),
        }
    }

    #[test]
    fn test_merge_aggregate_avg_empty() {
        // Edge case: empty values in avg should return 0
        let results = vec![ShardResult::success(0, QueryResult::Rows(vec![]), 100)];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Avg))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 0),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_query_plan_get_only_no_key() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        // "GET" without a key should fall through to Unknown -> Local
        let plan = planner.plan("GET");
        match plan {
            QueryPlan::Local { .. } => {},
            _ => panic!("Expected Local plan for GET without key"),
        }
    }

    #[test]
    fn test_query_plan_node_get_only() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        // "NODE GET" without a key should fall through to Unknown -> Local
        let plan = planner.plan("NODE GET");
        match plan {
            QueryPlan::Local { .. } => {},
            _ => panic!("Expected Local plan for NODE GET without key"),
        }
    }

    #[test]
    fn test_merge_union_other_result_types() {
        // Test with other QueryResult types that fall through to empty handling
        let results = vec![
            ShardResult::success(0, QueryResult::Path(vec![1, 2, 3]), 100),
            ShardResult::success(1, QueryResult::Value("test".to_string()), 150),
        ];

        let merged = ResultMerger::merge(results, &MergeStrategy::Union).unwrap();
        // Non-row, non-node, non-edge, non-similar types fall through
        assert!(matches!(merged, QueryResult::Empty));
    }

    #[test]
    fn test_stats_record_remote() {
        let mut stats = DistributedQueryStats::default();
        let plan = QueryPlan::Remote {
            shard: 1,
            query: "GET key".to_string(),
        };

        stats.record_query(&plan, 100, 0);

        assert_eq!(stats.queries_executed, 1);
        assert_eq!(stats.remote_queries, 1);
        assert_eq!(stats.shards_contacted, 1);
    }

    #[test]
    fn test_extract_key_get_at_end() {
        // "GET" at end without following key
        assert!(QueryPlanner::extract_key("something GET").is_none());
    }

    #[test]
    fn test_plan_with_embedding_empty_partitioner() {
        // Create partitioner with no nodes to trigger empty shards fallback
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = ConsistentHashPartitioner::new(config);
        let partitioner: Arc<dyn Partitioner + Send + Sync> = Arc::new(partitioner);
        let planner = QueryPlanner::new(partitioner, 0);

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        // With empty shards, should fall back to plan()
        let plan = planner.plan_with_embedding("SIMILAR key TOP 10", &embedding);

        // Falls back to plan() which returns Local for unknown query when no shards
        match plan {
            QueryPlan::Local { .. } | QueryPlan::ScatterGather { .. } => {},
            _ => panic!("Expected Local or ScatterGather plan"),
        }
    }

    #[test]
    fn test_all_shards_empty() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = ConsistentHashPartitioner::new(config);
        let partitioner: Arc<dyn Partitioner + Send + Sync> = Arc::new(partitioner);
        let planner = QueryPlanner::new(partitioner, 0);

        let shards = planner.all_shards();
        assert!(shards.is_empty());
    }

    #[test]
    fn test_plan_select_with_empty_partitioner() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = ConsistentHashPartitioner::new(config);
        let partitioner: Arc<dyn Partitioner + Send + Sync> = Arc::new(partitioner);
        let planner = QueryPlanner::new(partitioner, 0);

        // SELECT with no shards
        let plan = planner.plan("SELECT * FROM users");
        match plan {
            QueryPlan::ScatterGather { shards, .. } => {
                assert!(shards.is_empty());
            },
            _ => panic!("Expected ScatterGather plan"),
        }
    }

    #[test]
    fn test_get_with_trailing_space_no_key() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        // "GET " with trailing space but no key - triggers the GET block
        // but extract_key returns None since split_whitespace gives ["GET"]
        let plan = planner.plan("GET ");
        match plan {
            QueryPlan::Local { .. } => {},
            _ => panic!("Expected Local plan for GET without key"),
        }
    }

    #[test]
    fn test_merge_aggregate_unparseable_value() {
        // Test with Value string that cannot be parsed as i64
        let results = vec![
            ShardResult::success(0, QueryResult::Value("not_a_number".to_string()), 100),
            ShardResult::success(1, QueryResult::Count(100), 150),
        ];

        let merged =
            ResultMerger::merge(results, &MergeStrategy::Aggregate(AggregateFunction::Sum))
                .unwrap();
        match merged {
            QueryResult::Count(n) => assert_eq!(n, 100), // Only the Count value is used
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_node_get_trailing_space() {
        let partitioner = create_test_partitioner();
        let planner = QueryPlanner::new(partitioner, 0);

        // "NODE GET " triggers the block but extract_key fails
        let plan = planner.plan("NODE GET ");
        match plan {
            QueryPlan::Local { .. } => {},
            _ => panic!("Expected Local plan"),
        }
    }

    #[test]
    fn test_debug_impls() {
        // Test Debug implementations for coverage
        let config = DistributedQueryConfig::default();
        let _ = format!("{:?}", config);

        let plan_local = QueryPlan::Local {
            query: "test".to_string(),
        };
        let plan_remote = QueryPlan::Remote {
            shard: 0,
            query: "test".to_string(),
        };
        let plan_scatter = QueryPlan::ScatterGather {
            shards: vec![0, 1],
            query: "test".to_string(),
            merge: MergeStrategy::Union,
        };
        let _ = format!("{:?}", plan_local);
        let _ = format!("{:?}", plan_remote);
        let _ = format!("{:?}", plan_scatter);

        let _ = format!("{:?}", MergeStrategy::TopK(10));
        let _ = format!("{:?}", MergeStrategy::Aggregate(AggregateFunction::Count));
        let _ = format!("{:?}", MergeStrategy::FirstNonEmpty);
        let _ = format!("{:?}", MergeStrategy::Concat);

        let _ = format!("{:?}", AggregateFunction::Max);
        let _ = format!("{:?}", AggregateFunction::Min);

        let result = ShardResult::success(0, QueryResult::Empty, 100);
        let _ = format!("{:?}", result);

        let stats = DistributedQueryStats::default();
        let _ = format!("{:?}", stats);
    }

    #[test]
    fn test_shard_result_clone() {
        let result = ShardResult::success(0, QueryResult::Count(10), 100);
        let cloned = result.clone();
        assert_eq!(cloned.shard, result.shard);
    }

    #[test]
    fn test_config_clone() {
        let config = DistributedQueryConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_concurrent, config.max_concurrent);
    }

    #[test]
    fn test_stats_clone() {
        let mut stats = DistributedQueryStats::default();
        stats.queries_executed = 10;
        let cloned = stats.clone();
        assert_eq!(cloned.queries_executed, 10);
    }

    #[test]
    fn test_merge_strategy_clone() {
        let strategy = MergeStrategy::TopK(5);
        let cloned = strategy.clone();
        assert!(matches!(cloned, MergeStrategy::TopK(5)));
    }

    #[test]
    fn test_aggregate_function_copy() {
        let func = AggregateFunction::Sum;
        let copied: AggregateFunction = func;
        assert_eq!(copied, AggregateFunction::Sum);
    }

    #[test]
    fn test_query_plan_clone() {
        let plan = QueryPlan::Local {
            query: "test".to_string(),
        };
        let cloned = plan.clone();
        assert!(matches!(cloned, QueryPlan::Local { .. }));
    }
}
