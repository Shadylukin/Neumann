// SPDX-License-Identifier: MIT OR Apache-2.0
//! `CLUSTER` statement execution plus distributed-query dispatch helpers
//! (`try_execute_distributed`, `execute_on_shard`, `execute_scatter_gather`,
//! `execute_parsed_local`) shared by the synchronous and async public APIs.

use neumann_parser::{self as parser, ClusterOp, ClusterStmt};
use tensor_chain::ClusterOrchestrator;
use tokio::runtime::Runtime;

use crate::distributed::{MergeStrategy, QueryPlan, ResultMerger, ShardId, ShardResult};
use crate::{QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Attempt distributed execution. Returns `None` when the query should be
/// executed locally (no cluster, or planner picked local shard).
pub fn try_execute_distributed(router: &QueryRouter, command: &str) -> Option<Result<QueryResult>> {
    let planner = router.distributed_planner.as_ref()?;
    let cluster = router.cluster.as_ref()?;
    let runtime = router.cluster_runtime.as_ref()?;

    let plan = planner.plan(command);

    match plan {
        QueryPlan::Local { .. } => None,
        QueryPlan::Remote { shard, query } => {
            Some(execute_on_shard(router, runtime, cluster, shard, &query))
        },
        QueryPlan::ScatterGather {
            shards,
            query,
            merge,
        } => Some(execute_scatter_gather(
            router, runtime, cluster, &shards, &query, &merge,
        )),
    }
}

/// Forward a query to a single remote shard and decode the result.
pub fn execute_on_shard(
    router: &QueryRouter,
    runtime: &Runtime,
    cluster: &ClusterOrchestrator,
    shard: ShardId,
    query: &str,
) -> Result<QueryResult> {
    let nodes = cluster.membership().view().nodes;
    let node_id = nodes
        .get(shard)
        .map(|n| n.node_id.clone())
        .ok_or_else(|| RouterError::InvalidArgument(format!("Shard {shard} not found")))?;

    let request = tensor_chain::QueryRequest {
        query_id: rand::random(),
        query: query.to_string(),
        shard_id: shard,
        embedding: None,
        timeout_ms: router.distributed_config.shard_timeout_ms,
    };

    let response = runtime.block_on(async {
        cluster
            .send_query(&node_id, request)
            .await
            .map_err(|e| RouterError::ChainError(e.to_string()))
    })?;

    if response.success {
        bitcode::deserialize(&response.result)
            .map_err(|e| RouterError::ChainError(format!("Deserialization error: {e}")))
    } else {
        Err(RouterError::ChainError(
            response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()),
        ))
    }
}

/// Run a query against every shard in parallel and merge the results.
#[allow(
    clippy::too_many_lines,
    reason = "scatter-gather covers local + remote shard paths"
)]
pub fn execute_scatter_gather(
    router: &QueryRouter,
    runtime: &Runtime,
    cluster: &ClusterOrchestrator,
    shards: &[ShardId],
    query: &str,
    merge_strategy: &MergeStrategy,
) -> Result<QueryResult> {
    let nodes = cluster.membership().view().nodes;

    let results: Vec<ShardResult> = runtime.block_on(async {
        let mut results = Vec::with_capacity(shards.len());

        for &shard in shards {
            let start = std::time::Instant::now();

            if shard == router.local_shard_id {
                match execute_parsed_local(router, query) {
                    Ok(result) => {
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "microseconds won't exceed u64::MAX"
                        )]
                        let elapsed = start.elapsed().as_micros() as u64;
                        results.push(ShardResult::success(shard, result, elapsed));
                    },
                    Err(e) => {
                        if router.distributed_config.fail_fast {
                            return vec![ShardResult::error(shard, e.to_string())];
                        }
                        results.push(ShardResult::error(shard, e.to_string()));
                    },
                }
                continue;
            }

            let Some(n) = nodes.get(shard) else {
                results.push(ShardResult::error(
                    shard,
                    format!("Shard {shard} not found"),
                ));
                continue;
            };
            let node_id = n.node_id.clone();

            let request = tensor_chain::QueryRequest {
                query_id: rand::random(),
                query: query.to_string(),
                shard_id: shard,
                embedding: None,
                timeout_ms: router.distributed_config.shard_timeout_ms,
            };

            match cluster.send_query(&node_id, request).await {
                Ok(response) => {
                    if response.success {
                        match bitcode::deserialize(&response.result) {
                            Ok(result) => {
                                results.push(ShardResult::success(
                                    shard,
                                    result,
                                    response.execution_time_us,
                                ));
                            },
                            Err(e) => {
                                results.push(ShardResult::error(
                                    shard,
                                    format!("Deserialization error: {e}"),
                                ));
                            },
                        }
                    } else {
                        results.push(ShardResult::error(
                            shard,
                            response
                                .error
                                .unwrap_or_else(|| "Unknown error".to_string()),
                        ));
                    }
                },
                Err(e) => {
                    results.push(ShardResult::error(shard, e.to_string()));
                },
            }
        }

        results
    });

    ResultMerger::merge(results, merge_strategy)
}

/// Parse and execute a query locally, bypassing distributed routing.
pub fn execute_parsed_local(router: &QueryRouter, command: &str) -> Result<QueryResult> {
    let stmt = parser::parse(command)
        .map_err(|e| RouterError::ParseError(e.format_with_source(command)))?;
    router.execute_statement(&stmt)
}

/// Execute a `CLUSTER ...` statement.
#[allow(clippy::too_many_lines, reason = "covers every CLUSTER sub-op")]
pub fn exec_cluster(router: &QueryRouter, stmt: &ClusterStmt) -> Result<QueryResult> {
    match &stmt.operation {
        ClusterOp::Connect { addresses } => {
            let addr_str = expr::eval_string_expr(addresses)?;
            Err(RouterError::InvalidArgument(format!(
                "CLUSTER CONNECT '{addr_str}' requires shell support. Use the shell command or call router.init_cluster() from code."
            )))
        },
        ClusterOp::Disconnect => {
            if router.cluster.is_none() {
                return Err(RouterError::InvalidArgument(
                    "Not connected to cluster".to_string(),
                ));
            }
            Err(RouterError::InvalidArgument(
                "CLUSTER DISCONNECT requires shell support. Use the shell command or call router.shutdown_cluster() from code.".to_string(),
            ))
        },
        ClusterOp::Status => router.cluster.as_ref().map_or_else(
            || {
                Ok(QueryResult::Value(
                    "Cluster: single-node mode (not connected)".to_string(),
                ))
            },
            |cluster| {
                let node_id = cluster.node_id();
                let is_leader = cluster.is_leader();
                let leader = cluster
                    .current_leader()
                    .unwrap_or_else(|| "(unknown)".to_string());
                let height = cluster.chain_height();
                let commit_idx = cluster.commit_index();

                let status = format!(
                    "Cluster Status:\n  Node ID: {}\n  Role: {}\n  Leader: {}\n  Chain Height: {}\n  Commit Index: {}",
                    node_id,
                    if is_leader { "Leader" } else { "Follower" },
                    leader,
                    height,
                    commit_idx
                );
                Ok(QueryResult::Value(status))
            },
        ),
        ClusterOp::Nodes => router.cluster.as_ref().map_or_else(
            || {
                Ok(QueryResult::Value(
                    "No cluster nodes (single-node mode)".to_string(),
                ))
            },
            |cluster| {
                let membership = cluster.membership();
                let view = membership.view();
                let local_id = cluster.node_id();

                let mut nodes = vec![format!("  {} (self)", local_id)];
                for node in &view.nodes {
                    if &node.node_id != local_id {
                        let status = if view.healthy_nodes.contains(&node.node_id) {
                            "healthy"
                        } else if view.failed_nodes.contains(&node.node_id) {
                            "failed"
                        } else {
                            "unknown"
                        };
                        nodes.push(format!("  {} - {}", node.node_id, status));
                    }
                }

                Ok(QueryResult::Value(format!(
                    "Cluster Nodes ({}):\n{}",
                    nodes.len(),
                    nodes.join("\n")
                )))
            },
        ),
        ClusterOp::Leader => router.cluster.as_ref().map_or_else(
            || {
                Ok(QueryResult::Value(
                    "No leader (single-node mode)".to_string(),
                ))
            },
            |cluster| {
                let leader = cluster.current_leader();
                let is_self = cluster.is_leader();

                leader.map_or_else(
                    || {
                        Ok(QueryResult::Value(
                            "No leader elected (election in progress)".to_string(),
                        ))
                    },
                    |leader_id| {
                        if is_self {
                            Ok(QueryResult::Value(format!(
                                "Leader: {leader_id} (this node)"
                            )))
                        } else {
                            Ok(QueryResult::Value(format!("Leader: {leader_id}")))
                        }
                    },
                )
            },
        ),
    }
}

/// Serialize a query result via `execute_parsed` for cross-cluster RPC.
pub fn execute_for_cluster(
    router: &QueryRouter,
    query: &str,
) -> std::result::Result<Vec<u8>, String> {
    let result = router.execute_parsed(query).map_err(|e| e.to_string())?;
    bitcode::serialize(&result).map_err(|e| format!("Serialization error: {e}"))
}
