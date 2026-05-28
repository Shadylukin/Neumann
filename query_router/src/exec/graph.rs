// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph statement execution: NODE/EDGE/NEIGHBORS/PATH plus the extended
//! GraphAlgorithm/GraphConstraint/GraphIndex/GraphAggregate/GraphPattern/GraphBatch
//! family.
//!
//! These methods stay on `impl QueryRouter` (split into a sibling impl block) so
//! they can call other private router methods (`self.expr_*`, `self.graph`, etc.)
//! unchanged. The dispatcher in `lib.rs` calls them as `router.exec_node(...)`.

#![allow(
    clippy::too_many_lines,
    reason = "match dispatchers cover many sub-ops"
)]

use std::collections::HashMap;

use graph_engine::{
    CentralityConfig, CommunityConfig, Constraint, ConstraintTarget as GConstraintTarget,
    ConstraintType as GConstraintType, Direction, EdgeInput, GraphError, NodeInput, PageRankConfig,
    PropertyValue,
};
use neumann_parser::{
    self as parser, AggregateFunction, ConstraintTarget, ConstraintType,
    Direction as ParsedDirection, EdgeOp, EdgeStmt, Expr, GraphAggregateOp, GraphAggregateStmt,
    GraphAlgorithmOp, GraphAlgorithmStmt, GraphBatchOp, GraphBatchStmt, GraphConstraintOp,
    GraphConstraintStmt, GraphIndexOp, GraphIndexStmt, GraphPatternOp, GraphPatternStmt,
    NeighborsStmt, NodeOp, NodeStmt, PathStmt,
};
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::result::{
    AggregateResultValue, BatchOperationResult, BindingValue, CentralityItem, CentralityResult,
    CentralityType, CommunityItem, CommunityResult, ConstraintInfo, EdgeResult, NodeResult,
    PageRankItem, PageRankResult, PatternMatchBinding, PatternMatchResultValue,
    PatternMatchStatsValue, SimilarResult,
};
use crate::{QueryResult, QueryRouter, Result, RouterError};

impl QueryRouter {
    #[allow(clippy::too_many_lines)] // Graph algorithm dispatch requires handling many algorithm types
    pub(crate) fn exec_graph_algorithm(&self, stmt: &GraphAlgorithmStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphAlgorithmOp::PageRank {
                damping,
                tolerance,
                max_iterations,
                direction,
                edge_type,
            } => {
                let config = PageRankConfig {
                    damping: damping
                        .as_ref()
                        .map(|e| self.expr_to_f64(e))
                        .transpose()?
                        .unwrap_or(0.85),
                    tolerance: tolerance
                        .as_ref()
                        .map(|e| self.expr_to_f64(e))
                        .transpose()?
                        .unwrap_or(1e-6),
                    max_iterations: max_iterations
                        .as_ref()
                        .map(|e| self.expr_to_usize(e))
                        .transpose()?
                        .unwrap_or(100),
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Outgoing, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                };
                let result = self.graph.pagerank(Some(config))?;
                let items: Vec<PageRankItem> = result
                    .scores
                    .into_iter()
                    .map(|(node_id, score)| PageRankItem { node_id, score })
                    .collect();
                Ok(QueryResult::PageRank(PageRankResult {
                    items,
                    iterations: result.iterations,
                    convergence: result.convergence,
                    converged: result.converged,
                }))
            },
            GraphAlgorithmOp::BetweennessCentrality {
                sampling_ratio,
                direction,
                edge_type,
            } => {
                let config = CentralityConfig {
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Both, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                    sampling_ratio: sampling_ratio
                        .as_ref()
                        .map(|e| self.expr_to_f64(e))
                        .transpose()?
                        .unwrap_or(1.0),
                    max_iterations: 100,
                    tolerance: 1e-6,
                };
                let result = self.graph.betweenness_centrality(Some(config))?;
                let items: Vec<CentralityItem> = result
                    .scores
                    .into_iter()
                    .map(|(node_id, score)| CentralityItem { node_id, score })
                    .collect();
                Ok(QueryResult::Centrality(CentralityResult {
                    items,
                    centrality_type: CentralityType::Betweenness,
                    iterations: result.iterations,
                    converged: result.converged,
                    sample_count: result.sample_count,
                }))
            },
            GraphAlgorithmOp::ClosenessCentrality {
                direction,
                edge_type,
            } => {
                let config = CentralityConfig {
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Both, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                    sampling_ratio: 1.0,
                    max_iterations: 100,
                    tolerance: 1e-6,
                };
                let result = self.graph.closeness_centrality(Some(config))?;
                let items: Vec<CentralityItem> = result
                    .scores
                    .into_iter()
                    .map(|(node_id, score)| CentralityItem { node_id, score })
                    .collect();
                Ok(QueryResult::Centrality(CentralityResult {
                    items,
                    centrality_type: CentralityType::Closeness,
                    iterations: result.iterations,
                    converged: result.converged,
                    sample_count: result.sample_count,
                }))
            },
            GraphAlgorithmOp::EigenvectorCentrality {
                max_iterations,
                tolerance,
                direction,
                edge_type,
            } => {
                let config = CentralityConfig {
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Both, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                    sampling_ratio: 1.0,
                    max_iterations: max_iterations
                        .as_ref()
                        .map(|e| self.expr_to_usize(e))
                        .transpose()?
                        .unwrap_or(100),
                    tolerance: tolerance
                        .as_ref()
                        .map(|e| self.expr_to_f64(e))
                        .transpose()?
                        .unwrap_or(1e-6),
                };
                let result = self.graph.eigenvector_centrality(Some(config))?;
                let items: Vec<CentralityItem> = result
                    .scores
                    .into_iter()
                    .map(|(node_id, score)| CentralityItem { node_id, score })
                    .collect();
                Ok(QueryResult::Centrality(CentralityResult {
                    items,
                    centrality_type: CentralityType::Eigenvector,
                    iterations: result.iterations,
                    converged: result.converged,
                    sample_count: result.sample_count,
                }))
            },
            GraphAlgorithmOp::LouvainCommunities {
                resolution,
                max_passes,
                direction,
                edge_type,
            } => {
                let config = CommunityConfig {
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Both, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                    resolution: resolution
                        .as_ref()
                        .map(|e| self.expr_to_f64(e))
                        .transpose()?
                        .unwrap_or(1.0),
                    max_passes: max_passes
                        .as_ref()
                        .map(|e| self.expr_to_usize(e))
                        .transpose()?
                        .unwrap_or(10),
                    max_iterations: 100,
                    seed: None,
                };
                let result = self.graph.louvain_communities(Some(config))?;
                let items: Vec<CommunityItem> = result
                    .communities
                    .iter()
                    .map(|(&node_id, &community_id)| CommunityItem {
                        node_id,
                        community_id,
                    })
                    .collect();
                Ok(QueryResult::Communities(CommunityResult {
                    items,
                    members: result.members,
                    community_count: result.community_count,
                    modularity: result.modularity,
                    passes: result.passes,
                    iterations: result.iterations,
                }))
            },
            GraphAlgorithmOp::LabelPropagation {
                max_iterations,
                direction,
                edge_type,
            } => {
                let config = CommunityConfig {
                    direction: direction
                        .as_ref()
                        .map_or(Direction::Both, |d| self.convert_parsed_direction(d)),
                    edge_type: edge_type.as_ref().map(|e| e.name.clone()),
                    resolution: 1.0,
                    max_passes: 10,
                    max_iterations: max_iterations
                        .as_ref()
                        .map(|e| self.expr_to_usize(e))
                        .transpose()?
                        .unwrap_or(100),
                    seed: None,
                };
                let result = self.graph.label_propagation(Some(config))?;
                let items: Vec<CommunityItem> = result
                    .communities
                    .iter()
                    .map(|(&node_id, &community_id)| CommunityItem {
                        node_id,
                        community_id,
                    })
                    .collect();
                Ok(QueryResult::Communities(CommunityResult {
                    items,
                    members: result.members,
                    community_count: result.community_count,
                    modularity: result.modularity,
                    passes: result.passes,
                    iterations: result.iterations,
                }))
            },
        }
    }

    pub(crate) fn exec_graph_constraint(&self, stmt: &GraphConstraintStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphConstraintOp::Create {
                name,
                target,
                property,
                constraint_type,
            } => {
                let g_target = match target {
                    ConstraintTarget::Node { label } => {
                        label.as_ref().map_or(GConstraintTarget::AllNodes, |l| {
                            GConstraintTarget::NodeLabel(l.name.clone())
                        })
                    },
                    ConstraintTarget::Edge { edge_type } => {
                        edge_type.as_ref().map_or(GConstraintTarget::AllEdges, |t| {
                            GConstraintTarget::EdgeType(t.name.clone())
                        })
                    },
                };
                let g_type = match constraint_type {
                    ConstraintType::Unique => GConstraintType::Unique,
                    ConstraintType::Exists => GConstraintType::Exists,
                    ConstraintType::Type(t) => {
                        use graph_engine::PropertyValueType;
                        let type_name = t.to_uppercase();
                        match type_name.as_str() {
                            "INT" | "INTEGER" => {
                                GConstraintType::PropertyType(PropertyValueType::Int)
                            },
                            "FLOAT" | "DOUBLE" => {
                                GConstraintType::PropertyType(PropertyValueType::Float)
                            },
                            "BOOL" | "BOOLEAN" => {
                                GConstraintType::PropertyType(PropertyValueType::Bool)
                            },
                            // Default to String for "STRING" and any unrecognized types
                            _ => GConstraintType::PropertyType(PropertyValueType::String),
                        }
                    },
                };
                let constraint = Constraint {
                    name: name.name.clone(),
                    target: g_target,
                    property: property.name.clone(),
                    constraint_type: g_type,
                };
                self.graph.create_constraint(constraint)?;
                Ok(QueryResult::Empty)
            },
            GraphConstraintOp::Drop { name } => {
                self.graph.drop_constraint(&name.name)?;
                Ok(QueryResult::Empty)
            },
            GraphConstraintOp::List => {
                let constraints = self.graph.list_constraints();
                let results: Vec<ConstraintInfo> = constraints
                    .into_iter()
                    .map(|c| ConstraintInfo {
                        name: c.name,
                        target: match c.target {
                            GConstraintTarget::NodeLabel(l) => format!("Node({l})"),
                            GConstraintTarget::EdgeType(t) => format!("Edge({t})"),
                            GConstraintTarget::AllNodes => "AllNodes".to_string(),
                            GConstraintTarget::AllEdges => "AllEdges".to_string(),
                        },
                        property: c.property,
                        constraint_type: match c.constraint_type {
                            GConstraintType::Unique => "UNIQUE".to_string(),
                            GConstraintType::Exists => "EXISTS".to_string(),
                            GConstraintType::PropertyType(t) => format!("TYPE({t:?})"),
                        },
                    })
                    .collect();
                Ok(QueryResult::Constraints(results))
            },
            GraphConstraintOp::Get { name } => match self.graph.get_constraint(&name.name) {
                Some(c) => {
                    let info = ConstraintInfo {
                        name: c.name,
                        target: match c.target {
                            GConstraintTarget::NodeLabel(l) => format!("Node({l})"),
                            GConstraintTarget::EdgeType(t) => format!("Edge({t})"),
                            GConstraintTarget::AllNodes => "AllNodes".to_string(),
                            GConstraintTarget::AllEdges => "AllEdges".to_string(),
                        },
                        property: c.property,
                        constraint_type: match c.constraint_type {
                            GConstraintType::Unique => "UNIQUE".to_string(),
                            GConstraintType::Exists => "EXISTS".to_string(),
                            GConstraintType::PropertyType(t) => format!("TYPE({t:?})"),
                        },
                    };
                    Ok(QueryResult::Constraints(vec![info]))
                },
                None => Ok(QueryResult::Constraints(vec![])),
            },
        }
    }

    pub(crate) fn exec_graph_index(&self, stmt: &GraphIndexStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphIndexOp::CreateNodeProperty { property } => {
                self.graph.create_node_property_index(&property.name)?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::CreateEdgeProperty { property } => {
                self.graph.create_edge_property_index(&property.name)?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::CreateLabel => {
                self.graph.create_label_index()?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::CreateEdgeType => {
                self.graph.create_edge_type_index()?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::DropNode { property } => {
                self.graph.drop_node_index(&property.name)?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::DropEdge { property } => {
                self.graph.drop_edge_index(&property.name)?;
                Ok(QueryResult::Empty)
            },
            GraphIndexOp::ShowNodeIndexes => {
                let indexes = self.graph.get_indexed_node_properties();
                Ok(QueryResult::GraphIndexes(indexes))
            },
            GraphIndexOp::ShowEdgeIndexes => {
                let indexes = self.graph.get_indexed_edge_properties();
                Ok(QueryResult::GraphIndexes(indexes))
            },
        }
    }

    pub(crate) fn exec_graph_aggregate(&self, stmt: &GraphAggregateStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphAggregateOp::CountNodes { label } => {
                let count = match label {
                    Some(l) => self.graph.count_nodes_by_label(&l.name)?,
                    None => self.graph.count_nodes(),
                };
                Ok(QueryResult::Aggregate(AggregateResultValue::Count(count)))
            },
            GraphAggregateOp::CountEdges { edge_type } => {
                let count = match edge_type {
                    Some(t) => self.graph.count_edges_by_type(&t.name)?,
                    None => self.graph.count_edges(),
                };
                Ok(QueryResult::Aggregate(AggregateResultValue::Count(count)))
            },
            GraphAggregateOp::AggregateNodeProperty {
                function,
                property,
                label,
                ..
            } => {
                // Get the aggregate result based on whether we filter by label
                let agg = match label {
                    Some(l) => self
                        .graph
                        .aggregate_node_property_by_label(&l.name, &property.name)?,
                    None => self.graph.aggregate_node_property(&property.name),
                };
                let result = match function {
                    AggregateFunction::Sum => AggregateResultValue::Sum(agg.sum.unwrap_or(0.0)),
                    AggregateFunction::Avg => AggregateResultValue::Avg(agg.avg.unwrap_or(0.0)),
                    AggregateFunction::Min => AggregateResultValue::Min(
                        self.property_value_to_f64(agg.min).unwrap_or(0.0),
                    ),
                    AggregateFunction::Max => AggregateResultValue::Max(
                        self.property_value_to_f64(agg.max).unwrap_or(0.0),
                    ),
                    AggregateFunction::Count => AggregateResultValue::Count(agg.count),
                };
                Ok(QueryResult::Aggregate(result))
            },
            GraphAggregateOp::AggregateEdgeProperty {
                function,
                property,
                edge_type,
                ..
            } => {
                // Get the aggregate result based on whether we filter by edge type
                let agg = match edge_type {
                    Some(t) => self
                        .graph
                        .aggregate_edge_property_by_type(&t.name, &property.name)?,
                    None => self.graph.aggregate_edge_property(&property.name),
                };
                let result = match function {
                    AggregateFunction::Sum => AggregateResultValue::Sum(agg.sum.unwrap_or(0.0)),
                    AggregateFunction::Avg => AggregateResultValue::Avg(agg.avg.unwrap_or(0.0)),
                    AggregateFunction::Min => AggregateResultValue::Min(
                        self.property_value_to_f64(agg.min).unwrap_or(0.0),
                    ),
                    AggregateFunction::Max => AggregateResultValue::Max(
                        self.property_value_to_f64(agg.max).unwrap_or(0.0),
                    ),
                    AggregateFunction::Count => AggregateResultValue::Count(agg.count),
                };
                Ok(QueryResult::Aggregate(result))
            },
        }
    }

    pub(crate) fn exec_graph_pattern(&self, stmt: &GraphPatternStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphPatternOp::Match { pattern, limit } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, limit.as_ref())?;
                let result = self.graph.match_pattern(&gp)?;
                Ok(QueryResult::PatternMatch(
                    self.convert_pattern_match_result(&result),
                ))
            },
            GraphPatternOp::Count { pattern } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, None)?;
                let count = self.graph.count_pattern_matches(&gp)?;
                Ok(QueryResult::Aggregate(AggregateResultValue::Count(count)))
            },
            GraphPatternOp::Exists { pattern } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, None)?;
                let exists = self.graph.pattern_exists(&gp)?;
                Ok(QueryResult::Value(exists.to_string()))
            },
        }
    }

    fn pattern_spec_to_graph_pattern(
        &self,
        pattern: &parser::PatternSpec,
        limit: Option<&Expr>,
    ) -> Result<graph_engine::Pattern> {
        use graph_engine::{EdgePattern, NodePattern, PathPattern, Pattern};

        if pattern.nodes.is_empty() {
            return Err(RouterError::InvalidArgument(
                "Pattern must have at least one node".to_string(),
            ));
        }

        // Build node patterns from AST
        let build_node_pattern = |spec: &parser::NodePatternSpec| -> NodePattern {
            let mut np = NodePattern::new();
            if let Some(alias) = &spec.alias {
                np = np.variable(&alias.name);
            }
            if let Some(label) = &spec.label {
                np = np.label(&label.name);
            }
            np
        };

        // If there are no edges, return a pattern that matches just the first node
        if pattern.edges.is_empty() {
            let start = build_node_pattern(&pattern.nodes[0]);
            // Create a minimal pattern with just one node (edge and end required by API)
            let path = PathPattern::new(start, EdgePattern::new(), NodePattern::new());
            let mut gp = Pattern::new(path);
            if let Some(lim) = limit {
                gp = gp.limit(self.expr_to_usize(lim)?);
            }
            return Ok(gp);
        }

        // Build path from edges - edges reference nodes by index
        let first_edge = &pattern.edges[0];
        let start_node = build_node_pattern(&pattern.nodes[first_edge.from_node]);

        let edge =
            EdgePattern::new().direction(self.convert_parsed_direction(&first_edge.direction));
        let edge = if let Some(alias) = &first_edge.alias {
            edge.variable(&alias.name)
        } else {
            edge
        };
        let edge = if let Some(et) = &first_edge.edge_type {
            edge.edge_type(&et.name)
        } else {
            edge
        };

        let end_node = build_node_pattern(&pattern.nodes[first_edge.to_node]);
        let mut path = PathPattern::new(start_node, edge, end_node);

        // Extend path with remaining edges
        for edge_spec in pattern.edges.iter().skip(1) {
            let edge =
                EdgePattern::new().direction(self.convert_parsed_direction(&edge_spec.direction));
            let edge = if let Some(alias) = &edge_spec.alias {
                edge.variable(&alias.name)
            } else {
                edge
            };
            let edge = if let Some(et) = &edge_spec.edge_type {
                edge.edge_type(&et.name)
            } else {
                edge
            };

            let target_node = build_node_pattern(&pattern.nodes[edge_spec.to_node]);
            path = path.extend(edge, target_node);
        }

        let mut gp = Pattern::new(path);
        if let Some(lim) = limit {
            gp = gp.limit(self.expr_to_usize(lim)?);
        }

        Ok(gp)
    }

    #[allow(clippy::unused_self)] // Method signature for API consistency
    fn convert_pattern_match_result(
        &self,
        result: &graph_engine::PatternMatchResult,
    ) -> PatternMatchResultValue {
        use graph_engine::Binding;

        let matches = result
            .matches
            .iter()
            .map(|m| {
                let bindings = m
                    .bindings
                    .iter()
                    .map(|(k, v)| {
                        let binding = match v {
                            Binding::Node(n) => BindingValue::Node {
                                id: n.id,
                                label: n.labels.join(", "),
                            },
                            Binding::Edge(e) => BindingValue::Edge {
                                id: e.id,
                                edge_type: e.edge_type.clone(),
                                from: e.from,
                                to: e.to,
                            },
                            Binding::Path(p) => BindingValue::Path {
                                nodes: p.nodes.clone(),
                                edges: p.edges.clone(),
                                length: p.nodes.len().saturating_sub(1),
                            },
                        };
                        (k.clone(), binding)
                    })
                    .collect();
                PatternMatchBinding { bindings }
            })
            .collect();

        PatternMatchResultValue {
            matches,
            stats: PatternMatchStatsValue {
                matches_found: result.stats.matches_found,
                nodes_evaluated: result.stats.nodes_evaluated,
                edges_evaluated: result.stats.edges_evaluated,
                truncated: result.stats.truncated,
            },
        }
    }

    #[allow(clippy::too_many_lines)] // Batch operations require handling multiple node/edge scenarios
    pub(crate) fn exec_graph_batch(&self, stmt: &GraphBatchStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphBatchOp::CreateNodes { nodes } => {
                let inputs: Vec<NodeInput> = nodes
                    .iter()
                    .map(|n| {
                        let props = n
                            .properties
                            .iter()
                            .map(|p| {
                                let pv = self
                                    .expr_to_property_value(&p.value)
                                    .unwrap_or(PropertyValue::Null);
                                (p.key.name.clone(), pv)
                            })
                            .collect();
                        NodeInput {
                            labels: n.labels.iter().map(|l| l.name.clone()).collect(),
                            properties: props,
                        }
                    })
                    .collect();
                let result = self.graph.batch_create_nodes(inputs)?;
                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "CREATE_NODES".to_string(),
                    affected_count: result.count,
                    created_ids: Some(result.created_ids),
                }))
            },
            GraphBatchOp::CreateEdges { edges } => {
                let inputs: Vec<EdgeInput> = edges
                    .iter()
                    .map(|e| {
                        let from_id = self.expr_to_u64(&e.from_id).unwrap_or(0);
                        let to_id = self.expr_to_u64(&e.to_id).unwrap_or(0);
                        let props = e
                            .properties
                            .iter()
                            .map(|p| {
                                let pv = self
                                    .expr_to_property_value(&p.value)
                                    .unwrap_or(PropertyValue::Null);
                                (p.key.name.clone(), pv)
                            })
                            .collect();
                        EdgeInput {
                            from: from_id,
                            to: to_id,
                            edge_type: e.edge_type.name.clone(),
                            properties: props,
                            directed: true,
                        }
                    })
                    .collect();
                let result = self.graph.batch_create_edges(inputs)?;
                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "CREATE_EDGES".to_string(),
                    affected_count: result.count,
                    created_ids: Some(result.created_ids),
                }))
            },
            GraphBatchOp::DeleteNodes { ids } => {
                let node_ids: Vec<u64> = ids
                    .iter()
                    .filter_map(|e| self.expr_to_u64(e).ok())
                    .collect();

                if !node_ids.is_empty() {
                    // Checkpoint protection for batch delete
                    let sample_data: Vec<String> =
                        node_ids.iter().map(|id| format!("node {id}")).collect();
                    let op = DestructiveOp::NodeDelete {
                        node_id: node_ids[0],
                        edge_count: node_ids.len().saturating_sub(1),
                    };

                    match self.protect_destructive_op(
                        &format!("BATCH DELETE NODES ({})", node_ids.len()),
                        op,
                        sample_data,
                    ) {
                        ProtectedOpResult::Proceed => {},
                        ProtectedOpResult::Cancelled => {
                            return Err(RouterError::CheckpointError(
                                "Operation cancelled by user".to_string(),
                            ));
                        },
                    }
                }

                let result = self.graph.batch_delete_nodes(node_ids)?;
                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "DELETE_NODES".to_string(),
                    affected_count: result.count,
                    created_ids: None,
                }))
            },
            GraphBatchOp::DeleteEdges { ids } => {
                let edge_ids: Vec<u64> = ids
                    .iter()
                    .filter_map(|e| self.expr_to_u64(e).ok())
                    .collect();

                if !edge_ids.is_empty() {
                    // Checkpoint protection for batch delete
                    let sample_data: Vec<String> =
                        edge_ids.iter().map(|id| format!("edge {id}")).collect();
                    let op = DestructiveOp::EdgeDelete {
                        edge_id: edge_ids[0],
                    };

                    match self.protect_destructive_op(
                        &format!("BATCH DELETE EDGES ({})", edge_ids.len()),
                        op,
                        sample_data,
                    ) {
                        ProtectedOpResult::Proceed => {},
                        ProtectedOpResult::Cancelled => {
                            return Err(RouterError::CheckpointError(
                                "Operation cancelled by user".to_string(),
                            ));
                        },
                    }
                }

                let result = self.graph.batch_delete_edges(edge_ids)?;
                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "DELETE_EDGES".to_string(),
                    affected_count: result.count,
                    created_ids: None,
                }))
            },
            GraphBatchOp::UpdateNodes { updates } => {
                #[allow(clippy::type_complexity)]
                let update_inputs: Vec<(
                    u64,
                    Option<Vec<String>>,
                    HashMap<String, PropertyValue>,
                )> = updates
                    .iter()
                    .filter_map(|u| {
                        let id = self.expr_to_u64(&u.id).ok()?;
                        let props: HashMap<String, PropertyValue> = u
                            .properties
                            .iter()
                            .map(|p| {
                                let pv = self
                                    .expr_to_property_value(&p.value)
                                    .unwrap_or(PropertyValue::Null);
                                (p.key.name.clone(), pv)
                            })
                            .collect();
                        Some((id, None, props))
                    })
                    .collect();
                let count = self.graph.batch_update_nodes(update_inputs)?;
                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "UPDATE_NODES".to_string(),
                    affected_count: count,
                    created_ids: None,
                }))
            },
        }
    }

    #[allow(clippy::unused_self)] // Method signature for API consistency
    #[allow(clippy::trivially_copy_pass_by_ref)] // API consistency with other direction converters
    const fn convert_parsed_direction(&self, dir: &ParsedDirection) -> Direction {
        match dir {
            ParsedDirection::Outgoing => Direction::Outgoing,
            ParsedDirection::Incoming => Direction::Incoming,
            ParsedDirection::Both => Direction::Both,
        }
    }

    pub(crate) fn exec_node(&self, node: &NodeStmt) -> Result<QueryResult> {
        match &node.operation {
            NodeOp::Create { label, properties } => {
                let props = self.properties_to_map(properties)?;
                let id = self.graph.create_node(&label.name, props)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            NodeOp::Get { id } => {
                let node_id = self.expr_to_u64(id)?;
                let node = self.graph.get_node(node_id)?;
                let properties: HashMap<String, String> = node
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::property_to_string(v)))
                    .collect();
                Ok(QueryResult::Nodes(vec![NodeResult {
                    id: node.id,
                    label: node.labels.join(":"),
                    properties,
                }]))
            },
            NodeOp::Delete { id } => {
                let node_id = self.expr_to_u64(id)?;

                // Collect node info for preview
                let (edge_count, sample_data) = self.collect_node_info(node_id);

                // Check for auto-checkpoint protection
                let op = DestructiveOp::NodeDelete {
                    node_id,
                    edge_count,
                };

                match self.protect_destructive_op(
                    &format!("NODE DELETE {node_id}"),
                    op,
                    sample_data,
                ) {
                    ProtectedOpResult::Proceed => {},
                    ProtectedOpResult::Cancelled => {
                        return Err(RouterError::CheckpointError(
                            "Operation cancelled by user".to_string(),
                        ));
                    },
                }

                self.graph.delete_node(node_id)?;
                Ok(QueryResult::Count(1))
            },
            NodeOp::List {
                label,
                limit,
                offset,
            } => {
                // List all nodes with optional label filter and pagination
                let label_filter = label.as_ref().map(|l| l.name.as_str());
                let limit_val = limit
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(1000);
                let offset_val = offset
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(0);
                let unified_items =
                    self.scan_find_nodes(label_filter, None, limit_val, offset_val)?;

                // Convert UnifiedItem to NodeResult
                let nodes: Vec<NodeResult> = unified_items
                    .into_iter()
                    .map(|item| {
                        let id = item.id.parse::<u64>().unwrap_or(0);
                        let label = item.data.get("label").cloned().unwrap_or_default();
                        let properties: HashMap<String, String> = item
                            .data
                            .into_iter()
                            .filter(|(k, _)| k != "label")
                            .collect();
                        NodeResult {
                            id,
                            label,
                            properties,
                        }
                    })
                    .collect();

                Ok(QueryResult::Nodes(nodes))
            },
        }
    }

    pub(crate) fn exec_edge(&self, edge: &EdgeStmt) -> Result<QueryResult> {
        match &edge.operation {
            EdgeOp::Create {
                from_id,
                to_id,
                edge_type,
                properties,
            } => {
                let from = self.expr_to_u64(from_id)?;
                let to = self.expr_to_u64(to_id)?;
                let props = self.properties_to_map(properties)?;
                let id = self
                    .graph
                    .create_edge(from, to, &edge_type.name, props, true)?;
                Ok(QueryResult::Ids(vec![id]))
            },
            EdgeOp::Get { id } => {
                let edge_id = self.expr_to_u64(id)?;
                let edge = self.graph.get_edge(edge_id)?;
                Ok(QueryResult::Edges(vec![EdgeResult {
                    id: edge.id,
                    from: edge.from,
                    to: edge.to,
                    label: edge.edge_type,
                }]))
            },
            EdgeOp::Delete { id } => {
                let edge_id = self.expr_to_u64(id)?;
                let sample_data = self.collect_edge_info(edge_id);
                let op = DestructiveOp::EdgeDelete { edge_id };

                match self.protect_destructive_op(
                    &format!("EDGE DELETE {edge_id}"),
                    op,
                    sample_data,
                ) {
                    ProtectedOpResult::Proceed => {},
                    ProtectedOpResult::Cancelled => {
                        return Err(RouterError::CheckpointError(
                            "Operation cancelled by user".to_string(),
                        ));
                    },
                }

                self.graph.delete_edge(edge_id)?;
                Ok(QueryResult::Count(1))
            },
            EdgeOp::List {
                edge_type,
                limit,
                offset,
            } => {
                // List all edges with optional type filter and pagination
                let type_filter = edge_type.as_ref().map(|t| t.name.as_str());
                let limit_val = limit
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(1000);
                let offset_val = offset
                    .as_ref()
                    .map(|e| self.expr_to_usize(e))
                    .transpose()?
                    .unwrap_or(0);
                let unified_items =
                    self.scan_find_edges(type_filter, None, limit_val, offset_val)?;

                // Convert UnifiedItem to EdgeResult
                let edges: Vec<EdgeResult> = unified_items
                    .into_iter()
                    .map(|item| {
                        let id = item.id.parse::<u64>().unwrap_or(0);
                        let from = item
                            .data
                            .get("from")
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(0);
                        let to = item
                            .data
                            .get("to")
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(0);
                        let label = item.data.get("type").cloned().unwrap_or_default();
                        EdgeResult {
                            id,
                            from,
                            to,
                            label,
                        }
                    })
                    .collect();

                Ok(QueryResult::Edges(edges))
            },
        }
    }

    pub(crate) fn exec_neighbors(&self, neighbors: &NeighborsStmt) -> Result<QueryResult> {
        // Handle NEIGHBORS...BY SIMILARITY cross-engine query
        if let Some(ref similarity_vec) = neighbors.by_similarity {
            // For BY SIMILARITY queries, node_id should be a string key (entity identifier)
            let entity_key = self.expr_to_string(&neighbors.node_id)?;

            let query: Vec<f32> = similarity_vec
                .iter()
                .map(|e| self.expr_to_f32(e))
                .collect::<Result<_>>()?;

            let top_k = neighbors
                .limit
                .as_ref()
                .map(|e| self.expr_to_usize(e))
                .transpose()?
                .unwrap_or(10);

            // Use the cross-engine find_neighbors_by_similarity method
            let items = self.find_neighbors_by_similarity(&entity_key, &query, top_k)?;

            let results: Vec<SimilarResult> = items
                .into_iter()
                .map(|item| SimilarResult {
                    key: item.id,
                    score: item.score.unwrap_or(0.0),
                })
                .collect();

            return Ok(QueryResult::Similar(results));
        }

        // Standard neighbors query
        let node_id = self.expr_to_u64(&neighbors.node_id)?;

        let direction = match neighbors.direction {
            ParsedDirection::Outgoing => Direction::Outgoing,
            ParsedDirection::Incoming => Direction::Incoming,
            ParsedDirection::Both => Direction::Both,
        };

        let edge_type = neighbors.edge_type.as_ref().map(|e| e.name.as_str());
        let neighbor_nodes = self.graph.neighbors(node_id, edge_type, direction, None)?;
        let neighbor_ids: Vec<u64> = neighbor_nodes.iter().map(|n| n.id).collect();

        Ok(QueryResult::Ids(neighbor_ids))
    }

    pub(crate) fn exec_path(&self, path: &PathStmt) -> Result<QueryResult> {
        let from = self.expr_to_u64(&path.from_id)?;
        let to = self.expr_to_u64(&path.to_id)?;

        match self.graph.find_path(from, to, None) {
            Ok(path) => Ok(QueryResult::Path(path.nodes)),
            Err(GraphError::PathNotFound) => Ok(QueryResult::Path(vec![])),
            Err(e) => Err(e.into()),
        }
    }
}
