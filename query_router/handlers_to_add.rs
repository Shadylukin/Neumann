// SPDX-License-Identifier: MIT OR Apache-2.0
    // ========== Extended Graph Statement Handlers ==========

    fn exec_graph_algorithm(&self, stmt: &GraphAlgorithmStmt) -> Result<QueryResult> {
        use graph_engine::{CentralityConfig, CommunityConfig, PageRankConfig};

        match &stmt.operation {
            GraphAlgorithmOp::PageRank {
                damping,
                tolerance,
                max_iterations,
                ..
            } => {
                let mut config = PageRankConfig::new();
                if let Some(d) = damping {
                    config = config.damping(self.expr_to_float(d)?);
                }
                if let Some(t) = tolerance {
                    config = config.tolerance(self.expr_to_float(t)?);
                }
                if let Some(m) = max_iterations {
                    config = config.max_iterations(self.expr_to_usize(m)?);
                }

                let result = self.graph.pagerank(Some(config))?;
                Ok(QueryResult::PageRank(result))
            }
            GraphAlgorithmOp::BetweennessCentrality { sampling_ratio, .. } => {
                let mut config = CentralityConfig::new();
                if let Some(s) = sampling_ratio {
                    config = config.sampling_ratio(self.expr_to_float(s)?);
                }

                let result = self.graph.betweenness_centrality(Some(config))?;
                Ok(QueryResult::Centrality(result))
            }
            GraphAlgorithmOp::ClosenessCentrality { .. } => {
                let result = self.graph.closeness_centrality(None)?;
                Ok(QueryResult::Centrality(result))
            }
            GraphAlgorithmOp::EigenvectorCentrality {
                max_iterations,
                tolerance,
                ..
            } => {
                let mut config = CentralityConfig::new();
                if let Some(m) = max_iterations {
                    config = config.max_iterations(self.expr_to_usize(m)?);
                }
                if let Some(t) = tolerance {
                    config = config.tolerance(self.expr_to_float(t)?);
                }

                let result = self.graph.eigenvector_centrality(Some(config))?;
                Ok(QueryResult::Centrality(result))
            }
            GraphAlgorithmOp::LouvainCommunities {
                resolution,
                max_passes,
                ..
            } => {
                let mut config = CommunityConfig::default();
                if let Some(r) = resolution {
                    config.resolution = self.expr_to_float(r)?;
                }
                if let Some(p) = max_passes {
                    config.max_passes = self.expr_to_usize(p)?;
                }

                let result = self.graph.louvain_communities(Some(config))?;
                Ok(QueryResult::Community(result))
            }
            GraphAlgorithmOp::LabelPropagation { max_iterations, .. } => {
                let mut config = CommunityConfig::default();
                if let Some(m) = max_iterations {
                    config.max_iterations = self.expr_to_usize(m)?;
                }

                let result = self.graph.label_propagation(Some(config))?;
                Ok(QueryResult::Community(result))
            }
        }
    }

    fn exec_graph_constraint(&self, stmt: &GraphConstraintStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphConstraintOp::Create {
                name,
                target,
                property,
                constraint_type,
            } => {
                let target_enum = match target {
                    ConstraintTargetAst::Node(label) => {
                        graph_engine::ConstraintTarget::Node(label.name.clone())
                    }
                    ConstraintTargetAst::Edge(edge_type) => {
                        graph_engine::ConstraintTarget::Edge(edge_type.name.clone())
                    }
                };
                let ctype = match constraint_type {
                    ConstraintTypeAst::Unique => graph_engine::ConstraintType::Unique,
                    ConstraintTypeAst::Exists => graph_engine::ConstraintType::Exists,
                    ConstraintTypeAst::Type(pt) => {
                        let pt_enum = match pt {
                            neumann_parser::PropertyTypeAst::String => {
                                graph_engine::PropertyValueType::String
                            }
                            neumann_parser::PropertyTypeAst::Integer => {
                                graph_engine::PropertyValueType::Int
                            }
                            neumann_parser::PropertyTypeAst::Float => {
                                graph_engine::PropertyValueType::Float
                            }
                            neumann_parser::PropertyTypeAst::Boolean => {
                                graph_engine::PropertyValueType::Bool
                            }
                        };
                        graph_engine::ConstraintType::Type(pt_enum)
                    }
                };

                let constraint = graph_engine::Constraint {
                    name: name.name.clone(),
                    target: target_enum,
                    property: property.name.clone(),
                    constraint_type: ctype,
                };
                self.graph.create_constraint(constraint)?;
                Ok(QueryResult::Empty)
            }
            GraphConstraintOp::Drop { name } => {
                self.graph.drop_constraint(&name.name)?;
                Ok(QueryResult::Empty)
            }
            GraphConstraintOp::List => {
                let constraints = self.graph.list_constraints();
                let results: Vec<ConstraintInfoResult> = constraints
                    .into_iter()
                    .map(|c| ConstraintInfoResult {
                        name: c.name,
                        target: format!("{:?}", c.target),
                        property: c.property,
                        constraint_type: format!("{:?}", c.constraint_type),
                    })
                    .collect();
                Ok(QueryResult::Constraints(results))
            }
            GraphConstraintOp::Get { name } => {
                if let Some(c) = self.graph.get_constraint(&name.name) {
                    let result = ConstraintInfoResult {
                        name: c.name,
                        target: format!("{:?}", c.target),
                        property: c.property,
                        constraint_type: format!("{:?}", c.constraint_type),
                    };
                    Ok(QueryResult::Constraints(vec![result]))
                } else {
                    Err(RouterError::NotFound(format!(
                        "Constraint '{}' not found",
                        name.name
                    )))
                }
            }
        }
    }

    fn exec_graph_index(&self, stmt: &GraphIndexStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphIndexOp::CreateNodeProperty { property } => {
                self.graph.create_node_property_index(&property.name)?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::CreateEdgeProperty { property } => {
                self.graph.create_edge_property_index(&property.name)?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::CreateLabel => {
                self.graph.create_label_index()?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::CreateEdgeType => {
                self.graph.create_edge_type_index()?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::DropNode { property } => {
                self.graph.drop_node_index(&property.name)?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::DropEdge { property } => {
                self.graph.drop_edge_index(&property.name)?;
                Ok(QueryResult::Empty)
            }
            GraphIndexOp::ShowNodeIndexes => {
                let props = self.graph.get_indexed_node_properties();
                Ok(QueryResult::GraphIndexes(props))
            }
            GraphIndexOp::ShowEdgeIndexes => {
                let props = self.graph.get_indexed_edge_properties();
                Ok(QueryResult::GraphIndexes(props))
            }
        }
    }

    fn exec_graph_aggregate(&self, stmt: &GraphAggregateStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphAggregateOp::CountNodes { label } => {
                let count = match label {
                    Some(l) => self.graph.count_nodes_by_label(&l.name)?,
                    None => self.graph.count_nodes(),
                };
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "count_nodes".to_string(),
                    value: GraphAggregateValue::Count(count),
                }))
            }
            GraphAggregateOp::CountEdges { edge_type } => {
                let count = match edge_type {
                    Some(t) => self.graph.count_edges_by_type(&t.name)?,
                    None => self.graph.count_edges(),
                };
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "count_edges".to_string(),
                    value: GraphAggregateValue::Count(count),
                }))
            }
            GraphAggregateOp::SumNodeProperty { property, .. } => {
                let sum = self.graph.sum_node_property(&property.name);
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "sum".to_string(),
                    value: GraphAggregateValue::Float(sum.unwrap_or(0.0)),
                }))
            }
            GraphAggregateOp::AvgNodeProperty { property, .. } => {
                let avg = self.graph.avg_node_property(&property.name);
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "avg".to_string(),
                    value: GraphAggregateValue::Float(avg.unwrap_or(f64::NAN)),
                }))
            }
            GraphAggregateOp::SumEdgeProperty { property, .. } => {
                let sum = self.graph.sum_edge_property(&property.name);
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "sum".to_string(),
                    value: GraphAggregateValue::Float(sum.unwrap_or(0.0)),
                }))
            }
            GraphAggregateOp::AvgEdgeProperty { property, .. } => {
                let avg = self.graph.avg_edge_property(&property.name);
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "avg".to_string(),
                    value: GraphAggregateValue::Float(avg.unwrap_or(f64::NAN)),
                }))
            }
        }
    }

    fn exec_graph_pattern(&self, stmt: &GraphPatternStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphPatternOp::Match { pattern, limit } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, limit)?;
                let result = self.graph.match_pattern(&gp)?;
                Ok(QueryResult::PatternMatch(result))
            }
            GraphPatternOp::Count { pattern } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, &None)?;
                let count = self.graph.count_pattern_matches(&gp)?;
                Ok(QueryResult::GraphAggregate(GraphAggregateResult {
                    operation: "pattern_count".to_string(),
                    value: GraphAggregateValue::Count(count),
                }))
            }
            GraphPatternOp::Exists { pattern } => {
                let gp = self.pattern_spec_to_graph_pattern(pattern, &None)?;
                let exists = self.graph.pattern_exists(&gp)?;
                Ok(QueryResult::Value(exists.to_string()))
            }
        }
    }

    fn exec_graph_batch(&self, stmt: &GraphBatchStmt) -> Result<QueryResult> {
        match &stmt.operation {
            GraphBatchOp::CreateNodes { nodes } => {
                let node_inputs: Vec<graph_engine::NodeInput> = nodes
                    .iter()
                    .map(|n| {
                        let props: HashMap<String, graph_engine::PropertyValue> = n
                            .properties
                            .iter()
                            .map(|(k, v)| (k.clone(), self.expr_to_property_value(v)))
                            .collect();
                        graph_engine::NodeInput::new(
                            n.labels.iter().map(|l| l.name.clone()).collect(),
                            props,
                        )
                    })
                    .collect();
                let result = self.graph.batch_create_nodes(node_inputs)?;
                Ok(QueryResult::GraphBatch(GraphBatchResult {
                    operation: "create_nodes".to_string(),
                    affected_count: result.count,
                    created_ids: Some(result.created_ids),
                }))
            }
            GraphBatchOp::CreateEdges { edges } => {
                let edge_inputs: Vec<graph_engine::EdgeInput> = edges
                    .iter()
                    .filter_map(|e| {
                        let from = self.expr_to_u64(&e.from_id).ok()?;
                        let to = self.expr_to_u64(&e.to_id).ok()?;
                        let props: HashMap<String, graph_engine::PropertyValue> = e
                            .properties
                            .iter()
                            .map(|(k, v)| (k.clone(), self.expr_to_property_value(v)))
                            .collect();
                        Some(graph_engine::EdgeInput::new(
                            from,
                            to,
                            e.edge_type.name.clone(),
                            props,
                            true, // directed by default
                        ))
                    })
                    .collect();
                let result = self.graph.batch_create_edges(edge_inputs)?;
                Ok(QueryResult::GraphBatch(GraphBatchResult {
                    operation: "create_edges".to_string(),
                    affected_count: result.count,
                    created_ids: Some(result.created_ids),
                }))
            }
            GraphBatchOp::DeleteNodes { ids } => {
                let node_ids: Vec<u64> = ids
                    .iter()
                    .filter_map(|e| self.expr_to_u64(e).ok())
                    .collect();
                let result = self.graph.batch_delete_nodes(node_ids)?;
                Ok(QueryResult::GraphBatch(GraphBatchResult {
                    operation: "delete_nodes".to_string(),
                    affected_count: result.count,
                    created_ids: None,
                }))
            }
            GraphBatchOp::DeleteEdges { ids } => {
                let edge_ids: Vec<u64> = ids
                    .iter()
                    .filter_map(|e| self.expr_to_u64(e).ok())
                    .collect();
                let result = self.graph.batch_delete_edges(edge_ids)?;
                Ok(QueryResult::GraphBatch(GraphBatchResult {
                    operation: "delete_edges".to_string(),
                    affected_count: result.count,
                    created_ids: None,
                }))
            }
            GraphBatchOp::UpdateNodes { updates } => {
                let update_tuples: Vec<(u64, Option<Vec<String>>, HashMap<String, graph_engine::PropertyValue>)> = updates
                    .iter()
                    .filter_map(|u| {
                        let id = self.expr_to_u64(&u.id).ok()?;
                        let props: HashMap<String, graph_engine::PropertyValue> = u
                            .properties
                            .iter()
                            .map(|(k, v)| (k.clone(), self.expr_to_property_value(v)))
                            .collect();
                        Some((id, None, props))
                    })
                    .collect();
                let updated = self.graph.batch_update_nodes(update_tuples)?;
                Ok(QueryResult::GraphBatch(GraphBatchResult {
                    operation: "update_nodes".to_string(),
                    affected_count: updated,
                    created_ids: None,
                }))
            }
        }
    }

    fn expr_to_float(&self, expr: &Expr) -> Result<f64> {
        match &expr.kind {
            ExprKind::Literal(neumann_parser::Literal::Integer(i)) => Ok(*i as f64),
            ExprKind::Literal(neumann_parser::Literal::Float(f)) => Ok(*f),
            _ => Err(RouterError::InvalidArgument(
                "Expected numeric literal".to_string(),
            )),
        }
    }

    fn expr_to_usize(&self, expr: &Expr) -> Result<usize> {
        match &expr.kind {
            ExprKind::Literal(neumann_parser::Literal::Integer(i)) => {
                Ok(*i as usize)
            }
            _ => Err(RouterError::InvalidArgument(
                "Expected integer literal".to_string(),
            )),
        }
    }

    fn expr_to_u64(&self, expr: &Expr) -> Result<u64> {
        match &expr.kind {
            ExprKind::Literal(neumann_parser::Literal::Integer(i)) => Ok(*i as u64),
            _ => Err(RouterError::InvalidArgument(
                "Expected integer literal".to_string(),
            )),
        }
    }

    fn expr_to_property_value(&self, expr: &Expr) -> graph_engine::PropertyValue {
        match &expr.kind {
            ExprKind::Literal(neumann_parser::Literal::Integer(i)) => {
                graph_engine::PropertyValue::Int(*i)
            }
            ExprKind::Literal(neumann_parser::Literal::Float(f)) => {
                graph_engine::PropertyValue::Float(*f)
            }
            ExprKind::Literal(neumann_parser::Literal::String(s)) => {
                graph_engine::PropertyValue::String(s.clone())
            }
            ExprKind::Literal(neumann_parser::Literal::Boolean(b)) => {
                graph_engine::PropertyValue::Bool(*b)
            }
            _ => graph_engine::PropertyValue::String(format!("{:?}", expr)),
        }
    }

    fn pattern_spec_to_graph_pattern(
        &self,
        pattern: &PatternSpec,
        limit: &Option<Expr>,
    ) -> Result<graph_engine::Pattern> {
        // Build the graph pattern from AST PatternSpec
        let mut path = graph_engine::PathPattern::new();

        // Start node
        let start_node = graph_engine::NodePattern::new()
            .variable(pattern.start.variable.as_ref().map(|v| v.name.clone()))
            .label(pattern.start.label.as_ref().map(|l| l.name.clone()));
        path = path.node(start_node);

        // Subsequent edges and nodes
        for elem in &pattern.elements {
            let edge = graph_engine::EdgePattern::new()
                .variable(elem.edge.variable.as_ref().map(|v| v.name.clone()))
                .edge_type(elem.edge.edge_type.as_ref().map(|t| t.name.clone()))
                .direction(match elem.edge.direction {
                    neumann_parser::Direction::Outgoing => graph_engine::Direction::Outgoing,
                    neumann_parser::Direction::Incoming => graph_engine::Direction::Incoming,
                    neumann_parser::Direction::Both => graph_engine::Direction::Both,
                });

            let node = graph_engine::NodePattern::new()
                .variable(elem.node.variable.as_ref().map(|v| v.name.clone()))
                .label(elem.node.label.as_ref().map(|l| l.name.clone()));

            path = path.edge(edge).node(node);
        }

        let mut gp = graph_engine::Pattern::new(path);
        if let Some(lim) = limit {
            gp = gp.limit(self.expr_to_usize(lim)?);
        }

        Ok(gp)
    }

