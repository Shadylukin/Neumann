// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified cross-engine FIND / ENTITY statement execution plus helpers
//! for connecting entities and creating embedded entities.

#![allow(
    clippy::too_many_lines,
    reason = "match dispatchers cover many sub-ops"
)]

use std::collections::HashMap;

use neumann_parser::{EntityOp, EntityStmt, FindPattern, FindStmt};
use relational_engine::Condition;
use tensor_unified::{FindPattern as UnifiedFindPattern, UnifiedItem};
use tokio::runtime::Runtime;

use crate::result::{BatchOperationResult, UnifiedResult};
use crate::{QueryResult, QueryRouter, Result, RouterError};

impl QueryRouter {
    /// Store a unified entity with relational, graph, and vector data.
    ///
    /// Delegates to `UnifiedEngine::create_entity()`.
    ///
    /// # Errors
    ///
    /// Returns an error if unified engine is not initialized or entity creation fails.
    pub fn create_unified_entity(
        &self,
        key: &str,
        fields: HashMap<String, String>,
        embedding: Option<Vec<f32>>,
    ) -> Result<()> {
        let has_embedding = embedding.is_some();
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        runtime
            .block_on(unified.create_entity(key, fields, embedding))
            .map_err(|e| RouterError::VectorError(e.to_string()))?;

        if has_embedding {
            self.bump_vector_generation();
        }
        Ok(())
    }

    /// Connect two entities with an edge.
    ///
    /// Delegates to `UnifiedEngine::connect_entities()`.
    ///
    /// # Errors
    ///
    /// Returns an error if unified engine is not initialized or connection fails.
    pub fn connect_entities(
        &self,
        from_key: &str,
        to_key: &str,
        edge_type: &str,
    ) -> Result<String> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        runtime
            .block_on(unified.connect_entities(from_key, to_key, edge_type))
            .map(|edge_id| format!("edge:{edge_type}:{edge_id}"))
            .map_err(|e| RouterError::GraphError(e.to_string()))
    }

    pub(crate) fn exec_find(&self, find: &FindStmt) -> Result<QueryResult> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        let limit = find
            .limit
            .as_ref()
            .map(|e| self.expr_to_usize(e))
            .transpose()?;

        let has_similar = find.similar_to.is_some();
        let has_connected = find.connected_to.is_some();

        // SIMILAR TO and CONNECTED TO are only supported with FIND NODE
        if (has_similar || has_connected) && !matches!(find.pattern, FindPattern::Nodes { .. }) {
            return Err(RouterError::InvalidArgument(
                "SIMILAR TO and CONNECTED TO are only supported with FIND NODE".to_string(),
            ));
        }

        // Cross-engine path: delegate to find_nodes_hybrid
        if has_similar || has_connected {
            let FindPattern::Nodes { ref label } = find.pattern else {
                unreachable!()
            };

            let condition = find
                .where_clause
                .as_ref()
                .map(|expr| self.expr_to_condition(expr))
                .transpose()?;

            let similar_key = find
                .similar_to
                .as_ref()
                .map(|e| self.expr_to_string(e))
                .transpose()?;

            let connected_key = find
                .connected_to
                .as_ref()
                .map(|e| self.expr_to_string(e))
                .transpose()?;

            let effective_limit = limit.unwrap_or(100);
            let label_filter = label.as_ref().map(|l| l.name.as_str());

            let items = runtime
                .block_on(unified.find_nodes_hybrid(
                    label_filter,
                    condition.as_ref(),
                    similar_key.as_deref(),
                    connected_key.as_deref(),
                    effective_limit,
                ))
                .map_err(|e| RouterError::GraphError(e.to_string()))?;

            let description = format!(
                "Found {} node{}",
                items.len(),
                if items.len() == 1 { "" } else { "s" }
            );

            return Ok(QueryResult::Unified(UnifiedResult { description, items }));
        }

        let pattern = self.convert_find_pattern(&find.pattern);

        // Handle WHERE clause by using find_nodes/find_edges/find_rows with condition
        if let Some(ref where_expr) = find.where_clause {
            let condition = self.expr_to_condition(where_expr)?;
            let effective_limit = limit.unwrap_or(100);

            let (items, entity_type) = match &find.pattern {
                FindPattern::Nodes { label } => {
                    let label_filter = label.as_ref().map(|l| l.name.as_str());
                    let items =
                        self.scan_find_nodes(label_filter, Some(&condition), effective_limit, 0)?;
                    (items, "node")
                },
                FindPattern::Edges { edge_type } => {
                    let type_filter = edge_type.as_ref().map(|t| t.name.as_str());
                    let items =
                        self.scan_find_edges(type_filter, Some(&condition), effective_limit, 0)?;
                    (items, "edge")
                },
                FindPattern::Rows { table } => {
                    let items =
                        self.scan_find_rows(&table.name, Some(&condition), effective_limit)?;
                    (items, "row")
                },
                FindPattern::Path { .. } => (Vec::new(), "path"),
            };

            let description = format!(
                "Found {} {}{}",
                items.len(),
                entity_type,
                if items.len() == 1 { "" } else { "s" }
            );
            return Ok(QueryResult::Unified(UnifiedResult { description, items }));
        }

        // No WHERE clause - delegate to unified.find()
        let result = runtime
            .block_on(unified.find(&pattern, limit))
            .map_err(|e| RouterError::GraphError(e.to_string()))?;

        Ok(QueryResult::Unified(UnifiedResult {
            description: result.description,
            items: result.items,
        }))
    }

    #[allow(clippy::too_many_lines)] // Entity operations require handling create, get, update, delete, and link
    pub(crate) fn exec_entity(&self, entity: &EntityStmt) -> Result<QueryResult> {
        match &entity.operation {
            EntityOp::Create {
                key,
                properties,
                embedding,
            } => {
                let key_str = self.expr_to_string(key)?;

                // Convert properties to HashMap
                let fields: HashMap<String, String> = properties
                    .iter()
                    .filter_map(|p| {
                        self.expr_to_string(&p.value)
                            .ok()
                            .map(|v| (p.key.name.clone(), v))
                    })
                    .collect();

                // Convert embedding if present
                let emb = if let Some(vec_exprs) = embedding {
                    let embedding_vec: Result<Vec<f32>> =
                        vec_exprs.iter().map(|e| self.expr_to_f32(e)).collect();
                    Some(embedding_vec?)
                } else {
                    None
                };

                // Use the existing create_unified_entity method
                // (create_unified_entity already bumps vector_generation if embedding is present)
                self.create_unified_entity(&key_str, fields, emb)?;

                Ok(QueryResult::Value(format!("Entity '{key_str}' created")))
            },
            EntityOp::Get { key } => {
                let key_str = self.expr_to_string(key)?;

                // Try to get from unified engine if available
                if let Some(ref unified) = self.unified {
                    let runtime =
                        Runtime::new().map_err(|e| RouterError::InvalidArgument(e.to_string()))?;

                    let item = runtime
                        .block_on(unified.get_entity(&key_str))
                        .map_err(|e| RouterError::NotFound(e.to_string()))?;

                    return Ok(QueryResult::Unified(UnifiedResult {
                        description: format!("Entity: {key_str}"),
                        items: vec![item],
                    }));
                }

                // Fall back to looking up data directly
                let mut data = HashMap::new();
                data.insert("key".to_string(), key_str.clone());

                // Try to find in vector store
                if let Ok(embedding) = self.vector.get_embedding(&key_str) {
                    let item = UnifiedItem {
                        id: key_str.clone(),
                        source: "vector".to_string(),
                        data,
                        embedding: Some(embedding),
                        score: None,
                    };
                    return Ok(QueryResult::Unified(UnifiedResult {
                        description: format!("Entity: {key_str}"),
                        items: vec![item],
                    }));
                }

                Err(RouterError::NotFound(format!(
                    "Entity '{key_str}' not found"
                )))
            },
            EntityOp::Connect {
                from_key,
                to_key,
                edge_type,
            } => {
                let from_str = self.expr_to_string(from_key)?;
                let to_str = self.expr_to_string(to_key)?;
                let edge_type_str = &edge_type.name;

                // Use the existing connect_entities method
                let edge_key = self.connect_entities(&from_str, &to_str, edge_type_str)?;

                Ok(QueryResult::Value(format!(
                    "Connected '{from_str}' -> '{to_str}' with edge '{edge_key}'"
                )))
            },
            EntityOp::Batch { entities } => {
                #[allow(clippy::type_complexity)]
                let items: Vec<(
                    String,
                    HashMap<String, String>,
                    Option<Vec<f32>>,
                )> = entities
                    .iter()
                    .map(|e| {
                        let key = self.expr_to_string(&e.key)?;
                        let props: HashMap<String, String> = e
                            .properties
                            .iter()
                            .filter_map(|p| {
                                self.expr_to_string(&p.value)
                                    .ok()
                                    .map(|v| (p.key.name.clone(), v))
                            })
                            .collect();
                        let emb = e
                            .embedding
                            .as_ref()
                            .map(|v| v.iter().map(|ex| self.expr_to_f32(ex)).collect())
                            .transpose()?;
                        Ok((key, props, emb))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let has_embeddings = items.iter().any(|(_, _, emb)| emb.is_some());
                let unified = self.require_unified()?;
                let runtime = Self::create_runtime()?;
                let batch_result = runtime
                    .block_on(unified.create_entities_batch(items))
                    .map_err(|e| RouterError::VectorError(e.to_string()))?;

                if has_embeddings {
                    self.bump_vector_generation();
                }

                Ok(QueryResult::BatchResult(BatchOperationResult {
                    operation: "ENTITY CREATE".to_string(),
                    affected_count: batch_result.count,
                    created_ids: None, // Entities use string keys, not numeric IDs
                }))
            },
            EntityOp::Update {
                key,
                properties,
                embedding,
            } => {
                let key_str = self.expr_to_string(key)?;

                // Convert properties to HashMap
                let fields: HashMap<String, String> = properties
                    .iter()
                    .filter_map(|p| {
                        self.expr_to_string(&p.value)
                            .ok()
                            .map(|v| (p.key.name.clone(), v))
                    })
                    .collect();

                // Convert embedding if present
                let emb = if let Some(vec_exprs) = embedding {
                    let embedding_vec: Result<Vec<f32>> =
                        vec_exprs.iter().map(|e| self.expr_to_f32(e)).collect();
                    Some(embedding_vec?)
                } else {
                    None
                };

                let has_embedding = emb.is_some();
                let unified = self.require_unified()?;
                let runtime = Self::create_runtime()?;
                runtime
                    .block_on(unified.update_entity(&key_str, fields, emb))
                    .map_err(|e| RouterError::NotFound(e.to_string()))?;

                if has_embedding {
                    self.bump_vector_generation();
                }

                Ok(QueryResult::Value(format!("Entity '{key_str}' updated")))
            },
            EntityOp::Delete { key } => {
                let key_str = self.expr_to_string(key)?;

                let unified = self.require_unified()?;
                let runtime = Self::create_runtime()?;
                runtime
                    .block_on(unified.delete_entity(&key_str))
                    .map_err(|e| RouterError::NotFound(e.to_string()))?;

                self.bump_vector_generation();

                Ok(QueryResult::Value(format!("Entity '{key_str}' deleted")))
            },
        }
    }

    /// Delegates to `UnifiedEngine::find_nodes()`.
    pub(crate) fn scan_find_nodes(
        &self,
        label_filter: Option<&str>,
        condition: Option<&Condition>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        let items = runtime
            .block_on(unified.find_nodes(label_filter, condition))
            .map_err(|e| RouterError::GraphError(e.to_string()))?;

        let items: Vec<UnifiedItem> = items.into_iter().skip(offset).take(limit).collect();
        Ok(items)
    }

    /// Delegates to `UnifiedEngine::find_edges()`.
    pub(crate) fn scan_find_edges(
        &self,
        type_filter: Option<&str>,
        condition: Option<&Condition>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        let items = runtime
            .block_on(unified.find_edges(type_filter, condition))
            .map_err(|e| RouterError::GraphError(e.to_string()))?;

        let items: Vec<UnifiedItem> = items.into_iter().skip(offset).take(limit).collect();
        Ok(items)
    }

    /// Delegates to `UnifiedEngine::find_rows()`.
    pub(crate) fn scan_find_rows(
        &self,
        table: &str,
        condition: Option<&Condition>,
        limit: usize,
    ) -> Result<Vec<UnifiedItem>> {
        let unified = self.require_unified()?;
        let runtime = Self::create_runtime()?;

        let mut items = runtime
            .block_on(unified.find_rows(table, condition))
            .map_err(|e| RouterError::RelationalError(e.to_string()))?;

        items.truncate(limit);
        Ok(items)
    }

    /// Converts AST `FindPattern` to unified `FindPattern`.
    #[allow(clippy::unused_self)] // Method signature for API consistency
    pub(crate) fn convert_find_pattern(&self, ast_pattern: &FindPattern) -> UnifiedFindPattern {
        match ast_pattern {
            FindPattern::Nodes { label } => UnifiedFindPattern::Nodes {
                label: label.as_ref().map(|l| l.name.clone()),
            },
            FindPattern::Edges { edge_type } => UnifiedFindPattern::Edges {
                edge_type: edge_type.as_ref().map(|t| t.name.clone()),
            },
            FindPattern::Rows { table } => UnifiedFindPattern::Rows {
                table: table.name.clone(),
            },
            FindPattern::Path { from, edge, to } => UnifiedFindPattern::Path {
                from: from.as_ref().map(|f| f.name.clone()),
                edge: edge.as_ref().map(|e| e.name.clone()),
                to: to.as_ref().map(|t| t.name.clone()),
            },
        }
    }
}
