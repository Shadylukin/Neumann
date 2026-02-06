//! Cypher query execution for the query router.
//!
//! This module provides execution logic for Cypher-style graph queries:
//! - MATCH: Pattern matching with WHERE filtering and RETURN projection
//! - CREATE: Node and relationship creation
//! - DELETE: Node and relationship deletion (with optional DETACH)
//! - MERGE: Upsert operations

use std::collections::HashMap;

use graph_engine::{Direction, GraphEngine, Node, PropertyValue};
use neumann_parser::{
    CypherCreateStmt, CypherDeleteStmt, CypherDirection, CypherElement, CypherMatchStmt,
    CypherMergeStmt, CypherNode, CypherPattern, CypherReturn, Expr, ExprKind, Literal,
};

use crate::{NodeResult, QueryResult, Result, RouterError};

/// Binding context for pattern matching - maps variable names to matched entities.
#[derive(Debug, Clone, Default)]
pub struct BindingContext {
    /// Node variable bindings: variable name -> node ID
    nodes: HashMap<String, u64>,
    /// Relationship variable bindings: variable name -> edge ID
    edges: HashMap<String, u64>,
}

impl BindingContext {
    fn bind_node(&mut self, var: &str, id: u64) {
        self.nodes.insert(var.to_string(), id);
    }

    fn bind_edge(&mut self, var: &str, id: u64) {
        self.edges.insert(var.to_string(), id);
    }

    fn get_node(&self, var: &str) -> Option<u64> {
        self.nodes.get(var).copied()
    }
}

/// Execute a Cypher MATCH statement.
///
/// # Errors
///
/// Returns an error if pattern matching or projection fails.
pub fn exec_cypher_match(graph: &GraphEngine, stmt: &CypherMatchStmt) -> Result<QueryResult> {
    // For each pattern, find all matching subgraphs
    let mut all_bindings: Vec<BindingContext> = vec![BindingContext::default()];

    for pattern in &stmt.patterns {
        all_bindings = match_pattern(graph, pattern, all_bindings)?;
    }

    // Apply WHERE clause filter
    if let Some(ref where_expr) = stmt.where_clause {
        all_bindings.retain(|ctx| evaluate_where_clause(graph, where_expr, ctx).unwrap_or(false));
    }

    // Project results according to RETURN clause
    let results = project_return(graph, &stmt.return_clause, &all_bindings);

    // Apply ORDER BY, SKIP, LIMIT if present
    let mut results = results;

    // Apply SKIP
    if let Some(ref skip_expr) = stmt.skip {
        if let Some(skip_val) = eval_int_expr(skip_expr) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // skip_val.max(0) ensures non-negative; truncation acceptable for pagination
            let skip = skip_val.max(0) as usize;
            if skip < results.len() {
                results = results.into_iter().skip(skip).collect();
            } else {
                results.clear();
            }
        }
    }

    // Apply LIMIT
    if let Some(ref limit_expr) = stmt.limit {
        if let Some(limit_val) = eval_int_expr(limit_expr) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // limit_val.max(0) ensures non-negative; truncation acceptable for pagination
            let limit = limit_val.max(0) as usize;
            results.truncate(limit);
        }
    }

    Ok(QueryResult::Nodes(results))
}

/// Execute a Cypher CREATE statement.
///
/// # Errors
///
/// Returns an error if node or relationship creation fails.
pub fn exec_cypher_create(graph: &GraphEngine, stmt: &CypherCreateStmt) -> Result<QueryResult> {
    let mut created_ids = Vec::new();
    let mut ctx = BindingContext::default();

    for pattern in &stmt.patterns {
        create_pattern(graph, pattern, &mut ctx, &mut created_ids)?;
    }

    Ok(QueryResult::Ids(created_ids))
}

/// Execute a Cypher DELETE statement.
///
/// # Errors
///
/// Returns an error if node or edge deletion fails.
pub fn exec_cypher_delete(graph: &GraphEngine, stmt: &CypherDeleteStmt) -> Result<QueryResult> {
    let mut deleted_count = 0;

    for var_expr in &stmt.variables {
        // The variable should be an identifier that we can resolve
        if let ExprKind::Ident(ref ident) = var_expr.kind {
            // For now, we treat the identifier as a node ID directly
            // In a full implementation, we'd look up from a binding context
            if let Ok(id) = ident.name.parse::<u64>() {
                if stmt.detach {
                    // Delete all edges connected to this node first
                    let outgoing = graph.neighbors(id, None, Direction::Outgoing, None)?;
                    for neighbor in outgoing {
                        // Delete outgoing edges
                        if let Ok(edges) = graph
                            .find_edges_by_property("from", &PropertyValue::Int(id.cast_signed()))
                        {
                            for edge in edges {
                                if edge.to == neighbor.id {
                                    let _ = graph.delete_edge(edge.id);
                                    deleted_count += 1;
                                }
                            }
                        }
                    }
                    let incoming = graph.neighbors(id, None, Direction::Incoming, None)?;
                    for neighbor in incoming {
                        // Delete incoming edges
                        if let Ok(edges) = graph
                            .find_edges_by_property("to", &PropertyValue::Int(id.cast_signed()))
                        {
                            for edge in edges {
                                if edge.from == neighbor.id {
                                    let _ = graph.delete_edge(edge.id);
                                    deleted_count += 1;
                                }
                            }
                        }
                    }
                }
                graph.delete_node(id)?;
                deleted_count += 1;
            }
        }
    }

    Ok(QueryResult::Count(deleted_count))
}

/// Execute a Cypher MERGE statement.
///
/// # Errors
///
/// Returns an error if pattern matching, creation, or SET operations fail.
pub fn exec_cypher_merge(graph: &GraphEngine, stmt: &CypherMergeStmt) -> Result<QueryResult> {
    // Try to match the pattern first
    let bindings = match_pattern(graph, &stmt.pattern, vec![BindingContext::default()])?;

    if bindings.is_empty() {
        // Pattern not found - create it and apply ON CREATE SET
        let mut ctx = BindingContext::default();
        let mut created_ids = Vec::new();
        create_pattern(graph, &stmt.pattern, &mut ctx, &mut created_ids)?;

        // Apply ON CREATE SET items
        for set_item in &stmt.on_create {
            apply_set_item(graph, set_item, &ctx)?;
        }

        Ok(QueryResult::Ids(created_ids))
    } else {
        // Pattern found - apply ON MATCH SET to each binding
        let mut updated_ids = Vec::new();
        for ctx in &bindings {
            for set_item in &stmt.on_match {
                apply_set_item(graph, set_item, ctx)?;
            }
            // Collect matched node IDs
            updated_ids.extend(ctx.nodes.values().copied());
        }

        Ok(QueryResult::Ids(updated_ids))
    }
}

// =============================================================================
// Pattern Matching
// =============================================================================

fn match_pattern(
    graph: &GraphEngine,
    pattern: &CypherPattern,
    existing_bindings: Vec<BindingContext>,
) -> Result<Vec<BindingContext>> {
    let mut result_bindings = Vec::new();

    for mut ctx in existing_bindings {
        let pattern_bindings = match_pattern_elements(graph, &pattern.elements, &mut ctx)?;
        result_bindings.extend(pattern_bindings);
    }

    Ok(result_bindings)
}

fn match_pattern_elements(
    graph: &GraphEngine,
    elements: &[CypherElement],
    ctx: &mut BindingContext,
) -> Result<Vec<BindingContext>> {
    if elements.is_empty() {
        return Ok(vec![ctx.clone()]);
    }

    let mut results = Vec::new();

    // First element must be a node
    if let Some(CypherElement::Node(node)) = elements.first() {
        let matching_nodes = find_matching_nodes(graph, node, ctx)?;

        for matched_node in matching_nodes {
            let mut new_ctx = ctx.clone();
            if let Some(ref var) = node.variable {
                new_ctx.bind_node(&var.name, matched_node.id);
            }

            // If there are more elements (relationship + node), continue matching
            if elements.len() > 1 {
                let remaining = &elements[1..];
                let sub_results =
                    match_relationship_chain(graph, matched_node.id, remaining, &mut new_ctx)?;
                results.extend(sub_results);
            } else {
                results.push(new_ctx);
            }
        }
    }

    Ok(results)
}

fn match_relationship_chain(
    graph: &GraphEngine,
    current_node_id: u64,
    elements: &[CypherElement],
    ctx: &mut BindingContext,
) -> Result<Vec<BindingContext>> {
    if elements.len() < 2 {
        return Ok(vec![ctx.clone()]);
    }

    let mut results = Vec::new();

    // elements[0] should be a relationship, elements[1] should be a node
    if let (CypherElement::Rel(rel), CypherElement::Node(target_node)) =
        (&elements[0], &elements[1])
    {
        let direction = cypher_direction_to_engine(rel.direction);
        let edge_type = rel.rel_types.first().map(|t| t.name.as_str());

        // Handle variable-length relationships
        let (min_hops, max_hops) = match &rel.var_length {
            Some(vl) => (vl.min.unwrap_or(1), vl.max.unwrap_or(10)),
            None => (1, 1),
        };

        // Find neighbors within the hop range
        let neighbors = find_neighbors_in_range(
            graph,
            current_node_id,
            edge_type,
            direction,
            min_hops,
            max_hops,
        )?;

        for (neighbor_id, edge_id) in neighbors {
            // Check if the neighbor matches the target node pattern
            if let Ok(neighbor) = graph.get_node(neighbor_id) {
                if node_matches_pattern(&neighbor, target_node) {
                    let mut new_ctx = ctx.clone();

                    if let Some(ref var) = rel.variable {
                        new_ctx.bind_edge(&var.name, edge_id);
                    }
                    if let Some(ref var) = target_node.variable {
                        new_ctx.bind_node(&var.name, neighbor_id);
                    }

                    // Continue with remaining elements
                    if elements.len() > 2 {
                        let remaining = &elements[2..];
                        let sub_results =
                            match_relationship_chain(graph, neighbor_id, remaining, &mut new_ctx)?;
                        results.extend(sub_results);
                    } else {
                        results.push(new_ctx);
                    }
                }
            }
        }
    }

    Ok(results)
}

fn find_matching_nodes(
    graph: &GraphEngine,
    pattern: &CypherNode,
    ctx: &BindingContext,
) -> Result<Vec<Node>> {
    // Check if variable is already bound
    if let Some(ref var) = pattern.variable {
        if let Some(bound_id) = ctx.get_node(&var.name) {
            if let Ok(node) = graph.get_node(bound_id) {
                if node_matches_pattern(&node, pattern) {
                    return Ok(vec![node]);
                }
            }
            return Ok(vec![]);
        }
    }

    // Find nodes by label
    if let Some(label) = pattern.labels.first() {
        let nodes = graph.find_nodes_by_label(&label.name)?;
        let filtered: Vec<Node> = nodes
            .into_iter()
            .filter(|n| node_matches_pattern(n, pattern))
            .collect();
        Ok(filtered)
    } else {
        // No label specified - this is expensive, return empty for safety
        // A full implementation would scan all nodes
        Ok(Vec::new())
    }
}

fn node_matches_pattern(node: &Node, pattern: &CypherNode) -> bool {
    // Check labels - Node has labels: Vec<String>
    for label in &pattern.labels {
        if !node.labels.contains(&label.name) {
            return false;
        }
    }

    // Check inline properties
    for prop in &pattern.properties {
        if let Some(node_val) = node.properties.get(&prop.key.name) {
            if let Some(pattern_val) = expr_to_property_value(&prop.value) {
                if *node_val != pattern_val {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    true
}

fn find_neighbors_in_range(
    graph: &GraphEngine,
    start: u64,
    edge_type: Option<&str>,
    direction: Direction,
    min_hops: u32,
    max_hops: u32,
) -> Result<Vec<(u64, u64)>> {
    let mut results = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut frontier: Vec<(u64, u32, u64)> = vec![(start, 0, 0)]; // (node_id, depth, edge_id)

    while let Some((current, depth, edge_id)) = frontier.pop() {
        if depth >= min_hops && depth <= max_hops && current != start {
            results.push((current, edge_id));
        }

        if depth < max_hops && visited.insert(current) {
            let neighbors = graph.neighbors(current, edge_type, direction, None)?;
            for neighbor in neighbors {
                // For simplicity, we use 0 as edge ID - a full impl would track actual edge IDs
                frontier.push((neighbor.id, depth + 1, 0));
            }
        }
    }

    Ok(results)
}

// =============================================================================
// Pattern Creation
// =============================================================================

fn create_pattern(
    graph: &GraphEngine,
    pattern: &CypherPattern,
    ctx: &mut BindingContext,
    created_ids: &mut Vec<u64>,
) -> Result<()> {
    let mut prev_node_id: Option<u64> = None;

    for element in &pattern.elements {
        match element {
            CypherElement::Node(node) => {
                // Check if already bound
                let node_id = if let Some(ref var) = node.variable {
                    if let Some(existing) = ctx.get_node(&var.name) {
                        existing
                    } else {
                        let id = create_node(graph, node)?;
                        ctx.bind_node(&var.name, id);
                        created_ids.push(id);
                        id
                    }
                } else {
                    let id = create_node(graph, node)?;
                    created_ids.push(id);
                    id
                };
                prev_node_id = Some(node_id);
            },
            CypherElement::Rel(rel) => {
                // Will be created when we see the next node
                // Store rel info for the next iteration
                if let Some(from_id) = prev_node_id {
                    // We need the next node, so we peek ahead
                    // For now, just skip - the edge will be created by looking back
                    let _ = (from_id, rel);
                }
            },
        }
    }

    // Create edges between consecutive nodes
    let mut i = 0;
    while i < pattern.elements.len() {
        if i + 2 < pattern.elements.len() {
            if let (CypherElement::Node(n1), CypherElement::Rel(rel), CypherElement::Node(n2)) = (
                &pattern.elements[i],
                &pattern.elements[i + 1],
                &pattern.elements[i + 2],
            ) {
                let from_id = n1
                    .variable
                    .as_ref()
                    .and_then(|v| ctx.get_node(&v.name))
                    .ok_or_else(|| RouterError::InvalidArgument("Node not bound".to_string()))?;
                let to_id = n2
                    .variable
                    .as_ref()
                    .and_then(|v| ctx.get_node(&v.name))
                    .ok_or_else(|| RouterError::InvalidArgument("Node not bound".to_string()))?;

                let edge_type = rel.rel_types.first().map_or("RELATED", |t| t.name.as_str());

                let props = rel
                    .properties
                    .iter()
                    .filter_map(|p| {
                        expr_to_property_value(&p.value).map(|v| (p.key.name.clone(), v))
                    })
                    .collect();

                let (actual_from, actual_to) = match rel.direction {
                    CypherDirection::Incoming => (to_id, from_id),
                    _ => (from_id, to_id),
                };

                let edge_id = graph.create_edge(actual_from, actual_to, edge_type, props, true)?;
                created_ids.push(edge_id);

                if let Some(ref var) = rel.variable {
                    ctx.bind_edge(&var.name, edge_id);
                }
            }
        }
        i += 1;
    }

    Ok(())
}

fn create_node(graph: &GraphEngine, pattern: &CypherNode) -> Result<u64> {
    let label = pattern.labels.first().map_or("Node", |l| l.name.as_str());

    let props: HashMap<String, PropertyValue> = pattern
        .properties
        .iter()
        .filter_map(|p| expr_to_property_value(&p.value).map(|v| (p.key.name.clone(), v)))
        .collect();

    let id = graph.create_node(label, props)?;
    Ok(id)
}

// =============================================================================
// Result Projection
// =============================================================================

fn project_return(
    graph: &GraphEngine,
    return_clause: &CypherReturn,
    bindings: &[BindingContext],
) -> Vec<NodeResult> {
    let mut results = Vec::new();

    for ctx in bindings {
        // For each return item, extract the value
        for item in &return_clause.items {
            if let ExprKind::Ident(ref ident) = item.expr.kind {
                // Return a bound node
                if let Some(node_id) = ctx.nodes.get(&ident.name) {
                    if let Ok(node) = graph.get_node(*node_id) {
                        let label = node.labels.first().cloned().unwrap_or_default();
                        results.push(NodeResult {
                            id: node.id,
                            label,
                            properties: node
                                .properties
                                .iter()
                                .map(|(k, v)| (k.clone(), property_value_to_string(v)))
                                .collect(),
                        });
                    }
                }
            } else if let ExprKind::Qualified(ref base, ref field) = item.expr.kind {
                // Return a property: n.name
                if let ExprKind::Ident(ref ident) = base.kind {
                    if let Some(node_id) = ctx.nodes.get(&ident.name) {
                        if let Ok(node) = graph.get_node(*node_id) {
                            if let Some(val) = node.properties.get(&field.name) {
                                // Create a result with just this property
                                let mut props = HashMap::new();
                                props.insert(field.name.clone(), property_value_to_string(val));
                                let label = node.labels.first().cloned().unwrap_or_default();
                                results.push(NodeResult {
                                    id: node.id,
                                    label,
                                    properties: props,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply DISTINCT if specified
    if return_clause.distinct {
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.id));
    }

    results
}

// =============================================================================
// WHERE Clause Evaluation
// =============================================================================

fn evaluate_where_clause(graph: &GraphEngine, expr: &Expr, ctx: &BindingContext) -> Result<bool> {
    match &expr.kind {
        ExprKind::Binary(left, op, right) => {
            use neumann_parser::BinaryOp;
            match op {
                BinaryOp::And => {
                    let l = evaluate_where_clause(graph, left, ctx)?;
                    let r = evaluate_where_clause(graph, right, ctx)?;
                    Ok(l && r)
                },
                BinaryOp::Or => {
                    let l = evaluate_where_clause(graph, left, ctx)?;
                    let r = evaluate_where_clause(graph, right, ctx)?;
                    Ok(l || r)
                },
                BinaryOp::Eq => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(l == r)
                },
                BinaryOp::Ne => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(l != r)
                },
                BinaryOp::Lt => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(compare_values(&l, &r) == Some(std::cmp::Ordering::Less))
                },
                BinaryOp::Le => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(matches!(
                        compare_values(&l, &r),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    ))
                },
                BinaryOp::Gt => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(compare_values(&l, &r) == Some(std::cmp::Ordering::Greater))
                },
                BinaryOp::Ge => {
                    let l = eval_expr_value(graph, left, ctx);
                    let r = eval_expr_value(graph, right, ctx);
                    Ok(matches!(
                        compare_values(&l, &r),
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                    ))
                },
                _ => Ok(true), // Unsupported operator - pass through
            }
        },
        ExprKind::Unary(op, inner) => {
            use neumann_parser::UnaryOp;
            match op {
                UnaryOp::Not => {
                    let val = evaluate_where_clause(graph, inner, ctx)?;
                    Ok(!val)
                },
                _ => Ok(true),
            }
        },
        _ => Ok(true), // Unsupported expression type - pass through
    }
}

fn eval_expr_value(graph: &GraphEngine, expr: &Expr, ctx: &BindingContext) -> PropertyValue {
    match &expr.kind {
        ExprKind::Literal(lit) => literal_to_property_value(lit),
        ExprKind::Qualified(base, field) => {
            if let ExprKind::Ident(ref ident) = base.kind {
                if let Some(node_id) = ctx.nodes.get(&ident.name) {
                    if let Ok(node) = graph.get_node(*node_id) {
                        if let Some(val) = node.properties.get(&field.name) {
                            return val.clone();
                        }
                    }
                }
            }
            PropertyValue::Null
        },
        ExprKind::Ident(ident) => {
            // Identifier alone - might be a node reference
            if let Some(node_id) = ctx.nodes.get(&ident.name) {
                PropertyValue::Int((*node_id).cast_signed())
            } else {
                PropertyValue::Null
            }
        },
        _ => PropertyValue::Null,
    }
}

// =============================================================================
// SET Item Application
// =============================================================================

fn apply_set_item(
    graph: &GraphEngine,
    set_item: &neumann_parser::CypherSetItem,
    ctx: &BindingContext,
) -> Result<()> {
    // set_item.property is like n.name, set_item.value is the new value
    if let ExprKind::Qualified(ref base, ref field) = set_item.property.kind {
        if let ExprKind::Ident(ref ident) = base.kind {
            if let Some(node_id) = ctx.nodes.get(&ident.name) {
                if let Some(val) = expr_to_property_value(&set_item.value) {
                    // Get current node, update properties, and save
                    let node = graph.get_node(*node_id)?;
                    let mut props = node.properties.clone();
                    props.insert(field.name.clone(), val);
                    graph.update_node(*node_id, None, props)?;
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

fn cypher_direction_to_engine(dir: CypherDirection) -> Direction {
    match dir {
        CypherDirection::Outgoing => Direction::Outgoing,
        CypherDirection::Incoming => Direction::Incoming,
        CypherDirection::Undirected => Direction::Both,
    }
}

fn expr_to_property_value(expr: &Expr) -> Option<PropertyValue> {
    match &expr.kind {
        ExprKind::Literal(lit) => Some(literal_to_property_value(lit)),
        _ => None,
    }
}

fn literal_to_property_value(lit: &Literal) -> PropertyValue {
    match lit {
        Literal::Null => PropertyValue::Null,
        Literal::Boolean(b) => PropertyValue::Bool(*b),
        Literal::Integer(i) => PropertyValue::Int(*i),
        Literal::Float(f) => PropertyValue::Float(*f),
        Literal::String(s) => PropertyValue::String(s.clone()),
    }
}

fn property_value_to_string(val: &PropertyValue) -> String {
    match val {
        PropertyValue::Null => "null".to_string(),
        PropertyValue::Bool(b) => b.to_string(),
        PropertyValue::Int(i) => i.to_string(),
        PropertyValue::Float(f) => f.to_string(),
        PropertyValue::String(s) => s.clone(),
        PropertyValue::List(items) => {
            let parts: Vec<String> = items.iter().map(property_value_to_string).collect();
            format!("[{}]", parts.join(", "))
        },
        PropertyValue::Map(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", k, property_value_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        },
        PropertyValue::DateTime(dt) => dt.to_string(),
        PropertyValue::Bytes(b) => format!("<{} bytes>", b.len()),
        PropertyValue::Point { lat, lon } => format!("POINT({lat}, {lon})"),
    }
}

fn compare_values(a: &PropertyValue, b: &PropertyValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (PropertyValue::Int(a), PropertyValue::Int(b)) => Some(a.cmp(b)),
        (PropertyValue::Float(a), PropertyValue::Float(b)) => a.partial_cmp(b),
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for comparison
        (PropertyValue::Int(a), PropertyValue::Float(b)) => (*a as f64).partial_cmp(b),
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for comparison
        (PropertyValue::Float(a), PropertyValue::Int(b)) => a.partial_cmp(&(*b as f64)),
        (PropertyValue::String(a), PropertyValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn eval_int_expr(expr: &Expr) -> Option<i64> {
    if let ExprKind::Literal(Literal::Integer(n)) = &expr.kind {
        Some(*n)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neumann_parser::{
        BinaryOp, CypherReturnItem, CypherVarLength, Ident, Property, Span, UnaryOp,
    };

    fn make_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_expr_ident(name: &str) -> Expr {
        Expr {
            kind: ExprKind::Ident(make_ident(name)),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_expr_int(n: i64) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Integer(n)),
            span: Span::from_offsets(0, 0),
        }
    }

    #[allow(dead_code)]
    fn make_expr_float(f: f64) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Float(f)),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_expr_string(s: &str) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::String(s.to_string())),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_expr_qualified(base: &str, field: &str) -> Expr {
        Expr {
            kind: ExprKind::Qualified(Box::new(make_expr_ident(base)), make_ident(field)),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_binary_expr(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr {
            kind: ExprKind::Binary(Box::new(left), op, Box::new(right)),
            span: Span::from_offsets(0, 0),
        }
    }

    fn make_unary_expr(op: UnaryOp, inner: Expr) -> Expr {
        Expr {
            kind: ExprKind::Unary(op, Box::new(inner)),
            span: Span::from_offsets(0, 0),
        }
    }

    // =========================================================================
    // BindingContext tests
    // =========================================================================

    #[test]
    fn test_binding_context() {
        let mut ctx = BindingContext::default();
        ctx.bind_node("n", 42);
        ctx.bind_edge("r", 100);

        assert_eq!(ctx.get_node("n"), Some(42));
        assert_eq!(ctx.get_node("m"), None);
    }

    #[test]
    fn test_binding_context_overwrite() {
        let mut ctx = BindingContext::default();
        ctx.bind_node("n", 42);
        ctx.bind_node("n", 99);
        assert_eq!(ctx.get_node("n"), Some(99));
    }

    // =========================================================================
    // Direction conversion tests
    // =========================================================================

    #[test]
    fn test_cypher_direction_conversion() {
        assert!(matches!(
            cypher_direction_to_engine(CypherDirection::Outgoing),
            Direction::Outgoing
        ));
        assert!(matches!(
            cypher_direction_to_engine(CypherDirection::Incoming),
            Direction::Incoming
        ));
        assert!(matches!(
            cypher_direction_to_engine(CypherDirection::Undirected),
            Direction::Both
        ));
    }

    // =========================================================================
    // Literal conversion tests
    // =========================================================================

    #[test]
    fn test_literal_conversion() {
        assert!(matches!(
            literal_to_property_value(&Literal::Boolean(true)),
            PropertyValue::Bool(true)
        ));
        assert!(matches!(
            literal_to_property_value(&Literal::Integer(42)),
            PropertyValue::Int(42)
        ));
        assert!(matches!(
            literal_to_property_value(&Literal::String("test".to_string())),
            PropertyValue::String(_)
        ));
    }

    #[test]
    fn test_literal_conversion_null() {
        assert!(matches!(
            literal_to_property_value(&Literal::Null),
            PropertyValue::Null
        ));
    }

    #[test]
    fn test_literal_conversion_float() {
        let result = literal_to_property_value(&Literal::Float(3.14));
        assert!(matches!(result, PropertyValue::Float(f) if (f - 3.14).abs() < 0.001));
    }

    // =========================================================================
    // Property value to string tests
    // =========================================================================

    #[test]
    fn test_property_value_to_string() {
        assert_eq!(property_value_to_string(&PropertyValue::Null), "null");
        assert_eq!(property_value_to_string(&PropertyValue::Bool(true)), "true");
        assert_eq!(
            property_value_to_string(&PropertyValue::Bool(false)),
            "false"
        );
        assert_eq!(property_value_to_string(&PropertyValue::Int(42)), "42");
        assert_eq!(
            property_value_to_string(&PropertyValue::String("test".to_string())),
            "test"
        );
    }

    #[test]
    fn test_property_value_to_string_float() {
        let result = property_value_to_string(&PropertyValue::Float(3.14));
        assert!(result.starts_with("3.14"));
    }

    #[test]
    fn test_property_value_to_string_list() {
        let list = PropertyValue::List(vec![
            PropertyValue::Int(1),
            PropertyValue::Int(2),
            PropertyValue::Int(3),
        ]);
        assert_eq!(property_value_to_string(&list), "[1, 2, 3]");
    }

    #[test]
    fn test_property_value_to_string_map() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), PropertyValue::Int(42));
        let val = PropertyValue::Map(map);
        assert_eq!(property_value_to_string(&val), "{key: 42}");
    }

    #[test]
    fn test_property_value_to_string_datetime() {
        let val = PropertyValue::DateTime(1234567890);
        assert_eq!(property_value_to_string(&val), "1234567890");
    }

    #[test]
    fn test_property_value_to_string_bytes() {
        let val = PropertyValue::Bytes(vec![1, 2, 3, 4, 5]);
        assert_eq!(property_value_to_string(&val), "<5 bytes>");
    }

    #[test]
    fn test_property_value_to_string_point() {
        let val = PropertyValue::Point {
            lat: 40.7128,
            lon: -74.0060,
        };
        assert!(property_value_to_string(&val).contains("POINT"));
    }

    // =========================================================================
    // Compare values tests
    // =========================================================================

    #[test]
    fn test_compare_values() {
        assert_eq!(
            compare_values(&PropertyValue::Int(1), &PropertyValue::Int(2)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&PropertyValue::Int(2), &PropertyValue::Int(2)),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_values(&PropertyValue::Int(3), &PropertyValue::Int(2)),
            Some(std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn test_compare_values_float() {
        assert_eq!(
            compare_values(&PropertyValue::Float(1.0), &PropertyValue::Float(2.0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&PropertyValue::Float(2.0), &PropertyValue::Float(2.0)),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_values_int_float() {
        assert_eq!(
            compare_values(&PropertyValue::Int(1), &PropertyValue::Float(2.0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&PropertyValue::Float(1.0), &PropertyValue::Int(2)),
            Some(std::cmp::Ordering::Less)
        );
    }

    #[test]
    fn test_compare_values_string() {
        assert_eq!(
            compare_values(
                &PropertyValue::String("a".to_string()),
                &PropertyValue::String("b".to_string())
            ),
            Some(std::cmp::Ordering::Less)
        );
    }

    #[test]
    fn test_compare_values_incompatible() {
        assert_eq!(
            compare_values(
                &PropertyValue::Int(1),
                &PropertyValue::String("a".to_string())
            ),
            None
        );
        assert_eq!(
            compare_values(&PropertyValue::Null, &PropertyValue::Int(1)),
            None
        );
    }

    // =========================================================================
    // eval_int_expr tests
    // =========================================================================

    #[test]
    fn test_eval_int_expr_valid() {
        let expr = make_expr_int(42);
        assert_eq!(eval_int_expr(&expr), Some(42));
    }

    #[test]
    fn test_eval_int_expr_non_integer() {
        let expr = make_expr_string("hello");
        assert_eq!(eval_int_expr(&expr), None);
    }

    // =========================================================================
    // expr_to_property_value tests
    // =========================================================================

    #[test]
    fn test_expr_to_property_value_literal() {
        let expr = make_expr_int(42);
        assert!(matches!(
            expr_to_property_value(&expr),
            Some(PropertyValue::Int(42))
        ));
    }

    #[test]
    fn test_expr_to_property_value_non_literal() {
        let expr = make_expr_ident("foo");
        assert!(expr_to_property_value(&expr).is_none());
    }

    // =========================================================================
    // exec_cypher_create tests
    // =========================================================================

    #[test]
    fn test_exec_cypher_create_single_node() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Ids(ids) = result.unwrap() {
            assert_eq!(ids.len(), 1);
        } else {
            panic!("Expected Ids result");
        }
    }

    #[test]
    fn test_exec_cypher_create_node_with_properties() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![Property {
                        key: make_ident("name"),
                        value: make_expr_string("Alice"),
                    }],
                })],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_create_node_without_variable() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: None,
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_create_node_without_label() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![],
                    properties: vec![],
                })],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_create_with_relationship() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: Some(make_ident("r")),
                        rel_types: vec![make_ident("KNOWS")],
                        direction: CypherDirection::Outgoing,
                        var_length: None,
                        properties: vec![],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Ids(ids) = result.unwrap() {
            assert_eq!(ids.len(), 3); // 2 nodes + 1 edge
        }
    }

    #[test]
    fn test_exec_cypher_create_with_incoming_relationship() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: None,
                        rel_types: vec![make_ident("KNOWS")],
                        direction: CypherDirection::Incoming,
                        var_length: None,
                        properties: vec![],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_create_relationship_without_type() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: None,
                        rel_types: vec![],
                        direction: CypherDirection::Outgoing,
                        var_length: None,
                        properties: vec![],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_create_relationship_with_properties() {
        let graph = GraphEngine::new();
        let stmt = CypherCreateStmt {
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: None,
                        rel_types: vec![make_ident("KNOWS")],
                        direction: CypherDirection::Outgoing,
                        var_length: None,
                        properties: vec![Property {
                            key: make_ident("since"),
                            value: make_expr_int(2020),
                        }],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
        };

        let result = exec_cypher_create(&graph, &stmt);
        assert!(result.is_ok());
    }

    // =========================================================================
    // exec_cypher_match tests
    // =========================================================================

    #[test]
    fn test_exec_cypher_match_simple() {
        let graph = GraphEngine::new();
        // Create a node first
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_skip() {
        let graph = GraphEngine::new();
        graph.create_node("Person", HashMap::new()).unwrap();
        graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: Some(make_expr_int(1)),
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Nodes(nodes) = result.unwrap() {
            assert_eq!(nodes.len(), 1);
        }
    }

    #[test]
    fn test_exec_cypher_match_with_skip_exceeds_results() {
        let graph = GraphEngine::new();
        graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: Some(make_expr_int(100)),
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Nodes(nodes) = result.unwrap() {
            assert_eq!(nodes.len(), 0);
        }
    }

    #[test]
    fn test_exec_cypher_match_with_limit() {
        let graph = GraphEngine::new();
        graph.create_node("Person", HashMap::new()).unwrap();
        graph.create_node("Person", HashMap::new()).unwrap();
        graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: Some(make_expr_int(2)),
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Nodes(nodes) = result.unwrap() {
            assert_eq!(nodes.len(), 2);
        }
    }

    #[test]
    fn test_exec_cypher_match_with_where_eq() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Eq,
                make_expr_int(30),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_ne() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Ne,
                make_expr_int(25),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_lt() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(20));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Lt,
                make_expr_int(30),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_le() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Le,
                make_expr_int(30),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_gt() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(40));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Gt,
                make_expr_int(30),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_ge() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_expr_qualified("n", "age"),
                BinaryOp::Ge,
                make_expr_int(30),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_and() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        props.insert("active".to_string(), PropertyValue::Bool(true));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_binary_expr(
                    make_expr_qualified("n", "age"),
                    BinaryOp::Eq,
                    make_expr_int(30),
                ),
                BinaryOp::And,
                make_binary_expr(
                    make_expr_qualified("n", "active"),
                    BinaryOp::Eq,
                    Expr {
                        kind: ExprKind::Literal(Literal::Boolean(true)),
                        span: Span::from_offsets(0, 0),
                    },
                ),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_or() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_binary_expr(
                make_binary_expr(
                    make_expr_qualified("n", "age"),
                    BinaryOp::Eq,
                    make_expr_int(30),
                ),
                BinaryOp::Or,
                make_binary_expr(
                    make_expr_qualified("n", "age"),
                    BinaryOp::Eq,
                    make_expr_int(40),
                ),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_where_not() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: Some(Box::new(make_unary_expr(
                UnaryOp::Not,
                make_binary_expr(
                    make_expr_qualified("n", "age"),
                    BinaryOp::Eq,
                    make_expr_int(25),
                ),
            ))),
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_distinct() {
        let graph = GraphEngine::new();
        graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: true,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_return_property() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_qualified("n", "name"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_relationship() {
        let graph = GraphEngine::new();
        let id1 = graph.create_node("Person", HashMap::new()).unwrap();
        let id2 = graph.create_node("Person", HashMap::new()).unwrap();
        graph
            .create_edge(id1, id2, "KNOWS", HashMap::new(), true)
            .unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: Some(make_ident("r")),
                        rel_types: vec![make_ident("KNOWS")],
                        direction: CypherDirection::Outgoing,
                        var_length: None,
                        properties: vec![],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("b"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_var_length() {
        let graph = GraphEngine::new();
        let id1 = graph.create_node("Person", HashMap::new()).unwrap();
        let id2 = graph.create_node("Person", HashMap::new()).unwrap();
        let id3 = graph.create_node("Person", HashMap::new()).unwrap();
        graph
            .create_edge(id1, id2, "KNOWS", HashMap::new(), true)
            .unwrap();
        graph
            .create_edge(id2, id3, "KNOWS", HashMap::new(), true)
            .unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("a")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                    CypherElement::Rel(neumann_parser::CypherRel {
                        variable: None,
                        rel_types: vec![make_ident("KNOWS")],
                        direction: CypherDirection::Outgoing,
                        var_length: Some(CypherVarLength::range(Some(1), Some(3))),
                        properties: vec![],
                    }),
                    CypherElement::Node(CypherNode {
                        variable: Some(make_ident("b")),
                        labels: vec![make_ident("Person")],
                        properties: vec![],
                    }),
                ],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("b"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_empty_pattern() {
        let graph = GraphEngine::new();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_no_label() {
        let graph = GraphEngine::new();
        graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![],
                    properties: vec![],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_match_with_inline_property() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMatchStmt {
            optional: false,
            patterns: vec![CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![Property {
                        key: make_ident("name"),
                        value: make_expr_string("Alice"),
                    }],
                })],
            }],
            where_clause: None,
            return_clause: CypherReturn {
                distinct: false,
                items: vec![CypherReturnItem {
                    expr: make_expr_ident("n"),
                    alias: None,
                }],
            },
            order_by: vec![],
            skip: None,
            limit: None,
        };

        let result = exec_cypher_match(&graph, &stmt);
        assert!(result.is_ok());
    }

    // =========================================================================
    // exec_cypher_delete tests
    // =========================================================================

    #[test]
    fn test_exec_cypher_delete_simple() {
        let graph = GraphEngine::new();
        let id = graph.create_node("Person", HashMap::new()).unwrap();

        let stmt = CypherDeleteStmt {
            detach: false,
            variables: vec![Expr {
                kind: ExprKind::Ident(Ident {
                    name: id.to_string(),
                    span: Span::from_offsets(0, 0),
                }),
                span: Span::from_offsets(0, 0),
            }],
        };

        let result = exec_cypher_delete(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Count(count) = result.unwrap() {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_exec_cypher_delete_non_numeric_id() {
        let graph = GraphEngine::new();

        let stmt = CypherDeleteStmt {
            detach: false,
            variables: vec![make_expr_ident("not_a_number")],
        };

        let result = exec_cypher_delete(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Count(count) = result.unwrap() {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_exec_cypher_delete_non_ident() {
        let graph = GraphEngine::new();

        let stmt = CypherDeleteStmt {
            detach: false,
            variables: vec![make_expr_int(42)],
        };

        let result = exec_cypher_delete(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Count(count) = result.unwrap() {
            assert_eq!(count, 0);
        }
    }

    // =========================================================================
    // exec_cypher_merge tests
    // =========================================================================

    #[test]
    fn test_exec_cypher_merge_create() {
        let graph = GraphEngine::new();

        let stmt = CypherMergeStmt {
            pattern: CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![Property {
                        key: make_ident("name"),
                        value: make_expr_string("Alice"),
                    }],
                })],
            },
            on_create: vec![],
            on_match: vec![],
        };

        let result = exec_cypher_merge(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Ids(ids) = result.unwrap() {
            assert_eq!(ids.len(), 1);
        }
    }

    #[test]
    fn test_exec_cypher_merge_match() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMergeStmt {
            pattern: CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![Property {
                        key: make_ident("name"),
                        value: make_expr_string("Alice"),
                    }],
                })],
            },
            on_create: vec![],
            on_match: vec![],
        };

        let result = exec_cypher_merge(&graph, &stmt);
        assert!(result.is_ok());
        if let QueryResult::Ids(ids) = result.unwrap() {
            assert_eq!(ids.len(), 1);
        }
    }

    #[test]
    fn test_exec_cypher_merge_with_on_create() {
        let graph = GraphEngine::new();

        let stmt = CypherMergeStmt {
            pattern: CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![],
                })],
            },
            on_create: vec![neumann_parser::CypherSetItem {
                property: make_expr_qualified("n", "created"),
                value: make_expr_int(12345),
            }],
            on_match: vec![],
        };

        let result = exec_cypher_merge(&graph, &stmt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exec_cypher_merge_with_on_match() {
        let graph = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        graph.create_node("Person", props).unwrap();

        let stmt = CypherMergeStmt {
            pattern: CypherPattern {
                variable: None,
                elements: vec![CypherElement::Node(CypherNode {
                    variable: Some(make_ident("n")),
                    labels: vec![make_ident("Person")],
                    properties: vec![Property {
                        key: make_ident("name"),
                        value: make_expr_string("Alice"),
                    }],
                })],
            },
            on_create: vec![],
            on_match: vec![neumann_parser::CypherSetItem {
                property: make_expr_qualified("n", "updated"),
                value: make_expr_int(99999),
            }],
        };

        let result = exec_cypher_merge(&graph, &stmt);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_eval_expr_value_ident_not_bound() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_expr_ident("unknown");

        let result = eval_expr_value(&graph, &expr, &ctx);
        assert!(matches!(result, PropertyValue::Null));
    }

    #[test]
    fn test_eval_expr_value_qualified_not_bound() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_expr_qualified("unknown", "prop");

        let result = eval_expr_value(&graph, &expr, &ctx);
        assert!(matches!(result, PropertyValue::Null));
    }

    #[test]
    fn test_eval_expr_value_other() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_binary_expr(make_expr_int(1), BinaryOp::Add, make_expr_int(2));

        let result = eval_expr_value(&graph, &expr, &ctx);
        assert!(matches!(result, PropertyValue::Null));
    }

    #[test]
    fn test_evaluate_where_clause_unsupported_binary_op() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_binary_expr(make_expr_int(1), BinaryOp::Add, make_expr_int(2));

        let result = evaluate_where_clause(&graph, &expr, &ctx);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Unsupported ops pass through as true
    }

    #[test]
    fn test_evaluate_where_clause_unsupported_unary_op() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_unary_expr(UnaryOp::Neg, make_expr_int(42));

        let result = evaluate_where_clause(&graph, &expr, &ctx);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Unsupported ops pass through as true
    }

    #[test]
    fn test_evaluate_where_clause_unsupported_expr() {
        let graph = GraphEngine::new();
        let ctx = BindingContext::default();
        let expr = make_expr_int(42);

        let result = evaluate_where_clause(&graph, &expr, &ctx);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Unsupported exprs pass through as true
    }

    #[test]
    fn test_node_matches_pattern_missing_property() {
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };

        let pattern = CypherNode {
            variable: None,
            labels: vec![make_ident("Person")],
            properties: vec![Property {
                key: make_ident("name"),
                value: make_expr_string("Alice"),
            }],
        };

        assert!(!node_matches_pattern(&node, &pattern));
    }

    #[test]
    fn test_node_matches_pattern_wrong_property_value() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };

        let pattern = CypherNode {
            variable: None,
            labels: vec![make_ident("Person")],
            properties: vec![Property {
                key: make_ident("name"),
                value: make_expr_string("Alice"),
            }],
        };

        assert!(!node_matches_pattern(&node, &pattern));
    }

    #[test]
    fn test_node_matches_pattern_missing_label() {
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };

        let pattern = CypherNode {
            variable: None,
            labels: vec![make_ident("Person"), make_ident("Employee")],
            properties: vec![],
        };

        assert!(!node_matches_pattern(&node, &pattern));
    }
}
