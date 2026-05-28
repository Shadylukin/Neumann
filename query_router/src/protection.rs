// SPDX-License-Identifier: MIT OR Apache-2.0
//! Destructive-operation protection: confirm-or-checkpoint before drops/deletes
//! and gather sample data for previews.
//!
//! These helpers are called from every dispatcher arm that performs a
//! destructive operation (DROP TABLE, DROP INDEX, DELETE, NODE DELETE,
//! EDGE DELETE, etc.). They live in their own module because they cut across
//! SQL, graph, vector, blob, vault, and chain.

use graph_engine::Direction;
use relational_engine::Condition;
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::QueryRouter;

/// Check and optionally create checkpoint before a destructive operation.
///
/// If a checkpoint manager is configured and auto-checkpoint is enabled,
/// this generates a preview, prompts for confirmation via the configured
/// handler, and creates an auto-checkpoint before returning [`Proceed`].
///
/// [`Proceed`]: ProtectedOpResult::Proceed
pub fn protect_destructive_op(
    router: &QueryRouter,
    command: &str,
    op: DestructiveOp,
    sample_data: Vec<String>,
) -> ProtectedOpResult {
    // If no checkpoint manager, proceed without protection
    let Some(checkpoint) = router.checkpoint.as_ref() else {
        return ProtectedOpResult::Proceed;
    };

    // Check if auto-checkpoint is enabled
    if !checkpoint.auto_checkpoint_enabled() {
        return ProtectedOpResult::Proceed;
    }

    // Generate preview
    let preview = checkpoint.generate_preview(&op, sample_data);

    // Request confirmation (may prompt user via handler)
    if !checkpoint.request_confirmation(&op, &preview) {
        return ProtectedOpResult::Cancelled;
    }

    // Create auto-checkpoint before operation
    let store = router.vector.store();
    if let Err(e) = checkpoint.create_auto(command, op, preview, store) {
        // Log but don't fail - checkpoint is best-effort
        eprintln!("Warning: Failed to create auto-checkpoint: {e}");
    }

    ProtectedOpResult::Proceed
}

/// Collect sample data for a relational delete preview.
pub fn collect_delete_sample(
    router: &QueryRouter,
    table: &str,
    condition: &Condition,
    limit: usize,
) -> (usize, Vec<String>) {
    let Ok(rows) = router.relational.select(table, condition.clone()) else {
        return (0, vec![]);
    };

    let count = rows.len();
    let sample: Vec<String> = rows
        .into_iter()
        .take(limit)
        .map(|row| {
            let pairs: Vec<String> = row
                .values
                .iter()
                .map(|(k, v)| format!("{k}={v:?}"))
                .collect();
            format!("_id={}, {}", row.id, pairs.join(", "))
        })
        .collect();

    (count, sample)
}

/// Collect sample data for a DROP TABLE preview.
pub fn collect_table_sample(
    router: &QueryRouter,
    table: &str,
    limit: usize,
) -> (usize, Vec<String>) {
    collect_delete_sample(router, table, &Condition::True, limit)
}

/// Collect info about a node for deletion preview.
pub fn collect_node_info(router: &QueryRouter, node_id: u64) -> (usize, Vec<String>) {
    // Count connected edges (neighbors returns Vec<Node>)
    let edge_count = router
        .graph
        .neighbors(node_id, None, Direction::Both, None)
        .map_or(0, |nodes| nodes.len());

    // Get node label and properties for sample
    let sample = match router.graph.get_node(node_id) {
        Ok(node) => {
            let props: Vec<String> = node
                .properties
                .iter()
                .take(3)
                .map(|(k, v)| format!("{k}={v:?}"))
                .collect();
            vec![format!(
                "label='{}', {}",
                node.labels.join(":"),
                props.join(", ")
            )]
        },
        Err(_) => vec![],
    };

    (edge_count, sample)
}

/// Collect info about an edge for deletion preview.
pub fn collect_edge_info(router: &QueryRouter, edge_id: u64) -> Vec<String> {
    match router.graph.get_edge(edge_id) {
        Ok(edge) => {
            let props: Vec<String> = edge
                .properties
                .iter()
                .take(3)
                .map(|(k, v)| format!("{k}={v:?}"))
                .collect();
            vec![format!(
                "type='{}', from={}, to={}, {}",
                edge.edge_type,
                edge.from,
                edge.to,
                props.join(", ")
            )]
        },
        Err(_) => vec![],
    }
}
