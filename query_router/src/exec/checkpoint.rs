// SPDX-License-Identifier: MIT OR Apache-2.0
//! `CHECKPOINT`, `CHECKPOINTS`, and `ROLLBACK` statement execution.

use neumann_parser::{CheckpointStmt, CheckpointsStmt, RollbackStmt};

use crate::result::CheckpointInfo;
use crate::{QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Execute a `CHECKPOINT [name]` statement.
pub fn exec_checkpoint(router: &QueryRouter, stmt: &CheckpointStmt) -> Result<QueryResult> {
    let checkpoint = router.checkpoint.as_ref().ok_or_else(|| {
        RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
    })?;

    let name = stmt.name.as_ref().map(expr::eval_string_expr).transpose()?;

    let store = router.vector.store();
    let checkpoint_id = checkpoint.create(name.as_deref(), store)?;

    Ok(QueryResult::Value(format!(
        "Checkpoint created: {checkpoint_id}"
    )))
}

/// Execute a `ROLLBACK TO <id|name>` statement.
pub fn exec_rollback(router: &QueryRouter, stmt: &RollbackStmt) -> Result<QueryResult> {
    let checkpoint = router.checkpoint.as_ref().ok_or_else(|| {
        RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
    })?;

    let target = expr::eval_string_expr(&stmt.target)?;

    let store = router.vector.store();
    checkpoint.rollback(&target, store)?;

    Ok(QueryResult::Value(format!(
        "Rolled back to checkpoint: {target}"
    )))
}

/// Execute a `CHECKPOINTS [LIMIT n]` listing.
pub fn exec_checkpoints(router: &QueryRouter, stmt: &CheckpointsStmt) -> Result<QueryResult> {
    let checkpoint = router.checkpoint.as_ref().ok_or_else(|| {
        RouterError::CheckpointError("Checkpoint manager not initialized".to_string())
    })?;

    let limit = stmt.limit.as_ref().map(expr::expr_to_usize).transpose()?;
    let limit_opt = limit.or(Some(10));

    let checkpoints = checkpoint.list(limit_opt)?;

    let info_list: Vec<CheckpointInfo> = checkpoints
        .into_iter()
        .map(|cp| CheckpointInfo {
            id: cp.id,
            name: cp.name,
            created_at: cp.created_at,
            is_auto: cp.trigger.is_some(),
        })
        .collect();

    Ok(QueryResult::CheckpointList(info_list))
}
