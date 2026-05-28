// SPDX-License-Identifier: MIT OR Apache-2.0
//! `CHAIN` statement execution.

use neumann_parser::{ChainOp, ChainStmt};

use crate::result::{
    ChainBlockInfo, ChainCodebookInfo, ChainDriftResult, ChainHistoryEntry, ChainResult,
    ChainTransitionAnalysis,
};
use crate::{QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Execute a `CHAIN ...` statement. Requires an authenticated identity.
#[allow(clippy::too_many_lines, reason = "covers every CHAIN sub-op")]
pub fn exec_chain(router: &QueryRouter, stmt: &ChainStmt) -> Result<QueryResult> {
    let _identity = router.require_identity()?;

    let chain = router
        .chain
        .as_ref()
        .ok_or_else(|| RouterError::ChainError("Chain not initialized".to_string()))?;

    match &stmt.operation {
        ChainOp::Begin => {
            let workspace = chain.begin()?;
            Ok(QueryResult::Chain(ChainResult::TransactionBegun {
                tx_id: workspace.id().to_string(),
            }))
        },
        ChainOp::Commit => Ok(QueryResult::Chain(ChainResult::Committed {
            block_hash: "pending".to_string(),
            height: chain.height(),
        })),
        ChainOp::Rollback { height } => {
            let h = expr::expr_to_u64(height)?;
            Ok(QueryResult::Chain(ChainResult::RolledBack { to_height: h }))
        },
        ChainOp::History { key } => {
            let key_str = expr::eval_string_expr(key)?;
            let history = chain.history(&key_str)?;
            let entries: Vec<ChainHistoryEntry> = history
                .into_iter()
                .map(|(height, tx)| ChainHistoryEntry {
                    height,
                    transaction_type: format!("{tx:?}"),
                    data: None,
                })
                .collect();
            Ok(QueryResult::Chain(ChainResult::History(entries)))
        },
        ChainOp::Similar { embedding, limit } => {
            let _embedding: Vec<f32> = embedding
                .iter()
                .map(expr::expr_to_f32)
                .collect::<Result<Vec<_>>>()?;
            let _limit = limit.as_ref().map(expr::expr_to_usize).transpose()?;
            Ok(QueryResult::Chain(ChainResult::Similar(vec![])))
        },
        ChainOp::Drift {
            from_height,
            to_height,
        } => {
            let from_h = expr::expr_to_u64(from_height)?;
            let to_h = expr::expr_to_u64(to_height)?;
            Ok(QueryResult::Chain(ChainResult::Drift(ChainDriftResult {
                from_height: from_h,
                to_height: to_h,
                total_drift: 0.0,
                avg_drift_per_block: 0.0,
                max_drift: 0.0,
            })))
        },
        ChainOp::Height => Ok(QueryResult::Chain(ChainResult::Height(chain.height()))),
        ChainOp::Tip => {
            let hash = chain.tip_hash();
            let height = chain.height();
            Ok(QueryResult::Chain(ChainResult::Tip {
                hash: hex::encode(hash),
                height,
            }))
        },
        ChainOp::Block { height } => {
            let h = expr::expr_to_u64(height)?;
            if let Some(block) = chain.get_block(h)? {
                Ok(QueryResult::Chain(ChainResult::Block(ChainBlockInfo {
                    height: h,
                    hash: hex::encode(block.hash()),
                    prev_hash: hex::encode(block.header.prev_hash),
                    timestamp: block.header.timestamp,
                    transaction_count: block.transactions.len(),
                    proposer: block.header.proposer,
                })))
            } else {
                Err(RouterError::ChainError(format!("Block {h} not found")))
            }
        },
        ChainOp::Verify => match chain.verify() {
            Ok(()) => Ok(QueryResult::Chain(ChainResult::Verified {
                ok: true,
                errors: vec![],
            })),
            Err(e) => Ok(QueryResult::Chain(ChainResult::Verified {
                ok: false,
                errors: vec![e.to_string()],
            })),
        },
        ChainOp::ShowCodebookGlobal => Ok(QueryResult::Chain(ChainResult::Codebook(
            ChainCodebookInfo {
                scope: "global".to_string(),
                entry_count: 0,
                dimension: 0,
                domain: None,
            },
        ))),
        ChainOp::ShowCodebookLocal { domain } => {
            let domain_str = expr::eval_string_expr(domain)?;
            Ok(QueryResult::Chain(ChainResult::Codebook(
                ChainCodebookInfo {
                    scope: "local".to_string(),
                    entry_count: 0,
                    dimension: 0,
                    domain: Some(domain_str),
                },
            )))
        },
        ChainOp::AnalyzeTransitions => Ok(QueryResult::Chain(ChainResult::TransitionAnalysis(
            ChainTransitionAnalysis {
                total_transitions: 0,
                valid_transitions: 0,
                invalid_transitions: 0,
                avg_validity_score: 0.0,
            },
        ))),
    }
}
