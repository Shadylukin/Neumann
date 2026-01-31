// SPDX-License-Identifier: MIT OR Apache-2.0
//! Blockchain/chain result formatting.

#![allow(clippy::needless_borrows_for_generic_args)]

use crate::output::TableBuilder;
use crate::style::{styled, Icons, Theme};
use query_router::{
    ChainBlockInfo, ChainCodebookInfo, ChainDriftResult, ChainHistoryEntry, ChainResult,
    ChainSimilarResult, ChainTransitionAnalysis,
};
use std::fmt::Write;

/// Formats chain operation results.
#[must_use]
pub fn format_chain_result(result: &ChainResult, theme: &Theme, icons: &Icons) -> String {
    match result {
        ChainResult::TransactionBegun { tx_id } => {
            format!(
                "{} Chain transaction started: {}",
                icons.chain,
                styled(tx_id, theme.id)
            )
        },
        ChainResult::Committed { block_hash, height } => {
            format!(
                "{} Committed block {} at height {}",
                icons.success,
                styled(block_hash, theme.id),
                styled(height, theme.number)
            )
        },
        ChainResult::RolledBack { to_height } => {
            format!(
                "{} Chain rolled back to height {}",
                icons.warning,
                styled(to_height, theme.number)
            )
        },
        ChainResult::History(entries) => format_chain_history(entries, theme),
        ChainResult::Similar(results) => format_chain_similar(results, theme),
        ChainResult::Drift(drift) => format_chain_drift(drift, theme),
        ChainResult::Height(h) => {
            format!("Chain height: {}", styled(h, theme.number))
        },
        ChainResult::Tip { hash, height } => {
            format!(
                "Chain tip: {} at height {}",
                styled(hash, theme.id),
                styled(height, theme.number)
            )
        },
        ChainResult::Block(info) => format_chain_block(info, theme),
        ChainResult::Codebook(info) => format_chain_codebook(info, theme),
        ChainResult::Verified { ok, errors } => {
            if *ok {
                format!("{} Chain verified: OK", icons.success)
            } else {
                let mut output = format!("{} Chain verification failed:\n", icons.error);
                for err in errors {
                    let _ = writeln!(output, "  - {}", styled(err, theme.error));
                }
                output.trim_end().to_string()
            }
        },
        ChainResult::TransitionAnalysis(analysis) => format_chain_transitions(analysis, theme),
    }
}

/// Formats chain history entries.
#[must_use]
pub fn format_chain_history(entries: &[ChainHistoryEntry], theme: &Theme) -> String {
    if entries.is_empty() {
        return styled("No history found for key", theme.muted);
    }

    let mut output = format!("{}\n", styled("Chain History:", theme.header));

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["Height", "Transaction"]);

    for entry in entries {
        builder.add_row(vec![
            styled(entry.height, theme.number),
            styled(&entry.transaction_type, theme.keyword),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

/// Formats chain similarity results.
#[must_use]
pub fn format_chain_similar(results: &[ChainSimilarResult], theme: &Theme) -> String {
    if results.is_empty() {
        return styled("No similar blocks found", theme.muted);
    }

    let mut output = format!("{}\n", styled("Similar Blocks:", theme.header));

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["Height", "Hash", "Similarity"]);

    for r in results {
        builder.add_row(vec![
            styled(r.height, theme.number),
            styled(&r.block_hash, theme.id),
            styled(format!("{:.4}", r.similarity), theme.number),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

/// Formats chain drift analysis.
#[must_use]
pub fn format_chain_drift(drift: &ChainDriftResult, theme: &Theme) -> String {
    let mut output = format!("{}\n", styled("Chain Drift Analysis:", theme.header));
    let _ = writeln!(
        output,
        "  From height:     {}",
        styled(drift.from_height, theme.number)
    );
    let _ = writeln!(
        output,
        "  To height:       {}",
        styled(drift.to_height, theme.number)
    );
    let _ = writeln!(
        output,
        "  Total drift:     {}",
        styled(format!("{:.4}", drift.total_drift), theme.number)
    );
    let _ = writeln!(
        output,
        "  Avg drift/block: {}",
        styled(format!("{:.4}", drift.avg_drift_per_block), theme.number)
    );
    let _ = write!(
        output,
        "  Max drift:       {}",
        styled(format!("{:.4}", drift.max_drift), theme.number)
    );
    output
}

/// Formats chain block info.
#[must_use]
pub fn format_chain_block(info: &ChainBlockInfo, theme: &Theme) -> String {
    let mut output = format!("{}\n", styled("Block Info:", theme.header));
    let _ = writeln!(
        output,
        "  Height:       {}",
        styled(info.height, theme.number)
    );
    let _ = writeln!(output, "  Hash:         {}", styled(&info.hash, theme.id));
    let _ = writeln!(
        output,
        "  Prev Hash:    {}",
        styled(&info.prev_hash, theme.id)
    );
    let _ = writeln!(
        output,
        "  Timestamp:    {}",
        styled(&info.timestamp, theme.muted)
    );
    let _ = writeln!(
        output,
        "  Transactions: {}",
        styled(info.transaction_count, theme.number)
    );
    let _ = write!(
        output,
        "  Proposer:     {}",
        styled(&info.proposer, theme.label)
    );
    output
}

/// Formats codebook info.
#[must_use]
pub fn format_chain_codebook(info: &ChainCodebookInfo, theme: &Theme) -> String {
    let mut output = format!("{}\n", styled("Codebook Info:", theme.header));
    let _ = writeln!(
        output,
        "  Scope:     {}",
        styled(&info.scope, theme.keyword)
    );
    let _ = writeln!(
        output,
        "  Entries:   {}",
        styled(info.entry_count, theme.number)
    );
    let _ = writeln!(
        output,
        "  Dimension: {}",
        styled(info.dimension, theme.number)
    );
    if let Some(domain) = &info.domain {
        let _ = write!(output, "  Domain:    {}", styled(domain, theme.label));
    } else {
        output = output.trim_end().to_string();
    }
    output
}

/// Formats transition analysis.
#[must_use]
pub fn format_chain_transitions(analysis: &ChainTransitionAnalysis, theme: &Theme) -> String {
    let mut output = format!("{}\n", styled("Transition Analysis:", theme.header));
    let _ = writeln!(
        output,
        "  Total transitions:   {}",
        styled(analysis.total_transitions, theme.number)
    );
    let _ = writeln!(
        output,
        "  Valid transitions:   {}",
        styled(analysis.valid_transitions, theme.number)
    );
    let _ = writeln!(
        output,
        "  Invalid transitions: {}",
        styled(analysis.invalid_transitions, theme.number)
    );
    let _ = write!(
        output,
        "  Avg validity score:  {}",
        styled(format!("{:.4}", analysis.avg_validity_score), theme.number)
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chain_history_empty() {
        let theme = Theme::plain();
        let result = format_chain_history(&[], &theme);
        assert!(result.contains("No history"));
    }

    #[test]
    fn test_format_chain_history_with_entries() {
        let theme = Theme::plain();
        let entries = vec![ChainHistoryEntry {
            height: 42,
            transaction_type: "PUT".to_string(),
            data: None,
        }];
        let result = format_chain_history(&entries, &theme);
        assert!(result.contains("42"));
        assert!(result.contains("PUT"));
    }

    #[test]
    fn test_format_chain_similar_empty() {
        let theme = Theme::plain();
        let result = format_chain_similar(&[], &theme);
        assert!(result.contains("No similar"));
    }

    #[test]
    fn test_format_chain_drift() {
        let theme = Theme::plain();
        let drift = ChainDriftResult {
            from_height: 1,
            to_height: 10,
            total_drift: 0.5,
            avg_drift_per_block: 0.05,
            max_drift: 0.1,
        };
        let result = format_chain_drift(&drift, &theme);
        assert!(result.contains("From height"));
        assert!(result.contains("Total drift"));
    }

    #[test]
    fn test_format_chain_block() {
        let theme = Theme::plain();
        let info = ChainBlockInfo {
            height: 42,
            hash: "abc123".to_string(),
            prev_hash: "def456".to_string(),
            timestamp: 1704067200, // 2024-01-01 00:00:00 UTC
            transaction_count: 5,
            proposer: "node1".to_string(),
        };
        let result = format_chain_block(&info, &theme);
        assert!(result.contains("42"));
        assert!(result.contains("abc123"));
    }

    #[test]
    fn test_format_chain_codebook() {
        let theme = Theme::plain();
        let info = ChainCodebookInfo {
            scope: "global".to_string(),
            entry_count: 256,
            dimension: 128,
            domain: Some("embeddings".to_string()),
        };
        let result = format_chain_codebook(&info, &theme);
        assert!(result.contains("global"));
        assert!(result.contains("256"));
        assert!(result.contains("128"));
    }

    #[test]
    fn test_format_chain_transitions() {
        let theme = Theme::plain();
        let analysis = ChainTransitionAnalysis {
            total_transitions: 100,
            valid_transitions: 95,
            invalid_transitions: 5,
            avg_validity_score: 0.95,
        };
        let result = format_chain_transitions(&analysis, &theme);
        assert!(result.contains("100"));
        assert!(result.contains("95"));
    }
}
