// SPDX-License-Identifier: MIT OR Apache-2.0
//! Output formatting for shell results.

#![allow(clippy::format_push_string)]
//!
//! Provides styled, Unicode-aware formatting for all query result types.

mod blob;
mod chain;
mod graph;
mod help;
mod rows;
mod table;
mod vector;

pub use blob::{format_artifact_info, format_artifact_list, format_blob, format_blob_stats};
pub use chain::format_chain_result;
pub use graph::{
    format_aggregate, format_batch_result, format_centrality, format_communities,
    format_constraints, format_edges, format_graph_indexes, format_nodes, format_pagerank,
    format_path, format_pattern_match,
};
pub use help::format_help;
pub use rows::format_rows;
pub use table::TableBuilder;
pub use vector::{format_similar, format_unified};

use crate::style::{Icons, Theme};
use query_router::{CheckpointInfo, QueryResult};

/// Formats a query result for display with the given theme and icons.
#[must_use]
pub fn format_result(result: &QueryResult, theme: &Theme, icons: &Icons) -> String {
    match result {
        QueryResult::Empty => format_ok(theme, icons),
        QueryResult::Value(s) => s.clone(),
        QueryResult::Count(n) => format_count(*n, theme),
        QueryResult::Ids(ids) => format_ids(ids, theme),
        QueryResult::Rows(rows) => format_rows(rows, theme),
        QueryResult::Nodes(nodes) => format_nodes(nodes, theme, icons),
        QueryResult::Edges(edges) => format_edges(edges, theme, icons),
        QueryResult::Path(path) => format_path(path, theme, icons),
        QueryResult::Similar(results) => format_similar(results, theme),
        QueryResult::Unified(unified) => format_unified(unified, theme),
        QueryResult::TableList(tables) => format_table_list(tables, theme),
        QueryResult::Blob(data) => format_blob(data, theme),
        QueryResult::ArtifactInfo(info) => format_artifact_info(info, theme),
        QueryResult::ArtifactList(ids) => format_artifact_list(ids, theme),
        QueryResult::BlobStats(stats) => format_blob_stats(stats, theme),
        QueryResult::CheckpointList(checkpoints) => format_checkpoint_list(checkpoints, theme),
        QueryResult::Chain(chain) => format_chain_result(chain, theme, icons),
        QueryResult::PageRank(ref result) => format_pagerank(result, theme),
        QueryResult::Centrality(ref result) => format_centrality(result, theme),
        QueryResult::Communities(ref result) => format_communities(result, theme),
        QueryResult::Constraints(constraints) => format_constraints(constraints, theme),
        QueryResult::GraphIndexes(indexes) => format_graph_indexes(indexes, theme),
        QueryResult::Aggregate(agg) => format_aggregate(agg, theme),
        QueryResult::BatchResult(batch) => format_batch_result(batch, theme, icons),
        QueryResult::PatternMatch(pm) => format_pattern_match(pm, theme),
    }
}

/// Formats an OK result with success styling.
fn format_ok(theme: &Theme, icons: &Icons) -> String {
    use crate::style::styled;
    format!("{} {}", icons.success, styled("OK", theme.success))
}

/// Formats a count result.
fn format_count(n: usize, theme: &Theme) -> String {
    use crate::style::styled;
    if n == 1 {
        format!("{} row affected", styled("1", theme.number))
    } else {
        format!("{} rows affected", styled(n, theme.number))
    }
}

/// Formats a list of IDs.
fn format_ids(ids: &[u64], theme: &Theme) -> String {
    use crate::style::styled;
    if ids.is_empty() {
        styled("(no results)", theme.muted)
    } else if ids.len() == 1 {
        format!("ID: {}", styled(ids[0], theme.id))
    } else {
        let id_list: Vec<String> = ids.iter().map(|id| styled(*id, theme.id)).collect();
        format!("IDs: {}", id_list.join(", "))
    }
}

/// Formats a list of table names.
fn format_table_list(tables: &[String], theme: &Theme) -> String {
    use crate::style::styled;
    if tables.is_empty() {
        styled("No tables found.", theme.muted)
    } else {
        let mut output = styled("Tables:", theme.header);
        output.push('\n');
        for table in tables {
            output.push_str(&format!("  {}\n", styled(table, theme.keyword)));
        }
        output.trim_end().to_string()
    }
}

/// Formats a checkpoint list.
fn format_checkpoint_list(checkpoints: &[CheckpointInfo], theme: &Theme) -> String {
    use crate::style::styled;
    use std::fmt::Write;

    if checkpoints.is_empty() {
        return styled("No checkpoints found", theme.muted);
    }

    let mut output = String::new();
    let _ = writeln!(output, "{}", styled("Checkpoints:", theme.header));

    // Build table
    let mut builder = TableBuilder::new();
    builder.add_header(vec!["ID", "Name", "Created", "Type"]);

    for cp in checkpoints {
        let created = format_timestamp(cp.created_at);
        let cp_type = if cp.is_auto { "auto" } else { "manual" };
        builder.add_row(vec![
            cp.id[..cp.id.len().min(36)].to_string(),
            cp.name[..cp.name.len().min(28)].to_string(),
            created,
            cp_type.to_string(),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

/// Formats a Unix timestamp as relative time.
pub fn format_timestamp(unix_secs: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if unix_secs == 0 {
        return "unknown".to_string();
    }

    let diff = now.saturating_sub(unix_secs);

    if diff < 60 {
        format!("{diff}s ago")
    } else if diff < 3600 {
        let mins = diff / 60;
        format!("{mins}m ago")
    } else if diff < 86400 {
        let hours = diff / 3600;
        format!("{hours}h ago")
    } else {
        let days = diff / 86400;
        format!("{days}d ago")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_ok() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_ok(&theme, &icons);
        assert!(result.contains("OK"));
    }

    #[test]
    fn test_format_count_singular() {
        let theme = Theme::plain();
        let result = format_count(1, &theme);
        assert!(result.contains("1 row affected"));
    }

    #[test]
    fn test_format_count_plural() {
        let theme = Theme::plain();
        let result = format_count(5, &theme);
        assert!(result.contains("5 rows affected"));
    }

    #[test]
    fn test_format_ids_empty() {
        let theme = Theme::plain();
        let result = format_ids(&[], &theme);
        assert!(result.contains("no results"));
    }

    #[test]
    fn test_format_ids_single() {
        let theme = Theme::plain();
        let result = format_ids(&[42], &theme);
        assert!(result.contains("ID:"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_format_ids_multiple() {
        let theme = Theme::plain();
        let result = format_ids(&[1, 2, 3], &theme);
        assert!(result.contains("IDs:"));
    }

    #[test]
    fn test_format_table_list_empty() {
        let theme = Theme::plain();
        let result = format_table_list(&[], &theme);
        assert!(result.contains("No tables"));
    }

    #[test]
    fn test_format_table_list_with_tables() {
        let theme = Theme::plain();
        let result = format_table_list(&["users".to_string(), "orders".to_string()], &theme);
        assert!(result.contains("Tables:"));
        assert!(result.contains("users"));
        assert!(result.contains("orders"));
    }

    #[test]
    fn test_format_timestamp_recent() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let result = format_timestamp(now - 30);
        assert!(result.contains("s ago"));
    }

    #[test]
    fn test_format_timestamp_minutes() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let result = format_timestamp(now - 300);
        assert!(result.contains("m ago"));
    }

    #[test]
    fn test_format_timestamp_unknown() {
        let result = format_timestamp(0);
        assert_eq!(result, "unknown");
    }
}
