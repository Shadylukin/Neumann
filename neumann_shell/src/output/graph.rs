// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph result formatting (nodes, edges, paths, algorithms).

use crate::output::TableBuilder;
use crate::style::{styled, Icons, Theme};
use query_router::{
    AggregateResultValue, BatchOperationResult, CentralityResult, CommunityResult, ConstraintInfo,
    PageRankResult, PatternMatchResultValue,
};
use std::collections::HashMap;
use std::fmt::Write;

/// Formats graph nodes as a styled table.
#[must_use]
pub fn format_nodes(nodes: &[query_router::NodeResult], theme: &Theme, icons: &Icons) -> String {
    if nodes.is_empty() {
        return styled("(0 nodes)", theme.muted);
    }

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["ID", "Label", "Properties"]);

    for node in nodes {
        let props = format_properties(&node.properties, theme);
        builder.add_row(vec![
            styled(node.id, theme.id),
            styled(&node.label, theme.label),
            props,
        ]);
    }

    let label = if nodes.len() == 1 { "node" } else { "nodes" };
    format!(
        "{} {}\n{}",
        icons.node,
        styled("Nodes", theme.header),
        builder.build_with_count(theme, nodes.len(), label)
    )
}

/// Formats graph edges as a styled table.
#[must_use]
pub fn format_edges(edges: &[query_router::EdgeResult], theme: &Theme, icons: &Icons) -> String {
    if edges.is_empty() {
        return styled("(0 edges)", theme.muted);
    }

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["ID", "From", "To", "Type"]);

    for edge in edges {
        builder.add_row(vec![
            styled(edge.id, theme.id),
            styled(edge.from, theme.number),
            styled(edge.to, theme.number),
            styled(&edge.label, theme.label),
        ]);
    }

    let label = if edges.len() == 1 { "edge" } else { "edges" };
    format!(
        "{} {}\n{}",
        icons.edge,
        styled("Edges", theme.header),
        builder.build_with_count(theme, edges.len(), label)
    )
}

/// Formats a graph path.
#[must_use]
pub fn format_path(path: &[u64], theme: &Theme, icons: &Icons) -> String {
    if path.is_empty() {
        return styled("(no path found)", theme.muted);
    }

    let path_str: Vec<String> = path.iter().map(|id| styled(*id, theme.id)).collect();

    format!(
        "{} Path: {}",
        icons.arrow,
        path_str.join(&format!(" {} ", styled("->", theme.muted)))
    )
}

/// Formats node/edge properties.
fn format_properties(props: &HashMap<String, String>, theme: &Theme) -> String {
    if props.is_empty() {
        styled("{}", theme.muted)
    } else {
        let formatted: Vec<String> = props
            .iter()
            .map(|(k, v)| format!("{}: {}", styled(k, theme.keyword), styled(v, theme.string)))
            .collect();
        format!("{{{}}}", formatted.join(", "))
    }
}

/// Formats `PageRank` algorithm results.
#[must_use]
pub fn format_pagerank(result: &PageRankResult, theme: &Theme) -> String {
    if result.items.is_empty() {
        return styled("No PageRank results", theme.muted);
    }

    let mut output = format!("{}\n", styled("PageRank Results:", theme.header));

    let mut sorted = result.items.clone();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["Node", "Score"]);

    for item in &sorted {
        builder.add_row(vec![
            styled(item.node_id, theme.id),
            styled(format!("{:.6}", item.score), theme.number),
        ]);
    }

    output.push_str(&builder.build(theme));
    let _ = write!(
        output,
        "\n  Iterations: {}, Converged: {}",
        styled(result.iterations, theme.number),
        styled(result.converged, theme.keyword)
    );
    output
}

/// Formats centrality algorithm results.
#[must_use]
pub fn format_centrality(result: &CentralityResult, theme: &Theme) -> String {
    if result.items.is_empty() {
        return styled("No centrality results", theme.muted);
    }

    let mut output = format!(
        "{}\n",
        styled(
            format!("{:?} Results:", result.centrality_type),
            theme.header
        )
    );

    let mut sorted = result.items.clone();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["Node", "Score"]);

    for item in &sorted {
        builder.add_row(vec![
            styled(item.node_id, theme.id),
            styled(format!("{:.6}", item.score), theme.number),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

/// Formats community detection results.
#[must_use]
pub fn format_communities(result: &CommunityResult, theme: &Theme) -> String {
    if result.members.is_empty() {
        return styled("No communities found", theme.muted);
    }

    let mut output = format!("{}\n", styled("Communities:", theme.header));

    let mut sorted: Vec<_> = result.members.iter().collect();
    sorted.sort_by_key(|(id, _)| *id);

    for (community_id, nodes) in sorted {
        let node_list: Vec<String> = nodes.iter().map(|id| styled(*id, theme.id)).collect();
        let _ = writeln!(
            output,
            "  Community {}: [{}]",
            styled(community_id, theme.label),
            node_list.join(", ")
        );
    }

    if let Some(modularity) = result.modularity {
        let _ = write!(
            output,
            "  Modularity: {}",
            styled(format!("{modularity:.6}"), theme.number)
        );
    }
    output.trim_end().to_string()
}

/// Formats graph constraints.
#[must_use]
pub fn format_constraints(constraints: &[ConstraintInfo], theme: &Theme) -> String {
    if constraints.is_empty() {
        return styled("No constraints found", theme.muted);
    }

    let mut output = format!("{}\n", styled("Constraints:", theme.header));

    for c in constraints {
        let _ = writeln!(
            output,
            "  {} on {} property '{}' ({})",
            styled(&c.name, theme.keyword),
            styled(&c.target, theme.label),
            styled(&c.property, theme.string),
            styled(&c.constraint_type, theme.muted)
        );
    }
    output.trim_end().to_string()
}

/// Formats graph indexes.
#[must_use]
pub fn format_graph_indexes(indexes: &[String], theme: &Theme) -> String {
    if indexes.is_empty() {
        return styled("No indexes found", theme.muted);
    }

    let mut output = format!("{}\n", styled("Graph Indexes:", theme.header));
    for idx in indexes {
        let _ = writeln!(output, "  {}", styled(idx, theme.keyword));
    }
    output.trim_end().to_string()
}

/// Formats aggregate results.
#[must_use]
pub fn format_aggregate(agg: &AggregateResultValue, theme: &Theme) -> String {
    match agg {
        AggregateResultValue::Count(n) => {
            format!("Count: {}", styled(n, theme.number))
        },
        AggregateResultValue::Sum(v) => {
            format!("Sum: {}", styled(v, theme.number))
        },
        AggregateResultValue::Avg(v) => {
            format!("Avg: {}", styled(v, theme.number))
        },
        AggregateResultValue::Min(v) => {
            format!("Min: {}", styled(v, theme.number))
        },
        AggregateResultValue::Max(v) => {
            format!("Max: {}", styled(v, theme.number))
        },
    }
}

/// Formats batch operation results.
#[must_use]
pub fn format_batch_result(batch: &BatchOperationResult, theme: &Theme, icons: &Icons) -> String {
    let mut output = format!(
        "{} {}: {} affected",
        icons.success,
        styled(&batch.operation, theme.keyword),
        styled(batch.affected_count, theme.number)
    );

    if let Some(ids) = &batch.created_ids {
        if !ids.is_empty() {
            let id_str: Vec<String> = ids.iter().map(|id| styled(*id, theme.id)).collect();
            let _ = write!(output, " (IDs: {})", id_str.join(", "));
        }
    }
    output
}

/// Formats pattern match results.
#[must_use]
pub fn format_pattern_match(pm: &PatternMatchResultValue, theme: &Theme) -> String {
    if pm.matches.is_empty() {
        return styled("No pattern matches found", theme.muted);
    }

    let mut output = format!(
        "{}\n",
        styled(
            format!("Pattern Matches ({} found):", pm.stats.matches_found),
            theme.header
        )
    );

    for (i, m) in pm.matches.iter().enumerate() {
        let _ = writeln!(
            output,
            "  {}:",
            styled(format!("Match {}", i + 1), theme.label)
        );

        for (var, binding) in &m.bindings {
            let desc = match binding {
                query_router::BindingValue::Node { id, label } => {
                    format!(
                        "Node {} ({})",
                        styled(id, theme.id),
                        styled(label, theme.label)
                    )
                },
                query_router::BindingValue::Edge {
                    id,
                    edge_type,
                    from,
                    to,
                } => format!(
                    "Edge {} ({}) {} -> {}",
                    styled(id, theme.id),
                    styled(edge_type, theme.label),
                    styled(from, theme.number),
                    styled(to, theme.number)
                ),
                query_router::BindingValue::Path { nodes, length, .. } => {
                    format!(
                        "Path (length {}, {} nodes)",
                        styled(length, theme.number),
                        styled(nodes.len(), theme.number)
                    )
                },
            };
            let _ = writeln!(output, "    {}: {}", styled(var, theme.keyword), desc);
        }
    }

    if pm.stats.truncated {
        let _ = writeln!(output, "  {}", styled("(results truncated)", theme.warning));
    }
    output.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_nodes_empty() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_nodes(&[], &theme, &icons);
        assert!(result.contains("0 nodes"));
    }

    #[test]
    fn test_format_edges_empty() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_edges(&[], &theme, &icons);
        assert!(result.contains("0 edges"));
    }

    #[test]
    fn test_format_path_empty() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_path(&[], &theme, &icons);
        assert!(result.contains("no path"));
    }

    #[test]
    fn test_format_path_with_nodes() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_path(&[1, 2, 3], &theme, &icons);
        assert!(result.contains("1"));
        assert!(result.contains("2"));
        assert!(result.contains("3"));
        assert!(result.contains("->"));
    }

    #[test]
    fn test_format_properties_empty() {
        let theme = Theme::plain();
        let result = format_properties(&HashMap::new(), &theme);
        assert!(result.contains("{}"));
    }

    #[test]
    fn test_format_properties_with_data() {
        let theme = Theme::plain();
        let mut props = HashMap::new();
        props.insert("name".to_string(), "Alice".to_string());
        let result = format_properties(&props, &theme);
        assert!(result.contains("name"));
        assert!(result.contains("Alice"));
    }

    #[test]
    fn test_format_aggregate_count() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Count(42), &theme);
        assert!(result.contains("Count"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_format_constraints_empty() {
        let theme = Theme::plain();
        let result = format_constraints(&[], &theme);
        assert!(result.contains("No constraints"));
    }

    #[test]
    fn test_format_graph_indexes_empty() {
        let theme = Theme::plain();
        let result = format_graph_indexes(&[], &theme);
        assert!(result.contains("No indexes"));
    }
}
