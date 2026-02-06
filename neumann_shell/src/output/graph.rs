// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
    use query_router::{
        BindingValue, CentralityItem, CentralityType, PageRankItem, PatternMatchBinding,
        PatternMatchStatsValue,
    };

    #[test]
    fn test_format_nodes_empty() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_nodes(&[], &theme, &icons);
        assert!(result.contains("0 nodes"));
    }

    #[test]
    fn test_format_nodes_single() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let mut props = HashMap::new();
        props.insert("name".to_string(), "Alice".to_string());
        let nodes = vec![query_router::NodeResult {
            id: 1,
            label: "person".to_string(),
            properties: props,
        }];
        let result = format_nodes(&nodes, &theme, &icons);
        assert!(result.contains("1 node"));
        assert!(result.contains("person"));
        assert!(result.contains("Alice"));
    }

    #[test]
    fn test_format_nodes_multiple() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let nodes = vec![
            query_router::NodeResult {
                id: 1,
                label: "person".to_string(),
                properties: HashMap::new(),
            },
            query_router::NodeResult {
                id: 2,
                label: "company".to_string(),
                properties: HashMap::new(),
            },
        ];
        let result = format_nodes(&nodes, &theme, &icons);
        assert!(result.contains("2 nodes"));
    }

    #[test]
    fn test_format_edges_empty() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_edges(&[], &theme, &icons);
        assert!(result.contains("0 edges"));
    }

    #[test]
    fn test_format_edges_single() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let edges = vec![query_router::EdgeResult {
            id: 1,
            from: 10,
            to: 20,
            label: "knows".to_string(),
        }];
        let result = format_edges(&edges, &theme, &icons);
        assert!(result.contains("1 edge"));
        assert!(result.contains("knows"));
        assert!(result.contains("10"));
        assert!(result.contains("20"));
    }

    #[test]
    fn test_format_edges_multiple() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let edges = vec![
            query_router::EdgeResult {
                id: 1,
                from: 10,
                to: 20,
                label: "knows".to_string(),
            },
            query_router::EdgeResult {
                id: 2,
                from: 20,
                to: 30,
                label: "works_at".to_string(),
            },
        ];
        let result = format_edges(&edges, &theme, &icons);
        assert!(result.contains("2 edges"));
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
        assert!(result.contains('1'));
        assert!(result.contains('2'));
        assert!(result.contains('3'));
        assert!(result.contains("->"));
    }

    #[test]
    fn test_format_path_single_node() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let result = format_path(&[42], &theme, &icons);
        assert!(result.contains("42"));
        assert!(result.contains("Path"));
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
    fn test_format_properties_multiple() {
        let theme = Theme::plain();
        let mut props = HashMap::new();
        props.insert("name".to_string(), "Alice".to_string());
        props.insert("age".to_string(), "30".to_string());
        let result = format_properties(&props, &theme);
        assert!(result.contains("name"));
        assert!(result.contains("age"));
    }

    #[test]
    fn test_format_aggregate_count() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Count(42), &theme);
        assert!(result.contains("Count"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_format_aggregate_sum() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Sum(123.5), &theme);
        assert!(result.contains("Sum"));
        assert!(result.contains("123.5"));
    }

    #[test]
    fn test_format_aggregate_avg() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Avg(45.67), &theme);
        assert!(result.contains("Avg"));
        assert!(result.contains("45.67"));
    }

    #[test]
    fn test_format_aggregate_min() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Min(1.0), &theme);
        assert!(result.contains("Min"));
        assert!(result.contains('1'));
    }

    #[test]
    fn test_format_aggregate_max() {
        let theme = Theme::plain();
        let result = format_aggregate(&AggregateResultValue::Max(999.0), &theme);
        assert!(result.contains("Max"));
        assert!(result.contains("999"));
    }

    #[test]
    fn test_format_constraints_empty() {
        let theme = Theme::plain();
        let result = format_constraints(&[], &theme);
        assert!(result.contains("No constraints"));
    }

    #[test]
    fn test_format_constraints_with_data() {
        let theme = Theme::plain();
        let constraints = vec![ConstraintInfo {
            name: "unique_email".to_string(),
            target: "user".to_string(),
            property: "email".to_string(),
            constraint_type: "unique".to_string(),
        }];
        let result = format_constraints(&constraints, &theme);
        assert!(result.contains("Constraints"));
        assert!(result.contains("unique_email"));
        assert!(result.contains("user"));
        assert!(result.contains("email"));
    }

    #[test]
    fn test_format_constraints_multiple() {
        let theme = Theme::plain();
        let constraints = vec![
            ConstraintInfo {
                name: "unique_email".to_string(),
                target: "user".to_string(),
                property: "email".to_string(),
                constraint_type: "unique".to_string(),
            },
            ConstraintInfo {
                name: "not_null_name".to_string(),
                target: "user".to_string(),
                property: "name".to_string(),
                constraint_type: "not_null".to_string(),
            },
        ];
        let result = format_constraints(&constraints, &theme);
        assert!(result.contains("unique_email"));
        assert!(result.contains("not_null_name"));
    }

    #[test]
    fn test_format_graph_indexes_empty() {
        let theme = Theme::plain();
        let result = format_graph_indexes(&[], &theme);
        assert!(result.contains("No indexes"));
    }

    #[test]
    fn test_format_graph_indexes_with_data() {
        let theme = Theme::plain();
        let indexes = vec!["idx_user_email".to_string(), "idx_order_date".to_string()];
        let result = format_graph_indexes(&indexes, &theme);
        assert!(result.contains("Graph Indexes"));
        assert!(result.contains("idx_user_email"));
        assert!(result.contains("idx_order_date"));
    }

    #[test]
    fn test_format_pagerank_empty() {
        let theme = Theme::plain();
        let result = PageRankResult {
            items: vec![],
            iterations: 0,
            convergence: 0.0,
            converged: false,
        };
        let output = format_pagerank(&result, &theme);
        assert!(output.contains("No PageRank results"));
    }

    #[test]
    fn test_format_pagerank_with_data() {
        let theme = Theme::plain();
        let result = PageRankResult {
            items: vec![
                PageRankItem {
                    node_id: 1,
                    score: 0.5,
                },
                PageRankItem {
                    node_id: 2,
                    score: 0.3,
                },
                PageRankItem {
                    node_id: 3,
                    score: 0.2,
                },
            ],
            iterations: 10,
            convergence: 0.001,
            converged: true,
        };
        let output = format_pagerank(&result, &theme);
        assert!(output.contains("PageRank Results"));
        assert!(output.contains("Iterations: 10"));
        assert!(output.contains("Converged: true"));
    }

    #[test]
    fn test_format_centrality_empty() {
        let theme = Theme::plain();
        let result = CentralityResult {
            centrality_type: CentralityType::Betweenness,
            items: vec![],
            iterations: None,
            converged: None,
            sample_count: None,
        };
        let output = format_centrality(&result, &theme);
        assert!(output.contains("No centrality results"));
    }

    #[test]
    fn test_format_centrality_with_data() {
        let theme = Theme::plain();
        let result = CentralityResult {
            centrality_type: CentralityType::Betweenness,
            items: vec![
                CentralityItem {
                    node_id: 1,
                    score: 0.8,
                },
                CentralityItem {
                    node_id: 2,
                    score: 0.5,
                },
            ],
            iterations: Some(10),
            converged: Some(true),
            sample_count: None,
        };
        let output = format_centrality(&result, &theme);
        assert!(output.contains("Betweenness"));
    }

    #[test]
    fn test_format_communities_empty() {
        let theme = Theme::plain();
        let result = CommunityResult {
            items: vec![],
            members: HashMap::new(),
            community_count: 0,
            modularity: None,
            passes: None,
            iterations: None,
        };
        let output = format_communities(&result, &theme);
        assert!(output.contains("No communities"));
    }

    #[test]
    fn test_format_communities_with_data() {
        let theme = Theme::plain();
        let mut members = HashMap::new();
        members.insert(0, vec![1, 2, 3]);
        members.insert(1, vec![4, 5]);
        let result = CommunityResult {
            items: vec![],
            members,
            community_count: 2,
            modularity: Some(0.75),
            passes: None,
            iterations: None,
        };
        let output = format_communities(&result, &theme);
        assert!(output.contains("Communities"));
        assert!(output.contains("Community 0"));
        assert!(output.contains("Community 1"));
        assert!(output.contains("Modularity: 0.75"));
    }

    #[test]
    fn test_format_communities_no_modularity() {
        let theme = Theme::plain();
        let mut members = HashMap::new();
        members.insert(0, vec![1, 2]);
        let result = CommunityResult {
            items: vec![],
            members,
            community_count: 1,
            modularity: None,
            passes: None,
            iterations: None,
        };
        let output = format_communities(&result, &theme);
        assert!(output.contains("Communities"));
        assert!(!output.contains("Modularity"));
    }

    #[test]
    fn test_format_batch_result() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let batch = BatchOperationResult {
            operation: "CREATE".to_string(),
            affected_count: 5,
            created_ids: None,
        };
        let output = format_batch_result(&batch, &theme, &icons);
        assert!(output.contains("CREATE"));
        assert!(output.contains("5 affected"));
    }

    #[test]
    fn test_format_batch_result_with_ids() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let batch = BatchOperationResult {
            operation: "CREATE".to_string(),
            affected_count: 3,
            created_ids: Some(vec![10, 11, 12]),
        };
        let output = format_batch_result(&batch, &theme, &icons);
        assert!(output.contains("CREATE"));
        assert!(output.contains("3 affected"));
        assert!(output.contains("IDs"));
        assert!(output.contains("10"));
        assert!(output.contains("11"));
        assert!(output.contains("12"));
    }

    #[test]
    fn test_format_batch_result_empty_ids() {
        let theme = Theme::plain();
        let icons = Icons::ASCII;
        let batch = BatchOperationResult {
            operation: "DELETE".to_string(),
            affected_count: 0,
            created_ids: Some(vec![]),
        };
        let output = format_batch_result(&batch, &theme, &icons);
        assert!(output.contains("DELETE"));
        assert!(!output.contains("IDs"));
    }

    #[test]
    fn test_format_pattern_match_empty() {
        let theme = Theme::plain();
        let pm = PatternMatchResultValue {
            matches: vec![],
            stats: PatternMatchStatsValue {
                matches_found: 0,
                nodes_evaluated: 0,
                edges_evaluated: 0,
                truncated: false,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("No pattern matches"));
    }

    #[test]
    fn test_format_pattern_match_with_node() {
        let theme = Theme::plain();
        let mut bindings = HashMap::new();
        bindings.insert(
            "n".to_string(),
            BindingValue::Node {
                id: 1,
                label: "person".to_string(),
            },
        );
        let pm = PatternMatchResultValue {
            matches: vec![PatternMatchBinding { bindings }],
            stats: PatternMatchStatsValue {
                matches_found: 1,
                nodes_evaluated: 10,
                edges_evaluated: 5,
                truncated: false,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("Pattern Matches"));
        assert!(output.contains("Match 1"));
        assert!(output.contains("Node"));
        assert!(output.contains("person"));
    }

    #[test]
    fn test_format_pattern_match_with_edge() {
        let theme = Theme::plain();
        let mut bindings = HashMap::new();
        bindings.insert(
            "e".to_string(),
            BindingValue::Edge {
                id: 1,
                edge_type: "knows".to_string(),
                from: 10,
                to: 20,
            },
        );
        let pm = PatternMatchResultValue {
            matches: vec![PatternMatchBinding { bindings }],
            stats: PatternMatchStatsValue {
                matches_found: 1,
                nodes_evaluated: 10,
                edges_evaluated: 5,
                truncated: false,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("Edge"));
        assert!(output.contains("knows"));
        assert!(output.contains("10"));
        assert!(output.contains("20"));
    }

    #[test]
    fn test_format_pattern_match_with_path() {
        let theme = Theme::plain();
        let mut bindings = HashMap::new();
        bindings.insert(
            "p".to_string(),
            BindingValue::Path {
                nodes: vec![1, 2, 3],
                edges: vec![10, 11],
                length: 2,
            },
        );
        let pm = PatternMatchResultValue {
            matches: vec![PatternMatchBinding { bindings }],
            stats: PatternMatchStatsValue {
                matches_found: 1,
                nodes_evaluated: 10,
                edges_evaluated: 5,
                truncated: false,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("Path"));
        assert!(output.contains("length 2"));
        assert!(output.contains("3 nodes"));
    }

    #[test]
    fn test_format_pattern_match_truncated() {
        let theme = Theme::plain();
        let pm = PatternMatchResultValue {
            matches: vec![PatternMatchBinding {
                bindings: HashMap::new(),
            }],
            stats: PatternMatchStatsValue {
                matches_found: 100,
                nodes_evaluated: 1000,
                edges_evaluated: 500,
                truncated: true,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("truncated"));
    }

    #[test]
    fn test_format_pattern_match_multiple() {
        let theme = Theme::plain();
        let mut bindings1 = HashMap::new();
        bindings1.insert(
            "n".to_string(),
            BindingValue::Node {
                id: 1,
                label: "person".to_string(),
            },
        );
        let mut bindings2 = HashMap::new();
        bindings2.insert(
            "n".to_string(),
            BindingValue::Node {
                id: 2,
                label: "company".to_string(),
            },
        );
        let pm = PatternMatchResultValue {
            matches: vec![
                PatternMatchBinding {
                    bindings: bindings1,
                },
                PatternMatchBinding {
                    bindings: bindings2,
                },
            ],
            stats: PatternMatchStatsValue {
                matches_found: 2,
                nodes_evaluated: 20,
                edges_evaluated: 10,
                truncated: false,
            },
        };
        let output = format_pattern_match(&pm, &theme);
        assert!(output.contains("Match 1"));
        assert!(output.contains("Match 2"));
    }
}
