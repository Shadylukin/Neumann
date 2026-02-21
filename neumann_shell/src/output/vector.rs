// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Vector/embedding result formatting.

#![allow(clippy::format_push_string)]

use crate::output::TableBuilder;
use crate::style::{styled, Theme};
use std::fmt::Write;

/// Formats similar embedding results.
#[must_use]
pub fn format_similar(results: &[query_router::SimilarResult], theme: &Theme) -> String {
    if results.is_empty() {
        return styled("(no similar embeddings)", theme.muted);
    }

    let mut output = format!("{}\n", styled("Similar:", theme.header));

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["#", "Key", "Similarity"]);

    for (i, r) in results.iter().enumerate() {
        builder.add_row(vec![
            styled(i + 1, theme.muted),
            styled(&r.key, theme.id),
            styled(format!("{:.4}", r.score), theme.number),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

/// Formats unified query results.
#[must_use]
pub fn format_unified(unified: &query_router::UnifiedResult, theme: &Theme) -> String {
    let mut output = String::new();

    output.push_str(&styled(&unified.description, theme.header));
    output.push('\n');

    if unified.items.is_empty() {
        output.push_str(&styled("(no items)\n", theme.muted));
        return output;
    }

    for item in &unified.items {
        let _ = write!(
            output,
            "  [{}] ({})",
            styled(&item.id, theme.id),
            styled(&item.source, theme.label)
        );

        if !item.data.is_empty() {
            let fields: Vec<String> = item
                .data
                .iter()
                .map(|(k, v)| format!("{}: {}", styled(k, theme.keyword), styled(v, theme.string)))
                .collect();
            output.push_str(&format!(" {{{}}}", fields.join(", ")));
        }

        if let Some(score) = item.score {
            let _ = write!(
                output,
                " [score: {}]",
                styled(format!("{score:.4}"), theme.number)
            );
        }

        if let Some(ref emb) = item.embedding {
            let _ = write!(output, " [embedding: {}D]", styled(emb.len(), theme.muted));
        }

        output.push('\n');
    }

    let _ = writeln!(
        output,
        "({} items)",
        styled(unified.items.len(), theme.number)
    );
    output
}

/// Formats spatial query results.
#[must_use]
pub fn format_spatial(results: &[query_router::SpatialResult], theme: &Theme) -> String {
    if results.is_empty() {
        return styled("(no spatial results)", theme.muted);
    }

    let mut output = format!("{}\n", styled("Spatial:", theme.header));

    let mut builder = TableBuilder::new();
    builder.add_header(vec!["#", "Key", "Distance", "X", "Y", "W", "H"]);

    for (i, r) in results.iter().enumerate() {
        builder.add_row(vec![
            styled(i + 1, theme.muted),
            styled(&r.key, theme.id),
            styled(format!("{:.4}", r.distance), theme.number),
            styled(format!("{:.1}", r.x), theme.number),
            styled(format!("{:.1}", r.y), theme.number),
            styled(format!("{:.1}", r.width), theme.number),
            styled(format!("{:.1}", r.height), theme.number),
        ]);
    }

    output.push_str(&builder.build(theme));
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_unified::UnifiedItem;

    #[test]
    fn test_format_similar_empty() {
        let theme = Theme::plain();
        let result = format_similar(&[], &theme);
        assert!(result.contains("no similar"));
    }

    #[test]
    fn test_format_similar_with_results() {
        let theme = Theme::plain();
        let results = vec![
            query_router::SimilarResult {
                key: "doc1".to_string(),
                score: 0.95,
            },
            query_router::SimilarResult {
                key: "doc2".to_string(),
                score: 0.85,
            },
        ];
        let result = format_similar(&results, &theme);
        assert!(result.contains("Similar:"));
        assert!(result.contains("doc1"));
        assert!(result.contains("doc2"));
        assert!(result.contains("0.95"));
        assert!(result.contains("0.85"));
    }

    #[test]
    fn test_format_similar_single() {
        let theme = Theme::plain();
        let results = vec![query_router::SimilarResult {
            key: "single_doc".to_string(),
            score: 0.99,
        }];
        let result = format_similar(&results, &theme);
        assert!(result.contains("single_doc"));
        assert!(result.contains("0.99"));
    }

    #[test]
    fn test_format_unified_empty() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Test".to_string(),
            items: vec![],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("no items"));
    }

    #[test]
    fn test_format_unified_with_items() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Test Results".to_string(),
            items: vec![UnifiedItem {
                id: "item1".to_string(),
                source: "relational".to_string(),
                data: std::collections::HashMap::new(),
                score: None,
                embedding: None,
            }],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("Test Results"));
        assert!(result.contains("item1"));
        assert!(result.contains("relational"));
        assert!(result.contains("1 items"));
    }

    #[test]
    fn test_format_unified_with_score() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Scored Results".to_string(),
            items: vec![UnifiedItem {
                id: "item1".to_string(),
                source: "vector".to_string(),
                data: std::collections::HashMap::new(),
                score: Some(0.95),
                embedding: None,
            }],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("score"));
        assert!(result.contains("0.95"));
    }

    #[test]
    fn test_format_unified_with_embedding() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Embedded Results".to_string(),
            items: vec![UnifiedItem {
                id: "item1".to_string(),
                source: "vector".to_string(),
                data: std::collections::HashMap::new(),
                score: None,
                embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            }],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("embedding"));
        assert!(result.contains("4D"));
    }

    #[test]
    fn test_format_unified_with_data() {
        let theme = Theme::plain();
        let mut data = std::collections::HashMap::new();
        data.insert("name".to_string(), "Alice".to_string());
        data.insert("age".to_string(), "30".to_string());
        let unified = query_router::UnifiedResult {
            description: "Data Results".to_string(),
            items: vec![UnifiedItem {
                id: "item1".to_string(),
                source: "relational".to_string(),
                data,
                score: None,
                embedding: None,
            }],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("name"));
        assert!(result.contains("Alice"));
    }

    #[test]
    fn test_format_unified_multiple_items() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Multiple Results".to_string(),
            items: vec![
                UnifiedItem {
                    id: "item1".to_string(),
                    source: "relational".to_string(),
                    data: std::collections::HashMap::new(),
                    score: None,
                    embedding: None,
                },
                UnifiedItem {
                    id: "item2".to_string(),
                    source: "graph".to_string(),
                    data: std::collections::HashMap::new(),
                    score: Some(0.8),
                    embedding: None,
                },
            ],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("item1"));
        assert!(result.contains("item2"));
        assert!(result.contains("2 items"));
    }

    #[test]
    fn test_format_spatial_empty() {
        let theme = Theme::plain();
        let result = format_spatial(&[], &theme);
        assert!(result.contains("no spatial results"));
    }

    #[test]
    fn test_format_spatial_with_results() {
        let theme = Theme::plain();
        let results = vec![
            query_router::SpatialResult {
                key: "park".to_string(),
                distance: 1.5,
                x: 10.0,
                y: 20.0,
                width: 5.0,
                height: 3.0,
            },
            query_router::SpatialResult {
                key: "lake".to_string(),
                distance: 4.2,
                x: 30.0,
                y: 40.0,
                width: 10.0,
                height: 8.0,
            },
        ];
        let result = format_spatial(&results, &theme);
        assert!(result.contains("Spatial:"));
        assert!(result.contains("park"));
        assert!(result.contains("lake"));
        assert!(result.contains("1.5000"));
        assert!(result.contains("4.2000"));
    }

    #[test]
    fn test_format_spatial_single() {
        let theme = Theme::plain();
        let results = vec![query_router::SpatialResult {
            key: "point".to_string(),
            distance: 0.0,
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        }];
        let result = format_spatial(&results, &theme);
        assert!(result.contains("point"));
        assert!(result.contains("0.0000"));
    }
}
