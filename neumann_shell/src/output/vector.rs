// SPDX-License-Identifier: MIT OR Apache-2.0
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_format_unified_empty() {
        let theme = Theme::plain();
        let unified = query_router::UnifiedResult {
            description: "Test".to_string(),
            items: vec![],
        };
        let result = format_unified(&unified, &theme);
        assert!(result.contains("no items"));
    }

    // Note: test_format_unified_with_items is omitted because UnifiedItem
    // is not publicly exported from query_router. The empty case is tested above.
    // Full testing is done via integration tests.
}
