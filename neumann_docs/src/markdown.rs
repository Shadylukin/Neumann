// SPDX-License-Identifier: MIT OR Apache-2.0
//! Markdown parsing utilities for extracting document metadata and links.

use regex::Regex;
use std::path::Path;

/// Parsed documentation file with extracted metadata.
#[derive(Debug, Clone)]
pub struct ParsedDoc {
    /// Relative path from docs root.
    pub path: String,
    /// Document title (first H1 heading).
    pub title: String,
    /// Category derived from path (e.g., "architecture", "concepts").
    pub category: String,
    /// Full content for embedding.
    pub content: String,
    /// File size in bytes.
    pub size: usize,
    /// Word count.
    pub word_count: usize,
    /// Links to other documents (relative paths).
    pub links: Vec<String>,
}

/// Extract the title from markdown content (first H1 heading).
fn extract_title(content: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            return title.trim().to_string();
        }
    }
    "Untitled".to_string()
}

/// Extract category from file path.
fn extract_category(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() >= 2 {
        parts[0].to_string()
    } else {
        "root".to_string()
    }
}

/// Count words in text content.
fn count_words(content: &str) -> usize {
    content.split_whitespace().filter(|w| !w.is_empty()).count()
}

/// Extract links to other markdown files from content.
///
/// Matches patterns like `[text](path.md)` and `[text](./path.md)`.
#[must_use]
pub fn extract_links(content: &str) -> Vec<String> {
    let re = Regex::new(r"\[.*?\]\(([^)]*\.md)\)").expect("valid regex");
    re.captures_iter(content)
        .filter_map(|cap| {
            let link = cap.get(1)?.as_str();
            // Skip external links
            if link.starts_with("http://") || link.starts_with("https://") {
                return None;
            }
            // Normalize relative paths
            let normalized = link.trim_start_matches("./").trim_start_matches("../");
            Some(normalized.to_string())
        })
        .collect()
}

/// Parse a markdown file and extract all metadata.
pub fn parse_markdown(path: &Path, docs_root: &Path) -> anyhow::Result<ParsedDoc> {
    let content = std::fs::read_to_string(path)?;
    let relative_path = path
        .strip_prefix(docs_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    let title = extract_title(&content);
    let category = extract_category(&relative_path);
    let size = content.len();
    let word_count = count_words(&content);
    let links = extract_links(&content);

    Ok(ParsedDoc {
        path: relative_path,
        title,
        category,
        content,
        size,
        word_count,
        links,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title() {
        let content = "# My Title\n\nSome content here.";
        assert_eq!(extract_title(content), "My Title");
    }

    #[test]
    fn test_extract_title_with_leading_whitespace() {
        let content = "  # Whitespace Title  \n\nContent.";
        assert_eq!(extract_title(content), "Whitespace Title");
    }

    #[test]
    fn test_extract_title_fallback() {
        let content = "No heading here.\nJust text.";
        assert_eq!(extract_title(content), "Untitled");
    }

    #[test]
    fn test_extract_category() {
        assert_eq!(extract_category("architecture/overview.md"), "architecture");
        assert_eq!(extract_category("concepts/sparse-vectors.md"), "concepts");
        assert_eq!(extract_category("introduction.md"), "root");
    }

    #[test]
    fn test_count_words() {
        assert_eq!(count_words("one two three"), 3);
        assert_eq!(count_words("  spaced   out  "), 2);
        assert_eq!(count_words(""), 0);
    }

    #[test]
    fn test_extract_links() {
        let content = r#"
See [overview](architecture/overview.md) and [concepts](./concepts/intro.md).
Also check [external](https://example.com/doc.md) which is ignored.
"#;
        let links = extract_links(content);
        assert_eq!(links.len(), 2);
        assert!(links.contains(&"architecture/overview.md".to_string()));
        assert!(links.contains(&"concepts/intro.md".to_string()));
    }

    #[test]
    fn test_extract_links_no_matches() {
        let content = "No links here at all.";
        let links = extract_links(content);
        assert!(links.is_empty());
    }

    #[test]
    fn test_extract_links_with_parent_path() {
        let content = "See [parent](../other/doc.md) for more.";
        let links = extract_links(content);
        assert_eq!(links, vec!["other/doc.md"]);
    }
}
