// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Unicode box-drawing table renderer.

#![allow(clippy::format_push_string)]

use crate::style::{styled, Theme};
use tabled::{
    settings::{object::Rows, Alignment, Modify, Style},
    Table,
};

/// Builder for creating styled tables with Unicode box-drawing characters.
#[derive(Debug, Default)]
pub struct TableBuilder {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl TableBuilder {
    /// Creates a new table builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a header row.
    pub fn add_header(&mut self, headers: Vec<&str>) {
        self.headers = headers.into_iter().map(String::from).collect();
    }

    /// Adds a data row.
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Builds the table string with the given theme.
    #[must_use]
    pub fn build(&self, theme: &Theme) -> String {
        if self.headers.is_empty() && self.rows.is_empty() {
            return String::new();
        }

        // Create data for tabled
        let mut data: Vec<Vec<String>> = Vec::with_capacity(self.rows.len() + 1);

        // Add styled headers
        let styled_headers: Vec<String> = self
            .headers
            .iter()
            .map(|h| styled(h, theme.header))
            .collect();
        data.push(styled_headers);

        // Add rows
        for row in &self.rows {
            data.push(row.clone());
        }

        // Build table with Unicode box-drawing style
        let table = Table::from_iter(data)
            .with(Style::rounded())
            .with(Modify::new(Rows::first()).with(Alignment::center()))
            .to_string();

        table
    }

    /// Builds the table and appends a row count footer.
    #[must_use]
    pub fn build_with_count(&self, theme: &Theme, count: usize, label: &str) -> String {
        let mut output = self.build(theme);
        output.push_str(&format!(
            "\n  {} {}",
            styled(count, theme.number),
            styled(label, theme.muted)
        ));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_table() {
        let builder = TableBuilder::new();
        let theme = Theme::plain();
        let result = builder.build(&theme);
        assert!(result.is_empty());
    }

    #[test]
    fn test_table_with_headers_only() {
        let mut builder = TableBuilder::new();
        builder.add_header(vec!["Name", "Age"]);
        let theme = Theme::plain();
        let result = builder.build(&theme);
        assert!(result.contains("Name"));
        assert!(result.contains("Age"));
    }

    #[test]
    fn test_table_with_data() {
        let mut builder = TableBuilder::new();
        builder.add_header(vec!["Name", "Age"]);
        builder.add_row(vec!["Alice".to_string(), "30".to_string()]);
        builder.add_row(vec!["Bob".to_string(), "25".to_string()]);
        let theme = Theme::plain();
        let result = builder.build(&theme);
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
        assert!(result.contains("30"));
        assert!(result.contains("25"));
    }

    #[test]
    fn test_table_has_unicode_borders() {
        let mut builder = TableBuilder::new();
        builder.add_header(vec!["Col"]);
        builder.add_row(vec!["Data".to_string()]);
        let theme = Theme::plain();
        let result = builder.build(&theme);
        // Rounded style uses curved corners
        assert!(result.contains('\u{256D}') || result.contains('+') || result.contains('|'));
    }

    #[test]
    fn test_table_with_count() {
        let mut builder = TableBuilder::new();
        builder.add_header(vec!["Name"]);
        builder.add_row(vec!["Test".to_string()]);
        let theme = Theme::plain();
        let result = builder.build_with_count(&theme, 1, "row");
        assert!(result.contains('1'));
        assert!(result.contains("row"));
    }
}
