// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Progress spinners for long-running operations.

use crate::style::{styled, Theme};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Spinner for showing progress during operations.
pub struct Spinner {
    bar: ProgressBar,
    theme: Theme,
}

impl Spinner {
    /// Creates a new spinner with the given message.
    #[must_use]
    pub fn new(message: &str, theme: Theme) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::default_spinner()
                .tick_strings(&["*", "o", "O", "o"])
                .template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(Duration::from_millis(100));

        Self { bar, theme }
    }

    /// Updates the spinner message.
    #[allow(dead_code)]
    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    /// Finishes the spinner with a success message.
    pub fn finish_success(&self, message: &str) {
        self.bar.finish_with_message(format!(
            "{} {}",
            styled("[ok]", self.theme.success),
            message
        ));
    }

    /// Finishes the spinner with an error message.
    pub fn finish_error(&self, message: &str) {
        self.bar
            .finish_with_message(format!("{} {}", styled("[!!]", self.theme.error), message));
    }

    /// Finishes the spinner and clears it.
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        if !self.bar.is_finished() {
            self.bar.finish_and_clear();
        }
    }
}

/// Progress bar for showing completion percentage.
#[allow(dead_code)]
pub struct ProgressIndicator {
    bar: ProgressBar,
    theme: Theme,
}

#[allow(dead_code)]
impl ProgressIndicator {
    /// Creates a new progress bar with the given total.
    #[must_use]
    pub fn new(total: u64, message: &str, theme: Theme) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                .expect("valid template")
                .progress_chars("=>-"),
        );
        bar.set_message(message.to_string());

        Self { bar, theme }
    }

    /// Increments the progress by 1.
    pub fn inc(&self) {
        self.bar.inc(1);
    }

    /// Increments the progress by the given amount.
    pub fn inc_by(&self, delta: u64) {
        self.bar.inc(delta);
    }

    /// Sets the current position.
    pub fn set_position(&self, pos: u64) {
        self.bar.set_position(pos);
    }

    /// Updates the message.
    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    /// Finishes the progress bar with a success message.
    pub fn finish_success(&self, message: &str) {
        self.bar.finish_with_message(format!(
            "{} {}",
            styled("[ok]", self.theme.success),
            message
        ));
    }

    /// Finishes the progress bar with an error message.
    pub fn finish_error(&self, message: &str) {
        self.bar
            .finish_with_message(format!("{} {}", styled("[!!]", self.theme.error), message));
    }
}

impl Drop for ProgressIndicator {
    fn drop(&mut self) {
        if !self.bar.is_finished() {
            self.bar.finish_and_clear();
        }
    }
}

/// Creates a spinner for the given operation type.
pub fn operation_spinner(operation: &str, theme: &Theme) -> Spinner {
    let message = match operation.to_uppercase().as_str() {
        "LOAD" => "Loading snapshot...",
        "SAVE" => "Saving snapshot...",
        "SAVE COMPRESSED" => "Compressing and saving...",
        "CLUSTER CONNECT" => "Connecting to cluster...",
        "EMBED BUILD INDEX" => "Building HNSW index...",
        "BLOB PUT" => "Uploading...",
        "BLOB GC" => "Running garbage collection...",
        "GRAPH ALGORITHM PAGERANK" => "Computing PageRank...",
        "GRAPH ALGORITHM BETWEENNESS" => "Computing betweenness centrality...",
        "GRAPH ALGORITHM CLOSENESS" => "Computing closeness centrality...",
        "GRAPH ALGORITHM LOUVAIN" => "Detecting communities...",
        "CHAIN VERIFY" => "Verifying chain integrity...",
        _ => "Processing...",
    };
    Spinner::new(message, theme.clone())
}

/// Checks if an operation should show a spinner.
#[must_use]
pub fn needs_spinner(command: &str) -> bool {
    let upper = command.to_uppercase();
    upper.starts_with("LOAD ")
        || upper.starts_with("SAVE ")
        || upper.starts_with("CLUSTER CONNECT")
        || upper.contains("BUILD INDEX")
        || upper.starts_with("BLOB PUT")
        || upper.starts_with("BLOB GC")
        || upper.contains("ALGORITHM")
        || upper.starts_with("CHAIN VERIFY")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinner_creation() {
        let theme = Theme::plain();
        let spinner = Spinner::new("Testing...", theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_spinner_messages() {
        let theme = Theme::plain();
        let spinner = Spinner::new("Start", theme);
        spinner.set_message("Middle");
        spinner.finish_success("Done");
    }

    #[test]
    fn test_spinner_finish_error() {
        let theme = Theme::plain();
        let spinner = Spinner::new("Testing", theme);
        spinner.finish_error("Failed");
    }

    #[test]
    fn test_spinner_drop_unfinished() {
        let theme = Theme::plain();
        let _spinner = Spinner::new("Testing", theme);
        // Drop without finishing - should call finish_and_clear
    }

    #[test]
    fn test_progress_indicator() {
        let theme = Theme::plain();
        let progress = ProgressIndicator::new(100, "Processing", theme);
        progress.inc();
        progress.inc_by(10);
        progress.set_position(50);
        progress.finish_success("Complete");
    }

    #[test]
    fn test_progress_indicator_set_message() {
        let theme = Theme::plain();
        let progress = ProgressIndicator::new(100, "Start", theme);
        progress.set_message("Updated");
        progress.finish_success("Done");
    }

    #[test]
    fn test_progress_indicator_finish_error() {
        let theme = Theme::plain();
        let progress = ProgressIndicator::new(100, "Processing", theme);
        progress.finish_error("Error occurred");
    }

    #[test]
    fn test_progress_indicator_drop_unfinished() {
        let theme = Theme::plain();
        let _progress = ProgressIndicator::new(100, "Processing", theme);
        // Drop without finishing
    }

    #[test]
    fn test_operation_spinner_load() {
        let theme = Theme::plain();
        let spinner = operation_spinner("LOAD", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_save() {
        let theme = Theme::plain();
        let spinner = operation_spinner("SAVE", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_save_compressed() {
        let theme = Theme::plain();
        let spinner = operation_spinner("SAVE COMPRESSED", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_cluster_connect() {
        let theme = Theme::plain();
        let spinner = operation_spinner("CLUSTER CONNECT", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_embed_build_index() {
        let theme = Theme::plain();
        let spinner = operation_spinner("EMBED BUILD INDEX", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_blob_put() {
        let theme = Theme::plain();
        let spinner = operation_spinner("BLOB PUT", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_blob_gc() {
        let theme = Theme::plain();
        let spinner = operation_spinner("BLOB GC", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_pagerank() {
        let theme = Theme::plain();
        let spinner = operation_spinner("GRAPH ALGORITHM PAGERANK", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_betweenness() {
        let theme = Theme::plain();
        let spinner = operation_spinner("GRAPH ALGORITHM BETWEENNESS", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_closeness() {
        let theme = Theme::plain();
        let spinner = operation_spinner("GRAPH ALGORITHM CLOSENESS", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_louvain() {
        let theme = Theme::plain();
        let spinner = operation_spinner("GRAPH ALGORITHM LOUVAIN", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_chain_verify() {
        let theme = Theme::plain();
        let spinner = operation_spinner("CHAIN VERIFY", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_operation_spinner_unknown() {
        let theme = Theme::plain();
        let spinner = operation_spinner("UNKNOWN OPERATION", &theme);
        spinner.finish_and_clear();
    }

    #[test]
    fn test_needs_spinner() {
        assert!(needs_spinner("LOAD 'backup.bin'"));
        assert!(needs_spinner("SAVE 'backup.bin'"));
        assert!(needs_spinner("CLUSTER CONNECT 'node@addr'"));
        assert!(needs_spinner("EMBED BUILD INDEX"));
        assert!(needs_spinner("BLOB PUT file.bin"));
        assert!(needs_spinner("BLOB GC"));
        assert!(needs_spinner("GRAPH ALGORITHM PAGERANK"));
        assert!(needs_spinner("CHAIN VERIFY"));
        assert!(!needs_spinner("SELECT * FROM users"));
        assert!(!needs_spinner("NODE CREATE person {}"));
    }
}
