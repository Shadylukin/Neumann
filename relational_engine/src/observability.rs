//! Observability infrastructure for relational engine.
//!
//! Provides structured logging, metrics, and query tracing for monitoring
//! query performance and detecting missing indexes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use dashmap::DashMap;
use tracing::warn;

/// Query execution metrics for observability.
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Table name.
    pub table: String,
    /// Operation type (select, insert, update, delete, join, etc.).
    pub operation: &'static str,
    /// Number of rows scanned.
    pub rows_scanned: usize,
    /// Number of rows returned.
    pub rows_returned: usize,
    /// Index used for the query, if any.
    pub index_used: Option<String>,
    /// Query execution duration.
    pub duration: Duration,
}

impl QueryMetrics {
    /// Creates new query metrics.
    #[must_use]
    pub fn new(table: impl Into<String>, operation: &'static str) -> Self {
        Self {
            table: table.into(),
            operation,
            rows_scanned: 0,
            rows_returned: 0,
            index_used: None,
            duration: Duration::ZERO,
        }
    }

    /// Sets the number of rows scanned.
    #[must_use]
    pub const fn with_rows_scanned(mut self, count: usize) -> Self {
        self.rows_scanned = count;
        self
    }

    /// Sets the number of rows returned.
    #[must_use]
    pub const fn with_rows_returned(mut self, count: usize) -> Self {
        self.rows_returned = count;
        self
    }

    /// Sets the index used.
    #[must_use]
    pub fn with_index(mut self, index: impl Into<String>) -> Self {
        self.index_used = Some(index.into());
        self
    }

    /// Sets the query duration.
    #[must_use]
    pub const fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Report of index misses for a specific table/column combination.
#[derive(Debug, Clone)]
pub struct IndexMissReport {
    /// Table name.
    pub table: String,
    /// Column name.
    pub column: String,
    /// Number of queries that missed this index.
    pub miss_count: u64,
    /// Number of queries that hit this index.
    pub hit_count: u64,
}

/// Tracks index usage to detect missing indexes.
#[derive(Debug, Default)]
pub struct IndexTracker {
    /// Tracks hits per (table, column).
    hits: DashMap<(String, String), AtomicU64>,
    /// Tracks misses per (table, column).
    misses: DashMap<(String, String), AtomicU64>,
}

impl IndexTracker {
    /// Creates a new index tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an index hit for a table/column combination.
    pub fn record_hit(&self, table: &str, column: &str) {
        let key = (table.to_string(), column.to_string());
        self.hits
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records an index miss for a table/column combination.
    pub fn record_miss(&self, table: &str, column: &str) {
        let key = (table.to_string(), column.to_string());
        self.misses
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Returns reports for all columns with index misses.
    #[must_use]
    pub fn report_misses(&self) -> Vec<IndexMissReport> {
        self.misses
            .iter()
            .map(|entry| {
                let (table, column) = entry.key();
                let miss_count = entry.value().load(Ordering::Relaxed);
                let hit_count = self
                    .hits
                    .get(&(table.clone(), column.clone()))
                    .map_or(0, |v| v.load(Ordering::Relaxed));
                IndexMissReport {
                    table: table.clone(),
                    column: column.clone(),
                    miss_count,
                    hit_count,
                }
            })
            .filter(|r| r.miss_count > 0)
            .collect()
    }

    /// Returns total hit count across all indexes.
    #[must_use]
    pub fn total_hits(&self) -> u64 {
        self.hits
            .iter()
            .map(|entry| entry.value().load(Ordering::Relaxed))
            .sum()
    }

    /// Returns total miss count across all indexes.
    #[must_use]
    pub fn total_misses(&self) -> u64 {
        self.misses
            .iter()
            .map(|entry| entry.value().load(Ordering::Relaxed))
            .sum()
    }

    /// Resets all counters.
    pub fn reset(&self) {
        self.hits.clear();
        self.misses.clear();
    }
}

/// Checks if a query is slow and logs a warning if so.
///
/// This function should be called after each query with its metrics.
/// If the query duration exceeds the threshold, a warning is logged
/// with relevant context for debugging.
pub fn check_slow_query(metrics: &QueryMetrics, threshold_ms: u64) {
    let duration_ms = metrics.duration.as_millis();
    if duration_ms > u128::from(threshold_ms) {
        warn!(
            table = %metrics.table,
            operation = %metrics.operation,
            duration_ms = %duration_ms,
            rows_scanned = %metrics.rows_scanned,
            rows_returned = %metrics.rows_returned,
            index_used = ?metrics.index_used,
            "slow query detected"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_metrics_builder() {
        let metrics = QueryMetrics::new("users", "select")
            .with_rows_scanned(1000)
            .with_rows_returned(50)
            .with_index("idx_user_id")
            .with_duration(Duration::from_millis(25));

        assert_eq!(metrics.table, "users");
        assert_eq!(metrics.operation, "select");
        assert_eq!(metrics.rows_scanned, 1000);
        assert_eq!(metrics.rows_returned, 50);
        assert_eq!(metrics.index_used, Some("idx_user_id".to_string()));
        assert_eq!(metrics.duration, Duration::from_millis(25));
    }

    #[test]
    fn test_index_tracker_hits() {
        let tracker = IndexTracker::new();

        tracker.record_hit("users", "id");
        tracker.record_hit("users", "id");
        tracker.record_hit("users", "email");

        assert_eq!(tracker.total_hits(), 3);
        assert_eq!(tracker.total_misses(), 0);
    }

    #[test]
    fn test_index_tracker_misses() {
        let tracker = IndexTracker::new();

        tracker.record_miss("users", "name");
        tracker.record_miss("users", "name");
        tracker.record_miss("orders", "status");

        assert_eq!(tracker.total_misses(), 3);
        let reports = tracker.report_misses();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_index_tracker_mixed() {
        let tracker = IndexTracker::new();

        tracker.record_hit("users", "id");
        tracker.record_hit("users", "id");
        tracker.record_miss("users", "id");

        let reports = tracker.report_misses();
        assert_eq!(reports.len(), 1);
        let report = &reports[0];
        assert_eq!(report.table, "users");
        assert_eq!(report.column, "id");
        assert_eq!(report.hit_count, 2);
        assert_eq!(report.miss_count, 1);
    }

    #[test]
    fn test_index_tracker_reset() {
        let tracker = IndexTracker::new();

        tracker.record_hit("users", "id");
        tracker.record_miss("users", "name");

        assert_eq!(tracker.total_hits(), 1);
        assert_eq!(tracker.total_misses(), 1);

        tracker.reset();

        assert_eq!(tracker.total_hits(), 0);
        assert_eq!(tracker.total_misses(), 0);
    }

    #[test]
    fn test_check_slow_query_below_threshold() {
        // Should not panic or warn for fast queries
        let metrics = QueryMetrics::new("users", "select").with_duration(Duration::from_millis(50));
        check_slow_query(&metrics, 100);
    }

    #[test]
    fn test_check_slow_query_above_threshold() {
        // Should log warning for slow queries
        let metrics = QueryMetrics::new("users", "select")
            .with_rows_scanned(10000)
            .with_duration(Duration::from_millis(150));
        check_slow_query(&metrics, 100);
    }
}
