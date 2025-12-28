//! Stress test utilities for Neumann.
//!
//! Provides data generators, latency histograms, memory tracking,
//! and test configuration presets for 1M entity scale stress testing.

pub mod config;
pub mod generators;
pub mod metrics;

pub use config::{endurance_config, full_config, quick_config, ScaleLevel, StressConfig};
pub use generators::{generate_embeddings, generate_sparse_embeddings, generate_tensor_data};
pub use metrics::{LatencyHistogram, LatencySnapshot, ThroughputCounter};

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string.
pub fn format_duration(secs: f64) -> String {
    if secs >= 3600.0 {
        let hours = secs / 3600.0;
        format!("{:.1}h", hours)
    } else if secs >= 60.0 {
        let mins = secs / 60.0;
        format!("{:.1}m", mins)
    } else if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.2}ms", secs * 1000.0)
    }
}
