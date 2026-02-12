// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / GB as f64;
        format!("{val:.2} GB")
    } else if bytes >= MB {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / MB as f64;
        format!("{val:.2} MB")
    } else if bytes >= KB {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / KB as f64;
        format!("{val:.2} KB")
    } else {
        format!("{bytes} B")
    }
}

/// Format duration as human-readable string.
#[must_use]
pub fn format_duration(secs: f64) -> String {
    if secs >= 3600.0 {
        let hours = secs / 3600.0;
        format!("{hours:.1}h")
    } else if secs >= 60.0 {
        let mins = secs / 60.0;
        format!("{mins:.1}m")
    } else if secs >= 1.0 {
        format!("{secs:.2}s")
    } else {
        format!("{:.2}ms", secs * 1000.0)
    }
}
