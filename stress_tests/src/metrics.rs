// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Latency and throughput metrics for stress tests.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use hdrhistogram::Histogram;

/// Latency histogram with p50/p99/p999 percentiles.
pub struct LatencyHistogram {
    histogram: Histogram<u64>,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            histogram: Histogram::new(3).expect("histogram creation"),
        }
    }

    /// Record a latency measurement.
    pub fn record(&mut self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        let _ = self.histogram.record(micros);
    }

    /// Get a snapshot of the current statistics.
    pub fn snapshot(&self) -> LatencySnapshot {
        LatencySnapshot {
            count: self.histogram.len(),
            p50: Duration::from_micros(self.histogram.value_at_quantile(0.5)),
            p99: Duration::from_micros(self.histogram.value_at_quantile(0.99)),
            p999: Duration::from_micros(self.histogram.value_at_quantile(0.999)),
            max: Duration::from_micros(self.histogram.max()),
            mean: Duration::from_micros(self.histogram.mean() as u64),
        }
    }

    /// Merge another histogram into this one.
    pub fn merge(&mut self, other: &LatencyHistogram) {
        let _ = self.histogram.add(&other.histogram);
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of latency statistics.
#[derive(Debug, Clone)]
pub struct LatencySnapshot {
    pub count: u64,
    pub p50: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub max: Duration,
    pub mean: Duration,
}

impl std::fmt::Display for LatencySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} p50={:?} p99={:?} p999={:?} max={:?}",
            self.count, self.p50, self.p99, self.p999, self.max
        )
    }
}

/// Thread-safe throughput counter.
pub struct ThroughputCounter {
    count: AtomicU64,
    start: Instant,
}

impl ThroughputCounter {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    /// Increment the counter by one.
    pub fn inc(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a specified amount.
    pub fn add(&self, n: u64) {
        self.count.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current count.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the elapsed time since creation.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Calculate throughput as ops/sec.
    pub fn throughput(&self) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.count() as f64 / elapsed
        } else {
            0.0
        }
    }
}

impl Default for ThroughputCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_histogram() {
        let mut hist = LatencyHistogram::new();
        for i in 1..=100 {
            hist.record(Duration::from_micros(i));
        }
        let snap = hist.snapshot();
        assert_eq!(snap.count, 100);
        assert!(snap.p50 >= Duration::from_micros(40));
        assert!(snap.p50 <= Duration::from_micros(60));
    }

    #[test]
    fn test_throughput_counter() {
        let counter = ThroughputCounter::new();
        counter.add(100);
        assert_eq!(counter.count(), 100);
    }
}
