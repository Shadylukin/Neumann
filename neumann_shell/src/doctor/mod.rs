// SPDX-License-Identifier: MIT OR Apache-2.0
//! System diagnostics for the Neumann shell.
//!
//! The doctor module provides health checks and system validation.

pub mod check;
mod cluster;
mod hnsw;
mod network;
pub mod output;
pub mod storage;
mod tls;
pub mod wal;

use check::{CheckResult, CheckStatus};
use query_router::QueryRouter;
use wal::WalInfo;

/// Context for running diagnostics.
pub struct DiagnosticContext<'a, W: WalInfo = crate::wal::Wal> {
    /// Reference to the query router.
    pub router: &'a QueryRouter,
    /// Optional reference to the WAL.
    pub wal: Option<&'a W>,
    /// Network latency threshold in milliseconds.
    pub latency_threshold_ms: u64,
}

impl<'a, W: WalInfo> DiagnosticContext<'a, W> {
    /// Default latency threshold in milliseconds.
    const DEFAULT_LATENCY_THRESHOLD_MS: u64 = 50;

    /// Creates a new diagnostic context.
    pub const fn new(router: &'a QueryRouter, wal: Option<&'a W>) -> Self {
        Self {
            router,
            wal,
            latency_threshold_ms: Self::DEFAULT_LATENCY_THRESHOLD_MS,
        }
    }

    /// Sets a custom latency threshold.
    #[must_use]
    #[cfg(test)]
    pub const fn with_latency_threshold(mut self, threshold_ms: u64) -> Self {
        self.latency_threshold_ms = threshold_ms;
        self
    }
}

/// Summary of diagnostic results.
#[derive(Debug, Clone, Default)]
pub struct ReportSummary {
    /// Number of healthy checks.
    pub healthy: usize,
    /// Number of warning checks.
    pub warnings: usize,
    /// Number of error checks.
    pub errors: usize,
    /// Number of skipped checks.
    pub skipped: usize,
}

/// Complete diagnostic report.
#[derive(Debug, Clone)]
pub struct DoctorReport {
    /// Individual check results.
    pub results: Vec<CheckResult>,
    /// Summary statistics.
    pub summary: ReportSummary,
}

impl DoctorReport {
    /// Returns true if all non-skipped checks are healthy.
    #[must_use]
    #[cfg(test)]
    pub const fn is_healthy(&self) -> bool {
        self.summary.errors == 0 && self.summary.warnings == 0
    }

    /// Returns true if there are any errors.
    #[must_use]
    #[cfg(test)]
    pub const fn has_errors(&self) -> bool {
        self.summary.errors > 0
    }

    /// Returns true if there are any warnings.
    #[must_use]
    #[cfg(test)]
    pub const fn has_warnings(&self) -> bool {
        self.summary.warnings > 0
    }
}

/// Runs all diagnostic checks and returns a report.
pub fn run_diagnostics<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> DoctorReport {
    let results = vec![
        storage::check_storage(ctx),
        wal::check_wal(ctx),
        hnsw::check_hnsw(ctx),
        cluster::check_cluster(ctx),
        network::check_network(ctx),
        tls::check_tls(ctx),
    ];

    // Calculate summary
    let summary = calculate_summary(&results);

    DoctorReport { results, summary }
}

/// Calculates summary statistics from check results.
fn calculate_summary(results: &[CheckResult]) -> ReportSummary {
    let mut summary = ReportSummary::default();

    for result in results {
        match result.status {
            CheckStatus::Healthy => summary.healthy += 1,
            CheckStatus::Warning => summary.warnings += 1,
            CheckStatus::Error => summary.errors += 1,
            CheckStatus::Skipped => summary.skipped += 1,
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_context_new() {
        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);

        assert_eq!(
            ctx.latency_threshold_ms,
            DiagnosticContext::<crate::wal::Wal>::DEFAULT_LATENCY_THRESHOLD_MS
        );
        assert!(ctx.wal.is_none());
    }

    #[test]
    fn test_diagnostic_context_with_latency_threshold() {
        let router = QueryRouter::new();
        let ctx =
            DiagnosticContext::<crate::wal::Wal>::new(&router, None).with_latency_threshold(100);

        assert_eq!(ctx.latency_threshold_ms, 100);
    }

    #[test]
    fn test_doctor_report_is_healthy() {
        let report = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 5,
                warnings: 0,
                errors: 0,
                skipped: 1,
            },
        };

        assert!(report.is_healthy());
    }

    #[test]
    fn test_doctor_report_is_healthy_with_warnings() {
        let report = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 4,
                warnings: 1,
                errors: 0,
                skipped: 1,
            },
        };

        assert!(!report.is_healthy());
    }

    #[test]
    fn test_doctor_report_is_healthy_with_errors() {
        let report = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 4,
                warnings: 0,
                errors: 1,
                skipped: 1,
            },
        };

        assert!(!report.is_healthy());
    }

    #[test]
    fn test_doctor_report_has_errors() {
        let report_with = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 0,
                warnings: 0,
                errors: 1,
                skipped: 0,
            },
        };
        let report_without = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 5,
                warnings: 1,
                errors: 0,
                skipped: 0,
            },
        };

        assert!(report_with.has_errors());
        assert!(!report_without.has_errors());
    }

    #[test]
    fn test_doctor_report_has_warnings() {
        let report_with = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 0,
                warnings: 1,
                errors: 0,
                skipped: 0,
            },
        };
        let report_without = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 5,
                warnings: 0,
                errors: 1,
                skipped: 0,
            },
        };

        assert!(report_with.has_warnings());
        assert!(!report_without.has_warnings());
    }

    #[test]
    fn test_calculate_summary() {
        let results = vec![
            CheckResult::healthy("A", "OK"),
            CheckResult::healthy("B", "OK"),
            CheckResult::warning("C", "Warn"),
            CheckResult::error("D", "Err"),
            CheckResult::skipped("E", "Skip"),
        ];

        let summary = calculate_summary(&results);

        assert_eq!(summary.healthy, 2);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.errors, 1);
        assert_eq!(summary.skipped, 1);
    }

    #[test]
    fn test_calculate_summary_empty() {
        let summary = calculate_summary(&[]);

        assert_eq!(summary.healthy, 0);
        assert_eq!(summary.warnings, 0);
        assert_eq!(summary.errors, 0);
        assert_eq!(summary.skipped, 0);
    }

    #[test]
    fn test_run_diagnostics() {
        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);

        let report = run_diagnostics(&ctx);

        // Should have 6 checks total
        assert_eq!(report.results.len(), 6);

        // Verify all check names are present
        let names: Vec<_> = report.results.iter().map(|r| r.name).collect();
        assert!(names.contains(&"Storage"));
        assert!(names.contains(&"WAL"));
        assert!(names.contains(&"HNSW"));
        assert!(names.contains(&"Cluster"));
        assert!(names.contains(&"Network"));
        assert!(names.contains(&"TLS"));
    }

    #[test]
    fn test_run_diagnostics_no_cluster() {
        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);

        let report = run_diagnostics(&ctx);

        // Cluster, Network, and TLS should be skipped when not connected
        let cluster = report.results.iter().find(|r| r.name == "Cluster").unwrap();
        let network = report.results.iter().find(|r| r.name == "Network").unwrap();
        let tls = report.results.iter().find(|r| r.name == "TLS").unwrap();

        assert_eq!(cluster.status, CheckStatus::Skipped);
        assert_eq!(network.status, CheckStatus::Skipped);
        assert_eq!(tls.status, CheckStatus::Skipped);
    }

    #[test]
    fn test_report_summary_default() {
        let summary = ReportSummary::default();
        assert_eq!(summary.healthy, 0);
        assert_eq!(summary.warnings, 0);
        assert_eq!(summary.errors, 0);
        assert_eq!(summary.skipped, 0);
    }

    #[test]
    fn test_report_summary_debug() {
        let summary = ReportSummary {
            healthy: 1,
            warnings: 2,
            errors: 3,
            skipped: 4,
        };
        let debug = format!("{summary:?}");
        assert!(debug.contains("ReportSummary"));
    }

    #[test]
    fn test_report_summary_clone() {
        let summary = ReportSummary {
            healthy: 1,
            warnings: 2,
            errors: 3,
            skipped: 4,
        };
        let cloned = summary;
        assert_eq!(cloned.healthy, 1);
        assert_eq!(cloned.warnings, 2);
        assert_eq!(cloned.errors, 3);
        assert_eq!(cloned.skipped, 4);
    }

    #[test]
    fn test_doctor_report_debug() {
        let report = DoctorReport {
            results: vec![CheckResult::healthy("Test", "OK")],
            summary: ReportSummary::default(),
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("DoctorReport"));
    }

    #[test]
    fn test_doctor_report_clone() {
        let report = DoctorReport {
            results: vec![CheckResult::healthy("Test", "OK")],
            summary: ReportSummary {
                healthy: 1,
                warnings: 0,
                errors: 0,
                skipped: 0,
            },
        };
        let cloned = report;
        assert_eq!(cloned.results.len(), 1);
        assert_eq!(cloned.summary.healthy, 1);
    }
}
