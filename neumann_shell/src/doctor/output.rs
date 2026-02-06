// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Output formatting for doctor report.

use std::fmt::Write;

use crate::style::{styled, Icons, Theme};

use super::check::{CheckResult, CheckStatus};
use super::DoctorReport;

/// Formats a doctor report for display.
pub fn format_report(report: &DoctorReport, theme: &Theme, icons: &Icons) -> String {
    let mut output = String::new();

    for result in &report.results {
        let line = format_check_result(result, theme, icons);
        let _ = writeln!(output, "{line}");
    }

    // Add summary
    let summary = &report.summary;
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "{}",
        styled(
            format!(
                "{} healthy, {} warnings, {} errors, {} skipped",
                summary.healthy, summary.warnings, summary.errors, summary.skipped
            ),
            theme.muted
        )
    );

    output
}

/// Formats a single check result.
fn format_check_result(result: &CheckResult, theme: &Theme, icons: &Icons) -> String {
    let (icon, style) = match result.status {
        CheckStatus::Healthy => (icons.success, theme.success),
        CheckStatus::Warning => (icons.warning, theme.warning),
        CheckStatus::Error => (icons.error, theme.error),
        CheckStatus::Skipped => (icons.info, theme.muted),
    };

    let mut line = format!(
        "{} {} {}",
        styled(icon, style),
        styled(result.name, theme.highlight),
        result.message
    );

    if let Some(details) = &result.details {
        let _ = write!(line, " ({})", styled(details, theme.muted));
    }

    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::{check::CheckResult, ReportSummary};

    fn test_icons() -> &'static Icons {
        Icons::plain()
    }

    fn test_theme() -> Theme {
        Theme::plain()
    }

    #[test]
    fn test_format_report_empty() {
        let report = DoctorReport {
            results: vec![],
            summary: ReportSummary {
                healthy: 0,
                warnings: 0,
                errors: 0,
                skipped: 0,
            },
        };

        let output = format_report(&report, &test_theme(), test_icons());
        assert!(output.contains("0 healthy"));
        assert!(output.contains("0 warnings"));
        assert!(output.contains("0 errors"));
        assert!(output.contains("0 skipped"));
    }

    #[test]
    fn test_format_report_all_statuses() {
        let report = DoctorReport {
            results: vec![
                CheckResult::healthy("Storage", "OK"),
                CheckResult::warning("WAL", "Large"),
                CheckResult::error("Network", "Failed"),
                CheckResult::skipped("TLS", "Not connected"),
            ],
            summary: ReportSummary {
                healthy: 1,
                warnings: 1,
                errors: 1,
                skipped: 1,
            },
        };

        let output = format_report(&report, &test_theme(), test_icons());
        assert!(output.contains("Storage"));
        assert!(output.contains("WAL"));
        assert!(output.contains("Network"));
        assert!(output.contains("TLS"));
        assert!(output.contains("1 healthy"));
        assert!(output.contains("1 warnings"));
        assert!(output.contains("1 errors"));
        assert!(output.contains("1 skipped"));
    }

    #[test]
    fn test_format_report_with_details() {
        let report = DoctorReport {
            results: vec![CheckResult::healthy("Storage", "OK").with_details("Extra info")],
            summary: ReportSummary {
                healthy: 1,
                warnings: 0,
                errors: 0,
                skipped: 0,
            },
        };

        let output = format_report(&report, &test_theme(), test_icons());
        assert!(output.contains("Extra info"));
    }

    #[test]
    fn test_format_check_result_healthy() {
        let result = CheckResult::healthy("Test", "All good");
        let line = format_check_result(&result, &test_theme(), test_icons());
        assert!(line.contains("[ok]"));
        assert!(line.contains("Test"));
        assert!(line.contains("All good"));
    }

    #[test]
    fn test_format_check_result_warning() {
        let result = CheckResult::warning("Test", "Needs attention");
        let line = format_check_result(&result, &test_theme(), test_icons());
        assert!(line.contains("[!]"));
        assert!(line.contains("Needs attention"));
    }

    #[test]
    fn test_format_check_result_error() {
        let result = CheckResult::error("Test", "Failed");
        let line = format_check_result(&result, &test_theme(), test_icons());
        assert!(line.contains("[!!]"));
        assert!(line.contains("Failed"));
    }

    #[test]
    fn test_format_check_result_skipped() {
        let result = CheckResult::skipped("Test", "Not applicable");
        let line = format_check_result(&result, &test_theme(), test_icons());
        assert!(line.contains("[i]"));
        assert!(line.contains("Not applicable"));
    }

    #[test]
    fn test_format_check_result_with_details() {
        let result = CheckResult::healthy("Test", "OK").with_details("More info");
        let line = format_check_result(&result, &test_theme(), test_icons());
        assert!(line.contains("More info"));
        assert!(line.contains('('));
        assert!(line.contains(')'));
    }
}
