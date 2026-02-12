// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! TLS certificate expiration check.

use std::path::Path;

use super::check::CheckResult;
use super::wal::WalInfo;
use super::DiagnosticContext;

/// Days threshold for TLS certificate warning.
const TLS_WARNING_DAYS: u64 = 30;

/// Days threshold for TLS certificate error.
const TLS_ERROR_DAYS: u64 = 7;

/// Checks TLS certificate expiration status.
///
/// Returns Skipped when not in cluster mode, as TLS is only used for cluster communication.
pub fn check_tls<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    if !ctx.router.is_cluster_active() {
        return CheckResult::skipped("TLS", "Cluster not connected");
    }

    // Try to get certificate path from cluster config
    let Some(cert_path) = ctx.router.tls_cert_path() else {
        return CheckResult::skipped("TLS", "TLS not configured");
    };

    // Read and parse certificate
    match check_certificate_expiry(&cert_path) {
        Ok(days) => tls_status_for_days(days),
        Err(e) => CheckResult::warning("TLS", "Cannot verify certificate").with_details(e),
    }
}

/// Checks the expiration date of a certificate file.
fn check_certificate_expiry(path: &Path) -> std::result::Result<u64, String> {
    use x509_parser::pem::parse_x509_pem;

    let pem_data = std::fs::read(path).map_err(|e| format!("Failed to read cert: {e}"))?;

    let (_, pem) = parse_x509_pem(&pem_data).map_err(|e| format!("Failed to parse PEM: {e}"))?;

    let cert = pem
        .parse_x509()
        .map_err(|e| format!("Failed to parse X.509: {e}"))?;

    let not_after = cert.validity().not_after.timestamp();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Handle expired certificates (negative timestamp or past expiration)
    let Some(not_after_secs) = u64::try_from(not_after).ok() else {
        return Ok(0); // Negative timestamp means already expired
    };

    if not_after_secs < now {
        return Ok(0);
    }

    let seconds_remaining = not_after_secs.saturating_sub(now);
    let days_remaining = seconds_remaining / 86400;
    Ok(days_remaining)
}

/// Returns the appropriate check result based on days until expiration.
fn tls_status_for_days(days: u64) -> CheckResult {
    if days <= TLS_ERROR_DAYS {
        CheckResult::error("TLS", format!("Certificate expires in {days} days"))
            .with_details("Renew certificate immediately to avoid service disruption")
    } else if days <= TLS_WARNING_DAYS {
        CheckResult::warning("TLS", format!("Certificate expires in {days} days"))
            .with_details("Plan certificate renewal soon")
    } else {
        CheckResult::healthy(
            "TLS",
            format!("Certificates valid (expires in {days} days)"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::check::CheckStatus;

    #[test]
    fn test_check_tls_not_connected() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_tls(&ctx);

        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.name, "TLS");
        assert!(result.message.contains("not connected"));
    }

    #[test]
    fn test_tls_status_healthy() {
        let result = tls_status_for_days(90);
        assert_eq!(result.status, CheckStatus::Healthy);
        assert!(result.message.contains("90 days"));
    }

    #[test]
    fn test_tls_status_healthy_boundary() {
        let result = tls_status_for_days(31);
        assert_eq!(result.status, CheckStatus::Healthy);
    }

    #[test]
    fn test_tls_status_warning() {
        let result = tls_status_for_days(30);
        assert_eq!(result.status, CheckStatus::Warning);
        assert!(result.message.contains("30 days"));
    }

    #[test]
    fn test_tls_status_warning_boundary() {
        let result = tls_status_for_days(8);
        assert_eq!(result.status, CheckStatus::Warning);
    }

    #[test]
    fn test_tls_status_error() {
        let result = tls_status_for_days(7);
        assert_eq!(result.status, CheckStatus::Error);
        assert!(result.message.contains("7 days"));
    }

    #[test]
    fn test_tls_status_error_critical() {
        let result = tls_status_for_days(1);
        assert_eq!(result.status, CheckStatus::Error);
        assert!(result.details.is_some());
    }

    #[test]
    fn test_tls_status_error_expired() {
        let result = tls_status_for_days(0);
        assert_eq!(result.status, CheckStatus::Error);
    }
}
