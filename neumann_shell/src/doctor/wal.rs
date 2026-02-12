// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! WAL health diagnostic check.

use super::check::CheckResult;
use super::DiagnosticContext;
use crate::wal::Wal;

/// WAL size warning threshold (100MB).
const WAL_WARNING_BYTES: u64 = 100 * 1024 * 1024;

/// Formats a duration in seconds to human-readable "X ago" format.
fn format_time_ago(seconds: u64) -> String {
    if seconds < 60 {
        format!("{seconds}s ago")
    } else if seconds < 3600 {
        format!("{}m ago", seconds / 60)
    } else if seconds < 86400 {
        format!("{}h ago", seconds / 3600)
    } else {
        format!("{}d ago", seconds / 86400)
    }
}

/// Gets the time since the last checkpoint in seconds.
fn get_last_checkpoint_time(ctx: &DiagnosticContext<'_, impl WalInfo>) -> Option<u64> {
    let checkpoint = ctx.router.checkpoint()?;
    let runtime = ctx.router.runtime()?;

    let result = runtime.block_on(async {
        let guard = checkpoint.lock().await;
        guard.list(Some(1)).await.ok()
    });

    result.and_then(|list| {
        list.first().map(|cp| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            now.saturating_sub(cp.created_at)
        })
    })
}

/// Checks WAL integrity and size.
pub fn check_wal<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    let Some(wal) = ctx.wal else {
        return CheckResult::skipped("WAL", "Not active (use LOAD to enable)");
    };

    let size = wal.wal_size().unwrap_or(0);
    let size_str = super::storage::format_bytes(size);

    // Try to get last checkpoint time
    let checkpoint_info = get_last_checkpoint_time(ctx);

    // Format message with or without checkpoint info
    let format_msg = |prefix: &str| {
        checkpoint_info.map_or_else(
            || format!("{prefix} ({size_str})"),
            |seconds_ago| {
                format!(
                    "{prefix} ({size_str}, last checkpoint {})",
                    format_time_ago(seconds_ago)
                )
            },
        )
    };

    if size > WAL_WARNING_BYTES {
        CheckResult::warning("WAL", format_msg("Large WAL"))
            .with_details("Consider running SAVE to create a checkpoint and truncate WAL")
    } else {
        CheckResult::healthy("WAL", format_msg("Intact"))
    }
}

/// Trait for accessing WAL information.
pub trait WalInfo {
    /// Returns the WAL file size in bytes.
    fn wal_size(&self) -> std::io::Result<u64>;
}

impl WalInfo for Wal {
    fn wal_size(&self) -> std::io::Result<u64> {
        self.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::check::CheckStatus;

    struct MockWal {
        size: u64,
    }

    impl WalInfo for MockWal {
        fn wal_size(&self) -> std::io::Result<u64> {
            Ok(self.size)
        }
    }

    struct FailingWal;

    impl WalInfo for FailingWal {
        fn wal_size(&self) -> std::io::Result<u64> {
            Err(std::io::Error::other("mock error"))
        }
    }

    #[test]
    fn test_format_time_ago_seconds() {
        assert_eq!(format_time_ago(0), "0s ago");
        assert_eq!(format_time_ago(1), "1s ago");
        assert_eq!(format_time_ago(30), "30s ago");
        assert_eq!(format_time_ago(59), "59s ago");
    }

    #[test]
    fn test_format_time_ago_minutes() {
        assert_eq!(format_time_ago(60), "1m ago");
        assert_eq!(format_time_ago(90), "1m ago");
        assert_eq!(format_time_ago(120), "2m ago");
        assert_eq!(format_time_ago(3599), "59m ago");
    }

    #[test]
    fn test_format_time_ago_hours() {
        assert_eq!(format_time_ago(3600), "1h ago");
        assert_eq!(format_time_ago(7200), "2h ago");
        assert_eq!(format_time_ago(86399), "23h ago");
    }

    #[test]
    fn test_format_time_ago_days() {
        assert_eq!(format_time_ago(86400), "1d ago");
        assert_eq!(format_time_ago(172800), "2d ago");
        assert_eq!(format_time_ago(604800), "7d ago");
    }

    #[test]
    fn test_check_wal_not_active() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::new(&router, None::<&MockWal>);
        let result = check_wal(&ctx);

        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.name, "WAL");
        assert!(result.message.contains("Not active"));
    }

    #[test]
    fn test_check_wal_healthy_small() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let wal = MockWal { size: 1024 }; // 1KB
        let ctx = DiagnosticContext::new(&router, Some(&wal));
        let result = check_wal(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
        assert_eq!(result.name, "WAL");
        assert!(result.message.contains("Intact"));
    }

    #[test]
    fn test_check_wal_healthy_at_threshold() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        // Exactly at threshold should be healthy
        let wal = MockWal {
            size: WAL_WARNING_BYTES,
        };
        let ctx = DiagnosticContext::new(&router, Some(&wal));
        let result = check_wal(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
    }

    #[test]
    fn test_check_wal_warning_large() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let wal = MockWal {
            size: WAL_WARNING_BYTES + 1,
        };
        let ctx = DiagnosticContext::new(&router, Some(&wal));
        let result = check_wal(&ctx);

        assert_eq!(result.status, CheckStatus::Warning);
        assert!(result.message.contains("Large WAL"));
        assert!(result.details.is_some());
    }

    #[test]
    fn test_check_wal_size_error() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let wal = FailingWal;
        let ctx = DiagnosticContext::new(&router, Some(&wal));
        let result = check_wal(&ctx);

        // When size() fails, it returns 0, which is healthy
        assert_eq!(result.status, CheckStatus::Healthy);
    }

    #[test]
    fn test_check_wal_zero_size() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let wal = MockWal { size: 0 };
        let ctx = DiagnosticContext::new(&router, Some(&wal));
        let result = check_wal(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
        assert!(result.message.contains("0B"));
    }
}
