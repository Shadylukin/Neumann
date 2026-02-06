// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Storage health diagnostic check.

use super::check::CheckResult;
#[cfg(test)]
use super::check::CheckStatus;
use super::wal::WalInfo;
use super::DiagnosticContext;

/// Disk usage warning threshold (85%).
const DISK_WARNING_PERCENT: u64 = 85;

/// Disk usage error threshold (95%).
const DISK_ERROR_PERCENT: u64 = 95;

/// Checks storage health including disk space and entity counts.
pub fn check_storage<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    // Get entity count from the store
    let entity_count = ctx.router.vector().store().len();

    // Try to get disk space info
    match sys_info::disk_info() {
        Ok(disk) => {
            let total_kb = disk.total;
            let free_kb = disk.free;

            if total_kb == 0 {
                return CheckResult::healthy("Storage", format!("{entity_count} entities"));
            }

            let used_kb = total_kb.saturating_sub(free_kb);
            let used_percent = (used_kb * 100) / total_kb;

            let total_str = format_bytes(total_kb * 1024);
            let used_str = format_bytes(used_kb * 1024);
            let free_str = format_bytes(free_kb * 1024);

            let message =
                format!("{used_str} used, {free_str} available ({entity_count} entities)");

            let details = format!("Total: {total_str}, Used: {used_percent}%");

            if used_percent >= DISK_ERROR_PERCENT {
                CheckResult::error("Storage", message).with_details(details)
            } else if used_percent >= DISK_WARNING_PERCENT {
                CheckResult::warning("Storage", message).with_details(details)
            } else {
                CheckResult::healthy("Storage", message).with_details(details)
            }
        },
        Err(_) => {
            // Fallback: just report entity count without disk info
            CheckResult::healthy("Storage", format!("{entity_count} entities stored"))
        },
    }
}

/// Formats bytes into a human-readable string.
#[allow(clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1}TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

/// Checks disk usage status based on percentage.
#[must_use]
#[cfg(test)]
pub const fn disk_status(used_percent: u64) -> CheckStatus {
    if used_percent >= DISK_ERROR_PERCENT {
        CheckStatus::Error
    } else if used_percent >= DISK_WARNING_PERCENT {
        CheckStatus::Warning
    } else {
        CheckStatus::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(100), "100B");
        assert_eq!(format_bytes(1023), "1023B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(2048), "2.0KB");
        assert_eq!(format_bytes(1536), "1.5KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
        assert_eq!(format_bytes(1024 * 1024 * 100), "100.0MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 23), "23.0GB");
    }

    #[test]
    fn test_format_bytes_terabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 1024), "1.0TB");
    }

    #[test]
    fn test_disk_status_healthy() {
        assert_eq!(disk_status(0), CheckStatus::Healthy);
        assert_eq!(disk_status(50), CheckStatus::Healthy);
        assert_eq!(disk_status(84), CheckStatus::Healthy);
    }

    #[test]
    fn test_disk_status_warning() {
        assert_eq!(disk_status(85), CheckStatus::Warning);
        assert_eq!(disk_status(90), CheckStatus::Warning);
        assert_eq!(disk_status(94), CheckStatus::Warning);
    }

    #[test]
    fn test_disk_status_error() {
        assert_eq!(disk_status(95), CheckStatus::Error);
        assert_eq!(disk_status(99), CheckStatus::Error);
        assert_eq!(disk_status(100), CheckStatus::Error);
    }

    #[test]
    fn test_check_storage_with_router() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_storage(&ctx);

        // Should return a valid result regardless of disk info availability
        assert!(matches!(
            result.status,
            CheckStatus::Healthy | CheckStatus::Warning | CheckStatus::Error
        ));
        assert_eq!(result.name, "Storage");
        assert!(!result.message.is_empty());
    }
}
