// SPDX-License-Identifier: MIT OR Apache-2.0
//! Check result types for diagnostic checks.

/// Status of a diagnostic check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Check passed successfully.
    Healthy,
    /// Check passed with warnings.
    Warning,
    /// Check failed.
    Error,
    /// Check was skipped (not applicable).
    Skipped,
}

/// Result of a single diagnostic check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Name of the check.
    pub name: &'static str,
    /// Status of the check.
    pub status: CheckStatus,
    /// Human-readable message describing the result.
    pub message: String,
    /// Optional additional details.
    pub details: Option<String>,
}

impl CheckResult {
    /// Creates a healthy check result.
    #[must_use]
    pub fn healthy(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Healthy,
            message: message.into(),
            details: None,
        }
    }

    /// Creates a warning check result.
    #[must_use]
    pub fn warning(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Warning,
            message: message.into(),
            details: None,
        }
    }

    /// Creates an error check result.
    #[must_use]
    pub fn error(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Error,
            message: message.into(),
            details: None,
        }
    }

    /// Creates a skipped check result.
    #[must_use]
    pub fn skipped(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Skipped,
            message: message.into(),
            details: None,
        }
    }

    /// Adds details to the check result.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_healthy() {
        let result = CheckResult::healthy("Storage", "All good");
        assert_eq!(result.name, "Storage");
        assert_eq!(result.status, CheckStatus::Healthy);
        assert_eq!(result.message, "All good");
        assert!(result.details.is_none());
    }

    #[test]
    fn test_check_result_warning() {
        let result = CheckResult::warning("Disk", "Running low");
        assert_eq!(result.name, "Disk");
        assert_eq!(result.status, CheckStatus::Warning);
        assert_eq!(result.message, "Running low");
    }

    #[test]
    fn test_check_result_error() {
        let result = CheckResult::error("Network", "Connection failed");
        assert_eq!(result.name, "Network");
        assert_eq!(result.status, CheckStatus::Error);
        assert_eq!(result.message, "Connection failed");
    }

    #[test]
    fn test_check_result_skipped() {
        let result = CheckResult::skipped("Cluster", "Not connected");
        assert_eq!(result.name, "Cluster");
        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.message, "Not connected");
    }

    #[test]
    fn test_check_result_with_details() {
        let result = CheckResult::healthy("Test", "OK").with_details("Extra info");
        assert_eq!(result.details, Some("Extra info".to_string()));
    }

    #[test]
    fn test_check_status_equality() {
        assert_eq!(CheckStatus::Healthy, CheckStatus::Healthy);
        assert_eq!(CheckStatus::Warning, CheckStatus::Warning);
        assert_eq!(CheckStatus::Error, CheckStatus::Error);
        assert_eq!(CheckStatus::Skipped, CheckStatus::Skipped);
        assert_ne!(CheckStatus::Healthy, CheckStatus::Warning);
        assert_ne!(CheckStatus::Error, CheckStatus::Skipped);
    }

    #[test]
    fn test_check_status_debug() {
        let debug = format!("{:?}", CheckStatus::Healthy);
        assert!(debug.contains("Healthy"));
    }

    #[test]
    fn test_check_status_clone() {
        let status = CheckStatus::Warning;
        let cloned = status;
        assert_eq!(cloned, CheckStatus::Warning);
    }

    #[test]
    fn test_check_result_debug() {
        let result = CheckResult::healthy("Test", "OK");
        let debug = format!("{result:?}");
        assert!(debug.contains("CheckResult"));
    }

    #[test]
    fn test_check_result_clone() {
        let result = CheckResult::healthy("Test", "OK").with_details("Details");
        let cloned = result;
        assert_eq!(cloned.name, "Test");
        assert_eq!(cloned.message, "OK");
        assert_eq!(cloned.details, Some("Details".to_string()));
    }

    #[test]
    fn test_check_result_with_string_message() {
        let msg = String::from("Dynamic message");
        let result = CheckResult::healthy("Test", msg);
        assert_eq!(result.message, "Dynamic message");
    }

    #[test]
    fn test_check_result_with_string_details() {
        let details = String::from("Dynamic details");
        let result = CheckResult::healthy("Test", "OK").with_details(details);
        assert_eq!(result.details, Some("Dynamic details".to_string()));
    }
}
