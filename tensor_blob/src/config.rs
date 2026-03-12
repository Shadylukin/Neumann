// SPDX-License-Identifier: MIT OR Apache-2.0
use std::time::Duration;

use crate::error::{BlobError, Result};

/// Configuration for the blob store.
#[derive(Debug, Clone)]
pub struct BlobConfig {
    /// Size of each chunk in bytes. Default: 1MB.
    pub chunk_size: usize,
    /// Maximum artifact size in bytes. None means unlimited.
    pub max_artifact_size: Option<usize>,
    /// Maximum number of artifacts. None means unlimited.
    pub max_artifacts: Option<usize>,
    /// Garbage collection check interval.
    pub gc_interval: Duration,
    /// Number of chunks to process per GC cycle.
    pub gc_batch_size: usize,
    /// Minimum age before a chunk can be garbage collected.
    pub gc_min_age: Duration,
    /// Default content type for artifacts without explicit type.
    pub default_content_type: String,
}

impl Default for BlobConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1MB
            max_artifact_size: None,
            max_artifacts: None,
            gc_interval: Duration::from_secs(300), // 5 minutes
            gc_batch_size: 100,
            gc_min_age: Duration::from_secs(60), // 1 minute
            default_content_type: "application/octet-stream".to_string(),
        }
    }
}

impl BlobConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    #[must_use]
    pub const fn with_max_artifact_size(mut self, size: usize) -> Self {
        self.max_artifact_size = Some(size);
        self
    }

    #[must_use]
    pub const fn with_max_artifacts(mut self, count: usize) -> Self {
        self.max_artifacts = Some(count);
        self
    }

    #[must_use]
    pub const fn with_gc_interval(mut self, interval: Duration) -> Self {
        self.gc_interval = interval;
        self
    }

    #[must_use]
    pub const fn with_gc_batch_size(mut self, size: usize) -> Self {
        self.gc_batch_size = size;
        self
    }

    #[must_use]
    pub const fn with_gc_min_age(mut self, age: Duration) -> Self {
        self.gc_min_age = age;
        self
    }

    #[must_use]
    pub fn with_default_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.default_content_type = content_type.into();
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_size` or `gc_batch_size` is zero.
    pub fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(BlobError::InvalidConfig(
                "chunk_size must be > 0".to_string(),
            ));
        }
        if self.gc_batch_size == 0 {
            return Err(BlobError::InvalidConfig(
                "gc_batch_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for garbage collection.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Check interval for background GC.
    pub check_interval: Duration,
    /// Number of chunks to process per cycle.
    pub batch_size: usize,
    /// Minimum age before a chunk can be collected.
    pub min_age: Duration,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(300),
            batch_size: 100,
            min_age: Duration::from_secs(60),
        }
    }
}

impl From<&BlobConfig> for GcConfig {
    fn from(config: &BlobConfig) -> Self {
        Self {
            check_interval: config.gc_interval,
            batch_size: config.gc_batch_size,
            min_age: config.gc_min_age,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BlobConfig::default();
        assert_eq!(config.chunk_size, 1024 * 1024);
        assert!(config.max_artifact_size.is_none());
        assert!(config.max_artifacts.is_none());
        assert_eq!(config.gc_interval, Duration::from_secs(300));
        assert_eq!(config.gc_batch_size, 100);
        assert_eq!(config.gc_min_age, Duration::from_secs(60));
        assert_eq!(config.default_content_type, "application/octet-stream");
    }

    #[test]
    fn test_builder_pattern() {
        let config = BlobConfig::new()
            .with_chunk_size(512 * 1024)
            .with_max_artifact_size(100 * 1024 * 1024)
            .with_max_artifacts(1000)
            .with_gc_interval(Duration::from_secs(60))
            .with_gc_batch_size(50)
            .with_gc_min_age(Duration::from_secs(30))
            .with_default_content_type("text/plain");

        assert_eq!(config.chunk_size, 512 * 1024);
        assert_eq!(config.max_artifact_size, Some(100 * 1024 * 1024));
        assert_eq!(config.max_artifacts, Some(1000));
        assert_eq!(config.gc_interval, Duration::from_secs(60));
        assert_eq!(config.gc_batch_size, 50);
        assert_eq!(config.gc_min_age, Duration::from_secs(30));
        assert_eq!(config.default_content_type, "text/plain");
    }

    #[test]
    fn test_validate_valid() {
        let config = BlobConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_chunk_size() {
        let config = BlobConfig::new().with_chunk_size(0);
        let result = config.validate();
        assert!(matches!(result, Err(BlobError::InvalidConfig(_))));
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let config = BlobConfig::new().with_gc_batch_size(0);
        let result = config.validate();
        assert!(matches!(result, Err(BlobError::InvalidConfig(_))));
    }

    #[test]
    fn test_gc_config_from_blob_config() {
        let blob_config = BlobConfig::new()
            .with_gc_interval(Duration::from_secs(120))
            .with_gc_batch_size(200)
            .with_gc_min_age(Duration::from_secs(90));

        let gc_config = GcConfig::from(&blob_config);
        assert_eq!(gc_config.check_interval, Duration::from_secs(120));
        assert_eq!(gc_config.batch_size, 200);
        assert_eq!(gc_config.min_age, Duration::from_secs(90));
    }
}
