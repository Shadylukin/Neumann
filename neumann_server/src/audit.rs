// SPDX-License-Identifier: MIT OR Apache-2.0
//! Audit logging for server events.
//!
//! Records authentication events, queries, and blob operations for compliance and debugging.

#![allow(clippy::missing_panics_doc)]

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// Configuration for audit logging.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Enable audit logging.
    pub enabled: bool,
    /// Log successful authentications.
    pub log_success: bool,
    /// Log failed authentications.
    pub log_failure: bool,
    /// Log query executions.
    pub log_queries: bool,
    /// Log blob operations.
    pub log_blob_ops: bool,
    /// Log vector operations.
    pub log_vector_ops: bool,
    /// Maximum entries to retain (0 = unlimited).
    pub max_entries: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_success: true,
            log_failure: true,
            log_queries: false,
            log_blob_ops: true,
            log_vector_ops: true,
            max_entries: 100_000,
        }
    }
}

impl AuditConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable query logging.
    #[must_use]
    pub const fn with_query_logging(mut self) -> Self {
        self.log_queries = true;
        self
    }

    /// Enable vector operations logging.
    #[must_use]
    pub const fn with_vector_logging(mut self) -> Self {
        self.log_vector_ops = true;
        self
    }

    /// Disable vector operations logging.
    #[must_use]
    pub const fn without_vector_logging(mut self) -> Self {
        self.log_vector_ops = false;
        self
    }

    /// Set maximum entries to retain.
    #[must_use]
    pub const fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Disable audit logging.
    #[must_use]
    pub const fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Audit event types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditEvent {
    /// Successful authentication.
    AuthSuccess {
        /// The authenticated identity.
        identity: String,
    },
    /// Failed authentication attempt.
    AuthFailure {
        /// Reason for failure.
        reason: String,
    },
    /// Query execution.
    QueryExecuted {
        /// Identity that executed the query (if authenticated).
        identity: Option<String>,
        /// The query string.
        query: String,
    },
    /// Blob upload.
    BlobUpload {
        /// Identity that uploaded (if authenticated).
        identity: Option<String>,
        /// The artifact ID.
        artifact_id: String,
        /// Size in bytes.
        size: usize,
    },
    /// Blob download.
    BlobDownload {
        /// Identity that downloaded (if authenticated).
        identity: Option<String>,
        /// The artifact ID.
        artifact_id: String,
    },
    /// Blob deletion.
    BlobDelete {
        /// Identity that deleted (if authenticated).
        identity: Option<String>,
        /// The artifact ID.
        artifact_id: String,
    },
    /// Rate limit exceeded.
    RateLimited {
        /// The rate-limited identity.
        identity: String,
        /// The operation that was limited.
        operation: String,
    },
    /// Vector upsert operation.
    VectorUpsert {
        /// Identity that performed the upsert (if authenticated).
        identity: Option<String>,
        /// The collection name.
        collection: String,
        /// Number of points upserted.
        count: usize,
    },
    /// Vector query operation.
    VectorQuery {
        /// Identity that performed the query (if authenticated).
        identity: Option<String>,
        /// The collection name.
        collection: String,
        /// Number of results requested.
        limit: usize,
    },
    /// Vector delete operation.
    VectorDelete {
        /// Identity that performed the delete (if authenticated).
        identity: Option<String>,
        /// The collection name.
        collection: String,
        /// Number of points deleted.
        count: usize,
    },
    /// Collection created.
    CollectionCreated {
        /// Identity that created the collection (if authenticated).
        identity: Option<String>,
        /// The collection name.
        collection: String,
    },
    /// Collection deleted.
    CollectionDeleted {
        /// Identity that deleted the collection (if authenticated).
        identity: Option<String>,
        /// The collection name.
        collection: String,
    },
}

/// Audit entry with timestamp and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID.
    pub id: u64,
    /// The audit event.
    pub event: AuditEvent,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Remote address of the client (if available).
    pub remote_addr: Option<String>,
}

/// Audit logger for server events.
pub struct AuditLogger {
    entries: DashMap<u64, AuditEntry>,
    counter: AtomicU64,
    config: AuditConfig,
}

impl AuditLogger {
    /// Create a new audit logger with the given configuration.
    #[must_use]
    pub fn new(config: AuditConfig) -> Self {
        Self {
            entries: DashMap::new(),
            counter: AtomicU64::new(0),
            config,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    fn should_log(&self, event: &AuditEvent) -> bool {
        if !self.config.enabled {
            return false;
        }

        match event {
            AuditEvent::AuthSuccess { .. } => self.config.log_success,
            AuditEvent::AuthFailure { .. } => self.config.log_failure,
            AuditEvent::QueryExecuted { .. } => self.config.log_queries,
            AuditEvent::BlobUpload { .. }
            | AuditEvent::BlobDownload { .. }
            | AuditEvent::BlobDelete { .. } => self.config.log_blob_ops,
            AuditEvent::VectorUpsert { .. }
            | AuditEvent::VectorQuery { .. }
            | AuditEvent::VectorDelete { .. }
            | AuditEvent::CollectionCreated { .. }
            | AuditEvent::CollectionDeleted { .. } => self.config.log_vector_ops,
            AuditEvent::RateLimited { .. } => true,
        }
    }

    fn enforce_max_entries(&self) {
        if self.config.max_entries == 0 {
            return;
        }

        let current_count = self.entries.len();
        if current_count <= self.config.max_entries {
            return;
        }

        // Remove oldest entries (lowest IDs)
        let to_remove = current_count - self.config.max_entries;
        let mut ids: Vec<u64> = self.entries.iter().map(|e| *e.key()).collect();
        ids.sort_unstable();

        for id in ids.into_iter().take(to_remove) {
            self.entries.remove(&id);
        }
    }

    /// Record an audit event (best-effort, never fails).
    pub fn record(&self, event: AuditEvent, remote_addr: Option<&str>) {
        if !self.should_log(&event) {
            return;
        }

        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let entry = AuditEntry {
            id,
            event,
            timestamp: Self::now_millis(),
            remote_addr: remote_addr.map(ToString::to_string),
        };

        self.entries.insert(id, entry);
        self.enforce_max_entries();
    }

    /// Query events by identity.
    #[must_use]
    pub fn by_identity(&self, identity: &str) -> Vec<AuditEntry> {
        self.entries
            .iter()
            .filter(|e| Self::entry_has_identity(&e.event, identity))
            .map(|e| e.clone())
            .collect()
    }

    fn entry_has_identity(event: &AuditEvent, identity: &str) -> bool {
        match event {
            AuditEvent::AuthSuccess { identity: id }
            | AuditEvent::RateLimited { identity: id, .. } => id == identity,
            AuditEvent::QueryExecuted {
                identity: Some(id), ..
            }
            | AuditEvent::BlobUpload {
                identity: Some(id), ..
            }
            | AuditEvent::BlobDownload {
                identity: Some(id), ..
            }
            | AuditEvent::BlobDelete {
                identity: Some(id), ..
            }
            | AuditEvent::VectorUpsert {
                identity: Some(id), ..
            }
            | AuditEvent::VectorQuery {
                identity: Some(id), ..
            }
            | AuditEvent::VectorDelete {
                identity: Some(id), ..
            }
            | AuditEvent::CollectionCreated {
                identity: Some(id), ..
            }
            | AuditEvent::CollectionDeleted {
                identity: Some(id), ..
            } => id == identity,
            _ => false,
        }
    }

    /// Query events since timestamp.
    #[must_use]
    pub fn since(&self, since_millis: i64) -> Vec<AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= since_millis)
            .map(|e| e.clone())
            .collect()
    }

    /// Get recent events.
    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let mut entries: Vec<_> = self.entries.iter().map(|e| e.clone()).collect();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        entries
    }

    /// Get total event count.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Check if audit logging is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &AuditConfig {
        &self.config
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_auth_success() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:alice".to_string(),
            },
            Some("127.0.0.1"),
        );

        assert_eq!(logger.count(), 1);

        let entries = logger.by_identity("user:alice");
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].event, AuditEvent::AuthSuccess { .. }));
        assert_eq!(entries[0].remote_addr, Some("127.0.0.1".to_string()));
    }

    #[test]
    fn test_record_auth_failure() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::AuthFailure {
                reason: "invalid API key".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_record_query_executed() {
        let logger = AuditLogger::new(AuditConfig::default().with_query_logging());

        logger.record(
            AuditEvent::QueryExecuted {
                identity: Some("user:alice".to_string()),
                query: "SELECT users".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 1);

        let entries = logger.by_identity("user:alice");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_record_query_not_logged_by_default() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::QueryExecuted {
                identity: Some("user:alice".to_string()),
                query: "SELECT users".to_string(),
            },
            None,
        );

        // Queries not logged by default
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_record_blob_operations() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::BlobUpload {
                identity: Some("user:alice".to_string()),
                artifact_id: "abc123".to_string(),
                size: 1024,
            },
            None,
        );

        logger.record(
            AuditEvent::BlobDownload {
                identity: Some("user:alice".to_string()),
                artifact_id: "abc123".to_string(),
            },
            None,
        );

        logger.record(
            AuditEvent::BlobDelete {
                identity: Some("user:alice".to_string()),
                artifact_id: "abc123".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 3);
    }

    #[test]
    fn test_record_rate_limited() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::RateLimited {
                identity: "user:alice".to_string(),
                operation: "request".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_query_by_identity() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:alice".to_string(),
            },
            None,
        );
        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:bob".to_string(),
            },
            None,
        );
        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:alice".to_string(),
            },
            None,
        );

        let alice_entries = logger.by_identity("user:alice");
        assert_eq!(alice_entries.len(), 2);

        let bob_entries = logger.by_identity("user:bob");
        assert_eq!(bob_entries.len(), 1);
    }

    #[test]
    fn test_query_since_timestamp() {
        let logger = AuditLogger::new(AuditConfig::default());

        let before = AuditLogger::now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));

        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:alice".to_string(),
            },
            None,
        );

        let entries = logger.since(before);
        assert_eq!(entries.len(), 1);

        let entries = logger.since(AuditLogger::now_millis() + 1000);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_recent_entries() {
        let logger = AuditLogger::new(AuditConfig::default());

        for i in 0..10 {
            logger.record(
                AuditEvent::AuthSuccess {
                    identity: format!("user:{i}"),
                },
                None,
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let recent = logger.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first
        assert!(recent[0].timestamp >= recent[1].timestamp);
        assert!(recent[1].timestamp >= recent[2].timestamp);
    }

    #[test]
    fn test_max_entries_enforcement() {
        let logger = AuditLogger::new(AuditConfig::default().with_max_entries(5));

        for i in 0..10 {
            logger.record(
                AuditEvent::AuthSuccess {
                    identity: format!("user:{i}"),
                },
                None,
            );
        }

        assert!(logger.count() <= 5);
    }

    #[test]
    fn test_disabled_no_recording() {
        let logger = AuditLogger::new(AuditConfig::default().disabled());

        logger.record(
            AuditEvent::AuthSuccess {
                identity: "user:alice".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_is_enabled() {
        let enabled = AuditLogger::new(AuditConfig::default());
        assert!(enabled.is_enabled());

        let disabled = AuditLogger::new(AuditConfig::default().disabled());
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_config_accessor() {
        let config = AuditConfig::default().with_query_logging();
        let logger = AuditLogger::new(config);

        assert!(logger.config().log_queries);
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();

        assert!(config.enabled);
        assert!(config.log_success);
        assert!(config.log_failure);
        assert!(!config.log_queries);
        assert!(config.log_blob_ops);
        assert!(config.log_vector_ops);
        assert_eq!(config.max_entries, 100_000);
    }

    #[test]
    fn test_vector_upsert_event() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::VectorUpsert {
                identity: Some("user:alice".to_string()),
                collection: "embeddings".to_string(),
                count: 10,
            },
            None,
        );

        assert_eq!(logger.count(), 1);
        let entries = logger.by_identity("user:alice");
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].event, AuditEvent::VectorUpsert { .. }));
    }

    #[test]
    fn test_vector_query_event() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::VectorQuery {
                identity: Some("user:alice".to_string()),
                collection: "embeddings".to_string(),
                limit: 10,
            },
            None,
        );

        assert_eq!(logger.count(), 1);
        let entries = logger.by_identity("user:alice");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_vector_delete_event() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::VectorDelete {
                identity: Some("user:alice".to_string()),
                collection: "embeddings".to_string(),
                count: 5,
            },
            None,
        );

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_collection_created_event() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::CollectionCreated {
                identity: Some("user:alice".to_string()),
                collection: "new_collection".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_collection_deleted_event() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::CollectionDeleted {
                identity: Some("user:alice".to_string()),
                collection: "old_collection".to_string(),
            },
            None,
        );

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_log_vector_ops_disabled() {
        let logger = AuditLogger::new(AuditConfig::default().without_vector_logging());

        logger.record(
            AuditEvent::VectorUpsert {
                identity: Some("user:alice".to_string()),
                collection: "embeddings".to_string(),
                count: 10,
            },
            None,
        );

        logger.record(
            AuditEvent::CollectionCreated {
                identity: Some("user:alice".to_string()),
                collection: "new_collection".to_string(),
            },
            None,
        );

        // Vector events should not be logged when log_vector_ops is false
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_auth_failure_not_matched_by_identity() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::AuthFailure {
                reason: "invalid key".to_string(),
            },
            None,
        );

        // Auth failures don't have an identity
        let entries = logger.by_identity("user:alice");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_anonymous_operations() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.record(
            AuditEvent::BlobDownload {
                identity: None,
                artifact_id: "abc123".to_string(),
            },
            None,
        );

        // Anonymous operations don't match any identity
        let entries = logger.by_identity("user:alice");
        assert!(entries.is_empty());

        // But they are still recorded
        assert_eq!(logger.count(), 1);
    }
}
