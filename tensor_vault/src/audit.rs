// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Audit logging for vault operations.
//!
//! Records all vault operations for compliance and forensics.

#![allow(clippy::missing_panics_doc)]

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

static AUDIT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Audit entry representing a single vault operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditEntry {
    /// The entity that performed the operation.
    pub entity: String,
    /// The secret key that was accessed.
    pub secret_key: String,
    /// The operation performed.
    pub operation: AuditOperation,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// Types of auditable operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditOperation {
    Get,
    Set,
    Delete,
    Rotate,
    Grant { to: String, permission: String },
    Revoke { from: String },
    List,
    RotateMasterKey { secrets_count: usize },
}

impl AuditOperation {
    fn as_str(&self) -> &str {
        match self {
            Self::Get => "get",
            Self::Set => "set",
            Self::Delete => "delete",
            Self::Rotate => "rotate",
            Self::Grant { .. } => "grant",
            Self::Revoke { .. } => "revoke",
            Self::List => "list",
            Self::RotateMasterKey { .. } => "rotate_master_key",
        }
    }

    fn from_tensor(tensor: &TensorData) -> Option<Self> {
        let op_type = match tensor.get("_op") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.as_str(),
            _ => return None,
        };

        match op_type {
            "get" => Some(Self::Get),
            "set" => Some(Self::Set),
            "delete" => Some(Self::Delete),
            "rotate" => Some(Self::Rotate),
            "list" => Some(Self::List),
            "grant" => {
                let to = match tensor.get("_target") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                let permission = match tensor.get("_permission") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => "admin".to_string(),
                };
                Some(Self::Grant { to, permission })
            },
            "revoke" => {
                let from = match tensor.get("_target") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                Some(Self::Revoke { from })
            },
            "rotate_master_key" => {
                let secrets_count = match tensor.get("_secrets_count") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let count = *n as usize;
                        count
                    },
                    _ => 0,
                };
                Some(Self::RotateMasterKey { secrets_count })
            },
            _ => None,
        }
    }
}

/// Audit log for tracking vault operations.
pub struct AuditLog<'a> {
    store: &'a TensorStore,
}

/// Prefix for audit entries in the store.
const AUDIT_PREFIX: &str = "_va:";

impl<'a> AuditLog<'a> {
    pub fn new(store: &'a TensorStore) -> Self {
        Self { store }
    }

    fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Record an operation.
    pub fn record(&self, entity: &str, secret_key: &str, operation: &AuditOperation) {
        let timestamp = Self::now_millis();
        let counter = AUDIT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let key = format!("{AUDIT_PREFIX}{timestamp}:{counter}");

        let mut tensor = TensorData::new();
        tensor.set(
            "_entity",
            TensorValue::Scalar(ScalarValue::String(entity.into())),
        );
        tensor.set(
            "_secret",
            TensorValue::Scalar(ScalarValue::String(secret_key.into())),
        );
        tensor.set(
            "_op",
            TensorValue::Scalar(ScalarValue::String(operation.as_str().into())),
        );
        tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(timestamp)));

        // Store additional info for grant/revoke
        match operation {
            AuditOperation::Grant { to, permission } => {
                tensor.set(
                    "_target",
                    TensorValue::Scalar(ScalarValue::String(to.clone())),
                );
                tensor.set(
                    "_permission",
                    TensorValue::Scalar(ScalarValue::String(permission.clone())),
                );
            },
            AuditOperation::Revoke { from } => {
                tensor.set(
                    "_target",
                    TensorValue::Scalar(ScalarValue::String(from.clone())),
                );
            },
            AuditOperation::RotateMasterKey { secrets_count } => {
                tensor.set(
                    "_secrets_count",
                    TensorValue::Scalar(ScalarValue::Int(*secrets_count as i64)),
                );
            },
            _ => {},
        }

        // Best effort - audit failures don't block operations
        let _ = self.store.put(&key, tensor);
    }

    /// Query audit entries for a specific secret.
    pub fn by_secret(&self, secret_key: &str) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.secret_key == secret_key)
            .collect()
    }

    /// Query audit entries by entity (who performed operations).
    pub fn by_entity(&self, entity: &str) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.entity == entity)
            .collect()
    }

    /// Query audit entries since a timestamp (unix millis).
    pub fn since(&self, since_millis: i64) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.timestamp >= since_millis)
            .collect()
    }

    /// Query audit entries within a time range.
    pub fn between(&self, start_millis: i64, end_millis: i64) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.timestamp >= start_millis && e.timestamp <= end_millis)
            .collect()
    }

    /// Get recent audit entries (last N).
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let mut entries = self.scan();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        entries
    }

    fn scan(&self) -> Vec<AuditEntry> {
        let keys = self.store.scan(AUDIT_PREFIX);
        let mut entries = Vec::new();

        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(entry) = Self::tensor_to_entry(&tensor) {
                    entries.push(entry);
                }
            }
        }

        entries
    }

    fn tensor_to_entry(tensor: &TensorData) -> Option<AuditEntry> {
        let entity = match tensor.get("_entity") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return None,
        };
        let secret_key = match tensor.get("_secret") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return None,
        };
        let timestamp = match tensor.get("_ts") {
            Some(TensorValue::Scalar(ScalarValue::Int(t))) => *t,
            _ => return None,
        };
        let operation = AuditOperation::from_tensor(tensor)?;

        Some(AuditEntry {
            entity,
            secret_key,
            operation,
            timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> TensorStore {
        TensorStore::new()
    }

    #[test]
    fn test_record_and_query_by_secret() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        log.record("user:alice", "api_key", &AuditOperation::Get);
        log.record("user:bob", "api_key", &AuditOperation::Get);
        log.record("user:alice", "other_key", &AuditOperation::Set);

        let entries = log.by_secret("api_key");
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            assert_eq!(entry.secret_key, "api_key");
        }
    }

    #[test]
    fn test_query_by_entity() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        log.record("user:alice", "key1", &AuditOperation::Get);
        log.record("user:alice", "key2", &AuditOperation::Set);
        log.record("user:bob", "key1", &AuditOperation::Get);

        let entries = log.by_entity("user:alice");
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            assert_eq!(entry.entity, "user:alice");
        }
    }

    #[test]
    fn test_query_since() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        let before = AuditLog::now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));

        log.record("user:alice", "key", &AuditOperation::Get);

        let entries = log.since(before);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_grant_operation_details() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Grant {
                to: "user:alice".to_string(),
                permission: "read".to_string(),
            },
        );

        let entries = log.by_secret("secret");
        assert_eq!(entries.len(), 1);

        match &entries[0].operation {
            AuditOperation::Grant { to, permission } => {
                assert_eq!(to, "user:alice");
                assert_eq!(permission, "read");
            },
            _ => panic!("Expected Grant operation"),
        }
    }

    #[test]
    fn test_revoke_operation_details() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Revoke {
                from: "user:alice".to_string(),
            },
        );

        let entries = log.by_secret("secret");
        assert_eq!(entries.len(), 1);

        match &entries[0].operation {
            AuditOperation::Revoke { from } => {
                assert_eq!(from, "user:alice");
            },
            _ => panic!("Expected Revoke operation"),
        }
    }

    #[test]
    fn test_recent_entries() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        for i in 0..10 {
            log.record(&format!("user:{i}"), "key", &AuditOperation::Get);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first
        assert!(recent[0].timestamp >= recent[1].timestamp);
        assert!(recent[1].timestamp >= recent[2].timestamp);
    }

    #[test]
    fn test_all_operation_types() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        log.record("u", "k", &AuditOperation::Get);
        log.record("u", "k", &AuditOperation::Set);
        log.record("u", "k", &AuditOperation::Delete);
        log.record("u", "k", &AuditOperation::Rotate);
        log.record("u", "k", &AuditOperation::List);
        log.record(
            "u",
            "k",
            &AuditOperation::Grant {
                to: "x".to_string(),
                permission: "write".to_string(),
            },
        );
        log.record(
            "u",
            "k",
            &AuditOperation::Revoke {
                from: "x".to_string(),
            },
        );

        let entries = log.by_secret("k");
        assert_eq!(entries.len(), 7);
    }

    #[test]
    fn test_empty_results() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        assert!(log.by_secret("nonexistent").is_empty());
        assert!(log.by_entity("unknown").is_empty());
        assert!(log.recent(10).is_empty());
    }

    #[test]
    fn test_between_range() {
        let store = create_test_store();
        let log = AuditLog::new(&store);

        let t1 = AuditLog::now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));

        log.record("user:alice", "key", &AuditOperation::Get);

        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = AuditLog::now_millis();

        std::thread::sleep(std::time::Duration::from_millis(10));
        log.record("user:bob", "key", &AuditOperation::Set);

        let entries = log.between(t1, t2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }
}
