//! Incremental (append-only) snapshot format.
//!
//! Supports delta snapshots that only contain changes since a base snapshot.
//! Enables efficient incremental backups and replication.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::use_self)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::return_self_not_must_use)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::format::{CompressedEntry, CompressedSnapshot, FormatError, Header};

/// Magic bytes for delta snapshot format.
pub const DELTA_MAGIC: [u8; 4] = *b"NEUD";

/// Delta format version.
pub const DELTA_VERSION: u16 = 1;

/// Errors from delta operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeltaError {
    #[error("base snapshot not found: {0}")]
    BaseNotFound(String),
    #[error("sequence gap: expected {expected}, got {got}")]
    SequenceGap { expected: u64, got: u64 },
    #[error("delta chain too long: {len} (max {max})")]
    ChainTooLong { len: usize, max: usize },
    #[error("format error: {0}")]
    Format(String),
}

impl From<FormatError> for DeltaError {
    fn from(e: FormatError) -> Self {
        DeltaError::Format(e.to_string())
    }
}

/// Type of change in a delta entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChangeType {
    /// Entry was added or updated.
    Put,
    /// Entry was deleted.
    Delete,
}

/// A single change in a delta snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeltaEntry {
    /// Key of the changed entry.
    pub key: String,
    /// Type of change.
    pub change: ChangeType,
    /// The new value (None for Delete).
    pub value: Option<CompressedEntry>,
    /// Sequence number for ordering.
    pub sequence: u64,
}

/// Header for delta snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeltaHeader {
    pub magic: [u8; 4],
    pub version: u16,
    /// Identifier of the base snapshot.
    pub base_id: String,
    /// Sequence range: (start, end) inclusive.
    pub sequence_range: (u64, u64),
    /// Number of changes.
    pub change_count: u64,
    /// Timestamp (Unix seconds).
    pub created_at: u64,
}

impl DeltaHeader {
    fn validate(&self) -> Result<(), DeltaError> {
        if self.magic != DELTA_MAGIC {
            return Err(DeltaError::Format("invalid delta magic bytes".into()));
        }
        if self.version > DELTA_VERSION {
            return Err(DeltaError::Format(format!(
                "unsupported delta version: {}",
                self.version
            )));
        }
        Ok(())
    }
}

/// Complete delta snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeltaSnapshot {
    pub header: DeltaHeader,
    pub entries: Vec<DeltaEntry>,
}

impl DeltaSnapshot {
    /// Serialize to bytes.
    pub fn serialize(&self) -> Result<Vec<u8>, DeltaError> {
        bincode::serialize(self).map_err(|e| DeltaError::Format(e.to_string()))
    }

    /// Deserialize from bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, DeltaError> {
        let delta: Self =
            bincode::deserialize(bytes).map_err(|e| DeltaError::Format(e.to_string()))?;
        delta.header.validate()?;
        Ok(delta)
    }
}

/// Builder for creating delta snapshots.
pub struct DeltaBuilder {
    base_id: String,
    start_sequence: u64,
    current_sequence: u64,
    entries: Vec<DeltaEntry>,
}

impl DeltaBuilder {
    /// Create a new delta builder referencing a base snapshot.
    pub fn new(base_id: impl Into<String>, start_sequence: u64) -> Self {
        Self {
            base_id: base_id.into(),
            start_sequence,
            current_sequence: start_sequence,
            entries: Vec::new(),
        }
    }

    /// Record a put (add/update) operation.
    pub fn put(&mut self, key: impl Into<String>, entry: CompressedEntry) {
        self.entries.push(DeltaEntry {
            key: key.into(),
            change: ChangeType::Put,
            value: Some(entry),
            sequence: self.current_sequence,
        });
        self.current_sequence += 1;
    }

    /// Record a delete operation.
    pub fn delete(&mut self, key: impl Into<String>) {
        self.entries.push(DeltaEntry {
            key: key.into(),
            change: ChangeType::Delete,
            value: None,
            sequence: self.current_sequence,
        });
        self.current_sequence += 1;
    }

    #[must_use]
    pub fn change_count(&self) -> usize {
        self.entries.len()
    }

    /// Build the delta snapshot.
    #[must_use]
    pub fn build(self) -> DeltaSnapshot {
        let end_sequence = if self.entries.is_empty() {
            self.start_sequence
        } else {
            self.current_sequence - 1
        };

        DeltaSnapshot {
            header: DeltaHeader {
                magic: DELTA_MAGIC,
                version: DELTA_VERSION,
                base_id: self.base_id,
                sequence_range: (self.start_sequence, end_sequence),
                change_count: self.entries.len() as u64,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            },
            entries: self.entries,
        }
    }
}

/// Apply a delta to a base snapshot, producing a new full snapshot.
pub fn apply_delta(
    base: &CompressedSnapshot,
    delta: &DeltaSnapshot,
) -> Result<CompressedSnapshot, DeltaError> {
    // Build map from base entries
    let mut entries: HashMap<String, CompressedEntry> = base
        .entries
        .iter()
        .map(|e| (e.key.clone(), e.clone()))
        .collect();

    // Apply delta changes
    for delta_entry in &delta.entries {
        match delta_entry.change {
            ChangeType::Put => {
                if let Some(value) = &delta_entry.value {
                    entries.insert(delta_entry.key.clone(), value.clone());
                }
            },
            ChangeType::Delete => {
                entries.remove(&delta_entry.key);
            },
        }
    }

    let entry_vec: Vec<_> = entries.into_values().collect();
    let header = Header::new(base.header.config.clone(), entry_vec.len() as u64);

    Ok(CompressedSnapshot {
        header,
        entries: entry_vec,
    })
}

/// Merge multiple deltas into a single delta.
/// Deltas must be in sequence order.
pub fn merge_deltas(deltas: &[DeltaSnapshot]) -> Result<DeltaSnapshot, DeltaError> {
    if deltas.is_empty() {
        return Err(DeltaError::Format("no deltas to merge".into()));
    }

    let base_id = deltas[0].header.base_id.clone();
    let start_sequence = deltas[0].header.sequence_range.0;

    // Track latest state for each key
    let mut latest: HashMap<String, DeltaEntry> = HashMap::new();

    for delta in deltas {
        for entry in &delta.entries {
            latest.insert(entry.key.clone(), entry.clone());
        }
    }

    // Rebuild entries in sequence order
    let mut entries: Vec<_> = latest.into_values().collect();
    entries.sort_by_key(|e| e.sequence);

    let end_sequence = entries.last().map(|e| e.sequence).unwrap_or(start_sequence);

    Ok(DeltaSnapshot {
        header: DeltaHeader {
            magic: DELTA_MAGIC,
            version: DELTA_VERSION,
            base_id,
            sequence_range: (start_sequence, end_sequence),
            change_count: entries.len() as u64,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        },
        entries,
    })
}

/// Create a delta by comparing two snapshots.
pub fn diff_snapshots(
    old: &CompressedSnapshot,
    new: &CompressedSnapshot,
    base_ref: impl Into<String>,
) -> DeltaSnapshot {
    let old_keys: HashMap<String, &CompressedEntry> =
        old.entries.iter().map(|e| (e.key.clone(), e)).collect();
    let new_keys: HashMap<String, &CompressedEntry> =
        new.entries.iter().map(|e| (e.key.clone(), e)).collect();

    let all_keys: HashSet<_> = old_keys.keys().chain(new_keys.keys()).cloned().collect();

    let mut builder = DeltaBuilder::new(base_ref, 0);

    for key in all_keys {
        match (old_keys.get(&key), new_keys.get(&key)) {
            (None, Some(new_entry)) => {
                // Added
                builder.put(key, (*new_entry).clone());
            },
            (Some(_), None) => {
                // Deleted
                builder.delete(key);
            },
            (Some(old_entry), Some(new_entry)) => {
                // Check if changed (simple equality check)
                if *old_entry != *new_entry {
                    builder.put(key, (*new_entry).clone());
                }
            },
            (None, None) => unreachable!(),
        }
    }

    builder.build()
}

/// Chain of deltas with efficient lookup.
pub struct DeltaChain {
    base: CompressedSnapshot,
    deltas: Vec<DeltaSnapshot>,
    /// Maximum chain length before compaction is recommended.
    max_chain_len: usize,
}

impl DeltaChain {
    /// Create a new chain with a base snapshot.
    pub fn new(base: CompressedSnapshot) -> Self {
        Self {
            base,
            deltas: Vec::new(),
            max_chain_len: 100,
        }
    }

    /// Set maximum chain length.
    pub fn with_max_chain_len(mut self, len: usize) -> Self {
        self.max_chain_len = len;
        self
    }

    /// Add a delta to the chain.
    pub fn push(&mut self, delta: DeltaSnapshot) -> Result<(), DeltaError> {
        if self.deltas.len() >= self.max_chain_len {
            return Err(DeltaError::ChainTooLong {
                len: self.deltas.len() + 1,
                max: self.max_chain_len,
            });
        }
        self.deltas.push(delta);
        Ok(())
    }

    /// Get the current logical state of a key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<CompressedEntry> {
        // Check deltas in reverse order for latest state
        for delta in self.deltas.iter().rev() {
            for entry in delta.entries.iter().rev() {
                if entry.key == key {
                    return match entry.change {
                        ChangeType::Put => entry.value.clone(),
                        ChangeType::Delete => None,
                    };
                }
            }
        }

        // Fall back to base
        self.base.entries.iter().find(|e| e.key == key).cloned()
    }

    /// Compact all deltas into a new base snapshot.
    pub fn compact(&self) -> Result<CompressedSnapshot, DeltaError> {
        let mut current = self.base.clone();
        for delta in &self.deltas {
            current = apply_delta(&current, delta)?;
        }
        Ok(current)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Check if compaction is recommended.
    #[must_use]
    pub fn should_compact(&self, threshold: usize) -> bool {
        self.deltas.len() >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        format::{CompressedScalar, CompressedValue},
        CompressionConfig,
    };

    fn make_entry(key: &str, value: i64) -> CompressedEntry {
        CompressedEntry {
            key: key.to_string(),
            fields: HashMap::from([(
                "value".to_string(),
                CompressedValue::Scalar(CompressedScalar::Int(value)),
            )]),
        }
    }

    fn make_base_snapshot(entries: Vec<(&str, i64)>) -> CompressedSnapshot {
        let entries: Vec<_> = entries.into_iter().map(|(k, v)| make_entry(k, v)).collect();
        let header = Header::new(CompressionConfig::default(), entries.len() as u64);
        CompressedSnapshot { header, entries }
    }

    #[test]
    fn test_delta_builder_new() {
        let builder = DeltaBuilder::new("base_123", 0);
        assert_eq!(builder.change_count(), 0);
    }

    #[test]
    fn test_delta_builder_put() {
        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("key1", make_entry("key1", 42));
        assert_eq!(builder.change_count(), 1);
    }

    #[test]
    fn test_delta_builder_delete() {
        let mut builder = DeltaBuilder::new("base", 0);
        builder.delete("key1");
        assert_eq!(builder.change_count(), 1);
    }

    #[test]
    fn test_delta_build() {
        let mut builder = DeltaBuilder::new("base_abc", 10);
        builder.put("a", make_entry("a", 1));
        builder.delete("b");
        let delta = builder.build();

        assert_eq!(delta.header.base_id, "base_abc");
        assert_eq!(delta.header.sequence_range, (10, 11));
        assert_eq!(delta.header.change_count, 2);
        assert_eq!(delta.entries.len(), 2);
    }

    #[test]
    fn test_delta_serialize_deserialize() {
        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("x", make_entry("x", 100));
        let delta = builder.build();

        let bytes = delta.serialize().unwrap();
        let restored = DeltaSnapshot::deserialize(&bytes).unwrap();

        assert_eq!(delta.header.base_id, restored.header.base_id);
        assert_eq!(delta.entries.len(), restored.entries.len());
    }

    #[test]
    fn test_apply_delta_puts() {
        let base = make_base_snapshot(vec![("a", 1), ("b", 2)]);

        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("c", make_entry("c", 3)); // Add new
        builder.put("a", make_entry("a", 10)); // Update existing
        let delta = builder.build();

        let result = apply_delta(&base, &delta).unwrap();

        assert_eq!(result.entries.len(), 3);
        let keys: HashSet<_> = result.entries.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains("a"));
        assert!(keys.contains("b"));
        assert!(keys.contains("c"));
    }

    #[test]
    fn test_apply_delta_deletes() {
        let base = make_base_snapshot(vec![("a", 1), ("b", 2), ("c", 3)]);

        let mut builder = DeltaBuilder::new("base", 0);
        builder.delete("b");
        let delta = builder.build();

        let result = apply_delta(&base, &delta).unwrap();

        assert_eq!(result.entries.len(), 2);
        let keys: HashSet<_> = result.entries.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains("a"));
        assert!(!keys.contains("b"));
        assert!(keys.contains("c"));
    }

    #[test]
    fn test_apply_delta_mixed() {
        let base = make_base_snapshot(vec![("a", 1), ("b", 2)]);

        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("c", make_entry("c", 3));
        builder.delete("a");
        builder.put("b", make_entry("b", 20));
        let delta = builder.build();

        let result = apply_delta(&base, &delta).unwrap();

        assert_eq!(result.entries.len(), 2);
        let keys: HashSet<_> = result.entries.iter().map(|e| e.key.as_str()).collect();
        assert!(!keys.contains("a"));
        assert!(keys.contains("b"));
        assert!(keys.contains("c"));
    }

    #[test]
    fn test_merge_deltas() {
        let mut b1 = DeltaBuilder::new("base", 0);
        b1.put("a", make_entry("a", 1));
        b1.put("b", make_entry("b", 2));
        let delta1 = b1.build();

        let mut b2 = DeltaBuilder::new("base", 2);
        b2.put("c", make_entry("c", 3));
        b2.delete("a");
        let delta2 = b2.build();

        let merged = merge_deltas(&[delta1, delta2]).unwrap();

        assert_eq!(merged.entries.len(), 3); // b=put, c=put, a=delete
    }

    #[test]
    fn test_diff_snapshots() {
        let old = make_base_snapshot(vec![("a", 1), ("b", 2), ("c", 3)]);
        let new = make_base_snapshot(vec![("a", 10), ("c", 3), ("d", 4)]); // a changed, b deleted, d added

        let delta = diff_snapshots(&old, &new, "old_snapshot");

        // Should have: a=put(changed), b=delete, d=put(new)
        assert_eq!(delta.entries.len(), 3);

        let changes: HashMap<_, _> = delta
            .entries
            .iter()
            .map(|e| (e.key.as_str(), e.change))
            .collect();
        assert_eq!(changes.get("a"), Some(&ChangeType::Put));
        assert_eq!(changes.get("b"), Some(&ChangeType::Delete));
        assert_eq!(changes.get("d"), Some(&ChangeType::Put));
    }

    #[test]
    fn test_delta_chain_get() {
        let base = make_base_snapshot(vec![("a", 1), ("b", 2)]);
        let mut chain = DeltaChain::new(base);

        // Add delta that modifies 'a' and deletes 'b'
        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("a", make_entry("a", 100));
        builder.delete("b");
        chain.push(builder.build()).unwrap();

        // Get from chain
        let a = chain.get("a").unwrap();
        assert_eq!(a.key, "a");
        assert!(chain.get("b").is_none()); // Deleted
    }

    #[test]
    fn test_delta_chain_compact() {
        let base = make_base_snapshot(vec![("a", 1)]);
        let mut chain = DeltaChain::new(base);

        let mut b1 = DeltaBuilder::new("base", 0);
        b1.put("b", make_entry("b", 2));
        chain.push(b1.build()).unwrap();

        let mut b2 = DeltaBuilder::new("base", 1);
        b2.put("c", make_entry("c", 3));
        chain.push(b2.build()).unwrap();

        let compacted = chain.compact().unwrap();
        assert_eq!(compacted.entries.len(), 3);
    }

    #[test]
    fn test_delta_chain_should_compact() {
        let base = make_base_snapshot(vec![]);
        let chain = DeltaChain::new(base);

        assert!(!chain.should_compact(5));
    }

    #[test]
    fn test_delta_chain_too_long() {
        let base = make_base_snapshot(vec![]);
        let mut chain = DeltaChain::new(base).with_max_chain_len(2);

        chain.push(DeltaBuilder::new("b", 0).build()).unwrap();
        chain.push(DeltaBuilder::new("b", 0).build()).unwrap();

        let result = chain.push(DeltaBuilder::new("b", 0).build());
        assert!(matches!(result, Err(DeltaError::ChainTooLong { .. })));
    }

    #[test]
    fn test_delta_header_validation() {
        let mut builder = DeltaBuilder::new("base", 0);
        builder.put("x", make_entry("x", 1));
        let mut delta = builder.build();

        // Valid header
        assert!(delta.header.validate().is_ok());

        // Invalid magic
        delta.header.magic = *b"BAAD";
        assert!(delta.header.validate().is_err());
    }

    #[test]
    fn test_empty_delta() {
        let builder = DeltaBuilder::new("base", 5);
        let delta = builder.build();

        assert_eq!(delta.header.change_count, 0);
        assert_eq!(delta.header.sequence_range, (5, 5));
        assert!(delta.entries.is_empty());
    }
}
