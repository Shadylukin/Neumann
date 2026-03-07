// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Trait and types for pluggable checkpoint storage backends.

use crate::{
    error::Result,
    state::{CheckpointInfo, CheckpointState},
};

/// Abstraction over checkpoint persistence.
///
/// Implementations must be thread-safe (`Send + Sync`) and synchronous.
/// The file-based backend (`FileCheckpointStore`) is the primary implementation.
pub trait CheckpointStore: Send + Sync {
    /// Persist a checkpoint, returning an opaque storage ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written to the backing store.
    fn store(&self, state: &CheckpointState) -> Result<String>;

    /// Load a checkpoint by ID or name.
    ///
    /// Resolution order: exact ID match first, then newest-by-name.
    ///
    /// # Errors
    ///
    /// Returns `CheckpointError::NotFound` if no matching checkpoint exists,
    /// or a storage/deserialization error if the checkpoint cannot be read.
    fn load(&self, id_or_name: &str) -> Result<CheckpointState>;

    /// List all checkpoints, newest first.
    ///
    /// If `limit` is `Some(n)`, return at most `n` entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store cannot be enumerated.
    fn list(&self, limit: Option<usize>) -> Result<Vec<CheckpointInfo>>;

    /// Delete a checkpoint by ID or name.
    ///
    /// # Errors
    ///
    /// Returns `CheckpointError::NotFound` if no matching checkpoint exists,
    /// or a storage error if the file cannot be removed.
    fn delete(&self, id_or_name: &str) -> Result<()>;
}
