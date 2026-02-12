// SPDX-License-Identifier: MIT OR Apache-2.0
//! Seal/unseal operations for vault key zeroization.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{Result, VaultError};

/// Storage key for persisted seal state.
const SEAL_STATE_KEY: &str = "_vault:seal_state";

/// Seal state shared across the vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealState {
    /// Whether the vault is sealed.
    pub sealed: bool,
}

/// Atomic seal flag for zero-cost sealed checks.
pub struct SealGuard {
    sealed: Arc<AtomicBool>,
}

impl SealGuard {
    pub fn new() -> Self {
        Self {
            sealed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed.load(Ordering::Acquire)
    }

    pub fn seal(&self) {
        self.sealed.store(true, Ordering::Release);
    }

    pub fn unseal(&self) {
        self.sealed.store(false, Ordering::Release);
    }

    /// Create a `SealGuard` initialized from persisted state.
    ///
    /// Returns unsealed if no persisted state exists (first-run compatibility).
    pub fn from_store(store: &TensorStore) -> Self {
        let guard = Self::new();
        if Self::load(store) {
            guard.seal();
        }
        guard
    }

    /// Persist the current seal state to the store.
    pub fn persist(&self, store: &TensorStore) {
        let state = SealState {
            sealed: self.is_sealed(),
        };
        let json = serde_json::to_string(&state).unwrap_or_default();
        let mut tensor = TensorData::new();
        tensor.set("_state", TensorValue::Scalar(ScalarValue::String(json)));
        let _ = store.put(SEAL_STATE_KEY, tensor);
    }

    /// Load persisted seal state. Returns `false` (unsealed) if not found.
    fn load(store: &TensorStore) -> bool {
        store
            .get(SEAL_STATE_KEY)
            .ok()
            .and_then(|tensor| {
                if let Some(TensorValue::Scalar(ScalarValue::String(json))) = tensor.get("_state") {
                    serde_json::from_str::<SealState>(json).ok()
                } else {
                    None
                }
            })
            .is_some_and(|s| s.sealed)
    }

    /// Check if vault is sealed and return error if so.
    pub fn check_sealed(&self) -> Result<()> {
        if self.is_sealed() {
            Err(VaultError::Sealed(
                "vault is sealed; unseal required".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

impl Clone for SealGuard {
    fn clone(&self) -> Self {
        Self {
            sealed: Arc::clone(&self.sealed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_guard_default_unsealed() {
        let guard = SealGuard::new();
        assert!(!guard.is_sealed());
        assert!(guard.check_sealed().is_ok());
    }

    #[test]
    fn test_seal_and_unseal() {
        let guard = SealGuard::new();

        guard.seal();
        assert!(guard.is_sealed());
        assert!(guard.check_sealed().is_err());

        guard.unseal();
        assert!(!guard.is_sealed());
        assert!(guard.check_sealed().is_ok());
    }

    #[test]
    fn test_seal_error_type() {
        let guard = SealGuard::new();
        guard.seal();

        let err = guard.check_sealed().unwrap_err();
        assert!(matches!(err, VaultError::Sealed(_)));
    }

    #[test]
    fn test_idempotent_seal() {
        let guard = SealGuard::new();
        guard.seal();
        guard.seal(); // Double seal is fine
        assert!(guard.is_sealed());

        guard.unseal();
        guard.unseal(); // Double unseal is fine
        assert!(!guard.is_sealed());
    }

    #[test]
    fn test_clone_shares_state() {
        let guard1 = SealGuard::new();
        let guard2 = guard1.clone();

        guard1.seal();
        assert!(guard2.is_sealed());

        guard2.unseal();
        assert!(!guard1.is_sealed());
    }

    #[test]
    fn test_concurrent_seal() {
        let guard = SealGuard::new();
        let g1 = guard.clone();
        let g2 = guard.clone();

        let h1 = std::thread::spawn(move || {
            g1.seal();
        });

        let h2 = std::thread::spawn(move || g2.is_sealed());

        h1.join().unwrap();
        let _ = h2.join().unwrap();
        // After seal thread completes, guard should be sealed
        assert!(guard.is_sealed());
    }

    #[test]
    fn test_persist_sealed() {
        let store = TensorStore::new();
        let guard = SealGuard::new();
        guard.seal();
        guard.persist(&store);

        let loaded = SealGuard::from_store(&store);
        assert!(loaded.is_sealed());
    }

    #[test]
    fn test_persist_unsealed() {
        let store = TensorStore::new();
        let guard = SealGuard::new();
        guard.seal();
        guard.persist(&store);

        guard.unseal();
        guard.persist(&store);

        let loaded = SealGuard::from_store(&store);
        assert!(!loaded.is_sealed());
    }

    #[test]
    fn test_default_unsealed_on_empty_store() {
        let store = TensorStore::new();
        let guard = SealGuard::from_store(&store);
        assert!(!guard.is_sealed());
    }

    #[test]
    fn test_from_store_roundtrip() {
        let store = TensorStore::new();

        let guard = SealGuard::new();
        guard.seal();
        guard.persist(&store);

        let loaded = SealGuard::from_store(&store);
        assert!(loaded.is_sealed());

        loaded.unseal();
        loaded.persist(&store);

        let reloaded = SealGuard::from_store(&store);
        assert!(!reloaded.is_sealed());
    }

    #[test]
    fn test_seal_state_serialization() {
        let state = SealState { sealed: true };
        let json = serde_json::to_string(&state).unwrap();
        let deser: SealState = serde_json::from_str(&json).unwrap();
        assert!(deser.sealed);
    }
}
