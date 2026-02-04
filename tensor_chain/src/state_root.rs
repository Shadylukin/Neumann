// SPDX-License-Identifier: MIT OR Apache-2.0
//! Utilities for computing and validating state roots.

use sha2::{Digest, Sha256};

use crate::{
    block::BlockHash,
    error::{ChainError, Result},
};

/// Compute a deterministic state root for the current TensorStore contents.
///
/// The state root is the SHA-256 hash of all keys and values, iterated in
/// sorted order to avoid dependence on internal storage layout.
pub fn compute_state_root(store: &tensor_store::TensorStore) -> Result<BlockHash> {
    let mut hasher = Sha256::new();
    let mut keys = store.scan("");
    keys.sort();

    for key in keys {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());

        let data = store
            .get(&key)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        let mut field_keys: Vec<&String> = data.keys().collect();
        field_keys.sort();
        hasher.update((field_keys.len() as u64).to_le_bytes());

        for field_key in field_keys {
            hasher.update((field_key.len() as u64).to_le_bytes());
            hasher.update(field_key.as_bytes());

            let value = data
                .get(field_key)
                .ok_or_else(|| ChainError::StorageError("missing field".to_string()))?;
            let value_bytes = bitcode::serialize(value)
                .map_err(|e| ChainError::SerializationError(e.to_string()))?;
            hasher.update((value_bytes.len() as u64).to_le_bytes());
            hasher.update(value_bytes);
        }
    }

    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_state_root_changes_with_store() {
        let store = tensor_store::TensorStore::new();
        let root_empty = compute_state_root(&store).unwrap();

        let mut data = tensor_store::TensorData::new();
        data.set(
            "data",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(vec![1, 2, 3])),
        );
        store.put("key1", data).unwrap();

        let root_after = compute_state_root(&store).unwrap();
        assert_ne!(root_empty, root_after);
    }
}
