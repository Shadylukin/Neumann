// SPDX-License-Identifier: MIT OR Apache-2.0
//! Point-in-time recovery: snapshot and restore vault state.

use std::time::{SystemTime, UNIX_EPOCH};

use rand::RngCore;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{encryption::Cipher, Result, VaultError};

/// Storage prefix for snapshot metadata.
const SNAP_PREFIX: &str = "_vsnap:";
/// Storage prefix for snapshot data.
const SNAPDATA_PREFIX: &str = "_vsnapdata:";
/// Storage prefix for vault secrets (matches vault.rs PREFIX).
const VAULT_PREFIX: &str = "_vk:";

/// Metadata about a vault snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultSnapshot {
    /// Unique snapshot identifier.
    pub id: String,
    /// When the snapshot was created (unix millis).
    pub created_at_ms: i64,
    /// Number of secrets in the snapshot.
    pub secret_count: usize,
    /// Human-readable label.
    pub label: String,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn generate_snap_id() -> String {
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("snap_{}", hex_encode(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Encrypt a JSON value string for snapshot storage.
///
/// Returns base64 of `0x01 || nonce(12) || ciphertext`.
fn encrypt_snap_value(cipher: &Cipher, json: &str) -> Result<String> {
    use base64::Engine;
    let (ciphertext, nonce) = cipher.encrypt(json.as_bytes())?;
    let mut buf = Vec::with_capacity(1 + nonce.len() + ciphertext.len());
    buf.push(0x01); // encrypted marker
    buf.extend_from_slice(&nonce);
    buf.extend_from_slice(&ciphertext);
    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

/// Decrypt a snapshot value string.
///
/// Detects `0x01` prefix after base64 decode for encrypted entries;
/// otherwise treats as legacy plaintext JSON.
fn decrypt_snap_value(cipher: &Cipher, encoded: &str) -> Result<String> {
    use base64::Engine;
    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) {
        if bytes.first() == Some(&0x01) && bytes.len() > 13 {
            let nonce = &bytes[1..13];
            let ciphertext = &bytes[13..];
            let plaintext = cipher.decrypt(ciphertext, nonce)?;
            return String::from_utf8(plaintext)
                .map_err(|e| VaultError::CryptoError(format!("invalid UTF-8: {e}")));
        }
    }
    // Legacy plaintext or non-base64 -- return as-is
    Ok(encoded.to_string())
}

/// Compute a Blake2b HMAC over snapshot entry data for integrity.
fn compute_snap_hmac(entries: &[String], hmac_key: Option<&[u8]>) -> String {
    use blake2::digest::Mac;
    let key = hmac_key.unwrap_or(b"neumann-vault-snapshot-hmac");
    let mut mac = blake2::Blake2bMac512::new_from_slice(key).expect("valid key length");
    for entry in entries {
        mac.update(entry.as_bytes());
    }
    let result = mac.finalize().into_bytes();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

/// Create a snapshot of all vault secrets.
///
/// When `cipher` is `Some`, each value entry is encrypted before storage.
pub fn create_snapshot(
    store: &TensorStore,
    label: &str,
    cipher: Option<&Cipher>,
    hmac_key: Option<&[u8]>,
) -> Result<VaultSnapshot> {
    let id = generate_snap_id();
    let now = now_ms();

    // Collect all vault secret entries (keys starting with VAULT_PREFIX)
    let vault_keys: Vec<String> = store.scan(VAULT_PREFIX);

    let secret_count = vault_keys.len();

    // Store the snapshot data: serialized list of (key, tensor_json) pairs
    let mut snap_data = TensorData::new();
    let mut key_list = Vec::new();
    let mut hmac_inputs = Vec::new();

    for (i, vk) in vault_keys.iter().enumerate() {
        if let Ok(tensor) = store.get(vk) {
            // Store each secret's raw data under a numbered field
            let json = serde_json::to_string(&tensor_to_map(&tensor)).unwrap_or_default();

            let stored_value = if let Some(c) = cipher {
                encrypt_snap_value(c, &json)?
            } else {
                json.clone()
            };

            hmac_inputs.push(stored_value.clone());

            snap_data.set(
                format!("_k{i}"),
                TensorValue::Scalar(ScalarValue::String(vk.clone())),
            );
            snap_data.set(
                format!("_v{i}"),
                TensorValue::Scalar(ScalarValue::String(stored_value)),
            );

            // Also snapshot the blob data pointed to by this secret
            if let Some(TensorValue::Pointers(blobs)) = tensor.get("_versions") {
                for blob_key in blobs {
                    if let Ok(blob_tensor) = store.get(blob_key) {
                        let blob_json =
                            serde_json::to_string(&tensor_to_map(&blob_tensor)).unwrap_or_default();
                        key_list.push((blob_key.clone(), blob_json));
                    }
                }
            } else if let Some(TensorValue::Pointer(blob_key)) = tensor.get("_blob") {
                if let Ok(blob_tensor) = store.get(blob_key) {
                    let blob_json =
                        serde_json::to_string(&tensor_to_map(&blob_tensor)).unwrap_or_default();
                    key_list.push((blob_key.clone(), blob_json));
                }
            }
        }
    }

    // Store blob data
    let blob_offset = vault_keys.len();
    for (i, (bk, bv)) in key_list.iter().enumerate() {
        let idx = blob_offset + i;

        let stored_value = if let Some(c) = cipher {
            encrypt_snap_value(c, bv)?
        } else {
            bv.clone()
        };

        hmac_inputs.push(stored_value.clone());

        snap_data.set(
            format!("_k{idx}"),
            TensorValue::Scalar(ScalarValue::String(bk.clone())),
        );
        snap_data.set(
            format!("_v{idx}"),
            TensorValue::Scalar(ScalarValue::String(stored_value)),
        );
    }

    let total_entries = vault_keys.len() + key_list.len();
    snap_data.set(
        "_count",
        TensorValue::Scalar(ScalarValue::Int(total_entries as i64)),
    );

    // Compute and store HMAC for integrity
    let hmac = compute_snap_hmac(&hmac_inputs, hmac_key);
    snap_data.set("_hmac", TensorValue::Scalar(ScalarValue::String(hmac)));

    let data_key = format!("{SNAPDATA_PREFIX}{id}");
    store
        .put(&data_key, snap_data)
        .map_err(|e| VaultError::StorageError(e.to_string()))?;

    // Store snapshot metadata
    let snapshot = VaultSnapshot {
        id: id.clone(),
        created_at_ms: now,
        secret_count,
        label: label.to_string(),
    };

    let meta_key = format!("{SNAP_PREFIX}{id}");
    let mut meta = TensorData::new();
    meta.set("_created_at", TensorValue::Scalar(ScalarValue::Int(now)));
    meta.set(
        "_count",
        TensorValue::Scalar(ScalarValue::Int(secret_count as i64)),
    );
    meta.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String(label.to_string())),
    );
    store
        .put(&meta_key, meta)
        .map_err(|e| VaultError::StorageError(e.to_string()))?;

    Ok(snapshot)
}

/// Restore vault state from a snapshot. Returns number of entries restored.
///
/// When `cipher` is `Some`, encrypted entries (0x01 prefix) are decrypted.
/// Legacy plaintext entries are restored as-is.
pub fn restore_snapshot(
    store: &TensorStore,
    snapshot_id: &str,
    cipher: Option<&Cipher>,
    hmac_key: Option<&[u8]>,
) -> Result<usize> {
    let data_key = format!("{SNAPDATA_PREFIX}{snapshot_id}");
    let snap_data = store
        .get(&data_key)
        .map_err(|_| VaultError::NotFound(format!("snapshot: {snapshot_id}")))?;

    let count = match snap_data.get("_count") {
        Some(TensorValue::Scalar(ScalarValue::Int(n))) => usize::try_from(*n).unwrap_or(0),
        _ => 0,
    };

    // Verify HMAC before restoring
    if let Some(TensorValue::Scalar(ScalarValue::String(stored_hmac))) = snap_data.get("_hmac") {
        let mut hmac_inputs = Vec::new();
        for i in 0..count {
            if let Some(TensorValue::Scalar(ScalarValue::String(val))) =
                snap_data.get(&format!("_v{i}"))
            {
                hmac_inputs.push(val.clone());
            }
        }
        let computed = compute_snap_hmac(&hmac_inputs, hmac_key);
        if computed != *stored_hmac {
            return Err(VaultError::CryptoError(
                "snapshot HMAC verification failed".to_string(),
            ));
        }
    }

    let mut restored = 0;
    for i in 0..count {
        let key_field = format!("_k{i}");
        let val_field = format!("_v{i}");

        let Some(TensorValue::Scalar(ScalarValue::String(key))) = snap_data.get(&key_field) else {
            continue;
        };
        let Some(TensorValue::Scalar(ScalarValue::String(stored_val))) = snap_data.get(&val_field)
        else {
            continue;
        };

        let json = if let Some(c) = cipher {
            decrypt_snap_value(c, stored_val)?
        } else {
            stored_val.clone()
        };

        if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, SerValue>>(&json)
        {
            let tensor = map_to_tensor(&map);
            if store.put(key, tensor).is_ok() {
                restored += 1;
            }
        }
    }

    Ok(restored)
}

/// List all snapshots.
pub fn list_snapshots(store: &TensorStore) -> Vec<VaultSnapshot> {
    let mut results = Vec::new();
    for key in store.scan(SNAP_PREFIX) {
        if let Some(id) = key.strip_prefix(SNAP_PREFIX) {
            if let Ok(tensor) = store.get(&key) {
                let created_at_ms = match tensor.get("_created_at") {
                    Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
                    _ => 0,
                };
                let secret_count = match tensor.get("_count") {
                    Some(TensorValue::Scalar(ScalarValue::Int(v))) => {
                        usize::try_from(*v).unwrap_or(0)
                    },
                    _ => 0,
                };
                let label = match tensor.get("_label") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                results.push(VaultSnapshot {
                    id: id.to_string(),
                    created_at_ms,
                    secret_count,
                    label,
                });
            }
        }
    }
    results.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    results
}

/// Delete a snapshot and its data.
pub fn delete_snapshot(store: &TensorStore, snapshot_id: &str) {
    let meta_key = format!("{SNAP_PREFIX}{snapshot_id}");
    let data_key = format!("{SNAPDATA_PREFIX}{snapshot_id}");
    store.delete(&meta_key).ok();
    store.delete(&data_key).ok();
}

// Serialization helpers for TensorData round-tripping through JSON.
#[derive(Serialize, Deserialize)]
enum SerValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Pointer(String),
    Pointers(Vec<String>),
    Null,
}

fn tensor_to_map(tensor: &TensorData) -> std::collections::HashMap<String, SerValue> {
    let mut map = std::collections::HashMap::new();
    for field in tensor.keys() {
        if let Some(val) = tensor.get(field) {
            let ser = match val {
                TensorValue::Scalar(ScalarValue::String(s)) => SerValue::String(s.clone()),
                TensorValue::Scalar(ScalarValue::Int(i)) => SerValue::Int(*i),
                TensorValue::Scalar(ScalarValue::Float(f)) => SerValue::Float(*f),
                TensorValue::Scalar(ScalarValue::Bool(b)) => SerValue::Bool(*b),
                TensorValue::Scalar(ScalarValue::Bytes(b)) => SerValue::Bytes(b.clone()),
                TensorValue::Scalar(ScalarValue::Null) => SerValue::Null,
                TensorValue::Pointer(p) => SerValue::Pointer(p.clone()),
                TensorValue::Pointers(ps) => SerValue::Pointers(ps.clone()),
                _ => continue,
            };
            map.insert(field.clone(), ser);
        }
    }
    map
}

fn map_to_tensor(map: &std::collections::HashMap<String, SerValue>) -> TensorData {
    let mut tensor = TensorData::new();
    for (field, val) in map {
        match val {
            SerValue::String(s) => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::String(s.clone())));
            },
            SerValue::Int(i) => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::Int(*i)));
            },
            SerValue::Float(f) => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::Float(*f)));
            },
            SerValue::Bool(b) => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::Bool(*b)));
            },
            SerValue::Bytes(b) => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::Bytes(b.clone())));
            },
            SerValue::Pointer(p) => {
                tensor.set(field, TensorValue::Pointer(p.clone()));
            },
            SerValue::Pointers(ps) => {
                tensor.set(field, TensorValue::Pointers(ps.clone()));
            },
            SerValue::Null => {
                tensor.set(field, TensorValue::Scalar(ScalarValue::Null));
            },
        }
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_store_with_secrets() -> TensorStore {
        let store = TensorStore::new();

        // Simulate vault secrets
        let mut t1 = TensorData::new();
        t1.set("_blob", TensorValue::Pointer("blob_001".to_string()));
        t1.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
        );
        store.put(&format!("{VAULT_PREFIX}secret1"), t1).unwrap();

        let mut blob = TensorData::new();
        blob.set(
            "_data",
            TensorValue::Scalar(ScalarValue::Bytes(vec![10, 20, 30])),
        );
        store.put("blob_001", blob).unwrap();

        let mut t2 = TensorData::new();
        t2.set("_blob", TensorValue::Pointer("blob_002".to_string()));
        store.put(&format!("{VAULT_PREFIX}secret2"), t2).unwrap();

        let mut blob2 = TensorData::new();
        blob2.set(
            "_data",
            TensorValue::Scalar(ScalarValue::Bytes(vec![40, 50, 60])),
        );
        store.put("blob_002", blob2).unwrap();

        store
    }

    #[test]
    fn test_create_snapshot() {
        let store = setup_store_with_secrets();
        let snap = create_snapshot(&store, "test backup", None, None).unwrap();

        assert!(snap.id.starts_with("snap_"));
        assert_eq!(snap.secret_count, 2);
        assert_eq!(snap.label, "test backup");
        assert!(snap.created_at_ms > 0);
    }

    #[test]
    fn test_list_snapshots() {
        let store = setup_store_with_secrets();
        create_snapshot(&store, "snap1", None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        create_snapshot(&store, "snap2", None, None).unwrap();

        let snaps = list_snapshots(&store);
        assert_eq!(snaps.len(), 2);
        // Most recent first
        assert!(snaps[0].created_at_ms >= snaps[1].created_at_ms);
    }

    #[test]
    fn test_delete_snapshot() {
        let store = setup_store_with_secrets();
        let snap = create_snapshot(&store, "to delete", None, None).unwrap();

        delete_snapshot(&store, &snap.id);
        let snaps = list_snapshots(&store);
        assert!(snaps.is_empty());
    }

    #[test]
    fn test_restore_snapshot() {
        let store = setup_store_with_secrets();
        let snap = create_snapshot(&store, "backup", None, None).unwrap();

        // Delete the vault secrets
        for key in store.scan(VAULT_PREFIX) {
            store.delete(&key).ok();
        }
        for key in store.scan("blob_") {
            store.delete(&key).ok();
        }

        // Verify they're gone
        let remaining = store.scan(VAULT_PREFIX);
        assert!(remaining.is_empty());

        // Restore
        let count = restore_snapshot(&store, &snap.id, None, None).unwrap();
        assert!(count > 0);

        // Verify secrets are back
        let restored = store.scan(VAULT_PREFIX);
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_restore_nonexistent_snapshot() {
        let store = TensorStore::new();
        let result = restore_snapshot(&store, "nonexistent", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_preserves_data() {
        let store = setup_store_with_secrets();
        let snap = create_snapshot(&store, "verify data", None, None).unwrap();

        // Read original blob data
        let original = store.get("blob_001").unwrap();
        let original_data = match original.get("_data") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => panic!("missing blob data"),
        };

        // Delete and restore
        store.delete("blob_001").ok();
        restore_snapshot(&store, &snap.id, None, None).unwrap();

        let restored = store.get("blob_001").unwrap();
        let restored_data = match restored.get("_data") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => panic!("missing restored data"),
        };

        assert_eq!(original_data, restored_data);
    }

    #[test]
    fn test_empty_store_snapshot() {
        let store = TensorStore::new();
        let snap = create_snapshot(&store, "empty", None, None).unwrap();
        assert_eq!(snap.secret_count, 0);
    }

    #[test]
    fn test_multiple_independent_snapshots() {
        let store = setup_store_with_secrets();
        let snap1 = create_snapshot(&store, "first", None, None).unwrap();

        // Add another secret
        let mut t3 = TensorData::new();
        t3.set("_blob", TensorValue::Pointer("blob_003".to_string()));
        store.put(&format!("{VAULT_PREFIX}secret3"), t3).unwrap();

        let snap2 = create_snapshot(&store, "second", None, None).unwrap();

        assert_eq!(snap1.secret_count, 2);
        assert_eq!(snap2.secret_count, 3);
        assert_ne!(snap1.id, snap2.id);
    }

    #[test]
    fn test_snapshot_serialization() {
        let snap = VaultSnapshot {
            id: "snap_abc".to_string(),
            created_at_ms: 12345,
            secret_count: 10,
            label: "test".to_string(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let deser: VaultSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "snap_abc");
        assert_eq!(deser.secret_count, 10);
    }

    #[test]
    fn test_snapshot_with_versioned_secrets() {
        let store = TensorStore::new();

        let mut t = TensorData::new();
        t.set("_blob", TensorValue::Pointer("blob_v2".to_string()));
        t.set(
            "_versions",
            TensorValue::Pointers(vec!["blob_v1".to_string(), "blob_v2".to_string()]),
        );
        store.put(&format!("{VAULT_PREFIX}versioned"), t).unwrap();

        for v in ["blob_v1", "blob_v2"] {
            let mut blob = TensorData::new();
            blob.set(
                "_data",
                TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
            );
            store.put(v, blob).unwrap();
        }

        let snap = create_snapshot(&store, "versioned", None, None).unwrap();
        assert_eq!(snap.secret_count, 1);
    }

    fn test_cipher() -> Cipher {
        use crate::key::{MasterKey, KEY_SIZE};
        Cipher::from_raw_key(MasterKey::from_bytes([42u8; KEY_SIZE]).snapshot_key())
    }

    #[test]
    fn test_encrypted_snapshot_roundtrip() {
        let store = setup_store_with_secrets();
        let cipher = test_cipher();

        let snap = create_snapshot(&store, "encrypted", Some(&cipher), None).unwrap();
        assert_eq!(snap.secret_count, 2);

        // Delete originals
        for key in store.scan(VAULT_PREFIX) {
            store.delete(&key).ok();
        }
        for key in store.scan("blob_") {
            store.delete(&key).ok();
        }

        // Restore with cipher
        let count = restore_snapshot(&store, &snap.id, Some(&cipher), None).unwrap();
        assert!(count > 0);

        // Verify data is back
        let restored = store.scan(VAULT_PREFIX);
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_encrypted_snapshot_values_not_plaintext() {
        let store = setup_store_with_secrets();
        let cipher = test_cipher();

        let snap = create_snapshot(&store, "enc-check", Some(&cipher), None).unwrap();

        // Read raw snapshot data and verify values are not plaintext JSON
        let data_key = format!("{SNAPDATA_PREFIX}{}", snap.id);
        let snap_data = store.get(&data_key).unwrap();

        if let Some(TensorValue::Scalar(ScalarValue::String(val))) = snap_data.get("_v0") {
            // Should be base64-encoded, not raw JSON
            assert!(
                !val.starts_with('{'),
                "encrypted snapshot value should not be plaintext JSON"
            );
        }
    }

    #[test]
    fn test_encrypted_snapshot_has_hmac() {
        let store = setup_store_with_secrets();
        let cipher = test_cipher();

        let snap = create_snapshot(&store, "hmac-check", Some(&cipher), None).unwrap();

        let data_key = format!("{SNAPDATA_PREFIX}{}", snap.id);
        let snap_data = store.get(&data_key).unwrap();

        match snap_data.get("_hmac") {
            Some(TensorValue::Scalar(ScalarValue::String(hmac))) => {
                assert!(!hmac.is_empty(), "HMAC should not be empty");
            },
            _ => panic!("HMAC field missing from snapshot data"),
        }
    }

    #[test]
    fn test_plaintext_snapshot_still_works() {
        let store = setup_store_with_secrets();

        let snap = create_snapshot(&store, "plain", None, None).unwrap();

        for key in store.scan(VAULT_PREFIX) {
            store.delete(&key).ok();
        }
        for key in store.scan("blob_") {
            store.delete(&key).ok();
        }

        let count = restore_snapshot(&store, &snap.id, None, None).unwrap();
        assert!(count > 0);
    }
}
