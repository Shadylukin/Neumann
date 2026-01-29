// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{
    snapshot_load, snapshot_save, ScalarValue, SlabRouter, TensorData, TensorValue,
};

#[derive(Arbitrary, Debug)]
struct Entry {
    key: String,
    id: i64,
    name: String,
}

#[derive(Arbitrary, Debug)]
struct Input {
    entries: Vec<Entry>,
}

fn create_tensor(id: i64, name: &str) -> TensorData {
    let mut data = TensorData::new();
    data.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(name.to_string())),
    );
    data
}

fuzz_target!(|input: Input| {
    // Limit entries to prevent OOM/timeout
    if input.entries.len() > 50 {
        return;
    }

    // Create unique temp file for this run
    let temp_dir = std::env::temp_dir();
    let file_name = format!("fuzz_snapshot_{:x}.bin", std::process::id());
    let path = temp_dir.join(file_name);

    // Create router with entries
    let router = SlabRouter::new();
    let mut expected = std::collections::HashMap::new();

    for entry in &input.entries {
        // Skip invalid keys
        if entry.key.is_empty() || entry.key.len() > 128 {
            continue;
        }
        // Skip keys with control characters
        if entry.key.chars().any(|c| c.is_control()) {
            continue;
        }
        let tensor = create_tensor(entry.id, &entry.name);
        let _ = router.put(&entry.key, tensor);
        expected.insert(entry.key.clone(), (entry.id, entry.name.clone()));
    }

    // Save snapshot
    if snapshot_save(&router, &path).is_err() {
        let _ = std::fs::remove_file(&path);
        return;
    }

    // Load snapshot
    let loaded: SlabRouter = match snapshot_load(&path) {
        Ok(r) => r,
        Err(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        },
    };

    // Verify all expected keys exist and have correct values
    for (key, (expected_id, expected_name)) in &expected {
        assert!(loaded.exists(key), "Key {} missing after roundtrip", key);

        let data = loaded.get(key).expect("Key should exist");

        if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = data.get("id") {
            assert_eq!(
                *id, *expected_id,
                "ID mismatch for key {}: expected {}, got {}",
                key, expected_id, id
            );
        }

        if let Some(TensorValue::Scalar(ScalarValue::String(name))) = data.get("name") {
            assert_eq!(
                name, expected_name,
                "Name mismatch for key {}: expected {}, got {}",
                key, expected_name, name
            );
        }
    }

    // Verify count matches
    assert_eq!(
        loaded.len(),
        expected.len(),
        "Entry count mismatch: expected {}, got {}",
        expected.len(),
        loaded.len()
    );

    // Cleanup
    let _ = std::fs::remove_file(&path);
});
