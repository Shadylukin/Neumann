#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{ScalarValue, SlabRouter, TensorData, TensorValue};

#[derive(Arbitrary, Debug, Clone)]
enum Op {
    Put { key: String, value: i64 },
    Get { key: String },
    Delete { key: String },
    Exists { key: String },
    Scan { prefix: String },
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fn create_tensor(value: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set("id", TensorValue::Scalar(ScalarValue::Int(value)));
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(format!("item_{}", value))),
    );
    data
}

fuzz_target!(|input: Input| {
    // Limit ops to prevent timeout
    if input.ops.len() > 100 {
        return;
    }

    let router = SlabRouter::new();
    let mut expected_keys = std::collections::HashMap::new();

    for op in input.ops {
        match op {
            Op::Put { key, value } => {
                // Skip very long keys
                if key.len() > 256 {
                    continue;
                }
                let tensor = create_tensor(value);
                let _ = router.put(&key, tensor);
                expected_keys.insert(key, value);
            },
            Op::Get { key } => {
                let result = router.get(&key);
                if let Some(&expected_value) = expected_keys.get(&key) {
                    // Key should exist
                    assert!(
                        result.is_ok(),
                        "Expected key {} to exist with value {}",
                        key,
                        expected_value
                    );
                    let data = result.unwrap();
                    if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = data.get("id") {
                        assert_eq!(
                            *id, expected_value,
                            "Value mismatch for key {}: expected {}, got {}",
                            key, expected_value, id
                        );
                    }
                }
            },
            Op::Delete { key } => {
                let _ = router.delete(&key);
                expected_keys.remove(&key);
            },
            Op::Exists { key } => {
                let exists = router.exists(&key);
                let should_exist = expected_keys.contains_key(&key);
                assert_eq!(
                    exists, should_exist,
                    "Exists mismatch for key {}: expected {}, got {}",
                    key, should_exist, exists
                );
            },
            Op::Scan { prefix } => {
                if prefix.len() > 64 {
                    continue;
                }
                let results = router.scan(&prefix);
                // Verify all returned keys start with prefix
                for key in results {
                    assert!(
                        key.starts_with(&prefix),
                        "Scan returned key {} that doesn't start with prefix {}",
                        key,
                        prefix
                    );
                }
            },
        }
    }

    // Verify final state
    assert_eq!(
        router.len(),
        expected_keys.len(),
        "Router len {} != expected keys len {}",
        router.len(),
        expected_keys.len()
    );
});
