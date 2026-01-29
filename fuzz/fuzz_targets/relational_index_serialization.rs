// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{ScalarValue, TensorData, TensorValue};

#[derive(Arbitrary, Debug)]
struct IndexInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    ValidRoundtrip { row_ids: Vec<u64> },
    DecodeRaw { raw_bytes: Vec<u8> },
    LegacyVectorFormat { values: Vec<f32> },
    EmptyIds,
    MalformedBytes { raw_bytes: Vec<u8> },
}

fuzz_target!(|input: IndexInput| {
    match input.test_case {
        TestCase::ValidRoundtrip { row_ids } => {
            // Limit size to prevent OOM
            let row_ids: Vec<u64> = row_ids.into_iter().take(10_000).collect();

            // Serialize
            let tensor = id_list_to_tensor(&row_ids);

            // Deserialize
            let recovered = tensor_to_id_list(&tensor);
            assert!(recovered.is_ok(), "Valid roundtrip should succeed");
            assert_eq!(recovered.unwrap(), row_ids, "Roundtrip mismatch");
        }
        TestCase::DecodeRaw { raw_bytes } => {
            let raw_bytes: Vec<u8> = raw_bytes.into_iter().take(80_000).collect();
            let mut tensor = TensorData::new();
            tensor.set("ids", TensorValue::Scalar(ScalarValue::Bytes(raw_bytes)));
            // Should not panic, may return error
            let _ = tensor_to_id_list(&tensor);
        }
        TestCase::LegacyVectorFormat { values } => {
            let values: Vec<f32> = values.into_iter().take(10_000).collect();
            let mut tensor = TensorData::new();
            tensor.set("ids", TensorValue::Vector(values.clone()));
            let result = tensor_to_id_list(&tensor);
            // Legacy format should be handled gracefully
            if let Ok(recovered) = result {
                assert_eq!(recovered.len(), values.len());
            }
        }
        TestCase::EmptyIds => {
            let tensor = TensorData::new();
            let result = tensor_to_id_list(&tensor);
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }
        TestCase::MalformedBytes { raw_bytes } => {
            // Ensure bytes are not a multiple of 8
            let mut raw_bytes: Vec<u8> = raw_bytes.into_iter().take(10_000).collect();
            if raw_bytes.len() % 8 == 0 && !raw_bytes.is_empty() {
                raw_bytes.push(0);
            }
            if raw_bytes.is_empty() {
                raw_bytes.push(0);
            }
            let mut tensor = TensorData::new();
            tensor.set("ids", TensorValue::Scalar(ScalarValue::Bytes(raw_bytes)));
            let result = tensor_to_id_list(&tensor);
            // Should return error for malformed bytes
            assert!(result.is_err());
        }
    }
});

fn id_list_to_tensor(ids: &[u64]) -> TensorData {
    let mut tensor = TensorData::new();
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    tensor.set("ids", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
    tensor
}

fn tensor_to_id_list(tensor: &TensorData) -> Result<Vec<u64>, &'static str> {
    match tensor.get("ids") {
        Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) => {
            if bytes.len() % 8 != 0 {
                return Err("bytes not multiple of 8");
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let arr: [u8; 8] = chunk.try_into().expect("chunk is 8 bytes");
                    u64::from_le_bytes(arr)
                })
                .collect())
        }
        Some(TensorValue::Vector(v)) => {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok(v.iter().map(|f| *f as u64).collect())
        }
        Some(_) => Err("unexpected type"),
        None => Ok(Vec::new()),
    }
}
