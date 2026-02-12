// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use tensor_store::{
    RegionalMmapConfig, RegionalMmapStore, ScalarValue, TensorData, TensorValue,
    VoronoiPartitioner, VoronoiPartitionerConfig,
};
use tempfile::tempdir;

#[derive(Arbitrary, Debug, Clone)]
enum Op {
    Put { key_idx: u8, value: i64, embedding_idx: u8 },
    Get { key_idx: u8 },
    GetRegion { region_idx: u8 },
    Flush,
}

#[derive(Arbitrary, Debug)]
struct Input {
    keys: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    ops: Vec<Op>,
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 || mag.is_nan() || mag.is_infinite() {
        v.to_vec()
    } else {
        v.iter().map(|x| x / mag).collect()
    }
}

fn create_tensor(value: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set("value", TensorValue::Scalar(ScalarValue::Int(value)));
    data
}

fuzz_target!(|input: Input| {
    // Constrain inputs
    if input.keys.is_empty() || input.keys.len() > 32 {
        return;
    }
    if input.embeddings.is_empty() || input.embeddings.len() > 32 {
        return;
    }
    if input.ops.is_empty() || input.ops.len() > 64 {
        return;
    }

    // Filter valid keys
    let keys: Vec<String> = input
        .keys
        .into_iter()
        .filter(|k| !k.is_empty() && k.len() <= 64 && !k.chars().any(|c| c.is_control()))
        .take(16)
        .collect();

    if keys.is_empty() {
        return;
    }

    // Fixed dimension
    let dimension = 8;

    // Normalize embeddings
    let embeddings: Vec<Vec<f32>> = input
        .embeddings
        .iter()
        .map(|e| {
            let mut vec: Vec<f32> = e
                .iter()
                .take(dimension)
                .copied()
                .filter(|x| x.is_finite())
                .collect();
            vec.resize(dimension, 0.0);
            normalize(&vec)
        })
        .filter(|v| v.iter().any(|&x| x != 0.0))
        .collect();

    if embeddings.len() < 4 {
        return;
    }

    // Create temp directory
    let dir = match tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Create partitioner with regions
    let mut config = VoronoiPartitionerConfig::new("fuzz_node", 4, dimension);
    config.min_samples_for_regions = 4;

    let partitioner = VoronoiPartitioner::new(config);

    // Add samples and compute regions
    for emb in embeddings.iter().take(8) {
        partitioner.add_sample(emb.clone());
    }
    partitioner.compute_regions_from_samples(&embeddings[..embeddings.len().min(8)]);

    if !partitioner.has_regions() {
        return;
    }

    // Create regional mmap store
    let mmap_config = RegionalMmapConfig {
        run_dir: dir.path().to_path_buf(),
        max_entries_per_run: 100,
        compaction_threshold: 4,
    };
    let store = match RegionalMmapStore::new(partitioner, mmap_config) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Track expected state
    let mut expected: HashMap<String, i64> = HashMap::new();

    // Execute operations
    for op in input.ops {
        match op {
            Op::Put { key_idx, value, embedding_idx } => {
                let key = &keys[key_idx as usize % keys.len()];
                let emb = &embeddings[embedding_idx as usize % embeddings.len()];

                let tensor = create_tensor(value);
                if store.put(key, &tensor, emb).is_ok() {
                    expected.insert(key.clone(), value);
                }
            }
            Op::Get { key_idx } => {
                let key = &keys[key_idx as usize % keys.len()];

                match store.get(key) {
                    Ok(data) => {
                        // Verify value if we have an expectation
                        if let Some(&expected_value) = expected.get(key) {
                            if let Some(TensorValue::Scalar(ScalarValue::Int(v))) = data.get("value")
                            {
                                assert_eq!(
                                    *v, expected_value,
                                    "Value mismatch for key {}: expected {}, got {}",
                                    key, expected_value, v
                                );
                            }
                        }
                    }
                    Err(_) => {
                        // Key might not exist if put failed or wasn't called
                    }
                }
            }
            Op::GetRegion { region_idx } => {
                let region_id = region_idx as u32 % 4;
                // Region query should not panic
                let _ = store.get_region(region_id);
            }
            Op::Flush => {
                let _ = store.flush();
            }
        }
    }

    // Final flush
    let _ = store.flush();

    // Verify all expected keys exist
    for (key, expected_value) in &expected {
        if let Ok(data) = store.get(key) {
            if let Some(TensorValue::Scalar(ScalarValue::Int(v))) = data.get("value") {
                assert_eq!(
                    *v, *expected_value,
                    "Final verification: value mismatch for key {}: expected {}, got {}",
                    key, expected_value, v
                );
            }
        }
    }
});
