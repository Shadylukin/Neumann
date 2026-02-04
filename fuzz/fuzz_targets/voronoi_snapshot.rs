// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{VoronoiPartitioner, VoronoiPartitionerConfig};

#[derive(Arbitrary, Debug)]
struct Sample {
    values: Vec<f32>,
}

#[derive(Arbitrary, Debug)]
struct Input {
    node_id: String,
    num_regions: u8,
    dimension: u8,
    samples: Vec<Sample>,
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 || mag.is_nan() || mag.is_infinite() {
        v.to_vec()
    } else {
        v.iter().map(|x| x / mag).collect()
    }
}

fuzz_target!(|input: Input| {
    // Constrain inputs for performance
    let dimension = (input.dimension as usize).clamp(4, 64);
    let num_regions = (input.num_regions as usize).clamp(2, 16);

    // Limit samples
    if input.samples.is_empty() || input.samples.len() > 100 {
        return;
    }

    // Skip invalid node IDs
    if input.node_id.is_empty() || input.node_id.len() > 64 {
        return;
    }
    if input.node_id.chars().any(|c| c.is_control()) {
        return;
    }

    // Prepare samples
    let samples: Vec<Vec<f32>> = input
        .samples
        .iter()
        .map(|s| {
            let mut vec: Vec<f32> = s
                .values
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

    if samples.len() < num_regions {
        return;
    }

    // Create partitioner with config requiring fewer samples
    let mut config = VoronoiPartitionerConfig::new(&input.node_id, num_regions, dimension);
    config.min_samples_for_regions = num_regions;

    let partitioner = VoronoiPartitioner::new(config);

    // Add samples
    for sample in &samples {
        partitioner.add_sample(sample.clone());
    }

    // Compute regions from samples
    partitioner.compute_regions_from_samples(&samples);

    if !partitioner.has_regions() {
        return;
    }

    // Take snapshot
    let snapshot = partitioner.snapshot();

    // Restore from snapshot
    let restored = VoronoiPartitioner::restore(snapshot.clone());

    // Verify restoration
    assert_eq!(
        restored.has_regions(),
        partitioner.has_regions(),
        "has_regions mismatch after restore"
    );

    // Verify region assignment is consistent
    for sample in &samples {
        let original_region = partitioner.region_id_for_embedding(sample);
        let restored_region = restored.region_id_for_embedding(sample);
        assert_eq!(
            original_region, restored_region,
            "Region assignment mismatch for sample"
        );
    }

    // Verify locality key generation
    if let Some(key1) = partitioner.locality_key_for_embedding(&samples[0]) {
        if let Some(key2) = restored.locality_key_for_embedding(&samples[0]) {
            // Keys should be in same region
            assert_eq!(
                key1.region_id(),
                key2.region_id(),
                "Locality key region mismatch"
            );
        }
    }

    // Double roundtrip to verify stability
    let snapshot2 = restored.snapshot();
    let restored2 = VoronoiPartitioner::restore(snapshot2);

    for sample in &samples {
        let r1 = restored.region_id_for_embedding(sample);
        let r2 = restored2.region_id_for_embedding(sample);
        assert_eq!(r1, r2, "Double roundtrip region mismatch");
    }
});
