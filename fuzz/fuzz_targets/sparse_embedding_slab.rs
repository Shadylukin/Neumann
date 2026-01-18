#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{EmbeddingSlab, EntityId};

#[derive(Arbitrary, Debug)]
struct SlabInput {
    // Limit dimension to avoid OOM
    dimension: u8,
    embeddings: Vec<EmbeddingInput>,
}

#[derive(Arbitrary, Debug)]
struct EmbeddingInput {
    entity_id: u32,
    values: Vec<f32>,
}

fuzz_target!(|input: SlabInput| {
    let dimension = input.dimension.max(1) as usize;
    if dimension > 1024 {
        return; // Avoid extremely large dimensions
    }

    let slab = EmbeddingSlab::new(dimension, 10);

    // Add embeddings (limit to prevent OOM)
    for (i, emb_input) in input.embeddings.iter().take(100).enumerate() {
        let mut values: Vec<f32> = emb_input
            .values
            .iter()
            .take(dimension)
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(dimension)
            .collect();

        // Filter out NaN/Inf values
        for v in &mut values {
            if v.is_nan() || v.is_infinite() {
                *v = 0.0;
            }
        }

        let entity = EntityId::new(i as u64);
        let _ = slab.set(entity, &values);
    }

    // Snapshot and serialize
    let snapshot = slab.snapshot();
    let serialized = match bincode::serialize(&snapshot) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Deserialize
    let deserialized = match bincode::deserialize(&serialized) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Restore
    let restored = EmbeddingSlab::restore(deserialized);

    // Verify count matches
    assert_eq!(slab.len(), restored.len());

    // Verify each embedding can be retrieved
    for i in 0..slab.len() {
        let entity = EntityId::new(i as u64);
        if let (Some(orig), Some(rest)) = (slab.get(entity), restored.get(entity)) {
            assert_eq!(orig.len(), rest.len());
            for (a, b) in orig.iter().zip(rest.iter()) {
                if a.is_nan() || b.is_nan() {
                    continue;
                }
                assert!(
                    (a - b).abs() < 1e-5,
                    "Mismatch at entity {}: {} != {}",
                    i,
                    a,
                    b
                );
            }
        }
    }
});
