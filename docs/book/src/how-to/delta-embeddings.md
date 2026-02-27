# How to Use Delta-Encoded Embeddings

This guide shows how to compress clustered embeddings using delta encoding with
the `ArchetypeRegistry`.

**See also**: [Delta Vectors Explanation](../explanation/delta-vectors.md) |
[HNSW Algorithm](../explanation/hnsw-algorithm.md) |
[API Reference](../reference/api/tensor-store.md)

---

## When to Use

Delta encoding is effective when your embeddings cluster around common patterns:

- Document embeddings from the same corpus or domain
- Product embeddings within categories
- User embeddings with shared behavioral patterns

If embeddings are uniformly distributed (no clusters), delta encoding provides
little compression and may slow down distance computation.

## 1. Create an Archetype Registry

```rust
let mut registry = ArchetypeRegistry::new(16);  // max 16 archetypes
```

The maximum archetype count should match the expected number of clusters. Start
with a small number (5-10) and increase if coverage analysis shows poor fit.

## 2. Discover Archetypes from Existing Data

Run k-means clustering over a representative sample of embeddings:

```rust
use tensor_store::{KMeansConfig, KMeansInit};

let config = KMeansConfig {
    max_iterations: 100,
    convergence_threshold: 1e-4,
    seed: 42,
    init_method: KMeansInit::KMeansPlusPlus,  // Better spread, slightly slower
};

// Discover 5 archetypes from the embedding collection
registry.discover_archetypes(&embeddings, 5, config);
```

For large collections, pass a random sample (e.g., 10,000 vectors) rather than
the full dataset. The archetypes only need to be representative, not exact.

## 3. Encode Vectors as Deltas

```rust
// Single vector
let delta = registry.encode(&vector, 0.01)?;  // threshold = 0.01

// Batch encoding
let results = registry.encode_batch(&embeddings, 0.01);
for (delta, compression_ratio) in results {
    println!("Archetype {}, compression: {:.2}x",
             delta.archetype_id(), compression_ratio);
}
```

The `threshold` parameter (0.01 in this example) controls precision:

| Threshold | Effect |
| --- | --- |
| 0.001 | High precision, more delta entries, less compression |
| 0.01 | Good balance for most use cases |
| 0.05 | Aggressive compression, some precision loss |

## 4. Analyze Coverage

Check how well the archetypes fit your data:

```rust
let stats = registry.analyze_coverage(&vectors, 0.01);
println!("Average similarity to nearest archetype: {:.4}", stats.avg_similarity);
println!("Average compression ratio: {:.2}x", stats.avg_compression_ratio);
println!("Per-archetype usage: {:?}", stats.archetype_usage);
```

Interpret the results:

- **avg_similarity > 0.95**: excellent fit, expect 5-10x compression
- **avg_similarity 0.85-0.95**: good fit, expect 3-5x compression
- **avg_similarity < 0.85**: poor fit, consider more archetypes or dense storage

## 5. Persist the Registry

Save archetypes to TensorStore so they survive restarts:

```rust
// Save
registry.save_to_store(&store)?;

// Load
let registry = ArchetypeRegistry::load_from_store(&store, 16)?;
```

## 6. Use Optimized Distance Computation

When searching against delta-encoded vectors, precompute the archetype-query dot
product to speed up per-vector distance:

```rust
// Precompute once per archetype per search
let archetype_dot_query = dot_product(&archetype, &query);

// O(nnz_delta) per vector instead of O(dimension)
let similarity = delta.dot_dense_with_precomputed(&query, archetype_dot_query);
```

For two delta vectors sharing the same archetype:

```rust
let similarity = a.dot_same_archetype(&b, &archetype, archetype_magnitude_sq);
```

## Common Issues

**"DeltaRequiresRegistry" error**: You are using delta storage without a
registered archetype. Ensure the registry is populated before encoding.

**"ArchetypeNotFound" error**: The delta references an archetype ID that is not
in the current registry. This can happen after loading a snapshot with a
different registry. Rebuild or reload the registry.

**Poor compression**: If `avg_compression_ratio` is below 2x, the data does not
cluster well. Use dense or sparse storage instead.
