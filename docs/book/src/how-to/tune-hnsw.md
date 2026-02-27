# How to Tune HNSW Parameters

This guide covers practical parameter tuning for the HNSW approximate nearest
neighbor index in `tensor_store`.

**See also**: [HNSW Algorithm Explanation](../explanation/hnsw-algorithm.md) |
[API Reference](../reference/api/tensor-store.md)

---

## 1. Start with a Preset

Use one of the built-in presets as a starting point:

```rust
// High recall (slower, more accurate) -- use for quality-critical search
HNSWConfig::high_recall()
// m=32, m0=64, ef_construction=400, ef_search=200

// High speed (faster, lower recall) -- use for latency-critical search
HNSWConfig::high_speed()
// m=8, m0=16, ef_construction=100, ef_search=20

// Default -- balanced
HNSWConfig::default()
// m=16, m0=32, ef_construction=200, ef_search=50
```

## 2. Understand the Parameters

| Parameter | Effect of Increasing | Effect of Decreasing |
| --- | --- | --- |
| `m` | Higher recall, more memory, slower insert | Lower recall, less memory, faster insert |
| `ef_construction` | Better graph quality, slower build | Faster build, slightly lower recall |
| `ef_search` | Higher recall, slower search | Faster search, lower recall |
| `max_nodes` | Higher capacity limit | Lower memory ceiling |

**m** (connections per layer) is the most impactful parameter. It determines
both memory usage and graph connectivity. Values below 8 produce disconnected
graphs; values above 48 waste memory with diminishing returns.

**ef_construction** controls build quality. Higher values explore more
candidates during insertion, producing better neighbor connections. This is a
one-time cost; increase it when build time is not a constraint.

**ef_search** is the runtime knob. It controls how many candidates are
considered during each search query. This is the easiest parameter to tune
dynamically based on required recall.

## 3. Custom Configuration

```rust
HNSWConfig {
    m: 24,
    m0: 48,
    ef_construction: 300,
    ef_search: 100,
    ..Default::default()
}
```

Keep `m0 = 2 * m` as a rule of thumb. Layer 0 benefits from denser connections
because it is where the final (most precise) search happens.

## 4. Insert Vectors

```rust
let index = HNSWIndex::with_config(config);

// Dense vectors
index.insert(vec![0.1, 0.2, 0.3]);

// Sparse vectors
index.insert_sparse(sparse_vec);

// Auto-select dense/sparse based on sparsity
index.insert_auto(mixed_vec);

// With capacity checking (recommended for production)
match index.try_insert(vec) {
    Ok(id) => println!("Inserted as node {}", id),
    Err(EmbeddingStorageError::CapacityExceeded { limit, current }) => {
        println!("Index full: {} / {}", current, limit);
    }
}
```

## 5. Search with Custom ef

Override `ef_search` per query for dynamic recall/latency trade-offs:

```rust
// Use configured ef_search (from HNSWConfig)
let results = index.search(&query, 10);

// Override ef_search for this query only
let results = index.search_with_ef(&query, 10, 100);

for (id, similarity) in results {
    println!("Node {}: {:.4}", id, similarity);
}
```

Higher `ef_search` values find more accurate results at the cost of higher
latency. For a 100k-vector index:

| ef_search | Recall@10 | Latency |
| --- | --- | --- |
| 20 | ~85% | ~0.5ms |
| 50 | ~95% | ~1.2ms |
| 100 | ~98% | ~2.5ms |
| 200 | ~99.5% | ~5ms |

(Approximate values; actual results depend on data distribution and dimension.)

## 6. Choose a Storage Type

Pick the storage type based on your data characteristics:

| Data Pattern | Storage | Config |
| --- | --- | --- |
| Dense embeddings (384-dim, few zeros) | Dense | Default |
| Sparse features (>50% zeros) | Sparse | `sparsity_threshold: 0.5` |
| Clustered embeddings (same domain) | Delta | Use ArchetypeRegistry |
| High-dimensional (768+, memory constrained) | TensorTrain | Requires tensor_compress |

## 7. Monitor Performance

Use HNSW access stats to diagnose issues:

```rust
let stats = hnsw.access_stats()?;
println!("Layer 0 ratio: {:.2}", stats.layer0_ratio());
println!("Avg distances per search: {:.0}", stats.avg_distances_per_search);
println!("Searches/sec: {:.0}", stats.searches_per_second());
```

- **High layer0_ratio (>0.9)**: upper layers are not helping. Consider
  increasing `m` for better skip connections.
- **High avg_distances_per_search**: try reducing `ef_search` if recall is
  acceptable, or check if vectors are poorly distributed.
