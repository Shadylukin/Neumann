# Sparse Vectors

Sparse vectors are a memory-efficient representation for high-dimensional data
where most values are zero.

## When to Use Sparse Vectors

| Use Case | Dense | Sparse |
| --- | --- | --- |
| Low dimensions (<100) | Preferred | Overhead |
| High dimensions (>1000) | Memory intensive | Preferred |
| Most values non-zero | Preferred | Overhead |
| <10% values non-zero | Wasteful | Preferred |

## SparseVector Type

```rust
pub struct SparseVector {
    dimension: usize,
    indices: Vec<usize>,
    values: Vec<f32>,
}
```

### Memory Comparison

For a 10,000-dimensional vector with 100 non-zero values:

| Representation | Memory |
| --- | --- |
| Dense `Vec<f32>` | 40,000 bytes |
| Sparse | ~800 bytes |
| Savings | 98% |

## Operations

### Creation

```rust
// From dense
let sparse = SparseVector::from_dense(&[0.0, 0.5, 0.0, 0.3, 0.0]);

// Incremental
let mut sparse = SparseVector::new(1000);
sparse.set(42, 0.5);
sparse.set(100, 0.3);
```

### Arithmetic

```rust
// Subtraction (for deltas)
let delta = new_state.sub(&old_state);

// Weighted average
let blended = SparseVector::weighted_average(&[
    (&vec_a, 0.7),
    (&vec_b, 0.3),
]);

// Orthogonal projection
let residual = vec.project_orthogonal(&basis);
```

### Similarity Metrics

| Metric | Formula | Range |
| --- | --- | --- |
| Cosine | `a.b / (‖a‖ * ‖b‖)` | -1 to 1 |
| Euclidean | `sqrt(sum((a-b)^2))` | 0 to inf |
| Jaccard | `‖A ∩ B‖ / ‖A ∪ B‖` | 0 to 1 |
| Angular | `acos(cosine) / pi` | 0 to 1 |

```rust
let sim = vec_a.cosine_similarity(&vec_b);
let dist = vec_a.euclidean_distance(&vec_b);
let jacc = vec_a.jaccard_index(&vec_b);
```

## HNSW Index

Hierarchical Navigable Small World for approximate nearest neighbor search:

```rust
let mut index = HNSWIndex::new(HNSWConfig::default());

// Insert
index.insert("doc_1", sparse_vec_1);
index.insert("doc_2", sparse_vec_2);

// Search
let results = index.search(&query_vec, 10); // top 10
```

### Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `m` | 16 | Max connections per layer |
| `ef_construction` | 200 | Build-time search width |
| `ef_search` | 50 | Query-time search width |

## Delta Encoding

For tracking state changes:

```rust
// Compute delta between states
let delta = DeltaVector::from_diff(&old_embedding, &new_embedding);

// Apply delta
let new_state = old_state.add(&delta.to_sparse());

// Check if orthogonal (non-conflicting)
if delta_a.is_orthogonal(&delta_b) {
    // Can merge automatically
}
```

## Compression

Sparse vectors compress well:

| Method | Ratio | Speed |
| --- | --- | --- |
| Varint indices | 2-4x | Fast |
| Quantization (int8) | 4x | Fast |
| Binary quantization | 32x | Very fast |
