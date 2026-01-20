# HNSW Stress Tests

Stress tests for the Hierarchical Navigable Small World (HNSW) index, targeting
1M vector scale.

## Test Suite

| Test | Scale | Description |
| --- | --- | --- |
| `stress_hnsw_1m_vectors` | 1M 128d vectors | Build 1M vector index |
| `stress_hnsw_100k_concurrent_build` | 100K vectors, 16 threads | Concurrent index construction |
| `stress_hnsw_search_during_insert` | 50K vectors, 4+4 threads | Concurrent search during insert |
| `stress_hnsw_recall_under_load` | 10K vectors | Verify recall@10 under load |

## Results

| Test | Key Metric | Result |
| --- | --- | --- |
| 1M vectors | Insert throughput | **3,372 vectors/sec** |
| 1M vectors | Search latency (p50) | **0.11ms** |
| 100K concurrent | Insert throughput | **1,155 vectors/sec** |
| Recall@10 | Average recall | **99.8%** (min 90%) |

## Running

```bash
# Run all HNSW stress tests
cargo test --release -p stress_tests --test hnsw_stress -- --ignored --nocapture

# Run specific test
cargo test --release -p stress_tests stress_hnsw_1m_vectors -- --ignored --nocapture
```

## 1M Vector Index Build

Tests building an HNSW index with 1 million 128-dimensional vectors.

**What it validates:**

- Memory efficiency at scale
- Index build time scalability
- Search accuracy after large insertions

**Expected behavior:**

- Linear memory growth with vector count
- Sub-linear search time (O(log n))
- Recall@10 > 95%

## Concurrent Index Build

Tests building an HNSW index with 16 concurrent writer threads.

**What it validates:**

- Thread-safety of HNSW insert operations
- Performance under contention
- Correctness with concurrent modifications

**Expected behavior:**

- All inserted vectors are findable
- No panics or data races
- Throughput scales with thread count (with diminishing returns)

## Search During Insert

Tests searching the index while new vectors are being inserted concurrently.

**What it validates:**

- Read/write concurrency safety
- Search accuracy with ongoing modifications
- Latency stability under load

**Expected behavior:**

- Searches return valid results
- No stale or corrupted results
- Latency remains bounded

## Recall Under Load

Tests search recall accuracy under sustained concurrent load.

**What it validates:**

- HNSW recall guarantees under stress
- Accuracy with high query volume
- Configuration impact on recall

**Expected behavior:**

- Average recall@10 > 95%
- Minimum recall@10 > 90%
- High_recall config > default config recall

## Performance Tuning

### HNSW Configuration Impact

| Config | Insert Speed | Search Speed | Recall | Memory |
| --- | --- | --- | --- | --- |
| high_speed | Fastest | Fastest | Lower | Lower |
| default | Medium | Medium | Good | Medium |
| high_recall | Slowest | Slowest | Highest | Higher |

### Scaling Recommendations

| Scale | Recommendation |
| --- | --- |
| < 100K | Use default config |
| 100K - 1M | Consider high_speed if latency-critical |
| > 1M | Shard across multiple indexes |
