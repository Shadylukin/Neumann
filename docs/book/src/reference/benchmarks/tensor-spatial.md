# tensor_spatial Benchmarks

R-tree spatial index benchmarks covering insert, bulk load, query,
removal, split strategy comparison, and mixed workloads in 2D and 3D.

All benchmarks use seeded RNG (`StdRng::seed_from_u64(42)`) for
reproducibility. Set `BENCH_HEAVY=1` for 1M-scale tests.

## Running

```bash
# Standard run
cargo bench -p tensor_spatial --bench spatial_bench

# Quick validation
cargo bench -p tensor_spatial --bench spatial_bench \
    -- --warm-up-time 1 --measurement-time 2 --sample-size 10

# Heavy (1M scale)
BENCH_HEAVY=1 cargo bench -p tensor_spatial --bench spatial_bench
```

<!-- BENCH:START -->
## Insert (incremental, RStar default)

| Size | Time |
| --- | --- |
| 1,000 | 570 us |
| 10,000 | 7.3 ms |
| 100,000 | 88 ms |

## Insert (Linear at 10K)

| Size | Time |
| --- | --- |
| 10,000 | 4.4 ms |

## Bulk Load (STR)

| Size | Time |
| --- | --- |
| 1,000 | 72 us |
| 10,000 | 870 us |
| 100,000 | 10 ms |

## Region Query (5% area)

| Size | Time |
| --- | --- |
| 10,000 | 4.2 us |
| 100,000 | 6.9 us |

## Nearest Query (k=10)

| Size | Time |
| --- | --- |
| 10,000 | 580 ns |
| 100,000 | 610 ns |

## Radius Query (r=50)

| Size | Time |
| --- | --- |
| 10,000 | 2.1 us |
| 100,000 | 6.9 us |

## Remove (100 entries from 10K)

| Operation | Time |
| --- | --- |
| remove 100 | 37 us |

## Query Strategy Comparison (100K, pre-built indexes)

| Strategy | Region 5% | Nearest k=10 | Radius r=50 |
| --- | --- | --- | --- |
| Linear | 7.6 us | 630 ns | 7.2 us |
| RStar | 6.7 us | 610 ns | 6.9 us |

## Nearest k Sweep (RStar, 100K)

| k | Time |
| --- | --- |
| 1 | 230 ns |
| 10 | 600 ns |
| 100 | 4.2 us |

## 3D Operations

| Operation | Size | Time |
| --- | --- | --- |
| insert | 10,000 | 7.3 ms |
| insert | 100,000 | 94 ms |
| region | 100,000 | 3.8 us |
| nearest k=10 | 100,000 | 1.4 us |
| radius r=50 | 100,000 | 950 ns |

## Mixed Workload (10K)

| Operation | Time |
| --- | --- |
| insert + delete 50% + reinsert + query | 11 ms |
<!-- BENCH:END -->

## Analysis

- **Insert scales sub-linearly**: 100K is ~12x the cost of 10K
  (not 10x) due to deeper tree traversal at larger sizes.
- **Bulk load is 8-9x faster** than incremental insert at 100K,
  since STR avoids per-entry split overhead.
- **RStar queries are 10-15% faster** than Linear at 100K due to
  less bounding-box overlap. The insert cost trade-off is worthwhile
  for read-heavy workloads.
- **Nearest-neighbor scales well**: k=10 at 100K takes only 610 ns
  thanks to priority-queue pruning with `min_dist_sq`.
- **3D nearest is ~2x slower** than 2D at the same size, reflecting
  the extra dimension in distance computation and larger node
  bounding boxes.
- **Branchless `min_dist_sq`** compiles to `fmaxnm` + `fmadd` on
  aarch64, eliminating branch misprediction in the hot query path.

## Benchmark Groups

| Group | What It Measures |
| --- | --- |
| `insert` | Incremental insert (RStar default, Linear at 10K) |
| `bulk_load` | STR bulk load |
| `region_query` | Region intersection query (5% area window) |
| `nearest_query` | k-NN query (k=10) |
| `radius_query` | Radius query (r=50) |
| `remove` | Removal with condense_tree |
| `split_strategies` | Combined insert+query (historical baseline) |
| `query_strategies` | Per-operation Linear vs RStar on pre-built indexes |
| `nearest_k_sizes` | k=1, k=10, k=100 sweep |
| `3d` | 3D insert, region, nearest, radius |
| `mixed_workload` | Insert-delete-reinsert-query cycle |
