# relational_engine Benchmarks

The relational engine provides SQL-like operations on top of tensor_store, with optional hash indexes for accelerated equality lookups and tensor-native condition evaluation.

## Row Insertion

| Count | Time | Throughput |
|-------|------|------------|
| 100 | 462us | 216K rows/s |
| 1,000 | 3.1ms | 319K rows/s |
| 5,000 | 15.6ms | 320K rows/s |

## Batch Insertion

| Count | Time | Throughput |
|-------|------|------------|
| 100 | 282us | 355K rows/s |
| 1,000 | 1.45ms | 688K rows/s |
| 5,000 | 7.26ms | 688K rows/s |

## Select Full Scan

| Rows | Time | Throughput |
|------|------|------------|
| 100 | 119us | 841K rows/s |
| 1,000 | 995us | 1.01M rows/s |
| 5,000 | 5.27ms | 949K rows/s |

## Select with Index vs Without (5,000 rows)

| Query Type | With Index | Without Index | Speedup |
|------------|------------|---------------|---------|
| Equality (2% match) | 105us | 4.23ms | **40x** |
| By _id (single row) | 2.93us | 4.70ms | **1,604x** |

## Select Filtered - No Index (5,000 rows)

| Filter Type | Time |
|-------------|------|
| Range (20% match) | 4.16ms |
| Compound AND | 4.42ms |

## Index Creation (parallel)

| Rows | Time |
|------|------|
| 100 | 554us |
| 1,000 | 2.75ms |
| 5,000 | 12.3ms |

## Update/Delete (1,000 rows, 10% affected)

| Operation | Time |
|-----------|------|
| Update | 1.74ms |
| Delete | 2.14ms |

## Join Performance (hash join)

| Tables | Result Rows | Time |
|--------|-------------|------|
| 50 users x 500 posts | 500 | 1.78ms |
| 100 users x 1000 posts | 1,000 | 1.50ms |
| 100 users x 5000 posts | 5,000 | 32.2ms |

## JOIN Types (10K x 10K rows)

| JOIN Type | Time | Throughput |
|-----------|------|------------|
| INNER JOIN | 45ms | 2.2M rows/s |
| LEFT JOIN | 52ms | 1.9M rows/s |
| RIGHT JOIN | 51ms | 1.9M rows/s |
| FULL JOIN | 68ms | 1.5M rows/s |
| CROSS JOIN | 180ms | 555K rows/s |
| NATURAL JOIN | 48ms | 2.1M rows/s |

## Aggregate Functions (1M rows, SIMD-accelerated)

| Function | Time | Notes |
|----------|------|-------|
| COUNT(*) | 2.1ms | O(1) via counter |
| SUM(col) | 8.5ms | SIMD i64x4 |
| AVG(col) | 8.7ms | SIMD i64x4 |
| MIN(col) | 12ms | Full scan |
| MAX(col) | 12ms | Full scan |

## GROUP BY Performance (100K rows)

| Groups | Time | Notes |
|--------|------|-------|
| 10 | 15ms | Parallel aggregation |
| 100 | 18ms | Hash-based grouping |
| 1,000 | 25ms | Low per-group overhead |
| 10,000 | 45ms | High cardinality |

## Row Count

| Rows | Time |
|------|------|
| 100 | 49us |
| 1,000 | 462us |
| 5,000 | 2.95ms |

## Analysis

- **Index acceleration**: Hash indexes provide O(1) lookup for equality conditions
  - 40x speedup for equality queries matching 2% of rows
  - 1,604x speedup for single-row _id lookups
- **Full scan cost**: Without index, O(n) for all queries (parallelized for >1000 rows)
- **Batch insert**: 2x faster than individual inserts (688K/s vs 320K/s)
- **Tensor-native evaluation**: `evaluate_tensor()` evaluates conditions directly on TensorData, avoiding Row conversion for non-matching rows
- **Parallel operations**: update/delete/create_index use rayon for condition evaluation
- **Index maintenance**: Small overhead on insert/update/delete to maintain indexes
- **Join complexity**: O(n+m) hash join for INNER/LEFT/RIGHT/NATURAL; O(n*m) for CROSS
- **Aggregate functions**: SUM/AVG use SIMD i64x4 vectors for 4x throughput improvement
- **GROUP BY**: Hash-based grouping with parallel per-group aggregation

## Competitor Comparison

| Operation | Neumann | SQLite | DuckDB | Notes |
|-----------|---------|--------|--------|-------|
| Point lookup (indexed) | 2.9us | ~3us | ~30us | B-tree optimized |
| Full scan (5K rows) | 5.3ms | ~15ms | ~2ms | DuckDB columnar wins |
| Aggregation (1M rows) | 8.5ms | ~200ms | ~12ms | SIMD-accelerated |
| Hash join (10Kx10K) | 45ms | ~500ms | ~35ms | Parallel execution |
| Insert (single row) | 3.1us | ~2us | ~5us | SQLite B-tree optimal |
| Batch insert (1K rows) | 1.5ms | ~8ms | ~3ms | Neumann batch-optimized |

## Design Trade-offs

- **vs SQLite**: Neumann trades SQLite's proven stability for tensor-native storage and SIMD acceleration. SQLite wins on point lookups; Neumann wins on analytics.
- **vs DuckDB**: Similar columnar design. DuckDB has more mature query optimizer; Neumann has tighter tensor integration and lower memory footprint.
- **Unique to Neumann**: Unified tensor storage enables cross-engine queries (relational + graph + vector) without data movement.
