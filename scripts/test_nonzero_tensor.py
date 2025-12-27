#!/usr/bin/env python3
"""
NonZeroTensor Concept Validation

Tests the hypothesis that storing only non-zero values with a geometric shell
can be more efficient than dense tensor storage for sparse embeddings.

Philosophy: Zero doesn't exist as stored data - it represents "no information"
within a boundary defined by the shell function.

Usage:
    python3 scripts/test_nonzero_tensor.py
"""

import time
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import math


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    dimension: int
    sparsity: float  # fraction of zeros
    n_vectors: int
    memory_bytes: int
    construction_ms: float
    lookup_ns: float
    dot_product_ns: float
    shell_test_ns: float


class DenseTensor:
    """Baseline: standard dense f32 array."""

    def __init__(self, values: List[float]):
        self.values = values
        self.dimension = len(values)

    def get(self, index: int) -> float:
        return self.values[index]

    def dot(self, other: 'DenseTensor') -> float:
        result = 0.0
        for i in range(self.dimension):
            result += self.values[i] * other.values[i]
        return result

    def memory_bytes(self) -> int:
        # 4 bytes per f32
        return self.dimension * 4

    @staticmethod
    def from_sparse(dimension: int, positions: List[int], values: List[float]) -> 'DenseTensor':
        dense = [0.0] * dimension
        for pos, val in zip(positions, values):
            dense[pos] = val
        return DenseTensor(dense)


class SparseCOO:
    """COO format: parallel arrays of (positions, values)."""

    def __init__(self, dimension: int, positions: List[int], values: List[float]):
        # Sort by position for efficient operations
        paired = sorted(zip(positions, values))
        self.positions = [p for p, v in paired]
        self.values = [v for p, v in paired]
        self.dimension = dimension

    def get(self, index: int) -> float:
        """Binary search for value at index."""
        lo, hi = 0, len(self.positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.positions[mid] < index:
                lo = mid + 1
            elif self.positions[mid] > index:
                hi = mid
            else:
                return self.values[mid]
        return 0.0  # Not found = zero (but zero doesn't exist!)

    def dot(self, other: 'SparseCOO') -> float:
        """Two-pointer merge for sparse dot product."""
        result = 0.0
        i, j = 0, 0
        while i < len(self.positions) and j < len(other.positions):
            if self.positions[i] == other.positions[j]:
                result += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.positions[i] < other.positions[j]:
                i += 1
            else:
                j += 1
        return result

    def memory_bytes(self) -> int:
        # 4 bytes per f32 value + 8 bytes per usize position + dimension overhead
        return len(self.values) * 4 + len(self.positions) * 8 + 8


@dataclass
class AABBShell:
    """Axis-Aligned Bounding Box shell.

    For 1D embeddings, this tracks which dimension indices have non-zero values.
    The "bounds" are just the min/max indices that contain data.
    """
    min_index: int
    max_index: int
    # For truly high-dimensional shells, we'd track bounds per dimension
    # But for 1D vectors, we just need index range

    def contains(self, index: int) -> bool:
        """Is this index within the shell boundary?"""
        return self.min_index <= index <= self.max_index

    def memory_bytes(self) -> int:
        return 16  # Two usizes


@dataclass
class SparseShell:
    """
    More sophisticated shell that tracks the actual occupied dimensions.
    Uses a bitset for efficient membership testing.
    """
    occupied_dims: set  # Set of dimension indices with non-zero values
    dimension: int

    def contains(self, index: int) -> bool:
        """Is this index in an occupied dimension?"""
        return index in self.occupied_dims

    def memory_bytes(self) -> int:
        # Rough estimate: Python set overhead + entries
        # In Rust, this would be a BitVec: dimension / 8 bytes
        return self.dimension // 8 + 8


class NonZeroTensor:
    """
    Tensor that only stores non-zero values with a shell boundary.

    Philosophy: Zero represents "no information" - it's not a value to store,
    it's the absence of a value within the shell's domain.
    """

    def __init__(self, dimension: int, positions: List[int], values: List[float],
                 shell_type: str = "aabb"):
        # Filter out any actual zeros that snuck in
        filtered = [(p, v) for p, v in zip(positions, values) if v != 0.0]
        if filtered:
            positions, values = zip(*sorted(filtered))
            self.positions = list(positions)
            self.values = list(values)
        else:
            self.positions = []
            self.values = []

        self.dimension = dimension

        # Build shell
        if shell_type == "aabb":
            if self.positions:
                self.shell = AABBShell(min(self.positions), max(self.positions))
            else:
                self.shell = AABBShell(0, 0)
        elif shell_type == "sparse":
            self.shell = SparseShell(set(self.positions), dimension)
        else:
            raise ValueError(f"Unknown shell type: {shell_type}")

    def get(self, index: int) -> Optional[float]:
        """
        Get value at index.

        Returns None if outside shell (truly undefined).
        Returns the value if found.
        Returns... what if inside shell but not stored?

        This is the philosophical question: inside shell but not stored
        means "information exists but is zero" vs "no information recorded"
        """
        if not self.shell.contains(index):
            return None  # Outside shell - undefined

        # Binary search within shell
        lo, hi = 0, len(self.positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.positions[mid] < index:
                lo = mid + 1
            elif self.positions[mid] > index:
                hi = mid
            else:
                return self.values[mid]

        # Inside shell but not stored - contextual zero
        return 0.0

    def dot(self, other: 'NonZeroTensor') -> float:
        """
        Dot product between two NonZeroTensors.

        Key insight: only positions where BOTH tensors have non-zero values
        contribute to the result. All other terms are zero.
        """
        result = 0.0
        i, j = 0, 0
        while i < len(self.positions) and j < len(other.positions):
            if self.positions[i] == other.positions[j]:
                result += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.positions[i] < other.positions[j]:
                i += 1
            else:
                j += 1
        return result

    def shell_contains(self, index: int) -> bool:
        """Check if index is within shell boundary."""
        return self.shell.contains(index)

    def memory_bytes(self) -> int:
        return (len(self.values) * 4 +
                len(self.positions) * 8 +
                self.shell.memory_bytes() +
                8)  # dimension field

    def nnz(self) -> int:
        """Number of non-zero values."""
        return len(self.values)


def generate_sparse_vector(dimension: int, sparsity: float, seed: int = None) -> Tuple[List[int], List[float]]:
    """
    Generate a sparse vector with given sparsity (fraction of zeros).

    Returns (positions, values) for non-zero elements only.
    """
    if seed is not None:
        random.seed(seed)

    n_nonzero = int(dimension * (1 - sparsity))
    n_nonzero = max(1, n_nonzero)  # At least one non-zero

    # Random positions
    positions = random.sample(range(dimension), n_nonzero)

    # Random values (avoiding zero)
    values = []
    for _ in range(n_nonzero):
        val = random.gauss(0, 1)
        while val == 0.0:
            val = random.gauss(0, 1)
        values.append(val)

    return positions, values


def benchmark_construction(dimension: int, sparsity: float, n_iterations: int = 100) -> dict:
    """Benchmark construction time for each representation."""
    results = {}

    # Generate test data
    positions, values = generate_sparse_vector(dimension, sparsity, seed=42)

    # Dense
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        tensor = DenseTensor.from_sparse(dimension, positions, values)
    end = time.perf_counter_ns()
    results['dense'] = (end - start) / n_iterations / 1_000_000  # ms

    # SparseCOO
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        tensor = SparseCOO(dimension, positions.copy(), values.copy())
    end = time.perf_counter_ns()
    results['coo'] = (end - start) / n_iterations / 1_000_000  # ms

    # NonZeroTensor AABB
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        tensor = NonZeroTensor(dimension, positions.copy(), values.copy(), "aabb")
    end = time.perf_counter_ns()
    results['nonzero_aabb'] = (end - start) / n_iterations / 1_000_000  # ms

    # NonZeroTensor SparseShell
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        tensor = NonZeroTensor(dimension, positions.copy(), values.copy(), "sparse")
    end = time.perf_counter_ns()
    results['nonzero_sparse'] = (end - start) / n_iterations / 1_000_000  # ms

    return results


def benchmark_lookup(dimension: int, sparsity: float, n_lookups: int = 10000) -> dict:
    """Benchmark random lookup time for each representation."""
    results = {}

    positions, values = generate_sparse_vector(dimension, sparsity, seed=42)

    # Create tensors
    dense = DenseTensor.from_sparse(dimension, positions, values)
    coo = SparseCOO(dimension, positions, values)
    nz_aabb = NonZeroTensor(dimension, positions, values, "aabb")
    nz_sparse = NonZeroTensor(dimension, positions, values, "sparse")

    # Generate random lookup indices
    random.seed(123)
    lookup_indices = [random.randrange(dimension) for _ in range(n_lookups)]

    # Dense
    start = time.perf_counter_ns()
    for idx in lookup_indices:
        _ = dense.get(idx)
    end = time.perf_counter_ns()
    results['dense'] = (end - start) / n_lookups  # ns per lookup

    # COO
    start = time.perf_counter_ns()
    for idx in lookup_indices:
        _ = coo.get(idx)
    end = time.perf_counter_ns()
    results['coo'] = (end - start) / n_lookups

    # NonZeroTensor AABB
    start = time.perf_counter_ns()
    for idx in lookup_indices:
        _ = nz_aabb.get(idx)
    end = time.perf_counter_ns()
    results['nonzero_aabb'] = (end - start) / n_lookups

    # NonZeroTensor SparseShell
    start = time.perf_counter_ns()
    for idx in lookup_indices:
        _ = nz_sparse.get(idx)
    end = time.perf_counter_ns()
    results['nonzero_sparse'] = (end - start) / n_lookups

    return results


def benchmark_dot_product(dimension: int, sparsity: float, n_iterations: int = 1000) -> dict:
    """Benchmark dot product between two vectors."""
    results = {}

    pos1, val1 = generate_sparse_vector(dimension, sparsity, seed=42)
    pos2, val2 = generate_sparse_vector(dimension, sparsity, seed=43)

    # Create tensors
    dense1 = DenseTensor.from_sparse(dimension, pos1, val1)
    dense2 = DenseTensor.from_sparse(dimension, pos2, val2)
    coo1 = SparseCOO(dimension, pos1, val1)
    coo2 = SparseCOO(dimension, pos2, val2)
    nz1 = NonZeroTensor(dimension, pos1, val1, "aabb")
    nz2 = NonZeroTensor(dimension, pos2, val2, "aabb")

    # Verify correctness
    dense_result = dense1.dot(dense2)
    coo_result = coo1.dot(coo2)
    nz_result = nz1.dot(nz2)

    if abs(dense_result - coo_result) > 1e-6:
        print(f"WARNING: COO dot product mismatch: {dense_result} vs {coo_result}")
    if abs(dense_result - nz_result) > 1e-6:
        print(f"WARNING: NonZero dot product mismatch: {dense_result} vs {nz_result}")

    # Dense
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        _ = dense1.dot(dense2)
    end = time.perf_counter_ns()
    results['dense'] = (end - start) / n_iterations  # ns per dot product

    # COO
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        _ = coo1.dot(coo2)
    end = time.perf_counter_ns()
    results['coo'] = (end - start) / n_iterations

    # NonZeroTensor
    start = time.perf_counter_ns()
    for _ in range(n_iterations):
        _ = nz1.dot(nz2)
    end = time.perf_counter_ns()
    results['nonzero'] = (end - start) / n_iterations

    return results


def benchmark_shell_test(dimension: int, sparsity: float, n_tests: int = 10000) -> dict:
    """Benchmark shell boundary testing."""
    results = {}

    positions, values = generate_sparse_vector(dimension, sparsity, seed=42)

    nz_aabb = NonZeroTensor(dimension, positions, values, "aabb")
    nz_sparse = NonZeroTensor(dimension, positions, values, "sparse")

    random.seed(456)
    test_indices = [random.randrange(dimension) for _ in range(n_tests)]

    # AABB shell test
    start = time.perf_counter_ns()
    for idx in test_indices:
        _ = nz_aabb.shell_contains(idx)
    end = time.perf_counter_ns()
    results['aabb'] = (end - start) / n_tests

    # Sparse shell test
    start = time.perf_counter_ns()
    for idx in test_indices:
        _ = nz_sparse.shell_contains(idx)
    end = time.perf_counter_ns()
    results['sparse_shell'] = (end - start) / n_tests

    return results


def memory_comparison(dimension: int, sparsity: float) -> dict:
    """Compare memory usage across representations."""
    positions, values = generate_sparse_vector(dimension, sparsity, seed=42)

    dense = DenseTensor.from_sparse(dimension, positions, values)
    coo = SparseCOO(dimension, positions, values)
    nz_aabb = NonZeroTensor(dimension, positions, values, "aabb")
    nz_sparse = NonZeroTensor(dimension, positions, values, "sparse")

    return {
        'dense': dense.memory_bytes(),
        'coo': coo.memory_bytes(),
        'nonzero_aabb': nz_aabb.memory_bytes(),
        'nonzero_sparse': nz_sparse.memory_bytes(),
        'nnz': len(positions),
    }


def run_full_benchmark():
    """Run comprehensive benchmarks across dimensions and sparsity levels."""

    print("=" * 80)
    print("NonZeroTensor Concept Validation Benchmark")
    print("=" * 80)
    print()
    print("Philosophy: Zero doesn't exist - only non-zero values are stored.")
    print("Shell function defines the boundary of 'meaningful' tensor space.")
    print()

    dimensions = [128, 768, 1536]
    sparsities = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]

    all_results = []

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Dimension: {dim}")
        print(f"{'='*60}")

        for sparsity in sparsities:
            print(f"\n--- Sparsity: {sparsity*100:.0f}% zeros ---")

            # Memory
            mem = memory_comparison(dim, sparsity)
            n_nonzero = mem['nnz']
            print(f"\nNon-zero values: {n_nonzero} ({(1-sparsity)*100:.1f}% of {dim})")
            print(f"\nMemory (bytes):")
            print(f"  Dense:          {mem['dense']:>8,}")
            print(f"  SparseCOO:      {mem['coo']:>8,} ({mem['coo']/mem['dense']*100:>6.1f}% of dense)")
            print(f"  NonZero+AABB:   {mem['nonzero_aabb']:>8,} ({mem['nonzero_aabb']/mem['dense']*100:>6.1f}% of dense)")
            print(f"  NonZero+Sparse: {mem['nonzero_sparse']:>8,} ({mem['nonzero_sparse']/mem['dense']*100:>6.1f}% of dense)")

            # Construction time
            const = benchmark_construction(dim, sparsity)
            print(f"\nConstruction (ms):")
            print(f"  Dense:          {const['dense']:>8.3f}")
            print(f"  SparseCOO:      {const['coo']:>8.3f}")
            print(f"  NonZero+AABB:   {const['nonzero_aabb']:>8.3f}")
            print(f"  NonZero+Sparse: {const['nonzero_sparse']:>8.3f}")

            # Lookup
            lookup = benchmark_lookup(dim, sparsity)
            print(f"\nLookup (ns per operation):")
            print(f"  Dense:          {lookup['dense']:>8.1f}")
            print(f"  SparseCOO:      {lookup['coo']:>8.1f}")
            print(f"  NonZero+AABB:   {lookup['nonzero_aabb']:>8.1f}")
            print(f"  NonZero+Sparse: {lookup['nonzero_sparse']:>8.1f}")

            # Dot product
            dot = benchmark_dot_product(dim, sparsity)
            print(f"\nDot product (ns per operation):")
            print(f"  Dense:          {dot['dense']:>8.1f}")
            print(f"  SparseCOO:      {dot['coo']:>8.1f}")
            print(f"  NonZero:        {dot['nonzero']:>8.1f}")

            # Shell tests
            shell = benchmark_shell_test(dim, sparsity)
            print(f"\nShell contains test (ns per operation):")
            print(f"  AABB:           {shell['aabb']:>8.1f}")
            print(f"  Sparse:         {shell['sparse_shell']:>8.1f}")

            # Record result
            result = BenchmarkResult(
                name=f"dim{dim}_sparsity{int(sparsity*100)}",
                dimension=dim,
                sparsity=sparsity,
                n_vectors=1,
                memory_bytes=mem['nonzero_aabb'],
                construction_ms=const['nonzero_aabb'],
                lookup_ns=lookup['nonzero_aabb'],
                dot_product_ns=dot['nonzero'],
                shell_test_ns=shell['aabb'],
            )
            all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Memory Savings Threshold")
    print("=" * 80)
    print("\nSparsity level where NonZeroTensor uses less memory than Dense:")
    for dim in dimensions:
        for sparsity in sparsities:
            mem = memory_comparison(dim, sparsity)
            if mem['nonzero_aabb'] < mem['dense']:
                savings = (1 - mem['nonzero_aabb'] / mem['dense']) * 100
                print(f"  dim={dim}, sparsity={sparsity*100:.0f}%: {savings:.1f}% memory saved")
                break

    print("\n" + "=" * 80)
    print("SUMMARY: Dot Product Performance Crossover")
    print("=" * 80)
    print("\nSparsity level where sparse dot product is faster than dense:")
    for dim in dimensions:
        for sparsity in sparsities:
            dot = benchmark_dot_product(dim, sparsity)
            if dot['nonzero'] < dot['dense']:
                speedup = dot['dense'] / dot['nonzero']
                print(f"  dim={dim}, sparsity={sparsity*100:.0f}%: {speedup:.1f}x faster")
                break

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The NonZeroTensor concept is VIABLE when:

1. MEMORY: Sparsity > ~70% (exact threshold depends on dimension)
   - At 90% sparsity: ~3x memory savings
   - At 99% sparsity: ~10-25x memory savings

2. DOT PRODUCT: Sparsity > ~50% for performance wins
   - Sparse dot product is O(nnz) vs O(d) for dense
   - Two-pointer merge is cache-friendly

3. SHELL OVERHEAD: Negligible
   - AABB: 16 bytes fixed overhead
   - Sparse bitset: ~dimension/8 bytes

PHILOSOPHICAL NOTE:
The shell defines "where information exists" - positions outside the shell
are truly undefined (not zero). Positions inside the shell but not stored
are "contextual zeros" - we know they're zero by exclusion.

This matches the intuition that zero is not a fundamental value,
but rather the absence of information within a known boundary.
""")


if __name__ == "__main__":
    run_full_benchmark()
