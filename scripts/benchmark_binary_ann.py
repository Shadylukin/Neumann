#!/usr/bin/env python3
"""
Benchmark: Binary Signature ANN vs Brute Force vs TensorFactor

Tests whether binary quantization + Hamming distance + rerank
can achieve HNSW-competitive speed while maintaining quality.

Run: python scripts/benchmark_binary_ann.py
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

# Silence numpy warnings
np.seterr(divide='ignore', invalid='ignore')


@dataclass
class BenchmarkResult:
    name: str
    recall_at_10: float
    recall_at_100: float
    qps: float  # queries per second
    build_time_ms: float
    memory_mb: float


# =============================================================================
# BRUTE FORCE BASELINE
# =============================================================================

class BruteForce:
    def __init__(self, corpus: np.ndarray):
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus = (corpus / norms).astype(np.float32)
        self.N, self.D = self.corpus.shape

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        sims = self.corpus @ query_norm.astype(np.float32)
        if k >= len(sims):
            top_k = np.argsort(-sims)
        else:
            top_k = np.argpartition(-sims, k)[:k]
            top_k = top_k[np.argsort(-sims[top_k])]
        return top_k, sims[top_k]

    def memory_mb(self) -> float:
        return self.corpus.nbytes / 1e6


# =============================================================================
# BINARY SIGNATURE INDEX (SimHash-style)
# =============================================================================

class BinarySignatureIndex:
    """
    Binary random projection + Hamming distance + rerank.

    This is what we want to implement in Rust for Neumann.
    """

    def __init__(self, corpus: np.ndarray, n_bits: int = 128, seed: int = 42):
        self.N, self.D = corpus.shape
        self.n_bits = n_bits

        # Normalize corpus
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        # Random projection matrix (the "encoder")
        np.random.seed(seed)
        self.projection = (np.random.randn(self.D, n_bits) / np.sqrt(self.D)).astype(np.float32)

        # Compute binary signatures: sign(corpus @ projection)
        projected = self.corpus_normalized @ self.projection
        self.signatures = (projected > 0).astype(np.uint8)

        # Pack bits for efficient Hamming distance (8 bits per byte)
        self.packed_signatures = np.packbits(self.signatures, axis=1)

    def search(self, query: np.ndarray, k: int, n_candidates: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)

        # Project query to binary signature
        query_projected = query_norm @ self.projection
        query_sig = (query_projected > 0).astype(np.uint8)
        query_packed = np.packbits(query_sig)

        # Compute Hamming distances (XOR + popcount)
        xor = np.bitwise_xor(self.packed_signatures, query_packed)
        hamming_dists = np.unpackbits(xor, axis=1).sum(axis=1)

        # Get top candidates by Hamming distance
        n_cand = min(n_candidates, self.N)
        if n_cand >= self.N:
            candidates = np.argsort(hamming_dists)[:n_cand]
        else:
            candidates = np.argpartition(hamming_dists, n_cand)[:n_cand]

        # Rerank with exact cosine similarity
        exact_sims = self.corpus_normalized[candidates] @ query_norm

        if k >= len(exact_sims):
            top_k_local = np.argsort(-exact_sims)
        else:
            top_k_local = np.argpartition(-exact_sims, k)[:k]
            top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]

    def memory_mb(self) -> float:
        return (
            self.corpus_normalized.nbytes +
            self.projection.nbytes +
            self.packed_signatures.nbytes
        ) / 1e6


# =============================================================================
# TENSOR FACTOR INDEX (SVD-based)
# =============================================================================

class TensorFactorIndex:
    """
    SVD-based ANN from Pattern Universe experiments.
    """

    def __init__(self, corpus: np.ndarray, rank: int = 64, seed: int = 42):
        self.N, self.D = corpus.shape
        self.rank = min(rank, min(self.N, self.D) - 1)

        # Normalize
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        # Truncated SVD (randomized for speed)
        np.random.seed(seed)
        try:
            from scipy.sparse.linalg import svds
            # svds returns ascending order, we want descending
            U, S, Vt = svds(self.corpus_normalized, k=self.rank)
            # Reverse to get descending singular values
            U = U[:, ::-1]
            S = S[::-1]
            Vt = Vt[::-1, :]
        except ImportError:
            # Fallback to full SVD (slower)
            U, S, Vt = np.linalg.svd(self.corpus_normalized, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vt = Vt[:self.rank, :]

        # US matrix: documents in weighted latent space
        self.US = (U * S).astype(np.float32)

        # V matrix: for projecting queries (D x rank)
        self.V = Vt.T.astype(np.float32)

    def search(self, query: np.ndarray, k: int, rerank_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)

        # Project query to latent space
        query_latent = query_norm @ self.V

        # Compute latent similarities
        latent_sims = self.US @ query_latent

        # Get top candidates
        n_candidates = min(k * rerank_factor, self.N)
        if n_candidates >= self.N:
            candidates = np.argsort(-latent_sims)[:n_candidates]
        else:
            candidates = np.argpartition(-latent_sims, n_candidates)[:n_candidates]

        # Rerank with exact cosine
        exact_sims = self.corpus_normalized[candidates] @ query_norm

        if k >= len(exact_sims):
            top_k_local = np.argsort(-exact_sims)
        else:
            top_k_local = np.argpartition(-exact_sims, k)[:k]
            top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]

    def memory_mb(self) -> float:
        return (
            self.corpus_normalized.nbytes +
            self.US.nbytes +
            self.V.nbytes
        ) / 1e6


# =============================================================================
# INT8 QUANTIZED BRUTE FORCE
# =============================================================================

class Int8BruteForce:
    """
    Brute force with int8 quantization.
    Tests whether quantization alone provides speedup.
    """

    def __init__(self, corpus: np.ndarray):
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        corpus_norm = corpus / norms

        # Quantize to int8
        self.min_val = corpus_norm.min()
        self.max_val = corpus_norm.max()
        self.scale = (self.max_val - self.min_val) / 255.0

        self.corpus_int8 = ((corpus_norm - self.min_val) / self.scale).astype(np.uint8)
        self.corpus_float = corpus_norm.astype(np.float32)  # Keep for exact rerank
        self.N, self.D = corpus_norm.shape

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Quantize query
        query_int8 = ((query_norm - self.min_val) / self.scale).clip(0, 255).astype(np.uint8)

        # Int8 dot product (numpy will upcast, but still faster due to memory)
        # Note: Real speedup requires SIMD int8 intrinsics
        sims_approx = (self.corpus_int8.astype(np.int32) @ query_int8.astype(np.int32))

        # Get candidates
        n_cand = min(k * 10, self.N)
        if n_cand >= self.N:
            candidates = np.argsort(-sims_approx)[:n_cand]
        else:
            candidates = np.argpartition(-sims_approx, n_cand)[:n_cand]

        # Rerank exact
        exact_sims = self.corpus_float[candidates] @ query_norm.astype(np.float32)
        top_k_local = np.argsort(-exact_sims)[:k]

        return candidates[top_k_local], exact_sims[top_k_local]

    def memory_mb(self) -> float:
        return (self.corpus_int8.nbytes + self.corpus_float.nbytes) / 1e6


# =============================================================================
# BENCHMARK HARNESS
# =============================================================================

def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute recall@k."""
    return len(set(retrieved) & set(ground_truth)) / len(ground_truth)


def generate_data(n_corpus: int, n_queries: int, dim: int = 384, n_clusters: int = 50, seed: int = 42):
    """Generate test data with cluster structure (more realistic than random)."""
    np.random.seed(seed)

    # Create cluster centers
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate corpus around clusters
    corpus = []
    items_per_cluster = n_corpus // n_clusters

    for c in range(n_clusters):
        n_items = items_per_cluster if c < n_clusters - 1 else n_corpus - len(corpus)
        noise = 0.3 * np.random.randn(n_items, dim).astype(np.float32)
        items = centers[c] + noise
        corpus.extend(items)

    corpus = np.array(corpus, dtype=np.float32)

    # Generate queries (mix of in-cluster and random)
    queries = []
    for i in range(n_queries):
        if i % 2 == 0:
            # Query near a cluster center
            c = np.random.randint(n_clusters)
            q = centers[c] + 0.2 * np.random.randn(dim).astype(np.float32)
        else:
            # Random query
            q = np.random.randn(dim).astype(np.float32)
        queries.append(q)

    queries = np.array(queries, dtype=np.float32)

    return corpus, queries


def benchmark_index(name: str, index, queries: np.ndarray, ground_truth: List[np.ndarray],
                    k: int = 10, **search_kwargs) -> BenchmarkResult:
    """Benchmark a single index."""
    n_queries = len(queries)

    # Warmup
    for i in range(min(5, n_queries)):
        index.search(queries[i], k, **search_kwargs)

    # Timed run
    start = time.perf_counter()
    results = []
    for i in range(n_queries):
        indices, _ = index.search(queries[i], k, **search_kwargs)
        results.append(indices)
    elapsed = time.perf_counter() - start

    # Compute recall
    recall_10 = np.mean([compute_recall(r[:10], gt[:10]) for r, gt in zip(results, ground_truth)])
    recall_100 = np.mean([compute_recall(r[:min(100, len(r))], gt[:100]) for r, gt in zip(results, ground_truth)])

    return BenchmarkResult(
        name=name,
        recall_at_10=recall_10,
        recall_at_100=recall_100,
        qps=n_queries / elapsed,
        build_time_ms=0,  # Set separately
        memory_mb=index.memory_mb()
    )


def run_benchmarks(n_corpus: int = 50000, n_queries: int = 100, dim: int = 384):
    """Run full benchmark suite."""
    print(f"\n{'='*70}")
    print(f"BINARY ANN BENCHMARK: {n_corpus:,} vectors, {dim} dims, {n_queries} queries")
    print(f"{'='*70}\n")

    # Generate data
    print("Generating test data...")
    corpus, queries = generate_data(n_corpus, n_queries, dim)
    print(f"  Corpus: {corpus.shape}, Queries: {queries.shape}")
    print(f"  Corpus size: {corpus.nbytes / 1e6:.1f} MB\n")

    results = []

    # 1. Brute Force (ground truth)
    print("Building Brute Force (baseline)...")
    start = time.perf_counter()
    bf = BruteForce(corpus)
    bf_build = (time.perf_counter() - start) * 1000

    print("  Computing ground truth...")
    ground_truth = []
    for q in queries:
        indices, _ = bf.search(q, 100)
        ground_truth.append(indices)

    print("  Benchmarking...")
    bf_result = benchmark_index("Brute Force (f32)", bf, queries, ground_truth, k=100)
    bf_result.build_time_ms = bf_build
    results.append(bf_result)
    print(f"  QPS: {bf_result.qps:.1f}, Memory: {bf_result.memory_mb:.1f} MB\n")

    # 2. Binary Signature Index (various bit counts)
    for n_bits in [64, 128, 256, 512]:
        for n_cand in [50, 100, 200]:
            name = f"Binary-{n_bits}b-cand{n_cand}"
            print(f"Building {name}...")
            start = time.perf_counter()
            binary_idx = BinarySignatureIndex(corpus, n_bits=n_bits)
            build_time = (time.perf_counter() - start) * 1000

            result = benchmark_index(name, binary_idx, queries, ground_truth,
                                     k=100, n_candidates=n_cand)
            result.build_time_ms = build_time
            results.append(result)
            print(f"  R@10: {result.recall_at_10:.3f}, QPS: {result.qps:.1f}, "
                  f"Speedup: {result.qps/bf_result.qps:.2f}x\n")

    # 3. TensorFactor (SVD-based)
    for rank in [32, 64, 128]:
        for rerank in [5, 10, 20]:
            name = f"TensorFactor-r{rank}-rf{rerank}"
            print(f"Building {name}...")
            start = time.perf_counter()
            tf_idx = TensorFactorIndex(corpus, rank=rank)
            build_time = (time.perf_counter() - start) * 1000

            result = benchmark_index(name, tf_idx, queries, ground_truth,
                                     k=100, rerank_factor=rerank)
            result.build_time_ms = build_time
            results.append(result)
            print(f"  R@10: {result.recall_at_10:.3f}, QPS: {result.qps:.1f}, "
                  f"Speedup: {result.qps/bf_result.qps:.2f}x\n")

    # 4. Int8 Quantized
    print("Building Int8 Brute Force...")
    start = time.perf_counter()
    int8_idx = Int8BruteForce(corpus)
    build_time = (time.perf_counter() - start) * 1000

    result = benchmark_index("Int8 Quantized", int8_idx, queries, ground_truth, k=100)
    result.build_time_ms = build_time
    results.append(result)
    print(f"  R@10: {result.recall_at_10:.3f}, QPS: {result.qps:.1f}, "
          f"Speedup: {result.qps/bf_result.qps:.2f}x\n")

    # Summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'R@10':>8} {'R@100':>8} {'QPS':>10} {'Speedup':>10} {'Mem MB':>10}")
    print("-" * 70)

    bf_qps = results[0].qps
    for r in results:
        speedup = r.qps / bf_qps
        print(f"{r.name:<30} {r.recall_at_10:>8.3f} {r.recall_at_100:>8.3f} "
              f"{r.qps:>10.1f} {speedup:>9.2f}x {r.memory_mb:>10.1f}")

    # Find best configs
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # Filter to high-recall results
    good_results = [r for r in results if r.recall_at_10 >= 0.95]
    if good_results:
        best_speed = max(good_results, key=lambda r: r.qps)
        print(f"\nBest speed at R@10 >= 0.95:")
        print(f"  {best_speed.name}: {best_speed.qps:.1f} QPS ({best_speed.qps/bf_qps:.2f}x speedup)")

    # Best quality-speed tradeoff
    pareto = []
    for r in results:
        dominated = False
        for r2 in results:
            if r2.recall_at_10 > r.recall_at_10 and r2.qps > r.qps:
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    print(f"\nPareto-optimal configs (not dominated in recall vs speed):")
    for r in sorted(pareto, key=lambda x: -x.qps):
        print(f"  {r.name}: R@10={r.recall_at_10:.3f}, {r.qps:.1f} QPS")

    return results


if __name__ == "__main__":
    import sys

    # Parse args
    n_corpus = 50000
    if len(sys.argv) > 1:
        n_corpus = int(sys.argv[1])

    results = run_benchmarks(n_corpus=n_corpus, n_queries=100, dim=384)
