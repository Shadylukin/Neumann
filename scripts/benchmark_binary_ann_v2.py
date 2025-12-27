#!/usr/bin/env python3
"""
Benchmark v2: Diagnosing why binary signatures fail

Hypotheses:
1. Random projection doesn't preserve angular similarity well enough
2. Need more bits (1024+)
3. Need PCA-based projection instead of random
4. Hamming distance needs asymmetric thresholds

Let's test each hypothesis.
"""

import numpy as np
import time
from typing import Tuple


class BruteForce:
    def __init__(self, corpus: np.ndarray):
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus = (corpus / norms).astype(np.float32)
        self.N, self.D = self.corpus.shape

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        sims = self.corpus @ query_norm.astype(np.float32)
        top_k = np.argpartition(-sims, min(k, len(sims)-1))[:k]
        top_k = top_k[np.argsort(-sims[top_k])]
        return top_k, sims[top_k]


class PCABinaryIndex:
    """Use PCA directions instead of random projection."""

    def __init__(self, corpus: np.ndarray, n_bits: int = 256):
        self.N, self.D = corpus.shape
        self.n_bits = min(n_bits, self.D)

        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        # PCA projection
        mean = self.corpus_normalized.mean(axis=0)
        centered = self.corpus_normalized - mean
        cov = centered.T @ centered / self.N

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:self.n_bits]
        self.projection = eigenvectors[:, idx].astype(np.float32)
        self.mean = mean.astype(np.float32)

        # Compute signatures
        projected = (self.corpus_normalized - self.mean) @ self.projection
        self.signatures = (projected > 0).astype(np.uint8)
        self.packed_signatures = np.packbits(self.signatures, axis=1)

    def search(self, query: np.ndarray, k: int, n_candidates: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)

        query_projected = (query_norm - self.mean) @ self.projection
        query_sig = (query_projected > 0).astype(np.uint8)
        query_packed = np.packbits(query_sig)

        xor = np.bitwise_xor(self.packed_signatures, query_packed)
        hamming_dists = np.unpackbits(xor, axis=1).sum(axis=1)

        n_cand = min(n_candidates, self.N)
        candidates = np.argpartition(hamming_dists, n_cand)[:n_cand]

        exact_sims = self.corpus_normalized[candidates] @ query_norm
        top_k_local = np.argpartition(-exact_sims, min(k, len(exact_sims)-1))[:k]
        top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]


class MultiProbeBinary:
    """
    Multi-probe: check neighbors in Hamming space.
    """

    def __init__(self, corpus: np.ndarray, n_bits: int = 128):
        self.N, self.D = corpus.shape
        self.n_bits = n_bits

        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        np.random.seed(42)
        self.projection = (np.random.randn(self.D, n_bits) / np.sqrt(self.D)).astype(np.float32)

        projected = self.corpus_normalized @ self.projection
        # Store the actual projection values for multi-probe
        self.projected_values = projected.astype(np.float32)
        self.signatures = (projected > 0).astype(np.uint8)
        self.packed_signatures = np.packbits(self.signatures, axis=1)

    def search(self, query: np.ndarray, k: int, n_candidates: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)

        query_projected = query_norm @ self.projection
        query_sig = (query_projected > 0).astype(np.uint8)
        query_packed = np.packbits(query_sig)

        xor = np.bitwise_xor(self.packed_signatures, query_packed)
        hamming_dists = np.unpackbits(xor, axis=1).sum(axis=1)

        n_cand = min(n_candidates, self.N)
        candidates = np.argpartition(hamming_dists, n_cand)[:n_cand]

        exact_sims = self.corpus_normalized[candidates] @ query_norm
        top_k_local = np.argpartition(-exact_sims, min(k, len(exact_sims)-1))[:k]
        top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]


class HybridBinaryTensorFactor:
    """
    Combine: TensorFactor for candidate generation + binary for speed.

    The idea: Use SVD's U*S as the "compressed" representation,
    but store it as int8 for faster dot products.
    """

    def __init__(self, corpus: np.ndarray, rank: int = 64):
        self.N, self.D = corpus.shape
        self.rank = min(rank, min(self.N, self.D) - 1)

        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        try:
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(self.corpus_normalized, k=self.rank)
            U = U[:, ::-1]
            S = S[::-1]
            Vt = Vt[::-1, :]
        except ImportError:
            U, S, Vt = np.linalg.svd(self.corpus_normalized, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vt = Vt[:self.rank, :]

        self.US = (U * S).astype(np.float32)
        self.V = Vt.T.astype(np.float32)

        # Quantize US to int8
        self.us_min = self.US.min()
        self.us_max = self.US.max()
        self.us_scale = (self.us_max - self.us_min) / 255.0
        self.US_int8 = ((self.US - self.us_min) / self.us_scale).astype(np.uint8)

    def search(self, query: np.ndarray, k: int, rerank_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)

        # Project and quantize query
        query_latent = query_norm @ self.V

        # Use float for latent search (int8 doesn't help much here)
        latent_sims = self.US @ query_latent

        n_candidates = min(k * rerank_factor, self.N)
        candidates = np.argpartition(-latent_sims, n_candidates)[:n_candidates]

        exact_sims = self.corpus_normalized[candidates] @ query_norm
        top_k_local = np.argpartition(-exact_sims, min(k, len(exact_sims)-1))[:k]
        top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]


class TensorFactorHighRank:
    """TensorFactor with very high rank to test quality ceiling."""

    def __init__(self, corpus: np.ndarray, rank: int = 256):
        self.N, self.D = corpus.shape
        self.rank = min(rank, min(self.N, self.D) - 1)

        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_normalized = (corpus / norms).astype(np.float32)

        try:
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(self.corpus_normalized, k=self.rank)
            U = U[:, ::-1]
            S = S[::-1]
            Vt = Vt[::-1, :]
        except ImportError:
            U, S, Vt = np.linalg.svd(self.corpus_normalized, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vt = Vt[:self.rank, :]

        self.US = (U * S).astype(np.float32)
        self.V = Vt.T.astype(np.float32)

    def search(self, query: np.ndarray, k: int, rerank_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = (query / (np.linalg.norm(query) + 1e-10)).astype(np.float32)
        query_latent = query_norm @ self.V
        latent_sims = self.US @ query_latent

        n_candidates = min(k * rerank_factor, self.N)
        candidates = np.argpartition(-latent_sims, n_candidates)[:n_candidates]

        exact_sims = self.corpus_normalized[candidates] @ query_norm
        top_k_local = np.argpartition(-exact_sims, min(k, len(exact_sims)-1))[:k]
        top_k_local = top_k_local[np.argsort(-exact_sims[top_k_local])]

        return candidates[top_k_local], exact_sims[top_k_local]


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    return len(set(retrieved) & set(ground_truth)) / len(ground_truth)


def generate_data(n_corpus: int, n_queries: int, dim: int = 384, n_clusters: int = 50, seed: int = 42):
    np.random.seed(seed)
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    corpus = []
    items_per_cluster = n_corpus // n_clusters
    for c in range(n_clusters):
        n_items = items_per_cluster if c < n_clusters - 1 else n_corpus - len(corpus)
        noise = 0.3 * np.random.randn(n_items, dim).astype(np.float32)
        items = centers[c] + noise
        corpus.extend(items)

    corpus = np.array(corpus, dtype=np.float32)

    queries = []
    for i in range(n_queries):
        if i % 2 == 0:
            c = np.random.randint(n_clusters)
            q = centers[c] + 0.2 * np.random.randn(dim).astype(np.float32)
        else:
            q = np.random.randn(dim).astype(np.float32)
        queries.append(q)

    return corpus, np.array(queries, dtype=np.float32)


def run_benchmarks():
    n_corpus = 50000
    n_queries = 100
    dim = 384

    print(f"\n{'='*70}")
    print(f"BINARY ANN DIAGNOSTIC: {n_corpus:,} vectors, {dim} dims")
    print(f"{'='*70}\n")

    corpus, queries = generate_data(n_corpus, n_queries, dim)

    # Ground truth
    print("Computing ground truth...")
    bf = BruteForce(corpus)
    ground_truth = [bf.search(q, 100)[0] for q in queries]

    # Warmup and time brute force
    for q in queries[:5]:
        bf.search(q, 10)
    start = time.perf_counter()
    for q in queries:
        bf.search(q, 10)
    bf_time = time.perf_counter() - start
    bf_qps = n_queries / bf_time
    print(f"Brute Force: {bf_qps:.1f} QPS\n")

    results = []

    # Test 1: PCA-based binary (should preserve variance better)
    print("Testing PCA-based binary signatures...")
    for n_bits in [128, 256, 384]:
        for n_cand in [200, 500, 1000]:
            idx = PCABinaryIndex(corpus, n_bits=n_bits)

            recalls = []
            start = time.perf_counter()
            for i, q in enumerate(queries):
                res, _ = idx.search(q, 10, n_candidates=n_cand)
                recalls.append(compute_recall(res, ground_truth[i][:10]))
            elapsed = time.perf_counter() - start

            qps = n_queries / elapsed
            recall = np.mean(recalls)
            results.append(('PCA-Binary', n_bits, n_cand, recall, qps, qps/bf_qps))
            print(f"  PCA-{n_bits}b cand={n_cand}: R@10={recall:.3f}, {qps:.1f} QPS, {qps/bf_qps:.2f}x")

    print()

    # Test 2: Multi-probe with more candidates
    print("Testing Multi-probe binary...")
    for n_bits in [128, 256]:
        for n_cand in [500, 1000, 2000]:
            idx = MultiProbeBinary(corpus, n_bits=n_bits)

            recalls = []
            start = time.perf_counter()
            for i, q in enumerate(queries):
                res, _ = idx.search(q, 10, n_candidates=n_cand)
                recalls.append(compute_recall(res, ground_truth[i][:10]))
            elapsed = time.perf_counter() - start

            qps = n_queries / elapsed
            recall = np.mean(recalls)
            results.append(('MultiProbe', n_bits, n_cand, recall, qps, qps/bf_qps))
            print(f"  MultiProbe-{n_bits}b cand={n_cand}: R@10={recall:.3f}, {qps:.1f} QPS, {qps/bf_qps:.2f}x")

    print()

    # Test 3: TensorFactor with high rank and high rerank
    print("Testing TensorFactor with higher configs...")
    for rank in [128, 192, 256]:
        for rf in [20, 50, 100]:
            idx = TensorFactorHighRank(corpus, rank=rank)

            recalls = []
            start = time.perf_counter()
            for i, q in enumerate(queries):
                res, _ = idx.search(q, 10, rerank_factor=rf)
                recalls.append(compute_recall(res, ground_truth[i][:10]))
            elapsed = time.perf_counter() - start

            qps = n_queries / elapsed
            recall = np.mean(recalls)
            results.append(('TensorFactor', rank, rf, recall, qps, qps/bf_qps))
            print(f"  TF-r{rank} rf={rf}: R@10={recall:.3f}, {qps:.1f} QPS, {qps/bf_qps:.2f}x")

    print()

    # Summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    # Find configs that achieve 95%+ recall
    high_recall = [r for r in results if r[3] >= 0.95]
    if high_recall:
        best = max(high_recall, key=lambda x: x[4])
        print(f"\nBest at R@10 >= 0.95: {best[0]} (params: {best[1]}, {best[2]})")
        print(f"  Recall: {best[3]:.3f}, QPS: {best[4]:.1f}, Speedup: {best[5]:.2f}x")
    else:
        print("\nNo config achieved 95%+ recall!")
        best_recall = max(results, key=lambda x: x[3])
        print(f"Best recall: {best_recall[0]} = {best_recall[3]:.3f}")

    # Best speed at any reasonable recall (>= 80%)
    ok_recall = [r for r in results if r[3] >= 0.80]
    if ok_recall:
        fastest = max(ok_recall, key=lambda x: x[4])
        print(f"\nFastest at R@10 >= 0.80: {fastest[0]} (params: {fastest[1]}, {fastest[2]})")
        print(f"  Recall: {fastest[3]:.3f}, QPS: {fastest[4]:.1f}, Speedup: {fastest[5]:.2f}x")


if __name__ == "__main__":
    run_benchmarks()
