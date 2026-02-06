#!/usr/bin/env python3
"""
Generate a skewed dataset V3 to demonstrate weighted vs unweighted global centroid.

Key design:
- Big cluster at origin
- Many small clusters ALL at +10 on first axis (biased in ONE direction)
- Small clusters spread out in other dimensions to keep them distinct

This creates the bias:
- Unweighted global centroid pulled toward +10 on axis 0
- Weighted global centroid stays near origin (dominated by big cluster)
"""

import numpy as np
import struct
import os

# Parameters
D = 384  # dimensions
N_BIG = 1_000_000  # vectors in big cluster
M_SMALL = 256  # number of small clusters (more clusters to inflate centroid count)
N_SMALL = 50  # vectors per small cluster (small to ensure 1 centroid each)
SIGMA = 0.3  # standard deviation
OFFSET = 10.0  # offset for small clusters on first axis

N_QUERIES = 1000  # number of query vectors

# Output paths
OUTPUT_DIR = "/Users/alirezaheidarikhazaei/ann-prototypes/data"
DOC_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v3-dataset.fvec")
QUERY_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v3-queries.fvec")

def write_fvec(path, vectors):
    """Write vectors to fvec format (dimension + float32 values per vector)."""
    n, d = vectors.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(vectors[i].astype(np.float32).tobytes())
    print(f"Written {n} vectors of dimension {d} to {path}")
    print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")

def main():
    np.random.seed(42)

    total_small = M_SMALL * N_SMALL
    total = N_BIG + total_small
    
    print(f"Generating skewed dataset V3...")
    print(f"  Dimensions: {D}")
    print(f"  Big cluster: {N_BIG} vectors at origin")
    print(f"  Small clusters: {M_SMALL} clusters x {N_SMALL} vectors = {total_small} vectors")
    print(f"  All small clusters at +{OFFSET} on axis 0")
    print(f"  Total vectors: {total}")
    print()

    # Generate big cluster centered at origin
    print("Generating big cluster...")
    big_cluster = np.random.randn(N_BIG, D).astype(np.float32) * SIGMA

    # Generate small clusters - all at +10 on first axis, spread in other dims
    print("Generating small clusters...")
    small_clusters = []
    
    for i in range(M_SMALL):
        # Center: +10 on axis 0, random positions on other axes to separate clusters
        center = np.zeros(D, dtype=np.float32)
        center[0] = OFFSET
        # Spread clusters apart on axes 1-10 to keep them distinct
        center[1:11] = np.random.randn(10) * 2.0  # Larger spread on secondary axes
        
        # Generate vectors around this center
        cluster = center + np.random.randn(N_SMALL, D).astype(np.float32) * SIGMA
        small_clusters.append(cluster)

    small_clusters = np.vstack(small_clusters)

    # Combine all vectors
    print("Combining clusters...")
    all_vectors = np.vstack([big_cluster, small_clusters])

    # Shuffle
    print("Shuffling vectors...")
    indices = np.random.permutation(len(all_vectors))
    all_vectors = all_vectors[indices]

    # Generate queries from the big cluster
    print(f"Generating {N_QUERIES} query vectors from big cluster distribution...")
    queries = np.random.randn(N_QUERIES, D).astype(np.float32) * SIGMA

    # Statistics
    print()
    print("Dataset statistics:")
    print(f"  Big cluster mean[0]: {big_cluster[:, 0].mean():.4f}")
    print(f"  Small clusters mean[0]: {small_clusters[:, 0].mean():.4f}")
    print(f"  All vectors mean[0]: {all_vectors[:, 0].mean():.4f}")
    print(f"  Query vectors mean[0]: {queries[:, 0].mean():.4f}")
    print()
    
    # Expected centroid behavior with cluster_size=128
    n_centroids_big = N_BIG // 128
    n_centroids_small = M_SMALL  # Each small cluster should get ~1 centroid
    
    # Unweighted mean of centroids on axis 0
    # Big cluster centroids: ~0
    # Small cluster centroids: ~+10
    unweighted_mean_0 = (n_centroids_big * 0 + n_centroids_small * OFFSET) / (n_centroids_big + n_centroids_small)
    
    # Weighted mean on axis 0
    # Each centroid from big cluster represents ~128 vectors
    # Each centroid from small clusters represents ~50 vectors
    weighted_mean_0 = (N_BIG * 0 + total_small * OFFSET) / total
    
    print("Expected centroid behavior (with cluster_size=128):")
    print(f"  Big cluster centroids: ~{n_centroids_big} (at axis0 ≈ 0)")
    print(f"  Small cluster centroids: ~{n_centroids_small} (at axis0 ≈ {OFFSET})")
    print(f"  Total centroids: ~{n_centroids_big + n_centroids_small}")
    print()
    print(f"  UNWEIGHTED global centroid axis0: {unweighted_mean_0:.4f}")
    print(f"  WEIGHTED global centroid axis0: {weighted_mean_0:.4f}")
    print(f"  True data mean axis0: {weighted_mean_0:.4f}")
    print()
    print(f"  Bias from unweighted: {unweighted_mean_0 - weighted_mean_0:.4f}")
    print()

    # Write to files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_fvec(DOC_VECTORS_PATH, all_vectors)
    write_fvec(QUERY_VECTORS_PATH, queries)

    print()
    print("Done!")
    print(f"Doc vectors: {DOC_VECTORS_PATH}")
    print(f"Query vectors: {QUERY_VECTORS_PATH}")

if __name__ == "__main__":
    main()
