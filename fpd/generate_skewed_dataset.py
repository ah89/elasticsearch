#!/usr/bin/env python3
"""
Generate a skewed dataset to demonstrate weighted vs unweighted global centroid.

Dataset structure:
- 1 big cluster: 1,000,000 vectors centered at origin (0)
- 128 small clusters: 200 vectors each, centered around +10 on first axis

This creates extreme bias for unweighted global centroid:
- True mean ≈ 0 (dominated by big cluster)
- Unweighted centroid mean ≈ +10 (many small clusters)
"""

import numpy as np
import struct
import os

# Parameters
D = 384  # dimensions
N_BIG = 1_000_000  # vectors in big cluster
M_SMALL = 128  # number of small clusters
N_SMALL = 200  # vectors per small cluster
SIGMA = 0.05  # standard deviation
OFFSET = 10.0  # offset for small clusters on first axis

N_QUERIES = 1000  # number of query vectors

# Output paths
OUTPUT_DIR = "/Users/alirezaheidarikhazaei/ann-prototypes/data"
DOC_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-dataset.fvec")
QUERY_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-queries.fvec")

def write_fvec(path, vectors):
    """Write vectors to fvec format (dimension + float32 values per vector)."""
    n, d = vectors.shape
    with open(path, 'wb') as f:
        for i in range(n):
            # Write dimension as int32
            f.write(struct.pack('i', d))
            # Write vector as float32
            f.write(vectors[i].astype(np.float32).tobytes())
    print(f"Written {n} vectors of dimension {d} to {path}")
    print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")

def main():
    np.random.seed(42)  # For reproducibility
    
    print(f"Generating skewed dataset...")
    print(f"  Dimensions: {D}")
    print(f"  Big cluster: {N_BIG} vectors at origin")
    print(f"  Small clusters: {M_SMALL} clusters x {N_SMALL} vectors = {M_SMALL * N_SMALL} vectors at +{OFFSET}")
    print(f"  Total vectors: {N_BIG + M_SMALL * N_SMALL}")
    print()
    
    # Generate big cluster centered at origin
    print("Generating big cluster...")
    big_cluster = np.random.randn(N_BIG, D).astype(np.float32) * SIGMA
    
    # Generate small clusters centered around +10 on first axis
    print("Generating small clusters...")
    small_clusters = []
    for i in range(M_SMALL):
        # Center for this small cluster: +10 on first axis, small jitter on others
        center = np.zeros(D, dtype=np.float32)
        center[0] = OFFSET + np.random.randn() * 0.1  # Small jitter around 10
        center[1:10] = np.random.randn(9) * 0.1  # Small jitter on a few dimensions
        
        # Generate vectors around this center
        cluster = center + np.random.randn(N_SMALL, D).astype(np.float32) * SIGMA
        small_clusters.append(cluster)
    
    small_clusters = np.vstack(small_clusters)
    
    # Combine all vectors
    print("Combining clusters...")
    all_vectors = np.vstack([big_cluster, small_clusters])
    
    # Shuffle to mix big and small cluster vectors
    print("Shuffling vectors...")
    indices = np.random.permutation(len(all_vectors))
    all_vectors = all_vectors[indices]
    
    # Generate queries from the big cluster (around origin)
    print(f"Generating {N_QUERIES} query vectors from big cluster distribution...")
    queries = np.random.randn(N_QUERIES, D).astype(np.float32) * SIGMA
    
    # Statistics
    print()
    print("Dataset statistics:")
    print(f"  All vectors mean[0]: {all_vectors[:, 0].mean():.4f}")
    print(f"  All vectors std[0]: {all_vectors[:, 0].std():.4f}")
    print(f"  Big cluster mean[0]: {big_cluster[:, 0].mean():.4f}")
    print(f"  Small clusters mean[0]: {small_clusters[:, 0].mean():.4f}")
    print(f"  Query vectors mean[0]: {queries[:, 0].mean():.4f}")
    print()
    
    # True mean of all vectors (should be close to 0)
    true_mean = all_vectors.mean(axis=0)
    print(f"  True global mean[0]: {true_mean[0]:.4f}")
    
    # Approximate unweighted centroid mean (assuming ~equal cluster sizes after k-means)
    # With cluster_size=128, we expect ~8000 centroids from big cluster, ~128 from small clusters
    # But small clusters are dense, so they might get fewer centroids
    # The key is that many centroids will be around +10
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
