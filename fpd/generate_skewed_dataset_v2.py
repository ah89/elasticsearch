#!/usr/bin/env python3
"""
Generate a skewed dataset V2 to demonstrate weighted vs unweighted global centroid.

Key changes from V1:
- Larger sigma (0.2) for better separation within clusters
- Small clusters spread out across vector space (not all at +10)
- This ensures each small cluster gets its own centroid

Dataset structure:
- 1 big cluster: 1,000,000 vectors centered at origin (0)
- 128 small clusters: 100 vectors each, spread across the space
"""

import numpy as np
import struct
import os

# Parameters
D = 384  # dimensions
N_BIG = 1_000_000  # vectors in big cluster
M_SMALL = 128  # number of small clusters
N_SMALL = 100  # vectors per small cluster (smaller to ensure 1 centroid each)
SIGMA = 0.2  # larger standard deviation for better separation
OFFSET_BASE = 10.0  # base offset for small clusters

N_QUERIES = 1000  # number of query vectors

# Output paths
OUTPUT_DIR = "/Users/alirezaheidarikhazaei/ann-prototypes/data"
DOC_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v2-dataset.fvec")
QUERY_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v2-queries.fvec")

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

    print(f"Generating skewed dataset V2...")
    print(f"  Dimensions: {D}")
    print(f"  Big cluster: {N_BIG} vectors at origin, sigma={SIGMA}")
    print(f"  Small clusters: {M_SMALL} clusters x {N_SMALL} vectors = {M_SMALL * N_SMALL} vectors")
    print(f"  Total vectors: {N_BIG + M_SMALL * N_SMALL}")
    print()

    # Generate big cluster centered at origin with larger sigma
    print("Generating big cluster...")
    big_cluster = np.random.randn(N_BIG, D).astype(np.float32) * SIGMA

    # Generate small clusters spread across different regions
    # Each cluster will be at a different location in the high-dimensional space
    print("Generating small clusters (spread out in space)...")
    small_clusters = []
    
    # Create centers for small clusters that are spread out
    # Use random directions in the high-dimensional space
    for i in range(M_SMALL):
        # Each small cluster center is in a unique direction
        # The center is at distance OFFSET_BASE from origin
        direction = np.random.randn(D).astype(np.float32)
        direction = direction / np.linalg.norm(direction)  # Normalize
        center = direction * OFFSET_BASE  # Move in that direction
        
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
    print(f"  Big cluster:")
    print(f"    Mean norm: {np.linalg.norm(big_cluster, axis=1).mean():.4f}")
    print(f"    Mean[0]: {big_cluster[:, 0].mean():.4f}")
    print(f"  Small clusters:")
    print(f"    Mean norm: {np.linalg.norm(small_clusters, axis=1).mean():.4f}")
    print(f"    Mean[0]: {small_clusters[:, 0].mean():.4f}")
    print(f"  Queries:")
    print(f"    Mean norm: {np.linalg.norm(queries, axis=1).mean():.4f}")
    print()

    # Compute expected global centroid positions
    # With cluster_size=128 and N_SMALL=100, each small cluster should get ~1 centroid
    # Big cluster has ~7800 centroids (1M/128)
    # Small clusters have ~128 centroids (one per cluster)
    
    # Unweighted mean of centroids:
    # - Big cluster centroids are near origin (mean norm ~0)
    # - Small cluster centroids are at distance ~10 from origin
    # Unweighted mean = (7800 * 0 + 128 * average_small_centroid) / 7928
    
    # For small clusters spread in random directions, their centroid mean is ~0
    # BUT their mean DISTANCE from origin is ~10
    
    print("Expected centroid behavior:")
    print(f"  With cluster_size=128:")
    print(f"    Big cluster centroids: ~{N_BIG // 128} (all near origin)")
    print(f"    Small cluster centroids: ~{M_SMALL} (spread at distance ~{OFFSET_BASE})")
    print(f"  Total centroids: ~{N_BIG // 128 + M_SMALL}")
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
