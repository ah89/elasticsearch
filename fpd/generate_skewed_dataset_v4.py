#!/usr/bin/env python3
"""
Generate extreme skewed dataset V4 to maximize weighted vs unweighted bias.

Key design:
- Keep small clusters at/below MAXK (128) so they can be represented early
- Use larger cluster_size to reduce total centroids and increase unweighted bias
- Keep small clusters tiny so weighted mean stays near the big cluster
"""

import numpy as np
import struct
import os

# Parameters - designed to maximize weighted vs unweighted divergence
D = 960  # dimensions (match GIST)
N_BIG = 1_000_000  # vectors in big cluster
# Keep small cluster count at/below MAXK so they can be represented early
M_SMALL = 64  # small clusters (each should capture a centroid if far enough)
N_SMALL = 1000  # vectors per small cluster
SIGMA = 0.05  # very tight clusters
# Large offset so small clusters are expensive to ignore in k-means
OFFSET = 20000.0  # offset for small clusters on first axis
# Larger cluster size reduces total centroids -> unweighted mean shifts more
CLUSTER_SIZE = 16384

N_QUERIES = 1000  # number of query vectors

# Output paths
OUTPUT_DIR = "/tmp/knn-skewed"
DOC_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v4-dataset.fvec")
QUERY_VECTORS_PATH = os.path.join(OUTPUT_DIR, "skewed-v4-queries.fvec")

def write_fvec(path, vectors):
    """Write vectors to fvec format."""
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
    
    print(f"Generating EXTREME skewed dataset V4...")
    print(f"  Dimensions: {D}")
    print(f"  Big cluster: {N_BIG} vectors at origin")
    print(f"  Small clusters: {M_SMALL} clusters x {N_SMALL} vectors = {total_small} vectors")
    print(f"  All small clusters at +{OFFSET} on axis 0")
    print(f"  Total vectors: {total}")
    print()

    # Generate big cluster centered at origin
    print("Generating big cluster...")
    big_cluster = np.random.randn(N_BIG, D).astype(np.float32) * SIGMA

    # Generate small clusters - all at +OFFSET on axis 0, with unique orthogonal offsets
    print("Generating small clusters...")
    small_clusters = []
    
    for i in range(M_SMALL):
        # Base center: large +OFFSET on axis 0 for directional bias
        center = np.zeros(D, dtype=np.float32)
        center[0] = OFFSET
        # Give each small cluster its own axis to force separation while keeping axis0 bias
        # With M_SMALL=64 and D=960, axis (i+1) is always valid.
        center[i + 1] = OFFSET
        
        # Generate vectors around this center
        cluster = center + np.random.randn(N_SMALL, D).astype(np.float32) * SIGMA
        small_clusters.append(cluster)

    small_clusters = np.vstack(small_clusters)

    # Combine and shuffle
    print("Combining and shuffling...")
    all_vectors = np.vstack([big_cluster, small_clusters])
    indices = np.random.permutation(len(all_vectors))
    all_vectors = all_vectors[indices]

    # Generate queries from the big cluster
    print(f"Generating {N_QUERIES} query vectors...")
    queries = np.random.randn(N_QUERIES, D).astype(np.float32) * SIGMA

    # Calculate expected behavior (rough estimate)
    # Note: actual centroid allocation is algorithm-dependent; this is a sanity check.
    n_centroids_est = int((total + (CLUSTER_SIZE / 2.0)) / CLUSTER_SIZE)
    n_centroids_big = max(n_centroids_est - M_SMALL, 1)
    n_centroids_small = M_SMALL  # if each small cluster captures a centroid
    
    unweighted_mean_0 = (n_centroids_big * 0 + n_centroids_small * OFFSET) / (n_centroids_big + n_centroids_small)
    weighted_mean_0 = (N_BIG * 0 + total_small * OFFSET) / total

    print()
    print("=" * 60)
    print(f"EXPECTED CENTROID BEHAVIOR (with cluster_size={CLUSTER_SIZE}):")
    print("=" * 60)
    print(f"  Estimated total centroids: ~{n_centroids_est}")
    print(f"  Big cluster centroids (est): ~{n_centroids_big} (at axis0 ≈ 0)")
    print(f"  Small cluster centroids (if 1 each): ~{n_centroids_small} (at axis0 ≈ {OFFSET})")
    print(f"  Total centroids (est): ~{n_centroids_big + n_centroids_small}")
    print()
    print(f"  ** UNWEIGHTED global centroid axis0: {unweighted_mean_0:.2f} **")
    print(f"  ** WEIGHTED global centroid axis0: {weighted_mean_0:.2f} **")
    print(f"  ** BIAS (unweighted - weighted): {unweighted_mean_0 - weighted_mean_0:.2f} **")
    print()
    print(f"  Queries are around axis0 ≈ 0")
    print(f"  True neighbors are in big cluster (axis0 ≈ 0)")
    print(f"  Unweighted centroid is biased toward +{unweighted_mean_0:.2f}")
    print("=" * 60)

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
