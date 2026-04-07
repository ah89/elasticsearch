/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.coreset;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Kernel-herding coreset in a Reproducing Kernel Hilbert Space (RKHS) for a
 * single continual-learning domain.
 *
 * <p>The domain is summarised as {@code k} support points chosen by greedy
 * herding that minimises the Maximum Mean Discrepancy (MMD) between the coreset
 * and the full domain distribution in an RBF-kernel RKHS.  Random Fourier
 * Features (Rahimi &amp; Recht, 2007) are used to approximate the kernel so that
 * the mean embedding {@code μ_P} can be computed in {@code O(n * rffDim)} time
 * rather than {@code O(n^2)}.
 *
 * <p>The RBF bandwidth is estimated via the <em>median heuristic</em>: the
 * median pairwise Euclidean distance on a sub-sample of up to 500 embeddings.
 *
 * <p>Overlap between two domains is measured via the MMD-based similarity
 * {@code exp(-MMD² / (2σ²))}, which equals 1 when the two distributions are
 * identical and decays toward 0 as they diverge.
 *
 * <p>Time complexity of {@link #fit}: {@code O(n * rffDim + k * (n - k/2) * rffDim)}.
 */
public class HilbertCoreset implements GeometricCoreset {

    public static final String TYPE = "hilbert";

    /** Dimensionality of Random Fourier Feature approximation. */
    private static final int RFF_DIM = 256;

    private final int dimension;
    /** Herding support points, shape [k][d]. */
    private final float[][] support;
    /** Uniform weights (1/k each). */
    private final float[] weights;
    /** RBF bandwidth estimated via median heuristic. */
    private final float bandwidth;
    /**
     * Random Fourier Feature projection matrix, shape [rffDim][d].
     * Stored per-coreset so that serialised coresets are self-contained.
     */
    private final float[][] rffW;
    /** Random Fourier Feature phase offsets, length rffDim. */
    private final float[] rffB;

    /**
     * Constructs a Hilbert coreset from pre-computed herding support points.
     *
     * @param dimension embedding dimensionality
     * @param support   herding support points, shape [k][d]
     * @param weights   point weights, length k, must sum to 1
     * @param bandwidth RBF kernel bandwidth
     * @param rffW      RFF projection matrix, shape [rffDim][d]
     * @param rffB      RFF phase offsets, length rffDim
     */
    public HilbertCoreset(int dimension, float[][] support, float[] weights, float bandwidth, float[][] rffW, float[] rffB) {
        this.dimension = dimension;
        this.support = support;
        this.weights = weights;
        this.bandwidth = bandwidth;
        this.rffW = rffW;
        this.rffB = rffB;
    }

    /** Deserialisation constructor. */
    public HilbertCoreset(StreamInput in) throws IOException {
        this.dimension = in.readVInt();
        int k = in.readVInt();
        this.support = new float[k][dimension];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimension; j++) {
                this.support[i][j] = in.readFloat();
            }
        }
        this.weights = new float[k];
        for (int i = 0; i < k; i++) {
            this.weights[i] = in.readFloat();
        }
        this.bandwidth = in.readFloat();
        int rffDim = in.readVInt();
        this.rffW = new float[rffDim][dimension];
        for (int i = 0; i < rffDim; i++) {
            for (int j = 0; j < dimension; j++) {
                this.rffW[i][j] = in.readFloat();
            }
        }
        this.rffB = new float[rffDim];
        for (int i = 0; i < rffDim; i++) {
            this.rffB[i] = in.readFloat();
        }
    }

    // -------------------------------------------------------------------------
    // Factory: greedy kernel herding
    // -------------------------------------------------------------------------

    /**
     * Builds a Hilbert coreset via greedy kernel herding.
     *
     * <ol>
     *   <li>Estimate RBF bandwidth via the median pairwise distance on a
     *       sub-sample of at most 500 embeddings.</li>
     *   <li>Compute Random Fourier Feature (RFF) approximations for all
     *       embeddings.</li>
     *   <li>Greedily select {@code k} points that minimise MMD between the
     *       coreset and the full distribution mean embedding.</li>
     * </ol>
     *
     * @param embeddings the embedding vectors to summarise
     * @param k          number of support points
     * @param rng        random source for RFF matrix and bandwidth sub-sample
     * @return fitted {@link HilbertCoreset}
     */
    public static HilbertCoreset fit(float[][] embeddings, int k, Random rng) {
        if (embeddings.length == 0) {
            throw new IllegalArgumentException("Cannot fit HilbertCoreset to empty embedding set");
        }
        int n = embeddings.length;
        int d = embeddings[0].length;
        k = Math.min(k, n);

        // 1. Bandwidth via median heuristic on a sub-sample
        float bandwidth = estimateBandwidth(embeddings, rng);

        // 2. Build RFF matrix: rffW[i] ~ N(0, 1/bandwidth²), rffB[i] ~ Uniform[0, 2π]
        float[][] rffW = new float[RFF_DIM][d];
        float[] rffB = new float[RFF_DIM];
        for (int i = 0; i < RFF_DIM; i++) {
            for (int j = 0; j < d; j++) {
                rffW[i][j] = (float) (rng.nextGaussian() / bandwidth);
            }
            rffB[i] = (float) (rng.nextDouble() * 2.0 * Math.PI);
        }

        // 3. Compute RFF features for all embeddings: Phi[i] = sqrt(2/rffDim) * cos(X @ rffW.T + rffB)
        float scale = (float) Math.sqrt(2.0 / RFF_DIM);
        float[][] phi = new float[n][RFF_DIM];
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < RFF_DIM; r++) {
                float proj = dot(embeddings[i], rffW[r]) + rffB[r];
                phi[i][r] = scale * (float) Math.cos(proj);
            }
        }

        // 4. Compute target mean embedding μ_P
        float[] muP = new float[RFF_DIM];
        for (float[] phiRow : phi) {
            for (int r = 0; r < RFF_DIM; r++) {
                muP[r] += phiRow[r];
            }
        }
        for (int r = 0; r < RFF_DIM; r++) {
            muP[r] /= n;
        }

        // 5. Greedy herding: select the point maximising dot(phi[i], residual)
        boolean[] chosen = new boolean[n];
        int[] chosenIndices = new int[k];
        float[] residual = Arrays.copyOf(muP, RFF_DIM);

        // Running mean of chosen RFF features
        float[] chosenMean = new float[RFF_DIM];

        for (int iter = 0; iter < k; iter++) {
            int best = -1;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (chosen[i] == false) {
                    float score = dot(phi[i], residual);
                    if (score > bestScore) {
                        bestScore = score;
                        best = i;
                    }
                }
            }
            chosen[best] = true;
            chosenIndices[iter] = best;

            // Update running mean and residual
            int t = iter + 1;
            for (int r = 0; r < RFF_DIM; r++) {
                chosenMean[r] = chosenMean[r] * (t - 1) / t + phi[best][r] / t;
                residual[r] = muP[r] - chosenMean[r];
            }
        }

        float[][] support = new float[k][d];
        for (int i = 0; i < k; i++) {
            support[i] = Arrays.copyOf(embeddings[chosenIndices[i]], d);
        }
        float[] weights = new float[k];
        Arrays.fill(weights, 1.0f / k);

        return new HilbertCoreset(d, support, weights, bandwidth, rffW, rffB);
    }

    /**
     * Estimates the RBF bandwidth as the median pairwise Euclidean distance on a
     * random sub-sample of at most 500 embeddings.
     */
    private static float estimateBandwidth(float[][] embeddings, Random rng) {
        int n = embeddings.length;
        int subN = Math.min(500, n);
        float[][] sub = new float[subN][];
        // Reservoir-sample subN points
        for (int i = 0; i < subN; i++) {
            sub[i] = embeddings[i];
        }
        for (int i = subN; i < n; i++) {
            int j = rng.nextInt(i + 1);
            if (j < subN) {
                sub[j] = embeddings[i];
            }
        }

        // Collect all pairwise squared distances (upper triangle)
        int pairs = subN * (subN - 1) / 2;
        float[] dists = new float[pairs];
        int idx = 0;
        for (int i = 0; i < subN; i++) {
            for (int j = i + 1; j < subN; j++) {
                dists[idx++] = euclideanDistance(sub[i], sub[j]);
            }
        }
        Arrays.sort(dists);
        float median = dists[pairs / 2];
        return Math.max(median, 1e-6f);
    }

    // -------------------------------------------------------------------------
    // GeometricCoreset implementation
    // -------------------------------------------------------------------------

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public List<float[]> getCentroids() {
        List<float[]> centroids = new ArrayList<>(support.length);
        for (float[] s : support) {
            centroids.add(Arrays.copyOf(s, dimension));
        }
        return centroids;
    }

    /**
     * Computes overlap via the MMD-based similarity
     * {@code exp(-MMD²(this, other) / (2 * bandwidth²))}, where MMD is
     * computed analytically from the RBF kernel evaluated on support points.
     */
    @Override
    public float computeOverlap(GeometricCoreset other) {
        if (other instanceof HilbertCoreset otherH) {
            return computeHilbertOverlap(otherH);
        }
        return computeCentroidProxyOverlap(other);
    }

    private float computeHilbertOverlap(HilbertCoreset other) {
        // MMD² = E[K(x,x')] - 2*E[K(x,y')] + E[K(y,y')]
        // where x ~ this, y ~ other, K is RBF with shared bandwidth
        float avgBw = (this.bandwidth + other.bandwidth) * 0.5f;
        double k11 = kernelMean(this.support, this.support, this.weights, this.weights, avgBw);
        double k12 = kernelMean(this.support, other.support, this.weights, other.weights, avgBw);
        double k22 = kernelMean(other.support, other.support, other.weights, other.weights, avgBw);
        double mmd2 = Math.max(0.0, k11 - 2.0 * k12 + k22);
        return (float) Math.exp(-mmd2 / (2.0 * (double) avgBw * avgBw));
    }

    /**
     * Computes the weighted kernel mean: sum_i sum_j w1[i] * w2[j] * K(a[i], b[j]).
     */
    private static double kernelMean(float[][] a, float[][] b, float[] w1, float[] w2, float bandwidth) {
        double result = 0;
        double inv2bw2 = 1.0 / (2.0 * (double) bandwidth * bandwidth);
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double d2 = squaredEuclidean(a[i], b[j]);
                result += (double) w1[i] * w2[j] * Math.exp(-d2 * inv2bw2);
            }
        }
        return result;
    }

    private float computeCentroidProxyOverlap(GeometricCoreset other) {
        List<float[]> myCentroids = getCentroids();
        List<float[]> otherCentroids = other.getCentroids();
        double maxSim = 0;
        for (float[] c1 : myCentroids) {
            for (float[] c2 : otherCentroids) {
                double dist = euclideanDistance(c1, c2);
                double sim = 1.0 / (1.0 + dist);
                if (sim > maxSim) {
                    maxSim = sim;
                }
            }
        }
        return (float) maxSim;
    }

    /**
     * Generates boundary-aware negatives by sampling from the support point
     * nearest to {@code targetCentroid} and displacing by {@code margin} toward
     * the target.
     */
    @Override
    public List<float[]> sampleBoundaryNegatives(float[] targetCentroid, int count, float margin) {
        Random rng = new Random();
        List<float[]> negatives = new ArrayList<>(count);

        // Find the support point nearest to the target centroid
        int nearest = 0;
        float minDist = Float.MAX_VALUE;
        for (int i = 0; i < support.length; i++) {
            float dist = euclideanDistance(support[i], targetCentroid);
            if (dist < minDist) {
                minDist = dist;
                nearest = i;
            }
        }
        float[] nearestPoint = support[nearest];

        // Direction from nearest support point toward target
        float[] dir = new float[dimension];
        double norm = 0;
        for (int i = 0; i < dimension; i++) {
            dir[i] = targetCentroid[i] - nearestPoint[i];
            norm += (double) dir[i] * dir[i];
        }
        norm = Math.sqrt(Math.max(norm, 1e-12));
        for (int i = 0; i < dimension; i++) {
            dir[i] = (float) (dir[i] / norm);
        }

        for (int s = 0; s < count; s++) {
            float[] perturbation = randomUnitVector(dimension, rng);
            float[] sample = new float[dimension];
            // Use bandwidth as the natural scale for boundary proximity
            float offset = bandwidth * margin;
            for (int i = 0; i < dimension; i++) {
                float blended = 0.9f * dir[i] + 0.1f * perturbation[i];
                sample[i] = nearestPoint[i] + offset * blended;
            }
            negatives.add(sample);
        }
        return negatives;
    }

    // -------------------------------------------------------------------------
    // Serialisation
    // -------------------------------------------------------------------------

    @Override
    public String getWriteableName() {
        return TYPE;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(dimension);
        int k = support.length;
        out.writeVInt(k);
        for (float[] s : support) {
            for (float v : s) {
                out.writeFloat(v);
            }
        }
        for (float w : weights) {
            out.writeFloat(w);
        }
        out.writeFloat(bandwidth);
        int rffDim = rffW.length;
        out.writeVInt(rffDim);
        for (float[] row : rffW) {
            for (float v : row) {
                out.writeFloat(v);
            }
        }
        for (float v : rffB) {
            out.writeFloat(v);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("type", TYPE);
        builder.field("dimension", dimension);
        builder.field("num_support", support.length);
        builder.field("bandwidth", bandwidth);
        builder.field("rff_dim", rffW.length);
        builder.startArray("support");
        for (int i = 0; i < support.length; i++) {
            builder.startObject();
            builder.array("point", support[i]);
            builder.field("weight", weights[i]);
            builder.endObject();
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public float[][] getSupport() {
        float[][] copy = new float[support.length][];
        for (int i = 0; i < support.length; i++) {
            copy[i] = Arrays.copyOf(support[i], support[i].length);
        }
        return copy;
    }

    public float[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public float getBandwidth() {
        return bandwidth;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static float[] randomUnitVector(int d, Random rng) {
        float[] v = new float[d];
        double norm = 0;
        for (int i = 0; i < d; i++) {
            v[i] = (float) rng.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(Math.max(norm, 1e-12));
        for (int i = 0; i < d; i++) {
            v[i] = (float) (v[i] / norm);
        }
        return v;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static float euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    private static double squaredEuclidean(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
