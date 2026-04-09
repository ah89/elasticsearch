/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.coreset;

import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Gaussian Mixture Model coreset for a single continual-learning domain.
 *
 * <p>Each domain is summarised as a mixture of {@code K} diagonal Gaussians fitted
 * to the domain's embedding distribution via the EM algorithm.  Storage per domain
 * is {@code O(K * d)} floats (means + diagonal variances + weights), which is orders
 * of magnitude smaller than retaining raw embeddings.
 *
 * <p>This is the recommended default coreset type in ContLoRA because it offers the
 * best accuracy / memory tradeoff: GMM captures multi-modal distributions, supports
 * analytic Bhattacharyya overlap computation, and enables fast, boundary-aware
 * negative sampling.
 */
public class GmmCoreset implements GeometricCoreset {

    public static final String TYPE = "gmm";

    /** Minimum per-dimension variance to prevent numerical collapse. */
    private static final float MIN_VARIANCE = 1e-6f;
    /** EM convergence threshold on relative log-likelihood change (matches sklearn default). */
    private static final double EM_CONVERGENCE_THRESHOLD = 1e-3;
    /** Maximum EM iterations — diagonal GMMs converge in 5-10 iterations typically. */
    private static final int MAX_EM_ITERATIONS = 20;
    /** Maximum samples used for EM fitting — bounds EM cost at O(MAX_EM_SAMPLES * K * d). */
    private static final int MAX_EM_SAMPLES = 2000;

    private final int dimension;
    private final float[] weights;
    /** Means: shape [K][d]. */
    private final float[][] means;
    /** Diagonal variances: shape [K][d]. */
    private final float[][] variances;

    /**
     * Constructs a GMM coreset from pre-fitted parameters.
     *
     * @param dimension embedding dimensionality
     * @param weights   mixture weights, length K, must sum to 1
     * @param means     component means, shape [K][d]
     * @param variances diagonal variances, shape [K][d], all entries &gt; 0
     */
    public GmmCoreset(int dimension, float[] weights, float[][] means, float[][] variances) {
        this.dimension = dimension;
        this.weights = weights;
        this.means = means;
        this.variances = variances;
    }

    /** Deserialisation constructor. */
    public GmmCoreset(StreamInput in) throws IOException {
        this.dimension = in.readVInt();
        int k = in.readVInt();
        this.weights = new float[k];
        for (int i = 0; i < k; i++) {
            this.weights[i] = in.readFloat();
        }
        this.means = new float[k][dimension];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimension; j++) {
                this.means[i][j] = in.readFloat();
            }
        }
        this.variances = new float[k][dimension];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimension; j++) {
                this.variances[i][j] = in.readFloat();
            }
        }
    }

    // -------------------------------------------------------------------------
    // Factory: fit GMM via k-means++ initialisation + EM
    // -------------------------------------------------------------------------

    /**
     * Fits a GMM coreset to the provided embeddings using k-means++ initialisation
     * followed by EM.
     *
     * <p>When the dataset exceeds {@link #MAX_EM_SAMPLES} embeddings, a random
     * subsample is used for EM fitting.  This bounds EM cost at O(MAX_EM_SAMPLES * k * d)
     * per iteration while preserving accuracy for diagonal-covariance GMMs.
     *
     * <p>Convergence is checked using relative log-likelihood change to avoid
     * iteration count depending on absolute scale.
     *
     * @param embeddings array of embedding vectors, each of length {@code d}
     * @param k          number of mixture components
     * @param rng        random source for reproducible results
     * @return fitted {@link GmmCoreset}
     */
    public static GmmCoreset fit(float[][] embeddings, int k, Random rng) {
        if (embeddings.length == 0) {
            throw new IllegalArgumentException("Cannot fit GMM to empty embedding set");
        }
        int d = embeddings[0].length;
        int n = embeddings.length;
        k = Math.min(k, n);

        // Subsample for EM fitting if dataset is large
        float[][] emData;
        if (n > MAX_EM_SAMPLES) {
            emData = subsample(embeddings, MAX_EM_SAMPLES, rng);
        } else {
            emData = embeddings;
        }
        int nEM = emData.length;

        // --- k-means++ initialisation ---
        float[][] means = kMeansPlusPlusInit(emData, k, rng);

        // --- Initialise equal weights and unit variances ---
        float[] weights = new float[k];
        Arrays.fill(weights, 1.0f / k);
        float[][] variances = new float[k][d];
        for (float[] row : variances) {
            Arrays.fill(row, 1.0f);
        }

        // --- EM iterations with SIMD-accelerated E-step and M-step ---
        double prevLogLikelihood = Double.NEGATIVE_INFINITY;
        double[][] responsibilities = new double[nEM][k];

        // Reusable per-component float arrays for E-step decomposition
        float[][] invVarF = new float[k][d];     // 1/variance
        float[][] muInvVarF = new float[k][d];   // means * invVar
        float[] muSqInvVarF = new float[k];      // sum(means^2 * invVar) per component
        double[] logConst = new double[k];

        // Precompute x_i^2 for Lucene-accelerated E-step: mah = dot(x², iv) - 2·dot(x, μ·iv) + μ²·iv
        float[][] emDataSq = new float[nEM][d];
        for (int i = 0; i < nEM; i++) {
            for (int dd = 0; dd < d; dd++) {
                emDataSq[i][dd] = emData[i][dd] * emData[i][dd];
            }
        }

        // Reusable M-step accumulators
        float[] nkF = new float[k];
        float[][] sumX = new float[k][d];
        float[][] sumXsq = new float[k][d];

        for (int iter = 0; iter < MAX_EM_ITERATIONS; iter++) {
            // Precompute per-component constants
            for (int j = 0; j < k; j++) {
                double logDet = 0;
                float muSqIv = 0;
                for (int dd = 0; dd < d; dd++) {
                    float v = Math.max(variances[j][dd], MIN_VARIANCE);
                    float iv = 1.0f / v;
                    invVarF[j][dd] = iv;
                    muInvVarF[j][dd] = means[j][dd] * iv;
                    muSqIv += means[j][dd] * muInvVarF[j][dd];
                    logDet += Math.log(v);
                }
                muSqInvVarF[j] = muSqIv;
                logConst[j] = Math.log(weights[j] + 1e-300) - 0.5 * (d * LOG_2PI + logDet);
            }

            // ── E-step: Mahalanobis via Lucene SIMD dot products ──
            // mah(x,j) = dot(x², iv[j]) - 2·dot(x, μ·iv[j]) + μ²·iv[j]
            double totalLogLikelihood = 0;
            for (int i = 0; i < nEM; i++) {
                double maxLog = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < k; j++) {
                    float t1 = VectorUtil.dotProduct(emDataSq[i], invVarF[j]);
                    float t2 = VectorUtil.dotProduct(emData[i], muInvVarF[j]);
                    double mahal = t1 - 2.0 * t2 + muSqInvVarF[j];
                    double lr = logConst[j] - 0.5 * mahal;
                    responsibilities[i][j] = lr;
                    if (lr > maxLog) maxLog = lr;
                }
                // log-sum-exp normalisation
                double logSum = 0;
                for (int j = 0; j < k; j++) logSum += Math.exp(responsibilities[i][j] - maxLog);
                logSum = maxLog + Math.log(logSum);
                totalLogLikelihood += logSum;
                for (int j = 0; j < k; j++) responsibilities[i][j] = Math.exp(responsibilities[i][j] - logSum);
            }

            // ── M-step: sample-major accumulation ──
            Arrays.fill(nkF, 0f);
            for (int j = 0; j < k; j++) {
                Arrays.fill(sumX[j], 0f);
                Arrays.fill(sumXsq[j], 0f);
            }

            for (int i = 0; i < nEM; i++) {
                float[] xi = emData[i];
                double[] ri = responsibilities[i];
                for (int j = 0; j < k; j++) {
                    float rij = (float) ri[j];
                    nkF[j] += rij;
                    float[] sj = sumX[j];
                    float[] sqj = sumXsq[j];
                    for (int dd = 0; dd < d; dd++) {
                        float xd = xi[dd];
                        sj[dd] += rij * xd;
                        sqj[dd] += rij * xd * xd;
                    }
                }
            }

            for (int j = 0; j < k; j++) {
                float nkj = Math.max(nkF[j], 1e-10f);
                weights[j] = nkF[j] / nEM;
                for (int dd = 0; dd < d; dd++) {
                    float mu = sumX[j][dd] / nkj;
                    means[j][dd] = mu;
                    variances[j][dd] = Math.max(sumXsq[j][dd] / nkj - mu * mu, MIN_VARIANCE);
                }
            }

            // Relative convergence check
            if (Math.abs(totalLogLikelihood - prevLogLikelihood) / Math.max(Math.abs(totalLogLikelihood), 1.0)
                < EM_CONVERGENCE_THRESHOLD) {
                break;
            }
            prevLogLikelihood = totalLogLikelihood;
        }

        return new GmmCoreset(d, weights, means, variances);
    }

    private static final double LOG_2PI = Math.log(2 * Math.PI);

    /** Subsample embeddings using Fisher-Yates partial shuffle. */
    private static float[][] subsample(float[][] embeddings, int n, Random rng) {
        int total = embeddings.length;
        int[] indices = new int[total];
        for (int i = 0; i < total; i++) {
            indices[i] = i;
        }
        float[][] result = new float[n][];
        for (int i = 0; i < n; i++) {
            int j = i + rng.nextInt(total - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            result[i] = embeddings[indices[i]];
        }
        return result;
    }

    /** k-means++ centre initialisation. */
    private static float[][] kMeansPlusPlusInit(float[][] embeddings, int k, Random rng) {
        int n = embeddings.length;
        int d = embeddings[0].length;
        float[][] centres = new float[k][d];

        // Pick first centre uniformly at random
        centres[0] = Arrays.copyOf(embeddings[rng.nextInt(n)], d);

        double[] minDistSq = new double[n];
        Arrays.fill(minDistSq, Double.MAX_VALUE);

        for (int c = 1; c < k; c++) {
            // Update minimum squared distances to any chosen centre
            for (int i = 0; i < n; i++) {
                double dist = squaredEuclidean(embeddings[i], centres[c - 1]);
                if (dist < minDistSq[i]) {
                    minDistSq[i] = dist;
                }
            }
            // Sample next centre proportional to distance squared
            double total = 0;
            for (double v : minDistSq) {
                total += v;
            }
            double threshold = rng.nextDouble() * total;
            double cumulative = 0;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                cumulative += minDistSq[i];
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }
            centres[c] = Arrays.copyOf(embeddings[chosen], d);
        }
        return centres;
    }

    /**
     * Log-density of a diagonal Gaussian at point {@code x}.
     * Used for single-point scoring (novelty detection).
     */
    static double logGaussianDiag(float[] x, float[] mean, float[] variance) {
        int d = x.length;
        double logDet = 0;
        double mahal = 0;
        for (int i = 0; i < d; i++) {
            double v = Math.max(variance[i], MIN_VARIANCE);
            logDet += Math.log(v);
            double diff = x[i] - mean[i];
            mahal += diff * diff / v;
        }
        return -0.5 * (d * LOG_2PI + logDet + mahal);
    }

    /**
     * Computes the log-likelihood of a single point under this GMM.
     * Uses Lucene-accelerated dot products for the Mahalanobis decomposition.
     *
     * @param x embedding vector of length {@code dimension}
     * @return log p(x | GMM)
     */
    public double logLikelihood(float[] x) {
        int k = weights.length;
        int d = dimension;

        // Precompute x² once
        float[] xSq = new float[d];
        for (int dd = 0; dd < d; dd++) xSq[dd] = x[dd] * x[dd];

        double maxLog = Double.NEGATIVE_INFINITY;
        double[] logResp = new double[k];
        for (int j = 0; j < k; j++) {
            // Build iv and mu_iv for this component
            float[] iv = new float[d];
            float[] muIv = new float[d];
            double logDet = 0;
            float muSqIv = 0;
            for (int dd = 0; dd < d; dd++) {
                float v = Math.max(variances[j][dd], MIN_VARIANCE);
                iv[dd] = 1.0f / v;
                muIv[dd] = means[j][dd] * iv[dd];
                muSqIv += means[j][dd] * muIv[dd];
                logDet += Math.log(v);
            }
            float t1 = VectorUtil.dotProduct(xSq, iv);
            float t2 = VectorUtil.dotProduct(x, muIv);
            double mahal = t1 - 2.0 * t2 + muSqIv;
            logResp[j] = Math.log(weights[j] + 1e-300) - 0.5 * (d * LOG_2PI + logDet + mahal);
            if (logResp[j] > maxLog) maxLog = logResp[j];
        }
        double sum = 0;
        for (int j = 0; j < k; j++) sum += Math.exp(logResp[j] - maxLog);
        return maxLog + Math.log(sum);
    }

    /**
     * Computes log-likelihoods for a batch of points under this GMM.
     * Uses SIMD-accelerated Mahalanobis decomposition with precomputed constants.
     *
     * @param batch array of embedding vectors, each of length {@code dimension}
     * @return array of log p(x | GMM) for each input
     */
    public double[] batchLogLikelihood(float[][] batch) {
        int m = batch.length;
        int k = weights.length;
        int d = dimension;

        // Precompute per-component constants
        float[][] ivF = new float[k][d];
        float[][] muIvF = new float[k][d];
        float[] muSqIvF = new float[k];
        double[] logConst = new double[k];
        for (int j = 0; j < k; j++) {
            double logDet = 0;
            float msiv = 0;
            for (int dd = 0; dd < d; dd++) {
                float v = Math.max(variances[j][dd], MIN_VARIANCE);
                float iv = 1.0f / v;
                ivF[j][dd] = iv;
                muIvF[j][dd] = means[j][dd] * iv;
                msiv += means[j][dd] * muIvF[j][dd];
                logDet += Math.log(v);
            }
            muSqIvF[j] = msiv;
            logConst[j] = Math.log(weights[j] + 1e-300) - 0.5 * (d * LOG_2PI + logDet);
        }

        // Precompute x² for all batch samples
        float[][] batchSq = new float[m][d];
        for (int i = 0; i < m; i++) {
            for (int dd = 0; dd < d; dd++) {
                batchSq[i][dd] = batch[i][dd] * batch[i][dd];
            }
        }

        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            double maxLog = Double.NEGATIVE_INFINITY;
            double[] lr = new double[k];
            for (int j = 0; j < k; j++) {
                float t1 = VectorUtil.dotProduct(batchSq[i], ivF[j]);
                float t2 = VectorUtil.dotProduct(batch[i], muIvF[j]);
                lr[j] = logConst[j] - 0.5 * (t1 - 2.0 * t2 + muSqIvF[j]);
                if (lr[j] > maxLog) maxLog = lr[j];
            }
            double sum = 0;
            for (int j = 0; j < k; j++) sum += Math.exp(lr[j] - maxLog);
            result[i] = maxLog + Math.log(sum);
        }
        return result;
    }

    private static double squaredEuclidean(float[] a, float[] b) {
        return VectorUtil.squareDistance(a, b);
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
        List<float[]> centroids = new ArrayList<>(means.length);
        for (float[] mean : means) {
            centroids.add(Arrays.copyOf(mean, mean.length));
        }
        return centroids;
    }

    /**
     * Computes overlap as 1 minus the minimum per-component Bhattacharyya
     * coefficient, approximating the full GMM–GMM overlap via a cross-product sum.
     */
    @Override
    public float computeOverlap(GeometricCoreset other) {
        if (other instanceof GmmCoreset otherGmm) {
            return computeGmmOverlap(otherGmm);
        }
        // Fallback: centroid distance proxy
        return computeCentroidProxyOverlap(other);
    }

    private float computeGmmOverlap(GmmCoreset other) {
        int k1 = weights.length;
        int k2 = other.weights.length;
        double bc = 0;
        for (int i = 0; i < k1; i++) {
            for (int j = 0; j < k2; j++) {
                double componentBC = bhattacharyyaCoeffDiagGaussian(means[i], variances[i], other.means[j], other.variances[j]);
                bc += Math.sqrt((double) weights[i] * other.weights[j]) * componentBC;
            }
        }
        return (float) Math.min(1.0, Math.max(0.0, bc));
    }

    /**
     * Bhattacharyya coefficient between two diagonal Gaussians.
     * BC = exp(-1/8 * (mu1 - mu2)^T * Sigma^{-1} * (mu1 - mu2) - 1/2 * log(det(Sigma) / sqrt(det(Sigma1)*det(Sigma2))))
     * where Sigma = (Sigma1 + Sigma2) / 2 (element-wise for diagonal case).
     */
    private static double bhattacharyyaCoeffDiagGaussian(float[] mu1, float[] var1, float[] mu2, float[] var2) {
        int d = mu1.length;
        double mahal = 0;
        double logDetRatio = 0;
        for (int i = 0; i < d; i++) {
            double v1 = Math.max(var1[i], MIN_VARIANCE);
            double v2 = Math.max(var2[i], MIN_VARIANCE);
            double avgVar = 0.5 * (v1 + v2);
            double diff = mu1[i] - mu2[i];
            mahal += diff * diff / avgVar;
            logDetRatio += Math.log(avgVar) - 0.5 * Math.log(v1) - 0.5 * Math.log(v2);
        }
        return Math.exp(-0.125 * mahal - 0.5 * logDetRatio);
    }

    @Override
    public double scoreSample(float[] embedding) {
        return logLikelihood(embedding);
    }

    @Override
    public double[] batchScoreSamples(float[][] batch) {
        return batchLogLikelihood(batch);
    }

    /** Centroid-distance proxy overlap for cross-type comparisons. */
    private float computeCentroidProxyOverlap(GeometricCoreset other) {
        List<float[]> myCentroids = getCentroids();
        List<float[]> otherCentroids = other.getCentroids();
        double maxSim = 0;
        for (float[] c1 : myCentroids) {
            for (float[] c2 : otherCentroids) {
                double sim = cosineSimilarity(c1, c2);
                if (sim > maxSim) {
                    maxSim = sim;
                }
            }
        }
        return (float) maxSim;
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            normA += (double) a[i] * a[i];
            normB += (double) b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom < 1e-12 ? 0 : dot / denom;
    }

    /**
     * Generates boundary-aware negatives by sampling from the GMM component
     * closest to {@code targetCentroid} and projecting each sample to the
     * component boundary offset by {@code margin}.
     */
    @Override
    public List<float[]> sampleBoundaryNegatives(float[] targetCentroid, int count, float margin) {
        Random rng = new Random();
        List<float[]> negatives = new ArrayList<>(count);

        // Find the component closest to the target centroid
        int nearest = nearestComponent(targetCentroid);

        for (int s = 0; s < count; s++) {
            // Sample from the nearest component
            float[] sample = sampleFromComponent(nearest, rng);
            // Project to the boundary: interpolate between component mean and
            // target, stopping at distance margin outside the component mean
            float[] boundary = projectToBoundary(sample, means[nearest], targetCentroid, margin);
            negatives.add(boundary);
        }
        return negatives;
    }

    private int nearestComponent(float[] target) {
        int nearest = 0;
        double minDist = Double.MAX_VALUE;
        for (int j = 0; j < means.length; j++) {
            double d = squaredEuclidean(means[j], target);
            if (d < minDist) {
                minDist = d;
                nearest = j;
            }
        }
        return nearest;
    }

    /** Samples from a diagonal Gaussian component using the Box-Muller transform. */
    private float[] sampleFromComponent(int j, Random rng) {
        float[] sample = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            double std = Math.sqrt(Math.max(variances[j][d], MIN_VARIANCE));
            sample[d] = (float) (means[j][d] + rng.nextGaussian() * std);
        }
        return sample;
    }

    /**
     * Moves {@code sample} outward from {@code componentMean} in the direction
     * of {@code targetCentroid} by {@code margin} units.
     */
    private static float[] projectToBoundary(float[] sample, float[] componentMean, float[] targetCentroid, float margin) {
        int d = sample.length;
        // Direction from component mean toward target centroid
        float[] dir = new float[d];
        double norm = 0;
        for (int i = 0; i < d; i++) {
            dir[i] = targetCentroid[i] - componentMean[i];
            norm += (double) dir[i] * dir[i];
        }
        if (norm < 1e-12) {
            return Arrays.copyOf(sample, d);
        }
        norm = Math.sqrt(norm);
        float[] boundary = new float[d];
        for (int i = 0; i < d; i++) {
            boundary[i] = sample[i] + (float) (margin * dir[i] / norm);
        }
        return boundary;
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
        int k = weights.length;
        out.writeVInt(k);
        for (float w : weights) {
            out.writeFloat(w);
        }
        for (float[] row : means) {
            for (float v : row) {
                out.writeFloat(v);
            }
        }
        for (float[] row : variances) {
            for (float v : row) {
                out.writeFloat(v);
            }
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("type", TYPE);
        builder.field("dimension", dimension);
        builder.field("num_components", weights.length);
        builder.startArray("components");
        for (int j = 0; j < weights.length; j++) {
            builder.startObject();
            builder.field("weight", weights[j]);
            builder.array("mean", means[j]);
            builder.array("variance", variances[j]);
            builder.endObject();
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    public float[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public float[][] getMeans() {
        float[][] copy = new float[means.length][];
        for (int i = 0; i < means.length; i++) {
            copy[i] = Arrays.copyOf(means[i], means[i].length);
        }
        return copy;
    }

    public float[][] getVariances() {
        float[][] copy = new float[variances.length][];
        for (int i = 0; i < variances.length; i++) {
            copy[i] = Arrays.copyOf(variances[i], variances[i].length);
        }
        return copy;
    }
}
