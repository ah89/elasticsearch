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
 * k-center coreset for a single continual-learning domain.
 *
 * <p>The domain is summarised as {@code k} centre points with associated covering
 * radii constructed via the farthest-first traversal (Gonzalez) algorithm.  This
 * provides a 2-approximation to the optimal k-center clustering and guarantees that
 * every embedding in the domain lies within distance {@code max(radii)} of its
 * nearest centre.
 *
 * <p>Boundary-aware negatives are sampled from annuli of thickness {@code margin}
 * just outside each covering ball, biased toward the centre closest to the target
 * domain centroid.
 */
public class KCenterCoreset implements GeometricCoreset {

    public static final String TYPE = "k_center";

    private final int dimension;
    /** Centre points selected by farthest-first traversal, shape [k][d]. */
    private final float[][] centers;
    /** Covering radius for each centre (max distance to any assigned point). */
    private final float[] radii;

    /**
     * Constructs a k-center coreset from pre-computed centres and radii.
     *
     * @param dimension embedding dimensionality
     * @param centers   coreset centre points, shape [k][d]
     * @param radii     covering radius per centre, length k
     */
    public KCenterCoreset(int dimension, float[][] centers, float[] radii) {
        this.dimension = dimension;
        this.centers = centers;
        this.radii = radii;
    }

    /** Deserialisation constructor. */
    public KCenterCoreset(StreamInput in) throws IOException {
        this.dimension = in.readVInt();
        int k = in.readVInt();
        this.centers = new float[k][dimension];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimension; j++) {
                this.centers[i][j] = in.readFloat();
            }
        }
        this.radii = new float[k];
        for (int i = 0; i < k; i++) {
            this.radii[i] = in.readFloat();
        }
    }

    // -------------------------------------------------------------------------
    // Factory: farthest-first traversal
    // -------------------------------------------------------------------------

    /**
     * Builds a k-center coreset using greedy farthest-first traversal.
     *
     * <p>Time complexity: {@code O(n * k * d)}.  For large {@code n}, subsample
     * before calling (see the {@code +Sampling} configuration in ContLoRA).
     *
     * @param embeddings the embedding vectors to summarise
     * @param k          number of centre points
     * @param rng        random source for the initial centre selection
     * @return fitted {@link KCenterCoreset}
     */
    /**
     * Builds a k-center coreset using greedy farthest-first traversal with
     * inline assignment tracking.
     *
     * <p>The covering radii are computed during the traversal itself (each point's
     * minimum distance to its nearest centre is already maintained in {@code minDist}).
     * This eliminates the separate {@code O(n * k * d)} radii computation pass,
     * reducing total time from {@code O(n * k * d)} to {@code O(n * k * d)} with a
     * constant factor of 1 instead of 2.
     */
    public static KCenterCoreset fit(float[][] embeddings, int k, Random rng) {
        if (embeddings.length == 0) {
            throw new IllegalArgumentException("Cannot fit k-center coreset to empty embedding set");
        }
        int n = embeddings.length;
        int d = embeddings[0].length;
        k = Math.min(k, n);

        int[] centerIndices = new int[k];
        float[] minDist = new float[n];
        int[] assignment = new int[n]; // which centre each point is assigned to

        // Pick first centre at random
        centerIndices[0] = rng.nextInt(n);
        float[] firstCenter = embeddings[centerIndices[0]];
        for (int i = 0; i < n; i++) {
            minDist[i] = euclideanDistance(embeddings[i], firstCenter);
            assignment[i] = 0;
        }

        for (int c = 1; c < k; c++) {
            // Next centre = farthest point from all current centres
            int farthest = 0;
            float maxDist = -1;
            for (int i = 0; i < n; i++) {
                if (minDist[i] > maxDist) {
                    maxDist = minDist[i];
                    farthest = i;
                }
            }
            centerIndices[c] = farthest;
            // Update min distances and assignments
            float[] newCenter = embeddings[farthest];
            for (int i = 0; i < n; i++) {
                float dist = euclideanDistance(embeddings[i], newCenter);
                if (dist < minDist[i]) {
                    minDist[i] = dist;
                    assignment[i] = c;
                }
            }
        }

        // Build centre array
        float[][] centers = new float[k][d];
        for (int c = 0; c < k; c++) {
            centers[c] = Arrays.copyOf(embeddings[centerIndices[c]], d);
        }

        // Compute radii from tracked assignments — no re-scan needed
        float[] radii = new float[k];
        for (int i = 0; i < n; i++) {
            int c = assignment[i];
            if (minDist[i] > radii[c]) {
                radii[c] = minDist[i];
            }
        }

        return new KCenterCoreset(d, centers, radii);
    }

    private static float euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
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
        List<float[]> centroids = new ArrayList<>(centers.length);
        for (float[] center : centers) {
            centroids.add(Arrays.copyOf(center, center.length));
        }
        return centroids;
    }

    /**
     * Overlap as the fraction of centre pairs whose covering balls intersect,
     * weighted by the smaller of the two radii.
     */
    @Override
    public float computeOverlap(GeometricCoreset other) {
        if (other instanceof KCenterCoreset otherKC) {
            return computeKCenterOverlap(otherKC);
        }
        return computeCentroidProxyOverlap(other);
    }

    private float computeKCenterOverlap(KCenterCoreset other) {
        int k1 = centers.length;
        int k2 = other.centers.length;
        double totalOverlap = 0;
        double totalWeight = 0;
        for (int i = 0; i < k1; i++) {
            for (int j = 0; j < k2; j++) {
                float dist = euclideanDistance(centers[i], other.centers[j]);
                float combinedRadius = radii[i] + other.radii[j];
                if (dist < combinedRadius) {
                    double weight = (double) radii[i] * other.radii[j];
                    double overlap = 1.0 - dist / Math.max(combinedRadius, 1e-6);
                    totalOverlap += weight * overlap;
                }
                totalWeight += (double) radii[i] * other.radii[j];
            }
        }
        if (totalWeight < 1e-12) {
            return 0.0f;
        }
        return (float) Math.min(1.0, totalOverlap / totalWeight);
    }

    /**
     * Returns the negative min-distance to any centre (higher = closer = more familiar).
     */
    @Override
    public double scoreSample(float[] embedding) {
        float minDist = Float.MAX_VALUE;
        for (float[] center : centers) {
            float dist = euclideanDistance(embedding, center);
            if (dist < minDist) {
                minDist = dist;
            }
        }
        return -minDist;
    }

    @Override
    public double[] batchScoreSamples(float[][] batch) {
        double[] scores = new double[batch.length];
        for (int i = 0; i < batch.length; i++) {
            float minDist = Float.MAX_VALUE;
            for (float[] center : centers) {
                float dist = euclideanDistance(batch[i], center);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            scores[i] = -minDist;
        }
        return scores;
    }

    private float computeCentroidProxyOverlap(GeometricCoreset other) {
        List<float[]> otherCentroids = other.getCentroids();
        double maxSim = 0;
        for (float[] c1 : centers) {
            for (float[] c2 : otherCentroids) {
                double dist = euclideanDistance(c1, c2);
                // Convert distance to similarity: 1 / (1 + dist)
                double sim = 1.0 / (1.0 + dist);
                if (sim > maxSim) {
                    maxSim = sim;
                }
            }
        }
        return (float) maxSim;
    }

    /**
     * Samples boundary negatives from the annulus {@code (radius, radius + margin]}
     * of the centre closest to {@code targetCentroid}.
     */
    @Override
    public List<float[]> sampleBoundaryNegatives(float[] targetCentroid, int count, float margin) {
        Random rng = new Random();
        List<float[]> negatives = new ArrayList<>(count);

        int nearestCenter = findNearestCenter(targetCentroid);
        float[] center = centers[nearestCenter];
        float innerRadius = radii[nearestCenter];

        for (int s = 0; s < count; s++) {
            // Sample a random direction
            float[] direction = randomUnitVector(dimension, rng);
            // Place the sample at innerRadius + margin in that direction
            float sampleRadius = innerRadius + margin * (0.5f + rng.nextFloat() * 0.5f);
            float[] sample = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                sample[i] = center[i] + sampleRadius * direction[i];
            }
            negatives.add(sample);
        }
        return negatives;
    }

    private int findNearestCenter(float[] query) {
        int nearest = 0;
        float minDist = Float.MAX_VALUE;
        for (int c = 0; c < centers.length; c++) {
            float dist = euclideanDistance(centers[c], query);
            if (dist < minDist) {
                minDist = dist;
                nearest = c;
            }
        }
        return nearest;
    }

    /** Generates a uniformly random unit vector in R^d via Gaussian normalisation. */
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
        int k = centers.length;
        out.writeVInt(k);
        for (float[] center : centers) {
            for (float v : center) {
                out.writeFloat(v);
            }
        }
        for (float r : radii) {
            out.writeFloat(r);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("type", TYPE);
        builder.field("dimension", dimension);
        builder.field("num_centers", centers.length);
        builder.field("max_radius", maxRadius());
        builder.startArray("centers");
        for (int c = 0; c < centers.length; c++) {
            builder.startObject();
            builder.array("centroid", centers[c]);
            builder.field("radius", radii[c]);
            builder.endObject();
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    /** Returns the maximum covering radius across all centres. */
    public float maxRadius() {
        float max = 0;
        for (float r : radii) {
            if (r > max) {
                max = r;
            }
        }
        return max;
    }

    public float[][] getCenters() {
        float[][] copy = new float[centers.length][];
        for (int i = 0; i < centers.length; i++) {
            copy[i] = Arrays.copyOf(centers[i], centers[i].length);
        }
        return copy;
    }

    public float[] getRadii() {
        return Arrays.copyOf(radii, radii.length);
    }
}
