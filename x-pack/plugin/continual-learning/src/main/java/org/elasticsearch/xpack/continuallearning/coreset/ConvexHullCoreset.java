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
 * Directional-extrema convex-hull proxy coreset for a single continual-learning domain.
 *
 * <p>An exact convex hull in high-dimensional spaces (e.g.\ d=768) is both
 * computationally prohibitive and has exponentially many vertices.  This
 * implementation approximates the boundary of the domain's convex hull by
 * projecting the embeddings onto {@code k} random unit directions and selecting
 * the point that maximises each projection — the <em>directional extremum</em>.
 * The result is a compact set of boundary-representative points that can be
 * computed in {@code O(k * n * d)} time without any convex-hull library.
 *
 * <p>Overlap between two domains is measured as the fraction of this domain's
 * boundary vertices that fall within the combined covering sphere of the other
 * domain (centroid ± radius).
 *
 * <p>Note: because random-direction extrema in very high dimensions concentrate
 * near the same region of the unit sphere, this coreset type is most useful as a
 * diversity baseline in the ContLoRA RQ4 ablation, not as a production default.
 * Use {@link GmmCoreset} or {@link KCenterCoreset} for production deployments.
 */
public class ConvexHullCoreset implements GeometricCoreset {

    public static final String TYPE = "convex_hull";

    private final int dimension;
    /** Boundary-representative vertex points, shape [k][d]. */
    private final float[][] vertices;
    /** Mean of all embeddings used during fitting. */
    private final float[] centroid;
    /** Maximum distance from centroid to any vertex. */
    private final float radius;

    /**
     * Constructs a convex-hull coreset from pre-computed boundary points.
     *
     * @param dimension embedding dimensionality
     * @param vertices  boundary-representative points, shape [k][d]
     * @param centroid  mean embedding of the domain, length d
     * @param radius    maximum distance from centroid to any vertex
     */
    public ConvexHullCoreset(int dimension, float[][] vertices, float[] centroid, float radius) {
        this.dimension = dimension;
        this.vertices = vertices;
        this.centroid = centroid;
        this.radius = radius;
    }

    /** Deserialisation constructor. */
    public ConvexHullCoreset(StreamInput in) throws IOException {
        this.dimension = in.readVInt();
        int k = in.readVInt();
        this.vertices = new float[k][dimension];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimension; j++) {
                this.vertices[i][j] = in.readFloat();
            }
        }
        this.centroid = new float[dimension];
        for (int j = 0; j < dimension; j++) {
            this.centroid[j] = in.readFloat();
        }
        this.radius = in.readFloat();
    }

    // -------------------------------------------------------------------------
    // Factory: directional-extrema approximation
    // -------------------------------------------------------------------------

    /**
     * Builds a convex-hull proxy coreset by selecting the directional extremum
     * for {@code 2*k} random unit directions, then deduplicating.
     *
     * <p>Time complexity: {@code O(k * n * d)}.
     *
     * @param embeddings the embedding vectors to summarise
     * @param k          target number of boundary vertices
     * @param rng        random source for generating probe directions
     * @return fitted {@link ConvexHullCoreset}
     */
    public static ConvexHullCoreset fit(float[][] embeddings, int k, Random rng) {
        if (embeddings.length == 0) {
            throw new IllegalArgumentException("Cannot fit ConvexHullCoreset to empty embedding set");
        }
        int n = embeddings.length;
        int d = embeddings[0].length;
        k = Math.min(k, n);

        // Generate 2*k random unit directions and find the argmax for each
        int numDirections = k * 2;
        boolean[] selected = new boolean[n];
        int selectedCount = 0;

        for (int dir = 0; dir < numDirections && selectedCount < k; dir++) {
            float[] direction = randomUnitVector(d, rng);
            int best = 0;
            float bestProj = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                float proj = dot(embeddings[i], direction);
                if (proj > bestProj) {
                    bestProj = proj;
                    best = i;
                }
            }
            if (selected[best] == false) {
                selected[best] = true;
                selectedCount++;
            }
        }

        // Fill remaining slots with random points if deduplication left gaps
        if (selectedCount < k) {
            for (int i = 0; i < n && selectedCount < k; i++) {
                if (selected[i] == false) {
                    selected[i] = true;
                    selectedCount++;
                }
            }
        }

        float[][] vertices = new float[selectedCount][d];
        int vi = 0;
        for (int i = 0; i < n; i++) {
            if (selected[i]) {
                vertices[vi++] = Arrays.copyOf(embeddings[i], d);
            }
        }

        // Compute centroid as mean of all embeddings
        float[] centroid = new float[d];
        for (float[] emb : embeddings) {
            for (int j = 0; j < d; j++) {
                centroid[j] += emb[j];
            }
        }
        for (int j = 0; j < d; j++) {
            centroid[j] /= n;
        }

        // Radius = max distance from centroid to any vertex
        float radius = 0;
        for (float[] v : vertices) {
            float dist = euclideanDistance(v, centroid);
            if (dist > radius) {
                radius = dist;
            }
        }

        return new ConvexHullCoreset(d, vertices, centroid, radius);
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
        List<float[]> result = new ArrayList<>(vertices.length + 1);
        result.add(Arrays.copyOf(centroid, dimension));
        for (float[] v : vertices) {
            result.add(Arrays.copyOf(v, dimension));
        }
        return result;
    }

    /**
     * Overlap as the fraction of this domain's boundary vertices that lie within
     * the combined covering sphere (centroid ± (this.radius + other.radius)) of
     * the other domain.
     */
    @Override
    public float computeOverlap(GeometricCoreset other) {
        if (other instanceof ConvexHullCoreset otherCH) {
            return computeConvexHullOverlap(otherCH);
        }
        return computeCentroidProxyOverlap(other);
    }

    private float computeConvexHullOverlap(ConvexHullCoreset other) {
        float combinedRadius = this.radius + other.radius;
        int inside = 0;
        for (float[] v : vertices) {
            float dist = euclideanDistance(v, other.centroid);
            if (dist < combinedRadius) {
                inside++;
            }
        }
        return (float) inside / vertices.length;
    }

    private float computeCentroidProxyOverlap(GeometricCoreset other) {
        List<float[]> otherCentroids = other.getCentroids();
        double maxSim = 0;
        for (float[] c2 : otherCentroids) {
            double dist = euclideanDistance(centroid, c2);
            double sim = 1.0 / (1.0 + dist);
            if (sim > maxSim) {
                maxSim = sim;
            }
        }
        return (float) maxSim;
    }

    /**
     * Generates boundary-aware negatives by placing samples at distance
     * {@code radius + margin} from the centroid, in the direction of
     * {@code targetCentroid}.
     */
    @Override
    public List<float[]> sampleBoundaryNegatives(float[] targetCentroid, int count, float margin) {
        Random rng = new Random();
        List<float[]> negatives = new ArrayList<>(count);

        // Direction from centroid toward target
        float[] dir = new float[dimension];
        double norm = 0;
        for (int i = 0; i < dimension; i++) {
            dir[i] = targetCentroid[i] - centroid[i];
            norm += (double) dir[i] * dir[i];
        }
        norm = Math.sqrt(Math.max(norm, 1e-12));
        for (int i = 0; i < dimension; i++) {
            dir[i] = (float) (dir[i] / norm);
        }

        for (int s = 0; s < count; s++) {
            // Small random perturbation around the boundary direction
            float[] perturbation = randomUnitVector(dimension, rng);
            float sampleRadius = radius + margin * (0.5f + rng.nextFloat() * 0.5f);
            float[] sample = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                // Blend direction (90%) with random perturbation (10%)
                float blended = 0.9f * dir[i] + 0.1f * perturbation[i];
                sample[i] = centroid[i] + sampleRadius * blended;
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
        out.writeVInt(vertices.length);
        for (float[] v : vertices) {
            for (float val : v) {
                out.writeFloat(val);
            }
        }
        for (float val : centroid) {
            out.writeFloat(val);
        }
        out.writeFloat(radius);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("type", TYPE);
        builder.field("dimension", dimension);
        builder.field("num_vertices", vertices.length);
        builder.field("radius", radius);
        builder.array("centroid", centroid);
        builder.startArray("vertices");
        for (float[] v : vertices) {
            builder.startObject();
            builder.array("point", v);
            builder.endObject();
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public float[][] getVertices() {
        float[][] copy = new float[vertices.length][];
        for (int i = 0; i < vertices.length; i++) {
            copy[i] = Arrays.copyOf(vertices[i], vertices[i].length);
        }
        return copy;
    }

    public float[] getDomainCentroid() {
        return Arrays.copyOf(centroid, dimension);
    }

    public float getRadius() {
        return radius;
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
}
