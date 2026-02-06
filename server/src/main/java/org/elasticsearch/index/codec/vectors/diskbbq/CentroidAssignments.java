/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

public record CentroidAssignments(
    int numCentroids,
    float[][] centroids,
    int[] assignments,
    int[] overspillAssignments,
    float[] globalCentroid
) {

    public CentroidAssignments(int dims, float[][] centroids, int[] assignments, int[] overspillAssignments) {
        this(centroids.length, centroids, assignments, overspillAssignments, computeGlobalCentroid(dims, centroids));
        assert assignments.length == overspillAssignments.length || overspillAssignments.length == 0
            : "assignments and overspillAssignments must have the same length";

    }

    public static CentroidAssignments create(
        int dims,
        float[][] centroids,
        int[] assignments,
        int[] overspillAssignments,
        float[] globalCentroid
    ) {
        assert assignments.length == overspillAssignments.length || overspillAssignments.length == 0
            : "assignments and overspillAssignments must have the same length";
        return new CentroidAssignments(centroids.length, centroids, assignments, overspillAssignments, globalCentroid);
    }

    public static float[] computeGlobalCentroid(int dims, float[][] centroids) {
        final float[] globalCentroid = new float[dims];
        // TODO: push this logic into vector util?
        for (float[] centroid : centroids) {
            assert centroid.length == dims;
            for (int j = 0; j < centroid.length; j++) {
                globalCentroid[j] += centroid[j];
            }
        }
        for (int j = 0; j < globalCentroid.length; j++) {
            globalCentroid[j] /= centroids.length;
        }
        return globalCentroid;
    }

    public static float[] computeWeightedGlobalCentroid(int dims, float[][] centroids, int[] assignments) {
        if (centroids.length == 0) {
            return new float[dims];
        }
        final int[] centroidCounts = new int[centroids.length];
        for (int assignment : assignments) {
            centroidCounts[assignment]++;
        }
        long total = assignments.length;
        if (total == 0) {
            return computeGlobalCentroid(dims, centroids);
        }
        final float[] globalCentroid = new float[dims];
        for (int i = 0; i < centroids.length; i++) {
            float[] centroid = centroids[i];
            assert centroid.length == dims;
            int weight = centroidCounts[i];
            if (weight == 0) {
                continue;
            }
            for (int j = 0; j < centroid.length; j++) {
                globalCentroid[j] += centroid[j] * weight;
            }
        }
        float invTotal = 1f / total;
        for (int j = 0; j < globalCentroid.length; j++) {
            globalCentroid[j] *= invTotal;
        }
        return globalCentroid;
    }
}
