/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.elasticsearch.test.ESTestCase;

public class CentroidAssignmentsTests extends ESTestCase {

    public void testWeightedGlobalCentroidSkewedAssignments() {
        float[][] centroids = new float[][] { new float[] { 0f, 0f }, new float[] { 10f, 0f } };
        int[] assignments = new int[101];
        for (int i = 0; i < 100; i++) {
            assignments[i] = 0;
        }
        assignments[100] = 1;

        float[] unweighted = CentroidAssignments.computeGlobalCentroid(2, centroids);
        float[] weighted = CentroidAssignments.computeWeightedGlobalCentroid(2, centroids, assignments);

        assertEquals(5f, unweighted[0], 1e-6f);
        assertEquals(0f, unweighted[1], 1e-6f);

        assertEquals(10f / 101f, weighted[0], 1e-6f);
        assertEquals(0f, weighted[1], 1e-6f);
    }
}
