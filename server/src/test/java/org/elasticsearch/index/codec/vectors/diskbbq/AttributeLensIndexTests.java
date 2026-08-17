/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.LongValues;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Random;

/**
 * Checks the three stored numbers against brute-force references, and that the range they imply
 * actually contains the true query-to-sub-centroid similarity — the property the cluster ordering
 * relies on.
 */
public class AttributeLensIndexTests extends ESTestCase {

    private static final int DIM = 24;

    /** Minimal dense FloatVectorValues over an in-memory array. */
    private static FloatVectorValues vectorValues(float[][] vectors) {
        return new FloatVectorValues() {
            @Override
            public float[] vectorValue(int ord) {
                return vectors[ord];
            }

            @Override
            public int dimension() {
                return DIM;
            }

            @Override
            public int size() {
                return vectors.length;
            }

            @Override
            public FloatVectorValues copy() {
                return this;
            }

            @Override
            public DocIndexIterator iterator() {
                return createDenseIterator();
            }
        };
    }

    public void testStoredNumbersMatchBruteForce() throws IOException {
        final Random random = new Random(7);
        final int numDocs = 400;
        final int numCentroids = 2;
        final float[][] vectors = new float[numDocs][];
        final int[] cluster = new int[numDocs];
        final int[] docValues = new int[numDocs];
        final int[] docValueOffsets = new int[numDocs + 1];
        final float[][] base = new float[numCentroids][DIM];
        base[0][0] = 1f;
        base[1][1] = 1f;
        for (int i = 0; i < numDocs; i++) {
            final int c = i % numCentroids;
            cluster[i] = c;
            final float[] x = new float[DIM];
            for (int j = 0; j < DIM; j++) {
                x[j] = base[c][j] + 0.1f * (float) random.nextGaussian();
            }
            final int value = random.nextInt(2);
            if (value == 1) {
                x[2] += 0.5f;
            }
            vectors[i] = x;
            docValues[i] = value;
            docValueOffsets[i] = i;
        }
        docValueOffsets[numDocs] = numDocs;
        final LongValues centroidOf = new LongValues() {
            @Override
            public long get(long ord) {
                return cluster[(int) ord];
            }
        };
        final AttributeLensIndex lens = AttributeLensIndex.build(
            vectorValues(vectors),
            centroidOf,
            null, // no overspill in these fixtures: every vector is stored only in its owning cluster
            docValues,
            docValueOffsets,
            2,
            numCentroids
        );

        for (int c = 0; c < numCentroids; c++) {
            final float[] mu = new float[DIM];
            final float[] muA = new float[DIM];
            int n = 0;
            int nA = 0;
            for (int i = 0; i < numDocs; i++) {
                if (cluster[i] != c) {
                    continue;
                }
                for (int j = 0; j < DIM; j++) {
                    mu[j] += vectors[i][j];
                }
                n++;
                if (docValues[i] == 1) {
                    for (int j = 0; j < DIM; j++) {
                        muA[j] += vectors[i][j];
                    }
                    nA++;
                }
            }
            double muSq = 0;
            double muASq = 0;
            double dot = 0;
            double deltaSq = 0;
            for (int j = 0; j < DIM; j++) {
                mu[j] /= n;
                muA[j] /= nA;
                muSq += (double) mu[j] * mu[j];
                muASq += (double) muA[j] * muA[j];
                dot += (double) mu[j] * muA[j];
                deltaSq += ((double) muA[j] - mu[j]) * (muA[j] - mu[j]);
            }
            assertEquals("x for cluster " + c, (double) nA / n, lens.proportion(1, c), 1e-6);
            // y and z are checked through the range they imply, below
            assertTrue(deltaSq > 0);
            assertTrue(dot / Math.sqrt(muSq * muASq) <= 1.0 + 1e-6);
        }
    }

    /**
     * The derived range must bracket the truth: for random queries, the exact similarity to the
     * value's sub-centroid has to lie between the nearest and farthest points the stored numbers
     * allow. If it ever falls outside, the ordering is built on a bound that does not hold.
     */
    public void testDerivedRangeContainsTheTruth() throws IOException {
        final Random random = new Random(19);
        final int numDocs = 1200;
        final int numCentroids = 4;
        final float[][] vectors = new float[numDocs][];
        final int[] cluster = new int[numDocs];
        final int[] docValues = new int[numDocs];
        final int[] docValueOffsets = new int[numDocs + 1];
        for (int i = 0; i < numDocs; i++) {
            final int c = i % numCentroids;
            cluster[i] = c;
            final float[] x = new float[DIM];
            x[c] = 1f;
            for (int j = 0; j < DIM; j++) {
                x[j] += 0.3f * (float) random.nextGaussian();
            }
            vectors[i] = x;
            docValues[i] = random.nextInt(3);
            docValueOffsets[i] = i;
        }
        docValueOffsets[numDocs] = numDocs;
        final LongValues centroidOf = new LongValues() {
            @Override
            public long get(long ord) {
                return cluster[(int) ord];
            }
        };
        final AttributeLensIndex lens = AttributeLensIndex.build(
            vectorValues(vectors),
            centroidOf,
            null, // no overspill in these fixtures: every vector is stored only in its owning cluster
            docValues,
            docValueOffsets,
            3,
            numCentroids
        );

        // exact centroids and sub-centroids for value 1
        final float[][] mu = new float[numCentroids][DIM];
        final float[][] muA = new float[numCentroids][DIM];
        final int[] n = new int[numCentroids];
        final int[] nA = new int[numCentroids];
        for (int i = 0; i < numDocs; i++) {
            final int c = cluster[i];
            for (int j = 0; j < DIM; j++) {
                mu[c][j] += vectors[i][j];
            }
            n[c]++;
            if (docValues[i] == 1) {
                for (int j = 0; j < DIM; j++) {
                    muA[c][j] += vectors[i][j];
                }
                nA[c]++;
            }
        }
        for (int c = 0; c < numCentroids; c++) {
            for (int j = 0; j < DIM; j++) {
                mu[c][j] /= n[c];
                muA[c][j] /= nA[c];
            }
        }

        final float[] mult = new float[numCentroids];
        final float[] add = new float[numCentroids];
        final float[] perp = new float[numCentroids];
        final float[] invNorm = new float[numCentroids];
        for (int trial = 0; trial < 40; trial++) {
            final float[] q = new float[DIM];
            double norm = 0;
            for (int j = 0; j < DIM; j++) {
                q[j] = (float) random.nextGaussian();
                norm += (double) q[j] * q[j];
            }
            final float inv = (float) (1.0 / Math.sqrt(norm));
            for (int j = 0; j < DIM; j++) {
                q[j] *= inv;
            }
            assertEquals(numCentroids, lens.scatterCoefficients(1, 0f, mult, add, null, perp, invNorm));
            for (int c = 0; c < numCentroids; c++) {
                double sim = 0;
                double truth = 0;
                for (int j = 0; j < DIM; j++) {
                    sim += (double) q[j] * mu[c][j];
                    truth += (double) q[j] * muA[c][j];
                }
                final double cosine = Math.clamp(sim * invNorm[c], -1.0, 1.0);
                final double halfWidth = perp[c] * Math.sqrt(Math.max(0, 1 - cosine * cosine));
                final double mid = mult[c] * sim;
                assertTrue(
                    "truth " + truth + " outside [" + (mid - halfWidth) + ", " + (mid + halfWidth) + "] for cluster " + c,
                    truth >= mid - halfWidth - 1e-4 && truth <= mid + halfWidth + 1e-4
                );
            }
        }
    }
}
