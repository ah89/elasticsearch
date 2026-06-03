/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.test.ESTestCase;

public class PrefixQuantizedCentroidsTests extends ESTestCase {

    public void testPrefixViewMatchesStandaloneQuantization() throws Exception {
        int dim = 512;
        int numCentroids = 4;
        float[][] centroids = randomCentroids(numCentroids, dim);
        float[] global = randomFloatVector(dim);

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        PrefixQuantizedCentroids splitQuantized = new PrefixQuantizedCentroids(
            CentroidSupplier.fromArray(centroids, KMeansResult.emptyFloat(), dim),
            dim,
            quantizer,
            global
        );

        int prefixLen = PrefixLayout.prefixLength(dim);
        float[] globalPrefix = new float[prefixLen];
        PrefixLayout.copyPrefix(global, globalPrefix);

        QuantizedVectorValues prefixView = splitQuantized.prefixView();
        assertEquals(numCentroids, prefixView.count());

        for (int i = 0; i < numCentroids; i++) {
            byte[] viewBytes = prefixView.next().clone();
            OptimizedScalarQuantizer.QuantizationResult viewCorr = prefixView.getCorrections();

            float[] standalonePrefix = new float[prefixLen];
            PrefixLayout.copyPrefix(centroids[i], standalonePrefix);
            int[] expectedInts = new int[prefixLen];
            OptimizedScalarQuantizer.QuantizationResult expectedCorr = quantizer.scalarQuantize(
                standalonePrefix,
                new float[prefixLen],
                expectedInts,
                (byte) 7,
                globalPrefix
            );
            byte[] expectedBytes = new byte[prefixLen];
            for (int j = 0; j < prefixLen; j++) {
                expectedBytes[j] = (byte) expectedInts[j];
            }

            assertArrayEquals("centroid " + i + " prefix bytes mismatch", expectedBytes, viewBytes);
            assertEquals(expectedCorr.lowerInterval(), viewCorr.lowerInterval(), 0.0f);
            assertEquals(expectedCorr.upperInterval(), viewCorr.upperInterval(), 0.0f);
            assertEquals(expectedCorr.quantizedComponentSum(), viewCorr.quantizedComponentSum());
            assertEquals(expectedCorr.additionalCorrection(), viewCorr.additionalCorrection(), 0.0f);
        }
    }

    public void testSuffixViewMatchesStandaloneQuantization() throws Exception {
        int dim = 768;
        int numCentroids = 3;
        float[][] centroids = randomCentroids(numCentroids, dim);
        float[] global = randomFloatVector(dim);

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        PrefixQuantizedCentroids splitQuantized = new PrefixQuantizedCentroids(
            CentroidSupplier.fromArray(centroids, KMeansResult.emptyFloat(), dim),
            dim,
            quantizer,
            global
        );

        int suffixLen = PrefixLayout.suffixLength(dim);
        assertTrue("dim 768 should produce a non-empty suffix", suffixLen > 0);

        float[] globalSuffix = new float[suffixLen];
        PrefixLayout.copySuffix(global, globalSuffix);

        QuantizedVectorValues suffixView = splitQuantized.suffixView();
        assertNotNull(suffixView);
        assertEquals(numCentroids, suffixView.count());

        for (int i = 0; i < numCentroids; i++) {
            byte[] viewBytes = suffixView.next().clone();
            OptimizedScalarQuantizer.QuantizationResult viewCorr = suffixView.getCorrections();

            float[] standaloneSuffix = new float[suffixLen];
            PrefixLayout.copySuffix(centroids[i], standaloneSuffix);
            int[] expectedInts = new int[suffixLen];
            OptimizedScalarQuantizer.QuantizationResult expectedCorr = quantizer.scalarQuantize(
                standaloneSuffix,
                new float[suffixLen],
                expectedInts,
                (byte) 7,
                globalSuffix
            );
            byte[] expectedBytes = new byte[suffixLen];
            for (int j = 0; j < suffixLen; j++) {
                expectedBytes[j] = (byte) expectedInts[j];
            }

            assertArrayEquals("centroid " + i + " suffix bytes mismatch", expectedBytes, viewBytes);
            assertEquals(expectedCorr.lowerInterval(), viewCorr.lowerInterval(), 0.0f);
            assertEquals(expectedCorr.upperInterval(), viewCorr.upperInterval(), 0.0f);
            assertEquals(expectedCorr.quantizedComponentSum(), viewCorr.quantizedComponentSum());
            assertEquals(expectedCorr.additionalCorrection(), viewCorr.additionalCorrection(), 0.0f);
        }
    }

    public void testSmallDimensionDisablesSuffixView() throws Exception {
        int dim = randomIntBetween(1, PrefixLayout.MIN_DIMENSION - 1);
        int numCentroids = 2;
        float[][] centroids = randomCentroids(numCentroids, dim);
        float[] global = randomFloatVector(dim);

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        PrefixQuantizedCentroids splitQuantized = new PrefixQuantizedCentroids(
            CentroidSupplier.fromArray(centroids, KMeansResult.emptyFloat(), dim),
            dim,
            quantizer,
            global
        );

        assertNull("suffix view should be null when split is disabled", splitQuantized.suffixView());

        QuantizedVectorValues prefixView = splitQuantized.prefixView();
        assertEquals(numCentroids, prefixView.count());

        for (int i = 0; i < numCentroids; i++) {
            byte[] viewBytes = prefixView.next().clone();
            assertEquals(dim, viewBytes.length);

            int[] expectedInts = new int[dim];
            quantizer.scalarQuantize(centroids[i], new float[dim], expectedInts, (byte) 7, global);
            byte[] expectedBytes = new byte[dim];
            for (int j = 0; j < dim; j++) {
                expectedBytes[j] = (byte) expectedInts[j];
            }
            assertArrayEquals("centroid " + i + " full-vector bytes mismatch", expectedBytes, viewBytes);
        }
    }

    public void testGlobalCentroidLengthIsValidated() {
        int dim = 512;
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        CentroidSupplier supplier = CentroidSupplier.fromArray(randomCentroids(1, dim), KMeansResult.emptyFloat(), dim);
        float[] wrongLengthGlobal = new float[dim + 1];
        expectThrows(IllegalArgumentException.class, () -> new PrefixQuantizedCentroids(supplier, dim, quantizer, wrongLengthGlobal));
    }

    private float[][] randomCentroids(int n, int dim) {
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = randomFloatVector(dim);
        }
        return out;
    }

    private float[] randomFloatVector(int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = randomFloat();
        }
        return v;
    }
}



