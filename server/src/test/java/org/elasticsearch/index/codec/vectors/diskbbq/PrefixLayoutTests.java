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

public class PrefixLayoutTests extends ESTestCase {

    public void testSmallVectorDisablesSplit() {
        // below MIN_DIMENSION the whole vector is the prefix (no split)
        int dim = randomIntBetween(1, PrefixLayout.MIN_DIMENSION - 1);
        assertEquals(dim, PrefixLayout.prefixLength(dim));
        assertEquals(0, PrefixLayout.suffixLength(dim));
        assertFalse(PrefixLayout.isEnabled(dim));
    }

    public void testLargeVectorSplitsRoughlyInHalf() {
        // at MIN_DIMENSION or above, prefix is ~50% of the vector
        int dim = randomIntBetween(PrefixLayout.MIN_DIMENSION, 4096);
        int prefix = PrefixLayout.prefixLength(dim);
        assertTrue("prefix should be less than full vector", prefix < dim);
        assertTrue("prefix should be at least one block", prefix >= PrefixLayout.BLOCK_SIZE);
        assertEquals(dim, prefix + PrefixLayout.suffixLength(dim));
        assertTrue(PrefixLayout.isEnabled(dim));
    }

    public void testPrefixAlignedToBlockMultiple() {
        // prefix length is always a multiple of BLOCK_SIZE when split is enabled
        int dim = randomIntBetween(PrefixLayout.MIN_DIMENSION, 4096);
        assertEquals(0, PrefixLayout.prefixLength(dim) % PrefixLayout.BLOCK_SIZE);
    }

    public void testCopyPrefixCopiesLeadingDims() {
        int dim = 512;
        float[] src = randomFloatVector(dim);
        float[] prefix = new float[PrefixLayout.prefixLength(dim)];
        PrefixLayout.copyPrefix(src, prefix);
        for (int i = 0; i < prefix.length; i++) {
            assertEquals(src[i], prefix[i], 0.0f);
        }
    }

    public void testCopySuffixCopiesTrailingDims() {
        int dim = 512;
        float[] src = randomFloatVector(dim);
        float[] suffix = new float[PrefixLayout.suffixLength(dim)];
        PrefixLayout.copySuffix(src, suffix);
        int prefixLen = PrefixLayout.prefixLength(dim);
        for (int i = 0; i < suffix.length; i++) {
            assertEquals(src[prefixLen + i], suffix[i], 0.0f);
        }
    }

    public void testCopyPrefixRejectsWrongOutputSize() {
        int dim = 512;
        float[] src = randomFloatVector(dim);
        float[] wrongSize = new float[PrefixLayout.prefixLength(dim) + 1];
        expectThrows(IllegalArgumentException.class, () -> PrefixLayout.copyPrefix(src, wrongSize));
    }

    public void testCopySuffixRejectsWrongOutputSize() {
        int dim = 512;
        float[] src = randomFloatVector(dim);
        float[] wrongSize = new float[PrefixLayout.suffixLength(dim) + 1];
        expectThrows(IllegalArgumentException.class, () -> PrefixLayout.copySuffix(src, wrongSize));
    }

    private float[] randomFloatVector(int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = randomFloat();
        }
        return v;
    }
}
