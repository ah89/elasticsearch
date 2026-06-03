/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

public class PrefixLayout {
    public static final int MIN_DIMENSION = 128;

    public static final int BLOCK_SIZE = 64;

    private static final float PREFIX_RATIO = 0.5f;

    private PrefixLayout() {}

    public static int prefixLength(int dimension) {
        if (dimension < MIN_DIMENSION) {
            return dimension;
        }
        int raw = Math.round(dimension * PREFIX_RATIO);
        int rounded = ((raw + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        return Math.min(rounded, dimension);
    }

    public static boolean isEnabled(int dimension) {
        return prefixLength(dimension) < dimension;
    }

    public static int suffixLength(int dimension) {
        return dimension - prefixLength(dimension);
    }

    public static void copyPrefix(float[] vector, float[] out) {
        int expected = prefixLength(vector.length);
        if (out.length != expected) {
            throw new IllegalArgumentException("out length [" + out.length + "] does not match prefix length [" + expected + "]");
        }
        System.arraycopy(vector, 0, out, 0, expected);
    }

    public static void copySuffix(float[] vector, float[] out) {
        int expected = suffixLength(vector.length);
        if (out.length != expected) {
            throw new IllegalArgumentException("out length [" + out.length + "] does not match suffix length [" + expected + "]");
        }
        System.arraycopy(vector, prefixLength(vector.length), out, 0, expected);
    }
}
