/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;

import java.io.IOException;

/**
 * Quantizes each centroid as two independent halves: a prefix half and a suffix half.
 *
 * <p>Each half is treated as its own vector for the purpose of optimized scalar quantization,
 * meaning it gets its own {@code lowerInterval}, {@code upperInterval}, {@code quantizedComponentSum}
 * and {@code additionalCorrection}. This is what lets us score the prefix region of the centroid
 * file in isolation later on, without having to read suffix bytes.
 *
 * <p>The split point comes from {@link PrefixLayout}. The supplied {@code globalCentroid} is split
 * the same way so each half uses the matching reference vector during quantization.
 *
 * <p>Both views are single-pass iterators over the same underlying {@link CentroidSupplier}.
 * They are independent: callers typically iterate the prefix view to write the prefix region,
 * then iterate the suffix view to write the suffix region.
 *
 * <p>When {@link PrefixLayout#isEnabled(int)} is {@code false} the suffix half is empty.
 * In that case {@link #suffixView()} returns {@code null} and {@link #prefixView()} yields the
 * full-dimensional quantization (equivalent to the non-split path).
 */
public final class PrefixQuantizedCentroids {

    private final HalfView prefixView;
    private final HalfView suffixView;

    public PrefixQuantizedCentroids(
        CentroidSupplier supplier,
        int dimension,
        OptimizedScalarQuantizer quantizer,
        float[] globalCentroid
    ) {
        if (globalCentroid.length != dimension) {
            throw new IllegalArgumentException(
                "globalCentroid length [" + globalCentroid.length + "] does not match dimension [" + dimension + "]"
            );
        }
        final int prefixLen = PrefixLayout.prefixLength(dimension);
        final int suffixLen = PrefixLayout.suffixLength(dimension);

        final float[] globalPrefix = new float[prefixLen];
        PrefixLayout.copyPrefix(globalCentroid, globalPrefix);
        this.prefixView = new HalfView(supplier, quantizer, 0, prefixLen, globalPrefix);

        if (suffixLen == 0) {
            this.suffixView = null;
        } else {
            final float[] globalSuffix = new float[suffixLen];
            PrefixLayout.copySuffix(globalCentroid, globalSuffix);
            this.suffixView = new HalfView(supplier, quantizer, prefixLen, suffixLen, globalSuffix);
        }
    }

    /**
     * Iterator over the prefix half of every centroid, quantized as a standalone vector.
     * When the layout has no split (small dimension), this iterates the full vector.
     */
    public QuantizedVectorValues prefixView() {
        return prefixView;
    }

    /**
     * Iterator over the suffix half of every centroid, quantized as a standalone vector.
     * Returns {@code null} when the layout has no split, i.e. when there is no suffix to write.
     */
    public QuantizedVectorValues suffixView() {
        return suffixView;
    }

    /**
     * Single-pass quantized view over one fixed slice ({@code offset .. offset+length}) of each
     * centroid produced by the underlying supplier.
     */
    private static final class HalfView implements QuantizedVectorValues {
        private final CentroidSupplier supplier;
        private final OptimizedScalarQuantizer quantizer;
        private final int offset;
        private final int length;
        private final float[] globalHalf;
        private final float[] sliceScratch;
        private final float[] residualScratch;
        private final int[] quantizedIntScratch;
        private final byte[] quantizedBytes;
        private int currOrd = -1;
        private OptimizedScalarQuantizer.QuantizationResult corrections;

        HalfView(CentroidSupplier supplier, OptimizedScalarQuantizer quantizer, int offset, int length, float[] globalHalf) {
            this.supplier = supplier;
            this.quantizer = quantizer;
            this.offset = offset;
            this.length = length;
            this.globalHalf = globalHalf;
            this.sliceScratch = new float[length];
            this.residualScratch = new float[length];
            this.quantizedIntScratch = new int[length];
            this.quantizedBytes = new byte[length];
        }

        @Override
        public int count() {
            return supplier.size();
        }

        @Override
        public byte[] next() throws IOException {
            if (currOrd >= count() - 1) {
                throw new IllegalStateException("No more vectors to read, current ord: " + currOrd + ", count: " + count());
            }
            currOrd++;
            float[] full = supplier.centroid(currOrd);
            System.arraycopy(full, offset, sliceScratch, 0, length);
            corrections = quantizer.scalarQuantize(sliceScratch, residualScratch, quantizedIntScratch, (byte) 7, globalHalf);
            for (int i = 0; i < length; i++) {
                quantizedBytes[i] = (byte) quantizedIntScratch[i];
            }
            return quantizedBytes;
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            return corrections;
        }
    }
}

