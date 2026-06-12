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
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.test.ESTestCase;

import java.util.Arrays;

import static org.hamcrest.Matchers.lessThan;

public class PrefixSuffixScoreCombinerTests extends ESTestCase {

    private static final float EPS = 1e-6f;
    private static final float SEVEN_BIT_SCALE = 1f / 127f;

    public void testDotProductExactRecombination() {
        // Pick raw centered dot products for prefix and suffix halves; full = sum of halves.
        float rawPrefix = 0.3f;
        float rawSuffix = 0.5f;
        float prefixScore = (1f + rawPrefix) / 2f; // 0.65
        float suffixScore = (1f + rawSuffix) / 2f; // 0.75
        float expectedFull = (1f + rawPrefix + rawSuffix) / 2f; // 0.9
        float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.DOT_PRODUCT, prefixScore, suffixScore);
        assertEquals(expectedFull, combined, EPS);
    }

    public void testCosineMatchesDotProduct() {
        float prefixScore = 0.4f;
        float suffixScore = 0.6f;
        assertEquals(
            PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.DOT_PRODUCT, prefixScore, suffixScore),
            PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.COSINE, prefixScore, suffixScore),
            EPS
        );
    }

    public void testDotProductClampsAtZero() {
        // Two halves that together would imply a negative raw sum < -1 must clamp at 0.
        float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.DOT_PRODUCT, 0.0f, 0.0f);
        assertEquals(0f, combined, 0f);
    }

    public void testEuclideanExactRecombination() {
        float dpSq = 1f; // prefix squared distance
        float dsSq = 3f; // suffix squared distance
        float prefixScore = 1f / (1f + dpSq); // 0.5
        float suffixScore = 1f / (1f + dsSq); // 0.25
        float expectedFull = 1f / (1f + dpSq + dsSq); // 0.2
        float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.EUCLIDEAN, prefixScore, suffixScore);
        assertEquals(expectedFull, combined, EPS);
    }

    public void testEuclideanZeroHalfYieldsZero() {
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.EUCLIDEAN, 0f, 0.5f), 0f);
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.EUCLIDEAN, 0.5f, 0f), 0f);
    }

    public void testMaxInnerProductPositiveSum() {
        // raw_p = 2, raw_s = 3 -> full_raw = 5; scaled: scaleMIP(2)=3, scaleMIP(3)=4, scaleMIP(5)=6
        float combined = PrefixSuffixScoreCombiner.combine(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            VectorUtil.scaleMaxInnerProductScore(2f),
            VectorUtil.scaleMaxInnerProductScore(3f)
        );
        assertEquals(VectorUtil.scaleMaxInnerProductScore(5f), combined, EPS);
    }

    public void testMaxInnerProductMixedSigns() {
        // raw_p = 2, raw_s = -3 -> full_raw = -1; scaled: scaleMIP(2)=3, scaleMIP(-3)=1/4, scaleMIP(-1)=1/2
        float prefixScaled = VectorUtil.scaleMaxInnerProductScore(2f);
        float suffixScaled = VectorUtil.scaleMaxInnerProductScore(-3f);
        float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, prefixScaled, suffixScaled);
        assertEquals(VectorUtil.scaleMaxInnerProductScore(-1f), combined, EPS);
    }

    public void testMaxInnerProductBothNegative() {
        // raw_p = -2, raw_s = -5 -> full_raw = -7
        float prefixScaled = VectorUtil.scaleMaxInnerProductScore(-2f);
        float suffixScaled = VectorUtil.scaleMaxInnerProductScore(-5f);
        float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, prefixScaled, suffixScaled);
        assertEquals(VectorUtil.scaleMaxInnerProductScore(-7f), combined, EPS);
    }

    /** Round-trip the inversion via combine() for randomly chosen raw scalar pairs. */
    public void testMaxInnerProductRandomRoundTrip() {
        for (int i = 0; i < 100; i++) {
            float rawPrefix = randomFloatBetween(-10f, 10f, true);
            float rawSuffix = randomFloatBetween(-10f, 10f, true);
            float prefixScaled = VectorUtil.scaleMaxInnerProductScore(rawPrefix);
            float suffixScaled = VectorUtil.scaleMaxInnerProductScore(rawSuffix);
            float expectedFull = VectorUtil.scaleMaxInnerProductScore(rawPrefix + rawSuffix);
            float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, prefixScaled, suffixScaled);
            assertEquals("rawP=" + rawPrefix + " rawS=" + rawSuffix, expectedFull, combined, 1e-4f);
        }
    }

    public void testDotProductRandomRoundTrip() {
        for (int i = 0; i < 100; i++) {
            // raw in [-1, 1] for valid dot-product scores; sum may fall outside, in which case
            // applyCorrections clamps at 0, so we mirror that here.
            float rawPrefix = randomFloatBetween(-1f, 1f, true);
            float rawSuffix = randomFloatBetween(-1f, 1f, true);
            float prefixScore = (1f + rawPrefix) / 2f;
            float suffixScore = (1f + rawSuffix) / 2f;
            float expectedFull = Math.max((1f + rawPrefix + rawSuffix) / 2f, 0f);
            float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.DOT_PRODUCT, prefixScore, suffixScore);
            assertEquals(expectedFull, combined, EPS);
        }
    }

    public void testEuclideanRandomRoundTrip() {
        for (int i = 0; i < 100; i++) {
            float dpSq = randomFloatBetween(1e-3f, 50f, true);
            float dsSq = randomFloatBetween(1e-3f, 50f, true);
            float prefixScore = 1f / (1f + dpSq);
            float suffixScore = 1f / (1f + dsSq);
            float expectedFull = 1f / (1f + dpSq + dsSq);
            float combined = PrefixSuffixScoreCombiner.combine(VectorSimilarityFunction.EUCLIDEAN, prefixScore, suffixScore);
            assertEquals(expectedFull, combined, 1e-5f);
        }
    }

    /**
     * Degenerate / non-finite inputs must collapse to the worst-possible score (0) rather than the
     * best (1). A bad refinement that promotes garbage above legitimate candidates would silently
     * corrupt ranking; documenting and pinning the fail-safe ensures the guard cannot regress.
     */
    public void testEuclideanGuardsAgainstNonFiniteAndDegenerateInputs() {
        VectorSimilarityFunction sim = VectorSimilarityFunction.EUCLIDEAN;
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(sim, Float.NaN, 0.5f), 0f);
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(sim, 0.5f, Float.NaN), 0f);
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(sim, Float.POSITIVE_INFINITY, 0.5f), 0f);
        // Two halves both > 1 (e.g. from quantization noise) drive denom = 1/p + 1/s - 1 below 1
        // and potentially to or below 0. Either way it is degenerate input.
        assertEquals(0f, PrefixSuffixScoreCombiner.combine(sim, 2f, 2f), 0f);
    }

    /**
     * MAX_IP inversion must not blow up on a non-positive scaled input (which would correspond to a
     * raw score of negative infinity). We return -Infinity from invertMip, and after summing and
     * re-scaling that produces 0; the test pins this so a future refactor cannot regress to NaN.
     */
    public void testMaxInnerProductGuardsAgainstNonPositiveScaledInput() {
        float combined = PrefixSuffixScoreCombiner.combine(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            0f, // would invert to -Infinity
            VectorUtil.scaleMaxInnerProductScore(2f)
        );
        assertFalse("combined must be finite, got " + combined, Float.isNaN(combined));
        assertEquals(0f, combined, 0f);
    }

    /**
     * Ground-truth: drive a real OSQ quantization for full-dim and for the two halves, score each
     * via the same formula {@code ES92Int7VectorsScorer#applyCorrections} uses, then combine. This
     * is the only test that catches a bug where the algebra is self-consistent but does not
     * actually reconstruct what the production scorer produces (e.g. mis-handling of per-half
     * {@code centroidDp} / {@code additionalCorrection}).
     *
     * <p>Vectors are unit-normalised so per-half scores stay in the un-clamped regime of
     * {@code Math.max((1 + score) / 2f, 0)}. The combine formula cannot recover information lost
     * to that floor; in production that only matters for far centroids which never survive top-K
     * refinement, but in this test it would mask the algebra we are trying to validate.
     *
     * <p>Per-half OSQ rounding makes the equality approximate; we assert on average relative error
     * across many trials so quantization noise averages out but a systematic algebra bias (which
     * would show up as huge average error) still fails loudly.
     */
    public void testCombineApproximatesFullOSQScoreDotProduct() {
        assertApproximatesFullOSQScore(VectorSimilarityFunction.DOT_PRODUCT);
    }

    public void testCombineApproximatesFullOSQScoreEuclidean() {
        // EUCLIDEAN scores 1/(1+d^2) are inherently in (0, 1] so the un-clamped regime is
        // automatic; we still use unit vectors for consistency with the DOT_PRODUCT variant.
        assertApproximatesFullOSQScore(VectorSimilarityFunction.EUCLIDEAN);
    }

    private void assertApproximatesFullOSQScore(VectorSimilarityFunction sim) {
        final int trials = 25;
        float totalRel = 0f;
        for (int t = 0; t < trials; t++) {
            // Even half-dims so each half is well-conditioned for OSQ; also matches the scorer's
            // even-dimension requirement.
            int prefixDim = randomIntBetween(32, 128) * 2;
            int suffixDim = randomIntBetween(32, 128) * 2;
            int dim = prefixDim + suffixDim;

            float[] target = randomUnitVector(dim);
            float[] query = randomUnitVector(dim);
            float[] centroid = randomUnitVector(dim);

            float full = osqScore(target, query, centroid, sim);
            float pref = osqScore(
                Arrays.copyOfRange(target, 0, prefixDim),
                Arrays.copyOfRange(query, 0, prefixDim),
                Arrays.copyOfRange(centroid, 0, prefixDim),
                sim
            );
            float suf = osqScore(
                Arrays.copyOfRange(target, prefixDim, dim),
                Arrays.copyOfRange(query, prefixDim, dim),
                Arrays.copyOfRange(centroid, prefixDim, dim),
                sim
            );
            float combined = PrefixSuffixScoreCombiner.combine(sim, pref, suf);

            float denom = Math.max(Math.abs(full), 1e-6f);
            totalRel += Math.abs(combined - full) / denom;
        }
        float avgRel = totalRel / trials;
        // 10% average relative error is well above 7-bit OSQ noise but well below what an
        // off-by-constant algebra bug would produce.
        assertThat("avg relative error for " + sim + " was " + avgRel, avgRel, lessThan(0.10f));
    }

    /**
     * Runs OSQ quantization and the corrected-score formula without needing an {@code IndexInput},
     * so the test stays a pure unit test. Mirrors
     * {@code org.elasticsearch.simdvec.ES92Int7VectorsScorer#applyCorrections}; if the upstream
     * formula ever changes, this duplication surfaces the divergence quickly.
     */
    private static float osqScore(float[] target, float[] query, float[] centroid, VectorSimilarityFunction sim) {
        int dim = target.length;
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(sim);
        float[] residual = new float[dim];
        int[] scratch = new int[dim];

        OptimizedScalarQuantizer.QuantizationResult tRes = quantizer.scalarQuantize(target, residual, scratch, (byte) 7, centroid);
        byte[] tBytes = new byte[dim];
        for (int i = 0; i < dim; i++) {
            tBytes[i] = (byte) scratch[i];
        }
        OptimizedScalarQuantizer.QuantizationResult qRes = quantizer.scalarQuantize(query, residual, scratch, (byte) 7, centroid);
        byte[] qBytes = new byte[dim];
        for (int i = 0; i < dim; i++) {
            qBytes[i] = (byte) scratch[i];
        }
        long rawDot = 0;
        for (int i = 0; i < dim; i++) {
            rawDot += (long) tBytes[i] * qBytes[i];
        }
        float centroidDp = VectorUtil.dotProduct(centroid, centroid);

        float ax = tRes.lowerInterval();
        float lx = (tRes.upperInterval() - ax) * SEVEN_BIT_SCALE;
        float ay = qRes.lowerInterval();
        float ly = (qRes.upperInterval() - ay) * SEVEN_BIT_SCALE;
        float y1 = qRes.quantizedComponentSum();
        float score = ax * ay * dim + ay * lx * tRes.quantizedComponentSum() + ax * ly * y1 + lx * ly * (float) rawDot;
        if (sim == VectorSimilarityFunction.EUCLIDEAN) {
            score = qRes.additionalCorrection() + tRes.additionalCorrection() - 2 * score;
            return Math.max(1f / (1f + score), 0f);
        }
        score += qRes.additionalCorrection() + tRes.additionalCorrection() - centroidDp;
        if (sim == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            return VectorUtil.scaleMaxInnerProductScore(score);
        }
        return Math.max((1f + score) / 2f, 0f);
    }

    private float[] randomUnitVector(int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = randomFloat() * 2f - 1f;
        }
        VectorUtil.l2normalize(v);
        return v;
    }
}
