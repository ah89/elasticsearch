/*
 * Copyright Elasticsearch B.V., the OpenSearch Project and/or its contributors
 * under one or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side Public
 * License v 1"; you may not use this file except in compliance with, at your election,
 * the "Elastic License 2.0", the "GNU Affero General Public License v3.0 only", or
 * the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

/**
 * Combines a score produced by scoring only the prefix bytes of a quantized vector with a score
 * produced by scoring only the suffix bytes into the score that the full-dimensional scorer would
 * have produced, modulo the rounding error introduced by quantising each half independently.
 *
 * <p>The algebra below works because the underlying centered dot product and squared L2 distance
 * are both linear in the dimension partition: if {@code D = P + S} and we split a query/target
 * pair at offset {@code P}, then
 * <ul>
 *   <li>{@code <x, q> = <x[0:P], q[0:P]> + <x[P:D], q[P:D]>}</li>
 *   <li>{@code |x - q|^2 = |x[0:P] - q[0:P]|^2 + |x[P:D] - q[P:D]|^2}</li>
 * </ul>
 * The per-similarity normalisations in {@code ES92Int7VectorsScorer#applyCorrections} are then
 * inverted, summed in the raw domain, and re-applied. Errors come exclusively from each half
 * carrying its own OSQ interval / quantised-sum corrections (and being lossy in different ways);
 * they vanish as quantisation becomes lossless.
 *
 * <p><b>Clamping caveat.</b> {@code applyCorrections} floors DOT_PRODUCT and COSINE scores at 0
 * (via {@code Math.max((1 + score) / 2f, 0)}) and EUCLIDEAN scores at 0 (via
 * {@code Math.max(1 / (1 + score), 0)}). The inversion performed here cannot recover information
 * lost to that floor. In practice this matters only for "far" centroids whose halves saturate;
 * such candidates would not survive top-K refinement anyway, so combine() may return an
 * approximate result for them but will never be relied on for ranking among close candidates.
 * Tests that validate the algebra against full-dim OSQ scores must use unit-normalised vectors
 * (for DOT_PRODUCT / COSINE) so that no half clamps.
 */
public final class PrefixSuffixScoreCombiner {

    private PrefixSuffixScoreCombiner() {}

    /**
     * @param sim similarity used for both prefix and suffix scoring; must match
     * @param prefixScore score produced by scoring only the prefix bytes (output of
     *                    {@code applyCorrections} with the prefix scorer)
     * @param suffixScore score produced by scoring only the suffix bytes
     * @return the combined score, clamped to the per-similarity valid range
     */
    public static float combine(VectorSimilarityFunction sim, float prefixScore, float suffixScore) {
        return switch (sim) {
            case DOT_PRODUCT, COSINE -> Math.max(prefixScore + suffixScore - 0.5f, 0f);
            case EUCLIDEAN -> combineEuclidean(prefixScore, suffixScore);
            case MAXIMUM_INNER_PRODUCT -> VectorUtil.scaleMaxInnerProductScore(invertMip(prefixScore) + invertMip(suffixScore));
        };
    }

    private static float combineEuclidean(float prefixScore, float suffixScore) {
        // Each half is 1 / (1 + d^2_half); combined squared distance is the sum of halves'
        // squared distances. If either half saturated to 0 (infinite distance), or either input
        // is non-finite (NaN / Infinity), the combined score is also 0 - a bad refinement must
        // never out-rank a legitimate candidate.
        if (Float.isFinite(prefixScore) == false || Float.isFinite(suffixScore) == false || prefixScore <= 0f || suffixScore <= 0f) {
            return 0f;
        }
        float denom = (1f / prefixScore) + (1f / suffixScore) - 1f;
        // For valid inputs in (0, 1] both reciprocals are >= 1, so denom is always >= 1.
        // A non-finite or non-positive denom therefore signals invalid/degenerate input
        // (e.g. a per-half score that drifted above 1 from numerical noise). Same fallback.
        if (Float.isFinite(denom) == false || denom <= 0f) {
            return 0f;
        }
        return Math.max(1f / denom, 0f);
    }

    private static float invertMip(float scaledScore) {
        // Inverse of VectorUtil.scaleMaxInnerProductScore:
        // raw >= 0 -> scaled = raw + 1
        // raw < 0 -> scaled = 1 / (1 - raw) (always in (0, 1))
        if (scaledScore >= 1f) {
            return scaledScore - 1f;
        }
        // scaled in (0, 1) -> raw = 1 - 1/scaled (negative)
        // Guard against pathological zero input.
        if (scaledScore <= 0f) {
            return Float.NEGATIVE_INFINITY;
        }
        return 1f - 1f / scaledScore;
    }
}
