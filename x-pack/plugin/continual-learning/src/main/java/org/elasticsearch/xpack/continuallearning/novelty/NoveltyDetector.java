/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.novelty;

import org.elasticsearch.xpack.continuallearning.coreset.GeometricCoreset;

import java.util.List;

/**
 * Determines whether an incoming domain is genuinely novel relative to all
 * previously seen domains.
 *
 * <p>The novelty score for an incoming coreset {@code H_{t+1}} is
 * <pre>
 *   Novelty(H_{t+1}) = 1 − max_{i ≤ t} Overlap(H_{t+1}, H_i)
 * </pre>
 * where {@code Overlap} is the coreset-type-specific overlap metric
 * (e.g.\ Bhattacharyya coefficient for GMMs, radius-overlap fraction for
 * k-center coresets).
 *
 * <p>If the novelty score exceeds the configured {@link #threshold} the domain
 * is treated as new and a fresh LoRA adapter is allocated.  Otherwise the
 * incoming data is merged into the most similar existing domain.
 *
 * <p>When JL projection is active, all overlap computations occur in the
 * projected {@code d'}-dimensional space.  Per Theorem 5 of ContLoRA, the
 * overlap estimates distort by at most {@code O(ε_JL)}.  The caller is
 * expected to tighten the threshold by {@code 2 * ε_JL} to compensate.
 */
public class NoveltyDetector {

    /**
     * Default novelty threshold.  Domains with novelty below this value are
     * considered duplicates.  Calibrated for FPR ≤ 5% on held-out validation
     * data (see §4.3 of the ContLoRA paper).
     */
    public static final float DEFAULT_THRESHOLD = 0.5f;

    private final float threshold;

    /**
     * Creates a novelty detector with the given threshold.
     *
     * @param threshold novelty scores above this value trigger a new LoRA allocation
     */
    public NoveltyDetector(float threshold) {
        this.threshold = threshold;
    }

    /** Creates a novelty detector with the {@link #DEFAULT_THRESHOLD}. */
    public NoveltyDetector() {
        this(DEFAULT_THRESHOLD);
    }

    /**
     * Computes the novelty of an incoming domain coreset relative to all
     * previously stored domain coresets.
     *
     * @param incoming       coreset for the new domain
     * @param existingDomains list of coresets for all previously seen domains
     * @return novelty score in [0, 1]; higher means more novel
     */
    public float computeNovelty(GeometricCoreset incoming, List<GeometricCoreset> existingDomains) {
        if (existingDomains.isEmpty()) {
            return 1.0f;
        }
        float maxOverlap = 0f;
        for (GeometricCoreset existing : existingDomains) {
            float overlap = incoming.computeOverlap(existing);
            if (overlap > maxOverlap) {
                maxOverlap = overlap;
            }
        }
        return 1.0f - maxOverlap;
    }

    /**
     * Returns {@code true} if the incoming domain should receive a new LoRA
     * adapter, i.e. its novelty score exceeds the configured threshold.
     *
     * @param incoming       coreset for the new domain
     * @param existingDomains coresets for all previously seen domains
     */
    public boolean isNovel(GeometricCoreset incoming, List<GeometricCoreset> existingDomains) {
        return computeNovelty(incoming, existingDomains) >= threshold;
    }

    /**
     * Identifies the most similar existing domain for potential merging when the
     * incoming domain is not considered novel.
     *
     * @param incoming       coreset for the new domain
     * @param existingDomains coresets for all previously seen domains
     * @return index into {@code existingDomains} of the closest domain, or -1 if
     *         {@code existingDomains} is empty
     */
    public int findClosestDomain(GeometricCoreset incoming, List<GeometricCoreset> existingDomains) {
        if (existingDomains.isEmpty()) {
            return -1;
        }
        int closest = 0;
        float maxOverlap = -1f;
        for (int i = 0; i < existingDomains.size(); i++) {
            float overlap = incoming.computeOverlap(existingDomains.get(i));
            if (overlap > maxOverlap) {
                maxOverlap = overlap;
                closest = i;
            }
        }
        return closest;
    }

    /**
     * Returns the configured novelty threshold.
     */
    public float getThreshold() {
        return threshold;
    }

    /**
     * Returns a tightened threshold adjusted for JL projection distortion.
     * Per the ContLoRA paper (§4.3), when JL is active the threshold should be
     * reduced by {@code 2 * ε_JL} to maintain the same FPR guarantee.
     *
     * @param epsilonJL the JL distortion parameter used during projection
     * @return adjusted threshold
     */
    public float adjustedThresholdForJL(float epsilonJL) {
        return Math.max(0.0f, threshold - 2.0f * epsilonJL);
    }

    // =========================================================================
    // Per-sample novelty scoring via Fisher Discriminant Gap
    // =========================================================================

    /**
     * Computes the Fisher Discriminant Gap novelty score for a single embedding
     * against all existing domain coresets.
     *
     * <p>The Fisher Gap cancels the "ambient density" contribution that arises from
     * concentration of measure in high-dimensional ViT embedding spaces (d ≫ 1).
     * For each sample x, instead of using the raw maximum log-likelihood across
     * tasks, we compute:
     *
     * <pre>
     *   FisherGap(x) = max_j [LL_j(x) − mean_{i≠j} LL_i(x)]
     * </pre>
     *
     * <p>This is equivalent to projecting onto the first Fisher discriminant direction
     * in the T-dimensional space of per-task log-likelihoods.  The ambient LL term
     * (which is approximately constant across tasks due to concentration of measure)
     * cancels, isolating the task-discriminative signal.
     *
     * <p>For novel samples, even when one task's GMM gives high LL (absorption), the
     * Fisher Gap is small because other tasks give similarly high LL.  For known
     * samples from the correct task, the Fisher Gap is large because one task's GMM
     * is distinctively better.
     *
     * @param embedding       the query embedding vector
     * @param existingDomains list of coresets for all previously seen domains
     * @return Fisher Gap score; higher = more likely to belong to an existing domain
     *         (i.e. LESS novel)
     */
    public float computeFisherGap(float[] embedding, List<GeometricCoreset> existingDomains) {
        if (existingDomains.isEmpty()) {
            return 0f;
        }
        int T = existingDomains.size();
        double[] scores = new double[T];
        double sumScores = 0;
        for (int j = 0; j < T; j++) {
            scores[j] = existingDomains.get(j).scoreSample(embedding);
            sumScores += scores[j];
        }
        if (T == 1) {
            return (float) scores[0];
        }
        // Fisher gap: max_j [score_j - mean_{i≠j} score_i]
        double bestGap = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < T; j++) {
            double meanOthers = (sumScores - scores[j]) / (T - 1);
            double gap = scores[j] - meanOthers;
            if (gap > bestGap) {
                bestGap = gap;
            }
        }
        return (float) bestGap;
    }

    /**
     * Batch version of {@link #computeFisherGap} for multiple embeddings.
     *
     * @param embeddings      array of query embedding vectors
     * @param existingDomains list of coresets for all previously seen domains
     * @return array of Fisher Gap scores, one per embedding
     */
    public float[] batchFisherGap(float[][] embeddings, List<GeometricCoreset> existingDomains) {
        if (existingDomains.isEmpty()) {
            float[] result = new float[embeddings.length];
            return result;
        }
        int T = existingDomains.size();
        int m = embeddings.length;

        // Compute per-task scores for all embeddings
        double[][] perTaskScores = new double[T][];
        for (int j = 0; j < T; j++) {
            perTaskScores[j] = existingDomains.get(j).batchScoreSamples(embeddings);
        }

        // Compute Fisher gap for each embedding
        float[] result = new float[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < T; j++) {
                sum += perTaskScores[j][i];
            }
            double bestGap = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < T; j++) {
                double meanOthers = (sum - perTaskScores[j][i]) / (T - 1);
                double gap = perTaskScores[j][i] - meanOthers;
                if (gap > bestGap) {
                    bestGap = gap;
                }
            }
            result[i] = (float) bestGap;
        }
        return result;
    }
}
