/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.coreset;

import org.elasticsearch.common.io.stream.NamedWriteable;
import org.elasticsearch.xcontent.ToXContentObject;

import java.util.List;

/**
 * A geometric summary of an embedding-space domain distribution used for
 * ContLoRA continual learning.
 *
 * <p>Each domain encountered during continual learning is compressed into a
 * {@code GeometricCoreset} instead of retaining raw training embeddings.  The
 * coreset serves two purposes:
 * <ol>
 *   <li>Novelty detection — deciding whether new data constitutes a genuinely
 *       new domain or overlaps sufficiently with an existing one.</li>
 *   <li>Boundary-aware negative sampling — generating hard negatives near
 *       domain decision boundaries for training sparse LoRA selectors.</li>
 * </ol>
 *
 * <p>Implementations must be serialisable via {@link NamedWriteable} so that
 * coresets survive node restarts and can be stored in system indices.
 */
public interface GeometricCoreset extends NamedWriteable, ToXContentObject {

    /**
     * Returns the coreset type identifier, e.g. {@code "gmm"} or {@code "k_center"}.
     * Must match the {@link NamedWriteable#getWriteableName()} value.
     */
    String getType();

    /**
     * Returns the dimensionality of the embedding space in which this coreset
     * was constructed.  All embedding vectors passed to this coreset must have
     * this length.
     */
    int getDimension();

    /**
     * Returns a set of representative centroids for this domain.  Used by
     * {@link org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector}
     * to compute a quick centroid-level overlap estimate before the full
     * overlap computation.
     */
    List<float[]> getCentroids();

    /**
     * Computes a normalised overlap score in [0, 1] between this coreset and
     * {@code other}.  A score of 0 means the two domains are fully separated;
     * a score of 1 means they are identical.
     *
     * <p>The concrete formula depends on the coreset type (e.g.\ Bhattacharyya
     * coefficient for GMMs, radius-overlap fraction for k-center).
     *
     * @param other the coreset for the domain to compare against; must have the
     *              same {@link #getDimension()} as this coreset
     */
    float computeOverlap(GeometricCoreset other);

    /**
     * Generates {@code count} boundary-aware negative samples for selector
     * training.  Negatives are placed near the boundary of this domain's
     * region in the direction of {@code targetCentroid}, offset outward by
     * {@code margin}.
     *
     * @param targetCentroid centroid of the new domain being trained
     * @param count          number of negative samples to generate
     * @param margin         boundary offset distance (in embedding space units)
     * @return list of {@code count} negative embedding vectors
     */
    List<float[]> sampleBoundaryNegatives(float[] targetCentroid, int count, float margin);

    /**
     * Computes a familiarity score for a single embedding.
     * Higher values indicate the embedding is more likely to belong to this domain.
     *
     * <p>The default implementation delegates to {@link #computeOverlap} via a
     * single-centroid proxy coreset.  Concrete implementations should override
     * this with type-specific efficient scoring.
     *
     * @param embedding the query embedding vector
     * @return familiarity score (higher = more familiar / less novel)
     */
    default double scoreSample(float[] embedding) {
        // Default: cosine similarity to nearest centroid
        List<float[]> centroids = getCentroids();
        double maxSim = Double.NEGATIVE_INFINITY;
        for (float[] centroid : centroids) {
            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < embedding.length; i++) {
                dot += (double) embedding[i] * centroid[i];
                normA += (double) embedding[i] * embedding[i];
                normB += (double) centroid[i] * centroid[i];
            }
            double denom = Math.sqrt(normA) * Math.sqrt(normB);
            double sim = denom < 1e-12 ? 0 : dot / denom;
            if (sim > maxSim) {
                maxSim = sim;
            }
        }
        return maxSim;
    }

    /**
     * Batch-scores multiple embeddings for familiarity with this domain.
     * The default implementation calls {@link #scoreSample} per element.
     * Concrete implementations should override for amortised performance.
     *
     * @param batch array of embedding vectors
     * @return array of familiarity scores, one per input
     */
    default double[] batchScoreSamples(float[][] batch) {
        double[] scores = new double[batch.length];
        for (int i = 0; i < batch.length; i++) {
            scores[i] = scoreSample(batch[i]);
        }
        return scores;
    }
}
