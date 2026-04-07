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
}
