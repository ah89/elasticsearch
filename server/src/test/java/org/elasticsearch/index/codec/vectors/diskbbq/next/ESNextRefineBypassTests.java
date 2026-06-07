/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.elasticsearch.test.ESTestCase;

/**
 * Unit tests for {@link ESNextDiskBBQVectorsReader#shouldBypassRefinement(float[], int, float)}.
 *
 * <p>The helper is a pure function over a best-first sorted prefix-score window and a relative-gap
 * threshold. End-to-end recall behavior is validated by benchmarks; these tests cover only the
 * decision boundary and the documented disable / no-op cases.
 */
public class ESNextRefineBypassTests extends ESTestCase {

    public void testBypassTriggersWhenRelativeGapExceedsRatio() {
        // best=1.0, worst=0.4 → relative gap = 0.6 / 1.0 = 0.6, above ratio 0.25 → bypass
        float[] scores = { 1.0f, 0.9f, 0.7f, 0.4f };
        assertTrue(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 4, 0.25f));
    }

    public void testBypassSkippedWhenRelativeGapBelowRatio() {
        // best=1.0, worst=0.9 → relative gap = 0.1 / 1.0 = 0.1, below ratio 0.25 → no bypass
        float[] scores = { 1.0f, 0.95f, 0.92f, 0.9f };
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 4, 0.25f));
    }

    public void testInfiniteRatioDisablesBypass() {
        // Even a maximally wide gap must not trigger bypass when the caller passes +Inf as the
        // ratio. This is the documented escape hatch for restoring pre-bypass behavior.
        float[] scores = { 1.0f, 0.0f };
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 2, Float.POSITIVE_INFINITY));
    }

    public void testNonPositiveRatioDisablesBypass() {
        // Ratios of zero or below must not trigger bypass — they represent "always refine".
        float[] scores = { 1.0f, 0.0f };
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 2, 0.0f));
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 2, -1.0f));
    }

    public void testSingleEntryNeverBypasses() {
        // A one-element window has no second element to compare against; the gap is undefined,
        // so the helper must conservatively return false regardless of the configured ratio.
        float[] scores = { 1.0f };
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 1, 0.001f));
    }

    public void testZeroRefineCountNeverBypasses() {
        // refineCount == 0 means the queue was empty; the helper must not read array slots that
        // don't exist. The early-return on refineCount <= 1 guards this.
        float[] scores = {};
        assertFalse(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 0, 0.001f));
    }

    public void testHandlesNegativeBestScoreViaAbs() {
        // For un-normalized dot_product, scores can be negative. The denominator uses |best| so
        // the relative-gap test stays well-defined: best=-0.4, worst=-1.0 → gap = 0.6, |best| = 0.4,
        // relativeGap = 1.5, which exceeds ratio 0.25 → bypass.
        float[] scores = { -0.4f, -0.6f, -0.8f, -1.0f };
        assertTrue(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 4, 0.25f));
    }

    public void testHandlesNearZeroBestScoreViaEpsilon() {
        // best ≈ 0 would normally divide by zero; the helper clamps the denominator at EPS so
        // the comparison still yields a meaningful (very large) ratio. Caller's ratio knob still
        // controls the decision.
        float[] scores = { 1e-20f, 0.0f, -1e-10f };
        // gap = 1e-20f - (-1e-10f) ≈ 1e-10f, divided by max(|1e-20f|, 1e-12f) = 1e-12f → ratio ≈ 100
        assertTrue(ESNextDiskBBQVectorsReader.shouldBypassRefinement(scores, 3, 1.0f));
    }
}
