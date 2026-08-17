/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;

import java.io.IOException;
import java.util.Arrays;

/**
 * Three numbers per (cluster, attribute value), describing where that attribute's vectors sit
 * inside the cluster. Writing {@code mu} for the cluster centroid (length {@code m}) and {@code nu}
 * for the centroid of just the members carrying the value:
 *
 * <ol>
 *   <li><b>x</b> — the fraction of the cluster's members that carry the value;</li>
 *   <li><b>mult</b> — {@code (nu . mu_hat) / m}, the component of the sub-centroid along the
 *       centroid direction, expressed so that {@code mult * sim} is the expected similarity of a
 *       matching member;</li>
 *   <li><b>perp</b> — {@code |nu_perp|}, how far the sub-centroid sits off that axis.</li>
 * </ol>
 *
 * <p>Held as three {@code float}s, so a pair costs 12 bytes of payload beside its 4-byte cluster id.
 * For a query whose similarity to the centroid is {@code sim} and whose cosine to it is
 * {@code s_hat = sim/m}, the similarity to the value's own centroid is confined to
 *
 * <pre>  mult*sim  -+  perp*sqrt(1 - s_hat^2)</pre>
 *
 * the nearest and farthest points of the circle the sub-centroid may lie on. The midpoint is the
 * expectation over the unknown rotation and is what orders the clusters; the half-width bounds how
 * wrong that can be, and its standard deviation is that half-width over {@code sqrt(dim - 1)}.
 *
 * <p><b>Why these two numbers and not {@code (y, z)}.</b> An earlier revision stored the distance
 * {@code y = |nu - mu|} and the cosine {@code z} between the two centroids, and recovered the
 * decomposition per query by solving {@code |nu|^2 - 2*m*z*|nu| + (m^2 - y^2) = 0}. That quadratic
 * has two roots and {@code (y, z)} does not say which one is the truth: taking the larger was wrong
 * for 13.9% of pairs on arXiv (mean relative error 0.25%, worst 10.8%). The parallel/perpendicular
 * pair is the same three floats, is exact because the build has {@code nu} in hand, and leaves
 * nothing for the query to derive — the coefficients are query-independent, so all of that work
 * belongs at build time regardless.
 *
 * <p>Alongside them the build derives an additive <em>yield</em> per pair — how far above the
 * sub-centroid the best of the cluster's {@code n} matching members is expected to sit,
 * {@code sigma_A * sqrt(2 ln n)}. It is a function of the stored numbers and the cluster statistics
 * only, so it too is precomputed rather than recomputed per query.
 */
public final class AttributeLensIndex {

    /** Payload bytes per (cluster, value) pair: the three numbers plus the cluster id. */
    public static final int BYTES_PER_PAIR = 3 * Float.BYTES + Integer.BYTES;

    /** Sentinel cached when the inputs are unavailable, so the build is attempted once. */
    public static final AttributeLensIndex UNSUPPORTED = new AttributeLensIndex(
        new int[1],
        new int[0],
        new float[0],
        new float[0],
        new float[0],
        new float[0],
        new boolean[0],
        new float[0],
        0
    );

    // CSR by value: pairs [valueOffsets[v] .. valueOffsets[v+1]) hold that value's clusters
    private final int[] valueOffsets;
    private final int[] pairCluster;
    /** x: fraction of the cluster's members carrying the value */
    private final float[] pairX;
    /** mult: {@code (nu . mu_hat)/m}, so {@code mult * sim} is the expected member similarity */
    private final float[] pairMult;
    /** perp: {@code |nu_perp|}, the half-width coefficient of the reachable range */
    private final float[] pairPerp;
    /**
     * Derived, not stored: {@code sigma_A * sqrt(2 ln n)}, the expected lift of the best of the
     * cluster's n matching members above their sub-centroid, in similarity units.
     */
    private final float[] pairYield;
    /**
     * One bit per pair: whether the cluster <em>owns</em> at least one document carrying the value,
     * as opposed to only holding SOAR overspill copies of documents owned elsewhere. Only used by
     * the centroid gate, and only to keep it recall-identical to the gates it replaces: an
     * overspill-only cluster can be skipped without losing a document, because every copy it holds
     * belongs to a document whose owning cluster is accepted anyway. Leaving those clusters in
     * would quietly enlarge the accept set and, through the follow-on drain, the effective budget —
     * which reads as a recall gain but is really extra work.
     */
    private final boolean[] pairOwns;

    /** |mu_c| per cluster (mean of the members, not unit length) */
    private final float[] centroidNorm;
    private final int numCentroids;
    private final int maxPairsPerValue;

    private AttributeLensIndex(
        int[] valueOffsets,
        int[] pairCluster,
        float[] pairX,
        float[] pairMult,
        float[] pairPerp,
        float[] pairYield,
        boolean[] pairOwns,
        float[] centroidNorm,
        int numCentroids
    ) {
        this.valueOffsets = valueOffsets;
        this.pairCluster = pairCluster;
        this.pairX = pairX;
        this.pairMult = pairMult;
        this.pairPerp = pairPerp;
        this.pairYield = pairYield;
        this.pairOwns = pairOwns;
        this.centroidNorm = centroidNorm;
        this.numCentroids = numCentroids;
        int max = 0;
        for (int v = 0; v + 1 < valueOffsets.length; v++) {
            max = Math.max(max, valueOffsets[v + 1] - valueOffsets[v]);
        }
        this.maxPairsPerValue = max;
    }

    public int numValues() {
        return valueOffsets.length - 1;
    }

    public int numCentroids() {
        return numCentroids;
    }

    /**
     * Every array actually held, not just the ones the on-disk layout would need. {@link #BYTES_PER_PAIR}
     * is the design figure — the three stored numbers plus the cluster id — while the resident
     * structure also carries the derived yield and the ownership bit per pair. Counting only the
     * stored bytes would understate the heap by a quarter.
     */
    public long ramBytesUsed() {
        final long perPair = BYTES_PER_PAIR + Float.BYTES + 1; // stored 16 + yield 4 + owns 1
        return (long) valueOffsets.length * Integer.BYTES + (long) pairCluster.length * perPair + (long) centroidNorm.length * Float.BYTES;
    }

    /** Fraction of cluster {@code c} carrying {@code value}, 0 when absent. */
    public float proportion(int value, int c) {
        if (value < 0 || value >= numValues()) {
            return 0f;
        }
        for (int i = valueOffsets[value]; i < valueOffsets[value + 1]; i++) {
            if (pairCluster[i] == c) {
                return pairX[i];
            }
        }
        return 0f;
    }

    /**
     * Scatters this value's per-cluster ordering coefficients into centroid-indexed arrays:
     * {@code mult} so that {@code mult * sim} is the midpoint of the reachable similarity range, and
     * {@code add} the precomputed yield scaled by {@code beta}.
     *
     * <p>{@code perp} and {@code invNorm} are the half-width of that range and the {@code 1/|mu_c|}
     * that turns {@code sim} into a cosine. The ordering key does not use them — measurement puts the
     * optimal optimism weight at zero, see the class javadoc — but they are what makes the range's
     * containment property checkable, so they remain available as an optional output rather than
     * being dropped along with the term that applied them.
     *
     * <p>Every one of these is a function of the index alone, so the query does no arithmetic here
     * beyond the scale on the yield — this is a gather, not a derivation.
     *
     * <p>When {@code accept} is non-null the same pass also fills the exact centroid gate — the
     * clusters owning at least one document that carries the value. That is the whole gate: the
     * inversion the ordering needs is the inversion the gate needs, so the two come out of one
     * read and no walk over the filter's matching documents is required.
     *
     * @param accept if non-null, cleared and filled with the gate's accepted centroids
     * @return the number of clusters holding the value
     */
    public int scatterCoefficients(int value, float beta, float[] mult, float[] add, FixedBitSet accept, float[] perp, float[] invNorm) {
        if (value < 0 || value >= numValues()) {
            return 0;
        }
        final int from = valueOffsets[value];
        final int count = valueOffsets[value + 1] - from;
        if (accept != null) {
            accept.clear();
        }
        for (int i = 0; i < count; i++) {
            final int c = pairCluster[from + i];
            mult[c] = pairMult[from + i];
            add[c] = beta * pairYield[from + i];
            if (perp != null) {
                perp[c] = pairPerp[from + i];
                final float norm = centroidNorm[c];
                invNorm[c] = norm > 0 ? 1f / norm : 0f;
            }
            if (accept != null && pairOwns[from + i]) {
                accept.set(c);
            }
        }
        return count;
    }

    /** Largest number of clusters any single value spans, i.e. the scratch size a query can need. */
    public int maxPairsPerValue() {
        return maxPairsPerValue;
    }

    /**
     * Builds the three numbers with one pass per value over that value's documents, so peak scratch
     * is one accumulator per <em>touched</em> cluster rather than per (cluster, value) pair.
     *
     * <p>All three numbers describe <em>physical membership</em>: the copies a posting list actually
     * stores, SOAR overspill included. Deriving them from ownership instead makes x a ratio of two
     * quantities that are both wrong -- a cluster physically holds more vectors than it owns, and
     * more matching vectors than it owns -- and the resulting proportion is a false signal for the
     * early stop, which halts a scan once it believes it has seen every match the cluster holds.
     * The same reasoning applies to the sub-centroid behind mult and perp: it has to describe the
     * vectors that will actually be scanned, not a subset of them. The cluster centroid itself stays
     * ownership-derived, because it must reproduce the centroid stored in the index -- that is the
     * vector the query is scored against, and the sub-centroid is measured relative to it.
     *
     * @param values raw vectors, random access by ordinal
     * @param centroidOfOrdinal packed doc-&gt;centroid lookup
     * @param geometry posting-list geometry naming each document's overspill cluster; when null the
     *                 numbers fall back to ownership, which understates dense clusters
     * @param docValues CSR value ordinals per document
     * @param docValueOffsets CSR offsets, length {@code maxDoc + 1}
     */
    public static AttributeLensIndex build(
        FloatVectorValues values,
        LongValues centroidOfOrdinal,
        MatchFetchIndex geometry,
        int[] docValues,
        int[] docValueOffsets,
        int numValues,
        int numCentroids
    ) throws IOException {
        final int dim = values.dimension();
        final int maxDoc = docValueOffsets.length - 1;

        // pass 1: cluster centroids, their lengths, and the within-cluster spread
        final float[][] centroidSum = new float[numCentroids][dim];
        final int[] clusterSize = new int[numCentroids];
        final int[] clusterOfDoc = new int[maxDoc];
        final int[] ordOfDoc = new int[maxDoc];
        Arrays.fill(clusterOfDoc, -1);
        Arrays.fill(ordOfDoc, -1);
        final KnnVectorValues.DocIndexIterator it = values.iterator();
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            if (doc >= maxDoc) {
                break;
            }
            final int c = (int) centroidOfOrdinal.get(it.index());
            clusterOfDoc[doc] = c;
            ordOfDoc[doc] = it.index();
            final float[] x = values.vectorValue(it.index());
            final float[] sum = centroidSum[c];
            for (int j = 0; j < dim; j++) {
                sum[j] += x[j];
            }
            clusterSize[c]++;
        }
        final float[] centroidNorm = new float[numCentroids];
        final double inverseSqrtDim = 1.0 / Math.sqrt(dim);
        for (int c = 0; c < numCentroids; c++) {
            if (clusterSize[c] == 0) {
                continue;
            }
            final float[] sum = centroidSum[c];
            double sq = 0;
            for (int j = 0; j < dim; j++) {
                sum[j] /= clusterSize[c];
                sq += (double) sum[j] * sum[j];
            }
            centroidNorm[c] = (float) Math.sqrt(sq);
        }

        // second cluster physically holding each document, and the resulting per-cluster member
        // counts -- the denominator of x
        final int[] overspillOfDoc = new int[maxDoc];
        Arrays.fill(overspillOfDoc, -1);
        final int[] memberCount = new int[numCentroids];
        for (int doc = 0; doc < maxDoc; doc++) {
            if (clusterOfDoc[doc] < 0) {
                continue;
            }
            memberCount[clusterOfDoc[doc]]++;
            final int spill = geometry == null ? -1 : geometry.overspillCentroid(doc);
            if (spill >= 0) {
                overspillOfDoc[doc] = spill;
                memberCount[spill]++;
            }
        }

        // invert doc->values into (value -> memberships), one entry per copy, so each value takes
        // one pass and a document counts toward every cluster that stores it
        final int[] docsPerValue = new int[numValues + 1];
        for (int doc = 0; doc < maxDoc; doc++) {
            if (clusterOfDoc[doc] < 0) {
                continue;
            }
            final int copies = overspillOfDoc[doc] >= 0 ? 2 : 1;
            for (int i = docValueOffsets[doc]; i < docValueOffsets[doc + 1]; i++) {
                final int v = docValues[i];
                if (v >= 0 && v < numValues) {
                    docsPerValue[v + 1] += copies;
                }
            }
        }
        for (int v = 0; v < numValues; v++) {
            docsPerValue[v + 1] += docsPerValue[v];
        }
        final int[] valueClusters = new int[docsPerValue[numValues]];
        final int[] fill = new int[numValues];
        for (int doc = 0; doc < maxDoc; doc++) {
            if (clusterOfDoc[doc] < 0) {
                continue;
            }
            for (int i = docValueOffsets[doc]; i < docValueOffsets[doc + 1]; i++) {
                final int v = docValues[i];
                if (v < 0 || v >= numValues) {
                    continue;
                }
                valueClusters[docsPerValue[v] + fill[v]++] = clusterOfDoc[doc];
                if (overspillOfDoc[doc] >= 0) {
                    valueClusters[docsPerValue[v] + fill[v]++] = overspillOfDoc[doc];
                }
            }
        }

        // size and fill the pair CSR: which clusters each value touches, ascending so the
        // accumulation pass below can find a pair's slot by binary search
        final int[] valueOffsets = new int[numValues + 1];
        final boolean[] seen = new boolean[numCentroids];
        final int[] touched = new int[numCentroids];
        for (int v = 0; v < numValues; v++) {
            int pairs = 0;
            for (int i = docsPerValue[v]; i < docsPerValue[v + 1]; i++) {
                final int c = valueClusters[i];
                if (seen[c] == false) {
                    seen[c] = true;
                    touched[pairs++] = c;
                }
            }
            for (int t = 0; t < pairs; t++) {
                seen[touched[t]] = false;
            }
            valueOffsets[v + 1] = valueOffsets[v] + pairs;
        }
        final int totalPairs = valueOffsets[numValues];
        final int[] pairCluster = new int[totalPairs];
        for (int v = 0; v < numValues; v++) {
            int pairs = 0;
            for (int i = docsPerValue[v]; i < docsPerValue[v + 1]; i++) {
                final int c = valueClusters[i];
                if (seen[c] == false) {
                    seen[c] = true;
                    touched[pairs++] = c;
                }
            }
            Arrays.sort(touched, 0, pairs);
            System.arraycopy(touched, 0, pairCluster, valueOffsets[v], pairs);
            for (int t = 0; t < pairs; t++) {
                seen[touched[t]] = false;
            }
        }
        final float[] pairX = new float[totalPairs];
        final float[] pairMult = new float[totalPairs];
        final float[] pairPerp = new float[totalPairs];
        final float[] pairYield = new float[totalPairs];
        final boolean[] pairOwns = new boolean[totalPairs];

        // pass 2: the sub-centroids, accumulated in DOCUMENT order rather than value order.
        //
        // Accumulating per value reads a vector once per membership -- once for each value the
        // document carries, and again for its SOAR overspill copy -- scattered across the file,
        // three-ish reads per document on a multi-label corpus. Since every membership of a
        // document adds the same vector, one sequential pass that fans each vector out to its pair
        // accumulators reads it once, in file order, and computes its squared norm once instead of
        // once per membership. The accumulators are per pair rather than per cluster, so they are
        // sized by the CSR: on arXiv 3556 pairs x 384 dims is 5.5 MB. Values are processed in
        // batches when that would not fit, one sequential pass each, which keeps a
        // high-cardinality field bounded instead of quadratic.
        final int pairsInFlight = Math.max(1, (int) (MAX_ACCUMULATOR_BYTES / ((long) dim * Float.BYTES)));
        final float[][] pairSum = new float[totalPairs][];
        final int[] pairCount = new int[totalPairs];
        final double[] pairSqNormSum = new double[totalPairs];
        int valueFrom = 0;
        while (valueFrom < numValues) {
            int valueTo = valueFrom + 1;
            while (valueTo < numValues && valueOffsets[valueTo + 1] - valueOffsets[valueFrom] <= pairsInFlight) {
                valueTo++;
            }
            for (int doc = 0; doc < maxDoc; doc++) {
                if (clusterOfDoc[doc] < 0 || docValueOffsets[doc] == docValueOffsets[doc + 1]) {
                    continue;
                }
                float[] x = null;
                double sq = 0;
                for (int i = docValueOffsets[doc]; i < docValueOffsets[doc + 1]; i++) {
                    final int v = docValues[i];
                    if (v < valueFrom || v >= valueTo) {
                        continue;
                    }
                    if (x == null) {
                        x = values.vectorValue(ordOfDoc[doc]);
                        for (int j = 0; j < dim; j++) {
                            sq += (double) x[j] * x[j];
                        }
                    }
                    final int owner = slotOf(pairCluster, valueOffsets[v], valueOffsets[v + 1], clusterOfDoc[doc]);
                    accumulate(pairSum, pairCount, pairSqNormSum, owner, x, sq, dim);
                    pairOwns[owner] = true;
                    if (overspillOfDoc[doc] >= 0) {
                        final int spill = slotOf(pairCluster, valueOffsets[v], valueOffsets[v + 1], overspillOfDoc[doc]);
                        accumulate(pairSum, pairCount, pairSqNormSum, spill, x, sq, dim);
                    }
                }
            }
            // reduce this batch's pairs to the three numbers plus the derived yield, then release
            // the accumulators before the next batch allocates its own
            for (int p = valueOffsets[valueFrom]; p < valueOffsets[valueTo]; p++) {
                final int c = pairCluster[p];
                final float[] sub = pairSum[p];
                final int n = pairCount[p];
                if (sub == null || n == 0) {
                    continue;
                }
                final float[] mu = centroidSum[c];
                final float m = centroidNorm[c];
                double dot = 0;
                double subSq = 0;
                for (int j = 0; j < dim; j++) {
                    final double s = sub[j] / n;
                    dot += s * mu[j];
                    subSq += s * s;
                }
                // parallel component of the sub-centroid, nu . mu_hat, and what is left over
                final double parallel = m > 0 ? dot / m : 0;
                final double perpendicular = Math.sqrt(Math.max(0, subSq - parallel * parallel));
                pairX[p] = (float) n / memberCount[c];
                pairMult[p] = m > 0 ? (float) (parallel / m) : 1f;
                pairPerp[p] = (float) perpendicular;
                // yield: the expected lift of the best of n draws above their own mean. The scatter
                // that governs it is the matching members' spread about the sub-centroid, not the
                // whole cluster's about the cluster centroid -- a value that clumps inside its
                // cluster is exactly the case the lens exists for, and there the two differ most.
                final float sigma = (float) (Math.sqrt(Math.max(0, pairSqNormSum[p] / n - subSq)) * inverseSqrtDim);
                pairYield[p] = sigma * (float) Math.sqrt(2.0 * Math.log(Math.max(2, n)));
                pairSum[p] = null;
            }
            valueFrom = valueTo;
        }

        return new AttributeLensIndex(
            valueOffsets,
            pairCluster,
            pairX,
            pairMult,
            pairPerp,
            pairYield,
            pairOwns,
            centroidNorm,
            numCentroids
        );
    }

    /**
     * How much heap the sub-centroid accumulators may hold at once. Sets how many values one
     * sequential pass covers; anything beyond it costs another pass, not more memory.
     */
    private static final long MAX_ACCUMULATOR_BYTES = 64L << 20;

    /** Slot of {@code cluster} within a value's ascending pair range. */
    private static int slotOf(int[] pairCluster, int from, int to, int cluster) {
        final int at = Arrays.binarySearch(pairCluster, from, to, cluster);
        assert at >= 0 : "cluster " + cluster + " missing from the pair CSR";
        return at;
    }

    private static void accumulate(float[][] sums, int[] counts, double[] sqNorms, int slot, float[] x, double sq, int dim) {
        float[] sum = sums[slot];
        if (sum == null) {
            sum = sums[slot] = new float[dim];
        }
        for (int j = 0; j < dim; j++) {
            sum[j] += x[j];
        }
        counts[slot]++;
        sqNorms[slot] += sq;
    }

}
