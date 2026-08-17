/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.simdvec.ES92Int7VectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata.NO_ORDINAL;

public class FlatCentroidIndex {

    /** Maximum number of filter-matching documents sampled to estimate per-centroid match density. */
    static final int MAX_SAMPLED_DOCS = 4096;
    /**
     * Minimum ratio of the largest per-centroid sample count over the mean for the density signal to be
     * used. With a filter that is uncorrelated with the clustering, per-centroid counts are approximately
     * Poisson around the mean and the max rarely exceeds a small multiple of it; in that regime the
     * "density" is sampling noise and boosting by it scrambles the score-ordered centroid visit order.
     * Only a strongly concentrated distribution — e.g. produced by attribute-fused clustering, where
     * matches pile into few category-pure posting lists — carries signal worth reordering for.
     */
    static final float MIN_DENSITY_CONCENTRATION = 4f;
    /**
     * POC-only hook (see {@code ESNextDiskBBQFusedClusteringPocTests}): forces the exact centroid
     * counted-gate path in the mid-selectivity regime, emulating Mariano's method — the
     * cluster-level per-field value-set bitmap of the hash-fingerprint filtered-search proposal,
     * upgraded from a binary presence gate to exact per-centroid match counts. A production
     * implementation would precompute per-cluster per-value counts at merge time; here the
     * O(matches) counting walk runs on first use and is cached per (filter, segment) on the
     * {@link IVFVectorsReader.FieldEntry}, so repeated filters — the common production shape, and
     * why Lucene's own query cache exists — get the same query-time cost profile as merge-built
     * bitmaps while still supporting arbitrary filters. Never enabled outside the POC harness.
     */
    public static volatile boolean POC_FORCE_EXACT_CENTROID_PRUNING = false;

    /**
     * POC-only: the summarized field's per-document value ordinals in CSR form —
     * {@code POC_DOC_VALUES[POC_DOC_VALUE_OFFSETS[doc] .. POC_DOC_VALUE_OFFSETS[doc+1])} are the
     * ordinals carried by {@code doc}. Multi-valued because real attribute data is: a document can
     * carry several categories, and a cluster contains a value if <em>any</em> of its documents does.
     * Stands in for the doc-values read a production implementation would perform while building the
     * summary at merge time.
     */
    public static volatile int[] POC_DOC_VALUES = null;

    /** POC-only: CSR offsets into {@link #POC_DOC_VALUES}, length {@code maxDoc + 1}. */
    public static volatile int[] POC_DOC_VALUE_OFFSETS = null;

    /** POC-only: number of distinct discretized values in {@link #POC_DOC_VALUES}. */
    public static volatile int POC_NUM_VALUES = 0;

    /**
     * POC-only: the current query's valid value set — the {@code S_i} of the proposal. Stands in for
     * the query inspection a production implementation would perform.
     */
    public static final ThreadLocal<int[]> POC_QUERY_VALUE_SET = new ThreadLocal<>();

    /**
     * POC-only: live heap held by centroid-gate structures — the per-segment lookups/summaries plus
     * whatever the per-filter caches currently retain. Maintained incrementally (cache evictions
     * subtract) so the harness can report a memory column alongside QPS and recall, making the gates'
     * very different memory profiles visible: per-filter caches grow with workload diversity, while
     * the per-segment structures are fixed by segment size or field cardinality.
     */
    public static final AtomicLong POC_GATE_RAM_BYTES = new AtomicLong();

    /**
     * POC-only hook enabling the FetchGate (see {@link MatchFetchIndex}): in the selective regime,
     * instead of gating full posting-list scans, the search jumps straight to each matching
     * document's quantized bytes and scores only the matches, in centroid-distance order, with the
     * visit budget counted in matches scored. Outside that regime the search falls back to the
     * forced exact centroid gate, so the arm is never worse than the gates it generalizes. Never
     * enabled outside the POC harness.
     */
    public static volatile boolean POC_FETCH_GATE = false;

    /**
     * POC-only: a precomputed accept-centroids set for the current query, installed by the fetch
     * path before it builds this index so the centroid iterator only surfaces (in distance order)
     * the posting lists that hold at least one match. Identical accept-set semantics to the counted
     * gate — the set is derived from the same doc-&gt;centroid walk — it just arrives precomputed
     * because the fetch path already needed the full per-centroid match lists.
     */
    public static final ThreadLocal<FixedBitSet> POC_FETCH_ACCEPT_CENTROIDS = new ThreadLocal<>();

    /**
     * POC-only: the accept-centroids set derived from the lens index alone, installed by
     * {@code prepareLens}. The lens stores its pairs inverted by value, so the clusters holding a
     * value are already listed contiguously: reading them <em>is</em> the exact gate, and it costs
     * one pass over that list. No walk over the filter's matching documents (what the counted gate
     * does), and no per-field value bitmap to build and keep current (what the cluster-summary gate
     * does) — the structure that supplies the ordering supplies the gate for free.
     *
     * <p>Recall-safety rests on the same precondition the ordering already relies on: the query's
     * value set must <em>cover</em> the filter, i.e. every document the filter accepts carries at
     * least one of its values. Under that precondition every cluster physically storing a matching
     * document is listed, SOAR overspill copies included, so the set can never prune a cluster that
     * holds a match. Where the value set is only a superset of the filter the gate is merely less
     * selective, never wrong.
     */
    public static final ThreadLocal<FixedBitSet> POC_LENS_ACCEPT_CENTROIDS = new ThreadLocal<>();

    /**
     * POC-only hook enabling the LensGate: per-(cluster, value) sub-centroid statistics
     * (see {@link AttributeLensIndex}) turn the cluster ranking from "nearest cluster centroid"
     * into "nearest <em>attribute sub-centroid</em>, upper-bounded", so filtered queries spend
     * their budget on the clusters whose matching mass is actually near the query.
     */
    public static volatile boolean POC_LENS_GATE = false;

    /**
     * POC-only: per-centroid ordering coefficients for the current query, installed by the lens
     * path. When set, a centroid's queue key becomes {@code mult[ord] * sim + add[ord]} where
     * {@code sim} is the raw similarity recovered from the centroid score — the upper bound on the
     * query's similarity to the filter value's sub-centroid in that cluster. The raw content score
     * is preserved for {@link PostingMetadata} exactly as the density path does.
     */
    public static final ThreadLocal<float[]> POC_LENS_MULT = new ThreadLocal<>();

    /** POC-only: additive part of the lens ordering key, see {@link #POC_LENS_MULT}. */
    public static final ThreadLocal<float[]> POC_LENS_ADD = new ThreadLocal<>();

    private final FieldInfo fieldInfo;
    private final IVFVectorsReader.FieldEntry fieldEntry;
    private final int numCentroids;
    private final IndexInput centroids;
    private final float visitRatio;
    private final FixedBitSet acceptCentroids;
    // sampled per-centroid filter-match density in [0,1]; non-null only in the mid-selectivity sampled mode
    // LensGate ordering coefficients for the current query (see POC_LENS_MULT), or null
    private final float[] lensMult;
    private final float[] lensAdd;
    // raw (unboosted) centroid scores, recorded so PostingMetadata carries the exact content score;
    // allocated lazily only when an ordering transform (density or lens) is active
    private final float[] rawScores;
    private final int numParents;
    private final FixedBitSet acceptParents;
    private final int bulkSize;
    private final byte[] quantized;
    private final OptimizedScalarQuantizer.QuantizationResult queryParams;

    public FlatCentroidIndex(
        FieldInfo fieldInfo,
        IVFVectorsReader.FieldEntry fieldEntry,
        int numCentroids,
        IndexInput centroids,
        float[] targetQuery,
        AcceptDocs acceptDocs,
        float approximateCost,
        FloatVectorValues values,
        float visitRatio
    ) throws IOException {
        this.fieldInfo = fieldInfo;
        this.fieldEntry = fieldEntry;
        this.numCentroids = numCentroids;
        this.centroids = centroids;
        this.visitRatio = visitRatio;

        // build optimization filters if possible
        CentroidFilter centroidFilter = getCentroidFilter(centroids, numCentroids, values, acceptDocs, approximateCost, fieldEntry);
        acceptCentroids = centroidFilter.acceptCentroids();
        lensMult = POC_LENS_MULT.get();
        lensAdd = POC_LENS_ADD.get();
        rawScores = (lensMult == null) ? null : new float[numCentroids];
        numParents = centroids.readVInt();
        acceptParents = getParentCentroidFilter(centroids, numParents, numCentroids, acceptDocs, fieldEntry.numSlices());

        // build centroid search helpers
        bulkSize = fieldEntry.getBulkSize();
        OptimizedScalarQuantizer scalarQuantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        int[] scratch = new int[targetQuery.length];
        queryParams = scalarQuantizer.scalarQuantize(
            targetQuery,
            new float[targetQuery.length],
            scratch,
            (byte) 7,
            fieldEntry.globalCentroid()
        );
        quantized = new byte[targetQuery.length];
        for (int i = 0; i < quantized.length; i++) {
            quantized[i] = (byte) scratch[i];
        }
    }

    /**
     * Result of inspecting the filter against the doc-to-centroid lookup: an exact centroid pruning
     * bitset, or null when no gate applies. Skips posting lists that provably contain no match; among
     * the surviving posting lists the content-score visit order is preserved.
     */
    private record CentroidFilter(FixedBitSet acceptCentroids) {
        static final CentroidFilter NONE = new CentroidFilter(null);
    }

    private static CentroidFilter getCentroidFilter(
        IndexInput centroids,
        int numCentroids,
        FloatVectorValues values,
        AcceptDocs acceptDocs,
        float approximateCost,
        IVFVectorsReader.FieldEntry fieldEntry
    ) throws IOException {
        float approximateDocsPerCentroid = approximateCost / numCentroids;
        if (approximateDocsPerCentroid <= 1.25) {
            // TODO: we need to make this call to build the iterator, otherwise accept docs breaks all together
            approximateDocsPerCentroid = (float) acceptDocs.cost() / numCentroids;
        }
        final int bitsRequired = DirectWriter.bitsRequired(numCentroids);
        final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
        long fp = centroids.getFilePointer();
        final FixedBitSet fetchAcceptCentroids = POC_FETCH_ACCEPT_CENTROIDS.get();
        final CentroidFilter centroidFilter;
        final FixedBitSet lensAcceptCentroids = POC_LENS_ACCEPT_CENTROIDS.get();
        if (fetchAcceptCentroids != null) {
            // the fetch path precomputed the accept set from its per-centroid match lists;
            // nothing to derive here
            centroidFilter = new CentroidFilter(fetchAcceptCentroids);
        } else if (lensAcceptCentroids != null) {
            // read straight off the lens index's value -> clusters inversion (see the field's
            // javadoc). As with the counted gate, a set that accepts everything prunes nothing, so
            // hand back NONE and let the search skip the per-centroid check entirely.
            centroidFilter = lensAcceptCentroids.cardinality() == numCentroids
                ? CentroidFilter.NONE
                : new CentroidFilter(lensAcceptCentroids);
        } else if (numCentroids == 1 || acceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll) {
            centroidFilter = CentroidFilter.NONE;
        } else if (approximateDocsPerCentroid <= 1.25 || POC_FORCE_EXACT_CENTROID_PRUNING) {
            // Restrictive filter: we expect many centroids to contain no matching document, so exact
            // pruning pays off. Three interchangeable implementations (identical accept sets, different
            // cost profiles) — the cluster value-set summary answers from index-side data alone; the
            // counted gate walks the matches through the doc->centroid lookup and caches per filter;
            // the SliceGate scans the filter's bitset words and reads centroid ids straight from an
            // unpacked copy of the same lookup.
            centroidFilter = countedGate(centroids, numCentroids, values, acceptDocs, fieldEntry, fp, sizeLookup, bitsRequired);
        } else {
            // Broad filter: post-filtering during posting visits handles this fine.
            centroidFilter = CentroidFilter.NONE;
        }
        centroids.seek(fp + sizeLookup);
        return centroidFilter;
    }

    /**
     * Builds (or reuses) exact per-centroid filter-match counts and derives from them an exact centroid
     * pruning bitset. The counts are cached on the {@link IVFVectorsReader.FieldEntry} keyed by a
     * content fingerprint of the filter's matching-doc set, so the O(matches) counting walk is paid
     * once per (filter, segment): production workloads repeat filters (that is what Lucene's query
     * cache exists for), giving later queries the same query-time cost profile as bitmaps precomputed
     * at merge time, while supporting arbitrary filters.
     */
    private static CentroidFilter countedGate(
        IndexInput centroids,
        int numCentroids,
        FloatVectorValues values,
        AcceptDocs acceptDocs,
        IVFVectorsReader.FieldEntry fieldEntry,
        long fp,
        long sizeLookup,
        int bitsRequired
    ) throws IOException {
        final Bits bits = acceptDocs.bits();
        final Long fingerprint = filterFingerprint(bits, acceptDocs);
        int[] counts = null;
        if (fingerprint != null) {
            final IVFVectorsReader.FieldEntry.CachedCentroidCounts cached = fieldEntry.filterCentroidCountsCache().get(fingerprint);
            // A fingerprint match alone is not enough to reuse counts: the key is a lossy hash, and
            // reusing another filter's counts would silently drop centroids that do hold matches.
            if (cached != null && sameFilter(cached.source().get(), bits)) {
                counts = cached.counts();
            }
        }
        if (counts == null) {
            counts = new int[numCentroids];
            final KnnVectorValues.DocIndexIterator docIndexIterator = values.iterator();
            final DocIdSetIterator iterator = ConjunctionUtils.intersectIterators(List.of(acceptDocs.iterator(), docIndexIterator));
            final LongValues longValues = DirectReader.getInstance(centroids.randomAccessSlice(fp, sizeLookup), bitsRequired);
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                counts[(int) longValues.get(docIndexIterator.index())]++;
            }
            if (fingerprint != null) {
                if (fieldEntry.filterCentroidCountsCache()
                    .put(fingerprint, new IVFVectorsReader.FieldEntry.CachedCentroidCounts(new WeakReference<>(bits), counts)) == null) {
                    POC_GATE_RAM_BYTES.addAndGet(IVFVectorsReader.FieldEntry.CACHE_ENTRY_OVERHEAD + (long) counts.length * Integer.BYTES);
                }
            }
        }
        final FixedBitSet acceptCentroids = new FixedBitSet(numCentroids);
        int accepted = 0;
        for (int i = 0; i < numCentroids; i++) {
            if (counts[i] > 0) {
                acceptCentroids.set(i);
                accepted++;
            }
        }
        // Gate only, deliberately: the bitset already encodes the match/no-match separation, and among
        // *surviving* clusters the content-score visit order is the right one — experiments stacking a
        // count- or fraction-based density boost on top of the gate consistently scrambled that order
        // (counts mirror cluster-size jitter; fractions are near-uniform over category-pure survivors)
        // and lost up to 16 recall points. The sampled density path below serves the un-gated regime,
        // where density also encodes the separation the bitset encodes here.
        //
        // A gate that accepts every centroid prunes nothing, so hand back NONE rather than a
        // full bitset: the search then skips the per-centroid accept check entirely. The other two
        // gate implementations already short-circuit this way, and leaving this one out made the
        // comparison between them unfair in exactly the broad-filter cells where no gate can help.
        return accepted == numCentroids ? CentroidFilter.NONE : new CentroidFilter(acceptCentroids);
    }

    /**
     * Whether cached counts may be reused for the current filter. Identity is the fast path and the
     * common one: when Lucene's query cache is warm it hands back the very same {@link FixedBitSet}
     * instance for a repeated filter (see {@code ESAcceptDocs.ScorerSupplierAcceptDocs}), so a hot
     * filter verifies in one reference comparison. Otherwise the two bitsets are compared word by word,
     * which is still far cheaper than the random-access counting walk it avoids. Anything that cannot
     * be verified cheaply is treated as a miss and recomputed, so a hash collision can never surface as
     * a wrong gate.
     */
    public static boolean sameFilter(Bits cached, Bits current) {
        if (cached == null || current == null) {
            return false;
        }
        if (cached == current) {
            return true;
        }
        if (cached.length() != current.length()) {
            return false;
        }
        if (cached instanceof FixedBitSet a && current instanceof FixedBitSet b) {
            return Arrays.equals(a.getBits(), b.getBits());
        }
        return false;
    }

    /**
     * Content fingerprint of the filter's matching-doc set, used as the counts-cache key. Hashes the
     * accept bitset's words (strided for very large segments) together with its length, so the same
     * logical filter maps to the same key across query instances even when each query materializes its
     * own bitset. Returns {@code null} — no caching — when the filter exposes no random-access bits.
     *
     * <p>This is deliberately a hash and not an identity: striding means it can collide, which is why
     * every hit is confirmed by {@link #sameFilter} before the counts are trusted. The hash only has to
     * be a good <em>index</em>; correctness comes from the verification step.
     */
    public static Long filterFingerprint(Bits bits, AcceptDocs acceptDocs) throws IOException {
        if (bits == null) {
            return null;
        }
        long h = 1125899906842597L;
        if (bits instanceof FixedBitSet fixedBitSet) {
            final long[] words = fixedBitSet.getBits();
            final int stride = Math.max(1, words.length / 1024);
            for (int i = 0; i < words.length; i += stride) {
                h = 31 * h + words[i];
            }
        } else {
            final int length = bits.length();
            final int stride = Math.max(1, length / 4096);
            for (int i = 0; i < length; i += stride) {
                h = 31 * h + (bits.get(i) ? 1 : 0);
            }
        }
        // mixing the exact match count in means a collision needs two different filters with identical
        // sampled bit patterns *and* identical cardinality
        return 31 * (31 * h + bits.length()) + acceptDocs.cost();
    }

    private static FixedBitSet getParentCentroidFilter(
        IndexInput centroids,
        int numParents,
        int numCentroids,
        AcceptDocs acceptDocs,
        int numSlices
    ) throws IOException {
        if (numSlices <= 0) {
            return null;
        }
        long fp = centroids.getFilePointer();
        FixedBitSet acceptParents = null;
        if (acceptDocs instanceof ESAcceptDocs esAcceptDocs) {
            // build a parent centroids filter
            int slice = esAcceptDocs.sliceOrd();
            // a slice must be provided
            assert slice >= 0 && slice < numSlices : "sliceOrd out of range for centroid slices";
            final int startOffset;
            final int endOffset;
            if (slice == 0) {
                startOffset = 0;
                endOffset = centroids.readInt();
            } else {
                centroids.skipBytes((long) (slice - 1) * Integer.BYTES);
                startOffset = centroids.readInt();
                endOffset = centroids.readInt();
            }
            if (numParents > 0) {
                acceptParents = new FixedBitSet(numParents);
                assert startOffset >= 0 && endOffset <= numParents;
            } else {
                acceptParents = new FixedBitSet(numCentroids);
                assert startOffset >= 0 && endOffset <= numCentroids;
            }
            acceptParents.set(startOffset, endOffset);
        }
        centroids.seek(fp + (long) numSlices * Integer.BYTES);
        return acceptParents;
    }

    public CentroidIterator getIterator() throws IOException {
        final ES92Int7VectorsScorer scorer = ESVectorUtil.getES92Int7VectorsScorer(centroids, fieldInfo.getVectorDimension(), bulkSize);
        // build iterator
        if (numParents > 0) {
            // equivalent to (float) centroidsPerParentCluster / 2
            float centroidOversampling = (float) fieldEntry.numCentroids() / (2 * numParents);
            return getCentroidIteratorWithParents(
                fieldInfo,
                centroids,
                numParents,
                numCentroids,
                scorer,
                quantized,
                queryParams,
                fieldEntry.globalCentroidDp(),
                visitRatio * centroidOversampling,
                acceptParents,
                acceptCentroids,
                lensMult,
                lensAdd,
                rawScores,
                bulkSize
            );
        } else {
            if (acceptCentroids != null && acceptParents != null) {
                acceptCentroids.and(acceptParents);
            }
            return getCentroidIteratorNoParent(
                fieldInfo,
                centroids,
                numCentroids,
                scorer,
                quantized,
                queryParams,
                fieldEntry.globalCentroidDp(),
                acceptCentroids != null ? acceptCentroids : acceptParents,
                lensMult,
                lensAdd,
                rawScores,
                bulkSize
            );
        }
    }

    private static CentroidIterator getCentroidIteratorNoParent(
        FieldInfo fieldInfo,
        IndexInput centroids,
        int numCentroids,
        ES92Int7VectorsScorer scorer,
        byte[] quantizeQuery,
        OptimizedScalarQuantizer.QuantizationResult queryParams,
        float globalCentroidDp,
        FixedBitSet acceptCentroids,
        float[] lensMult,
        float[] lensAdd,
        float[] rawScores,
        int bulkSize
    ) throws IOException {
        final NeighborQueue neighborQueue = new NeighborQueue(numCentroids, true);
        final long centroidQuantizeSize = fieldInfo.getVectorDimension() + 3 * Float.BYTES + Integer.BYTES;
        score(
            neighborQueue,
            numCentroids,
            0,
            scorer,
            centroids,
            centroidQuantizeSize,
            quantizeQuery,
            queryParams,
            globalCentroidDp,
            fieldInfo.getVectorSimilarityFunction(),
            new float[bulkSize],
            acceptCentroids,
            lensMult,
            lensAdd,
            rawScores,
            bulkSize
        );
        long offset = centroids.getFilePointer();
        return new CentroidIterator() {
            @Override
            public boolean hasNext() {
                return neighborQueue.size() > 0;
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                long centroidOrdinalAndScore = neighborQueue.popRaw();
                int centroidOrd = neighborQueue.decodeNodeId(centroidOrdinalAndScore);
                // the queue may be ordered by a density-boosted key; PostingMetadata must carry the raw
                // content score because it is inverted downstream for scoring corrections
                float score = rawScores != null ? rawScores[centroidOrd] : neighborQueue.decodeScore(centroidOrdinalAndScore);
                centroids.seek(offset + (long) Long.BYTES * 2 * centroidOrd);
                long postingListOffset = centroids.readLong();
                long postingListLength = centroids.readLong();
                // NO_ORDINAL indicates that the global centroid should be used for query quantization
                return new PostingMetadata(postingListOffset, postingListLength, NO_ORDINAL, score);
            }
        };
    }

    private static CentroidIterator getCentroidIteratorWithParents(
        FieldInfo fieldInfo,
        IndexInput centroids,
        int numParents,
        int numCentroids,
        ES92Int7VectorsScorer scorer,
        byte[] quantizeQuery,
        OptimizedScalarQuantizer.QuantizationResult queryParams,
        float globalCentroidDp,
        float centroidRatio,
        FixedBitSet acceptParents,
        FixedBitSet acceptCentroids,
        float[] lensMult,
        float[] lensAdd,
        float[] rawScores,
        int bulkSize
    ) throws IOException {
        // build the three queues we are going to use
        final long rawParentSize = (long) fieldInfo.getVectorDimension() * Float.BYTES;
        final long centroidQuantizeSize = fieldInfo.getVectorDimension() + 3 * Float.BYTES + Integer.BYTES;
        final NeighborQueue parentsQueue = new NeighborQueue(numParents, true);
        final int maxChildrenSize = centroids.readVInt();
        final NeighborQueue currentParentQueue = new NeighborQueue(maxChildrenSize, true);
        final int bufferSize = (int) Math.clamp(centroidRatio * numCentroids, 1, numCentroids);
        final int numCentroidsFiltered = acceptCentroids == null ? numCentroids : acceptCentroids.cardinality();
        if (numCentroidsFiltered == 0) {
            // TODO maybe this makes CentroidIterator polymorphic?
            return new CentroidIterator() {
                @Override
                public boolean hasNext() {
                    return false;
                }

                @Override
                public PostingMetadata nextPosting() {
                    return null;
                }
            };
        }
        final float[] scores = new float[bulkSize];
        final NeighborQueue neighborQueue;
        if (acceptCentroids != null && numCentroidsFiltered <= bufferSize) {
            // we are collecting every non-filter centroid, therefore we do not need to score the
            // parents. We give each of them the same score.
            neighborQueue = new NeighborQueue(numCentroidsFiltered, true);
            for (int i = 0; i < numParents; i++) {
                if (acceptParents == null || acceptParents.get(i)) {
                    parentsQueue.add(i, 0.5f);
                }
            }
            centroids.skipBytes((centroidQuantizeSize + rawParentSize) * numParents);
        } else {
            neighborQueue = new NeighborQueue(bufferSize, true);
            // score the parents (never lens-boosted: the lens is defined per child centroid)
            centroids.skipBytes(rawParentSize * numParents);
            score(
                parentsQueue,
                numParents,
                0,
                scorer,
                centroids,
                centroidQuantizeSize,
                quantizeQuery,
                queryParams,
                globalCentroidDp,
                fieldInfo.getVectorSimilarityFunction(),
                scores,
                acceptParents,
                null,
                null,
                null,
                bulkSize
            );
        }

        final long offset = centroids.getFilePointer();
        final long childrenOffset = offset + (long) Long.BYTES * numParents;
        // populate the children's queue by reading parents one by one
        while (parentsQueue.size() > 0 && neighborQueue.size() < bufferSize) {
            final int pop = parentsQueue.pop();
            populateOneChildrenGroup(
                currentParentQueue,
                centroids,
                offset + 2L * Integer.BYTES * pop,
                childrenOffset,
                centroidQuantizeSize,
                fieldInfo,
                scorer,
                quantizeQuery,
                queryParams,
                globalCentroidDp,
                scores,
                acceptCentroids,
                lensMult,
                lensAdd,
                rawScores,
                bulkSize
            );
            while (currentParentQueue.size() > 0 && neighborQueue.size() < bufferSize) {
                final float score = currentParentQueue.topScore();
                final int children = currentParentQueue.pop();
                neighborQueue.add(children, score);
            }
        }
        final long childrenFileOffsets = childrenOffset + centroidQuantizeSize * numCentroids;
        return new CentroidIterator() {

            @Override
            public boolean hasNext() {
                return neighborQueue.size() > 0;
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                long centroidOrdinalAndScore = nextCentroid();
                int centroidOrdinal = neighborQueue.decodeNodeId(centroidOrdinalAndScore);
                // the queue may be ordered by a density-boosted key; PostingMetadata must carry the raw
                // content score because it is inverted downstream for scoring corrections
                float score = rawScores != null ? rawScores[centroidOrdinal] : neighborQueue.decodeScore(centroidOrdinalAndScore);
                centroids.seek(childrenFileOffsets + (long) (Long.BYTES * 2 + Integer.BYTES) * centroidOrdinal);
                long postingListOffset = centroids.readLong();
                long postingListLength = centroids.readLong();
                int parentOrd = centroids.readInt();
                return new PostingMetadata(postingListOffset, postingListLength, parentOrd, score);
            }

            private long nextCentroid() throws IOException {
                if (currentParentQueue.size() > 0) {
                    // return next centroid and maybe add a children from the current parent queue
                    return neighborQueue.popRawAndAddRaw(currentParentQueue.popRaw());
                } else if (parentsQueue.size() > 0) {
                    // current parent queue is empty, populate it again with the next parent
                    int pop = parentsQueue.pop();
                    populateOneChildrenGroup(
                        currentParentQueue,
                        centroids,
                        offset + 2L * Integer.BYTES * pop,
                        childrenOffset,
                        centroidQuantizeSize,
                        fieldInfo,
                        scorer,
                        quantizeQuery,
                        queryParams,
                        globalCentroidDp,
                        scores,
                        acceptCentroids,
                        lensMult,
                        lensAdd,
                        rawScores,
                        bulkSize
                    );
                    return nextCentroid();
                } else {
                    return neighborQueue.popRaw();
                }
            }
        };
    }

    private static void populateOneChildrenGroup(
        NeighborQueue neighborQueue,
        IndexInput centroids,
        long parentOffset,
        long childrenOffset,
        long centroidQuantizeSize,
        FieldInfo fieldInfo,
        ES92Int7VectorsScorer scorer,
        byte[] quantizeQuery,
        OptimizedScalarQuantizer.QuantizationResult queryParams,
        float globalCentroidDp,
        float[] scores,
        FixedBitSet acceptCentroids,
        float[] lensMult,
        float[] lensAdd,
        float[] rawScores,
        int bulkSize
    ) throws IOException {
        centroids.seek(parentOffset);
        int childrenOrdinal = centroids.readInt();
        int numChildren = centroids.readInt();
        centroids.seek(childrenOffset + centroidQuantizeSize * childrenOrdinal);
        score(
            neighborQueue,
            numChildren,
            childrenOrdinal,
            scorer,
            centroids,
            centroidQuantizeSize,
            quantizeQuery,
            queryParams,
            globalCentroidDp,
            fieldInfo.getVectorSimilarityFunction(),
            scores,
            acceptCentroids,
            lensMult,
            lensAdd,
            rawScores,
            bulkSize
        );
    }

    private static void score(
        NeighborQueue neighborQueue,
        int size,
        int scoresOffset,
        ES92Int7VectorsScorer scorer,
        IndexInput centroids,
        long centroidQuantizeSize,
        byte[] quantizeQuery,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        float centroidDp,
        VectorSimilarityFunction similarityFunction,
        float[] scores,
        FixedBitSet acceptCentroids,
        float[] lensMult,
        float[] lensAdd,
        float[] rawScores,
        int bulkSize
    ) throws IOException {
        int limit = size - bulkSize + 1;
        int i = 0;
        for (; i < limit; i += bulkSize) {
            if (acceptCentroids == null || acceptCentroids.cardinality(scoresOffset + i, scoresOffset + i + bulkSize) > 0) {
                scorer.scoreBulk(
                    quantizeQuery,
                    queryCorrections.lowerInterval(),
                    queryCorrections.upperInterval(),
                    queryCorrections.quantizedComponentSum(),
                    queryCorrections.additionalCorrection(),
                    similarityFunction,
                    centroidDp,
                    scores,
                    bulkSize
                );
                for (int j = 0; j < bulkSize; j++) {
                    int centroidOrd = scoresOffset + i + j;
                    if (acceptCentroids == null || acceptCentroids.get(centroidOrd)) {
                        neighborQueue.add(
                            centroidOrd,
                            orderingKey(scores[j], centroidOrd, lensMult, lensAdd, rawScores, similarityFunction)
                        );
                    }
                }
            } else {
                centroids.skipBytes(bulkSize * centroidQuantizeSize);
            }
        }

        int tailBulkSize = size - i;
        if (tailBulkSize > 0) {
            if (acceptCentroids == null || acceptCentroids.cardinality(scoresOffset + i, scoresOffset + i + tailBulkSize) > 0) {
                scorer.scoreBulk(
                    quantizeQuery,
                    queryCorrections.lowerInterval(),
                    queryCorrections.upperInterval(),
                    queryCorrections.quantizedComponentSum(),
                    queryCorrections.additionalCorrection(),
                    similarityFunction,
                    centroidDp,
                    scores,
                    tailBulkSize
                );
                for (int j = 0; j < tailBulkSize; j++) {
                    int centroidOrd = scoresOffset + i + j;
                    if (acceptCentroids == null || acceptCentroids.get(centroidOrd)) {
                        neighborQueue.add(
                            centroidOrd,
                            orderingKey(scores[j], centroidOrd, lensMult, lensAdd, rawScores, similarityFunction)
                        );
                    }
                }
            } else {
                centroids.skipBytes(tailBulkSize * centroidQuantizeSize);
            }
        }
    }

    /**
     * Computes the queue ordering key for a centroid. Two transforms exist, mutually exclusive:
     * <ul>
     *   <li>Lens (see {@link AttributeLensIndex}): the key becomes
     *       {@code mult[ord] * sim + add[ord]}, the upper bound on the query's similarity to the
     *       filter value's sub-centroid in this cluster, where {@code sim} is the raw similarity
     *       recovered from the centroid score.</li>
     *   <li>Sampled match density (mid-selectivity fused-clustering mode): a multiplicative
     *       boost.</li>
     * </ul>
     * Either way the raw content score is recorded in {@code rawScores} so
     * {@link PostingMetadata} carries the exact score for downstream corrections. With an
     * uninformative filter the lens coefficients shrink to (1, 0) and the ordering reduces to the
     * content ordering.
     */
    private static float orderingKey(
        float score,
        int centroidOrd,
        float[] lensMult,
        float[] lensAdd,
        float[] rawScores,
        VectorSimilarityFunction similarityFunction
    ) {
        if (lensMult != null) {
            rawScores[centroidOrd] = score;
            // invert the score transform to raw similarity (see resetPostingsScorer); the lens is
            // only engaged for dot-product-family similarities
            final float sim = switch (similarityFunction) {
                case COSINE, DOT_PRODUCT -> 2 * score - 1;
                case MAXIMUM_INNER_PRODUCT -> score - 1;
                case EUCLIDEAN -> score;
            };
            return lensMult[centroidOrd] * sim + lensAdd[centroidOrd];
        }
        return score;
    }
}
