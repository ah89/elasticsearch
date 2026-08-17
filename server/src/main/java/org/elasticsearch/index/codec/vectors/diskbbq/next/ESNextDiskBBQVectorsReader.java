/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CalibrationAwareReader;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfAutoCalibration;
import org.elasticsearch.index.codec.vectors.diskbbq.MatchFetchIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.PrefetchingCentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer.DEFAULT_LAMBDA;
import static org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata.NO_ORDINAL;
import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * Default implementation of {@link IVFVectorsReader}. It scores the posting lists centroids using
 * brute force and then scores the top ones using the posting list.
 */
public class ESNextDiskBBQVectorsReader extends IVFVectorsReader<ESNextDiskBBQVectorsReader.NextFieldEntry>
    implements
        VectorPreconditioner,
        CalibrationAwareReader {

    public ESNextDiskBBQVectorsReader(SegmentReadState state, GenericFlatVectorReaders.LoadFlatVectorsReader getFormatReader)
        throws IOException {
        super(
            state,
            getFormatReader,
            ESNextDiskBBQVectorsFormat.NAME,
            ESNextDiskBBQVectorsFormat.CENTROID_EXTENSION,
            ESNextDiskBBQVectorsFormat.CLUSTER_EXTENSION,
            ESNextDiskBBQVectorsFormat.IVF_META_EXTENSION,
            ESNextDiskBBQVectorsFormat.VERSION_START,
            ESNextDiskBBQVectorsFormat.VERSION_CURRENT,
            ESNextDiskBBQVectorsFormat.VERSION_DIRECT_IO,
            ESNextDiskBBQVectorsFormat.DYNAMIC_VISIT_RATIO
        );
    }

    /**
     * Maximum matches scored per cluster visit, as a multiple of {@code k}; 0 drains each cluster.
     * See the use site for why breadth can beat depth under a tight budget.
     */
    private static final int FETCH_PER_CLUSTER_CAP_K = Integer.parseInt(System.getProperty("es.poc.fetch.capK", "0"));

    /**
     * How the fetch path spends its visit allowance. All three are charged in the same unit the
     * scan-based gates use -- posting-list entries -- so the number is comparable across arms.
     *
     * <p>MATCHES charges only the documents that pass the filter. Non-matching entries are free,
     * which is what the fetch mechanics actually deliver, but it means the arm does far more work
     * than a scan given the same allowance and so cannot be compared against one at equal settings.
     *
     * <p>ENTRIES charges every entry of every cluster opened, matching or not. This is exactly the
     * scan's rule, so both arms stop after covering the same amount of posting data -- but it hands
     * back none of the savings from skipping non-matches, which is the fetch path's whole point.
     *
     * <p>BLOCKS charges the entries of each 32-document block the fetch actually reads, and nothing
     * for blocks it skips entirely. Both paths read posting data in 32-document blocks, and a scan
     * reads every block of every cluster it opens, so for the scan BLOCKS and ENTRIES are the same
     * number: its budget is untouched. For the fetch they diverge exactly by the blocks it managed
     * to skip. That makes it the honest middle ground. Where the filter is dense the fetch skips
     * nothing, is charged the full cluster, and stops where the scan stops. Where the filter is
     * sparse it skips most of each cluster and earns proportionally more clusters for the same
     * bytes read. The compensation for selective filters is earned in proportion to work genuinely
     * avoided, rather than granted by a separate hand-tuned phase as the scan does.
     */
    private enum BudgetMode {
        MATCHES,
        ENTRIES,
        BLOCKS
    }

    /**
     * Per-thread set of documents already scored this query, so each is scored exactly once however
     * many clusters store a copy of it.
     *
     * <p>The scan has no such set: it scores every copy it meets and collects each one, so a
     * SOAR-overspilled document can take two slots in the collector. That is why a k=10 search can
     * come back with 8 distinct documents. Scoring once is both cheaper and strictly better formed.
     *
     * <p>Which copy gets scored is then whichever cluster is visited first, and the two do not score
     * alike -- one document's copies estimated 0.374 and 0.295 on a merged segment, because k-means
     * puts a vector in the cluster that minimizes its residual, making the owning cluster's copy the
     * better-quantized one. Preferring the owning copy was tried and is worse overall: it means
     * scoring a document again when its owning cluster turns up later, which reintroduces exactly
     * the duplicate collector entries this set exists to remove.
     */
    private static final ThreadLocal<FixedBitSet> VISITED_SCRATCH = new ThreadLocal<>();

    /** Every copy after the first is redundant: the same vector, already scored this query. */
    private static boolean worthScoring(MatchFetchIndex.MatchView view, int at, FixedBitSet visited) {
        return visited.get(view.docs()[at]) == false;
    }

    private static FixedBitSet visitedScratch(int maxDoc) {
        FixedBitSet set = VISITED_SCRATCH.get();
        if (set == null || set.length() < maxDoc) {
            set = new FixedBitSet(maxDoc);
            VISITED_SCRATCH.set(set);
        } else {
            set.clear();
        }
        return set;
    }

    private static final BudgetMode FETCH_BUDGET_MODE = BudgetMode.valueOf(
        System.getProperty("es.poc.fetch.budget", "matches").toUpperCase(Locale.ROOT)
    );

    CentroidIterator getPostingListPrefetchIterator(CentroidIterator centroidIterator, IndexInput postingListSlice) throws IOException {
        // TODO we may want to prefetch more than one postings list, however, we will likely want to place a limit
        // so we don't bother prefetching many lists we won't end up scoring
        return new PrefetchingCentroidIterator(centroidIterator, postingListSlice);
    }

    @Override
    protected int getNumberOfVectors(NextFieldEntry entry, FloatVectorValues values, IndexInput centroidSlice, ESAcceptDocs esAcceptDocs)
        throws IOException {
        int size = values.size();
        assert esAcceptDocs == null
            || entry.numSlices >= 0 && esAcceptDocs.sliceOrd() >= 0
            || entry.numSlices == -1 && esAcceptDocs.sliceOrd() == -1;
        if (entry.numSlices > 0) {
            long fp = centroidSlice.getFilePointer();
            final int bitsRequired = DirectWriter.bitsRequired(entry.maxSliceSize);
            final long sizeLookup = DirectWriter.bytesRequired(entry.numSlices, bitsRequired);
            if (esAcceptDocs != null) {
                int sliceOrd = esAcceptDocs.sliceOrd();
                assert sliceOrd < entry.numSlices : "sliceOrd out of range for centroid slices";
                final LongValues longValues = DirectReader.getInstance(centroidSlice.randomAccessSlice(fp, sizeLookup), bitsRequired);
                size = (int) longValues.get(sliceOrd);
            }
            centroidSlice.seek(fp + sizeLookup);
        }
        return size;
    }

    @Override
    public float getOversampleFactor(FieldInfo fieldInfo) {
        final NextFieldEntry e = fields.get(fieldInfo.number);
        if (e == null) {
            return IvfAutoCalibration.NO_CALIBRATED_OVERSAMPLE;
        }
        float r = e.rescoreOversample();
        return Float.isFinite(r) ? r : IvfAutoCalibration.NO_CALIBRATED_OVERSAMPLE;
    }

    @Override
    public boolean shouldPrecondition(FieldInfo fieldInfo) {
        final NextFieldEntry e = fields.get(fieldInfo.number);
        return e != null && e.preconditionerLength() > 0;
    }

    @Override
    public QuantEncoding getQuantEncoding(FieldInfo fieldInfo) {
        final NextFieldEntry e = fields.get(fieldInfo.number);
        return e == null ? null : e.quantEncoding();
    }

    // visible for testing
    NextFieldEntry fieldEntry(String fieldName) {
        final FieldInfo info = fieldInfos.fieldInfo(fieldName);
        return info == null ? null : fields.get(info.number);
    }

    @Override
    public CentroidIterator getCentroidIterator(
        FieldInfo fieldInfo,
        int numCentroids,
        IndexInput centroids,
        float[] targetQuery,
        IndexInput postingListSlice,
        AcceptDocs acceptDocs,
        float approximateCost,
        FloatVectorValues values,
        float visitRatio
    ) throws IOException {
        ESNextDiskBBQVectorsReader.NextFieldEntry fieldEntry = fields.get(fieldInfo.number);
        var iterator = switch (fieldEntry.centroidIndexFormat()) {
            case FLAT -> new FlatCentroidIndex(
                fieldInfo,
                fieldEntry,
                numCentroids,
                centroids,
                targetQuery,
                acceptDocs,
                approximateCost,
                values,
                visitRatio
            ).getIterator();
        };
        return getPostingListPrefetchIterator(iterator, postingListSlice);
    }

    @Override
    protected NextFieldEntry doReadField(
        IndexInput input,
        String rawVectorFormat,
        boolean useDirectIOReads,
        VectorSimilarityFunction similarityFunction,
        VectorEncoding vectorEncoding,
        int numCentroids,
        long centroidOffset,
        long centroidLength,
        long postingListOffset,
        long postingListLength,
        float[] globalCentroid,
        float globalCentroidDp
    ) throws IOException {
        int bulkSize = input.readInt();
        CentroidIndexFormat centroidIndexFormat = CentroidIndexFormat.fromId(input.readInt());
        QuantEncoding quantEncoding = QuantEncoding.fromId(input.readInt());
        long preconditionerLength = input.readLong();
        long preconditionerOffset = -1;
        if (preconditionerLength > 0) {
            preconditionerOffset = input.readLong();
        }
        int numSlices = input.readInt();
        int maxSliceSize = 0;
        if (numSlices > 0) {
            maxSliceSize = input.readVInt();
        }
        float rescoreOversample = Float.intBitsToFloat(input.readInt());
        return new NextFieldEntry(
            rawVectorFormat,
            useDirectIOReads,
            similarityFunction,
            vectorEncoding,
            numCentroids,
            centroidOffset,
            centroidLength,
            postingListOffset,
            postingListLength,
            globalCentroid,
            globalCentroidDp,
            centroidIndexFormat,
            quantEncoding,
            bulkSize,
            preconditionerOffset,
            preconditionerLength,
            numSlices,
            maxSliceSize,
            rescoreOversample
        );
    }

    @Override
    public Preconditioner getPreconditioner(FieldInfo fieldInfo) throws IOException {
        final NextFieldEntry fieldEntry = fields.get(fieldInfo.number);
        // only seems possible in tests
        if (fieldEntry == null) {
            return null;
        }
        long preconditionerOffset = fieldEntry.preconditionerOffset();
        long preconditionerLength = fieldEntry.preconditionerLength();
        if (preconditionerLength > 0) {
            IndexInput ivfPreconditionerSlice = ivfCentroids.slice("preconditioner", preconditionerOffset, preconditionerLength);
            if (ivfPreconditionerSlice != null) {
                ivfPreconditionerSlice.seek(0);
                return Preconditioner.read(ivfPreconditionerSlice);
            }
        }
        return null;
    }

    @Override
    public CentroidData readCentroidData(String fieldName) throws IOException {
        FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
        if (fieldInfo == null) {
            return null;
        }
        NextFieldEntry entry = fields.get(fieldInfo.number);
        if (entry == null || entry.numCentroids() == 0) {
            return null;
        }
        int dimension = fieldInfo.getVectorDimension();
        int numCentroids = entry.numCentroids();
        FloatVectorValues vectorValues = getFloatVectorValues(fieldInfo.name);
        int numVectors = vectorValues != null ? vectorValues.size() : 0;
        int[] clusterSizes = new int[numCentroids];

        long rawCentroidsSize = (long) numCentroids * dimension * Float.BYTES;
        IndexInput centroidsSlice = null;
        boolean success = false;
        try (IndexInput centroidSlice = entry.centroidSlice(ivfCentroids); IndexInput postingSlice = entry.postingListSlice(ivfClusters)) {
            long[] postingOffsets = readPostingListOffsets(centroidSlice, numVectors, numCentroids, dimension);

            // First pass: read cluster sizes only (from the posting slice).
            for (int c = 0; c < numCentroids; c++) {
                postingSlice.seek(postingOffsets[c] + Integer.BYTES);
                clusterSizes[c] = postingSlice.readVInt();
            }

            // The raw centroids live contiguously at the end of the centroid data; slice that
            // region and hand it to the streaming view. The slice owns its own resources and
            // outlives the parent centroidSlice.
            long centroidsOffset = centroidSlice.length() - rawCentroidsSize;
            centroidsSlice = centroidSlice.slice("centroids-raw", centroidsOffset, rawCentroidsSize);
            KMeansFloatVectorValues centroids = KMeansFloatVectorValues.build(centroidsSlice, null, numCentroids, dimension);
            CentroidData data = new CentroidData(centroids, clusterSizes, entry.globalCentroid(), centroidsSlice);
            success = true;
            return data;
        } finally {
            if (success == false && centroidsSlice != null) {
                centroidsSlice.close();
            }
        }
    }

    /**
     * FetchGate (POC): scores the filter's matching documents by fetching their quantized bytes
     * directly out of the posting lists, in centroid-distance order, with the visit budget counted
     * in matches scored. See {@link MatchFetchIndex} for the structures and the rationale. Declines
     * (returns {@code false}) on layouts the geometry does not model — sliced segments, non-flat
     * centroid indexes — and when the filter exposes no random-access bits; the caller then runs
     * the standard gated search.
     */
    @Override
    protected boolean fetchSearch(
        FieldInfo fieldInfo,
        NextFieldEntry entry,
        FloatVectorValues values,
        float[] target,
        AcceptDocs acceptDocs,
        KnnCollector knnCollector,
        float visitRatio,
        int numVectors
    ) throws IOException {
        if (entry.numSlices != -1 || entry.centroidIndexFormat() != CentroidIndexFormat.FLAT || entry.numCentroids() <= 1) {
            return false;
        }
        final Bits bits = acceptDocs.bits();
        if (bits == null) {
            return false;
        }
        final int numCentroids = entry.numCentroids();
        MatchFetchIndex fetchIndex = entry.matchFetchIndex();
        if (fetchIndex == null) {
            synchronized (entry) {
                fetchIndex = entry.matchFetchIndex();
                if (fetchIndex == null) {
                    fetchIndex = buildMatchFetchIndex(fieldInfo, entry, values);
                    entry.matchFetchIndex(fetchIndex);
                    if (fetchIndex != MatchFetchIndex.UNSUPPORTED) {
                        FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(fetchIndex.ramBytesUsed());
                    }
                }
            }
        }
        if (fetchIndex == MatchFetchIndex.UNSUPPORTED) {
            return false;
        }
        // Per-filter match view, memoized with the same content-verified fingerprint discipline as
        // the counted gate's counts cache; a miss pays one O(matches) walk — small by definition in
        // the selective regime where this path fires.
        MatchFetchIndex.MatchView view = null;
        // Key the match-view cache off the query's value set when one is available, exactly as
        // Mariano's gate does. The content fingerprint below has to hash the filter bitset's words
        // on every query (~1,563 words for a 100k-doc segment) purely to find a cache entry — work
        // the scan-based gate never performs, and enough to account for the last few percent of
        // throughput on a mid-selectivity workload. A value set identifies the filter exactly, so
        // no verification pass is needed either.
        final int[] cacheValueSet = FlatCentroidIndex.POC_QUERY_VALUE_SET.get();
        final Long fingerprint = cacheValueSet != null && cacheValueSet.length > 0
            ? (long) Arrays.hashCode(cacheValueSet) ^ 0x5DEECE66DL
            : FlatCentroidIndex.filterFingerprint(bits, acceptDocs);
        if (fingerprint != null) {
            final FieldEntry.CachedMatchView cached = entry.matchViewCache().get(fingerprint);
            if (cached != null && (cacheValueSet != null || FlatCentroidIndex.sameFilter(cached.source().get(), bits))) {
                view = cached.view();
            }
        }
        if (view == null) {
            final KnnVectorValues.DocIndexIterator docIndexIterator = values.iterator();
            final DocIdSetIterator matching = ConjunctionUtils.intersectIterators(List.of(acceptDocs.iterator(), docIndexIterator));
            final IndexInput centroidSlice = entry.centroidSlice(ivfCentroids);
            final int bitsRequired = DirectWriter.bitsRequired(numCentroids);
            final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
            final LongValues centroidOfOrdinal = DirectReader.getInstance(centroidSlice.randomAccessSlice(0, sizeLookup), bitsRequired);
            view = MatchFetchIndex.MatchView.build(
                matching,
                docIndexIterator,
                centroidOfOrdinal,
                fetchIndex,
                numCentroids,
                Math.max(0, acceptDocs.cost())
            );
            if (fingerprint != null) {
                if (entry.matchViewCache().put(fingerprint, new FieldEntry.CachedMatchView(new WeakReference<>(bits), view)) == null) {
                    FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(FieldEntry.CACHE_ENTRY_OVERHEAD + view.ramBytesUsed());
                }
            }
        }
        if (view.totalMatches() == 0) {
            // no matching document carries a vector: the empty result is the correct result
            return true;
        }
        runFetch(fieldInfo, entry, values, target, acceptDocs, knnCollector, visitRatio, numVectors, fetchIndex, view);
        return true;
    }

    private void runFetch(
        FieldInfo fieldInfo,
        NextFieldEntry entry,
        FloatVectorValues values,
        float[] target,
        AcceptDocs acceptDocs,
        KnnCollector knnCollector,
        float visitRatio,
        int numVectors,
        MatchFetchIndex fetchIndex,
        MatchFetchIndex.MatchView view
    ) throws IOException {
        // The budget is counted in matches scored — the only per-document work this path performs.
        // The 2x mirrors maxVectorsToVisit's SOAR allowance: the scan's cap is 2 * ratio * N posting
        // entries where a document can appear twice, and in dense match regions (filter aligned with
        // the query) nearly every entry it visits is a match. Scoring each document exactly once,
        // the fetch needs the same doubled cap to see as many distinct matches as the scan does.
        final long matchBudget = Math.max(knnCollector.k(), (long) Math.ceil(2.0 * visitRatio * numVectors));
        final BudgetMode budgetMode = FETCH_BUDGET_MODE;
        // 0 disables the cap and drains each cluster fully (the original depth-first behaviour)
        final int perClusterCap = FETCH_PER_CLUSTER_CAP_K > 0 ? FETCH_PER_CLUSTER_CAP_K * knnCollector.k() : 0;
        final QuantEncoding quantEncoding = entry.quantEncoding();
        final VectorSimilarityFunction similarityFunction = fieldInfo.getVectorSimilarityFunction();
        final long quantizedVectorByteSize = quantEncoding.getDocPackedLength(fieldInfo.getVectorDimension());

        // Distance-ordered iterator over exactly the posting lists that hold >= 1 match: the accept
        // set is handed to the centroid index precomputed (it falls out of the match view), so the
        // ordering machinery is identical to the gated scan's.
        final CentroidIterator centroidIterator;
        FlatCentroidIndex.POC_FETCH_ACCEPT_CENTROIDS.set(view.acceptCentroids());
        try {
            centroidIterator = new FlatCentroidIndex(
                fieldInfo,
                entry,
                entry.numCentroids(),
                entry.centroidSlice(ivfCentroids),
                target,
                acceptDocs,
                view.totalMatches(),
                values,
                visitRatio
            ).getIterator();
        } finally {
            FlatCentroidIndex.POC_FETCH_ACCEPT_CENTROIDS.remove();
        }

        // query quantization per query-centroid ordinal, exactly as the posting visitor does
        final IndexInput centroidSlice = entry.centroidSlice(ivfCentroids);
        final int bitsRequired = DirectWriter.bitsRequired(entry.numCentroids());
        centroidSlice.skipBytes(DirectWriter.bytesRequired(values.size(), bitsRequired));
        final int numParents = centroidSlice.readVInt();
        final QueryQuantizer queryQuantizer;
        if (numParents > 0) {
            centroidSlice.readVInt(); // longest posting list, unused
            final IndexInput parentsSlice = centroidSlice.slice(
                "parents-slice",
                centroidSlice.getFilePointer(),
                (long) numParents * fieldInfo.getVectorDimension() * Float.BYTES
            );
            queryQuantizer = new QueryQuantizer(quantEncoding, fieldInfo, target, parentsSlice, entry.globalCentroid());
        } else {
            queryQuantizer = new QueryQuantizer(quantEncoding, fieldInfo, target, null, entry.globalCentroid());
        }

        final IndexInput postingSlice = entry.postingListSlice(ivfClusters);
        final ES940OSQVectorsScorer scorer = ESVectorUtil.getES940OSQVectorsScorer(
            postingSlice,
            quantEncoding.queryBits(),
            quantEncoding.bits(),
            fieldInfo.getVectorDimension(),
            (int) quantizedVectorByteSize,
            BULK_SIZE,
            quantEncoding.bits() == 2 || quantEncoding.bits() == 4
                ? ES940OSQVectorsScorer.BitEncoding.PACKED
                : ES940OSQVectorsScorer.BitEncoding.STRIPED
        );
        final float[] scoreScratch = new float[BULK_SIZE];
        final int[] slotScratch = new int[BULK_SIZE];

        final FixedBitSet visited = visitedScratch(fetchIndex.maxDoc());
        long scored = 0;
        long budgetSpent = 0;
        while (centroidIterator.hasNext()
            && (budgetSpent < matchBudget || knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {
            final PostingMetadata metadata = centroidIterator.nextPosting();
            final int centroid = fetchIndex.centroidForOffset(metadata.offset());
            assert centroid >= 0 : "posting offset not in fetch geometry";
            final int from = view.csrOffsets()[centroid];
            int to = view.csrOffsets()[centroid + 1];
            if (from == to) {
                continue;
            }
            // Breadth-first budget: a cluster holding hundreds of matches would otherwise drain the
            // whole budget before the next-nearest cluster is opened at all, even though a cluster
            // with a single match may well hold the nearest neighbour. Capping how many of a
            // cluster's matches are scored in one visit spreads the budget over more of the
            // distance-ordered clusters. The cap is a multiple of k because k sets how many results
            // any one cluster could plausibly contribute.
            if (perClusterCap > 0 && to - from > perClusterCap) {
                to = from + perClusterCap;
            }
            postingSlice.seek(metadata.offset());
            final float centroidToParentSqDist = Float.intBitsToFloat(postingSlice.readInt());
            final int postingEntries = postingSlice.readVInt();
            if (budgetMode == BudgetMode.ENTRIES) {
                budgetSpent += postingEntries;
            }
            final float rawScore = metadata.documentCentroidScore();
            // identical raw-similarity reconstruction to the posting visitor's resetPostingsScorer
            final float centroidDistance = switch (similarityFunction) {
                case EUCLIDEAN -> ((1 / rawScore) - 1) - centroidToParentSqDist;
                case COSINE, DOT_PRODUCT -> 2 * rawScore - 1;
                case MAXIMUM_INNER_PRODUCT -> rawScore - 1;
            };
            queryQuantizer.reset(metadata.queryCentroidOrdinal());
            queryQuantizer.quantizeQueryIfNecessary();
            final OptimizedScalarQuantizer.QuantizationResult queryCorrections = queryQuantizer.getQueryCorrections();
            final byte[] quantizedTarget = queryQuantizer.getQuantizedTarget();
            // Matches within a cluster are in posting order, so same-block matches are consecutive.
            // Score a whole block in one call: quantizeScoreBulkOffsets walks the block once,
            // skipping unselected vectors with skipBytes rather than reading them, and the four
            // correction columns are read sequentially in bulk. That replaces the previous five
            // scattered seeks per matching document with a single seek per block, which is where
            // the scan-based gates were winning on throughput despite scoring far more documents.
            int i = from;
            int clusterScored = 0;
            while (i < to) {
                final int firstDoc = view.docs()[i];
                assert fetchIndex.hasDoc(firstDoc) : "match view names a doc the fetch geometry does not know";
                final long blockPtr = fetchIndex.blockOffset(firstDoc, view.occurrence()[i]);
                final int count = fetchIndex.blockCount(firstDoc, view.occurrence()[i]);
                int end = i + 1;
                while (end < to && fetchIndex.blockOffset(view.docs()[end], view.occurrence()[end]) == blockPtr) {
                    end++;
                }
                // Score the block's matching slots. quantizeScoreBulkOffsets scores each selected
                // document with the scalar kernel but SKIPS the rest with skipBytes rather than
                // reading them, and pulls the four correction columns sequentially. Scoring the
                // whole block with the SIMD kernel instead was measured 1.8x SLOWER here: at one or
                // two matches per block the 8 KB of vector bytes it must read dominates the cheaper
                // arithmetic. Memory traffic, not the popcount, is the binding cost.
                int numSlots = 0;
                for (int j = i; j < end; j++) {
                    if (worthScoring(view, j, visited)) {
                        // ascending by construction: doc order within a block is slot order
                        slotScratch[numSlots++] = fetchIndex.slot(view.docs()[j], view.occurrence()[j]);
                    }
                }
                if (numSlots == 0) {
                    i = end;
                    continue;
                }
                if (budgetMode == BudgetMode.BLOCKS) {
                    // charged for the whole block: reading one match in it pulls all of its bytes
                    budgetSpent += count;
                }
                postingSlice.seek(blockPtr);
                final float maxScore = scorer.scoreBulkOffsets(
                    quantizedTarget,
                    queryCorrections.lowerInterval(),
                    queryCorrections.upperInterval(),
                    queryCorrections.quantizedComponentSum(),
                    centroidDistance,
                    similarityFunction,
                    0f,
                    slotScratch,
                    numSlots,
                    scoreScratch,
                    count
                );
                final boolean competitive = knnCollector.minCompetitiveSimilarity() < maxScore;
                for (int j = i; j < end; j++) {
                    if (worthScoring(view, j, visited) == false) {
                        continue;
                    }
                    final int doc = view.docs()[j];
                    if (competitive) {
                        knnCollector.collect(doc, scoreScratch[fetchIndex.slot(doc, view.occurrence()[j])]);
                    }
                    visited.set(doc);
                    clusterScored++;
                }
                i = end;
            }
            scored += clusterScored;
            if (budgetMode == BudgetMode.MATCHES) {
                budgetSpent += clusterScored;
            }
            knnCollector.incVisitedCount(clusterScored);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }
    }

    @Override
    protected MatchFetchIndex matchFetchGeometry(FieldInfo fieldInfo, NextFieldEntry entry, FloatVectorValues values) throws IOException {
        MatchFetchIndex geometry = entry.matchFetchIndex();
        if (geometry == null) {
            synchronized (entry) {
                geometry = entry.matchFetchIndex();
                if (geometry == null) {
                    geometry = buildMatchFetchIndex(fieldInfo, entry, values);
                    entry.matchFetchIndex(geometry);
                    // charged here as well as on the fetch path: the lens needs this geometry to
                    // know each document's SOAR overspill cluster, so when the lens builds it first
                    // the heap belongs to the lens and must not go unreported
                    if (geometry != MatchFetchIndex.UNSUPPORTED) {
                        FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(geometry.ramBytesUsed());
                    }
                }
            }
        }
        return geometry == MatchFetchIndex.UNSUPPORTED ? null : geometry;
    }

    private MatchFetchIndex buildMatchFetchIndex(FieldInfo fieldInfo, NextFieldEntry entry, FloatVectorValues values) throws IOException {
        final int numCentroids = entry.numCentroids();
        // ordinals are in doc order, so the last ordinal's doc bounds the posting lists' doc space
        final int maxDoc = values.ordToDoc(values.size() - 1) + 1;
        final int[] docToOrdinal = new int[maxDoc];
        Arrays.fill(docToOrdinal, -1);
        final KnnVectorValues.DocIndexIterator it = values.iterator();
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            docToOrdinal[doc] = it.index();
        }
        final IndexInput centroidSlice = entry.centroidSlice(ivfCentroids);
        final int bitsRequired = DirectWriter.bitsRequired(numCentroids);
        final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
        final LongValues centroidOfOrdinal = DirectReader.getInstance(centroidSlice.randomAccessSlice(0, sizeLookup), bitsRequired);
        final long[] postingOffsets = readPostingListOffsets(
            entry.centroidSlice(ivfCentroids),
            values.size(),
            numCentroids,
            fieldInfo.getVectorDimension()
        );
        final MatchFetchIndex built = MatchFetchIndex.build(
            entry.postingListSlice(ivfClusters),
            postingOffsets,
            centroidOfOrdinal,
            docToOrdinal,
            maxDoc,
            BULK_SIZE,
            entry.quantEncoding().getDocPackedLength(fieldInfo.getVectorDimension())
        );
        return built == null ? MatchFetchIndex.UNSUPPORTED : built;
    }

    private static long[] readPostingListOffsets(IndexInput centroidSlice, int numVectors, int numCentroids, int dimension)
        throws IOException {
        long[] offsets = new long[numCentroids];
        int bitsRequired = DirectWriter.bitsRequired(numCentroids);
        long sizeLookup = DirectWriter.bytesRequired(numVectors, bitsRequired);
        centroidSlice.seek(sizeLookup);
        int numParents = centroidSlice.readVInt();
        long rawCentroidsSize = (long) numCentroids * dimension * Float.BYTES;
        long offsetTableEntrySize = numParents == 0 ? 2L * Long.BYTES : 2L * Long.BYTES + Integer.BYTES;
        long offsetTableStart = centroidSlice.length() - rawCentroidsSize - offsetTableEntrySize * numCentroids;

        centroidSlice.seek(offsetTableStart);
        for (int i = 0; i < numCentroids; i++) {
            offsets[i] = centroidSlice.readLong();
            centroidSlice.readLong();
            if (numParents > 0) {
                centroidSlice.readInt();
            }
        }
        return offsets;
    }

    public static class NextFieldEntry extends FieldEntry {
        private final CentroidIndexFormat centroidIndexFormat;
        private final QuantEncoding quantEncoding;
        protected final long preconditionerOffset;
        protected final long preconditionerLength;
        // -1 "not sliced".
        // 0 "sliced but on flush".
        // > 0 "sliced but on merge, is the number of slices".
        final int numSlices;
        final int maxSliceSize;
        private final float rescoreOversample;

        NextFieldEntry(
            String rawVectorFormat,
            boolean doDirectIOReads,
            VectorSimilarityFunction similarityFunction,
            VectorEncoding vectorEncoding,
            int numCentroids,
            long centroidOffset,
            long centroidLength,
            long postingListOffset,
            long postingListLength,
            float[] globalCentroid,
            float globalCentroidDp,
            CentroidIndexFormat centroidIndexFormat,
            QuantEncoding quantEncoding,
            int bulkSize,
            long preconditionerOffset,
            long preconditionerLength,
            int numSlices,
            int maxSliceSize,
            float rescoreOversample
        ) {
            super(
                rawVectorFormat,
                doDirectIOReads,
                similarityFunction,
                vectorEncoding,
                numCentroids,
                centroidOffset,
                centroidLength,
                postingListOffset,
                postingListLength,
                globalCentroid,
                globalCentroidDp,
                bulkSize
            );
            this.centroidIndexFormat = centroidIndexFormat;
            this.quantEncoding = quantEncoding;
            this.preconditionerOffset = preconditionerOffset;
            this.preconditionerLength = preconditionerLength;
            this.numSlices = numSlices;
            this.maxSliceSize = maxSliceSize;
            this.rescoreOversample = rescoreOversample;
        }

        public CentroidIndexFormat centroidIndexFormat() {
            return centroidIndexFormat;
        }

        public QuantEncoding quantEncoding() {
            return quantEncoding;
        }

        public long preconditionerOffset() {
            return preconditionerOffset;
        }

        public long preconditionerLength() {
            return preconditionerLength;
        }

        public float rescoreOversample() {
            return rescoreOversample;
        }

        @Override
        public int numSlices() {
            return numSlices;
        }

    }

    @Override
    protected long maxVectorsToVisit(NextFieldEntry entry, float visitRatio, int numVectors) {
        return switch (entry.centroidIndexFormat()) {
            case FLAT -> super.maxVectorsToVisit(entry, visitRatio, numVectors);
        };
    }

    @Override
    public PostingVisitor getPostingVisitor(
        FieldInfo fieldInfo,
        FloatVectorValues values,
        IndexInput indexInput,
        float[] target,
        Bits needsScoring,
        IndexInput centroidSlice,
        ESAcceptDocs acceptDocs
    ) throws IOException {
        NextFieldEntry entry = fields.get(fieldInfo.number);
        if (entry.numSlices > 0) {
            final int bitsRequired = DirectWriter.bitsRequired(entry.maxSliceSize);
            final long sizeLookup = DirectWriter.bytesRequired(entry.numSlices, bitsRequired);
            centroidSlice.skipBytes(sizeLookup);
        }
        final int bitsRequired = DirectWriter.bitsRequired(entry.numCentroids());
        final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
        centroidSlice.skipBytes(sizeLookup);
        QuantEncoding quantEncoding = entry.quantEncoding();
        int numParents = centroidSlice.readVInt();
        if (entry.numSlices > 0) {
            // skip slice offsets
            centroidSlice.skipBytes((long) entry.numSlices * Integer.BYTES);
        }
        final QueryQuantizer queryQuantizer;
        if (numParents > 0) {
            // unused
            int longestPostingList = centroidSlice.readVInt();
            IndexInput parentsSlice = centroidSlice.slice(
                "parents-slice",
                centroidSlice.getFilePointer(),
                (long) numParents * fieldInfo.getVectorDimension() * Float.BYTES
            );
            queryQuantizer = new QueryQuantizer(quantEncoding, fieldInfo, target, parentsSlice, entry.globalCentroid());
        } else {
            queryQuantizer = new QueryQuantizer(quantEncoding, fieldInfo, target, null, entry.globalCentroid());
        }
        if (entry.numSlices == 0) {
            // should only happen in sliced flushed segments
            assert entry.numCentroids() == 1;
            int startDoc;
            int endDoc;
            if (acceptDocs == null) {
                startDoc = 0;
                endDoc = values.ordToDoc(values.size() - 1) + 1;
            } else {
                ESAcceptDocs.SliceAcceptDocs sliceAcceptDocs = acceptDocs.sliceAcceptDocs();
                startDoc = sliceAcceptDocs.startDoc();
                endDoc = sliceAcceptDocs.endDoc();
            }
            return new SlicedMemorySegmentPostingsVisitor(
                queryQuantizer,
                quantEncoding,
                indexInput,
                entry,
                fieldInfo,
                needsScoring,
                values,
                startDoc,
                endDoc
            );

        } else {
            return new MemorySegmentPostingsVisitor(queryQuantizer, quantEncoding, indexInput, entry, fieldInfo, needsScoring);
        }
    }

    private record QueryQuantizerResult(OptimizedScalarQuantizer.QuantizationResult queryCorrections, byte[] quantizedTarget) {}

    private static final int QUERY_CACHE_SIZE = 16;

    private static class QueryQuantizer {
        private final LinkedHashMap<Integer, QueryQuantizerResult> cache;
        private final QuantEncoding quantEncoding;
        private final float[] target;
        private final float[] scratch;
        private final int[] quantizationScratch;
        private final OptimizedScalarQuantizer quantizer;
        private final IndexInput parentsSlice;
        private final float[] globalCentroid;
        private final float[] centroidScratch;
        private int currentCentroidOrdinal = -2;
        private int nextCentroidOrdinal = -1;
        private byte[] evictedQuantizedQuery = null;
        private QueryQuantizerResult result = null;

        QueryQuantizer(QuantEncoding quantEncoding, FieldInfo fieldInfo, float[] target, IndexInput parentsSlice, float[] globalCentroid) {
            this.quantEncoding = quantEncoding;
            this.target = target;
            this.scratch = new float[fieldInfo.getVectorDimension()];
            this.centroidScratch = new float[fieldInfo.getVectorDimension()];
            this.quantizationScratch = new int[quantEncoding.discretizedDimensions(fieldInfo.getVectorDimension())];
            this.quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction(), DEFAULT_LAMBDA, 1);
            this.parentsSlice = parentsSlice;
            this.globalCentroid = globalCentroid;
            this.cache = new LinkedHashMap<>(QUERY_CACHE_SIZE, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, QueryQuantizerResult> eldest) {
                    if (size() > QUERY_CACHE_SIZE) {
                        evictedQuantizedQuery = eldest.getValue().quantizedTarget();
                        return true;
                    }
                    return false;
                }
            };
        }

        void reset(int centroidOrdinal) {
            this.nextCentroidOrdinal = centroidOrdinal;
        }

        void quantizeQueryIfNecessary() throws IOException {
            if (nextCentroidOrdinal != currentCentroidOrdinal) {
                var quantized = cache.get(nextCentroidOrdinal);
                if (quantized != null) {
                    result = quantized;
                    currentCentroidOrdinal = nextCentroidOrdinal;
                    return;
                }
                // reuse the evicted byte array to reduce allocations
                final byte[] quantizedQuery = Objects.requireNonNullElseGet(
                    evictedQuantizedQuery,
                    () -> new byte[quantEncoding.getQueryPackedLength(target.length)]
                );
                final float[] queryCentroid;
                if (parentsSlice != null) {
                    assert nextCentroidOrdinal >= 0;
                    parentsSlice.seek((long) nextCentroidOrdinal * centroidScratch.length * Float.BYTES);
                    parentsSlice.readFloats(centroidScratch, 0, centroidScratch.length);
                    queryCentroid = centroidScratch;
                } else {
                    assert nextCentroidOrdinal == NO_ORDINAL;
                    queryCentroid = globalCentroid;
                }
                OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizer.scalarQuantize(
                    target,
                    scratch,
                    quantizationScratch,
                    quantEncoding.queryBits(),
                    queryCentroid
                );
                quantEncoding.packQuery(quantizationScratch, quantizedQuery);
                currentCentroidOrdinal = nextCentroidOrdinal;
                result = new QueryQuantizerResult(queryCorrections, quantizedQuery);
                cache.put(nextCentroidOrdinal, result);
            }
        }

        OptimizedScalarQuantizer.QuantizationResult getQueryCorrections() {
            return result.queryCorrections();
        }

        byte[] getQuantizedTarget() {
            return result.quantizedTarget();
        }
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        // TODO: override if adding new files
        return super.getOffHeapByteSize(fieldInfo);
    }

    private static class SlicedMemorySegmentPostingsVisitor extends MemorySegmentPostingsVisitor {
        final int startDocId;
        final int endDocId;
        final FloatVectorValues floatVectorValues;

        SlicedMemorySegmentPostingsVisitor(
            QueryQuantizer queryQuantizer,
            QuantEncoding quantEncoding,
            IndexInput indexInput,
            FieldEntry entry,
            FieldInfo fieldInfo,
            Bits acceptDocs,
            FloatVectorValues values,
            int startDocId,
            int endDocId
        ) throws IOException {
            super(queryQuantizer, quantEncoding, indexInput, entry, fieldInfo, acceptDocs);
            this.startDocId = startDocId;
            this.endDocId = endDocId;
            this.floatVectorValues = values;
        }

        @Override
        public int resetPostingsScorer(PostingMetadata metadata) throws IOException {
            int totalVectors = super.resetPostingsScorer(metadata);
            int totalBlocks = totalVectors / BULK_SIZE;
            KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
            if (iterator.advance(startDocId) >= endDocId) {
                this.vectors = 0;
                return 0;
            }
            int minOrd = iterator.index();
            int docId = iterator.advance(endDocId);
            int maxOrd;
            if (docId == DocIdSetIterator.NO_MORE_DOCS) {
                maxOrd = floatVectorValues.size();
            } else {
                maxOrd = iterator.index();
            }
            assert maxOrd - minOrd <= totalVectors;
            int startBlock = minOrd / BULK_SIZE;
            int endBlock = (maxOrd - 1) / BULK_SIZE;
            if (endBlock == totalBlocks) {
                this.vectors = totalVectors - startBlock * BULK_SIZE;
            } else {
                this.vectors = (1 + endBlock - startBlock) * BULK_SIZE;
            }
            docBase = startBlock * BULK_SIZE;
            slicePos += startBlock * BULK_SIZE * quantizedByteLength;
            return this.vectors;
        }

        @Override
        protected int docToBulkScore(int[] docIds, int[] offsets, Bits acceptDocs, int bulkSize) {
            int docToScore = 0;
            for (int i = 0; i < bulkSize; i++) {
                if (docIds[i] == -1 || (acceptDocs != null && acceptDocs.get(docIds[i]) == false)) {
                    docIds[i] = -1;
                } else {
                    offsets[docToScore] = i;
                    docToScore++;
                }
            }
            return docToScore;
        }

        @Override
        protected void readDocIds(int count) {
            for (int j = 0; j < count; j++) {
                int docId = floatVectorValues.ordToDoc(docBase++);
                if (docId >= startDocId && docId < endDocId) {
                    docIdsScratch[j] = docId;
                } else {
                    docIdsScratch[j] = -1;
                }
            }
        }
    }

    private static class MemorySegmentPostingsVisitor implements PostingVisitor {
        final long quantizedByteLength;
        final IndexInput indexInput;
        final FieldEntry entry;
        final FieldInfo fieldInfo;
        final Bits acceptDocs;
        private final ES940OSQVectorsScorer osqVectorsScorer;
        final float[] scores = new float[BULK_SIZE];
        final float[] correctionsLower = new float[BULK_SIZE];
        final float[] correctionsUpper = new float[BULK_SIZE];
        final int[] correctionsSum = new int[BULK_SIZE];
        final float[] correctionsAdd = new float[BULK_SIZE];
        final int[] docIdsScratch = new int[BULK_SIZE];
        final int[] offsetsScratch = new int[BULK_SIZE];
        byte docEncoding;
        int docBase = 0;

        int vectors;
        float centroidToParentSqDist;
        float centroidDistance;
        long slicePos;
        /**
         * How many documents in this posting list carry the query's attribute, or -1 when unknown.
         * Once that many have been scored the rest of the list provably holds no further match, so
         * scanning it is pure waste — this is the count the cluster value-set summary already has,
         * used to cut the tail of every cluster short.
         */

        private final QueryQuantizer queryQuantizer;
        final DocIdsWriter idsWriter = new DocIdsWriter();
        final VectorSimilarityFunction similarityFunction;
        final long quantizedVectorByteSize;

        MemorySegmentPostingsVisitor(
            QueryQuantizer queryQuantizer,
            QuantEncoding quantEncoding,
            IndexInput indexInput,
            FieldEntry entry,
            FieldInfo fieldInfo,
            Bits acceptDocs
        ) throws IOException {
            this.queryQuantizer = queryQuantizer;
            this.indexInput = indexInput;
            this.similarityFunction = fieldInfo.getVectorSimilarityFunction();
            this.entry = entry;
            this.fieldInfo = fieldInfo;
            this.acceptDocs = acceptDocs;
            quantizedVectorByteSize = quantEncoding.getDocPackedLength(fieldInfo.getVectorDimension());
            quantizedByteLength = quantizedVectorByteSize + (Float.BYTES * 3) + Integer.BYTES;
            osqVectorsScorer = ESVectorUtil.getES940OSQVectorsScorer(
                indexInput,
                quantEncoding.queryBits(),
                quantEncoding.bits(),
                fieldInfo.getVectorDimension(),
                (int) quantizedVectorByteSize,
                BULK_SIZE,
                quantEncoding.bits() == 2 || quantEncoding.bits() == 4
                    ? ES940OSQVectorsScorer.BitEncoding.PACKED
                    : ES940OSQVectorsScorer.BitEncoding.STRIPED
            );
        }

        @Override
        public int resetPostingsScorer(PostingMetadata metadata) throws IOException {
            float score = metadata.documentCentroidScore();
            indexInput.seek(metadata.offset());
            centroidToParentSqDist = Float.intBitsToFloat(indexInput.readInt());
            vectors = indexInput.readVInt();
            docEncoding = indexInput.readByte();
            docBase = 0;
            slicePos = indexInput.getFilePointer();
            // The score is the transformed score used when searching the centroids.
            // we need to convert it back to the raw similarity to be used as part of
            // final corrections
            centroidDistance = switch (similarityFunction) {
                case EUCLIDEAN -> ((1 / score) - 1) - centroidToParentSqDist;
                case COSINE, DOT_PRODUCT -> 2 * score - 1;
                case MAXIMUM_INNER_PRODUCT -> score - 1;
            };
            queryQuantizer.reset(metadata.queryCentroidOrdinal());
            return vectors;
        }

        private float scoreIndividually(int bulkSize) throws IOException {
            float maxScore = Float.NEGATIVE_INFINITY;
            // score individually, first the quantized byte chunk
            for (int j = 0; j < bulkSize; j++) {
                int doc = docIdsScratch[j];
                if (doc != -1) {
                    float qcDist = osqVectorsScorer.quantizeScore(queryQuantizer.getQuantizedTarget());
                    scores[j] = qcDist;
                } else {
                    indexInput.skipBytes(quantizedVectorByteSize);
                }
            }
            // read in all corrections
            indexInput.readFloats(correctionsLower, 0, bulkSize);
            indexInput.readFloats(correctionsUpper, 0, bulkSize);
            for (int j = 0; j < bulkSize; j++) {
                correctionsSum[j] = indexInput.readInt();
            }
            indexInput.readFloats(correctionsAdd, 0, bulkSize);
            // Now apply corrections
            for (int j = 0; j < bulkSize; j++) {
                int doc = docIdsScratch[j];
                if (doc != -1) {
                    scores[j] = osqVectorsScorer.applyCorrectionsIndividually(
                        queryQuantizer.getQueryCorrections().lowerInterval(),
                        queryQuantizer.getQueryCorrections().upperInterval(),
                        queryQuantizer.getQueryCorrections().quantizedComponentSum(),
                        centroidDistance,
                        fieldInfo.getVectorSimilarityFunction(),
                        0,
                        correctionsLower[j],
                        correctionsUpper[j],
                        correctionsSum[j],
                        correctionsAdd[j],
                        scores[j]
                    );
                    if (scores[j] > maxScore) {
                        maxScore = scores[j];
                    }
                }
            }
            return maxScore;
        }

        protected int docToBulkScore(int[] docIds, int[] offsets, Bits acceptDocs, int bulkSize) {
            if (acceptDocs == null) {
                return bulkSize;
            }
            int docToScore = 0;
            for (int i = 0; i < bulkSize; i++) {
                if (docIds[i] == -1 || acceptDocs.get(docIds[i]) == false) {
                    docIds[i] = -1;
                } else {
                    offsets[docToScore] = i;
                    docToScore++;
                }
            }
            return docToScore;
        }

        protected void collectBulk(KnnCollector knnCollector, float[] scores, int bulkSize, int docsToBulkScore, float maxScore) {
            if (knnCollector instanceof BulkKnnCollector bulkCollector) {
                if (docsToBulkScore == bulkSize) {
                    bulkCollector.bulkCollect(docIdsScratch, scores, bulkSize, maxScore);
                    return;
                }
                for (int i = 0; i < docsToBulkScore; i++) {
                    int offset = offsetsScratch[i];
                    docIdsScratch[i] = docIdsScratch[offset];
                    scores[i] = scores[offset];
                }
                bulkCollector.bulkCollect(docIdsScratch, scores, docsToBulkScore, maxScore);
                return;
            }
            for (int i = 0; i < bulkSize; i++) {
                final int doc = docIdsScratch[i];
                if (doc != -1) {
                    knnCollector.collect(doc, scores[i]);
                }
            }
        }

        protected void readDocIds(int count) throws IOException {
            idsWriter.readInts(indexInput, count, docEncoding, docIdsScratch);
            // reconstitute from the deltas
            for (int j = 0; j < count; j++) {
                docBase += docIdsScratch[j];
                docIdsScratch[j] = docBase;
            }
        }

        @Override
        public int visit(KnnCollector knnCollector) throws IOException {
            indexInput.seek(slicePos);
            // block processing
            int scoredDocs = 0;
            int limit = vectors - BULK_SIZE + 1;
            int i = 0;
            // read Docs
            for (; i < limit; i += BULK_SIZE) {
                // read the doc ids
                readDocIds(BULK_SIZE);
                final int docsToBulkScore = docToBulkScore(docIdsScratch, offsetsScratch, acceptDocs, BULK_SIZE);
                if (docsToBulkScore == 0) {
                    indexInput.skipBytes(quantizedByteLength * BULK_SIZE);
                    continue;
                }
                queryQuantizer.quantizeQueryIfNecessary();
                final float maxScore;
                if (docsToBulkScore == 1) {
                    maxScore = scoreIndividually(BULK_SIZE);
                } else if (docsToBulkScore < BULK_SIZE) {
                    maxScore = osqVectorsScorer.scoreBulkOffsets(
                        queryQuantizer.getQuantizedTarget(),
                        queryQuantizer.getQueryCorrections().lowerInterval(),
                        queryQuantizer.getQueryCorrections().upperInterval(),
                        queryQuantizer.getQueryCorrections().quantizedComponentSum(),
                        centroidDistance,
                        fieldInfo.getVectorSimilarityFunction(),
                        0f,
                        offsetsScratch,
                        docsToBulkScore,
                        scores,
                        BULK_SIZE
                    );
                } else {
                    maxScore = osqVectorsScorer.scoreBulk(
                        queryQuantizer.getQuantizedTarget(),
                        queryQuantizer.getQueryCorrections().lowerInterval(),
                        queryQuantizer.getQueryCorrections().upperInterval(),
                        queryQuantizer.getQueryCorrections().quantizedComponentSum(),
                        centroidDistance,
                        fieldInfo.getVectorSimilarityFunction(),
                        0f,
                        scores
                    );
                }
                if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                    collectBulk(knnCollector, scores, BULK_SIZE, docsToBulkScore, maxScore);
                }
                scoredDocs += docsToBulkScore;
            }
            // bulk process tail
            if (i < vectors) {
                int tailSize = vectors - i;
                readDocIds(tailSize);
                final int docsToBulkScore = docToBulkScore(docIdsScratch, offsetsScratch, acceptDocs, tailSize);
                if (docsToBulkScore == 0) {
                    indexInput.skipBytes(quantizedByteLength * tailSize);
                } else {
                    queryQuantizer.quantizeQueryIfNecessary();
                    final float maxScore;
                    if (docsToBulkScore == 1) {
                        maxScore = scoreIndividually(tailSize);
                    } else if (docsToBulkScore < tailSize) {
                        maxScore = osqVectorsScorer.scoreBulkOffsets(
                            queryQuantizer.getQuantizedTarget(),
                            queryQuantizer.getQueryCorrections().lowerInterval(),
                            queryQuantizer.getQueryCorrections().upperInterval(),
                            queryQuantizer.getQueryCorrections().quantizedComponentSum(),
                            centroidDistance,
                            fieldInfo.getVectorSimilarityFunction(),
                            0f,
                            offsetsScratch,
                            docsToBulkScore,
                            scores,
                            tailSize
                        );
                    } else {
                        maxScore = osqVectorsScorer.scoreBulk(
                            queryQuantizer.getQuantizedTarget(),
                            queryQuantizer.getQueryCorrections().lowerInterval(),
                            queryQuantizer.getQueryCorrections().upperInterval(),
                            queryQuantizer.getQueryCorrections().quantizedComponentSum(),
                            centroidDistance,
                            fieldInfo.getVectorSimilarityFunction(),
                            0f,
                            scores,
                            tailSize
                        );
                    }
                    if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                        collectBulk(knnCollector, scores, tailSize, docsToBulkScore, maxScore);
                    }
                    scoredDocs += docsToBulkScore;
                }
            }
            if (scoredDocs > 0) {
                knnCollector.incVisitedCount(scoredDocs);
            }
            return scoredDocs;
        }
    }

}
