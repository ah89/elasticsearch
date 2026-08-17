/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValues;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;

import java.io.Closeable;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;

/**
 * Reader for IVF vectors. This reader is used to read the IVF vectors from the index.
 */
public abstract class IVFVectorsReader<E extends IVFVectorsReader.FieldEntry> extends KnnVectorsReader {

    // Two-Signal Model constants for dynamic visit ratio computation.
    // Computes a visit ratio from the num_candidates/k ratio and k magnitude.
    private static final double V_MIN = 0.003;
    private static final double V_MAX = 0.04;
    private static final double LOG1P_R_MAX = Math.log1p(10.0);
    private static final double LOG1P_K_MAX = Math.log1p(10_000.0);
    private static final double RATIO_WEIGHT = 0.85;
    private static final double K_WEIGHT = 0.15;

    // Segment-size cap constants.
    // Empirical power-law curve calibrated on GIST-1M, Wiki-Cohere-1M, and MSMarco-130M datasets.
    // Caps the visit ratio for large segments where fewer clusters need visiting to achieve the target recall.
    // Produces ~10% cap for small segments (100K), ~4.5% at 1M, and ~2-3% for large segments (5-10M).
    private static final double CAP_COEFFICIENT = 0.045;
    private static final int CAP_REF_SIZE = 1_000_000;
    private static final double CAP_EXPONENT = 0.35;
    static final float DEFAULT_TARGET_RECALL = 0.9f;

    protected final IndexInput ivfCentroids, ivfClusters;
    private final SegmentReadState state;
    protected final FieldInfos fieldInfos;
    protected final IntObjectHashMap<E> fields;
    private final GenericFlatVectorReaders genericReaders;
    private final String centroidExtension;
    private final String clusterExtension;
    private final int versionDirectIo;
    private final float dynamicVisitRatio;
    protected int versionMeta = -1;

    @SuppressWarnings("this-escape")
    protected IVFVectorsReader(
        SegmentReadState state,
        GenericFlatVectorReaders.LoadFlatVectorsReader loadReader,
        String codecName,
        String centroidExtension,
        String clusterExtension,
        String metaExtension,
        int versionStart,
        int versionCurrent,
        int versionDirectIo,
        float dynamicVisitRatio
    ) throws IOException {
        this.state = state;
        this.fieldInfos = state.fieldInfos;
        this.fields = new IntObjectHashMap<>();
        this.genericReaders = new GenericFlatVectorReaders();
        this.centroidExtension = centroidExtension;
        this.clusterExtension = clusterExtension;
        this.versionDirectIo = versionDirectIo;
        this.dynamicVisitRatio = dynamicVisitRatio;
        String meta = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, metaExtension);

        int versionMeta = -1;
        try (ChecksumIndexInput ivfMeta = state.directory.openChecksumInput(meta)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    ivfMeta,
                    codecName,
                    versionStart,
                    versionCurrent,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                this.versionMeta = versionMeta;
                readFields(ivfMeta, versionMeta, genericReaders, loadReader);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(ivfMeta, priorE);
            }
            ivfCentroids = openDataInput(state, versionMeta, centroidExtension, codecName, versionStart, versionCurrent, state.context);
            ivfClusters = openDataInput(state, versionMeta, clusterExtension, codecName, versionStart, versionCurrent, state.context);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(this);
            throw t;
        }
    }

    public abstract CentroidIterator getCentroidIterator(
        FieldInfo fieldInfo,
        int numCentroids,
        IndexInput centroids,
        float[] target,
        IndexInput postingListSlice,
        AcceptDocs acceptDocs,
        float approximateCost,
        FloatVectorValues values,
        float visitRatio
    ) throws IOException;

    /** Get the number of vectors to search, which is typically the total number of vectors in the segment or the
     *  number of vectors in a slice if the segment is sliced.*/
    protected int getNumberOfVectors(E entry, FloatVectorValues values, IndexInput centroidSlice, ESAcceptDocs esAcceptDocs)
        throws IOException {
        return values.size();
    }

    protected static IndexInput openDataInput(
        SegmentReadState state,
        int versionMeta,
        String fileExtension,
        String codecName,
        int versionStart,
        int versionCurrent,
        IOContext context
    ) throws IOException {
        final String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
        final IndexInput in = state.directory.openInput(fileName, context);
        try {
            final int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                versionStart,
                versionCurrent,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    "Format versions mismatch: meta=" + versionMeta + ", " + codecName + "=" + versionVectorData,
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            return in;
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(in);
            throw t;
        }
    }

    private void readFields(
        ChecksumIndexInput meta,
        int versionMeta,
        GenericFlatVectorReaders genericFields,
        GenericFlatVectorReaders.LoadFlatVectorsReader loadReader
    ) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }

            E fieldEntry = readField(meta, info, versionMeta);
            genericFields.loadField(fieldNumber, fieldEntry, loadReader);

            fields.put(info.number, fieldEntry);
        }
    }

    private E readField(IndexInput input, FieldInfo info, int versionMeta) throws IOException {
        final String rawVectorFormat = input.readString();
        final boolean useDirectIOReads = versionMeta >= versionDirectIo && input.readByte() == 1;
        final VectorEncoding vectorEncoding = readVectorEncoding(input);
        final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
        if (similarityFunction != info.getVectorSimilarityFunction()) {
            throw new IllegalStateException(
                "Inconsistent vector similarity function for field=\""
                    + info.name
                    + "\"; "
                    + similarityFunction
                    + " != "
                    + info.getVectorSimilarityFunction()
            );
        }
        final int numCentroids = input.readInt();
        final long centroidOffset = input.readLong();
        final long centroidLength = input.readLong();
        final float[] globalCentroid = new float[info.getVectorDimension()];
        long postingListOffset = -1;
        long postingListLength = 0;
        float globalCentroidDp = 0;
        if (centroidLength > 0) {
            postingListOffset = input.readLong();
            postingListLength = input.readLong();
            input.readFloats(globalCentroid, 0, globalCentroid.length);
            globalCentroidDp = Float.intBitsToFloat(input.readInt());
        }
        return doReadField(
            input,
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
            globalCentroidDp
        );
    }

    protected abstract E doReadField(
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
    ) throws IOException;

    private static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
        final int i = input.readInt();
        if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
            throw new IllegalArgumentException("invalid distance function: " + i);
        }
        return SIMILARITY_FUNCTIONS.get(i);
    }

    private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
        final int encodingId = input.readInt();
        if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
            throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
        }
        return VectorEncoding.values()[encodingId];
    }

    @Override
    public final void checkIntegrity() throws IOException {
        for (var reader : genericReaders.allReaders()) {
            reader.checkIntegrity();
        }
        CodecUtil.checksumEntireFile(ivfCentroids);
        CodecUtil.checksumEntireFile(ivfClusters);
    }

    protected FlatVectorsReader getReaderForField(String field) {
        FieldInfo info = fieldInfos.fieldInfo(field);
        if (info == null) throw new IllegalArgumentException("Could not find field [" + field + "]");
        return genericReaders.getReaderForField(info.number);
    }

    @Override
    public final FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return getReaderForField(field).getFloatVectorValues(field);
    }

    @Override
    public final ByteVectorValues getByteVectorValues(String field) throws IOException {
        return getReaderForField(field).getByteVectorValues(field);
    }

    @Override
    public final void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        if (fieldInfo == null || fieldInfo.getVectorDimension() == 0) {
            return;
        }
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
            getReaderForField(field).search(field, target, knnCollector, acceptDocs);
            return;
        }
        final E entry = fields.get(fieldInfo.number);
        if (hasNoVectors(fieldInfo, entry)) {
            return;
        }
        if (fieldInfo.getVectorDimension() != target.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + target.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }

        final ESAcceptDocs esAcceptDocs;
        if (acceptDocs instanceof ESAcceptDocs) {
            esAcceptDocs = (ESAcceptDocs) acceptDocs;
        } else {
            esAcceptDocs = null;
        }

        final FloatVectorValues values = getFloatVectorValues(field);
        final IndexInput centroids = entry.centroidSlice(ivfCentroids);
        final int numVectors = getNumberOfVectors(entry, values, centroids, esAcceptDocs);
        if (numVectors == 0) {
            return; // nothing more to do if there are no vectors in this segment / slice
        }
        final float approximateCost;
        if (esAcceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll) {
            approximateCost = numVectors;
        } else {
            approximateCost = esAcceptDocs == null ? acceptDocs.cost() : esAcceptDocs.approximateCost();
        }
        float percentFiltered = Math.clamp(approximateCost / numVectors, 0f, 1f);
        int k = knnCollector.k();
        int numCands = k;
        float visitRatio = dynamicVisitRatio;
        // Search strategy may be null if this is being called from checkIndex (e.g. from a test)
        if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy ivfSearchStrategy) {
            visitRatio = ivfSearchStrategy.getVisitRatio();
            numCands = ivfSearchStrategy.getNumCands();
            k = ivfSearchStrategy.getK();
        }

        if (visitRatio == dynamicVisitRatio) {
            visitRatio = Math.min(computeDynamicVisitRatio(numCands, k), computeSegmentSizeCap(numVectors));
        }
        // LensGate (POC): when the filter's value set is known, install per-centroid ordering
        // coefficients so both the fetch path and the normal scan below rank clusters by proximity
        // to the filter value's sub-centroid rather than to the cluster centroid. On uninformative
        // filters the coefficients shrink to identity (see AttributeLensIndex), so the ordering
        // degenerates to the plain one and the query proceeds as if the lens were absent.
        // Exact scan stop: tell the posting visitor how many matching documents each posting list
        // physically holds, so it can leave a cluster the moment it has seen them all instead of
        // reading the tail. Counts are of posting-list membership, which includes SOAR overspill
        // copies — owner counts would under-report and the stop would drop results.
        boolean lensEngaged = false;
        if (FlatCentroidIndex.POC_LENS_GATE && percentFiltered < 1f) {
            final float[][] coefficients = prepareLens(fieldInfo, entry, values, target, visitRatio);
            if (coefficients != null) {
                FlatCentroidIndex.POC_LENS_MULT.set(coefficients[0]);
                FlatCentroidIndex.POC_LENS_ADD.set(coefficients[1]);
                lensEngaged = true;
            }
        }
        try {
            searchWithBudget(
                fieldInfo,
                entry,
                values,
                target,
                acceptDocs,
                esAcceptDocs,
                knnCollector,
                visitRatio,
                numVectors,
                percentFiltered,
                approximateCost,
                centroids
            );
        } finally {
            if (lensEngaged) {
                FlatCentroidIndex.POC_LENS_MULT.remove();
                FlatCentroidIndex.POC_LENS_ADD.remove();
                // unpublished, not discarded: the bitset itself stays in its own scratch for reuse
                FlatCentroidIndex.POC_LENS_ACCEPT_CENTROIDS.remove();
            }
        }
    }

    private void searchWithBudget(
        FieldInfo fieldInfo,
        E entry,
        FloatVectorValues values,
        float[] target,
        AcceptDocs acceptDocs,
        ESAcceptDocs esAcceptDocs,
        KnnCollector knnCollector,
        float visitRatio,
        int numVectors,
        float percentFiltered,
        float approximateCost,
        IndexInput centroids
    ) throws IOException {
        // FetchGate (POC): in the selective regime, replace the posting-list scan with direct
        // fetches of the matching documents' quantized bytes. Outside its regime (broad filters,
        // unsupported layouts) it declines and the search proceeds unchanged — including whatever
        // centroid gate is active — so the fetch arm can never do worse than the gate it extends.
        if (FlatCentroidIndex.POC_FETCH_GATE
            && percentFiltered <= FETCH_MAX_SELECTIVITY
            && fetchSearch(fieldInfo, entry, values, target, acceptDocs, knnCollector, visitRatio, numVectors)) {
            return;
        }
        long maxVectorVisited = maxVectorsToVisit(entry, visitRatio, numVectors);
        IndexInput postListSlice = entry.postingListSlice(ivfClusters);
        CentroidIterator centroidPrefetchingIterator = getCentroidIterator(
            fieldInfo,
            entry.numCentroids,
            centroids,
            target,
            postListSlice,
            acceptDocs,
            approximateCost,
            values,
            visitRatio
        );
        Bits acceptDocsBits = acceptDocs.bits();
        PostingVisitor scorer = getPostingVisitor(
            fieldInfo,
            values,
            postListSlice,
            target,
            acceptDocsBits,
            entry.centroidSlice(ivfCentroids),
            esAcceptDocs
        );
        long expectedDocs = 0;
        long actualDocs = 0;
        // initially we visit only the "centroids to search"
        // Note, numCollected is doing the bare minimum here.
        // TODO do we need to handle nested doc counts similarly to how we handle
        // filtering? E.g. keep exploring until we hit an expected number of parent documents vs. child vectors?
        while (centroidPrefetchingIterator.hasNext()
            && (maxVectorVisited > expectedDocs || knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {
            PostingMetadata postingMetadata = centroidPrefetchingIterator.nextPosting();
            expectedDocs += scorer.resetPostingsScorer(postingMetadata);
            actualDocs += scorer.visit(knnCollector);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }
        if (acceptDocsBits != null) {
            // TODO Adjust the value here when using centroid filtering
            float unfilteredRatioVisited = (float) expectedDocs / numVectors;
            int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
            float expectedScored = Math.min(2 * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
            while (centroidPrefetchingIterator.hasNext() && (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
                PostingMetadata postingMetadata = centroidPrefetchingIterator.nextPosting();
                scorer.resetPostingsScorer(postingMetadata);
                actualDocs += scorer.visit(knnCollector);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        }
    }

    /**
     * The cap on the number of (posting-member) vectors the search loop may visit. The default accounts for
     * SOAR overspill, which can place a vector in up to two postings, by allowing 2x the visit-ratio budget.
     * Subclasses may override to use a different budgeting model (e.g. an experiment-only posting/head-count
     * budget where the centroid iterator's own bound governs how many postings are drained).
     */
    protected long maxVectorsToVisit(E entry, float visitRatio, int numVectors) {
        return (long) (2.0 * visitRatio * numVectors);
    }

    /**
     * Above this filter selectivity the FetchGate declines and the query takes the normal (gated)
     * scan path. The crossover is set by block density: at selectivity s a 16-document posting block
     * holds 16s matches in expectation, and once that approaches 1 the scan touches roughly the same
     * blocks the fetch would while scoring them with the bulk SIMD kernel, which per scored document
     * beats individual fetches (measured: at 10% the gated scan is 1.4-3x faster at equal recall; at
     * 2% and below the fetch dominates). 5% keeps the fetch in the regime where most of the scan's
     * decode work is provably wasted. The check is per query, so mixed workloads split query by
     * query between fetch and gate.
     */
    protected static final float FETCH_MAX_SELECTIVITY = 0.05f;

    /**
     * FetchGate hook (POC, see {@link MatchFetchIndex}): score the filter's matching documents
     * directly instead of scanning posting lists. Implementations return {@code false} to decline —
     * unsupported layout, no random-access filter bits, out-of-regime — in which case the caller
     * runs the standard search. Only invoked when {@link FlatCentroidIndex#POC_FETCH_GATE} is set.
     *
     * @return true if the search was fully handled
     */
    protected boolean fetchSearch(
        FieldInfo fieldInfo,
        E entry,
        FloatVectorValues values,
        float[] target,
        AcceptDocs acceptDocs,
        KnnCollector knnCollector,
        float visitRatio,
        int numVectors
    ) throws IOException {
        return false;
    }

    /** Scales the expected-maximum term; 1 is the derived value. POC-tunable to probe sensitivity. */
    protected static final float LENS_BETA = Float.parseFloat(System.getProperty("es.poc.lens.beta", "1"));

    /**
     * Whether the lens also answers the centroid gate. Its value-&gt;clusters inversion already names
     * exactly the clusters that can hold a match, so the counted gate's walk over the filter's
     * matching documents is redundant work. On by default; a knob so the saving can be measured.
     */
    protected static final boolean LENS_GATE = Boolean.parseBoolean(System.getProperty("es.poc.lens.gate", "true"));

    private final ThreadLocal<FixedBitSet> lensAcceptScratch = new ThreadLocal<>();

    private FixedBitSet acceptScratch(int numCentroids) {
        FixedBitSet scratch = lensAcceptScratch.get();
        if (scratch == null || scratch.length() != numCentroids) {
            scratch = new FixedBitSet(numCentroids);
            lensAcceptScratch.set(scratch);
        }
        return scratch;
    }

    /** Per-thread {mult, add} ordering-coefficient scratch, see {@link #prepareLens}. */
    private final ThreadLocal<float[][]> lensScratch = ThreadLocal.withInitial(() -> new float[2][]);

    /**
     * Visit ratio below which the log-count term is switched off entirely, and the ratio at which
     * it reaches full weight; it ramps linearly between them.
     *
     * <p>Ranking by expected yield only pays once the budget allows visiting enough clusters for
     * yield to dominate proximity. At a very tight budget only a handful of posting lists are
     * opened and the nearest ones are exactly the right ones — measured on the correlated scenario,
     * <em>any</em> non-zero count weight there costs ~3 recall points at k=100, while full weight
     * at a 5% budget gains 4. Recall is deterministic per configuration, so those are real effects
     * rather than run-to-run scatter. The dead zone keeps the tight-budget end of the operating
     * curve bit-identical to the plain ordering and lets the count term act only where it wins.
     */
    protected static final float LENS_COUNT_MIN_RATIO = Float.parseFloat(System.getProperty("es.poc.lens.countMinRatio", "0.02"));

    protected static final float LENS_COUNT_FULL_WEIGHT_RATIO = Float.parseFloat(System.getProperty("es.poc.lens.countRampRatio", "0.05"));

    /**
     * LensGate setup (POC, see {@link AttributeLensIndex}): builds (once per segment) the
     * per-(cluster, value) sub-centroid statistics and derives the current query's per-centroid
     * ordering coefficients from them. Returns {@code {mult, add}} or {@code null} when the lens
     * does not apply — no single-value filter available, non-dot-family similarity, sliced or
     * degenerate layouts — in which case the search proceeds with the plain ordering.
     */
    /**
     * Posting-list geometry for the segment, or null when the format cannot provide it. Only the
     * concrete reader knows the posting layout, so the walk lives there.
     */
    protected MatchFetchIndex matchFetchGeometry(FieldInfo fieldInfo, E entry, FloatVectorValues values) throws IOException {
        return null;
    }

    protected float[][] prepareLens(FieldInfo fieldInfo, E entry, FloatVectorValues values, float[] target, float visitRatio)
        throws IOException {
        if (fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN) {
            return null;
        }
        if (entry.numSlices() != -1 || entry.numCentroids() <= 1) {
            return null;
        }
        final int[] valueSet = FlatCentroidIndex.POC_QUERY_VALUE_SET.get();
        if (valueSet == null || valueSet.length != 1) {
            return null;
        }
        final int[] docValues = FlatCentroidIndex.POC_DOC_VALUES;
        final int[] docValueOffsets = FlatCentroidIndex.POC_DOC_VALUE_OFFSETS;
        final int numValues = FlatCentroidIndex.POC_NUM_VALUES;
        if (docValues == null || docValueOffsets == null || numValues <= 0) {
            return null;
        }
        AttributeLensIndex lens = entry.attributeLensIndex();
        if (lens == null) {
            synchronized (entry) {
                lens = entry.attributeLensIndex();
                if (lens == null) {
                    final IndexInput centroidSlice = entry.centroidSlice(ivfCentroids);
                    final int bitsRequired = DirectWriter.bitsRequired(entry.numCentroids());
                    final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
                    final LongValues centroidOfOrdinal = DirectReader.getInstance(
                        centroidSlice.randomAccessSlice(0, sizeLookup),
                        bitsRequired
                    );
                    // membership, not ownership: x must be the fraction of what a posting list
                    // actually stores, or the early stop trusts a proportion that cannot be reached
                    lens = AttributeLensIndex.build(
                        values,
                        centroidOfOrdinal,
                        matchFetchGeometry(fieldInfo, entry, values),
                        docValues,
                        docValueOffsets,
                        numValues,
                        entry.numCentroids()
                    );
                    entry.attributeLensIndex(lens);
                    FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(lens.ramBytesUsed());
                }
            }
        }
        if (lens == AttributeLensIndex.UNSUPPORTED) {
            return null;
        }
        // Reuse per-thread scratch instead of allocating and clearing two arrays per query: at a
        // tight budget a filtered query runs in ~100us, and an O(numCentroids) allocate-and-fill
        // per query was costing ~20% of it — enough to hide the ordering's benefit entirely.
        // No clearing is needed: the centroid gate accepts exactly the clusters that hold the
        // value, which are exactly the entries filled below, so no stale entry is ever read.
        final float[][] scratch = lensScratch.get();
        final float[] mult;
        final float[] add;
        if (scratch[0] != null && scratch[0].length == entry.numCentroids()) {
            mult = scratch[0];
            add = scratch[1];
        } else {
            mult = new float[entry.numCentroids()];
            add = new float[entry.numCentroids()];
            scratch[0] = mult;
            scratch[1] = add;
        }
        Arrays.fill(mult, 1f);
        Arrays.fill(add, 0f);
        final float span = Math.max(1e-6f, LENS_COUNT_FULL_WEIGHT_RATIO - LENS_COUNT_MIN_RATIO);
        final float ramp = Math.clamp((visitRatio - LENS_COUNT_MIN_RATIO) / span, 0f, 1f);
        // Scatter the reachable range's coefficients for the clusters holding this value. mult*sim
        // is the range's midpoint (the expectation over the sub-centroid's unknown rotation); perp
        // is its half-width before the query-angle factor, which the ordering key applies once sim
        // is known; add is the precomputed expected-maximum lift, scaled by the budget ramp. All of
        // it is query-independent, so the query only gathers -- see AttributeLensIndex.
        // the same pass that scatters the coefficients fills the exact centroid gate, so the walk
        // over the filter's matching documents that the counted gate would do is not needed at all
        final FixedBitSet accept = LENS_GATE ? acceptScratch(entry.numCentroids()) : null;
        if (lens.scatterCoefficients(valueSet[0], LENS_BETA * ramp, mult, add, accept, null, null) == 0) {
            return null;
        }
        if (accept != null) {
            FlatCentroidIndex.POC_LENS_ACCEPT_CENTROIDS.set(accept);
        }
        return new float[][] { mult, add };
    }

    private static boolean hasNoVectors(FieldInfo fieldInfo, FieldEntry fieldEntry) {
        return fieldInfo.getVectorDimension() == 0
            || fieldEntry == null
            || (fieldEntry.numCentroids() == 0 && fieldEntry.postingListLength == 0L && fieldEntry.centroidLength == 0L);
    }

    /**
     * Computes the dynamic visit ratio using the Two-Signal model.
     * The formula blends the num_candidates/k ratio signal with the k magnitude signal.
     */
    static float computeDynamicVisitRatio(int numCands, int k) {
        double r = (double) numCands / Math.max(k, 1);
        double z = RATIO_WEIGHT * logScale(r - 1.0, LOG1P_R_MAX) + K_WEIGHT * logScale(k, LOG1P_K_MAX);
        return (float) (V_MIN + (V_MAX - V_MIN) * z);
    }

    private static double logScale(double value, double log1pMax) {
        return Math.clamp(Math.log1p(value) / log1pMax, 0.0, 1.0);
    }

    /**
     * Computes a segment-size-aware cap on the visit ratio.
     * Larger segments have better-formed IVF clusters and need a lower visit ratio to achieve the target recall.
     * The power-law curve is calibrated on multi-dataset experiments (GIST-1M, Wiki-Cohere, MSMarco-130M).
     * <p>
     * Formula: cap = {@link #CAP_COEFFICIENT} * ({@link #CAP_REF_SIZE} / numVectors)^{@link #CAP_EXPONENT}
     *              * (0.1 / (1 - targetRecall))
     *
     * @param numVectors number of vectors in the segment
     * @return the upper-bound visit ratio for this segment size
     */
    static float computeSegmentSizeCap(int numVectors) {
        if (numVectors <= 0) {
            return (float) V_MAX;
        }
        double sizeScale = Math.pow((double) CAP_REF_SIZE / numVectors, CAP_EXPONENT);
        double recallScale = 0.1 / (1.0 - DEFAULT_TARGET_RECALL);
        return (float) Math.min(1.0, CAP_COEFFICIENT * sizeScale * recallScale);
    }

    @Override
    public final void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        final ByteVectorValues values = getReaderForField(field).getByteVectorValues(field);
        for (int i = 0; i < values.size(); i++) {
            final float score = fieldInfo.getVectorSimilarityFunction().compare(target, values.vectorValue(i));
            knnCollector.collect(values.ordToDoc(i), score);
            if (knnCollector.earlyTerminated()) {
                return;
            }
        }
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        var raw = getReaderForField(fieldInfo.name).getOffHeapByteSize(fieldInfo);
        FieldEntry fe = fields.get(fieldInfo.number);
        if (fe == null) {
            assert fieldInfo.getVectorEncoding() == VectorEncoding.BYTE;
            return raw;
        }

        var centroidsClusters = Map.of(centroidExtension, fe.centroidLength, clusterExtension, fe.postingListLength);
        return KnnVectorsReader.mergeOffHeapByteSizeMaps(raw, centroidsClusters);
    }

    @Override
    public void close() throws IOException {
        List<Closeable> closeables = new ArrayList<>(genericReaders.allReaders());
        Collections.addAll(closeables, ivfCentroids, ivfClusters);
        IOUtils.close(closeables);
    }

    protected static class FieldEntry implements GenericFlatVectorReaders.Field {
        protected final String rawVectorFormatName;
        protected final boolean useDirectIOReads;
        protected final VectorSimilarityFunction similarityFunction;
        protected final VectorEncoding vectorEncoding;
        protected final int numCentroids;
        protected final long centroidOffset;
        protected final long centroidLength;
        protected final long postingListOffset;
        protected final long postingListLength;
        protected final float[] globalCentroid;
        protected final float globalCentroidDp;
        protected final int bulkSize;

        public FieldEntry(
            String rawVectorFormatName,
            boolean useDirectIOReads,
            VectorSimilarityFunction similarityFunction,
            VectorEncoding vectorEncoding,
            int numCentroids,
            long centroidOffset,
            long centroidLength,
            long postingListOffset,
            long postingListLength,
            float[] globalCentroid,
            float globalCentroidDp,
            int bulkSize
        ) {
            this.rawVectorFormatName = rawVectorFormatName;
            this.useDirectIOReads = useDirectIOReads;
            this.similarityFunction = similarityFunction;
            this.vectorEncoding = vectorEncoding;
            this.numCentroids = numCentroids;
            this.centroidOffset = centroidOffset;
            this.centroidLength = centroidLength;
            this.postingListOffset = postingListOffset;
            this.postingListLength = postingListLength;
            this.globalCentroid = globalCentroid;
            this.globalCentroidDp = globalCentroidDp;
            this.bulkSize = bulkSize;
        }

        @Override
        public String rawVectorFormatName() {
            return rawVectorFormatName;
        }

        @Override
        public boolean useDirectIOReads() {
            return useDirectIOReads;
        }

        public int numCentroids() {
            return numCentroids;
        }

        public float[] globalCentroid() {
            return globalCentroid;
        }

        public float globalCentroidDp() {
            return globalCentroidDp;
        }

        public VectorSimilarityFunction similarityFunction() {
            return similarityFunction;
        }

        public IndexInput centroidSlice(IndexInput centroidFile) throws IOException {
            return centroidFile.slice("centroids", centroidOffset, centroidLength);
        }

        public IndexInput postingListSlice(IndexInput postingListFile) throws IOException {
            return postingListFile.slice("postingLists", postingListOffset, postingListLength);
        }

        public int getBulkSize() {
            return bulkSize;
        }

        public int numSlices() {
            return -1;
        }

        /**
         * Cached per-centroid match counts plus a weak reference to the accept-docs instance they were
         * derived from. The reference is what makes a hit <em>sound</em>: the cache key is a lossy hash
         * of the matching-doc set, so two different filters can collide, and silently reusing the wrong
         * counts would zero out centroids that do hold matches — losing results with no error. A hit is
         * therefore only trusted after confirming the stored source is the same bitset, or has identical
         * content (see {@code FlatCentroidIndex#sameFilter}). The reference is weak so that remembering
         * counts never pins a filter bitset in memory.
         */
        public record CachedCentroidCounts(WeakReference<Bits> source, int[] counts) {}

        /**
         * Approximate per-entry cost of a filter cache slot beyond its payload: the map node, the boxed
         * key, the record, and the weak reference. Counted so the memory column reflects that a
         * per-filter cache is not free even when its payload is small.
         */
        public static final long CACHE_ENTRY_OVERHEAD = 96L;

        /**
         * Cache of exact per-centroid filter-match counts, keyed by a fingerprint of the filter's
         * matching-doc set (see {@code FlatCentroidIndex#filterFingerprint}). Real workloads repeat
         * the same filters (Lucene's query cache exists for the same reason), so the O(matches) walk
         * that builds the counts is paid once per (filter, segment) and every later query gets the
         * exact centroid gate for free — the arbitrary-filter equivalent of precomputing per-cluster
         * per-value bitmaps at merge time. Bounded LRU: one {@code int} per centroid per remembered
         * filter.
         */
        private final Map<Long, CachedCentroidCounts> filterCentroidCountsCache = Collections.synchronizedMap(
            new LinkedHashMap<>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, CachedCentroidCounts> eldest) {
                    if (size() > 64) {
                        FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(
                            -CACHE_ENTRY_OVERHEAD - (long) eldest.getValue().counts().length * Integer.BYTES
                        );
                        return true;
                    }
                    return false;
                }
            }
        );

        public Map<Long, CachedCentroidCounts> filterCentroidCountsCache() {
            return filterCentroidCountsCache;
        }

        /**
         * A memoized SliceGate result: the accept-centroids bitset ({@code null} when the gate
         * accepts everything and prunes nothing) plus the weak source reference that makes reuse
         * sound (same verification contract as {@link CachedCentroidCounts}). Entries are ~40 bytes
         * (one bit per centroid), so this is strictly cheaper than remembering counts.
         *
         * <p>The memo is an optimization, never a dependency: on repeated filters a verified hit
         * costs one identity comparison — matching the counted gate's warm path — while a miss falls
         * back to the SliceGate compute, which is fast and deterministic on its own. This is what
         * closes the repeated-filter gap without reintroducing the workload dependence the SliceGate
         * exists to remove: worst case (never-repeating filters) is unchanged.
         */
        public record CachedGate(WeakReference<Bits> source, FixedBitSet acceptCentroids) {}

        private final Map<Long, CachedGate> sliceGateCache = Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, CachedGate> eldest) {
                if (size() > 64) {
                    final FixedBitSet gate = eldest.getValue().acceptCentroids();
                    FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(-CACHE_ENTRY_OVERHEAD - (gate == null ? 0 : gate.ramBytesUsed()));
                    return true;
                }
                return false;
            }
        });

        public Map<Long, CachedGate> sliceGateCache() {
            return sliceGateCache;
        }

        /**
         * Lazily built posting-block geometry for the FetchGate (see {@link MatchFetchIndex}).
         * Per-segment and filter-agnostic: one sequential sweep of
         * the posting lists on first use serves every later filter. A production format would write
         * it at merge time.
         */
        private volatile MatchFetchIndex matchFetchIndex;

        public MatchFetchIndex matchFetchIndex() {
            return matchFetchIndex;
        }

        public void matchFetchIndex(MatchFetchIndex index) {
            this.matchFetchIndex = index;
        }

        /**
         * A memoized per-filter match view (matching docs grouped by centroid) plus the weak source
         * reference that makes reuse sound — the same verification contract as
         * {@link CachedCentroidCounts}. The view is the counted gate's walk with locations kept,
         * so caching it has the same cost profile as caching the counts.
         */
        public record CachedMatchView(WeakReference<Bits> source, MatchFetchIndex.MatchView view) {}

        private final Map<Long, CachedMatchView> matchViewCache = Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, CachedMatchView> eldest) {
                if (size() > 64) {
                    FlatCentroidIndex.POC_GATE_RAM_BYTES.addAndGet(-CACHE_ENTRY_OVERHEAD - eldest.getValue().view().ramBytesUsed());
                    return true;
                }
                return false;
            }
        });

        public Map<Long, CachedMatchView> matchViewCache() {
            return matchViewCache;
        }

        /**
         * Lazily built per-(cluster, value) sub-centroid statistics for the LensGate (see
         * {@link AttributeLensIndex}). Per-segment and filter-agnostic; a production format would
         * compute it during the merge that builds the posting lists.
         */
        private volatile AttributeLensIndex attributeLensIndex;

        public AttributeLensIndex attributeLensIndex() {
            return attributeLensIndex;
        }

        public void attributeLensIndex(AttributeLensIndex index) {
            this.attributeLensIndex = index;
        }

    }

    /**
     * Read the raw centroids and cluster sizes for the given field from this segment.
     * Used by the adaptive merge strategy to bootstrap K-means with prior segment centroids.
     * Implementations may return {@code null} if the format does not support reading centroid data
     * (e.g. because the layout differs from the writer that consumes this data).
     *
     * @param fieldName the vector field to read centroids for
     * @return centroid data, or {@code null} if unavailable
     */
    public abstract CentroidData readCentroidData(String fieldName) throws IOException;

    /**
     * Container for centroid data read from an existing segment. The centroid vectors are
     * exposed as a streaming {@link ClusteringFloatVectorValues}
     * so the merge path can iterate them without materializing the full {@code float[N][dim]}
     * on the heap. The optional {@code backing} {@link IndexInput} owns any sliced resources
     * required by the streaming view; {@link #close()} releases it.
     */
    public static final class CentroidData implements Closeable {
        private final int numCentroids;
        private final ClusteringFloatVectorValues centroids;
        private final int[] clusterSizes;
        private final float[] globalCentroid;
        private final IndexInput backing;

        public CentroidData(ClusteringFloatVectorValues centroids, int[] clusterSizes, float[] globalCentroid, IndexInput backing) {
            assert centroids.size() == clusterSizes.length;
            this.numCentroids = centroids.size();
            this.centroids = centroids;
            this.clusterSizes = clusterSizes;
            this.globalCentroid = globalCentroid;
            this.backing = backing;
        }

        public int numCentroids() {
            return numCentroids;
        }

        public ClusteringFloatVectorValues centroids() {
            return centroids;
        }

        public int[] clusterSizes() {
            return clusterSizes;
        }

        public float[] globalCentroid() {
            return globalCentroid;
        }

        @Override
        public void close() throws IOException {
            if (backing != null) {
                backing.close();
            }
        }
    }

    public abstract PostingVisitor getPostingVisitor(
        FieldInfo fieldInfo,
        FloatVectorValues values,
        IndexInput postingsLists,
        float[] target,
        Bits needsScoring,
        IndexInput centroidSlice,
        ESAcceptDocs acceptDocs
    ) throws IOException;

    public interface PostingVisitor {
        /** returns the number of documents in the posting list */
        int resetPostingsScorer(PostingMetadata metadata) throws IOException;

        /** returns the number of scored documents */
        int visit(KnnCollector collector) throws IOException;
    }

}
