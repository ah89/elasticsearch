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
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.DiskBBQBulkWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.PrefetchingCentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.PrefixLayout;
import org.elasticsearch.index.codec.vectors.diskbbq.PrefixSuffixScoreCombiner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.simdvec.ES92Int7VectorsScorer;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
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
        VectorPreconditioner {

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

    /**
     * Rescore-style oversample from mivf (NaN if unset or missing field entry), for merging with
     * query and mapping in {@code DenseVectorFieldMapper}; not used for centroid visit ratio.
     */
    public float getRescoreOversample(FieldInfo fieldInfo) {
        final NextFieldEntry e = fields.get(fieldInfo.number);
        return e == null ? Float.NaN : e.rescoreOversample();
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
        final NextFieldEntry fieldEntry = fields.get(fieldInfo.number);
        // build optmization filters if possible
        final FixedBitSet acceptCentroids = getCentroidFilter(centroids, numCentroids, values, acceptDocs, approximateCost);
        final int numParents = centroids.readVInt();
        final FixedBitSet acceptParents = getParentCentroidFilter(centroids, numParents, numCentroids, acceptDocs, fieldEntry.numSlices);
        // build centroid search helpers
        final int bulkSize = fieldEntry.getBulkSize();
        final int dimension = fieldInfo.getVectorDimension();
        final OptimizedScalarQuantizer scalarQuantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        final boolean splitLayout = useSplitLayout(dimension, versionMeta, fieldInfo.getVectorSimilarityFunction());
        final boolean needsFullContext = (splitLayout == false) || numParents > 0;
        final ScoringContext fullContext = needsFullContext
            ? buildScoringContext(
                centroids,
                scalarQuantizer,
                targetQuery,
                fieldEntry.globalCentroid(),
                0,
                dimension,
                bulkSize,
                /* trailingDim */ 0
            )
            : null;
        final ScoringContext childrenContext = splitLayout
            ? buildScoringContext(
                centroids,
                scalarQuantizer,
                targetQuery,
                fieldEntry.globalCentroid(),
                0,
                PrefixLayout.prefixLength(dimension),
                bulkSize,
                PrefixLayout.suffixLength(dimension)
            )
            : fullContext;
        // Suffix-only scorer for top-K refinement: same OSQ quantization as the children scorer
        // but over the suffix slice of the global centroid.
        final ScoringContext suffixContext = splitLayout
            ? buildScoringContext(
                centroids,
                scalarQuantizer,
                targetQuery,
                fieldEntry.globalCentroid(),
                PrefixLayout.prefixLength(dimension),
                PrefixLayout.suffixLength(dimension),
                bulkSize,
                /* trailingDim */ 0
            )
            : null;
        // build iterator
        CentroidIterator centroidIterator;
        if (numParents > 0) {
            // equivalent to (float) centroidsPerParentCluster / 2
            float centroidOversampling = (float) fieldEntry.numCentroids() / (2 * numParents);
            centroidIterator = getCentroidIteratorWithParents(
                fieldInfo,
                centroids,
                numParents,
                numCentroids,
                fullContext,
                childrenContext,
                suffixContext,
                visitRatio * centroidOversampling,
                acceptParents,
                acceptCentroids,
                bulkSize
            );
        } else {
            if (acceptCentroids != null && acceptParents != null) {
                acceptCentroids.and(acceptParents);
            }
            centroidIterator = getCentroidIteratorNoParent(
                fieldInfo,
                centroids,
                numCentroids,
                childrenContext,
                suffixContext,
                acceptCentroids != null ? acceptCentroids : acceptParents,
                bulkSize
            );
        }
        return getPostingListPrefetchIterator(centroidIterator, postingListSlice);
    }

    /**
     * Smallest dimension where splitting into a prefix/suffix is worth the extra per-query work
     * (second scoring context, refinement pass, needsScore mask, combiner).
     */
    static final int MIN_PROFITABLE_SPLIT_DIMENSION = 512;

    /**
     * Whether to use the v2 prefix/suffix split layout. Must match
     * {@code ESNextDiskBBQVectorsWriter#useSplitLayout} exactly or the reader will misread the file.
     *
     * <p>EUCLIDEAN is out. Until preconditioning lands, EUCLIDEAN
     * goes through the full-dim path. COSINE / DOT_PRODUCT keep the split.
     *
     * <p>Dimension floor: see {@link #MIN_PROFITABLE_SPLIT_DIMENSION}.
     */
    private static boolean useSplitLayout(int dimension, int version, VectorSimilarityFunction similarityFunction) {
        return version >= ESNextDiskBBQVectorsFormat.VERSION_PREFIX_SPLIT_CENTROIDS
            && PrefixLayout.isEnabled(dimension)
            && dimension >= MIN_PROFITABLE_SPLIT_DIMENSION
            && similarityFunction != VectorSimilarityFunction.EUCLIDEAN;
    }

    /**
     * Bundle of the per-region state needed to score a centroid (or a slice of one). The
     * {@code scorer} is wired to a specific dimension at construction so split readers carry two
     * separate contexts. {@code bytesPerVector} is the per-vector byte cost of the data this
     * scorer reads (full dim for the unsplit / parents / full context, prefix dim for the
     * children context, suffix dim for the suffix context). {@code trailingBytesPerVector} is
     * the per-vector cost of any sibling region that physically trails this scorer's data on
     * disk; non-zero only for the children context of a v2 split layout, whose data is followed
     * by the global suffix region.
     */
    private record ScoringContext(
        ES92Int7VectorsScorer scorer,
        byte[] quantizedQuery,
        OptimizedScalarQuantizer.QuantizationResult queryParams,
        long bytesPerVector,
        long trailingBytesPerVector,
        float centroidDp
    ) {}

    private static ScoringContext buildScoringContext(
        IndexInput centroids,
        OptimizedScalarQuantizer scalarQuantizer,
        float[] targetQuery,
        float[] globalCentroid,
        int from,
        int scorerDim,
        int bulkSize,
        int trailingDim
    ) throws IOException {
        // OptimizedScalarQuantizer requires every array (vector, residual, destination) to be the
        // same length, so we cannot share buffers across contexts of different dim. The only
        // allocation we can skip cheaply is the slice copy when the request already covers the
        // whole array — taken below for the full-dim context, leaving the split-context path with
        // two unavoidable float[scorerDim] slice copies plus the required per-context scratch/
        // residual/quantized arrays. An offset/length variant of scalarQuantize would remove
        // those copies.
        final float[] queryView = (from == 0 && scorerDim == targetQuery.length) ? targetQuery : sliceCopy(targetQuery, from, scorerDim);
        final float[] centroidView = (from == 0 && scorerDim == globalCentroid.length)
            ? globalCentroid
            : sliceCopy(globalCentroid, from, scorerDim);
        final int[] scratch = new int[scorerDim];
        final OptimizedScalarQuantizer.QuantizationResult queryParams = scalarQuantizer.scalarQuantize(
            queryView,
            new float[scorerDim],
            scratch,
            (byte) 7,
            centroidView
        );
        final byte[] quantized = new byte[scorerDim];
        for (int i = 0; i < scorerDim; i++) {
            quantized[i] = (byte) scratch[i];
        }
        return new ScoringContext(
            ESVectorUtil.getES92Int7VectorsScorer(centroids, scorerDim, bulkSize),
            quantized,
            queryParams,
            DiskBBQBulkWriter.largeBitBytesPerVector(scorerDim),
            trailingDim == 0 ? 0L : DiskBBQBulkWriter.largeBitBytesPerVector(trailingDim),
            // Dot product of the slice of the global centroid that this scorer reads against
            // itself — matches the convention used by the bulk scorer's centroidDp argument and
            // by ES92Int7VectorsScorer#applyCorrections to recover the un-centered score.
            ESVectorUtil.dotProduct(centroidView, centroidView)
        );
    }

    private static float[] sliceCopy(float[] src, int from, int length) {
        float[] out = new float[length];
        System.arraycopy(src, from, out, 0, length);
        return out;
    }

    private FixedBitSet getCentroidFilter(
        IndexInput centroids,
        int numCentroids,
        FloatVectorValues values,
        AcceptDocs acceptDocs,
        float approximateCost
    ) throws IOException {
        float approximateDocsPerCentroid = approximateCost / numCentroids;
        if (approximateDocsPerCentroid <= 1.25) {
            // TODO: we need to make this call to build the iterator, otherwise accept docs breaks all together
            approximateDocsPerCentroid = (float) acceptDocs.cost() / numCentroids;
        }
        final int bitsRequired = DirectWriter.bitsRequired(numCentroids);
        final long sizeLookup = DirectWriter.bytesRequired(values.size(), bitsRequired);
        long fp = centroids.getFilePointer();
        final FixedBitSet acceptCentroids;
        if (approximateDocsPerCentroid > 1.25 || numCentroids == 1 || acceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll) {
            // only apply centroid filtering when we expect some / many centroids will not have
            // any matching document.
            acceptCentroids = null;
        } else {
            acceptCentroids = new FixedBitSet(numCentroids);
            final KnnVectorValues.DocIndexIterator docIndexIterator = values.iterator();
            final DocIdSetIterator iterator = ConjunctionUtils.intersectIterators(List.of(acceptDocs.iterator(), docIndexIterator));
            final LongValues longValues = DirectReader.getInstance(centroids.randomAccessSlice(fp, sizeLookup), bitsRequired);
            int doc = iterator.nextDoc();
            for (; doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                acceptCentroids.set((int) longValues.get(docIndexIterator.index()));
            }
        }
        centroids.seek(fp + sizeLookup);
        return acceptCentroids;
    }

    private FixedBitSet getParentCentroidFilter(
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
        ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding = ESNextDiskBBQVectorsFormat.QuantEncoding.fromId(input.readInt());
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
    public CentroidData readCentroidData(FieldInfo fieldInfo) throws IOException {
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
        private final ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding;
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
            ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
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
            this.quantEncoding = quantEncoding;
            this.preconditionerOffset = preconditionerOffset;
            this.preconditionerLength = preconditionerLength;
            this.numSlices = numSlices;
            this.maxSliceSize = maxSliceSize;
            this.rescoreOversample = rescoreOversample;
        }

        public ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding() {
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
    }

    private static CentroidIterator getCentroidIteratorNoParent(
        FieldInfo fieldInfo,
        IndexInput centroids,
        int numCentroids,
        ScoringContext ctx,
        ScoringContext suffixContext,
        FixedBitSet acceptCentroids,
        int bulkSize
    ) throws IOException {
        assert suffixContext == null || ctx.trailingBytesPerVector() == suffixContext.bytesPerVector();
        final NeighborQueue neighborQueue = new NeighborQueue(numCentroids, true);
        score(
            neighborQueue,
            numCentroids,
            0,
            centroids,
            ctx,
            fieldInfo.getVectorSimilarityFunction(),
            new float[bulkSize],
            acceptCentroids,
            bulkSize
        );
        final long suffixRegionStart = centroids.getFilePointer();
        if (suffixContext != null && neighborQueue.size() > 0) {
            refineTopKWithSuffix(
                neighborQueue,
                centroids,
                suffixRegionStart,
                numCentroids,
                suffixContext,
                bulkSize,
                fieldInfo.getVectorSimilarityFunction()
            );
        }
        // Land at the posting-offsets table. When the segment has a suffix region this skips past
        // it; when it doesn't, ctx.trailingBytesPerVector() == 0 and the seek is a no-op.
        final long offset = suffixRegionStart + (long) numCentroids * ctx.trailingBytesPerVector();
        centroids.seek(offset);
        return new CentroidIterator() {
            @Override
            public boolean hasNext() {
                return neighborQueue.size() > 0;
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                long centroidOrdinalAndScore = neighborQueue.popRaw();
                int centroidOrd = neighborQueue.decodeNodeId(centroidOrdinalAndScore);
                float score = neighborQueue.decodeScore(centroidOrdinalAndScore);
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
        ScoringContext parentsCtx,
        ScoringContext childrenCtx,
        ScoringContext suffixContext,
        float centroidRatio,
        FixedBitSet acceptParents,
        FixedBitSet acceptCentroids,
        int bulkSize
    ) throws IOException {
        // children suffix bytes form a global region (not grouped by parent) located at
        // childrenOffset + numCentroids * childrenCtx.bytesPerVector(); top-K refinement seeks
        // into it after the initial fill below.
        assert suffixContext == null || childrenCtx.trailingBytesPerVector() == suffixContext.bytesPerVector();
        // build the three queues we are going to use
        final long rawParentSize = (long) fieldInfo.getVectorDimension() * Float.BYTES;
        final long childrenTotalBytesPerVector = childrenCtx.bytesPerVector() + childrenCtx.trailingBytesPerVector();
        final NeighborQueue parentsQueue = new NeighborQueue(numParents, true);
        final int maxChildrenSize = centroids.readVInt();
        final NeighborQueue currentParentQueue = new NeighborQueue(maxChildrenSize, true);
        final int bufferSize = (int) Math.min(Math.max(centroidRatio * numCentroids, 1), numCentroids);
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
            // Skip past raw parents + quantized parents. Parents are NOT split, so this uses the
            // parents' full-dim byte cost from parentsCtx.
            centroids.skipBytes((parentsCtx.bytesPerVector() + rawParentSize) * numParents);
        } else {
            neighborQueue = new NeighborQueue(bufferSize, true);
            // score the parents (always full-dim, never split)
            centroids.skipBytes(rawParentSize * numParents);
            score(
                parentsQueue,
                numParents,
                0,
                centroids,
                parentsCtx,
                fieldInfo.getVectorSimilarityFunction(),
                scores,
                acceptParents,
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
                childrenCtx,
                fieldInfo,
                scores,
                acceptCentroids,
                bulkSize
            );
            while (currentParentQueue.size() > 0 && neighborQueue.size() < bufferSize) {
                final float score = currentParentQueue.topScore();
                final int children = currentParentQueue.pop();
                neighborQueue.add(children, score);
            }
        }
        // Refine the initial fill using the global suffix region. Children populated on-demand
        // later by the iterator's nextCentroid keep their prefix-only scores; they come from
        // successively-worse parents, so the marginal recall benefit of refining them is small.
        if (suffixContext != null && neighborQueue.size() > 0) {
            final long childrenSuffixRegionStart = childrenOffset + childrenCtx.bytesPerVector() * numCentroids;
            refineTopKWithSuffix(
                neighborQueue,
                centroids,
                childrenSuffixRegionStart,
                numCentroids,
                suffixContext,
                bulkSize,
                fieldInfo.getVectorSimilarityFunction()
            );
        }
        // The posting offsets table follows the entire children section (prefix region + suffix
        // region for split layouts). childrenTotalBytesPerVector covers both regions per child.
        final long childrenFileOffsets = childrenOffset + childrenTotalBytesPerVector * numCentroids;
        return new CentroidIterator() {

            @Override
            public boolean hasNext() {
                return neighborQueue.size() > 0;
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                long centroidOrdinalAndScore = nextCentroid();
                int centroidOrdinal = neighborQueue.decodeNodeId(centroidOrdinalAndScore);
                float score = neighborQueue.decodeScore(centroidOrdinalAndScore);
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
                        childrenCtx,
                        fieldInfo,
                        scores,
                        acceptCentroids,
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
        ScoringContext childrenCtx,
        FieldInfo fieldInfo,
        float[] scores,
        FixedBitSet acceptCentroids,
        int bulkSize
    ) throws IOException {
        centroids.seek(parentOffset);
        int childrenOrdinal = centroids.readInt();
        int numChildren = centroids.readInt();
        // The writer lays out ALL children's prefix bytes contiguously across parent groups
        // (see ESNextDiskBBQVectorsWriter.writeCentroidsWithParents), so we seek into the global
        // prefix region using only the prefix-byte stride. Suffix bytes for these children live
        // in a separate global suffix region addressed by random-access seek elsewhere.
        centroids.seek(childrenOffset + childrenCtx.bytesPerVector() * childrenOrdinal);
        score(
            neighborQueue,
            numChildren,
            childrenOrdinal,
            centroids,
            childrenCtx,
            fieldInfo.getVectorSimilarityFunction(),
            scores,
            acceptCentroids,
            bulkSize
        );
    }

    private static void score(
        NeighborQueue neighborQueue,
        int size,
        int scoresOffset,
        IndexInput centroids,
        ScoringContext ctx,
        VectorSimilarityFunction similarityFunction,
        float[] scores,
        FixedBitSet acceptCentroids,
        int bulkSize
    ) throws IOException {
        // Score `size` consecutive vectors at the current `centroids` position. The scorer's
        // wrapped IndexInput is the same `centroids` we manipulate above, so it advances by
        // exactly ctx.bytesPerVector() per vector. For split layouts the caller handles any
        // trailing suffix region:
        // * no-parents path: skip the trailing suffix region once after this call.
        // * with-parents children path: do nothing — suffix region is global and is addressed
        // by random-access seek elsewhere.
        final long bytesPerVector = ctx.bytesPerVector();
        final byte[] quantizeQuery = ctx.quantizedQuery();
        final OptimizedScalarQuantizer.QuantizationResult queryCorrections = ctx.queryParams();
        // Pull centroidDp from the context so it always matches the dimension slice this scorer
        // reads. Passing a full-vector centroidDp into a prefix-only or suffix-only scorer
        // biases every score by `D_global - D_slice`, which compounds across prefix + suffix in
        // PrefixSuffixScoreCombiner and clamps refined scores to zero.
        final float centroidDp = ctx.centroidDp();
        int limit = size - bulkSize + 1;
        int i = 0;
        for (; i < limit; i += bulkSize) {
            if (acceptCentroids == null || acceptCentroids.cardinality(scoresOffset + i, scoresOffset + i + bulkSize) > 0) {
                ctx.scorer()
                    .scoreBulk(
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
                        neighborQueue.add(centroidOrd, scores[j]);
                    }
                }
            } else {
                centroids.skipBytes(bulkSize * bytesPerVector);
            }
        }

        int tailBulkSize = size - i;
        if (tailBulkSize > 0) {
            if (acceptCentroids == null || acceptCentroids.cardinality(scoresOffset + i, scoresOffset + i + tailBulkSize) > 0) {
                ctx.scorer()
                    .scoreBulk(
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
                        neighborQueue.add(centroidOrd, scores[j]);
                    }
                }
            } else {
                centroids.skipBytes(tailBulkSize * bytesPerVector);
            }
        }
    }

    /**
     * Refine count tuning. We refine top {@code max(REFINE_MIN, queueSize / REFINE_FRACTION)}
     * centroids after the prefix pass: refining more improves recall, refining fewer cuts cost.
     * Picked as a heuristic and worth re-tuning with benchmarks.
     */
    private static final int REFINE_MIN = 64;

    private static final int REFINE_FRACTION = 10;

    /**
     * Drains the top of {@code neighborQueue} (already populated by prefix-only scoring), reads
     * matching suffix bytes via {@code suffixContext} for any block that contains a top-K
     * centroid, combines prefix + suffix via {@link PrefixSuffixScoreCombiner}, and pushes the
     * refined entries back into the queue. Remaining (un-refined) entries keep their prefix-only
     * score; the heap order naturally reflects whichever score is higher.
     *
     * <p><b>Layout-driven access pattern:</b> the suffix region is written by
     * {@code LargeBitEncodedDiskBBQBulkWriter} in block-interleaved layout
     * ({@code [bulkSize * dim bytes][bulkSize * 3 floats][bulkSize * 1 int]} per block). Random
     * access via {@link ES92Int7VectorsScorer#score} would land on raw bytes but read garbage
     * correction values from the neighbouring vector. The only safe reader is
     * {@link ES92Int7VectorsScorer#scoreBulk}, which consumes one full block at a time. So we
     * walk the suffix region sequentially: for blocks that contain a top-K centroid we bulk-
     * score, for blocks that do not we {@code skipBytes} past them.
     *
     * <p><b>Worst case:</b> if every top-K centroid lands in a different block we bulk-score
     * {@code refineCount * bulkSize} suffix vectors instead of {@code refineCount}; in
     * common IVF sizings ({@code numCentroids >> refineCount * bulkSize}) this is still well
     * under reading the entire suffix region.
     */
    private static void refineTopKWithSuffix(
        NeighborQueue neighborQueue,
        IndexInput centroids,
        long suffixRegionStart,
        int numCentroids,
        ScoringContext suffixContext,
        int bulkSize,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        final int refineCount = Math.min(neighborQueue.size(), Math.max(REFINE_MIN, neighborQueue.size() / REFINE_FRACTION));
        if (refineCount == 0) {
            return;
        }
        final int[] refinedIds = new int[refineCount];
        final float[] refinedPrefixScores = new float[refineCount];
        // Drain the prefix-only top-K and remember which centroid ordinals need refinement.
        // needsScore lets us decide per-block "any of mine here?" with one array lookup per
        // vector inside the sequential pass below.
        final boolean[] needsScore = new boolean[numCentroids];
        for (int i = 0; i < refineCount; i++) {
            final long raw = neighborQueue.popRaw();
            final int centroidOrd = neighborQueue.decodeNodeId(raw);
            refinedIds[i] = centroidOrd;
            refinedPrefixScores[i] = neighborQueue.decodeScore(raw);
            needsScore[centroidOrd] = true;
        }
        // Score per-block, leaving an entry only for centroids in needsScore.
        final float[] suffixScores = new float[numCentroids];
        final float[] scratch = new float[bulkSize];
        // suffixContext.bytesPerVector() is the per-vector cost of the suffix scorer's data,
        // which IS the suffix stride on disk (raw dim bytes + 16-byte correction footer).
        final long suffixStride = suffixContext.bytesPerVector();
        final OptimizedScalarQuantizer.QuantizationResult queryParams = suffixContext.queryParams();
        final byte[] quantizedQuery = suffixContext.quantizedQuery();
        // centroidDp must match the suffix slice that this scorer reads — using the full-vector
        // centroidDp here biases every refined score downward by a constant, which
        // PrefixSuffixScoreCombiner#combine then doubles when summing prefix + suffix. The
        // double-bias clamps refined scores to zero, demoting the actually-good centroids below
        // the un-refined ones and collapsing recall.
        final float centroidDp = suffixContext.centroidDp();
        centroids.seek(suffixRegionStart);
        int i = 0;
        final int fullBlockLimit = numCentroids - bulkSize + 1;
        for (; i < fullBlockLimit; i += bulkSize) {
            if (anyNeeded(needsScore, i, bulkSize)) {
                suffixContext.scorer()
                    .scoreBulk(
                        quantizedQuery,
                        queryParams.lowerInterval(),
                        queryParams.upperInterval(),
                        queryParams.quantizedComponentSum(),
                        queryParams.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        scratch,
                        bulkSize
                    );
                System.arraycopy(scratch, 0, suffixScores, i, bulkSize);
            } else {
                centroids.skipBytes((long) bulkSize * suffixStride);
            }
        }
        final int tail = numCentroids - i;
        if (tail > 0) {
            if (anyNeeded(needsScore, i, tail)) {
                // LargeBitEncodedDiskBBQBulkWriter writes the trailing partial block with the
                // same interleaved layout but with bulkSize=tail, so scoreBulk(tail) reads it
                // exactly.
                suffixContext.scorer()
                    .scoreBulk(
                        quantizedQuery,
                        queryParams.lowerInterval(),
                        queryParams.upperInterval(),
                        queryParams.quantizedComponentSum(),
                        queryParams.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        scratch,
                        tail
                    );
                System.arraycopy(scratch, 0, suffixScores, i, tail);
            } else {
                centroids.skipBytes((long) tail * suffixStride);
            }
        }
        // Push refined entries back into the queue.
        for (int k = 0; k < refineCount; k++) {
            final int ord = refinedIds[k];
            final float refined = PrefixSuffixScoreCombiner.combine(similarityFunction, refinedPrefixScores[k], suffixScores[ord]);
            neighborQueue.add(ord, refined);
        }
    }

    private static boolean anyNeeded(boolean[] needsScore, int from, int len) {
        for (int j = 0; j < len; j++) {
            if (needsScore[from + j]) {
                return true;
            }
        }
        return false;
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
        ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding = entry.quantEncoding();
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
        private final ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding;
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

        QueryQuantizer(
            ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
            FieldInfo fieldInfo,
            float[] target,
            IndexInput parentsSlice,
            float[] globalCentroid
        ) {
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
            ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
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

        private final QueryQuantizer queryQuantizer;
        final DocIdsWriter idsWriter = new DocIdsWriter();
        final VectorSimilarityFunction similarityFunction;
        final long quantizedVectorByteSize;

        MemorySegmentPostingsVisitor(
            QueryQuantizer queryQuantizer,
            ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
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
                quantEncoding.bits() == 4 ? ES940OSQVectorsScorer.BitEncoding.PACKED : ES940OSQVectorsScorer.BitEncoding.STRIPED
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
