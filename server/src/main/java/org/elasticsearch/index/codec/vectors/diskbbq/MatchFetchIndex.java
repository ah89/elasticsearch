/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.internal.hppc.LongIntHashMap;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;

import java.io.IOException;
import java.util.Arrays;

/**
 * Per-segment geometry that lets a filtered search score a document's quantized vector by jumping
 * straight to its bytes inside the posting file — the "FetchGate".
 *
 * <p>The existing cluster gates (Mariano's value-set bitmap, the counted gate, the SliceGate) all
 * answer the same binary question — "does this posting list contain at least one filter match?" —
 * and therefore all pay the same fundamental cost once a posting list survives: the whole list is
 * scanned, and under a selective filter almost every entry scanned fails the filter. The scan cannot
 * stop before it has seen enough <em>matching</em> documents, so the number of posting entries
 * visited has a floor of roughly {@code k / selectivity} however good the gate is. When the filter
 * is uncorrelated with the clustering no posting list is empty, the gate prunes nothing, and that
 * floor is the entire query cost.
 *
 * <p>This class removes the scan instead of gating it. One sequential sweep over the posting lists
 * (once per segment, filter-agnostic — the same cost class as unpacking the SliceGate's lookup)
 * records for every document the file offset of its posting block, its slot within the block, and
 * the block's size. With that geometry, a filtered search can iterate the surviving posting lists in
 * centroid-distance order exactly as before, but score <em>only the matching documents</em> of each
 * list: seek to the slot, run the same quantized scorer over the same bytes, apply the same per-doc
 * corrections. Scores are bit-identical to the full scan's, no false negatives are possible, and the
 * per-query work becomes proportional to matches scored rather than posting entries visited.
 *
 * <p>Documents placed in two posting lists by SOAR overspill are recorded once, at the occurrence in
 * the cluster the packed doc&rarr;centroid lookup names (the same cluster the counted gate and the
 * cluster bitmaps attribute the document to), so the fetch path scores each match exactly once and
 * agrees with the gates about which cluster holds it.
 *
 * <p>This POC builds the geometry on first use and keeps it on the field entry; a production format
 * would write it at merge time next to the doc&rarr;centroid lookup it complements — it is a
 * permutation of information the index already has, roughly {@code log2(postingBytes)} bits per
 * document.
 */
public final class MatchFetchIndex {

    private static final long PTR_MASK = (1L << 46) - 1;
    private static final int SLOT_SHIFT = 46;
    private static final int COUNT_SHIFT = 52;
    private static final long PRESENT = 1L << 63;

    /**
     * Sentinel cached on the field entry when the segment's layout cannot be modeled, so the build
     * is attempted once rather than once per query.
     */
    public static final MatchFetchIndex UNSUPPORTED = new MatchFetchIndex(new long[0], new long[0], new int[0], new LongIntHashMap(0), 0);

    /** Occurrence selector: the copy stored in the document's own (k-means owning) cluster. */
    public static final int PRIMARY = 0;
    /** Occurrence selector: the SOAR overspill copy, stored in a second cluster. */
    public static final int OVERSPILL = 1;

    /**
     * Per-document packed geometry: bits 0..45 the file offset of the quantized section of the
     * document's posting block, bits 46..51 the slot within the block, bits 52..57 the block's
     * vector count (tail blocks are short), bit 63 presence. Zero means "no vector recorded".
     */
    private final long[] docMeta;
    /**
     * The same packing for the document's SOAR overspill copy, zero when it has none. Recording it
     * is what lets a cluster visit cover every match <em>physically stored</em> in that cluster.
     * Indexing only the owning cluster's copy makes a document reachable through one cluster only,
     * so a visit to the cluster holding its overspill copy silently skips it -- measured at 1399 of
     * 2575 documents covered, and 8 recall points, where the filter sits on the query's own region.
     */
    private final long[] altMeta;
    /** Cluster holding each document's overspill copy, -1 when it has none. */
    private final int[] altCentroid;
    /** Resolves a posting list's file offset back to its centroid ordinal. */
    private final LongIntHashMap centroidByOffset;
    private final int numCentroids;

    private MatchFetchIndex(long[] docMeta, long[] altMeta, int[] altCentroid, LongIntHashMap centroidByOffset, int numCentroids) {
        this.docMeta = docMeta;
        this.altMeta = altMeta;
        this.altCentroid = altCentroid;
        this.centroidByOffset = centroidByOffset;
        this.numCentroids = numCentroids;
    }

    public int maxDoc() {
        return docMeta.length;
    }

    /** Cluster holding {@code doc}'s overspill copy, or -1 when it has none. */
    public int overspillCentroid(int doc) {
        return doc < altCentroid.length ? altCentroid[doc] : -1;
    }

    public int numCentroids() {
        return numCentroids;
    }

    /** @return the centroid ordinal whose posting list starts at {@code offset}, or -1 if unknown */
    public int centroidForOffset(long offset) {
        return centroidByOffset.getOrDefault(offset, -1);
    }

    public boolean hasDoc(int doc) {
        return doc < docMeta.length && (docMeta[doc] & PRESENT) != 0;
    }

    private long packed(int doc, int occurrence) {
        return occurrence == PRIMARY ? docMeta[doc] : altMeta[doc];
    }

    /** File offset of the quantized section of the block holding this copy of {@code doc}. */
    public long blockOffset(int doc, int occurrence) {
        return packed(doc, occurrence) & PTR_MASK;
    }

    /** Slot of this copy of {@code doc} within its block. */
    public int slot(int doc, int occurrence) {
        return (int) ((packed(doc, occurrence) >>> SLOT_SHIFT) & 0x3F);
    }

    /** Number of vectors in the block holding this copy of {@code doc} (short for tail blocks). */
    public int blockCount(int doc, int occurrence) {
        return (int) ((packed(doc, occurrence) >>> COUNT_SHIFT) & 0x3F);
    }

    public long ramBytesUsed() {
        return (long) docMeta.length * Long.BYTES + (long) altMeta.length * Long.BYTES + (long) altCentroid.length * Integer.BYTES
            + (long) centroidByOffset.size() * (Long.BYTES + Integer.BYTES + 16);
    }

    /**
     * Builds the geometry with one sequential sweep over every posting list, mirroring the block
     * layout the posting visitor reads: per block, the delta-encoded doc ids, then
     * {@code count * quantizedVectorByteSize} quantized vectors, then four correction columns of
     * {@code count} entries each.
     *
     * @param postingSlice the posting-list slice, positioned anywhere (this method seeks)
     * @param postingOffsets per-centroid posting list file offsets
     * @param centroidOfOrdinal packed doc&rarr;centroid lookup, addressed by vector ordinal
     * @param docToOrdinal vector ordinal per doc id, -1 when the doc has no vector
     * @param maxDoc doc-id space bound
     * @param bulkSize the writer's block size ({@code BULK_SIZE})
     * @param quantizedVectorByteSize packed doc vector length in bytes
     * @return the geometry, or {@code null} when the layout cannot be recorded (block counts or
     *         offsets exceeding the packed field widths — not expected for any real segment)
     */
    public static MatchFetchIndex build(
        IndexInput postingSlice,
        long[] postingOffsets,
        LongValues centroidOfOrdinal,
        int[] docToOrdinal,
        int maxDoc,
        int bulkSize,
        long quantizedVectorByteSize
    ) throws IOException {
        if (bulkSize > 63) {
            return null;
        }
        final long quantizedByteLength = quantizedVectorByteSize + 3L * Float.BYTES + Integer.BYTES;
        final long[] docMeta = new long[maxDoc];
        final long[] altMeta = new long[maxDoc];
        final int[] altCentroid = new int[maxDoc];
        Arrays.fill(altCentroid, -1);
        final LongIntHashMap centroidByOffset = new LongIntHashMap(postingOffsets.length);
        final DocIdsWriter idsWriter = new DocIdsWriter();
        final int[] docIdsScratch = new int[bulkSize];
        for (int c = 0; c < postingOffsets.length; c++) {
            centroidByOffset.put(postingOffsets[c], c);
            postingSlice.seek(postingOffsets[c]);
            postingSlice.readInt(); // centroid-to-parent distance, not needed here
            final int vectors = postingSlice.readVInt();
            final byte docEncoding = postingSlice.readByte();
            int docBase = 0;
            for (int i = 0; i < vectors;) {
                final int count = Math.min(bulkSize, vectors - i);
                idsWriter.readInts(postingSlice, count, docEncoding, docIdsScratch);
                for (int j = 0; j < count; j++) {
                    docBase += docIdsScratch[j];
                    docIdsScratch[j] = docBase;
                }
                final long blockPtr = postingSlice.getFilePointer();
                if ((blockPtr & ~PTR_MASK) != 0) {
                    return null;
                }
                for (int j = 0; j < count; j++) {
                    final int doc = docIdsScratch[j];
                    if (doc >= maxDoc) {
                        return null;
                    }
                    final int ord = docToOrdinal[doc];
                    if (ord < 0) {
                        continue;
                    }
                    // Record every occurrence, keyed by which cluster holds it. The owning cluster's
                    // copy is the primary; a copy found in any other cluster is the SOAR overspill.
                    // Both are needed for a cluster visit to see every match stored in it; scoring
                    // each document once is enforced at query time instead, by the visited set.
                    final long entry = PRESENT | ((long) count << COUNT_SHIFT) | ((long) j << SLOT_SHIFT) | blockPtr;
                    if ((int) centroidOfOrdinal.get(ord) == c) {
                        docMeta[doc] = entry;
                    } else {
                        // SOAR writes at most one overspill copy per vector
                        assert altCentroid[doc] == -1 : "more than one overspill copy for doc " + doc;
                        altMeta[doc] = entry;
                        altCentroid[doc] = c;
                    }
                }
                postingSlice.skipBytes(quantizedByteLength * count);
                i += count;
            }
        }
        return new MatchFetchIndex(docMeta, altMeta, altCentroid, centroidByOffset, postingOffsets.length);
    }

    /**
     * The filter's matches grouped by centroid: {@code docs[csrOffsets[c] .. csrOffsets[c+1])} are
     * the matching doc ids in centroid {@code c}'s posting list. Built with one walk of the filter's
     * matching documents through the packed doc&rarr;centroid lookup — the same walk the counted
     * gate performs, storing where the matches are instead of only how many there are. Like the
     * counted gate's counts, views are cached per (filter, segment) on the field entry; unlike it,
     * the fetch path only fires in the selective regime, so a cold (never-seen) filter pays a walk
     * proportional to its own small match count.
     */
    public record MatchView(int[] csrOffsets, int[] docs, byte[] occurrence, int distinctMatches) {

        /**
         * Distinct matching documents. This is the filtered-cardinality figure the ordering and
         * budget machinery wants -- {@code docs.length} over-counts, because a document with a SOAR
         * overspill copy is listed under both clusters that physically hold it.
         */
        public int totalMatches() {
            return distinctMatches;
        }

        /** Listed (document, cluster) pairs, i.e. how many copies the view can reach. */
        public int occurrences() {
            return docs.length;
        }

        public int count(int centroid) {
            return csrOffsets[centroid + 1] - csrOffsets[centroid];
        }

        /** Accept-set over centroids — by construction identical to the counted gate's. */
        public FixedBitSet acceptCentroids() {
            final FixedBitSet accept = new FixedBitSet(csrOffsets.length - 1);
            for (int c = 0; c < csrOffsets.length - 1; c++) {
                if (csrOffsets[c + 1] > csrOffsets[c]) {
                    accept.set(c);
                }
            }
            return accept;
        }

        public long ramBytesUsed() {
            return (long) (csrOffsets.length + docs.length) * Integer.BYTES + occurrence.length;
        }

        /**
         * One pass over {@code matchingDocs} (already intersected with the docs that carry vectors)
         * collecting (doc, centroid) pairs, then a counting sort by centroid.
         *
         * @param docIndexIterator the vector values iterator backing {@code matchingDocs}'s
         *                         conjunction, whose {@code index()} yields the current ordinal
         */
        public static MatchView build(
            DocIdSetIterator matchingDocs,
            KnnVectorValues.DocIndexIterator docIndexIterator,
            LongValues centroidOfOrdinal,
            MatchFetchIndex geometry,
            int numCentroids,
            int expectedMatches
        ) throws IOException {
            int[] docs = new int[Math.max(16, 2 * expectedMatches)];
            int[] centroids = new int[docs.length];
            byte[] occurrences = new byte[docs.length];
            int m = 0;
            int distinct = 0;
            for (int doc = matchingDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = matchingDocs.nextDoc()) {
                // room for both of this document's copies
                if (m + 2 > docs.length) {
                    docs = ArrayUtil.grow(docs, m + 2);
                    centroids = ArrayUtil.growExact(centroids, docs.length);
                    occurrences = ArrayUtil.growExact(occurrences, docs.length);
                }
                distinct++;
                docs[m] = doc;
                centroids[m] = (int) centroidOfOrdinal.get(docIndexIterator.index());
                occurrences[m] = PRIMARY;
                m++;
                // list the overspill copy under the cluster that physically holds it, so visiting
                // that cluster reaches this document even though it is owned elsewhere
                final int spill = geometry.overspillCentroid(doc);
                if (spill >= 0) {
                    docs[m] = doc;
                    centroids[m] = spill;
                    occurrences[m] = OVERSPILL;
                    m++;
                }
            }
            final int[] csrOffsets = new int[numCentroids + 1];
            for (int i = 0; i < m; i++) {
                csrOffsets[centroids[i] + 1]++;
            }
            for (int c = 0; c < numCentroids; c++) {
                csrOffsets[c + 1] += csrOffsets[c];
            }
            final int[] grouped = new int[m];
            final byte[] groupedOcc = new byte[m];
            final int[] fill = new int[numCentroids];
            for (int i = 0; i < m; i++) {
                final int c = centroids[i];
                final int at = csrOffsets[c] + fill[c]++;
                grouped[at] = docs[i];
                groupedOcc[at] = occurrences[i];
            }
            // within a cluster the counting sort preserves input order, which is ascending by doc
            // id, and posting lists are doc-id ordered -- so slots ascend with position, as the
            // block-grouped scoring loop assumes
            return new MatchView(csrOffsets, grouped, groupedOcc, distinct);
        }
    }
}
