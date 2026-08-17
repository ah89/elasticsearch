/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidIndex;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Correctness tests for the FetchGate POC (see {@code MatchFetchIndex}): the fetch path must return
 * exactly the documents the gated scan path returns, with the same quantized scores, because it
 * reads the same bytes and applies the same corrections — it only skips the posting entries that
 * fail the filter. Runs the same scenario against a parent-structured centroid index (small
 * clusters) and a flat one (large clusters) since the two use different query-quantization inputs.
 */
public class ESNextDiskBBQMatchFetchTests extends ESTestCase {

    static {
        LogConfigurator.configureESLogging();
    }

    private static final String VECTOR_FIELD = "vector";
    private static final String ATTR_FIELD = "attr";
    private static final int NUM_DOCS = 6_000;
    private static final int DIM = 32;

    public void testFetchMatchesGatedScanWithParents() throws Exception {
        // 6000 docs / 64 per cluster = ~94 centroids -> parent structure engaged;
        // 64 categories (~1.6% selectivity) keeps matches sparse: mostly one per posting block
        runFetchVersusScan(new ESNextDiskBBQVectorsFormat(64, 8, null), 64);
    }

    public void testFetchMatchesGatedScanFlat() throws Exception {
        // 6000 docs / 1024 per cluster = ~6 centroids -> no parents; 24 categories (~4.2%
        // selectivity, still inside the fetch regime) makes multi-match blocks common, so the
        // batched correction-column path is exercised alongside the single-match path
        runFetchVersusScan(new ESNextDiskBBQVectorsFormat(1024, 16, null), 24);
    }

    private void runFetchVersusScan(ESNextDiskBBQVectorsFormat format, int numCategories) throws Exception {
        final Random random = new Random(20260810);
        final float[][] vectors = new float[NUM_DOCS][];
        final int[] categories = new int[NUM_DOCS];
        for (int i = 0; i < NUM_DOCS; i++) {
            vectors[i] = randomUnitVector(random, DIM);
            categories[i] = random.nextInt(numCategories);
        }
        try (Directory dir = FSDirectory.open(createTempDir("fetch-gate"))) {
            buildMergedIndex(dir, format, vectors, categories);
            try (IndexReader reader = DirectoryReader.open(dir)) {
                assertEquals("expected a single segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final int[] indexedCategories = readIndexedCategories(leaf);
                int fetchHitsTotal = 0;
                int scanHitsTotal = 0;
                for (int trial = 0; trial < 8; trial++) {
                    final float[] query = randomUnitVector(random, DIM);
                    final int category = random.nextInt(numCategories);
                    final Set<Integer> matches = new HashSet<>();
                    for (int doc = 0; doc < indexedCategories.length; doc++) {
                        if (indexedCategories[doc] == category) {
                            matches.add(doc);
                        }
                    }
                    assertFalse("degenerate trial: category matches nothing", matches.isEmpty());
                    // the scan can score a SOAR-overspilled doc twice, so the collector needs room
                    // for two entries per match before every distinct doc is guaranteed a slot
                    final int k = 2 * matches.size() + 16;
                    // Reference: full budget AND no centroid pruning, so every copy of every match
                    // is scored. The fetch reaches a document through either cluster that stores it,
                    // including clusters a pruned scan skips, so only an exhaustive scan enumerates
                    // all the scores a copy can legitimately produce.
                    final Map<Integer, Set<Float>> scan = search(leaf, query, category, k, 1.0f, false, false);
                    final Map<Integer, Set<Float>> fetch = search(leaf, query, category, k, 1.0f, true);
                    assertEquals("scan must surface every match under a full budget", matches, scan.keySet());
                    assertEquals("fetch must surface every match under a full budget (no false negatives)", matches, fetch.keySet());
                    // Score parity with the scan is deliberately NOT asserted. A document stored in
                    // two clusters has two quantized copies that estimate its similarity
                    // differently (0.374 vs 0.295 was measured here), and the two paths no longer
                    // read the same copy: the fetch reaches a document through either cluster that
                    // stores it, including clusters the scan's centroid pruning skips. What must
                    // hold is that scoring each match once still ranks them as well as the scan
                    // does, so that is what is checked, against an exact brute-force ranking.
                    for (Set<Float> scores : fetch.values()) {
                        assertEquals("fetch scores each match exactly once", 1, scores.size());
                    }
                    final List<Integer> truth = bruteForceRanking(vectors, query, matches);
                    final int compareAt = Math.min(10, truth.size());
                    final Set<Integer> ideal = new HashSet<>(truth.subList(0, compareAt));
                    // Compared in aggregate, not per trial. Which of a document's two copies gets
                    // scored depends on cluster visit order, so a single trial can go either way by
                    // a position or two; what must not happen is a systematic loss.
                    fetchHitsTotal += topKOverlap(fetch, compareAt, ideal);
                    scanHitsTotal += topKOverlap(scan, compareAt, ideal);

                    // partial budget: correct result shape, still no false positives
                    final int smallK = Math.min(10, matches.size());
                    final Map<Integer, Set<Float>> partial = search(leaf, query, category, smallK, 0.01f, true);
                    assertEquals("collector must fill to k even past the match budget", smallK, partial.size());
                    for (int doc : partial.keySet()) {
                        assertTrue("fetch returned a doc that fails the filter", matches.contains(doc));
                    }
                }
                // canary: the fetch structures must actually have been engaged, otherwise this test
                // compared the scan path against itself
                assertTrue(
                    "fetch ranking is systematically worse than the scan's across trials: " + fetchHitsTotal + " vs " + scanHitsTotal,
                    fetchHitsTotal >= scanHitsTotal - 2
                );
                assertNotNull("fetch geometry was never built — fetch path did not engage", fieldEntry(leaf).matchFetchIndex());
            }
        }
    }

    /** Exact ranking of the matching docs by similarity, best first. */
    private static List<Integer> bruteForceRanking(float[][] vectors, float[] query, Set<Integer> matches) {
        final List<Integer> ranked = new ArrayList<>(matches);
        ranked.sort((a, b) -> Float.compare(similarity(vectors[b], query), similarity(vectors[a], query)));
        return ranked;
    }

    private static float similarity(float[] vector, float[] query) {
        return VectorSimilarityFunction.EUCLIDEAN.compare(vector, query);
    }

    /** How many of the arm's own top {@code n} scoring docs are in the exact top {@code n}. */
    private static int topKOverlap(Map<Integer, Set<Float>> byDoc, int n, Set<Integer> ideal) {
        final List<Map.Entry<Integer, Set<Float>>> ranked = new ArrayList<>(byDoc.entrySet());
        ranked.sort((a, b) -> Float.compare(Collections.max(b.getValue()), Collections.max(a.getValue())));
        int hits = 0;
        for (int i = 0; i < Math.min(n, ranked.size()); i++) {
            if (ideal.contains(ranked.get(i).getKey())) {
                hits++;
            }
        }
        return hits;
    }

    /** Runs one search with the POC flags of the chosen arm and groups the collected scores by doc. */
    private static Map<Integer, Set<Float>> search(LeafReader leaf, float[] query, int category, int k, float visitRatio, boolean fetch)
        throws IOException {
        return search(leaf, query, category, k, visitRatio, fetch, true);
    }

    private static Map<Integer, Set<Float>> search(
        LeafReader leaf,
        float[] query,
        int category,
        int k,
        float visitRatio,
        boolean fetch,
        boolean prune
    ) throws IOException {
        try {
            FlatCentroidIndex.POC_FORCE_EXACT_CENTROID_PRUNING = prune;
            FlatCentroidIndex.POC_FETCH_GATE = fetch;
            final KnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, new IVFKnnSearchStrategy(visitRatio, k, k, null));
            final String term = Integer.toString(category);
            leaf.searchNearestVectors(
                VECTOR_FIELD,
                query,
                collector,
                AcceptDocs.fromIteratorSupplier(() -> leaf.postings(new Term(ATTR_FIELD, term)), leaf.getLiveDocs(), leaf.maxDoc())
            );
            final TopDocs topDocs = collector.topDocs();
            final Map<Integer, Set<Float>> byDoc = new HashMap<>();
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                byDoc.computeIfAbsent(scoreDoc.doc, d -> new HashSet<>()).add(scoreDoc.score);
            }
            return byDoc;
        } finally {
            FlatCentroidIndex.POC_FORCE_EXACT_CENTROID_PRUNING = false;
            FlatCentroidIndex.POC_FETCH_GATE = false;
        }
    }

    /** Two flush segments force-merged into one merge-built segment, like the benchmark harness builds. */
    private static void buildMergedIndex(Directory dir, ESNextDiskBBQVectorsFormat format, float[][] vectors, int[] categories)
        throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(NUM_DOCS / 2 + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                final String term = Integer.toString(categories[i]);
                doc.add(new StringField(ATTR_FIELD, term, Field.Store.NO));
                doc.add(new SortedDocValuesField(ATTR_FIELD, newBytesRef(term)));
                writer.addDocument(doc);
                if (i == NUM_DOCS / 2) {
                    writer.commit();
                }
            }
            writer.commit();
            writer.forceMerge(1);
        }
    }

    private static int[] readIndexedCategories(LeafReader leafReader) throws IOException {
        final int[] categories = new int[leafReader.maxDoc()];
        Arrays.fill(categories, -1);
        final SortedDocValues values = DocValues.getSorted(leafReader, ATTR_FIELD);
        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            categories[doc] = Integer.parseInt(values.lookupOrd(values.ordValue()).utf8ToString());
        }
        return categories;
    }

    private static ESNextDiskBBQVectorsReader.NextFieldEntry fieldEntry(LeafReader leafReader) {
        org.apache.lucene.codecs.KnnVectorsReader vectorsReader = ((org.apache.lucene.index.CodecReader) leafReader).getVectorReader();
        if (vectorsReader instanceof org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
            vectorsReader = fieldsReader.getFieldReader(VECTOR_FIELD);
        }
        return ((ESNextDiskBBQVectorsReader) vectorsReader).fieldEntry(VECTOR_FIELD);
    }

    private static float[] randomUnitVector(Random random, int dim) {
        final float[] vector = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            vector[i] = (float) random.nextGaussian();
            norm += vector[i] * vector[i];
        }
        final float inverseNorm = (float) (1.0 / Math.sqrt(norm));
        for (int i = 0; i < dim; i++) {
            vector[i] *= inverseNorm;
        }
        return vector;
    }
}
