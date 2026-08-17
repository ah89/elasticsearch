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
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidIndex;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * End-to-end correctness of the LensGate: reordering the cluster visit must never change WHAT can
 * be found, only WHEN. Under a full budget the lens arm must return exactly the same document set
 * as the forced-gate scan; under a partial budget it must return only filter matches. The
 * attribute here is deliberately informative (value == planted region) so the lens coefficients
 * are non-trivial, and the vectors are unit-normalized as the cosine benchmark's are.
 */
public class ESNextDiskBBQLensGateTests extends ESTestCase {

    static {
        LogConfigurator.configureESLogging();
    }

    private static final String VECTOR_FIELD = "vector";
    private static final String ATTR_FIELD = "attr";
    private static final int NUM_DOCS = 6_000;
    private static final int DIM = 32;
    private static final int NUM_REGIONS = 24;

    public void testLensReturnsSameSetUnderFullBudget() throws Exception {
        final Random random = new Random(20260811);
        final float[][] regionCenters = new float[NUM_REGIONS][];
        for (int r = 0; r < NUM_REGIONS; r++) {
            regionCenters[r] = randomUnitVector(random, DIM);
        }
        final float[][] vectors = new float[NUM_DOCS][];
        final int[] categories = new int[NUM_DOCS];
        for (int i = 0; i < NUM_DOCS; i++) {
            final int r = random.nextInt(NUM_REGIONS);
            categories[i] = r;
            final float[] x = new float[DIM];
            for (int j = 0; j < DIM; j++) {
                x[j] = regionCenters[r][j] + 0.35f * (float) random.nextGaussian();
            }
            normalize(x);
            vectors[i] = x;
        }
        try (Directory dir = FSDirectory.open(createTempDir("lens-gate"))) {
            buildMergedIndex(dir, new ESNextDiskBBQVectorsFormat(64, 8, null), vectors, categories);
            try (IndexReader reader = DirectoryReader.open(dir)) {
                assertEquals(1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final int[] indexedCategories = readIndexedCategories(leaf);
                publishValueChannel(indexedCategories, NUM_REGIONS);
                try {
                    for (int trial = 0; trial < 6; trial++) {
                        final float[] query = randomUnitVector(random, DIM);
                        final int category = random.nextInt(NUM_REGIONS);
                        final Set<Integer> matches = new HashSet<>();
                        for (int doc = 0; doc < indexedCategories.length; doc++) {
                            if (indexedCategories[doc] == category) {
                                matches.add(doc);
                            }
                        }
                        assertFalse(matches.isEmpty());
                        final int k = 2 * matches.size() + 16;
                        final Set<Integer> scan = search(leaf, query, category, k, 1.0f, false);
                        final Set<Integer> lens = search(leaf, query, category, k, 1.0f, true);
                        assertEquals("full budget: lens must find exactly the scan's set", scan, lens);
                        assertEquals(matches, lens);
                        final Set<Integer> partial = search(leaf, query, category, 10, 0.02f, true);
                        assertEquals(10, partial.size());
                        for (int doc : partial) {
                            assertTrue("lens returned a non-matching doc", matches.contains(doc));
                        }
                    }
                    assertNotNull("lens statistics were never built — lens path did not engage", fieldEntry(leaf).attributeLensIndex());
                } finally {
                    FlatCentroidIndex.POC_DOC_VALUES = null;
                    FlatCentroidIndex.POC_DOC_VALUE_OFFSETS = null;
                    FlatCentroidIndex.POC_NUM_VALUES = 0;
                    FlatCentroidIndex.POC_QUERY_VALUE_SET.remove();
                }
            }
        }
    }

    /** Publishes the per-doc single-valued category ordinals in the CSR form the POC channel expects. */
    private static void publishValueChannel(int[] categories, int numValues) {
        final int[] offsets = new int[categories.length + 1];
        final int[] values = new int[categories.length];
        for (int doc = 0; doc < categories.length; doc++) {
            offsets[doc] = doc;
            values[doc] = categories[doc];
        }
        offsets[categories.length] = categories.length;
        FlatCentroidIndex.POC_DOC_VALUES = values;
        FlatCentroidIndex.POC_DOC_VALUE_OFFSETS = offsets;
        FlatCentroidIndex.POC_NUM_VALUES = numValues;
    }

    private static Set<Integer> search(LeafReader leaf, float[] query, int category, int k, float visitRatio, boolean lens)
        throws IOException {
        try {
            FlatCentroidIndex.POC_FORCE_EXACT_CENTROID_PRUNING = true;
            FlatCentroidIndex.POC_FETCH_GATE = lens;
            FlatCentroidIndex.POC_LENS_GATE = lens;
            FlatCentroidIndex.POC_QUERY_VALUE_SET.set(lens ? new int[] { category } : null);
            final KnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, new IVFKnnSearchStrategy(visitRatio, k, k, null));
            final String term = Integer.toString(category);
            leaf.searchNearestVectors(
                VECTOR_FIELD,
                query,
                collector,
                AcceptDocs.fromIteratorSupplier(() -> leaf.postings(new Term(ATTR_FIELD, term)), leaf.getLiveDocs(), leaf.maxDoc())
            );
            final Set<Integer> docs = new HashSet<>();
            for (ScoreDoc scoreDoc : collector.topDocs().scoreDocs) {
                docs.add(scoreDoc.doc);
            }
            return docs;
        } finally {
            FlatCentroidIndex.POC_FORCE_EXACT_CENTROID_PRUNING = false;
            FlatCentroidIndex.POC_FETCH_GATE = false;
            FlatCentroidIndex.POC_LENS_GATE = false;
            FlatCentroidIndex.POC_QUERY_VALUE_SET.remove();
        }
    }

    private static void buildMergedIndex(Directory dir, ESNextDiskBBQVectorsFormat format, float[][] vectors, int[] categories)
        throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(NUM_DOCS / 2 + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[i], VectorSimilarityFunction.DOT_PRODUCT));
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

    private static void normalize(float[] x) {
        double norm = 0;
        for (float v : x) {
            norm += v * v;
        }
        final float inv = (float) (1.0 / Math.sqrt(norm));
        for (int j = 0; j < x.length; j++) {
            x[j] *= inv;
        }
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
