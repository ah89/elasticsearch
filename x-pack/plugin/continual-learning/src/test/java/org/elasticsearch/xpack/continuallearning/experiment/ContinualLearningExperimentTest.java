/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.experiment;

import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.continuallearning.coreset.GmmCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.KCenterCoreset;
import org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector;
import org.elasticsearch.xpack.continuallearning.projection.JLProjection;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Controlled experiment comparing ContLoRA against a raw-replay baseline
 * (representative of the CL-LoRA family) on novelty detection.
 *
 * <h2>Experimental setup</h2>
 * <ul>
 *   <li>10 domains, each represented as a Gaussian cluster in R^{@value #DIM}.</li>
 *   <li>{@value #TRAIN_PER_DOMAIN} training embeddings per domain for summary construction.</li>
 *   <li>{@value #TEST_PER_DOMAIN} held-out embeddings per domain for evaluation.</li>
 *   <li>Novel-domain probes: embeddings from a fresh Gaussian not seen during training.</li>
 * </ul>
 *
 * <h2>Methods compared</h2>
 * <ol>
 *   <li><b>ContLoRA-GMM</b>: Gaussian Mixture Model coreset (K={@value #GMM_COMPONENTS}).</li>
 *   <li><b>ContLoRA-GMM+JL</b>: GMM coreset with JL projection to d'={@value #JL_DIM}.</li>
 *   <li><b>ContLoRA-kCenter</b>: k-center coreset (k={@value #KCENTER_K}).</li>
 *   <li><b>CL-LoRA (1% replay)</b>: store {@value #REPLAY_SIZE} raw embeddings per domain,
 *       detect via cosine similarity — representative of rehearsal-based LoRA methods
 *       (e.g. CL-LoRA, EASE with LoRA) that rely on a small replay buffer.</li>
 *   <li><b>Raw Full</b>: store all training embeddings (theoretical upper bound).</li>
 * </ol>
 *
 * <h2>Metrics</h2>
 * <ul>
 *   <li>Memory per domain (bytes of domain summary storage).</li>
 *   <li>Novelty-detection F1 (binary: novel vs. known).</li>
 *   <li>Build time (ms to construct domain summary).</li>
 *   <li>Detection time (ms to run novelty check over all domains).</li>
 * </ul>
 *
 * <p>Results are printed as a Markdown table so they can be pasted directly into
 * experiment logs or the paper.
 */
public class ContinualLearningExperimentTest extends ESTestCase {

    // -------------------------------------------------------------------------
    // Experiment hyper-parameters
    // -------------------------------------------------------------------------

    /** Embedding dimension (mirrors ViT-B/16 CLS token dimension downscaled for speed). */
    private static final int DIM = 256;
    /** Number of sequential learning stages / domains. */
    private static final int NUM_DOMAINS = 10;
    /** Training embeddings per domain for summary construction. */
    private static final int TRAIN_PER_DOMAIN = 2000;
    /** Held-out embeddings per domain for evaluation. */
    private static final int TEST_PER_DOMAIN = 500;
    /** Number of novel-domain probe embeddings per evaluation round. */
    private static final int NOVEL_PROBES = 500;
    /** Number of GMM components. */
    private static final int GMM_COMPONENTS = 20;
    /** Number of k-center points. */
    private static final int KCENTER_K = 50;
    /** JL output dimension. */
    private static final int JL_DIM = 64;
    /** Raw replay buffer size (≈ 1% of TRAIN_PER_DOMAIN, representative of CL-LoRA). */
    private static final int REPLAY_SIZE = 20;
    /** Cosine similarity threshold for replay-based novelty detection. */
    private static final float REPLAY_NOVELTY_THRESHOLD = 0.85f;
    /** Coreset-based novelty threshold. */
    private static final float CORESET_NOVELTY_THRESHOLD = 0.5f;
    /** Standard deviation of intra-domain Gaussian noise. */
    private static final double INTRA_DOMAIN_STD = 0.3;
    /** Spacing between domain centres (controls inter-domain separation). */
    private static final double DOMAIN_CENTRE_SCALE = 3.0;

    // -------------------------------------------------------------------------
    // Main experiment entry point
    // -------------------------------------------------------------------------

    public void testExperiment() {
        Random rng = new Random(42L);
        JLProjection jlMatrix = new JLProjection(DIM, JL_DIM, new Random(7L));

        // Generate domain centres: uniformly placed in a unit hypercube and scaled
        float[][] domainCentres = generateDomainCentres(NUM_DOMAINS, DIM, rng);

        // ----------------------------------------------------------------
        // Phase 1: Build domain summaries (training phase)
        // ----------------------------------------------------------------

        List<GmmCoreset> gmmCoresets = new ArrayList<>();
        List<GmmCoreset> gmmJlCoresets = new ArrayList<>();
        List<KCenterCoreset> kcCoresets = new ArrayList<>();
        List<float[][]> replayBuffers = new ArrayList<>();
        List<float[][]> fullBuffers = new ArrayList<>();

        long[] buildTimeGmm = new long[NUM_DOMAINS];
        long[] buildTimeGmmJl = new long[NUM_DOMAINS];
        long[] buildTimeKc = new long[NUM_DOMAINS];
        long[] buildTimeReplay = new long[NUM_DOMAINS];

        for (int t = 0; t < NUM_DOMAINS; t++) {
            float[][] trainEmbeddings = generateDomainEmbeddings(domainCentres[t], TRAIN_PER_DOMAIN, rng);

            // ContLoRA-GMM
            long t0 = System.nanoTime();
            GmmCoreset gmm = GmmCoreset.fit(trainEmbeddings, GMM_COMPONENTS, new Random(rng.nextLong()));
            buildTimeGmm[t] = System.nanoTime() - t0;
            gmmCoresets.add(gmm);

            // ContLoRA-GMM+JL
            t0 = System.nanoTime();
            float[][] projectedEmbeddings = jlMatrix.projectBatch(trainEmbeddings);
            GmmCoreset gmmJl = GmmCoreset.fit(projectedEmbeddings, GMM_COMPONENTS, new Random(rng.nextLong()));
            buildTimeGmmJl[t] = System.nanoTime() - t0;
            gmmJlCoresets.add(gmmJl);

            // ContLoRA-kCenter
            t0 = System.nanoTime();
            KCenterCoreset kc = KCenterCoreset.fit(trainEmbeddings, KCENTER_K, new Random(rng.nextLong()));
            buildTimeKc[t] = System.nanoTime() - t0;
            kcCoresets.add(kc);

            // CL-LoRA replay: store REPLAY_SIZE random embeddings
            t0 = System.nanoTime();
            float[][] replay = subsample(trainEmbeddings, REPLAY_SIZE, rng);
            buildTimeReplay[t] = System.nanoTime() - t0;
            replayBuffers.add(replay);

            // Full raw storage (upper bound)
            fullBuffers.add(trainEmbeddings);
        }

        // ----------------------------------------------------------------
        // Phase 2: Evaluate novelty detection (sequential, domain by domain)
        // ----------------------------------------------------------------

        // After seeing all NUM_DOMAINS domains, evaluate on held-out test set.
        // Positive class: embeddings from a NEW domain (novel = true).
        // Negative class: embeddings from EXISTING domains (novel = false).

        float[][] novelCentre = generateDomainCentres(1, DIM, new Random(999L));
        float[][] novelEmbeddings = generateDomainEmbeddings(novelCentre[0], NOVEL_PROBES, new Random(rng.nextLong()));

        // Known-domain test embeddings
        List<float[][]> knownTestSets = new ArrayList<>();
        for (int t = 0; t < NUM_DOMAINS; t++) {
            knownTestSets.add(generateDomainEmbeddings(domainCentres[t], TEST_PER_DOMAIN, new Random(rng.nextLong())));
        }

        NoveltyDetector detector = new NoveltyDetector(CORESET_NOVELTY_THRESHOLD);

        F1Result gmmF1 = evaluateGmmNovelty(gmmCoresets, knownTestSets, novelEmbeddings, detector, jlMatrix, false);
        F1Result gmmJlF1 = evaluateGmmNovelty(gmmJlCoresets, knownTestSets, novelEmbeddings, detector, jlMatrix, true);
        F1Result kcF1 = evaluateKcNovelty(kcCoresets, knownTestSets, novelEmbeddings, detector);
        F1Result replayF1 = evaluateReplayNovelty(replayBuffers, knownTestSets, novelEmbeddings, REPLAY_NOVELTY_THRESHOLD);

        // ----------------------------------------------------------------
        // Phase 3: Measure detection time (average over 100 probes)
        // ----------------------------------------------------------------

        long detectTimeGmm = measureDetectionTimeGmm(gmmCoresets, novelEmbeddings, GMM_COMPONENTS, detector, jlMatrix, false);
        long detectTimeGmmJl = measureDetectionTimeGmm(gmmJlCoresets, novelEmbeddings, GMM_COMPONENTS, detector, jlMatrix, true);
        long detectTimeKc = measureDetectionTimeKc(kcCoresets, novelEmbeddings, detector);
        long detectTimeReplay = measureDetectionTimeReplay(replayBuffers, novelEmbeddings, REPLAY_NOVELTY_THRESHOLD);

        // ----------------------------------------------------------------
        // Phase 4: Memory accounting
        // ----------------------------------------------------------------

        long memGmm = gmmMemoryBytes(GMM_COMPONENTS, DIM);
        long memGmmJl = gmmMemoryBytes(GMM_COMPONENTS, JL_DIM);
        long memKc = kcMemoryBytes(KCENTER_K, DIM);
        long memReplay = replayMemoryBytes(REPLAY_SIZE, DIM);
        long memFull = replayMemoryBytes(TRAIN_PER_DOMAIN, DIM);

        // ----------------------------------------------------------------
        // Phase 5: Print results as Markdown table
        // ----------------------------------------------------------------

        printResultsTable(
            new String[] { "ContLoRA-GMM", "ContLoRA-GMM+JL", "ContLoRA-kCenter", "CL-LoRA (1% replay)", "Raw Full" },
            new F1Result[] { gmmF1, gmmJlF1, kcF1, replayF1, null },
            new long[] { memGmm, memGmmJl, memKc, memReplay, memFull },
            new long[] {
                avgBuildTime(buildTimeGmm),
                avgBuildTime(buildTimeGmmJl),
                avgBuildTime(buildTimeKc),
                avgBuildTime(buildTimeReplay),
                0L },
            new long[] { detectTimeGmm, detectTimeGmmJl, detectTimeKc, detectTimeReplay, 0L }
        );

        // ----------------------------------------------------------------
        // Assertions: ContLoRA-GMM must beat replay on memory and match on F1
        // ----------------------------------------------------------------

        assertTrue("ContLoRA-GMM must use less memory than 1% replay buffer", memGmm < memReplay);
        assertNotNull("GMM coreset novelty F1 must be computable", gmmF1);
        assertTrue("ContLoRA-GMM F1 must be >= CL-LoRA replay F1 minus a 5pp tolerance", gmmF1.f1() >= replayF1.f1() - 0.05f);
        assertTrue("ContLoRA-GMM+JL must use less memory than plain GMM", memGmmJl < memGmm);
    }

    // =========================================================================
    // Synthetic data generation
    // =========================================================================

    /**
     * Generates {@code numDomains} random domain centres, spaced at
     * {@value #DOMAIN_CENTRE_SCALE} standard deviations apart on average.
     */
    private static float[][] generateDomainCentres(int numDomains, int dim, Random rng) {
        float[][] centres = new float[numDomains][dim];
        for (int i = 0; i < numDomains; i++) {
            for (int j = 0; j < dim; j++) {
                centres[i][j] = (float) (rng.nextGaussian() * DOMAIN_CENTRE_SCALE);
            }
        }
        return centres;
    }

    /**
     * Generates {@code n} embeddings sampled from N(centre, σ²I).
     */
    private static float[][] generateDomainEmbeddings(float[] centre, int n, Random rng) {
        int d = centre.length;
        float[][] embeddings = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                embeddings[i][j] = centre[j] + (float) (rng.nextGaussian() * INTRA_DOMAIN_STD);
            }
        }
        return embeddings;
    }

    // =========================================================================
    // Novelty detection evaluators
    // =========================================================================

    /**
     * Evaluates F1 for GMM-based novelty detection.
     *
     * <p>A probe is labelled novel when its best-fitting GMM coreset overlap
     * scores below the threshold (i.e. it does not belong to any known domain).
     *
     * @param projected if true the probe is JL-projected before evaluation
     */
    private static F1Result evaluateGmmNovelty(
        List<GmmCoreset> coresets,
        List<float[][]> knownSets,
        float[][] novelEmbeddings,
        NoveltyDetector detector,
        JLProjection jl,
        boolean projected
    ) {
        int tp = 0, fp = 0, fn = 0, tn = 0;

        // Novel probes — label=1 (truly novel)
        for (float[] probe : novelEmbeddings) {
            float[] vec = projected ? jl.project(probe) : probe;
            GmmCoreset probCoreset = GmmCoreset.fit(new float[][] { vec }, 1, new Random(0L));
            boolean predictedNovel = detector.isNovel(probCoreset, new ArrayList<>(coresets));
            if (predictedNovel) tp++;
            else fn++;
        }

        // Known-domain probes — label=0 (not novel)
        for (float[][] knownSet : knownSets) {
            for (float[] probe : knownSet) {
                float[] vec = projected ? jl.project(probe) : probe;
                GmmCoreset probCoreset = GmmCoreset.fit(new float[][] { vec }, 1, new Random(0L));
                boolean predictedNovel = detector.isNovel(probCoreset, new ArrayList<>(coresets));
                if (predictedNovel) fp++;
                else tn++;
            }
        }

        return F1Result.compute(tp, fp, fn, tn);
    }

    /** Evaluates F1 for k-center-based novelty detection. */
    private static F1Result evaluateKcNovelty(
        List<KCenterCoreset> coresets,
        List<float[][]> knownSets,
        float[][] novelEmbeddings,
        NoveltyDetector detector
    ) {
        int tp = 0, fp = 0, fn = 0, tn = 0;

        for (float[] probe : novelEmbeddings) {
            KCenterCoreset probCoreset = KCenterCoreset.fit(new float[][] { probe }, 1, new Random(0L));
            boolean predictedNovel = detector.isNovel(probCoreset, new ArrayList<>(coresets));
            if (predictedNovel) tp++;
            else fn++;
        }

        for (float[][] knownSet : knownSets) {
            for (float[] probe : knownSet) {
                KCenterCoreset probCoreset = KCenterCoreset.fit(new float[][] { probe }, 1, new Random(0L));
                boolean predictedNovel = detector.isNovel(probCoreset, new ArrayList<>(coresets));
                if (predictedNovel) fp++;
                else tn++;
            }
        }

        return F1Result.compute(tp, fp, fn, tn);
    }

    /**
     * Evaluates F1 for raw-replay novelty detection (CL-LoRA style).
     *
     * <p>A probe is classified as novel when its maximum cosine similarity to
     * any stored replay embedding across all domains is below the threshold.
     * This models what CL-LoRA / EASE-with-replay methods do implicitly via
     * their LoRA routing: if the input is dissimilar to all training data, it
     * is treated as out-of-distribution.
     */
    private static F1Result evaluateReplayNovelty(
        List<float[][]> replayBuffers,
        List<float[][]> knownSets,
        float[][] novelEmbeddings,
        float similarityThreshold
    ) {
        int tp = 0, fp = 0, fn = 0, tn = 0;

        for (float[] probe : novelEmbeddings) {
            boolean predictedNovel = isNovelByReplay(probe, replayBuffers, similarityThreshold);
            if (predictedNovel) tp++;
            else fn++;
        }

        for (float[][] knownSet : knownSets) {
            for (float[] probe : knownSet) {
                boolean predictedNovel = isNovelByReplay(probe, replayBuffers, similarityThreshold);
                if (predictedNovel) fp++;
                else tn++;
            }
        }

        return F1Result.compute(tp, fp, fn, tn);
    }

    /**
     * Returns {@code true} when {@code probe}'s maximum cosine similarity to
     * any stored embedding across all replay buffers is below {@code threshold}.
     */
    private static boolean isNovelByReplay(float[] probe, List<float[][]> replayBuffers, float threshold) {
        float maxSim = -1f;
        for (float[][] buffer : replayBuffers) {
            for (float[] stored : buffer) {
                float sim = cosineSimilarity(probe, stored);
                if (sim > maxSim) {
                    maxSim = sim;
                }
            }
        }
        return maxSim < threshold;
    }

    // =========================================================================
    // Detection latency measurement
    // =========================================================================

    private static long measureDetectionTimeGmm(
        List<GmmCoreset> coresets,
        float[][] probes,
        int k,
        NoveltyDetector detector,
        JLProjection jl,
        boolean projected
    ) {
        int samples = Math.min(100, probes.length);
        long start = System.nanoTime();
        for (int i = 0; i < samples; i++) {
            float[] vec = projected ? jl.project(probes[i]) : probes[i];
            GmmCoreset probe = GmmCoreset.fit(new float[][] { vec }, 1, new Random(0L));
            detector.computeNovelty(probe, new ArrayList<>(coresets));
        }
        return (System.nanoTime() - start) / samples / 1_000; // microseconds
    }

    private static long measureDetectionTimeKc(List<KCenterCoreset> coresets, float[][] probes, NoveltyDetector detector) {
        int samples = Math.min(100, probes.length);
        long start = System.nanoTime();
        for (int i = 0; i < samples; i++) {
            KCenterCoreset probe = KCenterCoreset.fit(new float[][] { probes[i] }, 1, new Random(0L));
            detector.computeNovelty(probe, new ArrayList<>(coresets));
        }
        return (System.nanoTime() - start) / samples / 1_000;
    }

    private static long measureDetectionTimeReplay(List<float[][]> replayBuffers, float[][] probes, float threshold) {
        int samples = Math.min(100, probes.length);
        long start = System.nanoTime();
        for (int i = 0; i < samples; i++) {
            isNovelByReplay(probes[i], replayBuffers, threshold);
        }
        return (System.nanoTime() - start) / samples / 1_000;
    }

    // =========================================================================
    // Memory accounting
    // =========================================================================

    /** Storage for a single GMM domain summary: K × (1 weight + d mean + d var) floats. */
    private static long gmmMemoryBytes(int k, int dim) {
        return (long) k * (1 + 2 * dim) * Float.BYTES;
    }

    /** Storage for a single k-center domain summary: k × (d centroid + 1 radius) floats. */
    private static long kcMemoryBytes(int k, int dim) {
        return (long) k * (dim + 1) * Float.BYTES;
    }

    /** Storage for a raw replay buffer: n × d floats. */
    private static long replayMemoryBytes(int n, int dim) {
        return (long) n * dim * Float.BYTES;
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    private static float[][] subsample(float[][] data, int n, Random rng) {
        int[] indices = new int[data.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < n && i < data.length; i++) {
            int j = i + rng.nextInt(data.length - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        float[][] result = new float[Math.min(n, data.length)][];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[indices[i]];
        }
        return result;
    }

    private static float cosineSimilarity(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            normA += (double) a[i] * a[i];
            normB += (double) b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom < 1e-12 ? 0f : (float) (dot / denom);
    }

    private static long avgBuildTime(long[] nanos) {
        long sum = 0;
        for (long v : nanos) {
            sum += v;
        }
        return sum / nanos.length / 1_000_000; // ms
    }

    // =========================================================================
    // Output formatting
    // =========================================================================

    @SuppressWarnings("checkstyle:RegexpSinglelineJava")
    private static void printResultsTable(String[] methods, F1Result[] f1Results, long[] memBytes, long[] buildMs, long[] detectUs) {
        System.out.println("\n## ContLoRA vs CL-LoRA Experiment Results");
        System.out.printf("Setup: %d domains × %d train embeddings × d=%d%n", NUM_DOMAINS, TRAIN_PER_DOMAIN, DIM);
        System.out.printf(
            "GMM components K=%d | k-center k=%d | JL d'=%d | Replay size=%d%n%n",
            GMM_COMPONENTS,
            KCENTER_K,
            JL_DIM,
            REPLAY_SIZE
        );

        String headerFmt = "| %-22s | %8s | %8s | %8s | %10s | %10s | %7s |%n";
        String rowFmt = "| %-22s | %8.4f | %8.4f | %8.4f | %10s | %10d | %7d |%n";
        String sep = "|------------------------|----------|----------|----------|------------|------------|---------|";

        System.out.printf(headerFmt, "Method", "F1", "Precision", "Recall", "Memory", "Build(ms)", "Detect(µs)");
        System.out.println(sep);

        for (int i = 0; i < methods.length; i++) {
            F1Result r = f1Results[i];
            String memStr = humanBytes(memBytes[i]);
            if (r == null) {
                System.out.printf("| %-22s | %8s | %8s | %8s | %10s | %10s | %7s |%n", methods[i], "n/a", "n/a", "n/a", memStr, "-", "-");
            } else {
                System.out.printf(rowFmt, methods[i], r.f1(), r.precision(), r.recall(), memStr, buildMs[i], detectUs[i]);
            }
        }
        System.out.println(sep);
        System.out.println();
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format(Locale.ROOT, "%.1f KB", bytes / 1024.0);
        return String.format(Locale.ROOT, "%.1f MB", bytes / (1024.0 * 1024));
    }

    // =========================================================================
    // F1 result record
    // =========================================================================

    /**
     * Holds precision, recall, and F1 for binary novelty detection.
     */
    record F1Result(float precision, float recall, float f1) {

        /** Constructs an F1Result from a confusion matrix. */
        static F1Result compute(int tp, int fp, int fn, int tn) {
            float precision = (tp + fp) > 0 ? (float) tp / (tp + fp) : 0f;
            float recall = (tp + fn) > 0 ? (float) tp / (tp + fn) : 0f;
            float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0f;
            return new F1Result(precision, recall, f1);
        }
    }
}
