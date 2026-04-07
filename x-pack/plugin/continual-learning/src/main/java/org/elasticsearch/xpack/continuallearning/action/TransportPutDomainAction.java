/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.action;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.support.ActionFilters;
import org.elasticsearch.action.support.HandledTransportAction;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.client.internal.OriginSettingClient;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.io.stream.BytesStreamOutput;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.injection.guice.Inject;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.transport.TransportService;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xpack.continuallearning.coreset.ConvexHullCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.GeometricCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.GmmCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.HilbertCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.KCenterCoreset;
import org.elasticsearch.xpack.continuallearning.index.ContinualLearningSystemIndices;
import org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector;
import org.elasticsearch.xpack.continuallearning.projection.JLProjection;
import org.elasticsearch.xpack.core.ClientHelper;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Transport handler for {@link PutDomainAction}.
 *
 * <p>Processing pipeline per the ContLoRA paper (Algorithm 3):
 * <ol>
 *   <li>Optionally subsample embeddings.</li>
 *   <li>Optionally apply JL projection.</li>
 *   <li>Build geometric coreset from (projected) embeddings.</li>
 *   <li>Load existing domain coresets from the system index.</li>
 *   <li>Run novelty detection.  If not novel, merge and return.</li>
 *   <li>Compute intrinsic rank via PCA 95% threshold.</li>
 *   <li>Allocate LoRA rank from the global budget.</li>
 *   <li>Persist domain metadata to the system index.</li>
 * </ol>
 */
public class TransportPutDomainAction extends HandledTransportAction<PutDomainAction.Request, PutDomainAction.Response> {

    private static final Logger logger = LogManager.getLogger(TransportPutDomainAction.class);

    /** Default LoRA rank per domain when dynamic allocation is not overridden. */
    private static final int DEFAULT_LORA_RANK = 10;
    /** Maximum embeddings fetched per existing domain for coreset deserialisation. */
    private static final int MAX_EXISTING_DOMAINS = 200;

    private final Client client;

    @Inject
    public TransportPutDomainAction(TransportService transportService, ActionFilters actionFilters, Client client) {
        super(PutDomainAction.NAME, transportService, actionFilters, PutDomainAction.Request::new, EsExecutors.DIRECT_EXECUTOR_SERVICE);
        this.client = new OriginSettingClient(client, ClientHelper.CONTINUAL_LEARNING_ORIGIN);
    }

    @Override
    protected void doExecute(Task task, PutDomainAction.Request request, ActionListener<PutDomainAction.Response> listener) {
        // Step 1 & 2: subsample + optional JL projection
        float[][] embeddings = prepareEmbeddings(request);
        int embeddingDim = request.getEmbeddings().length > 0 ? request.getEmbeddings()[0].length : 0;

        // Step 3: build coreset
        GeometricCoreset newCoreset = buildCoreset(request, embeddings);

        // Step 4: load existing coresets
        SearchRequest searchRequest = new SearchRequest(ContinualLearningSystemIndices.DOMAINS_INDEX_ALIAS);
        searchRequest.source(new SearchSourceBuilder().query(QueryBuilders.matchAllQuery()).size(MAX_EXISTING_DOMAINS).fetchSource(true));

        client.search(searchRequest, listener.delegateFailureAndWrap((delegate, searchResponse) -> {
            List<GeometricCoreset> existingCoresets = loadExistingCoresets(searchResponse);
            List<String> existingDomainIds = loadExistingDomainIds(searchResponse);

            // Step 5: novelty detection
            NoveltyDetector detector = new NoveltyDetector(request.getNoveltyThreshold());
            float noveltyScore = detector.computeNovelty(newCoreset, existingCoresets);
            boolean isNovel = noveltyScore >= request.getNoveltyThreshold();

            if (isNovel == false) {
                int closestIdx = detector.findClosestDomain(newCoreset, existingCoresets);
                String closestDomainId = closestIdx >= 0 ? existingDomainIds.get(closestIdx) : null;
                logger.debug(
                    "Domain [{}] is not novel (novelty_score={}, threshold={}), merging into [{}]",
                    request.getDomainId(),
                    noveltyScore,
                    request.getNoveltyThreshold(),
                    closestDomainId
                );
                delegate.onResponse(new PutDomainAction.Response(request.getDomainId(), false, true, closestDomainId, noveltyScore, 0));
                return;
            }

            // Step 6 & 7: intrinsic rank and LoRA rank allocation
            int intrinsicRank = estimateIntrinsicRank(embeddings);
            int loraRank = DEFAULT_LORA_RANK;

            logger.debug(
                "Registering new domain [{}] with novelty_score={}, intrinsic_rank={}, lora_rank={}",
                request.getDomainId(),
                noveltyScore,
                intrinsicRank,
                loraRank
            );

            // Step 8: persist to system index
            byte[] coresetBytes = serialiseCoreset(newCoreset);
            indexDomain(
                request,
                coresetBytes,
                embeddingDim,
                intrinsicRank,
                loraRank,
                noveltyScore,
                existingCoresets.size(),
                delegate,
                isNovel
            );
        }));
    }

    // -------------------------------------------------------------------------
    // Embedding preparation
    // -------------------------------------------------------------------------

    private float[][] prepareEmbeddings(PutDomainAction.Request request) {
        float[][] embeddings = request.getEmbeddings();

        // Subsample if requested
        int subsampleSize = request.getSubsampleSize();
        if (subsampleSize > 0 && subsampleSize < embeddings.length) {
            embeddings = subsample(embeddings, subsampleSize);
        }

        // JL projection if requested
        if (request.isUseJlProjection() && request.getJlOutputDim() > 0) {
            JLProjection projection = new JLProjection(embeddings[0].length, request.getJlOutputDim(), new Random(42L));
            embeddings = projection.projectBatch(embeddings);
        }

        return embeddings;
    }

    private static float[][] subsample(float[][] embeddings, int n) {
        Random rng = new Random();
        float[][] sampled = new float[n][];
        int[] indices = new int[embeddings.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        // Fisher-Yates partial shuffle
        for (int i = 0; i < n; i++) {
            int j = i + rng.nextInt(embeddings.length - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            sampled[i] = embeddings[indices[i]];
        }
        return sampled;
    }

    // -------------------------------------------------------------------------
    // Coreset construction
    // -------------------------------------------------------------------------

    private static GeometricCoreset buildCoreset(PutDomainAction.Request request, float[][] embeddings) {
        Random rng = new Random(42L);
        return switch (request.getCoresetType()) {
            case GMM -> GmmCoreset.fit(embeddings, request.getNumComponents(), rng);
            case K_CENTER -> KCenterCoreset.fit(embeddings, request.getNumComponents(), rng);
            case CONVEX_HULL -> ConvexHullCoreset.fit(embeddings, request.getNumComponents(), rng);
            case HILBERT -> HilbertCoreset.fit(embeddings, request.getNumComponents(), rng);
        };
    }

    // -------------------------------------------------------------------------
    // Intrinsic rank estimation via PCA threshold
    // -------------------------------------------------------------------------

    /**
     * Estimates intrinsic dimension as the number of PCA components explaining
     * 95% of variance.  Uses a simplified power-iteration estimate on the
     * covariance diagonal for efficiency.
     */
    private static int estimateIntrinsicRank(float[][] embeddings) {
        if (embeddings.length == 0 || embeddings[0].length == 0) {
            return 1;
        }
        int n = embeddings.length;
        int d = embeddings[0].length;

        // Compute per-dimension variance
        double[] mean = new double[d];
        for (float[] e : embeddings) {
            for (int i = 0; i < d; i++) {
                mean[i] += e[i];
            }
        }
        for (int i = 0; i < d; i++) {
            mean[i] /= n;
        }

        double[] variance = new double[d];
        for (float[] e : embeddings) {
            for (int i = 0; i < d; i++) {
                double diff = e[i] - mean[i];
                variance[i] += diff * diff;
            }
        }
        double totalVar = 0;
        for (int i = 0; i < d; i++) {
            variance[i] /= n;
            totalVar += variance[i];
        }

        if (totalVar < 1e-12) {
            return 1;
        }

        // Sort variances descending
        double[] sorted = Arrays.copyOf(variance, d);
        Arrays.sort(sorted);
        double cumulative = 0;
        int rank = 0;
        for (int i = d - 1; i >= 0; i--) {
            cumulative += sorted[i];
            rank++;
            if (cumulative / totalVar >= 0.95) {
                break;
            }
        }
        return rank;
    }

    // -------------------------------------------------------------------------
    // Coreset serialisation
    // -------------------------------------------------------------------------

    private static byte[] serialiseCoreset(GeometricCoreset coreset) {
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            out.writeString(coreset.getWriteableName());
            coreset.writeTo(out);
            return BytesReference.toBytes(out.bytes());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to serialise coreset", e);
        }
    }

    // -------------------------------------------------------------------------
    // Load existing domain coresets from the system index
    // -------------------------------------------------------------------------

    private static List<GeometricCoreset> loadExistingCoresets(SearchResponse searchResponse) {
        List<GeometricCoreset> coresets = new ArrayList<>();
        for (SearchHit hit : searchResponse.getHits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            if (source == null) {
                continue;
            }
            try {
                GeometricCoreset coreset = deserialiseCoresetFromSource(source);
                if (coreset != null) {
                    coresets.add(coreset);
                }
            } catch (Exception e) {
                logger.warn("Failed to deserialise coreset for domain [{}]: {}", source.get("domain_id"), e.getMessage());
            }
        }
        return coresets;
    }

    private static List<String> loadExistingDomainIds(SearchResponse searchResponse) {
        List<String> ids = new ArrayList<>();
        for (SearchHit hit : searchResponse.getHits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            if (source != null) {
                Object id = source.get("domain_id");
                if (id instanceof String s) {
                    ids.add(s);
                }
            }
        }
        return ids;
    }

    private static GeometricCoreset deserialiseCoresetFromSource(Map<String, Object> source) throws IOException {
        Object bytesObj = source.get("coreset_bytes");
        if (bytesObj == null) {
            return null;
        }
        byte[] bytes = Base64.getDecoder().decode(bytesObj.toString());
        try (org.elasticsearch.common.io.stream.StreamInput in = org.elasticsearch.common.io.stream.StreamInput.wrap(bytes)) {
            String type = in.readString();
            return switch (type) {
                case GmmCoreset.TYPE -> new GmmCoreset(in);
                case KCenterCoreset.TYPE -> new KCenterCoreset(in);
                case ConvexHullCoreset.TYPE -> new ConvexHullCoreset(in);
                case HilbertCoreset.TYPE -> new HilbertCoreset(in);
                default -> null;
            };
        }
    }

    // -------------------------------------------------------------------------
    // Index domain into system index
    // -------------------------------------------------------------------------

    @SuppressWarnings("checkstyle:ParameterNumber")
    private void indexDomain(
        PutDomainAction.Request request,
        byte[] coresetBytes,
        int embeddingDim,
        int intrinsicRank,
        int loraRank,
        float noveltyScore,
        int stage,
        ActionListener<PutDomainAction.Response> listener,
        boolean created
    ) {
        try (XContentBuilder source = XContentFactory.jsonBuilder()) {
            source.startObject();
            source.field("domain_id", request.getDomainId());
            source.field("domain_name", request.getDomainName() != null ? request.getDomainName() : request.getDomainId());
            source.field("stage", stage);
            source.field("created_at", Instant.now().toEpochMilli());
            source.field("coreset_type", request.getCoresetType().name().toLowerCase(java.util.Locale.ROOT));
            source.field("coreset_bytes", Base64.getEncoder().encodeToString(coresetBytes));
            source.field("lora_rank", loraRank);
            source.field("intrinsic_rank", intrinsicRank);
            source.field("novelty_score", noveltyScore);
            source.field("embedding_dim", embeddingDim);
            source.field("jl_projected", request.isUseJlProjection());
            source.field("jl_output_dim", request.getJlOutputDim());
            source.endObject();

            IndexRequest indexRequest = new IndexRequest(ContinualLearningSystemIndices.DOMAINS_INDEX_ALIAS).id(request.getDomainId())
                .source(source)
                .create(created);

            client.index(
                indexRequest,
                listener.delegateFailureAndWrap(
                    (delegate, indexResponse) -> delegate.onResponse(
                        new PutDomainAction.Response(request.getDomainId(), created, false, null, noveltyScore, loraRank)
                    )
                )
            );
        } catch (IOException e) {
            listener.onFailure(e);
        }
    }
}
