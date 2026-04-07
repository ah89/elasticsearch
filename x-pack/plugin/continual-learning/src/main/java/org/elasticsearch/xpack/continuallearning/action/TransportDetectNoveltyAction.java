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
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.support.ActionFilters;
import org.elasticsearch.action.support.HandledTransportAction;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.client.internal.OriginSettingClient;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.injection.guice.Inject;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.transport.TransportService;
import org.elasticsearch.xpack.continuallearning.coreset.ConvexHullCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.GeometricCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.GmmCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.HilbertCoreset;
import org.elasticsearch.xpack.continuallearning.coreset.KCenterCoreset;
import org.elasticsearch.xpack.continuallearning.index.ContinualLearningSystemIndices;
import org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector;
import org.elasticsearch.xpack.core.ClientHelper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Transport handler for {@link DetectNoveltyAction}.
 *
 * <p>Builds a temporary coreset from the request embeddings, loads all existing
 * domain coresets, and computes the novelty score without persisting anything.
 */
public class TransportDetectNoveltyAction extends HandledTransportAction<DetectNoveltyAction.Request, DetectNoveltyAction.Response> {

    private static final Logger logger = LogManager.getLogger(TransportDetectNoveltyAction.class);
    private static final int MAX_EXISTING_DOMAINS = 200;

    private final Client client;

    @Inject
    public TransportDetectNoveltyAction(TransportService transportService, ActionFilters actionFilters, Client client) {
        super(
            DetectNoveltyAction.NAME,
            transportService,
            actionFilters,
            DetectNoveltyAction.Request::new,
            EsExecutors.DIRECT_EXECUTOR_SERVICE
        );
        this.client = new OriginSettingClient(client, ClientHelper.CONTINUAL_LEARNING_ORIGIN);
    }

    @Override
    protected void doExecute(Task task, DetectNoveltyAction.Request request, ActionListener<DetectNoveltyAction.Response> listener) {
        // Build a temporary coreset for the incoming embeddings
        GeometricCoreset incomingCoreset = buildCoreset(request);

        // Load existing domain coresets
        SearchRequest searchRequest = new SearchRequest(ContinualLearningSystemIndices.DOMAINS_INDEX_ALIAS);
        searchRequest.source(new SearchSourceBuilder().query(QueryBuilders.matchAllQuery()).size(MAX_EXISTING_DOMAINS).fetchSource(true));

        client.search(searchRequest, listener.delegateFailureAndWrap((delegate, searchResponse) -> {
            List<GeometricCoreset> existingCoresets = new ArrayList<>();
            List<String> existingDomainIds = new ArrayList<>();

            for (SearchHit hit : searchResponse.getHits().getHits()) {
                Map<String, Object> source = hit.getSourceAsMap();
                if (source == null) {
                    continue;
                }
                try {
                    GeometricCoreset coreset = deserialiseCoresetFromSource(source);
                    if (coreset != null) {
                        existingCoresets.add(coreset);
                        Object id = source.get("domain_id");
                        existingDomainIds.add(id instanceof String s ? s : hit.getId());
                    }
                } catch (Exception e) {
                    logger.warn("Skipping malformed coreset for domain [{}]: {}", source.get("domain_id"), e.getMessage());
                }
            }

            NoveltyDetector detector = new NoveltyDetector(request.getNoveltyThreshold());
            float noveltyScore = detector.computeNovelty(incomingCoreset, existingCoresets);
            boolean isNovel = noveltyScore >= request.getNoveltyThreshold();

            // Find closest domain for diagnostics
            String closestDomainId = null;
            float closestOverlap = 0f;
            if (existingCoresets.isEmpty() == false) {
                int closestIdx = detector.findClosestDomain(incomingCoreset, existingCoresets);
                if (closestIdx >= 0) {
                    closestDomainId = existingDomainIds.get(closestIdx);
                    closestOverlap = incomingCoreset.computeOverlap(existingCoresets.get(closestIdx));
                }
            }

            logger.debug("Novelty detection: score={}, is_novel={}, closest_domain={}", noveltyScore, isNovel, closestDomainId);

            delegate.onResponse(new DetectNoveltyAction.Response(noveltyScore, isNovel, closestDomainId, closestOverlap));
        }));
    }

    private static GeometricCoreset buildCoreset(DetectNoveltyAction.Request request) {
        Random rng = new Random(42L);
        return switch (request.getCoresetType()) {
            case GMM -> GmmCoreset.fit(request.getEmbeddings(), request.getNumComponents(), rng);
            case K_CENTER -> KCenterCoreset.fit(request.getEmbeddings(), request.getNumComponents(), rng);
            case CONVEX_HULL -> ConvexHullCoreset.fit(request.getEmbeddings(), request.getNumComponents(), rng);
            case HILBERT -> HilbertCoreset.fit(request.getEmbeddings(), request.getNumComponents(), rng);
        };
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
}
