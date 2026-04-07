/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.rest;

import org.elasticsearch.client.internal.node.NodeClient;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.action.RestToXContentListener;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.continuallearning.action.DetectNoveltyAction;
import org.elasticsearch.xpack.continuallearning.domain.ContinualLearningDomain;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.rest.RestRequest.Method.POST;

/**
 * REST handler for computing novelty scores without persisting a new domain.
 *
 * <p>Endpoint: {@code POST /_continual_learning/novelty}
 *
 * <p>Example request body:
 * <pre>{@code
 * {
 *   "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
 *   "coreset_type": "gmm",
 *   "num_components": 50,
 *   "novelty_threshold": 0.5
 * }
 * }</pre>
 */
public class RestDetectNoveltyAction extends BaseRestHandler {

    @Override
    public List<Route> routes() {
        return List.of(new Route(POST, "/_continual_learning/novelty"));
    }

    @Override
    public String getName() {
        return "continual_learning_detect_novelty";
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        DetectNoveltyAction.Request request;
        try (XContentParser parser = restRequest.contentParser()) {
            request = parseRequest(parser);
        }
        return channel -> client.execute(DetectNoveltyAction.INSTANCE, request, new RestToXContentListener<>(channel));
    }

    @SuppressWarnings("unchecked")
    private static DetectNoveltyAction.Request parseRequest(XContentParser parser) throws IOException {
        Map<String, Object> body = parser.map();

        List<float[]> embeddingList = new ArrayList<>();
        Object embeddingsObj = body.get("embeddings");
        if (embeddingsObj instanceof List<?> rows) {
            for (Object row : rows) {
                if (row instanceof List<?> dims) {
                    float[] vec = new float[dims.size()];
                    for (int i = 0; i < dims.size(); i++) {
                        Object v = dims.get(i);
                        vec[i] = v instanceof Number n ? n.floatValue() : 0f;
                    }
                    embeddingList.add(vec);
                }
            }
        }
        float[][] embeddings = embeddingList.toArray(new float[0][]);

        String coresetTypeStr = (String) body.getOrDefault("coreset_type", "gmm");
        ContinualLearningDomain.CoresetType coresetType = ContinualLearningDomain.CoresetType.fromString(coresetTypeStr);

        int numComponents = body.containsKey("num_components") ? ((Number) body.get("num_components")).intValue() : 50;
        float noveltyThreshold = body.containsKey("novelty_threshold")
            ? ((Number) body.get("novelty_threshold")).floatValue()
            : DetectNoveltyAction.DEFAULT_NOVELTY_THRESHOLD;

        return new DetectNoveltyAction.Request(embeddings, coresetType, numComponents, noveltyThreshold);
    }
}
