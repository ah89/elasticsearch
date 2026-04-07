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
import org.elasticsearch.xpack.continuallearning.action.PutDomainAction;
import org.elasticsearch.xpack.continuallearning.domain.ContinualLearningDomain;
import org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.rest.RestRequest.Method.PUT;

/**
 * REST handler for registering a new continual-learning domain.
 *
 * <p>Endpoint: {@code PUT /_continual_learning/domains/{domain_id}}
 *
 * <p>Example request body:
 * <pre>{@code
 * {
 *   "domain_name": "product-catalogue-v2",
 *   "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
 *   "coreset_type": "gmm",
 *   "num_components": 50,
 *   "use_jl_projection": true,
 *   "jl_output_dim": 256,
 *   "subsample_size": 10000,
 *   "novelty_threshold": 0.5
 * }
 * }</pre>
 */
public class RestPutDomainAction extends BaseRestHandler {

    @Override
    public List<Route> routes() {
        return List.of(new Route(PUT, "/_continual_learning/domains/{domain_id}"));
    }

    @Override
    public String getName() {
        return "continual_learning_put_domain";
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        String domainId = restRequest.param("domain_id");

        PutDomainAction.Request request;
        try (XContentParser parser = restRequest.contentParser()) {
            request = parseRequest(domainId, parser);
        }

        return channel -> client.execute(PutDomainAction.INSTANCE, request, new RestToXContentListener<>(channel));
    }

    @SuppressWarnings("unchecked")
    private static PutDomainAction.Request parseRequest(String domainId, XContentParser parser) throws IOException {
        Map<String, Object> body = parser.map();

        String domainName = (String) body.getOrDefault("domain_name", domainId);

        // Parse embeddings: array of arrays
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
        boolean useJlProjection = body.containsKey("use_jl_projection") && Boolean.TRUE.equals(body.get("use_jl_projection"));
        int jlOutputDim = body.containsKey("jl_output_dim") ? ((Number) body.get("jl_output_dim")).intValue() : 256;
        int subsampleSize = body.containsKey("subsample_size") ? ((Number) body.get("subsample_size")).intValue() : 0;
        float noveltyThreshold = body.containsKey("novelty_threshold")
            ? ((Number) body.get("novelty_threshold")).floatValue()
            : NoveltyDetector.DEFAULT_THRESHOLD;

        return new PutDomainAction.Request(
            domainId,
            domainName,
            embeddings,
            coresetType,
            numComponents,
            useJlProjection,
            jlOutputDim,
            subsampleSize,
            noveltyThreshold
        );
    }
}
