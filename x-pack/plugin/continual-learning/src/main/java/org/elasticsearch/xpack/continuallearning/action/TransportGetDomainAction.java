/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.action;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.get.GetRequest;
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
import org.elasticsearch.xpack.continuallearning.domain.ContinualLearningDomain;
import org.elasticsearch.xpack.continuallearning.index.ContinualLearningSystemIndices;
import org.elasticsearch.xpack.core.ClientHelper;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Transport handler for {@link GetDomainAction}.
 *
 * <p>Fetches a single domain by ID or lists all registered domains from the
 * {@code .continual-learning-domains} system index.
 */
public class TransportGetDomainAction extends HandledTransportAction<GetDomainAction.Request, GetDomainAction.Response> {

    private static final Logger logger = LogManager.getLogger(TransportGetDomainAction.class);
    private static final int MAX_DOMAINS = 1000;

    private final Client client;

    @Inject
    public TransportGetDomainAction(TransportService transportService, ActionFilters actionFilters, Client client) {
        super(GetDomainAction.NAME, transportService, actionFilters, GetDomainAction.Request::new, EsExecutors.DIRECT_EXECUTOR_SERVICE);
        this.client = new OriginSettingClient(client, ClientHelper.CONTINUAL_LEARNING_ORIGIN);
    }

    @Override
    protected void doExecute(Task task, GetDomainAction.Request request, ActionListener<GetDomainAction.Response> listener) {
        if (request.isSingleDomainRequest()) {
            fetchSingleDomain(request.getDomainId(), listener);
        } else {
            fetchAllDomains(listener);
        }
    }

    private void fetchSingleDomain(String domainId, ActionListener<GetDomainAction.Response> listener) {
        GetRequest getRequest = new GetRequest(ContinualLearningSystemIndices.DOMAINS_INDEX_ALIAS, domainId);
        client.get(getRequest, listener.delegateFailureAndWrap((delegate, getResponse) -> {
            if (getResponse.isExists() == false) {
                delegate.onFailure(new ResourceNotFoundException("No continual-learning domain with id [{}]", domainId));
                return;
            }
            try {
                ContinualLearningDomain domain = domainFromSource(getResponse.getId(), getResponse.getSourceAsMap());
                delegate.onResponse(new GetDomainAction.Response(List.of(domain)));
            } catch (Exception e) {
                delegate.onFailure(e);
            }
        }));
    }

    private void fetchAllDomains(ActionListener<GetDomainAction.Response> listener) {
        SearchRequest searchRequest = new SearchRequest(ContinualLearningSystemIndices.DOMAINS_INDEX_ALIAS);
        searchRequest.source(
            new SearchSourceBuilder().query(QueryBuilders.matchAllQuery())
                .size(MAX_DOMAINS)
                .fetchSource(true)
                .sort("stage", org.elasticsearch.search.sort.SortOrder.ASC)
        );

        client.search(searchRequest, listener.delegateFailureAndWrap((delegate, searchResponse) -> {
            List<ContinualLearningDomain> domains = new ArrayList<>();
            for (SearchHit hit : searchResponse.getHits().getHits()) {
                try {
                    domains.add(domainFromSource(hit.getId(), hit.getSourceAsMap()));
                } catch (Exception e) {
                    logger.warn("Skipping malformed domain document [{}]: {}", hit.getId(), e.getMessage());
                }
            }
            delegate.onResponse(new GetDomainAction.Response(domains));
        }));
    }

    private static ContinualLearningDomain domainFromSource(String docId, Map<String, Object> source) {
        if (source == null) {
            throw new IllegalStateException("Source is empty for domain document [" + docId + "]");
        }

        String domainId = getString(source, "domain_id", docId);
        String domainName = getString(source, "domain_name", domainId);
        int stage = getInt(source, "stage", 0);
        long createdAt = getLong(source, "created_at", System.currentTimeMillis());
        ContinualLearningDomain.CoresetType coresetType = ContinualLearningDomain.CoresetType.fromString(
            getString(source, "coreset_type", "gmm")
        );
        byte[] coresetBytes = source.containsKey("coreset_bytes")
            ? Base64.getDecoder().decode(source.get("coreset_bytes").toString())
            : new byte[0];
        String loraAdapterId = (String) source.get("lora_adapter_id");
        int loraRank = getInt(source, "lora_rank", 0);
        int intrinsicRank = getInt(source, "intrinsic_rank", 0);
        float noveltyScore = getFloat(source, "novelty_score", 1.0f);
        int embeddingDim = getInt(source, "embedding_dim", 0);
        boolean jlProjected = getBoolean(source, "jl_projected", false);
        int jlOutputDim = getInt(source, "jl_output_dim", 0);

        return new ContinualLearningDomain(
            domainId,
            domainName,
            stage,
            createdAt,
            coresetType,
            coresetBytes,
            loraAdapterId,
            loraRank,
            intrinsicRank,
            noveltyScore,
            embeddingDim,
            jlProjected,
            jlOutputDim
        );
    }

    private static String getString(Map<String, Object> source, String key, String defaultValue) {
        Object v = source.get(key);
        return v instanceof String s ? s : defaultValue;
    }

    private static int getInt(Map<String, Object> source, String key, int defaultValue) {
        Object v = source.get(key);
        if (v instanceof Number n) {
            return n.intValue();
        }
        return defaultValue;
    }

    private static long getLong(Map<String, Object> source, String key, long defaultValue) {
        Object v = source.get(key);
        if (v instanceof Number n) {
            return n.longValue();
        }
        return defaultValue;
    }

    private static float getFloat(Map<String, Object> source, String key, float defaultValue) {
        Object v = source.get(key);
        if (v instanceof Number n) {
            return n.floatValue();
        }
        return defaultValue;
    }

    private static boolean getBoolean(Map<String, Object> source, String key, boolean defaultValue) {
        Object v = source.get(key);
        if (v instanceof Boolean b) {
            return b;
        }
        return defaultValue;
    }
}
