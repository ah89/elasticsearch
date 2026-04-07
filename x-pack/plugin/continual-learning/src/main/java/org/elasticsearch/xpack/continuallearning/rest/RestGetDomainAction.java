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
import org.elasticsearch.xpack.continuallearning.action.GetDomainAction;

import java.util.List;

import static org.elasticsearch.rest.RestRequest.Method.GET;

/**
 * REST handler for retrieving continual-learning domain metadata.
 *
 * <p>Endpoints:
 * <ul>
 *   <li>{@code GET /_continual_learning/domains/{domain_id}} — single domain</li>
 *   <li>{@code GET /_continual_learning/domains} — all registered domains</li>
 * </ul>
 */
public class RestGetDomainAction extends BaseRestHandler {

    @Override
    public List<Route> routes() {
        return List.of(new Route(GET, "/_continual_learning/domains/{domain_id}"), new Route(GET, "/_continual_learning/domains"));
    }

    @Override
    public String getName() {
        return "continual_learning_get_domain";
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) {
        String domainId = restRequest.param("domain_id");
        GetDomainAction.Request request = domainId != null ? new GetDomainAction.Request(domainId) : new GetDomainAction.Request();

        return channel -> client.execute(GetDomainAction.INSTANCE, request, new RestToXContentListener<>(channel));
    }
}
