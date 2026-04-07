/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.action;

import org.elasticsearch.action.ActionRequestValidationException;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.continuallearning.domain.ContinualLearningDomain;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * Action for retrieving continual-learning domain metadata.
 *
 * <p>Supports fetching a single domain by ID or listing all registered domains.
 *
 * <p>REST endpoints:
 * <ul>
 *   <li>{@code GET /_continual_learning/domains/{domain_id}} — single domain</li>
 *   <li>{@code GET /_continual_learning/domains} — all domains</li>
 * </ul>
 */
public class GetDomainAction extends ActionType<GetDomainAction.Response> {

    public static final GetDomainAction INSTANCE = new GetDomainAction();
    public static final String NAME = "cluster:admin/continual_learning/domain/get";

    private GetDomainAction() {
        super(NAME);
    }

    // -------------------------------------------------------------------------
    // Request
    // -------------------------------------------------------------------------

    public static class Request extends org.elasticsearch.action.ActionRequest {

        /**
         * Domain ID to fetch, or {@code null} to list all domains.
         */
        private final String domainId;

        public Request(String domainId) {
            this.domainId = domainId;
        }

        /** List-all request. */
        public Request() {
            this(null);
        }

        public Request(StreamInput in) throws IOException {
            super(in);
            this.domainId = in.readOptionalString();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeOptionalString(domainId);
        }

        @Override
        public ActionRequestValidationException validate() {
            return null;
        }

        public String getDomainId() {
            return domainId;
        }

        public boolean isSingleDomainRequest() {
            return domainId != null;
        }
    }

    // -------------------------------------------------------------------------
    // Response
    // -------------------------------------------------------------------------

    public static class Response extends ActionResponse implements ToXContentObject {

        private final List<ContinualLearningDomain> domains;

        public Response(List<ContinualLearningDomain> domains) {
            this.domains = domains;
        }

        public Response(StreamInput in) throws IOException {
            this.domains = in.readCollectionAsList(ContinualLearningDomain::new);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeCollection(domains);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject();
            builder.field("count", domains.size());
            builder.startArray("domains");
            for (ContinualLearningDomain domain : domains) {
                domain.toXContent(builder, params);
            }
            builder.endArray();
            builder.endObject();
            return builder;
        }

        public List<ContinualLearningDomain> getDomains() {
            return domains;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof Response other) {
                return Objects.equals(domains, other.domains);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(domains);
        }
    }
}
