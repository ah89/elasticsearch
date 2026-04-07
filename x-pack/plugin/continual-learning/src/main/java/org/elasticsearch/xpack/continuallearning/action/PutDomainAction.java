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
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.continuallearning.domain.ContinualLearningDomain;

import java.io.IOException;
import java.util.Objects;

/**
 * Action for registering a new continual-learning domain.
 *
 * <p>The request accepts a set of embedding vectors (or, optionally, pre-built
 * coreset parameters) for the domain.  The transport handler builds a geometric
 * coreset, performs novelty detection against all existing domains, and — if the
 * domain is novel — persists the metadata to the system index and signals the
 * caller to train a new LoRA adapter.
 *
 * <p>REST endpoint: {@code PUT /_continual_learning/domains/{domain_id}}
 */
public class PutDomainAction extends ActionType<PutDomainAction.Response> {

    public static final PutDomainAction INSTANCE = new PutDomainAction();
    public static final String NAME = "cluster:admin/continual_learning/domain/put";

    private PutDomainAction() {
        super(NAME);
    }

    // -------------------------------------------------------------------------
    // Request
    // -------------------------------------------------------------------------

    public static class Request extends org.elasticsearch.action.ActionRequest {

        private final String domainId;
        private final String domainName;
        /** Embedding vectors for coreset construction, shape [n][d]. */
        private final float[][] embeddings;
        private final ContinualLearningDomain.CoresetType coresetType;
        private final int numComponents;
        private final boolean useJlProjection;
        private final int jlOutputDim;
        private final int subsampleSize;
        private final float noveltyThreshold;

        public Request(
            String domainId,
            String domainName,
            float[][] embeddings,
            ContinualLearningDomain.CoresetType coresetType,
            int numComponents,
            boolean useJlProjection,
            int jlOutputDim,
            int subsampleSize,
            float noveltyThreshold
        ) {
            this.domainId = domainId;
            this.domainName = domainName;
            this.embeddings = embeddings;
            this.coresetType = coresetType;
            this.numComponents = numComponents;
            this.useJlProjection = useJlProjection;
            this.jlOutputDim = jlOutputDim;
            this.subsampleSize = subsampleSize;
            this.noveltyThreshold = noveltyThreshold;
        }

        public Request(StreamInput in) throws IOException {
            super(in);
            this.domainId = in.readString();
            this.domainName = in.readString();
            int n = in.readVInt();
            int d = n > 0 ? in.readVInt() : 0;
            this.embeddings = new float[n][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    this.embeddings[i][j] = in.readFloat();
                }
            }
            this.coresetType = ContinualLearningDomain.CoresetType.fromString(in.readString());
            this.numComponents = in.readVInt();
            this.useJlProjection = in.readBoolean();
            this.jlOutputDim = in.readVInt();
            this.subsampleSize = in.readVInt();
            this.noveltyThreshold = in.readFloat();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeString(domainId);
            out.writeString(domainName);
            out.writeVInt(embeddings.length);
            if (embeddings.length > 0) {
                out.writeVInt(embeddings[0].length);
                for (float[] row : embeddings) {
                    for (float v : row) {
                        out.writeFloat(v);
                    }
                }
            }
            out.writeString(coresetType.name());
            out.writeVInt(numComponents);
            out.writeBoolean(useJlProjection);
            out.writeVInt(jlOutputDim);
            out.writeVInt(subsampleSize);
            out.writeFloat(noveltyThreshold);
        }

        @Override
        public ActionRequestValidationException validate() {
            ActionRequestValidationException e = null;
            if (domainId == null || domainId.isBlank()) {
                e = addValidationError("domain_id is required", e);
            }
            if (embeddings == null || embeddings.length == 0) {
                e = addValidationError("embeddings must be non-empty", e);
            }
            if (numComponents < 1) {
                e = addValidationError("num_components must be >= 1", e);
            }
            if (useJlProjection && jlOutputDim < 1) {
                e = addValidationError("jl_output_dim must be >= 1 when use_jl_projection is true", e);
            }
            return e;
        }

        public String getDomainId() {
            return domainId;
        }

        public String getDomainName() {
            return domainName;
        }

        public float[][] getEmbeddings() {
            return embeddings;
        }

        public ContinualLearningDomain.CoresetType getCoresetType() {
            return coresetType;
        }

        public int getNumComponents() {
            return numComponents;
        }

        public boolean isUseJlProjection() {
            return useJlProjection;
        }

        public int getJlOutputDim() {
            return jlOutputDim;
        }

        public int getSubsampleSize() {
            return subsampleSize;
        }

        public float getNoveltyThreshold() {
            return noveltyThreshold;
        }
    }

    // -------------------------------------------------------------------------
    // Response
    // -------------------------------------------------------------------------

    public static class Response extends ActionResponse implements ToXContentObject {

        private final String domainId;
        private final boolean created;
        private final boolean merged;
        /** ID of the closest existing domain when merging. */
        private final String mergedIntoDomainId;
        private final float noveltyScore;
        private final int loraRank;

        public Response(String domainId, boolean created, boolean merged, String mergedIntoDomainId, float noveltyScore, int loraRank) {
            this.domainId = domainId;
            this.created = created;
            this.merged = merged;
            this.mergedIntoDomainId = mergedIntoDomainId;
            this.noveltyScore = noveltyScore;
            this.loraRank = loraRank;
        }

        public Response(StreamInput in) throws IOException {
            this.domainId = in.readString();
            this.created = in.readBoolean();
            this.merged = in.readBoolean();
            this.mergedIntoDomainId = in.readOptionalString();
            this.noveltyScore = in.readFloat();
            this.loraRank = in.readVInt();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeString(domainId);
            out.writeBoolean(created);
            out.writeBoolean(merged);
            out.writeOptionalString(mergedIntoDomainId);
            out.writeFloat(noveltyScore);
            out.writeVInt(loraRank);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject();
            builder.field("domain_id", domainId);
            builder.field("created", created);
            builder.field("merged", merged);
            if (mergedIntoDomainId != null) {
                builder.field("merged_into_domain_id", mergedIntoDomainId);
            }
            builder.field("novelty_score", noveltyScore);
            builder.field("lora_rank", loraRank);
            builder.endObject();
            return builder;
        }

        public RestStatus status() {
            return created ? RestStatus.CREATED : RestStatus.OK;
        }

        public String getDomainId() {
            return domainId;
        }

        public boolean isCreated() {
            return created;
        }

        public boolean isMerged() {
            return merged;
        }

        public String getMergedIntoDomainId() {
            return mergedIntoDomainId;
        }

        public float getNoveltyScore() {
            return noveltyScore;
        }

        public int getLoraRank() {
            return loraRank;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof Response other) {
                return created == other.created
                    && merged == other.merged
                    && Float.compare(noveltyScore, other.noveltyScore) == 0
                    && loraRank == other.loraRank
                    && Objects.equals(domainId, other.domainId)
                    && Objects.equals(mergedIntoDomainId, other.mergedIntoDomainId);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(domainId, created, merged, mergedIntoDomainId, noveltyScore, loraRank);
        }
    }
}
