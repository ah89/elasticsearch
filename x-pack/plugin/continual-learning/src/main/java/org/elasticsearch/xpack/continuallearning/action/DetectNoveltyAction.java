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
import org.elasticsearch.xpack.continuallearning.novelty.NoveltyDetector;

import java.io.IOException;
import java.util.Objects;

/**
 * Action for computing the novelty score of a set of embeddings against all
 * registered continual-learning domains, without persisting a new domain.
 *
 * <p>This is useful for pre-screening incoming data batches before committing to
 * a full {@link PutDomainAction} call, or for monitoring domain drift in production.
 *
 * <p>REST endpoint: {@code POST /_continual_learning/novelty}
 */
public class DetectNoveltyAction extends ActionType<DetectNoveltyAction.Response> {

    public static final DetectNoveltyAction INSTANCE = new DetectNoveltyAction();
    public static final String NAME = "cluster:admin/continual_learning/novelty/detect";

    private DetectNoveltyAction() {
        super(NAME);
    }

    // -------------------------------------------------------------------------
    // Request
    // -------------------------------------------------------------------------

    public static class Request extends org.elasticsearch.action.ActionRequest {

        private final float[][] embeddings;
        private final ContinualLearningDomain.CoresetType coresetType;
        private final int numComponents;
        private final float noveltyThreshold;

        public Request(float[][] embeddings, ContinualLearningDomain.CoresetType coresetType, int numComponents, float noveltyThreshold) {
            this.embeddings = embeddings;
            this.coresetType = coresetType;
            this.numComponents = numComponents;
            this.noveltyThreshold = noveltyThreshold;
        }

        public Request(StreamInput in) throws IOException {
            super(in);
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
            this.noveltyThreshold = in.readFloat();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
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
            out.writeFloat(noveltyThreshold);
        }

        @Override
        public ActionRequestValidationException validate() {
            if (embeddings == null || embeddings.length == 0) {
                return addValidationError("embeddings must be non-empty", null);
            }
            return null;
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

        public float getNoveltyThreshold() {
            return noveltyThreshold;
        }
    }

    // -------------------------------------------------------------------------
    // Response
    // -------------------------------------------------------------------------

    public static class Response extends ActionResponse implements ToXContentObject {

        private final float noveltyScore;
        private final boolean isNovel;
        private final String closestDomainId;
        private final float closestDomainOverlap;

        public Response(float noveltyScore, boolean isNovel, String closestDomainId, float closestDomainOverlap) {
            this.noveltyScore = noveltyScore;
            this.isNovel = isNovel;
            this.closestDomainId = closestDomainId;
            this.closestDomainOverlap = closestDomainOverlap;
        }

        public Response(StreamInput in) throws IOException {
            this.noveltyScore = in.readFloat();
            this.isNovel = in.readBoolean();
            this.closestDomainId = in.readOptionalString();
            this.closestDomainOverlap = in.readFloat();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeFloat(noveltyScore);
            out.writeBoolean(isNovel);
            out.writeOptionalString(closestDomainId);
            out.writeFloat(closestDomainOverlap);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject();
            builder.field("novelty_score", noveltyScore);
            builder.field("is_novel", isNovel);
            if (closestDomainId != null) {
                builder.field("closest_domain_id", closestDomainId);
                builder.field("closest_domain_overlap", closestDomainOverlap);
            }
            builder.endObject();
            return builder;
        }

        public float getNoveltyScore() {
            return noveltyScore;
        }

        public boolean isNovel() {
            return isNovel;
        }

        public String getClosestDomainId() {
            return closestDomainId;
        }

        public float getClosestDomainOverlap() {
            return closestDomainOverlap;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof Response other) {
                return isNovel == other.isNovel
                    && Float.compare(noveltyScore, other.noveltyScore) == 0
                    && Float.compare(closestDomainOverlap, other.closestDomainOverlap) == 0
                    && Objects.equals(closestDomainId, other.closestDomainId);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(noveltyScore, isNovel, closestDomainId, closestDomainOverlap);
        }
    }

    /** Default novelty threshold forwarded from {@link NoveltyDetector}. */
    public static final float DEFAULT_NOVELTY_THRESHOLD = NoveltyDetector.DEFAULT_THRESHOLD;
}
