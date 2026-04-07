/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.domain;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.time.Instant;

/**
 * Metadata for a single continual-learning domain stored in the
 * {@code .continual-learning-domains} system index.
 *
 * <p>A domain represents one stage of sequential fine-tuning.  Each domain is
 * associated with:
 * <ul>
 *   <li>A geometric coreset compressing its embedding distribution.</li>
 *   <li>An optional LoRA adapter identifier referencing the corresponding
 *       trained adapter stored in the ML trained-model registry.</li>
 *   <li>Rank metadata used by the global rank-budget constraint.</li>
 * </ul>
 */
public class ContinualLearningDomain implements Writeable, ToXContentObject {

    /** Identifies the coreset type used to summarise this domain's embeddings. */
    public enum CoresetType {
        GMM,
        K_CENTER;

        public static CoresetType fromString(String value) {
            return switch (value.toUpperCase(java.util.Locale.ROOT)) {
                case "GMM" -> GMM;
                case "K_CENTER" -> K_CENTER;
                default -> throw new IllegalArgumentException("Unknown coreset type [" + value + "]");
            };
        }
    }

    private final String domainId;
    private final String domainName;
    private final int stage;
    private final long createdAtMillis;
    private final CoresetType coresetType;
    /** Serialised coreset data stored as a raw byte representation. */
    private final byte[] coresetBytes;
    /** ML trained-model ID of the associated LoRA adapter, may be null. */
    private final String loraAdapterId;
    /** LoRA rank allocated to this domain's adapter. */
    private final int loraRank;
    /** Intrinsic dimension of this domain (PCA 95% explained variance). */
    private final int intrinsicRank;
    /** Novelty score computed when this domain was registered. */
    private final float noveltyScore;
    /** Embedding dimensionality. */
    private final int embeddingDim;
    /** Whether a JL projection was applied before coreset construction. */
    private final boolean jlProjected;
    /** Output dimension of the JL projection (0 if not projected). */
    private final int jlOutputDim;

    @SuppressWarnings("checkstyle:ParameterNumber")
    public ContinualLearningDomain(
        String domainId,
        String domainName,
        int stage,
        long createdAtMillis,
        CoresetType coresetType,
        byte[] coresetBytes,
        String loraAdapterId,
        int loraRank,
        int intrinsicRank,
        float noveltyScore,
        int embeddingDim,
        boolean jlProjected,
        int jlOutputDim
    ) {
        this.domainId = domainId;
        this.domainName = domainName;
        this.stage = stage;
        this.createdAtMillis = createdAtMillis;
        this.coresetType = coresetType;
        this.coresetBytes = coresetBytes;
        this.loraAdapterId = loraAdapterId;
        this.loraRank = loraRank;
        this.intrinsicRank = intrinsicRank;
        this.noveltyScore = noveltyScore;
        this.embeddingDim = embeddingDim;
        this.jlProjected = jlProjected;
        this.jlOutputDim = jlOutputDim;
    }

    public ContinualLearningDomain(StreamInput in) throws IOException {
        this.domainId = in.readString();
        this.domainName = in.readString();
        this.stage = in.readVInt();
        this.createdAtMillis = in.readLong();
        this.coresetType = CoresetType.fromString(in.readString());
        this.coresetBytes = in.readByteArray();
        this.loraAdapterId = in.readOptionalString();
        this.loraRank = in.readVInt();
        this.intrinsicRank = in.readVInt();
        this.noveltyScore = in.readFloat();
        this.embeddingDim = in.readVInt();
        this.jlProjected = in.readBoolean();
        this.jlOutputDim = in.readVInt();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(domainId);
        out.writeString(domainName);
        out.writeVInt(stage);
        out.writeLong(createdAtMillis);
        out.writeString(coresetType.name());
        out.writeByteArray(coresetBytes);
        out.writeOptionalString(loraAdapterId);
        out.writeVInt(loraRank);
        out.writeVInt(intrinsicRank);
        out.writeFloat(noveltyScore);
        out.writeVInt(embeddingDim);
        out.writeBoolean(jlProjected);
        out.writeVInt(jlOutputDim);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("domain_id", domainId);
        builder.field("domain_name", domainName);
        builder.field("stage", stage);
        builder.timeField("created_at", "created_at_string", createdAtMillis);
        builder.field("coreset_type", coresetType.name().toLowerCase(java.util.Locale.ROOT));
        builder.field("lora_rank", loraRank);
        builder.field("intrinsic_rank", intrinsicRank);
        builder.field("novelty_score", noveltyScore);
        builder.field("embedding_dim", embeddingDim);
        builder.field("jl_projected", jlProjected);
        if (jlProjected) {
            builder.field("jl_output_dim", jlOutputDim);
        }
        if (loraAdapterId != null) {
            builder.field("lora_adapter_id", loraAdapterId);
        }
        builder.endObject();
        return builder;
    }

    public String getDomainId() {
        return domainId;
    }

    public String getDomainName() {
        return domainName;
    }

    public int getStage() {
        return stage;
    }

    public Instant getCreatedAt() {
        return Instant.ofEpochMilli(createdAtMillis);
    }

    public CoresetType getCoresetType() {
        return coresetType;
    }

    public byte[] getCoresetBytes() {
        return coresetBytes;
    }

    public String getLoraAdapterId() {
        return loraAdapterId;
    }

    public int getLoraRank() {
        return loraRank;
    }

    public int getIntrinsicRank() {
        return intrinsicRank;
    }

    public float getNoveltyScore() {
        return noveltyScore;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    public boolean isJlProjected() {
        return jlProjected;
    }

    public int getJlOutputDim() {
        return jlOutputDim;
    }
}
