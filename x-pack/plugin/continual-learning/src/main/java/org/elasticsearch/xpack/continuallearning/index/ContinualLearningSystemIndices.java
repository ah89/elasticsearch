/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.index;

import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.indices.SystemIndexDescriptor;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;

/**
 * Defines the system indices used by the continual-learning plugin.
 *
 * <p>{@value #DOMAINS_INDEX_ALIAS} stores per-domain metadata including the
 * serialised geometric coreset, LoRA adapter reference, and rank information.
 * The index is managed internally by Elasticsearch; external clients should
 * use the ContLoRA REST API rather than writing to it directly.
 */
public final class ContinualLearningSystemIndices {

    /** Alias name exposed to external queries. */
    public static final String DOMAINS_INDEX_ALIAS = ".continual-learning-domains";

    /** Primary backing index (version-suffixed for future migrations). */
    static final String DOMAINS_INDEX_CONCRETE = ".continual-learning-domains-1";

    /** Index pattern matched by the system index descriptor. */
    static final String DOMAINS_INDEX_PATTERN = ".continual-learning-domains*";

    /** Current mapping version; increment when the mapping changes. */
    static final int CURRENT_MAPPING_VERSION = 1;

    private ContinualLearningSystemIndices() {}

    /**
     * Returns all {@link SystemIndexDescriptor}s registered by this plugin.
     */
    public static Collection<SystemIndexDescriptor> getSystemIndexDescriptors() {
        return List.of(domainsIndexDescriptor());
    }

    private static SystemIndexDescriptor domainsIndexDescriptor() {
        return SystemIndexDescriptor.builder()
            .setIndexPattern(DOMAINS_INDEX_PATTERN)
            .setPrimaryIndex(DOMAINS_INDEX_CONCRETE)
            .setAliasName(DOMAINS_INDEX_ALIAS)
            .setDescription("Geometric coresets and LoRA adapter metadata for continual learning domains")
            .setMappings(buildMappings())
            .setSettings(buildSettings())
            .setType(SystemIndexDescriptor.Type.INTERNAL_MANAGED)
            .setOrigin(org.elasticsearch.xpack.core.ClientHelper.CONTINUAL_LEARNING_ORIGIN)
            .build();
    }

    private static String buildMappings() {
        try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
            builder.startObject();
            builder.field("dynamic", false);
            builder.startObject("_meta");
            builder.field("version", CURRENT_MAPPING_VERSION);
            builder.endObject();
            builder.startObject("properties");

            addKeywordField(builder, "domain_id");
            addKeywordField(builder, "domain_name");
            builder.startObject("stage").field("type", "integer").endObject();
            builder.startObject("created_at").field("type", "date").field("format", "epoch_millis||strict_date_optional_time").endObject();
            addKeywordField(builder, "coreset_type");
            builder.startObject("coreset_bytes").field("type", "binary").field("doc_values", false).endObject();
            addKeywordField(builder, "lora_adapter_id");
            builder.startObject("lora_rank").field("type", "integer").endObject();
            builder.startObject("intrinsic_rank").field("type", "integer").endObject();
            builder.startObject("novelty_score").field("type", "float").endObject();
            builder.startObject("embedding_dim").field("type", "integer").endObject();
            builder.startObject("jl_projected").field("type", "boolean").endObject();
            builder.startObject("jl_output_dim").field("type", "integer").endObject();

            builder.endObject(); // properties
            builder.endObject();
            return builder.toString();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to build continual-learning domains index mappings", e);
        }
    }

    private static void addKeywordField(XContentBuilder builder, String name) throws IOException {
        builder.startObject(name).field("type", "keyword").endObject();
    }

    private static Settings buildSettings() {
        return Settings.builder()
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .put(IndexMetadata.SETTING_AUTO_EXPAND_REPLICAS, "0-1")
            .build();
    }
}
