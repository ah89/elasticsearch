/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning;

import org.elasticsearch.cluster.node.DiscoveryNodes;
import org.elasticsearch.features.NodeFeature;
import org.elasticsearch.indices.SystemIndexDescriptor;
import org.elasticsearch.plugins.ActionPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SystemIndexPlugin;
import org.elasticsearch.rest.RestHandler;
import org.elasticsearch.xpack.continuallearning.action.DetectNoveltyAction;
import org.elasticsearch.xpack.continuallearning.action.GetDomainAction;
import org.elasticsearch.xpack.continuallearning.action.PutDomainAction;
import org.elasticsearch.xpack.continuallearning.action.TransportDetectNoveltyAction;
import org.elasticsearch.xpack.continuallearning.action.TransportGetDomainAction;
import org.elasticsearch.xpack.continuallearning.action.TransportPutDomainAction;
import org.elasticsearch.xpack.continuallearning.index.ContinualLearningSystemIndices;
import org.elasticsearch.xpack.continuallearning.rest.RestDetectNoveltyAction;
import org.elasticsearch.xpack.continuallearning.rest.RestGetDomainAction;
import org.elasticsearch.xpack.continuallearning.rest.RestPutDomainAction;

import java.util.Collection;
import java.util.List;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Elasticsearch x-pack plugin implementing ContLoRA: continual learning via
 * geometric coresets and Low-Rank Adaptation.
 *
 * <p>This plugin exposes three REST endpoints:
 * <ul>
 *   <li>{@code PUT  /_continual_learning/domains/{domain_id}} — register a new
 *       domain, build a geometric coreset, run novelty detection, and persist
 *       domain metadata to the system index.</li>
 *   <li>{@code GET  /_continual_learning/domains[/{domain_id}]} — retrieve
 *       registered domain metadata (single or list).</li>
 *   <li>{@code POST /_continual_learning/novelty} — compute a novelty score for
 *       a batch of embeddings without persisting a new domain.</li>
 * </ul>
 *
 * <p>Domain metadata (including serialised geometric coresets and LoRA adapter
 * references) is stored in the managed system index
 * {@value ContinualLearningSystemIndices#DOMAINS_INDEX_ALIAS}.
 */
public class ContinualLearningPlugin extends Plugin implements ActionPlugin, SystemIndexPlugin {

    @Override
    public List<ActionHandler> getActions() {
        return List.of(
            new ActionHandler(PutDomainAction.INSTANCE, TransportPutDomainAction.class),
            new ActionHandler(GetDomainAction.INSTANCE, TransportGetDomainAction.class),
            new ActionHandler(DetectNoveltyAction.INSTANCE, TransportDetectNoveltyAction.class)
        );
    }

    @Override
    public List<RestHandler> getRestHandlers(
        RestHandlersServices restHandlersServices,
        Supplier<DiscoveryNodes> nodesInCluster,
        Predicate<NodeFeature> clusterSupportsFeature
    ) {
        return List.of(new RestPutDomainAction(), new RestGetDomainAction(), new RestDetectNoveltyAction());
    }

    @Override
    public Collection<SystemIndexDescriptor> getSystemIndexDescriptors(org.elasticsearch.common.settings.Settings settings) {
        return ContinualLearningSystemIndices.getSystemIndexDescriptors();
    }

    @Override
    public String getFeatureName() {
        return "continual_learning";
    }

    @Override
    public String getFeatureDescription() {
        return "Continual learning via geometric coresets and Low-Rank Adaptation (ContLoRA)";
    }
}
