/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.querysampling;

import org.elasticsearch.common.settings.Setting;

import java.util.List;

/** Settings of the query sampling pipeline; the sampling knobs are dynamic so it can be tuned without a restart. */
public final class QuerySamplingSettings {

    /** When false the pipeline does no work on the search path. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "xpack.query_sampling.enabled",
        false,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * Probability that an eligible kNN search is captured; each search is an independent coin flip. A node with little traffic
     * captures more, see {@link #MIN_CAPTURES_PER_HOUR}.
     */
    public static final Setting<Double> CAPTURE_RATE = Setting.doubleSetting(
        "xpack.query_sampling.capture_rate",
        0.01,
        0.0,
        1.0,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * Per-node floor of captures per hour, raising the rate above {@code capture_rate} for low-traffic nodes, so that a quiet
     * node or hour still gives samples. 0 means no floor. With a floor, a {@code capture_rate} of 0 does not stop captures:
     * switch sampling off with {@code xpack.query_sampling.enabled} for that.
     */
    public static final Setting<Long> MIN_CAPTURES_PER_HOUR = Setting.longSetting(
        "xpack.query_sampling.min_captures_per_hour",
        100L,
        0L,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * Most queries a node picks per hour, apart from the hottest ones; once it is reached the chance of a pick is cut to a
     * hundredth, not to 0, so a few more may pass. 0 means no limit.
     */
    public static final Setting<Long> MAX_PICKS_PER_HOUR = Setting.longSetting(
        "xpack.query_sampling.max_picks_per_hour",
        0L,
        0L,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * Picks per hour (apart from the hottest queries) that the acceptance scale is steered towards; unlike the
     * max it is only a target. 0 leaves the scale as set.
     */
    public static final Setting<Long> TARGET_PICKS_PER_HOUR = Setting.longSetting(
        "xpack.query_sampling.target_picks_per_hour",
        0L,
        0L,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * Number of clusters of query vectors per field used to balance the sample, see {@link #SPATIAL_BALANCE}. 0
     * disables. The default, 16, is what the offline replay of the sampler used. Built per node, in memory: before the
     * clusters are fitted, ten vectors per cluster wait for it, for each field, which is why the number is limited.
     */
    public static final Setting<Integer> SPATIAL_CLUSTERS = Setting.intSetting(
        "xpack.query_sampling.spatial_clusters",
        16,
        0,
        64,
        Setting.Property.NodeScope
    );

    /**
     * 0 follows the traffic, 1 (the default) picks as many queries from each cluster. No effect without {@link
     * #SPATIAL_CLUSTERS}.
     */
    public static final Setting<Double> SPATIAL_BALANCE = Setting.doubleSetting(
        "xpack.query_sampling.spatial_balance",
        1.0,
        0.0,
        1.0,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    private QuerySamplingSettings() {}

    public static List<Setting<?>> getSettings() {
        return List.of(
            ENABLED,
            CAPTURE_RATE,
            MIN_CAPTURES_PER_HOUR,
            MAX_PICKS_PER_HOUR,
            TARGET_PICKS_PER_HOUR,
            SPATIAL_CLUSTERS,
            SPATIAL_BALANCE
        );
    }
}
