/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.querysampling;

import java.util.function.Supplier;

/**
 * The counters of this node, read together as one {@link QuerySamplingStats}. It is a type of its own so that the stats
 * action can be given it as a component of the plugin; nodes do not coordinate, the stats are combined when they are read.
 */
public record QuerySamplingStatsSource(Supplier<QuerySamplingStats> stats) {}
