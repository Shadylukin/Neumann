// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! OpenTelemetry metrics for `GraphEngine`.
//!
//! This module provides metrics instrumentation for graph operations.
//! Enable the `metrics` feature to use this functionality.

use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter, Unit},
    KeyValue,
};

/// Metrics collector for `GraphEngine` operations.
pub struct GraphMetrics {
    /// Counter for node operations (create, update, delete).
    node_ops: Counter<u64>,
    /// Counter for edge operations (create, update, delete).
    edge_ops: Counter<u64>,
    /// Histogram for query latency in seconds.
    query_latency: Histogram<f64>,
    /// Counter for index lookups.
    index_lookups: Counter<u64>,
    /// Histogram for batch operation sizes.
    batch_size: Histogram<u64>,
    /// Counter for traversal operations.
    traversal_ops: Counter<u64>,
}

impl GraphMetrics {
    /// Creates a new metrics collector using the global meter provider.
    #[must_use]
    pub fn new() -> Self {
        let meter = global::meter("graph_engine");
        Self::with_meter(&meter)
    }

    /// Creates a new metrics collector with a specific meter.
    #[must_use]
    pub fn with_meter(meter: &Meter) -> Self {
        let node_ops = meter
            .u64_counter("graph_engine.node_operations")
            .with_description("Total number of node operations")
            .init();

        let edge_ops = meter
            .u64_counter("graph_engine.edge_operations")
            .with_description("Total number of edge operations")
            .init();

        let query_latency = meter
            .f64_histogram("graph_engine.query_latency")
            .with_description("Query latency in seconds")
            .with_unit(Unit::new("s"))
            .init();

        let index_lookups = meter
            .u64_counter("graph_engine.index_lookups")
            .with_description("Total number of index lookups")
            .init();

        let batch_size = meter
            .u64_histogram("graph_engine.batch_size")
            .with_description("Size of batch operations")
            .init();

        let traversal_ops = meter
            .u64_counter("graph_engine.traversal_operations")
            .with_description("Total number of graph traversal operations")
            .init();

        Self {
            node_ops,
            edge_ops,
            query_latency,
            index_lookups,
            batch_size,
            traversal_ops,
        }
    }

    /// Records a node operation.
    pub fn record_node_op(&self, op: &str) {
        self.node_ops.add(1, &[KeyValue::new("op", op.to_string())]);
    }

    /// Records an edge operation.
    pub fn record_edge_op(&self, op: &str) {
        self.edge_ops.add(1, &[KeyValue::new("op", op.to_string())]);
    }

    /// Records query latency.
    pub fn record_query_latency(&self, latency_secs: f64, query_type: &str) {
        self.query_latency
            .record(latency_secs, &[KeyValue::new("type", query_type.to_string())]);
    }

    /// Records an index lookup.
    pub fn record_index_lookup(&self, index_type: &str) {
        self.index_lookups
            .add(1, &[KeyValue::new("index", index_type.to_string())]);
    }

    /// Records a batch operation size.
    pub fn record_batch_size(&self, size: u64, op: &str) {
        self.batch_size
            .record(size, &[KeyValue::new("op", op.to_string())]);
    }

    /// Records a traversal operation.
    pub fn record_traversal(&self, algorithm: &str) {
        self.traversal_ops
            .add(1, &[KeyValue::new("algorithm", algorithm.to_string())]);
    }
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self::new()
    }
}
