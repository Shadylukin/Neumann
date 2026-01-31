// SPDX-License-Identifier: MIT OR Apache-2.0
//! OpenTelemetry metrics integration.
//!
//! This module provides metrics collection using OpenTelemetry with OTLP export.

use std::sync::Arc;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Histogram, Meter, MeterProvider};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::runtime::Tokio;

use crate::error::{Result, ServerError};

/// Configuration for metrics collection.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled.
    pub enabled: bool,
    /// OTLP endpoint URL for exporting metrics.
    pub otlp_endpoint: String,
    /// Service name for metrics identification.
    pub service_name: String,
    /// Export interval in seconds.
    pub export_interval_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            otlp_endpoint: "http://localhost:4317".to_string(),
            service_name: "neumann_server".to_string(),
            export_interval_secs: 60,
        }
    }
}

impl MetricsConfig {
    /// Create a new metrics configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable metrics collection.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the OTLP endpoint URL.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.otlp_endpoint = endpoint;
        self
    }

    /// Set the service name.
    #[must_use]
    pub fn with_service_name(mut self, name: String) -> Self {
        self.service_name = name;
        self
    }

    /// Set the export interval in seconds.
    #[must_use]
    pub fn with_export_interval(mut self, secs: u64) -> Self {
        self.export_interval_secs = secs;
        self
    }
}

/// Server metrics collection.
pub struct ServerMetrics {
    meter: Meter,
    /// Total number of requests received.
    pub requests_total: Counter<u64>,
    /// Number of successful requests.
    pub requests_success: Counter<u64>,
    /// Number of failed requests.
    pub requests_error: Counter<u64>,
    /// Number of authentication failures.
    pub auth_failures: Counter<u64>,
    /// Number of rate-limited requests.
    pub rate_limited: Counter<u64>,
    /// Query latency histogram in milliseconds.
    pub query_latency: Histogram<f64>,
    /// Blob operation latency histogram in milliseconds.
    pub blob_latency: Histogram<f64>,
    /// Vector operation latency histogram in milliseconds.
    pub vector_latency: Histogram<f64>,
}

impl ServerMetrics {
    /// Create a new metrics instance.
    #[must_use]
    pub fn new(meter: Meter) -> Self {
        let requests_total = meter
            .u64_counter("neumann.requests.total")
            .with_description("Total number of requests received")
            .init();

        let requests_success = meter
            .u64_counter("neumann.requests.success")
            .with_description("Number of successful requests")
            .init();

        let requests_error = meter
            .u64_counter("neumann.requests.error")
            .with_description("Number of failed requests")
            .init();

        let auth_failures = meter
            .u64_counter("neumann.auth.failures")
            .with_description("Number of authentication failures")
            .init();

        let rate_limited = meter
            .u64_counter("neumann.rate_limited")
            .with_description("Number of rate-limited requests")
            .init();

        let query_latency = meter
            .f64_histogram("neumann.query.latency_ms")
            .with_description("Query execution latency in milliseconds")
            .init();

        let blob_latency = meter
            .f64_histogram("neumann.blob.latency_ms")
            .with_description("Blob operation latency in milliseconds")
            .init();

        let vector_latency = meter
            .f64_histogram("neumann.vector.latency_ms")
            .with_description("Vector operation latency in milliseconds")
            .init();

        Self {
            meter,
            requests_total,
            requests_success,
            requests_error,
            auth_failures,
            rate_limited,
            query_latency,
            blob_latency,
            vector_latency,
        }
    }

    /// Record a request with its outcome.
    pub fn record_request(&self, service: &str, method: &str, success: bool, latency_ms: f64) {
        let attrs = [
            KeyValue::new("service", service.to_string()),
            KeyValue::new("method", method.to_string()),
        ];

        self.requests_total.add(1, &attrs);

        if success {
            self.requests_success.add(1, &attrs);
        } else {
            self.requests_error.add(1, &attrs);
        }

        // Also record the latency based on service type
        if service == "query" {
            self.query_latency.record(latency_ms, &attrs);
        } else if service == "blob" {
            self.blob_latency.record(latency_ms, &attrs);
        } else if service == "vector" {
            self.vector_latency.record(latency_ms, &attrs);
        }
    }

    /// Record query latency.
    pub fn record_query_latency(&self, method: &str, latency_ms: f64) {
        let attrs = [KeyValue::new("method", method.to_string())];
        self.query_latency.record(latency_ms, &attrs);
    }

    /// Record blob operation latency.
    pub fn record_blob_latency(&self, operation: &str, latency_ms: f64) {
        let attrs = [KeyValue::new("operation", operation.to_string())];
        self.blob_latency.record(latency_ms, &attrs);
    }

    /// Record vector operation latency.
    pub fn record_vector_latency(&self, operation: &str, latency_ms: f64) {
        let attrs = [KeyValue::new("operation", operation.to_string())];
        self.vector_latency.record(latency_ms, &attrs);
    }

    /// Record an authentication failure.
    pub fn record_auth_failure(&self, reason: &str) {
        let attrs = [KeyValue::new("reason", reason.to_string())];
        self.auth_failures.add(1, &attrs);
    }

    /// Record a rate-limited request.
    pub fn record_rate_limited(&self, identity: &str, operation: &str) {
        let attrs = [
            KeyValue::new("identity", identity.to_string()),
            KeyValue::new("operation", operation.to_string()),
        ];
        self.rate_limited.add(1, &attrs);
    }

    /// Get the underlying meter for custom metrics.
    #[must_use]
    pub fn meter(&self) -> &Meter {
        &self.meter
    }
}

/// Holder for the meter provider, keeps metrics pipeline alive.
pub struct MetricsHandle {
    provider: SdkMeterProvider,
    metrics: Arc<ServerMetrics>,
}

impl MetricsHandle {
    /// Get a reference to the server metrics.
    #[must_use]
    pub fn metrics(&self) -> &Arc<ServerMetrics> {
        &self.metrics
    }

    /// Shutdown the metrics pipeline, flushing any pending data.
    pub fn shutdown(self) -> Result<()> {
        self.provider
            .shutdown()
            .map_err(|e| ServerError::Internal(format!("metrics shutdown failed: {e}")))
    }
}

/// Initialize the metrics pipeline.
///
/// Returns a handle that must be kept alive for the duration of metrics collection.
pub fn init_metrics(config: &MetricsConfig) -> Result<MetricsHandle> {
    if !config.enabled {
        // Return a no-op metrics handle
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter(config.service_name.clone());
        let metrics = Arc::new(ServerMetrics::new(meter));
        return Ok(MetricsHandle { provider, metrics });
    }

    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build_metrics_exporter(
            Box::new(opentelemetry_sdk::metrics::reader::DefaultAggregationSelector::new()),
            Box::new(opentelemetry_sdk::metrics::reader::DefaultTemporalitySelector::new()),
        )
        .map_err(|e| ServerError::Config(format!("failed to create OTLP exporter: {e}")))?;

    let reader = PeriodicReader::builder(exporter, Tokio)
        .with_interval(Duration::from_secs(config.export_interval_secs))
        .build();

    let provider = SdkMeterProvider::builder().with_reader(reader).build();

    let meter = provider.meter(config.service_name.clone());
    let metrics = Arc::new(ServerMetrics::new(meter));

    tracing::info!(
        endpoint = %config.otlp_endpoint,
        service = %config.service_name,
        interval_secs = config.export_interval_secs,
        "Metrics initialized"
    );

    Ok(MetricsHandle { provider, metrics })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.otlp_endpoint, "http://localhost:4317");
        assert_eq!(config.service_name, "neumann_server");
        assert_eq!(config.export_interval_secs, 60);
    }

    #[test]
    fn test_metrics_config_builder() {
        let config = MetricsConfig::new()
            .with_enabled(false)
            .with_endpoint("http://otel:4317".to_string())
            .with_service_name("test_service".to_string())
            .with_export_interval(30);

        assert!(!config.enabled);
        assert_eq!(config.otlp_endpoint, "http://otel:4317");
        assert_eq!(config.service_name, "test_service");
        assert_eq!(config.export_interval_secs, 30);
    }

    #[test]
    fn test_metrics_disabled() {
        let config = MetricsConfig::new().with_enabled(false);
        let handle = init_metrics(&config).expect("should create disabled metrics");

        // Should still be able to record metrics (they just won't be exported)
        handle
            .metrics()
            .record_request("query", "execute", true, 10.0);
        handle.metrics().record_auth_failure("invalid_key");
        handle.metrics().record_rate_limited("user:test", "query");
    }

    #[test]
    fn test_server_metrics_new() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Should not panic
        metrics.record_request("query", "execute", true, 5.0);
        metrics.record_request("query", "execute", false, 10.0);
        metrics.record_query_latency("execute", 15.0);
        metrics.record_blob_latency("upload", 100.0);
        metrics.record_auth_failure("missing_key");
        metrics.record_rate_limited("user:alice", "query");
    }

    #[test]
    fn test_counter_increment() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Record multiple requests
        for _ in 0..5 {
            metrics.record_request("query", "execute", true, 1.0);
        }
        for _ in 0..3 {
            metrics.record_request("query", "execute", false, 1.0);
        }
    }

    #[test]
    fn test_histogram_recording() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Record various latencies
        metrics.record_query_latency("execute", 1.0);
        metrics.record_query_latency("execute", 10.0);
        metrics.record_query_latency("execute", 100.0);
        metrics.record_blob_latency("upload", 50.0);
        metrics.record_blob_latency("download", 25.0);
    }

    #[test]
    fn test_record_vector_latency() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Record various vector operation latencies
        metrics.record_vector_latency("upsert", 5.0);
        metrics.record_vector_latency("query", 10.0);
        metrics.record_vector_latency("delete", 2.0);
    }

    #[test]
    fn test_vector_latency_histogram_attributes() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Test with different operations
        metrics.record_vector_latency("upsert", 1.0);
        metrics.record_vector_latency("query", 2.0);
        metrics.record_vector_latency("delete", 3.0);
        metrics.record_vector_latency("scroll", 4.0);
    }

    #[test]
    fn test_record_request_vector_service() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Test recording vector service request
        metrics.record_request("vector", "upsert", true, 5.0);
        metrics.record_request("vector", "query", true, 10.0);
        metrics.record_request("vector", "query", false, 15.0);
    }

    #[test]
    fn test_meter_accessor() {
        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let metrics = ServerMetrics::new(meter);

        // Should be able to access the meter for custom metrics
        let _custom_counter = metrics.meter().u64_counter("custom.counter").init();
    }
}
