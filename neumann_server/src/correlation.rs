// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Request correlation and trace ID propagation.
//!
//! This module provides utilities for extracting and propagating trace IDs
//! across gRPC requests to enable distributed tracing.

use tonic::metadata::MetadataValue;
use tonic::{Request, Response};
use tracing::Span;
use uuid::Uuid;

/// Header name for trace ID in requests.
pub const TRACE_ID_HEADER: &str = "x-request-id";

/// Metadata key for trace ID in gRPC metadata.
pub const TRACE_ID_METADATA: &str = "x-request-id";

/// Extract trace ID from request metadata or generate a new one.
pub fn extract_or_generate<T>(request: &Request<T>) -> String {
    request
        .metadata()
        .get(TRACE_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map_or_else(|| Uuid::new_v4().to_string(), ToString::to_string)
}

/// Create a tracing span with trace ID and request metadata.
pub fn request_span(trace_id: &str, service: &str, method: &str) -> Span {
    tracing::info_span!(
        "grpc_request",
        trace_id = %trace_id,
        service = %service,
        method = %method,
    )
}

/// Add trace ID to response metadata.
pub fn add_trace_id_to_response<T>(response: &mut Response<T>, trace_id: &str) {
    if let Ok(value) = trace_id.parse::<MetadataValue<_>>() {
        response.metadata_mut().insert(TRACE_ID_METADATA, value);
    }
}

/// Extract trace ID from request and create a span.
///
/// Returns the trace ID and enters the span. The span will be active
/// for the duration of the returned guard.
pub fn start_request_span<T>(request: &Request<T>, service: &str, method: &str) -> RequestSpan {
    let trace_id = extract_or_generate(request);
    let span = request_span(&trace_id, service, method);
    let guard = span.clone().entered();

    tracing::debug!(
        trace_id = %trace_id,
        service = %service,
        method = %method,
        "Request started"
    );

    RequestSpan {
        trace_id,
        span,
        guard,
    }
}

/// A wrapper holding trace ID and its associated span.
pub struct RequestSpan {
    trace_id: String,
    span: Span,
    /// Guard that keeps the span entered. Dropped when `RequestSpan` is dropped.
    #[allow(dead_code)]
    guard: tracing::span::EnteredSpan,
}

impl RequestSpan {
    /// Get the trace ID.
    #[must_use]
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// Get a reference to the span.
    #[must_use]
    pub const fn span(&self) -> &Span {
        &self.span
    }

    /// Add the trace ID to a response.
    pub fn add_to_response<T>(&self, response: &mut Response<T>) {
        add_trace_id_to_response(response, &self.trace_id);
    }

    /// Create a response with the trace ID in metadata.
    pub fn into_response<T>(self, inner: T) -> Response<T> {
        let mut response = Response::new(inner);
        add_trace_id_to_response(&mut response, &self.trace_id);
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::metadata::MetadataValue;

    #[test]
    fn test_extract_from_header() {
        let mut request = Request::new(());
        let trace_id = "test-trace-id-12345";
        request.metadata_mut().insert(
            TRACE_ID_HEADER,
            MetadataValue::try_from(trace_id).expect("valid value"),
        );

        let extracted = extract_or_generate(&request);
        assert_eq!(extracted, trace_id);
    }

    #[test]
    fn test_generate_when_missing() {
        let request = Request::new(());
        let trace_id = extract_or_generate(&request);

        // Should be a valid UUID
        assert!(Uuid::parse_str(&trace_id).is_ok());
    }

    #[test]
    fn test_generate_when_empty() {
        let mut request = Request::new(());
        request.metadata_mut().insert(
            TRACE_ID_HEADER,
            MetadataValue::try_from("").expect("valid value"),
        );

        let trace_id = extract_or_generate(&request);

        // Should generate a new UUID, not use the empty string
        assert!(Uuid::parse_str(&trace_id).is_ok());
    }

    #[test]
    fn test_span_creation() {
        let span = request_span("trace-123", "QueryService", "execute");

        // The span should record the fields
        span.in_scope(|| {
            tracing::info!("Inside span");
        });
    }

    #[test]
    fn test_response_metadata() {
        let trace_id = "response-trace-id";
        let mut response = Response::new(());

        add_trace_id_to_response(&mut response, trace_id);

        let value = response
            .metadata()
            .get(TRACE_ID_METADATA)
            .expect("should have trace ID");
        assert_eq!(value.to_str().expect("valid str"), trace_id);
    }

    #[test]
    fn test_uuid_format() {
        let request = Request::new(());
        let trace_id = extract_or_generate(&request);

        // Verify it's a valid UUID v4 format
        let uuid = Uuid::parse_str(&trace_id).expect("should be valid UUID");
        assert_eq!(uuid.get_version_num(), 4);
    }

    #[test]
    fn test_request_span_trace_id() {
        let request = Request::new(());
        let req_span = start_request_span(&request, "TestService", "test");

        // Should have a valid UUID as trace ID
        assert!(Uuid::parse_str(req_span.trace_id()).is_ok());
    }

    #[test]
    fn test_request_span_with_existing_trace_id() {
        let mut request = Request::new(());
        let expected_trace_id = "existing-trace-123";
        request.metadata_mut().insert(
            TRACE_ID_HEADER,
            MetadataValue::try_from(expected_trace_id).expect("valid value"),
        );

        let req_span = start_request_span(&request, "TestService", "test");
        assert_eq!(req_span.trace_id(), expected_trace_id);
    }

    #[test]
    fn test_request_span_into_response() {
        let request = Request::new(());
        let req_span = start_request_span(&request, "TestService", "test");
        let trace_id = req_span.trace_id().to_string();

        let response = req_span.into_response("result");

        let value = response
            .metadata()
            .get(TRACE_ID_METADATA)
            .expect("should have trace ID");
        assert_eq!(value.to_str().expect("valid str"), trace_id);
        assert_eq!(*response.get_ref(), "result");
    }

    #[test]
    fn test_request_span_add_to_response() {
        let request = Request::new(());
        let req_span = start_request_span(&request, "TestService", "test");
        let trace_id = req_span.trace_id().to_string();

        let mut response = Response::new(42);
        req_span.add_to_response(&mut response);

        let value = response
            .metadata()
            .get(TRACE_ID_METADATA)
            .expect("should have trace ID");
        assert_eq!(value.to_str().expect("valid str"), trace_id);
    }

    #[test]
    fn test_span_accessor() {
        let request = Request::new(());
        let req_span = start_request_span(&request, "TestService", "test");

        // Should be able to access the span
        req_span.span().in_scope(|| {
            tracing::info!("In request span");
        });
    }
}
