// SPDX-License-Identifier: MIT OR Apache-2.0
//! REST API error types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

/// API error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error status string.
    pub status: String,
    /// Error code.
    pub code: u16,
    /// Human-readable error message.
    pub message: String,
}

impl ApiError {
    /// Create a new API error.
    #[must_use]
    pub fn new(status: impl Into<String>, code: u16, message: impl Into<String>) -> Self {
        Self {
            status: status.into(),
            code,
            message: message.into(),
        }
    }

    /// Create a bad request error.
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new("bad_request", 400, message)
    }

    /// Create an unauthorized error.
    #[must_use]
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new("unauthorized", 401, message)
    }

    /// Create a not found error.
    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("not_found", 404, message)
    }

    /// Create a conflict error.
    #[must_use]
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new("conflict", 409, message)
    }

    /// Create a rate limited error.
    #[must_use]
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new("rate_limited", 429, message)
    }

    /// Create an internal server error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("internal_error", 500, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.code {
            400 => StatusCode::BAD_REQUEST,
            401 => StatusCode::UNAUTHORIZED,
            404 => StatusCode::NOT_FOUND,
            409 => StatusCode::CONFLICT,
            429 => StatusCode::TOO_MANY_REQUESTS,
            500 => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(self)).into_response()
    }
}

/// Result type for REST API handlers.
pub type ApiResult<T> = Result<Json<T>, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_new() {
        let error = ApiError::new("test", 400, "test message");
        assert_eq!(error.status, "test");
        assert_eq!(error.code, 400);
        assert_eq!(error.message, "test message");
    }

    #[test]
    fn test_api_error_bad_request() {
        let error = ApiError::bad_request("invalid input");
        assert_eq!(error.status, "bad_request");
        assert_eq!(error.code, 400);
    }

    #[test]
    fn test_api_error_unauthorized() {
        let error = ApiError::unauthorized("missing api key");
        assert_eq!(error.status, "unauthorized");
        assert_eq!(error.code, 401);
    }

    #[test]
    fn test_api_error_not_found() {
        let error = ApiError::not_found("collection not found");
        assert_eq!(error.status, "not_found");
        assert_eq!(error.code, 404);
    }

    #[test]
    fn test_api_error_conflict() {
        let error = ApiError::conflict("collection already exists");
        assert_eq!(error.status, "conflict");
        assert_eq!(error.code, 409);
    }

    #[test]
    fn test_api_error_rate_limited() {
        let error = ApiError::rate_limited("too many requests");
        assert_eq!(error.status, "rate_limited");
        assert_eq!(error.code, 429);
    }

    #[test]
    fn test_api_error_internal() {
        let error = ApiError::internal("unexpected error");
        assert_eq!(error.status, "internal_error");
        assert_eq!(error.code, 500);
    }

    #[test]
    fn test_api_error_serialization() {
        let error = ApiError::bad_request("test");
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("bad_request"));
        assert!(json.contains("400"));
        assert!(json.contains("test"));
    }
}
