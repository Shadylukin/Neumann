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

impl From<query_router::RouterError> for ApiError {
    fn from(error: query_router::RouterError) -> Self {
        use query_router::RouterError;
        match &error {
            RouterError::AuthenticationRequired => Self::unauthorized(error.to_string()),
            RouterError::NotFound(_) => Self::not_found(error.to_string()),
            // Preserve current 500 "3D spatial index not configured" behavior from
            // neumann_server/src/rest/spatial3d.rs (the configuredness check).
            // Match-arm order matters: this fires before the generic InvalidArgument arm.
            RouterError::InvalidArgument(msg) if msg.to_lowercase().contains("not configured") => {
                Self::internal(error.to_string())
            },
            RouterError::ParseError(_)
            | RouterError::UnknownCommand(_)
            | RouterError::InvalidArgument(_)
            | RouterError::MissingArgument(_)
            | RouterError::TypeMismatch(_) => Self::bad_request(error.to_string()),
            RouterError::CheckpointError(msg) if msg.to_lowercase().contains("cancelled") => {
                Self::bad_request(error.to_string())
            },
            RouterError::VectorError(msg) if msg.to_lowercase().contains("not found") => {
                Self::not_found(error.to_string())
            },
            RouterError::VectorError(msg)
                if msg.to_lowercase().contains("already exists")
                    || msg.to_lowercase().contains("duplicate") =>
            {
                Self::conflict(error.to_string())
            },
            _ => Self::internal(error.to_string()),
        }
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

    #[test]
    fn test_api_error_deserialization() {
        let json = r#"{"status":"test_status","code":422,"message":"test message"}"#;
        let error: ApiError = serde_json::from_str(json).unwrap();
        assert_eq!(error.status, "test_status");
        assert_eq!(error.code, 422);
        assert_eq!(error.message, "test message");
    }

    #[test]
    fn test_api_error_into_response_bad_request() {
        let error = ApiError::bad_request("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_api_error_into_response_unauthorized() {
        let error = ApiError::unauthorized("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_api_error_into_response_not_found() {
        let error = ApiError::not_found("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_api_error_into_response_conflict() {
        let error = ApiError::conflict("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[test]
    fn test_api_error_into_response_rate_limited() {
        let error = ApiError::rate_limited("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_api_error_into_response_internal() {
        let error = ApiError::internal("test");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_api_error_into_response_unknown_code() {
        // Test the default case for unknown status codes
        let error = ApiError::new("unknown", 999, "unknown error");
        let response = error.into_response();
        // Unknown codes should map to INTERNAL_SERVER_ERROR
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_api_error_clone() {
        let error = ApiError::bad_request("test");
        let cloned = error.clone();
        assert_eq!(cloned.status, error.status);
        assert_eq!(cloned.code, error.code);
        assert_eq!(cloned.message, error.message);
    }

    #[test]
    fn test_api_error_debug() {
        let error = ApiError::bad_request("test message");
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ApiError"));
        assert!(debug_str.contains("bad_request"));
        assert!(debug_str.contains("400"));
    }

    // ===== Phase 1.5: From<RouterError> for ApiError =====

    #[test]
    fn from_router_error_authentication_required_is_401() {
        let api: ApiError = query_router::RouterError::AuthenticationRequired.into();
        assert_eq!(api.code, 401);
        assert_eq!(api.status, "unauthorized");
    }

    #[test]
    fn from_router_error_not_found_is_404() {
        let api: ApiError = query_router::RouterError::NotFound("entity 'foo'".to_string()).into();
        assert_eq!(api.code, 404);
        assert_eq!(api.status, "not_found");
    }

    #[test]
    fn from_router_error_parse_is_400() {
        let api: ApiError = query_router::RouterError::ParseError("bad syntax".to_string()).into();
        assert_eq!(api.code, 400);
        assert_eq!(api.status, "bad_request");
    }

    #[test]
    fn from_router_error_invalid_argument_not_configured_is_500() {
        let api: ApiError = query_router::RouterError::InvalidArgument(
            "3D spatial index not configured".to_string(),
        )
        .into();
        assert_eq!(
            api.code, 500,
            "'not configured' must preserve the legacy 500 from spatial3d.rs"
        );
        assert_eq!(api.status, "internal_error");
    }

    #[test]
    fn from_router_error_invalid_argument_general_is_400() {
        let api: ApiError =
            query_router::RouterError::InvalidArgument("missing radius".to_string()).into();
        assert_eq!(api.code, 400);
        assert_eq!(api.status, "bad_request");
    }

    #[test]
    fn from_router_error_vector_not_found_is_404() {
        let api: ApiError =
            query_router::RouterError::VectorError("key 'a' not found".to_string()).into();
        assert_eq!(api.code, 404);
    }

    #[test]
    fn from_router_error_vector_already_exists_is_409() {
        let api: ApiError =
            query_router::RouterError::VectorError("collection already exists".to_string()).into();
        assert_eq!(api.code, 409);
    }

    #[test]
    fn from_router_error_unknown_command_is_400() {
        let api: ApiError =
            query_router::RouterError::UnknownCommand("FROBNICATE".to_string()).into();
        assert_eq!(api.code, 400);
    }

    #[test]
    fn from_router_error_checkpoint_cancelled_is_400() {
        let api: ApiError =
            query_router::RouterError::CheckpointError("Operation cancelled by user".to_string())
                .into();
        assert_eq!(api.code, 400);
    }

    #[test]
    fn from_router_error_other_is_500() {
        let api: ApiError =
            query_router::RouterError::ChainError("network down".to_string()).into();
        assert_eq!(api.code, 500);
    }
}
