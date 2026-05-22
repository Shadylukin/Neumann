// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared authentication and rate-limiting helpers for REST and NLQ API handlers.

use std::sync::Arc;

use axum::http::HeaderMap;

use crate::config::AuthConfig;
use crate::rate_limit::{Operation, RateLimiter};
use crate::rest::error::ApiError;

/// Extract the API key from request headers.
///
/// Uses the configured header name (defaults to `x-api-key`).
pub fn extract_api_key(headers: &HeaderMap, auth_config: Option<&AuthConfig>) -> Option<String> {
    let header_name = auth_config.map_or("x-api-key", |c| c.api_key_header.as_str());

    headers
        .get(header_name)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

/// Validate authentication from request headers.
///
/// Returns `Ok(Some(identity))` for authenticated requests,
/// `Ok(None)` when auth is disabled or anonymous access is allowed,
/// or `Err(ApiError)` when auth is required but missing/invalid.
///
/// # Errors
///
/// Returns `ApiError::unauthorized` if an API key is required but not provided
/// or is invalid.
pub fn validate_auth(
    headers: &HeaderMap,
    auth_config: Option<&AuthConfig>,
) -> Result<Option<String>, ApiError> {
    let api_key = extract_api_key(headers, auth_config);

    match (auth_config, api_key) {
        (None, _) => Ok(None),
        (Some(config), None) => {
            if config.allow_anonymous {
                Ok(None)
            } else {
                Err(ApiError::unauthorized("API key required"))
            }
        },
        (Some(config), Some(key)) => config.validate_key(&key).map_or_else(
            || Err(ApiError::unauthorized("Invalid API key")),
            |identity| Ok(Some(identity.to_string())),
        ),
    }
}

/// Check rate limits for the given identity and operation.
///
/// # Errors
///
/// Returns `ApiError::rate_limited` if the rate limit is exceeded.
pub fn check_rate_limit(
    identity: Option<&str>,
    rate_limiter: Option<&Arc<RateLimiter>>,
    operation: Operation,
) -> Result<(), ApiError> {
    if let Some(limiter) = rate_limiter {
        if let Some(id) = identity {
            if let Err(msg) = limiter.check_and_record(id, operation) {
                tracing::warn!("Rate limited: {id} for {}", operation.as_str());
                return Err(ApiError::rate_limited(msg));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_auth_no_config() {
        let headers = HeaderMap::new();
        let result = validate_auth(&headers, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_validate_auth_missing_key_anonymous_allowed() {
        let config = AuthConfig {
            api_key_header: "x-api-key".to_string(),
            api_keys: vec![],
            allow_anonymous: true,
        };
        let headers = HeaderMap::new();
        let result = validate_auth(&headers, Some(&config));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_validate_auth_missing_key_required() {
        let config = AuthConfig {
            api_key_header: "x-api-key".to_string(),
            api_keys: vec![],
            allow_anonymous: false,
        };
        let headers = HeaderMap::new();
        let result = validate_auth(&headers, Some(&config));
        assert!(result.is_err());
    }

    #[test]
    fn test_check_rate_limit_no_limiter() {
        let result = check_rate_limit(Some("user1"), None, Operation::Query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rate_limit_no_identity() {
        let limiter = Arc::new(RateLimiter::new(
            crate::rate_limit::RateLimitConfig::default(),
        ));
        let result = check_rate_limit(None, Some(&limiter), Operation::Query);
        assert!(result.is_ok());
    }
}
