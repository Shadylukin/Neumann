// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Authentication middleware for API key validation.

use tonic::{Request, Status};

use crate::audit::{AuditEvent, AuditLogger};
use crate::config::AuthConfig;
use crate::rate_limit::{Operation, RateLimiter};

/// Extension type to carry authenticated identity through the request.
#[derive(Debug, Clone)]
pub struct AuthenticatedIdentity(pub Option<String>);

/// Validate an incoming request against the authentication configuration.
///
/// Returns the authenticated identity if valid, or an error status if not.
///
/// # Errors
///
/// Returns `Status::unauthenticated` if the API key is missing or invalid.
pub fn validate_request<T>(
    request: &Request<T>,
    config: &Option<AuthConfig>,
) -> Result<Option<String>, Status> {
    let Some(auth_config) = config else {
        // No auth configured - allow all requests
        return Ok(None);
    };

    // Try to extract API key from metadata
    let api_key = request
        .metadata()
        .get(&auth_config.api_key_header)
        .and_then(|v| v.to_str().ok());

    api_key.map_or_else(
        || {
            if auth_config.allow_anonymous {
                Ok(None)
            } else {
                Err(Status::unauthenticated("API key required"))
            }
        },
        |key| {
            auth_config.validate_key(key).map_or_else(
                || Err(Status::unauthenticated("invalid API key")),
                |identity| Ok(Some(identity.to_string())),
            )
        },
    )
}

/// Extract identity from request, falling back to query-level identity if available.
///
/// # Errors
///
/// Returns `Status::unauthenticated` if authentication fails.
pub fn extract_identity<T>(
    request: &Request<T>,
    query_identity: Option<&str>,
    config: &Option<AuthConfig>,
) -> Result<Option<String>, Status> {
    // First try request-level auth
    let request_identity = validate_request(request, config)?;

    // Query-level identity overrides if present
    Ok(query_identity.map(ToString::to_string).or(request_identity))
}

/// Validate request with rate limiting and audit logging.
///
/// This function combines authentication validation with rate limiting and audit logging.
/// It returns the authenticated identity if successful.
///
/// # Errors
///
/// Returns `Status::unauthenticated` if authentication fails, or
/// `Status::resource_exhausted` if the rate limit is exceeded.
pub fn validate_request_with_audit<T>(
    request: &Request<T>,
    auth_config: &Option<AuthConfig>,
    rate_limiter: Option<&RateLimiter>,
    audit_logger: Option<&AuditLogger>,
) -> Result<Option<String>, Status> {
    let remote_addr = request.remote_addr().map(|a| a.to_string());

    // Validate authentication
    let result = validate_request(request, auth_config);

    // Audit the result
    if let Some(logger) = audit_logger {
        match &result {
            Ok(Some(identity)) => {
                logger.record(
                    AuditEvent::AuthSuccess {
                        identity: identity.clone(),
                    },
                    remote_addr.as_deref(),
                );
            },
            Ok(None) => {
                // Anonymous access - no need to log
            },
            Err(status) => {
                logger.record(
                    AuditEvent::AuthFailure {
                        reason: status.message().to_string(),
                    },
                    remote_addr.as_deref(),
                );
            },
        }
    }

    // Check rate limit for all requests (authenticated and anonymous)
    if let Ok(ref identity_opt) = result {
        if let Some(limiter) = rate_limiter {
            // Use identity for authenticated requests, remote address for anonymous
            let rate_limit_key = identity_opt
                .as_deref()
                .unwrap_or_else(|| remote_addr.as_deref().unwrap_or("anonymous"));
            if let Err(msg) = limiter.check_and_record(rate_limit_key, Operation::Request) {
                if let Some(logger) = audit_logger {
                    logger.record(
                        AuditEvent::RateLimited {
                            identity: rate_limit_key.to_string(),
                            operation: "request".to_string(),
                        },
                        remote_addr.as_deref(),
                    );
                }
                return Err(Status::resource_exhausted(msg));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiKey;
    use tonic::metadata::MetadataValue;

    fn create_request_with_key(key: &str) -> Request<()> {
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("x-api-key", MetadataValue::try_from(key).unwrap());
        request
    }

    fn create_request_without_key() -> Request<()> {
        Request::new(())
    }

    #[test]
    fn test_no_auth_config_allows_all() {
        let request = create_request_without_key();
        let result = validate_request(&request, &None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_valid_api_key() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let request = create_request_with_key("test-api-key-12345678");
        let result = validate_request(&request, &Some(auth_config));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:alice".to_string()));
    }

    #[test]
    fn test_invalid_api_key() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let request = create_request_with_key("wrong-key");
        let result = validate_request(&request, &Some(auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_missing_key_with_anonymous_allowed() {
        let auth_config = AuthConfig::new().with_anonymous(true);

        let request = create_request_without_key();
        let result = validate_request(&request, &Some(auth_config));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_missing_key_without_anonymous() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let request = create_request_without_key();
        let result = validate_request(&request, &Some(auth_config));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_extract_identity_query_override() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let request = create_request_with_key("test-api-key-12345678");
        let result = extract_identity(&request, Some("user:bob"), &Some(auth_config));
        assert!(result.is_ok());
        // Query identity overrides request identity
        assert_eq!(result.unwrap(), Some("user:bob".to_string()));
    }

    #[test]
    fn test_extract_identity_fallback_to_request() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let request = create_request_with_key("test-api-key-12345678");
        let result = extract_identity(&request, None, &Some(auth_config));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:alice".to_string()));
    }

    #[test]
    fn test_custom_header() {
        let auth_config = AuthConfig::new()
            .with_header("authorization".to_string())
            .with_api_key(ApiKey::new(
                "Bearer token12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let mut request = Request::new(());
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from("Bearer token12345678").unwrap(),
        );

        let result = validate_request(&request, &Some(auth_config));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:alice".to_string()));
    }

    #[test]
    fn test_validate_with_rate_limit_success() {
        use crate::rate_limit::RateLimitConfig;

        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let limiter = RateLimiter::new(RateLimitConfig::default());
        let request = create_request_with_key("test-api-key-12345678");

        let result =
            validate_request_with_audit(&request, &Some(auth_config), Some(&limiter), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("user:alice".to_string()));
    }

    #[test]
    fn test_validate_with_rate_limit_exceeded() {
        use crate::rate_limit::RateLimitConfig;
        use std::time::Duration;

        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let limiter = RateLimiter::new(
            RateLimitConfig::new()
                .with_max_requests(1)
                .with_window(Duration::from_secs(60)),
        );

        // First request succeeds
        let request = create_request_with_key("test-api-key-12345678");
        let result =
            validate_request_with_audit(&request, &Some(auth_config.clone()), Some(&limiter), None);
        assert!(result.is_ok());

        // Second request should fail
        let request = create_request_with_key("test-api-key-12345678");
        let result =
            validate_request_with_audit(&request, &Some(auth_config), Some(&limiter), None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_validate_with_audit_logging() {
        use crate::audit::AuditConfig;

        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let logger = AuditLogger::new(AuditConfig::default());
        let request = create_request_with_key("test-api-key-12345678");

        let result = validate_request_with_audit(&request, &Some(auth_config), None, Some(&logger));
        assert!(result.is_ok());

        // Check audit log
        assert_eq!(logger.count(), 1);
        let entries = logger.by_identity("user:alice");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_auth_failure_audited() {
        use crate::audit::AuditConfig;

        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let logger = AuditLogger::new(AuditConfig::default());
        let request = create_request_with_key("wrong-key");

        let result = validate_request_with_audit(&request, &Some(auth_config), None, Some(&logger));
        assert!(result.is_err());

        // Auth failure should be logged
        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_rate_limit_audited() {
        use crate::audit::AuditConfig;
        use crate::rate_limit::RateLimitConfig;
        use std::time::Duration;

        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:alice".to_string(),
            ))
            .with_anonymous(false);

        let limiter = RateLimiter::new(
            RateLimitConfig::new()
                .with_max_requests(1)
                .with_window(Duration::from_secs(60)),
        );
        let logger = AuditLogger::new(AuditConfig::default());

        // First request succeeds
        let request = create_request_with_key("test-api-key-12345678");
        let _ = validate_request_with_audit(
            &request,
            &Some(auth_config.clone()),
            Some(&limiter),
            Some(&logger),
        );

        // Second request should be rate limited
        let request = create_request_with_key("test-api-key-12345678");
        let _ = validate_request_with_audit(
            &request,
            &Some(auth_config),
            Some(&limiter),
            Some(&logger),
        );

        // Should have 2 auth success + 1 rate limited
        assert_eq!(logger.count(), 3);
    }
}
