//! Authentication middleware for API key validation.

use tonic::{Request, Status};

use crate::config::AuthConfig;

/// Extension type to carry authenticated identity through the request.
#[derive(Debug, Clone)]
pub struct AuthenticatedIdentity(pub Option<String>);

/// Validate an incoming request against the authentication configuration.
///
/// Returns the authenticated identity if valid, or an error status if not.
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

    match api_key {
        Some(key) => {
            // Validate the key
            if let Some(identity) = auth_config.validate_key(key) {
                Ok(Some(identity.to_string()))
            } else {
                Err(Status::unauthenticated("invalid API key"))
            }
        },
        None => {
            // No API key provided
            if auth_config.allow_anonymous {
                Ok(None)
            } else {
                Err(Status::unauthenticated("API key required"))
            }
        },
    }
}

/// Extract identity from request, falling back to query-level identity if available.
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
}
