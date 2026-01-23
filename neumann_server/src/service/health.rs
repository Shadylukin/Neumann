//! Health check service implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::proto::{health_server::Health, HealthCheckRequest, HealthCheckResponse, ServingStatus};

/// Service health state shared across threads.
#[derive(Debug)]
pub struct HealthState {
    query_service_healthy: AtomicBool,
    blob_service_healthy: AtomicBool,
}

impl Default for HealthState {
    fn default() -> Self {
        Self {
            query_service_healthy: AtomicBool::new(true),
            blob_service_healthy: AtomicBool::new(true),
        }
    }
}

impl HealthState {
    /// Create a new health state with all services healthy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the query service health status.
    pub fn set_query_service_healthy(&self, healthy: bool) {
        self.query_service_healthy.store(healthy, Ordering::SeqCst);
    }

    /// Set the blob service health status.
    pub fn set_blob_service_healthy(&self, healthy: bool) {
        self.blob_service_healthy.store(healthy, Ordering::SeqCst);
    }

    /// Check if the query service is healthy.
    #[must_use]
    pub fn is_query_service_healthy(&self) -> bool {
        self.query_service_healthy.load(Ordering::SeqCst)
    }

    /// Check if the blob service is healthy.
    #[must_use]
    pub fn is_blob_service_healthy(&self) -> bool {
        self.blob_service_healthy.load(Ordering::SeqCst)
    }

    /// Check if all services are healthy.
    #[must_use]
    pub fn is_all_healthy(&self) -> bool {
        self.is_query_service_healthy() && self.is_blob_service_healthy()
    }
}

/// Implementation of the Health gRPC service.
#[derive(Debug, Clone)]
pub struct HealthServiceImpl {
    state: Arc<HealthState>,
}

impl Default for HealthServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthServiceImpl {
    /// Create a new health service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(HealthState::new()),
        }
    }

    /// Create a new health service with shared state.
    #[must_use]
    pub fn with_state(state: Arc<HealthState>) -> Self {
        Self { state }
    }

    /// Get a reference to the health state.
    #[must_use]
    pub fn state(&self) -> &Arc<HealthState> {
        &self.state
    }
}

#[tonic::async_trait]
impl Health for HealthServiceImpl {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let service = request.into_inner().service;

        let status = match service.as_deref() {
            Some("neumann.v1.QueryService") => {
                if self.state.is_query_service_healthy() {
                    ServingStatus::Serving
                } else {
                    ServingStatus::NotServing
                }
            },
            Some("neumann.v1.BlobService") => {
                if self.state.is_blob_service_healthy() {
                    ServingStatus::Serving
                } else {
                    ServingStatus::NotServing
                }
            },
            Some("") | None => {
                // Check overall health
                if self.state.is_all_healthy() {
                    ServingStatus::Serving
                } else {
                    ServingStatus::NotServing
                }
            },
            Some(unknown) => {
                tracing::warn!("Health check for unknown service: {}", unknown);
                ServingStatus::Unspecified
            },
        };

        Ok(Response::new(HealthCheckResponse {
            status: status.into(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_no_service() {
        let service = HealthServiceImpl::new();
        let request = Request::new(HealthCheckRequest { service: None });

        let response = service.check(request).await.unwrap();
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Serving)
        );
    }

    #[tokio::test]
    async fn test_health_check_empty_service() {
        let service = HealthServiceImpl::new();
        let request = Request::new(HealthCheckRequest {
            service: Some(String::new()),
        });

        let response = service.check(request).await.unwrap();
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Serving)
        );
    }

    #[tokio::test]
    async fn test_health_check_query_service() {
        let service = HealthServiceImpl::new();
        let request = Request::new(HealthCheckRequest {
            service: Some("neumann.v1.QueryService".to_string()),
        });

        let response = service.check(request).await.unwrap();
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Serving)
        );
    }

    #[tokio::test]
    async fn test_health_check_blob_service() {
        let service = HealthServiceImpl::new();
        let request = Request::new(HealthCheckRequest {
            service: Some("neumann.v1.BlobService".to_string()),
        });

        let response = service.check(request).await.unwrap();
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Serving)
        );
    }

    #[tokio::test]
    async fn test_health_check_unknown_service() {
        let service = HealthServiceImpl::new();
        let request = Request::new(HealthCheckRequest {
            service: Some("unknown.Service".to_string()),
        });

        let response = service.check(request).await.unwrap();
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Unspecified)
        );
    }

    #[tokio::test]
    async fn test_health_state_set_unhealthy() {
        let state = Arc::new(HealthState::new());
        let service = HealthServiceImpl::with_state(Arc::clone(&state));

        // Initially healthy
        assert!(state.is_query_service_healthy());
        assert!(state.is_blob_service_healthy());

        // Set query service unhealthy
        state.set_query_service_healthy(false);

        let request = Request::new(HealthCheckRequest {
            service: Some("neumann.v1.QueryService".to_string()),
        });
        let response = service
            .check(request)
            .await
            .expect("should return response");
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::NotServing)
        );

        // Overall health should be not serving
        let request = Request::new(HealthCheckRequest { service: None });
        let response = service
            .check(request)
            .await
            .expect("should return response");
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::NotServing)
        );

        // Blob service should still be serving
        let request = Request::new(HealthCheckRequest {
            service: Some("neumann.v1.BlobService".to_string()),
        });
        let response = service
            .check(request)
            .await
            .expect("should return response");
        assert_eq!(
            response.into_inner().status,
            i32::from(ServingStatus::Serving)
        );
    }

    #[test]
    fn test_health_state_operations() {
        let state = HealthState::new();

        assert!(state.is_query_service_healthy());
        assert!(state.is_blob_service_healthy());
        assert!(state.is_all_healthy());

        state.set_query_service_healthy(false);
        assert!(!state.is_query_service_healthy());
        assert!(state.is_blob_service_healthy());
        assert!(!state.is_all_healthy());

        state.set_query_service_healthy(true);
        state.set_blob_service_healthy(false);
        assert!(state.is_query_service_healthy());
        assert!(!state.is_blob_service_healthy());
        assert!(!state.is_all_healthy());

        state.set_blob_service_healthy(true);
        assert!(state.is_all_healthy());
    }
}
