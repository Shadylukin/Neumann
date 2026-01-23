//! Neumann Database Rust Client SDK
//!
//! This crate provides a Rust client for Neumann database with support for
//! both embedded (in-process) and remote (gRPC) modes.
//!
//! # Features
//!
//! - `embedded` - Enable in-process database via `QueryRouter`
//! - `remote` - Enable gRPC client (default)
//! - `full` - Enable both embedded and remote modes
//!
//! # Examples
//!
//! ## Remote Connection
//!
//! ```ignore
//! use neumann_client::{NeumannClient, ClientConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NeumannClient::connect("localhost:9200")
//!         .api_key("your-api-key")
//!         .build()
//!         .await?;
//!
//!     let result = client.execute("SELECT users").await?;
//!     println!("{:?}", result);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Embedded Mode
//!
//! ```ignore
//! use neumann_client::NeumannClient;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NeumannClient::embedded()?;
//!
//!     let result = client.execute_sync("CREATE TABLE users (name:string)")?;
//!     println!("{:?}", result);
//!
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![deny(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    rustdoc::broken_intra_doc_links
)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::significant_drop_tightening)]

mod error;

pub use error::{ClientError, Result};

#[cfg(feature = "embedded")]
use parking_lot::RwLock;
#[cfg(feature = "embedded")]
pub use query_router::QueryResult;
#[cfg(feature = "embedded")]
use query_router::QueryRouter;
#[cfg(feature = "embedded")]
use std::sync::Arc;

use zeroize::Zeroize;

#[cfg(feature = "remote")]
use std::time::Duration;
#[cfg(feature = "remote")]
use tonic::transport::Channel;

/// Generated protobuf types (remote feature only).
#[cfg(feature = "remote")]
#[allow(missing_docs)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
pub mod proto {
    tonic::include_proto!("neumann.v1");
}

#[cfg(feature = "remote")]
use proto::query_service_client::QueryServiceClient;

/// Client mode for Neumann database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientMode {
    /// In-process database using `QueryRouter`.
    Embedded,
    /// Remote database via gRPC.
    Remote,
}

/// Configuration for connecting to a remote server.
///
/// API keys are automatically zeroized on drop for security.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server address.
    pub address: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Whether to use TLS.
    pub tls: bool,
    /// Connection timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            address: "localhost:9200".to_string(),
            api_key: None,
            tls: false,
            timeout_ms: 30_000,
        }
    }
}

impl Drop for ClientConfig {
    fn drop(&mut self) {
        // Zeroize sensitive data on drop
        if let Some(ref mut key) = self.api_key {
            key.zeroize();
        }
    }
}

/// Builder for creating a remote client.
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Create a new builder with the given server address.
    #[must_use]
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            config: ClientConfig {
                address: address.into(),
                api_key: None,
                tls: false,
                timeout_ms: 30_000,
            },
        }
    }

    /// Set the API key for authentication.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Enable TLS encryption.
    #[must_use]
    pub fn with_tls(mut self) -> Self {
        self.config.tls = true;
        self
    }

    /// Set the connection timeout in milliseconds.
    #[must_use]
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.config.timeout_ms = ms;
        self
    }

    /// Build the client and connect to the server.
    #[cfg(feature = "remote")]
    pub async fn build(self) -> Result<NeumannClient> {
        NeumannClient::connect_remote(self.config).await
    }

    /// Build the client and connect to the server (blocking version).
    #[cfg(feature = "remote")]
    pub fn build_blocking(self) -> Result<NeumannClient> {
        // Use a temporary runtime for blocking connection
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| ClientError::Connection(e.to_string()))?;
        rt.block_on(self.build())
    }
}

/// Client for Neumann database.
pub struct NeumannClient {
    mode: ClientMode,
    #[cfg(feature = "embedded")]
    router: Option<Arc<RwLock<QueryRouter>>>,
    #[cfg(feature = "remote")]
    grpc_client: Option<QueryServiceClient<Channel>>,
    #[cfg(feature = "remote")]
    config: Option<ClientConfig>,
    #[cfg(feature = "remote")]
    connected: bool,
}

impl NeumannClient {
    /// Create a builder for connecting to a remote server.
    #[must_use]
    pub fn connect(address: impl Into<String>) -> ClientBuilder {
        ClientBuilder::new(address)
    }

    /// Create an embedded (in-process) client.
    #[cfg(feature = "embedded")]
    pub fn embedded() -> Result<Self> {
        Ok(Self {
            mode: ClientMode::Embedded,
            router: Some(Arc::new(RwLock::new(QueryRouter::new()))),
            #[cfg(feature = "remote")]
            grpc_client: None,
            #[cfg(feature = "remote")]
            config: None,
            #[cfg(feature = "remote")]
            connected: false,
        })
    }

    /// Create an embedded client with a custom router.
    #[cfg(feature = "embedded")]
    pub fn with_router(router: Arc<RwLock<QueryRouter>>) -> Self {
        Self {
            mode: ClientMode::Embedded,
            router: Some(router),
            #[cfg(feature = "remote")]
            grpc_client: None,
            #[cfg(feature = "remote")]
            config: None,
            #[cfg(feature = "remote")]
            connected: false,
        }
    }

    /// Connect to a remote server.
    #[cfg(feature = "remote")]
    async fn connect_remote(config: ClientConfig) -> Result<Self> {
        tracing::info!("Connecting to {}", config.address);

        // Build the endpoint
        let scheme = if config.tls { "https" } else { "http" };
        let uri = format!("{scheme}://{}", config.address);

        let mut endpoint = Channel::from_shared(uri)
            .map_err(|e| ClientError::Connection(format!("Invalid URI: {e}")))?
            .timeout(Duration::from_millis(config.timeout_ms))
            .connect_timeout(Duration::from_millis(config.timeout_ms));

        // Configure TLS if enabled
        if config.tls {
            let tls_config = tonic::transport::ClientTlsConfig::new();
            endpoint = endpoint
                .tls_config(tls_config)
                .map_err(|e| ClientError::Connection(format!("TLS configuration error: {e}")))?;
        }

        // Connect to the server
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| ClientError::Connection(format!("Failed to connect: {e}")))?;

        let grpc_client = QueryServiceClient::new(channel);

        Ok(Self {
            mode: ClientMode::Remote,
            #[cfg(feature = "embedded")]
            router: None,
            grpc_client: Some(grpc_client),
            config: Some(config),
            connected: true,
        })
    }

    /// Get the client mode.
    #[must_use]
    pub fn mode(&self) -> ClientMode {
        self.mode
    }

    /// Check if the client is connected (for remote mode).
    #[must_use]
    pub fn is_connected(&self) -> bool {
        match self.mode {
            ClientMode::Embedded => true,
            #[cfg(feature = "remote")]
            ClientMode::Remote => self.connected,
            #[cfg(not(feature = "remote"))]
            ClientMode::Remote => false,
        }
    }

    /// Execute a query synchronously (for embedded mode).
    #[cfg(feature = "embedded")]
    pub fn execute_sync(&self, query: &str) -> Result<QueryResult> {
        self.execute_sync_with_identity(query, None)
    }

    /// Execute a query synchronously with identity (for embedded mode).
    #[cfg(feature = "embedded")]
    pub fn execute_sync_with_identity(
        &self,
        query: &str,
        identity: Option<&str>,
    ) -> Result<QueryResult> {
        match self.mode {
            ClientMode::Embedded => {
                let router = self
                    .router
                    .as_ref()
                    .ok_or_else(|| ClientError::Internal("Router not initialized".to_string()))?;

                let mut router = router.write();
                if let Some(id) = identity {
                    router.set_identity(id);
                }

                router
                    .execute(query)
                    .map_err(|e| ClientError::Query(e.to_string()))
            },
            ClientMode::Remote => Err(ClientError::InvalidArgument(
                "Use execute() for remote mode".to_string(),
            )),
        }
    }

    /// Execute a query asynchronously (for remote mode).
    #[cfg(feature = "remote")]
    pub async fn execute(&self, query: &str) -> Result<RemoteQueryResult> {
        self.execute_with_identity(query, None).await
    }

    /// Execute a query asynchronously with identity (for remote mode).
    #[cfg(feature = "remote")]
    pub async fn execute_with_identity(
        &self,
        query: &str,
        identity: Option<&str>,
    ) -> Result<RemoteQueryResult> {
        match self.mode {
            #[cfg(feature = "embedded")]
            ClientMode::Embedded => Err(ClientError::InvalidArgument(
                "Use execute_sync() for embedded mode".to_string(),
            )),
            #[cfg(not(feature = "embedded"))]
            ClientMode::Embedded => Err(ClientError::Internal(
                "Embedded mode not available".to_string(),
            )),
            ClientMode::Remote => {
                if !self.connected {
                    return Err(ClientError::Connection("Not connected".to_string()));
                }

                let client = self.grpc_client.as_ref().ok_or_else(|| {
                    ClientError::Internal("gRPC client not initialized".to_string())
                })?;

                let mut request = tonic::Request::new(proto::QueryRequest {
                    query: query.to_string(),
                    identity: identity.map(ToString::to_string),
                });

                // Add API key to metadata if configured
                if let Some(ref config) = self.config {
                    if let Some(ref api_key) = config.api_key {
                        let value = api_key.parse().map_err(|_| {
                            ClientError::InvalidArgument("Invalid API key format".to_string())
                        })?;
                        request.metadata_mut().insert("x-api-key", value);
                    }
                }

                let response = client
                    .clone()
                    .execute(request)
                    .await
                    .map_err(|e| ClientError::Query(e.message().to_string()))?;

                Ok(RemoteQueryResult(response.into_inner()))
            },
        }
    }

    /// Execute multiple queries in a batch (for remote mode).
    #[cfg(feature = "remote")]
    pub async fn execute_batch(&self, queries: &[&str]) -> Result<Vec<RemoteQueryResult>> {
        self.execute_batch_with_identity(queries, None).await
    }

    /// Execute multiple queries in a batch with identity (for remote mode).
    #[cfg(feature = "remote")]
    pub async fn execute_batch_with_identity(
        &self,
        queries: &[&str],
        identity: Option<&str>,
    ) -> Result<Vec<RemoteQueryResult>> {
        match self.mode {
            #[cfg(feature = "embedded")]
            ClientMode::Embedded => Err(ClientError::InvalidArgument(
                "Use execute_sync() for embedded mode".to_string(),
            )),
            #[cfg(not(feature = "embedded"))]
            ClientMode::Embedded => Err(ClientError::Internal(
                "Embedded mode not available".to_string(),
            )),
            ClientMode::Remote => {
                if !self.connected {
                    return Err(ClientError::Connection("Not connected".to_string()));
                }

                let client = self.grpc_client.as_ref().ok_or_else(|| {
                    ClientError::Internal("gRPC client not initialized".to_string())
                })?;

                let batch_request = proto::BatchQueryRequest {
                    queries: queries
                        .iter()
                        .map(|q| proto::QueryRequest {
                            query: (*q).to_string(),
                            identity: identity.map(ToString::to_string),
                        })
                        .collect(),
                };

                let mut request = tonic::Request::new(batch_request);

                // Add API key to metadata if configured
                if let Some(ref config) = self.config {
                    if let Some(ref api_key) = config.api_key {
                        let value = api_key.parse().map_err(|_| {
                            ClientError::InvalidArgument("Invalid API key format".to_string())
                        })?;
                        request.metadata_mut().insert("x-api-key", value);
                    }
                }

                let response = client
                    .clone()
                    .execute_batch(request)
                    .await
                    .map_err(|e| ClientError::Query(e.message().to_string()))?;

                Ok(response
                    .into_inner()
                    .results
                    .into_iter()
                    .map(RemoteQueryResult)
                    .collect())
            },
        }
    }

    /// Close the client connection.
    pub fn close(&mut self) {
        #[cfg(feature = "remote")]
        {
            self.connected = false;
            self.grpc_client = None;
            // Config will be zeroized automatically when dropped
        }
        #[cfg(feature = "embedded")]
        {
            self.router = None;
        }
    }
}

impl Drop for NeumannClient {
    fn drop(&mut self) {
        self.close();
    }
}

/// Result from a remote query execution.
///
/// Wraps the proto `QueryResponse` to provide a type-safe API.
#[cfg(feature = "remote")]
#[derive(Debug)]
pub struct RemoteQueryResult(proto::QueryResponse);

#[cfg(feature = "remote")]
impl RemoteQueryResult {
    /// Check if the result contains an error.
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.0.error.is_some()
    }

    /// Get the error message if present.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.0.error.as_ref().map(|e| e.message.as_str())
    }

    /// Get the underlying proto response.
    #[must_use]
    pub fn into_inner(self) -> proto::QueryResponse {
        self.0
    }

    /// Get a reference to the underlying proto response.
    #[must_use]
    pub fn inner(&self) -> &proto::QueryResponse {
        &self.0
    }

    /// Check if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(
            self.0.result,
            Some(proto::query_response::Result::Empty(_)) | None
        )
    }

    /// Get the count if this is a count result.
    #[must_use]
    pub fn count(&self) -> Option<u64> {
        match &self.0.result {
            Some(proto::query_response::Result::Count(c)) => Some(c.count),
            _ => None,
        }
    }

    /// Get the rows if this is a rows result.
    #[must_use]
    pub fn rows(&self) -> Option<&[proto::Row]> {
        match &self.0.result {
            Some(proto::query_response::Result::Rows(r)) => Some(&r.rows),
            _ => None,
        }
    }

    /// Get the nodes if this is a nodes result.
    #[must_use]
    pub fn nodes(&self) -> Option<&[proto::Node]> {
        match &self.0.result {
            Some(proto::query_response::Result::Nodes(n)) => Some(&n.nodes),
            _ => None,
        }
    }

    /// Get the edges if this is an edges result.
    #[must_use]
    pub fn edges(&self) -> Option<&[proto::Edge]> {
        match &self.0.result {
            Some(proto::query_response::Result::Edges(e)) => Some(&e.edges),
            _ => None,
        }
    }

    /// Get the similar items if this is a similar result.
    #[must_use]
    pub fn similar(&self) -> Option<&[proto::SimilarItem]> {
        match &self.0.result {
            Some(proto::query_response::Result::Similar(s)) => Some(&s.items),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.address, "localhost:9200");
        assert!(config.api_key.is_none());
        assert!(!config.tls);
        assert_eq!(config.timeout_ms, 30_000);
    }

    #[test]
    fn test_client_builder() {
        let builder = NeumannClient::connect("server:9200")
            .api_key("test-key")
            .with_tls()
            .timeout_ms(60_000);

        assert_eq!(builder.config.address, "server:9200");
        assert_eq!(builder.config.api_key, Some("test-key".to_string()));
        assert!(builder.config.tls);
        assert_eq!(builder.config.timeout_ms, 60_000);
    }

    #[test]
    fn test_client_config_zeroize_on_drop() {
        // Test that config with API key is properly cleaned up on drop
        // This test mainly verifies that the Drop impl doesn't panic
        let config = ClientConfig {
            address: "localhost:9200".to_string(),
            api_key: Some("secret-api-key".to_string()),
            tls: false,
            timeout_ms: 30_000,
        };
        // Config will be zeroized when it goes out of scope
        drop(config);
    }

    #[test]
    fn test_client_config_explicit_zeroize() {
        // Test explicit zeroize
        let mut api_key = "secret-api-key".to_string();
        api_key.zeroize();
        assert!(api_key.is_empty());
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn test_embedded_client() {
        let client = NeumannClient::embedded().expect("should create embedded client");
        assert_eq!(client.mode(), ClientMode::Embedded);
        assert!(client.is_connected());
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn test_embedded_execute() {
        let client = NeumannClient::embedded().expect("should create embedded client");

        // Create a table
        let result = client
            .execute_sync("CREATE TABLE test_client (x:int)")
            .expect("should create table");
        // CREATE TABLE returns Empty
        assert!(matches!(result, QueryResult::Empty));

        // Insert data - returns Ids with the inserted row ID
        let result = client
            .execute_sync("INSERT test_client x=42")
            .expect("should insert row");
        assert!(
            matches!(
                result,
                QueryResult::Empty | QueryResult::Count(_) | QueryResult::Ids(_)
            ),
            "INSERT should return Empty, Count, or Ids, got {result:?}",
        );

        // Select data
        let result = client
            .execute_sync("SELECT test_client")
            .expect("should select rows");
        assert!(matches!(result, QueryResult::Rows(_)));
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn test_embedded_execute_with_identity() {
        let client = NeumannClient::embedded().expect("should create embedded client");

        let result = client
            .execute_sync_with_identity("CREATE TABLE id_test (x:int)", Some("test-user"))
            .expect("should create table with identity");
        assert!(matches!(result, QueryResult::Empty));
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn test_embedded_with_custom_router() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let client = NeumannClient::with_router(Arc::clone(&router));

        assert_eq!(client.mode(), ClientMode::Embedded);
        assert!(client.is_connected());

        // Execute through client
        let result = client
            .execute_sync("CREATE TABLE custom_router_test (x:int)")
            .expect("should create table");
        assert!(matches!(result, QueryResult::Empty));
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn test_embedded_close() {
        let mut client = NeumannClient::embedded().expect("should create embedded client");

        // Create table before close
        let result = client
            .execute_sync("CREATE TABLE close_test (x:int)")
            .expect("should create table");
        assert!(matches!(result, QueryResult::Empty));

        // Close clears the router
        client.close();

        // After close, execute should fail (router is None)
        let result = client.execute_sync("SELECT close_test");
        assert!(result.is_err());
    }

    #[test]
    fn test_client_mode_equality() {
        assert_eq!(ClientMode::Embedded, ClientMode::Embedded);
        assert_eq!(ClientMode::Remote, ClientMode::Remote);
        assert_ne!(ClientMode::Embedded, ClientMode::Remote);
    }

    #[test]
    fn test_client_mode_debug() {
        assert!(format!("{:?}", ClientMode::Embedded).contains("Embedded"));
        assert!(format!("{:?}", ClientMode::Remote).contains("Remote"));
    }

    #[test]
    fn test_client_config_debug() {
        let config = ClientConfig {
            address: "localhost:9200".to_string(),
            api_key: Some("secret".to_string()),
            tls: true,
            timeout_ms: 60_000,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("localhost:9200"));
        assert!(debug.contains("tls: true"));
    }

    #[test]
    fn test_client_config_clone() {
        let config1 = ClientConfig {
            address: "server:9200".to_string(),
            api_key: Some("key".to_string()),
            tls: true,
            timeout_ms: 45_000,
        };
        let config2 = config1.clone();
        assert_eq!(config2.address, "server:9200");
        assert_eq!(config2.api_key, Some("key".to_string()));
        assert!(config2.tls);
        assert_eq!(config2.timeout_ms, 45_000);
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_accessors() {
        // Test empty result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Empty(proto::EmptyResult {})),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.has_error());
        assert!(result.error_message().is_none());
        assert!(result.is_empty());
        assert!(result.count().is_none());
        assert!(result.rows().is_none());
        assert!(result.nodes().is_none());
        assert!(result.edges().is_none());
        assert!(result.similar().is_none());

        // Test count result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Count(proto::CountResult {
                count: 42,
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.is_empty());
        assert_eq!(result.count(), Some(42));

        // Test rows result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Rows(proto::RowsResult {
                rows: vec![proto::Row {
                    id: 1,
                    values: vec![],
                }],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(result.rows().is_some());
        assert_eq!(result.rows().unwrap().len(), 1);

        // Test nodes result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Nodes(proto::NodesResult {
                nodes: vec![proto::Node {
                    id: 1,
                    label: "Person".to_string(),
                    properties: std::collections::HashMap::new(),
                }],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(result.nodes().is_some());
        assert_eq!(result.nodes().unwrap().len(), 1);

        // Test edges result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Edges(proto::EdgesResult {
                edges: vec![proto::Edge {
                    id: 1,
                    from: 2,
                    to: 3,
                    label: "KNOWS".to_string(),
                }],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(result.edges().is_some());
        assert_eq!(result.edges().unwrap().len(), 1);

        // Test similar result
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Similar(
                proto::SimilarResult {
                    items: vec![proto::SimilarItem {
                        key: "item1".to_string(),
                        score: 0.95,
                    }],
                },
            )),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(result.similar().is_some());
        assert_eq!(result.similar().unwrap().len(), 1);
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_error() {
        let response = proto::QueryResponse {
            result: None,
            error: Some(proto::ErrorInfo {
                code: 1,
                message: "Something went wrong".to_string(),
                details: None,
            }),
        };
        let result = RemoteQueryResult(response);
        assert!(result.has_error());
        assert_eq!(result.error_message(), Some("Something went wrong"));
        assert!(result.is_empty()); // None result also counts as empty
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_into_inner() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Count(proto::CountResult {
                count: 10,
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);

        // Test inner() reference
        let inner_ref = result.inner();
        assert!(inner_ref.error.is_none());

        // Test into_inner() consumes the result
        let inner = result.into_inner();
        match inner.result {
            Some(proto::query_response::Result::Count(c)) => assert_eq!(c.count, 10),
            _ => panic!("Expected Count result"),
        }
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_debug() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Empty(proto::EmptyResult {})),
            error: None,
        };
        let result = RemoteQueryResult(response);
        let debug = format!("{:?}", result);
        assert!(debug.contains("RemoteQueryResult"));
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_none_result() {
        // Response with no result (None)
        let response = proto::QueryResponse {
            result: None,
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(result.is_empty());
        assert!(result.count().is_none());
        assert!(result.rows().is_none());
        assert!(result.nodes().is_none());
        assert!(result.edges().is_none());
        assert!(result.similar().is_none());
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_path() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Path(proto::PathResult {
                node_ids: vec![1, 2, 3],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.is_empty());
        // Path doesn't have a dedicated accessor, just verify it's not empty
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_table_list() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::TableList(
                proto::TableListResult {
                    tables: vec!["users".to_string(), "orders".to_string()],
                },
            )),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.is_empty());
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_blob() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Blob(proto::BlobResult {
                data: vec![1, 2, 3, 4],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.is_empty());
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_query_result_ids() {
        let response = proto::QueryResponse {
            result: Some(proto::query_response::Result::Ids(proto::IdsResult {
                ids: vec![1, 2, 3],
            })),
            error: None,
        };
        let result = RemoteQueryResult(response);
        assert!(!result.is_empty());
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_builder_without_api_key() {
        let builder = NeumannClient::connect("localhost:9200");
        assert!(builder.config.api_key.is_none());
        assert!(!builder.config.tls);
    }

    #[cfg(all(feature = "embedded", feature = "remote"))]
    #[tokio::test]
    async fn test_embedded_execute_async_returns_error() {
        let client = NeumannClient::embedded().expect("should create embedded client");

        // Calling async execute() on embedded mode should fail
        let result = client.execute("SELECT test").await;
        assert!(result.is_err());
        match result {
            Err(ClientError::InvalidArgument(msg)) => {
                assert!(msg.contains("embedded"));
            },
            other => panic!("Expected InvalidArgument, got {:?}", other),
        }
    }

    #[cfg(all(feature = "embedded", feature = "remote"))]
    #[tokio::test]
    async fn test_embedded_execute_batch_returns_error() {
        let client = NeumannClient::embedded().expect("should create embedded client");

        // Calling execute_batch() on embedded mode should fail
        let result = client.execute_batch(&["SELECT test"]).await;
        assert!(result.is_err());
    }
}
