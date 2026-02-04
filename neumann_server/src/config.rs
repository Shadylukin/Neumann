// SPDX-License-Identifier: MIT OR Apache-2.0
//! Server configuration types.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use crate::audit::AuditConfig;
use crate::error::{Result, ServerError};
use crate::memory::MemoryBudgetConfig;
use crate::metrics::MetricsConfig;
use crate::rate_limit::RateLimitConfig;
use crate::shutdown::ShutdownConfig;

// Environment variable names for configuration.

/// Bind address environment variable.
pub const ENV_BIND_ADDR: &str = "NEUMANN_BIND_ADDR";
/// Maximum message size environment variable.
pub const ENV_MAX_MESSAGE_SIZE: &str = "NEUMANN_MAX_MESSAGE_SIZE";
/// Maximum upload size environment variable.
pub const ENV_MAX_UPLOAD_SIZE: &str = "NEUMANN_MAX_UPLOAD_SIZE";
/// Enable gRPC-Web environment variable.
pub const ENV_ENABLE_GRPC_WEB: &str = "NEUMANN_ENABLE_GRPC_WEB";
/// Enable reflection environment variable.
pub const ENV_ENABLE_REFLECTION: &str = "NEUMANN_ENABLE_REFLECTION";
/// TLS certificate path environment variable.
pub const ENV_TLS_CERT_PATH: &str = "NEUMANN_TLS_CERT_PATH";
/// TLS key path environment variable.
pub const ENV_TLS_KEY_PATH: &str = "NEUMANN_TLS_KEY_PATH";
/// TLS CA certificate path environment variable.
pub const ENV_TLS_CA_CERT_PATH: &str = "NEUMANN_TLS_CA_CERT_PATH";
/// TLS require client certificate environment variable.
pub const ENV_TLS_REQUIRE_CLIENT_CERT: &str = "NEUMANN_TLS_REQUIRE_CLIENT_CERT";
/// Rate limit max requests environment variable.
pub const ENV_RATE_LIMIT_MAX_REQUESTS: &str = "NEUMANN_RATE_LIMIT_MAX_REQUESTS";
/// Rate limit max queries environment variable.
pub const ENV_RATE_LIMIT_MAX_QUERIES: &str = "NEUMANN_RATE_LIMIT_MAX_QUERIES";
/// Rate limit window seconds environment variable.
pub const ENV_RATE_LIMIT_WINDOW_SECS: &str = "NEUMANN_RATE_LIMIT_WINDOW_SECS";
/// Shutdown drain timeout seconds environment variable.
pub const ENV_SHUTDOWN_DRAIN_TIMEOUT_SECS: &str = "NEUMANN_SHUTDOWN_DRAIN_TIMEOUT_SECS";
/// Shutdown grace period seconds environment variable.
pub const ENV_SHUTDOWN_GRACE_PERIOD_SECS: &str = "NEUMANN_SHUTDOWN_GRACE_PERIOD_SECS";
/// Blob chunk size environment variable.
pub const ENV_BLOB_CHUNK_SIZE: &str = "NEUMANN_BLOB_CHUNK_SIZE";
/// Stream channel capacity environment variable.
pub const ENV_STREAM_CHANNEL_CAPACITY: &str = "NEUMANN_STREAM_CHANNEL_CAPACITY";
/// Max concurrent connections environment variable.
pub const ENV_MAX_CONCURRENT_CONNECTIONS: &str = "NEUMANN_MAX_CONCURRENT_CONNECTIONS";
/// Max concurrent streams per connection environment variable.
pub const ENV_MAX_CONCURRENT_STREAMS: &str = "NEUMANN_MAX_CONCURRENT_STREAMS";
/// Initial window size environment variable.
pub const ENV_INITIAL_WINDOW_SIZE: &str = "NEUMANN_INITIAL_WINDOW_SIZE";
/// Initial connection window size environment variable.
pub const ENV_INITIAL_CONNECTION_WINDOW_SIZE: &str = "NEUMANN_INITIAL_CONNECTION_WINDOW_SIZE";
/// Request timeout seconds environment variable.
pub const ENV_REQUEST_TIMEOUT_SECS: &str = "NEUMANN_REQUEST_TIMEOUT_SECS";
/// Memory budget max bytes environment variable.
pub const ENV_MEMORY_BUDGET_MAX_BYTES: &str = "NEUMANN_MEMORY_BUDGET_MAX_BYTES";
/// Memory budget enable load shedding environment variable.
pub const ENV_MEMORY_BUDGET_LOAD_SHEDDING: &str = "NEUMANN_MEMORY_BUDGET_LOAD_SHEDDING";

/// Environment variable parsing helpers.
mod env_parse {
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;

    use super::{Result, ServerError};

    /// Parse a socket address from an environment variable.
    pub fn parse_socket_addr(key: &str) -> Option<Result<SocketAddr>> {
        std::env::var(key).ok().map(|val| {
            val.parse()
                .map_err(|e| ServerError::Config(format!("invalid {key}: {e}")))
        })
    }

    /// Parse a usize from an environment variable.
    pub fn parse_usize(key: &str) -> Option<Result<usize>> {
        std::env::var(key).ok().map(|val| {
            val.parse()
                .map_err(|e| ServerError::Config(format!("invalid {key}: {e}")))
        })
    }

    /// Parse a u32 from an environment variable.
    pub fn parse_u32(key: &str) -> Option<Result<u32>> {
        std::env::var(key).ok().map(|val| {
            val.parse()
                .map_err(|e| ServerError::Config(format!("invalid {key}: {e}")))
        })
    }

    /// Parse a boolean from an environment variable.
    /// Accepts "true", "1", "yes", "on" as true (case-insensitive).
    /// Accepts "false", "0", "no", "off" as false (case-insensitive).
    pub fn parse_bool(key: &str) -> Option<Result<bool>> {
        std::env::var(key)
            .ok()
            .map(|val| match val.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Ok(true),
                "false" | "0" | "no" | "off" => Ok(false),
                _ => Err(ServerError::Config(format!(
                    "invalid {key}: expected boolean (true/false/1/0/yes/no/on/off)"
                ))),
            })
    }

    /// Parse a duration in seconds from an environment variable.
    pub fn parse_duration_secs(key: &str) -> Option<Result<Duration>> {
        std::env::var(key).ok().map(|val| {
            val.parse::<u64>()
                .map(Duration::from_secs)
                .map_err(|e| ServerError::Config(format!("invalid {key}: {e}")))
        })
    }

    /// Parse a path from an environment variable.
    pub fn parse_path(key: &str) -> Option<PathBuf> {
        std::env::var(key).ok().map(PathBuf::from)
    }
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind the server to.
    pub bind_addr: SocketAddr,
    /// TLS configuration (optional).
    pub tls: Option<TlsConfig>,
    /// Authentication configuration (optional).
    pub auth: Option<AuthConfig>,
    /// Maximum message size in bytes.
    pub max_message_size: usize,
    /// Maximum upload size for blob service in bytes.
    pub max_upload_size: usize,
    /// Enable gRPC-web support for browser clients.
    pub enable_grpc_web: bool,
    /// Enable reflection service for debugging.
    pub enable_reflection: bool,
    /// Blob streaming chunk size.
    pub blob_chunk_size: usize,
    /// Channel capacity for streaming responses (backpressure control).
    pub stream_channel_capacity: usize,
    /// Rate limiting configuration (optional).
    pub rate_limit: Option<RateLimitConfig>,
    /// Audit logging configuration (optional).
    pub audit: Option<AuditConfig>,
    /// Graceful shutdown configuration (optional).
    pub shutdown: Option<ShutdownConfig>,
    /// Metrics configuration (optional).
    pub metrics: Option<MetricsConfig>,
    /// Maximum concurrent connections (None = unlimited).
    pub max_concurrent_connections: Option<usize>,
    /// Maximum HTTP/2 streams per connection.
    pub max_concurrent_streams_per_connection: Option<u32>,
    /// HTTP/2 initial window size.
    pub initial_window_size: Option<u32>,
    /// HTTP/2 connection window size.
    pub initial_connection_window_size: Option<u32>,
    /// Request timeout.
    pub request_timeout: Option<Duration>,
    /// Memory budget configuration (optional).
    pub memory_budget: Option<MemoryBudgetConfig>,
    /// REST API bind address (optional, None disables REST API).
    pub rest_addr: Option<SocketAddr>,
    /// Web admin UI bind address (optional, None disables Web UI).
    pub web_addr: Option<SocketAddr>,
    /// Enhanced streaming configuration (optional).
    pub streaming: Option<StreamingConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:9200".parse().unwrap_or_else(|_| {
                // Safe fallback - this should never fail for a valid literal
                SocketAddr::from(([127, 0, 0, 1], 9200))
            }),
            tls: None,
            auth: None,
            max_message_size: 64 * 1024 * 1024, // 64MB
            max_upload_size: 512 * 1024 * 1024, // 512MB
            enable_grpc_web: true,
            enable_reflection: true,
            blob_chunk_size: 64 * 1024,  // 64KB chunks for streaming
            stream_channel_capacity: 32, // Bounded channel for backpressure
            rate_limit: None,
            audit: None,
            shutdown: None,
            metrics: None,
            max_concurrent_connections: None,
            max_concurrent_streams_per_connection: None,
            initial_window_size: None,
            initial_connection_window_size: None,
            request_timeout: None,
            memory_budget: None,
            rest_addr: None,
            web_addr: None,
            streaming: None,
        }
    }
}

impl ServerConfig {
    /// Create a new server configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables.
    ///
    /// Unset variables use defaults. Invalid values return an error.
    ///
    /// # Supported Environment Variables
    ///
    /// - `NEUMANN_BIND_ADDR` - Server bind address (e.g., "0.0.0.0:9200")
    /// - `NEUMANN_MAX_MESSAGE_SIZE` - Maximum message size in bytes
    /// - `NEUMANN_MAX_UPLOAD_SIZE` - Maximum upload size in bytes
    /// - `NEUMANN_ENABLE_GRPC_WEB` - Enable gRPC-Web (true/false)
    /// - `NEUMANN_ENABLE_REFLECTION` - Enable reflection (true/false)
    /// - `NEUMANN_TLS_CERT_PATH` - Path to TLS certificate
    /// - `NEUMANN_TLS_KEY_PATH` - Path to TLS private key
    /// - `NEUMANN_TLS_CA_CERT_PATH` - Path to CA certificate (optional)
    /// - `NEUMANN_RATE_LIMIT_MAX_REQUESTS` - Max requests per window
    /// - `NEUMANN_RATE_LIMIT_MAX_QUERIES` - Max queries per window
    /// - `NEUMANN_RATE_LIMIT_WINDOW_SECS` - Rate limit window in seconds
    /// - `NEUMANN_SHUTDOWN_DRAIN_TIMEOUT_SECS` - Shutdown drain timeout
    /// - `NEUMANN_SHUTDOWN_GRACE_PERIOD_SECS` - Shutdown grace period
    /// - `NEUMANN_BLOB_CHUNK_SIZE` - Blob streaming chunk size
    /// - `NEUMANN_STREAM_CHANNEL_CAPACITY` - Stream channel capacity
    /// - `NEUMANN_MAX_CONCURRENT_CONNECTIONS` - Max concurrent connections
    /// - `NEUMANN_MAX_CONCURRENT_STREAMS` - Max HTTP/2 streams per connection
    /// - `NEUMANN_INITIAL_WINDOW_SIZE` - HTTP/2 initial window size
    /// - `NEUMANN_INITIAL_CONNECTION_WINDOW_SIZE` - HTTP/2 connection window
    /// - `NEUMANN_REQUEST_TIMEOUT_SECS` - Request timeout in seconds
    /// - `NEUMANN_MEMORY_BUDGET_MAX_BYTES` - Memory budget in bytes
    /// - `NEUMANN_MEMORY_BUDGET_LOAD_SHEDDING` - Enable load shedding (true/false)
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        // Core server settings
        if let Some(result) = env_parse::parse_socket_addr(ENV_BIND_ADDR) {
            config.bind_addr = result?;
        }
        if let Some(result) = env_parse::parse_usize(ENV_MAX_MESSAGE_SIZE) {
            config.max_message_size = result?;
        }
        if let Some(result) = env_parse::parse_usize(ENV_MAX_UPLOAD_SIZE) {
            config.max_upload_size = result?;
        }
        if let Some(result) = env_parse::parse_bool(ENV_ENABLE_GRPC_WEB) {
            config.enable_grpc_web = result?;
        }
        if let Some(result) = env_parse::parse_bool(ENV_ENABLE_REFLECTION) {
            config.enable_reflection = result?;
        }
        if let Some(result) = env_parse::parse_usize(ENV_BLOB_CHUNK_SIZE) {
            config.blob_chunk_size = result?;
        }
        if let Some(result) = env_parse::parse_usize(ENV_STREAM_CHANNEL_CAPACITY) {
            config.stream_channel_capacity = result?;
        }

        // TLS configuration (requires both cert and key)
        let cert_path = env_parse::parse_path(ENV_TLS_CERT_PATH);
        let key_path = env_parse::parse_path(ENV_TLS_KEY_PATH);
        if let (Some(cert), Some(key)) = (cert_path, key_path) {
            let mut tls = TlsConfig::new(cert, key);
            if let Some(ca) = env_parse::parse_path(ENV_TLS_CA_CERT_PATH) {
                tls = tls.with_ca_cert(ca);
            }
            if let Some(result) = env_parse::parse_bool(ENV_TLS_REQUIRE_CLIENT_CERT) {
                tls = tls.with_required_client_cert(result?);
            }
            config.tls = Some(tls);
        }

        // Rate limiting configuration
        let has_rate_limit = std::env::var(ENV_RATE_LIMIT_MAX_REQUESTS).is_ok()
            || std::env::var(ENV_RATE_LIMIT_MAX_QUERIES).is_ok()
            || std::env::var(ENV_RATE_LIMIT_WINDOW_SECS).is_ok();

        if has_rate_limit {
            let mut rate_limit = RateLimitConfig::default();
            if let Some(result) = env_parse::parse_u32(ENV_RATE_LIMIT_MAX_REQUESTS) {
                rate_limit.max_requests = result?;
            }
            if let Some(result) = env_parse::parse_u32(ENV_RATE_LIMIT_MAX_QUERIES) {
                rate_limit.max_queries = result?;
            }
            if let Some(result) = env_parse::parse_duration_secs(ENV_RATE_LIMIT_WINDOW_SECS) {
                rate_limit.window = result?;
            }
            config.rate_limit = Some(rate_limit);
        }

        // Shutdown configuration
        let has_shutdown = std::env::var(ENV_SHUTDOWN_DRAIN_TIMEOUT_SECS).is_ok()
            || std::env::var(ENV_SHUTDOWN_GRACE_PERIOD_SECS).is_ok();

        if has_shutdown {
            let mut shutdown = ShutdownConfig::default();
            if let Some(result) = env_parse::parse_duration_secs(ENV_SHUTDOWN_DRAIN_TIMEOUT_SECS) {
                shutdown.drain_timeout = result?;
            }
            if let Some(result) = env_parse::parse_duration_secs(ENV_SHUTDOWN_GRACE_PERIOD_SECS) {
                shutdown.grace_period = result?;
            }
            config.shutdown = Some(shutdown);
        }

        // Connection limits
        if let Some(result) = env_parse::parse_usize(ENV_MAX_CONCURRENT_CONNECTIONS) {
            config.max_concurrent_connections = Some(result?);
        }
        if let Some(result) = env_parse::parse_u32(ENV_MAX_CONCURRENT_STREAMS) {
            config.max_concurrent_streams_per_connection = Some(result?);
        }
        if let Some(result) = env_parse::parse_u32(ENV_INITIAL_WINDOW_SIZE) {
            config.initial_window_size = Some(result?);
        }
        if let Some(result) = env_parse::parse_u32(ENV_INITIAL_CONNECTION_WINDOW_SIZE) {
            config.initial_connection_window_size = Some(result?);
        }
        if let Some(result) = env_parse::parse_duration_secs(ENV_REQUEST_TIMEOUT_SECS) {
            config.request_timeout = Some(result?);
        }

        // Memory budget configuration
        let has_memory = std::env::var(ENV_MEMORY_BUDGET_MAX_BYTES).is_ok()
            || std::env::var(ENV_MEMORY_BUDGET_LOAD_SHEDDING).is_ok();

        if has_memory {
            let mut memory = MemoryBudgetConfig::default();
            if let Some(result) = env_parse::parse_usize(ENV_MEMORY_BUDGET_MAX_BYTES) {
                memory.max_bytes = result?;
            }
            if let Some(result) = env_parse::parse_bool(ENV_MEMORY_BUDGET_LOAD_SHEDDING) {
                memory.enable_load_shedding = result?;
            }
            config.memory_budget = Some(memory);
        }

        Ok(config)
    }

    /// Set the bind address.
    #[must_use]
    pub fn with_bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Set TLS configuration.
    #[must_use]
    pub fn with_tls(mut self, tls: TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Set authentication configuration.
    #[must_use]
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Set maximum message size.
    #[must_use]
    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    /// Enable or disable gRPC-web support.
    #[must_use]
    pub fn with_grpc_web(mut self, enabled: bool) -> Self {
        self.enable_grpc_web = enabled;
        self
    }

    /// Enable or disable reflection service.
    #[must_use]
    pub fn with_reflection(mut self, enabled: bool) -> Self {
        self.enable_reflection = enabled;
        self
    }

    /// Set blob streaming chunk size.
    #[must_use]
    pub fn with_blob_chunk_size(mut self, size: usize) -> Self {
        self.blob_chunk_size = size;
        self
    }

    /// Set the maximum upload size for blob service.
    #[must_use]
    pub fn with_max_upload_size(mut self, size: usize) -> Self {
        self.max_upload_size = size;
        self
    }

    /// Set the channel capacity for streaming responses.
    ///
    /// Lower values provide better backpressure at the cost of throughput.
    /// Higher values allow more buffering but may use more memory.
    #[must_use]
    pub fn with_stream_channel_capacity(mut self, capacity: usize) -> Self {
        self.stream_channel_capacity = capacity;
        self
    }

    /// Set rate limiting configuration.
    #[must_use]
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }

    /// Set audit logging configuration.
    #[must_use]
    pub fn with_audit(mut self, config: AuditConfig) -> Self {
        self.audit = Some(config);
        self
    }

    /// Set graceful shutdown configuration.
    #[must_use]
    pub fn with_shutdown(mut self, config: ShutdownConfig) -> Self {
        self.shutdown = Some(config);
        self
    }

    /// Set metrics configuration.
    #[must_use]
    pub fn with_metrics(mut self, config: MetricsConfig) -> Self {
        self.metrics = Some(config);
        self
    }

    /// Set maximum concurrent connections.
    #[must_use]
    pub const fn with_max_concurrent_connections(mut self, max: usize) -> Self {
        self.max_concurrent_connections = Some(max);
        self
    }

    /// Set maximum concurrent HTTP/2 streams per connection.
    #[must_use]
    pub const fn with_max_concurrent_streams_per_connection(mut self, max: u32) -> Self {
        self.max_concurrent_streams_per_connection = Some(max);
        self
    }

    /// Set HTTP/2 initial window size.
    #[must_use]
    pub const fn with_initial_window_size(mut self, size: u32) -> Self {
        self.initial_window_size = Some(size);
        self
    }

    /// Set HTTP/2 connection window size.
    #[must_use]
    pub const fn with_initial_connection_window_size(mut self, size: u32) -> Self {
        self.initial_connection_window_size = Some(size);
        self
    }

    /// Set request timeout.
    #[must_use]
    pub const fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Set memory budget configuration.
    #[must_use]
    pub fn with_memory_budget(mut self, config: MemoryBudgetConfig) -> Self {
        self.memory_budget = Some(config);
        self
    }

    /// Set REST API bind address.
    #[must_use]
    pub fn with_rest_addr(mut self, addr: SocketAddr) -> Self {
        self.rest_addr = Some(addr);
        self
    }

    /// Set Web admin UI bind address.
    #[must_use]
    pub fn with_web_addr(mut self, addr: SocketAddr) -> Self {
        self.web_addr = Some(addr);
        self
    }

    /// Set streaming configuration.
    #[must_use]
    pub fn with_streaming(mut self, config: StreamingConfig) -> Self {
        self.streaming = Some(config);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_message_size == 0 {
            return Err(ServerError::Config(
                "max_message_size must be greater than 0".to_string(),
            ));
        }

        if self.blob_chunk_size == 0 {
            return Err(ServerError::Config(
                "blob_chunk_size must be greater than 0".to_string(),
            ));
        }

        if self.stream_channel_capacity == 0 {
            return Err(ServerError::Config(
                "stream_channel_capacity must be greater than 0".to_string(),
            ));
        }

        if let Some(ref tls) = self.tls {
            tls.validate()?;
        }

        if let Some(ref auth) = self.auth {
            auth.validate()?;
        }

        Ok(())
    }
}

/// Enhanced streaming configuration for large result sets.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Channel capacity for streaming responses (backpressure control).
    pub channel_capacity: usize,
    /// Maximum items to stream before requiring cursor continuation.
    pub max_stream_items: usize,
    /// Timeout for slow consumers before dropping connection.
    pub slow_consumer_timeout: Duration,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 32,
            max_stream_items: 10_000,
            slow_consumer_timeout: Duration::from_secs(30),
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set channel capacity.
    #[must_use]
    pub const fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Set maximum stream items.
    #[must_use]
    pub const fn with_max_stream_items(mut self, max: usize) -> Self {
        self.max_stream_items = max;
        self
    }

    /// Set slow consumer timeout.
    #[must_use]
    pub const fn with_slow_consumer_timeout(mut self, timeout: Duration) -> Self {
        self.slow_consumer_timeout = timeout;
        self
    }
}

/// TLS configuration for secure connections.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the certificate file (PEM format).
    pub cert_path: PathBuf,
    /// Path to the private key file (PEM format).
    pub key_path: PathBuf,
    /// Optional path to CA certificate for client authentication.
    pub ca_cert_path: Option<PathBuf>,
    /// Require client certificates.
    pub require_client_cert: bool,
}

impl TlsConfig {
    /// Create a new TLS configuration.
    #[must_use]
    pub fn new(cert_path: PathBuf, key_path: PathBuf) -> Self {
        Self {
            cert_path,
            key_path,
            ca_cert_path: None,
            require_client_cert: false,
        }
    }

    /// Set CA certificate for client authentication.
    #[must_use]
    pub fn with_ca_cert(mut self, path: PathBuf) -> Self {
        self.ca_cert_path = Some(path);
        self
    }

    /// Require client certificates.
    #[must_use]
    pub fn with_required_client_cert(mut self, required: bool) -> Self {
        self.require_client_cert = required;
        self
    }

    /// Validate the TLS configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.cert_path.exists() {
            return Err(ServerError::Config(format!(
                "certificate file not found: {}",
                self.cert_path.display()
            )));
        }

        if !self.key_path.exists() {
            return Err(ServerError::Config(format!(
                "key file not found: {}",
                self.key_path.display()
            )));
        }

        if let Some(ref ca_path) = self.ca_cert_path {
            if !ca_path.exists() {
                return Err(ServerError::Config(format!(
                    "CA certificate file not found: {}",
                    ca_path.display()
                )));
            }
        }

        if self.require_client_cert && self.ca_cert_path.is_none() {
            return Err(ServerError::Config(
                "require_client_cert is true but ca_cert_path is not set; \
                 cannot verify client certificates without a CA"
                    .to_string(),
            ));
        }

        Ok(())
    }
}

/// Authentication configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// API keys that are allowed to access the server.
    pub api_keys: Vec<ApiKey>,
    /// Header name for API key authentication.
    pub api_key_header: String,
    /// Allow unauthenticated access (for development).
    pub allow_anonymous: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            api_key_header: "x-api-key".to_string(),
            allow_anonymous: false,
        }
    }
}

impl AuthConfig {
    /// Create a new authentication configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an API key.
    #[must_use]
    pub fn with_api_key(mut self, key: ApiKey) -> Self {
        self.api_keys.push(key);
        self
    }

    /// Set the API key header name.
    #[must_use]
    pub fn with_header(mut self, header: String) -> Self {
        self.api_key_header = header;
        self
    }

    /// Allow anonymous access.
    #[must_use]
    pub fn with_anonymous(mut self, allowed: bool) -> Self {
        self.allow_anonymous = allowed;
        self
    }

    /// Validate the authentication configuration.
    pub fn validate(&self) -> Result<()> {
        if self.api_key_header.is_empty() {
            return Err(ServerError::Config(
                "api_key_header cannot be empty".to_string(),
            ));
        }

        if !self.allow_anonymous && self.api_keys.is_empty() {
            return Err(ServerError::Config(
                "at least one API key required when anonymous access is disabled".to_string(),
            ));
        }

        for key in &self.api_keys {
            key.validate()?;
        }

        Ok(())
    }

    /// Check if an API key is valid and return the associated identity.
    ///
    /// Uses constant-time comparison to prevent timing attacks, including
    /// key length. We iterate through all keys regardless of whether a
    /// match was found to prevent timing side channels.
    #[must_use]
    pub fn validate_key(&self, key: &str) -> Option<&str> {
        let key_bytes = key.as_bytes();

        // To prevent timing attacks, we must:
        // 1. Always check all keys (no early return on match)
        // 2. Use constant-time comparison that doesn't leak length
        let mut found_identity: Option<&str> = None;

        for api_key in &self.api_keys {
            let stored_bytes = api_key.key.as_bytes();

            // Constant-time length-safe comparison:
            // We compare byte-by-byte up to the longer key length,
            // padding the shorter one with zeros (which won't match printable keys).
            // The result is true only if lengths match AND all bytes match.
            let max_len = stored_bytes.len().max(key_bytes.len());

            let mut matches: u8 = 1;

            // Compare overlapping portion in constant time
            for i in 0..max_len {
                let stored_byte = if i < stored_bytes.len() {
                    stored_bytes[i]
                } else {
                    0
                };
                let key_byte = if i < key_bytes.len() { key_bytes[i] } else { 0 };
                matches &= u8::from(stored_byte == key_byte);
            }

            // Length must also match (constant-time via bitwise)
            let lengths_match = u8::from(stored_bytes.len() == key_bytes.len());
            matches &= lengths_match;

            // If match found, record identity (but don't return early)
            if matches == 1 {
                found_identity = Some(api_key.identity.as_str());
            }
        }

        found_identity
    }
}

/// An API key with associated identity.
#[derive(Debug, Clone)]
pub struct ApiKey {
    /// The API key value.
    pub key: String,
    /// The identity associated with this key.
    pub identity: String,
    /// Optional description for this key.
    pub description: Option<String>,
}

impl ApiKey {
    /// Create a new API key.
    #[must_use]
    pub fn new(key: String, identity: String) -> Self {
        Self {
            key,
            identity,
            description: None,
        }
    }

    /// Set description for this key.
    #[must_use]
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Validate the API key.
    pub fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(ServerError::Config("API key cannot be empty".to_string()));
        }

        if self.key.len() < 16 {
            return Err(ServerError::Config(
                "API key must be at least 16 characters".to_string(),
            ));
        }

        if self.identity.is_empty() {
            return Err(ServerError::Config(
                "API key identity cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_addr.port(), 9200);
        assert!(config.tls.is_none());
        assert!(config.auth.is_none());
        assert!(config.enable_grpc_web);
        assert!(config.enable_reflection);
    }

    #[test]
    fn test_config_builder() {
        let config = ServerConfig::new()
            .with_bind_addr("0.0.0.0:8080".parse().unwrap())
            .with_max_message_size(128 * 1024 * 1024)
            .with_grpc_web(false)
            .with_reflection(false)
            .with_blob_chunk_size(32 * 1024);

        assert_eq!(config.bind_addr.port(), 8080);
        assert_eq!(config.max_message_size, 128 * 1024 * 1024);
        assert!(!config.enable_grpc_web);
        assert!(!config.enable_reflection);
        assert_eq!(config.blob_chunk_size, 32 * 1024);
    }

    #[test]
    fn test_config_validation() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = ServerConfig::new().with_max_message_size(0);
        assert!(invalid_config.validate().is_err());

        let invalid_config = ServerConfig::new().with_blob_chunk_size(0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_auth_config_validation() {
        // Valid with anonymous access
        let auth = AuthConfig::new().with_anonymous(true);
        assert!(auth.validate().is_ok());

        // Invalid: no keys and no anonymous
        let auth = AuthConfig::new().with_anonymous(false);
        assert!(auth.validate().is_err());

        // Valid with API key
        let auth = AuthConfig::new()
            .with_anonymous(false)
            .with_api_key(ApiKey::new(
                "test-api-key-12345678".to_string(),
                "user:test".to_string(),
            ));
        assert!(auth.validate().is_ok());
    }

    #[test]
    fn test_api_key_validation() {
        // Valid key
        let key = ApiKey::new("test-api-key-12345678".to_string(), "user:test".to_string());
        assert!(key.validate().is_ok());

        // Key too short
        let key = ApiKey::new("short".to_string(), "user:test".to_string());
        assert!(key.validate().is_err());

        // Empty identity
        let key = ApiKey::new("test-api-key-12345678".to_string(), String::new());
        assert!(key.validate().is_err());
    }

    #[test]
    fn test_validate_key() {
        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));

        assert_eq!(
            auth.validate_key("test-api-key-12345678"),
            Some("user:alice")
        );
        assert_eq!(auth.validate_key("wrong-key"), None);
    }

    #[test]
    fn test_validate_key_different_lengths() {
        // Test that keys of different lengths don't match (constant-time)
        let auth = AuthConfig::new().with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ));

        // Shorter key
        assert_eq!(auth.validate_key("test-api-key"), None);
        // Longer key
        assert_eq!(auth.validate_key("test-api-key-12345678-extra"), None);
        // Same length, wrong content
        assert_eq!(auth.validate_key("test-api-key-XXXXXXXX"), None);
        // Exact match
        assert_eq!(
            auth.validate_key("test-api-key-12345678"),
            Some("user:alice")
        );
    }

    #[test]
    fn test_validate_key_multiple_keys() {
        // Verify all keys are checked (constant-time)
        let auth = AuthConfig::new()
            .with_api_key(ApiKey::new(
                "first-key-12345678".to_string(),
                "user:first".to_string(),
            ))
            .with_api_key(ApiKey::new(
                "second-key-1234567".to_string(),
                "user:second".to_string(),
            ))
            .with_api_key(ApiKey::new(
                "third-key-12345678".to_string(),
                "user:third".to_string(),
            ));

        assert_eq!(auth.validate_key("first-key-12345678"), Some("user:first"));
        assert_eq!(auth.validate_key("second-key-1234567"), Some("user:second"));
        assert_eq!(auth.validate_key("third-key-12345678"), Some("user:third"));
        assert_eq!(auth.validate_key("unknown-key-12345"), None);
    }

    #[test]
    fn test_tls_config() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        );
        assert!(tls.validate().is_ok());

        // Non-existent cert file
        let tls = TlsConfig::new(
            PathBuf::from("/nonexistent/cert.pem"),
            key_file.path().to_path_buf(),
        );
        assert!(tls.validate().is_err());
    }

    #[test]
    fn test_api_key_with_description() {
        let key = ApiKey::new("test-api-key-12345678".to_string(), "user:test".to_string())
            .with_description("Test API key".to_string());

        assert_eq!(key.description, Some("Test API key".to_string()));
    }

    #[test]
    fn test_tls_config_with_ca() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        let ca_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_ca_cert(ca_file.path().to_path_buf())
        .with_required_client_cert(true);

        assert!(tls.validate().is_ok());
        assert!(tls.require_client_cert);
    }

    #[test]
    fn test_require_client_cert_without_ca_fails_validation() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();

        // require_client_cert = true but no CA cert path
        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_required_client_cert(true);

        let result = tls.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("require_client_cert"));
        assert!(err.contains("ca_cert_path"));
    }

    #[test]
    fn test_require_client_cert_with_ca_passes_validation() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        let ca_file = NamedTempFile::new().unwrap();

        // require_client_cert = true with CA cert path
        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_ca_cert(ca_file.path().to_path_buf())
        .with_required_client_cert(true);

        assert!(tls.validate().is_ok());
    }

    #[test]
    fn test_optional_client_cert_with_ca_passes_validation() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        let ca_file = NamedTempFile::new().unwrap();

        // require_client_cert = false with CA cert path (optional mTLS)
        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_ca_cert(ca_file.path().to_path_buf())
        .with_required_client_cert(false);

        assert!(tls.validate().is_ok());
        assert!(!tls.require_client_cert);
    }

    #[test]
    fn test_auth_config_with_header() {
        let auth = AuthConfig::new()
            .with_header("Authorization".to_string())
            .with_anonymous(true);

        assert_eq!(auth.api_key_header, "Authorization");
        assert!(auth.validate().is_ok());
    }

    #[test]
    fn test_empty_header_validation() {
        let auth = AuthConfig {
            api_keys: Vec::new(),
            api_key_header: String::new(),
            allow_anonymous: true,
        };
        assert!(auth.validate().is_err());
    }

    #[test]
    fn test_server_config_with_rate_limit() {
        let config = ServerConfig::new().with_rate_limit(RateLimitConfig::default());

        assert!(config.rate_limit.is_some());
        let rate_limit = config.rate_limit.as_ref().unwrap();
        assert_eq!(rate_limit.max_requests, 1000);
    }

    #[test]
    fn test_server_config_with_audit() {
        let config = ServerConfig::new().with_audit(AuditConfig::default());

        assert!(config.audit.is_some());
        let audit = config.audit.as_ref().unwrap();
        assert!(audit.enabled);
    }

    #[test]
    fn test_server_config_with_both() {
        let config = ServerConfig::new()
            .with_rate_limit(RateLimitConfig::strict())
            .with_audit(AuditConfig::default().with_query_logging());

        assert!(config.rate_limit.is_some());
        assert!(config.audit.is_some());
        assert_eq!(config.rate_limit.as_ref().unwrap().max_requests, 10);
        assert!(config.audit.as_ref().unwrap().log_queries);
    }

    #[test]
    fn test_default_config_no_rate_limit_or_audit() {
        let config = ServerConfig::default();

        assert!(config.rate_limit.is_none());
        assert!(config.audit.is_none());
    }

    #[test]
    fn test_server_config_with_shutdown() {
        use std::time::Duration;

        let config = ServerConfig::new()
            .with_shutdown(ShutdownConfig::default().with_drain_timeout(Duration::from_secs(60)));

        assert!(config.shutdown.is_some());
        let shutdown = config.shutdown.as_ref().unwrap();
        assert_eq!(shutdown.drain_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_server_config_with_metrics() {
        let config = ServerConfig::new()
            .with_metrics(MetricsConfig::default().with_service_name("test_service".to_string()));

        assert!(config.metrics.is_some());
        let metrics = config.metrics.as_ref().unwrap();
        assert_eq!(metrics.service_name, "test_service");
    }

    #[test]
    fn test_default_config_no_shutdown_or_metrics() {
        let config = ServerConfig::default();

        assert!(config.shutdown.is_none());
        assert!(config.metrics.is_none());
    }

    #[test]
    fn test_server_config_with_max_concurrent_connections() {
        let config = ServerConfig::new().with_max_concurrent_connections(100);

        assert_eq!(config.max_concurrent_connections, Some(100));
    }

    #[test]
    fn test_server_config_with_max_concurrent_streams() {
        let config = ServerConfig::new().with_max_concurrent_streams_per_connection(50);

        assert_eq!(config.max_concurrent_streams_per_connection, Some(50));
    }

    #[test]
    fn test_server_config_with_window_sizes() {
        let config = ServerConfig::new()
            .with_initial_window_size(65535)
            .with_initial_connection_window_size(1024 * 1024);

        assert_eq!(config.initial_window_size, Some(65535));
        assert_eq!(config.initial_connection_window_size, Some(1024 * 1024));
    }

    #[test]
    fn test_server_config_with_request_timeout() {
        let config = ServerConfig::new().with_request_timeout(Duration::from_secs(30));

        assert_eq!(config.request_timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_server_config_with_memory_budget() {
        let config = ServerConfig::new().with_memory_budget(
            MemoryBudgetConfig::new()
                .with_max_bytes(512 * 1024 * 1024)
                .with_load_shedding(true),
        );

        assert!(config.memory_budget.is_some());
        let budget = config.memory_budget.as_ref().unwrap();
        assert_eq!(budget.max_bytes, 512 * 1024 * 1024);
        assert!(budget.enable_load_shedding);
    }

    #[test]
    fn test_default_config_no_connection_limits() {
        let config = ServerConfig::default();

        assert!(config.max_concurrent_connections.is_none());
        assert!(config.max_concurrent_streams_per_connection.is_none());
        assert!(config.initial_window_size.is_none());
        assert!(config.initial_connection_window_size.is_none());
        assert!(config.request_timeout.is_none());
        assert!(config.memory_budget.is_none());
    }

    #[test]
    fn test_server_config_all_connection_limits() {
        let config = ServerConfig::new()
            .with_max_concurrent_connections(200)
            .with_max_concurrent_streams_per_connection(100)
            .with_initial_window_size(65535)
            .with_initial_connection_window_size(2 * 1024 * 1024)
            .with_request_timeout(Duration::from_secs(60))
            .with_memory_budget(MemoryBudgetConfig::default());

        assert_eq!(config.max_concurrent_connections, Some(200));
        assert_eq!(config.max_concurrent_streams_per_connection, Some(100));
        assert_eq!(config.initial_window_size, Some(65535));
        assert_eq!(config.initial_connection_window_size, Some(2 * 1024 * 1024));
        assert_eq!(config.request_timeout, Some(Duration::from_secs(60)));
        assert!(config.memory_budget.is_some());
    }

    // Environment variable tests
    mod env_tests {
        use super::*;
        use std::sync::Mutex;

        // Use a mutex to ensure env var tests don't interfere with each other
        static ENV_MUTEX: Mutex<()> = Mutex::new(());

        fn with_env_vars<F, R>(vars: &[(&str, &str)], f: F) -> R
        where
            F: FnOnce() -> R,
        {
            let _guard = ENV_MUTEX
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            // Save and set env vars
            let saved: Vec<_> = vars
                .iter()
                .map(|(k, v)| {
                    let old = std::env::var(k).ok();
                    std::env::set_var(k, v);
                    (*k, old)
                })
                .collect();

            let result = f();

            // Restore env vars
            for (k, old) in saved {
                match old {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }

            result
        }

        fn without_env_vars<F, R>(keys: &[&str], f: F) -> R
        where
            F: FnOnce() -> R,
        {
            let _guard = ENV_MUTEX
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            // Save and remove env vars
            let saved: Vec<_> = keys
                .iter()
                .map(|k| {
                    let old = std::env::var(k).ok();
                    std::env::remove_var(k);
                    (*k, old)
                })
                .collect();

            let result = f();

            // Restore env vars
            for (k, old) in saved {
                if let Some(v) = old {
                    std::env::set_var(k, v);
                }
            }

            result
        }

        #[test]
        fn test_from_env_defaults() {
            // Ensure no env vars are set
            without_env_vars(
                &[
                    ENV_BIND_ADDR,
                    ENV_MAX_MESSAGE_SIZE,
                    ENV_ENABLE_GRPC_WEB,
                    ENV_RATE_LIMIT_MAX_REQUESTS,
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert_eq!(config.bind_addr.port(), 9200);
                    assert!(config.enable_grpc_web);
                    assert!(config.enable_reflection);
                    assert!(config.rate_limit.is_none());
                },
            );
        }

        #[test]
        fn test_from_env_bind_addr() {
            with_env_vars(&[(ENV_BIND_ADDR, "0.0.0.0:8080")], || {
                let config = ServerConfig::from_env().unwrap();
                assert_eq!(config.bind_addr.to_string(), "0.0.0.0:8080");
            });
        }

        #[test]
        fn test_from_env_invalid_bind_addr() {
            with_env_vars(&[(ENV_BIND_ADDR, "not-an-address")], || {
                let result = ServerConfig::from_env();
                assert!(result.is_err());
                let err = result.unwrap_err().to_string();
                assert!(err.contains(ENV_BIND_ADDR));
            });
        }

        #[test]
        fn test_from_env_max_message_size() {
            with_env_vars(&[(ENV_MAX_MESSAGE_SIZE, "1048576")], || {
                let config = ServerConfig::from_env().unwrap();
                assert_eq!(config.max_message_size, 1048576);
            });
        }

        #[test]
        fn test_from_env_invalid_max_message_size() {
            with_env_vars(&[(ENV_MAX_MESSAGE_SIZE, "not-a-number")], || {
                let result = ServerConfig::from_env();
                assert!(result.is_err());
            });
        }

        #[test]
        fn test_from_env_bool_true_variants() {
            for val in &["true", "1", "yes", "on", "TRUE", "YES", "ON"] {
                with_env_vars(&[(ENV_ENABLE_GRPC_WEB, val)], || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.enable_grpc_web, "failed for value: {val}");
                });
            }
        }

        #[test]
        fn test_from_env_bool_false_variants() {
            for val in &["false", "0", "no", "off", "FALSE", "NO", "OFF"] {
                with_env_vars(&[(ENV_ENABLE_GRPC_WEB, val)], || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(!config.enable_grpc_web, "failed for value: {val}");
                });
            }
        }

        #[test]
        fn test_from_env_invalid_bool() {
            with_env_vars(&[(ENV_ENABLE_GRPC_WEB, "maybe")], || {
                let result = ServerConfig::from_env();
                assert!(result.is_err());
                let err = result.unwrap_err().to_string();
                assert!(err.contains("boolean"));
            });
        }

        #[test]
        fn test_from_env_rate_limit() {
            with_env_vars(
                &[
                    (ENV_RATE_LIMIT_MAX_REQUESTS, "500"),
                    (ENV_RATE_LIMIT_MAX_QUERIES, "100"),
                    (ENV_RATE_LIMIT_WINDOW_SECS, "120"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.rate_limit.is_some());
                    let rate = config.rate_limit.unwrap();
                    assert_eq!(rate.max_requests, 500);
                    assert_eq!(rate.max_queries, 100);
                    assert_eq!(rate.window, Duration::from_secs(120));
                },
            );
        }

        #[test]
        fn test_from_env_shutdown() {
            with_env_vars(
                &[
                    (ENV_SHUTDOWN_DRAIN_TIMEOUT_SECS, "60"),
                    (ENV_SHUTDOWN_GRACE_PERIOD_SECS, "10"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.shutdown.is_some());
                    let shutdown = config.shutdown.unwrap();
                    assert_eq!(shutdown.drain_timeout, Duration::from_secs(60));
                    assert_eq!(shutdown.grace_period, Duration::from_secs(10));
                },
            );
        }

        #[test]
        fn test_from_env_connection_limits() {
            with_env_vars(
                &[
                    (ENV_MAX_CONCURRENT_CONNECTIONS, "100"),
                    (ENV_MAX_CONCURRENT_STREAMS, "50"),
                    (ENV_INITIAL_WINDOW_SIZE, "65535"),
                    (ENV_REQUEST_TIMEOUT_SECS, "30"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert_eq!(config.max_concurrent_connections, Some(100));
                    assert_eq!(config.max_concurrent_streams_per_connection, Some(50));
                    assert_eq!(config.initial_window_size, Some(65535));
                    assert_eq!(config.request_timeout, Some(Duration::from_secs(30)));
                },
            );
        }

        #[test]
        fn test_from_env_memory_budget() {
            with_env_vars(
                &[
                    (ENV_MEMORY_BUDGET_MAX_BYTES, "536870912"),
                    (ENV_MEMORY_BUDGET_LOAD_SHEDDING, "true"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.memory_budget.is_some());
                    let memory = config.memory_budget.unwrap();
                    assert_eq!(memory.max_bytes, 536_870_912);
                    assert!(memory.enable_load_shedding);
                },
            );
        }

        #[test]
        fn test_from_env_tls_requires_both_paths() {
            // Only cert path - should not create TLS config
            with_env_vars(&[(ENV_TLS_CERT_PATH, "/path/to/cert.pem")], || {
                let config = ServerConfig::from_env().unwrap();
                assert!(config.tls.is_none());
            });

            // Only key path - should not create TLS config
            with_env_vars(&[(ENV_TLS_KEY_PATH, "/path/to/key.pem")], || {
                let config = ServerConfig::from_env().unwrap();
                assert!(config.tls.is_none());
            });

            // Both paths - should create TLS config
            with_env_vars(
                &[
                    (ENV_TLS_CERT_PATH, "/path/to/cert.pem"),
                    (ENV_TLS_KEY_PATH, "/path/to/key.pem"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.tls.is_some());
                    let tls = config.tls.unwrap();
                    assert_eq!(tls.cert_path.to_string_lossy(), "/path/to/cert.pem");
                    assert_eq!(tls.key_path.to_string_lossy(), "/path/to/key.pem");
                },
            );
        }

        #[test]
        fn test_from_env_tls_with_ca() {
            with_env_vars(
                &[
                    (ENV_TLS_CERT_PATH, "/path/to/cert.pem"),
                    (ENV_TLS_KEY_PATH, "/path/to/key.pem"),
                    (ENV_TLS_CA_CERT_PATH, "/path/to/ca.pem"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.tls.is_some());
                    let tls = config.tls.unwrap();
                    assert!(tls.ca_cert_path.is_some());
                    assert_eq!(
                        tls.ca_cert_path.unwrap().to_string_lossy(),
                        "/path/to/ca.pem"
                    );
                },
            );
        }

        #[test]
        fn test_from_env_require_client_cert() {
            // Test with require_client_cert = true
            with_env_vars(
                &[
                    (ENV_TLS_CERT_PATH, "/path/to/cert.pem"),
                    (ENV_TLS_KEY_PATH, "/path/to/key.pem"),
                    (ENV_TLS_CA_CERT_PATH, "/path/to/ca.pem"),
                    (ENV_TLS_REQUIRE_CLIENT_CERT, "true"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.tls.is_some());
                    let tls = config.tls.unwrap();
                    assert!(tls.require_client_cert);
                },
            );

            // Test with require_client_cert = false
            with_env_vars(
                &[
                    (ENV_TLS_CERT_PATH, "/path/to/cert.pem"),
                    (ENV_TLS_KEY_PATH, "/path/to/key.pem"),
                    (ENV_TLS_CA_CERT_PATH, "/path/to/ca.pem"),
                    (ENV_TLS_REQUIRE_CLIENT_CERT, "false"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.tls.is_some());
                    let tls = config.tls.unwrap();
                    assert!(!tls.require_client_cert);
                },
            );

            // Test default (no env var) - should be false
            with_env_vars(
                &[
                    (ENV_TLS_CERT_PATH, "/path/to/cert.pem"),
                    (ENV_TLS_KEY_PATH, "/path/to/key.pem"),
                ],
                || {
                    let config = ServerConfig::from_env().unwrap();
                    assert!(config.tls.is_some());
                    let tls = config.tls.unwrap();
                    assert!(!tls.require_client_cert);
                },
            );
        }

        #[test]
        fn test_env_var_constants() {
            // Verify all env var constants follow the NEUMANN_ prefix convention
            assert!(ENV_BIND_ADDR.starts_with("NEUMANN_"));
            assert!(ENV_MAX_MESSAGE_SIZE.starts_with("NEUMANN_"));
            assert!(ENV_ENABLE_GRPC_WEB.starts_with("NEUMANN_"));
            assert!(ENV_TLS_CERT_PATH.starts_with("NEUMANN_"));
            assert!(ENV_TLS_REQUIRE_CLIENT_CERT.starts_with("NEUMANN_"));
            assert!(ENV_RATE_LIMIT_MAX_REQUESTS.starts_with("NEUMANN_"));
            assert!(ENV_SHUTDOWN_DRAIN_TIMEOUT_SECS.starts_with("NEUMANN_"));
            assert!(ENV_MAX_CONCURRENT_CONNECTIONS.starts_with("NEUMANN_"));
            assert!(ENV_MEMORY_BUDGET_MAX_BYTES.starts_with("NEUMANN_"));
        }
    }

    // ========== StreamingConfig tests ==========

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.channel_capacity, 32);
        assert_eq!(config.max_stream_items, 10_000);
        assert_eq!(config.slow_consumer_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_streaming_config_new() {
        let config = StreamingConfig::new();
        assert_eq!(config.channel_capacity, 32);
        assert_eq!(config.max_stream_items, 10_000);
    }

    #[test]
    fn test_streaming_config_with_channel_capacity() {
        let config = StreamingConfig::new().with_channel_capacity(64);
        assert_eq!(config.channel_capacity, 64);
    }

    #[test]
    fn test_streaming_config_with_max_stream_items() {
        let config = StreamingConfig::new().with_max_stream_items(5000);
        assert_eq!(config.max_stream_items, 5000);
    }

    #[test]
    fn test_streaming_config_with_slow_consumer_timeout() {
        let config = StreamingConfig::new().with_slow_consumer_timeout(Duration::from_secs(60));
        assert_eq!(config.slow_consumer_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_streaming_config_builder_chain() {
        let config = StreamingConfig::new()
            .with_channel_capacity(128)
            .with_max_stream_items(20_000)
            .with_slow_consumer_timeout(Duration::from_secs(120));

        assert_eq!(config.channel_capacity, 128);
        assert_eq!(config.max_stream_items, 20_000);
        assert_eq!(config.slow_consumer_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_server_config_with_streaming() {
        let streaming = StreamingConfig::new()
            .with_channel_capacity(64)
            .with_max_stream_items(5000);

        let config = ServerConfig::new().with_streaming(streaming);

        assert!(config.streaming.is_some());
        let s = config.streaming.as_ref().unwrap();
        assert_eq!(s.channel_capacity, 64);
        assert_eq!(s.max_stream_items, 5000);
    }

    #[test]
    fn test_server_config_with_rest_addr() {
        let config = ServerConfig::new().with_rest_addr("0.0.0.0:8080".parse().unwrap());
        assert!(config.rest_addr.is_some());
        assert_eq!(config.rest_addr.unwrap().port(), 8080);
    }

    #[test]
    fn test_server_config_with_web_addr() {
        let config = ServerConfig::new().with_web_addr("0.0.0.0:9000".parse().unwrap());
        assert!(config.web_addr.is_some());
        assert_eq!(config.web_addr.unwrap().port(), 9000);
    }

    #[test]
    fn test_server_config_with_max_upload_size() {
        let config = ServerConfig::new().with_max_upload_size(1024 * 1024 * 1024);
        assert_eq!(config.max_upload_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_server_config_with_stream_channel_capacity() {
        let config = ServerConfig::new().with_stream_channel_capacity(64);
        assert_eq!(config.stream_channel_capacity, 64);
    }

    #[test]
    fn test_server_config_validate_stream_channel_capacity_zero() {
        let config = ServerConfig::new().with_stream_channel_capacity(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tls_config_missing_key_file() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            PathBuf::from("/nonexistent/key.pem"),
        );
        assert!(tls.validate().is_err());
    }

    #[test]
    fn test_tls_config_missing_ca_file() {
        use tempfile::NamedTempFile;

        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();

        let tls = TlsConfig::new(
            cert_file.path().to_path_buf(),
            key_file.path().to_path_buf(),
        )
        .with_ca_cert(PathBuf::from("/nonexistent/ca.pem"));
        assert!(tls.validate().is_err());
    }

    #[test]
    fn test_api_key_empty() {
        let key = ApiKey::new(String::new(), "user:test".to_string());
        assert!(key.validate().is_err());
    }

    #[test]
    fn test_streaming_config_debug() {
        let config = StreamingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("StreamingConfig"));
        assert!(debug_str.contains("channel_capacity"));
    }

    #[test]
    fn test_streaming_config_clone() {
        let config = StreamingConfig::new()
            .with_channel_capacity(100)
            .with_max_stream_items(500);
        let cloned = config.clone();
        assert_eq!(cloned.channel_capacity, 100);
        assert_eq!(cloned.max_stream_items, 500);
    }
}
