//! TCP transport configuration.

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use super::{compression::CompressionConfig, error::TcpResult, rate_limit::RateLimitConfig};
use crate::block::NodeId;

/// Security mode for TLS configuration.
///
/// Controls the level of TLS enforcement:
/// - `Strict`: Production-ready. Requires TLS, mTLS, and NodeId verification.
/// - `Permissive`: TLS required but mTLS is optional.
/// - `Development`: No TLS required (logs warning). For local testing only.
/// - `Legacy`: Current behavior for migration (TLS optional).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SecurityMode {
    /// Production-ready: TLS + mTLS + NodeId verification required.
    #[default]
    Strict,
    /// TLS required but mTLS optional.
    Permissive,
    /// No TLS required (logs warning). For local development only.
    Development,
    /// Backward-compatible mode. TLS is optional.
    Legacy,
}

impl SecurityMode {
    pub fn requires_tls(&self) -> bool {
        matches!(self, Self::Strict | Self::Permissive)
    }

    pub fn requires_mtls(&self) -> bool {
        matches!(self, Self::Strict)
    }

    pub fn requires_node_id_verification(&self) -> bool {
        matches!(self, Self::Strict)
    }

    pub fn should_warn(&self) -> bool {
        matches!(self, Self::Development | Self::Legacy)
    }
}

/// Security configuration for TCP transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Security mode controlling TLS enforcement level.
    pub mode: SecurityMode,
    /// Whether to log warnings for insecure configurations.
    #[serde(default = "default_warn_on_insecure")]
    pub warn_on_insecure: bool,
}

fn default_warn_on_insecure() -> bool {
    true
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            mode: SecurityMode::default(),
            warn_on_insecure: true,
        }
    }
}

impl SecurityConfig {
    pub fn strict() -> Self {
        Self {
            mode: SecurityMode::Strict,
            warn_on_insecure: true,
        }
    }

    pub fn permissive() -> Self {
        Self {
            mode: SecurityMode::Permissive,
            warn_on_insecure: true,
        }
    }

    pub fn development() -> Self {
        Self {
            mode: SecurityMode::Development,
            warn_on_insecure: true,
        }
    }

    pub fn legacy() -> Self {
        Self {
            mode: SecurityMode::Legacy,
            warn_on_insecure: true,
        }
    }

    pub fn without_warnings(mut self) -> Self {
        self.warn_on_insecure = false;
        self
    }
}

/// Configuration for TCP transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpTransportConfig {
    /// Local address to bind for incoming connections.
    pub bind_address: SocketAddr,

    /// Local node ID.
    pub node_id: NodeId,

    /// Number of connections per peer in the pool.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,

    /// Connection timeout in milliseconds.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_ms: u64,

    /// Read/write timeout in milliseconds.
    #[serde(default = "default_io_timeout")]
    pub io_timeout_ms: u64,

    /// Maximum message size in bytes.
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,

    /// Enable TCP keepalive.
    #[serde(default = "default_keepalive")]
    pub keepalive: bool,

    /// Keepalive interval in seconds.
    #[serde(default = "default_keepalive_interval")]
    pub keepalive_interval_secs: u64,

    /// Reconnection configuration.
    #[serde(default)]
    pub reconnect: ReconnectConfig,

    /// Optional TLS configuration.
    #[serde(default)]
    pub tls: Option<TlsConfig>,

    /// Require TLS for all connections. If true and tls is None, connections will fail.
    /// Default is false for backwards compatibility, but should be enabled in production.
    #[serde(default)]
    pub require_tls: bool,

    /// Maximum queued outbound messages per peer.
    #[serde(default = "default_max_pending")]
    pub max_pending_messages: usize,

    /// Channel buffer size for incoming messages.
    #[serde(default = "default_recv_buffer")]
    pub recv_buffer_size: usize,

    /// Compression configuration.
    #[serde(default)]
    pub compression: CompressionConfig,

    /// Per-peer rate limiting configuration.
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// Security configuration controlling TLS enforcement.
    #[serde(default)]
    pub security: SecurityConfig,
}

fn default_pool_size() -> usize {
    2
}
fn default_connect_timeout() -> u64 {
    5000
}
fn default_io_timeout() -> u64 {
    30000
}
fn default_max_message_size() -> usize {
    16 * 1024 * 1024
} // 16 MB
fn default_keepalive() -> bool {
    true
}
fn default_keepalive_interval() -> u64 {
    30
}
fn default_max_pending() -> usize {
    1000
}
fn default_recv_buffer() -> usize {
    1000
}

impl Default for TcpTransportConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:9100".parse().unwrap(),
            node_id: String::new(),
            pool_size: default_pool_size(),
            connect_timeout_ms: default_connect_timeout(),
            io_timeout_ms: default_io_timeout(),
            max_message_size: default_max_message_size(),
            keepalive: default_keepalive(),
            keepalive_interval_secs: default_keepalive_interval(),
            reconnect: ReconnectConfig::default(),
            tls: None,
            require_tls: true, // Default to true for security
            max_pending_messages: default_max_pending(),
            recv_buffer_size: default_recv_buffer(),
            compression: CompressionConfig::default(),
            rate_limit: RateLimitConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl TcpTransportConfig {
    pub fn new(node_id: impl Into<NodeId>, bind_address: SocketAddr) -> Self {
        Self {
            node_id: node_id.into(),
            bind_address,
            ..Default::default()
        }
    }

    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout_ms = timeout.as_millis() as u64;
        self
    }

    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout_ms = timeout.as_millis() as u64;
        self
    }

    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    pub fn with_reconnect(mut self, config: ReconnectConfig) -> Self {
        self.reconnect = config;
        self
    }

    pub fn with_keepalive(mut self, enabled: bool) -> Self {
        self.keepalive = enabled;
        self
    }

    pub fn with_keepalive_interval_secs(mut self, secs: u64) -> Self {
        self.keepalive_interval_secs = secs;
        self
    }

    pub fn with_tls(mut self, config: TlsConfig) -> Self {
        self.tls = Some(config);
        self
    }

    pub fn with_require_tls(mut self, require: bool) -> Self {
        self.require_tls = require;
        self
    }

    /// Returns true if TLS is required based on both explicit require_tls and security mode.
    pub fn effective_require_tls(&self) -> bool {
        self.security.mode.requires_tls() || self.require_tls
    }

    pub fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub fn io_timeout(&self) -> Duration {
        Duration::from_millis(self.io_timeout_ms)
    }

    pub fn with_compression(mut self, config: CompressionConfig) -> Self {
        self.compression = config;
        self
    }

    pub fn compression_disabled(mut self) -> Self {
        self.compression.enabled = false;
        self
    }

    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = config;
        self
    }

    pub fn rate_limit_disabled(mut self) -> Self {
        self.rate_limit.enabled = false;
        self
    }

    pub fn with_security(mut self, config: SecurityConfig) -> Self {
        self.security = config;
        self
    }

    pub fn with_security_mode(mut self, mode: SecurityMode) -> Self {
        self.security.mode = mode;
        self
    }

    /// Validate security configuration against current TLS settings.
    ///
    /// Returns an error if the security mode requires TLS but TLS is not configured.
    pub fn validate_security(&self) -> TcpResult<()> {
        use super::error::TcpError;

        let mode = self.security.mode;

        // Log warnings for non-strict modes
        if self.security.warn_on_insecure && mode.should_warn() {
            tracing::warn!(
                "TCP transport using insecure mode {:?}. This is not recommended for production.",
                mode
            );
        }

        // Check TLS requirements
        if mode.requires_tls() && self.tls.is_none() {
            return Err(TcpError::TlsRequired {
                mode,
                reason: "TLS configuration is required but not provided".to_string(),
            });
        }

        // Check mTLS requirements
        if mode.requires_mtls() {
            if let Some(ref tls) = self.tls {
                if !tls.require_client_auth {
                    return Err(TcpError::MtlsRequired {
                        mode,
                        reason: "Mutual TLS (client auth) is required but not enabled".to_string(),
                    });
                }
            }
        }

        // Check NodeId verification requirements
        if mode.requires_node_id_verification() {
            if let Some(ref tls) = self.tls {
                if tls.node_id_verification == NodeIdVerification::None {
                    return Err(TcpError::NodeIdVerificationRequired {
                        mode,
                        reason: "NodeId verification is required but not configured".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if the current configuration is secure for production use.
    pub fn is_secure(&self) -> bool {
        self.validate_security().is_ok()
            && matches!(
                self.security.mode,
                SecurityMode::Strict | SecurityMode::Permissive
            )
    }
}

/// Configuration for automatic reconnection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectConfig {
    /// Enable automatic reconnection.
    #[serde(default = "default_reconnect_enabled")]
    pub enabled: bool,

    /// Initial backoff in milliseconds.
    #[serde(default = "default_initial_backoff")]
    pub initial_backoff_ms: u64,

    /// Maximum backoff in milliseconds.
    #[serde(default = "default_max_backoff")]
    pub max_backoff_ms: u64,

    /// Backoff multiplier.
    #[serde(default = "default_multiplier")]
    pub multiplier: f64,

    /// Maximum reconnection attempts (None = infinite).
    #[serde(default)]
    pub max_attempts: Option<usize>,

    /// Jitter factor (0.0 to 1.0) to randomize backoff.
    #[serde(default = "default_jitter")]
    pub jitter: f64,
}

fn default_reconnect_enabled() -> bool {
    true
}
fn default_initial_backoff() -> u64 {
    100
}
fn default_max_backoff() -> u64 {
    30000
}
fn default_multiplier() -> f64 {
    2.0
}
fn default_jitter() -> f64 {
    0.1
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            enabled: default_reconnect_enabled(),
            initial_backoff_ms: default_initial_backoff(),
            max_backoff_ms: default_max_backoff(),
            multiplier: default_multiplier(),
            max_attempts: None,
            jitter: default_jitter(),
        }
    }
}

impl ReconnectConfig {
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        let base = self.initial_backoff_ms as f64 * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max_backoff_ms as f64);

        // Apply jitter
        let jitter_range = capped * self.jitter;
        let jitter_offset = rand::random::<f64>() * jitter_range * 2.0 - jitter_range;
        let final_ms = (capped + jitter_offset).max(0.0);

        Duration::from_millis(final_ms as u64)
    }

    pub fn should_retry(&self, attempt: usize) -> bool {
        if !self.enabled {
            return false;
        }
        match self.max_attempts {
            Some(max) => attempt < max,
            None => true,
        }
    }
}

/// How to verify that a peer's NodeId matches their TLS certificate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeIdVerification {
    /// No verification - NodeId is trusted from handshake (testing only).
    #[default]
    None,
    /// NodeId must match certificate Common Name (CN).
    CommonName,
    /// NodeId must match a Subject Alternative Name (SAN) DNS entry.
    SubjectAltName,
}

/// TLS configuration for encrypted connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to certificate chain PEM file.
    pub cert_path: PathBuf,

    /// Path to private key PEM file.
    pub key_path: PathBuf,

    /// Path to CA certificate for peer verification.
    #[serde(default)]
    pub ca_cert_path: Option<PathBuf>,

    /// Require client certificates (mutual TLS).
    #[serde(default)]
    pub require_client_auth: bool,

    /// Skip certificate verification (for testing only).
    /// SECURITY: This field is ignored in release builds - verification is always enforced.
    #[serde(default)]
    pub insecure_skip_verify: bool,

    /// How to verify NodeId against TLS certificate.
    #[serde(default)]
    pub node_id_verification: NodeIdVerification,
}

impl TlsConfig {
    pub fn new(cert_path: impl Into<PathBuf>, key_path: impl Into<PathBuf>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            ca_cert_path: None,
            require_client_auth: false,
            insecure_skip_verify: false,
            node_id_verification: NodeIdVerification::default(),
        }
    }

    pub fn with_ca_cert(mut self, path: impl Into<PathBuf>) -> Self {
        self.ca_cert_path = Some(path.into());
        self
    }

    pub fn with_client_auth(mut self) -> Self {
        self.require_client_auth = true;
        self
    }

    pub fn with_node_id_verification(mut self, mode: NodeIdVerification) -> Self {
        self.node_id_verification = mode;
        self
    }

    /// Create a secure TLS configuration suitable for production.
    ///
    /// This constructor enforces:
    /// - Mutual TLS (client certificate required)
    /// - NodeId verification via Common Name
    /// - Certificate verification (no insecure skip)
    pub fn new_secure(
        cert_path: impl Into<PathBuf>,
        key_path: impl Into<PathBuf>,
        ca_cert_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            ca_cert_path: Some(ca_cert_path.into()),
            require_client_auth: true,
            insecure_skip_verify: false,
            node_id_verification: NodeIdVerification::CommonName,
        }
    }

    /// Check if certificate verification should be performed.
    /// SECURITY: In release builds, this ALWAYS returns true regardless of
    /// the `insecure_skip_verify` field, preventing TLS bypass in production.
    #[inline]
    pub fn should_verify(&self) -> bool {
        #[cfg(any(test, debug_assertions))]
        {
            !self.insecure_skip_verify
        }
        #[cfg(not(any(test, debug_assertions)))]
        {
            true // Always verify in release builds
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TcpTransportConfig::default();
        assert_eq!(config.pool_size, 2);
        assert_eq!(config.connect_timeout_ms, 5000);
        assert_eq!(config.max_message_size, 16 * 1024 * 1024);
        assert!(config.reconnect.enabled);
        assert!(config.keepalive);
        assert_eq!(config.keepalive_interval_secs, 30);
        assert!(config.tls.is_none());
        assert!(config.require_tls); // Default is true for security
    }

    #[test]
    fn test_config_builder() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_pool_size(4)
            .with_connect_timeout(Duration::from_secs(10))
            .with_max_message_size(1024 * 1024);

        assert_eq!(config.node_id, "node1");
        assert_eq!(config.pool_size, 4);
        assert_eq!(config.connect_timeout_ms, 10000);
        assert_eq!(config.max_message_size, 1024 * 1024);
    }

    #[test]
    fn test_config_io_timeout() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_io_timeout(Duration::from_secs(60));

        assert_eq!(config.io_timeout_ms, 60000);
        assert_eq!(config.io_timeout(), Duration::from_secs(60));
    }

    #[test]
    fn test_config_keepalive() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_keepalive(false)
            .with_keepalive_interval_secs(60);

        assert!(!config.keepalive);
        assert_eq!(config.keepalive_interval_secs, 60);
    }

    #[test]
    fn test_config_reconnect() {
        let reconnect = ReconnectConfig {
            enabled: false,
            max_attempts: Some(10),
            ..Default::default()
        };

        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_reconnect(reconnect);

        assert!(!config.reconnect.enabled);
        assert_eq!(config.reconnect.max_attempts, Some(10));
    }

    #[test]
    fn test_config_connect_timeout_duration() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_connect_timeout(Duration::from_millis(2500));

        assert_eq!(config.connect_timeout(), Duration::from_millis(2500));
    }

    #[test]
    fn test_reconnect_backoff() {
        let config = ReconnectConfig {
            enabled: true,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            multiplier: 2.0,
            max_attempts: Some(5),
            jitter: 0.0, // No jitter for predictable testing
        };

        // Attempt 0: 100ms
        let d0 = config.backoff_for_attempt(0);
        assert_eq!(d0.as_millis(), 100);

        // Attempt 1: 200ms
        let d1 = config.backoff_for_attempt(1);
        assert_eq!(d1.as_millis(), 200);

        // Attempt 2: 400ms
        let d2 = config.backoff_for_attempt(2);
        assert_eq!(d2.as_millis(), 400);

        // Attempt 10: capped at 5000ms
        let d10 = config.backoff_for_attempt(10);
        assert_eq!(d10.as_millis(), 5000);
    }

    #[test]
    fn test_reconnect_backoff_with_jitter() {
        let config = ReconnectConfig {
            enabled: true,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            multiplier: 2.0,
            max_attempts: None,
            jitter: 0.2, // 20% jitter
        };

        // Run multiple times to check jitter variability
        let d1 = config.backoff_for_attempt(0);
        // Should be within 20% of 1000ms: 800-1200ms
        assert!(d1.as_millis() >= 800 && d1.as_millis() <= 1200);
    }

    #[test]
    fn test_reconnect_should_retry() {
        let config = ReconnectConfig {
            enabled: true,
            max_attempts: Some(3),
            ..Default::default()
        };

        assert!(config.should_retry(0));
        assert!(config.should_retry(1));
        assert!(config.should_retry(2));
        assert!(!config.should_retry(3));

        // Disabled config
        let disabled = ReconnectConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(!disabled.should_retry(0));
    }

    #[test]
    fn test_reconnect_unlimited_retries() {
        let config = ReconnectConfig {
            enabled: true,
            max_attempts: None,
            ..Default::default()
        };

        assert!(config.should_retry(0));
        assert!(config.should_retry(100));
        assert!(config.should_retry(1000));
    }

    #[test]
    fn test_reconnect_default() {
        let config = ReconnectConfig::default();
        assert!(config.enabled);
        assert_eq!(config.initial_backoff_ms, 100);
        assert_eq!(config.max_backoff_ms, 30000);
        assert_eq!(config.multiplier, 2.0);
        assert!(config.max_attempts.is_none());
        assert_eq!(config.jitter, 0.1);
    }

    #[test]
    fn test_tls_config_new() {
        let tls = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");

        assert_eq!(tls.cert_path.to_str().unwrap(), "/path/to/cert.pem");
        assert_eq!(tls.key_path.to_str().unwrap(), "/path/to/key.pem");
        assert!(tls.ca_cert_path.is_none());
        assert!(!tls.require_client_auth);
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_config_with_ca_cert() {
        let tls =
            TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem").with_ca_cert("/path/to/ca.pem");

        assert!(tls.ca_cert_path.is_some());
        assert_eq!(
            tls.ca_cert_path.unwrap().to_str().unwrap(),
            "/path/to/ca.pem"
        );
    }

    #[test]
    fn test_tls_config_with_client_auth() {
        let tls = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem").with_client_auth();

        assert!(tls.require_client_auth);
    }

    #[test]
    fn test_tls_should_verify_default() {
        let tls = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");
        // Default should verify
        assert!(tls.should_verify());
    }

    #[test]
    fn test_tls_should_verify_in_test_mode() {
        // In test builds, insecure_skip_verify can disable verification
        let mut tls = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");
        tls.insecure_skip_verify = true;
        // In test mode (which this is), should_verify returns false
        assert!(!tls.should_verify());
    }

    #[test]
    fn test_config_with_tls() {
        let tls = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");
        let config =
            TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap()).with_tls(tls);

        assert!(config.tls.is_some());
        let tls = config.tls.unwrap();
        assert_eq!(tls.cert_path.to_str().unwrap(), "/path/to/cert.pem");
    }

    #[test]
    fn test_config_debug() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let debug = format!("{:?}", config);
        assert!(debug.contains("TcpTransportConfig"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_config_clone() {
        let config =
            TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap()).with_pool_size(4);
        let cloned = config.clone();

        assert_eq!(cloned.node_id, "node1");
        assert_eq!(cloned.pool_size, 4);
    }

    #[test]
    fn test_reconnect_debug() {
        let config = ReconnectConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ReconnectConfig"));
    }

    #[test]
    fn test_tls_debug() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        let debug = format!("{:?}", tls);
        assert!(debug.contains("TlsConfig"));
    }

    #[test]
    fn test_config_compression() {
        use crate::tcp::compression::CompressionMethod;

        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());

        assert!(config.compression.enabled);
        assert_eq!(config.compression.method, CompressionMethod::Lz4);
    }

    #[test]
    fn test_config_compression_disabled() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .compression_disabled();

        assert!(!config.compression.enabled);
    }

    #[test]
    fn test_config_with_compression() {
        use crate::tcp::compression::{CompressionConfig, CompressionMethod};

        let compression = CompressionConfig::default()
            .with_method(CompressionMethod::None)
            .with_min_size(1024);

        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_compression(compression);

        assert_eq!(config.compression.method, CompressionMethod::None);
        assert_eq!(config.compression.min_size, 1024);
    }

    #[test]
    fn test_config_rate_limit() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());

        assert!(config.rate_limit.enabled);
        assert_eq!(config.rate_limit.bucket_size, 100);
    }

    #[test]
    fn test_config_rate_limit_disabled() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .rate_limit_disabled();

        assert!(!config.rate_limit.enabled);
    }

    #[test]
    fn test_config_with_rate_limit() {
        use crate::tcp::rate_limit::RateLimitConfig;

        let rate_limit = RateLimitConfig::aggressive();

        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_rate_limit(rate_limit);

        assert_eq!(config.rate_limit.bucket_size, 50);
        assert_eq!(config.rate_limit.refill_rate, 25.0);
    }

    #[test]
    fn test_config_require_tls() {
        // Default should be true for security
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        assert!(config.require_tls);

        // Can disable require_tls
        let config = config.with_require_tls(false);
        assert!(!config.require_tls);

        // With TLS config and require_tls
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_require_tls(true);
        assert!(config.require_tls);
        assert!(config.tls.is_some());
    }

    #[test]
    fn test_effective_require_tls() {
        // Default config with Strict mode should require TLS
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        assert!(config.effective_require_tls());

        // Even with require_tls=false, Strict mode should require TLS
        let config = config.clone().with_require_tls(false);
        assert!(config.effective_require_tls()); // Strict mode requires TLS

        // With Development mode and require_tls=false, should not require TLS
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Development)
            .with_require_tls(false);
        assert!(!config.effective_require_tls());

        // With Development mode but require_tls=true, should require TLS
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Development)
            .with_require_tls(true);
        assert!(config.effective_require_tls());
    }

    #[test]
    fn test_node_id_verification_default() {
        let mode = NodeIdVerification::default();
        assert_eq!(mode, NodeIdVerification::None);
    }

    #[test]
    fn test_node_id_verification_variants() {
        let none = NodeIdVerification::None;
        let cn = NodeIdVerification::CommonName;
        let san = NodeIdVerification::SubjectAltName;

        assert_eq!(format!("{:?}", none), "None");
        assert_eq!(format!("{:?}", cn), "CommonName");
        assert_eq!(format!("{:?}", san), "SubjectAltName");
    }

    #[test]
    fn test_tls_config_with_node_id_verification() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem")
            .with_node_id_verification(NodeIdVerification::CommonName);

        assert_eq!(tls.node_id_verification, NodeIdVerification::CommonName);
    }

    #[test]
    fn test_tls_config_node_id_verification_default() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        assert_eq!(tls.node_id_verification, NodeIdVerification::None);
    }

    #[test]
    fn test_node_id_verification_serde() {
        // Use bincode for serialization since serde_json isn't a dependency
        let cn = NodeIdVerification::CommonName;
        let serialized = bitcode::serialize(&cn).unwrap();
        let deserialized: NodeIdVerification = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, NodeIdVerification::CommonName);
    }

    #[test]
    fn test_security_mode_strict_default() {
        let mode = SecurityMode::default();
        assert_eq!(mode, SecurityMode::Strict);
    }

    #[test]
    fn test_security_mode_requires_tls() {
        assert!(SecurityMode::Strict.requires_tls());
        assert!(SecurityMode::Permissive.requires_tls());
        assert!(!SecurityMode::Development.requires_tls());
        assert!(!SecurityMode::Legacy.requires_tls());
    }

    #[test]
    fn test_security_mode_requires_mtls() {
        assert!(SecurityMode::Strict.requires_mtls());
        assert!(!SecurityMode::Permissive.requires_mtls());
        assert!(!SecurityMode::Development.requires_mtls());
        assert!(!SecurityMode::Legacy.requires_mtls());
    }

    #[test]
    fn test_security_mode_requires_node_id_verification() {
        assert!(SecurityMode::Strict.requires_node_id_verification());
        assert!(!SecurityMode::Permissive.requires_node_id_verification());
        assert!(!SecurityMode::Development.requires_node_id_verification());
        assert!(!SecurityMode::Legacy.requires_node_id_verification());
    }

    #[test]
    fn test_security_mode_should_warn() {
        assert!(!SecurityMode::Strict.should_warn());
        assert!(!SecurityMode::Permissive.should_warn());
        assert!(SecurityMode::Development.should_warn());
        assert!(SecurityMode::Legacy.should_warn());
    }

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert_eq!(config.mode, SecurityMode::Strict);
        assert!(config.warn_on_insecure);
    }

    #[test]
    fn test_security_config_constructors() {
        let strict = SecurityConfig::strict();
        assert_eq!(strict.mode, SecurityMode::Strict);

        let permissive = SecurityConfig::permissive();
        assert_eq!(permissive.mode, SecurityMode::Permissive);

        let development = SecurityConfig::development();
        assert_eq!(development.mode, SecurityMode::Development);

        let legacy = SecurityConfig::legacy();
        assert_eq!(legacy.mode, SecurityMode::Legacy);
    }

    #[test]
    fn test_security_config_without_warnings() {
        let config = SecurityConfig::development().without_warnings();
        assert!(!config.warn_on_insecure);
    }

    #[test]
    fn test_tls_config_mtls_default() {
        // Ensure default TlsConfig has require_client_auth = false
        // (for backward compatibility with existing configs)
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        assert!(!tls.require_client_auth);
    }

    #[test]
    fn test_tls_config_new_secure() {
        let tls = TlsConfig::new_secure("/cert.pem", "/key.pem", "/ca.pem");

        assert!(tls.require_client_auth);
        assert!(tls.ca_cert_path.is_some());
        assert_eq!(tls.node_id_verification, NodeIdVerification::CommonName);
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_validate_security_fails_without_tls() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Strict);
        // No TLS configured but Strict mode requires it
        let result = config.validate_security();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_security_fails_without_mtls() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        // TLS is configured but require_client_auth is false
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_security_mode(SecurityMode::Strict);

        let result = config.validate_security();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_security_fails_without_node_id_verification() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem")
            .with_ca_cert("/ca.pem")
            .with_client_auth();
        // TLS + mTLS configured but NodeIdVerification is None
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_security_mode(SecurityMode::Strict);

        let result = config.validate_security();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_security_strict_mode_passes() {
        let tls = TlsConfig::new_secure("/cert.pem", "/key.pem", "/ca.pem");
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_security_mode(SecurityMode::Strict);

        let result = config.validate_security();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_security_permissive_mode() {
        let tls = TlsConfig::new("/cert.pem", "/key.pem");
        // Permissive only requires TLS, not mTLS
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_security_mode(SecurityMode::Permissive);

        let result = config.validate_security();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_security_development_mode() {
        // Development mode doesn't require TLS
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security(SecurityConfig::development().without_warnings());

        let result = config.validate_security();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_security_legacy_mode_backward_compatible() {
        // Legacy mode should allow no TLS for backward compatibility
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security(SecurityConfig::legacy().without_warnings());

        let result = config.validate_security();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_with_security() {
        let security = SecurityConfig::permissive();
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security(security);

        assert_eq!(config.security.mode, SecurityMode::Permissive);
    }

    #[test]
    fn test_config_with_security_mode() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Development);

        assert_eq!(config.security.mode, SecurityMode::Development);
    }

    #[test]
    fn test_config_is_secure() {
        // Without TLS, is_secure returns false
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Strict);
        assert!(!config.is_secure());

        // With proper TLS config, is_secure returns true
        let tls = TlsConfig::new_secure("/cert.pem", "/key.pem", "/ca.pem");
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_tls(tls)
            .with_security_mode(SecurityMode::Strict);
        assert!(config.is_secure());

        // Development mode is never secure for production
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_security_mode(SecurityMode::Development);
        assert!(!config.is_secure());
    }

    #[test]
    fn test_security_mode_serde() {
        let mode = SecurityMode::Strict;
        let serialized = bitcode::serialize(&mode).unwrap();
        let deserialized: SecurityMode = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, SecurityMode::Strict);

        let mode = SecurityMode::Development;
        let serialized = bitcode::serialize(&mode).unwrap();
        let deserialized: SecurityMode = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, SecurityMode::Development);
    }

    #[test]
    fn test_security_config_serde() {
        let config = SecurityConfig::permissive();
        let serialized = bitcode::serialize(&config).unwrap();
        let deserialized: SecurityConfig = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.mode, SecurityMode::Permissive);
        assert!(deserialized.warn_on_insecure);
    }
}
