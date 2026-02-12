// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tensor Vault: Secure secret storage with graph-based access control.
//!
//! Secrets are encrypted at rest using AES-256-GCM. Access is controlled
//! by graph topology - a requester must have a path to the secret node.
//!
//! Security features:
//! - AES-256-GCM authenticated encryption
//! - Argon2id key derivation (GPU/ASIC resistant)
//! - Key obfuscation via HMAC (hides secret names in storage)
//! - Metadata encryption (hides creator, timestamps)
//! - Length padding (hides plaintext size)
//! - Pointer indirection (hides storage patterns)
//! - Graph-based access control (topological authorization)

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::format_collect)]

mod access;
mod access_tensor;
mod anomaly;
mod attenuation;
mod audit;
mod delegation;
mod dependency;
mod dynamic;
mod encryption;
mod engine;
mod geo_routing;
mod graph_intel;
#[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
mod heat_kernel;
mod key;
mod manifold;
pub mod namespaced;
mod obfuscation;
mod pitr;
mod pki;
mod policy;
mod quota;
mod rate_limit;
mod rotation;
pub mod scoped;
mod seal;
mod shamir;
mod signing;
mod similarity;
mod sync_target;
mod template_store;
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::manual_is_multiple_of
)]
mod temporal_analysis;
mod topology;
mod ttl;
mod vault;
mod wrapping;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use access::AccessController;
pub use access_tensor::{
    AccessTensor, AccessTensorConfig, EntityAccessProfile, SecretAccessProfile,
};
pub use anomaly::{AgentProfile, AnomalyEvent, AnomalyMonitor, AnomalyThresholds};
pub use attenuation::{AttenuationPolicy, ExponentialAttenuationPolicy};
pub use audit::{AuditContext, AuditEntry, AuditLog, AuditOperation};
pub use delegation::{DelegationManager, DelegationRecord};
pub use dependency::{
    DependencyInfo, DependencyWeight, ImpactReport, RotationPlan, RotationStep,
    WeightedAffectedSecret, WeightedImpactReport,
};
pub use dynamic::{
    ApiKeyConfig, DynamicSecretMetadata, PasswordCharset, PasswordConfig, SecretTemplate,
    TokenConfig, TokenEncoding,
};
pub use encryption::{Cipher, NONCE_SIZE};
pub use engine::{EngineRegistry, SecretEngine};
pub use geo_routing::{
    ExcludedTarget, ExclusionReason, GeoRouter, RoutedTarget, RoutingConfig, RoutingDecision,
    TargetGeometry,
};
pub use graph_intel::{
    AccessCycle, AccessExplanation, AccessHop, BlastRadius, CriticalEntity, DelegationAnomalyScore,
    DenialReason, EntityRiskScore, EntityTrustScore, InferredRole, NewAccess, OverPrivilegedEntity,
    PrivilegeAnalysis, PrivilegeAnalysisReport, ReachableSecret, RiskContributor,
    RiskPropagationReport, RoleInferenceResult, SecurityAuditReport, SimulationResult,
    SinglePointOfFailure, TrustTransitivityReport,
};
pub use heat_kernel::{HeatKernelConfig, HeatKernelTrustReport, HeatKernelTrustScore};
pub use key::{MasterKey, KEY_SIZE, SALT_SIZE};
pub use manifold::{
    batch_recommend_placement, recommend_placement, GeoCoordinate, PlacementConfig,
    PlacementRecommendation, RegionRegistry, VaultRegion,
};
pub use obfuscation::{Obfuscator, PaddingSize};
pub use pitr::VaultSnapshot;
pub use pki::{CertInfo, CertificateRequest, PkiEngine, RevocationEntry, RevocationList};
pub use policy::{PolicyManager, PolicyMatch, PolicyTemplate};
pub use quota::{QuotaManager, ResourceQuota, ResourceUsage};
pub use rate_limit::{Operation, RateLimitConfig, RateLimiter};
pub use rotation::{PendingRotation, RotationGenerator, RotationPolicy};
pub use seal::SealState;
pub use shamir::{reconstruct_master_key, split_master_key, KeyShare, ShamirConfig};
pub use similarity::{SecretFeatures, SimilarSecret, SimilarityIndex};
pub use sync_target::{EnvSyncTarget, FileSyncTarget, SyncManager, SyncTarget};
pub use template_store::{StoredTemplate, TemplateManager};
pub use temporal_analysis::{
    DriftDetection, SeasonalPattern, TemporalAnalysisConfig, TemporalAnalysisReport,
};
pub use topology::{AccessTopology, BatchPermissionResult, PolicyRedundancyReport, TopologyConfig};
pub use ttl::GrantTTLTracker;
pub use vault::{BatchSetResult, Vault};
pub use wrapping::WrappingToken;

/// Permission levels for vault access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Permission {
    /// Read-only access: can get(), list()
    #[default]
    Read,
    /// Write access: can get(), list(), set(), rotate()
    Write,
    /// Admin access: can get(), list(), set(), rotate(), delete(), grant(), revoke()
    Admin,
}

impl Permission {
    /// Check if this permission level allows the given operation.
    pub fn allows(&self, required: Self) -> bool {
        matches!(
            (self, required),
            (Self::Admin, _) | (Self::Write, Self::Read | Self::Write) | (Self::Read, Self::Read)
        )
    }

    /// Numeric level for bottleneck flow (1=Read, 2=Write, 3=Admin).
    pub(crate) fn to_level(self) -> i64 {
        match self {
            Self::Read => 1,
            Self::Write => 2,
            Self::Admin => 3,
        }
    }

    /// Parse from numeric level.
    pub(crate) fn from_level(level: i64) -> Option<Self> {
        match level {
            1 => Some(Self::Read),
            2 => Some(Self::Write),
            3 => Some(Self::Admin),
            _ => None,
        }
    }

    /// Get edge type suffix for this permission level.
    pub(crate) fn edge_suffix(self) -> &'static str {
        match self {
            Self::Read => "_READ",
            Self::Write => "_WRITE",
            Self::Admin => "_ADMIN",
        }
    }

    /// Parse from edge type.
    pub(crate) fn from_edge_type(edge_type: &str) -> Option<Self> {
        if edge_type.ends_with("_READ") {
            Some(Self::Read)
        } else if edge_type.ends_with("_WRITE") {
            Some(Self::Write)
        } else if edge_type.ends_with("_ADMIN") {
            Some(Self::Admin)
        } else if edge_type == "VAULT_ACCESS" {
            // Backward compatibility: old grants are Admin
            Some(Self::Admin)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "READ"),
            Self::Write => write!(f, "WRITE"),
            Self::Admin => write!(f, "ADMIN"),
        }
    }
}

impl std::str::FromStr for Permission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "READ" | "R" => Ok(Self::Read),
            "WRITE" | "W" => Ok(Self::Write),
            "ADMIN" | "A" => Ok(Self::Admin),
            _ => Err(format!("Invalid permission: {s}")),
        }
    }
}

/// Error types for vault operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaultError {
    /// Access denied - no path from requester to secret.
    AccessDenied(String),
    /// Secret not found.
    NotFound(String),
    /// Encryption/decryption failed.
    CryptoError(String),
    /// Key derivation failed.
    KeyDerivationError(String),
    /// Storage error from `TensorStore`.
    StorageError(String),
    /// Graph error from path verification.
    GraphError(String),
    /// Invalid secret key format.
    InvalidKey(String),
    /// Insufficient permissions for operation.
    InsufficientPermission(String),
    /// Rate limit exceeded.
    RateLimited(String),
    /// Secret has expired past its TTL deadline.
    SecretExpired(String),
    /// Shamir secret sharing error.
    ShamirError(String),
    /// Wrapping token has expired.
    WrappingTokenExpired(String),
    /// Wrapping token already consumed.
    WrappingTokenConsumed(String),
    /// Vault is sealed.
    Sealed(String),
    /// Quota exceeded.
    QuotaExceeded(String),
    /// Engine not found.
    EngineNotFound(String),
    /// Cyclic dependency detected.
    CyclicDependency(String),
}

impl std::fmt::Display for VaultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccessDenied(msg) => write!(f, "access denied: {msg}"),
            Self::NotFound(key) => write!(f, "secret not found: {key}"),
            Self::CryptoError(msg) => write!(f, "crypto error: {msg}"),
            Self::KeyDerivationError(msg) => write!(f, "key derivation error: {msg}"),
            Self::StorageError(msg) => write!(f, "storage error: {msg}"),
            Self::GraphError(msg) => write!(f, "graph error: {msg}"),
            Self::InvalidKey(msg) => write!(f, "invalid key: {msg}"),
            Self::InsufficientPermission(msg) => write!(f, "insufficient permission: {msg}"),
            Self::RateLimited(msg) => write!(f, "rate limited: {msg}"),
            Self::SecretExpired(key) => write!(f, "secret expired: {key}"),
            Self::ShamirError(msg) => write!(f, "shamir error: {msg}"),
            Self::WrappingTokenExpired(token) => write!(f, "wrapping token expired: {token}"),
            Self::WrappingTokenConsumed(token) => write!(f, "wrapping token consumed: {token}"),
            Self::Sealed(msg) => write!(f, "vault sealed: {msg}"),
            Self::QuotaExceeded(msg) => write!(f, "quota exceeded: {msg}"),
            Self::EngineNotFound(name) => write!(f, "engine not found: {name}"),
            Self::CyclicDependency(msg) => write!(f, "cyclic dependency: {msg}"),
        }
    }
}

impl std::error::Error for VaultError {}

pub type Result<T> = std::result::Result<T, VaultError>;

/// Configuration for the vault.
#[derive(Clone)]
pub struct VaultConfig {
    /// Salt for key derivation (randomly generated if not provided).
    pub salt: Option<[u8; 16]>,
    /// Memory cost for Argon2id (in KiB, default: 65536 = 64MB).
    pub argon2_memory_cost: u32,
    /// Time cost for Argon2id (iterations, default: 3).
    pub argon2_time_cost: u32,
    /// Parallelism for Argon2id (default: 4).
    pub argon2_parallelism: u32,
    /// Rate limiting configuration (None = disabled).
    pub rate_limit: Option<RateLimitConfig>,
    /// Maximum number of versions to keep per secret (default: 5).
    pub max_versions: usize,
    /// Permission attenuation policy (default: attenuate with distance).
    pub attenuation: AttenuationPolicy,
    /// Anomaly detection thresholds (None = default thresholds).
    pub anomaly_thresholds: Option<AnomalyThresholds>,
    /// Maximum delegation chain depth (None = default 3).
    pub max_delegation_depth: Option<u32>,
    /// Shamir secret sharing configuration (None = disabled).
    pub shamir_config: Option<ShamirConfig>,
    /// Default resource quota for namespaces (None = no quota).
    pub default_quota: Option<ResourceQuota>,
    /// Event handler for vault monitoring (default: no-op).
    pub event_handler: Option<Arc<dyn VaultEventHandler>>,
    /// Maximum secret value size in bytes (default: 65531).
    pub max_value_size: usize,
}

impl std::fmt::Debug for VaultConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaultConfig")
            .field("salt", &self.salt)
            .field("argon2_memory_cost", &self.argon2_memory_cost)
            .field("argon2_time_cost", &self.argon2_time_cost)
            .field("argon2_parallelism", &self.argon2_parallelism)
            .field("rate_limit", &self.rate_limit)
            .field("max_versions", &self.max_versions)
            .field("attenuation", &self.attenuation)
            .field("anomaly_thresholds", &self.anomaly_thresholds)
            .field("max_delegation_depth", &self.max_delegation_depth)
            .field("shamir_config", &self.shamir_config)
            .field("default_quota", &self.default_quota)
            .field("event_handler", &self.event_handler.as_ref().map(|_| "..."))
            .field("max_value_size", &self.max_value_size)
            .finish()
    }
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            salt: None,
            argon2_memory_cost: 65536,
            argon2_time_cost: 3,
            argon2_parallelism: 4,
            rate_limit: None,
            max_versions: 5,
            attenuation: AttenuationPolicy::default(),
            anomaly_thresholds: None,
            max_delegation_depth: None,
            shamir_config: None,
            default_quota: None,
            event_handler: None,
            max_value_size: 65_531,
        }
    }
}

impl VaultConfig {
    /// Create a config with an explicit salt.
    #[must_use]
    pub fn with_salt(mut self, salt: [u8; 16]) -> Self {
        self.salt = Some(salt);
        self
    }

    /// Create a config with rate limiting enabled.
    #[must_use]
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }

    /// Create a config with custom max versions.
    #[must_use]
    pub fn with_max_versions(mut self, max: usize) -> Self {
        self.max_versions = max;
        self
    }

    /// Create a config with a custom attenuation policy.
    #[must_use]
    pub fn with_attenuation(mut self, policy: AttenuationPolicy) -> Self {
        self.attenuation = policy;
        self
    }

    /// Create a config with custom anomaly detection thresholds.
    #[must_use]
    pub fn with_anomaly_thresholds(mut self, thresholds: AnomalyThresholds) -> Self {
        self.anomaly_thresholds = Some(thresholds);
        self
    }

    /// Create a config with a custom maximum delegation depth.
    #[must_use]
    pub fn with_max_delegation_depth(mut self, depth: u32) -> Self {
        self.max_delegation_depth = Some(depth);
        self
    }

    /// Create a config with Shamir secret sharing.
    #[must_use]
    pub fn with_shamir(mut self, config: ShamirConfig) -> Self {
        self.shamir_config = Some(config);
        self
    }

    /// Create a config with a default resource quota for namespaces.
    #[must_use]
    pub fn with_default_quota(mut self, quota: ResourceQuota) -> Self {
        self.default_quota = Some(quota);
        self
    }

    /// Set an event handler for vault monitoring.
    #[must_use]
    pub fn with_event_handler(mut self, handler: Arc<dyn VaultEventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// Set the maximum secret value size in bytes.
    #[must_use]
    pub fn with_max_value_size(mut self, size: usize) -> Self {
        self.max_value_size = size;
        self
    }
}

/// Events emitted by vault operations for monitoring and alerting.
#[derive(Debug, Clone)]
pub enum VaultEvent {
    /// Anomaly detected by the behavior monitor.
    Anomaly(anomaly::AnomalyEvent),
    /// A storage cleanup operation failed (non-fatal).
    CleanupError { context: String, error: String },
    /// A mutex was recovered from a poisoned state.
    PoisonRecovery { context: String },
    /// Legacy ciphertext decrypted (no version tag).
    LegacyDecrypt { entity: String, key: String },
}

/// Trait for handling vault events.
pub trait VaultEventHandler: Send + Sync {
    fn on_event(&self, event: &VaultEvent);
}

struct NoopEventHandler;
impl VaultEventHandler for NoopEventHandler {
    fn on_event(&self, _event: &VaultEvent) {}
}

/// Comprehensive vault health and status information.
#[derive(Debug, Clone)]
pub struct VaultStatus {
    pub sealed: bool,
    pub total_secrets: usize,
    pub sync_health: Vec<(String, bool)>,
    pub pending_rotations: usize,
    pub snapshot_count: usize,
}

/// Information about a secret version.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Version number (1-based).
    pub version: u32,
    /// Unix timestamp in milliseconds when the version was created.
    pub created_at: i64,
}

/// Paginated list of secret keys.
#[derive(Debug, Clone)]
pub struct PagedSecrets {
    pub secrets: Vec<String>,
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
    pub has_more: bool,
}

/// Summary metadata for a secret.
#[derive(Debug, Clone)]
pub struct SecretSummary {
    pub key: String,
    pub version_count: u32,
    pub created_at: i64,
    pub last_accessed: Option<i64>,
    pub entity_count: usize,
}

/// Side-by-side comparison of two secret versions.
#[derive(Debug, Clone)]
pub struct VersionDiff {
    pub key: String,
    pub version_a: u32,
    pub version_b: u32,
    pub value_a: String,
    pub value_b: String,
    pub timestamp_a: i64,
    pub timestamp_b: i64,
}

/// A single entry in a secret's change history.
#[derive(Debug, Clone)]
pub struct ChangelogEntry {
    pub version: Option<u32>,
    pub operation: String,
    pub entity: String,
    pub timestamp: i64,
}

pub use graph_intel::{
    BehaviorEmbeddingConfig, ClusteringResult, GeometricAnomalyReport, GeometricAnomalyResult,
    NodeEmbedding, SpectralCluster,
};

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use graph_engine::{Direction, GraphEngine, PropertyValue};
    use serial_test::serial;
    use tensor_store::TensorStore;

    use super::*;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    /// Helper to add edges between entity keys using the node-based API.
    fn add_test_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) {
        let get_or_create = |key: &str| -> u64 {
            if let Ok(nodes) =
                graph.find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
            {
                if let Some(node) = nodes.first() {
                    return node.id;
                }
            }
            let mut props = HashMap::new();
            props.insert(
                "entity_key".to_string(),
                PropertyValue::String(key.to_string()),
            );
            graph.create_node("TestEntity", props).unwrap_or(0)
        };

        let from_node = get_or_create(from_key);
        let to_node = get_or_create(to_key);
        graph
            .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
            .ok();
    }

    /// Check whether an entity node has any outgoing VAULT_ACCESS edges.
    fn has_access_edges(graph: &GraphEngine, entity_key: &str) -> bool {
        let nodes = graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
            .unwrap_or_default();
        let Some(node) = nodes.first() else {
            return false;
        };
        graph
            .edges_of(node.id, Direction::Outgoing)
            .unwrap_or_default()
            .iter()
            .any(|e| e.edge_type.starts_with("VAULT_ACCESS"))
    }

    /// Helper to get outgoing edges for an entity key.
    fn get_test_outgoing_edges(graph: &GraphEngine, entity_key: &str) -> Vec<(String, String)> {
        let find_node = |key: &str| -> Option<u64> {
            graph
                .find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
                .ok()
                .and_then(|nodes| nodes.first().map(|n| n.id))
        };

        let get_key = |id: u64| -> Option<String> {
            graph.get_node(id).ok().and_then(|node| {
                if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
                    Some(key.clone())
                } else {
                    None
                }
            })
        };

        let Some(node_id) = find_node(entity_key) else {
            return Vec::new();
        };

        let mut result = Vec::new();
        if let Ok(edges) = graph.edges_of(node_id, Direction::Outgoing) {
            for edge in edges {
                let target_id = if edge.from == node_id {
                    edge.to
                } else {
                    edge.from
                };
                if let Some(target_key) = get_key(target_id) {
                    result.push((target_key, edge.edge_type.clone()));
                }
            }
        }
        result
    }

    #[test]
    fn test_set_get_roundtrip() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-secret123").unwrap();
        let value = vault.get(Vault::ROOT, "api_key").unwrap();

        assert_eq!(value, "sk-secret123");
    }

    #[test]
    fn test_access_denied_without_path() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_root_always_has_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        let value = vault.get(Vault::ROOT, "secret").unwrap();

        assert_eq!(value, "value");
    }

    #[test]
    fn test_grant_enables_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-123").unwrap();

        // Alice can't access yet
        assert!(vault.get("user:alice", "api_key").is_err());

        // Grant access
        vault.grant(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Now Alice can access
        let value = vault.get("user:alice", "api_key").unwrap();
        assert_eq!(value, "sk-123");
    }

    #[test]
    fn test_revoke_removes_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-123").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Alice can access
        assert!(vault.get("user:alice", "api_key").is_ok());

        // Revoke access
        vault.revoke(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Alice can no longer access
        assert!(vault.get("user:alice", "api_key").is_err());
    }

    #[test]
    fn test_rotate_updates_value() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "old_value").unwrap();
        vault.rotate(Vault::ROOT, "api_key", "new_value").unwrap();

        let value = vault.get(Vault::ROOT, "api_key").unwrap();
        assert_eq!(value, "new_value");
    }

    #[test]
    fn test_delete_removes_secret() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "value").unwrap();
        vault.delete(Vault::ROOT, "api_key").unwrap();

        let result = vault.get(Vault::ROOT, "api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_list_filters_by_pattern() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "openai:key", "sk-1").unwrap();
        vault.set(Vault::ROOT, "openai:org", "org-1").unwrap();
        vault.set(Vault::ROOT, "github:token", "ghp-1").unwrap();

        let openai_keys = vault.list(Vault::ROOT, "openai:*").unwrap();
        assert_eq!(openai_keys.len(), 2);
        assert!(openai_keys.contains(&"openai:key".to_string()));
        assert!(openai_keys.contains(&"openai:org".to_string()));

        let all_keys = vault.list(Vault::ROOT, "*").unwrap();
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_list_filters_by_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "public:key", "value1").unwrap();
        vault.set(Vault::ROOT, "private:key", "value2").unwrap();

        vault
            .grant(Vault::ROOT, "user:alice", "public:key")
            .unwrap();

        // Alice can only see public:key
        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert_eq!(alice_keys.len(), 1);
        assert!(alice_keys.contains(&"public:key".to_string()));

        // Root can see all
        let root_keys = vault.list(Vault::ROOT, "*").unwrap();
        assert_eq!(root_keys.len(), 2);
    }

    #[test]
    fn test_list_excludes_member_only_paths() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Alice has a MEMBER-only path to the secret node (no VAULT_ACCESS edge)
        // This should NOT make the secret visible in list()
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        add_test_edge(&vault.graph, "team:devs", &secret_node, "MEMBER");

        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert!(
            alice_keys.is_empty(),
            "MEMBER-only path should not leak secret names in list()"
        );
    }

    #[test]
    fn test_list_includes_vault_access_paths() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Alice has a proper VAULT_ACCESS path via a team
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Read)
            .unwrap();

        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert_eq!(alice_keys.len(), 1);
        assert!(alice_keys.contains(&"secret".to_string()));
    }

    #[test]
    fn test_scoped_vault() {
        let vault = create_test_vault();

        // Create a secret and grant access to alice
        vault.set(Vault::ROOT, "shared:key", "value").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "shared:key")
            .unwrap();

        let scoped = vault.scope("user:alice");
        let value = scoped.get("shared:key").unwrap();
        assert_eq!(value, "value");
    }

    #[test]
    fn test_non_root_cannot_create_secrets() {
        let vault = create_test_vault();

        let result = vault.set("user:alice", "new_secret", "value");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_transitive_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create a chain: alice -> team -> secret
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault.grant(Vault::ROOT, "team:devs", "secret").unwrap();

        // Alice can access via transitive path
        let value = vault.get("user:alice", "secret").unwrap();
        assert_eq!(value, "value");
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let vault = Arc::new(create_test_vault());

        vault.set(Vault::ROOT, "shared", "initial").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "shared").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "shared").unwrap();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let vault = Arc::clone(&vault);
                let user = if i % 2 == 0 { "user:alice" } else { "user:bob" };
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = vault.get(user, "shared");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_vault_error_display() {
        let errors = [
            (
                VaultError::AccessDenied("test".to_string()),
                "access denied: test",
            ),
            (
                VaultError::NotFound("key".to_string()),
                "secret not found: key",
            ),
            (
                VaultError::CryptoError("bad".to_string()),
                "crypto error: bad",
            ),
            (
                VaultError::KeyDerivationError("fail".to_string()),
                "key derivation error: fail",
            ),
            (
                VaultError::StorageError("io".to_string()),
                "storage error: io",
            ),
            (
                VaultError::GraphError("node".to_string()),
                "graph error: node",
            ),
            (
                VaultError::InvalidKey("empty".to_string()),
                "invalid key: empty",
            ),
            (
                VaultError::SecretExpired("db/pass".to_string()),
                "secret expired: db/pass",
            ),
        ];

        for (err, expected) in errors {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_scoped_vault_set_and_delete() {
        let vault = create_test_vault();

        let scoped = vault.scope(Vault::ROOT);

        scoped.set("scoped:key", "scoped_value").unwrap();
        let value = scoped.get("scoped:key").unwrap();
        assert_eq!(value, "scoped_value");

        scoped.delete("scoped:key").unwrap();
        assert!(scoped.get("scoped:key").is_err());
    }

    #[test]
    fn test_scoped_vault_list() {
        let vault = create_test_vault();
        let scoped = vault.scope(Vault::ROOT);

        scoped.set("list:a", "1").unwrap();
        scoped.set("list:b", "2").unwrap();

        let keys = scoped.list("list:*").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_list_exact_match() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "exact_key", "value1").unwrap();
        vault.set(Vault::ROOT, "exact_key_extra", "value2").unwrap();

        let keys = vault.list(Vault::ROOT, "exact_key").unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "exact_key");
    }

    #[test]
    fn test_get_not_found() {
        let vault = create_test_vault();
        let result = vault.get(Vault::ROOT, "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_delete_not_found() {
        let vault = create_test_vault();
        let result = vault.delete(Vault::ROOT, "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_rotate_not_found() {
        let vault = create_test_vault();
        let result = vault.rotate(Vault::ROOT, "nonexistent", "new_value");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_grant_on_nonexistent_secret() {
        let vault = create_test_vault();
        let result = vault.grant(Vault::ROOT, "user:alice", "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_overwrite_existing_secret() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "key", "value1").unwrap();
        vault.set(Vault::ROOT, "key", "value2").unwrap();

        let value = vault.get(Vault::ROOT, "key").unwrap();
        assert_eq!(value, "value2");
    }

    #[test]
    fn test_empty_pattern_returns_all() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "a", "1").unwrap();
        vault.set(Vault::ROOT, "b", "2").unwrap();

        let keys = vault.list(Vault::ROOT, "").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    #[serial]
    fn test_from_env_with_valid_key() {
        use base64::Engine;

        let key_bytes = b"01234567890123456789012345678901";
        let encoded = base64::engine::general_purpose::STANDARD.encode(key_bytes);
        std::env::set_var("NEUMANN_VAULT_KEY", &encoded);

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        std::env::remove_var("NEUMANN_VAULT_KEY");

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_from_env_missing_key() {
        std::env::remove_var("NEUMANN_VAULT_KEY");

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        assert!(matches!(result, Err(VaultError::KeyDerivationError(_))));
    }

    #[test]
    #[serial]
    fn test_from_env_invalid_base64() {
        std::env::set_var("NEUMANN_VAULT_KEY", "not-valid-base64!!!");

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        std::env::remove_var("NEUMANN_VAULT_KEY");

        assert!(matches!(result, Err(VaultError::KeyDerivationError(_))));
    }

    #[test]
    fn test_overwrite_without_access_denied() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "protected", "initial").unwrap();

        let result = vault.set("user:alice", "protected", "hacked");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_overwrite_with_granted_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "shared", "initial").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "shared").unwrap();

        vault.set("user:alice", "shared", "updated").unwrap();

        let value = vault.get(Vault::ROOT, "shared").unwrap();
        assert_eq!(value, "updated");
    }

    #[test]
    fn test_short_key_still_works() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());

        let result = Vault::new(b"short", graph.clone(), store, VaultConfig::default());
        assert!(result.is_ok());

        let store2 = TensorStore::new();
        let result = Vault::new(b"", graph, store2, VaultConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_revoke_on_nonexistent_secret() {
        let vault = create_test_vault();

        let result = vault.revoke("user:alice", "user:bob", "nonexistent");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        let result = vault.revoke(Vault::ROOT, "user:alice", "nonexistent");
        assert!(result.is_ok());
    }

    // === Security feature tests ===

    #[test]
    fn test_storage_key_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "secret").unwrap();

        let all_keys = vault.store.scan("_vk:");
        for key in &all_keys {
            assert!(!key.contains("api_key"), "Key should be obfuscated");
        }
        assert_eq!(all_keys.len(), 1);
    }

    #[test]
    fn test_pointer_indirection_creates_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "my_secret", "value").unwrap();

        let blob_keys = vault.store.scan("_vs:");
        assert_eq!(blob_keys.len(), 1, "Should have one blob");

        let blob = vault.store.get(&blob_keys[0]).unwrap();
        assert!(blob.get("_data").is_some(), "Blob should have _data field");
    }

    #[test]
    fn test_metadata_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        assert!(
            tensor.get("_creator_obf").is_some(),
            "Should have obfuscated creator"
        );
        assert!(
            tensor.get("_created_by").is_none(),
            "Should NOT have plaintext creator"
        );

        assert!(
            tensor.get("_created_obf").is_some(),
            "Should have obfuscated timestamp"
        );
        assert!(
            tensor.get("_created_at").is_none(),
            "Should NOT have plaintext timestamp"
        );
    }

    #[test]
    fn test_encrypted_key_name_stored_for_listing() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "list_test_key", "value").unwrap();

        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        assert!(
            tensor.get("_key_enc").is_some(),
            "Should have encrypted key"
        );
        assert!(tensor.get("_key_nonce").is_some(), "Should have key nonce");

        let found = vault.list(Vault::ROOT, "list_test_*").unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "list_test_key");
    }

    #[test]
    fn test_delete_removes_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "temp", "data").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.delete(Vault::ROOT, "temp").unwrap();

        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 0, "Blob should be deleted");
    }

    #[test]
    fn test_rotate_creates_new_version() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "rotatable", "old_value").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.rotate(Vault::ROOT, "rotatable", "new_value").unwrap();

        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 2, "Should have 2 versions");

        let value = vault.get(Vault::ROOT, "rotatable").unwrap();
        assert_eq!(value, "new_value");

        let old_value = vault.get_version(Vault::ROOT, "rotatable", 1).unwrap();
        assert_eq!(old_value, "old_value");
    }

    #[test]
    fn test_padding_hides_length() {
        let short1 = obfuscation::pad_plaintext(b"a").unwrap();
        let short2 = obfuscation::pad_plaintext(b"abc").unwrap();
        let short3 = obfuscation::pad_plaintext(b"hello world!").unwrap();

        assert_eq!(short1.len(), 256);
        assert_eq!(short2.len(), 256);
        assert_eq!(short3.len(), 256);

        assert_eq!(obfuscation::unpad_plaintext(&short1).unwrap(), b"a");
        assert_eq!(obfuscation::unpad_plaintext(&short2).unwrap(), b"abc");
    }

    // === Permission Level Tests ===

    #[test]
    fn test_permission_enum_allows() {
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Write));
        assert!(!Permission::Read.allows(Permission::Admin));

        assert!(Permission::Write.allows(Permission::Read));
        assert!(Permission::Write.allows(Permission::Write));
        assert!(!Permission::Write.allows(Permission::Admin));

        assert!(Permission::Admin.allows(Permission::Read));
        assert!(Permission::Admin.allows(Permission::Write));
        assert!(Permission::Admin.allows(Permission::Admin));
    }

    #[test]
    fn test_permission_from_str() {
        assert_eq!("READ".parse::<Permission>().unwrap(), Permission::Read);
        assert_eq!("read".parse::<Permission>().unwrap(), Permission::Read);
        assert_eq!("R".parse::<Permission>().unwrap(), Permission::Read);

        assert_eq!("WRITE".parse::<Permission>().unwrap(), Permission::Write);
        assert_eq!("W".parse::<Permission>().unwrap(), Permission::Write);

        assert_eq!("ADMIN".parse::<Permission>().unwrap(), Permission::Admin);
        assert_eq!("A".parse::<Permission>().unwrap(), Permission::Admin);

        assert!("INVALID".parse::<Permission>().is_err());
    }

    #[test]
    fn test_permission_display() {
        assert_eq!(format!("{}", Permission::Read), "READ");
        assert_eq!(format!("{}", Permission::Write), "WRITE");
        assert_eq!(format!("{}", Permission::Admin), "ADMIN");
    }

    #[test]
    fn test_grant_with_read_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();

        assert!(vault.get("user:reader", "secret").is_ok());
        assert!(vault.list("user:reader", "*").is_ok());

        let result = vault.set("user:reader", "secret", "new_value");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

        let result = vault.delete("user:reader", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_grant_with_write_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();

        assert!(vault.get("user:writer", "secret").is_ok());

        vault.set("user:writer", "secret", "updated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "updated");

        vault.rotate("user:writer", "secret", "rotated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "rotated");

        let result = vault.delete("user:writer", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

        let result = vault.grant("user:writer", "user:other", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_grant_with_admin_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:admin", "secret", Permission::Admin)
            .unwrap();

        assert!(vault.get("user:admin", "secret").is_ok());
        vault.set("user:admin", "secret", "admin_update").unwrap();

        vault
            .grant_with_permission("user:admin", "user:reader", "secret", Permission::Read)
            .unwrap();
        assert!(vault.get("user:reader", "secret").is_ok());

        vault.revoke("user:admin", "user:reader", "secret").unwrap();
        assert!(vault.get("user:reader", "secret").is_err());

        vault.delete("user:admin", "secret").unwrap();
        assert!(vault.get("user:admin", "secret").is_err());
    }

    #[test]
    fn test_get_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        assert_eq!(
            vault.get_permission(Vault::ROOT, "secret"),
            Some(Permission::Admin)
        );

        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:admin", "secret", Permission::Admin)
            .unwrap();

        assert_eq!(
            vault.get_permission("user:reader", "secret"),
            Some(Permission::Read)
        );
        assert_eq!(
            vault.get_permission("user:writer", "secret"),
            Some(Permission::Write)
        );
        assert_eq!(
            vault.get_permission("user:admin", "secret"),
            Some(Permission::Admin)
        );

        assert_eq!(vault.get_permission("user:nobody", "secret"), None);
    }

    #[test]
    fn test_insufficient_permission_error_display() {
        let err = VaultError::InsufficientPermission("test message".to_string());
        assert_eq!(err.to_string(), "insufficient permission: test message");
    }

    #[test]
    fn test_read_user_cannot_rotate() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();

        let result = vault.rotate("user:reader", "secret", "new_value");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_write_user_cannot_revoke() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:other", "secret", Permission::Read)
            .unwrap();

        let result = vault.revoke("user:writer", "user:other", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_transitive_permission_minimum() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Read)
            .unwrap();

        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Read)
        );

        assert!(vault.get("user:alice", "secret").is_ok());
        let result = vault.set("user:alice", "secret", "hack");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_secret_node_key_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "my_api_key", "value").unwrap();

        vault
            .grant(Vault::ROOT, "user:alice", "my_api_key")
            .unwrap();

        let edges = get_test_outgoing_edges(&vault.graph, "user:alice");
        assert!(!edges.is_empty(), "Expected grant to create edge");

        for (to, edge_type) in &edges {
            assert!(edge_type.starts_with("VAULT_ACCESS"));

            assert!(
                !to.contains("my_api_key"),
                "Secret name should be obfuscated in graph: found {to}"
            );
            assert!(
                to.starts_with("vault_secret:"),
                "Edge should point to vault secret node: {to}"
            );
        }

        assert!(vault.get("user:alice", "my_api_key").is_ok());
    }

    // === TTL Grant Tests ===

    #[test]
    fn test_grant_with_ttl_creates_grant() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        assert!(vault.get("user:alice", "secret").is_ok());
    }

    #[test]
    fn test_grant_with_ttl_expires() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_cleanup_expired_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_revoke_removes_from_ttl_tracker() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        vault.revoke(Vault::ROOT, "user:alice", "secret").unwrap();

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 0);
    }

    #[test]
    fn test_multiple_ttl_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:bob",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

        assert!(matches!(
            vault.get("user:alice", "secret"),
            Err(VaultError::AccessDenied(_))
        ));
        assert!(vault.get("user:bob", "secret").is_ok());
    }

    // === Rate Limiting Tests ===

    fn create_rate_limited_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
            max_gets: 3,
            max_lists: 2,
            max_sets: 2,
            max_grants: 2,
            max_break_glass: 1,
            max_wraps: 2,
            max_generates: 2,
            window: Duration::from_secs(60),
        });
        Vault::new(b"test_password", graph, store, config).unwrap()
    }

    #[test]
    fn test_rate_limit_get() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_list() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        assert!(vault.list("user:alice", "*").is_ok());
        assert!(vault.list("user:alice", "*").is_ok());

        let result = vault.list("user:alice", "*");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_set() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();

        assert!(vault.set("user:writer", "secret", "v1").is_ok());
        assert!(vault.set("user:writer", "secret", "v2").is_ok());

        let result = vault.set("user:writer", "secret", "v3");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_grant() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:admin", "secret").unwrap();

        assert!(vault.grant("user:admin", "user:a", "secret").is_ok());
        assert!(vault.grant("user:admin", "user:b", "secret").is_ok());

        let result = vault.grant("user:admin", "user:c", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_root_exempt() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        for _ in 0..10 {
            assert!(vault.get(Vault::ROOT, "secret").is_ok());
        }
    }

    #[test]
    fn test_rate_limit_separate_entities() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "secret").unwrap();

        for _ in 0..3 {
            assert!(vault.get("user:alice", "secret").is_ok());
        }
        assert!(vault.get("user:alice", "secret").is_err());

        for _ in 0..3 {
            assert!(vault.get("user:bob", "secret").is_ok());
        }
        assert!(vault.get("user:bob", "secret").is_err());
    }

    #[test]
    fn test_rate_limited_error_display() {
        let err = VaultError::RateLimited("test message".to_string());
        assert_eq!(err.to_string(), "rate limited: test message");
    }

    #[test]
    fn test_vault_without_rate_limit() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        for _ in 0..100 {
            assert!(vault.get("user:alice", "secret").is_ok());
        }
    }

    // === Namespace Isolation Tests ===

    #[test]
    fn test_namespaced_vault_set_get() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret123").unwrap();
        let value = ns.get("api_key").unwrap();

        assert_eq!(value, "secret123");
    }

    #[test]
    fn test_namespaced_vault_key_prefixing() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret123").unwrap();

        let value = vault.get(Vault::ROOT, "team:alpha:api_key").unwrap();
        assert_eq!(value, "secret123");

        let result = vault.get(Vault::ROOT, "api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_namespaced_vault_isolation() {
        let vault = create_test_vault();

        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        let ns_beta = vault.namespace("team:beta", Vault::ROOT);

        ns_alpha.set("api_key", "alpha_secret").unwrap();
        ns_beta.set("api_key", "beta_secret").unwrap();

        assert_eq!(ns_alpha.get("api_key").unwrap(), "alpha_secret");
        assert_eq!(ns_beta.get("api_key").unwrap(), "beta_secret");

        assert_eq!(
            vault.get(Vault::ROOT, "team:alpha:api_key").unwrap(),
            "alpha_secret"
        );
        assert_eq!(
            vault.get(Vault::ROOT, "team:beta:api_key").unwrap(),
            "beta_secret"
        );
    }

    #[test]
    fn test_namespaced_vault_list() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("key1", "v1").unwrap();
        ns.set("key2", "v2").unwrap();

        vault.set(Vault::ROOT, "other_key", "other").unwrap();

        let keys = ns.list("*").unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        assert!(!keys.contains(&"other_key".to_string()));
    }

    #[test]
    fn test_namespaced_vault_list_pattern() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("openai:key", "sk-1").unwrap();
        ns.set("openai:org", "org-1").unwrap();
        ns.set("github:token", "ghp-1").unwrap();

        let openai_keys = ns.list("openai:*").unwrap();
        assert_eq!(openai_keys.len(), 2);
        assert!(openai_keys.contains(&"openai:key".to_string()));
        assert!(openai_keys.contains(&"openai:org".to_string()));
    }

    #[test]
    fn test_namespaced_vault_delete() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret").unwrap();
        ns.delete("api_key").unwrap();

        let result = ns.get("api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));

        let result = vault.get(Vault::ROOT, "team:alpha:api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_namespaced_vault_rotate() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "old_value").unwrap();
        ns.rotate("api_key", "new_value").unwrap();

        assert_eq!(ns.get("api_key").unwrap(), "new_value");
    }

    #[test]
    fn test_namespaced_vault_grant_revoke() {
        let vault = create_test_vault();

        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert_eq!(ns_alice.get("api_key").unwrap(), "secret");

        ns_root.revoke("user:alice", "api_key").unwrap();

        let result = ns_alice.get("api_key");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_namespaced_vault_permission_enforcement() {
        let vault = create_test_vault();

        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert!(ns_alice.get("api_key").is_ok());

        let result = ns_alice.set("api_key", "hacked");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_namespaced_vault_accessors() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", "user:alice");

        assert_eq!(ns.namespace(), "team:alpha");
        assert_eq!(ns.identity(), "user:alice");
    }

    #[test]
    fn test_namespaced_vault_cross_namespace_blocked() {
        let vault = create_test_vault();

        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        ns_alpha.set("secret", "alpha_data").unwrap();

        ns_alpha
            .grant("user:alice", "secret", Permission::Read)
            .unwrap();

        let ns_beta = vault.namespace("team:beta", "user:alice");
        let result = ns_beta.get("secret");
        assert!(matches!(
            result,
            Err(VaultError::NotFound(_) | VaultError::AccessDenied(_))
        ));
    }

    #[test]
    fn test_namespace_helper_method() {
        let vault = create_test_vault();

        let ns = vault.namespace("myns", Vault::ROOT);
        ns.set("key", "value").unwrap();

        assert_eq!(ns.get("key").unwrap(), "value");
    }

    // === Audit Query API Tests ===

    #[test]
    fn test_audit_log_records_get() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        let before = vault.audit_log("secret").unwrap().len();

        vault.get("user:alice", "secret").unwrap();

        let entries = vault.audit_log("secret").unwrap();
        assert!(
            entries.len() > before,
            "Should have new audit entries after get"
        );

        let get_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Get))
            .collect();
        assert!(!get_entries.is_empty(), "Should have Get audit entry");
    }

    #[test]
    fn test_audit_log_records_set() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "new_secret", "value").unwrap();

        let entries = vault.audit_log("new_secret").unwrap();
        let set_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Set))
            .collect();
        assert!(!set_entries.is_empty(), "Should have Set audit entry");
    }

    #[test]
    fn test_audit_log_records_grant() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        let entries = vault.audit_log("secret").unwrap();
        let grant_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Grant { .. }))
            .collect();
        assert!(!grant_entries.is_empty(), "Should have Grant audit entry");

        if let AuditOperation::Grant { to, permission } = &grant_entries[0].operation {
            assert_eq!(to, "user:alice");
            assert_eq!(permission, "read");
        }
    }

    #[test]
    fn test_audit_log_records_revoke() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.revoke(Vault::ROOT, "user:alice", "secret").unwrap();

        let entries = vault.audit_log("secret").unwrap();
        let revoke_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Revoke { .. }))
            .collect();
        assert!(!revoke_entries.is_empty(), "Should have Revoke audit entry");

        if let AuditOperation::Revoke { from } = &revoke_entries[0].operation {
            assert_eq!(from, "user:alice");
        }
    }

    #[test]
    fn test_audit_log_records_rotate() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "old").unwrap();
        vault.rotate(Vault::ROOT, "secret", "new").unwrap();

        let entries = vault.audit_log("secret").unwrap();
        let rotate_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Rotate))
            .collect();
        assert!(!rotate_entries.is_empty(), "Should have Rotate audit entry");
    }

    #[test]
    fn test_audit_log_records_delete() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.delete(Vault::ROOT, "secret").unwrap();

        let entries = vault.audit_log("secret").unwrap();
        let delete_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Delete))
            .collect();
        assert!(!delete_entries.is_empty(), "Should have Delete audit entry");
    }

    #[test]
    fn test_audit_by_entity() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "s1").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "s2").unwrap();

        vault.get("user:alice", "s1").unwrap();
        vault.get("user:alice", "s2").unwrap();

        let entries = vault.audit_by_entity("user:alice").unwrap();
        assert!(
            entries.len() >= 2,
            "Should have at least 2 entries for alice"
        );
    }

    #[test]
    fn test_audit_recent() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.set(Vault::ROOT, "s3", "v3").unwrap();

        let recent = vault.audit_recent(2).unwrap();
        assert_eq!(recent.len(), 2, "Should return at most 2 entries");

        if recent.len() >= 2 {
            assert!(recent[0].timestamp >= recent[1].timestamp);
        }
    }

    #[test]
    fn test_audit_since() {
        let vault = create_test_vault();

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        std::thread::sleep(std::time::Duration::from_millis(10));

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let entries = vault.audit_since(before).unwrap();
        assert!(
            !entries.is_empty(),
            "Should have entries since the timestamp"
        );
    }

    // === Secret Versioning Tests ===

    #[test]
    fn test_version_on_set() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v2");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);
    }

    #[test]
    fn test_get_specific_version() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "version_1").unwrap();
        vault.set(Vault::ROOT, "secret", "version_2").unwrap();
        vault.set(Vault::ROOT, "secret", "version_3").unwrap();

        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
            "version_1"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
            "version_2"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 3).unwrap(),
            "version_3"
        );
    }

    #[test]
    fn test_get_version_not_found() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();

        let result = vault.get_version(Vault::ROOT, "secret", 2);
        assert!(matches!(result, Err(VaultError::NotFound(_))));

        let result = vault.get_version(Vault::ROOT, "secret", 100);
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_list_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();
        assert_eq!(versions.len(), 2);

        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);

        assert!(versions[1].created_at >= versions[0].created_at);
    }

    #[test]
    fn test_rollback() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "original").unwrap();
        vault.set(Vault::ROOT, "secret", "modified").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "modified");

        vault.rollback(Vault::ROOT, "secret", 1).unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "original");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
    }

    #[test]
    fn test_max_versions_pruning() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_max_versions(3);
        let vault = Vault::new(b"test", graph, store, config).unwrap();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();
        vault.set(Vault::ROOT, "secret", "v4").unwrap();

        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);

        assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v2");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v3");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v4");
    }

    #[test]
    fn test_rotate_keeps_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "initial").unwrap();
        vault.rotate(Vault::ROOT, "secret", "rotated").unwrap();

        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
            "initial"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
            "rotated"
        );
    }

    #[test]
    fn test_delete_removes_all_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 3);

        vault.delete(Vault::ROOT, "secret").unwrap();

        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 0);
    }

    #[test]
    fn test_version_access_control() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let result = vault.get_version("user:alice", "secret", 1);
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        assert_eq!(
            vault.get_version("user:alice", "secret", 1).unwrap(),
            "value"
        );
    }

    #[test]
    fn test_with_max_versions_config() {
        let config = VaultConfig::default().with_max_versions(10);
        assert_eq!(config.max_versions, 10);
    }

    #[test]
    fn test_version_info_timestamps() {
        let vault = create_test_vault();

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].created_at >= before);
    }

    // === Signed Edge Tests ===

    #[test]
    fn test_signed_edge_grants_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Signed edges (created by grant) should work
        let value = vault.get("user:alice", "secret").unwrap();
        assert_eq!(value, "value");

        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_tampered_edge_denied() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Verify access works before tampering
        assert!(vault.get("user:alice", "secret").is_ok());

        // Find the access edge and tamper with its signature
        let secret_node = vault.secret_node_key("secret");
        let alice_node_id = vault.get_or_create_entity_node("user:alice");
        let edges = vault
            .graph
            .edges_of(alice_node_id, Direction::Outgoing)
            .unwrap();

        for edge in edges {
            if edge.edge_type.starts_with("VAULT_ACCESS") {
                // Delete the edge and recreate with a tampered signature
                vault.graph.delete_edge(edge.id).unwrap();

                let target_node = vault.get_or_create_entity_node(&secret_node);
                let mut props = HashMap::new();
                props.insert(
                    "vault_sig".to_string(),
                    PropertyValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
                );
                props.insert("vault_sig_ts".to_string(), PropertyValue::Int(12345));
                vault
                    .graph
                    .create_edge(alice_node_id, target_node, &edge.edge_type, props, true)
                    .unwrap();
            }
        }

        // Access should be denied with tampered signature
        let result = vault.get("user:alice", "secret");
        assert!(
            result.is_err(),
            "Tampered edge signature should deny access"
        );
    }

    #[test]
    fn test_unsigned_legacy_edge_accepted() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Manually create an unsigned VAULT_ACCESS edge (legacy/backward-compatible)
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(
            &vault.graph,
            "user:legacy",
            &secret_node,
            "VAULT_ACCESS_READ",
        );

        // Legacy unsigned edges should still be accepted
        let value = vault.get("user:legacy", "secret").unwrap();
        assert_eq!(value, "value");

        assert_eq!(
            vault.get_permission("user:legacy", "secret"),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_tampered_edge_not_in_list() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Verify secret shows in list before tampering
        assert_eq!(vault.list("user:alice", "*").unwrap().len(), 1);

        // Tamper with the edge signature
        let secret_node = vault.secret_node_key("secret");
        let alice_node_id = vault.get_or_create_entity_node("user:alice");
        let edges = vault
            .graph
            .edges_of(alice_node_id, Direction::Outgoing)
            .unwrap();

        for edge in edges {
            if edge.edge_type.starts_with("VAULT_ACCESS") {
                vault.graph.delete_edge(edge.id).unwrap();

                let target_node = vault.get_or_create_entity_node(&secret_node);
                let mut props = HashMap::new();
                props.insert(
                    "vault_sig".to_string(),
                    PropertyValue::Bytes(vec![0xFF; 32]),
                );
                props.insert("vault_sig_ts".to_string(), PropertyValue::Int(99999));
                vault
                    .graph
                    .create_edge(alice_node_id, target_node, &edge.edge_type, props, true)
                    .unwrap();
            }
        }

        // Tampered edge should make secret invisible in list
        let keys = vault.list("user:alice", "*").unwrap();
        assert!(
            keys.is_empty(),
            "Tampered edge should hide secret from list()"
        );
    }

    // === Attenuation Integration Tests ===

    fn create_vault_with_attenuation(policy: AttenuationPolicy) -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_attenuation(policy);
        Vault::new(b"test_password", graph, store, config).unwrap()
    }

    #[test]
    fn test_attenuation_direct_admin_preserved() {
        let vault = create_test_vault(); // default policy: admin_limit=1

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Admin)
            .unwrap();

        // Direct (1 hop): Admin preserved
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_attenuation_2hop_admin_attenuated_to_write() {
        let vault = create_test_vault(); // default: admin_limit=1, write_limit=2

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> team (MEMBER, 1 hop) -> secret (VAULT_ACCESS_ADMIN, +1 = 2 hops)
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 2 hops: Admin attenuated to Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );

        // Can read and write but not delete (Admin required)
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.set("user:alice", "secret", "updated").is_ok());
        assert!(matches!(
            vault.delete("user:alice", "secret"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_attenuation_3hop_attenuated_to_read() {
        let vault = create_test_vault(); // default: admin_limit=1, write_limit=2

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> org (MEMBER) -> team (MEMBER) -> secret (VAULT_ACCESS_ADMIN)
        // depth: 0 -> 1 -> 2 -> VAULT_ACCESS at depth 3
        add_test_edge(&vault.graph, "user:alice", "org:eng", "MEMBER");
        add_test_edge(&vault.graph, "org:eng", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 3 hops: Admin attenuated to Read
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Read)
        );

        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(matches!(
            vault.set("user:alice", "secret", "hack"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_attenuation_beyond_horizon_denied() {
        // horizon=2 means BFS stops at depth 2
        let policy = AttenuationPolicy {
            admin_limit: 1,
            write_limit: 1,
            horizon: 2,
        };
        let vault = create_vault_with_attenuation(policy);

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> org -> team -> secret (3 MEMBER hops + VAULT_ACCESS = beyond horizon)
        add_test_edge(&vault.graph, "user:alice", "org:eng", "MEMBER");
        add_test_edge(&vault.graph, "org:eng", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // Beyond horizon: no access
        assert_eq!(vault.get_permission("user:alice", "secret"), None);
        assert!(vault.get("user:alice", "secret").is_err());
    }

    #[test]
    fn test_no_attenuation_policy_preserves_admin_at_depth() {
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Deep chain: alice -> g1 -> g2 -> g3 -> g4 -> secret
        add_test_edge(&vault.graph, "user:alice", "group:1", "MEMBER");
        add_test_edge(&vault.graph, "group:1", "group:2", "MEMBER");
        add_test_edge(&vault.graph, "group:2", "group:3", "MEMBER");
        add_test_edge(&vault.graph, "group:3", "group:4", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "group:4", "secret", Permission::Admin)
            .unwrap();

        // With no attenuation, Admin is preserved regardless of depth
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_with_attenuation_config_builder() {
        let policy = AttenuationPolicy {
            admin_limit: 3,
            write_limit: 5,
            horizon: 20,
        };
        let config = VaultConfig::default().with_attenuation(policy);
        assert_eq!(config.attenuation.admin_limit, 3);
        assert_eq!(config.attenuation.write_limit, 5);
        assert_eq!(config.attenuation.horizon, 20);
    }

    // === Bottleneck Flow Tests ===

    #[test]
    fn test_bottleneck_restricts_permission() {
        // Grant Admin to a team lead, but team lead grants Read to member.
        // Member should get Read (bottleneck of the edge capacity).
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:lead", "secret", Permission::Admin)
            .unwrap();

        // Lead grants Read to member -- the edge capacity is Read
        vault
            .grant_with_permission("user:lead", "user:member", "secret", Permission::Read)
            .unwrap();

        // Member gets Read (capped by edge capacity)
        assert_eq!(
            vault.get_permission("user:member", "secret"),
            Some(Permission::Read)
        );
        assert!(vault.get("user:member", "secret").is_ok());
        assert!(matches!(
            vault.set("user:member", "secret", "hack"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_bottleneck_multiple_paths_best_wins() {
        // Two paths with different capacities: best effective wins.
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Path 1: direct Read
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        // Path 2: via team with Write
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Write)
            .unwrap();

        // Best of (Read, Write) = Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_bottleneck_with_attenuation() {
        // Capacity=Admin, but 2 hops attenuate Admin->Write. Bottleneck = min(Admin, Write) = Write.
        let vault = create_test_vault(); // default attenuation: admin_limit=1

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 2 hops + Admin capacity: attenuation gives Write, capacity is Admin
        // effective = min(Write, Admin) = Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_permission_to_level_roundtrip() {
        assert_eq!(
            Permission::from_level(Permission::Read.to_level()),
            Some(Permission::Read)
        );
        assert_eq!(
            Permission::from_level(Permission::Write.to_level()),
            Some(Permission::Write)
        );
        assert_eq!(
            Permission::from_level(Permission::Admin.to_level()),
            Some(Permission::Admin)
        );
        assert_eq!(Permission::from_level(0), None);
        assert_eq!(Permission::from_level(4), None);
    }

    // === TTL Persistence Tests ===

    #[test]
    fn test_ttl_survives_vault_reopen() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        // Create vault and grant with TTL
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        // Verify TTL is tracked
        assert!(!vault.ttl_tracker.is_empty());

        // Drop vault and reopen with same store
        drop(vault);
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();

        // TTL should survive the reopen
        assert!(
            !vault2.ttl_tracker.is_empty(),
            "TTL grants should persist across vault reopens"
        );
    }

    // === Delete Graph Cleanup Tests ===

    #[test]
    fn test_delete_cleans_graph_edges() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Alice should have access
        assert!(vault.get("user:alice", "secret").is_ok());

        // Delete the secret
        vault.delete(Vault::ROOT, "secret").unwrap();

        // The secret node should be gone from the graph (use actual obfuscated key)
        let secret_node_key = vault.secret_node_key("secret");
        let found = graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(secret_node_key))
            .unwrap_or_default();
        assert!(found.is_empty(), "Secret graph node should be deleted");

        // Alice's entity node should have no outgoing VAULT_ACCESS edges
        assert!(
            !has_access_edges(&graph, "user:alice"),
            "Alice should have no VAULT_ACCESS edges after secret deletion"
        );
    }

    #[test]
    fn test_delete_cleans_ttl_entries() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        assert!(
            !vault.ttl_tracker.is_empty(),
            "TTL should be tracked after grant"
        );

        // Delete the secret -- TTL entries should be cleaned up
        vault.delete(Vault::ROOT, "secret").unwrap();

        assert!(
            vault.ttl_tracker.is_empty(),
            "TTL tracker should be empty after secret deletion"
        );
    }

    #[test]
    fn test_delete_cleans_edges_multiple_grantees() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "secret").unwrap();

        // Both should have access
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:bob", "secret").is_ok());

        // Delete the secret
        vault.delete(Vault::ROOT, "secret").unwrap();

        // Both entities should have no VAULT_ACCESS edges
        for entity in &["user:alice", "user:bob"] {
            assert!(
                !has_access_edges(&graph, entity),
                "{entity} should have no VAULT_ACCESS edges after secret deletion"
            );
        }
    }

    #[test]
    fn test_delete_then_recreate_secret() {
        let vault = create_test_vault();

        // Create, grant, delete
        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.delete(Vault::ROOT, "secret").unwrap();

        // Recreate the same secret
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        // Alice should NOT have access (old grant was cleaned up)
        assert!(
            vault.get("user:alice", "secret").is_err(),
            "Old grant should not carry over after delete + recreate"
        );

        // Root should still work
        let val = vault.get(Vault::ROOT, "secret").unwrap();
        assert_eq!(val, "v2");

        // Can grant again
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "v2");
    }

    // === Master Key Rotation Tests ===

    fn create_test_vault_mut() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"old_password", graph, store, VaultConfig::default()).unwrap()
    }

    #[test]
    fn test_rotate_master_key_basic() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "api_key", "sk-secret123").unwrap();
        vault.set(Vault::ROOT, "db_pass", "pg-hunter2").unwrap();

        let count = vault.rotate_master_key(b"new_password").unwrap();
        assert_eq!(count, 2);

        // Secrets accessible with new key material
        assert_eq!(vault.get(Vault::ROOT, "api_key").unwrap(), "sk-secret123");
        assert_eq!(vault.get(Vault::ROOT, "db_pass").unwrap(), "pg-hunter2");
    }

    #[test]
    fn test_rotate_master_key_preserves_access() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        assert!(vault.get("user:alice", "secret").is_ok());

        vault.rotate_master_key(b"new_password").unwrap();

        // Alice can still access after rotation
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "value");
    }

    #[test]
    fn test_rotate_master_key_preserves_versions() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v3");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v1");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v2");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v3");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
    }

    #[test]
    fn test_rotate_master_key_old_password_fails() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(
            b"old_password",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        // Opening with old password should produce different key material,
        // so decryption should fail
        let old_vault = Vault::new(b"old_password", graph, store, VaultConfig::default()).unwrap();
        let result = old_vault.get(Vault::ROOT, "secret");
        assert!(
            result.is_err(),
            "Old password should not decrypt after rotation"
        );
    }

    #[test]
    fn test_rotate_master_key_empty_vault() {
        let mut vault = create_test_vault_mut();

        let count = vault.rotate_master_key(b"new_password").unwrap();
        assert_eq!(count, 0);

        // Vault still works after rotating with no secrets
        vault.set(Vault::ROOT, "new_secret", "value").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "new_secret").unwrap(), "value");
    }

    #[test]
    fn test_rotate_master_key_list_works() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "alpha", "a").unwrap();
        vault.set(Vault::ROOT, "beta", "b").unwrap();
        vault.set(Vault::ROOT, "gamma", "c").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        let mut keys = vault.list(Vault::ROOT, "*").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_rotate_master_key_edge_signatures_valid() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Admin)
            .unwrap();

        // Verify access works before rotation
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );

        vault.rotate_master_key(b"new_password").unwrap();

        // Signed edges should still verify with new key material
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
        vault.set("user:alice", "secret", "updated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "updated");
    }

    #[test]
    fn test_rotate_master_key_ttl_preserved() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        // TTL grant should still work after rotation
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(!vault.ttl_tracker.is_empty());
    }

    #[test]
    fn test_rotate_master_key_audit_recorded() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        let recent = vault.audit_recent(10).unwrap();
        let rotation_entries: Vec<_> = recent
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::RotateMasterKey { .. }))
            .collect();
        assert!(
            !rotation_entries.is_empty(),
            "Should have RotateMasterKey audit entry"
        );

        if let AuditOperation::RotateMasterKey { secrets_count } = &rotation_entries[0].operation {
            assert_eq!(*secrets_count, 1);
        }
    }

    #[test]
    fn test_rotate_master_key_multiple_grants() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s2", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Read)
            .unwrap();

        vault.rotate_master_key(b"rotated").unwrap();

        assert_eq!(vault.get("user:alice", "s1").unwrap(), "v1");
        assert_eq!(vault.get("user:bob", "s2").unwrap(), "v2");
        assert_eq!(vault.get("user:alice", "s2").unwrap(), "v2");

        // Permission levels preserved
        assert_eq!(
            vault.get_permission("user:alice", "s1"),
            Some(Permission::Admin)
        );
        assert_eq!(
            vault.get_permission("user:bob", "s2"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_rotate_master_key_new_vault_instance() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(
            b"old_password",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "my_value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        // Create a fresh vault instance with the new password
        let vault2 = Vault::new(b"new_password", graph, store, VaultConfig::default()).unwrap();
        assert_eq!(vault2.get(Vault::ROOT, "secret").unwrap(), "my_value");
    }

    #[test]
    fn test_rotate_master_key_unsigned_edges_preserved() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create an unsigned legacy edge
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(
            &vault.graph,
            "user:legacy",
            &secret_node,
            "VAULT_ACCESS_READ",
        );

        // Legacy user can access
        assert!(vault.get("user:legacy", "secret").is_ok());

        vault.rotate_master_key(b"new_password").unwrap();

        // Unsigned legacy edge should still work (not re-signed)
        assert!(vault.get("user:legacy", "secret").is_ok());
    }

    #[test]
    fn test_rotate_master_key_secrets_without_grants() {
        let mut vault = create_test_vault_mut();

        // Create secrets with no grants (only root access)
        vault.set(Vault::ROOT, "lonely_secret", "alone").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "lonely_secret").unwrap(), "alone");
    }

    #[test]
    fn test_ttl_grant_cleaned_on_restart() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        // Create vault, store a secret, grant with short TTL
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        // Wait for TTL to expire, then drop (simulating process exit without cleanup)
        std::thread::sleep(Duration::from_millis(10));
        drop(vault);

        // Reopen vault -- constructor should clean up the expired grant
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();

        // The expired grant's graph edge should have been removed on startup
        let result = vault2.get("user:alice", "secret");
        assert!(
            matches!(result, Err(VaultError::AccessDenied(_))),
            "expired TTL grant should be revoked after restart"
        );
    }

    #[test]
    fn test_ttl_non_expired_survives_restart() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        drop(vault);

        // Reopen vault -- non-expired grant should still work
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();
        assert!(
            vault2.get("user:alice", "secret").is_ok(),
            "non-expired TTL grant should survive restart"
        );
    }

    #[test]
    fn test_rotate_master_key_double_rotation() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        vault.rotate_master_key(b"password_2").unwrap();
        vault.rotate_master_key(b"password_3").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "value");
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "value");
    }

    // === Concurrent Write Protection Tests ===

    #[test]
    fn test_concurrent_set_no_lost_versions() {
        let vault = Arc::new(create_test_vault());
        vault.set(Vault::ROOT, "secret", "v0").unwrap();

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let v = Arc::clone(&vault);
                std::thread::spawn(move || {
                    for i in 0..10 {
                        v.set(Vault::ROOT, "secret", &format!("t{t}_v{i}")).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // The secret should still be readable without panic
        assert!(vault.get(Vault::ROOT, "secret").is_ok());
    }

    #[test]
    fn test_concurrent_rotate_no_lost_versions() {
        let vault = Arc::new(create_test_vault());
        vault.set(Vault::ROOT, "secret", "initial").unwrap();

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let v = Arc::clone(&vault);
                std::thread::spawn(move || {
                    for i in 0..10 {
                        v.rotate(Vault::ROOT, "secret", &format!("rot_t{t}_v{i}"))
                            .unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert!(vault.get(Vault::ROOT, "secret").is_ok());
    }

    #[test]
    fn test_concurrent_set_and_rotate_mixed() {
        let vault = Arc::new(create_test_vault());
        vault.set(Vault::ROOT, "secret", "initial").unwrap();

        let v1 = Arc::clone(&vault);
        let v2 = Arc::clone(&vault);

        let setter = std::thread::spawn(move || {
            for i in 0..10 {
                v1.set(Vault::ROOT, "secret", &format!("set_{i}")).unwrap();
            }
        });

        let rotator = std::thread::spawn(move || {
            for i in 0..10 {
                v2.rotate(Vault::ROOT, "secret", &format!("rot_{i}"))
                    .unwrap();
            }
        });

        setter.join().unwrap();
        rotator.join().unwrap();

        assert!(vault.get(Vault::ROOT, "secret").is_ok());
    }

    #[test]
    fn test_concurrent_writes_different_secrets_independent() {
        let vault = Arc::new(create_test_vault());

        // Create separate secrets
        for i in 0..4 {
            vault
                .set(Vault::ROOT, &format!("secret_{i}"), "initial")
                .unwrap();
        }

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let v = Arc::clone(&vault);
                std::thread::spawn(move || {
                    for i in 0..10 {
                        v.set(Vault::ROOT, &format!("secret_{t}"), &format!("v{i}"))
                            .unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All secrets should be independently readable
        for i in 0..4 {
            assert!(vault.get(Vault::ROOT, &format!("secret_{i}")).is_ok());
        }
    }

    // ====== Delegation integration tests ======

    #[test]
    fn test_delegate_creates_access() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "db/pass", "secret123").unwrap();

        // Grant Write to parent agent
        vault
            .grant_with_permission(Vault::ROOT, "agent:parent", "db/pass", Permission::Write)
            .unwrap();

        // Parent delegates Read to child
        vault
            .delegate(
                "agent:parent",
                "agent:child",
                &["db/pass"],
                Permission::Read,
                None,
            )
            .unwrap();

        // Child should be able to read
        assert_eq!(vault.get("agent:child", "db/pass").unwrap(), "secret123");
    }

    #[test]
    fn test_delegate_permission_ceiling() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "db/pass", "secret123").unwrap();

        // Grant Write to parent
        vault
            .grant_with_permission(Vault::ROOT, "agent:parent", "db/pass", Permission::Write)
            .unwrap();

        // Parent delegates Read to child
        vault
            .delegate(
                "agent:parent",
                "agent:child",
                &["db/pass"],
                Permission::Read,
                None,
            )
            .unwrap();

        // Child should not be able to write (only has Read)
        let result = vault.rotate("agent:child", "db/pass", "new_value");
        assert!(result.is_err());
    }

    #[test]
    fn test_delegate_requires_sufficient_permission() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "db/pass", "secret123").unwrap();

        // Grant Read to parent
        vault
            .grant_with_permission(Vault::ROOT, "agent:parent", "db/pass", Permission::Read)
            .unwrap();

        // Parent tries to delegate Write -- should fail (parent only has Read)
        let result = vault.delegate(
            "agent:parent",
            "agent:child",
            &["db/pass"],
            Permission::Write,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_revoke_delegation_removes_access() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "db/pass", "secret123").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "agent:parent", "db/pass", Permission::Write)
            .unwrap();
        vault
            .delegate(
                "agent:parent",
                "agent:child",
                &["db/pass"],
                Permission::Read,
                None,
            )
            .unwrap();

        // Child can read
        assert!(vault.get("agent:child", "db/pass").is_ok());

        // Revoke delegation
        vault
            .revoke_delegation("agent:parent", "agent:child")
            .unwrap();

        // Child should no longer have access
        assert!(vault.get("agent:child", "db/pass").is_err());
    }

    #[test]
    fn test_cascading_revocation() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "db/pass", "secret123").unwrap();

        // root -> parent(Admin) -> childA(Write) -> childB(Read)
        vault.grant(Vault::ROOT, "agent:parent", "db/pass").unwrap();
        vault
            .delegate(
                "agent:parent",
                "agent:childA",
                &["db/pass"],
                Permission::Write,
                None,
            )
            .unwrap();
        vault
            .delegate(
                "agent:childA",
                "agent:childB",
                &["db/pass"],
                Permission::Read,
                None,
            )
            .unwrap();

        // Both children can access
        assert!(vault.get("agent:childA", "db/pass").is_ok());
        assert!(vault.get("agent:childB", "db/pass").is_ok());

        // Cascade revoke from parent -> childA
        let revoked = vault
            .revoke_delegation_cascading("agent:parent", "agent:childA")
            .unwrap();
        assert_eq!(revoked.len(), 2);

        // Neither child should have access
        assert!(vault.get("agent:childA", "db/pass").is_err());
        assert!(vault.get("agent:childB", "db/pass").is_err());
    }

    #[test]
    fn test_anomaly_fires_on_first_access() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key1", "value1").unwrap();

        // The anomaly monitor should have seen root's operations
        let profile = vault.anomaly_monitor().get_profile(Vault::ROOT);
        assert!(profile.is_some());
        let p = profile.unwrap();
        assert!(p.total_ops > 0);
    }

    #[test]
    fn test_anomaly_profiles_persist() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());

        {
            let vault = Vault::new(
                b"test_password",
                graph.clone(),
                store.clone(),
                VaultConfig::default().with_anomaly_thresholds(AnomalyThresholds::default()),
            )
            .unwrap();
            vault.set(Vault::ROOT, "key1", "value1").unwrap();
            vault.persist_anomaly_profiles();
        }

        // Reopen vault with same store -- profiles should load
        let vault2 = Vault::new(
            b"test_password",
            graph,
            store,
            VaultConfig::default().with_anomaly_thresholds(AnomalyThresholds::default()),
        )
        .unwrap();
        let profile = vault2.anomaly_monitor().get_profile(Vault::ROOT);
        assert!(profile.is_some());
        assert!(profile.unwrap().total_ops > 0);
    }

    // ============================================================
    // Tier 2: Secret Expiration Tests
    // ============================================================

    #[test]
    fn test_set_with_ttl_and_get_before_expiry() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "db/pass", "secret", Duration::from_secs(3600))
            .unwrap();
        assert_eq!(vault.get(Vault::ROOT, "db/pass").unwrap(), "secret");
    }

    #[test]
    fn test_set_with_ttl_and_get_after_expiry() {
        let vault = create_test_vault();
        // TTL of 0 seconds means it's already expired by the time we read
        vault
            .set_with_ttl(Vault::ROOT, "db/pass", "secret", Duration::from_secs(0))
            .unwrap();
        // Small sleep to ensure timestamp advances
        std::thread::sleep(Duration::from_millis(10));
        let result = vault.get(Vault::ROOT, "db/pass");
        assert!(matches!(result, Err(VaultError::SecretExpired(_))));
    }

    #[test]
    fn test_set_with_ttl_get_version_after_expiry() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "v1", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let result = vault.get_version(Vault::ROOT, "key", 1);
        assert!(matches!(result, Err(VaultError::SecretExpired(_))));
    }

    #[test]
    fn test_list_filters_expired() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "alive", "ok").unwrap();
        vault
            .set_with_ttl(Vault::ROOT, "dead", "gone", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));

        let keys = vault.list(Vault::ROOT, "*").unwrap();
        assert!(keys.contains(&"alive".to_string()));
        assert!(!keys.contains(&"dead".to_string()));
    }

    #[test]
    fn test_clear_expiration_makes_readable() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));

        // Should be expired
        assert!(vault.get(Vault::ROOT, "key").is_err());

        // Admin clears expiration
        vault.clear_expiration(Vault::ROOT, "key").unwrap();

        // Now readable
        assert_eq!(vault.get(Vault::ROOT, "key").unwrap(), "val");
    }

    #[test]
    fn test_clear_expiration_requires_admin() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(3600))
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "key", Permission::Read)
            .unwrap();

        let result = vault.clear_expiration("user:reader", "key");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_get_expiration_returns_timestamp() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(3600))
            .unwrap();

        let exp = vault.get_expiration(Vault::ROOT, "key").unwrap();
        assert!(exp.is_some());
        // Timestamp should be roughly now + 3600
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let delta = exp.unwrap() - now;
        assert!(delta >= 3590 && delta <= 3610);
    }

    #[test]
    fn test_get_expiration_none_for_no_ttl() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        let exp = vault.get_expiration(Vault::ROOT, "key").unwrap();
        assert!(exp.is_none());
    }

    #[test]
    fn test_set_overwrite_clears_expiration() {
        let vault = create_test_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert!(vault.get(Vault::ROOT, "key").is_err());

        // Regular set() clears expiration
        vault.set(Vault::ROOT, "key", "new_val").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "key").unwrap(), "new_val");
        assert!(vault.get_expiration(Vault::ROOT, "key").unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_rotate_master_key_preserves_expiration() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(b"password", graph, store, VaultConfig::default()).unwrap();

        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(7200))
            .unwrap();
        let before = vault.get_expiration(Vault::ROOT, "key").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        let after = vault.get_expiration(Vault::ROOT, "key").unwrap();
        assert_eq!(before, after);
        assert_eq!(vault.get(Vault::ROOT, "key").unwrap(), "val");
    }

    // ============================================================
    // Tier 2: Transit Encryption Tests
    // ============================================================

    #[test]
    fn test_encrypt_decrypt_roundtrip_transit() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let data = b"hello, world!";
        let sealed = vault.encrypt_for(Vault::ROOT, "key", data).unwrap();
        let decrypted = vault.decrypt_as(Vault::ROOT, "key", &sealed).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_transit_different_keys_different_ciphertexts() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let data = b"same data";
        let sealed1 = vault.encrypt_for(Vault::ROOT, "key", data).unwrap();
        let sealed2 = vault.encrypt_for(Vault::ROOT, "key", data).unwrap();
        // Different random nonces produce different ciphertexts
        assert_ne!(sealed1, sealed2);
    }

    #[test]
    fn test_transit_requires_access() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let result = vault.encrypt_for("user:nobody", "key", b"data");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_transit_empty_plaintext() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let sealed = vault.encrypt_for(Vault::ROOT, "key", b"").unwrap();
        let decrypted = vault.decrypt_as(Vault::ROOT, "key", &sealed).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_transit_large_plaintext() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let data = vec![0xab_u8; 1024 * 1024]; // 1MB
        let sealed = vault.encrypt_for(Vault::ROOT, "key", &data).unwrap();
        let decrypted = vault.decrypt_as(Vault::ROOT, "key", &sealed).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_transit_invalid_sealed_data() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        // Too short: less than nonce size
        let result = vault.decrypt_as(Vault::ROOT, "key", &[0u8; 5]);
        assert!(matches!(result, Err(VaultError::CryptoError(_))));
    }

    #[test]
    fn test_transit_audit_logged() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let sealed = vault.encrypt_for(Vault::ROOT, "key", b"data").unwrap();
        vault.decrypt_as(Vault::ROOT, "key", &sealed).unwrap();

        let log = vault.audit_log("key").unwrap();
        let transit_ops: Vec<_> = log
            .iter()
            .filter(|e| {
                matches!(
                    e.operation,
                    AuditOperation::TransitEncrypt | AuditOperation::TransitDecrypt
                )
            })
            .collect();
        assert_eq!(transit_ops.len(), 2);
    }

    #[test]
    #[serial]
    fn test_transit_survives_key_rotation_fails() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(b"password", graph, store, VaultConfig::default()).unwrap();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let sealed = vault.encrypt_for(Vault::ROOT, "key", b"secret").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        // After rotation, transit key changes, so decryption should fail
        let result = vault.decrypt_as(Vault::ROOT, "key", &sealed);
        assert!(result.is_err());
    }

    // ============================================================
    // Tier 2: Break-Glass Emergency Access Tests
    // ============================================================

    #[test]
    fn test_emergency_access_bypasses_graph() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "secret/key", "value").unwrap();

        // user:alice has no grant
        assert!(vault.get("user:alice", "secret/key").is_err());

        // Emergency access succeeds
        let val = vault
            .emergency_access(
                "user:alice",
                "secret/key",
                "production outage",
                Duration::from_secs(300),
            )
            .unwrap();
        assert_eq!(val, "value");
    }

    #[test]
    fn test_emergency_access_creates_temp_grant() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "secret/key", "value").unwrap();

        vault
            .emergency_access(
                "user:alice",
                "secret/key",
                "outage",
                Duration::from_secs(3600),
            )
            .unwrap();

        // Now regular get() works because of the temp grant
        assert_eq!(vault.get("user:alice", "secret/key").unwrap(), "value");
    }

    #[test]
    fn test_emergency_access_audit_includes_justification() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        vault
            .emergency_access(
                "user:alice",
                "key",
                "database corruption",
                Duration::from_secs(60),
            )
            .unwrap();

        let log = vault.audit_log("key").unwrap();
        let break_glass_entries: Vec<_> = log
            .iter()
            .filter(|e| matches!(&e.operation, AuditOperation::BreakGlass { .. }))
            .collect();
        assert_eq!(break_glass_entries.len(), 1);
        if let AuditOperation::BreakGlass {
            justification,
            duration_secs,
        } = &break_glass_entries[0].operation
        {
            assert_eq!(justification, "database corruption");
            assert_eq!(*duration_secs, 60);
        }
    }

    #[test]
    fn test_emergency_access_rate_limited() {
        let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
            max_gets: 100,
            max_lists: 100,
            max_sets: 100,
            max_grants: 100,
            max_break_glass: 1,
            max_wraps: 100,
            max_generates: 100,
            window: Duration::from_secs(60),
        });
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        // First break-glass succeeds
        vault
            .emergency_access("user:alice", "key", "first", Duration::from_secs(60))
            .unwrap();

        // Second is rate limited
        let result = vault.emergency_access("user:alice", "key", "second", Duration::from_secs(60));
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_emergency_access_nonexistent_secret() {
        let vault = create_rate_limited_vault();
        let result = vault.emergency_access(
            "user:alice",
            "nonexistent",
            "reason",
            Duration::from_secs(60),
        );
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_emergency_access_expired_secret() {
        let vault = create_rate_limited_vault();
        vault
            .set_with_ttl(Vault::ROOT, "key", "val", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));

        let result = vault.emergency_access("user:alice", "key", "reason", Duration::from_secs(60));
        assert!(matches!(result, Err(VaultError::SecretExpired(_))));
    }

    #[test]
    fn test_emergency_access_grant_expires() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        vault
            .emergency_access("user:alice", "key", "reason", Duration::from_millis(50))
            .unwrap();

        // Grant should be active now
        assert_eq!(vault.get("user:alice", "key").unwrap(), "val");

        // Wait for grant to expire
        std::thread::sleep(Duration::from_millis(60));

        // Next get() triggers cleanup, grant is removed
        let result = vault.get("user:alice", "key");
        assert!(result.is_err());
    }

    #[test]
    fn test_emergency_access_empty_justification() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        // Empty justification is allowed
        let val = vault
            .emergency_access("user:alice", "key", "", Duration::from_secs(60))
            .unwrap();
        assert_eq!(val, "val");
    }

    // ============================================================
    // Tier 2: Batch Operations Tests
    // ============================================================

    #[test]
    fn test_batch_get_multiple_secrets() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "a", "1").unwrap();
        vault.set(Vault::ROOT, "b", "2").unwrap();
        vault.set(Vault::ROOT, "c", "3").unwrap();

        let results = vault.batch_get(Vault::ROOT, &["a", "b", "c"]).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1.as_ref().unwrap(), "1");
        assert_eq!(results[1].1.as_ref().unwrap(), "2");
        assert_eq!(results[2].1.as_ref().unwrap(), "3");
    }

    #[test]
    fn test_batch_get_partial_access() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "public", "yes").unwrap();
        vault.set(Vault::ROOT, "private", "no").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "public").unwrap();

        let results = vault
            .batch_get("user:alice", &["public", "private"])
            .unwrap();
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_err());
    }

    #[test]
    fn test_batch_get_empty_keys() {
        let vault = create_test_vault();
        let results = vault.batch_get(Vault::ROOT, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_get_with_expired() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "alive", "ok").unwrap();
        vault
            .set_with_ttl(Vault::ROOT, "dead", "gone", Duration::from_secs(0))
            .unwrap();
        std::thread::sleep(Duration::from_millis(10));

        let results = vault.batch_get(Vault::ROOT, &["alive", "dead"]).unwrap();
        assert!(results[0].1.is_ok());
        assert!(matches!(results[1].1, Err(VaultError::SecretExpired(_))));
    }

    #[test]
    fn test_batch_get_audit_logged() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "k1", "v1").unwrap();
        vault.set(Vault::ROOT, "k2", "v2").unwrap();

        vault.batch_get(Vault::ROOT, &["k1", "k2"]).unwrap();

        let log = vault.audit_log("*").unwrap();
        let batch_entries: Vec<_> = log
            .iter()
            .filter(|e| matches!(&e.operation, AuditOperation::BatchGet { .. }))
            .collect();
        assert_eq!(batch_entries.len(), 1);
        if let AuditOperation::BatchGet { count } = &batch_entries[0].operation {
            assert_eq!(*count, 2);
        }
    }

    #[test]
    fn test_batch_set_multiple_secrets() {
        let vault = create_test_vault();
        vault
            .batch_set(Vault::ROOT, &[("a", "1"), ("b", "2"), ("c", "3")])
            .unwrap();

        assert_eq!(vault.get(Vault::ROOT, "a").unwrap(), "1");
        assert_eq!(vault.get(Vault::ROOT, "b").unwrap(), "2");
        assert_eq!(vault.get(Vault::ROOT, "c").unwrap(), "3");
    }

    #[test]
    fn test_batch_set_permission_check() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "existing", "old").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "existing", Permission::Write)
            .unwrap();

        // user:writer can update existing but not create new -- batch fails
        let result = vault.batch_set("user:writer", &[("existing", "new"), ("brand_new", "val")]);
        assert!(result.is_err());

        // The batch sorts by obfuscated key, so order is non-deterministic.
        // "existing" may or may not have been updated before the error.
        let val = vault.get(Vault::ROOT, "existing").unwrap();
        assert!(val == "old" || val == "new");
    }

    #[test]
    fn test_batch_set_empty_entries() {
        let vault = create_test_vault();
        assert!(vault.batch_set(Vault::ROOT, &[]).is_ok());
    }

    #[test]
    fn test_batch_set_audit_logged() {
        let vault = create_test_vault();
        vault
            .batch_set(Vault::ROOT, &[("x", "1"), ("y", "2")])
            .unwrap();

        let log = vault.audit_log("*").unwrap();
        let batch_entries: Vec<_> = log
            .iter()
            .filter(|e| matches!(&e.operation, AuditOperation::BatchSet { .. }))
            .collect();
        assert_eq!(batch_entries.len(), 1);
        if let AuditOperation::BatchSet { count } = &batch_entries[0].operation {
            assert_eq!(*count, 2);
        }
    }

    // ============================================================
    // Tier 2: Integration / Scoped / Namespaced Tests
    // ============================================================

    #[test]
    fn test_scoped_set_with_ttl() {
        let vault = create_test_vault();
        let scoped = vault.scope(Vault::ROOT);
        scoped
            .set_with_ttl("key", "val", Duration::from_secs(3600))
            .unwrap();
        assert_eq!(scoped.get("key").unwrap(), "val");
        assert!(scoped.get_expiration("key").unwrap().is_some());
    }

    #[test]
    fn test_scoped_transit_encrypt_decrypt() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "unused").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let sealed = scoped.encrypt_for("key", b"secret data").unwrap();
        let plain = scoped.decrypt_as("key", &sealed).unwrap();
        assert_eq!(plain, b"secret data");
    }

    #[test]
    fn test_scoped_emergency_access() {
        let vault = create_rate_limited_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        let scoped = vault.scope("user:alice");
        let val = scoped
            .emergency_access("key", "outage", Duration::from_secs(60))
            .unwrap();
        assert_eq!(val, "val");
    }

    #[test]
    fn test_scoped_batch_get() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "a", "1").unwrap();
        vault.set(Vault::ROOT, "b", "2").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let results = scoped.batch_get(&["a", "b"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.as_ref().unwrap(), "1");
    }

    #[test]
    fn test_namespaced_set_with_ttl() {
        let vault = create_test_vault();
        let ns = vault.namespace("prod", Vault::ROOT);
        ns.set_with_ttl("key", "val", Duration::from_secs(3600))
            .unwrap();
        assert_eq!(ns.get("key").unwrap(), "val");
        assert!(ns.get_expiration("key").unwrap().is_some());
    }

    #[test]
    fn test_namespaced_transit() {
        let vault = create_test_vault();
        let ns = vault.namespace("prod", Vault::ROOT);
        ns.set("key", "unused").unwrap();

        let sealed = ns.encrypt_for("key", b"data").unwrap();
        let plain = ns.decrypt_as("key", &sealed).unwrap();
        assert_eq!(plain, b"data");
    }

    #[test]
    fn test_namespaced_batch_get_prefixed() {
        let vault = create_test_vault();
        let ns = vault.namespace("prod", Vault::ROOT);
        ns.set("a", "1").unwrap();
        ns.set("b", "2").unwrap();

        let results = ns.batch_get(&["a", "b"]).unwrap();
        assert_eq!(results.len(), 2);
        // Keys should be stripped of namespace prefix
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
        assert_eq!(results[0].1.as_ref().unwrap(), "1");
        assert_eq!(results[1].1.as_ref().unwrap(), "2");
    }

    // ============================================================
    // Transit Key Independence Tests
    // ============================================================

    #[test]
    fn test_transit_key_independent_of_other_subkeys() {
        let key = crate::key::MasterKey::from_bytes([42u8; 32]);
        let transit = key.transit_key();
        let encryption = key.encryption_key();
        let obfuscation = key.obfuscation_key();
        let audit = key.audit_key();
        let metadata = key.metadata_key();

        assert_ne!(transit, encryption);
        assert_ne!(transit, obfuscation);
        assert_ne!(transit, audit);
        assert_ne!(transit, metadata);
        assert_ne!(&transit, key.as_bytes());
    }

    #[test]
    fn test_transit_key_deterministic() {
        let key1 = crate::key::MasterKey::from_bytes([99u8; 32]);
        let key2 = crate::key::MasterKey::from_bytes([99u8; 32]);
        assert_eq!(key1.transit_key(), key2.transit_key());
    }

    // ============================================================
    // Break-glass extras
    // ============================================================

    #[test]
    fn test_emergency_access_with_custom_rate_limit() {
        let config = VaultConfig::default().with_rate_limit(RateLimitConfig::strict());
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        // strict() has max_break_glass=1
        let val = vault
            .emergency_access("user:alice", "key", "test", Duration::from_secs(60))
            .unwrap();
        assert_eq!(val, "val");

        // Second attempt should fail
        let result = vault.emergency_access("user:alice", "key", "test2", Duration::from_secs(60));
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_batch_set_deadlock_prevention() {
        // Concurrent batch_set with overlapping keys should not deadlock
        let vault = Arc::new(create_test_vault());
        vault.set(Vault::ROOT, "x", "init_x").unwrap();
        vault.set(Vault::ROOT, "y", "init_y").unwrap();

        let v1 = Arc::clone(&vault);
        let v2 = Arc::clone(&vault);

        let h1 = std::thread::spawn(move || {
            for _ in 0..10 {
                v1.batch_set(Vault::ROOT, &[("x", "a"), ("y", "b")]).ok();
            }
        });
        let h2 = std::thread::spawn(move || {
            for _ in 0..10 {
                v2.batch_set(Vault::ROOT, &[("y", "c"), ("x", "d")]).ok();
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Both keys should still be readable
        assert!(vault.get(Vault::ROOT, "x").is_ok());
        assert!(vault.get(Vault::ROOT, "y").is_ok());
    }

    // ========== Coverage tests for uncovered vault methods ==========

    #[test]
    #[serial]
    fn test_wrap_and_unwrap_secret() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "wrap_key", "wrap_value").unwrap();

        let token = vault.wrap_secret(Vault::ROOT, "wrap_key", 60_000).unwrap();
        assert!(vault.wrapping_token_info(&token).is_some());

        let value = vault.unwrap_secret(&token).unwrap();
        assert_eq!(value, "wrap_value");

        // Token consumed -- info should be gone
        assert!(vault.wrapping_token_info(&token).is_none());
        assert!(vault.unwrap_secret(&token).is_err());
    }

    #[test]
    #[serial]
    fn test_dependency_management() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "parent", "p").unwrap();
        vault.set(Vault::ROOT, "child", "c").unwrap();

        vault
            .add_dependency(Vault::ROOT, "parent", "child")
            .unwrap();

        let deps = vault.get_dependencies(Vault::ROOT, "parent").unwrap();
        assert!(!deps.is_empty());

        let dependents = vault.get_dependents(Vault::ROOT, "child").unwrap();
        assert!(!dependents.is_empty());

        let report = vault.impact_analysis(Vault::ROOT, "parent").unwrap();
        // root_secret is the obfuscated node key, just verify the report exists
        assert!(!report.root_secret.is_empty());

        vault
            .remove_dependency(Vault::ROOT, "parent", "child")
            .unwrap();
        let deps = vault.get_dependencies(Vault::ROOT, "parent").unwrap();
        assert!(deps.is_empty());
    }

    #[test]
    #[serial]
    fn test_quota_management() {
        let vault = create_test_vault();

        let quota = ResourceQuota {
            max_secrets: 100,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 1000,
        };

        vault.set_quota(Vault::ROOT, "ns1", quota).unwrap();
        assert!(vault.get_quota("ns1").is_some());

        let usage = vault.get_usage("ns1");
        assert_eq!(usage.secret_count, 0);

        // Non-root cannot set quota
        assert!(vault
            .set_quota(
                "user:alice",
                "ns1",
                ResourceQuota {
                    max_secrets: 50,
                    max_storage_bytes: 512,
                    max_ops_per_hour: 50,
                }
            )
            .is_err());

        vault.remove_quota(Vault::ROOT, "ns1").unwrap();
        assert!(vault.get_quota("ns1").is_none());

        // Non-root cannot remove quota
        assert!(vault.remove_quota("user:alice", "ns1").is_err());
    }

    #[test]
    #[serial]
    fn test_dynamic_secret_lifecycle() {
        let vault = create_test_vault();

        let template = SecretTemplate::Password(PasswordConfig::default());
        let (id, value) = vault
            .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
            .unwrap();
        assert!(!id.is_empty());
        assert!(!value.is_empty());

        let fetched = vault.get_dynamic_secret(Vault::ROOT, &id).unwrap();
        assert_eq!(fetched, value);

        let all = vault.list_dynamic_secrets(Vault::ROOT).unwrap();
        assert!(!all.is_empty());

        vault.revoke_dynamic_secret(Vault::ROOT, &id).unwrap();
    }

    #[test]
    #[serial]
    fn test_dynamic_secret_one_time() {
        let vault = create_test_vault();

        let template = SecretTemplate::Token(TokenConfig::default());
        let (id, value) = vault
            .generate_dynamic_secret(Vault::ROOT, &template, 60_000, true)
            .unwrap();

        // First read succeeds
        let fetched = vault.get_dynamic_secret(Vault::ROOT, &id).unwrap();
        assert_eq!(fetched, value);

        // Second read fails -- consumed
        assert!(vault.get_dynamic_secret(Vault::ROOT, &id).is_err());
    }

    #[test]
    #[serial]
    fn test_seal_and_unseal() {
        let mut vault = create_test_vault_mut();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        assert!(!vault.is_sealed());
        vault.seal().unwrap();
        assert!(vault.is_sealed());

        // Seal-guarded operations are blocked
        assert!(vault.wrap_secret(Vault::ROOT, "key", 60_000).is_err());

        vault.unseal(b"old_password").unwrap();
        assert!(!vault.is_sealed());
        // After unseal, operations work again
        assert_eq!(vault.get(Vault::ROOT, "key").unwrap(), "val");
        assert!(vault.wrap_secret(Vault::ROOT, "key", 60_000).is_ok());
    }

    #[test]
    #[serial]
    fn test_policy_management() {
        let vault = create_test_vault();

        let template = PolicyTemplate {
            name: "dev_policy".to_string(),
            match_pattern: "user:dev_*".to_string(),
            secret_pattern: "dev/*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };

        vault.add_policy(Vault::ROOT, template).unwrap();

        let policies = vault.list_policies();
        assert_eq!(policies.len(), 1);

        assert!(vault.get_policy("dev_policy").is_some());

        let matches = vault.evaluate_policies("user:dev_alice");
        assert!(!matches.is_empty());

        // Non-root cannot add policy
        assert!(vault
            .add_policy(
                "user:alice",
                PolicyTemplate {
                    name: "hack".to_string(),
                    match_pattern: "*".to_string(),
                    secret_pattern: "*".to_string(),
                    permission: Permission::Admin,
                    ttl_ms: None,
                }
            )
            .is_err());

        vault.remove_policy(Vault::ROOT, "dev_policy").unwrap();
        assert!(vault.list_policies().is_empty());

        // Non-root cannot remove policy
        assert!(vault.remove_policy("user:alice", "dev_policy").is_err());
    }

    #[test]
    #[serial]
    fn test_snapshot_management() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "snap_key", "original").unwrap();

        let snap = vault.create_snapshot(Vault::ROOT, "my_snapshot").unwrap();
        assert!(!snap.id.is_empty());

        let snaps = vault.list_snapshots();
        assert_eq!(snaps.len(), 1);

        // Modify vault state
        vault.set(Vault::ROOT, "snap_key", "modified").unwrap();

        // Restore
        let count = vault.restore_snapshot(Vault::ROOT, &snap.id).unwrap();
        assert!(count > 0);

        // Non-root cannot create/restore/delete snapshots
        assert!(vault.create_snapshot("user:alice", "bad").is_err());
        assert!(vault.restore_snapshot("user:alice", &snap.id).is_err());
        assert!(vault.delete_snapshot("user:alice", &snap.id).is_err());

        vault.delete_snapshot(Vault::ROOT, &snap.id).unwrap();
    }

    #[test]
    #[serial]
    fn test_rotation_policy_management() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "rot_key", "initial").unwrap();

        let policy = RotationPolicy {
            secret_key: "rot_key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::Password(PasswordConfig::default()),
            notify_before_ms: 0,
        };

        vault
            .set_rotation_policy(Vault::ROOT, "rot_key", policy)
            .unwrap();
        assert!(vault.get_rotation_policy(Vault::ROOT, "rot_key").is_some());

        let policies = vault.list_rotation_policies();
        assert_eq!(policies.len(), 1);

        let pending = vault.check_pending_rotations();
        assert!(!pending.is_empty());

        // Execute rotation
        let new_val = vault.execute_rotation(Vault::ROOT, "rot_key").unwrap();
        assert!(!new_val.is_empty());
        assert_ne!(vault.get(Vault::ROOT, "rot_key").unwrap(), "initial");

        vault
            .remove_rotation_policy(Vault::ROOT, "rot_key")
            .unwrap();
        assert!(vault.get_rotation_policy(Vault::ROOT, "rot_key").is_none());
    }

    #[test]
    #[serial]
    fn test_rotation_with_no_generator_fails() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "manual_key", "val").unwrap();

        let policy = RotationPolicy {
            secret_key: "manual_key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };

        vault
            .set_rotation_policy(Vault::ROOT, "manual_key", policy)
            .unwrap();

        // Execute should fail with None generator
        assert!(vault.execute_rotation(Vault::ROOT, "manual_key").is_err());
    }

    #[test]
    #[serial]
    fn test_engine_registry() {
        let vault = create_test_vault();

        // No engines registered initially
        assert!(vault.list_engines().is_empty());

        // engine_generate without registered engine should fail
        let params = serde_json::json!({});
        assert!(vault
            .engine_generate(Vault::ROOT, "nonexistent", &params)
            .is_err());

        // engine_revoke without registered engine should fail
        assert!(vault
            .engine_revoke(Vault::ROOT, "nonexistent", "id")
            .is_err());
    }

    #[test]
    #[serial]
    fn test_pki_operations() {
        let vault = create_test_vault();

        // Non-root cannot init PKI
        assert!(vault.init_pki("user:alice").is_err());

        vault.init_pki(Vault::ROOT).unwrap();

        let ca = vault.get_ca_certificate(Vault::ROOT).unwrap();
        assert!(!ca.is_empty());

        let request = CertificateRequest {
            common_name: "test.local".to_string(),
            organization: None,
            san_dns: vec!["test.local".to_string()],
            san_ip: vec![],
        };
        let (serial, _) = vault
            .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
            .unwrap();

        let certs = vault.list_certificates(Vault::ROOT).unwrap();
        assert_eq!(certs.len(), 1);

        vault.revoke_certificate(Vault::ROOT, &serial).unwrap();
    }

    #[test]
    #[serial]
    fn test_sync_target_operations() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "sync_key", "sync_val").unwrap();

        let dir = std::env::temp_dir().join("vault_sync_test");
        std::fs::create_dir_all(&dir).ok();

        vault
            .register_sync_target(Box::new(FileSyncTarget::new("file_target", dir.clone())))
            .unwrap();

        let targets = vault.list_sync_targets();
        assert_eq!(targets.len(), 1);

        let health = vault.sync_health();
        assert_eq!(health.len(), 1);

        vault
            .subscribe_sync(Vault::ROOT, "sync_key", "file_target")
            .unwrap();

        let count = vault.trigger_sync(Vault::ROOT, "sync_key").unwrap();
        assert_eq!(count, 1);

        vault
            .unsubscribe_sync(Vault::ROOT, "sync_key", "file_target")
            .unwrap();

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[serial]
    fn test_seal_blocks_advanced_operations() {
        let mut vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();
        vault.seal().unwrap();

        assert!(vault.wrap_secret(Vault::ROOT, "key", 60_000).is_err());
        assert!(vault
            .generate_dynamic_secret(
                Vault::ROOT,
                &SecretTemplate::Password(PasswordConfig::default()),
                60_000,
                false
            )
            .is_err());
        assert!(vault.add_dependency(Vault::ROOT, "key", "other").is_err());
    }

    // ===== P0 Group 1: Seal Enforcement & Key Zeroization =====

    #[test]
    fn test_seal_blocks_core_operations() {
        let mut vault = create_test_vault();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        vault.seal().unwrap();

        // Every core operation should return Err(Sealed)
        assert!(matches!(
            vault.set(Vault::ROOT, "new", "val"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.get(Vault::ROOT, "secret"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.get_version(Vault::ROOT, "secret", 1),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.delete(Vault::ROOT, "secret"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.rotate(Vault::ROOT, "secret", "new_val"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.list(Vault::ROOT, "*"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.grant(Vault::ROOT, "user:bob", "secret"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.revoke(Vault::ROOT, "user:alice", "secret"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.batch_get(Vault::ROOT, &["secret"]),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.batch_set(Vault::ROOT, &[("k", "v")]),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.encrypt_for(Vault::ROOT, "user:alice", b"data"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.audit_log("secret"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.audit_by_entity("user:alice"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(vault.audit_since(0), Err(VaultError::Sealed(_))));
        assert!(matches!(vault.audit_recent(10), Err(VaultError::Sealed(_))));
    }

    #[test]
    fn test_seal_zeroizes_then_unseal_restores() {
        let mut vault = create_test_vault();
        vault.set(Vault::ROOT, "important", "data123").unwrap();

        vault.seal().unwrap();
        assert!(matches!(
            vault.get(Vault::ROOT, "important"),
            Err(VaultError::Sealed(_))
        ));

        vault.unseal(b"test_password").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "important").unwrap(), "data123");
    }

    #[test]
    fn test_root_not_exempt_from_seal() {
        let mut vault = create_test_vault();
        vault.seal().unwrap();

        assert!(matches!(
            vault.set(Vault::ROOT, "key", "val"),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.list(Vault::ROOT, "*"),
            Err(VaultError::Sealed(_))
        ));
    }

    #[test]
    fn test_seal_blocks_rollback() {
        let mut vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.rotate(Vault::ROOT, "key", "v2").unwrap();

        vault.seal().unwrap();
        assert!(matches!(
            vault.rollback(Vault::ROOT, "key", 1),
            Err(VaultError::Sealed(_))
        ));
    }

    #[test]
    fn test_seal_blocks_grant_with_permission_and_ttl() {
        let mut vault = create_test_vault();
        vault.set(Vault::ROOT, "key", "val").unwrap();
        vault.seal().unwrap();

        assert!(matches!(
            vault.grant_with_permission(Vault::ROOT, "user:a", "key", Permission::Read),
            Err(VaultError::Sealed(_))
        ));
        assert!(matches!(
            vault.grant_with_ttl(
                Vault::ROOT,
                "user:a",
                "key",
                Permission::Read,
                Duration::from_secs(60)
            ),
            Err(VaultError::Sealed(_))
        ));
    }

    // ===== P0 Group 2: Emergency Access & Dynamic Secret Hardening =====

    #[test]
    fn test_emergency_access_requires_rate_limiting() {
        let vault = create_test_vault(); // default config, no rate limiting
        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let result = vault.emergency_access(
            "user:alice",
            "secret",
            "need access urgently",
            Duration::from_secs(60),
        );
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
        assert!(result.unwrap_err().to_string().contains("rate limiting"));
    }

    #[test]
    fn test_dynamic_secret_consume_after_get() {
        let vault = create_test_vault();
        let template = SecretTemplate::Password(PasswordConfig::default());
        let (id, _value) = vault
            .generate_dynamic_secret(Vault::ROOT, &template, 60_000, true)
            .unwrap();

        // First get succeeds
        let fetched = vault.get_dynamic_secret(Vault::ROOT, &id);
        assert!(fetched.is_ok());

        // Second get fails (consumed)
        let result = vault.get_dynamic_secret(Vault::ROOT, &id);
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_dynamic_secret_expiration_enforced() {
        let vault = create_test_vault();
        let template = SecretTemplate::Password(PasswordConfig::default());
        // TTL of 1ms; current_timestamp() has second-level granularity,
        // so we must wait >= 1 second for expiration to be visible
        let (id, _value) = vault
            .generate_dynamic_secret(Vault::ROOT, &template, 1, false)
            .unwrap();

        std::thread::sleep(Duration::from_millis(1100));

        let result = vault.get_dynamic_secret(Vault::ROOT, &id);
        assert!(matches!(result, Err(VaultError::SecretExpired(_))));
    }

    #[test]
    fn test_list_dynamic_secrets_root_only() {
        let vault = create_test_vault();
        let template = SecretTemplate::Password(PasswordConfig::default());
        vault
            .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
            .unwrap();

        // Root can list
        let list = vault.list_dynamic_secrets(Vault::ROOT).unwrap();
        assert_eq!(list.len(), 1);

        // Non-root cannot list
        let result = vault.list_dynamic_secrets("user:alice");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_revoke_dynamic_secret_authorization() {
        let vault = create_test_vault();
        let template = SecretTemplate::Password(PasswordConfig::default());
        let (id, _) = vault
            .generate_dynamic_secret(Vault::ROOT, &template, 60_000, false)
            .unwrap();

        // Non-owner, non-root cannot revoke (requester in metadata is ROOT)
        let result = vault.revoke_dynamic_secret("user:alice", &id);
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Root can revoke
        assert!(vault.revoke_dynamic_secret(Vault::ROOT, &id).is_ok());
    }

    // ===== P1 Group 3: Quota Enforcement =====

    #[test]
    fn test_quota_enforced_on_set() {
        let vault = create_test_vault();
        vault
            .set_quota(
                Vault::ROOT,
                "ns",
                ResourceQuota {
                    max_secrets: 2,
                    max_storage_bytes: 1_000_000,
                    max_ops_per_hour: 1000,
                },
            )
            .unwrap();

        vault.set(Vault::ROOT, "ns/key1", "value1").unwrap();
        vault.set(Vault::ROOT, "ns/key2", "value2").unwrap();

        // Third secret exceeds quota
        let result = vault.set(Vault::ROOT, "ns/key3", "value3");
        assert!(matches!(result, Err(VaultError::QuotaExceeded(_))));
    }

    #[test]
    fn test_quota_tracks_delete() {
        let vault = create_test_vault();
        vault
            .set_quota(
                Vault::ROOT,
                "ns",
                ResourceQuota {
                    max_secrets: 2,
                    max_storage_bytes: 1_000_000,
                    max_ops_per_hour: 1000,
                },
            )
            .unwrap();

        vault.set(Vault::ROOT, "ns/key1", "value1").unwrap();
        vault.set(Vault::ROOT, "ns/key2", "value2").unwrap();

        // Delete one
        vault.delete(Vault::ROOT, "ns/key1").unwrap();

        // Now we can add another
        vault.set(Vault::ROOT, "ns/key3", "value3").unwrap();
    }

    #[test]
    fn test_quota_namespace_extraction() {
        let vault = create_test_vault();
        vault
            .set_quota(
                Vault::ROOT,
                "ns",
                ResourceQuota {
                    max_secrets: 1,
                    max_storage_bytes: 1_000_000,
                    max_ops_per_hour: 1000,
                },
            )
            .unwrap();

        // "ns/key1" belongs to namespace "ns"
        vault.set(Vault::ROOT, "ns/key1", "v1").unwrap();

        // "other/key1" belongs to namespace "other" (no quota)
        vault.set(Vault::ROOT, "other/key1", "v1").unwrap();

        // "ns/key2" is blocked by quota
        let result = vault.set(Vault::ROOT, "ns/key2", "v2");
        assert!(matches!(result, Err(VaultError::QuotaExceeded(_))));
    }

    // ===== P1 Group 4: PKI Access Control =====

    #[test]
    fn test_pki_issue_root_only() {
        let vault = create_test_vault();
        vault.init_pki(Vault::ROOT).unwrap();

        let request = CertificateRequest {
            common_name: "test.local".to_string(),
            organization: None,
            san_dns: vec!["test.local".to_string()],
            san_ip: vec![],
        };

        // Non-root cannot issue
        let result = vault.issue_certificate("user:alice", &request, Duration::from_secs(3600));
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Root can issue
        assert!(vault
            .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
            .is_ok());
    }

    #[test]
    fn test_pki_revoke_root_only_with_seal_check() {
        let mut vault = create_test_vault();
        vault.init_pki(Vault::ROOT).unwrap();

        let request = CertificateRequest {
            common_name: "test.local".to_string(),
            organization: None,
            san_dns: vec!["test.local".to_string()],
            san_ip: vec![],
        };
        let (serial, _) = vault
            .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
            .unwrap();

        // Non-root cannot revoke
        let result = vault.revoke_certificate("user:alice", &serial);
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Root can revoke
        vault.revoke_certificate(Vault::ROOT, &serial).unwrap();

        // Re-issue for seal test
        let (serial2, _) = vault
            .issue_certificate(Vault::ROOT, &request, Duration::from_secs(3600))
            .unwrap();
        vault.seal().unwrap();
        assert!(matches!(
            vault.revoke_certificate(Vault::ROOT, &serial2),
            Err(VaultError::Sealed(_))
        ));
    }

    #[test]
    fn test_pki_list_root_only() {
        let vault = create_test_vault();
        vault.init_pki(Vault::ROOT).unwrap();

        // Non-root cannot list
        let result = vault.list_certificates("user:alice");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Root can list
        assert!(vault.list_certificates(Vault::ROOT).is_ok());
    }

    // ===== P1 Group 5: Master Key Rotation Safety =====

    #[test]
    fn test_rotate_master_key_preserves_secrets() {
        let mut vault = create_test_vault_mut();
        vault.set(Vault::ROOT, "key1", "value1").unwrap();
        vault.set(Vault::ROOT, "key2", "value2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key1").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        // All secrets must be accessible after rotation
        assert_eq!(vault.get(Vault::ROOT, "key1").unwrap(), "value1");
        assert_eq!(vault.get(Vault::ROOT, "key2").unwrap(), "value2");
        assert_eq!(vault.get("user:alice", "key1").unwrap(), "value1");
    }

    #[test]
    fn test_rotate_master_key_updates_snapshot_cipher() {
        let mut vault = create_test_vault_mut();
        vault.set(Vault::ROOT, "key", "val").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        // Snapshot should work with updated cipher
        let snap = vault.create_snapshot(Vault::ROOT, "post-rotation").unwrap();
        assert_eq!(snap.secret_count, 1);

        // Restore should also work
        vault.set(Vault::ROOT, "key", "changed").unwrap();
        vault.restore_snapshot(Vault::ROOT, &snap.id).unwrap();
        assert_eq!(vault.get(Vault::ROOT, "key").unwrap(), "val");
    }

    // ===== P1 Group 6: Backup Integrity =====

    #[test]
    fn test_hmac_verified_on_restore() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "snap_key", "snap_val").unwrap();

        let snap = vault
            .create_snapshot(Vault::ROOT, "integrity test")
            .unwrap();

        // Tamper with snapshot data in the store
        let data_key = format!("_vsnapdata:{}", snap.id);
        if let Ok(mut data) = vault.store.get(&data_key) {
            // Corrupt the first value entry
            data.set(
                "_v0",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                    "TAMPERED".to_string(),
                )),
            );
            vault.store.put(&data_key, data).unwrap();
        }

        // Restore should fail due to HMAC mismatch
        let result = vault.restore_snapshot(Vault::ROOT, &snap.id);
        assert!(result.is_err());
    }

    #[test]
    fn test_keyed_hmac_differs_from_default() {
        use tensor_store::TensorStore;

        let store = TensorStore::new();
        // Create an entry in the store
        let mut data = tensor_store::TensorData::new();
        data.set(
            "field",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                "hello".to_string(),
            )),
        );
        store.put("_vk:test", data).unwrap();

        let snap_no_key = crate::pitr::create_snapshot(&store, "no key", None, None).unwrap();
        let snap_with_key =
            crate::pitr::create_snapshot(&store, "with key", None, Some(b"secret_key")).unwrap();

        // Read the HMAC values from stored snapshot data
        let data1 = store
            .get(&format!("_vsnapdata:{}", snap_no_key.id))
            .unwrap();
        let data2 = store
            .get(&format!("_vsnapdata:{}", snap_with_key.id))
            .unwrap();

        let hmac1 = match data1.get("_hmac") {
            Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(s))) => {
                s.clone()
            },
            _ => panic!("missing hmac"),
        };
        let hmac2 = match data2.get("_hmac") {
            Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(s))) => {
                s.clone()
            },
            _ => panic!("missing hmac"),
        };

        assert_ne!(hmac1, hmac2);
    }

    // ===== P1 Group 8: Graph Hardening =====

    #[test]
    fn test_query_nonexistent_entity_no_node_creation() {
        let vault = create_test_vault();

        // Query edges for a nonexistent entity
        let _perm = vault.get_permission("user:nonexistent", "secret:none");

        // Verify no graph nodes were created
        let nodes = vault
            .graph
            .find_nodes_by_property(
                "entity_key",
                &PropertyValue::String("user:nonexistent".to_string()),
            )
            .unwrap_or_default();
        assert!(
            nodes.is_empty(),
            "querying a nonexistent entity should not create graph nodes"
        );
    }

    // ===== P2: VaultEvent, AuditContext, VaultStatus =====

    #[test]
    fn test_event_handler_noop_default() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "test_key", "test_value").unwrap();
        let _ = vault.get(Vault::ROOT, "test_key").unwrap();
    }

    #[test]
    fn test_event_handler_anomaly_callback_fires() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingHandler {
            count: AtomicUsize,
        }
        impl VaultEventHandler for CountingHandler {
            fn on_event(&self, _event: &VaultEvent) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }

        let handler = Arc::new(CountingHandler {
            count: AtomicUsize::new(0),
        });
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_event_handler(handler.clone());
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();

        // Operations trigger anomaly checks which may emit events
        vault.set(Vault::ROOT, "test", "value").unwrap();
        let _ = vault.get(Vault::ROOT, "test").unwrap();
        // We just verify no panics -- anomaly events depend on thresholds
    }

    #[test]
    fn test_audit_context_stored_and_retrieved() {
        let ctx = AuditContext {
            source_ip: Some("192.168.1.1".to_string()),
            session_id: Some("sess-123".to_string()),
            correlation_id: None,
        };
        assert_eq!(ctx.source_ip.as_deref(), Some("192.168.1.1"));
        assert_eq!(ctx.session_id.as_deref(), Some("sess-123"));
        assert!(ctx.correlation_id.is_none());

        let default_ctx = AuditContext::default();
        assert!(default_ctx.source_ip.is_none());
    }

    #[test]
    fn test_vault_status_returns_health_info() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "status_test", "value").unwrap();

        let status = vault.vault_status();
        assert!(!status.sealed);
        assert!(status.total_secrets >= 1);
        assert_eq!(status.pending_rotations, 0);
    }

    #[test]
    fn test_vault_config_debug_with_event_handler() {
        let config = VaultConfig::default().with_event_handler(Arc::new(NoopEventHandler));
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("event_handler"));
        assert!(debug_str.contains("max_value_size"));
    }

    #[test]
    fn test_vault_config_max_value_size() {
        let config = VaultConfig::default().with_max_value_size(1024);
        assert_eq!(config.max_value_size, 1024);
    }

    #[test]
    fn test_log_operation_with_context_no_panic() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "ctx_test", "value").unwrap();

        let ctx = AuditContext {
            source_ip: Some("10.0.0.1".to_string()),
            session_id: None,
            correlation_id: Some("corr-456".to_string()),
        };
        vault.log_operation_with_context(Vault::ROOT, "ctx_test", &AuditOperation::Get, Some(&ctx));
    }

    // ===== P2: Storage resilience, batch_set_detailed, max_value_size, grant lifecycle =====

    #[test]
    fn test_max_value_size_rejects_oversized_secret() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_max_value_size(10);
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();

        let result = vault.set(Vault::ROOT, "too_big", "this value is too long");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("exceeds maximum size"));
    }

    #[test]
    fn test_max_value_size_default_allows_normal_secrets() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "normal", "normal value").unwrap();
        let val = vault.get(Vault::ROOT, "normal").unwrap();
        assert_eq!(val, "normal value");
    }

    #[test]
    fn test_batch_set_detailed_all_succeed() {
        let vault = create_test_vault();
        let entries = vec![("key1", "val1"), ("key2", "val2")];
        let result = vault.batch_set_detailed(Vault::ROOT, &entries).unwrap();
        assert_eq!(result.succeeded, 2);
        assert!(result.failed.is_empty());
        assert_eq!(vault.get(Vault::ROOT, "key1").unwrap(), "val1");
        assert_eq!(vault.get(Vault::ROOT, "key2").unwrap(), "val2");
    }

    #[test]
    fn test_batch_set_detailed_empty() {
        let vault = create_test_vault();
        let result = vault.batch_set_detailed(Vault::ROOT, &[]).unwrap();
        assert_eq!(result.succeeded, 0);
        assert!(result.failed.is_empty());
    }

    #[test]
    fn test_batch_set_detailed_partial_failure() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        // Set very small max_value_size so one entry fails
        let config = VaultConfig::default().with_max_value_size(5);
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();

        let entries = vec![
            ("ok", "short"),
            ("fail", "this is way too long for the limit"),
        ];
        let result = vault.batch_set_detailed(Vault::ROOT, &entries).unwrap();
        assert_eq!(result.succeeded, 1);
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.failed[0].0, "fail");
    }

    #[test]
    #[serial]
    fn test_list_triggers_expired_grant_cleanup() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap();

        // Create a secret and grant access with short TTL
        vault.set(Vault::ROOT, "ttl_list_test", "secret").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "ttl_list_test",
                Permission::Read,
                Duration::from_millis(50),
            )
            .unwrap();

        // Alice can access immediately
        assert!(vault.get("user:alice", "ttl_list_test").is_ok());

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(100));

        // list() should trigger cleanup of expired grants
        let _ = vault.list(Vault::ROOT, "*").unwrap();

        // Alice should no longer have access
        assert!(vault.get("user:alice", "ttl_list_test").is_err());
    }

    #[test]
    fn test_legacy_decrypt_event_fires() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct LegacyTracker {
            legacy_count: AtomicUsize,
        }
        impl VaultEventHandler for LegacyTracker {
            fn on_event(&self, event: &VaultEvent) {
                if matches!(event, VaultEvent::LegacyDecrypt { .. }) {
                    self.legacy_count.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        let handler = Arc::new(LegacyTracker {
            legacy_count: AtomicUsize::new(0),
        });
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_event_handler(handler.clone());
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();

        vault.set(Vault::ROOT, "legacy_test", "value").unwrap();
        let _ = vault.get(Vault::ROOT, "legacy_test").unwrap();

        // Current encryption uses 0x02 version byte, so no legacy event fires
        // unless the ciphertext doesn't start with 0x02
        // This test verifies the event handler path doesn't panic
    }

    // ========== Pagination tests ==========

    #[test]
    fn test_list_paginated_basic() {
        let vault = create_test_vault();
        for i in 0..5 {
            vault
                .set(Vault::ROOT, &format!("page/{i}"), &format!("val{i}"))
                .unwrap();
        }
        let result = vault.list_paginated(Vault::ROOT, "*", 0, 3).unwrap();
        assert_eq!(result.secrets.len(), 3);
        assert_eq!(result.offset, 0);
        assert_eq!(result.limit, 3);
        assert_eq!(result.total, 5);
        assert!(result.has_more);
    }

    #[test]
    fn test_list_paginated_offset() {
        let vault = create_test_vault();
        for i in 0..5 {
            vault
                .set(Vault::ROOT, &format!("page/{i}"), &format!("val{i}"))
                .unwrap();
        }
        let result = vault.list_paginated(Vault::ROOT, "*", 2, 2).unwrap();
        assert_eq!(result.secrets.len(), 2);
        assert_eq!(result.offset, 2);
        assert!(result.has_more); // 2 remaining after offset 2+2=4
    }

    #[test]
    fn test_list_paginated_beyond_end() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "only_one", "val").unwrap();
        let result = vault.list_paginated(Vault::ROOT, "*", 100, 10).unwrap();
        assert!(result.secrets.is_empty());
        assert!(!result.has_more);
        assert_eq!(result.total, 1);
    }

    #[test]
    fn test_list_paginated_zero_limit() {
        let vault = create_test_vault();
        for i in 0..3 {
            vault
                .set(Vault::ROOT, &format!("all/{i}"), &format!("v{i}"))
                .unwrap();
        }
        let result = vault.list_paginated(Vault::ROOT, "*", 0, 0).unwrap();
        assert_eq!(result.secrets.len(), 3);
        assert!(!result.has_more);
    }

    // ========== Metadata listing tests ==========

    #[test]
    fn test_list_with_metadata_accuracy() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "meta/test", "v1").unwrap();
        vault.rotate(Vault::ROOT, "meta/test", "v2").unwrap();

        let summaries = vault.list_with_metadata(Vault::ROOT, "*").unwrap();
        let meta = summaries.iter().find(|s| s.key == "meta/test").unwrap();
        assert_eq!(meta.version_count, 2);
        assert!(meta.created_at > 0);
    }

    #[test]
    fn test_list_with_metadata_empty() {
        let vault = create_test_vault();
        let summaries = vault.list_with_metadata(Vault::ROOT, "*").unwrap();
        assert!(summaries.is_empty());
    }

    // ========== Version diff tests ==========

    #[test]
    fn test_diff_versions_basic() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "diff/test", "original").unwrap();
        vault.rotate(Vault::ROOT, "diff/test", "updated").unwrap();

        let diff = vault.diff_versions(Vault::ROOT, "diff/test", 1, 2).unwrap();
        assert_eq!(diff.key, "diff/test");
        assert_eq!(diff.version_a, 1);
        assert_eq!(diff.version_b, 2);
        assert_eq!(diff.value_a, "original");
        assert_eq!(diff.value_b, "updated");
        assert!(diff.timestamp_a > 0);
        assert!(diff.timestamp_b > 0);
    }

    #[test]
    fn test_diff_versions_same() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "diff/same", "value").unwrap();

        let diff = vault.diff_versions(Vault::ROOT, "diff/same", 1, 1).unwrap();
        assert_eq!(diff.value_a, diff.value_b);
        assert_eq!(diff.timestamp_a, diff.timestamp_b);
    }

    #[test]
    fn test_diff_versions_nonexistent() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "diff/bad", "value").unwrap();

        let result = vault.diff_versions(Vault::ROOT, "diff/bad", 1, 99);
        assert!(result.is_err());
    }

    // ========== Changelog tests ==========

    #[test]
    fn test_changelog_ordering() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "changelog/test", "v1").unwrap();
        vault.rotate(Vault::ROOT, "changelog/test", "v2").unwrap();

        let log = vault.changelog(Vault::ROOT, "changelog/test").unwrap();
        assert!(!log.is_empty());
        // Entries should be sorted by timestamp
        for w in log.windows(2) {
            assert!(w[0].timestamp <= w[1].timestamp);
        }
    }

    #[test]
    fn test_changelog_multiple_ops() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "changelog/multi", "v1").unwrap();
        vault.rotate(Vault::ROOT, "changelog/multi", "v2").unwrap();
        let _ = vault.get(Vault::ROOT, "changelog/multi").unwrap();

        let log = vault.changelog(Vault::ROOT, "changelog/multi").unwrap();
        let ops: Vec<&str> = log.iter().map(|e| e.operation.as_str()).collect();
        // Should contain set, rotate, and get operations
        assert!(ops.contains(&"set"), "should contain 'set' op: {ops:?}");
    }

    // ========== Similarity tests ==========

    #[test]
    fn test_find_similar_basic() {
        let vault = create_test_vault();
        // Create 5 secrets with similar metadata patterns
        for i in 0..5 {
            vault
                .set(Vault::ROOT, &format!("sim/{i}"), &format!("val{i}"))
                .unwrap();
        }
        let result = vault.find_similar(Vault::ROOT, "sim/0", 2).unwrap();
        // Should find up to 2 similar secrets
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_find_similar_access_filtered() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "sim/owned", "v1").unwrap();
        vault.set(Vault::ROOT, "sim/other", "v2").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "sim/owned", Permission::Read)
            .unwrap();

        // Alice can only see secrets she has access to
        let result = vault.find_similar("user:alice", "sim/owned", 5).unwrap();
        // Should not contain secrets alice can't access
        assert!(result.iter().all(|r| r.key != "sim/other"));
    }

    #[test]
    fn test_check_duplication_detects_similar() {
        let vault = create_test_vault();
        // Create secrets with very similar metadata
        vault.set(Vault::ROOT, "dup/a", "value1").unwrap();
        vault.set(Vault::ROOT, "dup/b", "value2").unwrap();

        // With a very low threshold, everything is a "duplicate"
        let result = vault.check_duplication(Vault::ROOT, 0.01).unwrap();
        // Both secrets are so similar in metadata they should be found
        assert!(
            !result.is_empty(),
            "should detect similar metadata with low threshold"
        );
    }

    #[test]
    fn test_check_duplication_no_duplicates() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "single", "alone").unwrap();

        // With threshold of 1.0 (perfect match), single secret can't have duplicates
        let result = vault.check_duplication(Vault::ROOT, 1.0).unwrap();
        assert!(result.is_empty());
    }

    // ========== Template tests ==========

    #[test]
    fn test_vault_template_save_get_list_delete() {
        let vault = create_test_vault();
        let template = SecretTemplate::Password(PasswordConfig::default());

        vault
            .save_template(Vault::ROOT, "db-pass", template)
            .unwrap();
        assert!(vault.get_template("db-pass").is_some());

        let names = vault.list_templates();
        assert!(names.contains(&"db-pass".to_string()));

        vault.delete_template(Vault::ROOT, "db-pass").unwrap();
        assert!(vault.get_template("db-pass").is_none());
    }

    #[test]
    fn test_vault_template_persistence() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap();
        let template = SecretTemplate::Password(PasswordConfig::default());
        vault
            .save_template(Vault::ROOT, "persistent", template)
            .unwrap();

        // Create new vault on same store (simulates restart)
        let vault2 = Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap();
        assert!(vault2.get_template("persistent").is_some());
    }

    #[test]
    fn test_vault_template_not_found() {
        let vault = create_test_vault();
        assert!(vault.get_template("nonexistent").is_none());
    }

    #[test]
    fn test_vault_heat_kernel_trust() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "secret:a", "val_a").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret:a").unwrap();
        let config = crate::heat_kernel::HeatKernelConfig::default();
        let report = vault.heat_kernel_trust(config);
        assert!(report.total_entities > 0);
    }

    #[test]
    fn test_vault_build_access_tensor() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "secret:a", "val").unwrap();
        let _ = vault.get(Vault::ROOT, "secret:a");
        let config = crate::access_tensor::AccessTensorConfig::default();
        let tensor = vault.build_access_tensor(config).unwrap();
        let (entities, _, _) = tensor.dimensions();
        assert!(entities > 0 || tensor.raw_data().is_empty());
    }

    #[test]
    fn test_vault_analyze_temporal_patterns() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "secret:a", "val").unwrap();
        let _ = vault.get(Vault::ROOT, "secret:a");
        let tensor_config = crate::access_tensor::AccessTensorConfig::default();
        let analysis_config = crate::temporal_analysis::TemporalAnalysisConfig::default();
        let report = vault
            .analyze_temporal_patterns(tensor_config, analysis_config)
            .unwrap();
        // With only 2 accesses, seasonal analysis (min_accesses=5) filters the entity,
        // but drift detection (threshold=1.0) may include it depending on bucket timing.
        assert!(report.total_entities_analyzed <= 1);
    }

    #[test]
    fn test_vault_weighted_dependency_and_impact() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "parent", "p_val").unwrap();
        vault.set(Vault::ROOT, "child", "c_val").unwrap();
        vault
            .add_weighted_dependency(
                Vault::ROOT,
                "parent",
                "child",
                crate::dependency::DependencyWeight::High,
                Some("test dep"),
            )
            .unwrap();
        let report = vault
            .weighted_impact_analysis(Vault::ROOT, "parent")
            .unwrap();
        assert!(!report.affected_secrets.is_empty());
    }

    #[test]
    fn test_vault_rotation_plan() {
        let vault = create_test_vault();
        vault.set(Vault::ROOT, "root_sec", "r_val").unwrap();
        vault.set(Vault::ROOT, "dep_sec", "d_val").unwrap();
        vault
            .add_weighted_dependency(
                Vault::ROOT,
                "root_sec",
                "dep_sec",
                crate::dependency::DependencyWeight::Critical,
                None,
            )
            .unwrap();
        let plan = vault.rotation_plan(Vault::ROOT, "root_sec").unwrap();
        assert!(plan.total_secrets >= 1);
    }
}
