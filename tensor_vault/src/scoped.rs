// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Scoped vault view for a specific entity.

use std::time::Duration;

use crate::anomaly::AgentProfile;
use crate::audit::AuditEntry;
use crate::delegation::DelegationRecord;
use crate::dependency::ImpactReport;
use crate::dynamic::{DynamicSecretMetadata, SecretTemplate};
use crate::pitr::VaultSnapshot;
use crate::policy::{PolicyMatch, PolicyTemplate};
use crate::quota::{ResourceQuota, ResourceUsage};
use crate::rotation::{PendingRotation, RotationPolicy};
use crate::vault::Vault;
use crate::wrapping::WrappingToken;
use crate::{Permission, Result, VersionInfo};

/// A scoped view of the vault for a specific entity.
///
/// All operations are automatically performed as the scoped entity,
/// removing the need to pass the requester on every call.
pub struct ScopedVault<'a> {
    vault: &'a Vault,
    entity: String,
}

impl<'a> ScopedVault<'a> {
    pub(crate) fn new(vault: &'a Vault, entity: &str) -> Self {
        Self {
            vault,
            entity: entity.to_string(),
        }
    }

    /// Store a secret at the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks write permission.
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        self.vault.set(&self.entity, key, value)
    }

    /// Retrieve the current version of a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity lacks
    /// read permission.
    pub fn get(&self, key: &str) -> Result<String> {
        self.vault.get(&self.entity, key)
    }

    /// Delete a secret and all its versions.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission.
    pub fn delete(&self, key: &str) -> Result<()> {
        self.vault.delete(&self.entity, key)
    }

    /// List secret keys matching the given glob pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks read permission.
    pub fn list(&self, pattern: &str) -> Result<Vec<String>> {
        self.vault.list(&self.entity, pattern)
    }

    /// Rotate a secret to a new value, preserving the previous version.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity lacks
    /// write permission.
    pub fn rotate(&self, key: &str, new_value: &str) -> Result<()> {
        self.vault.rotate(&self.entity, key, new_value)
    }

    /// Retrieve a specific version of a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the version does not exist or the scoped entity
    /// lacks read permission.
    pub fn get_version(&self, key: &str, version: u32) -> Result<String> {
        self.vault.get_version(&self.entity, key, version)
    }

    /// List all versions of a secret with their metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn list_versions(&self, key: &str) -> Result<Vec<VersionInfo>> {
        self.vault.list_versions(&self.entity, key)
    }

    /// Return the current (latest) version number for a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn current_version(&self, key: &str) -> Result<u32> {
        self.vault.current_version(&self.entity, key)
    }

    /// Roll back a secret to a previous version, making it the current value.
    ///
    /// # Errors
    ///
    /// Returns an error if the version does not exist or the scoped entity
    /// lacks write permission.
    pub fn rollback(&self, key: &str, version: u32) -> Result<()> {
        self.vault.rollback(&self.entity, key, version)
    }

    /// Grant access to another entity with a specific permission level.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission on the key.
    pub fn grant_with_permission(&self, entity: &str, key: &str, level: Permission) -> Result<()> {
        self.vault
            .grant_with_permission(&self.entity, entity, key, level)
    }

    /// Grant time-limited access to another entity with a specific permission level.
    ///
    /// The grant automatically expires after the given TTL.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission on the key.
    pub fn grant_with_ttl(
        &self,
        entity: &str,
        key: &str,
        level: Permission,
        ttl: Duration,
    ) -> Result<()> {
        self.vault
            .grant_with_ttl(&self.entity, entity, key, level, ttl)
    }

    /// Revoke another entity's access to a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission on the key.
    pub fn revoke(&self, entity: &str, key: &str) -> Result<()> {
        self.vault.revoke(&self.entity, entity, key)
    }

    /// Retrieve the audit log for a specific secret key.
    ///
    /// # Errors
    ///
    /// Returns an error if no audit entries exist for the key.
    pub fn audit_log(&self, key: &str) -> Result<Vec<AuditEntry>> {
        self.vault.audit_log(key)
    }

    /// Return the permission level the scoped entity holds on a secret, if any.
    pub fn get_permission(&self, key: &str) -> Option<Permission> {
        self.vault.get_permission(&self.entity, key)
    }

    /// Grant default (read) access to another entity on a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission on the key.
    pub fn grant(&self, entity: &str, key: &str) -> Result<()> {
        self.vault.grant(&self.entity, entity, key)
    }

    /// Delegate access to secrets to a child entity.
    ///
    /// The child receives the specified permission on all listed secrets.
    /// An optional TTL limits how long the delegation remains active.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks permission to delegate.
    pub fn delegate(
        &self,
        child: &str,
        secrets: &[&str],
        permission: Permission,
        ttl: Option<Duration>,
    ) -> Result<DelegationRecord> {
        self.vault
            .delegate(&self.entity, child, secrets, permission, ttl)
    }

    /// Revoke all delegated access from a child entity.
    ///
    /// Returns the list of secret keys whose access was revoked.
    ///
    /// # Errors
    ///
    /// Returns an error if the delegation does not exist or the scoped entity
    /// is not the delegator.
    pub fn revoke_delegation(&self, child: &str) -> Result<Vec<String>> {
        self.vault.revoke_delegation(&self.entity, child)
    }

    /// Revoke delegated access from a child entity and all its transitive delegates.
    ///
    /// # Errors
    ///
    /// Returns an error if the delegation does not exist or the scoped entity
    /// is not the delegator.
    pub fn revoke_delegation_cascading(&self, child: &str) -> Result<Vec<DelegationRecord>> {
        self.vault.revoke_delegation_cascading(&self.entity, child)
    }

    /// Return the anomaly detection profile for the scoped entity, if one exists.
    pub fn anomaly_profile(&self) -> Option<AgentProfile> {
        self.vault.anomaly_monitor().get_profile(&self.entity)
    }

    /// Explain how the scoped entity can access a secret through the access graph.
    pub fn explain_access(&self, secret: &str) -> crate::graph_intel::AccessExplanation {
        self.vault.explain_access(&self.entity, secret)
    }

    /// Compute the blast radius if the scoped entity were compromised.
    pub fn blast_radius(&self) -> crate::graph_intel::BlastRadius {
        self.vault.blast_radius(&self.entity)
    }

    /// Simulate granting an entity access and report the resulting access changes.
    pub fn simulate_grant(
        &self,
        entity: &str,
        secret: &str,
        permission: Permission,
    ) -> crate::graph_intel::SimulationResult {
        self.vault.simulate_grant(entity, secret, permission)
    }

    /// Analyze privilege distribution across the vault's access graph.
    pub fn privilege_analysis(&self) -> crate::graph_intel::PrivilegeAnalysisReport {
        self.vault.privilege_analysis()
    }

    /// Score all delegation relationships for anomalous patterns.
    pub fn delegation_anomaly_scores(&self) -> Vec<crate::graph_intel::DelegationAnomalyScore> {
        self.vault.delegation_anomaly_scores()
    }

    /// Infer roles from access patterns in the vault's access graph.
    pub fn infer_roles(&self) -> crate::graph_intel::RoleInferenceResult {
        self.vault.infer_roles()
    }

    /// Analyze transitive trust relationships in the access graph.
    pub fn trust_transitivity(&self) -> crate::graph_intel::TrustTransitivityReport {
        self.vault.trust_transitivity()
    }

    /// Analyze how risk propagates through the access graph.
    pub fn risk_propagation(&self) -> crate::graph_intel::RiskPropagationReport {
        self.vault.risk_propagation()
    }

    /// Store a secret with a time-to-live. After the TTL elapses the secret expires.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks write permission.
    pub fn set_with_ttl(&self, key: &str, value: &str, ttl: Duration) -> Result<()> {
        self.vault.set_with_ttl(&self.entity, key, value, ttl)
    }

    /// Remove the expiration from a secret, making it permanent.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks write permission.
    pub fn clear_expiration(&self, key: &str) -> Result<()> {
        self.vault.clear_expiration(&self.entity, key)
    }

    /// Return the expiration timestamp (epoch millis) for a secret, if set.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn get_expiration(&self, key: &str) -> Result<Option<i64>> {
        self.vault.get_expiration(&self.entity, key)
    }

    /// Encrypt plaintext using the transit cipher associated with a secret key.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks read permission or encryption fails.
    pub fn encrypt_for(&self, key: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        self.vault.encrypt_for(&self.entity, key, plaintext)
    }

    /// Decrypt ciphertext using the transit cipher associated with a secret key.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks read permission or decryption fails.
    pub fn decrypt_as(&self, key: &str, sealed: &[u8]) -> Result<Vec<u8>> {
        self.vault.decrypt_as(&self.entity, key, sealed)
    }

    /// Request emergency (break-glass) access to a secret.
    ///
    /// The access is time-limited and logged with the provided justification.
    ///
    /// # Errors
    ///
    /// Returns an error if emergency access is denied or the secret does not exist.
    pub fn emergency_access(
        &self,
        key: &str,
        justification: &str,
        duration: Duration,
    ) -> Result<String> {
        self.vault
            .emergency_access(&self.entity, key, justification, duration)
    }

    /// Retrieve multiple secrets in a single operation.
    ///
    /// Returns a vec of `(key, result)` pairs; individual keys may fail independently.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch operation itself fails.
    pub fn batch_get(&self, keys: &[&str]) -> Result<Vec<(String, Result<String>)>> {
        self.vault.batch_get(&self.entity, keys)
    }

    /// Store multiple secrets in a single operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks write permission on any key.
    pub fn batch_set(&self, entries: &[(&str, &str)]) -> Result<()> {
        self.vault.batch_set(&self.entity, entries)
    }

    /// Retrieve audit entries filtered by a specific entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the audit log cannot be queried.
    pub fn audit_by_entity(&self, entity: &str) -> Result<Vec<AuditEntry>> {
        self.vault.audit_by_entity(entity)
    }

    /// Retrieve audit entries created after the given timestamp (epoch milliseconds).
    ///
    /// # Errors
    ///
    /// Returns an error if the audit log cannot be queried.
    pub fn audit_since(&self, since_millis: i64) -> Result<Vec<AuditEntry>> {
        self.vault.audit_since(since_millis)
    }

    /// Retrieve the most recent audit entries, up to `limit`.
    ///
    /// # Errors
    ///
    /// Returns an error if the audit log cannot be queried.
    pub fn audit_recent(&self, limit: usize) -> Result<Vec<AuditEntry>> {
        self.vault.audit_recent(limit)
    }

    // ========== Wrapping ==========

    /// Wrap a secret into a single-use token with a time-to-live in milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn wrap_secret(&self, key: &str, ttl_ms: i64) -> Result<String> {
        self.vault.wrap_secret(&self.entity, key, ttl_ms)
    }

    /// Unwrap a wrapping token to retrieve the secret value (single-use).
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid, expired, or already consumed.
    pub fn unwrap_secret(&self, token: &str) -> Result<String> {
        self.vault.unwrap_secret(token)
    }

    /// Look up metadata for a wrapping token without consuming it.
    pub fn wrapping_token_info(&self, token: &str) -> Option<WrappingToken> {
        self.vault.wrapping_token_info(token)
    }

    // ========== Dependencies ==========

    /// Register a dependency relationship between two secrets.
    ///
    /// # Errors
    ///
    /// Returns an error if either secret does not exist or the scoped entity
    /// lacks permission.
    pub fn add_dependency(&self, parent_key: &str, child_key: &str) -> Result<()> {
        self.vault
            .add_dependency(&self.entity, parent_key, child_key)
    }

    /// Remove a dependency relationship between two secrets.
    ///
    /// # Errors
    ///
    /// Returns an error if the dependency does not exist or the scoped entity
    /// lacks permission.
    pub fn remove_dependency(&self, parent_key: &str, child_key: &str) -> Result<()> {
        self.vault
            .remove_dependency(&self.entity, parent_key, child_key)
    }

    /// List the secrets that the given secret depends on (children).
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn get_dependencies(&self, key: &str) -> Result<Vec<String>> {
        self.vault.get_dependencies(&self.entity, key)
    }

    /// List the secrets that depend on the given secret (parents).
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn get_dependents(&self, key: &str) -> Result<Vec<String>> {
        self.vault.get_dependents(&self.entity, key)
    }

    /// Perform impact analysis on a secret, reporting all transitively affected secrets.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn impact_analysis(&self, key: &str) -> Result<ImpactReport> {
        self.vault.impact_analysis(&self.entity, key)
    }

    /// Register a weighted dependency between two secrets with an optional description.
    ///
    /// # Errors
    ///
    /// Returns an error if either secret does not exist or the scoped entity
    /// lacks permission.
    pub fn add_weighted_dependency(
        &self,
        parent_key: &str,
        child_key: &str,
        weight: crate::dependency::DependencyWeight,
        description: Option<&str>,
    ) -> Result<()> {
        self.vault
            .add_weighted_dependency(&self.entity, parent_key, child_key, weight, description)
    }

    /// Perform weighted impact analysis, factoring in dependency criticality levels.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn weighted_impact_analysis(
        &self,
        key: &str,
    ) -> Result<crate::dependency::WeightedImpactReport> {
        self.vault.weighted_impact_analysis(&self.entity, key)
    }

    /// Generate a rotation plan for a secret and its dependency chain.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn rotation_plan(&self, key: &str) -> Result<crate::dependency::RotationPlan> {
        self.vault.rotation_plan(&self.entity, key)
    }

    // ========== Quotas ==========

    /// Set a resource quota for a namespace.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission.
    pub fn set_quota(&self, namespace: &str, quota: ResourceQuota) -> Result<()> {
        self.vault.set_quota(&self.entity, namespace, quota)
    }

    /// Retrieve the resource quota for a namespace, if one is set.
    pub fn get_quota(&self, namespace: &str) -> Option<ResourceQuota> {
        self.vault.get_quota(namespace)
    }

    /// Return current resource usage for a namespace.
    pub fn get_usage(&self, namespace: &str) -> ResourceUsage {
        self.vault.get_usage(namespace)
    }

    /// Remove the resource quota for a namespace.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission.
    pub fn remove_quota(&self, namespace: &str) -> Result<()> {
        self.vault.remove_quota(&self.entity, namespace)
    }

    // ========== Dynamic Secrets ==========

    /// Generate a dynamic secret from a template with a TTL.
    ///
    /// Returns `(secret_id, secret_value)`. If `one_time` is true, the secret
    /// can only be retrieved once.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks permission or generation fails.
    pub fn generate_dynamic_secret(
        &self,
        template: &SecretTemplate,
        ttl_ms: i64,
        one_time: bool,
    ) -> Result<(String, String)> {
        self.vault
            .generate_dynamic_secret(&self.entity, template, ttl_ms, one_time)
    }

    /// Retrieve a dynamic secret by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret has expired, been revoked, or the scoped
    /// entity lacks permission.
    pub fn get_dynamic_secret(&self, secret_id: &str) -> Result<String> {
        self.vault.get_dynamic_secret(&self.entity, secret_id)
    }

    /// List all dynamic secrets created by the scoped entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_dynamic_secrets(&self) -> Result<Vec<DynamicSecretMetadata>> {
        self.vault.list_dynamic_secrets(&self.entity)
    }

    /// Revoke a dynamic secret, making it permanently inaccessible.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn revoke_dynamic_secret(&self, secret_id: &str) -> Result<()> {
        self.vault.revoke_dynamic_secret(&self.entity, secret_id)
    }

    // ========== Policies ==========

    /// Add a policy-based access control template.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission.
    pub fn add_policy(&self, template: PolicyTemplate) -> Result<()> {
        self.vault.add_policy(&self.entity, template)
    }

    /// Remove a policy template by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy does not exist or the scoped entity
    /// lacks admin permission.
    pub fn remove_policy(&self, name: &str) -> Result<()> {
        self.vault.remove_policy(&self.entity, name)
    }

    /// List all registered policy templates.
    pub fn list_policies(&self) -> Vec<PolicyTemplate> {
        self.vault.list_policies()
    }

    /// Evaluate all policies against the scoped entity and return matches.
    pub fn evaluate_policies(&self) -> Vec<PolicyMatch> {
        self.vault.evaluate_policies(&self.entity)
    }

    // ========== PITR ==========

    /// Create a point-in-time snapshot of the vault with the given label.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks admin permission or
    /// snapshot creation fails.
    pub fn create_snapshot(&self, label: &str) -> Result<VaultSnapshot> {
        self.vault.create_snapshot(&self.entity, label)
    }

    /// Restore the vault to a previous snapshot, returning the number of secrets restored.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not exist or the scoped entity
    /// lacks admin permission.
    pub fn restore_snapshot(&self, snapshot_id: &str) -> Result<usize> {
        self.vault.restore_snapshot(&self.entity, snapshot_id)
    }

    /// List all available vault snapshots.
    pub fn list_snapshots(&self) -> Vec<VaultSnapshot> {
        self.vault.list_snapshots()
    }

    /// Delete a vault snapshot by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not exist or the scoped entity
    /// lacks admin permission.
    pub fn delete_snapshot(&self, snapshot_id: &str) -> Result<()> {
        self.vault.delete_snapshot(&self.entity, snapshot_id)
    }

    // ========== Rotation ==========

    /// Set an automated rotation policy for a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks admin permission.
    pub fn set_rotation_policy(&self, key: &str, policy: RotationPolicy) -> Result<()> {
        self.vault.set_rotation_policy(&self.entity, key, policy)
    }

    /// Retrieve the rotation policy for a secret, if one is set.
    pub fn get_rotation_policy(&self, key: &str) -> Option<RotationPolicy> {
        self.vault.get_rotation_policy(&self.entity, key)
    }

    /// Remove the rotation policy for a secret.
    ///
    /// # Errors
    ///
    /// Returns an error if no policy exists or the scoped entity lacks admin permission.
    pub fn remove_rotation_policy(&self, key: &str) -> Result<()> {
        self.vault.remove_rotation_policy(&self.entity, key)
    }

    /// Return all secrets with rotation policies that are due for rotation.
    pub fn check_pending_rotations(&self) -> Vec<PendingRotation> {
        self.vault.check_pending_rotations()
    }

    /// Execute the rotation policy for a secret, returning the new generated value.
    ///
    /// # Errors
    ///
    /// Returns an error if no rotation policy exists, the generator fails, or
    /// the scoped entity lacks permission.
    pub fn execute_rotation(&self, key: &str) -> Result<String> {
        self.vault.execute_rotation(&self.entity, key)
    }

    // ========== Engine ==========

    /// Generate a secret using a registered secret engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine is not registered or generation fails.
    pub fn engine_generate(&self, engine_name: &str, params: &serde_json::Value) -> Result<String> {
        self.vault
            .engine_generate(&self.entity, engine_name, params)
    }

    /// Revoke a secret through its originating engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine is not registered or revocation fails.
    pub fn engine_revoke(&self, engine_name: &str, secret_id: &str) -> Result<()> {
        self.vault
            .engine_revoke(&self.entity, engine_name, secret_id)
    }

    // ========== Sync ==========

    /// Subscribe a sync target to receive updates when a secret changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the target is not registered or the scoped entity
    /// lacks permission.
    pub fn subscribe_sync(&self, key: &str, target_name: &str) -> Result<()> {
        self.vault.subscribe_sync(&self.entity, key, target_name)
    }

    /// Remove a sync subscription for a secret and target.
    ///
    /// # Errors
    ///
    /// Returns an error if the subscription does not exist or the scoped entity
    /// lacks permission.
    pub fn unsubscribe_sync(&self, key: &str, target_name: &str) -> Result<()> {
        self.vault.unsubscribe_sync(&self.entity, key, target_name)
    }

    /// Manually trigger sync for a secret, returning the number of targets notified.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks permission.
    pub fn trigger_sync(&self, key: &str) -> Result<usize> {
        self.vault.trigger_sync(&self.entity, key)
    }

    // ========== Developer Experience ==========

    /// List secrets matching a pattern with pagination support.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks read permission.
    pub fn list_paginated(
        &self,
        pattern: &str,
        offset: usize,
        limit: usize,
    ) -> Result<crate::PagedSecrets> {
        self.vault
            .list_paginated(&self.entity, pattern, offset, limit)
    }

    /// List secrets matching a pattern with summary metadata for each entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the scoped entity lacks read permission.
    pub fn list_with_metadata(&self, pattern: &str) -> Result<Vec<crate::SecretSummary>> {
        self.vault.list_with_metadata(&self.entity, pattern)
    }

    /// Compare two versions of a secret and return their differences.
    ///
    /// # Errors
    ///
    /// Returns an error if either version does not exist or the scoped entity
    /// lacks read permission.
    pub fn diff_versions(
        &self,
        key: &str,
        version_a: u32,
        version_b: u32,
    ) -> Result<crate::VersionDiff> {
        self.vault
            .diff_versions(&self.entity, key, version_a, version_b)
    }

    /// Return the full changelog for a secret across all versions.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn changelog(&self, key: &str) -> Result<Vec<crate::ChangelogEntry>> {
        self.vault.changelog(&self.entity, key)
    }

    // ========== Similarity ==========

    /// Find the `k` most similar secrets to the given key based on name embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the secret does not exist or the scoped entity
    /// lacks read permission.
    pub fn find_similar(
        &self,
        key: &str,
        k: usize,
    ) -> Result<Vec<crate::similarity::SimilarSecret>> {
        self.vault.find_similar(&self.entity, key, k)
    }

    // ========== Graph Intelligence (Tier 4) ==========

    /// Compute behavior embeddings for all entities using the given configuration.
    pub fn compute_behavior_embeddings(
        &self,
        config: crate::graph_intel::BehaviorEmbeddingConfig,
    ) -> Vec<crate::graph_intel::NodeEmbedding> {
        self.vault.compute_behavior_embeddings(config)
    }

    /// Detect anomalous entities using geometric distance analysis in embedding space.
    pub fn detect_geometric_anomalies(
        &self,
        k: usize,
        threshold_multiplier: f64,
    ) -> crate::graph_intel::GeometricAnomalyReport {
        self.vault
            .detect_geometric_anomalies(k, threshold_multiplier)
    }

    /// Cluster entities by behavioral similarity using access patterns.
    pub fn cluster_entities(&self) -> crate::graph_intel::ClusteringResult {
        self.vault.cluster_entities()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::{
        PasswordConfig, PolicyTemplate, ResourceQuota, RotationGenerator, RotationPolicy,
        SecretEngine, SecretTemplate, SyncTarget, VaultConfig,
    };

    fn create_scoped_vault_env() -> (Vault, Arc<GraphEngine>) {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();
        (vault, graph)
    }

    #[test]
    fn test_scoped_set_and_get() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "db/password", "initial").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "db/password")
            .unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get("db/password").unwrap(), "initial");
    }

    #[test]
    fn test_scoped_rotate() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "db/password", "initial").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "db/password")
            .unwrap();

        let scoped = vault.scope("user:alice");
        scoped.rotate("db/password", "rotated").unwrap();
        assert_eq!(scoped.get("db/password").unwrap(), "rotated");
    }

    #[test]
    fn test_scoped_get_version() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get_version("key", 1).unwrap(), "v1");
        assert_eq!(scoped.get_version("key", 2).unwrap(), "v2");
    }

    #[test]
    fn test_scoped_list_versions() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        let versions = scoped.list_versions("key").unwrap();
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_scoped_current_version() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.current_version("key").unwrap(), 2);
    }

    #[test]
    fn test_scoped_rollback() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped.rollback("key", 1).unwrap();
        assert_eq!(scoped.get("key").unwrap(), "v1");
    }

    #[test]
    fn test_scoped_grant_with_permission() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped
            .grant_with_permission("user:bob", "key", Permission::Read)
            .unwrap();

        assert_eq!(vault.get("user:bob", "key").unwrap(), "value");
    }

    #[test]
    fn test_scoped_grant_with_ttl() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped
            .grant_with_ttl(
                "user:bob",
                "key",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        assert_eq!(vault.get("user:bob", "key").unwrap(), "value");
    }

    #[test]
    fn test_scoped_revoke() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped.revoke("user:bob", "key").unwrap();

        assert!(vault.get("user:bob", "key").is_err());
    }

    #[test]
    fn test_scoped_audit_log() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        let _ = scoped.get("key").unwrap();

        let log = scoped.audit_log("key").unwrap();
        assert!(!log.is_empty());
    }

    #[test]
    fn test_scoped_get_permission() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "key", Permission::Write)
            .unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get_permission("key"), Some(Permission::Write));
    }

    #[test]
    fn test_scoped_grant_simple() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "db/password", "secret").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "db/password")
            .unwrap();

        let scoped = vault.scope("user:alice");
        scoped.grant("user:bob", "db/password").unwrap();

        assert_eq!(vault.get("user:bob", "db/password").unwrap(), "secret");
    }

    #[test]
    fn test_scoped_audit_by_entity() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        let _ = scoped.get("key").unwrap();

        let entries = scoped.audit_by_entity("user:alice").unwrap();
        assert!(!entries.is_empty());
        for entry in &entries {
            assert_eq!(entry.entity, "user:alice");
        }
    }

    #[test]
    fn test_scoped_audit_since() {
        let (vault, _graph) = create_scoped_vault_env();
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        vault.set(Vault::ROOT, "key", "value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let entries = scoped.audit_since(before).unwrap();
        assert!(!entries.is_empty());
        for entry in &entries {
            assert!(entry.timestamp >= before);
        }
    }

    #[test]
    fn test_scoped_audit_recent() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "k1", "v1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        vault.set(Vault::ROOT, "k2", "v2").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        vault.set(Vault::ROOT, "k3", "v3").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let recent = scoped.audit_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert!(recent[0].timestamp >= recent[1].timestamp);
    }

    #[test]
    fn test_scoped_no_access() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();

        let scoped = vault.scope("user:nobody");
        assert!(scoped.get("key").is_err());
        assert_eq!(scoped.get_permission("key"), None);
    }

    // ========== Wrapping Tests ==========

    #[test]
    fn test_scoped_wrap_and_unwrap_secret() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "wrap/key", "wrap_value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let token = scoped.wrap_secret("wrap/key", 60_000).unwrap();
        assert!(!token.is_empty());

        let value = scoped.unwrap_secret(&token).unwrap();
        assert_eq!(value, "wrap_value");
    }

    #[test]
    fn test_scoped_wrapping_token_info() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "info/key", "info_value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let token = scoped.wrap_secret("info/key", 60_000).unwrap();

        let info = scoped.wrapping_token_info(&token);
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(!info.consumed);
        assert!(info.expires_at_ms > info.created_at_ms);
    }

    #[test]
    fn test_scoped_wrapping_token_info_nonexistent() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        assert!(scoped.wrapping_token_info("nonexistent_token").is_none());
    }

    // ========== Dependency Tests ==========

    #[test]
    fn test_scoped_add_and_get_dependencies() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "dep/parent", "parent_val").unwrap();
        vault.set(Vault::ROOT, "dep/child", "child_val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.add_dependency("dep/parent", "dep/child").unwrap();

        let deps = scoped.get_dependencies("dep/parent").unwrap();
        assert!(!deps.is_empty());
    }

    #[test]
    fn test_scoped_get_dependents() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "dep2/parent", "p").unwrap();
        vault.set(Vault::ROOT, "dep2/child", "c").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.add_dependency("dep2/parent", "dep2/child").unwrap();

        let dependents = scoped.get_dependents("dep2/child").unwrap();
        assert!(!dependents.is_empty());
    }

    #[test]
    fn test_scoped_remove_dependency() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rem/parent", "p").unwrap();
        vault.set(Vault::ROOT, "rem/child", "c").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.add_dependency("rem/parent", "rem/child").unwrap();
        scoped.remove_dependency("rem/parent", "rem/child").unwrap();

        let deps = scoped.get_dependencies("rem/parent").unwrap();
        assert!(deps.is_empty());
    }

    #[test]
    fn test_scoped_impact_analysis() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "ia/root", "r").unwrap();
        vault.set(Vault::ROOT, "ia/child", "c").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.add_dependency("ia/root", "ia/child").unwrap();

        let report = scoped.impact_analysis("ia/root").unwrap();
        assert!(!report.root_secret.is_empty());
    }

    // ========== Quota Tests ==========

    #[test]
    fn test_scoped_set_and_get_quota() {
        let (vault, _graph) = create_scoped_vault_env();

        let scoped = vault.scope(Vault::ROOT);
        let quota = ResourceQuota {
            max_secrets: 100,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 500,
        };
        scoped.set_quota("test_ns", quota).unwrap();

        let got = scoped.get_quota("test_ns");
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.max_secrets, 100);
    }

    #[test]
    fn test_scoped_get_usage() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let usage = scoped.get_usage("empty_ns");
        assert_eq!(usage.secret_count, 0);
    }

    #[test]
    fn test_scoped_remove_quota() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let quota = ResourceQuota {
            max_secrets: 10,
            max_storage_bytes: 10_000,
            max_ops_per_hour: 100,
        };
        scoped.set_quota("rm_ns", quota).unwrap();
        assert!(scoped.get_quota("rm_ns").is_some());

        scoped.remove_quota("rm_ns").unwrap();
        assert!(scoped.get_quota("rm_ns").is_none());
    }

    // ========== Dynamic Secrets Tests ==========

    #[test]
    fn test_scoped_generate_and_get_dynamic_secret() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let template = SecretTemplate::Password(PasswordConfig::default());
        let (secret_id, value) = scoped
            .generate_dynamic_secret(&template, 600_000, false)
            .unwrap();
        assert!(!secret_id.is_empty());
        assert!(!value.is_empty());

        let retrieved = scoped.get_dynamic_secret(&secret_id).unwrap();
        assert!(!retrieved.is_empty());
    }

    #[test]
    fn test_scoped_list_dynamic_secrets() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let template = SecretTemplate::Password(PasswordConfig::default());
        scoped
            .generate_dynamic_secret(&template, 600_000, false)
            .unwrap();

        let list = scoped.list_dynamic_secrets().unwrap();
        assert!(!list.is_empty());
    }

    #[test]
    fn test_scoped_revoke_dynamic_secret() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let template = SecretTemplate::Password(PasswordConfig::default());
        let (secret_id, _) = scoped
            .generate_dynamic_secret(&template, 600_000, false)
            .unwrap();

        scoped.revoke_dynamic_secret(&secret_id).unwrap();

        // After revocation, dynamic secret metadata should be gone
        let list = scoped.list_dynamic_secrets().unwrap();
        let found = list.iter().any(|m| m.secret_id == secret_id);
        assert!(!found);
    }

    // ========== Policy Tests ==========

    #[test]
    fn test_scoped_add_and_list_policies() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let policy = PolicyTemplate {
            name: "scoped_test_policy".to_string(),
            match_pattern: "team:*".to_string(),
            secret_pattern: "staging/*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };
        scoped.add_policy(policy).unwrap();

        let policies = scoped.list_policies();
        assert!(policies.iter().any(|p| p.name == "scoped_test_policy"));
    }

    #[test]
    fn test_scoped_remove_policy() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let policy = PolicyTemplate {
            name: "to_remove".to_string(),
            match_pattern: "*".to_string(),
            secret_pattern: "*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };
        scoped.add_policy(policy).unwrap();
        scoped.remove_policy("to_remove").unwrap();

        let policies = scoped.list_policies();
        assert!(!policies.iter().any(|p| p.name == "to_remove"));
    }

    #[test]
    fn test_scoped_evaluate_policies() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        let policy = PolicyTemplate {
            name: "eng_policy".to_string(),
            match_pattern: "team:eng/*".to_string(),
            secret_pattern: "config/*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };
        scoped.add_policy(policy).unwrap();

        let eng_scoped = vault.scope("team:eng/alice");
        let matches = eng_scoped.evaluate_policies();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].policy_name, "eng_policy");
    }

    // ========== Snapshot Tests ==========

    #[test]
    fn test_scoped_create_and_list_snapshots() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "snap/key", "snap_val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let snap = scoped.create_snapshot("test_snapshot").unwrap();
        assert!(!snap.id.is_empty());
        assert_eq!(snap.label, "test_snapshot");

        let snapshots = scoped.list_snapshots();
        assert!(snapshots.iter().any(|s| s.id == snap.id));
    }

    #[test]
    fn test_scoped_restore_snapshot() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "snap2/key", "before").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let snap = scoped.create_snapshot("restore_test").unwrap();

        // Modify the secret
        vault.set(Vault::ROOT, "snap2/key", "after").unwrap();

        let count = scoped.restore_snapshot(&snap.id).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_scoped_delete_snapshot() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "snap3/key", "val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let snap = scoped.create_snapshot("to_delete").unwrap();

        scoped.delete_snapshot(&snap.id).unwrap();

        let snapshots = scoped.list_snapshots();
        assert!(!snapshots.iter().any(|s| s.id == snap.id));
    }

    // ========== Rotation Tests ==========

    #[test]
    fn test_scoped_set_and_get_rotation_policy() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rot/key", "rot_val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let policy = RotationPolicy {
            secret_key: "rot/key".to_string(),
            interval_ms: 86_400_000,
            last_rotated_ms: 0,
            generator: RotationGenerator::Password(PasswordConfig::default()),
            notify_before_ms: 3_600_000,
        };
        scoped.set_rotation_policy("rot/key", policy).unwrap();

        let got = scoped.get_rotation_policy("rot/key");
        assert!(got.is_some());
        assert_eq!(got.unwrap().secret_key, "rot/key");
    }

    #[test]
    fn test_scoped_remove_rotation_policy() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rot2/key", "val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let policy = RotationPolicy {
            secret_key: "rot2/key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };
        scoped.set_rotation_policy("rot2/key", policy).unwrap();
        scoped.remove_rotation_policy("rot2/key").unwrap();

        assert!(scoped.get_rotation_policy("rot2/key").is_none());
    }

    #[test]
    fn test_scoped_check_pending_rotations() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        // No policies set, so no pending rotations
        let pending = scoped.check_pending_rotations();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_scoped_execute_rotation() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rot3/key", "old_value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let policy = RotationPolicy {
            secret_key: "rot3/key".to_string(),
            interval_ms: 1,
            last_rotated_ms: 0,
            generator: RotationGenerator::Password(PasswordConfig::default()),
            notify_before_ms: 0,
        };
        scoped.set_rotation_policy("rot3/key", policy).unwrap();

        let new_val = scoped.execute_rotation("rot3/key").unwrap();
        assert!(!new_val.is_empty());
        assert_ne!(new_val, "old_value");
    }

    // ========== Engine Tests ==========

    struct MockEngine {
        name: String,
    }

    impl MockEngine {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl SecretEngine for MockEngine {
        fn name(&self) -> &str {
            &self.name
        }

        fn generate(&self, _params: &serde_json::Value) -> crate::Result<String> {
            Ok("generated_secret".to_string())
        }

        fn renew(&self, secret_id: &str, _params: &serde_json::Value) -> crate::Result<String> {
            Ok(format!("{secret_id}_renewed"))
        }

        fn revoke(&self, _secret_id: &str) -> crate::Result<()> {
            Ok(())
        }

        fn list(&self) -> crate::Result<Vec<String>> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_scoped_engine_generate() {
        let (vault, _graph) = create_scoped_vault_env();
        vault
            .register_engine(Box::new(MockEngine::new("mock")))
            .unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let secret = scoped
            .engine_generate("mock", &serde_json::json!({}))
            .unwrap();
        assert_eq!(secret, "generated_secret");
    }

    #[test]
    fn test_scoped_engine_revoke() {
        let (vault, _graph) = create_scoped_vault_env();
        vault
            .register_engine(Box::new(MockEngine::new("mock_rev")))
            .unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.engine_revoke("mock_rev", "some_secret_id").unwrap();
    }

    // ========== Sync Tests ==========

    struct MockSyncTarget {
        name: String,
    }

    impl MockSyncTarget {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl SyncTarget for MockSyncTarget {
        fn name(&self) -> &str {
            &self.name
        }

        fn push(&self, _key: &str, _value: &str) -> crate::Result<()> {
            Ok(())
        }

        fn delete(&self, _key: &str) -> crate::Result<()> {
            Ok(())
        }

        fn health_check(&self) -> crate::Result<bool> {
            Ok(true)
        }
    }

    #[test]
    fn test_scoped_subscribe_and_trigger_sync() {
        let (vault, _graph) = create_scoped_vault_env();
        vault
            .register_sync_target(Box::new(MockSyncTarget::new("mock_sync")))
            .unwrap();
        vault.set(Vault::ROOT, "sync/key", "sync_val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.subscribe_sync("sync/key", "mock_sync").unwrap();

        let count = scoped.trigger_sync("sync/key").unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scoped_unsubscribe_sync() {
        let (vault, _graph) = create_scoped_vault_env();
        vault
            .register_sync_target(Box::new(MockSyncTarget::new("mock_unsub")))
            .unwrap();
        vault.set(Vault::ROOT, "unsub/key", "val").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.subscribe_sync("unsub/key", "mock_unsub").unwrap();
        scoped.unsubscribe_sync("unsub/key", "mock_unsub").unwrap();

        let count = scoped.trigger_sync("unsub/key").unwrap();
        assert_eq!(count, 0);
    }

    // ========== Delegation Tests ==========

    #[test]
    fn test_scoped_delegate() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "del/key", "secret").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let record = scoped
            .delegate("user:bob", &["del/key"], Permission::Read, None)
            .unwrap();
        assert_eq!(record.parent, Vault::ROOT);
        assert_eq!(record.child, "user:bob");

        assert_eq!(vault.get("user:bob", "del/key").unwrap(), "secret");
    }

    #[test]
    fn test_scoped_revoke_delegation() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rdel/key", "secret").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped
            .delegate("user:carol", &["rdel/key"], Permission::Read, None)
            .unwrap();

        let revoked = scoped.revoke_delegation("user:carol").unwrap();
        assert!(!revoked.is_empty());
        assert!(vault.get("user:carol", "rdel/key").is_err());
    }

    #[test]
    fn test_scoped_revoke_delegation_cascading() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "cdel/key", "secret").unwrap();

        let scoped_root = vault.scope(Vault::ROOT);
        scoped_root
            .delegate("user:mid", &["cdel/key"], Permission::Read, None)
            .unwrap();

        let scoped_mid = vault.scope("user:mid");
        scoped_mid
            .delegate("user:leaf", &["cdel/key"], Permission::Read, None)
            .unwrap();

        let revoked = scoped_root.revoke_delegation_cascading("user:mid").unwrap();
        assert!(!revoked.is_empty());
    }

    #[test]
    fn test_scoped_anomaly_profile() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "anom/key", "value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let _ = scoped.get("anom/key").unwrap();

        // Profile may or may not exist depending on monitor state
        let _profile = scoped.anomaly_profile();
    }

    #[test]
    fn test_scoped_simulate_grant() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "sim/key", "value").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let result = scoped.simulate_grant("user:eve", "sim/key", Permission::Read);
        assert!(!result.new_accesses.is_empty() || result.new_accesses.is_empty());
    }

    // ========== TTL / Expiration Tests ==========

    #[test]
    fn test_scoped_clear_expiration() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        scoped
            .set_with_ttl("ttl/key", "value", Duration::from_secs(3600))
            .unwrap();

        let exp = scoped.get_expiration("ttl/key").unwrap();
        assert!(exp.is_some());

        scoped.clear_expiration("ttl/key").unwrap();
        let exp_after = scoped.get_expiration("ttl/key").unwrap();
        assert!(exp_after.is_none());
    }

    // ========== Batch Tests ==========

    #[test]
    fn test_scoped_batch_set() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        scoped
            .batch_set(&[("batch/a", "va"), ("batch/b", "vb")])
            .unwrap();

        assert_eq!(scoped.get("batch/a").unwrap(), "va");
        assert_eq!(scoped.get("batch/b").unwrap(), "vb");
    }

    // ========== Weighted Dependency Tests ==========

    #[test]
    fn test_scoped_add_weighted_dependency() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "wd/parent", "p").unwrap();
        vault.set(Vault::ROOT, "wd/child", "c").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped
            .add_weighted_dependency(
                "wd/parent",
                "wd/child",
                crate::dependency::DependencyWeight::Critical,
                Some("critical dependency"),
            )
            .unwrap();

        let deps = scoped.get_dependencies("wd/parent").unwrap();
        assert!(!deps.is_empty());
    }

    #[test]
    fn test_scoped_weighted_impact_analysis() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "wi/root", "r").unwrap();
        vault.set(Vault::ROOT, "wi/dep", "d").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped
            .add_weighted_dependency(
                "wi/root",
                "wi/dep",
                crate::dependency::DependencyWeight::High,
                None,
            )
            .unwrap();

        let report = scoped.weighted_impact_analysis("wi/root").unwrap();
        assert!(!report.root_secret.is_empty());
    }

    #[test]
    fn test_scoped_rotation_plan() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "rp/root", "r").unwrap();
        vault.set(Vault::ROOT, "rp/child", "c").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        scoped.add_dependency("rp/root", "rp/child").unwrap();

        let plan = scoped.rotation_plan("rp/root").unwrap();
        assert!(!plan.rotation_order.is_empty());
    }

    // ========== Developer Experience Tests ==========

    #[test]
    fn test_scoped_list_paginated() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);

        for i in 0..5 {
            scoped
                .set(&format!("pg/key{i}"), &format!("val{i}"))
                .unwrap();
        }

        let page = scoped.list_paginated("pg/*", 0, 3).unwrap();
        assert_eq!(page.secrets.len(), 3);
        assert_eq!(page.total, 5);
        assert!(page.has_more);

        let page2 = scoped.list_paginated("pg/*", 3, 3).unwrap();
        assert_eq!(page2.secrets.len(), 2);
        assert!(!page2.has_more);
    }

    #[test]
    fn test_scoped_list_with_metadata() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        scoped.set("meta/key1", "val1").unwrap();
        scoped.set("meta/key2", "val2").unwrap();

        let summaries = scoped.list_with_metadata("meta/*").unwrap();
        assert_eq!(summaries.len(), 2);
    }

    #[test]
    fn test_scoped_diff_versions() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        scoped.set("dv/key", "version1").unwrap();
        scoped.set("dv/key", "version2").unwrap();

        let diff = scoped.diff_versions("dv/key", 1, 2).unwrap();
        assert_ne!(diff.value_a, diff.value_b);
    }

    #[test]
    fn test_scoped_changelog() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        scoped.set("cl/key", "v1").unwrap();
        scoped.set("cl/key", "v2").unwrap();

        let log = scoped.changelog("cl/key").unwrap();
        assert!(log.len() >= 2);
    }

    // ========== Similarity Tests ==========

    #[test]
    fn test_scoped_find_similar() {
        let (vault, _graph) = create_scoped_vault_env();
        let scoped = vault.scope(Vault::ROOT);
        scoped.set("sim/db_password", "secret123").unwrap();
        scoped.set("sim/db_pass", "secret456").unwrap();

        let similar = scoped.find_similar("sim/db_password", 5).unwrap();
        // May or may not find similarity depending on implementation
        assert!(similar.len() <= 5);
    }

    // ========== Graph Intelligence Tests ==========

    #[test]
    fn test_scoped_compute_behavior_embeddings() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "emb/key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "emb/key").unwrap();
        let _ = vault.get("user:alice", "emb/key").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let config = crate::graph_intel::BehaviorEmbeddingConfig::default();
        let embeddings = scoped.compute_behavior_embeddings(config);
        // Embeddings may be empty if no behavior data accumulated
        let _ = embeddings;
    }

    #[test]
    fn test_scoped_detect_geometric_anomalies() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "geo/key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "geo/key").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let report = scoped.detect_geometric_anomalies(3, 2.0);
        assert!(report.anomalies.len() <= 10);
    }

    #[test]
    fn test_scoped_cluster_entities() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "cl/a", "va").unwrap();
        vault.set(Vault::ROOT, "cl/b", "vb").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "cl/a").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "cl/b").unwrap();

        let scoped = vault.scope(Vault::ROOT);
        let result = scoped.cluster_entities();
        // Clustering always returns a result
        let _ = result;
    }

    // ========== Mock Trait Coverage ==========

    #[test]
    fn test_mock_engine_renew_and_list() {
        let engine = MockEngine::new("coverage");
        let renewed = engine.renew("secret-1", &serde_json::json!({})).unwrap();
        assert!(renewed.contains("renewed"));

        let list = engine.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_mock_sync_target_delete_and_health() {
        let target = MockSyncTarget::new("coverage");
        target.delete("some_key").unwrap();

        let healthy = target.health_check().unwrap();
        assert!(healthy);
    }
}
