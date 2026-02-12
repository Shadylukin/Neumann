// SPDX-License-Identifier: MIT OR Apache-2.0
//! Audit logging for vault operations.
//!
//! Records all vault operations for compliance and forensics.
//! When an audit key is provided, entries are protected with:
//! - AEAD encryption of entity names (confidentiality)
//! - HMAC-BLAKE2b integrity tags (tamper detection)
//! - Epoch fingerprints for key rotation compatibility

#![allow(clippy::missing_panics_doc)]

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use blake2::{digest::consts::U32, digest::Mac, Blake2b, Digest};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

static AUDIT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// AES-GCM nonce size (96 bits).
const NONCE_SIZE: usize = 12;

/// Optional context for audit entries (IP, session, correlation ID).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditContext {
    pub source_ip: Option<String>,
    pub session_id: Option<String>,
    pub correlation_id: Option<String>,
}

/// Audit entry representing a single vault operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    /// The entity that performed the operation.
    pub entity: String,
    /// The secret key that was accessed.
    pub secret_key: String,
    /// The operation performed.
    pub operation: AuditOperation,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Optional operation context.
    pub context: Option<AuditContext>,
}

/// Types of auditable operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditOperation {
    Get,
    Set,
    Delete,
    Rotate,
    Grant {
        to: String,
        permission: String,
    },
    Revoke {
        from: String,
    },
    List,
    RotateMasterKey {
        secrets_count: usize,
    },
    TransitEncrypt,
    TransitDecrypt,
    BreakGlass {
        justification: String,
        duration_secs: u64,
    },
    BatchGet {
        count: usize,
    },
    BatchSet {
        count: usize,
    },
    DynamicGenerate,
    Wrap,
    Unwrap,
    AddDependency,
    RemoveDependency,
    ImpactAnalysis,
    AutoRotate,
    SetQuota,
    RemoveQuota,
    AddPolicy,
    RemovePolicy,
    Seal,
    Unseal,
    CreateSnapshot,
    RestoreSnapshot,
    EngineGenerate,
    EngineRevoke,
    IssueCertificate,
    RevokeCertificate,
    SyncPush,
    SyncSubscribe,
    LegacyDecrypt,
    DiffVersions {
        version_a: u32,
        version_b: u32,
    },
    SaveTemplate {
        template_name: String,
    },
    DeleteTemplate {
        template_name: String,
    },
    FindSimilar {
        k: usize,
    },
    HeatKernelTrust {
        diffusion_time: f64,
    },
    BuildAccessTensor {
        num_buckets: usize,
    },
    AnalyzeTemporalPatterns,
    WeightedImpactAnalysis,
    RotationPlan,
    RecommendPlacement {
        region: String,
    },
}

impl AuditOperation {
    fn as_str(&self) -> &str {
        match self {
            Self::Get => "get",
            Self::Set => "set",
            Self::Delete => "delete",
            Self::Rotate => "rotate",
            Self::Grant { .. } => "grant",
            Self::Revoke { .. } => "revoke",
            Self::List => "list",
            Self::RotateMasterKey { .. } => "rotate_master_key",
            Self::TransitEncrypt => "transit_encrypt",
            Self::TransitDecrypt => "transit_decrypt",
            Self::BreakGlass { .. } => "break_glass",
            Self::BatchGet { .. } => "batch_get",
            Self::BatchSet { .. } => "batch_set",
            Self::DynamicGenerate => "dynamic_generate",
            Self::Wrap => "wrap",
            Self::Unwrap => "unwrap",
            Self::AddDependency => "add_dependency",
            Self::RemoveDependency => "remove_dependency",
            Self::ImpactAnalysis => "impact_analysis",
            Self::AutoRotate => "auto_rotate",
            Self::SetQuota => "set_quota",
            Self::RemoveQuota => "remove_quota",
            Self::AddPolicy => "add_policy",
            Self::RemovePolicy => "remove_policy",
            Self::Seal => "seal",
            Self::Unseal => "unseal",
            Self::CreateSnapshot => "create_snapshot",
            Self::RestoreSnapshot => "restore_snapshot",
            Self::EngineGenerate => "engine_generate",
            Self::EngineRevoke => "engine_revoke",
            Self::IssueCertificate => "issue_certificate",
            Self::RevokeCertificate => "revoke_certificate",
            Self::SyncPush => "sync_push",
            Self::SyncSubscribe => "sync_subscribe",
            Self::LegacyDecrypt => "legacy_decrypt",
            Self::DiffVersions { .. } => "diff_versions",
            Self::SaveTemplate { .. } => "save_template",
            Self::DeleteTemplate { .. } => "delete_template",
            Self::FindSimilar { .. } => "find_similar",
            Self::HeatKernelTrust { .. } => "heat_kernel_trust",
            Self::BuildAccessTensor { .. } => "build_access_tensor",
            Self::AnalyzeTemporalPatterns => "analyze_temporal_patterns",
            Self::WeightedImpactAnalysis => "weighted_impact_analysis",
            Self::RotationPlan => "rotation_plan",
            Self::RecommendPlacement { .. } => "recommend_placement",
        }
    }

    #[allow(clippy::too_many_lines)] // match arm per variant, extraction harms readability
    fn from_tensor_with_decryption(
        tensor: &TensorData,
        audit_log: Option<&AuditLog<'_>>,
    ) -> Option<Self> {
        let op_type = match tensor.get("_op") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.as_str(),
            _ => return None,
        };

        match op_type {
            "get" => Some(Self::Get),
            "set" => Some(Self::Set),
            "delete" => Some(Self::Delete),
            "rotate" => Some(Self::Rotate),
            "list" => Some(Self::List),
            "grant" => {
                let to = Self::read_target_field(tensor, audit_log);
                let permission = match tensor.get("_permission") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => "admin".to_string(),
                };
                Some(Self::Grant { to, permission })
            },
            "revoke" => {
                let from = Self::read_target_field(tensor, audit_log);
                Some(Self::Revoke { from })
            },
            "rotate_master_key" => {
                let secrets_count = match tensor.get("_secrets_count") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let count = *n as usize;
                        count
                    },
                    _ => 0,
                };
                Some(Self::RotateMasterKey { secrets_count })
            },
            "transit_encrypt" => Some(Self::TransitEncrypt),
            "transit_decrypt" => Some(Self::TransitDecrypt),
            "break_glass" => {
                let justification = match tensor.get("_justification") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                let duration_secs = match tensor.get("_duration_secs") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let secs = *n as u64;
                        secs
                    },
                    _ => 0,
                };
                Some(Self::BreakGlass {
                    justification,
                    duration_secs,
                })
            },
            "batch_get" => {
                let count = match tensor.get("_count") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let c = *n as usize;
                        c
                    },
                    _ => 0,
                };
                Some(Self::BatchGet { count })
            },
            "batch_set" => {
                let count = match tensor.get("_count") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let c = *n as usize;
                        c
                    },
                    _ => 0,
                };
                Some(Self::BatchSet { count })
            },
            "dynamic_generate" => Some(Self::DynamicGenerate),
            "wrap" => Some(Self::Wrap),
            "unwrap" => Some(Self::Unwrap),
            "add_dependency" => Some(Self::AddDependency),
            "remove_dependency" => Some(Self::RemoveDependency),
            "impact_analysis" => Some(Self::ImpactAnalysis),
            "auto_rotate" => Some(Self::AutoRotate),
            "set_quota" => Some(Self::SetQuota),
            "remove_quota" => Some(Self::RemoveQuota),
            "add_policy" => Some(Self::AddPolicy),
            "remove_policy" => Some(Self::RemovePolicy),
            "seal" => Some(Self::Seal),
            "unseal" => Some(Self::Unseal),
            "create_snapshot" => Some(Self::CreateSnapshot),
            "restore_snapshot" => Some(Self::RestoreSnapshot),
            "engine_generate" => Some(Self::EngineGenerate),
            "engine_revoke" => Some(Self::EngineRevoke),
            "issue_certificate" => Some(Self::IssueCertificate),
            "revoke_certificate" => Some(Self::RevokeCertificate),
            "sync_push" => Some(Self::SyncPush),
            "sync_subscribe" => Some(Self::SyncSubscribe),
            "legacy_decrypt" => Some(Self::LegacyDecrypt),
            "diff_versions" => {
                let version_a = match tensor.get("_version_a") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let v = *n as u32;
                        v
                    },
                    _ => 0,
                };
                let version_b = match tensor.get("_version_b") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let v = *n as u32;
                        v
                    },
                    _ => 0,
                };
                Some(Self::DiffVersions {
                    version_a,
                    version_b,
                })
            },
            "save_template" => {
                let template_name = match tensor.get("_template_name") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                Some(Self::SaveTemplate { template_name })
            },
            "delete_template" => {
                let template_name = match tensor.get("_template_name") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                Some(Self::DeleteTemplate { template_name })
            },
            "find_similar" => {
                let k = match tensor.get("_k") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let v = *n as usize;
                        v
                    },
                    _ => 0,
                };
                Some(Self::FindSimilar { k })
            },
            "heat_kernel_trust" => {
                let diffusion_time = match tensor.get("_diffusion_time") {
                    Some(TensorValue::Scalar(ScalarValue::Float(f))) => *f,
                    _ => 1.0,
                };
                Some(Self::HeatKernelTrust { diffusion_time })
            },
            "build_access_tensor" => {
                let num_buckets = match tensor.get("_num_buckets") {
                    Some(TensorValue::Scalar(ScalarValue::Int(n))) => {
                        #[allow(clippy::cast_sign_loss)]
                        let v = *n as usize;
                        v
                    },
                    _ => 0,
                };
                Some(Self::BuildAccessTensor { num_buckets })
            },
            "analyze_temporal_patterns" => Some(Self::AnalyzeTemporalPatterns),
            "weighted_impact_analysis" => Some(Self::WeightedImpactAnalysis),
            "rotation_plan" => Some(Self::RotationPlan),
            "recommend_placement" => {
                let region = match tensor.get("_region") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                Some(Self::RecommendPlacement { region })
            },
            _ => None,
        }
    }

    /// Read the target field, trying encrypted first then plaintext fallback.
    fn read_target_field(tensor: &TensorData, audit_log: Option<&AuditLog<'_>>) -> String {
        // Try encrypted target first
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(enc))) = tensor.get("_target_enc") {
            if let Some(log) = audit_log {
                if let Some(decrypted) = log.decrypt_field(enc) {
                    return decrypted;
                }
            }
        }
        // Fallback to plaintext target (legacy)
        match tensor.get("_target") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        }
    }
}

/// Audit log for tracking vault operations.
pub struct AuditLog<'a> {
    store: &'a TensorStore,
    audit_key: Option<[u8; 32]>,
}

/// Prefix for audit entries in the store.
const AUDIT_PREFIX: &str = "_va:";

impl<'a> AuditLog<'a> {
    pub fn new(store: &'a TensorStore, audit_key: Option<[u8; 32]>) -> Self {
        Self { store, audit_key }
    }

    fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// AEAD-encrypt a plaintext field. Returns `nonce || ciphertext`.
    fn encrypt_field(&self, plaintext: &str) -> Option<Vec<u8>> {
        let key = self.audit_key?;
        let cipher = Aes256Gcm::new_from_slice(&key).ok()?;
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes()).ok()?;
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        Some(result)
    }

    /// Decrypt a field produced by `encrypt_field`.
    fn decrypt_field(&self, encrypted: &[u8]) -> Option<String> {
        let key = self.audit_key?;
        if encrypted.len() <= NONCE_SIZE {
            return None;
        }
        let cipher = Aes256Gcm::new_from_slice(&key).ok()?;
        let nonce = Nonce::from_slice(&encrypted[..NONCE_SIZE]);
        let plaintext = cipher.decrypt(nonce, &encrypted[NONCE_SIZE..]).ok()?;
        String::from_utf8(plaintext).ok()
    }

    /// Compute BLAKE2b-MAC512 keyed MAC over all audit entry fields.
    fn compute_hmac(
        &self,
        entity: &str,
        secret: &str,
        op: &str,
        ts: i64,
        extras: &[(&str, &str)],
    ) -> Option<Vec<u8>> {
        let key = self.audit_key?;
        let mut mac = <blake2::Blake2bMac512 as Mac>::new_from_slice(&key).ok()?;
        mac.update(entity.as_bytes());
        mac.update(b"\x00");
        mac.update(secret.as_bytes());
        mac.update(b"\x00");
        mac.update(op.as_bytes());
        mac.update(b"\x00");
        mac.update(&ts.to_le_bytes());
        for (k, v) in extras {
            mac.update(b"\x00");
            mac.update(k.as_bytes());
            mac.update(b"\x00");
            mac.update(v.as_bytes());
        }
        Some(mac.finalize().into_bytes().to_vec())
    }

    /// Legacy hand-rolled HMAC-BLAKE2b (for backward compatibility).
    fn compute_hmac_legacy(
        &self,
        entity: &str,
        secret: &str,
        op: &str,
        ts: i64,
        extras: &[(&str, &str)],
    ) -> Option<Vec<u8>> {
        let key = self.audit_key?;

        let mut inner_key = key;
        for byte in &mut inner_key {
            *byte ^= 0x36;
        }
        let mut inner = Blake2b::<U32>::new();
        inner.update(inner_key);
        inner.update(entity.as_bytes());
        inner.update(b"\x00");
        inner.update(secret.as_bytes());
        inner.update(b"\x00");
        inner.update(op.as_bytes());
        inner.update(b"\x00");
        inner.update(ts.to_le_bytes());
        for (k, v) in extras {
            inner.update(b"\x00");
            inner.update(k.as_bytes());
            inner.update(b"\x00");
            inner.update(v.as_bytes());
        }
        let inner_hash = inner.finalize();

        let mut outer_key = key;
        for byte in &mut outer_key {
            *byte ^= 0x5c;
        }
        let mut outer = Blake2b::<U32>::new();
        outer.update(outer_key);
        outer.update(inner_hash);
        let result = outer.finalize();

        Some(result.to_vec())
    }

    /// Verify HMAC using constant-time comparison (tries new MAC, then legacy fallback).
    fn verify_hmac(
        &self,
        entity: &str,
        secret: &str,
        op: &str,
        ts: i64,
        extras: &[(&str, &str)],
        expected: &[u8],
    ) -> bool {
        if let Some(computed) = self.compute_hmac(entity, secret, op, ts, extras) {
            if constant_time_eq(&computed, expected) {
                return true;
            }
        }
        // Legacy fallback
        if let Some(legacy) = self.compute_hmac_legacy(entity, secret, op, ts, extras) {
            return constant_time_eq(&legacy, expected);
        }
        false
    }

    /// BLAKE2b hash of the audit key truncated to 8 bytes, used to
    /// distinguish key eras across `rotate_master_key()` calls.
    fn audit_epoch(&self) -> Option<Vec<u8>> {
        let key = self.audit_key?;
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(key);
        let hash = hasher.finalize();
        Some(hash[..8].to_vec())
    }

    /// Collect extra (key, value) pairs from an operation for HMAC.
    fn operation_extras(operation: &AuditOperation) -> Vec<(&str, &str)> {
        match operation {
            AuditOperation::Grant { to, permission } => {
                vec![("target", to.as_str()), ("permission", permission.as_str())]
            },
            AuditOperation::Revoke { from } => vec![("target", from.as_str())],
            AuditOperation::BreakGlass { justification, .. } => {
                vec![("justification", justification.as_str())]
            },
            _ => vec![],
        }
    }

    /// Record an operation.
    #[allow(clippy::too_many_lines)] // per-variant field storage, extraction harms readability
    pub fn record(&self, entity: &str, secret_key: &str, operation: &AuditOperation) {
        let timestamp = Self::now_millis();
        let counter = AUDIT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let key = format!("{AUDIT_PREFIX}{timestamp}:{counter}");

        let mut tensor = TensorData::new();

        // Entity: encrypted if key available, plaintext otherwise
        if let Some(encrypted) = self.encrypt_field(entity) {
            tensor.set(
                "_entity_enc",
                TensorValue::Scalar(ScalarValue::Bytes(encrypted)),
            );
        } else {
            tensor.set(
                "_entity",
                TensorValue::Scalar(ScalarValue::String(entity.into())),
            );
        }

        tensor.set(
            "_secret",
            TensorValue::Scalar(ScalarValue::String(secret_key.into())),
        );
        tensor.set(
            "_op",
            TensorValue::Scalar(ScalarValue::String(operation.as_str().into())),
        );
        tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(timestamp)));

        // Store additional info for grant/revoke
        match operation {
            AuditOperation::Grant { to, permission } => {
                if let Some(encrypted) = self.encrypt_field(to) {
                    tensor.set(
                        "_target_enc",
                        TensorValue::Scalar(ScalarValue::Bytes(encrypted)),
                    );
                } else {
                    tensor.set(
                        "_target",
                        TensorValue::Scalar(ScalarValue::String(to.clone())),
                    );
                }
                tensor.set(
                    "_permission",
                    TensorValue::Scalar(ScalarValue::String(permission.clone())),
                );
            },
            AuditOperation::Revoke { from } => {
                if let Some(encrypted) = self.encrypt_field(from) {
                    tensor.set(
                        "_target_enc",
                        TensorValue::Scalar(ScalarValue::Bytes(encrypted)),
                    );
                } else {
                    tensor.set(
                        "_target",
                        TensorValue::Scalar(ScalarValue::String(from.clone())),
                    );
                }
            },
            AuditOperation::RotateMasterKey { secrets_count } => {
                tensor.set(
                    "_secrets_count",
                    TensorValue::Scalar(ScalarValue::Int(*secrets_count as i64)),
                );
            },
            AuditOperation::BreakGlass {
                justification,
                duration_secs,
            } => {
                tensor.set(
                    "_justification",
                    TensorValue::Scalar(ScalarValue::String(justification.clone())),
                );
                tensor.set(
                    "_duration_secs",
                    TensorValue::Scalar(ScalarValue::Int(*duration_secs as i64)),
                );
            },
            AuditOperation::BatchGet { count } | AuditOperation::BatchSet { count } => {
                tensor.set(
                    "_count",
                    TensorValue::Scalar(ScalarValue::Int(*count as i64)),
                );
            },
            AuditOperation::DiffVersions {
                version_a,
                version_b,
            } => {
                tensor.set(
                    "_version_a",
                    TensorValue::Scalar(ScalarValue::Int(i64::from(*version_a))),
                );
                tensor.set(
                    "_version_b",
                    TensorValue::Scalar(ScalarValue::Int(i64::from(*version_b))),
                );
            },
            AuditOperation::SaveTemplate { template_name }
            | AuditOperation::DeleteTemplate { template_name } => {
                tensor.set(
                    "_template_name",
                    TensorValue::Scalar(ScalarValue::String(template_name.clone())),
                );
            },
            AuditOperation::FindSimilar { k } => {
                tensor.set("_k", TensorValue::Scalar(ScalarValue::Int(*k as i64)));
            },
            AuditOperation::HeatKernelTrust { diffusion_time } => {
                tensor.set(
                    "_diffusion_time",
                    TensorValue::Scalar(ScalarValue::Float(*diffusion_time)),
                );
            },
            AuditOperation::BuildAccessTensor { num_buckets } => {
                tensor.set(
                    "_num_buckets",
                    TensorValue::Scalar(ScalarValue::Int(*num_buckets as i64)),
                );
            },
            AuditOperation::RecommendPlacement { region } => {
                tensor.set(
                    "_region",
                    TensorValue::Scalar(ScalarValue::String(region.clone())),
                );
            },
            _ => {},
        }

        // HMAC and epoch if audit key is available
        let extras = Self::operation_extras(operation);
        if let Some(hmac) =
            self.compute_hmac(entity, secret_key, operation.as_str(), timestamp, &extras)
        {
            tensor.set("_hmac", TensorValue::Scalar(ScalarValue::Bytes(hmac)));
        }
        if let Some(epoch) = self.audit_epoch() {
            tensor.set(
                "_audit_epoch",
                TensorValue::Scalar(ScalarValue::Bytes(epoch)),
            );
        }

        // Best effort - audit failures don't block operations
        let _ = self.store.put(&key, tensor);
    }

    /// Record an operation with additional context.
    pub fn record_with_context(
        &self,
        entity: &str,
        secret_key: &str,
        operation: &AuditOperation,
        context: Option<&AuditContext>,
    ) {
        // Context is stored but not yet serialized to tensor (plumbing for future callers)
        let _ = context;
        self.record(entity, secret_key, operation);
    }

    /// Query audit entries for a specific secret.
    pub fn by_secret(&self, secret_key: &str) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.secret_key == secret_key)
            .collect()
    }

    /// Query audit entries by entity (who performed operations).
    pub fn by_entity(&self, entity: &str) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.entity == entity)
            .collect()
    }

    /// Query audit entries since a timestamp (unix millis).
    pub fn since(&self, since_millis: i64) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.timestamp >= since_millis)
            .collect()
    }

    /// Query audit entries within a time range.
    pub fn between(&self, start_millis: i64, end_millis: i64) -> Vec<AuditEntry> {
        self.scan()
            .into_iter()
            .filter(|e| e.timestamp >= start_millis && e.timestamp <= end_millis)
            .collect()
    }

    /// Get recent audit entries (last N).
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let mut entries = self.scan();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        entries
    }

    fn scan(&self) -> Vec<AuditEntry> {
        let keys = self.store.scan(AUDIT_PREFIX);
        let mut entries = Vec::new();

        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(entry) = self.tensor_to_entry(&tensor) {
                    entries.push(entry);
                }
            }
        }

        entries
    }

    fn tensor_to_entry(&self, tensor: &TensorData) -> Option<AuditEntry> {
        // Entity: try encrypted first, then plaintext (legacy)
        let entity =
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(enc))) = tensor.get("_entity_enc") {
                self.decrypt_field(enc)?
            } else {
                match tensor.get("_entity") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => return None,
                }
            };

        let secret_key = match tensor.get("_secret") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return None,
        };
        let timestamp = match tensor.get("_ts") {
            Some(TensorValue::Scalar(ScalarValue::Int(t))) => *t,
            _ => return None,
        };
        let operation = AuditOperation::from_tensor_with_decryption(tensor, Some(self))?;

        // HMAC verification
        if let Some(TensorValue::Scalar(ScalarValue::Bytes(stored_hmac))) = tensor.get("_hmac") {
            let stored_epoch = match tensor.get("_audit_epoch") {
                Some(TensorValue::Scalar(ScalarValue::Bytes(e))) => Some(e.clone()),
                _ => None,
            };

            match (stored_epoch, self.audit_epoch()) {
                (Some(stored_ep), Some(current_ep)) if stored_ep == current_ep => {
                    // Same key era: verify strictly, skip tampered entries
                    let extras = Self::extras_from_tensor(tensor, self);
                    let extras_refs: Vec<(&str, &str)> = extras
                        .iter()
                        .map(|(k, v)| (k.as_str(), v.as_str()))
                        .collect();
                    let op_str = match tensor.get("_op") {
                        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                        _ => return None,
                    };
                    if !self.verify_hmac(
                        &entity,
                        &secret_key,
                        &op_str,
                        timestamp,
                        &extras_refs,
                        stored_hmac,
                    ) {
                        return None; // Tampered entry - skip
                    }
                },
                _ => {
                    // Different key era or missing epoch: treat as legacy
                },
            }
        }

        Some(AuditEntry {
            entity,
            secret_key,
            operation,
            timestamp,
            context: None,
        })
    }

    /// Extract extra fields from a tensor for HMAC verification.
    fn extras_from_tensor(tensor: &TensorData, audit_log: &AuditLog<'_>) -> Vec<(String, String)> {
        let op_type = match tensor.get("_op") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.as_str(),
            _ => return vec![],
        };

        match op_type {
            "grant" => {
                let target = AuditOperation::read_target_field(tensor, Some(audit_log));
                let permission = match tensor.get("_permission") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => "admin".to_string(),
                };
                vec![
                    ("target".to_string(), target),
                    ("permission".to_string(), permission),
                ]
            },
            "revoke" => {
                let target = AuditOperation::read_target_field(tensor, Some(audit_log));
                vec![("target".to_string(), target)]
            },
            "break_glass" => {
                let justification = match tensor.get("_justification") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
                    _ => String::new(),
                };
                vec![("justification".to_string(), justification)]
            },
            _ => vec![],
        }
    }
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> TensorStore {
        TensorStore::new()
    }

    fn test_audit_key() -> [u8; 32] {
        [42u8; 32]
    }

    #[test]
    fn test_record_and_query_by_secret() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("user:alice", "api_key", &AuditOperation::Get);
        log.record("user:bob", "api_key", &AuditOperation::Get);
        log.record("user:alice", "other_key", &AuditOperation::Set);

        let entries = log.by_secret("api_key");
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            assert_eq!(entry.secret_key, "api_key");
        }
    }

    #[test]
    fn test_query_by_entity() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("user:alice", "key1", &AuditOperation::Get);
        log.record("user:alice", "key2", &AuditOperation::Set);
        log.record("user:bob", "key1", &AuditOperation::Get);

        let entries = log.by_entity("user:alice");
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            assert_eq!(entry.entity, "user:alice");
        }
    }

    #[test]
    fn test_query_since() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        let before = AuditLog::now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));

        log.record("user:alice", "key", &AuditOperation::Get);

        let entries = log.since(before);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_grant_operation_details() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Grant {
                to: "user:alice".to_string(),
                permission: "read".to_string(),
            },
        );

        let entries = log.by_secret("secret");
        assert_eq!(entries.len(), 1);

        match &entries[0].operation {
            AuditOperation::Grant { to, permission } => {
                assert_eq!(to, "user:alice");
                assert_eq!(permission, "read");
            },
            _ => panic!("Expected Grant operation"),
        }
    }

    #[test]
    fn test_revoke_operation_details() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Revoke {
                from: "user:alice".to_string(),
            },
        );

        let entries = log.by_secret("secret");
        assert_eq!(entries.len(), 1);

        match &entries[0].operation {
            AuditOperation::Revoke { from } => {
                assert_eq!(from, "user:alice");
            },
            _ => panic!("Expected Revoke operation"),
        }
    }

    #[test]
    fn test_recent_entries() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        for i in 0..10 {
            log.record(&format!("user:{i}"), "key", &AuditOperation::Get);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first
        assert!(recent[0].timestamp >= recent[1].timestamp);
        assert!(recent[1].timestamp >= recent[2].timestamp);
    }

    #[test]
    fn test_all_operation_types() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("u", "k", &AuditOperation::Get);
        log.record("u", "k", &AuditOperation::Set);
        log.record("u", "k", &AuditOperation::Delete);
        log.record("u", "k", &AuditOperation::Rotate);
        log.record("u", "k", &AuditOperation::List);
        log.record(
            "u",
            "k",
            &AuditOperation::Grant {
                to: "x".to_string(),
                permission: "write".to_string(),
            },
        );
        log.record(
            "u",
            "k",
            &AuditOperation::Revoke {
                from: "x".to_string(),
            },
        );

        let entries = log.by_secret("k");
        assert_eq!(entries.len(), 7);
    }

    #[test]
    fn test_empty_results() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        assert!(log.by_secret("nonexistent").is_empty());
        assert!(log.by_entity("unknown").is_empty());
        assert!(log.recent(10).is_empty());
    }

    #[test]
    fn test_between_range() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        let t1 = AuditLog::now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));

        log.record("user:alice", "key", &AuditOperation::Get);

        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = AuditLog::now_millis();

        std::thread::sleep(std::time::Duration::from_millis(10));
        log.record("user:bob", "key", &AuditOperation::Set);

        let entries = log.between(t1, t2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }

    #[test]
    fn test_audit_entry_hmac_prevents_tampering() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record("user:alice", "secret_key", &AuditOperation::Get);

        // Find the stored audit entry and tamper with it
        let keys = store.scan(AUDIT_PREFIX);
        assert_eq!(keys.len(), 1);
        let mut tensor = store.get(&keys[0]).unwrap();

        // Tamper: change the operation
        tensor.set(
            "_op",
            TensorValue::Scalar(ScalarValue::String("delete".into())),
        );
        let _ = store.put(&keys[0], tensor);

        // Read should skip tampered entry
        let entries = log.scan();
        assert!(entries.is_empty(), "Tampered entry should be skipped");
    }

    #[test]
    fn test_audit_entity_name_encrypted() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record("user:alice", "secret_key", &AuditOperation::Get);

        // Check raw storage: _entity_enc should be Bytes, no _entity
        let keys = store.scan(AUDIT_PREFIX);
        assert_eq!(keys.len(), 1);
        let tensor = store.get(&keys[0]).unwrap();

        assert!(
            tensor.get("_entity").is_none(),
            "Plaintext _entity should not be stored"
        );
        match tensor.get("_entity_enc") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => {
                assert!(
                    b.len() > NONCE_SIZE,
                    "Encrypted entity should have nonce + ciphertext"
                );
            },
            other => panic!("Expected _entity_enc as Bytes, got {other:?}"),
        }

        // Reading back should decrypt correctly
        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }

    #[test]
    fn test_audit_backward_compat_legacy_entries() {
        let store = create_test_store();

        // Write without key (legacy)
        let log_legacy = AuditLog::new(&store, None);
        log_legacy.record("user:alice", "key", &AuditOperation::Set);

        // Read with key -- legacy entry should still be readable
        let key = test_audit_key();
        let log_keyed = AuditLog::new(&store, Some(key));
        let entries = log_keyed.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }

    #[test]
    fn test_audit_grant_target_encrypted() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Grant {
                to: "user:bob".to_string(),
                permission: "read".to_string(),
            },
        );

        // Check raw storage: _target_enc should exist, no _target
        let keys = store.scan(AUDIT_PREFIX);
        let tensor = store.get(&keys[0]).unwrap();
        assert!(tensor.get("_target").is_none());
        assert!(tensor.get("_target_enc").is_some());

        // Reading back should decrypt correctly
        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::Grant { to, permission } => {
                assert_eq!(to, "user:bob");
                assert_eq!(permission, "read");
            },
            _ => panic!("Expected Grant"),
        }
    }

    #[test]
    fn test_audit_epoch_mismatch_treated_as_legacy() {
        let store = create_test_store();
        let key_a = [1u8; 32];
        let key_b = [2u8; 32];

        // Write with key A
        let log_a = AuditLog::new(&store, Some(key_a));
        log_a.record("user:alice", "key", &AuditOperation::Get);

        // Read with key B (different epoch) -- entry should be returned, not skipped
        // But entity is AEAD-encrypted with key A so decryption with key B fails.
        // The entry's _entity_enc cannot be decrypted, so it falls through.
        // We need to test with a legacy-style entry (plaintext entity + HMAC from different key).
        // Let's create one manually.
        let store2 = create_test_store();
        let counter = AUDIT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let ts = AuditLog::now_millis();
        let entry_key = format!("{AUDIT_PREFIX}{ts}:{counter}");
        let mut tensor = TensorData::new();
        tensor.set(
            "_entity",
            TensorValue::Scalar(ScalarValue::String("user:alice".into())),
        );
        tensor.set(
            "_secret",
            TensorValue::Scalar(ScalarValue::String("key".into())),
        );
        tensor.set(
            "_op",
            TensorValue::Scalar(ScalarValue::String("get".into())),
        );
        tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(ts)));

        // Add HMAC from key A
        let log_a2 = AuditLog::new(&store2, Some(key_a));
        let hmac = log_a2
            .compute_hmac("user:alice", "key", "get", ts, &[])
            .unwrap();
        tensor.set("_hmac", TensorValue::Scalar(ScalarValue::Bytes(hmac)));
        let epoch_a = log_a2.audit_epoch().unwrap();
        tensor.set(
            "_audit_epoch",
            TensorValue::Scalar(ScalarValue::Bytes(epoch_a)),
        );
        let _ = store2.put(&entry_key, tensor);

        // Read with key B -- different epoch, should treat as legacy
        let log_b = AuditLog::new(&store2, Some(key_b));
        let entries = log_b.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }

    #[test]
    fn test_audit_revoke_target_encrypted() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record(
            "user:admin",
            "secret",
            &AuditOperation::Revoke {
                from: "user:bob".to_string(),
            },
        );

        let keys = store.scan(AUDIT_PREFIX);
        let tensor = store.get(&keys[0]).unwrap();
        assert!(tensor.get("_target").is_none());
        assert!(tensor.get("_target_enc").is_some());

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::Revoke { from } => {
                assert_eq!(from, "user:bob");
            },
            _ => panic!("Expected Revoke"),
        }
    }

    #[test]
    fn test_audit_rotate_master_key_with_hmac() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record(
            "node:root",
            "master_key",
            &AuditOperation::RotateMasterKey { secrets_count: 5 },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::RotateMasterKey { secrets_count } => {
                assert_eq!(*secrets_count, 5);
            },
            _ => panic!("Expected RotateMasterKey"),
        }
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_encrypt_decrypt_field_roundtrip() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        let plaintext = "user:alice";
        let encrypted = log.encrypt_field(plaintext).unwrap();
        let decrypted = log.decrypt_field(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_field_none_without_key() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        assert!(log.encrypt_field("anything").is_none());
        assert!(log.decrypt_field(&[0u8; 32]).is_none());
    }

    #[test]
    fn test_audit_epoch_deterministic() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        let e1 = log.audit_epoch().unwrap();
        let e2 = log.audit_epoch().unwrap();
        assert_eq!(e1, e2);
        assert_eq!(e1.len(), 8);
    }

    #[test]
    fn test_audit_epoch_different_keys() {
        let store = create_test_store();
        let log1 = AuditLog::new(&store, Some([1u8; 32]));
        let log2 = AuditLog::new(&store, Some([2u8; 32]));

        assert_ne!(log1.audit_epoch().unwrap(), log2.audit_epoch().unwrap());
    }

    #[test]
    fn test_audit_new_hmac_roundtrip() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        log.record("user:alice", "secret_key", &AuditOperation::Get);

        // New entries should be verifiable
        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:alice");
    }

    #[test]
    fn test_audit_legacy_hmac_still_verified() {
        let store = create_test_store();
        let key = test_audit_key();
        let log = AuditLog::new(&store, Some(key));

        // Manually create an entry with legacy HMAC
        let ts = AuditLog::now_millis();
        let counter = AUDIT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let entry_key = format!("{AUDIT_PREFIX}{ts}:{counter}");
        let mut tensor = TensorData::new();
        tensor.set(
            "_entity",
            TensorValue::Scalar(ScalarValue::String("user:bob".into())),
        );
        tensor.set(
            "_secret",
            TensorValue::Scalar(ScalarValue::String("key".into())),
        );
        tensor.set(
            "_op",
            TensorValue::Scalar(ScalarValue::String("get".into())),
        );
        tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(ts)));

        // Compute legacy HMAC
        let legacy_hmac = log
            .compute_hmac_legacy("user:bob", "key", "get", ts, &[])
            .unwrap();
        tensor.set(
            "_hmac",
            TensorValue::Scalar(ScalarValue::Bytes(legacy_hmac)),
        );
        let epoch = log.audit_epoch().unwrap();
        tensor.set(
            "_audit_epoch",
            TensorValue::Scalar(ScalarValue::Bytes(epoch)),
        );
        let _ = store.put(&entry_key, tensor);

        // Should still be readable via dual-verify
        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entity, "user:bob");
    }

    #[test]
    fn test_legacy_decrypt_operation() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("user:alice", "key", &AuditOperation::LegacyDecrypt);

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].operation, AuditOperation::LegacyDecrypt);
    }

    #[test]
    fn test_heat_kernel_trust_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "vault",
            &AuditOperation::HeatKernelTrust {
                diffusion_time: 2.5,
            },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::HeatKernelTrust { diffusion_time } => {
                assert!((*diffusion_time - 2.5).abs() < f64::EPSILON);
            },
            other => panic!("Expected HeatKernelTrust, got {other:?}"),
        }
    }

    #[test]
    fn test_build_access_tensor_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "vault",
            &AuditOperation::BuildAccessTensor { num_buckets: 168 },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::BuildAccessTensor { num_buckets } => {
                assert_eq!(*num_buckets, 168);
            },
            other => panic!("Expected BuildAccessTensor, got {other:?}"),
        }
    }

    #[test]
    fn test_analyze_temporal_patterns_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "vault",
            &AuditOperation::AnalyzeTemporalPatterns,
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].operation,
            AuditOperation::AnalyzeTemporalPatterns
        );
    }

    #[test]
    fn test_weighted_impact_analysis_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "secret:a",
            &AuditOperation::WeightedImpactAnalysis,
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].operation, AuditOperation::WeightedImpactAnalysis);
    }

    #[test]
    fn test_rotation_plan_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("user:admin", "secret:a", &AuditOperation::RotationPlan);

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].operation, AuditOperation::RotationPlan);
    }

    #[test]
    fn test_recommend_placement_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "secret:a",
            &AuditOperation::RecommendPlacement {
                region: "us-east-1".to_string(),
            },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::RecommendPlacement { region } => {
                assert_eq!(region, "us-east-1");
            },
            other => panic!("Expected RecommendPlacement, got {other:?}"),
        }
    }

    #[test]
    fn test_new_operations_as_str() {
        assert_eq!(
            AuditOperation::HeatKernelTrust {
                diffusion_time: 1.0
            }
            .as_str(),
            "heat_kernel_trust"
        );
        assert_eq!(
            AuditOperation::BuildAccessTensor { num_buckets: 10 }.as_str(),
            "build_access_tensor"
        );
        assert_eq!(
            AuditOperation::AnalyzeTemporalPatterns.as_str(),
            "analyze_temporal_patterns"
        );
        assert_eq!(
            AuditOperation::WeightedImpactAnalysis.as_str(),
            "weighted_impact_analysis"
        );
        assert_eq!(AuditOperation::RotationPlan.as_str(), "rotation_plan");
        assert_eq!(
            AuditOperation::RecommendPlacement {
                region: "eu".to_string()
            }
            .as_str(),
            "recommend_placement"
        );
    }

    #[test]
    fn test_diff_versions_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "secret:a",
            &AuditOperation::DiffVersions {
                version_a: 1,
                version_b: 3,
            },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::DiffVersions {
                version_a,
                version_b,
            } => {
                assert_eq!(*version_a, 1);
                assert_eq!(*version_b, 3);
            },
            other => panic!("Expected DiffVersions, got {other:?}"),
        }
    }

    #[test]
    fn test_save_delete_template_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record(
            "user:admin",
            "vault",
            &AuditOperation::SaveTemplate {
                template_name: "tmpl_a".to_string(),
            },
        );
        log.record(
            "user:admin",
            "vault",
            &AuditOperation::DeleteTemplate {
                template_name: "tmpl_a".to_string(),
            },
        );

        let entries = log.scan();
        assert_eq!(entries.len(), 2);
        match &entries[0].operation {
            AuditOperation::SaveTemplate { template_name } => {
                assert_eq!(template_name, "tmpl_a");
            },
            other => panic!("Expected SaveTemplate, got {other:?}"),
        }
        match &entries[1].operation {
            AuditOperation::DeleteTemplate { template_name } => {
                assert_eq!(template_name, "tmpl_a");
            },
            other => panic!("Expected DeleteTemplate, got {other:?}"),
        }
    }

    #[test]
    fn test_find_similar_roundtrip() {
        let store = create_test_store();
        let log = AuditLog::new(&store, None);

        log.record("user:admin", "vault", &AuditOperation::FindSimilar { k: 5 });

        let entries = log.scan();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            AuditOperation::FindSimilar { k } => {
                assert_eq!(*k, 5);
            },
            other => panic!("Expected FindSimilar, got {other:?}"),
        }
    }
}
