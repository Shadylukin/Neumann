// SPDX-License-Identifier: MIT OR Apache-2.0
//! PKI/Certificate engine for internal CA and certificate issuance.

use std::{
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use blake2::digest::Mac;

use rcgen::{Certificate, CertificateParams, DistinguishedName, DnType, IsCa, KeyPair, SanType};
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{engine::SecretEngine, Result, VaultError};

/// Storage prefix for certificates.
const PKI_PREFIX: &str = "_vpki:";
/// Storage key for CA material.
const CA_KEY: &str = "_vpki_ca:";

/// Information about an issued certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertInfo {
    /// Serial number (hex string).
    pub serial: String,
    /// Subject common name.
    pub subject: String,
    /// Issuer common name.
    pub issuer: String,
    /// Not-before timestamp (unix millis).
    pub not_before_ms: i64,
    /// Not-after timestamp (unix millis).
    pub not_after_ms: i64,
    /// Whether the certificate has been revoked.
    pub revoked: bool,
}

/// Certificate signing request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateRequest {
    /// Common name for the subject.
    pub common_name: String,
    /// Organization name.
    pub organization: Option<String>,
    /// DNS subject alternative names.
    pub san_dns: Vec<String>,
    /// IP subject alternative names.
    pub san_ip: Vec<String>,
}

/// A single revocation entry in the CRL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationEntry {
    /// Serial number of the revoked certificate.
    pub serial: String,
    /// Timestamp when the certificate was revoked (unix millis).
    pub revoked_at_ms: i64,
}

/// Certificate revocation list signed by the CA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationList {
    /// Issuer name.
    pub issuer: String,
    /// When this CRL was generated (unix millis).
    pub generated_at_ms: i64,
    /// Revoked certificate entries.
    pub entries: Vec<RevocationEntry>,
    /// Blake2b HMAC signature over the CRL content.
    pub signature: String,
}

/// Storage key for the certificate revocation list.
const CRL_KEY: &str = "_vpki:crl";

/// PKI engine implementing the `SecretEngine` trait.
pub struct PkiEngine {
    name: String,
    certs: Mutex<Vec<CertInfo>>,
}

impl Default for PkiEngine {
    fn default() -> Self {
        Self {
            name: "pki".to_string(),
            certs: Mutex::new(Vec::new()),
        }
    }
}

impl PkiEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the CA by generating a self-signed root certificate.
    pub fn init_ca(store: &TensorStore) -> Result<Vec<u8>> {
        let mut params = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "Neumann Vault Internal CA");
        dn.push(DnType::OrganizationName, "Neumann Vault");
        params.distinguished_name = dn;
        params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        params.not_before = rcgen::date_time_ymd(2024, 1, 1);
        params.not_after = rcgen::date_time_ymd(2034, 12, 31);

        let key_pair = KeyPair::generate()
            .map_err(|e| VaultError::CryptoError(format!("CA key generation failed: {e}")))?;

        let ca_cert = params.self_signed(&key_pair).map_err(|e| {
            VaultError::CryptoError(format!("CA certificate generation failed: {e}"))
        })?;

        let ca_pem = ca_cert.pem();
        let key_pem = key_pair.serialize_pem();

        // Store CA material
        let mut tensor = TensorData::new();
        tensor.set(
            "_ca_cert",
            TensorValue::Scalar(ScalarValue::String(ca_pem.clone())),
        );
        tensor.set("_ca_key", TensorValue::Scalar(ScalarValue::String(key_pem)));
        store
            .put(CA_KEY, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        Ok(ca_pem.into_bytes())
    }

    /// Load the CA key pair and reconstructed certificate from storage.
    fn load_ca(store: &TensorStore) -> Result<(KeyPair, Certificate)> {
        let ca_tensor = store
            .get(CA_KEY)
            .map_err(|_| VaultError::NotFound("CA not initialized".to_string()))?;

        let ca_key_pem = match ca_tensor.get("_ca_key") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return Err(VaultError::NotFound("CA key missing".to_string())),
        };

        let ca_key_pair = KeyPair::from_pem(&ca_key_pem)
            .map_err(|e| VaultError::CryptoError(format!("CA key parse failed: {e}")))?;

        let mut ca_params = CertificateParams::default();
        let mut ca_dn = DistinguishedName::new();
        ca_dn.push(DnType::CommonName, "Neumann Vault Internal CA");
        ca_dn.push(DnType::OrganizationName, "Neumann Vault");
        ca_params.distinguished_name = ca_dn;
        ca_params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params.not_before = rcgen::date_time_ymd(2024, 1, 1);
        ca_params.not_after = rcgen::date_time_ymd(2034, 12, 31);

        let ca_cert = ca_params
            .self_signed(&ca_key_pair)
            .map_err(|e| VaultError::CryptoError(format!("CA cert reconstruction failed: {e}")))?;

        Ok((ca_key_pair, ca_cert))
    }

    /// Issue a certificate signed by the CA.
    pub fn issue_certificate(
        store: &TensorStore,
        request: &CertificateRequest,
        ttl: Duration,
    ) -> Result<(String, Vec<u8>)> {
        let (ca_key_pair, ca_cert) = Self::load_ca(store)?;

        // Build the leaf certificate
        let mut params = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, &request.common_name);
        if let Some(org) = &request.organization {
            dn.push(DnType::OrganizationName, org);
        }
        params.distinguished_name = dn;
        params.is_ca = IsCa::NoCa;

        // SANs
        let mut sans = Vec::new();
        for dns in &request.san_dns {
            sans.push(SanType::DnsName(dns.clone().try_into().map_err(|e| {
                VaultError::CryptoError(format!("invalid DNS SAN: {e}"))
            })?));
        }
        for ip_str in &request.san_ip {
            let ip: std::net::IpAddr = ip_str
                .parse()
                .map_err(|e| VaultError::CryptoError(format!("invalid IP SAN: {e}")))?;
            sans.push(SanType::IpAddress(ip));
        }
        params.subject_alt_names = sans;

        // TTL
        let now = SystemTime::now();
        let not_after = now + ttl;
        let now_secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let after_secs = not_after
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate the leaf key pair and sign with CA
        let leaf_key = KeyPair::generate()
            .map_err(|e| VaultError::CryptoError(format!("leaf key generation failed: {e}")))?;

        let leaf_cert = params
            .signed_by(&leaf_key, &ca_cert, &ca_key_pair)
            .map_err(|e| VaultError::CryptoError(format!("certificate signing failed: {e}")))?;

        let cert_pem = leaf_cert.pem();

        // Generate a unique serial number using nanosecond timestamp
        let nanos = now
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let serial = format!("{nanos:032x}");

        // Store certificate info
        let cert_info = CertInfo {
            serial: serial.clone(),
            subject: request.common_name.clone(),
            issuer: "Neumann Vault Internal CA".to_string(),
            not_before_ms: i64::try_from(now_secs.saturating_mul(1000)).unwrap_or(i64::MAX),
            not_after_ms: i64::try_from(after_secs.saturating_mul(1000)).unwrap_or(i64::MAX),
            revoked: false,
        };

        let storage_key = format!("{PKI_PREFIX}{serial}");
        let mut tensor = TensorData::new();
        tensor.set(
            "_serial",
            TensorValue::Scalar(ScalarValue::String(serial.clone())),
        );
        tensor.set(
            "_subject",
            TensorValue::Scalar(ScalarValue::String(request.common_name.clone())),
        );
        tensor.set(
            "_issuer",
            TensorValue::Scalar(ScalarValue::String(cert_info.issuer.clone())),
        );
        tensor.set(
            "_not_before",
            TensorValue::Scalar(ScalarValue::Int(cert_info.not_before_ms)),
        );
        tensor.set(
            "_not_after",
            TensorValue::Scalar(ScalarValue::Int(cert_info.not_after_ms)),
        );
        tensor.set("_revoked", TensorValue::Scalar(ScalarValue::Bool(false)));
        tensor.set(
            "_cert_pem",
            TensorValue::Scalar(ScalarValue::String(cert_pem)),
        );
        tensor.set(
            "_key_pem",
            TensorValue::Scalar(ScalarValue::String(leaf_key.serialize_pem())),
        );

        store
            .put(&storage_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        Ok((serial, cert_info.issuer.into_bytes()))
    }

    /// Check whether a certificate is revoked by serial number.
    pub fn is_revoked(store: &TensorStore, serial: &str) -> bool {
        let storage_key = format!("{PKI_PREFIX}{serial}");
        store.get(&storage_key).ok().is_some_and(|tensor| {
            matches!(
                tensor.get("_revoked"),
                Some(TensorValue::Scalar(ScalarValue::Bool(true)))
            )
        })
    }

    /// Generate a CRL by scanning all certificates, persisting at `CRL_KEY`.
    #[allow(clippy::missing_panics_doc)] // Blake2b MAC with fixed key never fails
    pub fn generate_crl(store: &TensorStore) -> Result<RevocationList> {
        let mut entries = Vec::new();

        for key in store.scan(PKI_PREFIX) {
            // Skip the CRL key itself
            if key == CRL_KEY {
                continue;
            }
            if let Some(serial) = key.strip_prefix(PKI_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    let revoked = matches!(
                        tensor.get("_revoked"),
                        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
                    );
                    if revoked {
                        let revoked_at_ms = match tensor.get("_revoked_at") {
                            Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
                            _ => 0,
                        };
                        entries.push(RevocationEntry {
                            serial: serial.to_string(),
                            revoked_at_ms,
                        });
                    }
                }
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // Compute Blake2b HMAC over serialized entries + timestamp
        let payload = format!(
            "{now}:{}",
            serde_json::to_string(&entries).unwrap_or_default()
        );
        let ca_material = store
            .get(CA_KEY)
            .ok()
            .and_then(|t| match t.get("_ca_key") {
                Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let mut mac =
            blake2::Blake2bMac512::new_from_slice(ca_material.as_bytes()).unwrap_or_else(|_| {
                blake2::Blake2bMac512::new_from_slice(b"neumann-vault-crl-default")
                    .expect("valid key length")
            });
        mac.update(payload.as_bytes());
        let sig_bytes = mac.finalize().into_bytes();
        let signature: String = sig_bytes.iter().map(|b| format!("{b:02x}")).collect();

        let crl = RevocationList {
            issuer: "Neumann Vault Internal CA".to_string(),
            generated_at_ms: now,
            entries,
            signature,
        };

        // Persist
        let json = serde_json::to_string(&crl)
            .map_err(|e| VaultError::StorageError(format!("CRL serialization failed: {e}")))?;
        let mut tensor = TensorData::new();
        tensor.set("_crl", TensorValue::Scalar(ScalarValue::String(json)));
        store
            .put(CRL_KEY, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        Ok(crl)
    }

    /// Load the persisted CRL from storage.
    pub fn get_revocation_list(store: &TensorStore) -> Result<RevocationList> {
        let tensor = store
            .get(CRL_KEY)
            .map_err(|_| VaultError::NotFound("CRL not found".to_string()))?;

        match tensor.get("_crl") {
            Some(TensorValue::Scalar(ScalarValue::String(json))) => serde_json::from_str(json)
                .map_err(|e| VaultError::StorageError(format!("CRL parse failed: {e}"))),
            _ => Err(VaultError::NotFound("CRL data missing".to_string())),
        }
    }

    /// Revoke a certificate by serial number.
    pub fn revoke_certificate(store: &TensorStore, serial: &str) -> Result<()> {
        let storage_key = format!("{PKI_PREFIX}{serial}");
        let mut tensor = store
            .get(&storage_key)
            .map_err(|_| VaultError::NotFound(format!("certificate: {serial}")))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        tensor.set("_revoked", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("_revoked_at", TensorValue::Scalar(ScalarValue::Int(now)));
        store
            .put(&storage_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Auto-regenerate CRL after revocation
        Self::generate_crl(store)?;
        Ok(())
    }

    /// List all certificates.
    pub fn list_certificates(store: &TensorStore) -> Vec<CertInfo> {
        let mut certs = Vec::new();
        for key in store.scan(PKI_PREFIX) {
            if let Some(serial) = key.strip_prefix(PKI_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(info) = parse_cert_info(serial, &tensor) {
                        certs.push(info);
                    }
                }
            }
        }
        certs
    }

    /// Get the CA certificate PEM.
    pub fn get_ca_certificate(store: &TensorStore) -> Result<Vec<u8>> {
        let tensor = store
            .get(CA_KEY)
            .map_err(|_| VaultError::NotFound("CA not initialized".to_string()))?;

        match tensor.get("_ca_cert") {
            Some(TensorValue::Scalar(ScalarValue::String(pem))) => Ok(pem.as_bytes().to_vec()),
            _ => Err(VaultError::NotFound("CA certificate missing".to_string())),
        }
    }
}

impl SecretEngine for PkiEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(&self, params: &serde_json::Value) -> Result<String> {
        let cn = params["common_name"]
            .as_str()
            .unwrap_or("generated")
            .to_string();
        Ok(format!("pki_cert_{cn}"))
    }

    fn renew(&self, secret_id: &str, _params: &serde_json::Value) -> Result<String> {
        Ok(format!("{secret_id}_renewed"))
    }

    fn revoke(&self, secret_id: &str) -> Result<()> {
        let mut certs = self.certs.lock().unwrap();
        if let Some(cert) = certs.iter_mut().find(|c| c.serial == secret_id) {
            cert.revoked = true;
        }
        drop(certs);
        Ok(())
    }

    fn list(&self) -> Result<Vec<String>> {
        let certs = self.certs.lock().unwrap();
        Ok(certs.iter().map(|c| c.serial.clone()).collect())
    }
}

fn parse_cert_info(serial: &str, tensor: &TensorData) -> Option<CertInfo> {
    let subject = match tensor.get("_subject") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let issuer = match tensor.get("_issuer") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let not_before_ms = match tensor.get("_not_before") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    let not_after_ms = match tensor.get("_not_after") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    let revoked = matches!(
        tensor.get("_revoked"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    );

    Some(CertInfo {
        serial: serial.to_string(),
        subject,
        issuer,
        not_before_ms,
        not_after_ms,
        revoked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_ca() {
        let store = TensorStore::new();
        let ca_pem = PkiEngine::init_ca(&store).unwrap();
        assert!(!ca_pem.is_empty());

        let ca_str = String::from_utf8(ca_pem).unwrap();
        assert!(ca_str.contains("BEGIN CERTIFICATE"));
    }

    #[test]
    fn test_get_ca_certificate() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let ca = PkiEngine::get_ca_certificate(&store).unwrap();
        let pem = String::from_utf8(ca).unwrap();
        assert!(pem.contains("BEGIN CERTIFICATE"));
    }

    #[test]
    fn test_get_ca_without_init_fails() {
        let store = TensorStore::new();
        let result = PkiEngine::get_ca_certificate(&store);
        assert!(result.is_err());
    }

    #[test]
    fn test_issue_certificate() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "test.example.com".to_string(),
            organization: Some("Test Org".to_string()),
            san_dns: vec!["test.example.com".to_string()],
            san_ip: vec![],
        };

        let (serial, _) =
            PkiEngine::issue_certificate(&store, &request, Duration::from_secs(86400)).unwrap();
        assert!(!serial.is_empty());
    }

    #[test]
    fn test_list_certificates() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "svc.local".to_string(),
            organization: None,
            san_dns: vec!["svc.local".to_string()],
            san_ip: vec![],
        };

        PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();
        let certs = PkiEngine::list_certificates(&store);
        assert_eq!(certs.len(), 1);
        assert_eq!(certs[0].subject, "svc.local");
        assert!(!certs[0].revoked);
    }

    #[test]
    fn test_revoke_certificate() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "revoke-me.local".to_string(),
            organization: None,
            san_dns: vec!["revoke-me.local".to_string()],
            san_ip: vec![],
        };

        let (serial, _) =
            PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();

        PkiEngine::revoke_certificate(&store, &serial).unwrap();

        let certs = PkiEngine::list_certificates(&store);
        assert!(certs[0].revoked);
    }

    #[test]
    fn test_issue_with_ip_san() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "internal".to_string(),
            organization: None,
            san_dns: vec![],
            san_ip: vec!["192.168.1.1".to_string()],
        };

        let result = PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600));
        assert!(result.is_ok());
    }

    #[test]
    fn test_pki_engine_trait() {
        let engine = PkiEngine::new();
        assert_eq!(engine.name(), "pki");

        let result = engine.generate(&serde_json::json!({"common_name": "test"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_cert_info_serialization() {
        let info = CertInfo {
            serial: "001".to_string(),
            subject: "test.com".to_string(),
            issuer: "CA".to_string(),
            not_before_ms: 1000,
            not_after_ms: 2000,
            revoked: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deser: CertInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.serial, "001");
    }

    #[test]
    fn test_revoke_nonexistent() {
        let store = TensorStore::new();
        let result = PkiEngine::revoke_certificate(&store, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_certificates() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        for i in 0..3 {
            let request = CertificateRequest {
                common_name: format!("svc{i}.local"),
                organization: None,
                san_dns: vec![format!("svc{i}.local")],
                san_ip: vec![],
            };
            PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let certs = PkiEngine::list_certificates(&store);
        assert_eq!(certs.len(), 3);
    }

    #[test]
    fn test_is_revoked_true() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "revoke-check.local".to_string(),
            organization: None,
            san_dns: vec!["revoke-check.local".to_string()],
            san_ip: vec![],
        };
        let (serial, _) =
            PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();

        assert!(!PkiEngine::is_revoked(&store, &serial));
        PkiEngine::revoke_certificate(&store, &serial).unwrap();
        assert!(PkiEngine::is_revoked(&store, &serial));
    }

    #[test]
    fn test_is_revoked_nonexistent() {
        let store = TensorStore::new();
        assert!(!PkiEngine::is_revoked(&store, "nonexistent_serial"));
    }

    #[test]
    fn test_generate_and_get_crl() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "crl-test.local".to_string(),
            organization: None,
            san_dns: vec!["crl-test.local".to_string()],
            san_ip: vec![],
        };
        let (serial, _) =
            PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();
        PkiEngine::revoke_certificate(&store, &serial).unwrap();

        let crl = PkiEngine::get_revocation_list(&store).unwrap();
        assert_eq!(crl.issuer, "Neumann Vault Internal CA");
        assert_eq!(crl.entries.len(), 1);
        assert_eq!(crl.entries[0].serial, serial);
        assert!(!crl.signature.is_empty());
    }

    #[test]
    fn test_crl_auto_updates_on_revoke() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let mut serials = Vec::new();
        for i in 0..2 {
            let request = CertificateRequest {
                common_name: format!("auto-crl-{i}.local"),
                organization: None,
                san_dns: vec![format!("auto-crl-{i}.local")],
                san_ip: vec![],
            };
            let (s, _) =
                PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();
            serials.push(s);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        PkiEngine::revoke_certificate(&store, &serials[0]).unwrap();
        let crl1 = PkiEngine::get_revocation_list(&store).unwrap();
        assert_eq!(crl1.entries.len(), 1);

        PkiEngine::revoke_certificate(&store, &serials[1]).unwrap();
        let crl2 = PkiEngine::get_revocation_list(&store).unwrap();
        assert_eq!(crl2.entries.len(), 2);
    }

    #[test]
    fn test_crl_empty_when_no_revocations() {
        let store = TensorStore::new();
        PkiEngine::init_ca(&store).unwrap();

        let request = CertificateRequest {
            common_name: "active.local".to_string(),
            organization: None,
            san_dns: vec!["active.local".to_string()],
            san_ip: vec![],
        };
        PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600)).unwrap();

        let crl = PkiEngine::generate_crl(&store).unwrap();
        assert!(crl.entries.is_empty());
    }

    #[test]
    fn test_issue_without_ca_fails() {
        let store = TensorStore::new();
        let request = CertificateRequest {
            common_name: "test".to_string(),
            organization: None,
            san_dns: vec![],
            san_ip: vec![],
        };
        let result = PkiEngine::issue_certificate(&store, &request, Duration::from_secs(3600));
        assert!(result.is_err());
    }
}
