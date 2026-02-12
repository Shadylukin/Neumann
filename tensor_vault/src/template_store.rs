// SPDX-License-Identifier: MIT OR Apache-2.0
//! Persistent template storage for reusable secret generation templates.

use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::dynamic::SecretTemplate;
use crate::{Result, VaultError};

/// Storage prefix for template entries.
const TEMPLATE_PREFIX: &str = "_vtpl:";

/// A persisted secret template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTemplate {
    pub name: String,
    pub template: SecretTemplate,
    pub created_by: String,
    pub created_at: i64,
}

/// Thread-safe manager for secret generation templates.
pub struct TemplateManager {
    templates: DashMap<String, StoredTemplate>,
}

impl Default for TemplateManager {
    fn default() -> Self {
        Self {
            templates: DashMap::new(),
        }
    }
}

impl TemplateManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load templates from persistent storage.
    pub fn load(store: &TensorStore) -> Self {
        let manager = Self::new();
        for key in store.scan(TEMPLATE_PREFIX) {
            if let Some(name) = key.strip_prefix(TEMPLATE_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(stored) = deserialize_template(name, &tensor) {
                        manager.templates.insert(name.to_string(), stored);
                    }
                }
            }
        }
        manager
    }

    /// Save a template to storage.
    pub fn save(
        &self,
        store: &TensorStore,
        name: &str,
        template: SecretTemplate,
        creator: &str,
    ) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let stored = StoredTemplate {
            name: name.to_string(),
            template,
            created_by: creator.to_string(),
            created_at: now,
        };

        let key = format!("{TEMPLATE_PREFIX}{name}");
        let tensor = serialize_template(&stored);
        store
            .put(&key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;
        self.templates.insert(name.to_string(), stored);
        Ok(())
    }

    /// Get a template by name.
    pub fn get(&self, name: &str) -> Option<StoredTemplate> {
        self.templates.get(name).map(|t| t.clone())
    }

    /// List all template names.
    pub fn list(&self) -> Vec<String> {
        self.templates.iter().map(|e| e.key().clone()).collect()
    }

    /// Delete a template from storage.
    pub fn delete(&self, store: &TensorStore, name: &str) -> Result<()> {
        let key = format!("{TEMPLATE_PREFIX}{name}");
        store.delete(&key).ok();
        self.templates.remove(name);
        Ok(())
    }
}

fn serialize_template(stored: &StoredTemplate) -> TensorData {
    let mut tensor = TensorData::new();
    let json = serde_json::to_string(&stored.template).unwrap_or_default();
    tensor.set(
        "_template_json",
        TensorValue::Scalar(ScalarValue::String(json)),
    );
    tensor.set(
        "_created_by",
        TensorValue::Scalar(ScalarValue::String(stored.created_by.clone())),
    );
    tensor.set(
        "_created_at",
        TensorValue::Scalar(ScalarValue::Int(stored.created_at)),
    );
    tensor
}

fn deserialize_template(name: &str, tensor: &TensorData) -> Option<StoredTemplate> {
    let json = match tensor.get("_template_json") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let template: SecretTemplate = serde_json::from_str(&json).ok()?;
    let created_by = match tensor.get("_created_by") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => String::new(),
    };
    let created_at = match tensor.get("_created_at") {
        Some(TensorValue::Scalar(ScalarValue::Int(ts))) => *ts,
        _ => 0,
    };
    Some(StoredTemplate {
        name: name.to_string(),
        template,
        created_by,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic::{PasswordCharset, PasswordConfig};

    fn test_template() -> SecretTemplate {
        SecretTemplate::Password(PasswordConfig {
            length: 16,
            charset: PasswordCharset::Alphanumeric,
            require_uppercase: true,
            require_digit: true,
            require_special: false,
        })
    }

    #[test]
    fn test_template_save_get_list_delete() {
        let store = TensorStore::new();
        let mgr = TemplateManager::new();

        mgr.save(&store, "db-password", test_template(), "node:root")
            .unwrap();
        mgr.save(&store, "api-token", test_template(), "node:root")
            .unwrap();

        let got = mgr.get("db-password").unwrap();
        assert_eq!(got.name, "db-password");
        assert_eq!(got.created_by, "node:root");
        assert!(got.created_at > 0);

        let names = mgr.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"db-password".to_string()));
        assert!(names.contains(&"api-token".to_string()));

        mgr.delete(&store, "db-password").unwrap();
        assert!(mgr.get("db-password").is_none());
        assert_eq!(mgr.list().len(), 1);
    }

    #[test]
    fn test_template_persistence() {
        let store = TensorStore::new();
        let mgr = TemplateManager::new();
        mgr.save(&store, "persistent", test_template(), "admin")
            .unwrap();

        // Simulate restart: create a new manager and load from store
        let mgr2 = TemplateManager::load(&store);
        let loaded = mgr2.get("persistent").unwrap();
        assert_eq!(loaded.name, "persistent");
        assert_eq!(loaded.created_by, "admin");
    }

    #[test]
    fn test_template_not_found() {
        let mgr = TemplateManager::new();
        assert!(mgr.get("nonexistent").is_none());
    }
}
