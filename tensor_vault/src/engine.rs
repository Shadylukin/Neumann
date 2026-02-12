// SPDX-License-Identifier: MIT OR Apache-2.0
//! Plugin architecture for custom secret engines.

use dashmap::DashMap;

use crate::{Result, VaultError};

/// Trait for pluggable secret engines.
pub trait SecretEngine: Send + Sync {
    /// Engine name (must be unique).
    fn name(&self) -> &str;

    /// Generate a new secret from the given parameters.
    fn generate(&self, params: &serde_json::Value) -> Result<String>;

    /// Renew an existing secret.
    fn renew(&self, secret_id: &str, params: &serde_json::Value) -> Result<String>;

    /// Revoke a secret.
    fn revoke(&self, secret_id: &str) -> Result<()>;

    /// List active secrets managed by this engine.
    fn list(&self) -> Result<Vec<String>>;
}

/// Registry of secret engines.
pub struct EngineRegistry {
    engines: DashMap<String, Box<dyn SecretEngine>>,
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self {
            engines: DashMap::new(),
        }
    }
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new engine. Replaces any existing engine with the same name.
    pub fn register(&self, engine: Box<dyn SecretEngine>) -> Result<()> {
        let name = engine.name().to_string();
        self.engines.insert(name, engine);
        Ok(())
    }

    /// Unregister an engine by name.
    pub fn unregister(&self, name: &str) -> Result<()> {
        self.engines.remove(name);
        Ok(())
    }

    /// List all registered engine names.
    pub fn list_engines(&self) -> Vec<String> {
        self.engines.iter().map(|e| e.key().clone()).collect()
    }

    /// Generate a secret using the named engine.
    pub fn generate(&self, engine_name: &str, params: &serde_json::Value) -> Result<String> {
        let engine = self
            .engines
            .get(engine_name)
            .ok_or_else(|| VaultError::EngineNotFound(engine_name.to_string()))?;
        engine.generate(params)
    }

    /// Revoke a secret using the named engine.
    pub fn revoke(&self, engine_name: &str, secret_id: &str) -> Result<()> {
        let engine = self
            .engines
            .get(engine_name)
            .ok_or_else(|| VaultError::EngineNotFound(engine_name.to_string()))?;
        engine.revoke(secret_id)
    }

    /// Renew a secret using the named engine.
    pub fn renew(
        &self,
        engine_name: &str,
        secret_id: &str,
        params: &serde_json::Value,
    ) -> Result<String> {
        let engine = self
            .engines
            .get(engine_name)
            .ok_or_else(|| VaultError::EngineNotFound(engine_name.to_string()))?;
        engine.renew(secret_id, params)
    }

    /// List secrets from the named engine.
    pub fn list_secrets(&self, engine_name: &str) -> Result<Vec<String>> {
        let engine = self
            .engines
            .get(engine_name)
            .ok_or_else(|| VaultError::EngineNotFound(engine_name.to_string()))?;
        engine.list()
    }

    /// Check if an engine is registered.
    pub fn has_engine(&self, name: &str) -> bool {
        self.engines.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// A test engine that generates incrementing IDs.
    struct TestEngine {
        name: String,
        secrets: Mutex<Vec<String>>,
        counter: Mutex<u64>,
    }

    impl TestEngine {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                secrets: Mutex::new(Vec::new()),
                counter: Mutex::new(0),
            }
        }
    }

    impl SecretEngine for TestEngine {
        fn name(&self) -> &str {
            &self.name
        }

        fn generate(&self, _params: &serde_json::Value) -> Result<String> {
            let mut counter = self.counter.lock().unwrap();
            *counter += 1;
            let id = format!("{}_{}", self.name, counter);
            self.secrets.lock().unwrap().push(id.clone());
            Ok(id)
        }

        fn renew(&self, secret_id: &str, _params: &serde_json::Value) -> Result<String> {
            Ok(format!("{secret_id}_renewed"))
        }

        fn revoke(&self, secret_id: &str) -> Result<()> {
            let mut secrets = self.secrets.lock().unwrap();
            secrets.retain(|s| s != secret_id);
            Ok(())
        }

        fn list(&self) -> Result<Vec<String>> {
            Ok(self.secrets.lock().unwrap().clone())
        }
    }

    #[test]
    fn test_register_and_list() {
        let registry = EngineRegistry::new();
        registry
            .register(Box::new(TestEngine::new("test")))
            .unwrap();

        let engines = registry.list_engines();
        assert_eq!(engines.len(), 1);
        assert!(engines.contains(&"test".to_string()));
    }

    #[test]
    fn test_unregister() {
        let registry = EngineRegistry::new();
        registry
            .register(Box::new(TestEngine::new("test")))
            .unwrap();
        registry.unregister("test").unwrap();

        assert!(registry.list_engines().is_empty());
    }

    #[test]
    fn test_generate() {
        let registry = EngineRegistry::new();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();

        let secret = registry.generate("db", &serde_json::json!({})).unwrap();
        assert!(secret.starts_with("db_"));
    }

    #[test]
    fn test_generate_not_found() {
        let registry = EngineRegistry::new();
        let result = registry.generate("missing", &serde_json::json!({}));
        assert!(matches!(result, Err(VaultError::EngineNotFound(_))));
    }

    #[test]
    fn test_revoke() {
        let registry = EngineRegistry::new();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();

        let secret = registry.generate("db", &serde_json::json!({})).unwrap();
        registry.revoke("db", &secret).unwrap();

        let remaining = registry.list_secrets("db").unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_renew() {
        let registry = EngineRegistry::new();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();

        let renewed = registry
            .renew("db", "old_id", &serde_json::json!({}))
            .unwrap();
        assert_eq!(renewed, "old_id_renewed");
    }

    #[test]
    fn test_has_engine() {
        let registry = EngineRegistry::new();
        assert!(!registry.has_engine("test"));

        registry
            .register(Box::new(TestEngine::new("test")))
            .unwrap();
        assert!(registry.has_engine("test"));
    }

    #[test]
    fn test_multiple_engines() {
        let registry = EngineRegistry::new();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();
        registry
            .register(Box::new(TestEngine::new("cache")))
            .unwrap();

        assert_eq!(registry.list_engines().len(), 2);

        let db_secret = registry.generate("db", &serde_json::json!({})).unwrap();
        assert!(db_secret.starts_with("db_"));

        let cache_secret = registry.generate("cache", &serde_json::json!({})).unwrap();
        assert!(cache_secret.starts_with("cache_"));
    }

    #[test]
    fn test_replace_engine() {
        let registry = EngineRegistry::new();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();
        registry.register(Box::new(TestEngine::new("db"))).unwrap();

        assert_eq!(registry.list_engines().len(), 1);
    }

    #[test]
    fn test_list_secrets_not_found() {
        let registry = EngineRegistry::new();
        let result = registry.list_secrets("missing");
        assert!(matches!(result, Err(VaultError::EngineNotFound(_))));
    }
}
