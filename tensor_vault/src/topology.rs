// SPDX-License-Identifier: MIT OR Apache-2.0
//! Access topology extraction and tensor-compressed policy analysis.

use std::collections::HashMap;

use tensor_compress::tensor_train::{tt_decompose, TTConfig, TTVector};

use crate::vault::Vault;
use crate::{Permission, Result};

/// Configuration for access topology extraction.
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Whether to apply Tensor Train compression to the access matrix.
    pub enable_tt_compression: bool,
    /// Minimum matrix size (entities * secrets) to trigger TT compression.
    pub compression_threshold: usize,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            enable_tt_compression: true,
            compression_threshold: 10_000,
        }
    }
}

/// Access topology: a flat matrix encoding entity-secret permission relationships.
///
/// The matrix has shape [entities x secrets x 3] where the 3 channels are
/// (has_read, has_write, has_admin) as 0.0/1.0 floats.
pub struct AccessTopology {
    entity_index: HashMap<String, usize>,
    secret_index: HashMap<String, usize>,
    access_matrix: Vec<f32>,
    tt_compressed: Option<TTVector>,
    dimensions: (usize, usize, usize),
}

/// Result of a batch permission check.
#[derive(Debug, Clone)]
pub struct BatchPermissionResult {
    pub entity: String,
    pub secret: String,
    pub has_read: bool,
    pub has_write: bool,
    pub has_admin: bool,
}

/// Report on policy redundancy and merge opportunities.
#[derive(Debug, Clone)]
pub struct PolicyRedundancyReport {
    pub mergeable_groups: Vec<Vec<String>>,
    pub compression_ratio: f32,
    pub total_policies: usize,
}

impl AccessTopology {
    /// Build an access topology from the vault's current state.
    pub fn from_vault(vault: &Vault, config: TopologyConfig) -> Result<Self> {
        // Collect all accessible secrets (as root)
        let secrets = vault.list("node:root", "*").unwrap_or_default();

        // Collect all entity nodes from the graph
        let entities = collect_entities(vault);

        let num_entities = entities.len();
        let num_secrets = secrets.len();
        let channels = 3;

        if num_entities == 0 || num_secrets == 0 {
            return Ok(Self {
                entity_index: HashMap::new(),
                secret_index: HashMap::new(),
                access_matrix: Vec::new(),
                tt_compressed: None,
                dimensions: (0, 0, channels),
            });
        }

        let mut entity_index = HashMap::with_capacity(num_entities);
        let mut secret_index = HashMap::with_capacity(num_secrets);

        for (i, entity) in entities.iter().enumerate() {
            entity_index.insert(entity.clone(), i);
        }
        for (i, secret) in secrets.iter().enumerate() {
            secret_index.insert(secret.clone(), i);
        }

        // Build flat access matrix [entities x secrets x 3]
        let matrix_size = num_entities * num_secrets * channels;
        let mut access_matrix = vec![0.0_f32; matrix_size];

        for (ei, entity) in entities.iter().enumerate() {
            for (si, secret) in secrets.iter().enumerate() {
                if let Some(perm) = vault.get_permission(entity, secret) {
                    let base = (ei * num_secrets + si) * channels;
                    access_matrix[base] = if perm.allows(Permission::Read) {
                        1.0
                    } else {
                        0.0
                    };
                    access_matrix[base + 1] = if perm.allows(Permission::Write) {
                        1.0
                    } else {
                        0.0
                    };
                    access_matrix[base + 2] = if perm == Permission::Admin { 1.0 } else { 0.0 };
                }
            }
        }

        // Optionally apply TT compression
        let total_elements = num_entities * num_secrets;
        let tt_compressed =
            if config.enable_tt_compression && total_elements >= config.compression_threshold {
                try_tt_compress(&access_matrix)
            } else {
                None
            };

        Ok(Self {
            entity_index,
            secret_index,
            access_matrix,
            tt_compressed,
            dimensions: (num_entities, num_secrets, channels),
        })
    }

    /// Batch permission check for multiple (entity, secret) pairs.
    pub fn batch_check(&self, queries: &[(&str, &str)]) -> Vec<BatchPermissionResult> {
        let channels = self.dimensions.2;
        let num_secrets = self.dimensions.1;

        queries
            .iter()
            .map(|(entity, secret)| {
                let ei = self.entity_index.get(*entity);
                let si = self.secret_index.get(*secret);

                match (ei, si) {
                    (Some(&e), Some(&s)) => {
                        let base = (e * num_secrets + s) * channels;
                        BatchPermissionResult {
                            entity: (*entity).to_string(),
                            secret: (*secret).to_string(),
                            has_read: self.access_matrix.get(base).is_some_and(|&v| v > 0.5),
                            has_write: self.access_matrix.get(base + 1).is_some_and(|&v| v > 0.5),
                            has_admin: self.access_matrix.get(base + 2).is_some_and(|&v| v > 0.5),
                        }
                    },
                    _ => BatchPermissionResult {
                        entity: (*entity).to_string(),
                        secret: (*secret).to_string(),
                        has_read: false,
                        has_write: false,
                        has_admin: false,
                    },
                }
            })
            .collect()
    }

    /// Get compression statistics: (original_size, compressed_size, ratio).
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_stats(&self) -> (usize, usize, f32) {
        let original = self.access_matrix.len();
        if let Some(ref tt) = self.tt_compressed {
            let compressed = tt.storage_size();
            let ratio = if compressed > 0 {
                original as f32 / compressed as f32
            } else {
                0.0
            };
            (original, compressed, ratio)
        } else {
            (original, original, 1.0)
        }
    }

    /// Number of entities in the topology.
    pub fn num_entities(&self) -> usize {
        self.dimensions.0
    }

    /// Number of secrets in the topology.
    pub fn num_secrets(&self) -> usize {
        self.dimensions.1
    }
}

/// Analyze policy redundancy by computing pairwise Jaccard similarity.
#[allow(clippy::unnecessary_wraps, clippy::cast_precision_loss)]
pub fn analyze_policy_redundancy(vault: &Vault) -> Result<PolicyRedundancyReport> {
    let policies = vault.list_policies();
    let total_policies = policies.len();

    if total_policies < 2 {
        return Ok(PolicyRedundancyReport {
            mergeable_groups: Vec::new(),
            compression_ratio: 1.0,
            total_policies,
        });
    }

    // Compute pairwise Jaccard similarity on match pattern character sets
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut assigned = vec![false; total_policies];

    for i in 0..total_policies {
        if assigned[i] {
            continue;
        }
        let mut group = vec![policies[i].name.clone()];
        assigned[i] = true;

        for j in (i + 1)..total_policies {
            if assigned[j] {
                continue;
            }
            let sim = jaccard_similarity(&policies[i].match_pattern, &policies[j].match_pattern);
            if sim > 0.8 {
                group.push(policies[j].name.clone());
                assigned[j] = true;
            }
        }

        if group.len() > 1 {
            groups.push(group);
        }
    }

    let merged_count: usize = groups.iter().map(Vec::len).sum();
    let compression_ratio = if merged_count > 0 {
        total_policies as f32 / (total_policies - merged_count + groups.len()) as f32
    } else {
        1.0
    };

    Ok(PolicyRedundancyReport {
        mergeable_groups: groups,
        compression_ratio,
        total_policies,
    })
}

/// Compute Jaccard similarity between two strings (based on character bigrams).
#[allow(clippy::cast_precision_loss)]
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let bigrams_a: std::collections::HashSet<(char, char)> =
        a.chars().zip(a.chars().skip(1)).collect();
    let bigrams_b: std::collections::HashSet<(char, char)> =
        b.chars().zip(b.chars().skip(1)).collect();

    if bigrams_a.is_empty() && bigrams_b.is_empty() {
        return 1.0;
    }

    let intersection = bigrams_a.intersection(&bigrams_b).count();
    let union = bigrams_a.union(&bigrams_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Collect entity node keys from the vault's graph.
fn collect_entities(vault: &Vault) -> Vec<String> {
    use graph_engine::PropertyValue;

    let mut entities = Vec::new();

    if let Ok(node_ids) = vault.graph.get_all_node_ids() {
        for node_id in node_ids {
            if let Ok(node) = vault.graph.get_node(node_id) {
                if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
                    // Include entities that are not secret nodes
                    if !key.starts_with("_vk:") && !key.starts_with("vault_secret:") {
                        entities.push(key.clone());
                    }
                }
            }
        }
    }
    entities
}

/// Attempt TT compression on the access matrix.
fn try_tt_compress(matrix: &[f32]) -> Option<TTVector> {
    let len = matrix.len();
    if len < 4 {
        return None;
    }

    // Find a factorization of len into factors close to sqrt
    let shape = factorize_for_tt(len)?;
    let config = TTConfig {
        shape,
        max_rank: 4,
        tolerance: 1e-4,
    };

    tt_decompose(matrix, &config).ok()
}

/// Factorize n into 2-4 factors suitable for TT decomposition.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn factorize_for_tt(n: usize) -> Option<Vec<usize>> {
    if n == 0 {
        return None;
    }

    // Try to find 2-factor decomposition
    let sqrt_n = (n as f64).sqrt() as usize;
    for f in (2..=sqrt_n).rev() {
        if n.is_multiple_of(f) {
            let other = n / f;
            // Try to further factorize for better TT shape
            let sqrt_f = (f as f64).sqrt() as usize;
            for ff in (2..=sqrt_f).rev() {
                if f.is_multiple_of(ff) {
                    let sqrt_o = (other as f64).sqrt() as usize;
                    for fo in (2..=sqrt_o).rev() {
                        if other.is_multiple_of(fo) {
                            return Some(vec![ff, f / ff, fo, other / fo]);
                        }
                    }
                    return Some(vec![ff, f / ff, other]);
                }
            }
            return Some(vec![f, other]);
        }
    }

    // Prime number: pad to nearest factorizable size isn't worth it
    None
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::{Permission, VaultConfig};

    fn make_vault() -> (Vault, Arc<GraphEngine>) {
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
    fn test_topology_extraction_small() {
        let (vault, _graph) = make_vault();

        // Create secrets and entities
        vault.set("node:root", "secret/a", "val_a").unwrap();
        vault.set("node:root", "secret/b", "val_b").unwrap();
        vault.set("node:root", "secret/c", "val_c").unwrap();

        vault
            .grant_with_permission("node:root", "user:alice", "secret/a", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission("node:root", "user:alice", "secret/b", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission("node:root", "user:bob", "secret/c", Permission::Admin)
            .unwrap();

        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        assert!(topo.num_secrets() >= 3);
        assert!(topo.num_entities() >= 2);
    }

    #[test]
    fn test_topology_batch_check_granted() {
        let (vault, _graph) = make_vault();

        vault.set("node:root", "db/password", "secret123").unwrap();
        vault
            .grant_with_permission("node:root", "app:backend", "db/password", Permission::Write)
            .unwrap();

        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let results = topo.batch_check(&[("app:backend", "db/password")]);

        if let Some(r) = results.first() {
            if r.entity == "app:backend" {
                assert!(r.has_read);
                assert!(r.has_write);
                assert!(!r.has_admin);
            }
        }
    }

    #[test]
    fn test_topology_batch_check_denied() {
        let (vault, _graph) = make_vault();

        vault.set("node:root", "db/password", "secret").unwrap();

        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let results = topo.batch_check(&[("unknown:user", "db/password")]);

        assert_eq!(results.len(), 1);
        assert!(!results[0].has_read);
        assert!(!results[0].has_write);
        assert!(!results[0].has_admin);
    }

    #[test]
    fn test_topology_tt_compression_enabled() {
        // With a tiny matrix, TT compression won't trigger due to threshold
        let (vault, _graph) = make_vault();
        vault.set("node:root", "s1", "v1").unwrap();

        let config = TopologyConfig {
            enable_tt_compression: true,
            compression_threshold: 1, // Force compression even for small
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let (orig, _comp, _ratio) = topo.compression_stats();
        // For very small matrices, TT may fail to factorize
        assert!(orig > 0 || topo.num_secrets() == 0);
    }

    #[test]
    fn test_topology_tt_compression_disabled() {
        let (vault, _graph) = make_vault();
        vault.set("node:root", "s1", "v1").unwrap();

        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let (orig, comp, ratio) = topo.compression_stats();
        assert_eq!(orig, comp);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_topology_tt_below_threshold() {
        let (vault, _graph) = make_vault();
        vault.set("node:root", "s1", "v1").unwrap();

        let config = TopologyConfig {
            enable_tt_compression: true,
            compression_threshold: 1_000_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let (orig, comp, _) = topo.compression_stats();
        // Below threshold, no compression applied
        assert_eq!(orig, comp);
    }

    #[test]
    fn test_policy_redundancy_similar() {
        let (vault, _graph) = make_vault();

        // Add similar policies
        vault
            .add_policy(
                "node:root",
                crate::PolicyTemplate {
                    name: "dev-read-1".to_string(),
                    match_pattern: "team:engineering/*".to_string(),
                    secret_pattern: "staging/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();
        vault
            .add_policy(
                "node:root",
                crate::PolicyTemplate {
                    name: "dev-read-2".to_string(),
                    match_pattern: "team:engineering/*".to_string(),
                    secret_pattern: "staging/db/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let report = analyze_policy_redundancy(&vault).unwrap();
        assert_eq!(report.total_policies, 2);
        // The two policies have very similar match patterns
        assert!(!report.mergeable_groups.is_empty());
    }

    #[test]
    fn test_policy_redundancy_distinct() {
        let (vault, _graph) = make_vault();

        vault
            .add_policy(
                "node:root",
                crate::PolicyTemplate {
                    name: "dev-access".to_string(),
                    match_pattern: "team:engineering/*".to_string(),
                    secret_pattern: "staging/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();
        vault
            .add_policy(
                "node:root",
                crate::PolicyTemplate {
                    name: "ops-access".to_string(),
                    match_pattern: "svc:monitoring".to_string(),
                    secret_pattern: "prod/metrics/*".to_string(),
                    permission: Permission::Write,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let report = analyze_policy_redundancy(&vault).unwrap();
        assert_eq!(report.total_policies, 2);
        // Very different patterns should not be mergeable
        assert!(report.mergeable_groups.is_empty());
    }

    #[test]
    fn test_factorize_for_tt_basic() {
        // 12 = 2 * 2 * 3 (should find something)
        let f = factorize_for_tt(12);
        if let Some(factors) = f {
            let product: usize = factors.iter().product();
            assert_eq!(product, 12);
        }

        // Prime number: no factorization
        assert!(factorize_for_tt(7).is_none());

        // Zero
        assert!(factorize_for_tt(0).is_none());
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let sim = jaccard_similarity("team:engineering/*", "team:engineering/*");
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        let sim = jaccard_similarity("abc", "xyz");
        assert!(sim < 0.1);
    }

    #[test]
    fn test_jaccard_similarity_empty_strings() {
        // Both empty: should return 1.0 (identical)
        let sim = jaccard_similarity("", "");
        assert!((sim - 1.0).abs() < f32::EPSILON);

        // Single char strings have no bigrams, treated as empty
        let sim2 = jaccard_similarity("a", "b");
        assert!((sim2 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_topology_config_default() {
        let config = TopologyConfig::default();
        assert!(config.enable_tt_compression);
        assert_eq!(config.compression_threshold, 10_000);
    }

    #[test]
    fn test_topology_empty_vault() {
        let (vault, _graph) = make_vault();
        // Don't set any secrets -- empty vault
        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        assert_eq!(topo.num_entities(), 0);
        assert_eq!(topo.num_secrets(), 0);
        let (orig, comp, ratio) = topo.compression_stats();
        assert_eq!(orig, 0);
        assert_eq!(comp, 0);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_topology_admin_permission() {
        let (vault, _graph) = make_vault();
        vault.set("node:root", "admin/key", "val").unwrap();
        vault
            .grant_with_permission("node:root", "user:admin", "admin/key", Permission::Admin)
            .unwrap();

        let config = TopologyConfig {
            enable_tt_compression: false,
            compression_threshold: 10_000,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let results = topo.batch_check(&[("user:admin", "admin/key")]);
        if let Some(r) = results.first() {
            if r.entity == "user:admin" {
                assert!(r.has_admin);
                assert!(r.has_read);
                assert!(r.has_write);
            }
        }
    }

    #[test]
    fn test_policy_redundancy_single_policy() {
        let (vault, _graph) = make_vault();
        vault
            .add_policy(
                "node:root",
                crate::PolicyTemplate {
                    name: "only-one".to_string(),
                    match_pattern: "team:*".to_string(),
                    secret_pattern: "dev/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let report = analyze_policy_redundancy(&vault).unwrap();
        assert_eq!(report.total_policies, 1);
        assert!(report.mergeable_groups.is_empty());
        assert!((report.compression_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_policy_redundancy_zero_policies() {
        let (vault, _graph) = make_vault();
        let report = analyze_policy_redundancy(&vault).unwrap();
        assert_eq!(report.total_policies, 0);
        assert!(report.mergeable_groups.is_empty());
    }

    #[test]
    fn test_factorize_for_tt_three_factors() {
        // 24 = 2 * 3 * 4 or similar 3-factor decomposition
        let f = factorize_for_tt(24);
        assert!(f.is_some());
        let factors = f.unwrap();
        let product: usize = factors.iter().product();
        assert_eq!(product, 24);
        assert!(factors.len() >= 2);
    }

    #[test]
    fn test_factorize_for_tt_four_factors() {
        // 36 = 2 * 3 * 2 * 3 (should find 4-factor decomposition)
        let f = factorize_for_tt(36);
        assert!(f.is_some());
        let factors = f.unwrap();
        let product: usize = factors.iter().product();
        assert_eq!(product, 36);
    }

    #[test]
    fn test_factorize_for_tt_large_composite() {
        // 360 = 2^3 * 3^2 * 5 -- should decompose into 3-4 factors
        let f = factorize_for_tt(360);
        assert!(f.is_some());
        let factors = f.unwrap();
        let product: usize = factors.iter().product();
        assert_eq!(product, 360);
    }

    #[test]
    fn test_try_tt_compress_small() {
        // Very small matrix: should return None (< 4 elements)
        let data = vec![1.0, 2.0, 3.0];
        assert!(try_tt_compress(&data).is_none());

        // Small prime-sized: no factorization
        let data = vec![0.0; 7];
        assert!(try_tt_compress(&data).is_none());
    }

    #[test]
    fn test_try_tt_compress_factorizable() {
        // 12 elements -- should be factorizable (2*2*3 or 3*4)
        let data = vec![1.0_f32; 12];
        // Result depends on whether TT decomposition succeeds
        let _ = try_tt_compress(&data);
    }

    #[test]
    fn test_topology_with_tt_compression_triggered() {
        let (vault, _graph) = make_vault();

        // Create enough entities and secrets to trigger TT compression
        // threshold=1 forces it
        for i in 0..4 {
            let key = format!("tt/secret{i}");
            vault.set("node:root", &key, &format!("val{i}")).unwrap();
        }
        for i in 0..4 {
            let entity = format!("user:tt{i}");
            vault
                .grant_with_permission("node:root", &entity, "tt/secret0", Permission::Read)
                .unwrap();
        }

        let config = TopologyConfig {
            enable_tt_compression: true,
            compression_threshold: 1,
        };
        let topo = AccessTopology::from_vault(&vault, config).unwrap();
        let (orig, comp, ratio) = topo.compression_stats();
        // If TT compression succeeded, compressed < original
        // If not (factorization failed), they're equal
        assert!(orig > 0);
        assert!(comp > 0);
        assert!(ratio > 0.0);
    }
}
