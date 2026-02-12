// SPDX-License-Identifier: MIT OR Apache-2.0
//! Manifold-aware placement: geographic coordinates and latency-aware secret placement.

use std::collections::HashMap;

use dashmap::DashMap;
use graph_engine::{Direction, PropertyValue};
use serde::{Deserialize, Serialize};

use crate::vault::Vault;
use crate::{Result, VaultError};

/// A point in 2D or 3D coordinate space representing a geographic location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoCoordinate {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
}

/// A vault deployment region with capacity and inter-region latency metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultRegion {
    pub name: String,
    pub center: GeoCoordinate,
    pub capacity: usize,
    pub current_load: usize,
    /// Inter-region latencies in milliseconds, keyed by region name.
    pub latencies: HashMap<String, f64>,
}

/// Recommended placement for a single secret across available regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementRecommendation {
    pub secret_key: String,
    pub primary_region: String,
    pub replica_regions: Vec<String>,
    pub placement_score: f64,
    pub access_centroid: GeoCoordinate,
}

/// Tuning weights that control placement scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementConfig {
    pub locality_weight: f64,
    pub load_balance_weight: f64,
    pub replication_weight: f64,
    pub replica_count: usize,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            locality_weight: 0.7,
            load_balance_weight: 0.2,
            replication_weight: 0.1,
            replica_count: 2,
        }
    }
}

/// Thread-safe registry of deployment regions and entity geographic locations.
pub struct RegionRegistry {
    regions: DashMap<String, VaultRegion>,
    entity_locations: DashMap<String, GeoCoordinate>,
}

impl Default for RegionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionRegistry {
    pub fn new() -> Self {
        Self {
            regions: DashMap::new(),
            entity_locations: DashMap::new(),
        }
    }

    pub fn add_region(&self, region: VaultRegion) {
        self.regions.insert(region.name.clone(), region);
    }

    pub fn set_entity_location(&self, entity: &str, coord: GeoCoordinate) {
        self.entity_locations.insert(entity.to_string(), coord);
    }

    pub fn regions(&self) -> Vec<VaultRegion> {
        self.regions.iter().map(|r| r.value().clone()).collect()
    }

    pub fn entity_location(&self, entity: &str) -> Option<GeoCoordinate> {
        self.entity_locations.get(entity).map(|r| r.value().clone())
    }
}

/// Recommend the best region placement for a single secret.
///
/// Scores each known region by proximity to the secret's accessor centroid,
/// current load relative to capacity, and average replica latency. The region
/// with the lowest composite score becomes the primary; the next-lowest regions
/// (up to `config.replica_count`) become replicas.
///
/// # Errors
///
/// Returns `VaultError::NotFound` when the registry contains no regions or when
/// no placement centroid can be computed.
pub fn recommend_placement(
    vault: &Vault,
    registry: &RegionRegistry,
    secret_key: &str,
    config: &PlacementConfig,
) -> Result<PlacementRecommendation> {
    let regions = registry.regions();
    if regions.is_empty() {
        return Err(VaultError::NotFound("no regions registered".to_string()));
    }

    let centroid = compute_access_centroid(vault, registry, secret_key).unwrap_or(GeoCoordinate {
        x: 0.0,
        y: 0.0,
        z: None,
    });

    let mut scored: Vec<(String, f64)> = regions
        .iter()
        .map(|region| {
            let score = score_region(region, &centroid, &regions, config);
            (region.name.clone(), score)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let primary = scored[0].0.clone();
    let primary_score = scored[0].1;

    let replicas: Vec<String> = scored
        .iter()
        .skip(1)
        .take(config.replica_count)
        .map(|(name, _)| name.clone())
        .collect();

    Ok(PlacementRecommendation {
        secret_key: secret_key.to_string(),
        primary_region: primary,
        replica_regions: replicas,
        placement_score: primary_score,
        access_centroid: centroid,
    })
}

/// Recommend placement for every secret the vault root can list.
pub fn batch_recommend_placement(
    vault: &Vault,
    registry: &RegionRegistry,
    config: &PlacementConfig,
) -> Vec<PlacementRecommendation> {
    let secret_keys = vault.list(Vault::ROOT, "*").unwrap_or_default();

    secret_keys
        .iter()
        .filter_map(|key| recommend_placement(vault, registry, key, config).ok())
        .collect()
}

/// Compute the weighted geographic centroid of all entities that have
/// `VAULT_ACCESS*` edges in the graph and whose locations are registered.
#[allow(clippy::cast_precision_loss)] // entity count will never exceed 2^52
fn compute_access_centroid(
    vault: &Vault,
    registry: &RegionRegistry,
    _secret_key: &str,
) -> Option<GeoCoordinate> {
    let mut locations = Vec::new();

    if let Ok(node_ids) = vault.graph.get_all_node_ids() {
        for nid in &node_ids {
            if let Ok(node) = vault.graph.get_node(*nid) {
                if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
                    // Skip secret nodes -- they are not accessor entities
                    if key.starts_with("vault_secret:") || key.starts_with("_vk:") {
                        continue;
                    }
                    // Check outgoing edges for VAULT_ACCESS* types
                    if let Ok(edges) = vault.graph.edges_of(*nid, Direction::Outgoing) {
                        let has_access = edges
                            .iter()
                            .any(|edge| edge.edge_type.starts_with("VAULT_ACCESS"));
                        if has_access {
                            if let Some(coord) = registry.entity_location(key) {
                                locations.push(coord);
                            }
                        }
                    }
                }
            }
        }
    }

    if locations.is_empty() {
        return None;
    }

    let n = locations.len() as f64;
    let x = locations.iter().map(|c| c.x).sum::<f64>() / n;
    let y = locations.iter().map(|c| c.y).sum::<f64>() / n;
    let z_count = locations.iter().filter(|c| c.z.is_some()).count();
    let z = if z_count > 0 {
        let z_sum: f64 = locations.iter().filter_map(|c| c.z).sum();
        Some(z_sum / z_count as f64)
    } else {
        None
    };

    Some(GeoCoordinate { x, y, z })
}

/// Euclidean distance between two coordinates (including z when both have it).
fn geodesic_distance(a: &GeoCoordinate, b: &GeoCoordinate) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = match (a.z, b.z) {
        (Some(az), Some(bz)) => az - bz,
        _ => 0.0,
    };
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt()
}

/// Score a region for placement: lower is better.
#[allow(clippy::cast_precision_loss)] // region counts and capacities will never exceed 2^52
fn score_region(
    region: &VaultRegion,
    centroid: &GeoCoordinate,
    all_regions: &[VaultRegion],
    config: &PlacementConfig,
) -> f64 {
    // Locality: distance from centroid to region center, normalized by max distance
    let distance = geodesic_distance(centroid, &region.center);
    let max_distance = all_regions
        .iter()
        .map(|r| geodesic_distance(centroid, &r.center))
        .fold(f64::MIN, f64::max);
    let normalized_distance = if max_distance > 0.0 {
        distance / max_distance
    } else {
        0.0
    };

    // Load balance: ratio of current load to capacity
    let load_ratio = if region.capacity > 0 {
        region.current_load as f64 / region.capacity as f64
    } else {
        1.0
    };

    // Replication: average latency from this region to all others
    let avg_latency = if region.latencies.is_empty() {
        0.0
    } else {
        let total: f64 = region.latencies.values().sum();
        total / region.latencies.len() as f64
    };
    // Normalize latency against max observed average latency across all regions
    let max_avg_latency = all_regions
        .iter()
        .map(|r| {
            if r.latencies.is_empty() {
                0.0
            } else {
                let t: f64 = r.latencies.values().sum();
                t / r.latencies.len() as f64
            }
        })
        .fold(f64::MIN, f64::max);
    let normalized_latency = if max_avg_latency > 0.0 {
        avg_latency / max_avg_latency
    } else {
        0.0
    };

    config.locality_weight.mul_add(
        normalized_distance,
        config
            .load_balance_weight
            .mul_add(load_ratio, config.replication_weight * normalized_latency),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::VaultConfig;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    #[test]
    fn test_geo_distance_zero() {
        let a = GeoCoordinate {
            x: 5.0,
            y: 3.0,
            z: Some(1.0),
        };
        let d = geodesic_distance(&a, &a.clone());
        assert!((d - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_distance_known() {
        let a = GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        };
        let b = GeoCoordinate {
            x: 3.0,
            y: 4.0,
            z: None,
        };
        let d = geodesic_distance(&a, &b);
        assert!((d - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_distance_3d() {
        let a = GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: Some(0.0),
        };
        let b = GeoCoordinate {
            x: 1.0,
            y: 2.0,
            z: Some(2.0),
        };
        let d = geodesic_distance(&a, &b);
        assert!((d - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_centroid_single() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();
        registry.set_entity_location(
            "alice",
            GeoCoordinate {
                x: 7.0,
                y: 11.0,
                z: None,
            },
        );

        // Grant alice access so she appears as an accessor
        vault.set(Vault::ROOT, "my_secret", "value").unwrap();
        vault.grant(Vault::ROOT, "alice", "my_secret").unwrap();

        let centroid = compute_access_centroid(&vault, &registry, "my_secret");
        assert!(centroid.is_some());
        let c = centroid.unwrap();
        assert!((c.x - 7.0).abs() < f64::EPSILON);
        assert!((c.y - 11.0).abs() < f64::EPSILON);
        assert!(c.z.is_none());
    }

    #[test]
    fn test_centroid_multiple() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();

        vault.set(Vault::ROOT, "shared_secret", "data").unwrap();
        vault.grant(Vault::ROOT, "alice", "shared_secret").unwrap();
        vault.grant(Vault::ROOT, "bob", "shared_secret").unwrap();

        registry.set_entity_location(
            "alice",
            GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
        );
        registry.set_entity_location(
            "bob",
            GeoCoordinate {
                x: 2.0,
                y: 2.0,
                z: None,
            },
        );

        let centroid = compute_access_centroid(&vault, &registry, "shared_secret");
        assert!(centroid.is_some());
        let c = centroid.unwrap();
        assert!((c.x - 1.0).abs() < f64::EPSILON);
        assert!((c.y - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_placement_closer_region() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();

        vault.set(Vault::ROOT, "sec1", "val").unwrap();
        vault.grant(Vault::ROOT, "alice", "sec1").unwrap();

        registry.set_entity_location(
            "alice",
            GeoCoordinate {
                x: 1.0,
                y: 1.0,
                z: None,
            },
        );

        let close = VaultRegion {
            name: "close".to_string(),
            center: GeoCoordinate {
                x: 1.0,
                y: 1.0,
                z: None,
            },
            capacity: 100,
            current_load: 0,
            latencies: HashMap::new(),
        };
        let far = VaultRegion {
            name: "far".to_string(),
            center: GeoCoordinate {
                x: 100.0,
                y: 100.0,
                z: None,
            },
            capacity: 100,
            current_load: 0,
            latencies: HashMap::new(),
        };

        registry.add_region(close);
        registry.add_region(far);

        let config = PlacementConfig::default();
        let rec = recommend_placement(&vault, &registry, "sec1", &config).unwrap();
        assert_eq!(rec.primary_region, "close");
    }

    #[test]
    fn test_placement_load_balance() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();

        // No secret accessors -- centroid falls to origin so distance is equal
        // for two equidistant regions. The load difference should decide.
        let loaded = VaultRegion {
            name: "loaded".to_string(),
            center: GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            capacity: 100,
            current_load: 90,
            latencies: HashMap::new(),
        };
        let light = VaultRegion {
            name: "light".to_string(),
            center: GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            capacity: 100,
            current_load: 10,
            latencies: HashMap::new(),
        };

        registry.add_region(loaded);
        registry.add_region(light);

        let config = PlacementConfig {
            locality_weight: 0.0,
            load_balance_weight: 1.0,
            replication_weight: 0.0,
            replica_count: 1,
        };

        let rec = recommend_placement(&vault, &registry, "any_key", &config).unwrap();
        assert_eq!(rec.primary_region, "light");
    }

    #[test]
    fn test_placement_replicas() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();

        for i in 0..4 {
            let region = VaultRegion {
                name: format!("region_{i}"),
                center: GeoCoordinate {
                    x: f64::from(i) * 10.0,
                    y: 0.0,
                    z: None,
                },
                capacity: 100,
                current_load: 0,
                latencies: HashMap::new(),
            };
            registry.add_region(region);
        }

        let config = PlacementConfig {
            replica_count: 2,
            ..PlacementConfig::default()
        };
        let rec = recommend_placement(&vault, &registry, "k", &config).unwrap();
        assert_eq!(rec.replica_regions.len(), 2);
        // Replicas must not include the primary
        assert!(!rec.replica_regions.contains(&rec.primary_region));
    }

    #[test]
    fn test_region_registry() {
        let registry = RegionRegistry::new();
        assert!(registry.regions().is_empty());
        assert!(registry.entity_location("nobody").is_none());

        let r = VaultRegion {
            name: "us-east".to_string(),
            center: GeoCoordinate {
                x: -74.0,
                y: 40.7,
                z: None,
            },
            capacity: 500,
            current_load: 100,
            latencies: {
                let mut m = HashMap::new();
                m.insert("us-west".to_string(), 70.0);
                m
            },
        };
        registry.add_region(r);
        assert_eq!(registry.regions().len(), 1);
        assert_eq!(registry.regions()[0].name, "us-east");

        registry.set_entity_location(
            "alice",
            GeoCoordinate {
                x: 1.0,
                y: 2.0,
                z: Some(3.0),
            },
        );
        let loc = registry.entity_location("alice").unwrap();
        assert!((loc.x - 1.0).abs() < f64::EPSILON);
        assert!((loc.y - 2.0).abs() < f64::EPSILON);
        assert!((loc.z.unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_placement_config_default() {
        let config = PlacementConfig::default();
        assert!((config.locality_weight - 0.7).abs() < f64::EPSILON);
        assert!((config.load_balance_weight - 0.2).abs() < f64::EPSILON);
        assert!((config.replication_weight - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.replica_count, 2);
    }

    #[test]
    fn test_batch_recommend_placement() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();

        vault.set(Vault::ROOT, "s1", "a").unwrap();
        vault.set(Vault::ROOT, "s2", "b").unwrap();

        let region = VaultRegion {
            name: "default".to_string(),
            center: GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            capacity: 100,
            current_load: 0,
            latencies: HashMap::new(),
        };
        registry.add_region(region);

        let recs = batch_recommend_placement(&vault, &registry, &PlacementConfig::default());
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn test_recommend_no_regions() {
        let vault = create_test_vault();
        let registry = RegionRegistry::new();
        let config = PlacementConfig::default();
        let result = recommend_placement(&vault, &registry, "key", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_default() {
        let registry = RegionRegistry::default();
        assert!(registry.regions().is_empty());
        assert!(registry.entity_location("x").is_none());
    }
}
