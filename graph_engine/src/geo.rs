//! Geospatial index for Point properties.
//!
//! Provides spatial queries like radius search and bounding box search
//! using a grid-based spatial index.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::trivially_copy_pass_by_ref)]

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use crate::{GraphEngine, IndexTarget, Node, PropertyValue, Result};

/// A geospatial point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    #[must_use]
    pub const fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Calculate distance to another point in kilometers using Haversine formula.
    #[must_use]
    pub fn distance_km(&self, other: &Self) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other.lat.to_radians();
        let delta_lat = (other.lat - self.lat).to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let a = (lat1_rad.cos() * lat2_rad.cos()).mul_add(
            (delta_lon / 2.0).sin().powi(2),
            (delta_lat / 2.0).sin().powi(2),
        );
        let c = 2.0 * a.sqrt().asin();

        EARTH_RADIUS_KM * c
    }
}

impl From<(f64, f64)> for GeoPoint {
    fn from((lat, lon): (f64, f64)) -> Self {
        Self::new(lat, lon)
    }
}

/// Configuration for geospatial index.
#[derive(Debug, Clone)]
pub struct GeoConfig {
    /// Grid cell size in degrees (smaller = more precision, more memory).
    pub cell_size: f64,
}

impl Default for GeoConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.1, // ~11km at equator
        }
    }
}

impl GeoConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn cell_size(mut self, size: f64) -> Self {
        self.cell_size = size;
        self
    }
}

/// Grid cell coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    lat_cell: i32,
    lon_cell: i32,
}

/// Geospatial index using a grid-based structure.
pub struct GeoIndex {
    /// Property being indexed.
    property: String,
    /// Target (Node or Edge).
    target: IndexTarget,
    /// Grid: cell -> set of entity IDs.
    grid: RwLock<HashMap<GridCell, HashSet<u64>>>,
    /// Entity locations: entity ID -> point.
    locations: RwLock<HashMap<u64, GeoPoint>>,
    /// Configuration.
    config: GeoConfig,
}

impl GeoIndex {
    pub fn new(property: impl Into<String>, target: IndexTarget, config: GeoConfig) -> Self {
        Self {
            property: property.into(),
            target,
            grid: RwLock::new(HashMap::new()),
            locations: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Calculate grid cell for a point.
    fn point_to_cell(&self, point: &GeoPoint) -> GridCell {
        GridCell {
            lat_cell: (point.lat / self.config.cell_size).floor() as i32,
            lon_cell: (point.lon / self.config.cell_size).floor() as i32,
        }
    }

    /// Index an entity at a location.
    pub fn index_entity(&self, id: u64, point: GeoPoint) {
        let cell = self.point_to_cell(&point);

        let mut grid = self.grid.write();
        let mut locations = self.locations.write();

        // Remove from old cell if entity was previously indexed
        if let Some(old_point) = locations.remove(&id) {
            let old_cell = self.point_to_cell(&old_point);
            if let Some(ids) = grid.get_mut(&old_cell) {
                ids.remove(&id);
                if ids.is_empty() {
                    grid.remove(&old_cell);
                }
            }
        }

        // Add to new cell
        grid.entry(cell).or_default().insert(id);
        locations.insert(id, point);
    }

    /// Remove an entity from the index.
    pub fn remove_entity(&self, id: u64) {
        let mut grid = self.grid.write();
        let mut locations = self.locations.write();

        if let Some(point) = locations.remove(&id) {
            let cell = self.point_to_cell(&point);
            if let Some(ids) = grid.get_mut(&cell) {
                ids.remove(&id);
                if ids.is_empty() {
                    grid.remove(&cell);
                }
            }
        }
    }

    /// Find entities within a radius of a center point.
    pub fn find_within_radius(&self, center: GeoPoint, radius_km: f64) -> Vec<(u64, f64)> {
        // Calculate cell range to search
        // Approximate: 1 degree latitude = 111 km
        let lat_range = radius_km / 111.0;
        // Approximate: 1 degree longitude = 111 * cos(lat) km
        let lon_range = radius_km / (111.0 * center.lat.to_radians().cos().abs().max(0.01));

        let min_lat_cell = ((center.lat - lat_range) / self.config.cell_size).floor() as i32;
        let max_lat_cell = ((center.lat + lat_range) / self.config.cell_size).ceil() as i32;
        let min_lon_cell = ((center.lon - lon_range) / self.config.cell_size).floor() as i32;
        let max_lon_cell = ((center.lon + lon_range) / self.config.cell_size).ceil() as i32;

        let grid = self.grid.read();
        let locations = self.locations.read();

        let mut results = Vec::new();

        for lat_cell in min_lat_cell..=max_lat_cell {
            for lon_cell in min_lon_cell..=max_lon_cell {
                let cell = GridCell { lat_cell, lon_cell };
                if let Some(ids) = grid.get(&cell) {
                    for &id in ids {
                        if let Some(point) = locations.get(&id) {
                            let distance = center.distance_km(point);
                            if distance <= radius_km {
                                results.push((id, distance));
                            }
                        }
                    }
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find entities within a bounding box.
    pub fn find_in_bbox(&self, min: GeoPoint, max: GeoPoint) -> Vec<u64> {
        let min_lat_cell = (min.lat / self.config.cell_size).floor() as i32;
        let max_lat_cell = (max.lat / self.config.cell_size).ceil() as i32;
        let min_lon_cell = (min.lon / self.config.cell_size).floor() as i32;
        let max_lon_cell = (max.lon / self.config.cell_size).ceil() as i32;

        let grid = self.grid.read();
        let locations = self.locations.read();

        let mut results = Vec::new();

        for lat_cell in min_lat_cell..=max_lat_cell {
            for lon_cell in min_lon_cell..=max_lon_cell {
                let cell = GridCell { lat_cell, lon_cell };
                if let Some(ids) = grid.get(&cell) {
                    for &id in ids {
                        if let Some(point) = locations.get(&id) {
                            if point.lat >= min.lat
                                && point.lat <= max.lat
                                && point.lon >= min.lon
                                && point.lon <= max.lon
                            {
                                results.push(id);
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Find the k nearest neighbors to a point.
    pub fn find_nearest(&self, center: GeoPoint, k: usize) -> Vec<(u64, f64)> {
        let locations = self.locations.read();

        let mut results: Vec<(u64, f64)> = locations
            .iter()
            .map(|(&id, point)| (id, center.distance_km(point)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    #[must_use]
    pub fn property(&self) -> &str {
        &self.property
    }

    #[must_use]
    pub const fn target(&self) -> IndexTarget {
        self.target
    }
}

impl GraphEngine {
    /// Create a geospatial index on a node property containing Point values.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails.
    pub fn create_geo_index(&self, property: &str) -> Result<()> {
        self.create_geo_index_with_config(property, GeoConfig::default())
    }

    /// Create a geospatial index with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails.
    pub fn create_geo_index_with_config(&self, property: &str, config: GeoConfig) -> Result<()> {
        let index = GeoIndex::new(property, IndexTarget::Node, config);

        // Index all existing nodes
        let nodes = self.get_all_node_ids()?;
        for node_id in nodes {
            if let Ok(node) = self.get_node(node_id) {
                if let Some(PropertyValue::Point { lat, lon }) = node.properties.get(property) {
                    index.index_entity(node_id, GeoPoint::new(*lat, *lon));
                }
            }
        }

        self.geo_indexes.write().insert(property.to_string(), index);
        Ok(())
    }

    /// Find nodes within a radius of a center point.
    ///
    /// Returns nodes sorted by distance, with distance in kilometers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn find_within_radius(
        &self,
        property: &str,
        center: GeoPoint,
        radius_km: f64,
    ) -> Result<Vec<(Node, f64)>> {
        let indexes = self.geo_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "geo".to_string(),
                property: property.to_string(),
            })?;

        let results = index.find_within_radius(center, radius_km);
        results
            .into_iter()
            .map(|(id, dist)| self.get_node(id).map(|n| (n, dist)))
            .collect()
    }

    /// Find nodes within a bounding box.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn find_in_bbox(&self, property: &str, min: GeoPoint, max: GeoPoint) -> Result<Vec<Node>> {
        let indexes = self.geo_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "geo".to_string(),
                property: property.to_string(),
            })?;

        let ids = index.find_in_bbox(min, max);
        ids.into_iter().map(|id| self.get_node(id)).collect()
    }

    /// Find the k nearest nodes to a point.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn find_nearest(
        &self,
        property: &str,
        center: GeoPoint,
        k: usize,
    ) -> Result<Vec<(Node, f64)>> {
        let indexes = self.geo_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "geo".to_string(),
                property: property.to_string(),
            })?;

        let results = index.find_nearest(center, k);
        results
            .into_iter()
            .map(|(id, dist)| self.get_node(id).map(|n| (n, dist)))
            .collect()
    }

    /// Check if a geospatial index exists.
    #[must_use]
    pub fn has_geo_index(&self, property: &str) -> bool {
        self.geo_indexes.read().contains_key(property)
    }

    /// Drop a geospatial index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn drop_geo_index(&self, property: &str) -> Result<()> {
        self.geo_indexes
            .write()
            .remove(property)
            .map(|_| ())
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "geo".to_string(),
                property: property.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_point_distance() {
        // New York to Los Angeles (approximate)
        let nyc = GeoPoint::new(40.7128, -74.0060);
        let la = GeoPoint::new(34.0522, -118.2437);

        let distance = nyc.distance_km(&la);
        // Should be approximately 3944 km
        assert!((distance - 3944.0).abs() < 50.0);
    }

    #[test]
    fn test_geo_point_same_location() {
        let point = GeoPoint::new(51.5074, -0.1278);
        assert!(point.distance_km(&point) < 0.001);
    }

    #[test]
    fn test_geo_index_radius_search() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());

        // Add some points around London
        index.index_entity(1, GeoPoint::new(51.5074, -0.1278)); // London
        index.index_entity(2, GeoPoint::new(51.4545, -0.9781)); // Reading (~60km)
        index.index_entity(3, GeoPoint::new(52.4862, -1.8904)); // Birmingham (~160km)
        index.index_entity(4, GeoPoint::new(48.8566, 2.3522)); // Paris (~340km)

        // Search within 100km of London
        let results = index.find_within_radius(GeoPoint::new(51.5074, -0.1278), 100.0);
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

        assert!(ids.contains(&1)); // London
        assert!(ids.contains(&2)); // Reading
        assert!(!ids.contains(&3)); // Birmingham too far
        assert!(!ids.contains(&4)); // Paris too far
    }

    #[test]
    fn test_geo_index_bbox_search() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());

        index.index_entity(1, GeoPoint::new(51.5, -0.1));
        index.index_entity(2, GeoPoint::new(51.6, -0.2));
        index.index_entity(3, GeoPoint::new(52.0, 0.0));

        // Bounding box that includes points 1 and 2
        let results = index.find_in_bbox(GeoPoint::new(51.4, -0.3), GeoPoint::new(51.7, 0.0));

        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_geo_index_nearest() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());

        index.index_entity(1, GeoPoint::new(51.5, -0.1));
        index.index_entity(2, GeoPoint::new(51.6, -0.2));
        index.index_entity(3, GeoPoint::new(52.0, 0.0));

        let results = index.find_nearest(GeoPoint::new(51.5, -0.1), 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Closest point
    }

    #[test]
    fn test_geo_index_remove() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());

        index.index_entity(1, GeoPoint::new(51.5, -0.1));
        index.index_entity(2, GeoPoint::new(51.5, -0.1));

        let results = index.find_within_radius(GeoPoint::new(51.5, -0.1), 1.0);
        assert_eq!(results.len(), 2);

        index.remove_entity(1);

        let results = index.find_within_radius(GeoPoint::new(51.5, -0.1), 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_geo_index_update() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());

        index.index_entity(1, GeoPoint::new(51.5, -0.1));

        // Should find in London area
        let results = index.find_within_radius(GeoPoint::new(51.5, -0.1), 10.0);
        assert!(!results.is_empty());

        // Update to Paris
        index.index_entity(1, GeoPoint::new(48.8566, 2.3522));

        // Should not find in London area anymore
        let results = index.find_within_radius(GeoPoint::new(51.5, -0.1), 10.0);
        assert!(results.is_empty());

        // Should find in Paris area
        let results = index.find_within_radius(GeoPoint::new(48.8566, 2.3522), 10.0);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_graph_engine_geo_index() {
        let engine = GraphEngine::new();

        let n1 = engine
            .create_node(
                "Place",
                [(
                    "location".to_string(),
                    PropertyValue::Point {
                        lat: 51.5074,
                        lon: -0.1278,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let n2 = engine
            .create_node(
                "Place",
                [(
                    "location".to_string(),
                    PropertyValue::Point {
                        lat: 51.4545,
                        lon: -0.9781,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let _n3 = engine
            .create_node(
                "Place",
                [(
                    "location".to_string(),
                    PropertyValue::Point {
                        lat: 48.8566,
                        lon: 2.3522,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_geo_index("location").unwrap();

        // Search within 100km of London
        let results = engine
            .find_within_radius("location", GeoPoint::new(51.5074, -0.1278), 100.0)
            .unwrap();

        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|(n, _)| n.id).collect();
        assert!(ids.contains(&n1));
        assert!(ids.contains(&n2));
    }

    #[test]
    fn test_geo_config_builder() {
        let config = GeoConfig::new().cell_size(0.5);
        assert!((config.cell_size - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_index_property_and_target() {
        let index = GeoIndex::new("coords", IndexTarget::Node, GeoConfig::default());
        assert_eq!(index.property(), "coords");
        assert_eq!(index.target(), IndexTarget::Node);
    }

    #[test]
    fn test_geo_index_remove_nonexistent() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());
        index.remove_entity(999); // Should not panic
    }

    #[test]
    fn test_geo_index_empty_search() {
        let index = GeoIndex::new("location", IndexTarget::Node, GeoConfig::default());
        let results = index.find_within_radius(GeoPoint::new(0.0, 0.0), 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_graph_engine_find_in_bbox() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 51.5,
                        lon: -0.1,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 52.0,
                        lon: 0.0,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_geo_index("loc").unwrap();

        let results = engine
            .find_in_bbox("loc", GeoPoint::new(51.0, -0.5), GeoPoint::new(51.8, 0.5))
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_graph_engine_find_nearest() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 51.5,
                        lon: -0.1,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 51.6,
                        lon: -0.2,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 52.0,
                        lon: 0.0,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_geo_index("loc").unwrap();

        let results = engine
            .find_nearest("loc", GeoPoint::new(51.5, -0.1), 2)
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_graph_engine_has_and_drop_geo_index() {
        let engine = GraphEngine::new();

        assert!(!engine.has_geo_index("location"));
        engine.create_geo_index("location").unwrap();
        assert!(engine.has_geo_index("location"));
        engine.drop_geo_index("location").unwrap();
        assert!(!engine.has_geo_index("location"));
    }

    #[test]
    fn test_graph_engine_geo_index_not_found() {
        let engine = GraphEngine::new();

        let result = engine.find_within_radius("missing", GeoPoint::new(0.0, 0.0), 100.0);
        assert!(result.is_err());

        let result =
            engine.find_in_bbox("missing", GeoPoint::new(0.0, 0.0), GeoPoint::new(1.0, 1.0));
        assert!(result.is_err());

        let result = engine.find_nearest("missing", GeoPoint::new(0.0, 0.0), 10);
        assert!(result.is_err());

        let result = engine.drop_geo_index("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_engine_geo_with_config() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Place",
                [(
                    "loc".to_string(),
                    PropertyValue::Point {
                        lat: 51.5,
                        lon: -0.1,
                    },
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let config = GeoConfig::new().cell_size(0.5);
        engine.create_geo_index_with_config("loc", config).unwrap();

        let results = engine
            .find_within_radius("loc", GeoPoint::new(51.5, -0.1), 10.0)
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_geo_point_from_tuple() {
        let point: GeoPoint = (40.7, -74.0).into();
        assert!((point.lat - 40.7).abs() < f64::EPSILON);
        assert!((point.lon - -74.0).abs() < f64::EPSILON);
    }
}
