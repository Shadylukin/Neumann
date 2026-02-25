//! Generic N-dimensional spatial index backed by an R-tree.

use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::bbox::{BoundingBoxN, SpatialEntryN};
use crate::iter::SpatialIterN;
use crate::node::{str_build_nodes, InsertResult, NodeN};
use crate::{SpatialConfig, SpatialError, SplitStrategy};

/// A spatial index backed by an R-tree for efficient region and
/// nearest-neighbor queries in `D`-dimensional space.
pub struct SpatialIndexN<const D: usize, T> {
    /// The root node of the R-tree.
    pub(crate) root: NodeN<D, T>,
    /// Number of entries stored.
    len: usize,
    /// Node capacity configuration.
    config: SpatialConfig,
}

impl<const D: usize, T: Clone> Clone for SpatialIndexN<D, T> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            len: self.len,
            config: self.config,
        }
    }
}

impl<const D: usize, T> Default for SpatialIndexN<D, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, T> SpatialIndexN<D, T> {
    /// Creates a new, empty spatial index with default configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            root: NodeN::Leaf {
                entries: Vec::new(),
            },
            len: 0,
            config: SpatialConfig::DEFAULT,
        }
    }

    /// Creates a new, empty spatial index with the given configuration.
    #[must_use]
    pub const fn with_config(config: SpatialConfig) -> Self {
        Self {
            root: NodeN::Leaf {
                entries: Vec::new(),
            },
            len: 0,
            config,
        }
    }

    /// Returns the configuration used by this index.
    #[must_use]
    pub const fn config(&self) -> SpatialConfig {
        self.config
    }

    /// Inserts an entry into the index.
    pub fn insert(&mut self, entry: SpatialEntryN<D, T>) {
        self.len += 1;
        match self.root.insert_rstar(entry, self.config, true) {
            InsertResult::Ok => {},
            InsertResult::Split(sibling_bounds, sibling) => {
                self.promote_root(sibling_bounds, sibling);
            },
            InsertResult::Reinsert(entries) => {
                for e in entries {
                    self.insert_internal(e);
                }
            },
        }
    }

    /// Removes the first entry whose bounding box and data match the predicate.
    ///
    /// The `region` hint narrows the search to nodes overlapping it. After
    /// removal, underflowing nodes are condensed and their entries reinserted
    /// to maintain tree quality (Guttman's `condense_tree`).
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::NotFound`] if no matching entry exists.
    pub fn remove<F>(&mut self, region: BoundingBoxN<D>, pred: F) -> Result<(), SpatialError>
    where
        F: Fn(&SpatialEntryN<D, T>) -> bool,
    {
        let (found, orphans) = self.root.remove(region, &pred, self.config);
        if !found {
            return Err(SpatialError::NotFound);
        }
        self.len -= 1;
        self.shrink_root();
        for entry in orphans {
            self.insert_internal(entry);
        }
        Ok(())
    }

    /// Creates a new root from the current root and a sibling.
    fn promote_root(&mut self, sibling_bounds: BoundingBoxN<D>, sibling: NodeN<D, T>) {
        let old_root = std::mem::replace(
            &mut self.root,
            NodeN::Leaf {
                entries: Vec::new(),
            },
        );
        let old_bounds = old_root
            .bounds()
            .unwrap_or_else(BoundingBoxN::from_raw_zero);
        self.root = NodeN::Internal {
            children: vec![(old_bounds, old_root), (sibling_bounds, sibling)],
        };
    }

    /// Inserts an entry into the tree without incrementing `len`.
    ///
    /// Used to reinsert orphaned or force-reinserted entries that are already
    /// counted in `len`. Reinsertion is disabled (`allow_reinsert = false`) to
    /// prevent infinite loops.
    fn insert_internal(&mut self, entry: SpatialEntryN<D, T>) {
        match self.root.insert_rstar(entry, self.config, false) {
            InsertResult::Ok => {},
            InsertResult::Split(sb, sn) => self.promote_root(sb, sn),
            InsertResult::Reinsert(_) => {
                // Unreachable: allow_reinsert=false guarantees Split, not Reinsert.
                unreachable!("reinsert with allow_reinsert=false");
            },
        }
    }

    /// Collapses the root when it has 0 or 1 children.
    fn shrink_root(&mut self) {
        loop {
            match &mut self.root {
                NodeN::Internal { children } if children.is_empty() => {
                    self.root = NodeN::Leaf {
                        entries: Vec::new(),
                    };
                },
                NodeN::Internal { children } if children.len() == 1 => {
                    let (_, child) = children.pop().expect("single child");
                    self.root = child;
                },
                _ => break,
            }
        }
    }

    /// Returns all entries whose bounding box intersects `region`.
    #[must_use]
    pub fn query_region(&self, region: BoundingBoxN<D>) -> Vec<&SpatialEntryN<D, T>> {
        let mut results = Vec::new();
        self.root.query_region(region, &mut results);
        results
    }

    /// Returns the `k` entries nearest to `point`, ordered nearest-first.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. If fewer than `k` entries exist, all are returned.
    #[must_use]
    pub fn query_nearest_nd(&self, point: [f32; D], k: usize) -> Vec<&SpatialEntryN<D, T>> {
        let mut heap = BinaryHeap::new();
        self.root.query_nearest_heap(&point, &mut heap, k);
        let mut results: Vec<_> = heap.into_iter().map(|c| c.entry).collect();
        results.sort_by(|a, b| {
            let da = a.bounds.min_dist_sq_nd(&point);
            let db = b.bounds.min_dist_sq_nd(&point);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns all entries within `r` of `point`, sorted nearest-first.
    ///
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius_nd(&self, point: [f32; D], r: f32) -> Vec<&SpatialEntryN<D, T>> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(&point, r_sq, &mut results);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(entry, _)| entry).collect()
    }

    /// Returns all entries within `r` of `point` with their distances, sorted
    /// nearest-first.
    ///
    /// Each tuple contains `(entry, distance)` where distance is measured from
    /// the query point to the nearest edge of the bounding box (0 when inside).
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius_with_distances_nd(
        &self,
        point: [f32; D],
        r: f32,
    ) -> Vec<(&SpatialEntryN<D, T>, f32)> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(&point, r_sq, &mut results);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
            .into_iter()
            .map(|(entry, dist_sq)| (entry, dist_sq.sqrt()))
            .collect()
    }

    /// Returns the number of entries in the index.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the index contains no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes all entries from the index.
    pub fn clear(&mut self) {
        self.root = NodeN::Leaf {
            entries: Vec::new(),
        };
        self.len = 0;
    }

    /// Builds a spatial index from a pre-collected set of entries using the
    /// Sort-Tile-Recursive (STR) bulk-loading algorithm.
    ///
    /// The resulting tree has tighter bounding boxes and less overlap than
    /// one-at-a-time insertion, yielding better query performance for static
    /// datasets. The index can still be modified with [`insert`](Self::insert)
    /// and [`remove`](Self::remove) afterwards.
    #[must_use]
    pub fn bulk_load(entries: Vec<SpatialEntryN<D, T>>) -> Self {
        Self::bulk_load_with_config(entries, SpatialConfig::DEFAULT)
    }

    /// Builds a spatial index with the given configuration using the
    /// Sort-Tile-Recursive (STR) bulk-loading algorithm.
    #[must_use]
    pub fn bulk_load_with_config(entries: Vec<SpatialEntryN<D, T>>, config: SpatialConfig) -> Self {
        if entries.is_empty() {
            return Self::with_config(config);
        }
        let len = entries.len();
        let root = str_build_nodes(entries, config);
        Self { root, len, config }
    }

    /// Returns an iterator over references to all entries in the index.
    #[must_use]
    pub fn iter(&self) -> SpatialIterN<'_, D, T> {
        let mut entries = Vec::new();
        self.root.collect_all(&mut entries);
        SpatialIterN { entries, pos: 0 }
    }
}

// ---------------------------------------------------------------------------
// Specialized 2D convenience methods
// ---------------------------------------------------------------------------

impl<T> SpatialIndexN<2, T> {
    /// Returns the `k` entries nearest to the point `(x, y)`, ordered from
    /// nearest to farthest.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have
    /// distance 0. If fewer than `k` entries exist, all entries are returned.
    #[must_use]
    pub fn query_nearest(&self, x: f32, y: f32, k: usize) -> Vec<&SpatialEntryN<2, T>> {
        self.query_nearest_nd([x, y], k)
    }

    /// Returns all entries within `r` pixels of the point `(x, y)`, sorted
    /// nearest-first by edge distance.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have
    /// distance 0. Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius(&self, x: f32, y: f32, r: f32) -> Vec<&SpatialEntryN<2, T>> {
        self.query_within_radius_nd([x, y], r)
    }

    /// Returns all entries within `r` pixels of the point `(x, y)` with their
    /// distances, sorted nearest-first.
    ///
    /// Each tuple contains `(entry, distance)` where distance is measured from
    /// the query point to the nearest edge of the bounding box (0 when inside).
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius_with_distances(
        &self,
        x: f32,
        y: f32,
        r: f32,
    ) -> Vec<(&SpatialEntryN<2, T>, f32)> {
        self.query_within_radius_with_distances_nd([x, y], r)
    }
}

// ---------------------------------------------------------------------------
// Specialized 3D convenience methods
// ---------------------------------------------------------------------------

impl<T> SpatialIndexN<3, T> {
    /// Returns the `k` entries nearest to the point `(x, y, z)`, ordered from
    /// nearest to farthest.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have
    /// distance 0. If fewer than `k` entries exist, all entries are returned.
    #[must_use]
    pub fn query_nearest(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<&SpatialEntryN<3, T>> {
        self.query_nearest_nd([x, y, z], k)
    }

    /// Returns all entries within `r` units of the point `(x, y, z)`, sorted
    /// nearest-first by edge distance.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have
    /// distance 0. Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius(&self, x: f32, y: f32, z: f32, r: f32) -> Vec<&SpatialEntryN<3, T>> {
        self.query_within_radius_nd([x, y, z], r)
    }

    /// Returns all entries within `r` units of the point `(x, y, z)` with their
    /// distances, sorted nearest-first.
    ///
    /// Each tuple contains `(entry, distance)` where distance is measured from
    /// the query point to the nearest edge of the bounding box (0 when inside).
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius_with_distances(
        &self,
        x: f32,
        y: f32,
        z: f32,
        r: f32,
    ) -> Vec<(&SpatialEntryN<3, T>, f32)> {
        self.query_within_radius_with_distances_nd([x, y, z], r)
    }
}

// ---------------------------------------------------------------------------
// IntoIterator
// ---------------------------------------------------------------------------

impl<'a, const D: usize, T> IntoIterator for &'a SpatialIndexN<D, T> {
    type Item = &'a SpatialEntryN<D, T>;
    type IntoIter = SpatialIterN<'a, D, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

/// Version 3 serialization DTO that borrows entries to avoid cloning.
#[derive(Serialize)]
struct SpatialIndexDtoRefN<'a, const D: usize, T> {
    /// Serialization version tag.
    version: u8,
    /// Maximum entries per node.
    max_entries: usize,
    /// Split strategy: 0=Linear, 1=RStar.
    split_strategy: u8,
    /// Borrowed references to all entries.
    entries: Vec<&'a SpatialEntryN<D, T>>,
}

/// Version 3 deserialization DTO that owns entries.
#[derive(Deserialize)]
struct SpatialIndexDtoN<const D: usize, T> {
    /// Serialization version tag.
    version: u8,
    /// Maximum entries per node.
    max_entries: usize,
    /// Split strategy: 0=Linear, 1=RStar.
    split_strategy: u8,
    /// Owned entries to rebuild the index from.
    entries: Vec<SpatialEntryN<D, T>>,
}

impl<const D: usize, T: Serialize> Serialize for SpatialIndexN<D, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = Vec::with_capacity(self.len);
        self.root.collect_all(&mut entries);
        let dto = SpatialIndexDtoRefN {
            version: 3,
            max_entries: self.config.max_entries(),
            split_strategy: self.config.split_strategy() as u8,
            entries,
        };
        dto.serialize(serializer)
    }
}

impl<'de, const D: usize, T: Deserialize<'de>> Deserialize<'de> for SpatialIndexN<D, T> {
    fn deserialize<De: serde::Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let dto = SpatialIndexDtoN::<D, T>::deserialize(deserializer)?;
        if dto.version != 3 {
            return Err(serde::de::Error::custom("unsupported SpatialIndex version"));
        }
        let strategy = match dto.split_strategy {
            0 => SplitStrategy::Linear,
            1 => SplitStrategy::RStar,
            _ => return Err(serde::de::Error::custom("unsupported split strategy value")),
        };
        let config = SpatialConfig::with_strategy(dto.max_entries, strategy)
            .map_err(serde::de::Error::custom)?;
        Ok(Self::bulk_load_with_config(dto.entries, config))
    }
}
