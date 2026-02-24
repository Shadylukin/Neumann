//! Generic N-dimensional spatial index backed by an R-tree.

use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::bbox::{BoundingBoxN, SpatialEntryN};
use crate::iter::SpatialIterN;
use crate::node::NodeN;
use crate::SpatialError;

/// A spatial index backed by an R-tree for efficient region and
/// nearest-neighbor queries in `D`-dimensional space.
pub struct SpatialIndexN<const D: usize, T> {
    /// The root node of the R-tree.
    pub(crate) root: NodeN<D, T>,
    /// Number of entries stored.
    len: usize,
}

impl<const D: usize, T> Default for SpatialIndexN<D, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, T> SpatialIndexN<D, T> {
    /// Creates a new, empty spatial index.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            root: NodeN::Leaf {
                entries: Vec::new(),
            },
            len: 0,
        }
    }

    /// Inserts an entry into the index.
    pub fn insert(&mut self, entry: SpatialEntryN<D, T>) {
        if let Some((sibling_bounds, sibling)) = self.root.insert(entry) {
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
        self.len += 1;
    }

    /// Removes the first entry whose bounding box and data match the predicate.
    ///
    /// The `region` hint narrows the search to nodes overlapping it.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::NotFound`] if no matching entry exists.
    pub fn remove<F>(&mut self, region: BoundingBoxN<D>, pred: F) -> Result<(), SpatialError>
    where
        F: Fn(&SpatialEntryN<D, T>) -> bool,
    {
        if self.root.remove(region, &pred) {
            self.len -= 1;
            Ok(())
        } else {
            Err(SpatialError::NotFound)
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

/// Stable serialization DTO that borrows entries to avoid cloning.
#[derive(Serialize)]
struct SpatialIndexDtoRefN<'a, const D: usize, T> {
    /// Serialization version tag.
    version: u8,
    /// Borrowed references to all entries.
    entries: Vec<&'a SpatialEntryN<D, T>>,
}

/// Stable deserialization DTO that owns entries.
#[derive(Deserialize)]
struct SpatialIndexDtoN<const D: usize, T> {
    /// Serialization version tag.
    version: u8,
    /// Owned entries to rebuild the index from.
    entries: Vec<SpatialEntryN<D, T>>,
}

impl<const D: usize, T: Serialize> Serialize for SpatialIndexN<D, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = Vec::with_capacity(self.len);
        self.root.collect_all(&mut entries);
        let dto = SpatialIndexDtoRefN {
            version: 1,
            entries,
        };
        dto.serialize(serializer)
    }
}

impl<'de, const D: usize, T: Deserialize<'de>> Deserialize<'de> for SpatialIndexN<D, T> {
    fn deserialize<De: serde::Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let dto = SpatialIndexDtoN::<D, T>::deserialize(deserializer)?;
        if dto.version != 1 {
            return Err(serde::de::Error::custom("unsupported SpatialIndex version"));
        }
        let mut index = Self::new();
        for entry in dto.entries {
            index.insert(entry);
        }
        Ok(index)
    }
}
