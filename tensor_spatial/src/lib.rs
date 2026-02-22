//! R-tree spatial index for region and nearest-neighbor queries.
//!
//! Provides a node-based R-tree with linear split algorithm, supporting
//! insertion, removal, region queries, and k-nearest-neighbor lookups.

use std::collections::BinaryHeap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Maximum entries per R-tree node before splitting.
const MAX_ENTRIES: usize = 9;

/// Minimum entries per R-tree node after splitting.
const MIN_ENTRIES: usize = 4;

/// Errors that can occur during spatial operations.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SpatialError {
    /// A bounding box was constructed with negative dimensions.
    #[error("invalid bounding box: width and height must be non-negative")]
    InvalidBounds,

    /// The requested entry was not found in the index.
    #[error("entry not found in spatial index")]
    NotFound,

    /// A negative, NaN, or infinite radius was provided.
    #[error("invalid radius: must be non-negative and finite")]
    InvalidRadius,

    /// A 3D bounding box was constructed with negative dimensions.
    #[error("invalid 3D bounding box: width, height, and depth must be non-negative")]
    InvalidBounds3D,
}

/// An axis-aligned bounding box in 2D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// X coordinate of the lower-left corner.
    pub x: f32,
    /// Y coordinate of the lower-left corner.
    pub y: f32,
    /// Width of the bounding box.
    pub width: f32,
    /// Height of the bounding box.
    pub height: f32,
}

impl BoundingBox {
    /// Creates a new bounding box.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidBounds`] if width or height is negative.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, SpatialError> {
        if width < 0.0 || height < 0.0 {
            return Err(SpatialError::InvalidBounds);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Returns the center point of the bounding box as `(cx, cy)`.
    #[must_use]
    pub fn center(self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Returns the area of the bounding box.
    #[must_use]
    pub fn area(self) -> f32 {
        self.width * self.height
    }

    /// Returns `true` if the point `(px, py)` lies inside this bounding box.
    #[must_use]
    pub fn contains_point(self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    /// Returns `true` if this bounding box overlaps with `other`.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Returns the smallest bounding box that contains both `self` and `other`.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);
        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    /// Returns `true` if this bounding box has zero area.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.width == 0.0 || self.height == 0.0
    }

    /// Returns the minimum squared distance from point `(px, py)` to this box.
    #[must_use]
    pub fn min_dist_sq(self, px: f32, py: f32) -> f32 {
        let dx = if px < self.x {
            self.x - px
        } else if px > self.x + self.width {
            px - self.x - self.width
        } else {
            0.0
        };
        let dy = if py < self.y {
            self.y - py
        } else if py > self.y + self.height {
            py - self.y - self.height
        } else {
            0.0
        };
        dx.mul_add(dx, dy * dy)
    }
}

impl Serialize for BoundingBox {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.x, self.y, self.width, self.height).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BoundingBox {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (x, y, width, height) = <(f32, f32, f32, f32)>::deserialize(deserializer)?;
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
}

/// An entry in the spatial index pairing a bounding box with user data.
pub struct SpatialEntry<T> {
    /// The bounding box for this entry.
    pub bounds: BoundingBox,
    /// User-supplied data associated with this entry.
    pub data: T,
}

impl<T: Serialize> Serialize for SpatialEntry<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.bounds, &self.data).serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SpatialEntry<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (bounds, data) = <(BoundingBox, T)>::deserialize(deserializer)?;
        Ok(Self { bounds, data })
    }
}

impl<T: fmt::Debug> fmt::Debug for SpatialEntry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialEntry")
            .field("bounds", &self.bounds)
            .field("data", &self.data)
            .finish()
    }
}

impl<T: Clone> Clone for SpatialEntry<T> {
    fn clone(&self) -> Self {
        Self {
            bounds: self.bounds,
            data: self.data.clone(),
        }
    }
}

impl<T: PartialEq> PartialEq for SpatialEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds && self.data == other.data
    }
}

/// Internal R-tree node.
enum Node<T> {
    /// A leaf node containing spatial entries.
    Leaf { entries: Vec<SpatialEntry<T>> },
    /// An internal node containing child nodes with their bounding boxes.
    Internal { children: Vec<(BoundingBox, Self)> },
}

impl<T> Node<T> {
    /// Returns the bounding box enclosing all entries or children in this node.
    fn bounds(&self) -> Option<BoundingBox> {
        match self {
            Self::Leaf { entries } => {
                let mut iter = entries.iter().map(|e| e.bounds);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBox::union))
            },
            Self::Internal { children } => {
                let mut iter = children.iter().map(|(b, _)| *b);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBox::union))
            },
        }
    }

    /// Returns the number of data entries stored beneath this node.
    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::Leaf { entries } => entries.len(),
            Self::Internal { children } => children.iter().map(|(_, c)| c.len()).sum(),
        }
    }

    /// Collects all entries that intersect `region`.
    fn query_region<'a>(&'a self, region: BoundingBox, results: &mut Vec<&'a SpatialEntry<T>>) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.intersects(region) {
                        results.push(entry);
                    }
                }
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children {
                    if child_bounds.intersects(region) {
                        child.query_region(region, results);
                    }
                }
            },
        }
    }

    /// Pushes candidate entries onto the nearest-neighbor heap.
    fn query_nearest_heap<'a>(
        &'a self,
        px: f32,
        py: f32,
        heap: &mut BinaryHeap<NearestCandidate<'a, T>>,
        k: usize,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq(px, py);
                    if heap.len() < k {
                        heap.push(NearestCandidate { dist_sq, entry });
                    } else if let Some(worst) = heap.peek() {
                        if dist_sq < worst.dist_sq {
                            heap.pop();
                            heap.push(NearestCandidate { dist_sq, entry });
                        }
                    }
                }
            },
            Self::Internal { children } => {
                // Sort children by minimum distance to query point for pruning
                let mut child_dists: Vec<(f32, usize)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, (b, _))| (b.min_dist_sq(px, py), i))
                    .collect();
                child_dists
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (min_dist, idx) in child_dists {
                    // Prune: if we already have k results and this child's min distance
                    // exceeds the worst, skip it.
                    if heap.len() >= k {
                        if let Some(worst) = heap.peek() {
                            if min_dist > worst.dist_sq {
                                continue;
                            }
                        }
                    }
                    children[idx].1.query_nearest_heap(px, py, heap, k);
                }
            },
        }
    }

    /// Collects entries within a squared radius from a point.
    fn query_within_radius<'a>(
        &'a self,
        px: f32,
        py: f32,
        r_sq: f32,
        results: &mut Vec<(&'a SpatialEntry<T>, f32)>,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq(px, py);
                    if dist_sq <= r_sq {
                        results.push((entry, dist_sq));
                    }
                }
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children {
                    if child_bounds.min_dist_sq(px, py) <= r_sq {
                        child.query_within_radius(px, py, r_sq, results);
                    }
                }
            },
        }
    }

    /// Collects all entries in this subtree.
    fn collect_all<'a>(&'a self, out: &mut Vec<&'a SpatialEntry<T>>) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    out.push(entry);
                }
            },
            Self::Internal { children } => {
                for (_, child) in children {
                    child.collect_all(out);
                }
            },
        }
    }

    /// Inserts an entry, returning a split sibling if the node overflows.
    fn insert(&mut self, entry: SpatialEntry<T>) -> Option<(BoundingBox, Self)> {
        match self {
            Self::Leaf { entries } => {
                entries.push(entry);
                if entries.len() > MAX_ENTRIES {
                    Some(split_leaf(entries))
                } else {
                    None
                }
            },
            Self::Internal { children } => {
                // Choose the child whose bounding box needs least enlargement.
                let target = choose_subtree(children, entry.bounds);
                let split = children[target].1.insert(entry);
                // Update the child's bounding box.
                if let Some(b) = children[target].1.bounds() {
                    children[target].0 = b;
                }
                if let Some((sb, sn)) = split {
                    children.push((sb, sn));
                    if children.len() > MAX_ENTRIES {
                        Some(split_internal(children))
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        }
    }

    /// Removes the first entry matching the predicate. Returns `true` if found.
    fn remove<F>(&mut self, region: BoundingBox, pred: &F) -> bool
    where
        F: Fn(&SpatialEntry<T>) -> bool,
    {
        match self {
            Self::Leaf { entries } => {
                if let Some(pos) = entries.iter().position(pred) {
                    entries.remove(pos);
                    return true;
                }
                false
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children.iter_mut() {
                    if child_bounds.intersects(region) && child.remove(region, pred) {
                        // Update bounding box after removal.
                        if let Some(b) = child.bounds() {
                            *child_bounds = b;
                        }
                        return true;
                    }
                }
                false
            },
        }
    }
}

/// Candidate entry for nearest-neighbor search (max-heap by distance).
struct NearestCandidate<'a, T> {
    /// Squared distance from the query point to this entry's bounding box edge.
    dist_sq: f32,
    /// Reference to the spatial entry.
    entry: &'a SpatialEntry<T>,
}

impl<T> PartialEq for NearestCandidate<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl<T> Eq for NearestCandidate<'_, T> {}

impl<T> PartialOrd for NearestCandidate<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for NearestCandidate<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance comes first so we can pop the worst.
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Chooses the child whose bounding box needs the least area enlargement to
/// include `entry_bounds`.
fn choose_subtree<T>(children: &[(BoundingBox, Node<T>)], entry_bounds: BoundingBox) -> usize {
    children
        .iter()
        .enumerate()
        .min_by(|(_, (a_bb, _)), (_, (b_bb, _))| {
            let a_enlarge = a_bb.union(entry_bounds).area() - a_bb.area();
            let b_enlarge = b_bb.union(entry_bounds).area() - b_bb.area();
            a_enlarge
                .partial_cmp(&b_enlarge)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i)
}

/// Linear split for leaf nodes: picks the two most separated entries as seeds,
/// then distributes the rest by minimum area enlargement.
fn split_leaf<T>(entries: &mut Vec<SpatialEntry<T>>) -> (BoundingBox, Node<T>) {
    let (seed1, seed2) = pick_seeds_leaf(entries);
    let s2 = entries.swap_remove(seed2);
    let s1_idx = if seed1 == entries.len() {
        // seed1 was swapped into seed2's position by swap_remove
        seed2
    } else {
        seed1
    };
    let s1 = entries.swap_remove(s1_idx);

    let mut group1 = vec![s1];
    let mut group2 = vec![s2];
    let mut bb1 = group1[0].bounds;
    let mut bb2 = group2[0].bounds;

    while !entries.is_empty() {
        // If one group needs all remaining to reach minimum, give them all.
        if group1.len() + entries.len() == MIN_ENTRIES {
            group1.append(entries);
            break;
        }
        if group2.len() + entries.len() == MIN_ENTRIES {
            group2.append(entries);
            break;
        }

        let e = entries.pop().expect("entries is not empty");
        let enlarge1 = bb1.union(e.bounds).area() - bb1.area();
        let enlarge2 = bb2.union(e.bounds).area() - bb2.area();
        if enlarge1 <= enlarge2 {
            bb1 = bb1.union(e.bounds);
            group1.push(e);
        } else {
            bb2 = bb2.union(e.bounds);
            group2.push(e);
        }
    }

    *entries = group1;
    let sibling_bounds = group2
        .iter()
        .map(|e| e.bounds)
        .reduce(BoundingBox::union)
        .expect("group2 is not empty");
    (sibling_bounds, Node::Leaf { entries: group2 })
}

/// Linear split for internal nodes.
fn split_internal<T>(children: &mut Vec<(BoundingBox, Node<T>)>) -> (BoundingBox, Node<T>) {
    let (seed1, seed2) = pick_seeds_internal(children);
    let s2 = children.swap_remove(seed2);
    let s1_idx = if seed1 == children.len() {
        seed2
    } else {
        seed1
    };
    let s1 = children.swap_remove(s1_idx);

    let mut group1 = vec![s1];
    let mut group2 = vec![s2];
    let mut bb1 = group1[0].0;
    let mut bb2 = group2[0].0;

    while !children.is_empty() {
        if group1.len() + children.len() == MIN_ENTRIES {
            group1.append(children);
            break;
        }
        if group2.len() + children.len() == MIN_ENTRIES {
            group2.append(children);
            break;
        }

        let c = children.pop().expect("children is not empty");
        let enlarge1 = bb1.union(c.0).area() - bb1.area();
        let enlarge2 = bb2.union(c.0).area() - bb2.area();
        if enlarge1 <= enlarge2 {
            bb1 = bb1.union(c.0);
            group1.push(c);
        } else {
            bb2 = bb2.union(c.0);
            group2.push(c);
        }
    }

    *children = group1;

    let sibling_bb = group2
        .iter()
        .map(|(b, _)| *b)
        .reduce(BoundingBox::union)
        .expect("group2 is not empty");

    (sibling_bb, Node::Internal { children: group2 })
}

/// Picks two seed entries in a leaf with the largest separation along any axis.
fn pick_seeds_leaf<T>(entries: &[SpatialEntry<T>]) -> (usize, usize) {
    if entries.len() < 2 {
        return (0, entries.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, ei) in entries.iter().enumerate() {
        for (j, ej) in entries.iter().enumerate().skip(i + 1) {
            let combined = ei.bounds.union(ej.bounds).area();
            let waste = combined - ei.bounds.area() - ej.bounds.area();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}

/// Picks two seed children in an internal node with the largest separation.
fn pick_seeds_internal<T>(children: &[(BoundingBox, Node<T>)]) -> (usize, usize) {
    if children.len() < 2 {
        return (0, children.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, (bi, _)) in children.iter().enumerate() {
        for (j, (bj, _)) in children.iter().enumerate().skip(i + 1) {
            let combined = bi.union(*bj).area();
            let waste = combined - bi.area() - bj.area();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}

/// A spatial index backed by an R-tree for efficient region and nearest-neighbor
/// queries in 2D space.
pub struct SpatialIndex<T> {
    root: Node<T>,
    len: usize,
}

impl<T> Default for SpatialIndex<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SpatialIndex<T> {
    /// Creates a new, empty spatial index.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            root: Node::Leaf {
                entries: Vec::new(),
            },
            len: 0,
        }
    }

    /// Inserts an entry into the index.
    pub fn insert(&mut self, entry: SpatialEntry<T>) {
        if let Some((sibling_bounds, sibling)) = self.root.insert(entry) {
            // Root split: create a new root with the old root and sibling as children.
            let old_root = std::mem::replace(
                &mut self.root,
                Node::Leaf {
                    entries: Vec::new(),
                },
            );
            let old_bounds = old_root.bounds().unwrap_or(BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            });
            self.root = Node::Internal {
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
    pub fn remove<F>(&mut self, region: BoundingBox, pred: F) -> Result<(), SpatialError>
    where
        F: Fn(&SpatialEntry<T>) -> bool,
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
    pub fn query_region(&self, region: BoundingBox) -> Vec<&SpatialEntry<T>> {
        let mut results = Vec::new();
        self.root.query_region(region, &mut results);
        results
    }

    /// Returns the `k` entries nearest to the point `(x, y)`, ordered by distance
    /// from nearest to farthest.
    ///
    /// Distance is measured from the query point to the nearest edge of each entry's
    /// bounding box. Elements containing the query point have distance 0.
    /// If fewer than `k` entries exist, all entries are returned.
    #[must_use]
    pub fn query_nearest(&self, x: f32, y: f32, k: usize) -> Vec<&SpatialEntry<T>> {
        let mut heap = BinaryHeap::new();
        self.root.query_nearest_heap(x, y, &mut heap, k);
        let mut results: Vec<_> = heap.into_iter().map(|c| c.entry).collect();
        // Sort nearest-first (ascending edge distance).
        results.sort_by(|a, b| {
            let da = a.bounds.min_dist_sq(x, y);
            let db = b.bounds.min_dist_sq(x, y);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns all entries within `r` pixels of the point `(x, y)`, sorted
    /// nearest-first by edge distance.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have distance 0.
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius(&self, x: f32, y: f32, r: f32) -> Vec<&SpatialEntry<T>> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(x, y, r_sq, &mut results);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(entry, _)| entry).collect()
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
    ) -> Vec<(&SpatialEntry<T>, f32)> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(x, y, r_sq, &mut results);
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
        self.root = Node::Leaf {
            entries: Vec::new(),
        };
        self.len = 0;
    }

    /// Returns an iterator over references to all entries in the index.
    #[must_use]
    pub fn iter(&self) -> SpatialIter<'_, T> {
        let mut entries = Vec::new();
        self.root.collect_all(&mut entries);
        SpatialIter { entries, pos: 0 }
    }
}

impl<'a, T> IntoIterator for &'a SpatialIndex<T> {
    type Item = &'a SpatialEntry<T>;
    type IntoIter = SpatialIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over references to spatial entries.
pub struct SpatialIter<'a, T> {
    entries: Vec<&'a SpatialEntry<T>>,
    pos: usize,
}

impl<'a, T> Iterator for SpatialIter<'a, T> {
    type Item = &'a SpatialEntry<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let entry = self.entries[self.pos];
            self.pos += 1;
            Some(entry)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for SpatialIter<'_, T> {}

/// Stable serialization DTO that borrows entries to avoid cloning.
#[derive(Serialize)]
struct SpatialIndexDtoRef<'a, T> {
    version: u8,
    entries: Vec<&'a SpatialEntry<T>>,
}

/// Stable deserialization DTO that owns entries.
#[derive(Deserialize)]
struct SpatialIndexDto<T> {
    #[allow(dead_code)]
    version: u8,
    entries: Vec<SpatialEntry<T>>,
}

impl<T: Serialize> Serialize for SpatialIndex<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = Vec::with_capacity(self.len);
        self.root.collect_all(&mut entries);
        let dto = SpatialIndexDtoRef {
            version: 1,
            entries,
        };
        dto.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SpatialIndex<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let dto = SpatialIndexDto::<T>::deserialize(deserializer)?;
        let mut index = Self::new();
        for entry in dto.entries {
            index.insert(entry);
        }
        Ok(index)
    }
}

// ---------------------------------------------------------------------------
// 3D Types
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box in 3D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox3D {
    /// X coordinate of the lower corner.
    pub x: f32,
    /// Y coordinate of the lower corner.
    pub y: f32,
    /// Z coordinate of the lower corner.
    pub z: f32,
    /// Width of the bounding box (X extent).
    pub width: f32,
    /// Height of the bounding box (Y extent).
    pub height: f32,
    /// Depth of the bounding box (Z extent).
    pub depth: f32,
}

impl BoundingBox3D {
    /// Creates a new 3D bounding box.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidBounds3D`] if width, height, or depth is
    /// negative.
    pub fn new(
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<Self, SpatialError> {
        if width < 0.0 || height < 0.0 || depth < 0.0 {
            return Err(SpatialError::InvalidBounds3D);
        }
        Ok(Self {
            x,
            y,
            z,
            width,
            height,
            depth,
        })
    }

    /// Returns the center point of the bounding box as `(cx, cy, cz)`.
    #[must_use]
    pub fn center(self) -> (f32, f32, f32) {
        (
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
            self.z + self.depth / 2.0,
        )
    }

    /// Returns the volume of the bounding box.
    #[must_use]
    pub fn volume(self) -> f32 {
        self.width * self.height * self.depth
    }

    /// Returns `true` if the point `(px, py, pz)` lies inside this bounding box.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn contains_point(self, px: f32, py: f32, pz: f32) -> bool {
        px >= self.x
            && px <= self.x + self.width
            && py >= self.y
            && py <= self.y + self.height
            && pz >= self.z
            && pz <= self.z + self.depth
    }

    /// Returns `true` if this bounding box overlaps with `other`.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
            && self.z < other.z + other.depth
            && self.z + self.depth > other.z
    }

    /// Returns the smallest bounding box that contains both `self` and `other`.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let min_z = self.z.min(other.z);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);
        let max_z = (self.z + self.depth).max(other.z + other.depth);
        Self {
            x: min_x,
            y: min_y,
            z: min_z,
            width: max_x - min_x,
            height: max_y - min_y,
            depth: max_z - min_z,
        }
    }

    /// Returns `true` if this bounding box has zero volume.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.width == 0.0 || self.height == 0.0 || self.depth == 0.0
    }

    /// Returns the minimum squared distance from point `(px, py, pz)` to this
    /// box.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn min_dist_sq(self, px: f32, py: f32, pz: f32) -> f32 {
        let dx = if px < self.x {
            self.x - px
        } else if px > self.x + self.width {
            px - self.x - self.width
        } else {
            0.0
        };
        let dy = if py < self.y {
            self.y - py
        } else if py > self.y + self.height {
            py - self.y - self.height
        } else {
            0.0
        };
        let dz = if pz < self.z {
            self.z - pz
        } else if pz > self.z + self.depth {
            pz - self.z - self.depth
        } else {
            0.0
        };
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    }
}

impl Serialize for BoundingBox3D {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.x, self.y, self.z, self.width, self.height, self.depth).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BoundingBox3D {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (x, y, z, width, height, depth) =
            <(f32, f32, f32, f32, f32, f32)>::deserialize(deserializer)?;
        if width < 0.0 || height < 0.0 || depth < 0.0 {
            return Err(serde::de::Error::custom(
                "invalid 3D bounding box: width, height, and depth must be non-negative",
            ));
        }
        Ok(Self {
            x,
            y,
            z,
            width,
            height,
            depth,
        })
    }
}

/// An entry in the 3D spatial index pairing a bounding box with user data.
pub struct SpatialEntry3D<T> {
    /// The 3D bounding box for this entry.
    pub bounds: BoundingBox3D,
    /// User-supplied data associated with this entry.
    pub data: T,
}

impl<T: Serialize> Serialize for SpatialEntry3D<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.bounds, &self.data).serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SpatialEntry3D<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (bounds, data) = <(BoundingBox3D, T)>::deserialize(deserializer)?;
        Ok(Self { bounds, data })
    }
}

impl<T: fmt::Debug> fmt::Debug for SpatialEntry3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialEntry3D")
            .field("bounds", &self.bounds)
            .field("data", &self.data)
            .finish()
    }
}

impl<T: Clone> Clone for SpatialEntry3D<T> {
    fn clone(&self) -> Self {
        Self {
            bounds: self.bounds,
            data: self.data.clone(),
        }
    }
}

impl<T: PartialEq> PartialEq for SpatialEntry3D<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds && self.data == other.data
    }
}

/// Internal 3D R-tree node.
enum Node3D<T> {
    /// A leaf node containing 3D spatial entries.
    Leaf { entries: Vec<SpatialEntry3D<T>> },
    /// An internal node containing child nodes with their 3D bounding boxes.
    Internal {
        children: Vec<(BoundingBox3D, Self)>,
    },
}

impl<T> Node3D<T> {
    /// Returns the bounding box enclosing all entries or children in this node.
    fn bounds(&self) -> Option<BoundingBox3D> {
        match self {
            Self::Leaf { entries } => {
                let mut iter = entries.iter().map(|e| e.bounds);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBox3D::union))
            },
            Self::Internal { children } => {
                let mut iter = children.iter().map(|(b, _)| *b);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBox3D::union))
            },
        }
    }

    /// Returns the number of data entries stored beneath this node.
    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::Leaf { entries } => entries.len(),
            Self::Internal { children } => children.iter().map(|(_, c)| c.len()).sum(),
        }
    }

    /// Collects all entries that intersect `region`.
    fn query_region<'a>(&'a self, region: BoundingBox3D, results: &mut Vec<&'a SpatialEntry3D<T>>) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.intersects(region) {
                        results.push(entry);
                    }
                }
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children {
                    if child_bounds.intersects(region) {
                        child.query_region(region, results);
                    }
                }
            },
        }
    }

    /// Pushes candidate entries onto the nearest-neighbor heap.
    #[allow(clippy::similar_names)]
    fn query_nearest_heap<'a>(
        &'a self,
        px: f32,
        py: f32,
        pz: f32,
        heap: &mut BinaryHeap<NearestCandidate3D<'a, T>>,
        k: usize,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq(px, py, pz);
                    if heap.len() < k {
                        heap.push(NearestCandidate3D { dist_sq, entry });
                    } else if let Some(worst) = heap.peek() {
                        if dist_sq < worst.dist_sq {
                            heap.pop();
                            heap.push(NearestCandidate3D { dist_sq, entry });
                        }
                    }
                }
            },
            Self::Internal { children } => {
                let mut child_dists: Vec<(f32, usize)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, (b, _))| (b.min_dist_sq(px, py, pz), i))
                    .collect();
                child_dists
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (min_dist, idx) in child_dists {
                    if heap.len() >= k {
                        if let Some(worst) = heap.peek() {
                            if min_dist > worst.dist_sq {
                                continue;
                            }
                        }
                    }
                    children[idx].1.query_nearest_heap(px, py, pz, heap, k);
                }
            },
        }
    }

    /// Collects entries within a squared radius from a 3D point.
    #[allow(clippy::similar_names)]
    fn query_within_radius<'a>(
        &'a self,
        px: f32,
        py: f32,
        pz: f32,
        r_sq: f32,
        results: &mut Vec<(&'a SpatialEntry3D<T>, f32)>,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq(px, py, pz);
                    if dist_sq <= r_sq {
                        results.push((entry, dist_sq));
                    }
                }
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children {
                    if child_bounds.min_dist_sq(px, py, pz) <= r_sq {
                        child.query_within_radius(px, py, pz, r_sq, results);
                    }
                }
            },
        }
    }

    /// Collects all entries in this subtree.
    fn collect_all<'a>(&'a self, out: &mut Vec<&'a SpatialEntry3D<T>>) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    out.push(entry);
                }
            },
            Self::Internal { children } => {
                for (_, child) in children {
                    child.collect_all(out);
                }
            },
        }
    }

    /// Inserts an entry, returning a split sibling if the node overflows.
    fn insert(&mut self, entry: SpatialEntry3D<T>) -> Option<(BoundingBox3D, Self)> {
        match self {
            Self::Leaf { entries } => {
                entries.push(entry);
                if entries.len() > MAX_ENTRIES {
                    Some(split_leaf_3d(entries))
                } else {
                    None
                }
            },
            Self::Internal { children } => {
                let target = choose_subtree_3d(children, entry.bounds);
                let split = children[target].1.insert(entry);
                if let Some(b) = children[target].1.bounds() {
                    children[target].0 = b;
                }
                if let Some((sb, sn)) = split {
                    children.push((sb, sn));
                    if children.len() > MAX_ENTRIES {
                        Some(split_internal_3d(children))
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        }
    }

    /// Removes the first entry matching the predicate. Returns `true` if found.
    fn remove<F>(&mut self, region: BoundingBox3D, pred: &F) -> bool
    where
        F: Fn(&SpatialEntry3D<T>) -> bool,
    {
        match self {
            Self::Leaf { entries } => {
                if let Some(pos) = entries.iter().position(pred) {
                    entries.remove(pos);
                    return true;
                }
                false
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children.iter_mut() {
                    if child_bounds.intersects(region) && child.remove(region, pred) {
                        if let Some(b) = child.bounds() {
                            *child_bounds = b;
                        }
                        return true;
                    }
                }
                false
            },
        }
    }
}

/// Candidate entry for 3D nearest-neighbor search (max-heap by distance).
struct NearestCandidate3D<'a, T> {
    /// Squared distance from the query point to this entry's bounding box edge.
    dist_sq: f32,
    /// Reference to the 3D spatial entry.
    entry: &'a SpatialEntry3D<T>,
}

impl<T> PartialEq for NearestCandidate3D<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl<T> Eq for NearestCandidate3D<'_, T> {}

impl<T> PartialOrd for NearestCandidate3D<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for NearestCandidate3D<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance comes first so we can pop the worst.
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Chooses the 3D child whose bounding box needs the least volume enlargement.
fn choose_subtree_3d<T>(
    children: &[(BoundingBox3D, Node3D<T>)],
    entry_bounds: BoundingBox3D,
) -> usize {
    children
        .iter()
        .enumerate()
        .min_by(|(_, (a_bb, _)), (_, (b_bb, _))| {
            let a_enlarge = a_bb.union(entry_bounds).volume() - a_bb.volume();
            let b_enlarge = b_bb.union(entry_bounds).volume() - b_bb.volume();
            a_enlarge
                .partial_cmp(&b_enlarge)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i)
}

/// Linear split for 3D leaf nodes: picks the two most separated entries as
/// seeds, then distributes the rest by minimum volume enlargement.
fn split_leaf_3d<T>(entries: &mut Vec<SpatialEntry3D<T>>) -> (BoundingBox3D, Node3D<T>) {
    let (seed1, seed2) = pick_seeds_leaf_3d(entries);
    let s2 = entries.swap_remove(seed2);
    let s1_idx = if seed1 == entries.len() { seed2 } else { seed1 };
    let s1 = entries.swap_remove(s1_idx);

    let mut group1 = vec![s1];
    let mut group2 = vec![s2];
    let mut bb1 = group1[0].bounds;
    let mut bb2 = group2[0].bounds;

    while !entries.is_empty() {
        if group1.len() + entries.len() == MIN_ENTRIES {
            group1.append(entries);
            break;
        }
        if group2.len() + entries.len() == MIN_ENTRIES {
            group2.append(entries);
            break;
        }

        let e = entries.pop().expect("entries is not empty");
        let enlarge1 = bb1.union(e.bounds).volume() - bb1.volume();
        let enlarge2 = bb2.union(e.bounds).volume() - bb2.volume();
        if enlarge1 <= enlarge2 {
            bb1 = bb1.union(e.bounds);
            group1.push(e);
        } else {
            bb2 = bb2.union(e.bounds);
            group2.push(e);
        }
    }

    *entries = group1;
    let sibling_bounds = group2
        .iter()
        .map(|e| e.bounds)
        .reduce(BoundingBox3D::union)
        .expect("group2 is not empty");
    (sibling_bounds, Node3D::Leaf { entries: group2 })
}

/// Linear split for 3D internal nodes.
fn split_internal_3d<T>(
    children: &mut Vec<(BoundingBox3D, Node3D<T>)>,
) -> (BoundingBox3D, Node3D<T>) {
    let (seed1, seed2) = pick_seeds_internal_3d(children);
    let s2 = children.swap_remove(seed2);
    let s1_idx = if seed1 == children.len() {
        seed2
    } else {
        seed1
    };
    let s1 = children.swap_remove(s1_idx);

    let mut group1 = vec![s1];
    let mut group2 = vec![s2];
    let mut bb1 = group1[0].0;
    let mut bb2 = group2[0].0;

    while !children.is_empty() {
        if group1.len() + children.len() == MIN_ENTRIES {
            group1.append(children);
            break;
        }
        if group2.len() + children.len() == MIN_ENTRIES {
            group2.append(children);
            break;
        }

        let c = children.pop().expect("children is not empty");
        let enlarge1 = bb1.union(c.0).volume() - bb1.volume();
        let enlarge2 = bb2.union(c.0).volume() - bb2.volume();
        if enlarge1 <= enlarge2 {
            bb1 = bb1.union(c.0);
            group1.push(c);
        } else {
            bb2 = bb2.union(c.0);
            group2.push(c);
        }
    }

    *children = group1;

    let sibling_bb = group2
        .iter()
        .map(|(b, _)| *b)
        .reduce(BoundingBox3D::union)
        .expect("group2 is not empty");

    (sibling_bb, Node3D::Internal { children: group2 })
}

/// Picks two seed entries in a 3D leaf with the largest separation.
fn pick_seeds_leaf_3d<T>(entries: &[SpatialEntry3D<T>]) -> (usize, usize) {
    if entries.len() < 2 {
        return (0, entries.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, ei) in entries.iter().enumerate() {
        for (j, ej) in entries.iter().enumerate().skip(i + 1) {
            let combined = ei.bounds.union(ej.bounds).volume();
            let waste = combined - ei.bounds.volume() - ej.bounds.volume();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}

/// Picks two seed children in a 3D internal node with the largest separation.
fn pick_seeds_internal_3d<T>(children: &[(BoundingBox3D, Node3D<T>)]) -> (usize, usize) {
    if children.len() < 2 {
        return (0, children.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, (bi, _)) in children.iter().enumerate() {
        for (j, (bj, _)) in children.iter().enumerate().skip(i + 1) {
            let combined = bi.union(*bj).volume();
            let waste = combined - bi.volume() - bj.volume();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}

/// A 3D spatial index backed by an R-tree for efficient region and
/// nearest-neighbor queries in 3D space.
pub struct SpatialIndex3D<T> {
    root: Node3D<T>,
    len: usize,
}

impl<T> Default for SpatialIndex3D<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SpatialIndex3D<T> {
    /// Creates a new, empty 3D spatial index.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            root: Node3D::Leaf {
                entries: Vec::new(),
            },
            len: 0,
        }
    }

    /// Inserts an entry into the 3D index.
    pub fn insert(&mut self, entry: SpatialEntry3D<T>) {
        if let Some((sibling_bounds, sibling)) = self.root.insert(entry) {
            let old_root = std::mem::replace(
                &mut self.root,
                Node3D::Leaf {
                    entries: Vec::new(),
                },
            );
            let old_bounds = old_root.bounds().unwrap_or(BoundingBox3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                width: 0.0,
                height: 0.0,
                depth: 0.0,
            });
            self.root = Node3D::Internal {
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
    pub fn remove<F>(&mut self, region: BoundingBox3D, pred: F) -> Result<(), SpatialError>
    where
        F: Fn(&SpatialEntry3D<T>) -> bool,
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
    pub fn query_region(&self, region: BoundingBox3D) -> Vec<&SpatialEntry3D<T>> {
        let mut results = Vec::new();
        self.root.query_region(region, &mut results);
        results
    }

    /// Returns the `k` entries nearest to the point `(x, y, z)`, ordered by
    /// distance from nearest to farthest.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have distance 0.
    /// If fewer than `k` entries exist, all entries are returned.
    #[must_use]
    pub fn query_nearest(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<&SpatialEntry3D<T>> {
        let mut heap = BinaryHeap::new();
        self.root.query_nearest_heap(x, y, z, &mut heap, k);
        let mut results: Vec<_> = heap.into_iter().map(|c| c.entry).collect();
        results.sort_by(|a, b| {
            let da = a.bounds.min_dist_sq(x, y, z);
            let db = b.bounds.min_dist_sq(x, y, z);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns all entries within `r` units of the point `(x, y, z)`, sorted
    /// nearest-first by edge distance.
    ///
    /// Distance is measured from the query point to the nearest edge of each
    /// entry's bounding box. Elements containing the query point have distance 0.
    /// Returns an empty vector if `r < 0.0`.
    #[must_use]
    pub fn query_within_radius(&self, x: f32, y: f32, z: f32, r: f32) -> Vec<&SpatialEntry3D<T>> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(x, y, z, r_sq, &mut results);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(entry, _)| entry).collect()
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
    ) -> Vec<(&SpatialEntry3D<T>, f32)> {
        if r < 0.0 {
            return Vec::new();
        }
        let r_sq = r * r;
        let mut results = Vec::new();
        self.root.query_within_radius(x, y, z, r_sq, &mut results);
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
        self.root = Node3D::Leaf {
            entries: Vec::new(),
        };
        self.len = 0;
    }

    /// Returns an iterator over references to all entries in the index.
    #[must_use]
    pub fn iter(&self) -> SpatialIter3D<'_, T> {
        let mut entries = Vec::new();
        self.root.collect_all(&mut entries);
        SpatialIter3D { entries, pos: 0 }
    }
}

impl<'a, T> IntoIterator for &'a SpatialIndex3D<T> {
    type Item = &'a SpatialEntry3D<T>;
    type IntoIter = SpatialIter3D<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over references to 3D spatial entries.
pub struct SpatialIter3D<'a, T> {
    entries: Vec<&'a SpatialEntry3D<T>>,
    pos: usize,
}

impl<'a, T> Iterator for SpatialIter3D<'a, T> {
    type Item = &'a SpatialEntry3D<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let entry = self.entries[self.pos];
            self.pos += 1;
            Some(entry)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for SpatialIter3D<'_, T> {}

/// Stable serialization DTO for 3D index (borrows entries).
#[derive(Serialize)]
struct SpatialIndex3DDtoRef<'a, T> {
    version: u8,
    entries: Vec<&'a SpatialEntry3D<T>>,
}

/// Stable deserialization DTO for 3D index (owns entries).
#[derive(Deserialize)]
struct SpatialIndex3DDto<T> {
    version: u8,
    entries: Vec<SpatialEntry3D<T>>,
}

impl<T: Serialize> Serialize for SpatialIndex3D<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = Vec::with_capacity(self.len);
        self.root.collect_all(&mut entries);
        let dto = SpatialIndex3DDtoRef {
            version: 1,
            entries,
        };
        dto.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SpatialIndex3D<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let dto = SpatialIndex3DDto::<T>::deserialize(deserializer)?;
        if dto.version != 1 {
            return Err(serde::de::Error::custom(
                "unsupported SpatialIndex3D version",
            ));
        }
        let mut index = Self::new();
        for entry in dto.entries {
            index.insert(entry);
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_new_and_accessors() {
        let bb = BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_eq!(bb.x, 1.0);
        assert_eq!(bb.y, 2.0);
        assert_eq!(bb.width, 3.0);
        assert_eq!(bb.height, 4.0);
    }

    #[test]
    fn test_bounding_box_new_rejects_negative_width() {
        assert!(BoundingBox::new(0.0, 0.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_bounding_box_new_rejects_negative_height() {
        assert!(BoundingBox::new(0.0, 0.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_bounding_box_center() {
        let bb = BoundingBox::new(0.0, 0.0, 4.0, 6.0).unwrap();
        assert_eq!(bb.center(), (2.0, 3.0));
    }

    #[test]
    fn test_bounding_box_contains_point() {
        let bb = BoundingBox::new(1.0, 1.0, 3.0, 3.0).unwrap();
        // Inside
        assert!(bb.contains_point(2.0, 2.0));
        // On edge
        assert!(bb.contains_point(1.0, 1.0));
        assert!(bb.contains_point(4.0, 4.0));
        // Outside
        assert!(!bb.contains_point(0.0, 2.0));
        assert!(!bb.contains_point(5.0, 2.0));
        assert!(!bb.contains_point(2.0, 0.0));
        assert!(!bb.contains_point(2.0, 5.0));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
        let b = BoundingBox::new(1.0, 1.0, 2.0, 2.0).unwrap();
        assert!(a.intersects(b));
        assert!(b.intersects(a));

        let c = BoundingBox::new(5.0, 5.0, 1.0, 1.0).unwrap();
        assert!(!a.intersects(c));
        assert!(!c.intersects(a));
    }

    #[test]
    fn test_bounding_box_intersects_edge_touching() {
        let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
        let b = BoundingBox::new(2.0, 0.0, 2.0, 2.0).unwrap();
        assert!(!a.intersects(b));
    }

    #[test]
    fn test_bounding_box_union() {
        let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
        let b = BoundingBox::new(1.0, 1.0, 3.0, 3.0).unwrap();
        let u = a.union(b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, 0.0);
        assert_eq!(u.width, 4.0);
        assert_eq!(u.height, 4.0);
    }

    #[test]
    fn test_bounding_box_area() {
        let bb = BoundingBox::new(0.0, 0.0, 3.0, 5.0).unwrap();
        assert!((bb.area() - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_is_empty() {
        assert!(BoundingBox::new(0.0, 0.0, 0.0, 5.0).unwrap().is_empty());
        assert!(BoundingBox::new(0.0, 0.0, 5.0, 0.0).unwrap().is_empty());
        assert!(BoundingBox::new(0.0, 0.0, 0.0, 0.0).unwrap().is_empty());
        assert!(!BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap().is_empty());
    }

    #[test]
    fn test_spatial_entry() {
        let entry = SpatialEntry {
            bounds: BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap(),
            data: 42,
        };
        assert_eq!(entry.data, 42);
        assert_eq!(entry.bounds.x, 1.0);

        // Test Debug impl
        let debug_str = format!("{entry:?}");
        assert!(debug_str.contains("SpatialEntry"));

        // Test Clone impl
        let cloned = entry.clone();
        assert_eq!(cloned.data, entry.data);
        assert_eq!(cloned.bounds, entry.bounds);

        // Test PartialEq impl
        assert_eq!(entry, cloned);
    }

    #[test]
    fn test_spatial_index_empty() {
        let index: SpatialIndex<u32> = SpatialIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.iter().count(), 0);
    }

    #[test]
    fn test_spatial_index_default() {
        let index: SpatialIndex<u32> = SpatialIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn test_spatial_index_insert_and_query_region() {
        let mut index = SpatialIndex::new();

        for i in 0..20 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 20);

        let region = BoundingBox::new(0.0, 0.0, 12.0, 12.0).unwrap();
        let results = index.query_region(region);
        assert!(!results.is_empty());

        for entry in &results {
            assert!(entry.bounds.intersects(region));
        }
    }

    #[test]
    fn test_spatial_index_query_region_no_results() {
        let mut index = SpatialIndex::new();
        for i in 0..10 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let region = BoundingBox::new(100.0, 100.0, 1.0, 1.0).unwrap();
        let results = index.query_region(region);
        assert!(results.is_empty());
    }

    #[test]
    fn test_spatial_index_query_nearest() {
        let mut index = SpatialIndex::new();

        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
            data: "origin",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(10.0, 10.0, 1.0, 1.0).unwrap(),
            data: "far",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(2.0, 2.0, 1.0, 1.0).unwrap(),
            data: "near",
        });

        let results = index.query_nearest(0.0, 0.0, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data, "origin");
        assert_eq!(results[1].data, "near");
    }

    #[test]
    fn test_spatial_index_query_nearest_k_larger_than_n() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
            data: 1,
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(5.0, 5.0, 1.0, 1.0).unwrap(),
            data: 2,
        });

        let results = index.query_nearest(0.0, 0.0, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_remove() {
        let mut index = SpatialIndex::new();
        let bb = BoundingBox::new(1.0, 1.0, 2.0, 2.0).unwrap();
        index.insert(SpatialEntry {
            bounds: bb,
            data: 42u32,
        });
        assert_eq!(index.len(), 1);

        index.remove(bb, |e| e.data == 42).unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let result = index.remove(bb, |e| e.data == 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_index_remove_from_tree() {
        let mut index = SpatialIndex::new();
        for i in 0..20u32 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 20);

        let bb = BoundingBox::new(0.0, 0.0, 5.0, 5.0).unwrap();
        index.remove(bb, |e| e.data == 0).unwrap();
        assert_eq!(index.len(), 19);
    }

    #[test]
    fn test_spatial_index_clear() {
        let mut index = SpatialIndex::new();
        for i in 0..10 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 10);
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_iter() {
        let mut index = SpatialIndex::new();
        for i in 0..5 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all.len(), 5);
        let iter = index.iter();
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn test_spatial_index_into_iter() {
        let mut index = SpatialIndex::new();
        for i in 0..3 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let all: Vec<_> = (&index).into_iter().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_spatial_index_concurrent() {
        use std::sync::Arc;

        let index = Arc::new(parking_lot::Mutex::new(SpatialIndex::<u32>::new()));
        let mut handles = Vec::new();

        for thread_id in 0..4u32 {
            let idx = Arc::clone(&index);
            handles.push(std::thread::spawn(move || {
                for i in 0..100u32 {
                    let x = (thread_id * 100 + i) as f32;
                    idx.lock().insert(SpatialEntry {
                        bounds: BoundingBox::new(x, 0.0, 1.0, 1.0).unwrap(),
                        data: thread_id * 100 + i,
                    });
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(index.lock().len(), 400);
    }

    #[test]
    fn test_spatial_index_large_dataset() {
        let mut index = SpatialIndex::new();
        for i in 0..10_000u32 {
            let x = (i % 100) as f32;
            let y = (i / 100) as f32;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 10_000);

        let region = BoundingBox::new(0.0, 0.0, 5.5, 5.5).unwrap();
        let results = index.query_region(region);
        assert!(!results.is_empty());
        for entry in &results {
            assert!(entry.bounds.intersects(region));
        }

        let nearest = index.query_nearest(50.0, 50.0, 5);
        assert_eq!(nearest.len(), 5);
    }

    #[test]
    fn test_spatial_error_display() {
        let err = SpatialError::InvalidBounds;
        assert_eq!(
            err.to_string(),
            "invalid bounding box: width and height must be non-negative"
        );
        let err = SpatialError::NotFound;
        assert_eq!(err.to_string(), "entry not found in spatial index");
        let err = SpatialError::InvalidRadius;
        assert_eq!(
            err.to_string(),
            "invalid radius: must be non-negative and finite"
        );
        let err = SpatialError::InvalidBounds3D;
        assert_eq!(
            err.to_string(),
            "invalid 3D bounding box: width, height, and depth must be non-negative"
        );
    }

    #[test]
    fn test_bounding_box_min_dist_sq() {
        let bb = BoundingBox::new(2.0, 2.0, 3.0, 3.0).unwrap();
        // Point inside box: distance is 0
        assert_eq!(bb.min_dist_sq(3.0, 3.0), 0.0);
        // Point to the left
        assert!((bb.min_dist_sq(0.0, 3.0) - 4.0).abs() < f32::EPSILON);
        // Point below
        assert!((bb.min_dist_sq(3.0, 0.0) - 4.0).abs() < f32::EPSILON);
        // Point diagonal (lower-left corner: (0,0) -> nearest corner (2,2))
        assert!((bb.min_dist_sq(0.0, 0.0) - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_node_bounds_empty_leaf() {
        let node: Node<u32> = Node::Leaf {
            entries: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    #[test]
    fn test_node_bounds_empty_internal() {
        let node: Node<u32> = Node::Internal {
            children: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    #[test]
    fn test_query_nearest_empty_index() {
        let index: SpatialIndex<u32> = SpatialIndex::new();
        let results = index.query_nearest(0.0, 0.0, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_split_preserves_entries() {
        let mut index = SpatialIndex::new();
        for i in 0..15u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 15);
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all.len(), 15);
    }

    #[test]
    fn test_multiple_splits() {
        let mut index = SpatialIndex::new();
        for i in 0..100u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, i as f32, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 100);
        assert_eq!(index.iter().count(), 100);
    }

    #[test]
    fn test_internal_node_len() {
        let mut index = SpatialIndex::new();
        for i in 0..50u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 50);
        assert_eq!(index.root.len(), 50);
    }

    #[test]
    fn test_bounding_box_min_dist_sq_right_of_box() {
        let bb = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
        // Point to the right: (5, 1) -> nearest edge at x=2
        assert!((bb.min_dist_sq(5.0, 1.0) - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_min_dist_sq_is_pub() {
        // Verifies min_dist_sq is accessible from outside the crate
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap();
        let dist = bb.min_dist_sq(5.0, 5.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_bounding_box_min_dist_sq_above_box() {
        let bb = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
        // Point above: (1, 5) -> nearest edge at y=2
        assert!((bb.min_dist_sq(1.0, 5.0) - 9.0).abs() < f32::EPSILON);
    }

    // --- query_within_radius tests ---

    #[test]
    fn test_query_within_radius_point_inside() {
        let mut index = SpatialIndex::new();
        // Large element
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
            data: "large",
        });
        // Point (50,50) is inside -> distance 0, included for any r > 0
        let results = index.query_within_radius(50.0, 50.0, 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, "large");
    }

    #[test]
    fn test_query_within_radius_point_outside() {
        let mut index = SpatialIndex::new();
        // Element at (10, 10, 5, 5) -> right edge at x=15
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(10.0, 10.0, 5.0, 5.0).unwrap(),
            data: "box",
        });
        // Point (20, 12) -> nearest edge at x=15, distance = 5
        let results = index.query_within_radius(20.0, 12.0, 5.0);
        assert_eq!(results.len(), 1);
        // r=4.9 should not include it
        let results = index.query_within_radius(20.0, 12.0, 4.9);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_zero() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
            data: "inside",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(200.0, 200.0, 10.0, 10.0).unwrap(),
            data: "outside",
        });
        // r=0 should only return elements containing the point
        let results = index.query_within_radius(50.0, 50.0, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, "inside");
    }

    #[test]
    fn test_query_within_radius_negative() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
            data: 1,
        });
        let results = index.query_within_radius(50.0, 50.0, -1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_sorted_nearest_first() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(100.0, 0.0, 10.0, 10.0).unwrap(),
            data: "far",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(20.0, 0.0, 10.0, 10.0).unwrap(),
            data: "near",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
            data: "closest",
        });
        // Query from (0,0) with large radius
        let results = index.query_within_radius(0.0, 0.0, 200.0);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data, "closest");
        assert_eq!(results[1].data, "near");
        assert_eq!(results[2].data, "far");
    }

    #[test]
    fn test_query_within_radius_empty_index() {
        let index: SpatialIndex<u32> = SpatialIndex::new();
        let results = index.query_within_radius(0.0, 0.0, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_with_distances() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
            data: "inside",
        });
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(110.0, 0.0, 10.0, 10.0).unwrap(),
            data: "outside",
        });
        let results = index.query_within_radius_with_distances(50.0, 50.0, 200.0);
        assert_eq!(results.len(), 2);
        // First: inside element, distance 0
        assert_eq!(results[0].0.data, "inside");
        assert!((results[0].1 - 0.0).abs() < f32::EPSILON);
        // Second: outside element, dx=60 dy=40, distance = sqrt(5200) ≈ 72.1
        assert_eq!(results[1].0.data, "outside");
        let expected = (60.0_f32.powi(2) + 40.0_f32.powi(2)).sqrt();
        assert!((results[1].1 - expected).abs() < 0.1);
    }

    #[test]
    fn test_large_bbox_edge_distance_regression() {
        let mut index = SpatialIndex::new();
        // Large banner spanning full width
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 1000.0, 100.0).unwrap(),
            data: "banner",
        });
        // Small button far away
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(50.0, 200.0, 10.0, 10.0).unwrap(),
            data: "button",
        });

        // Point (50, 50) is inside the banner (distance 0)
        // but old center-distance would rank the small button closer
        // because banner center is (500, 50), button center is (55, 205)
        let nearest = index.query_nearest(50.0, 50.0, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(
            nearest[0].data, "banner",
            "Banner should be nearest (point is inside)"
        );
        assert_eq!(nearest[1].data, "button");

        // Radius query: r=50 should include banner (distance 0) but not button (distance ~150)
        let radius_results = index.query_within_radius(50.0, 50.0, 50.0);
        assert_eq!(radius_results.len(), 1);
        assert_eq!(radius_results[0].data, "banner");

        // With larger radius, should include both
        let radius_results = index.query_within_radius(50.0, 50.0, 200.0);
        assert_eq!(radius_results.len(), 2);
        assert_eq!(radius_results[0].data, "banner");
        assert_eq!(radius_results[1].data, "button");
    }

    #[test]
    fn test_query_within_radius_large_dataset_brute_force() {
        let mut index = SpatialIndex::new();
        for i in 0..10_000u32 {
            let x = (i % 100) as f32;
            let y = (i / 100) as f32;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
                data: i,
            });
        }

        let cx = 50.0_f32;
        let cy = 50.0_f32;
        let radius = 10.0_f32;

        let results = index.query_within_radius(cx, cy, radius);

        // Brute-force: check every entry
        let mut expected_count = 0;
        for entry in index.iter() {
            if entry.bounds.min_dist_sq(cx, cy) <= radius * radius {
                expected_count += 1;
            }
        }
        assert_eq!(results.len(), expected_count);

        // All results should be within radius
        for entry in &results {
            let dist = entry.bounds.min_dist_sq(cx, cy).sqrt();
            assert!(dist <= radius + f32::EPSILON);
        }
    }

    #[test]
    fn test_query_within_radius_all_entries() {
        let mut index = SpatialIndex::new();
        for i in 0..20u32 {
            let x = (i % 5) as f32;
            let y = (i / 5) as f32;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        // Radius large enough to capture everything
        let results = index.query_within_radius(2.5, 2.5, 1000.0);
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn test_query_within_radius_with_distances_negative() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
            data: 1,
        });
        let results = index.query_within_radius_with_distances(0.0, 0.0, -1.0);
        assert!(results.is_empty());
    }

    // --- serde roundtrip tests ---

    #[test]
    fn test_bounding_box_serde_roundtrip() {
        let bb = BoundingBox::new(1.5, 2.5, 10.0, 20.0).unwrap();
        let bytes = bitcode::serialize(&bb).unwrap();
        let restored: BoundingBox = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(bb, restored);
    }

    #[test]
    fn test_bounding_box_serde_zero_dimensions() {
        let bb = BoundingBox::new(0.0, 0.0, 0.0, 0.0).unwrap();
        let bytes = bitcode::serialize(&bb).unwrap();
        let restored: BoundingBox = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(bb, restored);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_spatial_entry_serde_roundtrip() {
        let entry = SpatialEntry {
            bounds: BoundingBox::new(3.0, 4.0, 5.0, 6.0).unwrap(),
            data: String::from("hello"),
        };
        let bytes = bitcode::serialize(&entry).unwrap();
        let restored: SpatialEntry<String> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(entry, restored);
    }

    #[test]
    fn test_spatial_index_serde_empty_roundtrip() {
        let index: SpatialIndex<u32> = SpatialIndex::new();
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_spatial_index_serde_single_entry_roundtrip() {
        let mut index = SpatialIndex::new();
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap(),
            data: 42u32,
        });
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 1);
        let region = BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap();
        let results = restored.query_region(region);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, 42);
    }

    #[test]
    fn test_spatial_index_serde_tree_with_splits_roundtrip() {
        let mut index = SpatialIndex::new();
        for i in 0..50u32 {
            let x = (i % 10) as f32 * 5.0;
            let y = (i / 10) as f32 * 5.0;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 2.0, 2.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 50);

        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 50);

        // Region query should return same count
        let region = BoundingBox::new(0.0, 0.0, 50.0, 50.0).unwrap();
        let orig_results = index.query_region(region);
        let rest_results = restored.query_region(region);
        assert_eq!(orig_results.len(), rest_results.len());

        // Nearest query: compare data sets (not ordering) to avoid tie flakiness
        let orig_nearest = index.query_nearest(25.0, 12.0, 5);
        let rest_nearest = restored.query_nearest(25.0, 12.0, 5);
        let mut orig_data: Vec<_> = orig_nearest.iter().map(|e| e.data).collect();
        let mut rest_data: Vec<_> = rest_nearest.iter().map(|e| e.data).collect();
        orig_data.sort_unstable();
        rest_data.sort_unstable();
        assert_eq!(orig_data, rest_data);
    }

    #[test]
    fn test_spatial_index_serde_string_data_roundtrip() {
        let mut index = SpatialIndex::new();
        for i in 0..10 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: format!("item_{i}"),
            });
        }
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex<String> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 10);

        let region = BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap();
        let mut orig_data: Vec<_> = index
            .query_region(region)
            .iter()
            .map(|e| &e.data)
            .cloned()
            .collect();
        let mut rest_data: Vec<_> = restored
            .query_region(region)
            .iter()
            .map(|e| &e.data)
            .cloned()
            .collect();
        orig_data.sort();
        rest_data.sort();
        assert_eq!(orig_data, rest_data);
    }

    #[test]
    fn test_spatial_index_serde_mutate_after_restore() {
        let mut index = SpatialIndex::new();
        for i in 0..5u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let bytes = bitcode::serialize(&index).unwrap();
        let mut restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 5);

        // Insert after restore
        restored.insert(SpatialEntry {
            bounds: BoundingBox::new(10.0, 10.0, 1.0, 1.0).unwrap(),
            data: 99,
        });
        assert_eq!(restored.len(), 6);

        // Remove after restore
        let bb = BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap();
        restored.remove(bb, |e| e.data == 0).unwrap();
        assert_eq!(restored.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 3D BoundingBox tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_3d_new_and_accessors() {
        let bb = BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap();
        assert_eq!(bb.x, 1.0);
        assert_eq!(bb.y, 2.0);
        assert_eq!(bb.z, 3.0);
        assert_eq!(bb.width, 4.0);
        assert_eq!(bb.height, 5.0);
        assert_eq!(bb.depth, 6.0);
    }

    #[test]
    fn test_bounding_box_3d_rejects_negative_width() {
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, -1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_bounding_box_3d_rejects_negative_height() {
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_bounding_box_3d_rejects_negative_depth() {
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_bounding_box_3d_center() {
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 4.0, 6.0, 8.0).unwrap();
        assert_eq!(bb.center(), (2.0, 3.0, 4.0));
    }

    #[test]
    fn test_bounding_box_3d_contains_point() {
        let bb = BoundingBox3D::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0).unwrap();
        // Inside
        assert!(bb.contains_point(2.0, 2.0, 2.0));
        // On edge / corner
        assert!(bb.contains_point(1.0, 1.0, 1.0));
        assert!(bb.contains_point(4.0, 4.0, 4.0));
        // Outside each axis
        assert!(!bb.contains_point(0.0, 2.0, 2.0));
        assert!(!bb.contains_point(5.0, 2.0, 2.0));
        assert!(!bb.contains_point(2.0, 0.0, 2.0));
        assert!(!bb.contains_point(2.0, 5.0, 2.0));
        assert!(!bb.contains_point(2.0, 2.0, 0.0));
        assert!(!bb.contains_point(2.0, 2.0, 5.0));
    }

    #[test]
    fn test_bounding_box_3d_intersects() {
        let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
        let b = BoundingBox3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0).unwrap();
        assert!(a.intersects(b));
        assert!(b.intersects(a));

        let c = BoundingBox3D::new(5.0, 5.0, 5.0, 1.0, 1.0, 1.0).unwrap();
        assert!(!a.intersects(c));
        assert!(!c.intersects(a));
    }

    #[test]
    fn test_bounding_box_3d_intersects_edge_touching() {
        let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
        // Touching on the X face
        let b = BoundingBox3D::new(2.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
        assert!(!a.intersects(b));
        // Touching on the Z face
        let c = BoundingBox3D::new(0.0, 0.0, 2.0, 2.0, 2.0, 2.0).unwrap();
        assert!(!a.intersects(c));
    }

    #[test]
    fn test_bounding_box_3d_union() {
        let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
        let b = BoundingBox3D::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0).unwrap();
        let u = a.union(b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, 0.0);
        assert_eq!(u.z, 0.0);
        assert_eq!(u.width, 4.0);
        assert_eq!(u.height, 4.0);
        assert_eq!(u.depth, 4.0);
    }

    #[test]
    fn test_bounding_box_3d_volume() {
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 3.0, 5.0, 7.0).unwrap();
        assert!((bb.volume() - 105.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_3d_is_empty() {
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 5.0, 5.0)
            .unwrap()
            .is_empty());
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 0.0, 5.0)
            .unwrap()
            .is_empty());
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 5.0, 0.0)
            .unwrap()
            .is_empty());
        assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap()
            .is_empty());
        assert!(!BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_bounding_box_3d_min_dist_sq_inside() {
        let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
        assert_eq!(bb.min_dist_sq(3.0, 3.0, 3.0), 0.0);
    }

    #[test]
    fn test_bounding_box_3d_min_dist_sq_each_axis() {
        let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
        // Left of box on X: (0,3,3) -> dx=2, distance^2=4
        assert!((bb.min_dist_sq(0.0, 3.0, 3.0) - 4.0).abs() < f32::EPSILON);
        // Below box on Y: (3,0,3) -> dy=2, distance^2=4
        assert!((bb.min_dist_sq(3.0, 0.0, 3.0) - 4.0).abs() < f32::EPSILON);
        // Behind box on Z: (3,3,0) -> dz=2, distance^2=4
        assert!((bb.min_dist_sq(3.0, 3.0, 0.0) - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_3d_min_dist_sq_diagonal() {
        let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
        // Point at origin (0,0,0) -> nearest corner (2,2,2), distance^2=12
        assert!((bb.min_dist_sq(0.0, 0.0, 0.0) - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_3d_min_dist_sq_beyond_box() {
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
        // Point to the right on X: (5, 1, 1) -> dx=3, distance^2=9
        assert!((bb.min_dist_sq(5.0, 1.0, 1.0) - 9.0).abs() < f32::EPSILON);
        // Point above on Y: (1, 5, 1) -> dy=3, distance^2=9
        assert!((bb.min_dist_sq(1.0, 5.0, 1.0) - 9.0).abs() < f32::EPSILON);
        // Point above on Z: (1, 1, 5) -> dz=3, distance^2=9
        assert!((bb.min_dist_sq(1.0, 1.0, 5.0) - 9.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 3D SpatialEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spatial_entry_3d() {
        let entry = SpatialEntry3D {
            bounds: BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap(),
            data: 42,
        };
        assert_eq!(entry.data, 42);
        assert_eq!(entry.bounds.x, 1.0);

        // Test Debug impl
        let debug_str = format!("{entry:?}");
        assert!(debug_str.contains("SpatialEntry3D"));

        // Test Clone impl
        let cloned = entry.clone();
        assert_eq!(cloned.data, entry.data);
        assert_eq!(cloned.bounds, entry.bounds);

        // Test PartialEq impl
        assert_eq!(entry, cloned);
    }

    // -----------------------------------------------------------------------
    // 3D SpatialIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spatial_index_3d_empty() {
        let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.iter().count(), 0);
    }

    #[test]
    fn test_spatial_index_3d_default() {
        let index: SpatialIndex3D<u32> = SpatialIndex3D::default();
        assert!(index.is_empty());
    }

    #[test]
    fn test_spatial_index_3d_insert_and_query_region() {
        let mut index = SpatialIndex3D::new();

        for i in 0..20 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            let z = (i % 3) as f32 * 10.0;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 20);

        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 12.0, 12.0, 12.0).unwrap();
        let results = index.query_region(region);
        assert!(!results.is_empty());

        for entry in &results {
            assert!(entry.bounds.intersects(region));
        }
    }

    #[test]
    fn test_spatial_index_3d_query_region_no_results() {
        let mut index = SpatialIndex3D::new();
        for i in 0..10 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let region = BoundingBox3D::new(100.0, 100.0, 100.0, 1.0, 1.0, 1.0).unwrap();
        let results = index.query_region(region);
        assert!(results.is_empty());
    }

    #[test]
    fn test_spatial_index_3d_query_nearest() {
        let mut index = SpatialIndex3D::new();

        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: "origin",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0).unwrap(),
            data: "far",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(2.0, 2.0, 2.0, 1.0, 1.0, 1.0).unwrap(),
            data: "near",
        });

        let results = index.query_nearest(0.0, 0.0, 0.0, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data, "origin");
        assert_eq!(results[1].data, "near");
    }

    #[test]
    fn test_spatial_index_3d_query_nearest_k_larger_than_n() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: 1,
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(5.0, 5.0, 5.0, 1.0, 1.0, 1.0).unwrap(),
            data: 2,
        });

        let results = index.query_nearest(0.0, 0.0, 0.0, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_3d_query_nearest_empty() {
        let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
        let results = index.query_nearest(0.0, 0.0, 0.0, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_spatial_index_3d_remove() {
        let mut index = SpatialIndex3D::new();
        let bb = BoundingBox3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0).unwrap();
        index.insert(SpatialEntry3D {
            bounds: bb,
            data: 42u32,
        });
        assert_eq!(index.len(), 1);

        index.remove(bb, |e| e.data == 42).unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let result = index.remove(bb, |e| e.data == 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_index_3d_remove_from_tree() {
        let mut index = SpatialIndex3D::new();
        for i in 0..20u32 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            let z = (i % 3) as f32 * 10.0;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 20);

        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 5.0, 5.0).unwrap();
        index.remove(bb, |e| e.data == 0).unwrap();
        assert_eq!(index.len(), 19);
    }

    #[test]
    fn test_spatial_index_3d_clear() {
        let mut index = SpatialIndex3D::new();
        for i in 0..10 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 10);
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_3d_iter() {
        let mut index = SpatialIndex3D::new();
        for i in 0..5 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all.len(), 5);
        let iter = index.iter();
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn test_spatial_index_3d_into_iter() {
        let mut index = SpatialIndex3D::new();
        for i in 0..3 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let all: Vec<_> = (&index).into_iter().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_spatial_index_3d_concurrent() {
        use std::sync::Arc;

        let index = Arc::new(parking_lot::Mutex::new(SpatialIndex3D::<u32>::new()));
        let mut handles = Vec::new();

        for thread_id in 0..4u32 {
            let idx = Arc::clone(&index);
            handles.push(std::thread::spawn(move || {
                for i in 0..100u32 {
                    let x = (thread_id * 100 + i) as f32;
                    idx.lock().insert(SpatialEntry3D {
                        bounds: BoundingBox3D::new(x, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                        data: thread_id * 100 + i,
                    });
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(index.lock().len(), 400);
    }

    #[test]
    fn test_spatial_index_3d_large_dataset() {
        let mut index = SpatialIndex3D::new();
        for i in 0..1_000u32 {
            let x = (i % 10) as f32;
            let y = ((i / 10) % 10) as f32;
            let z = (i / 100) as f32;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 0.5, 0.5, 0.5).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 1_000);

        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 5.5, 5.5, 5.5).unwrap();
        let results = index.query_region(region);
        assert!(!results.is_empty());
        for entry in &results {
            assert!(entry.bounds.intersects(region));
        }

        let nearest = index.query_nearest(5.0, 5.0, 5.0, 5);
        assert_eq!(nearest.len(), 5);
    }

    #[test]
    fn test_spatial_index_3d_split_preserves_entries() {
        let mut index = SpatialIndex3D::new();
        for i in 0..15u32 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 15);
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all.len(), 15);
    }

    #[test]
    fn test_spatial_index_3d_multiple_splits() {
        let mut index = SpatialIndex3D::new();
        for i in 0..100u32 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, i as f32, i as f32, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 100);
        assert_eq!(index.iter().count(), 100);
    }

    #[test]
    fn test_spatial_index_3d_internal_node_len() {
        let mut index = SpatialIndex3D::new();
        for i in 0..50u32 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 50);
        assert_eq!(index.root.len(), 50);
    }

    // -----------------------------------------------------------------------
    // 3D radius query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_within_radius_3d_point_inside() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
            data: "large",
        });
        let results = index.query_within_radius(50.0, 50.0, 50.0, 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, "large");
    }

    #[test]
    fn test_query_within_radius_3d_point_outside() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 5.0, 5.0, 5.0).unwrap(),
            data: "box",
        });
        // Point (20, 12, 12) -> nearest X edge at 15, distance = 5
        let results = index.query_within_radius(20.0, 12.0, 12.0, 5.0);
        assert_eq!(results.len(), 1);
        // r=4.9 should not include it
        let results = index.query_within_radius(20.0, 12.0, 12.0, 4.9);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_3d_zero() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
            data: "inside",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(200.0, 200.0, 200.0, 10.0, 10.0, 10.0).unwrap(),
            data: "outside",
        });
        let results = index.query_within_radius(50.0, 50.0, 50.0, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, "inside");
    }

    #[test]
    fn test_query_within_radius_3d_negative() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
            data: 1,
        });
        let results = index.query_within_radius(50.0, 50.0, 50.0, -1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_3d_sorted_nearest_first() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(100.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
            data: "far",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(20.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
            data: "near",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
            data: "closest",
        });
        let results = index.query_within_radius(0.0, 0.0, 0.0, 200.0);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data, "closest");
        assert_eq!(results[1].data, "near");
        assert_eq!(results[2].data, "far");
    }

    #[test]
    fn test_query_within_radius_3d_empty_index() {
        let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
        let results = index.query_within_radius(0.0, 0.0, 0.0, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_3d_with_distances() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
            data: "inside",
        });
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(110.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
            data: "outside",
        });
        let results = index.query_within_radius_with_distances(50.0, 50.0, 50.0, 200.0);
        assert_eq!(results.len(), 2);
        // Inside element, distance 0
        assert_eq!(results[0].0.data, "inside");
        assert!((results[0].1 - 0.0).abs() < f32::EPSILON);
        // Outside element: dx=60, dy=40, dz=40
        assert_eq!(results[1].0.data, "outside");
        let expected = (60.0_f32.powi(2) + 40.0_f32.powi(2) + 40.0_f32.powi(2)).sqrt();
        assert!((results[1].1 - expected).abs() < 0.1);
    }

    #[test]
    fn test_query_within_radius_3d_with_distances_negative() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
            data: 1,
        });
        let results = index.query_within_radius_with_distances(0.0, 0.0, 0.0, -1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_within_radius_3d_large_dataset_brute_force() {
        let mut index = SpatialIndex3D::new();
        for i in 0..1_000u32 {
            let x = (i % 10) as f32;
            let y = ((i / 10) % 10) as f32;
            let z = (i / 100) as f32;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 0.5, 0.5, 0.5).unwrap(),
                data: i,
            });
        }

        let cx = 5.0_f32;
        let cy = 5.0_f32;
        let cz = 5.0_f32;
        let radius = 3.0_f32;

        let results = index.query_within_radius(cx, cy, cz, radius);

        // Brute-force: check every entry
        let mut expected_count = 0;
        for entry in index.iter() {
            if entry.bounds.min_dist_sq(cx, cy, cz) <= radius * radius {
                expected_count += 1;
            }
        }
        assert_eq!(results.len(), expected_count);

        for entry in &results {
            let dist = entry.bounds.min_dist_sq(cx, cy, cz).sqrt();
            assert!(dist <= radius + f32::EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // 3D Node bounds tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_3d_bounds_empty_leaf() {
        let node: Node3D<u32> = Node3D::Leaf {
            entries: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    #[test]
    fn test_node_3d_bounds_empty_internal() {
        let node: Node3D<u32> = Node3D::Internal {
            children: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    // -----------------------------------------------------------------------
    // 3D serde roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_3d_serde_roundtrip() {
        let bb = BoundingBox3D::new(1.5, 2.5, 3.5, 10.0, 20.0, 30.0).unwrap();
        let bytes = bitcode::serialize(&bb).unwrap();
        let restored: BoundingBox3D = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(bb, restored);
    }

    #[test]
    fn test_bounding_box_3d_serde_zero_dimensions() {
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        let bytes = bitcode::serialize(&bb).unwrap();
        let restored: BoundingBox3D = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(bb, restored);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_spatial_entry_3d_serde_roundtrip() {
        let entry = SpatialEntry3D {
            bounds: BoundingBox3D::new(3.0, 4.0, 5.0, 6.0, 7.0, 8.0).unwrap(),
            data: String::from("hello"),
        };
        let bytes = bitcode::serialize(&entry).unwrap();
        let restored: SpatialEntry3D<String> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(entry, restored);
    }

    #[test]
    fn test_spatial_index_3d_serde_empty_roundtrip() {
        let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_spatial_index_3d_serde_single_entry_roundtrip() {
        let mut index = SpatialIndex3D::new();
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap(),
            data: 42u32,
        });
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 1);
        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap();
        let results = restored.query_region(region);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, 42);
    }

    #[test]
    fn test_spatial_index_3d_serde_tree_with_splits_roundtrip() {
        let mut index = SpatialIndex3D::new();
        for i in 0..50u32 {
            let x = (i % 10) as f32 * 5.0;
            let y = (i / 10) as f32 * 5.0;
            let z = (i % 5) as f32 * 5.0;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 2.0, 2.0, 2.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 50);

        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 50);

        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 50.0, 50.0, 50.0).unwrap();
        let orig_results = index.query_region(region);
        let rest_results = restored.query_region(region);
        assert_eq!(orig_results.len(), rest_results.len());

        let orig_nearest = index.query_nearest(25.0, 12.0, 10.0, 5);
        let rest_nearest = restored.query_nearest(25.0, 12.0, 10.0, 5);
        let mut orig_data: Vec<_> = orig_nearest.iter().map(|e| e.data).collect();
        let mut rest_data: Vec<_> = rest_nearest.iter().map(|e| e.data).collect();
        orig_data.sort_unstable();
        rest_data.sort_unstable();
        assert_eq!(orig_data, rest_data);
    }

    #[test]
    fn test_spatial_index_3d_serde_string_data_roundtrip() {
        let mut index = SpatialIndex3D::new();
        for i in 0..10 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: format!("item_{i}"),
            });
        }
        let bytes = bitcode::serialize(&index).unwrap();
        let restored: SpatialIndex3D<String> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 10);

        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap();
        let mut orig_data: Vec<_> = index
            .query_region(region)
            .iter()
            .map(|e| &e.data)
            .cloned()
            .collect();
        let mut rest_data: Vec<_> = restored
            .query_region(region)
            .iter()
            .map(|e| &e.data)
            .cloned()
            .collect();
        orig_data.sort();
        rest_data.sort();
        assert_eq!(orig_data, rest_data);
    }

    #[test]
    fn test_spatial_index_3d_serde_mutate_after_restore() {
        let mut index = SpatialIndex3D::new();
        for i in 0..5u32 {
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        let bytes = bitcode::serialize(&index).unwrap();
        let mut restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 5);

        restored.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0).unwrap(),
            data: 99,
        });
        assert_eq!(restored.len(), 6);

        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap();
        restored.remove(bb, |e| e.data == 0).unwrap();
        assert_eq!(restored.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Additional coverage: exercise failure paths on Internal nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_spatial_index_remove_not_found_on_internal() {
        let mut index = SpatialIndex::new();
        for i in 0..20u32 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        // Region overlaps children but predicate never matches
        let region = BoundingBox::new(0.0, 0.0, 200.0, 200.0).unwrap();
        let result = index.remove(region, |e| e.data == 999);
        assert!(result.is_err());
        assert_eq!(index.len(), 20);
    }

    #[test]
    fn test_spatial_index_3d_remove_not_found_on_internal() {
        let mut index = SpatialIndex3D::new();
        for i in 0..20u32 {
            let x = (i % 5) as f32 * 10.0;
            let y = (i / 5) as f32 * 10.0;
            let z = (i % 3) as f32 * 10.0;
            index.insert(SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
                data: i,
            });
        }
        // Region overlaps children but predicate never matches
        let region = BoundingBox3D::new(0.0, 0.0, 0.0, 200.0, 200.0, 200.0).unwrap();
        let result = index.remove(region, |e| e.data == 999);
        assert!(result.is_err());
        assert_eq!(index.len(), 20);
    }

    // -----------------------------------------------------------------------
    // Serde rejection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_3d_serde_rejects_negative_dims() {
        // Serialize a raw tuple with negative width
        let bytes =
            bitcode::serialize(&(1.0_f32, 2.0_f32, 3.0_f32, -1.0_f32, 5.0_f32, 6.0_f32)).unwrap();
        let result: Result<BoundingBox3D, _> = bitcode::deserialize(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_index_3d_serde_rejects_unknown_version() {
        #[derive(Serialize)]
        struct FakeDto {
            version: u8,
            entries: Vec<SpatialEntry3D<u32>>,
        }
        let fake = FakeDto {
            version: 99,
            entries: vec![],
        };
        let bytes = bitcode::serialize(&fake).unwrap();
        let result: Result<SpatialIndex3D<u32>, _> = bitcode::deserialize(&bytes);
        assert!(result.is_err());
    }
}
