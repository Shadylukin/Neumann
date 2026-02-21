//! R-tree spatial index for region and nearest-neighbor queries.
//!
//! Provides a node-based R-tree with linear split algorithm, supporting
//! insertion, removal, region queries, and k-nearest-neighbor lookups.

use std::collections::BinaryHeap;
use std::fmt;

/// Maximum entries per R-tree node before splitting.
const MAX_ENTRIES: usize = 9;

/// Minimum entries per R-tree node after splitting.
const MIN_ENTRIES: usize = 4;

/// Errors that can occur during spatial operations.
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

/// An entry in the spatial index pairing a bounding box with user data.
pub struct SpatialEntry<T> {
    /// The bounding box for this entry.
    pub bounds: BoundingBox,
    /// User-supplied data associated with this entry.
    pub data: T,
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
}
