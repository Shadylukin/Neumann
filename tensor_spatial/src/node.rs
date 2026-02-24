//! Internal R-tree node and free functions for splitting and subtree selection.

use std::collections::BinaryHeap;

use crate::bbox::{BoundingBoxN, SpatialEntryN};
use crate::{MAX_ENTRIES, MIN_ENTRIES};

/// Internal R-tree node, generic over dimension `D` and data type `T`.
pub enum NodeN<const D: usize, T> {
    /// A leaf node containing spatial entries.
    Leaf {
        /// Entries stored in this leaf.
        entries: Vec<SpatialEntryN<D, T>>,
    },
    /// An internal node containing child nodes with their bounding boxes.
    Internal {
        /// Child bounding boxes paired with their subtrees.
        children: Vec<(BoundingBoxN<D>, Self)>,
    },
}

impl<const D: usize, T> NodeN<D, T> {
    /// Returns the bounding box enclosing all entries or children in this node.
    pub fn bounds(&self) -> Option<BoundingBoxN<D>> {
        match self {
            Self::Leaf { entries } => {
                let mut iter = entries.iter().map(|e| e.bounds);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBoxN::union))
            },
            Self::Internal { children } => {
                let mut iter = children.iter().map(|(b, _)| *b);
                let first = iter.next()?;
                Some(iter.fold(first, BoundingBoxN::union))
            },
        }
    }

    /// Returns the number of data entries stored beneath this node.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        match self {
            Self::Leaf { entries } => entries.len(),
            Self::Internal { children } => children.iter().map(|(_, c)| c.len()).sum(),
        }
    }

    /// Collects all entries that intersect `region`.
    pub fn query_region<'a>(
        &'a self,
        region: BoundingBoxN<D>,
        results: &mut Vec<&'a SpatialEntryN<D, T>>,
    ) {
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
    pub fn query_nearest_heap<'a>(
        &'a self,
        point: &[f32; D],
        heap: &mut BinaryHeap<NearestCandidateN<'a, D, T>>,
        k: usize,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq_nd(point);
                    if heap.len() < k {
                        heap.push(NearestCandidateN { dist_sq, entry });
                    } else if let Some(worst) = heap.peek() {
                        if dist_sq < worst.dist_sq {
                            heap.pop();
                            heap.push(NearestCandidateN { dist_sq, entry });
                        }
                    }
                }
            },
            Self::Internal { children } => {
                let mut child_dists: Vec<(f32, usize)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, (b, _))| (b.min_dist_sq_nd(point), i))
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
                    children[idx].1.query_nearest_heap(point, heap, k);
                }
            },
        }
    }

    /// Collects entries within a squared radius from a point.
    pub fn query_within_radius<'a>(
        &'a self,
        point: &[f32; D],
        r_sq: f32,
        results: &mut Vec<(&'a SpatialEntryN<D, T>, f32)>,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.min_dist_sq_nd(point);
                    if dist_sq <= r_sq {
                        results.push((entry, dist_sq));
                    }
                }
            },
            Self::Internal { children } => {
                for (child_bounds, child) in children {
                    if child_bounds.min_dist_sq_nd(point) <= r_sq {
                        child.query_within_radius(point, r_sq, results);
                    }
                }
            },
        }
    }

    /// Collects references to all entries in this subtree.
    pub fn collect_all<'a>(&'a self, out: &mut Vec<&'a SpatialEntryN<D, T>>) {
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
    pub fn insert(&mut self, entry: SpatialEntryN<D, T>) -> Option<(BoundingBoxN<D>, Self)> {
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
                let target = choose_subtree(children, entry.bounds);
                let split = children[target].1.insert(entry);
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
    pub fn remove<F>(&mut self, region: BoundingBoxN<D>, pred: &F) -> bool
    where
        F: Fn(&SpatialEntryN<D, T>) -> bool,
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

// ---------------------------------------------------------------------------
// Nearest-neighbor candidate (max-heap by distance)
// ---------------------------------------------------------------------------

/// Candidate entry for nearest-neighbor search (max-heap by distance).
pub struct NearestCandidateN<'a, const D: usize, T> {
    /// Squared distance from the query point to this entry's bounding box edge.
    pub dist_sq: f32,
    /// Reference to the spatial entry.
    pub entry: &'a SpatialEntryN<D, T>,
}

impl<const D: usize, T> PartialEq for NearestCandidateN<'_, D, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl<const D: usize, T> Eq for NearestCandidateN<'_, D, T> {}

impl<const D: usize, T> PartialOrd for NearestCandidateN<'_, D, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const D: usize, T> Ord for NearestCandidateN<'_, D, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance comes first so we can pop the worst.
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Chooses the child whose bounding box needs the least enlargement to
/// include `entry_bounds`.
pub fn choose_subtree<const D: usize, T>(
    children: &[(BoundingBoxN<D>, NodeN<D, T>)],
    entry_bounds: BoundingBoxN<D>,
) -> usize {
    children
        .iter()
        .enumerate()
        .min_by(|(_, (a_bb, _)), (_, (b_bb, _))| {
            let a_enlarge = a_bb.union(entry_bounds).measure() - a_bb.measure();
            let b_enlarge = b_bb.union(entry_bounds).measure() - b_bb.measure();
            a_enlarge
                .partial_cmp(&b_enlarge)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i)
}

/// Linear split for leaf nodes: picks the two most separated entries as seeds,
/// then distributes the rest by minimum enlargement.
pub fn split_leaf<const D: usize, T>(
    entries: &mut Vec<SpatialEntryN<D, T>>,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
    let (seed1, seed2) = pick_seeds_leaf(entries);
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
        let enlarge1 = bb1.union(e.bounds).measure() - bb1.measure();
        let enlarge2 = bb2.union(e.bounds).measure() - bb2.measure();
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
        .reduce(BoundingBoxN::union)
        .expect("group2 is not empty");
    (sibling_bounds, NodeN::Leaf { entries: group2 })
}

/// Linear split for internal nodes.
pub fn split_internal<const D: usize, T>(
    children: &mut Vec<(BoundingBoxN<D>, NodeN<D, T>)>,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
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
        let enlarge1 = bb1.union(c.0).measure() - bb1.measure();
        let enlarge2 = bb2.union(c.0).measure() - bb2.measure();
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
        .reduce(BoundingBoxN::union)
        .expect("group2 is not empty");

    (sibling_bb, NodeN::Internal { children: group2 })
}

/// Picks two seed entries in a leaf with the largest separation along any axis.
fn pick_seeds_leaf<const D: usize, T>(entries: &[SpatialEntryN<D, T>]) -> (usize, usize) {
    if entries.len() < 2 {
        return (0, entries.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, ei) in entries.iter().enumerate() {
        for (j, ej) in entries.iter().enumerate().skip(i + 1) {
            let combined = ei.bounds.union(ej.bounds).measure();
            let waste = combined - ei.bounds.measure() - ej.bounds.measure();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}

/// Picks two seed children in an internal node with the largest separation.
fn pick_seeds_internal<const D: usize, T>(
    children: &[(BoundingBoxN<D>, NodeN<D, T>)],
) -> (usize, usize) {
    if children.len() < 2 {
        return (0, children.len().saturating_sub(1));
    }
    let mut best = (0, 1);
    let mut best_waste = f32::NEG_INFINITY;

    for (i, (bi, _)) in children.iter().enumerate() {
        for (j, (bj, _)) in children.iter().enumerate().skip(i + 1) {
            let combined = bi.union(*bj).measure();
            let waste = combined - bi.measure() - bj.measure();
            if waste > best_waste {
                best_waste = waste;
                best = (i, j);
            }
        }
    }
    best
}
