//! Internal R-tree node and free functions for splitting and subtree selection.

use std::collections::BinaryHeap;

use crate::bbox::{BoundingBoxN, SpatialEntryN};
use crate::{SpatialConfig, SplitStrategy};

/// Result of inserting into a node.
pub enum InsertResult<const D: usize, T> {
    /// Insertion completed without overflow.
    Ok,
    /// Node split; the sibling's bounding box and node are returned.
    Split(BoundingBoxN<D>, NodeN<D, T>),
    /// Forced reinsertion: extracted entries to reinsert at the root.
    Reinsert(Vec<SpatialEntryN<D, T>>),
}

/// Internal R-tree node, generic over dimension `D` and data type `T`.
#[derive(Clone)]
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

    /// Pushes candidate entries onto the nearest-neighbor heap using
    /// centroid distance for leaf scoring.
    ///
    /// Internal-node pruning still uses `min_dist_sq_nd` (a valid lower bound
    /// on centroid distance for any entry inside the subtree).
    pub fn query_nearest_by_centroid_heap<'a>(
        &'a self,
        point: &[f32; D],
        heap: &mut BinaryHeap<NearestCandidateN<'a, D, T>>,
        k: usize,
    ) {
        match self {
            Self::Leaf { entries } => {
                for entry in entries {
                    let dist_sq = entry.bounds.center_dist_sq_nd(point);
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
                    children[idx]
                        .1
                        .query_nearest_by_centroid_heap(point, heap, k);
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

    /// Inserts an entry, returning the overflow action.
    ///
    /// `allow_reinsert` controls whether R\*-tree forced reinsertion is
    /// permitted for leaf overflow (set to `false` during reinsertion to
    /// prevent infinite loops).
    pub(crate) fn insert_rstar(
        &mut self,
        entry: SpatialEntryN<D, T>,
        config: SpatialConfig,
        allow_reinsert: bool,
    ) -> InsertResult<D, T> {
        match self {
            Self::Leaf { entries } => {
                entries.push(entry);
                if entries.len() > config.max_entries() {
                    match config.split_strategy() {
                        SplitStrategy::RStar if allow_reinsert => {
                            InsertResult::Reinsert(rstar_reinsert_leaf(entries, config))
                        },
                        SplitStrategy::RStar => {
                            let (bb, sib) = rstar_split_leaf(entries, config);
                            InsertResult::Split(bb, sib)
                        },
                        SplitStrategy::Linear => {
                            let (bb, sib) = split_leaf(entries, config);
                            InsertResult::Split(bb, sib)
                        },
                    }
                } else {
                    InsertResult::Ok
                }
            },
            Self::Internal { children } => {
                let target = match config.split_strategy() {
                    SplitStrategy::RStar => rstar_choose_subtree(children, entry.bounds),
                    SplitStrategy::Linear => choose_subtree(children, entry.bounds),
                };
                let result = children[target]
                    .1
                    .insert_rstar(entry, config, allow_reinsert);
                if let Some(b) = children[target].1.bounds() {
                    children[target].0 = b;
                }
                match result {
                    InsertResult::Ok => InsertResult::Ok,
                    InsertResult::Split(sb, sn) => {
                        children.push((sb, sn));
                        if children.len() > config.max_entries() {
                            let (bb, sib) = match config.split_strategy() {
                                SplitStrategy::RStar => rstar_split_internal(children, config),
                                SplitStrategy::Linear => split_internal(children, config),
                            };
                            InsertResult::Split(bb, sib)
                        } else {
                            InsertResult::Ok
                        }
                    },
                    // Reinsert bubbles up unchanged from a leaf.
                    InsertResult::Reinsert(entries) => InsertResult::Reinsert(entries),
                }
            },
        }
    }

    /// Removes the first entry matching the predicate.
    ///
    /// Returns `(found, orphans)` where `found` indicates whether the entry
    /// was located and removed, and `orphans` contains entries displaced by
    /// underflow rebalancing that must be reinserted at the root level.
    pub fn remove<F>(
        &mut self,
        region: BoundingBoxN<D>,
        pred: &F,
        config: SpatialConfig,
    ) -> (bool, Vec<SpatialEntryN<D, T>>)
    where
        F: Fn(&SpatialEntryN<D, T>) -> bool,
    {
        match self {
            Self::Leaf { entries } => {
                let Some(pos) = entries
                    .iter()
                    .position(|e| e.bounds.intersects(region) && pred(e))
                else {
                    return (false, Vec::new());
                };
                entries.remove(pos);
                if entries.len() < config.min_entries() {
                    let orphans = std::mem::take(entries);
                    (true, orphans)
                } else {
                    (true, Vec::new())
                }
            },
            Self::Internal { children } => {
                let mut found_idx = None;
                let mut orphans = Vec::new();
                for (i, (child_bounds, child)) in children.iter_mut().enumerate() {
                    if !child_bounds.intersects(region) {
                        continue;
                    }
                    let (found, child_orphans) = child.remove(region, pred, config);
                    if !found {
                        continue;
                    }
                    orphans = child_orphans;
                    found_idx = Some(i);
                    break;
                }
                let Some(idx) = found_idx else {
                    return (false, Vec::new());
                };
                if children[idx].1.is_node_empty() {
                    children.swap_remove(idx);
                } else if let Some(b) = children[idx].1.bounds() {
                    children[idx].0 = b;
                }
                if children.len() < config.min_entries() && !children.is_empty() {
                    for (_, child) in children.drain(..) {
                        child.collect_all_into(&mut orphans);
                    }
                }
                (true, orphans)
            },
        }
    }

    /// Returns `true` if this node contains no entries or children.
    fn is_node_empty(&self) -> bool {
        match self {
            Self::Leaf { entries } => entries.is_empty(),
            Self::Internal { children } => children.is_empty(),
        }
    }

    /// Consumes this node's subtree, appending all leaf entries to `out`.
    fn collect_all_into(self, out: &mut Vec<SpatialEntryN<D, T>>) {
        match self {
            Self::Leaf { entries } => out.extend(entries),
            Self::Internal { children } => {
                for (_, child) in children {
                    child.collect_all_into(out);
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Nearest-neighbor candidate (max-heap by distance)
// ---------------------------------------------------------------------------

/// Candidate entry for nearest-neighbor search (max-heap by distance).
pub struct NearestCandidateN<'a, const D: usize, T> {
    /// Squared distance from the query point to this entry.
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
    config: SpatialConfig,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
    let min_entries = config.min_entries();
    let (seed1, seed2) = pick_seeds_leaf(entries);
    let s2 = entries.swap_remove(seed2);
    let s1_idx = if seed1 == entries.len() { seed2 } else { seed1 };
    let s1 = entries.swap_remove(s1_idx);

    let mut group1 = vec![s1];
    let mut group2 = vec![s2];
    let mut bb1 = group1[0].bounds;
    let mut bb2 = group2[0].bounds;

    while !entries.is_empty() {
        if group1.len() + entries.len() == min_entries {
            group1.append(entries);
            break;
        }
        if group2.len() + entries.len() == min_entries {
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
    config: SpatialConfig,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
    let min_entries = config.min_entries();
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
        if group1.len() + children.len() == min_entries {
            group1.append(children);
            break;
        }
        if group2.len() + children.len() == min_entries {
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

// ---------------------------------------------------------------------------
// R*-tree split, subtree selection, and forced reinsertion
// ---------------------------------------------------------------------------

/// R\*-tree subtree selection: for leaf-level insertion minimizes overlap
/// increase; for non-leaf minimizes area enlargement.
pub fn rstar_choose_subtree<const D: usize, T>(
    children: &[(BoundingBoxN<D>, NodeN<D, T>)],
    entry_bounds: BoundingBoxN<D>,
) -> usize {
    debug_assert!(!children.is_empty(), "children must not be empty");
    if children.is_empty() {
        return 0;
    }

    let is_leaf_level = matches!(children[0].1, NodeN::Leaf { .. });

    if is_leaf_level {
        // Minimize overlap increase with siblings; ties by enlargement, then area.
        children
            .iter()
            .enumerate()
            .min_by(|(i, (bb_i, _)), (j, (bb_j, _))| {
                let ov_inc_i = overlap_increase(children, *i, entry_bounds);
                let ov_inc_j = overlap_increase(children, *j, entry_bounds);
                ov_inc_i
                    .partial_cmp(&ov_inc_j)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let enlarge_i = bb_i.union(entry_bounds).measure() - bb_i.measure();
                        let enlarge_j = bb_j.union(entry_bounds).measure() - bb_j.measure();
                        enlarge_i
                            .partial_cmp(&enlarge_j)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        bb_i.measure()
                            .partial_cmp(&bb_j.measure())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .map_or(0, |(i, _)| i)
    } else {
        // Non-leaf: least enlargement, ties by smallest area.
        choose_subtree(children, entry_bounds)
    }
}

/// Computes the overlap increase for child `idx` when `entry_bounds` is added.
fn overlap_increase<const D: usize, T>(
    children: &[(BoundingBoxN<D>, NodeN<D, T>)],
    idx: usize,
    entry_bounds: BoundingBoxN<D>,
) -> f32 {
    let original = children[idx].0;
    let enlarged = original.union(entry_bounds);
    let mut original_overlap = 0.0_f32;
    let mut enlarged_overlap = 0.0_f32;
    for (j, (bb_j, _)) in children.iter().enumerate() {
        if j == idx {
            continue;
        }
        original_overlap += original.overlap_volume(*bb_j);
        enlarged_overlap += enlarged.overlap_volume(*bb_j);
    }
    enlarged_overlap - original_overlap
}

/// Chooses the best split axis (CSA) and split index (CSI) for an R\*-tree
/// split. Works on a mutable slice of bounding boxes with a key function
/// that extracts the `BoundingBoxN` from each element.
///
/// Returns `(best_axis, best_k, best_is_upper_ordering)`.
fn rstar_csa_csi<const D: usize, E>(
    items: &mut [E],
    bbox_fn: impl Fn(&E) -> BoundingBoxN<D>,
    min_e: usize,
) -> (usize, usize, bool) {
    // CSA: choose axis with minimum total margin.
    let mut best_axis = 0;
    let mut best_margin = f32::INFINITY;

    for axis in 0..D {
        let mut margin_sum = 0.0_f32;
        for is_upper in [false, true] {
            sort_by_axis(items, &bbox_fn, axis, is_upper);
            for k in min_e..=(items.len() - min_e) {
                let bb1 = items[..k]
                    .iter()
                    .map(&bbox_fn)
                    .reduce(BoundingBoxN::union)
                    .expect("non-empty");
                let bb2 = items[k..]
                    .iter()
                    .map(&bbox_fn)
                    .reduce(BoundingBoxN::union)
                    .expect("non-empty");
                margin_sum += bb1.margin() + bb2.margin();
            }
        }
        if margin_sum < best_margin {
            best_margin = margin_sum;
            best_axis = axis;
        }
    }

    // CSI: on the chosen axis, pick (ordering, k) minimizing overlap then area.
    let mut best_overlap = f32::INFINITY;
    let mut best_area = f32::INFINITY;
    let mut best_k = min_e;
    let mut best_is_upper = false;

    for is_upper in [false, true] {
        sort_by_axis(items, &bbox_fn, best_axis, is_upper);
        let valid_end = items.len() - min_e;
        for k in min_e..=valid_end {
            let bb1 = items[..k]
                .iter()
                .map(&bbox_fn)
                .reduce(BoundingBoxN::union)
                .expect("non-empty");
            let bb2 = items[k..]
                .iter()
                .map(&bbox_fn)
                .reduce(BoundingBoxN::union)
                .expect("non-empty");
            let ov = bb1.overlap_volume(bb2);
            let area = bb1.measure() + bb2.measure();
            if ov < best_overlap || ((ov - best_overlap).abs() < f32::EPSILON && area < best_area) {
                best_overlap = ov;
                best_area = area;
                best_k = k;
                best_is_upper = is_upper;
            }
        }
    }

    (best_axis, best_k, best_is_upper)
}

/// Sorts items by the lower or upper bound of their bounding box on `axis`.
fn sort_by_axis<const D: usize, E>(
    items: &mut [E],
    bbox_fn: &impl Fn(&E) -> BoundingBoxN<D>,
    axis: usize,
    upper: bool,
) {
    items.sort_by(|a, b| {
        let bb_a = bbox_fn(a);
        let bb_b = bbox_fn(b);
        let val_a = if upper {
            bb_a.origin[axis] + bb_a.extent[axis]
        } else {
            bb_a.origin[axis]
        };
        let val_b = if upper {
            bb_b.origin[axis] + bb_b.extent[axis]
        } else {
            bb_b.origin[axis]
        };
        val_a
            .partial_cmp(&val_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// R\*-tree split for leaf nodes using axis-based CSA/CSI.
///
/// Chooses the split axis with minimum total margin, then picks the split
/// index minimizing overlap (ties broken by area).
pub fn rstar_split_leaf<const D: usize, T>(
    entries: &mut Vec<SpatialEntryN<D, T>>,
    config: SpatialConfig,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
    let (best_axis, best_k, best_is_upper) =
        rstar_csa_csi(entries, |e| e.bounds, config.min_entries());
    sort_by_axis(
        entries,
        &|e: &SpatialEntryN<D, T>| e.bounds,
        best_axis,
        best_is_upper,
    );

    let group2: Vec<SpatialEntryN<D, T>> = entries.drain(best_k..).collect();
    let sibling_bounds = group2
        .iter()
        .map(|e| e.bounds)
        .reduce(BoundingBoxN::union)
        .expect("group2 is not empty");
    (sibling_bounds, NodeN::Leaf { entries: group2 })
}

/// R\*-tree split for internal nodes using axis-based CSA/CSI.
pub fn rstar_split_internal<const D: usize, T>(
    children: &mut Vec<(BoundingBoxN<D>, NodeN<D, T>)>,
    config: SpatialConfig,
) -> (BoundingBoxN<D>, NodeN<D, T>) {
    let (best_axis, best_k, best_is_upper) =
        rstar_csa_csi(children, |(b, _)| *b, config.min_entries());
    sort_by_axis(
        children,
        &|(b, _): &(BoundingBoxN<D>, NodeN<D, T>)| *b,
        best_axis,
        best_is_upper,
    );

    let group2: Vec<(BoundingBoxN<D>, NodeN<D, T>)> = children.drain(best_k..).collect();
    let sibling_bb = group2
        .iter()
        .map(|(b, _)| *b)
        .reduce(BoundingBoxN::union)
        .expect("group2 is not empty");
    (sibling_bb, NodeN::Internal { children: group2 })
}

/// Forced reinsertion for R\*-tree leaf overflow.
///
/// Removes ~30% of entries farthest from the node center and returns them for
/// root-level reinsertion.
pub fn rstar_reinsert_leaf<const D: usize, T>(
    entries: &mut Vec<SpatialEntryN<D, T>>,
    config: SpatialConfig,
) -> Vec<SpatialEntryN<D, T>> {
    // Compute the center of the bounding box of all entries.
    let all_bb = entries
        .iter()
        .map(|e| e.bounds)
        .reduce(BoundingBoxN::union)
        .expect("entries is non-empty");
    let center = all_bb.center_nd();

    // Sort by distance from center (descending) so farthest are at end.
    entries.sort_by(|a, b| {
        let da = dist_sq_to_center(&a.bounds.center_nd(), &center);
        let db = dist_sq_to_center(&b.bounds.center_nd(), &center);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Remove ~30% of entries (at least 1).
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    let remove_count = ((entries.len() as f64 * 0.3).ceil() as usize).max(1);
    let remove_count = remove_count.min(entries.len() - config.min_entries());

    // Take from the front (farthest from center).
    entries.drain(..remove_count).collect()
}

/// Squared Euclidean distance between two points.
fn dist_sq_to_center<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..D {
        let d = a[i] - b[i];
        sum = d.mul_add(d, sum);
    }
    sum
}

// ---------------------------------------------------------------------------
// STR (Sort-Tile-Recursive) bulk loading
// ---------------------------------------------------------------------------

/// Builds an R-tree from a pre-collected set of entries using the
/// Sort-Tile-Recursive (STR) algorithm.
///
/// The resulting tree has tighter bounding boxes and less overlap than
/// one-at-a-time top-down insertion, yielding better query performance
/// for static datasets.
pub fn str_build_nodes<const D: usize, T>(
    mut entries: Vec<SpatialEntryN<D, T>>,
    config: SpatialConfig,
) -> NodeN<D, T> {
    if entries.is_empty() {
        return NodeN::Leaf {
            entries: Vec::new(),
        };
    }
    let max = config.max_entries();
    if entries.len() <= max {
        return NodeN::Leaf { entries };
    }

    let mut leaves = str_partition::<D, T>(&mut entries, 0, config);
    while leaves.len() > max {
        leaves = pack_internal_level(leaves, config);
    }
    if leaves.len() == 1 {
        leaves.pop().expect("single element").1
    } else {
        NodeN::Internal { children: leaves }
    }
}

/// Recursively partitions entries into leaf-sized groups using the STR
/// tiling strategy, sorting along successive dimensions.
fn str_partition<const D: usize, T>(
    entries: &mut Vec<SpatialEntryN<D, T>>,
    dim: usize,
    config: SpatialConfig,
) -> Vec<(BoundingBoxN<D>, NodeN<D, T>)> {
    let max = config.max_entries();
    let remaining_dims = D - dim;

    if remaining_dims <= 1 {
        // Last dimension: chunk directly into leaves.
        let mut result = Vec::new();
        entries.sort_by(|a, b| {
            a.bounds.center_nd()[dim]
                .partial_cmp(&b.bounds.center_nd()[dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        while !entries.is_empty() {
            let take = max.min(entries.len());
            let chunk: Vec<SpatialEntryN<D, T>> = entries.drain(..take).collect();
            let bb = chunk
                .iter()
                .map(|e| e.bounds)
                .reduce(BoundingBoxN::union)
                .expect("chunk is non-empty");
            result.push((bb, NodeN::Leaf { entries: chunk }));
        }
        return result;
    }

    // Sort along current dimension.
    entries.sort_by(|a, b| {
        a.bounds.center_nd()[dim]
            .partial_cmp(&b.bounds.center_nd()[dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    #[allow(clippy::cast_precision_loss)]
    let num_leaves = (entries.len() as f64 / max as f64).ceil();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let num_slabs = num_leaves.powf(1.0 / remaining_dims as f64).ceil() as usize;
    let slab_size = (entries.len() + num_slabs - 1) / num_slabs.max(1);

    let mut result = Vec::new();
    while !entries.is_empty() {
        let take = slab_size.min(entries.len());
        let mut slab: Vec<SpatialEntryN<D, T>> = entries.drain(..take).collect();
        result.extend(str_partition::<D, T>(&mut slab, dim + 1, config));
    }
    result
}

/// Groups child nodes into internal nodes of at most `max_entries` children.
///
/// STR bulk packing does NOT enforce `min_entries` for the last chunk at each
/// level. STR packs all entries upfront, so the final chunk may have fewer
/// than `min_entries`. This differs from incremental insertion where underflow
/// triggers `condense_tree`.
fn pack_internal_level<const D: usize, T>(
    children: Vec<(BoundingBoxN<D>, NodeN<D, T>)>,
    config: SpatialConfig,
) -> Vec<(BoundingBoxN<D>, NodeN<D, T>)> {
    let max = config.max_entries();
    let mut result = Vec::new();
    let mut iter = children.into_iter().peekable();
    while iter.peek().is_some() {
        let group: Vec<(BoundingBoxN<D>, NodeN<D, T>)> = iter.by_ref().take(max).collect();
        let bb = group
            .iter()
            .map(|(b, _)| *b)
            .reduce(BoundingBoxN::union)
            .expect("group is non-empty");
        result.push((bb, NodeN::Internal { children: group }));
    }
    result
}
