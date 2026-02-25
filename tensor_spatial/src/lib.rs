//! R-tree spatial index for region and nearest-neighbor queries.
//!
//! Provides a generic N-dimensional R-tree with linear and R\*-tree split
//! algorithms, supporting insertion, removal, region queries, radius queries,
//! and k-nearest-neighbor lookups. The core types are parameterized by
//! `const D: usize` for the spatial dimension and `T` for user data.
//!
//! Type aliases preserve the existing 2D and 3D API:
//! - [`BoundingBox`] = `BoundingBoxN<2>`
//! - [`BoundingBox3D`] = `BoundingBoxN<3>`
//! - [`SpatialEntry`] = `SpatialEntryN<2, T>`
//! - [`SpatialEntry3D`] = `SpatialEntryN<3, T>`
//! - [`SpatialIndex`] = `SpatialIndexN<2, T>`
//! - [`SpatialIndex3D`] = `SpatialIndexN<3, T>`
//!
//! ## Concurrency
//!
//! `SpatialIndexN` has no internal synchronization. For concurrent access,
//! wrap it in `Arc<parking_lot::RwLock<>>`:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use parking_lot::RwLock;
//! use tensor_spatial::{SpatialIndex, SpatialEntry, BoundingBox};
//!
//! let index = Arc::new(RwLock::new(SpatialIndex::<String>::new()));
//!
//! // Read access (multiple concurrent readers allowed):
//! let region = BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap();
//! let results = index.read().query_region(region);
//!
//! // Write access (exclusive):
//! index.write().insert(SpatialEntry {
//!     bounds: BoundingBox::new(1.0, 2.0, 1.0, 1.0).unwrap(),
//!     data: "item".to_string(),
//! });
//! ```

mod bbox;
mod index;
mod iter;
mod node;

pub use bbox::{BoundingBoxN, SpatialEntryN};
pub use index::SpatialIndexN;
pub use iter::SpatialIterN;

/// Default maximum entries per R-tree node before splitting.
const DEFAULT_MAX_ENTRIES: usize = 9;

/// Default minimum entries per R-tree node after splitting.
const DEFAULT_MIN_ENTRIES: usize = 4;

/// Split strategy for R-tree overflow handling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Linear split: O(n^2) seed selection, greedy distribution.
    Linear = 0,
    /// R\*-tree: axis-based split, overlap-minimizing, forced reinsertion.
    #[default]
    RStar = 1,
}

/// Configuration for R-tree node capacity and split behavior.
///
/// Controls the maximum and minimum number of entries per node. Construction
/// is validated to ensure `max_entries >= 4` and `min_entries = max_entries / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialConfig {
    /// Maximum entries per node before splitting.
    max_entries: usize,
    /// Minimum entries per node; underflow triggers `condense_tree`.
    min_entries: usize,
    /// Strategy used for node splits on overflow.
    split_strategy: SplitStrategy,
}

impl SpatialConfig {
    /// Default configuration (max=9, min=4, `RStar` strategy).
    ///
    /// # Default Strategy Change
    ///
    /// `SpatialIndexN::new()` defaults to `RStar` (previously `Linear`).
    /// Deserialized v2 indexes preserve `Linear` for backward compatibility.
    /// Use [`with_strategy`](Self::with_strategy) to opt into `Linear` explicitly.
    pub const DEFAULT: Self = Self {
        max_entries: DEFAULT_MAX_ENTRIES,
        min_entries: DEFAULT_MIN_ENTRIES,
        split_strategy: SplitStrategy::RStar,
    };

    /// Creates a config with the given maximum node capacity and `RStar` strategy.
    ///
    /// `min_entries` is derived as `max_entries / 2`.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidConfig`] if `max_entries < 4`.
    pub const fn new(max_entries: usize) -> Result<Self, SpatialError> {
        if max_entries < 4 {
            return Err(SpatialError::InvalidConfig);
        }
        Ok(Self {
            max_entries,
            min_entries: max_entries / 2,
            split_strategy: SplitStrategy::RStar,
        })
    }

    /// Creates a config with the given maximum node capacity and split strategy.
    ///
    /// `min_entries` is derived as `max_entries / 2`.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidConfig`] if `max_entries < 4`.
    pub const fn with_strategy(
        max_entries: usize,
        strategy: SplitStrategy,
    ) -> Result<Self, SpatialError> {
        if max_entries < 4 {
            return Err(SpatialError::InvalidConfig);
        }
        Ok(Self {
            max_entries,
            min_entries: max_entries / 2,
            split_strategy: strategy,
        })
    }

    /// Returns the maximum entries per node.
    #[must_use]
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Returns the minimum entries per node.
    #[must_use]
    pub const fn min_entries(self) -> usize {
        self.min_entries
    }

    /// Returns the split strategy used on overflow.
    #[must_use]
    pub const fn split_strategy(self) -> SplitStrategy {
        self.split_strategy
    }
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ---------------------------------------------------------------------------
// Type aliases -- preserve the existing public API names
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box in 2D space.
pub type BoundingBox = BoundingBoxN<2>;

/// An axis-aligned bounding box in 3D space.
pub type BoundingBox3D = BoundingBoxN<3>;

/// An entry in the 2D spatial index pairing a bounding box with user data.
pub type SpatialEntry<T> = SpatialEntryN<2, T>;

/// An entry in the 3D spatial index pairing a bounding box with user data.
pub type SpatialEntry3D<T> = SpatialEntryN<3, T>;

/// A 2D spatial index backed by an R-tree.
pub type SpatialIndex<T> = SpatialIndexN<2, T>;

/// A 3D spatial index backed by an R-tree.
pub type SpatialIndex3D<T> = SpatialIndexN<3, T>;

/// Iterator over references to 2D spatial entries.
pub type SpatialIter<'a, T> = SpatialIterN<'a, 2, T>;

/// Iterator over references to 3D spatial entries.
#[allow(dead_code)]
pub type SpatialIter3D<'a, T> = SpatialIterN<'a, 3, T>;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

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

    /// An invalid spatial configuration was provided.
    #[error("invalid config: max_entries must be at least 4")]
    InvalidConfig,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::node::NodeN;
    use super::*;

    #[test]
    fn test_node_bounds_empty_leaf() {
        let node: NodeN<2, u32> = NodeN::Leaf {
            entries: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    #[test]
    fn test_node_bounds_empty_internal() {
        let node: NodeN<2, u32> = NodeN::Internal {
            children: Vec::new(),
        };
        assert!(node.bounds().is_none());
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
    fn test_node_3d_bounds_empty_leaf() {
        let node: NodeN<3, u32> = NodeN::Leaf {
            entries: Vec::new(),
        };
        assert!(node.bounds().is_none());
    }

    #[test]
    fn test_node_3d_bounds_empty_internal() {
        let node: NodeN<3, u32> = NodeN::Internal {
            children: Vec::new(),
        };
        assert!(node.bounds().is_none());
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

    #[test]
    fn test_bulk_load_tree_is_internal_for_large_input() {
        let entries: Vec<SpatialEntry<u32>> = (0..50)
            .map(|i| SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            })
            .collect();
        let index = SpatialIndex::bulk_load(entries);
        assert_eq!(index.len(), 50);
        assert_eq!(index.root.len(), 50);
        assert!(
            matches!(index.root, NodeN::Internal { .. }),
            "50 entries should produce an Internal root"
        );
    }

    #[test]
    fn test_remove_condense_leaf_underflow() {
        // Insert enough to create an internal root, then delete entries from
        // one spatial region to trigger leaf underflow and reinsertion.
        let mut index = SpatialIndex::new();
        for i in 0..20u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32 * 2.0, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert_eq!(index.len(), 20);

        // Delete 10 entries -- enough to trigger multiple leaf underflows.
        for i in 0..10u32 {
            let bb = BoundingBox::new(i as f32 * 2.0, 0.0, 1.0, 1.0).unwrap();
            index.remove(bb, |e| e.data == i).unwrap();
        }
        assert_eq!(index.len(), 10);
        assert_eq!(index.root.len(), 10);

        // All 10 remaining entries must still be queryable.
        let big_region = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
        let results = index.query_region(big_region);
        assert_eq!(results.len(), 10);
        let mut found: Vec<u32> = results.iter().map(|e| e.data).collect();
        found.sort_unstable();
        assert_eq!(found, (10..20).collect::<Vec<u32>>());
    }

    #[test]
    fn test_remove_condense_root_shrink() {
        // Build a tree with an internal root, then delete until the root
        // collapses to a leaf.
        let mut index = SpatialIndex::new();
        for i in 0..15u32 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
                data: i,
            });
        }
        assert!(matches!(index.root, NodeN::Internal { .. }));

        // Delete all but 3.
        for i in 0..12u32 {
            let bb = BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap();
            index.remove(bb, |e| e.data == i).unwrap();
        }
        assert_eq!(index.len(), 3);
        assert_eq!(index.root.len(), 3);

        // Root should have collapsed to a leaf (3 < min_entries).
        assert!(
            matches!(index.root, NodeN::Leaf { .. }),
            "root should be a Leaf after heavy deletion"
        );
    }
}
