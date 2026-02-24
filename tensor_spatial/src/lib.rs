//! R-tree spatial index for region and nearest-neighbor queries.
//!
//! Provides a generic N-dimensional R-tree with linear split algorithm,
//! supporting insertion, removal, region queries, and k-nearest-neighbor
//! lookups. The core types are parameterized by `const D: usize` for the
//! spatial dimension and `T` for user data.
//!
//! Type aliases preserve the existing 2D and 3D API:
//! - [`BoundingBox`] = `BoundingBoxN<2>`
//! - [`BoundingBox3D`] = `BoundingBoxN<3>`
//! - [`SpatialEntry`] = `SpatialEntryN<2, T>`
//! - [`SpatialEntry3D`] = `SpatialEntryN<3, T>`
//! - [`SpatialIndex`] = `SpatialIndexN<2, T>`
//! - [`SpatialIndex3D`] = `SpatialIndexN<3, T>`

mod bbox;
mod index;
mod iter;
mod node;

pub use bbox::{BoundingBoxN, SpatialEntryN};
pub use index::SpatialIndexN;
pub use iter::SpatialIterN;

/// Maximum entries per R-tree node before splitting.
const MAX_ENTRIES: usize = 9;

/// Minimum entries per R-tree node after splitting.
const MIN_ENTRIES: usize = 4;

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
}
