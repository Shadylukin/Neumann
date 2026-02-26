//! Generic N-dimensional bounding box and spatial entry types.

use std::fmt;

use serde::de::SeqAccess;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Serialize};

use crate::SpatialError;

/// An axis-aligned bounding box in `D`-dimensional space.
///
/// The box is defined by an `origin` (lower corner) and an `extent` (size per
/// axis). For 2D this corresponds to `(x, y, width, height)` and for 3D to
/// `(x, y, z, width, height, depth)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBoxN<const D: usize> {
    /// Lower corner coordinates (x, y, [z, ...]).
    pub(crate) origin: [f32; D],
    /// Size per axis (width, height, [depth, ...]).
    pub(crate) extent: [f32; D],
}

impl<const D: usize> BoundingBoxN<D> {
    /// Compile-time guard against zero-dimensional bounding boxes.
    const NON_ZERO_DIM: () = assert!(D > 0, "spatial dimension must be at least 1");

    /// Creates a bounding box from raw arrays without validating extents.
    ///
    /// Used by the deserialization path where validation is handled separately
    /// per dimension.
    const fn from_raw_unchecked(origin: [f32; D], extent: [f32; D]) -> Self {
        let () = Self::NON_ZERO_DIM;
        Self { origin, extent }
    }

    /// Creates a zero-origin, zero-extent bounding box.
    ///
    /// Used as a fallback when a node has no computable bounds.
    pub(crate) const fn from_raw_zero() -> Self {
        let () = Self::NON_ZERO_DIM;
        Self {
            origin: [0.0; D],
            extent: [0.0; D],
        }
    }

    /// Creates a bounding box from origin and extent arrays.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidBounds`] if any extent is negative.
    pub fn from_extents(origin: [f32; D], extent: [f32; D]) -> Result<Self, SpatialError> {
        let () = Self::NON_ZERO_DIM;
        for &e in &extent {
            if e < 0.0 {
                return Err(SpatialError::InvalidBounds);
            }
        }
        Ok(Self { origin, extent })
    }

    /// Returns the center point of the bounding box as an array.
    #[must_use]
    pub fn center_nd(self) -> [f32; D] {
        let mut c = [0.0_f32; D];
        for (c_i, (o_i, e_i)) in c.iter_mut().zip(self.origin.iter().zip(self.extent.iter())) {
            *c_i = o_i + e_i / 2.0;
        }
        c
    }

    /// Returns the hypervolume (product of all extents).
    ///
    /// For 2D this is area, for 3D this is volume.
    #[must_use]
    pub fn measure(self) -> f32 {
        self.extent.iter().product()
    }

    /// Returns `true` if `point` lies inside this bounding box (inclusive).
    #[must_use]
    pub fn contains_point_nd(self, point: &[f32; D]) -> bool {
        for ((&p, &o), &e) in point.iter().zip(self.origin.iter()).zip(self.extent.iter()) {
            if p < o || p > o + e {
                return false;
            }
        }
        true
    }

    /// Returns `true` if this bounding box overlaps with `other`.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        for i in 0..D {
            if self.origin[i] >= other.origin[i] + other.extent[i]
                || self.origin[i] + self.extent[i] <= other.origin[i]
            {
                return false;
            }
        }
        true
    }

    /// Returns the smallest bounding box containing both `self` and `other`.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let mut origin = [0.0_f32; D];
        let mut extent = [0.0_f32; D];
        for i in 0..D {
            let min = self.origin[i].min(other.origin[i]);
            let max = (self.origin[i] + self.extent[i]).max(other.origin[i] + other.extent[i]);
            origin[i] = min;
            extent[i] = max - min;
        }
        Self { origin, extent }
    }

    /// Returns the margin (sum of all extents).
    ///
    /// For 2D this is the semi-perimeter; used by the R\*-tree axis selection.
    #[must_use]
    pub fn margin(self) -> f32 {
        self.extent.iter().sum()
    }

    /// Returns the hypervolume of the intersection with `other`.
    ///
    /// Returns 0 if the boxes do not overlap on any axis.
    #[must_use]
    pub fn overlap_volume(self, other: Self) -> f32 {
        let mut vol = 1.0_f32;
        for i in 0..D {
            let lo = self.origin[i].max(other.origin[i]);
            let hi = (self.origin[i] + self.extent[i]).min(other.origin[i] + other.extent[i]);
            let overlap = (hi - lo).max(0.0);
            vol *= overlap;
        }
        vol
    }

    /// Returns `true` if any extent is zero (the box has zero hypervolume).
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.extent.contains(&0.0)
    }

    /// Returns the minimum squared distance from `point` to this box.
    #[must_use]
    pub fn min_dist_sq_nd(self, point: &[f32; D]) -> f32 {
        let mut sum = 0.0_f32;
        for ((&p, &o), &e) in point.iter().zip(self.origin.iter()).zip(self.extent.iter()) {
            let d = (o - p).max(0.0) + (p - o - e).max(0.0);
            sum = d.mul_add(d, sum);
        }
        sum
    }

    /// Returns the squared distance from `point` to this box's center.
    #[must_use]
    pub fn center_dist_sq_nd(self, point: &[f32; D]) -> f32 {
        let center = self.center_nd();
        let mut sum = 0.0_f32;
        for (&p, &c) in point.iter().zip(center.iter()) {
            let d = p - c;
            sum = d.mul_add(d, sum);
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Specialized 2D convenience methods
// ---------------------------------------------------------------------------

impl BoundingBoxN<2> {
    /// Creates a new 2D bounding box.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidBounds`] if width or height is negative.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, SpatialError> {
        Self::from_extents([x, y], [width, height])
    }

    /// X coordinate of the lower-left corner.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.origin[0]
    }

    /// Y coordinate of the lower-left corner.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.origin[1]
    }

    /// Width of the bounding box.
    #[must_use]
    pub const fn width(self) -> f32 {
        self.extent[0]
    }

    /// Height of the bounding box.
    #[must_use]
    pub const fn height(self) -> f32 {
        self.extent[1]
    }

    /// Returns the area of the bounding box.
    #[must_use]
    pub fn area(self) -> f32 {
        self.measure()
    }

    /// Returns the center point as `(cx, cy)`.
    #[must_use]
    pub fn center(self) -> (f32, f32) {
        self.center_nd().into()
    }

    /// Returns `true` if the point `(px, py)` lies inside this bounding box.
    #[must_use]
    pub fn contains_point(self, px: f32, py: f32) -> bool {
        self.contains_point_nd(&[px, py])
    }

    /// Returns the minimum squared distance from point `(px, py)` to this box.
    #[must_use]
    pub fn min_dist_sq(self, px: f32, py: f32) -> f32 {
        self.min_dist_sq_nd(&[px, py])
    }

    /// Returns the squared distance from point `(px, py)` to this box's center.
    #[must_use]
    pub fn center_dist_sq(self, px: f32, py: f32) -> f32 {
        self.center_dist_sq_nd(&[px, py])
    }
}

// ---------------------------------------------------------------------------
// Specialized 3D convenience methods
// ---------------------------------------------------------------------------

impl BoundingBoxN<3> {
    /// Creates a new 3D bounding box.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidBounds3D`] if width, height, or depth is
    /// negative.
    #[allow(clippy::similar_names)]
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
            origin: [x, y, z],
            extent: [width, height, depth],
        })
    }

    /// X coordinate of the lower corner.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.origin[0]
    }

    /// Y coordinate of the lower corner.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.origin[1]
    }

    /// Z coordinate of the lower corner.
    #[must_use]
    pub const fn z(self) -> f32 {
        self.origin[2]
    }

    /// Width of the bounding box (X extent).
    #[must_use]
    pub const fn width(self) -> f32 {
        self.extent[0]
    }

    /// Height of the bounding box (Y extent).
    #[must_use]
    pub const fn height(self) -> f32 {
        self.extent[1]
    }

    /// Depth of the bounding box (Z extent).
    #[must_use]
    pub const fn depth(self) -> f32 {
        self.extent[2]
    }

    /// Returns the volume of the bounding box.
    #[must_use]
    pub fn volume(self) -> f32 {
        self.measure()
    }

    /// Returns the center point as `(cx, cy, cz)`.
    #[must_use]
    pub fn center(self) -> (f32, f32, f32) {
        self.center_nd().into()
    }

    /// Returns `true` if the point `(px, py, pz)` lies inside this box.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn contains_point(self, px: f32, py: f32, pz: f32) -> bool {
        self.contains_point_nd(&[px, py, pz])
    }

    /// Returns the minimum squared distance from point `(px, py, pz)` to this
    /// box.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn min_dist_sq(self, px: f32, py: f32, pz: f32) -> f32 {
        self.min_dist_sq_nd(&[px, py, pz])
    }

    /// Returns the squared distance from point `(px, py, pz)` to this box's
    /// center.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn center_dist_sq(self, px: f32, py: f32, pz: f32) -> f32 {
        self.center_dist_sq_nd(&[px, py, pz])
    }
}

// ---------------------------------------------------------------------------
// Serde for BoundingBoxN<D>
// ---------------------------------------------------------------------------

impl<const D: usize> Serialize for BoundingBoxN<D> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(2 * D)?;
        for v in &self.origin {
            tup.serialize_element(v)?;
        }
        for v in &self.extent {
            tup.serialize_element(v)?;
        }
        tup.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for BoundingBoxN<D> {
    fn deserialize<De: serde::Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        struct BBoxVisitor<const D: usize>;

        impl<'de, const D: usize> serde::de::Visitor<'de> for BBoxVisitor<D> {
            type Value = BoundingBoxN<D>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a sequence of {} f32 values", 2 * D)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<BoundingBoxN<D>, A::Error> {
                let mut origin = [0.0_f32; D];
                let mut extent = [0.0_f32; D];
                for item in &mut origin {
                    *item = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                }
                for item in &mut extent {
                    *item = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(D, &self))?;
                }
                if extent.iter().any(|e| *e < 0.0) {
                    return Err(serde::de::Error::custom(
                        "invalid bounding box: all extents must be non-negative",
                    ));
                }
                Ok(BoundingBoxN::from_raw_unchecked(origin, extent))
            }
        }

        deserializer.deserialize_tuple(2 * D, BBoxVisitor::<D>)
    }
}

// ---------------------------------------------------------------------------
// SpatialEntryN<D, T>
// ---------------------------------------------------------------------------

/// An entry in the spatial index pairing a bounding box with user data.
pub struct SpatialEntryN<const D: usize, T> {
    /// The bounding box for this entry.
    pub bounds: BoundingBoxN<D>,
    /// User-supplied data associated with this entry.
    pub data: T,
}

impl<const D: usize, T: Serialize> Serialize for SpatialEntryN<D, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.bounds, &self.data).serialize(serializer)
    }
}

impl<'de, const D: usize, T: Deserialize<'de>> Deserialize<'de> for SpatialEntryN<D, T> {
    fn deserialize<De: serde::Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let (bounds, data) = <(BoundingBoxN<D>, T)>::deserialize(deserializer)?;
        Ok(Self { bounds, data })
    }
}

impl<const D: usize, T: fmt::Debug> fmt::Debug for SpatialEntryN<D, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialEntry")
            .field("bounds", &self.bounds)
            .field("data", &self.data)
            .finish()
    }
}

impl<const D: usize, T: Clone> Clone for SpatialEntryN<D, T> {
    fn clone(&self) -> Self {
        Self {
            bounds: self.bounds,
            data: self.data.clone(),
        }
    }
}

impl<const D: usize, T: PartialEq> PartialEq for SpatialEntryN<D, T> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds && self.data == other.data
    }
}
