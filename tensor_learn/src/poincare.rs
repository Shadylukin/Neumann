// SPDX-License-Identifier: MIT OR Apache-2.0
//! Poincare disk model for hyperbolic geometry.
//!
//! Uses f64 precision internally for numerical stability during training.
//! Points are projected onto the open unit disk at creation time.

use serde::{Deserialize, Serialize};

/// A point in the Poincare disk model (f64 precision).
///
/// The Poincare disk is the open unit disk { x : ||x|| < 1 } equipped
/// with the hyperbolic metric. Points are always projected to lie
/// strictly inside the disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincarePoint {
    /// Coordinates in the Poincare disk.
    coords: Vec<f64>,
}

/// Maximum norm for projection (just inside the boundary).
const MAX_NORM: f64 = 1.0 - 1e-6;

impl PoincarePoint {
    /// Create a new Poincare point, projecting onto the disk if necessary.
    #[must_use]
    pub fn new(coords: Vec<f64>) -> Self {
        let mut p = Self { coords };
        p.project_to_disk();
        p
    }

    /// Create the origin point in the given dimension.
    #[must_use]
    pub fn origin(dim: usize) -> Self {
        Self {
            coords: vec![0.0; dim],
        }
    }

    /// Squared norm of this point.
    #[must_use]
    pub fn norm_sq(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm of this point.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Whether this point lies strictly inside the unit disk.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.norm_sq() < 1.0
    }

    /// Access the coordinates.
    #[must_use]
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// Dimensionality of this point.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Project this point to lie strictly inside the unit disk.
    fn project_to_disk(&mut self) {
        let n = self.norm();
        if n >= MAX_NORM {
            let scale = MAX_NORM / n;
            for c in &mut self.coords {
                *c *= scale;
            }
        }
    }

    /// Convert to f32 vector for storage in vector engine.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.coords.iter().map(|&x| x as f32).collect()
    }

    /// Poincare distance to another point.
    ///
    /// `d(u,v) = (1/sqrt(c)) * arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))`
    #[must_use]
    pub fn distance(&self, other: &Self, curvature: f64) -> f64 {
        let diff_sq: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        if diff_sq < 1e-30 {
            return 0.0;
        }

        let norm_sq_a = self.norm_sq();
        let norm_sq_b = other.norm_sq();

        let denom = (1.0 - norm_sq_a) * (1.0 - norm_sq_b);
        if denom <= 0.0 {
            return f64::MAX;
        }

        let arg = 1.0 + 2.0 * diff_sq / denom;
        arg.acosh() / curvature.sqrt()
    }

    /// Mobius addition: u (+) v in the Poincare disk.
    ///
    /// The Mobius addition is the hyperbolic analog of vector addition.
    #[must_use]
    pub fn mobius_add(&self, other: &Self, curvature: f64) -> Self {
        let c = curvature;
        let u_sq = self.norm_sq();
        let v_sq = other.norm_sq();
        let uv: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum();

        let num_coeff_u = c.mul_add(v_sq, (2.0 * c).mul_add(uv, 1.0));
        let num_coeff_v = (-c).mul_add(u_sq, 1.0);
        let denom = (c * c * u_sq).mul_add(v_sq, (2.0 * c).mul_add(uv, 1.0));

        if denom.abs() < 1e-30 {
            return Self::origin(self.dim());
        }

        let coords: Vec<f64> = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(&u, &v)| (num_coeff_u * u + num_coeff_v * v) / denom)
            .collect();

        Self::new(coords)
    }

    /// Exponential map from tangent space at origin to the Poincare disk.
    ///
    /// Maps a tangent vector `v` at the origin to a point on the disk.
    #[must_use]
    pub fn exp_map(tangent: &[f64], curvature: f64) -> Self {
        let c_sqrt = curvature.sqrt();
        let v_norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

        if v_norm < 1e-15 {
            return Self::origin(tangent.len());
        }

        let scale = (c_sqrt * v_norm).tanh() / (c_sqrt * v_norm);
        let coords: Vec<f64> = tangent.iter().map(|&x| scale * x).collect();
        Self::new(coords)
    }

    /// Logarithmic map from the Poincare disk to tangent space at origin.
    ///
    /// Maps a point on the disk to a tangent vector at the origin.
    #[must_use]
    pub fn log_map(&self, curvature: f64) -> Vec<f64> {
        let c_sqrt = curvature.sqrt();
        let p_norm = self.norm();

        if p_norm < 1e-15 {
            return vec![0.0; self.dim()];
        }

        let scale = (c_sqrt * p_norm).atanh() / (c_sqrt * p_norm);
        self.coords.iter().map(|&x| scale * x).collect()
    }
}

impl PartialEq for PoincarePoint {
    fn eq(&self, other: &Self) -> bool {
        self.coords.len() == other.coords.len()
            && self
                .coords
                .iter()
                .zip(other.coords.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin() {
        let o = PoincarePoint::origin(3);
        assert_eq!(o.dim(), 3);
        assert_eq!(o.norm(), 0.0);
        assert!(o.is_valid());
    }

    #[test]
    fn test_new_inside_disk() {
        let p = PoincarePoint::new(vec![0.3, 0.4]);
        assert!(p.is_valid());
        assert!((p.norm() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_projection() {
        let p = PoincarePoint::new(vec![2.0, 0.0]);
        assert!(p.is_valid());
        assert!(p.norm() < 1.0);
    }

    #[test]
    fn test_distance_same_point() {
        let p = PoincarePoint::new(vec![0.3, 0.4]);
        assert_eq!(p.distance(&p, 1.0), 0.0);
    }

    #[test]
    fn test_distance_from_origin() {
        let o = PoincarePoint::origin(2);
        let p = PoincarePoint::new(vec![0.5, 0.0]);
        let d = o.distance(&p, 1.0);
        assert!(d > 0.0);
        // arcosh(1 + 2*0.25 / (1*0.75)) = arcosh(1 + 2/3)
        let expected = (1.0 + 2.0 * 0.25 / 0.75_f64).acosh();
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_distance_symmetric() {
        let a = PoincarePoint::new(vec![0.1, 0.2]);
        let b = PoincarePoint::new(vec![0.3, -0.1]);
        assert!((a.distance(&b, 1.0) - b.distance(&a, 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mobius_add_with_origin() {
        let o = PoincarePoint::origin(2);
        let p = PoincarePoint::new(vec![0.3, 0.4]);
        let result = o.mobius_add(&p, 1.0);
        assert!((result.coords[0] - p.coords[0]).abs() < 1e-10);
        assert!((result.coords[1] - p.coords[1]).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let tangent = vec![0.5, 0.3];
        let p = PoincarePoint::exp_map(&tangent, 1.0);
        let recovered = p.log_map(1.0);
        assert!((recovered[0] - tangent[0]).abs() < 1e-10);
        assert!((recovered[1] - tangent[1]).abs() < 1e-10);
    }

    #[test]
    fn test_exp_map_zero() {
        let p = PoincarePoint::exp_map(&[0.0, 0.0], 1.0);
        assert_eq!(p.norm(), 0.0);
    }

    #[test]
    fn test_log_map_origin() {
        let o = PoincarePoint::origin(3);
        let t = o.log_map(1.0);
        assert!(t.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_to_f32_vec() {
        let p = PoincarePoint::new(vec![0.1, 0.2, 0.3]);
        let v = p.to_f32_vec();
        assert_eq!(v.len(), 3);
        assert!((f64::from(v[0]) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_eq() {
        let a = PoincarePoint::new(vec![0.1, 0.2]);
        let b = PoincarePoint::new(vec![0.1, 0.2]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_curvature_scaling() {
        let o = PoincarePoint::origin(2);
        let p = PoincarePoint::new(vec![0.5, 0.0]);
        let d1 = o.distance(&p, 1.0);
        let d2 = o.distance(&p, 4.0);
        // Higher curvature means smaller distances
        assert!(d2 < d1);
    }

    #[test]
    fn test_triangle_inequality() {
        let a = PoincarePoint::new(vec![0.1, 0.0]);
        let b = PoincarePoint::new(vec![0.0, 0.2]);
        let c = PoincarePoint::new(vec![-0.1, 0.1]);
        let ab = a.distance(&b, 1.0);
        let bc = b.distance(&c, 1.0);
        let ac = a.distance(&c, 1.0);
        assert!(ac <= ab + bc + 1e-10);
    }

    #[test]
    fn test_mobius_add_stays_in_disk() {
        let a = PoincarePoint::new(vec![0.8, 0.0]);
        let b = PoincarePoint::new(vec![0.0, 0.8]);
        let result = a.mobius_add(&b, 1.0);
        assert!(result.is_valid());
    }

    #[test]
    fn test_serde_roundtrip() {
        let p = PoincarePoint::new(vec![0.3, 0.4]);
        let json = serde_json::to_string(&p).unwrap();
        let recovered: PoincarePoint = serde_json::from_str(&json).unwrap();
        assert_eq!(p, recovered);
    }
}
